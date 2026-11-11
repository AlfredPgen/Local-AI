"""Staging tests of the 2026-11-10 audit, package A3 (tiny_gpt.py: training loop, console output, experiments
table, compile, main). Target classes in test_tiny_gpt.py: TestPipeline for the tests that start training runs
as child processes, TestFixes for the unit tests.

    python -m unittest test_a2_a3
"""
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_prep  # noqa: E402
import tiny_gpt  # noqa: E402
import test_tiny_gpt as base  # noqa: E402 - as a module: a TestCase class imported by name would run twice

CLOCK = r"[A-Za-z]{2,4} \d{2}:\d{2}"  # clock(): 'Tue 12:41'


class Audit2TinyGptC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2_a3_")
        cls.cache = {}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---------------- the fixture of TestPipeline, built by the first test that trains ----------------
    @classmethod
    def pipeline(cls):
        """The five-source dataset, the run folder and the common arguments of TestPipeline, and a probe file with
        the first 24 probes."""
        if "pipeline" in cls.cache:
            return
        cls.corpus = base.Corpus(cls.tmp)
        cls.ds = os.path.join(cls.tmp, "dataset")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", cls.ds, "--md-dir", cls.corpus.md, "--md-dir", f"notes={cls.corpus.notes}",
                               "--tokenizer-weights", "md=3", "--wiki-dir", cls.corpus.wiki,
                               "--jsonl", cls.corpus.pes2o, "--jsonl-name", "pes2o",
                               "--wiki-max-docs", "30", "--parquet", cls.corpus.parquet, "--parquet-name", "web",
                               "--min-score", "2", "--include-keywords", "genetic*,allele*,protein,DNA,RNA,genome*",
                               "--keyword-min-distinct", "2", "--vocab-size", "700", "--val-fraction", "0.2",
                               "--max-doc-chars", "4000", "--boilerplate-min-docs", "20", "--threads", "2"])
        cls.runs = os.path.join(cls.tmp, "runs")
        os.makedirs(cls.runs)
        cls.log = os.path.join(cls.runs, "log.txt")
        cls.common = ["tiny_gpt.py", "--dataset", cls.ds, "--out-dir", cls.runs, "--log-file", cls.log,
                      "--device", "cpu", "--d-model", "64", "--layers", "2",
                      "--heads", "4", "--kv-heads", "2", "--ctx", "64", "--batch-size", "8", "--grad-accum", "1",
                      "--eval-every", "8", "--eval-tokens", "2048", "--save-every", "1000", "--sample-tokens", "8",
                      "--dropout", "0.1", "--tariff", "25"]
        cls.probes = os.path.join(cls.tmp, "probes24.tsv")
        with open(tiny_gpt.DEFAULT_PROBES, encoding="utf-8") as src, open(cls.probes, "w", encoding="utf-8") as dst:
            kept = 0
            for line in src:
                if line.startswith("#") or kept < 24:
                    dst.write(line)
                    kept += not line.startswith("#")
        cls.cache["pipeline"] = True

    def resume_args(self, name):
        return ["--name", name, "--out-dir", self.runs, "--log-file", self.log, "--device", "cpu", "--resume",
                "--eval-every", "8", "--eval-tokens", "2048", "--sample-tokens", "8", "--no-probes", "--tariff", "25"]

    def patched_run(self, mode, n, args):
        """TestPipeline.patched_run: tiny_gpt.main(args) in a child process with one fault injected."""
        script = os.path.join(self.tmp, "patched_run.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(base.PATCHED_RUN)
        return base.run([script, HERE, mode, str(n)] + args)

    def logged_run(self, args, patched=None):
        """(exit code, output, the part of log.txt this run appended): the log of a test class is cumulative."""
        size = os.path.getsize(self.log) if os.path.exists(self.log) else 0
        code, out = self.patched_run(*patched, args) if patched else base.run(args)
        with open(self.log, "rb") as handle:
            handle.seek(size)
            tail = handle.read().decode("utf-8").replace("\r\n", "\n")
        return code, out, tail

    def rows(self, name):
        with open(os.path.join(self.runs, f"{name}_metrics.jsonl"), encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def header(text):
        """The lines of a run header: from the RUN line to the closing bar (both left out)."""
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("RUN ") and " started " in line)
        end = next(i for i in range(start + 1, len(lines)) if lines[i] == "=" * 100)
        return lines[start + 1:end]

    # ---------------- console (section 2a of the plan) ----------------
    def console_runs(self):
        """One toy run with fact probes on the five-source dataset: 8 steps, evaluated every 4, stopped after 2
        and resumed. Returns (output, log part) of the first session and of the resumed one."""
        if "console" not in self.cache:
            self.pipeline()
            code, out, tail = self.logged_run(self.common + ["--name", "C", "--steps", "8", "--eval-every", "4",
                                                             "--stop-after-steps", "2", "--probes", self.probes])
            self.assertEqual(code, 0, out)
            code, out_r, tail_r = self.logged_run(["tiny_gpt.py", "--name", "C", "--out-dir", self.runs,
                                                   "--log-file", self.log, "--device", "cpu", "--resume",
                                                   "--eval-every", "4", "--sample-tokens", "8", "--probes",
                                                   self.probes, "--tariff", "25"])
            self.assertEqual(code, 0, out_r)
            self.cache["console"] = (out, tail, out_r, tail_r)
        return self.cache["console"]

    def test_no_per_source_values_on_the_console(self):
        """F217 (with F218, F261, F198): the evaluation line and the baseline line carry no per-source bracket;
        the values stay in <name>_metrics.jsonl."""
        out, _, out_r, _ = self.console_runs()
        self.assertNotRegex(out + out_r, r"@\d+ \[")
        self.assertRegex(out_r, r"step +8/8 \| train [\d.]+ \| val [\d.]+ @8 \| val\(mix\)")
        self.assertIn("baseline before training", out)  # a fresh run: the step-0 line is printed
        last = self.rows("C")[-1]
        self.assertEqual(last["step"], 8)
        self.assertEqual(sorted(last["val_per_source"]), ["md", "pes2o", "wiki"])
        self.assertEqual(sorted(last["bpb_per_source"]), ["md", "pes2o", "wiki"])
        self.assertFalse(hasattr(tiny_gpt, "_per_source"))


if __name__ == "__main__":
    unittest.main()
