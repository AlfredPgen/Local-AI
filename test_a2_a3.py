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

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_prep  # noqa: E402
import tiny_gpt  # noqa: E402
import test_tiny_gpt as base  # noqa: E402 - as a module: a TestCase class imported by name would run twice

CLOCK = r"[A-Za-z]{2,4} \d{2}:\d{2}"  # clock(): 'Tue 12:41'
# tiny_gpt.main(args) in a child process with --dashboard on: the best-file saves whose numbers (from 1) are in the
# comma list fail, and every dashboard launch is printed instead of started
DASHBOARD_RUN = r'''
import os
import sys

sys.path.insert(0, sys.argv[1])
import tiny_gpt  # noqa: E402

fail = {int(v) for v in sys.argv[2].split(",") if v}
calls = {"best": 0}
real_save = tiny_gpt.atomic_save


def save(obj, path):
    if obj.get("checkpoint_type") == "inference" and not path.endswith("_crash.pt"):
        calls["best"] += 1
        if calls["best"] in fail:
            raise OSError(28, "No space left on device (simulated)")
    return real_save(obj, path)


def launch(path, args, state):
    print(f"DASHBOARD LAUNCH {os.path.basename(path)} exists={os.path.exists(path)}")


tiny_gpt.device_memory_budget = lambda device: (8 * 2 ** 30, "8.0 GiB (test)")
tiny_gpt.atomic_save = save
tiny_gpt.launch_dashboard = launch
tiny_gpt.last_dashboard_push = lambda png_name: 0.0  # no git call
tiny_gpt.main(sys.argv[3:])
'''


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

    def test_run_header_is_short_on_screen_and_whole_in_the_log(self):
        """F243 (with F264, F256): the screen gets a short start-up block; the command, the versions, the attention
        and optimizer notes, the model and training settings as JSON, one per-source list, the energy method and
        the full checkpoint paths go to log.txt alone, which still holds every line of the screen."""
        import view_pt
        out, tail, out_r, tail_r = self.console_runs()
        for text in (out, out_r):
            head = self.header(text)
            joined = "\n".join(head)
            for gone in ("command: ", "train: {", "model: {", "metrics line (train = mean", "sampling share per source",
                         "attention: SDPA", "energy: ", "checkpoints: ", "weights: "):
                self.assertNotIn(gone, "".join(line for line in head if not line.startswith("plan: ")))
            starts = [line.split(":")[0] for line in head if not line.startswith("plan: ")]
            self.assertEqual([s for s in starts if s != "Earlier sessions of this run"],
                             ["device", "data", "model", "parameters", "budget", "schedule", "output"], joined)
            self.assertRegex(joined, r"(?m)^device: .* \| FP32 \| AdamW \| compile: off \(eager\)$")
            self.assertRegex(joined, r"(?m)^data: .*dataset \([0-9a-f]{12}\) \| [\d,]+ training tokens in 5 sources \| "
                                     r"validation: \d+ fixed windows of 64 tokens from 3 sources$")
            self.assertIn("model: d_model 64, 2 layers, 4 heads / 2 KV, ctx 64, vocabulary 700", head)
            self.assertRegex(joined, r"(?m)^parameters: [\d,]+ total, [\d,]+ non-embedding$")
            self.assertRegex(joined, r"(?m)^budget: 8 steps x 512 tokens \(8 x 1 x 64\) = 4,096 tokens = [\d.]+ "
                                     r"tokens/parameter, [\d.]+ epochs$")
            self.assertRegex(joined, r"(?m)^output: validation every 4 steps with 24 fact probes \| C\.pt every 1000 "
                                     r"steps and every 30 min, C_best\.pt on NEW BEST, in .*runs \| full settings: "
                                     r"log\.txt$")
        self.assertLessEqual(len(self.header(out_r)), 12, out_r)
        self.assertEqual(sum(line.startswith("Earlier sessions of this run: ") for line in self.header(out_r)), 1)
        self.assertNotIn("Earlier sessions", out)
        self.assertIn("measuring the untrained model on the fixed validation windows ...\n", out)
        self.assertFalse(hasattr(tiny_gpt.EvalSet, "describe"))
        for log_part, screen in ((tail, out), (tail_r, out_r)):
            for needle in ("command: ", "python ", " | tiny_gpt.py sha256 ", "device detail: attention: SDPA ",
                           " | optimizer: AdamW (", "model: {", "train: {", "energy: ", "checkpoints: ",
                           "per source (training tokens, sampling share; validation tokens, fixed windows): ",
                           "evaluation: every 4 steps on the fixed validation windows; fact probes: 24 from "
                           "probes24.tsv"):
                self.assertIn(needle, log_part)
            self.assertRegex(log_part, r"(?m)^checkpoints: .*[\\/]C\.pt every 1000 steps and every 30 min \(training "
                                       r"state\), .*C_best\.pt on NEW BEST \(inference only\); metrics: .*C_metrics\.jsonl$")
            self.assertRegex(log_part, r"per source \(training tokens, sampling share; validation tokens, fixed windows\): "
                                       r"md [\d,]+ [\d.]+%, [\d,]+ \d+ x 64; notes [\d,]+ [\d.]+%, no validation; ")
            self.assertRegex(log_part, r"={20,}\nRUN ")  # view_pt splits the sessions of a log here
            logged = set(log_part.splitlines())
            missing = [line for line in screen.splitlines() if line.strip() and line not in logged]
            self.assertEqual(missing, [], "log.txt holds every line of the screen")
        history, note = view_pt.history_from_log(self.log, os.path.join(self.runs, "C.pt"))
        self.assertEqual([row["step"] for row in history], [4, 8], note)

    def test_a_line_before_the_header_says_what_the_wait_is_for(self):
        """F224: one line before the dataset is checked and the model compiled, ahead of the header's bar."""
        out, tail, out_r, tail_r = self.console_runs()
        fresh, resumed = ("checking the dataset and compiling the model ...",
                          "loading C.pt, checking the dataset and compiling the model ...")
        self.assertIn(fresh, out.splitlines())
        self.assertIn(resumed, out_r.splitlines())
        self.assertNotIn("loading ", out[:out.index("=" * 100)])
        for text, line in ((out, fresh), (out_r, resumed), (tail, fresh), (tail_r, resumed)):
            self.assertLess(text.index(line), text.index("=" * 100))
            self.assertRegex(text, r"={20,}\nRUN ")

    def test_header_hash_of_the_script_ignores_line_endings(self):
        """F263: the 'tiny_gpt.py sha256' of the run header is the same for a checkout with LF and with CRLF line
        ends; file_sha256 (dataset and probe files, the stored probes_sha) still hashes the bytes."""
        lf, crlf = os.path.join(self.tmp, "lf.py"), os.path.join(self.tmp, "crlf.py")
        with open(lf, "wb") as handle:
            handle.write(b"a = 1\nb = 2\n")
        with open(crlf, "wb") as handle:
            handle.write(b"a = 1\r\nb = 2\r\n")
        self.assertEqual(tiny_gpt.source_sha256(lf), tiny_gpt.source_sha256(crlf))
        self.assertEqual(tiny_gpt.source_sha256(lf), tiny_gpt.file_sha256(lf))
        self.assertNotEqual(tiny_gpt.file_sha256(lf), tiny_gpt.file_sha256(crlf))
        self.assertEqual(tiny_gpt.source_sha256(os.path.join(self.tmp, "missing.py")), "?")
        _, tail, _, _ = self.console_runs()
        self.assertIn(f" | tiny_gpt.py sha256 {tiny_gpt.source_sha256(tiny_gpt.__file__)[:12]}\n", tail)

    def test_plan_lines_of_a_run_header_are_the_compact_ones(self):
        """F336 (call sites): a new run prints the planner's lines without what its header already says."""
        out, _, out_r, _ = self.console_runs()
        plan = [line for line in self.header(out) if line.startswith("plan: ")]
        self.assertTrue(any(line.startswith("plan: unique training tokens U = ") for line in plan), plan)
        self.assertTrue(any(line.startswith("plan: token budget D = ") for line in plan), plan)
        for line in plan:
            self.assertNotRegex(line, r"^plan: (batch:|weights:|ctx |ratios:)")
        self.assertNotRegex(out, r"plan: unique training tokens U = [\d,]+ \(")  # no per-source list
        self.assertEqual([line for line in self.header(out_r) if line.startswith("plan: ")], [])

    @staticmethod
    def metric_lines(text):
        return [line for line in text.splitlines() if re.match(r"step +\d+/\d+ \| train ", line)]

    def test_evaluation_line_without_perplexity_and_carbon(self):
        """F231: the evaluation line and the baseline line carry no perplexity and the energy without its carbon;
        both stay in the metrics file, and the carbon on the RUN end line."""
        out, _, out_r, _ = self.console_runs()
        self.assertEqual(len(self.metric_lines(out_r)), 2, out_r)
        self.assertNotRegex(out + out_r, r"(?m)^step +\d+/\d+ \|[^\n]*\| ppl ")
        self.assertNotRegex(out + out_r, r"(?m)^step +\d+/\d+ \|[^\n]*g CO2e")
        self.assertRegex(out_r, r"(?m)^step +8/8 \| train [^\n]*\| \d+\.\d{2} kWh, £\d+\.\d{2} \|")
        self.assertRegex(out_r, r"(?m)^RUN .* kWh, [\d,]+ g CO2e, £")
        last = self.rows("C")[-1]
        self.assertGreater(last["ppl"], 1.0)
        self.assertIn("co2e_g", last)

    def test_validation_losses_are_printed_to_four_decimals(self):
        """F229: val, prev and best on the evaluation line and val on the baseline line have four decimals (at
        three, a NEW BEST could show the same value as the one it beat); train and val(mix) keep three."""
        out, _, out_r, _ = self.console_runs()
        self.assertRegex(out, r"(?m)^step +0/8 \| baseline before training \| val \d+\.\d{4} @0 \|")
        self.assertRegex(out_r, r"(?m)^step +4/8 \| train \d+\.\d{3} \| val \d+\.\d{4} @4 \| val\(mix\) \d+\.\d{3} \|")
        self.assertRegex(out_r, r"(?m)^step +8/8 \|[^\n]*\| (NEW BEST \(prev \d+\.\d{4} @4\)|best \d+\.\d{4} @4)$")
        rows = {row["step"]: row for row in self.rows("C")}
        self.assertIn(f"| val {rows[8]['val_loss']:.4f} @8 |", out_r)
        self.assertIn(f"| val {rows[0]['val_loss']:.4f} @0 |", out)

    def test_evaluation_line_shows_the_fact_probe_accuracy(self):
        """F230: the fact-probe accuracy of an evaluation is on its line, after bpb; view_pt still reads the line."""
        import view_pt
        _, _, out_r, _ = self.console_runs()
        rows = {row["step"]: row for row in self.rows("C")}
        for line in self.metric_lines(out_r):
            self.assertRegex(line, r"\| bpb \d+\.\d{3} \| probes \d+\.\d% \| lr ")
            step, total, train, val, at, lr = view_pt.NEW_LINE.search(line).groups()
            self.assertEqual((step, total), (at, "8"))
            self.assertIn(f"| probes {rows[int(step)]['probe_acc']:.1%} |", line)

    def test_durations_in_days_and_clock_times(self):
        """F309: a duration of two days or more is printed in days and hours, and the evaluation line carries the
        clock time in place of the session's and the run's age (those stay on the RUN end line)."""
        self.assertEqual(tiny_gpt.fmt_dur(1_486_500), "17d 04h")
        self.assertEqual(tiny_gpt.fmt_dur(172_800), "2d 00h")
        self.assertEqual(tiny_gpt.fmt_dur(172_799), "47h 59m")
        self.assertRegex(tiny_gpt.clock(), "^" + CLOCK + "$")
        _, _, out_r, _ = self.console_runs()
        for line in self.metric_lines(out_r):
            self.assertRegex(line, r"\| " + CLOCK + r" \| ETA \d+m \d{2}s \| ")
            self.assertNotIn("elapsed", line)
        self.assertRegex(out_r, r"(?m)^RUN .* \| this session \d+:\d{2}:\d{2} \| whole run: \d+:\d{2}:\d{2} in total")

    def test_heartbeat_line_is_short_and_follows_the_first_step(self):
        """F234: the heartbeat line carries the step, the share done, the training loss, the speed, the clock time
        and the time to the next evaluation; a session prints one after its first completed step, unless that
        step has an evaluation or a --log-every line."""
        out, _, out_r, _ = self.console_runs()
        for text, step in ((out, 1), (out_r, 3)):
            beats = [line for line in text.splitlines() if line.startswith("  ... step")]
            self.assertEqual(len(beats), 1, text)
            self.assertRegex(beats[0], r"^  \.\.\. step +%d/8 \| +[\d.]+%% \| train \d+\.\d{3} \| [\d,]+ tok/s \| %s \| "
                                       r"next val in \d+m \d{2}s$" % (step, CLOCK))
            self.assertNotIn("tokens |", beats[0])
        code, out_p = base.run(self.common + ["--name", "H", "--steps", "2", "--eval-every", "2", "--log-every", "1",
                                              "--no-probes"])
        self.assertEqual(code, 0, out_p)
        self.assertNotIn("  ... step", out_p)
        self.assertRegex(out_p, r"(?m)^step +1/2 \| 50\.0%% \| train \d+\.\d{3} \| [\d,]+ tok/s \| %s \| next val in "
                                r"\d+m \d{2}s$" % CLOCK)
        code, out_q = base.run(self.common + ["--name", "H0", "--steps", "2", "--eval-every", "2", "--no-probes",
                                              "--heartbeat-seconds", "0"])
        self.assertEqual(code, 0, out_q)
        self.assertNotIn("  ... step", out_q)

    def test_end_line_without_an_evaluation_says_none(self):
        """F242: a session that ends before its first evaluation ends with 'best val none', not 'inf @None'."""
        out, _, out_r, _ = self.console_runs()
        self.assertRegex(out, r"(?m)^RUN \S+ stopped at \S+ \| step 2/8 \| best val none \| last val none \| this session ")
        self.assertNotIn("inf @None", out + out_r)
        self.assertRegex(out_r, r"(?m)^RUN \S+ completed at \S+ \| step 8/8 \| best val \d+\.\d{4} @\d \| last val "
                                r"\d+\.\d{4} @8 \| ")

    # ---------------- best-file save, dashboard launch ----------------
    def load(self, name):
        return torch.load(os.path.join(self.runs, name), map_location="cpu", weights_only=True)

    def test_a_failed_best_save_is_tolerated_three_times(self):
        """F249 (with F235): a save of the best file that fails (full disk, no memory for the copy) no longer ends
        the run. The evaluation is recorded as no best, the next better evaluation tries again, and three
        failures in a row end the run, as three failed periodic saves do."""
        self.pipeline()
        args = list(self.common[1:]) + ["--steps", "16", "--eval-every", "4", "--no-probes"]
        code, out = self.patched_run("bestfail", 2, args + ["--name", "BF"])
        self.assertEqual(code, 0, out)
        warned = [line for line in out.splitlines() if line.startswith("WARNING: could not save")]
        self.assertEqual(len(warned), 2, out)
        for i, line in enumerate(warned, 1):
            self.assertIn("BF_best.pt (OSError: ", line)
            self.assertIn(f"training continues, the next better evaluation tries again ({i}/3)", line)
        self.assertNotIn("inf @None", out)
        lines = self.metric_lines(out)
        self.assertEqual(len(lines), 4, out)
        self.assertTrue(lines[0].endswith(" | no best saved yet"), lines[0])
        self.assertTrue(lines[1].endswith(" | no best saved yet"), lines[1])
        self.assertTrue(lines[2].endswith(" | NEW BEST (first evaluation)"), lines[2])
        self.assertRegex(lines[3], r" \| (NEW BEST \(prev \d+\.\d{4} @12\)|best \d+\.\d{4} @12)$")
        rows = self.rows("BF")
        self.assertEqual([row["step"] for row in rows], [0, 4, 8, 12, 16], "one row per evaluation")
        for row in rows[1:3]:  # the evaluations whose best save failed
            self.assertIs(row["is_best"], False)
            self.assertIsNone(row["best_weights"])
        self.assertEqual((rows[3]["is_best"], rows[3]["best_weights"]), (True, "raw"))
        state, best = self.load("BF.pt"), self.load("BF_best.pt")
        self.assertEqual(state["step"], 16)
        self.assertIn(best["step"], (12, 16))
        self.assertEqual([(h["step"], h["is_best"]) for h in state["metrics"]["history"][1:4]],
                         [(4, False), (8, False), (12, True)])
        self.assertEqual(state["metrics"]["best_step"], best["step"])
        # three failures in a row end the run: a crash file, no best file, the third evaluation in the metrics
        code, out_f = self.patched_run("bestfail", 3, args + ["--name", "BG"])
        self.assertNotEqual(code, 0)
        self.assertEqual(out_f.count("WARNING: could not save"), 2, out_f)
        self.assertIn("(2/3)", out_f)
        self.assertIn("OSError", out_f)
        self.assertTrue(os.path.isfile(os.path.join(self.runs, "BG_crash.pt")))
        self.assertFalse(os.path.exists(os.path.join(self.runs, "BG_best.pt")))
        rows_f = self.rows("BG")
        self.assertEqual([(row["step"], row["is_best"], row.get("best_weights")) for row in rows_f[1:]],
                         [(4, False, None), (8, False, None), (12, False, None)])
        self.assertRegex(out_f, r"(?m)^RUN \S+ failed at \S+ \| step 12/16 \| best val none \| ")

    def test_dashboard_is_drawn_after_every_evaluation_with_a_best_file(self):
        """F368 (tiny_gpt.py half): the dashboard of the best file is launched after every evaluation, a new best
        or not, once that file exists, and after the evaluation's row is in the metrics file."""
        self.pipeline()
        script = os.path.join(self.tmp, "dashboard_run.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(DASHBOARD_RUN)
        args = list(self.common[1:]) + ["--steps", "12", "--eval-every", "4", "--no-probes", "--dashboard", "on"]
        # the second best save fails: that evaluation is no new best, and the best file of step 4 is drawn again
        code, out = base.run([script, HERE, "2"] + args + ["--name", "DA"])
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("DASHBOARD LAUNCH DA_best.pt exists=True"), 3, out)
        lines = [line for line in out.splitlines()
                 if line.startswith(("DASHBOARD LAUNCH", "WARNING: could not save")) or self.metric_lines(line)]
        self.assertEqual([line.split(" ")[0] for line in lines],
                         ["DASHBOARD", "step", "WARNING:", "DASHBOARD", "step", "DASHBOARD", "step"], out)
        self.assertRegex(lines[4], r" \| best \d+\.\d{4} @4$")
        # the first best save fails: nothing to draw until a best file exists
        code, out_n = base.run([script, HERE, "1"] + args + ["--name", "DB"])
        self.assertEqual(code, 0, out_n)
        self.assertEqual(out_n.count("DASHBOARD LAUNCH"), 2, out_n)
        self.assertNotIn("exists=False", out_n)
        first = self.metric_lines(out_n)[0]
        self.assertTrue(first.endswith(" | no best saved yet"), first)
        self.assertLess(out_n.index(first), out_n.index("DASHBOARD LAUNCH"))

    # ---------------- pause and save messages ----------------
    def test_second_ctrl_c_waits_for_the_pause_save(self):
        """F226 (with F250): Ctrl+C says that the state is being saved, and a second Ctrl+C during that save is
        held until the file is written instead of aborting it; the paused run resumes exactly."""
        self.pipeline()
        args = list(self.common[1:]) + ["--steps", "8", "--no-probes"]

        def lines(text):
            return re.findall(r"step +(\d+)/8 \| train ([\d.]+) \| val ([\d.]+)", text)

        code, out_u = self.patched_run("none", 0, args + ["--name", "IU"])
        self.assertEqual(code, 0, out_u)
        # Ctrl+C inside step 4 (its batch read), and a SIGINT while the pause save of step 3 is being written
        code, out = self.patched_run("interrupt_twice", 4, args + ["--name", "IT"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("Traceback", out)
        self.assertIn("Ctrl+C: saving the training state at step 3, please wait ...\n", out)
        self.assertIn("\nPaused after step 3 (last completed update); saved ", out)
        self.assertLess(out.index("Ctrl+C: saving"), out.index("Paused after step 3 "))
        self.assertRegex(out, r"(?m)^RUN \S+ interrupted at ")
        self.assertEqual(self.load("IT.pt")["step"], 3)
        code, out_r = self.patched_run("none", 0, self.resume_args("IT"))
        self.assertEqual(code, 0, out_r)
        self.assertTrue(lines(out_u))
        self.assertEqual(lines(out_u), lines(out_r), "the paused run must reproduce the uninterrupted one")

    def test_failed_save_warning_names_the_next_attempt(self):
        """F225 (with F237): the warning of a failed periodic save says when the next attempt is: in about a
        minute with --save-minutes, at the next multiple of --save-every with --save-minutes 0."""
        self.pipeline()
        args = list(self.common[1:]) + ["--steps", "8", "--no-probes", "--save-every", "3"]
        code, out = self.patched_run("savefail", 1, args + ["--name", "SW0", "--save-minutes", "0"])
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("WARNING: could not save"), 1, out)
        self.assertRegex(out, r"WARNING: could not save .*SW0\.pt \(OSError: .*\); training continues, next attempt "
                              r"at step 6 \(1/3\)\n")
        self.assertNotIn("in about a minute", out)
        self.assertEqual(self.load("SW0.pt")["step"], 8)
        code, out_m = self.patched_run("savefail", 1, args + ["--name", "SW1"])
        self.assertEqual(code, 0, out_m)
        self.assertRegex(out_m, r"WARNING: could not save .*SW1\.pt \(OSError: .*\); training continues, next attempt "
                                r"in about a minute \(1/3\)\n")

    # ---------------- the dataset of a resumed run ----------------
    def test_resume_takes_the_dataset_as_an_absolute_path(self):
        """F239 (with F246): a resumed run finds its dataset the way the other commands do (the stored path, also
        from this script's folder, or datasets/<name> with the same fingerprint) and uses it as an absolute
        path; --dataset is made absolute too; a dataset that is found nowhere keeps the stored text, so that
        _train's own messages explain it."""
        home = os.path.join(self.tmp, "repo239")
        stored = os.path.join(home, "datasets", "bio_x")
        elsewhere = os.path.join(self.tmp, "cwd239")
        os.makedirs(stored)
        os.makedirs(elsewhere)
        with open(os.path.join(stored, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"format": data_prep.FORMAT, "format_version": 1, "fingerprint": "abc"}, f)
        before = os.getcwd()
        os.chdir(elsewhere)
        try:
            with unittest.mock.patch.object(tiny_gpt, "HERE", home):
                for path in ("datasets\\bio_x", "datasets/bio_x", r"C:\Users\someone\proj\datasets\bio_x"):
                    got = tiny_gpt.resume_dataset_dir(None, {"dataset": {"path": path, "fingerprint": "abc"}})
                    self.assertTrue(os.path.isabs(got), got)
                    self.assertEqual(os.path.normcase(os.path.normpath(got)), os.path.normcase(stored))
                obj = {"dataset": {"path": "datasets/bio_x", "fingerprint": "abc"}}
                self.assertEqual(os.path.normcase(tiny_gpt.resume_dataset_dir("ds", obj)),
                                 os.path.normcase(os.path.join(os.getcwd(), "ds")))
                # found nowhere, or rebuilt since: the stored text, for the 'not found' and 'differs' messages
                for gone in ({"path": "datasets/gone", "fingerprint": "abc"},
                             {"path": "datasets/bio_x", "fingerprint": "zzz"}):
                    self.assertEqual(tiny_gpt.resume_dataset_dir(None, {"dataset": gone}), gone["path"])
        finally:
            os.chdir(before)

    @base.SLOW
    def test_resume_with_a_relative_dataset_stores_an_absolute_path(self):
        """F239: '--resume --dataset <relative path>' no longer writes the relative text into the checkpoint, so
        the run can be resumed again from any folder."""
        self.pipeline()
        script = os.path.join(HERE, "tiny_gpt.py")
        code, out = base.run(self.common + ["--name", "R239", "--steps", "6", "--eval-every", "2", "--no-probes",
                                            "--stop-after-steps", "2"])
        self.assertEqual(code, 0, out)
        resume = [script] + self.resume_args("R239")
        code, out = base.run(resume + ["--dataset", os.path.relpath(self.ds, self.tmp), "--stop-after-steps", "2"],
                             cwd=self.tmp)
        self.assertEqual(code, 0, out)
        stored = self.load("R239.pt")["dataset"]["path"]
        self.assertTrue(os.path.isabs(stored), stored)
        self.assertEqual(os.path.normcase(stored), os.path.normcase(self.ds))
        code, out = base.run(resume, cwd=self.runs)  # no --dataset, another folder
        self.assertEqual(code, 0, out)
        self.assertEqual(self.load("R239.pt")["step"], 6)

    # ---------------- notes about options a resumed or continued run does not take ----------------
    def notes_runs(self):
        """A finished toy run N (8 steps, z-loss off, RoPE base 500, a weight decay that is not the default),
        then two resumes of it that have nothing left to train and so only print their notes and header: one
        with options a resumed run ignores, one plain. Returns the two outputs."""
        if "notes" not in self.cache:
            self.pipeline()
            code, out = base.run(self.common + ["--name", "N", "--steps", "8", "--eval-every", "4", "--no-probes",
                                                "--z-loss", "0", "--rope-base", "500", "--weight-decay", "0.1"])
            self.assertEqual(code, 0, out)
            code, out_flags = base.run(["tiny_gpt.py"] + self.resume_args("N") + [
                "--grad-clip", "0.123", "--decay-frac", "0.5", "--weight-decay", "0.9", "--rope-base", "500",
                "--ffn-hidden", "64", "--beta2", "0.5", "--z-loss", "1e-4"])
            self.assertEqual(code, 0, out_flags)
            code, out_plain = base.run(["tiny_gpt.py"] + self.resume_args("N"))
            self.assertEqual(code, 0, out_plain)
            for text in (out_flags, out_plain):
                self.assertIn("nothing to train", text)
            self.cache["notes"] = (out_flags, out_plain)
        return self.cache["notes"]

    def init_runs(self):
        """Two short runs continued from N with --init-from: from its best file with options that belong to
        the checkpoint's architecture, and from its training state (step 8) for two more steps."""
        if "init" not in self.cache:
            self.notes_runs()
            short = self.common + ["--eval-every", "2", "--no-probes"]
            code, out_best = base.run(short + ["--name", "I1", "--init-from", os.path.join(self.runs, "N_best.pt"),
                                               "--steps", "2", "--d-model", "128", "--tie-embeddings", "off",
                                               "--rope-base", "2000"])
            self.assertEqual(code, 0, out_best)
            code, out_state = base.run(short + ["--name", "I2", "--init-from", os.path.join(self.runs, "N.pt"),
                                                "--train-tokens", "1024"])
            self.assertEqual(code, 0, out_state)
            self.cache["init"] = (out_best, out_state)
        return self.cache["init"]

    def test_ignored_options_are_named_on_resume_and_init_from(self):
        """F238: --resume names every option it does not take, also those with a default (when the value given
        is neither the default nor the run's own), and a plain --resume prints no note; --init-from says which
        architecture options the checkpoint overrides."""
        out_flags, out_plain = self.notes_runs()
        note = [line for line in out_flags.splitlines() if "ignored on --resume (the checkpoint's" in line]
        self.assertEqual(len(note), 1, out_flags)
        for flag in ("--grad-clip", "--decay-frac", "--weight-decay", "--rope-base", "--ffn-hidden", "--beta2"):
            self.assertIn(flag, note[0])
        for flag in ("--beta1", "--schedule", "--decay-shape", "--d-model"):
            self.assertNotIn(flag, note[0])
        self.assertNotIn("ignored on --resume", out_plain)  # although the run's weight decay is not the default
        self.assertEqual(self.load("N.pt")["train_config"]["weight_decay"], 0.1)
        out_best, _ = self.init_runs()
        plan = [line for line in self.header(out_best) if line.startswith("plan: ")]
        for start in ("plan: --d-model 128 ignored: ", "plan: --tie-embeddings off ignored: ",
                      "plan: --rope-base 2000 ignored: "):
            self.assertEqual(sum(line.startswith(start) for line in plan), 1, plan)
        self.assertFalse([line for line in plan if line.startswith(("plan: --layers", "plan: --heads",
                                                                    "plan: --kv-heads", "plan: --ffn-hidden"))])
        self.assertIn("model: d_model 64, 2 layers, 4 heads / 2 KV, ctx 64, vocabulary 700", self.header(out_best))

    def test_init_from_a_training_state_counts_its_own_steps(self):
        """F251 (with F222): after --init-from of a training state the budget line counts the steps and tokens of
        the new run, not the checkpoint's, and a plan WARNING says so when its first step is already in the
        decay of the continued schedule."""
        out_best, out_state = self.init_runs()
        head = self.header(out_state)
        self.assertRegex("\n".join(head), r"(?m)^budget: 2 steps \(9-10\) x 512 tokens \(8 x 1 x 64\) = 1,024 tokens = "
                                          r"[\d.]+ tokens/parameter, [\d.]+ epochs$")
        self.assertIn("schedule: WSD: warmup 1-1 (linear), stable to 8 at", "\n".join(head))
        warned = [line for line in head if line.startswith("plan: WARNING: ") and "decay" in line]
        self.assertEqual(len(warned), 1, head)
        self.assertIn("step 9", warned[0])
        self.assertIn("--steps", warned[0])
        self.assertRegex(out_state, r"(?m)^RUN \S+ completed at \S+ \| step 10/10 \| ")
        # from a best file the schedule starts at step 0: the whole run, no range, no warning
        head = self.header(out_best)
        self.assertRegex("\n".join(head), r"(?m)^budget: 2 steps x 512 tokens \(8 x 1 x 64\) = 1,024 tokens = ")
        self.assertFalse([line for line in head if line.startswith("plan: WARNING: ") and "decay" in line], head)
        # a resumed run keeps the whole run's count
        out_flags, _ = self.notes_runs()
        self.assertRegex(out_flags, r"(?m)^budget: 8 steps x 512 tokens \(8 x 1 x 64\) = 4,096 tokens = ")

    def test_z_loss_given_on_resume_is_noted_whatever_its_value(self):
        """F257: --z-loss has no default of its own to compare with, so a resumed run that is given the value a
        new run would use (1e-4) is told that its own setting stays; without the option nothing is said."""
        self.assertIsNone(tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp]).z_loss)
        out_flags, out_plain = self.notes_runs()
        self.assertIn("Note: --z-loss ignored on --resume; this run keeps z_loss 0.\n", out_flags)
        self.assertNotIn("--z-loss", out_plain)
        self.assertEqual(self.load("N.pt")["train_config"]["z_loss"], 0.0)
        _, tail, _, _ = self.console_runs()  # a new run without the option
        self.assertIn('"z_loss": 0.0001', tail)

    def test_resume_without_a_stored_eval_tokens_keeps_the_old_windows(self):
        """F228: a checkpoint from before eval_tokens was stored used 65,536 validation tokens per source; a
        resume with --eval-tokens keeps those windows and says that the option is ignored."""
        self.notes_runs()
        obj = self.load("N.pt")
        del obj["train_config"]["eval_tokens"]
        torch.save(obj, os.path.join(self.runs, "N228.pt"))
        code, out = base.run(["tiny_gpt.py", "--name", "N228", "--out-dir", self.runs, "--log-file", self.log,
                              "--device", "cpu", "--resume", "--eval-tokens", "1024", "--no-probes",
                              "--tariff", "25"])
        self.assertEqual(code, 0, out)
        self.assertIn("Note: --eval-tokens ignored on --resume; this run keeps 65,536 (its validation windows).", out)
        _, arrays = data_prep.open_token_files(self.ds, verify=False)

        def windows(tokens):
            return sum(w.shape[0] for w in tiny_gpt.EvalSet(arrays["val"], 64, tokens).windows.values())

        self.assertNotEqual(windows(65_536), windows(1024), "the toy sources must be long enough to tell")
        self.assertIn(f"| validation: {windows(65_536)} fixed windows of 64 tokens from 3 sources", out)

    def test_probe_file_hash_short_in_the_settings_and_whole_in_the_report(self):
        """F254 (guard: same values before and after): the probe file is hashed once; the settings keep the
        first 16 characters, the probe report of an evaluation the whole digest."""
        self.console_runs()
        state = self.load("C.pt")
        digest = tiny_gpt.file_sha256(self.probes)
        self.assertEqual(state["train_config"]["probes_sha"], digest[:16])
        self.assertEqual(state["metrics"]["last_eval"]["probes"]["file_sha256"], digest)
        self.notes_runs()
        self.assertIsNone(self.load("N.pt")["train_config"]["probes_sha"])  # --no-probes

    def test_init_from_says_when_the_rope_base_is_below_the_rule(self):
        """F341: --init-from keeps the checkpoint's RoPE base; when that is below the rule for the context of
        the new run, a plan line says so."""
        out_best, out_state = self.init_runs()
        for text in (out_best, out_state):
            plan = [line for line in self.header(text) if line.startswith("plan: RoPE base ")]
            self.assertEqual(len(plan), 1, text)
            self.assertIn("stays at the checkpoint's 500", plan[0])
            self.assertIn("ctx 64 (10,000)", plan[0])
        # a base at the rule (the run C of the console tests): no line
        self.console_runs()
        code, out = base.run(self.common + ["--name", "I3", "--init-from", os.path.join(self.runs, "C_best.pt"),
                                            "--steps", "2", "--eval-every", "2", "--no-probes"])
        self.assertEqual(code, 0, out)
        self.assertIn("plan: architecture taken from --init-from ", out)
        self.assertNotIn("RoPE base stays", out)


if __name__ == "__main__":
    unittest.main()
