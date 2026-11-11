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
import time
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
# tiny_gpt.main(args) in a child process whose transformer block fails at the places of the comma list: 'eval' (the
# first evaluation), 'oom<n>' (out of memory in the n-th batch drawn) or 'crash<n>' (another error there). The
# failing forward holds an activation of its own; whether that tensor is still alive is printed when the allocator
# is emptied for the retry and when the crash state is saved
FRAMES_RUN = r'''
import sys
import weakref

sys.path.insert(0, sys.argv[1])
import tiny_gpt  # noqa: E402

plan = [item for item in sys.argv[2].split(",") if item]
seen = {"draws": 0, "refs": []}
real_forward, real_next = tiny_gpt.TransformerBlock.forward, tiny_gpt.TokenBatcher.next
real_free, real_save = tiny_gpt.free_device_memory, tiny_gpt.atomic_save


def next_batch(self):
    seen["draws"] += 1
    return real_next(self)


def forward(self, x, rope, cache=None, mask=None):
    for where in ("eval",) if not self.training else (f"oom{seen['draws']}", f"crash{seen['draws']}"):
        if where in plan:
            plan.remove(where)
            activation = x * 2.0  # referenced by this frame alone
            seen["refs"].append((where, weakref.ref(activation)))
            raise RuntimeError("the block failed (simulated)" if where.startswith("crash")
                               else "CUDA out of memory (simulated)")
    return real_forward(self, x, rope, cache, mask)


def report():
    for where, ref in seen["refs"]:
        print(f"FAULT CHECK: {where}: the activation of the failed forward is "
              f"{'freed' if ref() is None else 'alive'}")
    seen["refs"].clear()


def free(device):
    report()
    return real_free(device)


def save(obj, path):
    if path.endswith("_crash.pt"):
        report()
    return real_save(obj, path)


tiny_gpt.device_memory_budget = lambda device: (8 * 2 ** 30, "8.0 GiB (test)")
tiny_gpt.TransformerBlock.forward = forward
tiny_gpt.TokenBatcher.next = next_batch
tiny_gpt.free_device_memory = free
tiny_gpt.atomic_save = save
tiny_gpt.main(sys.argv[3:])
'''
# tiny_gpt.main(args) in a child process whose torch.compile compiles nothing: it prints the cache folder it would
# use, and whether torch's compiler had been imported by then (the import is what writes the variable)
COMPILE_RUN = r'''
import os
import sys

import torch

sys.path.insert(0, sys.argv[1])
import tiny_gpt  # noqa: E402


def compile_stub(module, **kwargs):
    print(f"COMPILE CACHE dynamo={'torch._dynamo' in sys.modules} {os.environ.get('TORCHINDUCTOR_CACHE_DIR')}")
    return module


tiny_gpt.device_memory_budget = lambda device: (8 * 2 ** 30, "8.0 GiB (test)")
torch.compile = compile_stub
tiny_gpt.main(sys.argv[2:])
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
        the checkpoint's architecture (and a cosine schedule without warm-up, which decays from step 1), and
        from its training state (step 8) for two more steps."""
        if "init" not in self.cache:
            self.notes_runs()
            short = self.common + ["--eval-every", "2", "--no-probes"]
            code, out_best = base.run(short + ["--name", "I1", "--init-from", os.path.join(self.runs, "N_best.pt"),
                                               "--steps", "2", "--d-model", "128", "--tie-embeddings", "off",
                                               "--rope-base", "2000", "--schedule", "cosine",
                                               "--warmup-steps", "0"])
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
        # from a best file the schedule starts at step 0: the whole run, no range, and no warning although this
        # cosine schedule without warm-up is in its decay from the first step (nothing is continued)
        head = self.header(out_best)
        self.assertRegex("\n".join(head), r"(?m)^schedule: cosine: ")
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

    # ---------------- out-of-memory retries ----------------
    def test_failed_step_is_released_before_the_retry(self):
        """F236: the handlers that retry after an out-of-memory error (evaluation, first step, later step) and
        the crash path clear the frames of the failed call first; the exception's traceback used to keep the
        activations of the failed forward pass alive while the allocator was emptied."""
        self.pipeline()
        script = os.path.join(self.tmp, "frames_run.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(FRAMES_RUN)
        args = list(self.common[1:]) + ["--steps", "3", "--eval-every", "3", "--no-probes"]
        # the baseline evaluation, step 1 (the first batch drawn) and step 2 (the third: step 1 is drawn twice)
        code, out = base.run([script, HERE, "eval,oom1,oom3"] + args + ["--name", "FR"])
        self.assertEqual(code, 0, out)
        for line in ("Out of memory during evaluation; retrying with ",
                     "Out of memory at the first step; retrying with the loss computed ",
                     "Out of memory at step 2; retrying the same step (1/2)."):
            self.assertIn(line, out)
        checks = [line for line in out.splitlines() if line.startswith("FAULT CHECK: ")]
        self.assertEqual(checks, [f"FAULT CHECK: {where}: the activation of the failed forward is freed"
                                  for where in ("eval", "oom1", "oom3")], out)
        self.assertRegex(out, r"(?m)^RUN \S+ completed at \S+ \| step 3/3 \| ")
        # an error that ends the run: released before the crash state is written, and the traceback is whole
        code, out_c = base.run([script, HERE, "crash2"] + args + ["--name", "FC"])
        self.assertNotEqual(code, 0)
        self.assertIn("FAULT CHECK: crash2: the activation of the failed forward is freed\n", out_c)
        self.assertLess(out_c.index("FAULT CHECK: crash2"), out_c.index("FC_crash.pt for inspection"))
        self.assertIn("Traceback (most recent call last):", out_c)
        self.assertRegex(out_c, r'File ".*frames_run\.py", line \d+, in forward')
        self.assertRegex(out_c, r'File ".*tiny_gpt\.py", line \d+, in _train')
        self.assertIn("RuntimeError: the block failed (simulated)", out_c)
        self.assertRegex(out_c, r"(?m)^RUN \S+ failed at \S+ \| step 2/3 \| ")

    # ---------------- mixture of experts: the balance figures across a pause ----------------
    def test_moe_balance_figures_survive_a_pause_inside_an_interval(self):
        """F252: the sums behind moe_lb, moe_z, moe_load and dead_experts are saved with the training state, so
        the first metrics row after a pause between two evaluations covers the whole interval, as the training
        loss does; a checkpoint without the keys resumes as before."""
        self.pipeline()
        args = self.common + ["--experts", "4", "--steps", "16", "--eval-every", "8", "--no-probes"]
        code, out = base.run(args + ["--name", "MU"])
        self.assertEqual(code, 0, out)
        code, out = base.run(args + ["--name", "MS", "--stop-after-steps", "12"])
        self.assertEqual(code, 0, out)
        interval = self.load("MS.pt")["interval"]
        self.assertEqual(sorted(interval), ["counts", "lb_sum", "loss_sum", "micro", "z_sum"])
        self.assertEqual(interval["micro"], 4)
        # per MoE block: 4 steps of 512 tokens, 2 experts a token
        self.assertEqual({sum(row) for row in interval["counts"]}, {4 * 512 * 2})
        self.assertEqual({len(row) for row in interval["counts"]}, {4})
        code, out = base.run(["tiny_gpt.py"] + self.resume_args("MS"))
        self.assertEqual(code, 0, out)
        whole, paused = self.rows("MU")[-1], self.rows("MS")[-1]
        self.assertEqual((whole["step"], paused["step"]), (16, 16))
        self.assertEqual(paused["train_loss"], whole["train_loss"])
        for key in ("moe_lb", "moe_z", "moe_load_max"):
            self.assertAlmostEqual(paused[key], whole[key], delta=1e-6, msg=key)
        self.assertEqual(paused["moe_load"], whole["moe_load"])
        self.assertEqual(paused["dead_experts"], whole["dead_experts"])
        # a dense run keeps the two keys it had
        self.console_runs()
        self.assertEqual(sorted(self.load("C.pt")["interval"]), ["loss_sum", "micro"])

    # ---------------- the experiments table ----------------
    def test_a_broken_experiments_table_does_not_end_the_run(self):
        """F354: record_experiment is called when a run ends, before its last line. A table it cannot read (saved
        in another encoding) costs the row and one note, not a traceback; a row with a stray cell is carried
        through the one-time rewrite of the header, which then leaves one backup copy and no .tmp file."""
        import csv
        import glob

        def record(label, data):
            out = os.path.join(self.tmp, "exp_" + label)
            os.makedirs(os.path.join(out, "results"))
            path = os.path.join(out, "results", "experiments.csv")
            with open(path, "wb") as handle:
                handle.write(data)
            said = []
            for k in range(2):
                with redirect_stdout(io.StringIO()) as buffer:
                    tiny_gpt.record_experiment(out, {"event": "training run", "name": f"r{k}"})
                said.append(buffer.getvalue())
            self.assertFalse(os.path.exists(path + ".tmp"))
            with open(path, "rb") as handle:
                return path, said, handle.read()

        # not UTF-8 (saved from a spreadsheet program): untouched, both rows refused with a note
        data = "time,event,name\n2026,training run,caf\xa3\n".encode("cp1252")
        path, said, now = record("cp1252", data)
        self.assertEqual(now, data)
        self.assertEqual(glob.glob(path + ".before-columns-*"), [])
        for k, text in enumerate(said):
            self.assertIn(f"{path} could not be read or written (UnicodeDecodeError: ", text)
            self.assertIn(f"this row was not recorded: {{'event': 'training run', 'name': 'r{k}'}}", text)
        # an old header and a row with one cell too many: rewritten once, the stray cell stays in the backup
        data = b"time,event\n2026,training run,STRAY\n"
        path, said, now = record("extracell", data)
        self.assertEqual(said, ["", ""])
        backups = glob.glob(path + ".before-columns-*")
        self.assertEqual(len(backups), 1)
        with open(backups[0], "rb") as handle:
            self.assertEqual(handle.read(), data)
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], tiny_gpt.EXPERIMENT_COLUMNS)
        self.assertEqual({len(row) for row in rows}, {len(tiny_gpt.EXPERIMENT_COLUMNS)})
        at = rows[0].index("name")
        self.assertEqual([(row[1], row[at]) for row in rows[1:]],
                         [("training run", ""), ("training run", "r0"), ("training run", "r1")])
        self.assertNotIn(b"STRAY", now)

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_published_experiments_table_keeps_its_byte_order_mark(self):
        """F350: publishing rewrites results/experiments.csv in the repository; the rewritten file starts with
        the UTF-8 byte-order mark record_experiment creates it with (Excel reads the pound sign by it), once."""
        import csv
        import subprocess
        repo = os.path.join(self.tmp, "repo350")
        os.makedirs(repo)
        for cmd in (["init", "-q"], ["config", "user.name", "test"], ["config", "user.email", "test@example.invalid"],
                    ["config", "commit.gpgsign", "false"], ["config", "core.autocrlf", "false"]):
            done = subprocess.run(["git", "-C", repo] + cmd, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
        path = os.path.join(repo, "results", "experiments.csv")
        with unittest.mock.patch.object(tiny_gpt, "HERE", repo):  # a run of this folder: its table is published
            for k in range(2):
                with redirect_stdout(io.StringIO()) as buffer:
                    tiny_gpt.record_experiment(repo, {"event": "training run", "name": f"r{k}", "notes": "£1"})
                self.assertIn("experiments.csv on GitHub (results/experiments.csv): ", buffer.getvalue())
                with open(path, "rb") as handle:
                    data = handle.read()
                self.assertTrue(data.startswith(b"\xef\xbb\xbf"), data[:20])
                self.assertEqual(data.count(b"\xef\xbb\xbf"), 1)
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(row["name"], row["notes"]) for row in rows], [("r0", "£1"), ("r1", "£1")])
        shown = subprocess.run(["git", "-C", repo, "show", "HEAD:results/experiments.csv"], capture_output=True)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertEqual(shown.stdout, data, "the committed table is the file")

    # ---------------- torch.compile ----------------
    def test_compile_cache_stays_beside_the_script(self):
        """F326: a training run compiles with torch.compile's cache in .compile_cache beside the script (the
        default is the temp folder, which gets emptied) unless the user set TORCHINDUCTOR_CACHE_DIR. torch sets
        the variable itself, to the temp folder, when its compiler is first imported, and building an optimizer
        imports it: a run builds its optimizer before it compiles, so the value torch wrote must not win."""
        beside = os.path.join(tiny_gpt.HERE, ".compile_cache")
        # a whole run in a child process (torch writes the variable once per process): what torch.compile finds
        self.pipeline()
        script = os.path.join(self.tmp, "compile_run.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(COMPILE_RUN)
        args = list(self.common[1:]) + ["--steps", "2", "--eval-every", "2", "--no-probes", "--compile", "on"]
        given = os.path.join(self.tmp, "given_cache")
        for name, value, expected in (("CC", None, beside), ("CD", given, given)):
            with unittest.mock.patch.dict(os.environ):
                os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
                if value:
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = value
                code, out = base.run([script, HERE] + args + ["--name", name])
            self.assertEqual(code, 0, out)
            self.assertEqual([line for line in out.splitlines() if line.startswith("COMPILE CACHE ")],
                             ["COMPILE CACHE dynamo=True " + expected], out)
        # maybe_compile alone: nothing is set unless it compiles, and it puts the folder in place of torch's
        cpu, model = torch.device("cpu"), base.TestFixes.small_model(self)
        stub = unittest.mock.patch.object(tiny_gpt.torch, "compile", lambda module, **kwargs: module)
        with unittest.mock.patch.dict(os.environ), stub, \
                unittest.mock.patch.object(tiny_gpt, "COMPILE_CACHE_GIVEN", False):
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
            self.assertIs(tiny_gpt.maybe_compile(model, "off", cpu), model)
            self.assertNotIn("TORCHINDUCTOR_CACHE_DIR", os.environ, "nothing is compiled: nothing is set")
            self.assertIs(tiny_gpt.maybe_compile(model, "on", cpu, micro=2), model)
            self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], beside)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = "the folder torch chose"
            tiny_gpt.maybe_compile(model, "on", cpu, micro=2)
            self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], beside, "torch's own choice does not win")
        with unittest.mock.patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": "elsewhere"}), stub, \
                unittest.mock.patch.object(tiny_gpt, "COMPILE_CACHE_GIVEN", True):
            tiny_gpt.maybe_compile(model, "on", cpu, micro=2)
            self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], "elsewhere", "a setting of the user's wins")
        self.assertFalse(os.path.exists(os.path.join(HERE, ".compile_cache")), "this test compiles nothing")

    def test_compile_warm_up_has_the_layout_of_a_training_batch(self):
        """F328: the warm-up call of maybe_compile gets an input with the shape, strides and type of the batches
        TokenBatcher.next() returns (the first ctx columns of a (batch, ctx + 1) tensor). torch.compile guards
        on the strides, so a contiguous (batch, ctx) input made the first step compile the model again."""
        import functools

        import numpy as np
        from torch._dynamo.utils import counters
        cpu, model = torch.device("cpu"), base.TestFixes.small_model(self).train()
        batcher = tiny_gpt.TokenBatcher({"a": np.arange(5000, dtype=np.uint16) % 50}, {"a": 1.0}, model.ctx, 2, cpu, 1)
        seen = []

        def recording(module, **kwargs):
            def call(x, **kw):
                seen.append(x)
                return module(x, **kw)
            return call

        with unittest.mock.patch.dict(os.environ), unittest.mock.patch.object(tiny_gpt.torch, "compile", recording):
            tiny_gpt.maybe_compile(model, "on", cpu, micro=2)
        x, _ = batcher.next()
        self.assertEqual(len(seen), 1)
        self.assertEqual((seen[0].shape, seen[0].stride(), seen[0].dtype), (x.shape, x.stride(), x.dtype))
        self.assertEqual(x.stride(), (model.ctx + 1, 1))
        # the real thing, without a code generator: one graph for the warm-up and the first batches (torch makes
        # its cache folder for this backend too: a folder of this test, not the one beside the script)
        eager = functools.partial(torch.compile, backend="eager")
        given = {"TORCHINDUCTOR_CACHE_DIR": os.path.join(self.tmp, "eager_cache")}
        torch._dynamo.reset()
        counters.clear()
        try:
            with unittest.mock.patch.dict(os.environ, given),                     unittest.mock.patch.object(tiny_gpt, "COMPILE_CACHE_GIVEN", True),                     unittest.mock.patch.object(tiny_gpt.torch, "compile", eager):
                with redirect_stdout(io.StringIO()) as said:
                    compiled = tiny_gpt.maybe_compile(model, "on", cpu, micro=2)
            self.assertIsNot(compiled, model, said.getvalue())
            for _ in range(3):
                x, _ = batcher.next()
                with torch.autocast("cpu", dtype=torch.float32, enabled=False):  # as the training loop calls it
                    compiled(x).float().sum().backward()
                model.zero_grad(set_to_none=True)
            self.assertEqual(counters["stats"]["unique_graphs"], 1)
        finally:
            torch._dynamo.reset()
        self.assertFalse(os.path.exists(os.path.join(HERE, ".compile_cache")), "no cache beside the script")

    # ---------------- the log file and the command line ----------------
    def test_abbreviated_log_file_option_reaches_the_logger(self):
        """F332 (with F216): the log is opened before the arguments are parsed, from a look at --log-file alone.
        That look now abbreviates the way the full parser does, so '--log-f X' no longer logs to the default file
        while the parser takes X."""
        default = os.path.join(tiny_gpt.HERE, "log.txt")
        self.assertEqual(tiny_gpt._log_path_from(["--log-fi", "x.txt"]), "x.txt")
        self.assertEqual(tiny_gpt._log_path_from(["--name", "a", "--log-f", "X", "--resume"]), "X")
        self.assertEqual(tiny_gpt._log_path_from(["--log-fi=Y"]), "Y")
        self.assertEqual(tiny_gpt._log_path_from(["--log-every", "5"]), default)
        self.assertEqual(tiny_gpt._log_path_from(["--log-file", "a.txt", "--steps", "3"]), "a.txt")
        self.assertEqual(tiny_gpt._log_path_from(["--log-file=b.txt"]), "b.txt")
        self.assertEqual(tiny_gpt._log_path_from([]), default)
        self.assertEqual(tiny_gpt._log_path_from(["--plan", "--dataset", "d"]), default)
        # no value: the default log, where the full parser then reports the error
        with redirect_stdout(io.StringIO()) as out, unittest.mock.patch("sys.stderr", new=io.StringIO()) as err:
            self.assertEqual(tiny_gpt._log_path_from(["--name", "a", "--log-file"]), default)
        self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))
        # the two parsers agree
        for argv in (["--log-fi", "x.txt"], ["--log-f=Y"], ["--log-file", "a", "--log-file", "b"], []):
            args = tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp] + argv)
            self.assertEqual(tiny_gpt._log_path_from(argv), args.log_file, argv)

    def test_help_is_not_written_to_the_log(self):
        """F323: --help prints the options and exits without opening the log file; an argument error is still
        logged."""
        from contextlib import redirect_stderr
        for i, argv in enumerate((["--help"], ["-h"], ["--name", "x", "--help"])):
            log = os.path.join(self.tmp, f"help{i}.log")
            with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stop:
                tiny_gpt.main(argv + ["--log-file", log])
            self.assertEqual(stop.exception.code, 0)
            self.assertIn("--dashboard-push-hours", out.getvalue())
            self.assertNotIn("--- tinyGPT ", out.getvalue())
            self.assertFalse(os.path.exists(log), argv)
            self.assertNotIn("handle", tiny_gpt._LOG_STATE)
        log = os.path.join(self.tmp, "error.log")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
            tiny_gpt.main(["--no-such-option", "--log-file", log])
        self.assertEqual(stop.exception.code, 1)
        with open(log, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("--- tinyGPT ", text)
        self.assertIn("unrecognized arguments: --no-such-option", text)

    # ---------------- another training run beside this one ----------------
    def test_a_live_lock_of_another_run_is_a_warning(self):
        """F287 (with F312): a training run, and --plan with --time-budget-hours (it times training steps), warn
        when another run's lock is alive in the output folder or beside the script. A warning, never a refusal,
        and from the locks alone (gpu_busy(..., smi=False)): nvidia-smi's figures are also high on desktop load.
        With --device cpu nothing is asked."""
        import glob
        import signal
        out_dir = os.path.join(self.tmp, "busy287")
        os.makedirs(out_dir)
        warning = "WARNING: " + tiny_gpt.GPU_BUSY_MESSAGE + "\n"
        signals = [getattr(signal, name) for name in tiny_gpt.PAUSE_SIGNALS if hasattr(signal, name)]
        handlers = {number: signal.getsignal(number) for number in signals}  # cmd_train sets them for its run

        def train(argv, busy):
            args = tiny_gpt.parse_args(["--name", "b", "--out-dir", out_dir] + argv)
            with unittest.mock.patch.object(tiny_gpt, "gpu_busy", return_value=busy) as asked, \
                    unittest.mock.patch.object(tiny_gpt, "_train", return_value=None) as trained, \
                    redirect_stdout(io.StringIO()) as said:
                tiny_gpt.cmd_train(args)
            self.assertEqual(trained.call_count, 1, "a warning: the run goes on")
            return asked, said.getvalue()

        try:
            asked, said = train(["--device", "cuda"], True)
            self.assertEqual(said, warning)
            asked.assert_called_once_with(out_dir, tiny_gpt.HERE, smi=False)
            asked, said = train([], False)  # --device auto
            self.assertEqual((asked.call_count, said), (1, ""))
            asked, said = train(["--device", "cpu"], True)
            self.assertEqual((asked.call_count, said), (0, ""))
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
        self.assertEqual(glob.glob(os.path.join(out_dir, "*.train.lock")), [], "the run's own lock is released")

        # --plan: only when it times steps (--time-budget-hours), before the timing starts
        dataset = os.path.join(self.tmp, "ds287")
        os.makedirs(dataset)
        with open(os.path.join(dataset, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"format": data_prep.FORMAT, "format_version": 1, "fingerprint": "abc"}, f)

        def plan(argv, busy):
            args = tiny_gpt.parse_args(["--plan", "--dataset", dataset, "--out-dir", out_dir] + argv)
            with unittest.mock.patch.object(tiny_gpt, "gpu_busy", return_value=busy) as asked, \
                    unittest.mock.patch.object(tiny_gpt, "plan_with_time_budget",
                                               side_effect=KeyboardInterrupt("the planner starts")), \
                    redirect_stdout(io.StringIO()) as said, self.assertRaises(KeyboardInterrupt):
                tiny_gpt.cmd_plan(args)
            return asked, said.getvalue()

        asked, said = plan(["--time-budget-hours", "1"], True)
        self.assertEqual(said, warning)
        asked.assert_called_once_with(out_dir, tiny_gpt.HERE, smi=False)
        self.assertEqual(plan(["--time-budget-hours", "1"], False)[1], "")
        asked, said = plan([], True)
        self.assertEqual((asked.call_count, said), (0, ""))
        asked, said = plan(["--time-budget-hours", "1", "--device", "cpu"], True)
        self.assertEqual((asked.call_count, said), (0, ""))

    def test_gpu_busy_can_leave_nvidia_smi_out(self):
        """F287: gpu_busy(..., smi=False) answers from the lock files alone; the default still asks nvidia-smi."""
        folder = os.path.join(self.tmp, "smi287")
        os.makedirs(folder)
        loaded = unittest.mock.Mock(stdout="3000, 50\n")
        with unittest.mock.patch.object(tiny_gpt.subprocess, "run", return_value=loaded) as smi:
            self.assertTrue(tiny_gpt.gpu_busy(folder))
            self.assertEqual(smi.call_count, 1)
            self.assertFalse(tiny_gpt.gpu_busy(folder, smi=False))
            self.assertEqual(smi.call_count, 1, "not asked")
            with open(os.path.join(folder, "other.train.lock"), "w", encoding="utf-8") as f:
                json.dump({"pid": os.getppid(), "started": None}, f)  # a live process that is not this one
            self.assertTrue(tiny_gpt.gpu_busy(folder, smi=False))
            self.assertEqual(smi.call_count, 1)

    # ---------------- dashboard pushes ----------------
    def test_dashboard_is_pushed_every_twelve_hours_by_default(self):
        """F438: every pushed dashboard stays in the repository's history, so the default --dashboard-push-hours
        is 12 (it was 3); the option still takes any value."""
        args = tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp])
        self.assertEqual(args.dashboard_push_hours, 12.0)
        self.assertEqual(tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp, "--dashboard-push-hours", "3"])
                         .dashboard_push_hours, 3.0)
        with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            tiny_gpt.parse_args(["--help"])
        text = " ".join(out.getvalue().split())
        self.assertIn("push the dashboard to GitHub at most this often (0 = every time; default 12, ", text)
        launches = []
        state = {"proc": None, "pushed": time.time() - 4 * 3600}  # last pushed four hours ago
        with unittest.mock.patch.object(tiny_gpt.subprocess, "Popen", side_effect=lambda cmd, **kw: launches.append(cmd)):
            tiny_gpt.launch_dashboard(os.path.join(self.tmp, "X_best.pt"), args, state)
        self.assertEqual(len(launches), 1)
        self.assertIn("--no-push", launches[0])


if __name__ == "__main__":
    unittest.main()
