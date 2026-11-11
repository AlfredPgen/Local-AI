"""Staging tests of the 2026-11-10 audit, package A2 (tiny_gpt.py: planner, arguments, export/benchmark/generate
commands, signals). Target class in test_tiny_gpt.py: TestFixes.

    python -m unittest test_a2_a2
"""
import contextlib
import csv
import dataclasses
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
import unittest.mock

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiny_gpt  # noqa: E402

CPU = torch.device("cpu")
GIB = 2 ** 30


def toy_manifest(tokens=2_000_000, vocab=512, sources=("md", "wiki")):
    """A dataset manifest as plan_run reads it: `tokens` training tokens per source."""
    return {"tokenizer": {"vocab_size": vocab},
            "sources": {name: {"train": {"tokens": tokens}, "doc_tokens_percentiles": {"50": 100}}
                        for name in sources}}


def budget_of(gib):
    """device_memory_budget patched to a fixed number of GiB."""
    return unittest.mock.patch.object(tiny_gpt, "device_memory_budget",
                                      return_value=(int(gib * GIB), f"{gib:g} GiB (test)"))


def glob_tmp(folder):
    """Temporary files of an interrupted write (<name>.tmp-<pid>) in `folder`."""
    return sorted(name for name in os.listdir(folder) if ".tmp-" in name)


class Audit2TinyGptB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2_a2_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_plan_table_sizes_memory_for_the_optimizer(self):
        """F344 (with F278, F342): the 'batch that fits' column of --plan is the planner's own memory plan for the
        run's optimizer on this device, and the lr column follows the batch that plan reaches."""
        with budget_of(64):
            rows = tiny_gpt.plan_table(16384, CPU, None, 20.0, 4.0)
        big = rows[-1][2]
        # between what Muon and what AdamW need at batch 1 with every memory saving on
        cheapest = (big, 1, False, True, tiny_gpt.LOSS_CHUNK_TOKENS)
        budget = tiny_gpt.estimate_train_bytes(*cheapest, "muon") + 2 ** 20
        self.assertGreater(tiny_gpt.estimate_train_bytes(*cheapest, "adamw"), budget)
        with budget_of(budget / GIB):
            adamw = tiny_gpt.plan_table(16384, CPU, None, 20.0, 4.0)
            muon = tiny_gpt.plan_table(16384, CPU, None, 20.0, 4.0, optimizer="muon")
        self.assertEqual(adamw[-1][5], "does not fit")
        self.assertRegex(muon[-1][5], r"^\d+x\d+$")
        for unique, planned, c, n, tps, fits, lr in muon:
            micro, accum = (int(v) for v in fits.split("x"))
            self.assertLessEqual(micro * accum * c.ctx, tps)
            self.assertAlmostEqual(lr, tiny_gpt.heuristic_lr(n) * tiny_gpt.lr_batch_factor(n, micro * accum * c.ctx))
        self.assertFalse(hasattr(tiny_gpt, "choose_micro_batch"), "the one-caller wrapper is gone")
        # --plan passes the run's optimizer
        seen = {}
        real = tiny_gpt.plan_table

        def spy(*a, **k):
            seen["args"], seen["kwargs"] = a, k
            return real(*a, **k)

        manifest = toy_manifest()
        args = tiny_gpt.parse_args(["--name", "x", "--plan", "--optimizer", "muon", "--device", "cpu",
                                    "--out-dir", self.tmp])
        with unittest.mock.patch.object(tiny_gpt, "plan_table", spy), budget_of(64), \
                unittest.mock.patch.object(tiny_gpt, "ensure_dataset", return_value=self.tmp), \
                unittest.mock.patch.object(tiny_gpt.data_prep, "read_manifest", return_value=manifest), \
                contextlib.redirect_stdout(io.StringIO()):
            tiny_gpt.cmd_plan(args)
        self.assertIn("muon", list(seen["args"]) + list(seen["kwargs"].values()))

    # ---------------- compact plan lines for the run header ----------------
    BIG = {"tokenizer": {"vocab_size": 16384}, "sources": {
        "wiki": {"train": {"tokens": 1_430_000_000}, "doc_tokens_percentiles": {"50": 600}},
        "pmc": {"train": {"tokens": 1_105_000_000}, "doc_tokens_percentiles": {"50": 10_072}},
        "books": {"train": {"tokens": 20_000_000}, "doc_tokens_percentiles": {"50": 206_767}}}}

    def _plan(self, argv, manifest=None, gib=64, **kw):
        args = tiny_gpt.parse_args(["--name", "x"] + argv)
        with budget_of(gib):
            return tiny_gpt.plan_run(args, manifest or self.BIG, CPU, None, **kw)

    @staticmethod
    def _fake_measure(cfg, micro, accum, device, amp_dtype, attn_pref="auto", warmup=2, timed=3, checkpointing=False,
                      loss_chunk=0, settings=None):
        """A speed test that needs tokens/step x N x 1e-13 seconds per step and 1 GiB."""
        return micro * accum * cfg.ctx * tiny_gpt.count_params(cfg)[0] * 1e-13, GIB

    def _time_plan(self, argv, manifest, measure=None, gib=64, speed=(None, "no table (test)"), **kw):
        args = tiny_gpt.parse_args(["--name", "x", "--compile", "off"] + argv)
        with budget_of(gib), \
                unittest.mock.patch.object(tiny_gpt, "measure_seconds_per_step",
                                           side_effect=measure or self._fake_measure), \
                unittest.mock.patch.object(tiny_gpt, "sustained_speed", return_value=speed), \
                unittest.mock.patch.object(tiny_gpt, "have_triton", return_value=False):
            return tiny_gpt.plan_with_time_budget(args, manifest, CPU, None, **kw)

    def test_plan_lines_for_the_run_header_are_compact(self):
        """F336: plan_run(..., verbose=False) and plan_with_time_budget(..., verbose=False) leave out what the run
        header prints itself (per-source tokens, batch, weights, equal source shares) and fold the context line into
        the model line; the default (verbose=True, what --plan prints) is unchanged."""
        shape = ["--d-model", "1024", "--layers", "18", "--optimizer", "muon"]
        _, st, full = self._plan(shape)
        _, st_v, explicit = self._plan(shape, verbose=True)
        self.assertEqual(full, explicit)
        _, st_c, lines = self._plan(shape, verbose=False)
        self.assertEqual(st, st_c, "the settings do not depend on how the plan is worded")
        for prefix in ("batch:", "ratios:", "weights:", "ctx ", "sources:"):
            self.assertTrue(any(ln.startswith(prefix) for ln in full), prefix)
            self.assertFalse(any(ln.startswith(prefix) for ln in lines), (prefix, lines))
        self.assertTrue(lines[0].startswith("unique training tokens U = 2,555,000,000; vocabulary 16,384; "), lines[0])
        self.assertNotIn("wiki", lines[0])
        self.assertTrue(lines[1].startswith("token budget D = "), lines[1])
        model = [ln for ln in lines if ln.startswith("model: ")]
        self.assertEqual(len(model), 1)
        self.assertIn("ctx 1024 (default for d_model 1024; attention 15% of the training FLOPs per token), RoPE base",
                      model[0])
        for prefix in ("learning rate ", "decay timescale ", "memory: "):  # kept, word for word
            self.assertEqual([ln for ln in lines if ln.startswith(prefix)],
                             [ln for ln in full if ln.startswith(prefix)], prefix)
        self.assertEqual([ln for ln in lines if ln.startswith("note:")], [ln for ln in full if ln.startswith("note:")])
        self.assertTrue(any(ln.startswith("note: ctx 1024 is shorter than the median document of pmc") for ln in lines))
        # unequal source shares are news: the line stays, and so does every warning
        _, _, alpha_full = self._plan(["--mix-alpha", "0.8"])
        _, _, alpha = self._plan(["--mix-alpha", "0.8"], verbose=False)
        self.assertEqual([ln for ln in alpha if ln.startswith("sources: ")],
                         [ln for ln in alpha_full if ln.startswith("sources: ")])
        self.assertEqual(len([ln for ln in alpha if ln.startswith("sources: ")]), 1)
        self.assertEqual([ln for ln in alpha if ln.startswith("WARNING")],
                         [ln for ln in alpha_full if ln.startswith("WARNING")])
        self.assertTrue(any("books would be seen" in ln for ln in alpha))
        # a line appended before the main block (the --batch-size warning) is kept as well
        _, _, tight = self._plan(["--d-model", "1024", "--batch-size", "64"], gib=5.6, verbose=False)
        self.assertTrue(tight[0].startswith("WARNING: batch 64 x ctx 1024 may not fit"), tight[0])
        # a mixture of experts keeps its ratios line (per active and per total parameter: the header has neither)
        _, _, moe = self._plan(["--experts", "8"], verbose=False)
        self.assertEqual(len([ln for ln in moe if ln.startswith("ratios: ")]), 1)
        self.assertIn("per total parameter", [ln for ln in moe if ln.startswith("ratios: ")][0])
        # through the time budget: both returns carry the flag
        one = {"tokenizer": {"vocab_size": 16384},
               "sources": {"md": {"train": {"tokens": 1_000_000_000}, "doc_tokens_percentiles": {"90": 1000}}}}
        _, _, no_budget = self._time_plan([], one, verbose=False)
        self.assertFalse(any(ln.startswith(("batch:", "weights:", "ctx ")) for ln in no_budget), no_budget)
        _, st_t, timed = self._time_plan(["--time-budget-hours", "1"], one, verbose=False)
        _, st_tf, timed_full = self._time_plan(["--time-budget-hours", "1"], one)
        self.assertEqual(st_t, st_tf)
        self.assertFalse(any(ln.startswith(("batch:", "weights:", "ctx ")) for ln in timed), timed)
        self.assertTrue(any(ln.startswith("batch:") for ln in timed_full))
        budget_line = [ln for ln in timed if ln.startswith("token budget D")]
        self.assertEqual(len(budget_line), 1)
        self.assertIn("largest ladder shape that satisfies the data AND the time budget", budget_line[0])
        self.assertEqual([ln for ln in timed if ln.startswith("time budget")],
                         [ln for ln in timed_full if ln.startswith("time budget")])

    def test_time_budget_shape_note_survives_a_line_before_it(self):
        """F345: the shape note of a time-budget plan is rewritten in the 'token budget D' line wherever it stands
        (a note or warning printed before the main block used to shift it away from index 1)."""
        one = {"tokenizer": {"vocab_size": 16384},
               "sources": {"md": {"train": {"tokens": 1_000_000_000}, "doc_tokens_percentiles": {"90": 1000}}}}
        _, _, lines = self._time_plan(["--time-budget-hours", "1", "--lr", "1e-3", "--lr-scale", "2"], one)
        self.assertTrue(lines[0].startswith("--lr-scale 2 has no effect"), lines[0])
        self.assertTrue(any("satisfies the data AND the time budget" in ln for ln in lines), lines)
        self.assertFalse(any("ladder defaults for the rest" in ln for ln in lines), lines)
        _, _, fixed = self._time_plan(["--time-budget-hours", "1", "--d-model", "512", "--lr", "1e-3", "--lr-scale",
                                       "2"], one)
        self.assertTrue(any("command-line values, tokens capped by the time budget" in ln for ln in fixed), fixed)

    def test_time_budget_names_the_limit_that_binds(self):
        """F339: a time budget the data does not fill is not called a cap: no 'capped ... by --time-budget-hours',
        no 'tokens capped by the time budget', and the smallest-shape note says 'on this data', not 'in time'."""
        one = {"tokenizer": {"vocab_size": 16384},
               "sources": {"md": {"train": {"tokens": 1_000_000_000}, "doc_tokens_percentiles": {"90": 1000}}}}
        cfg_n, st_n, _ = self._time_plan([], one)
        cfg, st, lines = self._time_plan(["--time-budget-hours", "1000"], one)
        text = "\n".join(lines)
        self.assertIn("tokens fit in time", text)
        self.assertNotIn("capped", text)
        self.assertNotIn("even the smallest shape", text)
        self.assertEqual((cfg, st), (cfg_n, st_n), "time does not bind: the plan the data gives, as measured")
        _, st_x, lines_x = self._time_plan(["--time-budget-hours", "1000", "--d-model", "512"], one)
        self.assertNotIn("capped", "\n".join(lines_x))
        self.assertEqual(st_x, self._time_plan(["--d-model", "512"], one)[1])
        # too little data for the smallest shape, plenty of time: the data is the reason
        tiny = toy_manifest(tokens=50_000, vocab=16384)
        _, _, lines_d = self._time_plan(["--time-budget-hours", "1000"], tiny)
        note = [ln for ln in lines_d if "even the smallest shape" in ln]
        self.assertEqual(len(note), 1, lines_d)
        self.assertIn("on this data", note[0])
        self.assertNotIn("in time", note[0])
        self.assertNotIn("capped", "\n".join(lines_d))
        # time binds: both say so, as before
        _, st_h, lines_h = self._time_plan(["--time-budget-hours", "0.01", "--tokens-per-param", "1000"], one)
        text_h = "\n".join(lines_h)
        self.assertRegex(text_h, r"capped to [\d,]+ by --time-budget-hours")
        self.assertIn("cannot reach the tokens-per-parameter target in time", text_h)
        _, _, lines_e = self._time_plan(["--time-budget-hours", "1", "--d-model", "512"], one)
        self.assertIn("command-line values, tokens capped by the time budget", "\n".join(lines_e))

    def test_steps_alone_warns_about_huge_steps(self):
        """F338: --steps alone divides the whole token budget by the steps; when that gives more than 4x the
        planner's tokens per step the plan says so and names --tokens-per-step."""
        manifest = toy_manifest(tokens=50_000_000, vocab=4096)
        cfg, st, lines = self._plan(["--steps", "5"], manifest)
        rule = tiny_gpt.tokens_per_step_for(tiny_gpt.count_params(cfg)[2], 4 * 100_000_000)
        self.assertGreater(st["tokens_per_step"], 4 * rule)
        warning = [ln for ln in lines if ln.startswith("WARNING: --steps 5")]
        self.assertEqual(len(warning), 1, lines)
        self.assertIn("--tokens-per-step", warning[0])
        self.assertIn(f"{st['tokens_per_step']:,} tokens per step", warning[0])
        self.assertIn(f"{rule:,}", warning[0])
        # the budget named with the steps, a step count near the planner's own, or a batch fixed by hand: no warning
        steps = 4 * 100_000_000 // (2 * rule)
        for argv in (["--steps", "5", "--tokens-per-step", str(rule)], ["--steps", str(steps)],
                     ["--steps", "5", "--batch-size", "8", "--grad-accum", "1"], []):
            _, st_ok, lines_ok = self._plan(argv, manifest)
            self.assertLessEqual(st_ok["tokens_per_step"], 4 * rule, argv)
            self.assertFalse(any("WARNING" in ln for ln in lines_ok), (argv, lines_ok))
        # the example of the module docstring (and so of --help) plans a sane step
        example = [ln for ln in tiny_gpt.__doc__.splitlines() if "--name bio_small" in ln]
        self.assertEqual(len(example), 1)
        argv = example[0].split("--name bio_small", 1)[1].split()
        self.assertIn("--steps", argv)
        _, st_ex, lines_ex = self._plan(argv, self.BIG)
        self.assertFalse(any("WARNING" in ln for ln in lines_ex), lines_ex)

    def test_batch_size_keeps_the_memory_fallback_of_the_shape(self):
        """F337: --batch-size without shape options still lets the planner fall back to a smaller ladder shape
        when the data-chosen one cannot fit at batch 1 with every memory saving; an explicit shape is kept (with
        its warning), and a batch that is too large for a shape that fits keeps the warning."""
        manifest = {"tokenizer": {"vocab_size": 16384}, "sources": {
            "wiki": {"train": {"tokens": 1_430_000_000}}, "pmc": {"train": {"tokens": 1_105_000_000}},
            "fineweb": {"train": {"tokens": 810_000_000}}, "pes2o": {"train": {"tokens": 441_000_000}},
            "arxiv": {"train": {"tokens": 170_000_000}}, "books": {"train": {"tokens": 20_000_000}},
            "notes": {"train": {"tokens": 24_000_000}}}}

        def plan(argv):
            args = tiny_gpt.parse_args(["--name", "x", "--optimizer", "muon"] + argv)
            with budget_of(5.6):
                return tiny_gpt.plan_run(args, manifest, CPU, torch.bfloat16)

        auto, st_auto, lines_auto = plan([])
        self.assertEqual(auto.d_model, 1024)
        self.assertIn("does not fit in memory", "\n".join(lines_auto))
        for batch in ("1", "2", "64"):
            cfg, st, lines = plan(["--batch-size", batch])
            text = "\n".join(lines)
            self.assertEqual((cfg.d_model, cfg.n_layers, cfg.ctx), (auto.d_model, auto.n_layers, auto.ctx), batch)
            self.assertEqual(st["micro_batch"], int(batch))
            self.assertIn("does not fit in memory, so the largest that fits", text)
            self.assertEqual("WARNING: batch 64 x ctx 1024 may not fit" in text, batch == "64", text)
        kept, _, lines_kept = plan(["--d-model", "1536", "--batch-size", "2"])
        self.assertEqual(kept.d_model, 1536)
        self.assertIn("WARNING: batch 2 x ctx 2048 may not fit", "\n".join(lines_kept))
        # nothing on the ladder fits at batch 1: an error, as without --batch-size
        tiny = tiny_gpt.parse_args(["--name", "x", "--batch-size", "2"])
        with budget_of(0.01), self.assertRaises(SystemExit) as caught:
            tiny_gpt.plan_run(tiny, manifest, CPU, torch.bfloat16)
        self.assertIn("does not fit in memory even at batch 1", str(caught.exception))

    def test_time_budget_retries_memory_options_for_a_given_shape(self):
        """F347: with an explicit shape or --init-from (fixed_cfg) an out-of-memory speed test is repeated with
        the micro-batch halved down to 1 (which costs no compute: a plan that halving alone makes fit is the one
        it was before), and where that used to be the error, with activation checkpointing on and the batch as
        planned for it, then halved again; only a shape that fits with nothing is an error."""
        one = {"tokenizer": {"vocab_size": 16384},
               "sources": {"md": {"train": {"tokens": 1_000_000_000}, "doc_tokens_percentiles": {"90": 1000}}}}
        seen = []

        def measure(fails):
            def fake(cfg, micro, accum, device, amp_dtype, attn_pref="auto", warmup=2, timed=3, checkpointing=False,
                     loss_chunk=0, settings=None):
                seen.append((cfg.d_model, cfg.n_layers, micro, checkpointing))
                if fails(micro, checkpointing):
                    raise RuntimeError("CUDA out of memory (simulated)")
                return micro * accum * cfg.ctx * tiny_gpt.count_params(cfg)[0] * 1e-13, GIB
            return fake

        def plan(argv, fails, fixed_cfg=None):
            del seen[:]
            args = tiny_gpt.parse_args(["--name", "x", "--compile", "off", "--time-budget-hours", "1"] + argv)
            with budget_of(64), \
                    unittest.mock.patch.object(tiny_gpt, "measure_seconds_per_step", side_effect=measure(fails)), \
                    unittest.mock.patch.object(tiny_gpt, "sustained_speed", return_value=(None, "no table")), \
                    unittest.mock.patch.object(tiny_gpt, "have_triton", return_value=False):
                return tiny_gpt.plan_with_time_budget(args, one, CPU, None, fixed_cfg=fixed_cfg)

        # explicit shape that fits when the micro-batch is small enough: halved, no checkpointing, the same tokens
        # per step and training tokens as the plan whose speed test fits at once
        _, st_fits, _ = plan(["--d-model", "512"], lambda micro, ck: False)
        self.assertEqual([(micro, ck) for _, _, micro, ck in seen], [(st_fits["micro_batch"], False)])
        self.assertGreater(st_fits["micro_batch"], 16)
        cfg_s, st_s, lines_s = plan(["--d-model", "512"], lambda micro, ck: micro > 16 and not ck)
        self.assertEqual((st_s["micro_batch"], st_s["activation_checkpointing"]), (16, False))
        self.assertTrue(all(not ck for _, _, _, ck in seen), seen)
        self.assertEqual((st_s["tokens_per_step"], st_s["planned_train_tokens"], st_s["steps"]),
                         (st_fits["tokens_per_step"], st_fits["planned_train_tokens"], st_fits["steps"]))
        self.assertFalse(any("activation checkpointing on" in ln for ln in lines_s), lines_s)
        # out of memory at every micro-batch above 1, checkpointing or not: micro-batch 1 without checkpointing,
        # as before (checkpointing is not tried while halving still helps)
        cfg_1, st_1, _ = plan(["--d-model", "512"], lambda micro, ck: micro > 1)
        self.assertEqual((st_1["micro_batch"], st_1["activation_checkpointing"]), (1, False))
        self.assertTrue(all(not ck for _, _, _, ck in seen), seen)
        # explicit shape, out of memory unless checkpointing: halved down to 1 first (what used to end in the
        # error), then one more test with checkpointing on, which gives the plan of --activation-checkpointing on
        _, st_on, _ = plan(["--d-model", "512", "--activation-checkpointing", "on"], lambda micro, ck: False)
        test_on = list(seen)  # the one speed test of that plan
        cfg, st, lines = plan(["--d-model", "512"], lambda micro, ck: not ck)
        self.assertEqual(cfg.d_model, 512)
        self.assertTrue(st["activation_checkpointing"])
        micros = [micro for _, _, micro, _ in seen]
        self.assertEqual(micros[:-1], [micros[0] >> i for i in range(len(micros) - 1)])
        self.assertGreater(micros[0], 1)
        self.assertEqual(micros[-2], 1)
        self.assertEqual([ck for _, _, _, ck in seen], [False] * (len(seen) - 1) + [True])
        self.assertEqual(seen[-1:], test_on)
        self.assertEqual(st, st_on)
        self.assertTrue(any("d_model 512 ran out of memory in the speed test; measured again with activation "
                            "checkpointing on" in ln for ln in lines), lines)
        # only --layers given: the notes name the planned width, never 'd_model None'
        cfg_l, st_l, lines_l = plan(["--layers", "5"], lambda micro, ck: not ck)
        self.assertTrue(st_l["activation_checkpointing"])
        self.assertFalse(any("None" in ln for ln in lines_l), lines_l)
        self.assertTrue(any(ln.startswith("time budget: d_model ") and "ran out of memory in the speed test; "
                            "measured again with activation checkpointing on" in ln for ln in lines_l), lines_l)
        # checkpointing switched off on the command line: the micro-batch is halved instead
        cfg_h, st_h, lines_h = plan(["--d-model", "512", "--activation-checkpointing", "off"],
                                    lambda micro, ck: micro > 4)
        self.assertEqual((st_h["micro_batch"], st_h["activation_checkpointing"]), (4, False))
        self.assertTrue(all(not ck for _, _, _, ck in seen))
        # --init-from: the checkpoint's architecture is measured (not a ladder shape), with the same retries
        fixed = tiny_gpt.ModelConfig(vocab_size=16384, ctx=256, d_model=192, n_layers=3, n_heads=6, n_kv_heads=6)
        cfg_f, st_f, lines_f = plan([], lambda micro, ck: not ck, fixed_cfg=fixed)
        self.assertEqual((cfg_f.d_model, cfg_f.n_layers, cfg_f.ctx), (192, 3, 256))
        self.assertTrue(st_f["activation_checkpointing"])
        self.assertEqual({(d, layers) for d, layers, _, _ in seen}, {(192, 3)})
        self.assertTrue(any("d_model 192 ran out of memory" in ln for ln in lines_f), lines_f)
        # nothing helps: the error of the explicit branch, after checkpointing and every halving were tried
        with self.assertRaises(SystemExit) as caught:
            plan(["--d-model", "512"], lambda micro, ck: True)
        self.assertIn("runs out of memory on this device even at micro-batch 1", str(caught.exception))
        self.assertTrue(any(ck for _, _, _, ck in seen))
        self.assertEqual(seen[-1][2], 1)
        # a shape the planner itself refuses keeps the planner's message
        args = tiny_gpt.parse_args(["--name", "x", "--compile", "off", "--time-budget-hours", "1", "--d-model",
                                    "2560"])
        with budget_of(1), self.assertRaises(SystemExit) as refused, \
                unittest.mock.patch.object(tiny_gpt, "sustained_speed", return_value=(None, "no table")):
            tiny_gpt.plan_with_time_budget(args, one, CPU, None)
        self.assertIn("does not fit in memory even at batch 1", str(refused.exception))

    def test_sustained_speed_is_the_last_session_of_a_resumed_run(self):
        """F361: the rows of a resumed run are cumulative, so the planner's speed is the difference to the previous
        row of the same run (the last session), not the whole-run average; a different run under the same name
        (another shape, or the same shape started again from scratch) or a session too short to measure falls
        back to the whole run."""
        shape = {"event": "training run", "vocab": "16384", "ctx": "1024", "d_model": "1024", "layers": "18",
                 "heads": "16", "kv_heads": "4", "params": "216180000", "device": "cuda",
                 "activation_checkpointing": "off"}
        cfg = tiny_gpt.ModelConfig(vocab_size=16384, ctx=1024, d_model=1024, n_layers=18, n_heads=16, n_kv_heads=4)
        flops = tiny_gpt.flops_per_token(cfg)

        def speed_of(name, rows):
            out = os.path.join(self.tmp, name)
            os.makedirs(os.path.join(out, "results"))
            with open(os.path.join(out, "results", "experiments.csv"), "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
                writer.writeheader()
                writer.writerows(rows)
            return tiny_gpt.sustained_speed(out, torch.device("cuda"))

        def row(name, day, tokens, hours, **more):
            return {**shape, "name": name, "time": f"2026-11-{day:02d}T00:00:00", "tokens_seen": str(tokens),
                    "train_hours": str(hours), **more}

        # 100 h at 6,000 tokens/s, then two sessions of 4 h at 8,000 and 2 h at 9,000 tokens/s
        t1 = 100 * 3600 * 6000
        t2 = t1 + 4 * 3600 * 8000
        t3 = t2 + 2 * 3600 * 9000
        speed, note = speed_of("sessions", [row("run", 1, t1, 100), row("other", 2, 5_000_000, 1),
                                            row("run", 3, t2, 104), row("run", 4, t3, 106)])
        self.assertAlmostEqual(speed, 9000 * flops, delta=flops)
        self.assertIn("last session", note)
        self.assertIn("9,000 tokens/s", note)
        self.assertIn("run 'run' of 2026-11-04", note)
        # the activation-checkpointing flag of the row itself still sets the FLOP count
        speed_ck, note_ck = speed_of("sessions_ck", [row("run", 1, t1, 100),
                                                     row("run", 3, t2, 104, activation_checkpointing="on")])
        self.assertAlmostEqual(speed_ck, 8000 * tiny_gpt.flops_per_token(cfg, True), delta=flops)
        self.assertIn("activation checkpointing, last session", note_ck)
        # one row: the whole run, described as before
        speed_one, note_one = speed_of("one_row", [row("run", 1, t1, 100)])
        self.assertAlmostEqual(speed_one, 6000 * flops, delta=flops)
        self.assertNotIn("last session", note_one)
        # the same name with another shape is another run: the whole-run figure of the last row
        speed_shape, note_shape = speed_of("other_shape", [row("run", 1, t1, 100, d_model="768"),
                                                           row("run", 3, t2, 104)])
        self.assertAlmostEqual(speed_shape, t2 / (104 * 3600) * flops, delta=flops)
        self.assertNotIn("last session", note_shape)
        # a session under 12 minutes, and a run started again under its name (fewer tokens than the row before)
        speed_short, note_short = speed_of("short_session", [row("run", 1, t1, 100),
                                                             row("run", 3, t1 + 3_000_000, 100.1)])
        self.assertAlmostEqual(speed_short, (t1 + 3_000_000) / (100.1 * 3600) * flops, delta=flops)
        self.assertNotIn("last session", note_short)
        speed_new, note_new = speed_of("started_again", [row("run", 1, t1, 100), row("run", 3, 7_200_000, 1)])
        self.assertAlmostEqual(speed_new, 2000 * flops, delta=flops)
        self.assertNotIn("last session", note_new)
        # a run started again from scratch under its name and shape that got FURTHER than the old one (--overwrite):
        # more tokens and hours than the row before, but another run. Its whole training time fits between the two
        # rows, which no resumed run's does, so the whole-run figure of the new run stands (faster or slower)
        for case, old_speed, new_speed, when in (("again_faster", 6000, 8000, "2026-11-05T00:00:00"),
                                                 ("again_slower", 8000, 7000, "2026-11-05T00:00:00"),
                                                 ("again_at_once", 6000, 8000, "2026-11-01T12:30:00")):
            speed_again, note_again = speed_of(case, [
                row("run", 1, 10 * 3600 * old_speed, 10, status="finished"),
                row("run", 5, 12 * 3600 * new_speed, 12, status="finished", time=when)])
            self.assertAlmostEqual(speed_again, new_speed * flops, delta=flops, msg=case)
            self.assertIn(f"{new_speed:,} tokens/s)", note_again, case)
            self.assertNotIn("last session", note_again, case)
        # the same two rows written 3 hours apart can only be one run (12 hours of training do not fit in 3)
        speed_3h, note_3h = speed_of("resumed_3h", [row("run", 1, 10 * 3600 * 6000, 10),
                                                    row("run", 1, 10 * 3600 * 6000 + 2 * 3600 * 8000, 12,
                                                        time="2026-11-01T03:00:00")])
        self.assertAlmostEqual(speed_3h, 8000 * flops, delta=flops)
        self.assertIn("8,000 tokens/s, last session", note_3h)
        # a difference that is longer than the time between the rows, or rows without a usable time: the whole run
        for case, times in (("too_long", ("2026-11-01T00:00:00", "2026-11-01T01:00:00")), ("no_time", ("", "")),
                            ("mixed_time", ("2026-11-01T00:00:00+00:00", "2026-11-01T03:00:00"))):
            speed_w, note_w = speed_of(case, [row("run", 1, 10 * 3600 * 6000, 10, time=times[0]),
                                              row("run", 1, 10 * 3600 * 6000 + 2 * 3600 * 8000, 12, time=times[1])])
            self.assertAlmostEqual(speed_w, (10 * 6000 + 2 * 8000) / 12 * flops, delta=flops, msg=case)
            self.assertNotIn("last session", note_w, case)

    def test_resume_min_lr_ratio_that_already_holds_is_not_called_ignored(self):
        """F358: --min-lr-ratio on --resume that asks for the min_lr the run already has says so in every phase;
        'ignored' is only for a real change inside the decay."""
        settings = {"schedule": "wsd", "steps": 100, "lr": 1e-3, "min_lr": 0.0, "warmup_steps": 10,
                    "decay_frac": 0.2, "decay_shape": "1-sqrt"}
        for step in (5, 50, 90, 100):
            applied, note = tiny_gpt.resume_min_lr(dict(settings), step, 0.0)
            self.assertFalse(applied)
            self.assertIn("is already 0.00e+00", note, step)
            self.assertNotIn("ignored", note, step)
        changed = dict(settings)
        applied, note = tiny_gpt.resume_min_lr(changed, 90, 0.5)
        self.assertFalse(applied)
        self.assertIn("ignored (already in the decay phase)", note)
        self.assertEqual(changed["min_lr"], 0.0)
        applied, note = tiny_gpt.resume_min_lr(changed, 50, 0.5)
        self.assertTrue(applied)
        self.assertEqual(changed["min_lr"], 5e-4)

    def _refused(self, argv):
        """The message argparse prints when parse_args refuses `argv` (fails when it is accepted)."""
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit, msg=argv):
            tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp] + argv)
        return err.getvalue()

    def test_unchecked_inputs_are_refused(self):
        """F369: a RoPE base below 1, a negative z-loss weight or top-k, a token budget that is not a finite
        number and a warm-up as long as the planned run are refused with a message instead of a NaN model, a
        traceback or a run that never leaves its warm-up."""
        self.assertIn("--rope-base must be between", self._refused(["--rope-base", "-5"]))
        self.assertIn("--rope-base must be between", self._refused(["--rope-base", "0"]))
        self.assertIn("--rope-base must be between", self._refused(["--rope-base", "nan"]))
        self.assertIn("--z-loss must be between", self._refused(["--z-loss", "-1"]))
        self.assertIn("--top-k must be", self._refused(["--top-k", "-3"]))
        for flag in ("--train-tokens", "--epochs"):
            for value in ("nan", "inf"):
                self.assertIn("finite", self._refused([flag, value]), (flag, value))
        ok = tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp, "--rope-base", "500000", "--z-loss", "0",
                                  "--top-k", "0", "--train-tokens", "1e6"])
        self.assertEqual((ok.rope_base, ok.z_loss, ok.top_k, ok.train_tokens), (500000.0, 0.0, 0, 1e6))
        self.assertIsNone(tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp]).rope_base)
        # a warm-up as long as the planned run (no --steps for validate_args to compare with)
        manifest = toy_manifest(tokens=200_000, vocab=512)
        _, st, _ = self._plan([], manifest)
        with self.assertRaises(SystemExit) as caught:
            self._plan(["--warmup-steps", str(st["steps"])], manifest)
        self.assertIn("--warmup-steps", str(caught.exception))
        self.assertIn(f"{st['steps']:,}", str(caught.exception))
        _, st_ok, _ = self._plan(["--warmup-steps", str(st["steps"] - 1)], manifest)
        self.assertEqual((st_ok["warmup_steps"], st_ok["steps"]), (st["steps"] - 1, st["steps"]))
        self.assertLess(st["warmup_steps"], st["steps"], "the default warm-up is never the whole run")

    def test_stop_after_steps_must_be_positive(self):
        """F370 (guard: passes before and after): --stop-after-steps is checked with the other counts that must
        be positive, with the same message."""
        for value in ("0", "-2"):
            self.assertIn("--stop-after-steps must be positive", self._refused(["--stop-after-steps", value]))
        args = tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp, "--stop-after-steps", "3"])
        self.assertEqual(args.stop_after_steps, 3)
        self.assertIsNone(tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp]).stop_after_steps)

    def _safetensors_samples(self):
        torch.manual_seed(3)
        return {"f32": torch.randn(5, 7), "f16": torch.randn(4, 3).half(), "bf16": torch.randn(6, 2).bfloat16(),
                "i64": torch.arange(11), "i32": torch.arange(6, dtype=torch.int32).reshape(2, 3),
                "i16": torch.arange(5, dtype=torch.int16), "u8": torch.arange(9, dtype=torch.uint8),
                "bool": torch.tensor([[True, False], [False, True]]), "scalar": torch.tensor(2.5),
                "empty": torch.empty(0, 4), "strided": torch.randn(6, 8)[::2, 1::3], "grad": torch.nn.Parameter(
                    torch.randn(3, 3))}

    def test_write_safetensors_streams_the_tensors(self):
        """F373: write_safetensors writes each tensor's memory straight to the file (no bytes copy of the whole
        model first); the file is byte for byte what the copying writer produced."""
        import struct
        import tracemalloc
        tensors = self._safetensors_samples()
        path = os.path.join(self.tmp, "samples.safetensors")
        tiny_gpt.write_safetensors(tensors, path, {"format": "tinyGPT", "n": 3})
        # the reference: the format written by hand with a full copy of every tensor
        header, blobs, offset = {}, [], 0
        for name in sorted(tensors):
            t = tensors[name].detach().cpu().contiguous()
            raw = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()
            header[name] = {"dtype": tiny_gpt._ST_DTYPES[t.dtype], "shape": list(t.shape),
                            "data_offsets": [offset, offset + len(raw)]}
            blobs.append(raw)
            offset += len(raw)
        header["__metadata__"] = {"format": "tinyGPT", "n": "3"}
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))
        back, meta = tiny_gpt.read_safetensors(path)
        self.assertEqual(meta, {"format": "tinyGPT", "n": "3"})
        for name, t in tensors.items():
            self.assertEqual((back[name].dtype, back[name].shape), (t.dtype, t.shape), name)
            self.assertTrue(torch.equal(back[name], t.detach()), name)
        self.assertEqual(glob_tmp(self.tmp), [], "no temporary file is left")
        # 16 MiB of weights: the writer allocates far less than a copy of them
        big = {"w": torch.zeros(4 * 2 ** 20), "v": torch.ones(2 ** 20).bfloat16()}
        tracemalloc.start()
        try:
            tiny_gpt.write_safetensors(big, os.path.join(self.tmp, "big.safetensors"))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 2 * 2 ** 20, f"{peak:,} bytes allocated while writing 18 MiB of tensors")
        self.assertEqual(os.path.getsize(os.path.join(self.tmp, "big.safetensors")) % 8, 0)

    def test_read_safetensors_reads_each_tensor_into_its_own_memory(self):
        """F352: read_safetensors reads every tensor straight into the buffer it keeps (no copy of the whole file
        beside the tensors); the tensors are the same, and a file cut short is refused as corrupt."""
        import tracemalloc
        tensors = self._safetensors_samples()
        path = os.path.join(self.tmp, "read_samples.safetensors")
        tiny_gpt.write_safetensors(tensors, path, {"note": "x"})
        back, meta = tiny_gpt.read_safetensors(path)
        self.assertEqual(meta, {"note": "x"})
        self.assertEqual(sorted(back), sorted(tensors))
        for name, t in tensors.items():
            self.assertEqual((back[name].dtype, back[name].shape), (t.dtype, t.shape), name)
            self.assertTrue(torch.equal(back[name], t.detach()), name)
        back["f32"][0, 0] = 5.0  # its own writable memory
        self.assertEqual(float(back["f32"][0, 0]), 5.0)
        # 18 MiB of weights: the reader allocates about the tensors, not the file a second time
        big = {"w": torch.arange(4 * 2 ** 20, dtype=torch.float32), "v": torch.ones(2 ** 20).bfloat16()}
        big_path = os.path.join(self.tmp, "read_big.safetensors")
        tiny_gpt.write_safetensors(big, big_path)
        size = os.path.getsize(big_path)
        tracemalloc.start()
        try:
            big_back, _ = tiny_gpt.read_safetensors(big_path)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertTrue(torch.equal(big_back["w"], big["w"]) and torch.equal(big_back["v"], big["v"]))
        self.assertLess(peak, 1.25 * size, f"{peak:,} bytes allocated while reading a file of {size:,} bytes")
        # cut short inside the last tensor, and inside the first: SystemExit naming the file, never a short tensor
        with open(path, "rb") as handle:
            whole = handle.read()
        for cut in (len(whole) - 1, len(whole) - 200):
            short = os.path.join(self.tmp, f"read_short_{cut}.safetensors")
            with open(short, "wb") as handle:
                handle.write(whole[:cut])
            with self.assertRaises(SystemExit) as caught:
                tiny_gpt.read_safetensors(short)
            self.assertIn("corrupt entry", str(caught.exception))
            self.assertIn(short, str(caught.exception))
        # a file that shrinks after its size was read (the short count of readinto)
        real_fstat = os.fstat

        def longer(fd):
            st = real_fstat(fd)
            return os.stat_result(tuple(st[:6]) + (st.st_size + 64,) + tuple(st[7:10]))

        with unittest.mock.patch.object(tiny_gpt.os, "fstat", side_effect=longer), \
                unittest.mock.patch.object(tiny_gpt.json, "loads", side_effect=lambda raw: {
                    "x": {"dtype": "U8", "shape": [len(whole)], "data_offsets": [0, len(whole) - 8 - len(raw) + 64]}}):
            with self.assertRaises(SystemExit) as caught:
                tiny_gpt.read_safetensors(path)
        self.assertIn("corrupt entry x", str(caught.exception))

    # ---------------- export, benchmark and generate on a toy checkpoint ----------------
    _tok = None

    @classmethod
    def _toy_tokenizer(cls):
        """A small SentencePiece tokenizer in data_prep's 'lines' convention (trained once, in memory)."""
        if cls._tok is None:
            import sentencepiece as spm
            words = ("genetic drift allele frequency population selection mutation gene protein cell the of in "
                     "is a").split()
            lines = [" ".join(words[(i * 7 + j * 3) % len(words)] for j in range(8)) for i in range(200)]
            model = io.BytesIO()
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(lines), model_writer=model, model_type="bpe", vocab_size=300,
                byte_fallback=True, unk_id=0, bos_id=1, eos_id=2, pad_id=-1, user_defined_symbols=["\n", "\t"],
                remove_extra_whitespaces=False, allow_whitespace_only_pieces=True,
                normalization_rule_name="identity", hard_vocab_limit=False, num_threads=1, minloglevel=2)
            cls._tok = tiny_gpt.data_prep.Tokenizer(model.getvalue())
        return cls._tok

    def _toy_checkpoint(self, folder, name="A", heads=4, kv_heads=2, **extra):
        """<folder>/<name>_best.pt: an untrained toy model in the layout of the best file _train writes."""
        tok = self._toy_tokenizer()
        torch.manual_seed(1)
        cfg = tiny_gpt.ModelConfig(vocab_size=tok.vocab_size, ctx=16, d_model=32, n_layers=2, n_heads=heads,
                                   n_kv_heads=kv_heads)
        obj = {"format": tiny_gpt.CHECKPOINT_FORMAT, "format_version": tiny_gpt.FORMAT_VERSION,
               "checkpoint_type": "inference", "model_config": dataclasses.asdict(cfg),
               "model": tiny_gpt.TinyGPT(cfg).state_dict(), "tokenizer": {"proto": tok.proto, **tok.meta()},
               "step": 7, "metrics": {"best_val": 5.5, "best_step": 6}, "weights": "raw", **extra}
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{name}_best.pt")
        torch.save(obj, path)
        return path

    @staticmethod
    def _args(folder, *argv):
        return tiny_gpt.parse_args(["--name", "A", "--out-dir", folder, "--device", "cpu", *argv])

    @staticmethod
    def _quiet(command, args):
        """What `command(args)` prints."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            command(args)
        return out.getvalue()

    def test_commands_resolve_the_checkpoint_in_one_place(self):
        """F283 (with F279, F219, F285): --export, --benchmark and --generate take their checkpoint from
        resolve_checkpoint: a normalised path (an export folder typed with a trailing separator is the folder),
        and one 'No checkpoint' message that names the other runs of the folder without '--resume' (which would
        start training) and without the hint to a legacy file."""
        runs = os.path.join(self.tmp, "resolve")
        best = self._toy_checkpoint(runs)
        self.assertEqual(tiny_gpt.resolve_checkpoint(self._args(runs), "pass --checkpoint"), os.path.normpath(best))
        sloppy = best.replace(os.sep, "/").replace("/resolve/", "/resolve/./")
        self.assertEqual(tiny_gpt.resolve_checkpoint(self._args(runs, "--checkpoint", sloppy), "x"),
                         os.path.normpath(best))
        try:
            relative = os.path.relpath(best)
        except ValueError:  # Windows: the temporary folder is on another drive
            relative = None
        if relative:
            self.assertFalse(os.path.isabs(relative))
            self.assertEqual(tiny_gpt.resolve_checkpoint(self._args(runs, "--checkpoint", relative), "x"), relative,
                             "a relative path stays relative")
        os.replace(best, os.path.join(runs, "A.pt"))  # no best file: the training checkpoint
        self.assertEqual(tiny_gpt.resolve_checkpoint(self._args(runs), "x"), os.path.join(runs, "A.pt"))
        os.replace(os.path.join(runs, "A.pt"), best)
        # nothing to work on: the same message from the three commands
        empty = os.path.join(self.tmp, "resolve_empty")
        os.makedirs(empty)
        for name in ("oldrun.pt", "tiny_gpt_bpe_best.pt"):
            open(os.path.join(empty, name), "wb").close()
        missing = os.path.join(empty, "A.pt")
        for command, argv, hint in (
                (tiny_gpt.cmd_export, ["--export", os.path.join(empty, "out")], "pass --checkpoint"),
                (tiny_gpt.cmd_benchmark, ["--benchmark"], "pass --checkpoint"),
                (tiny_gpt.cmd_generate, ["--generate", "x"], "train first or pass --checkpoint")):
            with self.assertRaises(SystemExit) as caught:
                command(self._args(empty, *argv))
            self.assertEqual(str(caught.exception),
                             f"No checkpoint {missing}; {hint}. Training checkpoints in this folder: oldrun. "
                             "Pass the right one with --name, e.g. --name oldrun", argv)
        with self.assertRaises(SystemExit) as caught:
            tiny_gpt.cmd_generate(self._args(runs, "--generate", "x", "--checkpoint", os.path.join(runs, "no.pt")))
        self.assertEqual(str(caught.exception),
                         f"No checkpoint {os.path.join(runs, 'no.pt')}; train first or pass --checkpoint.")
        # a failed --resume still names --resume
        self.assertTrue(tiny_gpt._resume_hint(self._args(empty)).endswith("e.g. --name oldrun --resume"))
        # an export folder typed with a trailing separator (PowerShell tab completion)
        export = os.path.join(runs, "exp")
        self._quiet(tiny_gpt.cmd_export, self._args(runs, "--export", export))
        probes = os.path.join(runs, "probes.tsv")
        with open(probes, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("# toy probes\n"
                         "drift\teasy\tgenetic drift is a\tprotein\tcell|gene\n"
                         "drift\teasy\tthe allele frequency of\tselection\tmutation|population\n")
        out = self._quiet(tiny_gpt.cmd_benchmark, self._args(runs, "--benchmark", "--probes", probes, "--checkpoint",
                                                             export + os.sep))
        self.assertIn(f" on {export} (step 7, ", out)
        beside = [name for name in os.listdir(runs) if name.startswith("exp_benchmark_step7_")]
        self.assertEqual(len(beside), 1, os.listdir(runs))
        self.assertEqual(sorted(os.listdir(export)), ["config.json", "model.safetensors", "tokenizer.model"])
        with open(os.path.join(runs, "results", "experiments.csv"), encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(r["event"], r["name"]) for r in rows], [("benchmark", "exp")])
        self.assertEqual(int(rows[0]["params"]), tiny_gpt.count_params(tiny_gpt.ModelConfig(
            vocab_size=self._toy_tokenizer().vocab_size, ctx=16, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2))[0])
        out = self._quiet(tiny_gpt.cmd_generate, self._args(runs, "--generate", "genetic", "--length", "3",
                                                            "--checkpoint", export + os.sep))
        self.assertTrue(out.startswith(f"Checkpoint {export} (tinygpt, inference) | step 7 | "), out)

    def test_export_loads_the_checkpoint_once(self):
        """F281: --export builds its model with load_for_inference (the model holds the checkpoint's own tensors:
        one copy of the weights, and the optimizer state of a training checkpoint is dropped at once). The
        exported tensors are the checkpoint's, and an export folder can itself be exported."""
        runs = os.path.join(self.tmp, "export_once")
        best = self._toy_checkpoint(runs, optimizer={"state": torch.zeros(1000)}, checkpoint_type="training")
        source = torch.load(best, map_location="cpu", weights_only=True)["model"]
        first, second = os.path.join(runs, "exp1"), os.path.join(runs, "exp2")
        seen = {}
        real = tiny_gpt.model_from_checkpoint

        def spy(obj, *a, **k):
            seen["assign"] = k.get("assign", a[1] if len(a) > 1 else False)
            return real(obj, *a, **k)

        with unittest.mock.patch.object(tiny_gpt, "model_from_checkpoint", spy):
            out = self._quiet(tiny_gpt.cmd_export, self._args(runs, "--export", first))
        self.assertIs(seen["assign"], True, "the model takes the checkpoint's tensors instead of copying them")
        self.assertIn(f"verified {len(source) - 1} tensors by reading them back", out)
        self._quiet(tiny_gpt.cmd_export, self._args(runs, "--export", second, "--checkpoint", first))
        for folder in (first, second):
            tensors, meta = tiny_gpt.read_safetensors(os.path.join(folder, "model.safetensors"))
            self.assertEqual(sorted(tensors), sorted(k for k in source if k != "lm_head.weight"), folder)
            for name, t in tensors.items():
                self.assertEqual(t.dtype, torch.float32, name)
                self.assertTrue(torch.equal(t, source[name]), (folder, name))
            self.assertEqual(meta["tie_embeddings"], "True")
            with open(os.path.join(folder, "config.json"), encoding="utf-8") as handle:
                config = json.load(handle)
            self.assertEqual((config["step"], config["weights"], config["metrics"]),
                             (7, "raw", {"best_val": 5.5, "best_step": 6}), folder)

    def test_export_keeps_dataset_settings_and_weights_label(self):
        """F284: an export folder records the dataset (name and fingerprint only: no folder of this machine),
        the training settings and which weights it holds, and load_exported hands them on, so the dataset can
        be found again; a checkpoint or an older export folder without them still loads."""
        runs = os.path.join(self.tmp, "export_keys")
        dataset = os.path.join(runs, "ds_toy")
        os.makedirs(dataset)
        fingerprint = "f" * 64
        with open(os.path.join(dataset, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"format": tiny_gpt.data_prep.FORMAT, "format_version": tiny_gpt.data_prep.FORMAT_VERSION,
                       "fingerprint": fingerprint}, handle)
        settings = {"steps": 10, "optimizer": "muon", "betas": [0.9, 0.95], "lr": 1e-3, "doc_mask": False,
                    "mix": None}
        self._toy_checkpoint(runs, train_config=settings, weights="ema", ema_decay=0.999,
                             dataset={"path": dataset + os.sep, "fingerprint": fingerprint,
                                      "train_tokens": {"md": 1000}, "val_tokens": {"md": 100}})
        export = os.path.join(runs, "exp")
        self._quiet(tiny_gpt.cmd_export, self._args(runs, "--export", export))
        with open(os.path.join(export, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(config["dataset"], {"path": "ds_toy", "fingerprint": fingerprint})
        self.assertEqual(config["train_config"], settings)
        back = tiny_gpt.load_exported(export)
        self.assertEqual((back["dataset"], back["train_config"], back["weights"]),
                         (config["dataset"], settings, "ema"))
        with unittest.mock.patch.object(tiny_gpt, "HERE", runs):  # found by its name, checked by its fingerprint
            self.assertEqual(tiny_gpt.resolve_dataset(back), os.path.join(runs, "ds_toy"))
        self.assertEqual(tiny_gpt.load_for_inference(export)[3]["train_config"], settings)
        # a checkpoint that records neither, and an export folder written before the keys existed
        plain = os.path.join(self.tmp, "export_keys_plain")
        self._toy_checkpoint(plain)
        export_plain = os.path.join(plain, "exp")
        self._quiet(tiny_gpt.cmd_export, self._args(plain, "--export", export_plain))
        with open(os.path.join(export_plain, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertNotIn("dataset", config)
        self.assertNotIn("train_config", config)
        del config["weights"]
        with open(os.path.join(export_plain, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        old = tiny_gpt.load_exported(export_plain)
        self.assertEqual((old["dataset"], old["train_config"], old["weights"]), (None, None, "raw"))
        with self.assertRaises(SystemExit) as caught:
            tiny_gpt.resolve_dataset(old)
        self.assertIn("records no dataset", str(caught.exception))

    def test_export_records_only_the_name_of_a_dataset_stored_on_another_system(self):
        """F284: a checkpoint may carry a dataset path written on the other kind of system (trained on Windows,
        exported on the Mac, or the reverse). The export still records the folder's name alone, by the rule of
        resolve_dataset, never the whole foreign path with its user name."""
        import posixpath
        fingerprint = "f" * 64
        for case, stored in (("windows", "C:\\Users\\someone\\project\\datasets\\bio_all"),
                             ("windows_sep", "C:\\Users\\someone\\project\\datasets\\bio_all\\"),
                             ("posix", "/Users/someone/project/datasets/bio_all/"),
                             ("mixed", "C:\\Users\\someone/project/datasets\\bio_all")):
            runs = os.path.join(self.tmp, f"export_foreign_{case}")
            self._toy_checkpoint(runs, dataset={"path": stored, "fingerprint": fingerprint})
            args = self._args(runs, "--export", os.path.join(runs, "exp"))
            # os.path.basename as it is on macOS and Linux, where a backslash is no separator
            with unittest.mock.patch.object(tiny_gpt.os.path, "basename", posixpath.basename):
                self._quiet(tiny_gpt.cmd_export, args)
            with open(os.path.join(runs, "exp", "config.json"), encoding="utf-8") as handle:
                config = json.load(handle)
            self.assertEqual(config["dataset"], {"path": "bio_all", "fingerprint": fingerprint}, case)
            self.assertNotIn("someone", json.dumps(config["dataset"]), case)

    def test_export_names_the_attention_the_model_has(self):
        """F282: config.json of an export says 'grouped-query attention' with the head counts only when the
        key/value heads are shared; a model with as many key/value heads as query heads is 'multi-head attention'."""
        for name, heads, kv, want in (("gqa", 4, 2, "SwiGLU, grouped-query attention (4 query / 2 key-value heads), "),
                                      ("mha", 4, 4, "SwiGLU, multi-head attention, ")):
            runs = os.path.join(self.tmp, f"export_attn_{name}")
            self._toy_checkpoint(runs, heads=heads, kv_heads=kv)
            self._quiet(tiny_gpt.cmd_export, self._args(runs, "--export", os.path.join(runs, "exp")))
            with open(os.path.join(runs, "exp", "config.json"), encoding="utf-8") as handle:
                config = json.load(handle)
            self.assertIn(want, config["architecture"])
            self.assertEqual("grouped-query" in config["architecture"], kv < heads)
            self.assertEqual((config["model_config"]["n_heads"], config["model_config"]["n_kv_heads"]), (heads, kv))

    def test_benchmark_refuses_a_probe_file_without_probes(self):
        """F280: --benchmark with a --probes file that holds no probe line (comments, blank lines) stops with a
        message naming the file instead of a ZeroDivisionError; nothing is written."""
        runs = os.path.join(self.tmp, "no_probes")
        self._toy_checkpoint(runs)
        probes = os.path.join(runs, "empty.tsv")
        with open(probes, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("# category\tdifficulty\tprompt\tanswer\tdistractors\n\n# nothing else\n")
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stdout(io.StringIO()):
            tiny_gpt.cmd_benchmark(self._args(runs, "--benchmark", "--probes", probes))
        self.assertEqual(str(caught.exception), f"{probes}: no probes found")
        self.assertEqual(sorted(os.listdir(runs)), ["A_best.pt", "empty.tsv"])

    def test_generate_header_without_a_step(self):
        """F286: the --generate header of a checkpoint that has a best validation loss but no step count prints
        'step ?' instead of raising; with a step it is written with thousands separators in both forms."""
        runs = os.path.join(self.tmp, "no_step")
        best = self._toy_checkpoint(runs, step=None)
        obj = torch.load(best, map_location="cpu", weights_only=True)

        def header(**change):
            for key, value in change.items():
                if value is None:
                    obj.pop(key, None)
                else:
                    obj[key] = value
            torch.save(obj, best)
            out = self._quiet(tiny_gpt.cmd_generate, self._args(runs, "--generate", "genetic", "--length", "2"))
            return out.splitlines()[0]

        start = f"Checkpoint {best} (tinygpt, inference) | step "
        self.assertTrue(header().startswith(start + "? | best validation loss 5.5000 @6 | "), header())
        self.assertTrue(header(step=None).startswith(start + "? | best validation loss 5.5000 @6 | "))
        self.assertTrue(header(step=12345).startswith(start + "12,345 | best validation loss 5.5000 @6 | "))
        self.assertEqual(header(metrics={}), start + "12,345")
        self.assertEqual(header(step=None), start + "?")

    def test_home_short_masks_only_a_leading_home_folder(self):
        """F215 (home-masker half, not merged): _home_short, which the export and the tools use on one path,
        stays apart from _hide_home, the masker of the experiments table's text, because the two differ: it
        shortens the home folder only at the start of a path and only as a whole folder, so a folder beside the
        home folder whose name starts the same way, the home folder itself and a home string in the middle of a
        path read as before. On the paths a checkpoint usually has the two agree."""
        fake = os.path.join(os.path.splitdrive(self.tmp)[0] + os.sep, "Users", "lab")
        differ = [fake + "2" + os.sep + os.path.join("runs", "a.pt"),  # a folder beside it, same first letters
                  fake,  # the home folder itself
                  "copy" + os.sep + fake + os.sep + "a.pt"]  # the home string in the middle
        with unittest.mock.patch.object(tiny_gpt.os.path, "expanduser", return_value=fake):
            self.assertEqual([tiny_gpt._home_short(p) for p in differ], differ)
            self.assertEqual(tiny_gpt._home_short(os.path.join(fake, "runs", "a.pt")),
                             "~" + os.sep + os.path.join("runs", "a.pt"))
            self.assertNotEqual([tiny_gpt._hide_home(p) for p in differ], differ)  # why they are not one function
        home = os.path.expanduser("~")
        samples = [os.path.join(home, "runs", "A_best.pt"),  # inside the home folder
                   os.path.join(os.path.splitdrive(home)[0] + os.sep, "data", "runs", "A_best.pt"),  # outside it
                   os.path.join("runs", "A_best.pt"),  # relative
                   os.path.join(home, "runs", "exp") + os.sep,  # with a trailing separator
                   home.replace("\\", "/") + "/runs/A_best.pt"]
        want = ["~" + os.sep + os.path.join("runs", "A_best.pt"), samples[1], samples[2],
                "~" + os.sep + os.path.join("runs", "exp") + os.sep, "~/runs/A_best.pt"]
        if os.name == "nt":  # mixed case on Windows
            samples.append(os.path.join(home.swapcase(), "Runs", "A_best.pt"))
            want.append("~" + os.sep + os.path.join("Runs", "A_best.pt"))
        self.assertEqual([tiny_gpt._home_short(p) for p in samples], want)
        self.assertEqual([tiny_gpt._hide_home(p) for p in samples], want)
        # the export records its source with it
        runs = os.path.join(self.tmp, "home_mask")
        best = self._toy_checkpoint(runs)
        with unittest.mock.patch.object(tiny_gpt.os.path, "expanduser", return_value=self.tmp):
            self._quiet(tiny_gpt.cmd_export, self._args(runs, "--export", os.path.join(runs, "exp")))
        with open(os.path.join(runs, "exp", "config.json"), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["source_checkpoint"], "~" + best[len(self.tmp):])

    def test_sigterm_and_sighup_pause_like_ctrl_c(self):
        """F296: cmd_train gives SIGTERM and SIGHUP (kill, a closed terminal, logout) Python's Ctrl+C handler, so
        they take the pause path, and DeferInterrupt holds them like Ctrl+C until the update is done. Ctrl+Break
        (SIGBREAK) is left alone: it stays the hard stop. In a child process, because before the change the
        signal ended the process."""
        child = textwrap.dedent("""
            import os, signal, sys
            sys.path.insert(0, sys.argv[1])
            import tiny_gpt
            names = [name for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)]
            ctrl_break = getattr(signal, "SIGBREAK", None)
            before = signal.getsignal(ctrl_break) if ctrl_break is not None else None

            def fake_train(args):
                for name in names:
                    number, held = getattr(signal, name), tiny_gpt.DeferInterrupt()
                    with held:
                        signal.raise_signal(number)
                        print(name, "held", held.hit, flush=True)
                    print(name, "restored", signal.getsignal(number) is signal.default_int_handler, flush=True)
                    try:
                        held.deliver()
                    except KeyboardInterrupt:
                        print(name, "delivered", flush=True)
                    held.deliver()  # once only
                    try:
                        signal.raise_signal(number)  # outside an update: the pause path at once
                    except KeyboardInterrupt:
                        print(name, "pauses", flush=True)
                print("SIGBREAK untouched", ctrl_break is None or signal.getsignal(ctrl_break) is before, flush=True)

            tiny_gpt._train = fake_train
            tiny_gpt.cmd_train(tiny_gpt.parse_args(["--name", "sig", "--out-dir", sys.argv[2]]))
            print("locks left", [name for name in os.listdir(sys.argv[2]) if name.endswith(".lock")])
        """)
        folder = os.path.join(self.tmp, "signals")
        os.makedirs(folder)
        proc = subprocess.run([sys.executable, "-c", child, HERE, folder], cwd=folder, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=600,
                              env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        want = []
        for name in ("SIGTERM", "SIGHUP"):
            if hasattr(signal, name):
                want += [f"{name} held True", f"{name} restored True", f"{name} delivered", f"{name} pauses"]
        want += ["SIGBREAK untouched True", "locks left []"]
        self.assertEqual(proc.stdout.splitlines(), want, proc.stderr[-2000:])
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])

    def test_an_ignored_pause_signal_stays_ignored(self):
        """F296: a signal the run was started to ignore (nohup leaves SIGHUP ignored so that the run survives a
        closed terminal) is not given the Ctrl+C handler: it stays ignored inside cmd_train, also across an
        update held by DeferInterrupt, and the other pause signal still pauses. In a child process (it changes
        the handlers of its process)."""
        child = textwrap.dedent("""
            import os, signal, sys
            sys.path.insert(0, sys.argv[1])
            import tiny_gpt
            names = [name for name in tiny_gpt.PAUSE_SIGNALS if hasattr(signal, name)]
            ignored = getattr(signal, names[-1])  # SIGHUP where it exists (nohup), SIGTERM on Windows
            signal.signal(ignored, signal.SIG_IGN)

            def fake_train(args):
                print("handler", signal.getsignal(ignored) is signal.SIG_IGN, flush=True)
                held = tiny_gpt.DeferInterrupt()
                with held:
                    signal.raise_signal(ignored)
                print("held", held.hit, flush=True)
                held.deliver()
                signal.raise_signal(ignored)
                print("ignored", signal.getsignal(ignored) is signal.SIG_IGN, flush=True)
                for name in names[:-1]:
                    try:
                        signal.raise_signal(getattr(signal, name))
                    except KeyboardInterrupt:
                        print(name, "pauses", flush=True)

            tiny_gpt._train = fake_train
            tiny_gpt.cmd_train(tiny_gpt.parse_args(["--name", "sig", "--out-dir", sys.argv[2]]))
            print("locks left", [name for name in os.listdir(sys.argv[2]) if name.endswith(".lock")])
        """)
        folder = os.path.join(self.tmp, "signal_ignored")
        os.makedirs(folder)
        proc = subprocess.run([sys.executable, "-c", child, HERE, folder], cwd=folder, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=600,
                              env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        names = [name for name in tiny_gpt.PAUSE_SIGNALS if hasattr(signal, name)]
        want = ["handler True", "held False", "ignored True"] + [f"{name} pauses" for name in names[:-1]]
        self.assertEqual(proc.stdout.splitlines(), want + ["locks left []"], proc.stderr[-2000:])
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])

    def test_compatibility_options_of_the_original_script_are_gone(self):
        """F334, F288, F273: the six options kept for the original script (--data, --wiki-dir, --wiki-articles,
        --include-keywords, --tokenizer-vocab-size, --suppress) and the dataset auto-build behind them are
        removed. All six are refused as unrecognised: '--data' too, which argparse would otherwise read as an
        abbreviation of --dataset (an old '--data <md folder>' would be taken for a prepared dataset). The other
        abbreviations of --dataset still work. ensure_dataset is --dataset as an absolute path, or None, and
        starts no process."""
        md = os.path.join(self.tmp, "md_folder")
        os.makedirs(md)
        absent = os.path.join(self.tmp, "absent")
        for argv in (["--wiki-dir", self.tmp], ["--wiki-articles", "5"], ["--include-keywords", "gene"],
                     ["--tokenizer-vocab-size", "700"], ["--suppress"], ["--data", md], ["--data=" + md],
                     ["--data", absent]):
            self.assertIn("unrecognized arguments: " + argv[0], self._refused(argv), argv)
        self.assertIn("unrecognized arguments: --data", self._refused(["--dataset", md, "--data", md]))
        self.assertNotIn("--dataset folder not found", self._refused(["--data", absent]))
        args = tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp, "--datas", md])  # still an abbreviation
        self.assertEqual(args.dataset, md)
        self.assertEqual(tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp, "--dataset=" + md]).dataset, md)
        for name in ("data", "wiki_dir", "wiki_articles", "include_keywords", "tokenizer_vocab_size", "suppress"):
            self.assertFalse(hasattr(args, name), name)
        self.assertFalse(hasattr(tiny_gpt, "compat_dataset_dir"))
        with unittest.mock.patch.object(tiny_gpt.subprocess, "run") as started:
            self.assertEqual(tiny_gpt.ensure_dataset(args), os.path.abspath(md))
            args.dataset = "some_folder"
            self.assertEqual(tiny_gpt.ensure_dataset(args), os.path.abspath("some_folder"))
            none = tiny_gpt.parse_args(["--name", "no_data", "--out-dir", self.tmp, "--device", "cpu"])
            self.assertIsNone(tiny_gpt.ensure_dataset(none))
            # the two commands that need a dataset still say so
            with self.assertRaises(SystemExit) as caught:
                tiny_gpt.cmd_plan(none)
            self.assertEqual(str(caught.exception), "--plan needs --dataset.")
            with self.assertRaises(SystemExit) as caught:
                tiny_gpt._train(none)
            self.assertEqual(str(caught.exception), "No training data: pass --dataset <folder from data_prep.py>.")
            with self.assertRaises(SystemExit) as caught:  # a folder of Markdown files is no dataset
                tiny_gpt.cmd_plan(tiny_gpt.parse_args(["--name", "x", "--out-dir", self.tmp, "--dataset", md]))
            self.assertIn("is not a prepared dataset (manifest.json missing)", str(caught.exception))
        started.assert_not_called()
