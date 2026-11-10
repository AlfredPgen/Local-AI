"""Staging tests of the 2026-11-10 audit, package A2 (tiny_gpt.py: planner, arguments, export/benchmark/generate
commands, signals). Target class in test_tiny_gpt.py: TestFixes.

    python -m unittest test_a2_a2
"""
import argparse
import contextlib
import csv
import dataclasses
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiny_gpt  # noqa: E402
import test_tiny_gpt as base  # noqa: E402 - as a module: a TestCase class imported by name would run twice

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
