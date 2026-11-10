"""Staging tests of the 2026-11-10 audit, package A2 (tiny_gpt.py: planner, arguments, export/benchmark/generate
commands, signals). Target class in test_tiny_gpt.py: TestFixes.

    python -m unittest test_a2_a2
"""
import contextlib
import csv
import io
import json
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
        activation checkpointing first, then with the micro-batch halved, as on the ladder; only a shape that fits
        with nothing is an error."""
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

        # explicit shape, out of memory unless checkpointing: one retry, no halving
        cfg, st, lines = plan(["--d-model", "512"], lambda micro, ck: not ck)
        self.assertEqual(cfg.d_model, 512)
        self.assertTrue(st["activation_checkpointing"])
        self.assertEqual([ck for _, _, _, ck in seen], [False, True])
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
        row of the same run (the last session), not the whole-run average; a different run under the same name or
        a session too short to measure falls back to the whole run."""
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
