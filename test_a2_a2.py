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
