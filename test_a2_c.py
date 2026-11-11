"""Staging tests of audit package C (view_pt.py). Package T moves them into test_tiny_gpt.py
(target class TestPipeline; the pure-function tests to TestFixes) and deletes this file.

    python -m unittest -v test_a2_c
"""

import dataclasses
import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import unittest
import unittest.mock
import warnings
from contextlib import redirect_stdout

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiny_gpt  # noqa: E402
from test_tiny_gpt import types_ns  # noqa: E402

BANNED_TITLES = ("Adam update size", "Muon momentum size", "Validation perplexity")


def history_rows(n=60, tokens=True, full=True):
    """n evaluations 200 steps apart whose validation loss falls from 10.0 to 2.9; full=False keeps only what a
    log-derived or legacy history has (step, train_loss, val_loss, lr)."""
    out = []
    for i in range(n):
        val = 2.9 + 7.1 * math.exp(-i / 4)
        row = {"step": 200 * i, "train_loss": None if i == 0 else val - 0.05, "val_loss": val, "lr": 1e-3}
        if tokens:
            row["tokens"] = 200_000 * i
        if full:
            row.update({
                "val_mix": None if i < 20 else val - 0.02, "val_per_source": {"a": val - 0.1, "b": val + 0.9},
                "bpb": val / 3, "bpb_per_source": {"a": val / 3 - 0.05, "b": val / 3 + 0.3},
                "grad_norm": None if i == 0 else 0.3 + 2.0 / i, "tok_s": None if i == 0 else 5000.0 + 10 * (i % 7),
                "peak_mem_gib": 6.5, "energy_wh": {"gpu": 100.0 * i, "cpu": 20.0 * i, "ram": 1.0 * i},
                "cost_gbp": 0.03 * i, "train_seconds": 40.0 * 200 * i,
                "time": f"2026-11-{1 + i // 6:02d}T{4 * (i % 6):02d}:00:00+00:00",
                "probe_acc": None if i == 0 else 0.25 + 0.002 * i + 0.01 * (i % 3),
                "probe_floor": None if i == 0 else 0.24 + 0.01 * (i % 2),
                "probe_by_category": None if i == 0 else {"genetics": 0.30 + 0.01 * (i % 4), "cells": 0.5,
                                                          "evolution": 0.2}})
        out.append(row)
    return out


TRAIN_CONFIG = {"steps": 20_000, "tokens_per_step": 1000, "warmup_steps": 400, "grad_clip": 1.0, "schedule": "wsd",
                "lr": 1e-3, "min_lr": 1e-5, "decay_frac": 0.2, "decay_shape": "1-sqrt", "optimizer": "muon"}
DATASET = {"path": "datasets/none", "fingerprint": "0" * 12, "train_tokens": {"a": 4_500_000, "b": 500_000}}


def toy_file(folder, name, history=None, layers=2, **extra):
    """A tinyGPT checkpoint file of a tiny model (QK-norm on) with the given evaluation history."""
    torch.manual_seed(0)
    cfg = tiny_gpt.ModelConfig(vocab_size=50, ctx=16, d_model=32, n_layers=layers, n_heads=4, n_kv_heads=2,
                               qk_norm=True)
    history = history_rows() if history is None else history
    obj = {"format": tiny_gpt.CHECKPOINT_FORMAT, "format_version": tiny_gpt.FORMAT_VERSION,
           "checkpoint_type": "inference", "model_config": dataclasses.asdict(cfg),
           "model": tiny_gpt.TinyGPT(cfg).state_dict(), "step": history[-1]["step"] if history else 7,
           "train_config": dict(TRAIN_CONFIG), "dataset": dict(DATASET),
           "metrics": {"history": history, "best_val": history[-1]["val_loss"] if history else None,
                       "best_step": history[-1]["step"] if history else None, "last_eval": None}}
    obj.update(extra)
    path = os.path.join(folder, name)
    torch.save(obj, path)
    return path


def open_ck(path):
    import view_pt
    with unittest.mock.patch.object(tiny_gpt, "tokenizer_from_checkpoint", return_value=None):
        return view_pt.Checkpoint(path)


def viewer_args(path, *flags):
    import view_pt
    with unittest.mock.patch.object(sys, "argv", ["view_pt.py", path, *flags]):
        return view_pt.parse_args()


def run_main(*argv):
    """view_pt.main() in this process on a toy file (which stores no tokenizer), with the GitHub push replaced
    by a mock: (printed text, the mock)."""
    import view_pt
    with unittest.mock.patch.object(sys, "argv", ["view_pt.py", *argv]), redirect_stdout(io.StringIO()) as buf, \
            unittest.mock.patch.object(tiny_gpt, "tokenizer_from_checkpoint", return_value=None), \
            unittest.mock.patch.object(view_pt, "push_dashboard") as push:
        view_pt.main()
    return buf.getvalue(), push


def draw(ck, args):
    """create_dashboard in this process: (printed text, the figures it drew, left open for inspection)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import view_pt
    plt.close("all")
    with unittest.mock.patch.object(plt, "close"), redirect_stdout(io.StringIO()) as buf:
        out = view_pt.create_dashboard(ck, args)
    return out, buf.getvalue(), [plt.figure(n) for n in plt.get_fignums()]


def texts_of(fig):
    import matplotlib.text
    return [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]


def png_size(path):
    with open(path, "rb") as handle:
        return struct.unpack(">II", handle.read(24)[16:24])


def one_axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(8, 5))


def said(ax):
    """Title and every text of a panel on one line (subtitles are wrapped)."""
    return " | ".join([ax.get_title(loc="left")] + [t.get_text() for t in ax.texts]).replace(chr(10), " ")


class Audit2Viewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2c_")

    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt
        plt.close("all")
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    # ---------------- F368 (and the repeated steps of F325) ----------------
    def test_rows_after_appends_only_what_continues_the_checkpoint(self):
        import view_pt
        history = [{"step": s, "val_loss": 5.0 - s / 1000} for s in (0, 200, 400)]
        path = os.path.join(self.tmp, "R_metrics.jsonl")
        first = [{"run_id": "one", "step": s, "val_loss": 5.0 - s / 1000} for s in (0, 200, 400, 600, 800)]
        second = [{"run_id": "two", "step": 600, "val_loss": 4.35}]
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in first + second) + '{"run_id": "two", "step": 800, "val_')
        newer = view_pt.rows_after(history, path)
        self.assertEqual([(r["step"], r["run_id"], r["val_loss"]) for r in newer], [(600, "two", 4.35)],
                         "one row per step, the resumed session's 600, the abandoned 800 and the torn line gone")
        self.assertEqual([r["step"] for r in view_pt.rows_after(history[:2], path)], [400, 600])
        self.assertEqual(view_pt.rows_after([{"step": 300, "val_loss": 4.7}], path), [], "no row at the last step")
        self.assertEqual(view_pt.rows_after([{"step": 400, "val_loss": 4.61}], path), [], "another run's file")
        self.assertEqual(view_pt.rows_after(history, os.path.join(self.tmp, "none_metrics.jsonl")), [])
        self.assertEqual(view_pt.rows_after([], path), [])
        # the dashboard and the summary of a best file use them; a fine-tuned copy of the same file does not
        rows = history_rows(8)
        best = toy_file(self.tmp, "J_best.pt", rows[:5])
        with open(os.path.join(self.tmp, "J_metrics.jsonl"), "w", encoding="utf-8") as f:
            f.write("".join(json.dumps({"run_id": "one", **r}) + "\n" for r in rows))
        ck = open_ck(best)
        self.assertEqual([r["step"] for r in view_pt.newer_rows(ck)], [1000, 1200, 1400])
        with redirect_stdout(io.StringIO()) as buf:
            view_pt.print_summary(ck)
        self.assertIn("5 evaluations stored (steps 0-800) + 3 newer in J_metrics.jsonl (to step 1400)", buf.getvalue())
        args = viewer_args(best, "--plot", "--no-push", "--dpi", "50", "--out", os.path.join(self.tmp, "J.png"))
        _, text, figs = draw(ck, args)
        self.assertNotIn("panel failed", text)
        self.assertIn("weights: best @800; curves to step 1,400", " ".join(t.get_text() for t in texts_of(figs[0])))
        tuned = toy_file(self.tmp, "J_sft_best.pt", rows[:5], finetune={"method": "sft", "base": "J_best.pt"})
        shutil.copy(os.path.join(self.tmp, "J_metrics.jsonl"), os.path.join(self.tmp, "J_sft_metrics.jsonl"))
        self.assertEqual(view_pt.newer_rows(open_ck(tuned)), [])

    # ---------------- F419, F422 ----------------
    def test_val_mix_of_recomputes_the_training_mix(self):
        import view_pt
        ck = types_ns(obj={"dataset": {"train_tokens": {"a": 900, "b": 100}}, "train_config": {}})
        row = {"val_per_source": {"a": 2.0, "b": 4.0}}
        self.assertAlmostEqual(view_pt.val_mix_of(row, ck), 2.2, places=12)
        self.assertEqual(view_pt.val_mix_of({**row, "val_mix": 2.5}, ck), 2.5, "a stored value wins")
        self.assertAlmostEqual(view_pt.val_mix_of({"val_per_source": {"a": 2.0, "b": 4.0, "c": 9.0}}, ck), 2.2,
                               places=12, msg="a source the dataset does not know counts 0 (evaluate's mix_mean)")
        self.assertIsNone(view_pt.val_mix_of({"val_loss": 3.0}, ck))
        self.assertIsNone(view_pt.val_mix_of(row, types_ns(obj={"train_config": {}})), "no dataset in the file")
        self.assertIsNone(view_pt.val_mix_of({"val_per_source": {"c": 9.0}}, ck), "share sum 0")
        odd = types_ns(obj={"dataset": {"train_tokens": {"a": 900, "b": 100}}, "train_config": {"mix": "zzz=2"}})
        self.assertIsNone(view_pt.val_mix_of(row, odd), "a mix naming an unknown source")
        mixed = types_ns(obj={"dataset": {"train_tokens": {"a": 900, "b": 100}}, "train_config": {"mix": "b=9"}})
        self.assertAlmostEqual(view_pt.val_mix_of(row, mixed), 3.0, places=12)
        self.assertEqual(view_pt.source_shares(mixed), {"a": 0.5, "b": 0.5})
        self.assertEqual(view_pt.gap_points([{"step": 1, "val_mix": 2.71, "train_loss": 2.73}, {"step": 2, "train_loss": 2.7},
                                             {"step": 3, "val_mix": 2.6, "train_loss": None}], types_ns(obj={})),
                         [(1, 2.71 - 2.73)])
        # the loss panel still draws for the made-up mix: training and the balanced validation only
        path = toy_file(self.tmp, "M.pt", history_rows(12), train_config={**TRAIN_CONFIG, "mix": "zzz=2"})
        ck = open_ck(path)
        for h in ck.history:
            h["val_mix"] = None
        fig, ax = one_axes()
        view_pt.panel_loss(ax, ck.history, ck)
        self.assertEqual(len(ax.get_lines()), 2)

    # ---------------- F419, F420, F421 ----------------
    def test_loss_panel_axes_lines_and_marks(self):
        import view_pt
        rows = history_rows(60)
        for h in rows:
            h["val_per_source"] = {f"s{k}": h["val_loss"] + k / 10 for k in range(10)}
        ck = open_ck(toy_file(self.tmp, "L.pt", rows))
        fig, ax = one_axes()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            view_pt.panel_loss(ax, ck.history, ck)
            fig.canvas.draw()
        self.assertLessEqual(ax.get_ylim()[1], 4.0)
        self.assertLess(ax.get_ylim()[1] - ax.get_ylim()[0], 2)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertLessEqual(len(ax.get_lines()), 3)
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertFalse([name for name in labels if name.startswith("validation: ")], labels)
        self.assertEqual([name.split(",")[0].split(" (")[0] for name in labels], ["training", "validation", "validation"])
        self.assertIn("training mix", labels[1])
        self.assertIn("picks the best", labels[2])
        self.assertIn("tokens", ax.get_xlabel())
        self.assertIn("planned end", said(ax))
        self.assertIn("corpus size", said(ax))
        self.assertIn("above 4", said(ax))
        view_pt.finalize_legends(fig)
        # the mix line of a row without a stored value is the share-weighted mean of its per-source losses
        want = sum(0.1 * (rows[30]["val_loss"] + k / 10) for k in range(10))  # equal shares: no source is in the dataset
        ck.obj["dataset"]["train_tokens"] = {f"s{k}": 100 for k in range(10)}
        self.assertAlmostEqual(view_pt.val_mix_of({**rows[30], "val_mix": None}, ck), want, places=9)
        # a young run: three evaluations, step axis, every point inside the limits
        fig, ax = one_axes()
        view_pt.panel_loss(ax, ck.history[:3], ck)
        self.assertEqual(ax.get_xscale(), "linear")
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        for line in ax.get_lines():
            for x, y in line.get_xydata():
                self.assertTrue(x0 <= x <= x1 and y0 <= y <= y1, (x, y, ax.get_xlim(), ax.get_ylim()))
        self.assertIn("step", ax.get_xlabel())
        # no token counts: the step axis; no planned step count: no end mark
        fig, ax = one_axes()
        stepped = open_ck(toy_file(self.tmp, "L2.pt", history_rows(60, tokens=False)))
        view_pt.panel_loss(ax, stepped.history, stepped)
        self.assertIn("step", ax.get_xlabel())
        self.assertEqual(ax.get_xscale(), "linear")
        fig, ax = one_axes()
        ck.obj["train_config"].pop("steps")
        view_pt.panel_loss(ax, ck.history, ck)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertNotIn("planned end", said(ax))

    # ---------------- F421: the cut at 4.0 never hides where the run is now ----------------
    def test_loss_panel_shows_a_run_that_is_back_above_the_cut(self):
        import view_pt

        def drawn(rows):
            ck = open_ck(toy_file(self.tmp, "D.pt", rows))
            fig, ax = one_axes()
            view_pt.panel_loss(ax, ck.history, ck)
            return ax

        def back_up(rows):
            for h in rows:
                h.update(train_loss=6.0, val_loss=6.0, val_mix=6.0)
            return rows

        # the loss comes down to 2.9, then jumps to 6.0 and stays there: the whole curve, as before the cut existed
        rows = history_rows(80)
        ax = drawn(rows[:30] + back_up(rows[30:]))
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        self.assertGreater(y1, 6.0)
        self.assertEqual(len(ax.get_lines()), 3)
        for line in ax.get_lines():
            x, y = line.get_xydata()[-1]
            self.assertTrue(x0 <= x <= x1 and y0 <= y <= y1, (line.get_label(), x, y, ax.get_xlim(), ax.get_ylim()))
        self.assertNotIn("not shown", said(ax))
        self.assertIn("validation now 6.000", said(ax))
        # ten evaluations above the cut and back under it: the cut stays, and the subtitle counts them
        rows = history_rows(80)
        ax = drawn(rows[:30] + back_up(rows[30:40]) + rows[40:])
        self.assertLessEqual(ax.get_ylim()[1], 4.0)
        self.assertIn("17 evaluations above 4 not shown", said(ax), "7 of the first descent and the 10 later ones")

    # ---------------- F395, F411, F424, F367 ----------------
    def test_history_panels_draw_what_exists_and_say_what_is_missing(self):
        import view_pt
        log = os.path.join(self.tmp, "hist_log.txt")
        with open(log, "w", encoding="utf-8") as f:
            f.write("\n" + "=" * 30 + "\nRUN 1\ncheckpoints: /x/H_log.pt (best /x/H_log_best.pt)\n" + "".join(
                f"step {s:>5}/9000 | train {5 - s / 3000:.4f} | val {5.1 - s / 3000:.4f} @{s} | lr 1.0e-03\n"
                for s in range(200, 9001, 200)))
        full = history_rows(60)
        thin = history_rows(60, tokens=False, full=False)
        odd = history_rows(60)
        for i, h in enumerate(odd):
            h["grad_norm"] = (float("nan"), float("inf"), -1.0, 0.0)[i % 4]
        files = {"full": toy_file(self.tmp, "H.pt", full), "thin": toy_file(self.tmp, "H_thin.pt", thin),
                 "odd": toy_file(self.tmp, "H_odd.pt", odd), "log": toy_file(self.tmp, "H_log.pt", [], step=9000)}
        panels = {"gap": view_pt.panel_gap, "grad": view_pt.panel_grad_norm, "sources": view_pt.panel_sources,
                  "improvement": view_pt.panel_improvement, "speed": view_pt.panel_speed, "plan": view_pt.panel_plan,
                  "loss": view_pt.panel_loss}
        shown = {}
        for kind, path in files.items():
            ck = open_ck(path)
            args = viewer_args(path, "--plot", "--history-log", log) if kind == "log" else viewer_args(path, "--plot")
            history = view_pt.dashboard_history(ck, args)
            self.assertTrue(history, kind)
            for name, panel in panels.items():
                fig, ax = one_axes()
                panel(ax, history, ck)
                shown[kind, name] = said(ax)
            for name, panel in (("facts", view_pt.panel_facts), ("subjects", view_pt.panel_facts_subjects)):
                fig, ax = one_axes()
                panel(ax, history, ck, args)
                shown[kind, name] = said(ax)
            import matplotlib.pyplot as plt
            plt.close("all")
        self.assertEqual([h["step"] for h in view_pt.dashboard_history(open_ck(files["log"]), viewer_args(
            files["log"], "--plot", "--history-log", log))][:2], [200, 400])
        for kind in ("thin", "log"):
            self.assertIn("no gradient norms", shown[kind, "grad"])
            self.assertIn("no training speed", shown[kind, "speed"])
            self.assertIn("training mix", shown[kind, "gap"])
            self.assertIn("per-source", shown[kind, "sources"])
            self.assertIn("--panels all", shown[kind, "facts"])
            self.assertIn("step 11,800", shown["thin", "plan"])
        self.assertIn("no gradient norms", shown["odd", "grad"], "NaN, inf, negative and zero are not drawn")
        self.assertIn("before clipping", shown["full", "grad"])
        self.assertIn("worse on held-out text", shown["full", "gap"])
        self.assertIn("between evaluations", shown["full", "speed"])
        self.assertIn("kWh", shown["full", "speed"])
        self.assertIn("6.5 GiB", shown["full", "speed"])
        self.assertIn("of 20,000", shown["full", "plan"])
        self.assertIn("epoch", shown["full", "plan"])
        self.assertIn("h of training left", shown["full", "plan"])
        self.assertIn("decay starts at step 16,000", shown["full", "plan"])
        self.assertIn("done about 2026-", shown["full", "plan"], "the time stamps span more than 7 days")
        for key, text in shown.items():
            self.assertNotIn("panel failed", text, key)
        # speed: one point per evaluation that has a speed; improvement: the two 1,000-step means
        fig, ax = one_axes()
        ck = open_ck(files["full"])
        view_pt.panel_speed(ax, [{"step": 0, "tok_s": None}, {"step": 200, "tok_s": 5000.0}, {"step": 400, "tok_s": 7000.0}],
                            ck)
        self.assertEqual(len(ax.collections[0].get_offsets()), 2)
        self.assertEqual(ax.get_ylim()[0], 0)
        pts = [(s, 10.0 - s / 1000) for s in range(0, 4001, 500)]
        got = dict(view_pt.improvement_points(pts, start=400))
        self.assertEqual(sorted(got), [2500, 3000, 3500, 4000], "from 2,000 steps after the warm-up")
        self.assertAlmostEqual(got[3000], 1.0, places=12)
        self.assertEqual(view_pt.improvement_points(pts[:3]), [])
        # per 1,000 steps whatever --eval-every was: a loss that falls 1.0 per 1,000 steps gives 1.0
        for every in (200, 600, 800, 1000, 1500, 2000, 5000):
            line = [(s, 100.0 - s / 1000) for s in range(0, 60_001, every)]
            got = view_pt.improvement_points(line, start=400)
            self.assertGreaterEqual(len(got), 8, every)
            for s, v in got:
                self.assertAlmostEqual(v, 1.0, places=9, msg=(every, s))
        self.assertEqual(view_pt.improvement_width(pts), 1000)
        self.assertEqual(view_pt.improvement_width(line), 5000, "evaluations further apart than 1,000 steps")
        sparse = [dict(h, step=10 * h["step"], val_loss=100.0 - h["step"] / 100, val_mix=None, val_per_source=None)
                  for h in full]  # 2,000 steps apart, 1.0 down per 1,000 steps
        fig, ax = one_axes()
        view_pt.panel_improvement(ax, sparse, ck)
        self.assertNotIn("Drawn once", said(ax))
        self.assertIn("of the 2,000 steps before, minus that of the last 2,000, per 1,000 steps", said(ax))
        self.assertEqual(len(ax.get_lines()), 1)
        fig, ax = one_axes()
        view_pt.panel_improvement(ax, sparse[:5], ck)
        self.assertIn("at or after step 4,400", said(ax), "two 2,000-step windows after the warm-up of 400")
        plan = dict(view_pt.plan_lines(sparse, ck))
        self.assertIn("improving 1.0000 per 1,000 steps", plan)
        self.assertRegex(plan["improving 1.0000 per 1,000 steps"], r"^\d+\.\d{3} per 24 h of training$")
        # the Plan line of a validation loss that falls, rises, or does not move at all (no "-0.0e+00")
        for slope, want in ((-0.5, "improving 0.5000 per 1,000 steps"),
                            (0.5, "not improving (validation up 0.5000 per 1,000 steps)"),
                            (0.0, "not improving (validation flat: 0 per 1,000 steps)")):
            moved = [dict(h, val_loss=50.0 + slope * h["step"] / 1000) for h in full]
            self.assertEqual([head for head, _ in view_pt.plan_lines(moved, ck) if "improving" in head], [want])
        # by source: the last row's bits per byte, one bar each, the share in the label
        fig, ax = one_axes()
        view_pt.panel_sources(ax, full, ck)
        self.assertEqual(sorted(p.get_width() for p in ax.patches), sorted(full[-1]["bpb_per_source"].values()))
        self.assertIn("bits per byte", ax.get_xlabel())
        self.assertEqual(sorted(t.get_text() for t in ax.get_yticklabels()),
                         ["a (90.0% of training)", "b (10.0% of training)"])
        fig, ax = one_axes()
        view_pt.panel_sources(ax, [{"step": 1, "val_per_source": {"a": 2.0, "b": 3.0}}], ck)
        self.assertIn("nats", ax.get_xlabel())
        fig, ax = one_axes()
        view_pt.panel_sources(ax, [{"step": 1, "val_per_source": {"a": 2.0}}], ck)
        self.assertIn("one source", said(ax))

    # ---------------- F396, F399, F398, F430 ----------------
    def test_default_picture_needs_no_model_and_is_legible(self):
        import view_pt
        path = toy_file(self.tmp, "D_best.pt")
        ck = open_ck(path)
        png = os.path.join(self.tmp, "D.png")
        args = viewer_args(path, "--plot", "--no-push", "--out", png)
        self.assertEqual((args.panels, args.plot_tokens), ("history", False))
        boom = unittest.mock.Mock(side_effect=AssertionError("the default picture must not run the model"))
        with unittest.mock.patch.multiple(view_pt, build_model=boom, eval_windows=boom, forward_stats=boom):
            out, text, figs = draw(ck, args)
        self.assertEqual(out, png)
        self.assertNotIn("panel failed", text)
        self.assertEqual(text.count("Saved"), 1)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "D_tokens.png")))
        self.assertEqual(png_size(png), (3840, 2160))
        fig, = figs
        small = [(t.get_text()[:40], t.get_fontsize()) for t in texts_of(fig) if t.get_fontsize() < 9.5]
        self.assertFalse(small, small)
        titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
        self.assertEqual([t.split(":")[0] for t in titles],
                         ["Loss", "Validation minus training", "Gradient norm", "Validation by source", "Biology facts",
                          "Biology facts by subject", "Improvement per 1,000 steps", "Speed", "Plan"])
        for ax in fig.axes:
            if ax.get_title(loc="left"):
                self.assertGreaterEqual(ax.get_position().width * fig.get_figwidth(), 4, ax.get_title(loc="left"))
        said_all = " ".join(t.get_text() for t in texts_of(fig))
        self.assertNotIn("optimizer state is saved only", said_all)
        self.assertIn("D | step 11,800 of 20,000 (59%) | val 2.900", said_all)
        self.assertIn("drawn 20", said_all)
        # style_axes without a size keeps benchmark.py's 8 pt tick labels
        f2, ax = one_axes()
        view_pt.style_axes(ax)
        f2.canvas.draw()
        self.assertEqual({t.get_fontsize() for t in ax.get_xticklabels() + ax.get_yticklabels()}, {8.0})
        # a file without a history says where the other panels are
        bare = toy_file(self.tmp, "D_bare.pt", [])
        _, text, _ = draw(open_ck(bare), viewer_args(bare, "--plot", "--no-push", "--dpi", "50"))
        self.assertIn("weight and behaviour panels: add --panels all", text)
        self.assertNotIn("panel failed", text)
        # a fine-tuned file: its own curve in the loss panel, the base model's history in the others, the same line
        tuned = toy_file(self.tmp, "D_sft.pt", finetune={"method": "sft", "base": "D_best.pt", "history": [
            {"epoch": 1, "train_loss": 1.2, "val_loss": 1.3}, {"epoch": 2, "train_loss": 1.0, "val_loss": 1.25}]})
        _, text, figs = draw(open_ck(tuned), viewer_args(tuned, "--plot", "--no-push", "--dpi", "50"))
        self.assertIn("weight and behaviour panels: add --panels all", text)
        self.assertNotIn("panel failed", text)
        said_all = " ".join(t.get_text() for t in texts_of(figs[0]))
        self.assertIn("Fine-tuning loss (SFT)", said_all)
        self.assertIn("base model's pre-training", said_all)

    def test_all_panels_keep_the_wide_picture_and_drop_the_dead_ones(self):
        import view_pt
        self.assertFalse(hasattr(view_pt, "panel_ppl") or hasattr(view_pt, "panel_adam")
                         or hasattr(view_pt, "panel_confidence") or hasattr(view_pt.Checkpoint, "adam_names"))
        self.assertEqual(view_pt.BASE_STYLE["legend.fontsize"], 7)
        for name, extra in (("W.pt", {"checkpoint_type": "training", "optimizer": {"state": {}, "param_groups": []}}),
                            ("W_best.pt", {})):
            path = toy_file(self.tmp, name, history_rows(30), **extra)
            png = os.path.join(self.tmp, name + ".png")
            out, text, figs = draw(open_ck(path), viewer_args(path, "--plot", "--panels", "all", "--no-forward",
                                                               "--no-push", "--out", png))
            self.assertNotIn("panel failed", text)
            fig, = figs
            titles = [ax.get_title(loc="left").split(":")[0] for ax in fig.axes if ax.get_title(loc="left")]
            self.assertEqual(len(titles), 16, titles)
            self.assertEqual(titles[9:], ["Learning rate", "Calibration of the top prediction", "Weight scale per block",
                                          "RMSNorm gains by depth", "Singular value spectra",
                                          "Per-head projection scale", "Attention entropy"])
            said_all = " ".join(t.get_text() for t in texts_of(fig))
            for banned in BANNED_TITLES + ("optimizer state is saved only", "How opinionated"):
                self.assertNotIn(banned, said_all)
            self.assertEqual(png_size(png), (3900, 3450))
        # both pictures in one call, without a forward pass; only the dashboard's path is returned (and pushed)
        path = os.path.join(self.tmp, "W_best.pt")
        text, push = run_main(path, "--plot", "--plot-tokens", "--no-forward", "--dpi", "50", "--out",
                              os.path.join(self.tmp, "both.png"))
        self.assertNotIn("panel failed", text)
        self.assertEqual(text.count("Saved"), 2)
        self.assertEqual(push.call_args[0][0], os.path.join(self.tmp, "both.png"), "the dashboard, not the tokens")
        self.assertEqual(png_size(os.path.join(self.tmp, "both.png"))[0], 1280)
        self.assertEqual(png_size(os.path.join(self.tmp, "both_tokens.png"))[0], 1300)
        text, push = run_main(path, "--plot-tokens", "--no-forward", "--dpi", "50")
        self.assertEqual(text.count("Saved"), 1)
        push.assert_not_called()  # the token picture is never pushed
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "W_best_tokens.png")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "W_best_dashboard.png")))

    # ---------------- F398, F436 ----------------
    def test_header_names_the_run_and_whose_numbers_they_are(self):
        import view_pt
        rows = history_rows(30)
        ck = open_ck(toy_file(self.tmp, "A_best.pt", rows))
        line1, line2 = view_pt.dashboard_header(ck)
        self.assertTrue(line1.startswith("A | step 5,800 of 20,000 (29%) | val "), line1)
        self.assertIn("(best ", line1)
        self.assertIn("bits/byte", line1)
        self.assertRegex(line1, r"facts \d+% \(floor \d+%\) \| drawn \d{4}-\d\d-\d\d \d\d:\d\d$")
        self.assertIn("decoder-only Transformer", line2)
        self.assertRegex(line2, r"[\d,]+ parameters")
        generic = os.path.join(self.tmp, "generic.pt")
        torch.save({"w": torch.zeros(2, 2)}, generic)
        g = view_pt.Checkpoint(generic)
        self.assertEqual(view_pt.dashboard_header(g), (g.model_name, ""))
        tuned = open_ck(toy_file(self.tmp, "A_sft_best.pt", rows, finetune={"method": "sft", "base": "/x/A_best.pt"}))
        line1, line2 = view_pt.dashboard_header(tuned)
        self.assertIn("step 5,800 of 20,000 (29%) (base model's pre-training)", line1)
        self.assertIn("fine-tuned (SFT) from A_best.pt", line2)
        self.assertEqual(view_pt.derived_note(tuned), " (base model, before fine-tuning)")
        edited = open_ck(toy_file(self.tmp, "A_edit.pt", rows, edit={"chosen": {"kind": "neurons"}}))
        self.assertEqual(view_pt.derived_note(edited), " (original model, before the edit)")
        self.assertEqual(view_pt.derived_note(ck), "")
        self.assertIn("(original model, before the edit)", view_pt.dashboard_header(edited)[0])
        self.assertIn("original model, before the edit", view_pt._history_sub(edited, "x"))
        self.assertIn("base model pre-training (before SFT)", view_pt._history_sub(tuned, "x"))
        with redirect_stdout(io.StringIO()) as buf:
            view_pt.print_summary(edited)
        lines = {line.split(":")[0]: line for line in buf.getvalue().splitlines()}
        self.assertIn("best validation (original model, before the edit)", lines)
        self.assertIn("history (original model, before the edit)", lines)
        with redirect_stdout(io.StringIO()) as buf:
            view_pt.print_summary(tuned)
        self.assertIn("best validation (base model, before fine-tuning): loss", buf.getvalue())
        home = os.path.expanduser("~")
        hidden = open_ck(toy_file(self.tmp, "A_home.pt", rows, finetune={"method": "sft", "base": home + "/A_best.pt"}))
        self.assertNotIn(home, " ".join(view_pt.dashboard_header(hidden)))

    # ---------------- F417, F432 ----------------
    def test_fact_panels_smooth_the_curve_and_colour_only_clear_subjects(self):
        import view_pt
        from matplotlib.colors import to_rgba
        rows = history_rows(13)
        ck = open_ck(toy_file(self.tmp, "F.pt", rows))
        args = viewer_args(ck.path, "--plot")
        fig, ax = one_axes()
        view_pt.panel_facts(ax, rows, ck, args)
        acc = np.array([h["probe_acc"] for h in rows[1:]])
        want = np.array([acc[max(0, i - 4):i + 1].mean() for i in range(len(acc))])
        self.assertTrue(any(len(line.get_ydata()) == len(want) and np.allclose(line.get_ydata(), want)
                            for line in ax.get_lines()), "the 5-evaluation trailing mean of the accuracy")
        self.assertLess(ax.get_ylim()[1], 1)
        labels = " ".join(str(line.get_label()) for line in ax.get_lines())
        self.assertNotIn("confidence", labels)
        self.assertNotIn("top-1", labels)
        self.assertIn("by chance", said(ax))
        self.assertRegex(ax.get_title(loc="left"), r"^Biology facts: \d+% right, floor \d+%$")
        fig, ax = one_axes()
        view_pt.panel_facts(ax, rows[:4], ck, args)
        self.assertTrue(any(np.allclose(line.get_ydata(), acc[:3]) for line in ax.get_lines() if len(line.get_ydata()) == 3),
                        "fewer than 5 evaluations: the raw values")
        fig, ax = one_axes()
        older = [{k: v for k, v in h.items() if k != "probe_floor"} for h in rows]
        view_pt.panel_facts(ax, older, ck, args)
        self.assertRegex(ax.get_title(loc="left"), r"^Biology facts: \d+% right$")
        # scores a few points or less apart (and no chance line to widen the axis): no y label appears twice
        for spread in (0.0, 0.001, 0.02, 0.3):
            close = [dict(h, probe_acc=0.15 + spread * (i % 3) / 2, probe_floor=None) for i, h in enumerate(rows)]
            fig, ax = one_axes()
            view_pt.panel_facts(ax, close, ck, args)
            fig.canvas.draw()
            y0, y1 = ax.get_ylim()
            ticks = [t.get_text() for t, v in zip(ax.get_yticklabels(), ax.get_yticks()) if y0 <= v <= y1]
            self.assertGreaterEqual(len(ticks), 2, spread)
            self.assertEqual(len(set(ticks)), len(ticks), (spread, ticks))
            self.assertTrue(all(t.endswith("%") for t in ticks), ticks)
        self.assertIn("20%", ticks, "whole percents stay without decimals")
        tuned = open_ck(toy_file(self.tmp, "F_sft.pt", rows, finetune={"method": "sft", "base": "F.pt"}))
        fig, ax = one_axes()
        view_pt.panel_facts(ax, rows, tuned, args)
        self.assertIn("base model", said(ax))
        self.assertIn("--panels all", said(ax))
        # no per-evaluation scores: the stored report's single score with its interval, else where to get one
        sha = tiny_gpt.file_sha256(args.probes)
        report = {"file_sha256": sha, "n": 100, "correct": 40, "chance": 0.25, "floor": 0.3,
                  "category": {"genetics": {"n": 50, "correct": 30, "accuracy": 0.6, "low": 0.4, "high": 0.75,
                                            "floor": 0.5, "chance": 0.25},
                               "cells": {"n": 30, "correct": 24, "accuracy": 0.8, "low": 0.62, "high": 0.9,
                                         "floor": 0.4, "chance": 0.25},
                               "evolution": {"n": 20, "correct": 2, "accuracy": 0.1, "low": 0.03, "high": 0.3,
                                             "floor": 0.5, "chance": 0.25}},
                  "difficulty": {"easy": {"n": 100, "correct": 40, "accuracy": 0.4, "low": 0.3, "high": 0.5,
                                          "floor": 0.3, "chance": 0.25}}}
        stored = open_ck(toy_file(self.tmp, "F_stored.pt", [dict(h, probe_acc=None, probe_by_category=None) for h in rows]))
        stored.metrics["last_eval"] = {"step": stored.obj["step"], "probes": report}
        fig, ax = one_axes()
        view_pt.panel_facts(ax, stored.history, stored, args)
        self.assertIn("40% correct (31%-50%)", said(ax))
        fig, ax = one_axes()
        view_pt.panel_facts(ax, stored.history, ck, args)
        self.assertIn("--panels all", said(ax))
        # by subject: the mean of the last five evaluations; a colour only where the stored interval excludes the floor
        ck.metrics["last_eval"] = {"step": ck.obj["step"], "probes": report}
        fig, ax = one_axes()
        view_pt.panel_facts_subjects(ax, rows, ck, args)
        bars = {t.get_text().split(" (")[0]: p for t, p in zip(ax.get_yticklabels(), ax.patches)}
        self.assertAlmostEqual(bars["genetics"].get_width(), np.mean([h["probe_by_category"]["genetics"] for h in rows[-5:]]))
        colours = {name: p.get_facecolor()[:3] for name, p in bars.items()}
        self.assertEqual(colours["genetics"], to_rgba(view_pt.OTHER)[:3], "0.6 against a floor of 0.5, interval from 0.4")
        self.assertEqual(colours["cells"], to_rgba(view_pt.GOOD)[:3])
        self.assertEqual(colours["evolution"], to_rgba(view_pt.CRITICAL)[:3])
        fig, ax = one_axes()
        view_pt.panel_probes(ax, ck, None, None, args)
        scored = {t.get_text().split(" (")[0]: p.get_facecolor()[:3] for t, p in zip(ax.get_yticklabels(), ax.patches)}
        self.assertEqual(scored, colours, "the scored panel uses the same rule")
        self.assertEqual(view_pt.floor_colour({"low": 0.4, "high": 0.75, "floor": 0.5}), view_pt.OTHER)
        self.assertEqual(view_pt.floor_colour({"low": 0.4, "high": 0.75, "floor": None}), view_pt.OTHER)
        # F437: a best file that holds the EMA average. Its history rows are the raw weights' evaluations and so is
        # the stored report, so the two history panels use it; as this file's own score it is still refused
        ema = open_ck(toy_file(self.tmp, "F_ema_best.pt", rows, weights="ema", ema_decay=0.999))
        ema.metrics["last_eval"] = {"step": ema.obj["step"], "probes": report}
        self.assertIsNone(view_pt.probe_report(ema, None, None, args))
        self.assertIsNone(view_pt.stored_probes(ema, args))
        self.assertIs(view_pt.stored_probes(ema, args, raw_weights=True), report)
        fig, ax = one_axes()
        view_pt.panel_facts(ax, rows, ema, args)
        self.assertIn("chance 25%", said(ax))
        self.assertIn("easy 40%", said(ax))
        fig, ax = one_axes()
        view_pt.panel_facts_subjects(ax, rows, ema, args)
        self.assertEqual({t.get_text().split(" (")[0]: p.get_facecolor()[:3] for t, p in zip(ax.get_yticklabels(), ax.patches)},
                         colours)
        self.assertIn("genetics (n=50)", [t.get_text() for t in ax.get_yticklabels()])
        self.assertIn("clear of the subject's floor", said(ax))
        fig, ax = one_axes()
        view_pt.panel_facts(ax, stored.history, ema, args)
        self.assertNotIn("40% correct", said(ax), "without a curve the single stored score would pass as the average's")
        self.assertIn("--panels all", said(ax))
        for change in ({"finetune": {"method": "sft", "base": "F.pt"}}, {"edit": {"chosen": {}}}):
            other = open_ck(toy_file(self.tmp, "F_other.pt", rows, **change))
            other.metrics["last_eval"] = {"step": other.obj["step"], "probes": report}
            self.assertIsNone(view_pt.stored_probes(other, args, raw_weights=True), change)

    # ---------------- F428, F429, F414, F437, F392 ----------------
    def test_weight_and_behaviour_panels(self):
        import view_pt
        ck = open_ck(toy_file(self.tmp, "G.pt", history_rows(5), layers=18))
        fig, ax = one_axes()
        view_pt.panel_norm_gains(ax, ck)
        fig.canvas.draw()
        self.assertEqual([line.get_label() for line in ax.get_lines() if not line.get_label().startswith("_")],
                         ["attn_norm", "ffn_norm", "q_norm", "k_norm"])
        self.assertLessEqual(len([t for t in ax.get_xticklabels() if t.get_text()]), 20)
        self.assertIn("final norm", said(ax))
        self.assertEqual(ax.get_ylim()[0], 0, "the gains are read from zero")
        label = next(t for t in ax.texts if t.get_text() == "final norm")
        self.assertLess(label.get_window_extent().y1, ax.get_window_extent().y1, "the label stays inside the axes")
        profile = view_pt.norm_gain_profiles(ck)
        self.assertEqual([len(profile[k]) for k in ("attn_norm", "ffn_norm", "q_norm", "k_norm")], [18] * 4)
        ck.weights["blocks.3.ffn_norm.weight"] = ck.weights["blocks.3.ffn_norm.weight"] - 1.5
        fig, ax = one_axes()
        view_pt.panel_norm_gains(ax, ck)
        self.assertLess(ax.get_ylim()[0], -0.5, "a gain below zero is still inside the picture")
        ck.weights["blocks.3.ffn_norm.weight"] = ck.weights["blocks.3.ffn_norm.weight"] + 1.5
        # calibration: the run's own evaluation of this step wins over the 16-window forward pass
        cal = {"confidence": [0.1, 0.9], "accuracy": [0.1, 0.8], "count": [10.0, 30.0]}
        fstats = {"calibration": {"confidence": [0.2, 0.7], "accuracy": [0.2, 0.7], "count": [5.0, 5.0]}, "ece": 0.111,
                  "tokens": 16384}
        ck.metrics["last_eval"] = {"step": ck.obj["step"], "calibration": cal, "ece": 0.777}
        fig, ax = one_axes()
        view_pt.panel_calibration(ax, ck, fstats, "")
        self.assertIn("0.777", said(ax))
        self.assertIn(f"evaluation at step {ck.obj['step']:,}", said(ax))
        for change in ({"finetune": {"method": "sft"}}, {"edit": {"chosen": {}}}, {"weights": "ema"}):
            other = open_ck(toy_file(self.tmp, "G2.pt", history_rows(5), **change))
            other.metrics["last_eval"] = {"step": other.obj["step"], "calibration": cal, "ece": 0.777}
            fig, ax = one_axes()
            view_pt.panel_calibration(ax, other, fstats, "")
            self.assertIn("0.111", said(ax), change)
            self.assertIn("forward pass", said(ax), change)
            fig, ax = one_axes()
            view_pt.panel_calibration(ax, other, None, "")
            self.assertIn("0.777", said(ax), change)
        self.assertIn("(raw weights; this file holds the EMA average)", ax.get_title(loc="left"))
        ck.metrics["last_eval"]["step"] = 1  # an older evaluation: the forward pass
        fig, ax = one_axes()
        view_pt.panel_calibration(ax, ck, fstats, "")
        self.assertIn("0.111", said(ax))
        # an EMA best file: the stored fact benchmark and latest validation are the raw weights'
        sha = tiny_gpt.file_sha256(tiny_gpt.DEFAULT_PROBES)
        stored = {"file_sha256": sha, "accuracy": 0.5}
        args = types_ns(probes=tiny_gpt.DEFAULT_PROBES)
        raw = types_ns(metrics={"last_eval": {"step": 7, "probes": stored}}, obj={"step": 7}, finetune=None, edit=None)
        self.assertIs(view_pt.probe_report(raw, None, None, args), stored)
        raw.obj["weights"] = "ema"
        self.assertIsNone(view_pt.probe_report(raw, None, None, args), "the average is scored, not looked up")
        ema = open_ck(toy_file(self.tmp, "G_ema_best.pt", history_rows(5), weights="ema", ema_decay=0.999))
        ema.metrics["last_eval"] = {"loss": 2.5, "step": ema.obj["step"]}
        with redirect_stdout(io.StringIO()) as buf:
            view_pt.print_summary(ema)
        self.assertIn("latest validation (raw weights; this file holds the EMA average): loss 2.5", buf.getvalue())
        # the residual stream's size by depth comes out of the attention-entropy pass
        torch.manual_seed(1)
        cfg = tiny_gpt.ModelConfig(vocab_size=50, ctx=16, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2)
        model = tiny_gpt.TinyGPT(cfg).eval()
        window = torch.randint(0, 50, (17,), generator=torch.Generator().manual_seed(0))
        plain = view_pt.attention_entropy(model, window, torch.device("cpu"))
        ent, rms = view_pt.attention_entropy(model, window, torch.device("cpu"), with_rms=True)
        self.assertTrue(np.array_equal(plain, ent))
        self.assertEqual(len(rms), cfg.n_layers + 1)
        self.assertTrue(np.all(np.isfinite(rms)) and np.all(rms > 0), rms)
        fig, ax = one_axes()
        view_pt.panel_attention(ax, model, window[None], torch.device("cpu"),
                                __import__("matplotlib").colormaps["Blues"], "")
        self.assertIn("residual stream RMS", said(ax))

    # ---------------- F404 ----------------
    def test_history_from_log_starts_again_at_a_fresh_run_of_the_same_name(self):
        import view_pt
        bar = "=" * 100

        def session(run_id, note, *evals):
            return (f"\n{bar}\nRUN {run_id} started 2026-01-01T00:00:00+00:00 | {note}\n"
                    "checkpoints: C:\\x\\A.pt (best C:\\x\\A_best.pt)\n"
                    + "".join(f"step {s:5d}/400 | train {v:.4f} | val {v:.4f} @{s} | lr 1.0e-03\n" for s, v in evals))

        log = os.path.join(self.tmp, "overwritten_log.txt")
        with open(log, "w", encoding="utf-8") as f:
            f.write(session("a", "fresh run", (100, 9.1), (200, 8.1), (300, 7.1))
                    + session("b", "fresh run", (200, 5.1), (400, 4.1)))
        hist, label = view_pt.history_from_log(log, "A_best.pt", 400, "tinygpt")
        self.assertEqual([(h["step"], h["val_loss"]) for h in hist], [(200, 5.1), (400, 4.1)],
                         "the run that --overwrite replaced is not part of this curve")
        self.assertIn("1 session(s) of A", label)
        # a resumed session still continues the curve; a run begun with --init-from is a new one
        with open(log, "a", encoding="utf-8") as f:
            f.write(session("c", "RESUMED C:\\x\\A.pt at step 400", (600, 3.1)))
        hist, label = view_pt.history_from_log(log, "A.pt", None, "tinygpt")
        self.assertEqual([(h["step"], h["val_loss"]) for h in hist], [(200, 5.1), (400, 4.1), (600, 3.1)])
        self.assertIn("2 session(s) of A", label)
        with open(log, "a", encoding="utf-8") as f:
            f.write(session("d", "initialised from B.pt (tinygpt training, step 9); best/last validation reset "
                                 "because the validation set is new", (100, 2.1)))
        hist, label = view_pt.history_from_log(log, "A.pt", None, "tinygpt")
        self.assertEqual([(h["step"], h["val_loss"]) for h in hist], [(100, 2.1)])
        self.assertIn("1 session(s) of A", label)
        # the fresh start counts also when that session was stopped before its first evaluation and then resumed
        early = os.path.join(self.tmp, "stopped_early_log.txt")
        with open(early, "w", encoding="utf-8") as f:
            f.write(session("a", "fresh run", (100, 9.1), (200, 8.1), (300, 7.1)) + session("b", "fresh run")
                    + "step     0/400 | baseline before training | val 9.900 @0 | ppl 19930.4 | bpb 3.600\n")
        self.assertEqual(view_pt.history_from_log(early, "A.pt", None, "tinygpt"),
                         ([], "no evaluation in stopped_early_log.txt since the last fresh start of A.pt"))
        self.assertEqual(view_pt.history_from_log(early, "A.pt", None, None)[0], [], "not the replaced run's curve")
        with open(early, "a", encoding="utf-8") as f:
            f.write(session("c", "RESUMED C:\\x\\A.pt at step 50", (200, 5.1), (400, 4.1)))
        hist, label = view_pt.history_from_log(early, "A_best.pt", None, "tinygpt")
        self.assertEqual([(h["step"], h["val_loss"]) for h in hist], [(200, 5.1), (400, 4.1)])
        self.assertIn("1 session(s) of A", label)

    # ---------------- F401 ----------------
    def test_heatmap_cell_text_follows_the_drawn_colour(self):
        from matplotlib.colors import LinearSegmentedColormap
        import view_pt
        div = LinearSegmentedColormap.from_list("div_blue_red", view_pt.DIVERGING)
        seq = LinearSegmentedColormap.from_list("seq_blue", view_pt.BLUES)

        def colours(matrix, cmap, **limits):
            fig, ax = one_axes()
            view_pt.heatmap(ax, np.array(matrix), ["r0", "r1"], ["c0", "c1"], cmap, "value", **limits)
            return {t.get_text(): t.get_color() for t in ax.texts}

        got = colours([[-0.05, 0.05], [0.0, 0.04]], div, vmin=-0.05, vmax=0.05)
        self.assertEqual(got, {"-0.05": "#ffffff", "0.05": "#ffffff", "0": view_pt.INK, "0.04": "#ffffff"},
                         "the darkest cells of a diverging map with narrow limits get white text")
        # unchanged: the -1..1 similarity scale (white beyond 0.6) and a sequential map without fixed limits
        got = colours([[-0.7, 0.7], [0.5, -0.5]], div, vmin=-1, vmax=1)
        self.assertEqual(got, {"-0.7": "#ffffff", "0.7": "#ffffff", "0.5": view_pt.INK, "-0.5": view_pt.INK})
        got = colours([[1.0, 2.0], [3.0, 4.0]], seq)
        self.assertEqual(got, {"1": view_pt.INK, "2": view_pt.INK, "3": "#ffffff", "4": "#ffffff"})
        got = colours([[0.2, 0.9], [float("nan"), 0.4]], seq, vmin=0, vmax=1)
        self.assertEqual(got, {"0.2": view_pt.INK, "0.9": "#ffffff", "0.4": view_pt.INK}, "no text on an empty cell")

    # ---------------- F434 ----------------
    def test_out_is_checked_before_anything_is_loaded(self):
        path = toy_file(self.tmp, "O.pt", history_rows(3))

        def refused(*flags):
            with redirect_stdout(io.StringIO()), unittest.mock.patch.object(sys, "stderr", io.StringIO()) as err:
                with self.assertRaises(SystemExit, msg=flags) as stop:
                    viewer_args(path, *flags)
            self.assertEqual(stop.exception.code, 2, "an argument error, not a traceback")
            return err.getvalue()

        missing = os.path.join(self.tmp, "nodir")
        for flags in (("--plot",), ("--plot-tokens",), ("--html",)):
            said_ = refused(*flags, "--out", os.path.join(missing, "x.png"))
            self.assertIn("--out", said_)
            self.assertIn("does not exist", said_)
        self.assertFalse(os.path.exists(missing))
        self.assertIn(".bmp", refused("--plot", "--out", os.path.join(self.tmp, "x.bmp")))
        refused("--plot", "--html", "--out", os.path.join(self.tmp, "x.html"))
        # still accepted: no extension (a PNG), upper case, a bare name in the current folder, and the names the
        # other outputs only take their stem from
        for flags in (("--plot", "--out", os.path.join(self.tmp, "x")),
                      ("--plot", "--out", os.path.join(self.tmp, "x.PNG")),
                      ("--plot", "--out", os.path.join(self.tmp, "x.pdf")),
                      ("--plot", "--out", "x.png"),
                      ("--html", "--out", os.path.join(self.tmp, "x.html")),
                      ("--plot-tokens", "--out", os.path.join(self.tmp, "x.html"))):
            self.assertEqual(viewer_args(path, *flags).out, flags[-1])
        # --out without an output that uses it is ignored as before, wherever it points
        for flags in ((), ("--tensor", "token_embedding.weight"), ("--similar", "gene")):
            self.assertEqual(viewer_args(path, *flags, "--out", os.path.join(missing, "x.png")).out,
                             os.path.join(missing, "x.png"))

    # ---------------- F439 ----------------
    def test_push_finds_the_repository_through_a_linked_folder(self):
        import subprocess
        import view_pt
        root = os.path.realpath(os.path.join(self.tmp, "repo_real"))
        link = os.path.join(os.path.dirname(root), "repo_link")
        os.makedirs(os.path.join(root, "runs"))
        try:
            os.symlink(root, link, target_is_directory=True)
        except (OSError, NotImplementedError):  # Windows without the symlink privilege: a junction needs none
            try:
                import _winapi
                _winapi.CreateJunction(root, link)
            except (ImportError, AttributeError, OSError) as exc:
                self.skipTest(f"cannot link a folder here: {exc}")
        png = os.path.join(link, "runs", "A_best_dashboard.png")
        ck = types_ns(path=os.path.join(link, "runs", "A_best.pt"), obj={"step": 8})
        top = subprocess.CompletedProcess([], 0, stdout=root + "\n", stderr="")  # git names the real folder

        def pushed(ck_):
            with unittest.mock.patch.object(view_pt, "_git", return_value=top), redirect_stdout(io.StringIO()), \
                    unittest.mock.patch.object(tiny_gpt, "publish_to_github", return_value="pushed") as publish:
                view_pt.push_dashboard(png, ck_)
            return publish

        publish = pushed(ck)
        publish.assert_called_once()
        self.assertEqual(publish.call_args.args[0], "dashboards/A_best_dashboard.png")
        self.assertIn("A_best.pt at step 8", publish.call_args.args[1])
        # a checkpoint outside the repository is still left alone
        pushed(types_ns(path=os.path.join(self.tmp, "A_best.pt"), obj={"step": 8})).assert_not_called()

    # ---------------- F407 ----------------
    def test_tensor_table_is_not_printed_with_a_picture(self):
        path = toy_file(self.tmp, "T_best.pt", history_rows(6))
        png = os.path.join(self.tmp, "T.png")
        table = "named tensors"
        text, _ = run_main(path)
        self.assertIn(table, text, "the plain call still lists the tensors")
        self.assertIn("best validation: loss", text)
        text, _ = run_main(path, "--plot", "--no-push", "--dpi", "50", "--out", png)
        self.assertNotIn(table, text, "what the trainer's automatic dashboard appends to logs/dashboard.log")
        self.assertIn("best validation: loss", text, "the summary stays")
        self.assertIn("Saved", text)
        self.assertNotIn(table, run_main(path, "--html", "--out", png)[0])
        # a tensor or a token asked for by name: the table is printed as before
        text, _ = run_main(path, "--plot", "--no-push", "--dpi", "50", "--out", png, "--tensor", "lm_head.weight")
        self.assertIn(table, text)
        import view_pt
        argv = ["view_pt.py", path, "--plot", "--no-push", "--out", png, "--similar", "x"]
        with unittest.mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()) as buf, \
                unittest.mock.patch.object(tiny_gpt, "tokenizer_from_checkpoint", return_value=None), \
                self.assertRaises(SystemExit):  # a toy file has no tokenizer, so --similar stops after the table
            view_pt.main()
        self.assertIn(table, buf.getvalue())

    # ---------------- F406 ----------------
    def test_tensor_heading_names_an_untied_output_head_as_untied(self):
        torch.manual_seed(0)
        cfg = tiny_gpt.ModelConfig(vocab_size=50, ctx=16, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2,
                                   tie_embeddings=False)
        untied = toy_file(self.tmp, "untied.pt", history_rows(3), model_config=dataclasses.asdict(cfg),
                          model=tiny_gpt.TinyGPT(cfg).state_dict())
        text, _ = run_main(untied, "--tensor", "lm_head.weight")
        self.assertNotIn("output head (tied)", text)
        self.assertIn("\nlm_head.weight (output head):\n", text)
        text, _ = run_main(toy_file(self.tmp, "tied.pt", history_rows(3)), "--tensor", "lm_head.weight")
        self.assertIn("\nlm_head.weight (output head (tied)):\n", text)


if __name__ == "__main__":
    unittest.main()
