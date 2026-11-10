"""Staging tests of package D (benchmark.py) of the 2026-11 audit. Target class: TestBenchmarkSuite in
test_tiny_gpt.py (package T moves every test there and deletes this file)."""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class Audit2Benchmark(unittest.TestCase):
    """benchmark.py: offline only (no downloads, no sandbox, no real checkpoint)."""

    @classmethod
    def setUpClass(cls):
        import benchmark
        cls.b = benchmark
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2_bench_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def suite_run(self, label, items, extra=(), sandbox=None):
        """`benchmark.py run --device cpu --no-publish` on a tiny saved checkpoint, every output path inside the
        temporary folder. items = {benchmark name: [item, ...]}. Returns (exit code, stdout, rows of the results
        CSV, {file name: rows} of the per-item CSVs, rows of experiments.csv)."""
        import csv
        import dataclasses
        import json
        import torch
        import tiny_gpt
        from test_tiny_gpt import CharTok
        b = self.b
        root = os.path.join(self.tmp, label)
        bench, runs = os.path.join(root, "bench"), os.path.join(root, "bench", "runs")
        for name, its in items.items():
            os.makedirs(os.path.join(bench, name))
            b.write_items(os.path.join(bench, name, "items.jsonl"), its)
        with open(os.path.join(bench, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"benchmarks": {name: {"items": len(its)} for name, its in items.items()}}, handle)
        torch.manual_seed(0)
        cfg = tiny_gpt.ModelConfig(vocab_size=50, ctx=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2)
        ck = os.path.join(root, "model", "tiny_best.pt")
        os.makedirs(os.path.dirname(ck))
        torch.save({"format": tiny_gpt.CHECKPOINT_FORMAT, "format_version": tiny_gpt.FORMAT_VERSION,
                    "checkpoint_type": "inference", "model_config": dataclasses.asdict(cfg),
                    "model": tiny_gpt.TinyGPT(cfg).state_dict(), "step": 7}, ck)
        results = os.path.join(root, "results", "benchmarks.csv")
        out = io.StringIO()
        with unittest.mock.patch.object(b, "BENCH_DIR", bench), \
                unittest.mock.patch.object(b, "MANIFEST", os.path.join(bench, "manifest.json")), \
                unittest.mock.patch.object(b, "RUNS_DIR", runs), \
                unittest.mock.patch.object(b, "select_sandbox", return_value=sandbox), \
                unittest.mock.patch.object(tiny_gpt, "tokenizer_from_checkpoint", return_value=CharTok()), \
                unittest.mock.patch.object(tiny_gpt, "publish_experiments", tiny_gpt.publish_experiments), \
                redirect_stdout(out):
            code = b.main(["run", "--checkpoint", ck, "--benchmarks", ",".join(items), "--device", "cpu",
                           "--no-publish", "--results", results, *extra])

        def table(path):
            with open(path, encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))

        item_rows = {}
        for folder in (os.listdir(runs) if os.path.isdir(runs) else []):
            for name in os.listdir(os.path.join(runs, folder)):
                item_rows[name] = table(os.path.join(runs, folder, name))
        experiments = os.path.join(root, "model", "results", "experiments.csv")
        return (code, out.getvalue(), b.read_results(results), item_rows,
                table(experiments) if os.path.isfile(experiments) else [])

    @staticmethod
    def medcalc_items():
        return [{"id": f"medcalc-{i}", "question": "Score?", "options": None, "answer": answer, "category": "risk",
                 "kind": "numeric", "context": "A patient note.", "output_type": kind, "lower": answer,
                 "upper": answer, "calculator": "c"} for i, (answer, kind) in enumerate((("4", "integer"),
                                                                                        ("2.5", "decimal")))]

    @staticmethod
    def humaneval_item(**extra):
        return {"id": "h0", "question": "def f():\n", "options": None, "answer": "    return 1\n",
                "category": "python", "kind": "code", "style": "humaneval", "test": "def check(c): pass",
                "entry_point": "f", **extra}

    def mcq_items(self):
        options = ["alpha", "beta", "gamma"]
        return [self.b.mcq_item(f"t{i}", f"Which gene number {i}?", options, options[i % 3],
                                "genes" if i % 2 else "cells", ["biology"] if i < 3 else None) for i in range(6)]

    def test_experiments_row_has_the_device_type(self):
        """F008: the experiments.csv row of a benchmark run has the device type in `device`, as training rows
        do; the hardware label opens the notes."""
        import torch
        import tiny_gpt
        code, _, _, _, experiments = self.suite_run("f008", {"medqa": self.mcq_items()}, extra=["--no-floor"])
        self.assertEqual((code, len(experiments)), (0, 1))
        row = experiments[0]
        self.assertEqual((row["event"], row["device"]), ("benchmark suite", "cpu"))
        self.assertTrue(row["notes"].startswith(tiny_gpt.device_label(torch.device("cpu")) + "; medqa "),
                        row["notes"])

    def plot_figures(self, label, rows):
        """`benchmark.py plot` on a results table made of `rows` (benchmark, category, accuracy, floor or None,
        optional checkpoint/step/params): ({PNG name: its Figure}, stdout). The PNGs are drawn into the
        temporary folder."""
        import matplotlib.figure
        import tiny_gpt
        b = self.b
        path = os.path.join(self.tmp, label, "benchmarks.csv")
        full = []
        for r in rows:
            low, high = tiny_gpt.wilson_interval(round(r["accuracy"] * 20), 20)
            full.append({"time": "t", "checkpoint": "tinyGPT_best.pt", "step": 1000, "params": 1000000, "n": 20,
                         "low": low, "high": high, "chance": 0.25, "notes": "", **r,
                         "floor": "" if r.get("floor") is None else r["floor"]})
        b.append_results(path, full)
        figures, save, out = {}, matplotlib.figure.Figure.savefig, io.StringIO()

        def savefig(fig, name, *args, **kwargs):
            figures[os.path.basename(name)] = fig
            return save(fig, name, *args, **kwargs)

        with unittest.mock.patch.object(matplotlib.figure.Figure, "savefig", savefig), redirect_stdout(out):
            self.assertEqual(b.main(["plot", "--results", path, "--out-dir", os.path.join(self.tmp, label, "plots"),
                                     "--dpi", "40"]), 0)
        return figures, out.getvalue()

    def test_category_panel_without_a_floor(self):
        """F003: after `run --no-floor` the category bars are neutral (not all red) and neither the legend nor
        the axis speaks of a floor; a category with a floor is still green above it and red at or below."""
        import matplotlib.colors
        import view_pt as vp

        def panel(label, floors):
            rows = [{"benchmark": "mmlu", "category": "all", "accuracy": 0.3, "floor": floors[0]}]
            rows += [{"benchmark": "mmlu", "category": c, "accuracy": a, "floor": f}
                     for c, a, f in zip(("anatomy", "genetics", "virology"), (0.2, 0.4, 0.6), floors)]
            ax = self.plot_figures(label, rows)[0]["tinyGPT_best_categories.png"].axes[0]
            colours = [matplotlib.colors.to_hex(p.get_facecolor()[:3]) for p in ax.patches]
            return colours, [t.get_text() for t in ax.get_legend().get_texts()], ax.get_xlabel()

        colours, legend, xlabel = panel("f003_none", (None, None, None))
        self.assertEqual(colours, [vp.SERIES[0]] * 3)
        self.assertFalse([text for text in legend if "floor" in text], legend)
        self.assertEqual(xlabel, "accuracy")
        colours, legend, xlabel = panel("f003_mixed", (0.3, 0.3, None))  # sorted by accuracy: 0.2, 0.4, 0.6
        self.assertEqual(colours, [vp.CRITICAL, vp.GOOD, vp.SERIES[0]])
        self.assertTrue([text for text in legend if "floor" in text], legend)
        self.assertIn("green above floor", xlabel)

    def test_progress_plot_draws_one_line_per_model(self):
        """F004: the progress plot never joins two models into one line: one line per (benchmark, model), a model
        being a checkpoint name without _best/_final plus its parameter count; the line style tells the models
        apart, the colour the benchmarks, and from the ninth benchmark on the marker changes too."""
        import view_pt as vp

        def scored(checkpoint, params, steps, benches):
            return [{"checkpoint": checkpoint, "params": params, "step": step, "benchmark": bench, "category": "all",
                     "accuracy": 0.2 + 0.02 * k, "floor": 0.2} for step in steps for k, bench in enumerate(benches)]

        nine = [f"bench{k}" for k in range(9)]
        small = scored("m/small_best.pt", 1000000, (1000, 2000), nine)
        # one model (its _best and _final files): one line per benchmark, the bare name in the legend, inside
        ax = self.plot_figures("f004_one", small + scored("m/small_final.pt", 1000000, (3000,), nine)
                               )[0]["progress.png"].axes[0]
        self.assertEqual([line.get_label() for line in ax.lines], nine)
        self.assertEqual({len(line.get_xdata()) for line in ax.lines}, {3})
        self.assertEqual({line.get_linestyle() for line in ax.lines}, {"-"})
        self.assertEqual([line.get_color() for line in ax.lines], vp.SERIES + vp.SERIES[:1])
        self.assertEqual([line.get_marker() for line in ax.lines], ["o"] * 8 + ["s"])
        self.assertLess(ax.get_legend().get_bbox_to_anchor().x0, ax.bbox.x1)
        # two models scored at different steps: two lines per benchmark, the model in the label, legend outside
        ax = self.plot_figures("f004_two", small + scored("m/big_best.pt", 5000000, (1500, 2500), nine[:2])
                               )[0]["progress.png"].axes[0]
        self.assertEqual([(line.get_label(), line.get_linestyle(), list(line.get_xdata())) for line in ax.lines[:4]],
                         [("bench0: big (5.0M)", "-", [1500.0, 2500.0]), ("bench0: small (1.0M)", "--", [1000.0, 2000.0]),
                          ("bench1: big (5.0M)", "-", [1500.0, 2500.0]), ("bench1: small (1.0M)", "--", [1000.0, 2000.0])])
        self.assertEqual(len(ax.lines), 2 * 2 + 7)
        self.assertEqual(ax.lines[0].get_color(), ax.lines[1].get_color())
        self.assertGreater(ax.get_legend().get_bbox_to_anchor().x0, ax.bbox.x1)

    def test_medcalc_rows_name_the_integer_rule(self):
        """F016: MedCalc rows scored with the official integer rule say so in their notes (earlier rows of
        results/benchmarks.csv used rounding for the 34 'integer' items whose answer is not a whole number)."""
        code, _, rows, _, _ = self.suite_run("f016", {"medcalc": self.medcalc_items()})
        self.assertEqual(code, 0)
        self.assertEqual([r["benchmark"] for r in rows], ["medcalc"])
        self.assertIn("integer rule: official", rows[0]["notes"])

    def fake_download(self, label, force):
        """download_one on a made-up benchmark with huggingface_hub replaced by a stand-in (no network):
        (manifest entry, status, keyword arguments of every hf_hub_download call, items written)."""
        import json
        import types
        b = self.b
        bench = os.path.join(self.tmp, label)
        calls = []

        def hf_hub_download(repo, name, local_dir=None, **kwargs):
            calls.append(kwargs)
            path = os.path.join(local_dir, *name.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                for i in range(3):
                    handle.write(json.dumps({"n": i}) + "\n")
            return path

        class HfApi:
            def dataset_info(self, repo):
                return types.SimpleNamespace(gated=False, card_data=None, sha="rev1")

        def normalise(raw):
            items = [b.mcq_item(f"t{r['n']}", f"Question {r['n']}?", ["a", "b"], "a", "cat")
                     for r in b.read_table(raw[0])]
            return items + [dict(items[0], id="bad", answer="z")]

        spec = b._spec("toy", "Toy", "knowledge", "someone/toy", ["data/t.jsonl"], "MIT", normalise)
        hub = types.SimpleNamespace(HfApi=HfApi, hf_hub_download=hf_hub_download)
        with unittest.mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
                unittest.mock.patch.object(b, "BENCH_DIR", bench):
            entry, status = b.download_one(spec, {"benchmarks": {}}, force=force)
            items = b.read_items("toy")
        return entry, status, calls, items

    def test_download_force_downloads_again(self):
        """F012: `download --force` tells hf_hub_download to fetch the file again (it used to re-normalise the
        cached copy only); without --force nothing changes."""
        entry, status, calls, items = self.fake_download("f012_force", True)
        self.assertEqual((status, len(calls)), ("downloaded", 1))
        self.assertIs(calls[0].get("force_download"), True)
        self.assertEqual((calls[0]["repo_type"], calls[0]["revision"]), ("dataset", "rev1"))
        _, _, calls, _ = self.fake_download("f012_plain", False)
        self.assertFalse(calls[0].get("force_download"))

    def test_missing_module_hint_fits_the_sandbox(self):
        """F010: `sandbox-setup` fills the folder that only the bubblewrap jail mounts, so the line about skipped
        items points there for bubblewrap alone; Docker and the unshare fallback get the step that helps them."""
        b = self.b
        items = [self.humaneval_item(requires=["numpy"])]
        lacks = b.SandboxResult(True, "passed", 'MODS={"numpy": false}')
        boom = unittest.mock.Mock(side_effect=AssertionError("nothing may be executed"))
        old = "see `python benchmark.py sandbox-setup`"
        cases = (("docker", b.DockerSandbox("img"), "pass --docker-image <image with numpy> or --sandbox wsl"),
                 ("unshare", b.WSLSandbox("Ubuntu", "unshare"), "install bubblewrap; the unshare fallback cannot"),
                 ("bwrap", b.WSLSandbox("Ubuntu", "bwrap"), old))
        for label, sandbox, want in cases:
            with unittest.mock.patch.object(b, "_finish", return_value=lacks), \
                    unittest.mock.patch.object(b, "_run", boom), \
                    unittest.mock.patch.object(b, "_MODULE_CACHE", {}):
                code, out, rows, _, _ = self.suite_run(f"f010_{label}", {"humanevalplus": items}, sandbox=sandbox)
            line = next(text for text in out.splitlines() if "sandbox lacks numpy" in text)
            self.assertIn("(1 items skipped; " + want, line)
            self.assertEqual(old in line, label == "bwrap", line)
            self.assertEqual((code, rows), (0, []))
        boom.assert_not_called()
        self.assertIn("or unshare (weaker: no PID namespace)", " ".join(b.__doc__.split()))

    def test_refusal_names_this_machine(self):
        """F006: without a sandbox the refusal names Docker Desktop, WSL and the Windows host on Windows only; on
        a Mac or Linux it says 'start Docker' and 'this machine'. It starts with REFUSING everywhere."""
        import json
        b = self.b
        bench = os.path.join(self.tmp, "f006")
        os.makedirs(os.path.join(bench, "humaneval"))
        b.write_items(os.path.join(bench, "humaneval", "items.jsonl"), [self.humaneval_item()])
        with open(os.path.join(bench, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"benchmarks": {"humaneval": {"items": 1}}}, handle)
        args = b.parse_args(["run", "--checkpoint", "none.pt", "--benchmarks", "humaneval", "--device", "cpu"])
        said = {}
        for platform in ("win32", "darwin", "linux"):
            out = io.StringIO()
            with unittest.mock.patch.object(b, "BENCH_DIR", bench), \
                    unittest.mock.patch.object(b, "MANIFEST", os.path.join(bench, "manifest.json")), \
                    unittest.mock.patch.object(b, "select_sandbox", return_value=None), \
                    unittest.mock.patch.object(sys, "platform", platform), \
                    redirect_stdout(out):
                self.assertEqual(b.cmd_run(args), 1)
            said[platform] = out.getvalue()
            self.assertTrue(said[platform].startswith("REFUSING to run generated code for humaneval: no sandbox ("))
            self.assertIn("--unsafe-local-execution runs it in a plain subprocess", said[platform])
        self.assertIn("(start Docker Desktop, or install bubblewrap in a WSL distribution with python3). Generated "
                      "code is never run on the Windows host;", said["win32"])
        for platform in ("darwin", "linux"):
            self.assertNotIn("WSL", said[platform])
            self.assertNotIn("Windows", said[platform])
            self.assertIn("(start Docker). Generated code is never run on this machine;", said[platform])

    def test_generated_code_gets_no_stdin(self):
        """F020: the jailed program reads /dev/null, not the rest of the wrapper script piped to `bash -s` (which
        holds the clean-up lines), and the unsafe local runner gives its child no stdin either."""
        b = self.b
        for tool, end in (("bwrap", "python3 -s prog.py </dev/null"), ("unshare", "\"$d\" </dev/null")):
            script = b.WSLSandbox("Ubuntu", tool).script("input()\n", 5)
            jail = [line for line in script.splitlines() if "prlimit" in line]
            self.assertEqual(len(jail), 1, script)
            self.assertTrue(jail[0].endswith(end), jail[0])
            self.assertTrue(script.endswith("rc=$?\nrm -rf \"$d\"\nexit $rc\n"), "the clean-up lines follow the jail")
        with unittest.mock.patch.object(subprocess, "run", side_effect=subprocess.run) as spy:
            res = b.LocalSubprocess().run("input()\n", timeout=5)
        self.assertIs(spy.call_args.kwargs.get("stdin"), subprocess.DEVNULL)
        self.assertEqual((res.passed, res.status), (False, "failed"))
        self.assertIn("EOFError", res.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
