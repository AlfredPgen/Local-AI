"""Staging tests of the 2026-11-10 audit, package H: run_pipeline.py, estimate_dataset.py, gpu_check.py,
tools/*.py and guide/make_figures.py. They move into test_tiny_gpt.py (TestOtherScripts) when the packages
are merged.

    python -m unittest test_a2_h
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
import test_tiny_gpt as T  # noqa: E402


def fake_popen(scripts, started, killed):
    """A stand-in for subprocess.Popen: child k answers its wait() calls from scripts[k], where an exception
    class is raised and a number is the exit code (the last entry repeats)."""
    class Fake:
        def __init__(self, cmd, **kwargs):
            self.answers = list(scripts[len(started)])
            started.append(list(cmd))
            self.returncode = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def wait(self, timeout=None):
            answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            if isinstance(answer, type):
                raise answer()
            self.returncode = answer
            return answer

        def poll(self):
            return self.returncode

        def kill(self):
            killed.append(True)
    return Fake


class Audit2PipelineTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2h_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---------------- run_pipeline ----------------
    def test_run_waits_for_the_step_after_ctrl_c(self):
        """F194: Ctrl+C in run_pipeline.run does not kill the child (subprocess.call kills it 0.25 s later on
        macOS/Linux); run waits for it and returns 130 when the child itself exits 0, so no later step starts."""
        import run_pipeline
        started, killed = [], []
        fake = fake_popen([[KeyboardInterrupt, 0]], started, killed)
        with unittest.mock.patch.object(run_pipeline.subprocess, "Popen", fake), \
                redirect_stdout(io.StringIO()) as out:
            try:
                code = run_pipeline.run(["python", "child.py"])
            except KeyboardInterrupt:
                self.fail("the interrupt left run() while the child was still running")
        self.assertEqual((code, killed, started), (130, [], [["python", "child.py"]]))
        self.assertIn("Ctrl+C: waiting", out.getvalue())
        for want in (0, 3):  # no interrupt: the child's own code
            started.clear()
            fake = fake_popen([[want]], started, killed)
            with unittest.mock.patch.object(run_pipeline.subprocess, "Popen", fake), \
                    redirect_stdout(io.StringIO()) as out:
                self.assertEqual(run_pipeline.run(["python", "child.py"]), want)
            self.assertNotIn("Ctrl+C", out.getvalue())
        # an interrupted child that fails keeps its own code
        started.clear()
        fake = fake_popen([[KeyboardInterrupt, 3]], started, killed)
        with unittest.mock.patch.object(run_pipeline.subprocess, "Popen", fake), redirect_stdout(io.StringIO()):
            self.assertEqual(run_pipeline.run(["python", "child.py"]), 3)
        # continue: a build interrupted with Ctrl+C that still exits 0 does not go on to the training
        here = os.path.join(self.tmp, "f194")
        dataset = os.path.join(here, "datasets", "bio_all")
        os.makedirs(dataset, exist_ok=True)
        os.makedirs(os.path.join(here, "data"), exist_ok=True)
        open(os.path.join(dataset, "manifest.json"), "w").close()
        open(os.path.join(here, "R_best.pt"), "w").close()
        started.clear()
        fake = fake_popen([[KeyboardInterrupt, 0], [0]], started, killed)
        with unittest.mock.patch.multiple(run_pipeline, HERE=here, DATASET=dataset, DATA=os.path.join(here, "data")), \
                unittest.mock.patch.object(run_pipeline.subprocess, "Popen", fake), \
                unittest.mock.patch.object(run_pipeline, "gpu_busy", return_value=False), \
                redirect_stdout(io.StringIO()):
            try:
                code = run_pipeline.step_continue(1, "R", os.path.join(here, "new_ds"))
            except KeyboardInterrupt:
                self.fail("the interrupt left run() while the build was still running")
        self.assertEqual(code, 1)
        self.assertEqual([c[1] for c in started], ["data_prep.py"], "no training after an interrupted build")
        self.assertEqual(killed, [])

    def test_estimate_step_uses_the_built_datasets_tokenizer(self):
        """F191: `run_pipeline.py estimate` passes the built dataset as --tokenizer (the tokenizer that exists,
        not a new one at today's automatic vocabulary size); without a dataset it passes none."""
        import run_pipeline
        here = os.path.join(self.tmp, "f191")
        dataset = os.path.join(here, "datasets", "bio_all")
        os.makedirs(dataset, exist_ok=True)
        os.makedirs(os.path.join(here, "data"), exist_ok=True)
        calls = []

        def estimate():
            with unittest.mock.patch.multiple(run_pipeline, HERE=here, DATASET=dataset,
                                              DATA=os.path.join(here, "data")), \
                    unittest.mock.patch.object(run_pipeline, "run", side_effect=lambda cmd: calls.append(cmd) or 0), \
                    unittest.mock.patch.object(sys, "argv", ["run_pipeline.py", "estimate"]):
                self.assertEqual(run_pipeline.main(), 0)
            return calls[-1]

        self.assertNotIn("--tokenizer", estimate(), "no dataset yet: the estimate trains its own tokenizer")
        with open(os.path.join(dataset, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"prep_args": {"superbpe": True}, "tokenizer": {"vocab_size": 16384}}, f)
        cmd = estimate()
        self.assertEqual(cmd[1:4], ["estimate_dataset.py", "--tokenizer", dataset])
        with unittest.mock.patch.multiple(run_pipeline, HERE=here, DATASET=dataset, DATA=os.path.join(here, "data")):
            self.assertEqual(cmd[4:], run_pipeline.build_command()[4:])
            self.assertNotIn("--tokenizer", run_pipeline.build_command(), "the build command itself is unchanged")

    # ---------------- estimate_dataset ----------------
    def text_folder(self, name, n=8):
        folder = os.path.join(self.tmp, name)
        os.makedirs(folder, exist_ok=True)
        for i in range(n):
            with open(os.path.join(folder, f"doc_{i}.md"), "w", encoding="utf-8", newline="\n") as f:
                f.write(f"# Lecture note {i}\n\n" + T.paragraphs(900 + i, 30))
        return folder

    def run_estimate(self, argv):
        """estimate_dataset.main in this process; data_prep's worker globals and the process priority (which
        _init_worker lowers) are left as they were."""
        import data_prep as dp
        import estimate_dataset
        with unittest.mock.patch.multiple(dp, _W_KW=dp._W_KW, _W_CFG=dp._W_CFG, _lower_priority=lambda: None), \
                redirect_stdout(io.StringIO()) as out:
            estimate_dataset.main(argv)
        return out.getvalue()

    def tokenizer_model(self, name):
        """(path, pieces) of a small SentencePiece model with data_prep's line-break pieces."""
        import sentencepiece as spm
        work = os.path.join(self.tmp, name)
        os.makedirs(work, exist_ok=True)
        corpus = os.path.join(work, "corpus.txt")
        with open(corpus, "w", encoding="utf-8", newline="\n") as f:
            f.write(T.paragraphs(1, 60).replace("\n\n", "\n"))
        spm.SentencePieceTrainer.train(input=corpus, model_prefix=os.path.join(work, "tok"), model_type="bpe",
                                       vocab_size=120, hard_vocab_limit=False, user_defined_symbols=["\n", "\t"],
                                       unk_id=0, bos_id=1, eos_id=2, pad_id=-1, minloglevel=2)
        model = os.path.join(work, "tok.model")
        return model, spm.SentencePieceProcessor(model_file=model).get_piece_size()

    def test_estimate_with_a_tokenizer_sizes_models_for_its_vocabulary(self):
        """F191: with --tokenizer the model-size table is built for that tokenizer's vocabulary and one line
        names the vocabulary a rebuild would train; a tokenizer that cannot be read falls back to training."""
        import tiny_gpt
        folder = self.text_folder("f191_text")
        model, size = self.tokenizer_model("f191_tok")
        self.assertNotEqual(size, 4096)
        argv = ["--md-dir", folder, "--tflops", "1", "--hours", "1"]
        with unittest.mock.patch.object(tiny_gpt, "shape_ladder", wraps=tiny_gpt.shape_ladder) as ladder:
            out = self.run_estimate(["--tokenizer", model] + argv)
        ladder.assert_called_once_with(size)
        self.assertIn("a rebuild from scratch would train a 4,096-piece tokenizer", out)
        self.assertNotIn("training a", out)
        # a dataset folder whose manifest names no readable tokenizer: not an exit, a trained tokenizer
        broken = os.path.join(self.tmp, "f191_broken")
        os.makedirs(broken, exist_ok=True)
        with open(os.path.join(broken, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"prep_args": {}}, f)
        with unittest.mock.patch.object(tiny_gpt, "shape_ladder", wraps=tiny_gpt.shape_ladder) as ladder:
            out = self.run_estimate(["--tokenizer", broken] + argv)
        ladder.assert_called_once_with(4096)
        self.assertIn("cannot be read", out)
        self.assertIn("training a 4,096-piece SentencePiece tokenizer on the kept sample", out)
        self.assertNotIn("a rebuild from scratch", out)

    def test_estimate_honours_sample_fraction_and_tokenizer_from(self):
        """F148: data_prep's --sample-fraction scales the web sources before the --*-max-docs cap, and
        --tokenizer-from is the tokenizer measured with (both were parsed and ignored)."""
        import gzip
        import estimate_dataset
        import pyarrow as pa
        import pyarrow.parquet as pq
        t = estimate_dataset.Tally("fineweb", "parquet")
        t.total, t.sampled, t.kept, t.kept_chars, t.sample_fraction = 1000, 100, 50, 5000, 0.5
        self.assertEqual(t.estimate(), (250, 25000))
        t.max_docs = 200  # the cap applies to what the sample fraction leaves
        self.assertEqual(t.estimate(), (200, 20000))
        t.max_docs = 400
        self.assertEqual(t.estimate(), (250, 25000), "half the records cannot fill a cap of 400")
        md = estimate_dataset.Tally("md", "markdown")
        md.total, md.sampled, md.kept, md.kept_chars, md.total_bytes, md.sampled_bytes = 10, 10, 10, 900, 1000, 1000
        md.sample_fraction = 0.5
        self.assertEqual(md.estimate(), (10, 900), "data_prep does not sample text folders")
        # the sources take the fraction from the data_prep options
        work = os.path.join(self.tmp, "f148")
        os.makedirs(work, exist_ok=True)
        parquet, jsonl = os.path.join(work, "web.parquet"), os.path.join(work, "papers.json.gz")
        pq.write_table(pa.table({"id": [f"w{i}" for i in range(20)],
                                 "text": [T.BIO + " " + T.paragraphs(300 + i, 4) for i in range(20)]}), parquet)
        with gzip.open(jsonl, "wt", encoding="utf-8") as f:
            for i in range(12):
                f.write(json.dumps({"id": f"p{i}", "text": T.BIO + " " + T.paragraphs(700 + i, 5)}) + "\n")
        model, size = self.tokenizer_model("f148_tok")
        argv = ["--sample", "1", "--tflops", "1", "--hours", "1", "--parquet", parquet, "--parquet-name", "web",
                "--jsonl", jsonl, "--jsonl-name", "papers", "--tokenizer-from", model]

        def kept(out):
            rows = {ln.split()[0]: ln.split() for ln in out.splitlines() if ln.split()[:1] in (["web"], ["papers"])}
            return {name: (int(r[2]), int(r[3])) for name, r in rows.items()}  # (in source, kept docs)

        out = self.run_estimate(argv)
        self.assertEqual(kept(out), {"web": (20, 20), "papers": (12, 12)})
        self.assertIn(f"characters per token from tokenizer {model}", out)
        self.assertNotIn("training a", out)
        self.assertNotIn("a rebuild from scratch", out, "the build itself would reuse this tokenizer")
        self.assertEqual(kept(self.run_estimate(argv + ["--sample-fraction", "0.5"])),
                         {"web": (20, 10), "papers": (12, 6)})

    def test_estimate_excludes_gz_files_like_data_prep(self):
        """F150: an --md-exclude pattern is matched against a .gz file's name without '.gz', as data_prep does."""
        import collections
        import gzip
        import random
        import types
        import data_prep as dp
        import estimate_dataset
        folder = os.path.join(self.tmp, "f150")
        os.makedirs(os.path.join(folder, "sub"), exist_ok=True)
        with open(os.path.join(folder, "keep.md"), "w", encoding="utf-8") as f:
            f.write("# Keep\n\nGenetic drift.")
        with gzip.open(os.path.join(folder, "narration.lt.md.gz"), "wt", encoding="utf-8") as f:
            f.write("Lietuviškas tekstas.")
        with open(os.path.join(folder, "sub", "other.LT.md"), "w", encoding="utf-8") as f:
            f.write("Lietuviškas tekstas.")
        args = types.SimpleNamespace(md_glob=None, md_exclude=["*.lt.md"], md_recursive=True, sample=1.0,
                                     est_seed=0)
        with unittest.mock.patch.object(estimate_dataset, "examine"):
            tally = estimate_dataset.markdown_source("md", [folder], args, False, random.Random(0))
        stats = {"scanned": 0, "skipped": collections.Counter(), "cleaning": collections.Counter()}
        docs = list(dp.iter_markdown([folder], list(dp.MD_PATTERNS_DEFAULT), args.md_exclude, True, stats, []))
        self.assertEqual([d[0] for d in docs], ["keep.md"])
        self.assertEqual((tally.total, tally.sampled), (len(docs), len(docs)))

    def test_estimate_skips_a_gz_file_with_damaged_deflate_data(self):
        """F043: damaged deflate data raises zlib.error, which is no OSError; the estimate skips the file as it
        skips a truncated one (and as data_prep does)."""
        import gzip
        import random
        import types
        import zlib
        import estimate_dataset
        folder = os.path.join(self.tmp, "f043")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "plain.md"), "w", encoding="utf-8") as f:
            f.write("# Plain\n\nGenetic drift.")
        with gzip.open(os.path.join(folder, "damaged.md.gz"), "wt", encoding="utf-8") as f:
            f.write("# Packed\n\nAllele frequencies.")
        args = types.SimpleNamespace(md_glob=None, md_exclude=[], md_recursive=False, sample=1.0, est_seed=0)
        error = zlib.error("Error -3 while decompressing data: invalid distance too far back")
        with unittest.mock.patch.object(estimate_dataset, "examine") as examined, \
                unittest.mock.patch.object(estimate_dataset.gzip, "open", side_effect=error):
            tally = estimate_dataset.markdown_source("md", [folder], args, False, random.Random(0))
        self.assertEqual((tally.total, tally.sampled, examined.call_count), (2, 1, 1))

    def test_estimate_stores_a_random_sample_of_texts_and_measures_on_held_out_ones(self):
        """F153: the texts kept for the tokenizer are a reservoir sample of all kept documents (they were the
        first 6M characters, the start of the first files), and characters per token of a tokenizer trained
        here are measured on texts it was not trained on."""
        import data_prep as dp
        import estimate_dataset

        def fill(name, seed=0):
            tally = estimate_dataset.Tally(name, "parquet", seed)
            for i in range(200):  # first half short English, second half long technical text
                tally.keep(f"{i:03d} " + ("short english text. " * 5000 if i < 100 else "long technical text " * 10000))
            return tally

        t = fill("web")
        ids = [int(text[:3]) for text in t.texts]
        self.assertEqual((t.kept, len(ids), len(set(ids))), (200, 60, 60), "6M characters of 100k-character texts")
        self.assertTrue(20 <= sum(i >= 100 for i in ids) <= 40, ids)  # about half; the old store had none
        self.assertEqual(t.text_chars, sum(len(text) for text in t.texts))
        self.assertEqual(t.kept_chars, 100 * 100_004 + 100 * 200_004)
        self.assertEqual(ids, [int(text[:3]) for text in fill("web").texts], "the same sample every time")
        self.assertNotEqual(ids, [int(text[:3]) for text in fill("web", seed=1).texts])
        small = estimate_dataset.Tally("md", "markdown")
        for i in range(7):
            small.keep(f"text {i}")
        self.assertEqual(small.texts, [f"text {i}" for i in range(7)], "below the budget every text is stored")
        # every fifth stored text is held out; a source with fewer than five has none to spare
        self.assertEqual(estimate_dataset.split_texts(list("abcdefghijk")), (list("abcdfghik"), list("ej")))
        self.assertEqual(estimate_dataset.split_texts(list("abcd")), (list("abcd"), list("abcd")))
        folder = self.text_folder("f153_text", n=10)
        trained, measured = [], []
        real_lines, real_cpt = dp.tokenizer_sample_lines, estimate_dataset.chars_per_token
        argv = ["--md-dir", folder, "--tflops", "1", "--hours", "1"]
        with unittest.mock.patch.object(dp, "tokenizer_sample_lines",
                                        side_effect=lambda text, *a, **k: trained.append(text) or real_lines(
                                            text, *a, **k)), \
                unittest.mock.patch.object(estimate_dataset, "chars_per_token",
                                           side_effect=lambda tok, texts: measured.append(list(texts)) or real_cpt(
                                               tok, texts)):
            self.run_estimate(argv)
            self.assertEqual((len(trained), len(measured), len(measured[0])), (8, 1, 2))
            self.assertFalse(set(trained) & set(measured[0]), "measured on texts the tokenizer did not see")
            model, _ = self.tokenizer_model("f153_tok")
            self.run_estimate(argv + ["--tokenizer", model])
            self.assertEqual((len(trained), len(measured), len(measured[1])), (8, 2, 10), "a given tokenizer: all")

    def test_estimate_passes_the_seed_to_the_superbpe_stage(self):
        """F154: the SuperBPE stage of the estimate's tokenizer samples with --seed, as a build's does (it used
        superbpe_extend's own default, 1234, whatever --seed said)."""
        import types
        import data_prep as dp
        import estimate_dataset
        tally = estimate_dataset.Tally("md", "markdown")
        for i in range(6):
            tally.keep(T.paragraphs(40 + i, 30))
        args = types.SimpleNamespace(superbpe=True, superbpe_fraction=0.1, superbpe_max_words=4, seed=7,
                                     tokenizer_weights=None)
        work = os.path.join(self.tmp, "f154")
        os.makedirs(work, exist_ok=True)
        seen = {}

        def extend(proto, corpus, extra, max_words, **kwargs):
            seen.update(kwargs, extra=extra, max_words=max_words)
            return proto, []

        with unittest.mock.patch.object(dp, "superbpe_extend", extend):
            tok = estimate_dataset.train_tokenizer([tally], 600, work, args, {"md": tally.estimate()})
        self.assertEqual((seen.get("seed"), seen["extra"], seen["max_words"]), (7, 60, 4))
        self.assertGreater(tok.vocab_size, 100)


if __name__ == "__main__":
    unittest.main()
