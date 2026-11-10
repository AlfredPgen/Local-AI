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
        self.assertIn("a rebuild from scratch would train a 4,096-piece tokenizer (fewer tokens): python", out)
        self.assertNotIn("10 %", out, "a figure that holds for one pair of vocabulary sizes only")
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
        # only a dataset folder falls back: a mistyped path still stops the estimate, as it always did
        with self.assertRaises(SystemExit) as stop:
            self.run_estimate(["--tokenizer", os.path.join(self.tmp, "nosuch_folder")] + argv)
        self.assertIn("Could not read tokenizer from", str(stop.exception))

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
        """F153: the texts kept for the tokenizer are a random sample of all kept documents (they were the
        first 6M characters, the start of the first files) that holds the budget of characters whatever the
        order of long and short documents, and characters per token of a tokenizer trained here are measured
        on texts it was not trained on."""
        import data_prep as dp
        import estimate_dataset
        budget = estimate_dataset.TEXT_BUDGET

        def fill(name, seed=0):
            tally = estimate_dataset.Tally(name, "parquet", seed)
            for i in range(200):  # first half short English, second half long technical text
                tally.keep(f"{i:03d} " + ("short english text. " * 5000 if i < 100 else "long technical text " * 10000))
            return tally

        t = fill("web")
        ids = [int(text[:3]) for text in t.texts]
        self.assertEqual((t.kept, len(set(ids)), ids), (200, len(ids), sorted(ids)), "in the order they were kept")
        later = sum(i >= 100 for i in ids)
        self.assertTrue(0.3 * len(ids) <= later <= 0.7 * len(ids), ids)  # about half; the old store had none
        self.assertEqual(t.text_chars, sum(len(text) for text in t.texts))
        self.assertTrue(budget <= t.text_chars < budget + 200_004, t.text_chars)
        self.assertEqual(t.kept_chars, 100 * 100_004 + 100 * 200_004)
        self.assertEqual(ids, [int(text[:3]) for text in fill("web").texts], "the same sample every time")
        self.assertNotEqual(ids, [int(text[:3]) for text in fill("web", seed=1).texts])
        # long texts first and short ones later (Wikipedia: the first kept articles are 4 times the average):
        # a store with the number of documents that first filled the budget would end with a tenth of it
        t = estimate_dataset.Tally("wiki", "hf_arrow")
        for i in range(2040):
            t.keep(f"{i:04d} " + "long article text.. " * (10000 if i < 40 else 250))
        self.assertTrue(budget <= t.text_chars < budget + 200_005, t.text_chars)
        self.assertEqual(t.text_chars, sum(len(text) for text in t.texts))
        self.assertGreater(len(t.texts), 600, "mostly the short texts, which are 98 % of the documents")
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

    # ---------------- gpu_check ----------------
    def gpu_check_load(self, used, util):
        """(stdout, load_test mock) of `gpu_check.py --load 5` on a machine whose nvidia-smi reports `used` MiB
        and `util` % (both the script's own snapshot and tiny_gpt.gpu_busy's query)."""
        import types
        import gpu_check
        import tiny_gpt
        snapshot = dict.fromkeys(gpu_check.QUERY, "0")
        snapshot.update({"pstate": "P8", "utilization.gpu": str(util), "memory.used": str(used)})
        smi = types.SimpleNamespace(stdout=f"{used}, {util}\n", returncode=0)
        with unittest.mock.patch.object(gpu_check, "nvidia_smi", return_value=snapshot), \
                unittest.mock.patch.object(gpu_check, "windows_power", return_value={}), \
                unittest.mock.patch.object(gpu_check, "load_test") as load, \
                unittest.mock.patch.object(tiny_gpt.subprocess, "run", return_value=smi), \
                unittest.mock.patch.object(sys, "argv", ["gpu_check.py", "--load", "5"]), \
                redirect_stdout(io.StringIO()) as out:
            gpu_check.main()
        return out.getvalue(), load

    def test_gpu_check_uses_the_shared_busy_rule(self):
        """F166 (guard: passes before and after): gpu_check's busy test is tiny_gpt.gpu_busy's rule; 3000 MiB
        in use at 10 % or 25 % utilization at 100 MiB refuse the load test, an idle GPU runs it."""
        import tiny_gpt
        for used, util, busy in ((3000, 10, True), (100, 25, True), (400, 3, False)):
            out, load = self.gpu_check_load(used, util)
            self.assertEqual((tiny_gpt.GPU_BUSY_MESSAGE in out, load.called), (busy, not busy), (used, util))

    def test_gpu_check_busy_warning_says_the_caveat_once(self):
        """F165: the warning is the shared message plus the --force hint; the driver-reset caveat that the
        message already holds is not repeated."""
        import tiny_gpt
        out, _ = self.gpu_check_load(3000, 10)
        self.assertTrue(out.endswith("\n" + tiny_gpt.GPU_BUSY_MESSAGE + " Or pass --force.\n"), out[-400:])
        self.assertEqual(out.count("driver"), tiny_gpt.GPU_BUSY_MESSAGE.count("driver"))

    # ---------------- tools ----------------
    def test_keyword_covered_candidates_by_word_count_change_nothing(self):
        """F384 (guard: passes before and after): looking for covering terms only among terms with a word count
        that can cover gives what comparing every pair of terms gives."""
        import kw_covered

        def every_pair(terms, nested):
            shapes = {t: kw_covered.shape(t) for t in terms}
            covered, seen = {}, {}
            for t in terms:
                sa = shapes[t]
                if not sa:
                    continue
                key = (tuple(sa[0]), sa[1])
                if seen.get(key, t) != t:
                    covered[t] = seen[key]
                    continue
                seen[key] = t
                for u in sorted(terms):
                    su = shapes[u]
                    if u != t and u not in covered and su and su != sa and kw_covered.covers(su, sa, nested) \
                            and not any(kind == "suffix" for _, kind in su[0]):
                        covered[t] = u
                        break
            return covered

        terms = ["polymer*", "polymerase", "spatial transcriptomic*", "spatial transcriptomics", "CITE-seq",
                 "=CITE-seq", "=MET", "=Met", "regression", "linear regression", "*mab", "adalimumab",
                 "China Kadoorie", "China Kadoorie Biobank", "Kadoorie Biobank study group", "UK Biobank",
                 "UK Biobank study", "=UK", "T cell", "T cell receptor", "CAR T cell receptor*", "cell receptor",
                 "recept*", "receptor", "=T", "of the", "risk of the disease", "zeta", "Zeta", "Biobank stud*",
                 "linear mixed model", "mixed model*", "generalised linear mixed models", "model", "=DNA", "DNA"]
        for nested in (False, True):
            self.assertEqual(kw_covered.find(terms, nested), every_pair(terms, nested), nested)
        nested = kw_covered.find(terms, True)
        self.assertEqual((nested["UK Biobank study"], nested["T cell receptor"], nested["risk of the disease"],
                          nested["generalised linear mixed models"]),
                         ("Biobank stud*", "T cell", "of the", "mixed model*"))
        self.assertGreater(len(nested), len(kw_covered.find(terms, False)))

    def tool_main(self, module, argv):
        """stdout of module.main() run with the command line `argv`. The tools reconfigure sys.stdout, which an
        io.StringIO cannot do, so the output goes to a text wrapper around bytes."""
        raw = io.BytesIO()
        out = io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
        with unittest.mock.patch.object(sys, "argv", argv), unittest.mock.patch.object(sys, "stdout", out):
            module.main()
            out.flush()
        return raw.getvalue().decode("utf-8")

    def test_url_check_reports_a_row_without_a_url(self):
        """F385: a row whose url field is missing or empty is listed as failed ('no url') and not requested;
        it used to end the whole check with AttributeError before any report was printed."""
        import url_check
        path = os.path.join(self.tmp, "sources.tsv")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("# comment\ntool\turl\nalpha\t https://example.org/a \nbeta\ngamma\t\n")
        asked = []

        def check(url):
            asked.append(url)
            return {"status": 200, "final": url, "type": "text/html", "bytes_read": 5, "ok": True}

        with unittest.mock.patch.object(url_check, "check", check), \
                unittest.mock.patch.object(url_check.time, "sleep"):
            report = json.loads(self.tool_main(url_check, ["url_check.py", path]))
        self.assertEqual(asked, ["https://example.org/a"])
        self.assertEqual((report["sources"], report["failed"]), (3, 2))
        self.assertEqual(report["failures"],
                         [{"tool": "beta", "url": "", "status": None, "ok": False, "error": "no url"},
                          {"tool": "gamma", "url": "", "status": None, "ok": False, "error": "no url"}])
        self.assertEqual(report["redirected"], [])  # the stripped url is the one reported

    def test_edit_weights_copy_drops_the_training_state(self):
        """F377: the edited copy is an inference file; it keeps neither the optimizer state nor the source's
        averaged (EMA) weights, a second, unedited copy of every weight. main runs on a tiny model; the search
        for candidates (collect, loop_stats, quick_bpb) is replaced so that one candidate always wins."""
        import torch
        import edit_weights as ew
        model = T.TestOtherScripts.model(seed=3)
        obj = {"model": model.state_dict(), "step": 7, "checkpoint_type": "training",
               "metrics": {"history": [1], "last_eval": {"loss": 2.0, "step": 7}},
               "optimizer": {"state": 1}, "rng": {"python": 1}, "batcher": {"pos": 1}, "interval": {"n": 1},
               "ema": {"decay": 0.99, "state": {k: v.clone() for k, v in model.state_dict().items()}}}
        prompts = [(0,) + tuple(1 + (3 * r + c) % 40 for c in range(8)) for r in range(8)]

        class Stats:
            def __init__(self, dim):
                self.dim = dim

            def effect(self, li):
                diff = torch.randn(self.dim, generator=torch.Generator().manual_seed(li)).double()
                return diff, diff

        calls = []

        def loop_stats(seqs, start, eos):  # the original's two scores first, then a lower share per candidate
            calls.append(len(seqs))
            share = 0.5 if len(calls) <= 2 else 0.5 - 0.01 * len(calls)
            return {"repeat_token_share": share, "looping_texts": 0.0, "texts": len(seqs)}

        saved = {}
        dst = os.path.join(self.tmp, "edited.pt")
        with unittest.mock.patch.object(ew.tg, "load_for_inference",
                                        return_value=(model, T.CharTok(), model.cfg, obj)), \
                unittest.mock.patch.object(ew.tg, "resolve_dataset", return_value="dataset"), \
                unittest.mock.patch.object(ew, "prompts_from_validation", return_value=prompts), \
                unittest.mock.patch.object(ew.cm, "validation_documents", return_value=[("md", 0, "text")]), \
                unittest.mock.patch.object(ew, "collect", return_value=(
                    Stats(model.cfg.d_model), Stats(model.cfg.ffn_hidden),
                    {"loop_positions": 5, "fresh_positions": 9})), \
                unittest.mock.patch.object(ew, "loop_stats", loop_stats), \
                unittest.mock.patch.object(ew, "quick_bpb", return_value=1.0), \
                unittest.mock.patch.object(ew.tg, "atomic_save", lambda out, path: saved.update(out) or path), \
                redirect_stdout(io.StringIO()) as out:
            ew.main([os.path.join(self.tmp, "source.pt"), dst, "--device", "cpu", "--prompts", "4",
                     "--dev-prompts", "4", "--prompt-tokens", "8", "--new-tokens", "8", "--batch", "4",
                     "--layers", "2"])
        self.assertIn("Chose layer", out.getvalue())
        self.assertEqual((saved["checkpoint_type"], saved["step"], saved["metrics"]["last_eval"]),
                         ("inference", 7, None))
        self.assertIn("chosen", saved["edit"])
        self.assertEqual(sorted(set(obj) - set(saved)), ["batcher", "ema", "interval", "optimizer", "rng"])
        self.assertIn("ema", obj, "the loaded checkpoint dict itself is left as it was")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "edited_edit.json")))

    def test_loop_rates_per_text_are_computed_in_one_place(self):
        """F375: edit_weights.per_text_rates is the one per-text loop rate. loop_stats leaves out a text with
        no generated token before its end-of-text token (and reports one text at rate 0 when all are such);
        compare_edit keeps 0.0 for it, so that its paired arrays stay aligned. (The loop_stats values are a
        guard: the same before and after.)"""
        import torch
        import compare_edit
        import edit_weights as ew
        start, eos = 4, 9
        looping = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])  # 5 of the 8 generated tokens repeat a 4-gram
        empty = torch.tensor([1, 2, 3, 4, 9, 5, 6, 7, 8, 1, 2, 3])  # ends at once
        fresh = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13])
        seqs = [looping, empty, fresh]
        self.assertEqual(ew.loop_stats(seqs, start, eos),
                         {"repeat_token_share": 0.3125, "looping_texts": 0.5, "texts": 2})
        self.assertEqual(ew.loop_stats([empty, empty], start, eos),
                         {"repeat_token_share": 0.0, "looping_texts": 0.0, "texts": 1})
        self.assertEqual(ew.per_text_rates(seqs, start, eos).tolist(), [0.625, 0.0, 0.0])
        self.assertEqual(ew.per_text_rates(seqs, start, eos, keep_empty=False).tolist(), [0.625, 0.0])
        self.assertEqual(ew.per_text_rates([empty], start, eos, keep_empty=False).tolist(), [])
        self.assertFalse(hasattr(compare_edit, "per_text_rates"), "compare_edit uses edit_weights' function")

    def test_keyword_batch_tools_share_one_reader(self):
        """F382: kw_check and term_evidence read a batch file with one read_batch -> [(term, category)]. The
        kw_check report on a sample batch is a guard: the same before and after."""
        import kw_check
        import term_evidence
        here = os.path.join(self.tmp, "kw")
        os.makedirs(here, exist_ok=True)
        with open(os.path.join(here, "keywords.txt"), "w", encoding="utf-8", newline="\n") as handle:
            handle.write("# genomics\nPolymerase\nhelicase\n\n# proteins\n=MET\n")
        batch = os.path.join(here, "batch.txt")
        with open(batch, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("#comment without a space\n# genomics\npolymerase\n=MET\nab\ngen*\n  polymerase  \n\n"
                         "# new things\nsome new phrase\n+-\nof the\n")
        with unittest.mock.patch.object(kw_check, "HERE", here):
            report = json.loads(self.tool_main(kw_check, ["kw_check.py", batch, "0"]))
        self.assertEqual(report, {
            "batch": batch, "terms": 8, "flagged": 6,
            "flags": {"+-": ["parse_error: no words"], "=MET": ["duplicate"], "ab": ["short_token"],
                      "gen*": ["short_token", "short_prefix"], "of the": ["short_token"],
                      "polymerase": ["duplicate", "in_batch_twice", "duplicate"]},
            "categories": {"genomics": 5, "new things": 3}, "new_categories": ["new things"]})
        rows = [("polymerase", "genomics"), ("=MET", "genomics"), ("ab", "genomics"), ("gen*", "genomics"),
                ("polymerase", "genomics"), ("some new phrase", "new things"), ("+-", "new things"),
                ("of the", "new things")]
        self.assertEqual(term_evidence.read_batch(batch), rows)
        self.assertIs(kw_check.read_batch, term_evidence.read_batch)
        # each finds the shared reader by itself: it imports by path, with tools/ not on sys.path (as it always did)
        script = ("import importlib.util, os, sys\n"
                  "for name in ('kw_check', 'term_evidence'):\n"
                  "    spec = importlib.util.spec_from_file_location('by_path_' + name, os.path.join(sys.argv[1], "
                  "name + '.py'))\n"
                  "    module = importlib.util.module_from_spec(spec)\n"
                  "    spec.loader.exec_module(module)\n"
                  "    print(name, module.read_batch.__module__)\n")
        code, out = T.run(["-c", script, os.path.join(HERE, "tools")], cwd=self.tmp)
        self.assertEqual((code, out.split()), (0, ["kw_check", "_kw", "term_evidence", "_kw"]), out[-600:])

    def test_scripts_with_new_parsers_answer_help_and_do_no_work(self):
        """F146: guide/make_figures.py and four tools had no argument parser, so `--help` redrew the guide's
        figures, wrote files named '--help' or failed with FileNotFoundError. Each now prints its usage and
        docstring, exits 0 and writes nothing. A script whose source has no parser is not started here."""
        scripts = [os.path.join("guide", "make_figures.py")] + [
            os.path.join("tools", name) for name in ("term_evidence.py", "url_check.py", "kw_check.py",
                                                     "kw_covered.py")]
        figures = os.path.join(HERE, "guide", "figures")

        def stamps():
            return {name: os.stat(os.path.join(figures, name)).st_mtime_ns for name in sorted(os.listdir(figures))}

        before = stamps()
        for rel in scripts:
            with open(os.path.join(HERE, rel), encoding="utf-8") as handle:
                source = handle.read()
            self.assertIn("ArgumentParser(", source, rel)
            cwd = tempfile.mkdtemp(dir=self.tmp)
            code, out = T.run([os.path.join(HERE, rel), "--help"], cwd=cwd)
            self.assertEqual((code, os.listdir(cwd)), (0, []), (rel, out[-600:]))
            self.assertIn("usage: " + os.path.basename(rel), out)
            self.assertIn(source.split('"""')[1].strip().splitlines()[0], out, rel)
        self.assertEqual(stamps(), before, "--help must not redraw the guide's figures")

    def test_keyword_tools_take_their_arguments_through_a_parser(self):
        """F146: kw_covered takes --nested and --fix before or after the file (the file used to be whatever
        stood first); term_evidence takes OUT.tsv and one or more batch files."""
        from contextlib import redirect_stderr
        import kw_covered
        import term_evidence
        path = os.path.join(self.tmp, "covered.txt")
        text = "# words\npolymer*\npolymerase\nChina Kadoorie\nChina Kadoorie Biobank\n"
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        with redirect_stderr(io.StringIO()) as err:
            out = self.tool_main(kw_covered, ["kw_covered.py", "--nested", path])
        self.assertEqual(out, "China Kadoorie Biobank\tnested\tChina Kadoorie\npolymerase\tsame length\tpolymer*\n")
        self.assertIn("1 same-length covered terms, 1 nested", err.getvalue())
        with open(path, encoding="utf-8", newline="") as handle:
            self.assertEqual(handle.read(), text, "nothing is removed without --fix")
        with redirect_stderr(io.StringIO()):
            self.tool_main(kw_covered, ["kw_covered.py", path, "--fix"])
        with open(path, encoding="utf-8", newline="") as handle:
            self.assertEqual(handle.read(), text.replace("polymerase\n", ""))

        batches = []
        for name, body in (("a.txt", "# genes\nBRCA1\n*mab\n"), ("b.txt", "# methods\nlinear regression\n")):
            batches.append(os.path.join(self.tmp, name))
            with open(batches[-1], "w", encoding="utf-8", newline="\n") as handle:
                handle.write(body)
        table = os.path.join(self.tmp, "evidence.tsv")
        with unittest.mock.patch.object(term_evidence, "hits", lambda session, query: len(query)):
            said = self.tool_main(term_evidence, ["term_evidence.py", table] + batches)
        self.assertIn("3 terms, 2 queries to run", said)
        with open(table, encoding="utf-8") as handle:
            self.assertEqual(handle.read().splitlines(),
                             ["term\tcategory\tbatch\tquery\thits", 'BRCA1\tgenes\ta.txt\t"BRCA1"\t7',
                              "*mab\tgenes\ta.txt\t\t-1", 'linear regression\tmethods\tb.txt\t"linear regression"\t19'])
        with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as stop:
            self.tool_main(term_evidence, ["term_evidence.py", os.path.join(self.tmp, "none.tsv")])
        self.assertEqual(stop.exception.code, 2, "a table without a batch file is a usage error")
        self.assertIn("BATCH.txt", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "none.tsv")))


if __name__ == "__main__":
    unittest.main()
