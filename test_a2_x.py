"""Staging tests of the 2026-11-10 audit, package X: follower edits across files (callers of names that the
file packages added). They move into test_tiny_gpt.py (TestOtherScripts) when the packages are merged.

    python -m unittest test_a2_x
"""
import gzip
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from types import SimpleNamespace as types_ns

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
import test_a2_h as H  # noqa: E402
import test_tiny_gpt as T  # noqa: E402

KEYWORDS = "genetic*\nallele*\ndrift\nselection\n"
MATCH = ("Genetic drift and allele frequencies", "Genetic drift changes allele frequencies; selection acts too.")
OTHER = ("Cooking pasta", "Boil the water, add salt.")


class Audit2Followers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2x_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def keyword_file(self):
        path = os.path.join(self.tmp, "kw.txt")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(KEYWORDS)
        return path

    # ---------------- X1: callers that only call evaluate() build no RE2 patterns ----------------
    def filters_built(self, call):
        """The KeywordFilter objects `call` builds, as (object, positional arguments, keyword arguments), while
        building the RE2 prefilter patterns is an error."""
        import data_prep as dp
        real, built = dp.KeywordFilter, []

        def recording(*a, **k):
            built.append((real(*a, **k), a, k))
            return built[-1][0]

        with unittest.mock.patch.object(dp, "KeywordFilter", recording), \
                unittest.mock.patch.object(dp, "_prefilter_patterns",
                                           side_effect=AssertionError("the prefilter patterns were built")), \
                redirect_stdout(io.StringIO()):
            call()
        return built

    def test_callers_that_only_evaluate_build_no_prefilter_patterns(self):
        """X1 (F093, F143): the downloaders, posttrain.py and tools/kw_check.py only call
        KeywordFilter.evaluate, so they pass prefilter=False: no RE2 pattern is built at their start (minutes
        for the full keyword file) and evaluate() answers as the default object's does."""
        import data_prep as dp
        import download_arxiv
        import download_pmc
        import download_rxiv
        import kw_check
        import posttrain
        import pyarrow as pa
        import pyarrow.parquet as pq
        import youtube_transcripts
        keywords = self.keyword_file()
        work = os.path.join(self.tmp, "x1")
        os.makedirs(os.path.join(work, "kw_here"))
        papers = [MATCH, OTHER]
        records = os.path.join(work, "records.jsonl.gz")
        with gzip.open(records, "wt", encoding="utf-8") as f:
            for i, (title, abstract) in enumerate(papers):
                f.write(json.dumps({"pmcid": f"PMC{i}", "pmid": str(i), "year": "2020", "cited": i, "journal": "J",
                                    "lang": "eng", "title": title, "abstract": abstract}) + "\n")
        arxiv = [{"id": str(i), "created": "2021-01-01", "title": title, "abstract": abstract,
                  "categories": "q-bio.PE", "license": ""} for i, (title, abstract) in enumerate(papers)]
        rxiv = [{"doi": f"10.1101/{i}", "title": title, "abstract": abstract}
                for i, (title, abstract) in enumerate(papers)]
        shutil.copy(keywords, os.path.join(work, "kw_here", "keywords.txt"))
        batch = os.path.join(work, "batch.txt")
        with open(batch, "w", encoding="utf-8", newline="\n") as f:
            f.write("# new things\n" + KEYWORDS)
        web = os.path.join(work, "web.parquet")
        pq.write_table(pa.table({"text": [" ".join(MATCH), " ".join(OTHER)]}), web)
        data_args = types_ns(data_dir=os.path.join(work, "posttrain"), seed=0, keywords=keywords, pubmedqa=0,
                             medmcqa=0, mcq_pairs=0, smoltalk=0, ultrafeedback=0)

        def pmc_list():
            download_pmc.keyword_list(records, keywords, 3, 3, 100, "density", os.path.join(work, "pmc.tsv"))

        def arxiv_list():
            with unittest.mock.patch.object(download_arxiv, "read_cache", return_value=arxiv):
                download_arxiv.build_list(work, ["q-bio"], keywords, 3, 3, os.path.join(work, "arxiv.tsv"))

        def rxiv_list():
            with unittest.mock.patch.object(download_rxiv, "read_cache", return_value=rxiv):
                download_rxiv.build_list(work, keywords, 3, 3, os.path.join(work, "rxiv.tsv"))

        def posttrain_data():
            with unittest.mock.patch.multiple(
                    posttrain, eval_questions=lambda: ([], set()), pubmedqa_sft=lambda cap, ids: [],
                    medmcqa=lambda cap, pairs, rng: ([], [], {"cop_base": 0, "cop_check": "none"}),
                    sciq=lambda pairs, rng: ([], []), smoltalk=lambda cap, kw: ([], 0),
                    ultrafeedback=lambda cap, kw: ([], 0)):
                posttrain.build_data(data_args)

        def youtube_options():
            youtube_transcripts.parse_args(["@channel", "--out", os.path.join(work, "yt"), "--keywords", keywords])

        def keyword_check():
            raw = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")  # kw_check reconfigures sys.stdout
            with unittest.mock.patch.object(sys, "argv", ["kw_check.py", batch, "2"]), \
                    unittest.mock.patch.object(sys, "stdout", raw), \
                    unittest.mock.patch.object(kw_check, "HERE", os.path.join(work, "kw_here")), \
                    unittest.mock.patch.object(kw_check.glob, "glob", return_value=[web]):
                kw_check.main()

        texts = [MATCH, OTHER, ("", "allele " * 5), (None, "Selection of genetics after drift"), ("Drift", "")]
        sites = (("download_pmc.keyword_list", pmc_list, 1), ("download_arxiv.build_list", arxiv_list, 1),
                 ("download_rxiv.build_list", rxiv_list, 1), ("posttrain.build_data", posttrain_data, 1),
                 ("youtube_transcripts.parse_args", youtube_options, 2), ("tools/kw_check.py", keyword_check, 1))
        for name, call, count in sites:
            with self.subTest(site=name):
                built = self.filters_built(call)
                self.assertEqual(len(built), count)
                for kw, a, k in built:
                    self.assertIs(k.get("prefilter"), False)
                    self.assertEqual(kw.include_patterns, [])
                    default = dp.KeywordFilter(*a, **{key: v for key, v in k.items() if key != "prefilter"})
                    self.assertTrue(default.include_patterns, "the default object has the patterns")
                    for title, text in texts:
                        self.assertEqual(kw.evaluate(title, text), default.evaluate(title, text), (title, text))
                    self.assertEqual([kw.evaluate(*t)[0] for t in texts[:2]], [True, False])

    # ---------------- X2: the estimate gates the keyword prefilter as the build does ----------------
    def run_estimate(self, argv):
        return H.Audit2PipelineTools.run_estimate(self, argv)  # the helper of package H (one method after the fold)

    def tokenizer_model(self, name):
        return H.Audit2PipelineTools.tokenizer_model(self, name)

    def test_estimate_gates_the_keyword_prefilter_like_the_build(self):
        """X2 (F047): a source read in stream order ('first') gets the RE2 keyword counts only when the
        build's readers use them (data_prep._use_prefilter: not above PREFILTER_MAX_PATTERNS patterns); a
        ranked source ('top') always gets them, they are its ranking score. Either way the estimate keeps
        the documents the build keeps, with the build's skip reasons."""
        import data_prep as dp
        import estimate_dataset
        c = T.Corpus(os.path.join(self.tmp, "x2_corpus"))
        model, _ = self.tokenizer_model("x2_tok")
        common = ["--wiki-dir", c.wiki, "--parquet", c.parquet, "--parquet-name", "web", "--min-score", "2",
                  "--include-keywords", "genetic*,allele*,protein,DNA,RNA,genome*", "--keyword-min-distinct", "2"]
        real_scores, real_source = dp._vector_scores, estimate_dataset.table_source
        for mode, wiki_max in (("first", "0"), ("top", "30")):
            out = os.path.join(self.tmp, f"x2_build_{mode}")
            with redirect_stdout(io.StringIO()):
                dp.prepare(["--out", out, "--wiki-max-docs", wiki_max, "--vocab-size", "700", "--val-fraction", "0.2",
                            "--threads", "2", "--workers", "1"] + common)
            sources = dp.read_manifest(out)["sources"]
            self.assertEqual((sources["wiki"]["select"], sources["web"]["select"]), (mode, "first"))
            for limit in (0, 10 ** 9):  # prefilter never used / always used in stream order
                with_kw, tallies = [], {}

                def recorded(arrays, kw, *a, **k):
                    with_kw.append(kw is not None)
                    return real_scores(arrays, kw, *a, **k)

                def source(name, *a, **k):
                    tallies[name] = real_source(name, *a, **k)
                    return tallies[name]

                with unittest.mock.patch.object(dp, "PREFILTER_MAX_PATTERNS", limit), \
                        unittest.mock.patch.object(dp, "_vector_scores", recorded), \
                        unittest.mock.patch.object(estimate_dataset, "table_source", source):
                    self.run_estimate(["--sample", "1", "--tflops", "1", "--hours", "1", "--tokenizer-from", model,
                                       "--wiki-max-docs", wiki_max] + common)
                if limit:
                    self.assertEqual(set(with_kw), {True}, (mode, limit))
                else:  # only the ranked source is counted
                    self.assertEqual(set(with_kw), {False} if mode == "first" else {True, False}, (mode, limit))
                for name in ("wiki", "web"):
                    t, built = tallies[name], sources[name]
                    self.assertEqual((t.select, t.total, round(t.estimate()[0])),
                                     (built["select"], built["scanned"], built["kept_docs"]), (mode, limit, name))
                    self.assertEqual(t.reasons, {reason: n for reason, n in built["skipped"].items()
                                                 if n and reason in ("score_below_min", "keyword_min_hits")},
                                     (mode, limit, name))
            self.assertEqual(sources["wiki"]["skipped"]["keyword_min_hits"], 20)
            self.assertEqual(sources["web"]["skipped"], {"score_below_min": 12, "keyword_min_hits": 9})

    # ---------------- X3: the estimate trains its tokenizer with data_prep's options ----------------
    def test_estimate_tokenizer_is_trained_with_the_options_of_the_build(self):
        """X3 (F075, F154): estimate_dataset.train_tokenizer passes data_prep.SPM_TRAIN_OPTIONS (it had its own
        copy of the sixteen values) and adds only the five options that are the caller's; the SuperBPE share
        is data_prep.superbpe_extra's."""
        import sentencepiece as spm
        import data_prep as dp
        import estimate_dataset
        tally = estimate_dataset.Tally("md", "markdown")
        for i in range(6):
            tally.keep(T.paragraphs(40 + i, 30))
        real, calls = spm.SentencePieceTrainer.train, []

        def train(**options):
            calls.append(options)
            return real(**options)

        def trained(tag, superbpe, **patches):
            work = os.path.join(self.tmp, "x3_" + tag)
            os.makedirs(work)
            patches.setdefault("SPM_TRAIN_OPTIONS", dp.SPM_TRAIN_OPTIONS)  # patch.multiple wants one name
            args = types_ns(superbpe=superbpe, superbpe_fraction=0.1, superbpe_max_words=4, seed=7,
                            tokenizer_weights=None)
            with unittest.mock.patch.object(spm.SentencePieceTrainer, "train", train), \
                    unittest.mock.patch.multiple(dp, **patches):
                tok = estimate_dataset.train_tokenizer([tally], 600, work, args, {"md": tally.estimate()})
            return work, tok, calls[-1]

        work, tok, options = trained("plain", False)
        own = dict(input=os.path.join(work, "corpus.txt"), model_prefix=os.path.join(work, "spm"), vocab_size=600,
                   input_sentence_size=3_000_000, num_threads=os.cpu_count() or 4)
        self.assertEqual(options, dict(dp.SPM_TRAIN_OPTIONS, **own))
        self.assertEqual(len(options), 21)
        self.assertGreater(tok.vocab_size, 100)
        # the constant is what is passed, not a copy of today's values
        marked = dict(dp.SPM_TRAIN_OPTIONS, max_sentence_length=8192)
        self.assertEqual(trained("marked", False, SPM_TRAIN_OPTIONS=marked)[2]["max_sentence_length"], 8192)
        # SuperBPE: the first stage gets the vocabulary less the share of the second, both from data_prep
        extended = []

        def extend(proto, corpus, extra, max_words, **kwargs):
            extended.append(extra)
            return proto, []

        self.assertEqual(trained("superbpe", True, superbpe_extend=extend)[2]["vocab_size"], 540)
        self.assertEqual(trained("share", True, superbpe_extend=extend,
                                 superbpe_extra=lambda vocab, args: 7)[2]["vocab_size"], 593)
        self.assertEqual(extended, [60, 7])

    # ---------------- X4: the estimate's training hours use the planner's efficiency ----------------
    def test_estimate_hours_use_the_efficiency_of_the_time_budget(self):
        """X4 (F346, F149): the hours column and the tokens that fit an hours budget are computed with
        tiny_gpt.TIME_BUDGET_EFFICIENCY (two copies of 0.92 before)."""
        import tiny_gpt
        self.assertEqual(tiny_gpt.TIME_BUDGET_EFFICIENCY, 0.92)
        folder = H.Audit2PipelineTools.text_folder(self, "x4_text")
        model, _ = self.tokenizer_model("x4_tok")
        argv = ["--md-dir", folder, "--tokenizer", model, "--tflops", "0.00001", "--hours", "1",
                "--tokens-per-param", "0.0001"]

        def table(efficiency):
            with unittest.mock.patch.object(tiny_gpt, "TIME_BUDGET_EFFICIENCY", efficiency):
                lines = self.run_estimate(argv).splitlines()
            rows = {line[:16].strip(): line.split() for line in lines if line.startswith(("data only", "1 h budget"))}
            self.assertEqual(sorted(rows), ["1 h budget", "data only"])
            return float(rows["data only"][-2].replace(",", "")), rows["1 h budget"]

        hours, budget = table(0.92)
        half_hours, half_budget = table(0.46)
        self.assertGreater(hours, 100)
        self.assertAlmostEqual(half_hours / hours, 2, places=3)
        self.assertEqual((budget[-2], half_budget[-2]), ("1.0", "1.0"))
        self.assertNotEqual(budget, half_budget, "half the efficiency: a smaller model fits the hour")

    # ---------------- X5: the run lock is data_prep's lock with the run's message ----------------
    def test_run_lock_is_the_build_lock_with_the_message_of_the_run(self):
        """X5 (F266, F265): tiny_gpt.acquire_run_lock makes the folder and calls data_prep._acquire_lock, so a
        lock another trainer has created but not written yet counts as alive (the copy of the loop in
        tiny_gpt removed it as stale); the refusal names the run, not a data_prep.py build."""
        import time
        import data_prep
        import tiny_gpt
        args = types_ns(out_dir=os.path.join(self.tmp, "x5", "runs"), name="R")
        lock = os.path.join(args.out_dir, "R.train.lock")
        os.makedirs(args.out_dir)
        open(lock, "w").close()  # another start, between creating the lock and writing it
        with self.assertRaises(SystemExit) as stop:
            tiny_gpt.acquire_run_lock(args)
        said = str(stop.exception)
        self.assertEqual(said, f"Run 'R' in {args.out_dir} is being trained by another process (see {lock}). Wait "
                               "for it to finish or stop it first; delete the lock file only if no training is "
                               "running.")
        self.assertNotIn("data_prep.py run", said)
        self.assertEqual(os.path.getsize(lock), 0, "the other trainer's lock was not touched")
        old = time.time() - 60  # the same empty file left by a crash: stale after ten seconds
        os.utime(lock, (old, old))
        with unittest.mock.patch.object(data_prep, "_acquire_lock", wraps=data_prep._acquire_lock) as shared:
            self.assertEqual(tiny_gpt.acquire_run_lock(args), lock)
        shared.assert_called_once_with(lock, busy=said)
        with open(lock, encoding="utf-8") as f:
            self.assertEqual(sorted(json.load(f)), ["pid", "started"], "the lock file's format is unchanged")
        tiny_gpt.release_run_lock(lock)
        self.assertFalse(os.path.exists(lock))
        # a folder that does not exist yet is made
        fresh = types_ns(out_dir=os.path.join(self.tmp, "x5", "new", "runs"), name="R")
        tiny_gpt.release_run_lock(tiny_gpt.acquire_run_lock(fresh))
        self.assertEqual(os.listdir(fresh.out_dir), [])

    # ---------------- X6: the viewer loads and builds through tiny_gpt ----------------
    def test_viewer_model_and_file_come_from_the_loaders_of_tiny_gpt(self):
        """X6 (F405, F393, F329): view_pt.build_model is tiny_gpt.model_from_checkpoint(assign=True) and
        view_pt.load_checkpoint is tiny_gpt.load_any(mmap=True); the viewer had a copy of each. The model
        still holds the file's own tensors and gives the logits of an ordinary load, and it now gets the
        document mask of a run trained with --doc-mask on."""
        import torch
        import test_a2_c as C
        import tiny_gpt
        import view_pt
        folder = os.path.join(self.tmp, "x6")
        os.makedirs(folder)
        tok, cpu = T.CharTok(), torch.device("cpu")
        tokens = torch.tensor([[3, 4, tok.bos_id, 5, 6, 7, tok.bos_id, 8, 9, 10]])  # three documents in one window
        logits = {}
        for doc_mask in (False, True):
            path = C.toy_file(folder, f"mask_{doc_mask}.pt", train_config=dict(C.TRAIN_CONFIG, doc_mask=doc_mask))
            with unittest.mock.patch.object(tiny_gpt, "tokenizer_from_checkpoint", return_value=tok):
                ck = view_pt.Checkpoint(path)
                model = view_pt.build_model(ck, cpu)
                plain = tiny_gpt.model_from_checkpoint(tiny_gpt.safe_load(path))[0]
            self.assertFalse(model.training)
            self.assertEqual(model.doc_sep_id, tok.bos_id if doc_mask else None)
            self.assertEqual(model.token_embedding.weight.data_ptr(),
                             ck.obj["model"]["token_embedding.weight"].data_ptr(), "no second copy of the weights")
            with torch.no_grad():
                logits[doc_mask] = model(tokens)[0]
                self.assertTrue(torch.equal(logits[doc_mask], plain(tokens)[0]))
        self.assertFalse(torch.equal(logits[False], logits[True]), "the mask changes what a later document sees")
        with unittest.mock.patch.object(tiny_gpt, "load_any", return_value={"format": "none"}) as load:
            self.assertEqual(view_pt.load_checkpoint(path, True), {"format": "none"})
        load.assert_called_once_with(path, True, mmap=True)
        with open(view_pt.__file__, encoding="utf-8") as f:
            self.assertNotIn("attention_weights", f.read(), "a function that no longer exists")

    # ---------------- X7: edit_weights samples with tiny_gpt.sample_next ----------------
    def test_edit_weights_samples_through_tiny_gpt(self):
        """X7 (F311): tools/edit_weights.py draws its continuations with tiny_gpt.sample_next (it had a copy of
        the function): every token of a seeded batch is one that function returned."""
        import torch
        import edit_weights as ew
        import tiny_gpt
        model = T.TestFixes.small_model(None, seed=3)
        prompts = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 9, 9, 9)]
        real, drawn = tiny_gpt.sample_next, []

        def sample(logits, generator, *settings):
            self.assertEqual(settings, (0.8, 50, 0.95))
            drawn.append(real(logits, generator, *settings))
            return drawn[-1]

        with unittest.mock.patch.object(tiny_gpt, "sample_next", sample):
            out = ew.generate_batch(model, prompts, 6, torch.device("cpu"), None, seed=11)
        self.assertEqual(len(drawn), 6)
        self.assertFalse(hasattr(ew, "sample_next"), "one copy, in tiny_gpt")
        self.assertEqual(out[:, :4].tolist(), [list(p) for p in prompts])
        self.assertEqual(out[:, 4:].tolist(), torch.stack(drawn, dim=1).tolist())
        self.assertGreater(len(set(out[:, 4:].flatten().tolist())), 6, "sampled, not one repeated token")

    # ---------------- X8: the edit and its test never share a validation document ----------------
    def test_edit_and_its_comparison_use_disjoint_validation_documents(self):
        """X8 (F381): tools/edit_weights.py takes its prompts and its safety-check documents from validation
        documents with an even index within their source, tools/compare_edit.py its prompts and its documents
        from those with an odd index, whatever number each asks for (the split was by position in two lists
        of different spacing, which overlapped). Both mains run up to their first generation."""
        import numpy as np
        import compare_edit
        import compare_models
        import data_prep
        import edit_weights as ew
        sizes = {"big": 41, "mid": 7, "one": 1, "two": 2}
        doc = [0] + [1 + i % 40 for i in range(250)]  # a start token, then 250 characters of CharTok text
        arrays = {"val": {s: np.array(doc * n, dtype=np.uint16) for s, n in sizes.items()}}
        manifest = {"tokenizer": {"file": "tokenizer.model", "encode_mode": "lines"}}
        model = T.TestOtherScripts.model(seed=3)
        real, calls = compare_models.validation_documents, []

        def recorded(dataset, per_source, max_chars, layout=None, parity=None):
            docs = real(dataset, per_source, max_chars, layout, parity)
            calls.append((per_source, max_chars, parity, [(s, k) for s, k, _ in docs]))
            return docs

        class Reached(Exception):
            pass

        def generate_all(*a, **k):
            raise Reached

        def asked(main, argv):
            """The validation_documents calls of a tool's main up to its first generation."""
            del calls[:]
            with unittest.mock.patch.object(data_prep, "open_token_files",
                                            lambda folder, verify=True: (manifest, arrays)), \
                    unittest.mock.patch.object(data_prep.Tokenizer, "from_file", lambda *a, **k: T.CharTok()), \
                    unittest.mock.patch.object(compare_models, "validation_documents", recorded), \
                    unittest.mock.patch.object(ew.tg, "load_for_inference",
                                               return_value=(model, T.CharTok(), model.cfg, {"step": 7})), \
                    unittest.mock.patch.object(ew.tg, "resolve_dataset", return_value="dataset"), \
                    unittest.mock.patch.object(ew, "generate_all", generate_all), \
                    redirect_stdout(io.StringIO()), self.assertRaises(Reached):
                main(argv)
            return list(calls)

        sizes_of = ["--device", "cpu", "--prompt-tokens", "8", "--new-tokens", "4"]
        edit = asked(ew.main, [os.path.join(self.tmp, "x8_source.pt"), os.path.join(self.tmp, "x8_edited.pt"),
                               "--prompts", "30", "--dev-prompts", "10"] + sizes_of)
        test = asked(compare_edit.main, [os.path.join(self.tmp, "x8_source.pt"),
                                         os.path.join(self.tmp, "x8_edited.pt"), "--prompts", "20", "--docs", "5"]
                     + sizes_of)
        used, held_out = {d for c in edit for d in c[3]}, {d for c in test for d in c[3]}
        self.assertFalse(used & held_out, "a held-out document was used by the edit")
        self.assertEqual([c[:3] for c in edit], [(40, 4000, 0), (3, 6000, 0)])
        self.assertEqual([c[:3] for c in test], [(20, 4000, 1), (5, 20_000, 1)])
        self.assertTrue(all(k % 2 == 0 for _, k in used) and all(k % 2 == 1 for _, k in held_out))
        self.assertEqual(({s for s, _ in used}, {s for s, _ in held_out}), (set(sizes), {"big", "mid", "two"}))
        # the safety check gets --bpb-docs documents per source (fewer from a source that has fewer even ones)
        per_source = {s: sum(1 for name, _ in edit[1][3] if name == s) for s in sizes}
        self.assertEqual(per_source, {"big": 3, "mid": 3, "one": 1, "two": 1})
        for module in (ew, compare_edit):
            self.assertIn("index within their source", " ".join(module.__doc__.split()))
            self.assertNotIn("numbered", module.__doc__)
        self.assertIn("index within their source", " ".join(ew.prompts_from_validation.__doc__.split()))

    # ---------------- X12: the label writer counts columns as the probe reader does ----------------
    def test_label_writer_ignores_trailing_tabs_like_the_probe_reader(self):
        """X12 (F351): read_probes takes a three-column line that ends in tabs as the short form;
        calibrate_probes.rewrite_difficulties counted the empty fields as columns and wrote the difficulty
        over the ANSWER of such a line. It drops trailing empty fields first, as the reader does."""
        import calibrate_probes
        import tiny_gpt
        path = os.path.join(self.tmp, "x12_probes.tsv")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("# comment\nGenes are made of\tDNA\tsugar|fat\t\t\n\n"
                    "genetics\tunrated\tCells divide by\tmitosis\tfusion|decay\t \n")
        self.assertEqual([p["answer"] for p in tiny_gpt.read_probes(path)], ["DNA", "mitosis"])
        calibrate_probes.rewrite_difficulties(path, ["easy", "hard"])
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "# comment\ngeneral\teasy\tGenes are made of\tDNA\tsugar|fat\n\n"
                                       "genetics\thard\tCells divide by\tmitosis\tfusion|decay\n")
        self.assertEqual(tiny_gpt.read_probes(path), [
            {"category": "general", "difficulty": "easy", "prompt": "Genes are made of", "answer": "DNA",
             "distractors": ["sugar", "fat"]},
            {"category": "genetics", "difficulty": "hard", "prompt": "Cells divide by", "answer": "mitosis",
             "distractors": ["fusion", "decay"]}])

    # ---------------- X13: the viewer's dataset line asks tiny_gpt.resolve_dataset ----------------
    def test_summary_finds_the_dataset_of_an_export_folder_by_name_and_fingerprint(self):
        """X13 (F284): an export folder stores the name of its dataset and the fingerprint, not a path of
        this machine, so the viewer's 'dataset:' line asks tiny_gpt.resolve_dataset where it tested the
        stored path with os.path.isdir ('NOT FOUND' for every export), and it leaves the 'train tokens'
        part out when the entry has none ('train tokens None'). A folder that was rebuilt since is not the
        model's dataset."""
        import test_a2_c as C
        import data_prep
        import tiny_gpt
        import view_pt
        work = os.path.join(self.tmp, "x13")
        dataset, fingerprint = os.path.join(work, "datasets", "ds_toy"), "f" * 64
        os.makedirs(dataset)

        def manifest(value):
            with open(os.path.join(dataset, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"format": data_prep.FORMAT, "format_version": data_prep.FORMAT_VERSION,
                           "fingerprint": value}, f)

        def dataset_lines(path):
            with redirect_stdout(io.StringIO()) as out:
                view_pt.print_summary(view_pt.Checkpoint(path))
            return [line for line in out.getvalue().splitlines() if "dataset" in line.split("training settings")[0]]

        manifest(fingerprint)
        with open(self.tokenizer_model("x13_tok")[0], "rb") as f:
            tok = data_prep.Tokenizer(f.read())
        file = C.toy_file(work, "A_best.pt", history=[], tokenizer={"proto": tok.proto, **tok.meta()},
                          dataset={"path": dataset, "fingerprint": fingerprint, "train_tokens": {"md": 1000}})
        export = os.path.join(work, "exported")
        with redirect_stdout(io.StringIO()):
            tiny_gpt.cmd_export(tiny_gpt.parse_args(["--checkpoint", file, "--export", export, "--device", "cpu"]))
        self.assertEqual(tiny_gpt.load_exported(export)["dataset"], {"path": "ds_toy", "fingerprint": fingerprint})
        whole = [f"dataset: {dataset} (found) | fingerprint ffffffffffff | train tokens {{'md': 1000}}"]
        self.assertEqual(dataset_lines(file), whole, "the line of a training checkpoint is as it was")
        with unittest.mock.patch.object(tiny_gpt, "HERE", work):  # datasets/ds_toy beside the scripts
            self.assertEqual(dataset_lines(export), ["dataset: ds_toy (found) | fingerprint ffffffffffff"])
        missing = ["dataset: ds_toy (NOT FOUND) | fingerprint ffffffffffff"]
        with unittest.mock.patch.object(tiny_gpt, "HERE", export):  # no datasets/ds_toy there
            self.assertEqual(dataset_lines(export), missing)
        manifest("0" * 64)  # rebuilt: same name, another fingerprint
        with unittest.mock.patch.object(tiny_gpt, "HERE", work):
            self.assertEqual(dataset_lines(export), missing)
        self.assertEqual(dataset_lines(file), [whole[0].replace("(found)", "(NOT FOUND)")])

    # ---------------- X14: compare_models finds a model's training dataset as tiny_gpt does ----------------
    def test_training_overlap_finds_a_dataset_stored_by_name_in_datasets(self):
        """X14 (F284): a model from an export folder (or from another machine) records its training dataset
        as a name and a fingerprint. compare_models.training_overlap asks tiny_gpt.resolve_dataset for the
        folder, so the documents that model trained on are found in datasets/<name>/docs.tsv and left out;
        the bare name was tried from the current folder only ('not on this computer'). A folder of that name
        with another fingerprint, or a name without a fingerprint, is not taken."""
        import compare_models
        import data_prep
        import tiny_gpt
        work = os.path.join(self.tmp, "x14")
        dataset = os.path.join(work, "compared")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", dataset, "--md-dir", H.Audit2PipelineTools.text_folder(self, "x14_text", 30),
                               "--vocab-size", "700", "--val-fraction", "0.3", "--threads", "2", "--workers", "1"])
        layout = {}
        docs = compare_models.validation_documents(dataset, 4, 2000, layout)
        self.assertGreater(len(docs), 1)
        # the other model's dataset: the same documents, all of them in its training split
        other, fingerprint = os.path.join(work, "datasets", "ds_other"), "b" * 64
        os.makedirs(other)
        with open(os.path.join(dataset, "docs.tsv"), encoding="utf-8", newline="") as f:
            ledger = f.read()
        self.assertIn("\tval\t", ledger)
        with open(os.path.join(other, "docs.tsv"), "w", encoding="utf-8", newline="") as f:
            f.write(ledger.replace("\tval\t", "\ttrain\t"))

        def overlap(entry, written=fingerprint):
            with open(os.path.join(other, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"format": data_prep.FORMAT, "format_version": data_prep.FORMAT_VERSION,
                           "fingerprint": written}, f)
            with unittest.mock.patch.object(tiny_gpt, "HERE", work), redirect_stdout(io.StringIO()) as out:
                found = compare_models.training_overlap(docs, dataset, layout, {"B": entry})
            return found, out.getvalue()

        everything = set(range(len(docs)))
        found, said = overlap({"path": "ds_other", "fingerprint": fingerprint})
        self.assertEqual(found, (everything, {}, {}))
        self.assertIn(f"model B was trained on {other}: {len(docs)} of these documents", said)
        # stored on the other kind of system: the last part of the path is the name
        self.assertEqual(overlap({"path": "/data/sets/ds_other/", "fingerprint": fingerprint})[0],
                         (everything, {}, {}))
        # a full path that is the dataset is used as before
        self.assertEqual(overlap({"path": other, "fingerprint": fingerprint})[0], (everything, {}, {}))
        missing = {"B": "its training dataset ds_other (or its docs.tsv) is not on this computer"}
        self.assertEqual(overlap({"path": "ds_other", "fingerprint": fingerprint}, written="c" * 64),
                         ((set(), missing, {}), ""), "datasets/ds_other was rebuilt since")
        self.assertEqual(overlap({"path": "ds_other"}), ((set(), missing, {}), ""),
                         "without a fingerprint a folder found by its name cannot be checked")


if __name__ == "__main__":
    unittest.main()
