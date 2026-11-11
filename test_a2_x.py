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


if __name__ == "__main__":
    unittest.main()
