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
import test_tiny_gpt as T  # noqa: E402,F401

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


if __name__ == "__main__":
    unittest.main()
