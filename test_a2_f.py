"""Staged tests of audit 2, package F (download_docs.py, youtube_transcripts.py, convert_to_markdown.py).

Fast, CPU, temporary folders, no network. To be folded into test_tiny_gpt.TestOtherScripts.
"""

import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class Audit2DocsYoutubeConvert(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2f_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---- download_docs.py -------------------------------------------------------------------------------
    def test_docs_robots_429_is_not_allow_all(self):
        """F135: a robots.txt answered with 429 is an error for the row, not 'allow everything' for the host."""
        import download_docs as d
        http = d.Http((0.0, 0.0))

        def get(url, *a, **k):
            raise d.HttpError(429, url)

        with unittest.mock.patch.object(http, "get", get):
            with self.assertRaises(d.HttpError) as caught:
                http.allowed("https://a.org/docs/x.html")
        self.assertEqual(caught.exception.code, 429)
        self.assertEqual(http.robots, {})  # not cached: the next URL of the host asks again
        http = d.Http((0.0, 0.0))  # a 404 still means 'no robots.txt': everything allowed, and cached

        def missing(url, *a, **k):
            raise d.HttpError(404, url)

        with unittest.mock.patch.object(http, "get", missing):
            self.assertTrue(http.allowed("https://a.org/docs/x.html"))
        self.assertTrue(http.robots["https://a.org"].allow_all)

    def test_docs_pdf_tags_empty_script(self):
        """F129: an empty <sup></sup> or <sub> </sub> leaves no stray ^ or _."""
        import download_docs as d
        self.assertEqual(d.pdf_tags("x<sup></sup>y and a<sub> </sub>b"), "xy and ab")
        self.assertEqual(d.pdf_tags("x<sup>2</sup> a<sub>ij</sub>"), "x^2 a_{ij}")  # unchanged

    def test_docs_rd_keeps_r_escapes_in_code(self):
        """F128: in \\usage and \\examples a backslash escape of an R string ("\\t", "\\n", "\\d") is kept as
        written; in text sections an unknown macro still renders its (missing) first argument."""
        import download_docs as d
        md = d.rd_markdown(r'\name{f}\title{F}\usage{f(sep = "\t")}\examples{cat("a\n"); x <- gsub("\\d", "", y)}'
                           r'\value{V.}')
        self.assertIn(r'sep = "\t"', md)
        self.assertIn(r'cat("a\n")', md)
        self.assertIn(r'gsub("\d", "", y)', md)
        self.assertIn("## Value", md)
        text = d.rd_markdown(r"\name{f}\title{F}\description{\pkg{stats} and \unknownmacro}")
        self.assertEqual(text, "# F\n\n## Description\n\nstats and\n")
        # known Rd macros inside code keep their meaning
        self.assertIn("f(x, ...)", d.rd_markdown(r"\name{f}\title{F}\usage{f(x, \dots)}"))


if __name__ == "__main__":
    unittest.main()
