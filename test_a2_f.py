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


if __name__ == "__main__":
    unittest.main()
