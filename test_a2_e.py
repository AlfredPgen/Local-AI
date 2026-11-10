"""Staging tests of package E (download_arxiv.py, download_rxiv.py, download_pmc.py, download_wikipedia.py).

Offline: every test mocks the network and writes only into a temporary folder.
    python -m unittest -v test_a2_e
Target class when folded into test_tiny_gpt.py: TestOtherScripts (bioRxiv cases: TestRxiv).
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import download_arxiv as ax  # noqa: E402
import download_pmc as pmc  # noqa: E402
import download_rxiv as rx  # noqa: E402

PDF = b"%PDF-1.4 stub"
ROW = {"id": "2101.00001", "group": "biology", "year": "2021", "categories": "q-bio.PE", "licence": "",
       "title": "Drift", "abstract": "An abstract."}
TEX = "\\documentclass{article}\n\\begin{document}\n" + "Genetic drift changes allele frequencies. " * 40 + \
      "\n\\end{document}\n"


class Audit2Downloaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2e_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def convert(self, blob, **patches):
        """convert_paper with the network and the PDF child stubbed; returns the report row."""
        stubs = {"http_get": unittest.mock.Mock(return_value=PDF),
                 "pdf_markdown": unittest.mock.Mock(return_value="PDF text. " * 50)}
        stubs.update(patches)
        with unittest.mock.patch.multiple(ax, **stubs):
            return ax.convert_paper(dict(ROW), blob, self.tmp, 100, "pandoc")

    # ---------------- download_arxiv.py ----------------
    def test_f113_corrupt_gzip_eprint_takes_the_pdf_route(self):
        """F113: a gzip e-print with a damaged deflate stream (zlib.error) falls back to the PDF."""
        import zlib
        data = gzip.compress(TEX.encode() * 20, mtime=0)
        blob = data[:10] + bytes(b ^ 0xFF for b in data[10:60]) + data[60:]
        with self.assertRaises(zlib.error):  # the damage this test is about (not an OSError or EOFError)
            ax.tex_files(blob)
        result = self.convert(blob)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["method"].startswith("pdf (latex failed: error"), result["method"])
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, result["file"])))

    def test_f126_upper_case_tex_can_be_the_main_file(self):
        """F126: a file named MAIN.TEX is found as the main file."""
        self.assertEqual(ax.main_tex({"MAIN.TEX": r"\documentclass{article}\begin{document}x\end{document}"}),
                         "MAIN.TEX")


if __name__ == "__main__":
    unittest.main()
