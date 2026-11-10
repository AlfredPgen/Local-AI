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

    def arxiv_folders(self, rows, columns=None, sets=("q-bio",)):
        """(out, meta) for download_arxiv.main: a list.tsv with `rows` and a finished harvest of `sets`."""
        out, meta = os.path.join(self.tmp, "out"), os.path.join(self.tmp, "meta")
        os.makedirs(os.path.join(meta, "metadata"))
        for name in sets:
            with open(os.path.join(meta, "metadata", ax._set_file(name) + ".state.json"), "w", encoding="utf-8") as f:
                json.dump({"token": None, "pages": 1, "records": 5, "complete": True,
                           "harvested_until": "2026-01-01"}, f)
        with open(os.path.join(meta, "list.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(columns or ax.LIST_COLUMNS) + "\n")
            for row in rows:
                f.write("\t".join(row) + "\n")
        return out, meta

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

    def test_f120_failed_source_download_is_recorded_in_the_method(self):
        """F120: a paper converted from its PDF because the e-print download failed says so; a paper
        without source (404) is a plain 'pdf' as before."""
        failed = self.convert(ax.FETCH_FAILED)
        self.assertEqual((failed["status"], failed["method"]), ("ok", "pdf (source download failed)"))
        missing = self.convert(None)
        self.assertEqual((missing["status"], missing["method"]), ("ok", "pdf"))
        # main hands the marker over when the e-print request raises
        out, meta = self.arxiv_folders([[ROW["id"], "biology", "1.000", "2021", "q-bio.PE", "", "Drift",
                                         "An abstract."]])
        blobs = []

        def convert(row, blob, *rest):
            blobs.append(blob)
            return {"id": row["id"], "status": "ok", "chars": 1, "method": "pdf", "group": "", "year": "",
                    "categories": "", "licence": "", "file": ""}

        with unittest.mock.patch.multiple(ax, harvest=unittest.mock.Mock(return_value=0), pandoc_path=lambda: "pandoc",
                                          check_pdf_route=lambda: None, convert_paper=convert,
                                          http_get=unittest.mock.Mock(side_effect=OSError("no network"))), \
                redirect_stdout(io.StringIO()):
            ax.main(["--out", out, "--meta", meta, "--sets", "q-bio", "--wait", "0"])
        self.assertEqual(blobs, [ax.FETCH_FAILED])

    def test_f118_latex_failure_reason_is_recorded(self):
        """F118: the report's method says why LaTeX conversion failed (one line, bounded), not only the
        exception type; read_report does not take it for a PDF time-out."""
        boom = unittest.mock.Mock(side_effect=RuntimeError("Error at line 3:\tunknown\ncommand " + "x" * 200))
        result = self.convert(TEX.encode(), run_pandoc=boom)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["method"].startswith("pdf (latex failed: RuntimeError: Error at line 3: unknown command"),
                        result["method"])
        self.assertNotIn("\t", result["method"])
        self.assertNotIn("\n", result["method"])
        self.assertLessEqual(len(result["method"]), len("pdf (latex failed: RuntimeError: )") + 80)
        report = os.path.join(self.tmp, "report.tsv")
        with open(report, "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(ax.REPORT_COLUMNS) + "\n")
            f.write("\t".join(ax._flat(result[c]) for c in ax.REPORT_COLUMNS) + "\n")
        limits = {}
        self.assertEqual(ax.read_report(report, limits), {ROW["id"]: "ok"})
        self.assertEqual(limits, {})


if __name__ == "__main__":
    unittest.main()
