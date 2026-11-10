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

    def test_f124_pdf_route_drops_references_and_short_back_matter(self):
        """F124: Markdown from a PDF loses its reference list and short back-matter sections; a long
        section under a back-matter heading that is no reference list (PDF headings are often missing,
        so it may be body text) is kept whole."""
        intro, thanks, refs, appendix = "Drift changes allele frequencies.", "We thank the funders. " * 9, \
            "\n".join(f"[{i}] Wright S. 1931. Evolution in Mendelian populations. Genetics 16." for i in range(80)), \
            "Derivation of the variance."
        self.assertGreater(len(refs), 5000)
        md = (f"# Intro\n\n{intro}\n\n## **Acknowledgements**\n\n{thanks}\n\n# **References:**\n\n{refs}\n\n"
              f"## Appendix A\n\n{appendix}\n")
        cut = ax.drop_pdf_back_matter(md)
        self.assertEqual(cut, f"# Intro\n\n{intro}\n\n## Appendix A\n\n{appendix}\n")
        # a long section after a back-matter heading that is not "References": nothing is removed
        body = "The abstract and the results follow the title page here. " * 100
        long_md = f"# Title\n\n## **Correspondence**\n\n{body}\n\n# Discussion\n\nText.\n"
        self.assertGreater(len(body), ax.MAX_PDF_BACK_MATTER)
        self.assertEqual(ax.drop_pdf_back_matter(long_md), long_md)
        # the same heading over a short section is back matter
        self.assertEqual(ax.drop_pdf_back_matter("# Title\n\nText.\n\n## 7. Funding\n\nGrant 1.\n\n# Methods\n\nM.\n"),
                         "# Title\n\nText.\n\n# Methods\n\nM.\n")
        # a long reference list is dropped whatever its length, up to the next heading of any level
        self.assertEqual(ax.drop_pdf_back_matter(f"# A\n\na\n\n### References\n\n{refs}\n\n###### Table 3\n\nt\n"),
                         "# A\n\na\n\n###### Table 3\n\nt\n")
        plain = "# Intro\n\nText about references and funding.\n\n## Methods\n\nMore.\n\n**References**\n\n[1] x\n"
        self.assertEqual(ax.drop_pdf_back_matter(plain), plain)
        self.assertEqual(ax.drop_pdf_back_matter(""), "")

        def child(cmd, **kwargs):  # the PDF converter child: writes the Markdown file it is asked for
            with open(cmd[-1], "w", encoding="utf-8") as f:
                f.write(md.replace("\n", "\r\n") + "\r\nContact: someone@example.org\r\n")
            return unittest.mock.Mock(returncode=0, stderr=b"")

        with unittest.mock.patch.object(ax.subprocess, "run", side_effect=child):
            text = ax.pdf_markdown(PDF)
        self.assertEqual(text, f"# Intro\n\n{intro}\n\n## Appendix A\n\n{appendix}\n\nContact: [email]\n")

    LISTED = [["2101.00001", "biology", "1.500", "2021", "q-bio.PE", "", "Drift in small populations",
               "We model genetic drift."],
              ["2101.00002", "biology", "1.200", "2021", "q-bio.PE", "cc-by", "Allele: selection?", "An abstract, two."],
              ["2101.00003", "statistics", "0.900", "2022", "stat.ME", "", "Variance of estimators", ""]]

    def run_arxiv(self, out, meta, *options, loaded=None):
        """download_arxiv.main on a prepared folder: no network, LaTeX conversion replaced by a stand-in
        that builds the Markdown from the row's title and abstract. Returns the console text."""
        real_read = ax.read_list

        def read_list(*args, **kwargs):
            rows = real_read(*args, **kwargs)
            if loaded is not None:
                loaded.extend(rows)
            return rows

        def latex(blob, title, abstract, pandoc):
            return ax.clean_markdown("Body text. " * 400, title, abstract), "latex"

        with unittest.mock.patch.multiple(
                ax, harvest=unittest.mock.Mock(return_value=5), pandoc_path=lambda: "pandoc",
                check_pdf_route=lambda: None, http_get=unittest.mock.Mock(return_value=b"\\documentclass{article}"),
                latex_markdown=latex, read_list=read_list), redirect_stdout(io.StringIO()) as buf:
            ax.main(["--out", out, "--meta", meta, "--sets", "q-bio", "--wait", "0", *options])
        return buf.getvalue()

    def test_f125_title_and_abstract_are_loaded_for_the_papers_to_fetch_only(self):
        """F125: main keeps the title and abstract of the papers it is about to fetch, not of the whole
        list; their Markdown is unchanged."""
        out, meta = self.arxiv_folders(self.LISTED)

        def one_done():
            with open(os.path.join(meta, "report.tsv"), "w", encoding="utf-8", newline="\n") as f:
                f.write("\t".join(ax.REPORT_COLUMNS) + "\n")
                f.write("\t".join(["2101.00001", "ok", "9", "latex", "biology", "2021", "q-bio.PE", "", "x.md"]) + "\n")

        one_done()
        loaded = []
        self.run_arxiv(out, meta, loaded=loaded)
        body = ("Body text. " * 400).strip()
        expected = {"2101.00002__Allele selection.md": f"# Allele: selection?\n\n## Abstract\n\nAn abstract, two.\n\n{body}\n",
                    "2101.00003__Variance of estimators.md": f"# Variance of estimators\n\n{body}\n"}
        self.assertEqual(sorted(os.listdir(out)), sorted(expected))
        for name, text in expected.items():
            with open(os.path.join(out, name), encoding="utf-8", newline="") as f:
                self.assertEqual(f.read(), text, name)
        rows = {row["id"]: row for row in loaded}
        self.assertEqual(sorted(rows), [r[0] for r in self.LISTED])
        self.assertEqual(rows["2101.00001"], {"id": "2101.00001", "group": "biology", "score": "1.500", "year": "2021",
                                              "categories": "q-bio.PE", "licence": ""})
        self.assertEqual((rows["2101.00002"]["title"], rows["2101.00002"]["abstract"]),
                         ("Allele: selection?", "An abstract, two."))
        self.assertEqual((rows["2101.00003"]["title"], rows["2101.00003"]["abstract"]), ("Variance of estimators", ""))
        # the default read_list still gives whole rows (download_rxiv reads its list with it)
        whole = ax.read_list(os.path.join(meta, "list.tsv"))
        self.assertEqual([[row[c] for c in ax.LIST_COLUMNS] for row in whole], self.LISTED)
        # --max-papers counts the papers that are done, and only the rest get their text
        shutil.rmtree(out)
        one_done()
        loaded = []
        self.run_arxiv(out, meta, "--max-papers", "2", loaded=loaded)
        self.assertEqual(os.listdir(out), ["2101.00002__Allele selection.md"])
        self.assertEqual(["abstract" in row for row in loaded], [False, True, False])

    # ---------------- download_pmc.py ----------------
    QUERY = "OPEN_ACCESS:y AND TEST:y"

    @staticmethod
    def epmc_page(ids, cursor, total=0, cited=None):
        """One Europe PMC search answer (resultType core) for the PMC ids `ids`."""
        results = [{"pmcid": f"PMC{i}", "pmid": str(i), "pubYear": "2020", "citedByCount": (cited or {}).get(i, i),
                    "journalTitle": "J Test", "language": "eng", "title": f"Paper {i}",
                    "abstractText": f"Abstract of paper {i}."} for i in ids]
        return json.dumps({"hitCount": total or len(ids), "nextCursorMark": cursor,
                           "resultList": {"result": results}}).encode()

    @staticmethod
    def pmc_member(ids):
        """The gzip member harvest() writes for one page."""
        lines = [json.dumps({"pmcid": f"PMC{i}", "pmid": str(i), "year": "2020", "cited": i, "journal": "J Test",
                             "lang": "eng", "title": f"Paper {i}", "abstract": f"Abstract of paper {i}."})
                 for i in ids]
        return gzip.compress(("\n".join(lines) + "\n").encode("utf-8"))

    def pmc_cache(self, members, state=None):
        """A metadata folder with records.jsonl.gz made of `members` (bytes) and, if given, state.json."""
        folder = os.path.join(self.tmp, "metadata")
        os.makedirs(folder, exist_ok=True)
        records = os.path.join(folder, "records.jsonl.gz")
        with open(records, "wb") as f:
            for member in members:
                f.write(member)
        if state is not None:
            with open(os.path.join(folder, "state.json"), "w", encoding="utf-8") as f:
                json.dump(state, f)
        return folder, records

    @staticmethod
    def pmc_ids(records):
        return [int(r["pmcid"][3:]) for r in pmc.read_records(records)]

    def test_f140_pmc_cache_survives_a_cut_gzip_member(self):
        """F140: a page cut by a crash is cut off before the harvest goes on (state 'bytes'); a damaged
        cache is read up to the damage with a warning; a state from before 'bytes' truncates nothing."""
        first, second, third = self.pmc_member(range(1, 4)), self.pmc_member(range(4, 300)), [300, 301]
        # (a) member 1, a cut member 2, and the size after member 1 in the state: the resumed harvest
        # removes the cut page before appending page 3
        state = {"query": self.QUERY, "cursor": "c1", "records": 3, "complete": False, "bytes": len(first)}
        folder, records = self.pmc_cache([first, second[:40]], state)
        with unittest.mock.patch.object(pmc, "http_get", return_value=self.epmc_page(third, "c1")), \
                redirect_stdout(io.StringIO()):
            pmc.harvest(self.QUERY, folder)
        self.assertEqual(self.pmc_ids(records), [1, 2, 3, 300, 301])
        with open(os.path.join(folder, "state.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["bytes"], os.path.getsize(records))
        # (b) a cut tail and no harvest (--metadata-as-is): the records before the damage, and a warning
        folder, records = self.pmc_cache([first, second[:40]])
        with redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(self.pmc_ids(records), [1, 2, 3])
        self.assertIn("damaged", buf.getvalue())
        with self.assertRaises(FileNotFoundError):  # a missing file is not "damaged"
            self.pmc_ids(records + ".absent")
        # a line that is JSON but no record is skipped
        folder, records = self.pmc_cache([gzip.compress(b'{"x": 1}\n17\n'), first])
        self.assertEqual(self.pmc_ids(records), [1, 2, 3])
        # (c) a state written before 'bytes' existed: nothing is cut off
        state = {"query": self.QUERY, "cursor": "c1", "records": 299, "complete": False}
        folder, records = self.pmc_cache([first, second], state)
        with unittest.mock.patch.object(pmc, "http_get", return_value=self.epmc_page(third, "c1")), \
                redirect_stdout(io.StringIO()):
            pmc.harvest(self.QUERY, folder)
        self.assertEqual(self.pmc_ids(records), list(range(1, 302)))

    def epmc_server(self, pages, asked, fail_at=None):
        """A stand-in for download_pmc.http_get: `pages` maps a cursor to (ids, next cursor); the cursors
        asked for are appended to `asked`; request number fail_at (from 1) raises OSError."""
        import urllib.parse

        def get(url, *args, **kwargs):
            cursor = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["cursorMark"][0]
            asked.append(cursor)
            if fail_at is not None and len(asked) == fail_at:
                raise OSError("network down")
            ids, nxt = pages[cursor]
            return self.epmc_page(ids, nxt, total=sum(len(p[0]) for p in pages.values()))
        return get

    def test_f139_refresh_metadata_keeps_the_old_harvest_until_the_new_one_is_complete(self):
        """F139 (part A): --refresh-metadata harvests into records.jsonl.gz.new / state.json.new; the old
        harvest stays readable until the new one is complete, a rerun continues from the saved position,
        and a swap stopped half-way is finished at the next start."""
        folder = os.path.join(self.tmp, "metadata")
        records, state_path = os.path.join(folder, "records.jsonl.gz"), os.path.join(folder, "state.json")
        old = {"*": (range(1, 501), "c1"), "c1": (range(501, 1001), "c1")}
        new = {"*": (range(1, 501), "n1"), "n1": (range(501, 1002), "n1")}

        def state(path=state_path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        with unittest.mock.patch.object(pmc, "http_get", side_effect=self.epmc_server(old, [])), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(pmc.harvest(self.QUERY, folder), records)
        self.assertEqual(self.pmc_ids(records), list(range(1, 1001)))
        self.assertTrue(state()["complete"])
        self.assertRegex(state()["started"], r"^\d{4}-\d{2}-\d{2}$")
        # a refresh interrupted after one page: the old harvest is untouched
        asked = []
        with unittest.mock.patch.object(pmc, "http_get", side_effect=self.epmc_server(new, asked, fail_at=2)), \
                redirect_stdout(io.StringIO()), self.assertRaises(OSError):
            pmc.harvest(self.QUERY, folder, refresh=True)
        self.assertEqual(asked, ["*", "n1"])
        self.assertEqual(self.pmc_ids(records), list(range(1, 1001)))
        self.assertTrue(state()["complete"])
        self.assertEqual((state(state_path + ".new")["cursor"], state(state_path + ".new")["complete"]), ("n1", False))
        self.assertEqual(self.pmc_ids(records + ".new"), list(range(1, 501)))
        # without the flag the old harvest is used (no request) and the unfinished refresh is mentioned
        with unittest.mock.patch.object(pmc, "http_get", side_effect=AssertionError("no request expected")), \
                redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(pmc.harvest(self.QUERY, folder), records)
        self.assertIn("unfinished", buf.getvalue())
        self.assertIn("--refresh-metadata", buf.getvalue())
        # --metadata-as-is during a refresh selects from the old file
        seen = []
        with unittest.mock.patch.object(pmc, "keyword_list", side_effect=lambda path, *a, **k: seen.append(path) or []), \
                unittest.mock.patch.object(pmc, "http_get", side_effect=AssertionError("no request expected")), \
                redirect_stdout(io.StringIO()):
            pmc.main(["--out", os.path.join(self.tmp, "out"), "--meta", self.tmp, "--keywords", "unused.txt",
                      "--metadata-as-is", "--list-only"])
        self.assertEqual(seen, [records])
        self.assertEqual(self.pmc_ids(records), list(range(1, 1001)))
        # a rerun with the flag continues from the saved position, then the new harvest replaces the old
        asked = []
        with unittest.mock.patch.object(pmc, "http_get", side_effect=self.epmc_server(new, asked)), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(pmc.harvest(self.QUERY, folder, refresh=True), records)
        self.assertEqual(asked, ["n1"])
        self.assertEqual(sorted(os.listdir(folder)), ["records.jsonl.gz", "state.json"])
        self.assertEqual(self.pmc_ids(records), list(range(1, 1002)))
        self.assertEqual((state()["complete"], state()["records"]), (True, 1001))
        # the swap stopped after the records were renamed: the next start (with or without the flag) finishes it
        done = dict(state(), records=7)
        with open(state_path + ".new", "w", encoding="utf-8") as f:
            json.dump(done, f)
        with unittest.mock.patch.object(pmc, "http_get", side_effect=AssertionError("no request expected")), \
                redirect_stdout(io.StringIO()):
            pmc.harvest(self.QUERY, folder, refresh=True)
        self.assertEqual(sorted(os.listdir(folder)), ["records.jsonl.gz", "state.json"])
        self.assertEqual(state()["records"], 7)
        self.assertEqual(self.pmc_ids(records), list(range(1, 1002)))

    def test_f141_pmc_http_get_honours_retry_after(self):
        """F141: on 429/503 the PMC http_get waits as long as the server's Retry-After asks (at least
        its own backoff); without the header, and for other errors, the backoff is unchanged."""
        import email.message
        import urllib.error

        class Response:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, n=-1):
                return b"body"

        def answers(code, retry_after):
            headers = email.message.Message()
            if retry_after is not None:
                headers["Retry-After"] = retry_after
            return [urllib.error.HTTPError("https://example.org/x", code, "slow down", headers, None), Response()]

        def longest_sleep(code, retry_after):
            with unittest.mock.patch.object(pmc.urllib.request, "urlopen", side_effect=answers(code, retry_after)), \
                    unittest.mock.patch.object(pmc.time, "sleep") as sleep:
                self.assertEqual(pmc.http_get("https://example.org/x"), b"body")
            return max(call.args[0] for call in sleep.call_args_list)

        self.assertGreaterEqual(longest_sleep(429, "30"), 30)
        self.assertGreaterEqual(longest_sleep(503, "30"), 30)
        self.assertLess(longest_sleep(429, "1"), 4.01)  # shorter than the backoff: the backoff (3-4 s) stands
        self.assertLess(longest_sleep(429, None), 4.01)
        self.assertLess(longest_sleep(500, "30"), 4.01)  # Retry-After is honoured for 429 and 503 only
        self.assertLessEqual(longest_sleep(429, "999999"), 3600)

    KEYWORDS = "genetic*\nallele*\ndrift\nselection\n"
    MATCH = ("Genetic drift and allele frequencies", "Genetic drift changes allele frequencies; selection acts too.")

    def keyword_records(self, extra=()):
        """(records path, keyword file): matching and non-matching papers in several languages."""
        import data_prep
        papers = [(1, "eng", 10) + self.MATCH, (2, "eng", 99, "Cooking pasta", "Boil the water, add salt."),
                  (3, "", 30) + self.MATCH, (4, "fre", 40) + self.MATCH, (5, "English", 50) + self.MATCH,
                  (6, "ENG", 5, "Allele selection", "Genetic drift. " * 3 + "Padding words here. " * 40),
                  (7, "en", 70, "Drift", "Allele genetic selection drift."), (8, "ger", 80) + self.MATCH]
        papers += list(extra)
        lines = [json.dumps({"pmcid": f"PMC{i}", "pmid": str(i), "year": "2020", "cited": cited, "journal": "J Test",
                             "lang": lang, "title": title, "abstract": abstract})
                 for i, lang, cited, title, abstract in papers]
        folder, records = self.pmc_cache([gzip.compress(("\n".join(lines) + "\n").encode("utf-8"))])
        keywords = os.path.join(self.tmp, "kw.txt")
        with open(keywords, "w", encoding="utf-8") as f:
            f.write(self.KEYWORDS)
        return records, keywords, data_prep

    @staticmethod
    def reference_rows(dp, records, spec, min_hits, min_distinct, limit, rank, skip=()):
        """What keyword_list must return: the full check (KeywordFilter.evaluate) on every English paper."""
        import math
        kw = dp.KeywordFilter(dp.read_terms(spec), [], min_hits, min_distinct, title_weight=3)
        kept = []
        for rec in pmc.read_records(records):
            if rec["lang"] and not rec["lang"].lower().startswith("en"):
                continue
            keep, _, hits, _ = kw.evaluate(rec["title"], rec["abstract"])
            if keep:
                words = len(rec["title"].split()) + len(rec["abstract"].split())
                kept.append((hits / math.sqrt(max(words, 50)), rec))
        order = (lambda p: (-p[1]["cited"], -p[0], p[1]["pmcid"])) if rank == "cited" else \
            (lambda p: (-p[0], -p[1]["cited"], p[1]["pmcid"]))
        kept = [r for _, r in sorted(kept, key=order) if r["pmcid"] not in skip]
        return [(r["pmcid"], r["pmid"], r["year"], str(r["cited"]), r["journal"], r["title"]) for r in kept[:limit]]

    def test_f143_pmc_keyword_list_without_the_prefilter(self):
        """F143: keyword_list decides with the full check alone (no C++ prefilter pass) and gives the
        same rows; papers in another language are skipped, papers without a language are kept."""
        records, keywords, dp = self.keyword_records()
        path = os.path.join(self.tmp, "list.tsv")
        for rank in ("density", "cited"):
            with unittest.mock.patch.object(dp, "_vector_scores", side_effect=AssertionError("prefilter called")), \
                    redirect_stdout(io.StringIO()):
                rows = pmc.keyword_list(records, keywords, 3, 3, 100, rank, path)
            self.assertEqual(rows, self.reference_rows(dp, records, keywords, 3, 3, 100, rank))
            self.assertEqual(sorted(r[0] for r in rows), ["PMC1", "PMC3", "PMC5", "PMC6", "PMC7"])
            with open(path, encoding="utf-8") as f:
                self.assertEqual([tuple(line.split("\t")) for line in f.read().splitlines()[3:]], rows)

    def test_f144_pmc_scored_matches_are_reused(self):
        """F144: a --keywords run scores the metadata once (metadata/matches.tsv) and later runs select from
        that file; changed keywords, thresholds, rank or metadata, and --rebuild-list, score again; the rows
        are those of a fresh selection, also with --new and its skip set and with a comma-separated spec."""
        untitled = [(9, "eng", 60, "", "Genetic drift, allele selection and more genetic drift.")]
        records, keywords, dp = self.keyword_records(untitled)
        with open(os.path.join(self.tmp, "metadata", "state.json"), "w", encoding="utf-8") as f:
            json.dump({"query": pmc.HARVEST_QUERY, "cursor": "c", "records": 9, "complete": True}, f)
        out, listing = os.path.join(self.tmp, "out"), os.path.join(self.tmp, "list.tsv")
        real = dp.KeywordFilter.evaluate

        def run(*options, spec=keywords):
            """(KeywordFilter.evaluate calls, list rows, console text) of one --list-only run."""
            calls = []

            def evaluate(kw, title, text):
                calls.append(title)
                return real(kw, title, text)

            with unittest.mock.patch.object(dp.KeywordFilter, "evaluate", evaluate), \
                    unittest.mock.patch.object(pmc, "http_get", side_effect=AssertionError("no request expected")), \
                    redirect_stdout(io.StringIO()) as buf:
                pmc.main(["--out", out, "--meta", self.tmp, "--keywords", spec, "--list-only", *options])
            with open(listing, encoding="utf-8", newline="\n") as f:
                lines = f.read().split("\n")
            self.assertTrue(lines[0].startswith("# keywords: ") and lines[1].startswith("# metadata: ")
                            and lines[2].startswith("# complete: "), lines[:3])
            return len(calls), [tuple(line.split("\t")) for line in lines[3:] if line], buf.getvalue()

        def fresh(spec=keywords, min_hits=3, limit=100_000, rank="density", skip=()):
            return self.reference_rows(dp, records, spec, min_hits, 3, limit, rank, skip)

        calls, rows, text = run()
        self.assertEqual((calls, rows), (7, fresh()))  # 7: the nine papers without the two in another language
        self.assertIn(("PMC9", "9", "2020", "60", "J Test", ""), rows)  # an empty title keeps its field
        self.assertNotIn("--rebuild-list", text)
        calls, again, text = run()
        self.assertEqual((calls, again), (0, rows))
        self.assertIn("(--rebuild-list scores again)", text)
        self.assertEqual(run("--rebuild-list")[:2], (7, rows))
        self.assertEqual(run()[0], 0)
        # --new N: the best N that are not downloaded yet, from the cache; equal to a fresh selection
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"{rows[0][0]}__x.md"), "w", encoding="utf-8") as f:
            f.write("x")
        with open(os.path.join(self.tmp, "report.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(pmc.REPORT_COLUMNS) + "\n" + "\t".join([rows[2][0], "no_body", "0", "", "", "", "", ""])
                    + "\n" + "\t".join([rows[3][0], "error: x", "0", "", "", "", "", ""]) + "\n")
        skip = {rows[0][0], rows[2][0]}
        self.assertEqual(run("--new", "2")[:2], (0, fresh(limit=2, skip=skip)))
        self.assertEqual(run("--new", "2")[1], [rows[1], rows[3]])
        self.assertEqual(run("--max-papers", "3")[:2], (0, rows[:3]))
        # another rank, threshold, keyword list or metadata: scored again, then reused
        self.assertEqual(run("--rank", "cited")[:2], (7, fresh(rank="cited")))
        self.assertEqual(run("--rank", "cited", "--new", "2")[:2], (0, fresh(rank="cited", limit=2, skip=skip)))
        self.assertEqual(run("--min-hits", "9")[:2], (7, fresh(min_hits=9)))
        self.assertEqual(run("--min-hits", "9")[0], 0)
        listed = "genetic*,allele*,drift"
        self.assertEqual(run(spec=listed)[:2], (7, fresh(spec=listed)))
        self.assertEqual(run(spec=listed)[0], 0)
        self.assertEqual(run()[0], 7)
        with open(keywords, "a", encoding="utf-8") as f:
            f.write("pasta\nwater\nsalt\n")
        calls, edited, _ = run()
        self.assertEqual((calls, edited), (7, fresh()))
        self.assertIn("PMC2", [r[0] for r in edited])
        with open(records, "ab") as f:  # the harvest went on: one more page
            f.write(gzip.compress((json.dumps(
                {"pmcid": "PMC500", "pmid": "500", "year": "2021", "cited": 3, "journal": "J Test", "lang": "eng",
                 "title": "Selection", "abstract": "Genetic drift and allele selection."}) + "\n").encode("utf-8")))
        calls, grown, _ = run()
        self.assertEqual((calls, grown), (8, fresh()))
        self.assertEqual(run()[:2], (0, grown))
        self.assertFalse([name for name in os.listdir(os.path.join(self.tmp, "metadata")) if name.endswith(".part")])


if __name__ == "__main__":
    unittest.main()
