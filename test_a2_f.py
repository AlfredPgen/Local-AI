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

    # ---- youtube_transcripts.py -------------------------------------------------------------------------
    def test_youtube_video_spec_is_not_remembered_as_its_channel(self):
        """F445: a video, playlist or /streams URL is listed as itself on the next run too; only a plain channel
        (@handle, name, id, channel URL) is listed through its remembered channel address."""
        import json
        import data_prep
        import youtube_transcripts as yt
        channel_id = "UC" + "x" * 22
        asked = []

        class FakeYdl:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=False):
                asked.append(url)
                n = 1 if "watch?v=" in url else 2 if "list=" in url else 250
                videos = [{"id": f"v{k:010d}", "title": f"video {k}", "duration": 600} for k in range(n)]
                info = {"channel_id": channel_id, "channel": "Some Channel", "uploader_id": "@somechannel"}
                return {**videos[0], **info} if n == 1 else {**info, "id": "listing", "entries": videos}

        out = os.path.join(self.tmp, "f445", "lectures")
        meta = os.path.join(self.tmp, "f445", "_lectures_meta", "channels.json")
        specs = ("https://www.youtube.com/watch?v=abcdefghijk", "https://www.youtube.com/playlist?list=PL123",
                 "https://www.youtube.com/@somechannel/streams", "@somechannel",
                 "https://www.youtube.com/@somechannel/videos")
        # main runs in the test process here: its priority call is replaced, the test run keeps its own priority
        with unittest.mock.patch.object(yt, "ydl", lambda *a, **k: FakeYdl()), \
                unittest.mock.patch.object(data_prep, "_lower_priority") as lowered, \
                unittest.mock.patch("sys.stdout", new_callable=__import__("io").StringIO):
            for spec in specs:
                del asked[:]
                for _ in range(2):
                    yt.main([spec, "--list-only", "--keywords", "none", "--out", out])
                self.assertEqual(len(asked), 2, spec)
                if spec.startswith("http") and not spec.endswith("/videos"):
                    self.assertEqual(asked[1], asked[0], spec)
                else:  # a plain channel: its remembered address, so no handle guessing or name search again
                    self.assertEqual(asked[1], f"https://www.youtube.com/channel/{channel_id}/videos", spec)
        with open(meta, encoding="utf-8") as handle:
            memory = json.load(handle)
        for spec in specs:
            self.assertEqual(memory[spec]["name"], "Some Channel", spec)  # the name is kept for every kind
        self.assertEqual(lowered.call_count, 2 * len(specs))
        self.assertNotIn("url", memory[specs[0]])
        self.assertNotIn("url", memory[specs[1]])
        self.assertNotIn("url", memory[specs[2]])
        self.assertEqual(memory["@somechannel"]["url"], f"https://www.youtube.com/channel/{channel_id}")
        # an entry an earlier version wrote for a video URL (its whole channel) is harmless now
        with unittest.mock.patch.object(yt, "ydl", lambda *a, **k: FakeYdl()), \
                unittest.mock.patch.object(data_prep, "_lower_priority"), \
                unittest.mock.patch("sys.stdout", new_callable=__import__("io").StringIO):
            memory[specs[0]]["url"] = f"https://www.youtube.com/channel/{channel_id}"
            with open(meta, "w", encoding="utf-8") as handle:
                json.dump(memory, handle)
            del asked[:]
            yt.main([specs[0], "--list-only", "--keywords", "none", "--out", out])
        self.assertEqual(asked, [specs[0]])

    def test_youtube_gpu_busy_sees_train_lock(self):
        """F442: a live training lock beside the script means the GPU is busy, whatever nvidia-smi shows."""
        import json
        import subprocess
        import youtube_transcripts as yt
        here = os.path.join(self.tmp, "f442")
        os.makedirs(here)
        quiet = subprocess.CompletedProcess([], 0, stdout="5, 100\n", stderr="")  # 5 % used, 100 MiB held

        def no_nvidia(*a, **k):
            raise OSError("no nvidia-smi")

        path = list(sys.path)
        try:
            with unittest.mock.patch.object(yt, "HERE", here), unittest.mock.patch.object(yt.time, "sleep"):
                with unittest.mock.patch.object(yt.subprocess, "run", return_value=quiet):
                    self.assertIs(yt.gpu_busy(), False)  # no lock, an idle GPU
                with unittest.mock.patch.object(yt.subprocess, "run", no_nvidia):
                    self.assertIsNone(yt.gpu_busy())  # no lock, no NVIDIA GPU
                with open(os.path.join(here, "other.train.lock"), "w", encoding="utf-8") as handle:
                    json.dump({"pid": os.getppid()}, handle)
                with unittest.mock.patch.object(yt.subprocess, "run", return_value=quiet):
                    self.assertIs(yt.gpu_busy(), True)
                with unittest.mock.patch.object(yt.subprocess, "run", no_nvidia):
                    self.assertIs(yt.gpu_busy(), True)
                with open(os.path.join(here, "other.train.lock"), "w", encoding="utf-8") as handle:
                    json.dump({"pid": os.getpid()}, handle)  # a lock this process left: not another job
                with unittest.mock.patch.object(yt.subprocess, "run", return_value=quiet):
                    self.assertIs(yt.gpu_busy(), False)
        finally:
            sys.path[:] = path

    def test_youtube_priority_keeps_idle(self):
        """F448: a run started at idle priority is not raised to below-normal (data_prep's rule), and the file-name
        helper is download_pmc's."""
        import psutil
        import download_pmc
        import youtube_transcripts as yt

        class FakeYdl:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=False):
                return {"channel_id": "UC" + "y" * 22, "channel": "C", "id": "listing",
                        "entries": [{"id": "v0000000001", "duration": 60}]}

        proc = unittest.mock.Mock()
        proc.nice.return_value = psutil.IDLE_PRIORITY_CLASS if sys.platform == "win32" else 19
        with unittest.mock.patch.object(psutil, "Process", return_value=proc), \
                unittest.mock.patch.object(yt, "ydl", lambda *a, **k: FakeYdl()), \
                unittest.mock.patch("sys.stdout", new_callable=__import__("io").StringIO):
            yt.main(["@c", "--list-only", "--keywords", "none", "--out", os.path.join(self.tmp, "f448", "lectures")])
        self.assertTrue(proc.nice.called)  # the priority was read ...
        self.assertEqual([c for c in proc.nice.call_args_list if c.args or c.kwargs], [])  # ... and never set
        self.assertIs(yt.safe_name, download_pmc.safe_name)

    def test_list_videos_keeps_nested_order(self):
        """F443 (guard): nested playlists are flattened depth first, in the order YouTube lists them."""
        import youtube_transcripts as yt

        def video(letter):
            return {"id": letter * 11, "title": letter}

        info = {"id": "channel", "channel": "C", "channel_id": "UC" + "z" * 22,
                "entries": iter([video("a"),
                                 {"id": "tab", "entries": iter([video("b"),
                                                                {"id": "inner", "entries": iter([video("c")])},
                                                                video("d")])},
                                 video("e"), video("a")])}  # generators, as yt-dlp returns them; a repeat is dropped

        class FakeYdl:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=False):
                return info

        with unittest.mock.patch.object(yt, "ydl", lambda *a, **k: FakeYdl()):
            channel, entries = yt.list_videos("@c", None, 0)
        self.assertEqual([e["title"] for e in entries], ["a", "b", "c", "d", "e"])
        self.assertEqual(channel["name"], "C")

    # ---- convert_to_markdown.py -------------------------------------------------------------------------
    def test_gwas_readers_agree_and_take_the_median_first(self):
        """F027: the fast and the slow GWAS reader give the same counts, hits and lambda GC, and chi2.isf is
        called on the one or two middle values only, not on every variant."""
        import numpy as np
        import scipy.stats
        import convert_to_markdown as conv
        header = ["SNP", "CHR", "BP", "A1", "BETA", "P"]
        cols = conv.gwas_columns(header)
        folder = os.path.join(self.tmp, "f027")
        os.makedirs(folder)
        real_isf, sizes = scipy.stats.chi2.isf, []

        def isf(q, *a, **k):
            sizes.append(np.size(q))
            return real_isf(q, *a, **k)

        for count in (40, 41):  # an even and an odd number of valid rows
            rows, valid = [], []
            for k in range(count):
                p = (k + 0.5) / count
                rows.append([f"rs{k}", ("chr" if k % 3 == 0 else "") + str(1 + k % 5), 1000 + 7 * k, "A",
                             f"{0.01 * k:.2f}", f"{p:.6g}"])
                valid.append(float(f"{p:.6g}"))
            rows[3][1], rows[4][1] = "chrX", ""  # a named and an empty chromosome
            rows[5][5], valid[5] = "3e-9", 3e-9
            rows[6][5], valid[6] = "1e-320", 1e-300  # capped at 300 for lambda
            rows[7][5], valid[7] = "0", 1e-300
            rows += [["rsNA", "2", 5, "C", "0.5", "NA"], ["rsNoPos", "2", "", "C", "0.5", "0.5"]]  # both left out
            path = os.path.join(folder, f"gwas{count}.tsv")
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join("\t".join(map(str, r)) for r in [header] + rows) + "\n")
            del sizes[:]
            with unittest.mock.patch.object(scipy.stats.chi2, "isf", isf):
                fast = conv._gwas_fast(path, header, "\t", cols, False)
                slow = conv._gwas_slow(path, "\t", cols, False)
            self.assertEqual(len(sizes), 2, sizes)
            self.assertLessEqual(max(sizes), 2, sizes)
            want = float(np.median(real_isf(np.array(valid), 1)) / 0.4549)
            for name, (n, top, per_chr, lam) in (("fast", fast), ("slow", slow)):
                with self.subTest(reader=name, rows=count):
                    self.assertEqual(n, count)
                    self.assertAlmostEqual(lam, want, delta=1e-6)
                    self.assertEqual(per_chr["X"], 1)
                    self.assertEqual(per_chr[""], 1)
                    self.assertEqual(sum(per_chr.values()), count)
            self.assertEqual(fast[0], slow[0])
            self.assertEqual(dict(fast[2]), dict(slow[2]))
            self.assertEqual({type(c) for c in fast[2]}, {str})
            self.assertAlmostEqual(fast[3], slow[3], delta=1e-6)

            def plain(top):
                return sorted((round(nl, 9), chrom, pos, snp, float(beta), allele)
                              for nl, chrom, pos, snp, beta, allele in top)

            self.assertEqual(plain(fast[1]), plain(slow[1]))
            self.assertEqual([t[3] for t in sorted(fast[1], key=lambda t: t[2])], ["rs5", "rs6", "rs7"])
            self.assertEqual({type(t[1]) for t in fast[1]} | {type(t[2]) for t in fast[1]}, {str, int})
        empty = os.path.join(folder, "none.tsv")  # no valid row: lambda is nan in both readers
        with open(empty, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\t".join(header) + "\nrs1\t1\t5\tA\t0.1\tNA\n")
        for n, top, per_chr, lam in (conv._gwas_fast(empty, header, "\t", cols, False),
                                     conv._gwas_slow(empty, "\t", cols, False)):
            self.assertEqual((n, top, per_chr), (0, [], {}))
            self.assertTrue(np.isnan(lam))

    def test_by_type_sorts_txt_and_gz_by_their_header(self):
        """F039: with --by-type summary statistics go to tables as a .txt too, and a prose .gz goes to articles;
        a tab-separated table that is not summary statistics stays where it went before (.gz: tables, .txt:
        articles); an input an earlier run recorded in the other folder stays there (it is not converted a
        second time)."""
        import gzip
        import json
        import subprocess
        src, out = os.path.join(self.tmp, "f039", "src"), os.path.join(self.tmp, "f039", "out")
        os.makedirs(src)
        rows = ["CHROM\tGENPOS\tID\tALLELE0\tALLELE1\tBETA\tSE\tLOG10P"]
        rows += [f"{1 + k % 3}\t{1000 + k}\trs{k}\tA\tG\t0.01\t0.02\t{0.3 + (k % 7) / 10:.1f}" for k in range(30)]
        for name in ("a.txt", "old.txt"):
            with open(os.path.join(src, name), "w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(rows) + "\n")
        prose = "Genetic drift changes allele frequencies by chance. It is stronger in small populations.\n" * 5
        with gzip.open(os.path.join(src, "prose.txt.gz"), "wt", encoding="utf-8") as handle:
            handle.write(prose)
        with gzip.open(os.path.join(src, "sumstats.tsv.gz"), "wt", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(rows) + "\n")
        with open(os.path.join(src, "notes.txt"), "w", encoding="utf-8") as handle:
            handle.write(prose)
        counts = ["gene\tsample1\tsample2\tsample3"] + [f"g{k}\t{k}\t{k + 1}\t{k + 2}" for k in range(40)]
        with gzip.open(os.path.join(src, "counts.txt.gz"), "wt", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(counts) + "\n")  # a table that is not GWAS: a .gz is filed as a table, as before
        with open(os.path.join(src, "matrix.txt"), "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(counts) + "\n")  # the same table as a .txt: with the articles, as before
        os.makedirs(os.path.join(out, "articles"))  # old.txt was converted before, into articles
        with open(os.path.join(out, "articles", "old.md"), "w", encoding="utf-8") as handle:
            handle.write("# GWAS summary statistics: old\n")
        with open(os.path.join(out, "conversion_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"outputs": {"articles/old.md": os.path.join(src, "old.txt")}}, handle)
        # converted before as well, but no record is left (a lost manifest, a run older than manifests): a .txt
        # was put in articles then and a .gz in tables, and each still counts as done where it is
        shutil.copy(os.path.join(src, "old.txt"), os.path.join(src, "lost.txt"))
        shutil.copy(os.path.join(src, "prose.txt.gz"), os.path.join(src, "lostprose.txt.gz"))
        os.makedirs(os.path.join(out, "tables"))
        for name in ("articles/lost.md", "tables/lostprose.md"):
            with open(os.path.join(out, *name.split("/")), "w", encoding="utf-8") as handle:
                handle.write("made by an earlier run\n")
        done = subprocess.run([sys.executable, os.path.join(HERE, "convert_to_markdown.py"), src, "--out", out,
                               "--by-type"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=600, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        found = sorted(os.path.relpath(os.path.join(folder, name), out).replace(os.sep, "/")
                       for folder, _, names in os.walk(out) for name in names if name.endswith(".md"))
        self.assertEqual(found, ["articles/lost.md", "articles/matrix.md", "articles/notes.md", "articles/old.md",
                                 "articles/prose.md", "tables/a.md", "tables/counts.md", "tables/lostprose.md",
                                 "tables/sumstats.md"], done.stdout + done.stderr)
        with open(os.path.join(out, "tables", "a.md"), encoding="utf-8") as handle:
            self.assertIn("30 genetic variants", handle.read())
        for name in ("tables/counts.md", "articles/matrix.md"):  # both converted as a table, each filed as before
            with open(os.path.join(out, *name.split("/")), encoding="utf-8") as handle:
                self.assertIn("| gene | sample1 |", handle.read(), name)
        # --no-gwas: summary statistics are a plain file again and go where they went before: a .txt to articles,
        # a tab-separated .gz to tables
        import convert_to_markdown as conv
        import argparse
        spaced = os.path.join(self.tmp, "f039", "spaced.txt")
        with open(spaced, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(row.replace("\t", " ") for row in rows) + "\n")
        args = argparse.Namespace(out=out, by_type=True, name_prefix="", book_pages=150, no_gwas=True)

        def folder(path):
            return os.path.basename(os.path.dirname(conv.plan_target(path, args)))

        others = [os.path.join(src, name)
                  for name in ("sumstats.tsv.gz", "counts.txt.gz", "prose.txt.gz", "matrix.txt")]
        self.assertEqual([folder(os.path.join(src, "a.txt")), folder(spaced)], ["articles", "articles"])
        self.assertEqual([folder(path) for path in others], ["tables", "tables", "articles", "articles"])
        self.assertEqual(conv.convert_table_or_text(spaced, args)[1], "text")
        args.no_gwas = False
        self.assertEqual([folder(os.path.join(src, "a.txt")), folder(spaced)], ["tables", "tables"])
        self.assertEqual([folder(path) for path in others], ["tables", "tables", "articles", "articles"])
        # the unrecorded earlier output is kept only while it is nobody else's and nothing is at the new place
        lost, files = os.path.join(src, "lost.txt"), sorted(os.path.join(src, name) for name in os.listdir(src))

        def planned(owners=None):
            return os.path.relpath(dict(conv.plan_targets(files, args, owners))[lost], out).replace(os.sep, "/")

        self.assertEqual(planned(), "articles/lost.md")
        earlier = os.path.join(out, "articles", "lost.md")
        self.assertEqual(planned({conv._norm(earlier): (earlier, os.path.join(src, "notes.txt"))}), "tables/lost.md")
        with open(os.path.join(out, "tables", "lost.md"), "w", encoding="utf-8") as handle:
            handle.write("made by this version\n")
        self.assertEqual(planned(), "tables/lost.md")

    def test_text_encoding_is_chosen_as_before(self):
        """F039 (guard, passes before and after): the first lines of every .txt and .gz are now read when a
        --by-type batch is planned, so the encoding test counts in C, not character by character; it still
        chooses Windows-1252 only when invalid UTF-8 outnumbers valid non-ASCII UTF-8."""
        import gzip
        import convert_to_markdown as conv
        folder = os.path.join(self.tmp, "f039enc")
        os.makedirs(folder)
        squared = "h² ".encode("utf-8")  # one valid non-ASCII character
        mark = "� ".encode("utf-8")  # a replacement character that is in the file itself: not an invalid byte
        cases = {
            "ascii": (b"plain text\n" * 5, "utf-8-sig"),
            "utf8": ("Heritability h² of αβγ traits\n".encode("utf-8") * 50, "utf-8-sig"),
            "cp1252": ("naïve café – résumé\n".encode("cp1252") * 50, "cp1252"),
            "tie": (squared * 3 + b"\x96 \x97 \x93\n", "utf-8-sig"),  # 3 invalid bytes, 3 valid characters
            "more_invalid": (squared * 3 + b"\x96 \x97 \x93 \x94\n", "cp1252"),  # 4 invalid, 3 valid
            "own_marks": (mark * 3 + b"\x96\n", "cp1252"),  # 1 invalid, 0 valid: the 3 marks count as neither
            "own_marks_tie": (mark * 3 + squared + b"\x96\n", "utf-8-sig"),  # 1 invalid, 1 valid
            "cut": (b"a" * ((1 << 20) - 1) + "éé\n".encode("utf-8"), "utf-8-sig"),  # the MB ends inside a character
            "long": ((squared * 40 + b"\x96" * 41 + b"\n") * 3000, "cp1252"),  # 123,000 invalid, 120,000 valid
            "long_tie": ((squared * 41 + b"\x96" * 41 + b"\n") * 3000, "utf-8-sig"),
        }
        for name, (raw, want) in cases.items():
            path = os.path.join(folder, name + ".txt")
            with open(path, "wb") as handle:
                handle.write(raw)
            with gzip.open(path + ".gz", "wb") as handle:
                handle.write(raw)
            self.assertEqual([conv.text_encoding(path), conv.text_encoding(path + ".gz")], [want, want], name)

    def test_pdf_page_count_is_one_cached_function(self):
        """F038: one page counter (0 for a file that cannot be opened), each PDF opened once; --by-type sorts a
        PDF as before."""
        import argparse
        import pymupdf
        import convert_to_markdown as conv
        folder = os.path.join(self.tmp, "f038")
        os.makedirs(folder)
        two, broken = os.path.join(folder, "two.pdf"), os.path.join(folder, "broken.pdf")
        pdf = pymupdf.open()
        for k in range(2):
            pdf.new_page().insert_text((72, 72), f"Page {k + 1} about heritability.")
        pdf.save(two)
        pdf.close()
        with open(broken, "wb") as handle:
            handle.write(b"%PDF-1.7 not a pdf at all")
        self.assertFalse(hasattr(conv, "pdf_pages") or hasattr(conv, "_pdf_page_count"))
        real_open, opened = pymupdf.open, []

        def counting_open(*a, **k):
            opened.append(a[0] if a else None)
            return real_open(*a, **k)

        with unittest.mock.patch.object(pymupdf, "open", counting_open):
            self.assertEqual([conv.page_count(two) for _ in range(3)], [2, 2, 2])
            self.assertEqual([conv.page_count(broken) for _ in range(3)], [0, 0, 0])
            args = argparse.Namespace(out=os.path.join(folder, "out"), by_type=True, name_prefix="", book_pages=2,
                                      no_gwas=False)
            kinds = [os.path.basename(os.path.dirname(conv.plan_target(p, args))) for p in (two, broken)]
            args.book_pages = 3
            kinds.append(os.path.basename(os.path.dirname(conv.plan_target(two, args))))
            # the time limit main gives each PDF's process: per page, and a PDF that cannot be opened counts as one
            limits = []

            def child(cmd, timeout):  # in place of the process: the file and the limit it would get
                limits.append((os.path.basename(cmd[2]), timeout))
                return 0, "skipped\tnot converted in this test\n", []

            page, ocr_page = conv.PDF_SECONDS_PER_PAGE, conv.OCR_SECONDS_PER_PAGE
            for extra, want in ((["--timeout", "1"], [page, 2 * page]), (["--timeout", "1", "--ocr"],
                                                                         [ocr_page, 2 * ocr_page]), ([], [3600, 3600])):
                del limits[:]
                argv = ["convert_to_markdown.py", folder, "--out", os.path.join(folder, "out"), "--by-type"] + extra
                with unittest.mock.patch.object(sys, "argv", argv), \
                        unittest.mock.patch.object(conv, "_run_child", child), \
                        unittest.mock.patch.object(conv, "_lower_priority"), \
                        unittest.mock.patch("sys.stdout", new_callable=__import__("io").StringIO):
                    conv.main()
                self.assertEqual(limits, list(zip(["broken.pdf", "two.pdf"], want)), extra)
        self.assertEqual(kinds, ["books", "articles", "articles"])
        self.assertEqual(opened, [two, broken])  # once each, whatever asked: the folder, the names, the time limit

    def test_output_names_are_made_lazily_in_the_same_order(self):
        """F028: the candidate output names of an input come one at a time (the first is nearly always enough), in
        the order and with the last-resort name they always had."""
        import argparse
        import inspect
        import re
        import convert_to_markdown as conv
        root = os.path.join(self.tmp, "f028")
        out = os.path.join(root, "out")
        files = []
        for k in range(30):  # the same name in 30 folders that are all called 'sub': numbered names are needed
            folder = os.path.join(root, f"d{k:02d}", "sub")
            os.makedirs(folder)
            files.append(os.path.join(folder, "x.txt" if k else "x.md"))
            with open(files[-1], "w", encoding="utf-8") as handle:
                handle.write("Genetic drift.\n")
        args = argparse.Namespace(out=out, by_type=False, name_prefix="", book_pages=150, no_gwas=False)

        def reference(path):  # the list this function built before the change
            target = conv.plan_target(path, args)
            stem = target[:-3]
            ext = os.path.basename(path).lower().replace(".gz", "").rsplit(".", 1)[-1]
            parent = re.sub(r"[^\w.-]+", "_", os.path.basename(os.path.dirname(os.path.abspath(path)))) or "folder"
            head = [target] + ([] if path.endswith(".md") else [f"{stem}.{ext}.md"]) + [f"{stem}__{parent}.md"]
            return head + [f"{stem}__{k}.md" for k in range(2, 1000)]

        for path in files[:2]:
            self.assertEqual(list(conv._name_candidates(path, args)), reference(path))
        planned = conv.plan_targets(files, args)
        self.assertEqual([p for p, _ in planned], files)
        self.assertEqual([os.path.basename(t) for _, t in planned],
                         ["x.md", "x.txt.md", "x__sub.md"] + [f"x__{k}.md" for k in range(2, 29)])
        # a name recorded by an earlier run is kept when it is among the first 22 candidates, not beyond them
        names = reference(files[5])
        for index, kept in ((21, True), (22, False)):
            owners = {conv._norm(names[index]): (names[index], files[5])}
            target = dict(conv.plan_targets(files, args, owners))[files[5]]
            self.assertEqual(target == names[index], kept, (index, target))
        # every name taken: the last one, as before
        taken = {conv._norm(name): (name, os.path.join(root, "elsewhere", f"{k}.txt"))
                 for k, name in enumerate(reference(files[1]))}
        self.assertEqual(os.path.basename(dict(conv.plan_targets(files[1:2], args, taken))[files[1]]), "x__999.md")
        legacy = conv.legacy_outputs(files + [os.path.join(out, "x__sub.md"), os.path.join(out, "x__20.md"),
                                              os.path.join(out, "x__21.md")], args)
        self.assertEqual(sorted(os.path.basename(t) for t, _ in legacy.values()), ["x__20.md", "x__sub.md"])
        self.assertTrue(inspect.isgenerator(conv._name_candidates(files[0], args)))

    def test_pdf_layout_fast_reaches_the_child_process(self):
        """F031: --pdf-layout fast converts without pymupdf4llm's layout model, also in the child process every
        PDF runs in, and says so in the report; without the option nothing changes."""
        import argparse
        import subprocess
        import pymupdf
        import pymupdf4llm
        import convert_to_markdown as conv
        src = os.path.join(self.tmp, "f031", "src")
        os.makedirs(src)
        path = os.path.join(src, "paper.pdf")
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 80), "Genetic drift in small populations", fontsize=20)
        for k in range(8):
            page.insert_text((72, 120 + 16 * k), f"Sentence {k + 1}: allele frequencies change by chance alone.",
                             fontsize=11)
        pdf.save(path)
        pdf.close()
        with open(os.path.join(src, "notes.txt"), "w", encoding="utf-8") as handle:  # two inputs: a report is written
            handle.write("Selection changes allele frequencies in one direction.\n")
        before = pymupdf4llm._use_layout
        try:
            want = {"model": conv.convert_pdf(path, argparse.Namespace())}
            want["fast"] = conv.convert_pdf(path, argparse.Namespace(pdf_layout="fast"))
            self.assertFalse(pymupdf4llm._use_layout)
            # the switch is per process: a later conversion in it that asks for the model gets the model back
            self.assertEqual(conv.convert_pdf(path, argparse.Namespace()), want["model"])
            self.assertEqual(pymupdf4llm._use_layout, before)
        finally:
            pymupdf4llm.use_layout(before)  # whatever failed above, the other tests keep their mode
        for mode, extra in (("fast", ["--pdf-layout", "fast"]), ("model", [])):
            out = os.path.join(self.tmp, "f031", mode)
            done = subprocess.run([sys.executable, os.path.join(HERE, "convert_to_markdown.py"), src, "--out", out,
                                   "--workers", "2"] + extra, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=900, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            with open(os.path.join(out, "paper.md"), encoding="utf-8", newline="") as handle:
                self.assertEqual(handle.read(), want[mode].replace("\r\n", "\n").strip() + "\n", mode)
            with open(os.path.join(out, "conversion_report.tsv"), encoding="utf-8") as handle:
                report = handle.read()
            self.assertEqual(report.count("(fast layout)"), 1 if mode == "fast" else 0, report)  # the PDF's row only
        self.assertIn("Genetic drift in small populations", want["fast"])

    def test_pdf_layout_modes_share_the_heavy_page_and_ocr_paths(self):
        """F031 (guard): a plot-heavy page is read as plain text and a scanned page by --ocr in the same way with
        --pdf-layout model and fast; only the ordinary pages go through pymupdf4llm."""
        import argparse
        import contextlib
        import io
        import pymupdf
        import pymupdf4llm
        import convert_to_markdown as conv
        folder = os.path.join(self.tmp, "f031pages")
        os.makedirs(folder)
        path = os.path.join(folder, "mixed.pdf")
        pdf = pymupdf.open()
        page = pdf.new_page()  # 1: a plot of 600 shapes with a long caption
        for k in range(6):
            page.insert_text((72, 40 + 14 * k), f"Figure {k + 1}. Posterior draws of the heritability, sample {k + 1}.")
        for k in range(600):
            page.draw_rect(pymupdf.Rect(20 + (k % 40) * 13, 200 + (k // 40) * 13, 30 + (k % 40) * 13,
                                        210 + (k // 40) * 13), fill=(0.2, 0.4, 0.8))
        page = pdf.new_page()  # 2: an ordinary page of text
        for k in range(12):
            page.insert_text((72, 80 + 16 * k), f"Sentence {k + 1}: linkage disequilibrium decays with recombination.")
        page = pdf.new_page()  # 3: a scan (no text layer, one image over the whole page)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
        pixmap.clear_with(200)
        page.insert_image(page.rect, pixmap=pixmap)
        pdf.save(path)
        with pymupdf.open(path) as saved:
            self.assertEqual([conv._heavy_page(p) for p in saved], [True, False, False])
            caption = saved[0].get_text("text").strip()
        read = "Text of the scanned page as the OCR reads it. " * 8
        before, got = pymupdf4llm._use_layout, {}
        try:
            with unittest.mock.patch.object(conv, "find_tessdata", lambda: folder), \
                    unittest.mock.patch.object(conv, "_ocr_page", lambda page, tessdata, dpi=300: read.strip()):
                for mode in ("model", "fast"):
                    info, progress = {}, io.StringIO()
                    with contextlib.redirect_stderr(progress):
                        text = conv.convert_pdf(path, argparse.Namespace(pdf_layout=mode, ocr=True), info=info)
                    got[mode] = (text, info, progress.getvalue())
                    self.assertEqual(pymupdf4llm._use_layout, before and mode == "model", mode)
        finally:
            pymupdf4llm.use_layout(before)
        for mode, (text, info, progress) in got.items():
            self.assertEqual(info, {"pages": 3, "scan_pages": 1, "ocr_pages": 1}, mode)
            self.assertTrue(text.startswith(caption + "\n\n"), (mode, text[:200]))
            self.assertTrue(text.endswith("\n\n" + read.strip()), (mode, text[-200:]))
            self.assertIn("Sentence 12: linkage disequilibrium decays with recombination.", text, mode)
            self.assertEqual(progress, "  OCR mixed.pdf: 1 of 1 scanned pages\n", mode)

    def test_converter_child_crash_and_time_limit_cost_one_file(self):
        """F032 (guard): a file whose process ends with an error is reported 'skipped', a process past its time
        limit is stopped and reported 'timed out', and the batch goes on to the next file either way."""
        import time
        import convert_to_markdown as conv
        src, out = os.path.join(self.tmp, "f032", "src"), os.path.join(self.tmp, "f032", "out")
        os.makedirs(src)
        # what the process of each PDF does here, in place of converting it (the files are never read as PDFs)
        stand_in = {"a_crash.pdf": "import sys; print('half a line'); print('  OCR a_crash.pdf: 1 of 2 scanned pages', "
                                   "file=sys.stderr); print('MuPDF error: boom', file=sys.stderr); sys.exit(3)",
                    "b_silent.pdf": "import os; os._exit(5)",
                    "c_slow.pdf": "import time; time.sleep(600)"}
        for name in stand_in:
            with open(os.path.join(src, name), "wb") as handle:
                handle.write(b"%PDF-1.7 placeholder")
        with open(os.path.join(src, "d_notes.txt"), "w", encoding="utf-8") as handle:
            handle.write("Genetic drift changes allele frequencies by chance.\n")
        real_child, started = conv._run_child, []

        def child(cmd, timeout):  # the real runner (pipes, time limit, kill) on the stand-in process
            name = os.path.basename(cmd[2])
            started.append((name, timeout))
            return real_child([sys.executable, "-c", stand_in[name]], timeout)

        argv = ["convert_to_markdown.py", src, "--out", out, "--timeout", "1"]
        with unittest.mock.patch.object(sys, "argv", argv), unittest.mock.patch.object(conv, "_run_child", child), \
                unittest.mock.patch.object(conv, "_lower_priority"), \
                unittest.mock.patch("sys.stdout", new_callable=__import__("io").StringIO) as shown:
            began = time.monotonic()
            conv.main()
            took = time.monotonic() - began
        limit = conv.PDF_SECONDS_PER_PAGE  # --timeout 1 is raised to the time one PDF page gets
        self.assertEqual(started, [(name, limit) for name in sorted(stand_in)])
        self.assertLess(took, 300, "the slow process was stopped at its limit, not waited for")
        with open(os.path.join(out, "conversion_report.tsv"), encoding="utf-8") as handle:
            rows = [line.rstrip("\n").split("\t") for line in handle][1:]
        report = {os.path.basename(path): (status, detail) for path, status, detail in rows}
        self.assertEqual(report["a_crash.pdf"], ("skipped", "MuPDF error: boom"))
        self.assertEqual(report["b_silent.pdf"], ("skipped", "the converter crashed on this file (exit code 5)"))
        self.assertEqual(report["c_slow.pdf"],
                         ("skipped", f"timed out after {limit} s (raise --timeout to allow longer)"))
        self.assertEqual(report["d_notes.txt"][0], "converted (text)")  # the file after the three still converts
        with open(os.path.join(out, "d_notes.md"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Genetic drift changes allele frequencies by chance.\n")
        text = shown.getvalue()
        self.assertIn("  OCR a_crash.pdf: 1 of 2 scanned pages\n", text)  # a worker's OCR progress is passed on
        self.assertIn("[3/4] skipped                c_slow.pdf -> timed out after", text)
        self.assertIn("1 of 4 files converted; 3 skipped or already present", text)


if __name__ == "__main__":
    unittest.main()
