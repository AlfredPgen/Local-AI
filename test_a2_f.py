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
        with unittest.mock.patch.object(yt, "ydl", lambda *a, **k: FakeYdl()), \
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
        self.assertNotIn("url", memory[specs[0]])
        self.assertNotIn("url", memory[specs[1]])
        self.assertNotIn("url", memory[specs[2]])
        self.assertEqual(memory["@somechannel"]["url"], f"https://www.youtube.com/channel/{channel_id}")
        # an entry an earlier version wrote for a video URL (its whole channel) is harmless now
        with unittest.mock.patch.object(yt, "ydl", lambda *a, **k: FakeYdl()), \
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


if __name__ == "__main__":
    unittest.main()
