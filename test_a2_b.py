"""Staging tests of package B (data_prep.py) for the audit of 2026-11-10.

They move into test_tiny_gpt.py (TestPipeline for builds, TestFixes for unit tests) once every package
has landed; until then:
    python -m unittest -v test_a2_b
"""

import collections
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_prep  # noqa: E402
from test_tiny_gpt import Corpus  # noqa: E402


def read_file(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class Audit2DataPrep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2b_")
        cls._corpus = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def corpus(cls):
        """The messy test corpus of test_tiny_gpt, built once and only for the tests that need it."""
        if cls._corpus is None:
            cls._corpus = Corpus(os.path.join(cls.tmp, "corpus"))
        return cls._corpus

    def folder(self, name):
        path = os.path.join(self.tmp, name)
        os.makedirs(path)
        return path

    @classmethod
    def base_args(cls):
        """A two-source build (the text folders of the corpus), as test_resume_after_pass2 uses."""
        c = cls.corpus()
        return ["--md-dir", c.md, "--md-dir", f"notes={c.notes}", "--vocab-size", "600", "--val-fraction", "0.2",
                "--threads", "2", "--workers", "1"]

    @classmethod
    def plain(cls):
        """The uninterrupted build of base_args (made once): what a stopped and resumed build must equal."""
        out = os.path.join(cls.tmp, "plain")
        if not os.path.isdir(out):
            with redirect_stdout(io.StringIO()):
                data_prep.prepare(["--out", out] + cls.base_args())
        return out

    def stage(self, out):
        return data_prep._load_state(os.path.join(out, "_work", data_prep.STATE_NAME))["stage"]

    # ---------------- F057: Ctrl+C in the leakage audit stops the build, and --resume runs the audit ----------------
    def test_ctrl_c_in_the_leakage_audit_stops_a_resumable_build(self):
        plain = self.plain()
        self.assertTrue(os.path.isfile(os.path.join(plain, "leakage.tsv")))
        out = os.path.join(self.tmp, "f057")
        real = data_prep._audit_chunk

        def interrupt_once(texts):
            data_prep._audit_chunk = real
            raise KeyboardInterrupt

        data_prep._audit_chunk = interrupt_once
        try:
            with redirect_stdout(io.StringIO()) as buf, self.assertRaises(KeyboardInterrupt):
                data_prep.prepare(["--out", out] + self.base_args())
        finally:
            data_prep._audit_chunk = real
        self.assertEqual(self.stage(out), "pass2", "the split and the audit are redone by --resume")
        self.assertIn("--resume", buf.getvalue())
        self.assertFalse(os.path.exists(os.path.join(out, "manifest.json")))
        with redirect_stdout(io.StringIO()) as buf:
            data_prep.prepare(["--out", out, "--resume", "--threads", "2", "--workers", "1"])
        self.assertIn("Leakage audit:", buf.getvalue())
        self.assertEqual(data_prep.read_manifest(out)["fingerprint"], data_prep.read_manifest(plain)["fingerprint"])
        self.assertEqual(read_file(os.path.join(out, "leakage.tsv")), read_file(os.path.join(plain, "leakage.tsv")))

    # ---------------- F062: builds that share one --work-dir keep their spools apart ----------------
    def stop_after_pass2(self, argv):
        """prepare(argv) with the tokenizer training failing once: the build stops resumable; returns its output."""
        real = data_prep._train_tokenizer

        def fail_once(*a, **k):
            data_prep._train_tokenizer = real
            raise OSError("disk full (simulated)")

        data_prep._train_tokenizer = fail_once
        try:
            with redirect_stdout(io.StringIO()) as buf, self.assertRaises(OSError):
                data_prep.prepare(argv)
        finally:
            data_prep._train_tokenizer = real
        return buf.getvalue()

    def spools(self, folder):
        return sorted(os.path.relpath(os.path.join(root, name), folder) for root, _, names in os.walk(folder)
                      for name in names if name.startswith("spool") and name != "spool_x.txt")

    def test_a_second_build_in_the_same_work_dir_leaves_a_stopped_builds_spool(self):
        shared = self.folder("f062_work")
        mine = os.path.join(shared, "spool_x.txt")  # a file of the user in the folder: never removed
        with open(mine, "w") as f:
            f.write("keep me")
        h1, h2 = os.path.join(self.tmp, "f062_h1"), os.path.join(self.tmp, "f062_h2")
        self.stop_after_pass2(["--out", h1, "--work-dir", shared] + self.base_args())
        kept = self.spools(shared)
        self.assertEqual(len(kept), 1, kept)
        self.assertTrue(kept[0].startswith("f062_h1_") and os.path.dirname(kept[0]).endswith("_spool"), kept)
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", h2, "--work-dir", shared, "--scan-only"] + self.base_args())
        self.assertEqual(self.spools(shared), kept, "the other build removed only its own spool")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", h1, "--resume", "--threads", "2", "--workers", "1"])
        self.assertEqual(data_prep.read_manifest(h1)["fingerprint"],
                         data_prep.read_manifest(self.plain())["fingerprint"])
        self.assertEqual(self.spools(shared), [])
        self.assertEqual(os.listdir(shared), ["spool_x.txt"], "the sub-folders are gone, the user's file is not")

    def test_resume_finds_a_spool_in_the_work_dir_itself(self):
        """A build stopped before the sub-folders existed has its spool directly in --work-dir: the path
        saved in state.pkl is used, never a recomputed one."""
        shared = self.folder("f062_old")
        out = os.path.join(self.tmp, "f062_h3")
        self.stop_after_pass2(["--out", out, "--work-dir", shared] + self.base_args())
        state_path = os.path.join(out, "_work", data_prep.STATE_NAME)
        st = data_prep._load_state(state_path)
        old = os.path.join(shared, os.path.basename(st["spool2"]))
        os.replace(st["spool2"], old)
        os.rmdir(os.path.dirname(st["spool2"]))
        st["spool2"] = old
        data_prep._save_state(state_path, st)
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", out, "--resume", "--threads", "2", "--workers", "1"])
        self.assertEqual(data_prep.read_manifest(out)["fingerprint"],
                         data_prep.read_manifest(self.plain())["fingerprint"])
        self.assertFalse(os.path.exists(old))

    # ---------------- F080: a finished --keep-work dataset is finished ----------------
    def test_finished_keep_work_dataset_is_not_a_stopped_build(self):
        out = os.path.join(self.tmp, "f080")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", out, "--keep-work"] + self.base_args())
        self.assertTrue(os.path.isfile(os.path.join(out, "_work", data_prep.STATE_NAME)))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            data_prep.prepare(["--out", out, "--resume"])
        self.assertIn("finished dataset (manifest.json", err.getvalue())
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop:
            data_prep.prepare(["--out", out, "--keep-work"] + self.base_args())
        self.assertIn("already exists", str(stop.exception))
        self.assertNotIn("stopped after pass 2", str(stop.exception))

    # ---------------- F081: --resume names the options it ignores ----------------
    def test_resume_names_the_options_it_ignores(self):
        out = os.path.join(self.tmp, "f081")
        self.stop_after_pass2(["--out", out] + self.base_args())
        with redirect_stdout(io.StringIO()) as buf:
            args = data_prep.parse_args(["--out", out, "--resume", "--threads", "2", "--workers", "1", "--keep-work"])
        self.assertNotIn("ignored", buf.getvalue(), "the machine settings are not ignored")
        self.assertTrue(args.resume)
        with redirect_stdout(io.StringIO()) as buf:
            data_prep.prepare(["--out", out, "--resume", "--vocab-size", "900", "--md-exclude", "*.lt.md",
                               "--threads", "2", "--workers", "1"])
        note = [line for line in buf.getvalue().splitlines() if "ignored" in line]
        self.assertEqual(len(note), 1, buf.getvalue())
        self.assertIn("--vocab-size", note[0])
        self.assertIn("--md-exclude", note[0])
        for flag in ("--threads", "--workers", "--out", "--resume", "--val-fraction"):
            self.assertNotIn(flag + ",", note[0].split("ignored")[0])
        manifest = data_prep.read_manifest(out)
        self.assertEqual(manifest["tokenizer"]["vocab_size"], 600, "the saved build's vocabulary, not 900")
        self.assertEqual(manifest["fingerprint"], data_prep.read_manifest(self.plain())["fingerprint"])

    # ---------------- F079: numeric options are checked before anything is read ----------------
    def test_numeric_options_are_checked_at_the_start(self):
        folder = self.folder("f079")
        base = ["--out", os.path.join(folder, "out"), "--md-dir", folder]
        with redirect_stderr(io.StringIO()):
            data_prep.parse_args(base)  # the defaults pass
            for good in (["--near-dup-max-docs", "0"], ["--keyword-min-hits", "0"], ["--keyword-min-distinct", "0"],
                         ["--title-weight", "0"], ["--max-val-group-share", "1"], ["--boilerplate-min-docs", "2"],
                         ["--boilerplate-min-len", "1"]):
                data_prep.parse_args(base + good)
        for bad in (["--tokenizer-sample-chars", "0"], ["--val-max-chars", "-5"], ["--max-val-group-share", "-1"],
                    ["--near-dup-max-docs", "-3"], ["--boilerplate-min-len", "-1"], ["--boilerplate-min-len", "0"],
                    ["--max-val-group-share", "1.5"], ["--max-val-group-share", "0"], ["--boilerplate-min-docs", "1"],
                    ["--keyword-min-hits", "-1"], ["--keyword-min-distinct", "-1"], ["--title-weight", "-1"]):
            with self.subTest(bad=bad):
                with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                    data_prep.parse_args(base + bad)
                self.assertIn(bad[0], err.getvalue())
        # --resume: the machine settings of the new command are checked too
        stopped = os.path.join(folder, "stopped")
        os.makedirs(os.path.join(stopped, "_work"))
        open(os.path.join(stopped, "_work", data_prep.STATE_NAME), "wb").close()
        with redirect_stdout(io.StringIO()):
            data_prep.parse_args(["--out", stopped, "--resume", "--threads", "1", "--workers", "1"])
        for bad in (["--threads", "0"], ["--workers", "0"]):
            with self.subTest(bad=bad):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err, \
                        self.assertRaises(SystemExit):
                    data_prep.parse_args(["--out", stopped, "--resume"] + bad)
                self.assertIn(bad[0], err.getvalue())

    # ---------------- F102: the hint after a failure says what can be done ----------------
    def test_failure_hints_after_pass2(self):
        # a resume that cannot continue does not advise --resume again
        out = os.path.join(self.tmp, "f102_cut")
        self.stop_after_pass2(["--out", out] + self.base_args())
        spool2 = data_prep._load_state(os.path.join(out, "_work", data_prep.STATE_NAME))["spool2"]
        with open(spool2, "rb") as f:
            raw = f.read()
        with open(spool2, "wb") as f:
            f.write(raw[:len(raw) // 2])
        with redirect_stdout(io.StringIO()) as buf, self.assertRaises(SystemExit) as stop:
            data_prep.prepare(["--out", out, "--resume", "--threads", "2", "--workers", "1"])
        self.assertIn("cannot continue", str(stop.exception))
        self.assertNotIn("Continue from them", buf.getvalue())
        self.assertNotIn("--resume", buf.getvalue())
        # with --keep-work a build stopped after pass 2 gets the hint too
        out = os.path.join(self.tmp, "f102_keep")
        printed = self.stop_after_pass2(["--out", out, "--keep-work"] + self.base_args())
        self.assertIn("Continue from them", printed)
        self.assertIn("--resume", printed)
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", out, "--resume", "--threads", "2", "--workers", "1"])
        self.assertEqual(data_prep.read_manifest(out)["fingerprint"],
                         data_prep.read_manifest(self.plain())["fingerprint"])
        # and a --keep-work build stopped before pass 2 ended keeps its work folder, with no hint (guard)
        out = os.path.join(self.tmp, "f102_early")
        with unittest.mock.patch.object(data_prep, "_pass2_chunk", side_effect=KeyboardInterrupt):
            with redirect_stdout(io.StringIO()) as buf, self.assertRaises(KeyboardInterrupt):
                data_prep.prepare(["--out", out, "--keep-work"] + self.base_args())
        self.assertTrue(os.path.isdir(os.path.join(out, "_work")))
        self.assertNotIn("--resume", buf.getvalue())

    # ---------------- F101: a build stopped during pass 2 really removes its work files ----------------
    def test_build_stopped_in_pass2_leaves_no_work_files(self):
        real, calls = data_prep._pass2_chunk, []

        def interrupt_at_20(jobs):
            calls.append(1)
            if len(calls) == 20:
                raise KeyboardInterrupt
            return real(jobs)

        for name, extra in (("f101", []), ("f101_wd", ["--work-dir", self.folder("f101_work")])):
            out = os.path.join(self.tmp, name)
            del calls[:]
            with unittest.mock.patch.object(data_prep, "_pass2_chunk", interrupt_at_20):
                with redirect_stdout(io.StringIO()) as buf, self.assertRaises(KeyboardInterrupt):
                    data_prep.prepare(["--out", out] + extra + self.base_args())
            self.assertEqual(len(calls), 20)
            self.assertFalse(os.path.exists(os.path.join(out, "_work")), buf.getvalue())
            self.assertIn("removed the work spool", buf.getvalue())
            if extra:
                self.assertEqual(os.listdir(extra[1]), [], "the spool sub-folder of --work-dir is gone too")

    # ---------------- F061: a resumed build keeps the command and the time of the build ----------------
    def test_resumed_build_records_the_original_command_and_the_whole_time(self):
        out = os.path.join(self.tmp, "f061")
        first = ["data_prep.py", "--out", "f061", "--md-dir", "texts", "--vocab-size", "600"]
        with unittest.mock.patch.object(sys, "argv", first):
            self.stop_after_pass2(["--out", out] + self.base_args())
        state_path = os.path.join(out, "_work", data_prep.STATE_NAME)
        st = data_prep._load_state(state_path)
        self.assertEqual(st["command"], " ".join(first))
        self.assertGreater(st["seconds"], 0)
        st["seconds"] = 5000.0  # as if reading had taken 83 minutes
        data_prep._save_state(state_path, st)
        with unittest.mock.patch.object(sys, "argv", ["data_prep.py", "--out", "f061", "--resume"]):
            with redirect_stdout(io.StringIO()):
                data_prep.prepare(["--out", out, "--resume", "--threads", "2", "--workers", "1"])
        manifest = data_prep.read_manifest(out)
        self.assertEqual(manifest["command"], " ".join(first) + " (resumed)")
        self.assertEqual(manifest["fingerprint"], data_prep.read_manifest(self.plain())["fingerprint"])
        report = read_file(os.path.join(out, "report.md"))
        self.assertIn("Command: `" + " ".join(first) + " (resumed)`", report)
        seconds = int(re.search(r" in ([\d,]+) s\. Command", report).group(1).replace(",", ""))
        self.assertTrue(5000 <= seconds < 5600, seconds)
        # a state saved before these keys existed: the resume command and this session's time, as before
        out = os.path.join(self.tmp, "f061_old")
        self.stop_after_pass2(["--out", out] + self.base_args())
        state_path = os.path.join(out, "_work", data_prep.STATE_NAME)
        st = data_prep._load_state(state_path)
        del st["command"], st["seconds"]
        data_prep._save_state(state_path, st)
        with unittest.mock.patch.object(sys, "argv", ["data_prep.py", "--out", "f061_old", "--resume"]):
            with redirect_stdout(io.StringIO()):
                data_prep.prepare(["--out", out, "--resume", "--threads", "2", "--workers", "1"])
        self.assertEqual(data_prep.read_manifest(out)["command"], "data_prep.py --out f061_old --resume (resumed)")

    # ---------------- F104: a UTF-8 byte-order mark at the start of a keyword or JSONL file ----------------
    def test_byte_order_mark_is_not_part_of_the_first_line(self):
        folder = self.folder("f104")
        keywords = os.path.join(folder, "keywords.txt")
        with open(keywords, "wb") as f:
            f.write(b"\xef\xbb\xbf# biology terms\ngenome\n")
        self.assertEqual(data_prep.read_terms(keywords), ["genome"])
        records = "".join(json.dumps({"id": f"r{i}", "title": "", "text": f"Record {i} about alleles."}) + "\n"
                          for i in range(3))
        plain, packed = os.path.join(folder, "bom.jsonl"), os.path.join(folder, "bom.jsonl.gz")
        with open(plain, "wb") as f:
            f.write(b"\xef\xbb\xbf" + records.encode())
        with gzip.open(packed, "wb") as f:
            f.write(b"\xef\xbb\xbf" + records.encode())
        for path in (plain, packed):
            stats = data_prep._new_stats()
            docs = list(data_prep._jsonl_python(path, "text", "title", "id", 0, stats, None, 5000, 0, lambda: None))
            self.assertEqual([d[0] for d in docs], ["r0", "r1", "r2"], path)
            self.assertEqual(sum(stats["skipped"].values()), 0)

    # ---------------- F092: Ctrl+C during 'top' scoring does not wait for every file ----------------
    def test_ctrl_c_during_top_scoring_drops_the_files_not_started(self):
        started = []

        def slow_score(source, file_index, kw, min_score):
            started.append(file_index)
            time.sleep(0.5)
            empty = np.zeros(0)
            return collections.Counter({"scanned": 1}), (empty, empty, empty, empty)

        class StopAtSecondFile(io.StringIO):
            def write(self, text):
                if "scored file 2/12" in text:
                    raise KeyboardInterrupt
                return super().write(text)

        source = unittest.mock.Mock(paths=list(range(12)))
        t0 = time.time()
        with unittest.mock.patch.object(data_prep, "_score_file", slow_score):
            with redirect_stdout(StopAtSecondFile()), self.assertRaises(KeyboardInterrupt):
                list(data_prep.iter_table_source("wiki", source, None, 5, "top", None, 1.0, 0,
                                                 data_prep._new_stats(), threads=2))
            waited = time.time() - t0
            time.sleep(1.5)  # time enough for three more rounds, had the queued files not been cancelled
        self.assertLessEqual(len(started), 4, "2 scored + at most 2 running; the other 8 were never started")
        self.assertLess(waited, 2.0, "12 files at 0.5 s on 2 threads take 3 s when all are waited for")

    # ---------------- F265: a lock that exists but is not written yet belongs to a live process ----------------
    def test_lock_created_but_not_yet_written_is_not_stale(self):
        folder = self.folder("f265")
        lock = os.path.join(folder, data_prep.LOCK_NAME)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # another start, between creating and writing
        try:
            with self.assertRaises(SystemExit) as stop:
                data_prep._acquire_lock(lock)
            self.assertIn("is being built by another data_prep.py run", str(stop.exception))
            with self.assertRaises(SystemExit) as stop:
                data_prep._acquire_lock(lock, busy="Run 'x' is being trained by another process.")
            self.assertEqual(str(stop.exception), "Run 'x' is being trained by another process.")
        finally:
            os.close(fd)
        self.assertEqual(os.path.getsize(lock), 0, "the other process's lock was not touched")
        # the same empty file left by a crash is stale once it is older than ten seconds
        old = time.time() - 60
        os.utime(lock, (old, old))
        self.assertEqual(data_prep._acquire_lock(lock), lock)
        with open(lock, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["pid"], os.getpid())
        data_prep._release_lock(lock)
        self.assertFalse(os.path.exists(lock))

    # ---------------- F258: the token files are hashed in threads, with the same check ----------------
    def test_token_files_are_verified_in_threads_and_the_first_bad_file_is_named(self):
        ds = os.path.join(self.tmp, "f258")
        shutil.copytree(self.plain(), ds)
        real, threads = data_prep.sha256_file, []

        def recorded(path, *a, **k):
            threads.append(threading.get_ident())
            return real(path, *a, **k)

        with unittest.mock.patch.object(data_prep, "sha256_file", recorded):
            manifest, arrays = data_prep.open_token_files(ds, verify=True)
        files = [info[split]["file"] for info in manifest["sources"].values() for split in ("train", "val")
                 if info.get(split)]
        self.assertEqual(files[:3], ["train.md.bin", "val.md.bin", "train.notes.bin"])
        self.assertEqual(len(threads), len(files), "every token file is hashed once")
        self.assertNotIn(threading.main_thread().ident, threads, "hashed in worker threads, not one after another")
        self.assertEqual(sum(len(a) for a in arrays.values()), len(files))
        del arrays  # the memory maps: Windows does not let a mapped file be rewritten
        for name in ("val.md.bin", "train.notes.bin"):  # one changed byte in two files: the first one is named
            with open(os.path.join(ds, name), "r+b") as f:
                byte = f.read(1)
                f.seek(0)
                f.write(bytes([byte[0] ^ 1]))
        with self.assertRaises(SystemExit) as stop:
            data_prep.open_token_files(ds, verify=True)
        self.assertIn("val.md.bin failed its SHA-256 check", str(stop.exception))
        with unittest.mock.patch.object(data_prep, "sha256_file", side_effect=AssertionError("hashed")):
            data_prep.open_token_files(ds, verify=False)  # sizes only: nothing is hashed

    def test_an_error_in_the_leakage_audit_still_only_skips_it(self):
        out = os.path.join(self.tmp, "f057err")
        real = data_prep._audit_chunk

        def fail_once(texts):
            data_prep._audit_chunk = real
            raise ValueError("simulated")

        data_prep._audit_chunk = fail_once
        try:
            with redirect_stdout(io.StringIO()) as buf:
                data_prep.prepare(["--out", out] + self.base_args())
        finally:
            data_prep._audit_chunk = real
        self.assertIn("Leakage audit failed (ValueError)", buf.getvalue())
        self.assertIn("were not moved", buf.getvalue())
        self.assertIn("The leakage audit failed (ValueError) and was skipped.", read_file(os.path.join(out, "report.md")))
        self.assertTrue(os.path.isfile(os.path.join(out, "manifest.json")))
        self.assertFalse(os.path.exists(os.path.join(out, "leakage.tsv")))

    # ---------------- F043: a damaged .json.gz shard ends that file, not the build ----------------
    def jsonl_shard(self, path, n=4000):
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as f:
            for i in range(n):
                text = f"Genetic drift record {i}: " + hashlib.sha256(str(i).encode()).hexdigest() * 3
                f.write(json.dumps({"id": f"id{i}", "title": f"T{i}", "text": text}) + "\n")

    def test_truncated_jsonl_gz_is_counted_not_fatal(self):
        folder = self.folder("f043")
        intact, cut = os.path.join(folder, "intact.jsonl.gz"), os.path.join(folder, "cut.jsonl.gz")
        self.jsonl_shard(intact)
        with open(intact, "rb") as f:
            raw = f.read()
        with open(cut, "wb") as f:
            f.write(raw[:len(raw) // 2])
        stats = data_prep._new_stats()
        with redirect_stdout(io.StringIO()) as buf:
            docs = list(data_prep.iter_jsonl([cut, intact], "text", "title", "id", 0, stats))
        ids = [d[0] for d in docs]
        k = len(ids) - 4000
        self.assertTrue(500 < k < 3500, f"{k} records read before the cut")
        self.assertEqual(ids, [f"id{i}" for i in range(k)] + [f"id{i}" for i in range(4000)])
        self.assertEqual(stats["skipped"]["bad_gzip"], 1)
        self.assertEqual(len(stats["warnings"]), 1)
        self.assertIn("cut.jsonl.gz is damaged after record", stats["warnings"][0])
        self.assertIn("download it again", stats["warnings"][0])
        self.assertNotIn("garbled", stats["warnings"][0], "a plain truncation: what was read is sound")
        self.assertIn("WARNING", buf.getvalue())
        # the intact file alone: nothing counted, no warning
        stats = data_prep._new_stats()
        with redirect_stdout(io.StringIO()):
            docs = list(data_prep.iter_jsonl([intact], "text", "title", "id", 0, stats))
        self.assertEqual(len(docs), 4000)
        self.assertNotIn("bad_gzip", stats["skipped"])
        self.assertNotIn("warnings", stats)
        # in a build the warning becomes a note of the report
        out = os.path.join(folder, "scan")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", out, "--jsonl", cut, "--jsonl-name", "papers", "--scan-only",
                               "--threads", "2", "--workers", "1"])
        report = read_file(os.path.join(out, "report.md"))
        self.assertIn("bad_gzip 1", report)
        self.assertIn("**Note:** `papers`: cut.jsonl.gz is damaged after record", report)

    def test_jsonl_gz_with_a_failed_checksum_warns_of_garbled_records(self):
        folder = self.folder("f043crc")
        path = os.path.join(folder, "crc.jsonl.gz")
        self.jsonl_shard(path, 200)
        with open(path, "rb") as f:
            raw = bytearray(f.read())
        raw[-8] ^= 0xFF  # the stored CRC-32: every record still reads, the check at the end fails
        with open(path, "wb") as f:
            f.write(raw)
        stats = data_prep._new_stats()
        with redirect_stdout(io.StringIO()):
            docs = list(data_prep._jsonl_python(path, "text", "title", "id", 0, stats, None, 5000, 0, lambda: None))
        self.assertEqual(len(docs), 200)
        self.assertEqual(stats["skipped"]["bad_gzip"], 1)
        self.assertIn("may be garbled", stats["warnings"][0])

    # ---------------- F047: above PREFILTER_MAX_PATTERNS the RE2 prefilter is skipped, same documents ----------------
    def kept_rows(self, out):
        rows = [line.split("\t") for line in read_file(os.path.join(out, "docs.tsv")).splitlines()[1:]]
        return [row for row in rows if row[3] in ("train", "val")]

    def test_prefilter_gate_changes_no_kept_document(self):
        self.assertEqual(data_prep.PREFILTER_MAX_PATTERNS, 24)
        c = self.corpus()
        common = ["--md-dir", c.md, "--wiki-dir", c.wiki, "--jsonl", c.pes2o, "--jsonl-name", "pes2o",
                  "--parquet", c.parquet, "--parquet-name", "web", "--min-score", "2",
                  "--include-keywords", "genetic*,allele*,protein,DNA,RNA,genome*", "--keyword-min-distinct", "2",
                  "--vocab-size", "700", "--val-fraction", "0.2", "--threads", "2", "--workers", "1"]
        real = data_prep._vector_scores
        for mode, wiki_max in (("first", "0"), ("top", "30")):
            built = {}
            for limit in (0, 10 ** 9):  # prefilter never used / always used
                with_kw = []

                def recorded(arrays, kw, *a, **k):
                    with_kw.append(kw is not None)
                    return real(arrays, kw, *a, **k)

                out = os.path.join(self.tmp, f"f047_{mode}_{'on' if limit else 'off'}")
                with unittest.mock.patch.object(data_prep, "PREFILTER_MAX_PATTERNS", limit), \
                        unittest.mock.patch.object(data_prep, "_vector_scores", recorded), \
                        redirect_stdout(io.StringIO()):
                    data_prep.prepare(["--out", out, "--wiki-max-docs", wiki_max] + common)
                manifest = data_prep.read_manifest(out)
                built[limit] = (manifest["fingerprint"], self.kept_rows(out))
                self.assertTrue(with_kw, "the readers still ask for characters and scores")
                if mode == "first" and not limit:
                    self.assertFalse(any(with_kw), "no reader counted keywords: every rejection is the full check's")
                else:  # 'top' selection ranks by the RE2 counts whatever the limit
                    self.assertTrue(any(with_kw))
                self.assertEqual(manifest["sources"]["wiki"]["select"], mode)
            self.assertEqual(built[0][0], built[10 ** 9][0], mode)
            self.assertEqual(built[0][1], built[10 ** 9][1], mode)
            self.assertGreater(len(built[0][1]), 40)
        # the line-by-line JSONL reader has the same gate
        kw = data_prep.KeywordFilter(["genetic*", "allele*"], [], min_hits=1, min_distinct=1)
        lines = [(1, json.dumps({"id": "a", "title": "", "text": "Genetic drift changes allele frequencies."})),
                 (2, json.dumps({"id": "b", "title": "", "text": "A football match report."}))]
        for limit, ids, skipped in ((0, ["a", "b"], 0), (10 ** 9, ["a"], 1)):
            stats = data_prep._new_stats()
            with unittest.mock.patch.object(data_prep, "PREFILTER_MAX_PATTERNS", limit):
                docs = list(data_prep._jsonl_batch(lines, "x.jsonl", "text", "title", "id", stats, kw))
            self.assertEqual([d[0] for d in docs], ids)
            self.assertEqual(stats["skipped"]["keyword_min_hits"], skipped)

    # ---------------- F093: callers that never use the RE2 patterns do not build them ----------------
    def test_keyword_filter_without_prefilter_builds_no_patterns(self):
        terms = ["=MET", "*mab", "genom*", "Hardy-Weinberg", "genetic drift"]
        texts = [("", "Genomes and genomics under the Hardy Weinberg law."), ("MET", "A met gene; genetic drift."),
                 ("Antibodies", "Rituximab and other drugs."), ("", "A football match report.")]
        folder = self.folder("f093")
        cache = os.path.join(folder, "keywords.txt.prefilter.json")
        with unittest.mock.patch.object(data_prep, "_prefilter_patterns",
                                        side_effect=AssertionError("patterns were built")):
            plain = data_prep.KeywordFilter(terms, ["football"], 1, 1, prefilter=False, pattern_cache=cache)
        self.assertEqual(plain.include_patterns, [])
        self.assertFalse(os.path.exists(cache), "nothing is built, so nothing is cached")
        self.assertFalse(data_prep._use_prefilter(plain))
        full = data_prep.KeywordFilter(terms, ["football"], 1, 1)
        self.assertEqual(full.include_patterns, data_prep._prefilter_patterns(terms))
        self.assertTrue(plain.active)
        for title, text in texts:
            self.assertEqual(plain.evaluate(title, text), full.evaluate(title, text))
        with self.assertRaises(TypeError):  # keyword-only: a seventh positional argument is still pattern_cache's place
            data_prep.KeywordFilter(terms, [], 1, 1, 3, 100_000, None, False)

    # ---------------- F065: a wildcard next to a non-ASCII letter is found by the RE2 prefilter too ----------------
    def test_prefilter_finds_wildcard_terms_that_end_in_a_non_ascii_letter(self):
        import pyarrow as pa
        for term, text in (("café*", "a café here"), ("café*", "two cafés"), ("*école", "the école"),
                           ("*école", "an autoécole"), ("TNF-α*", "TNF-α rises"), ("=TNF-α*", "TNF-α rises")):
            with self.subTest(term=term, text=text):
                kw = data_prep.KeywordFilter([term], [], min_hits=1, min_distinct=1)
                exact = kw.evaluate("", text)[2]
                self.assertEqual(exact, 1)
                hits, _, _ = data_prep._vector_scores({"text": pa.array([text])}, kw, "text", "title", None, None)
                self.assertGreaterEqual(int(hits[0]), exact, kw.include_patterns)
        # next to ASCII letters the patterns are what they were
        self.assertEqual(data_prep._re2_term(["genom"], False, False, True), r"\bgenom\w*\b")
        self.assertEqual(data_prep._re2_term(["mab"], False, True, False), r"\b\w*mab\b")
        self.assertEqual(data_prep._re2_term(["MET"], True, False, False), r"(?-i:\bMET\b)")
        self.assertEqual(data_prep._re2_term(["genetic", "drif"], False, False, True), r"\bgenetic\W+drif\w*\b")
        self.assertEqual(data_prep._re2_term(["α", "synuclein"], False, False, False), r"α\W+synuclein\b")

    # ---------------- F086: the pattern cache is rebuilt when the pattern rules change ----------------
    def test_prefilter_cache_misses_when_the_pattern_version_changes(self):
        terms = ["genetic drift", "allele*"]
        cache = os.path.join(self.folder("f086"), "kw.txt.prefilter.json")
        fresh = data_prep.cached_prefilter_patterns(terms, cache)
        with open(cache, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["version"], data_prep.PREFILTER_VERSION)
        with unittest.mock.patch.object(data_prep, "_prefilter_patterns", side_effect=AssertionError("rebuilt")):
            self.assertEqual(data_prep.cached_prefilter_patterns(terms, cache), fresh, "same version: a hit")
        with unittest.mock.patch.object(data_prep, "PREFILTER_VERSION", data_prep.PREFILTER_VERSION + 1), \
                unittest.mock.patch.object(data_prep, "_prefilter_patterns", return_value=["(?:sentinel)"]) as built:
            self.assertEqual(data_prep.cached_prefilter_patterns(terms, cache), ["(?:sentinel)"])
            built.assert_called_once_with(terms)
            self.assertEqual(data_prep.cached_prefilter_patterns(terms, cache), ["(?:sentinel)"])
            built.assert_called_once_with(terms)  # the rewritten cache is a hit for the new version

    # ---------------- F041: wildcard terms are looked up by the lengths their three-letter head has ----------------
    def test_affix_lookup_by_head_counts_what_the_length_loop_counted(self):
        import random
        rng = random.Random(41)

        def word(lo, hi):
            return "".join(rng.choice("abcd") for _ in range(rng.randint(lo, hi)))

        freq = collections.Counter(word(1, 9) for _ in range(4000))
        for suffix in (False, True):
            for shortest in (3, 2, 1):  # with a term under three letters there is no head table: the old loop runs
                texts = sorted({word(shortest, 7) for _ in range(60)} | {word(shortest, shortest)})
                affixes = data_prep._Affixes(suffix)
                for text in texts:
                    affixes.add(text, ("*" + text) if suffix else (text + "*"))
                affixes.finish()
                if shortest >= 3:
                    self.assertEqual({h: list(v) for h, v in affixes.by_head.items()},
                                     {h: sorted({len(t) for t in texts if (t[-3:] if suffix else t[:3]) == h})
                                      for h in affixes.heads})
                else:
                    self.assertIsNone(affixes.by_head)
                expected = collections.Counter()
                for w, c in freq.items():
                    for text in texts:
                        if w.endswith(text) if suffix else w.startswith(text):
                            expected[("*" + text) if suffix else (text + "*")] += c * 3
                counted = collections.Counter()
                affixes.count(freq, counted, 3)
                self.assertEqual(counted, expected, (suffix, shortest))
                self.assertGreater(len(expected), 20)
        # through the matcher: prefix, suffix and a two-letter term together
        kw = data_prep.KeywordFilter(["genom*", "gen*", "*mab", "*ab", "io*"], [], min_hits=1, min_distinct=1)
        terms = kw.evaluate("", "Genomes, genes and ions: rituximab in the lab. Genomics again.")[3]
        self.assertEqual(dict(terms), {"genom*": 2, "gen*": 3, "*mab": 1, "*ab": 2, "io*": 1})

    # ---------------- F064: one bounded, ordered read-ahead for the table and the JSONL reader ----------------
    def test_read_ahead_keeps_the_order_and_the_bound(self):
        import random
        rng = random.Random(64)
        delays = [rng.random() * 0.02 for _ in range(40)]
        taken = []

        def source(n=40):
            for i in range(n):
                taken.append(i)
                yield i

        def work(i):
            time.sleep(delays[i])
            if i == 33:
                raise ValueError("item 33")
            return i, i * i

        for threads, window, bound in ((3, None, 6), (2, 5, 5), (0, None, 2)):
            del taken[:]
            got = []
            for result in data_prep._read_ahead(source(30), work, threads, window):
                got.append(result)
                self.assertLessEqual(len(taken) - len(got), bound, "items taken from the source but not yet used")
            self.assertEqual(got, [(i, i * i) for i in range(30)])
        # an error comes out at its own item, after everything before it
        del taken[:]
        got = []
        with self.assertRaises(ValueError):
            for result in data_prep._read_ahead(source(), work, 3):
                got.append(result)
        self.assertEqual(got, [(i, i * i) for i in range(33)])
        # a reader that stops early leaves no thread behind
        before = threading.active_count()
        ahead = data_prep._read_ahead(source(), work, 3)
        self.assertEqual([next(ahead), next(ahead)], [(0, 0), (1, 1)])
        self.assertGreater(threading.active_count(), before)
        ahead.close()
        self.assertEqual(threading.active_count(), before)

    # ---------------- F091: text files are opened ahead, in threads; the documents are the same ----------------
    @staticmethod
    def serial_markdown(dirs, patterns, excludes, recursive, stats, ledger, source="md"):
        """The reader as it was before the read-ahead, one file at a time: the reference of the test below."""
        import zlib
        seen, produced = set(), set()
        bases = [os.path.basename(os.path.normpath(d)) or f"folder{k + 1}" for k, d in enumerate(dirs)]
        if len({b.lower() for b in bases}) < len(bases):
            bases = [f"{k + 1}-{b}" for k, b in enumerate(bases)]
        excludes = [ex.lower().replace("\\", "/") for ex in excludes]
        for k, directory in enumerate(dirs):
            if not os.path.isdir(directory):
                raise SystemExit(f"--md-dir is not a folder: {directory}")
            for path in data_prep.list_text_files(directory, patterns, recursive):
                real = os.path.realpath(path)
                if real in seen or not os.path.isfile(real):
                    continue
                seen.add(real)
                rel = os.path.relpath(path, directory).replace(os.sep, "/")
                packed = rel.lower().endswith(".gz")
                if packed:
                    rel = rel[:-3]
                doc_id = f"{bases[k]}/{rel}" if len(dirs) > 1 else rel
                stats["scanned"] += 1
                if data_prep._excluded(path, rel, excludes):
                    stats["skipped"]["excluded_by_glob"] += 1
                    ledger.append(data_prep._ledger_row(source, doc_id, rel, "skipped", "excluded_by_glob",
                                                        os.path.getsize(path)))
                    continue
                size = os.path.getsize(path)
                if doc_id in produced:  # the rule of F089, added to this reference with it
                    name = os.path.basename(rel)
                    stats["skipped"]["duplicate_id"] += 1
                    ledger.append(data_prep._ledger_row(source, doc_id, rel, "skipped", f"duplicate_id: {name} and "
                                                        f"{name}.gz both present; first in sort order kept", size))
                    continue
                if size > 512 * 1024 * 1024:
                    stats["skipped"]["file_too_large"] += 1
                    ledger.append(data_prep._ledger_row(source, doc_id, rel, "skipped", "file_too_large", size))
                    continue
                with open(path, "rb") as handle:
                    raw = handle.read()
                if packed:
                    try:
                        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as unpacked:
                            raw = unpacked.read(512 * 1024 * 1024 + 1)
                    except (OSError, EOFError, zlib.error) as exc:
                        stats["skipped"]["bad_gzip"] += 1
                        ledger.append(data_prep._ledger_row(source, doc_id, rel, "skipped", f"bad_gzip: {exc}"[:80],
                                                            size))
                        continue
                    if len(raw) > 512 * 1024 * 1024:
                        stats["skipped"]["file_too_large"] += 1
                        ledger.append(data_prep._ledger_row(source, doc_id, rel, "skipped", "file_too_large",
                                                            len(raw)))
                        continue
                produced.add(doc_id)
                yield (doc_id, os.path.splitext(os.path.basename(rel))[0],
                       data_prep.decode_bytes(raw, stats["cleaning"]), {"path": path})

    def read_all(self, reader, dirs, recursive, **more):
        """(documents, stats, ledger, the SystemExit text if the reader stopped with one)."""
        stats, ledger, docs, stopped = data_prep._new_stats(), [], [], None
        try:
            for doc in reader(dirs, data_prep.MD_PATTERNS_DEFAULT, ["skip_*", "drafts/*"], recursive, stats, ledger,
                              "texts", **more):
                docs.append(doc)
        except SystemExit as exc:
            stopped = str(exc)
        return docs, stats, ledger, stopped

    def markdown_folder(self, name):
        root = self.folder(name)
        for sub in ("sub/deep", "drafts"):
            os.makedirs(os.path.join(root, sub))
        files = {"b.txt": "Plain text about alleles.\n", "empty.md": "", "skip_me.md": "excluded by its name",
                 "sub/c.md": "# C\n\nNested once.\n", "sub/deep/d.md": "# D\n\nNested twice.\n",
                 "drafts/e.md": "excluded by its path", "Upper.MD": "An upper-case ending.\n",
                 "big.md": "A long text. " * 500, "latin.md": b"caf\xe9 in Latin-1\n"}
        for rel, text in files.items():
            with open(os.path.join(root, rel), "wb") as f:
                f.write(text if isinstance(text, bytes) else text.encode("utf-8"))
        with gzip.open(os.path.join(root, "packed.md.gz"), "wb") as f:
            f.write(b"# Packed\n\nA compressed document.\n")
        with gzip.open(os.path.join(root, "sub", "c.md.gz"), "wb") as f:  # sub/c.md is there as well (F089)
            f.write(b"# C\n\nThe compressed copy of c.md.\n")
        with open(os.path.join(root, "bad.md.gz"), "wb") as f:
            f.write(b"this is not a gzip file")
        with open(os.path.join(root, "packed.md.gz"), "rb") as f:
            whole = f.read()
        with open(os.path.join(root, "cut.md.gz"), "wb") as f:
            f.write(whole[:len(whole) // 2])
        try:
            os.link(os.path.join(root, "b.txt"), os.path.join(root, "hard_link.txt"))
        except OSError:
            pass
        return root

    def test_markdown_read_ahead_gives_the_documents_of_the_serial_reader(self):
        root = self.markdown_folder("f091")
        sub, missing = os.path.join(root, "sub"), os.path.join(root, "no_such_folder")
        cases = [([root], True), ([root], False), ([root, root], True), ([root, sub], True), ([sub, root], True),
                 ([sub, missing, root], True)]
        for dirs, recursive in cases:
            expected = self.read_all(self.serial_markdown, dirs, recursive)
            self.assertEqual(expected[3] is not None, missing in dirs)
            for more in ({}, {"read_threads": 4, "read_window": 3, "read_ahead_bytes": 1000},
                         {"read_threads": 1, "read_window": 0, "read_ahead_bytes": 0}):
                with self.subTest(dirs=[os.path.relpath(d, root) for d in dirs], recursive=recursive, more=more):
                    found = self.read_all(data_prep.iter_markdown, dirs, recursive, **more)
                    self.assertEqual(found[0], expected[0])
                    self.assertEqual(found[1], expected[1])
                    self.assertEqual(found[2], expected[2])
                    self.assertEqual(found[3], expected[3])
        docs, stats, ledger, _ = self.read_all(data_prep.iter_markdown, [root], True)
        self.assertGreaterEqual(len(docs), 7)
        self.assertEqual(stats["skipped"]["excluded_by_glob"], 2)
        self.assertEqual(stats["skipped"]["bad_gzip"], 2)
        # the files are opened by the reader threads, never an excluded one, and only up to read_ahead_bytes
        real_open, opened = open, []

        def recorded(path, *a, **k):
            opened.append((os.path.basename(str(path)), threading.current_thread() is threading.main_thread()))
            return real_open(path, *a, **k)

        with unittest.mock.patch("builtins.open", recorded):
            self.read_all(data_prep.iter_markdown, [root], True, read_ahead_bytes=1000)
        by_name = dict(opened)
        self.assertEqual(len(opened), len(by_name), "every file is opened once")
        self.assertNotIn("skip_me.md", by_name)
        self.assertNotIn("e.md", by_name)
        self.assertTrue(by_name.pop("big.md"), "above read_ahead_bytes: read in turn by the main thread")
        self.assertFalse(any(by_name.values()), by_name)
        # a caller that stops reading early leaves no thread behind
        before = threading.active_count()
        reader = data_prep.iter_markdown([root], data_prep.MD_PATTERNS_DEFAULT, [], True, data_prep._new_stats(), [])
        next(reader)
        self.assertGreater(threading.active_count(), before)
        reader.close()
        self.assertEqual(threading.active_count(), before)

    # ---------------- F089: a.md and a.md.gz in one folder are one document id, so one document ----------------
    def test_file_and_its_gz_copy_give_one_document(self):
        folder = self.folder("f089")
        with open(os.path.join(folder, "a.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("The plain copy.\n")
        with gzip.open(os.path.join(folder, "a.md.gz"), "wb") as f:
            f.write(b"The compressed copy, edited later.\n")
        with gzip.open(os.path.join(folder, "only.md.gz"), "wb") as f:
            f.write(b"Compressed only.\n")
        stats, ledger = data_prep._new_stats(), []
        docs = list(data_prep.iter_markdown([folder], data_prep.MD_PATTERNS_DEFAULT, [], False, stats, ledger, "notes"))
        self.assertEqual([(d[0], d[2]) for d in docs], [("a.md", "The plain copy.\n"), ("only.md", "Compressed only.\n")])
        self.assertEqual(stats["scanned"], 3)
        self.assertEqual(stats["skipped"]["duplicate_id"], 1)
        self.assertEqual(len(ledger), 1)
        self.assertEqual((ledger[0]["source"], ledger[0]["doc_id"], ledger[0]["status"]), ("notes", "a.md", "skipped"))
        self.assertEqual(ledger[0]["reason"], "duplicate_id: a.md and a.md.gz both present; first in sort order kept")
        # an excluded copy does not stand in the way of the other one
        stats, ledger = data_prep._new_stats(), []
        docs = list(data_prep.iter_markdown([folder], data_prep.MD_PATTERNS_DEFAULT, ["*.md"], False, stats, ledger))
        self.assertEqual([d[0] for d in docs], [])
        stats, ledger = data_prep._new_stats(), []
        docs = list(data_prep.iter_markdown([folder], ["*.md.gz"], [], False, stats, ledger))
        self.assertEqual([(d[0], d[2]) for d in docs],
                         [("a.md", "The compressed copy, edited later.\n"), ("only.md", "Compressed only.\n")])
        self.assertNotIn("duplicate_id", stats["skipped"])
        # in a build: the id is once in docs.tsv as a document and once as a skipped row
        out = os.path.join(folder, "scan")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", out, "--md-dir", folder, "--scan-only", "--threads", "2", "--workers", "1",
                               "--min-chars", "1"])
        rows = [line.split("\t") for line in read_file(os.path.join(out, "docs.tsv")).splitlines()[1:]]
        self.assertEqual(sorted((r[1], r[3], r[4][:12]) for r in rows if r[1] == "a.md"),
                         [("a.md", "skipped", "duplicate_id"), ("a.md", "train", "")])
        self.assertIn("duplicate_id 1", read_file(os.path.join(out, "report.md")))

    # ---------------- F071: the fast boilerplate test is the structural rule, for every line ----------------
    def boilerplate_test_lines(self):
        """Documents (lists of lines) that reach every branch of the rule."""
        para = "We fitted a linear mixed model to every trait and estimated the additive variance from the kinship."
        docs = []
        c = self.corpus()
        for folder in (c.md, c.notes):
            for name in sorted(os.listdir(folder))[:12]:
                docs.append(read_file(os.path.join(folder, name)).split("\n"))
        heads = ["## Materials and Methods", "-----", "MATERIALS AND METHODS", "$$ \\hat{\\beta} = (X^T X)^{-1} X^T y $$",
                 "\\begin{equation} E = mc^2", "x \\end{align}", "Materials and Methods", "Etymology",
                 "This item is not available for download.", "one two three four five six seven eight nine",
                 "one two three four five six seven eight", "Please sign in to access the full text",
                 "<!-- Start of picture text -->", "| a | b |", "**Methods**", "(Results)", "`code` sample", "> quoted",
                 "#hashtag", "1. Introduction", "ÉTUDE", "étude", "A B", "x", "tab\tseparated\twords here",
                 "nine  words   with    wide     gaps      a  b   c", "UPPER CASE WITH MORE THAN EIGHT WORDS IN THE LINE"]
        heads += ["Ends with " + ch for ch in ".!?;:,"] + ["Chap. 5 Diagonalization", "Category: Genetics"]
        for nxt in (para, para[:80], para[:79], "Category: Genetics", ""):
            for head in heads:
                docs.append([head, "", nxt])
                docs.append([head, nxt])
                docs.append(["  " + head + "  ", "", "", "   " + nxt])
        docs.append(heads)  # each followed by a short line, the last one by nothing
        return docs

    def test_boilerplate_candidate_is_the_structural_rule(self):
        checked = candidates = 0
        for lines in self.boilerplate_test_lines():
            for idx, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                expected = not data_prep._structural_line(stripped, data_prep._next_nonblank(lines, idx))
                self.assertEqual(data_prep._boilerplate_candidate(stripped, lines, idx), expected, (stripped, lines))
                checked += 1
                candidates += expected
        self.assertGreater(checked, 1000)
        self.assertTrue(200 < candidates < checked - 200, (checked, candidates))  # both answers, many times
        # pass 2 looks the line up first and applies the rule only to a line that is in the table
        para = "We fitted a linear mixed model to every trait and estimated the additive variance from the kinship."
        banner, heading = "Please sign in to access the full text", "Materials and Methods"
        text = "\n".join([heading, "", para, "", banner, "", "## " + banner, "", heading, "", "Short line."])
        table = {data_prep._line_key(banner): 30, data_prep._line_key(heading): 30,
                 data_prep._line_key("## " + banner): 30}
        for mode, kept in (("strip", [heading, "", para, "", "## " + banner, "", "Short line."]), ("report", None)):
            data_prep._init_pass2({"md": table}, mode, 20, 1, False, 1)
            out, removed, sig = data_prep._pass2_doc("md", text)
            self.assertEqual(removed, [data_prep._line_key(banner), data_prep._line_key(heading)], mode)
            self.assertEqual(out, "\n".join(kept) if kept else text)
            self.assertIsNone(sig)

    # ---------------- F078: the tokenizer sample does not parse documents none of whose windows is kept ----------------
    def test_tokenizer_sample_skips_documents_without_a_kept_window(self):
        out = os.path.join(self.tmp, "f078")
        real_lines, real_shares = data_prep.tokenizer_sample_lines, data_prep.tokenizer_shares
        called, seen = [], {}

        def lines(text, source, doc_id, seed, prob, *a, **k):
            called.append((source, doc_id))
            return real_lines(text, source, doc_id, seed, prob, *a, **k)

        def shares(train_chars, weights):
            seen.update(chars=dict(train_chars), share=real_shares(train_chars, weights))
            return seen["share"]

        sample = 6000
        with unittest.mock.patch.object(data_prep, "tokenizer_sample_lines", lines), \
                unittest.mock.patch.object(data_prep, "tokenizer_shares", shares), redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", out, "--keep-work", "--tokenizer-sample-chars", str(sample),
                               "--tokenizer-weights", "md=3"] + self.base_args())
        st = data_prep._load_state(os.path.join(out, "_work", data_prep.STATE_NAME))
        prob = {n: min(1.0, sample * seen["share"][n] / max(seen["chars"][n], 1)) for n in seen["chars"]}
        self.assertTrue(all(0 < p < 0.5 for p in prob.values()), prob)
        expected, train = [], 0
        with data_prep._open_spool(st["spool2"], "r") as src:
            for i, line in enumerate(src):
                rec, doc = json.loads(line), st["docs"][i]
                self.assertEqual((doc[0], doc[1], doc[3]), (rec["s"], rec["id"], len(rec["x"])),
                                 "what the shortcut reads from the document table is what the spool holds")
                if st["split_of"][i] != data_prep.SPLIT_TRAIN:
                    continue
                train += 1
                expected.extend(real_lines(rec["x"], rec["s"], rec["id"], st["args"]["seed"], prob[rec["s"]]))
        with open(os.path.join(out, "_work", "tokenizer_corpus.txt"), "rb") as f:
            corpus = f.read().decode("utf-8")
        self.assertTrue(expected)
        self.assertEqual(corpus, "".join(piece + "\n" for piece in expected), "the text every document would give")
        self.assertGreater(len(called), 0)
        self.assertLess(len(called), train, "documents without a kept window are not parsed")
        self.assertEqual(len(called), len(set(called)))

    # ---------------- F058: the split works on arrays per source; same split as the per-group loop ----------------
    @staticmethod
    def split_per_group(args, docs, roots, dropped, src_index, src_of, chars_of, sources, cross_source, saw_train,
                        saw_val, split_of, forced_train, kept_split, settled_val):
        """The split as it was written before F058: one list of members, one sum and one assignment per group."""
        code = {"train": data_prep.SPLIT_TRAIN, "val": data_prep.SPLIT_VAL}

        def settled(r):
            if saw_train is None:
                return None
            return "train" if saw_train[r] else ("val" if saw_val[r] else None)

        for name, k in src_index.items():
            idxs = np.flatnonzero((src_of == k) & ~dropped).tolist()
            groups = collections.defaultdict(list)
            for i in idxs:
                groups[int(roots[i])].append(i)
            total_chars = int(chars_of[idxs].sum()) if idxs else 0
            budget = min(args.val_fraction * total_chars, args.val_max_chars)
            order = sorted(groups, key=lambda g: data_prep.stable_unit(args.seed, "split", f"{docs[g][0]}:{docs[g][1]}"))
            val_chars = 0
            if saw_train is not None:
                fresh = []
                for g in order:
                    target = settled(g)
                    if target is None:
                        fresh.append(g)
                        continue
                    kept_split[target] += len(groups[g])
                    split_of[groups[g]] = code[target]
                    if target == "val":
                        settled_val.add(g)
                        val_chars += int(chars_of[groups[g]].sum())
                order = fresh
            for g in order:
                chars = int(chars_of[groups[g]].sum())
                target = "train"
                if len(groups) >= 2 and val_chars < budget:
                    if g in cross_source:
                        forced_train["cross_source_cluster"] += 1
                    elif chars > args.max_val_group_share * budget:
                        forced_train["group_too_large_for_val"] += 1
                    elif val_chars + chars <= budget * 1.1:
                        target = "val"
                        val_chars += chars
                split_of[groups[g]] = code[target]
            sources[name]["val_chars_target"] = int(budget)
            sources[name]["val_chars"] = int(val_chars)
            sources[name]["train_chars"] = int(total_chars - val_chars)

    def test_split_on_arrays_is_the_split_of_the_per_group_loop(self):
        """50,000 documents in five sources (one with a single document, one with every document dropped),
        near-duplicate clusters inside and across sources, dropped documents, with and without an earlier split:
        _split_by_group gives the split, the per-source characters and the counters of the per-group loop."""
        import argparse
        rng = np.random.default_rng(58)
        n = 50_000
        names = ["wiki", "web", "pmc", "one", "none"]
        src_index = {name: k for k, name in enumerate(names)}
        src_of = rng.integers(0, 3, n).astype(np.int16)
        src_of[777] = 3
        src_of[rng.choice(np.setdiff1d(np.arange(n), [777]), 40, replace=False)] = 4
        chars_of = rng.integers(500, 20_000, n).astype(np.int64)
        docs = [[names[src_of[i]], f"doc{i}/file{i}.md", "t", int(chars_of[i])] for i in range(n)]
        roots = np.arange(n, dtype=np.int32)
        for i in sorted(rng.choice(np.arange(1, n), 4_000, replace=False).tolist()):
            near = int(rng.integers(max(0, i - 50), i)) if i % 2 else int(rng.integers(0, i))
            roots[i] = roots[near]  # the lowest index of the cluster, as lsh_clusters gives it
        dropped = rng.random(n) < 0.03
        dropped[src_of == 4] = True
        members = collections.defaultdict(set)
        for i in range(n):
            members[int(roots[i])].add(int(src_of[i]))
        cross_source = {r for r, m in members.items() if len(m) > 1}
        self.assertGreater(len(cross_source), 500)
        saw_train, saw_val = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        seen = rng.random(n)
        for r in members:
            saw_train[r] = seen[r] < 0.2
            saw_val[r] = 0.15 < seen[r] < 0.3  # some roots saw both: training wins
        cases = [(None, None, 0.05, 10 ** 12, 0.25), (saw_train, saw_val, 0.05, 10 ** 12, 0.25),
                 (saw_train, saw_val, 0.2, 3_000_000, 0.003), (None, None, 0.2, 3_000_000, 0.003)]
        reasons = collections.Counter()
        for st, sv, fraction, cap, share in cases:
            args = argparse.Namespace(seed=1234, val_fraction=fraction, val_max_chars=cap, max_val_group_share=share)
            got = []
            for split in (self.split_per_group, data_prep._split_by_group):
                split_of = np.zeros(n, dtype=np.int8)
                sources = {name: {} for name in names}
                forced, kept, settled_val = collections.Counter(), collections.Counter(), set()
                split(args, docs, roots, dropped, src_index, src_of, chars_of, sources, cross_source, st, sv,
                      split_of, forced, kept, settled_val)
                got.append((split_of, sources, forced, kept, settled_val))
            old, new = got
            label = f"previous split {st is not None}, fraction {fraction}, cap {cap}"
            self.assertTrue(np.array_equal(old[0], new[0]), label)
            self.assertEqual(old[1:], new[1:], label)
            self.assertEqual(json.dumps(new[1]), json.dumps(old[1]), "plain int values, as the manifest needs")
            self.assertTrue(all(type(g) is int for g in new[4]), "roots as plain int, as the saved state holds them")
            self.assertTrue((old[0][dropped] == data_prep.SPLIT_UNSET).all())
            self.assertGreater(int((new[0] == data_prep.SPLIT_VAL).sum()), 100, label)
            reasons.update(new[2])
            self.assertEqual(bool(new[3]), st is not None)
            self.assertEqual(new[1]["one"]["val_chars"], 0, "a source with one group has no validation part")
            self.assertEqual(new[1]["none"], {"val_chars_target": 0, "val_chars": 0, "train_chars": 0})
        self.assertTrue(reasons["cross_source_cluster"] and reasons["group_too_large_for_val"], reasons)

    # ---------------- F059: pass 3 encodes the next batch while the last one is written ----------------
    def test_pass3_in_small_batches_writes_the_token_files_of_a_plain_encode(self):
        """Guard. With the batch limit at 3,000 characters the toy corpus gives batches of short documents
        before and after documents above the limit (the whole-book path): every token file holds BOS + the
        tokens of the document's parts + EOS, document after document in spool order, and the dataset is the
        one built with the limit as it is."""
        c = self.corpus()
        argv = ["--md-dir", f"notes={c.notes}", "--md-dir", c.md, "--wiki-dir", c.wiki, "--vocab-size", "600",
                "--val-fraction", "0.2", "--max-doc-chars", "4000", "--threads", "2", "--workers", "1"]
        out, usual = os.path.join(self.tmp, "f059"), os.path.join(self.tmp, "f059_usual")
        with unittest.mock.patch.object(data_prep, "PASS3_BATCH_CHARS", 3_000), redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", out, "--keep-work"] + argv)
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", usual] + argv)
        manifest = data_prep.read_manifest(out)
        self.assertEqual(manifest["fingerprint"], data_prep.read_manifest(usual)["fingerprint"])
        state = data_prep._load_state(os.path.join(out, "_work", data_prep.STATE_NAME))
        tok = data_prep.Tokenizer.from_file(os.path.join(out, "tokenizer.model"))
        dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
        expected = collections.defaultdict(list)
        books, batches, batch_chars, batch_before_book = 0, 0, 0, False
        with data_prep._open_spool(state["spool2"], "r") as spool:
            for i, line in enumerate(spool):
                if state["dropped"][i]:
                    continue
                rec = json.loads(line)
                if len(rec["x"]) > 3_000:
                    batch_before_book = batch_before_book or batches > 0
                    books, batches, batch_chars = books + 1, batches + bool(batch_chars), 0
                else:
                    batch_chars += len(rec["x"])
                    if batch_chars > 3_000:
                        batches, batch_chars = batches + 1, 0
                ids = [tok.bos_id]
                for part in tok.encode_batch(data_prep.split_long_text(rec["x"], 4_000)):
                    ids.extend(part)
                ids.append(tok.eos_id)
                key = (data_prep.SPLIT_NAMES[state["split_of"][i]], rec["s"])
                expected[key].append(np.asarray(ids, dtype=dtype).tobytes())
        self.assertGreaterEqual(books, 1, "one document above the limit is needed")
        self.assertGreaterEqual(batches, 5, "several batches are needed")
        self.assertTrue(batch_before_book, "a batch must be under way when the first long document comes")
        self.assertGreater(batch_chars, 0, "and one after the last long document")
        names = sorted(f"{split}.{source}.bin" for split, source in expected)
        self.assertEqual(names, sorted(n for n in os.listdir(out) if n.endswith(".bin")))
        for (split, source), chunks in expected.items():
            with open(os.path.join(out, f"{split}.{source}.bin"), "rb") as handle:
                self.assertEqual(handle.read(), b"".join(chunks), f"{split}.{source}.bin")

    # ---------------- F107: Tokenizer.encode without the list call; same ids as encode_batch ----------------
    def test_encode_gives_the_ids_of_encode_batch_in_both_modes(self):
        """Guard (it passes before and after): encode(t) == encode_batch([t])[0] for ordinary and edge texts,
        in lines mode and in plain mode, and lines-mode text still survives the round trip."""
        path = os.path.join(self.plain(), "tokenizer.model")
        texts = ["", " ", "   \t  ", "\n", "\n\n\n", " \n ", "one line", "two lines\nof text",
                 "\nstarts and ends with a line break\n", "a blank line\n\nin the middle", "Windows\r\nline\r\nends\r\n",
                 "  indented\n\tcode  with   spaces  ", "The allele frequency changes by genetic drift.",
                 "naïve café α-helix 中文 \U0001f9ec", "x" * 5_000, "tab\tseparated\tcells\n" * 40]
        for mode in ("lines", "plain"):
            tok = data_prep.Tokenizer.from_file(path, mode)
            for text in texts:
                ids = tok.encode(text)
                self.assertEqual(ids, tok.encode_batch([text])[0], (mode, text[:40]))
                self.assertIs(type(ids), list, (mode, text[:40]))
                self.assertTrue(all(type(i) is int for i in ids), (mode, text[:40]))
        tok = data_prep.Tokenizer.from_file(path, "lines")
        self.assertEqual(tok.encode("a\nb").count(tok.newline_id), 1)
        self.assertEqual(tok.encode("\n\n"), [tok.newline_id, tok.newline_id])
        self.assertEqual(tok.decode(tok.encode("two lines\nof text")), "two lines\nof text")

    # ---------------- F095: a long English text behind front matter in another language is kept ----------------
    def test_english_body_behind_a_foreign_front_page_is_english(self):
        french = "l'été dernier les élèves ont étudié la génétique à l'université près de la forêt. "
        english = ("The frequency of an allele in a population changes from one generation to the next, and in a small "
                   "population most of that change is due to chance. ")
        front = french * (13_500 // len(french) + 1)
        body = english * (10_000 // len(english) + 1)
        self.assertFalse(data_prep.looks_english(front[:8_000]), "the front page alone is not English")
        self.assertTrue(data_prep.looks_english(front + body), "the middle of the text is English prose")
        self.assertFalse(data_prep.looks_english(french * (20_000 // len(french) + 1)), "French from start to end")
        # the middle window gets neither the short-text rule nor the ASCII rule of the first window
        self.assertFalse(data_prep.looks_english(front + "1 2 3 4 5 6 7 8 9 0 " * 1_000), "a middle without words")
        self.assertFalse(data_prep.looks_english(front + "xq zv kj wp " * 1_500), "an ASCII middle without function words")
        self.assertTrue(data_prep.looks_english(english * 100))
        self.assertTrue(data_prep.looks_english("| a | b |\n|---|---|\n" * 1_000), "tables and code are kept as before")

    # ---------------- F096: two different texts without a word are not near-duplicates ----------------
    WORDLESS = ("!!! ??? ... " * 40, "--- *** ((( " * 40)

    def test_texts_without_words_get_their_own_signatures(self):
        data_prep._init_pass2({}, "off", 20, 1, True, 1234)
        sigs = [data_prep._pass2_doc("md", text)[2] for text in self.WORDLESS]
        self.assertTrue(all(s is not None and s.shape == sigs[0].shape for s in sigs))
        self.assertFalse(np.array_equal(sigs[0], sigs[1]), "different texts, different signatures")
        self.assertTrue(np.array_equal(sigs[0], data_prep._pass2_doc("md", self.WORDLESS[0])[2]), "same text, same one")
        self.assertTrue(np.array_equal(sigs[0], data_prep._pass2_doc("md", "  !!!   ???  ...   " * 40)[2]),
                        "blanks do not count, as for the exact duplicates")
        # a text with words keeps the signature of its shingles
        text = "Genetic drift changes the frequency of an allele by chance in every generation. " * 5
        hasher, minhasher = data_prep.ShingleHasher(), data_prep.MinHasher(128, 1234)
        self.assertTrue(np.array_equal(data_prep._pass2_doc("md", text)[2],
                                       minhasher.signature(hasher.shingles(text, 5))))

    def test_near_dup_drop_keeps_two_different_texts_without_words(self):
        folder = self.folder("f096_in")
        for name, text in zip(("marks.md", "rules.md"), self.WORDLESS):
            with open(os.path.join(folder, name), "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text + "\n")
        out = os.path.join(self.tmp, "f096")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", out, "--md-dir", f"sym={folder}", "--near-dup", "drop"] + self.base_args())
        rows = [line.split("\t") for line in read_file(os.path.join(out, "docs.tsv")).splitlines()]
        status = {row[1]: (row[3], row[4]) for row in rows if row[0] == "sym"}
        self.assertEqual(len(status), 2, status)
        self.assertTrue(all(s in ("train", "val") for s, _ in status.values()), status)
        self.assertNotIn("\tsym\t", read_file(os.path.join(out, "duplicates.tsv")))


if __name__ == "__main__":
    unittest.main()
