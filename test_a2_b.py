"""Staging tests of package B (data_prep.py) for the audit of 2026-11-10.

They move into test_tiny_gpt.py (TestPipeline for builds, TestFixes for unit tests) once every package
has landed; until then:
    python -m unittest -v test_a2_b
"""

import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout

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


if __name__ == "__main__":
    unittest.main()
