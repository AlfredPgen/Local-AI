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
from contextlib import redirect_stdout

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
