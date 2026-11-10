"""Staging tests of package D (benchmark.py) of the 2026-11 audit. Target class: TestBenchmarkSuite in
test_tiny_gpt.py (package T moves every test there and deletes this file)."""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class Audit2Benchmark(unittest.TestCase):
    """benchmark.py: offline only (no downloads, no sandbox, no real checkpoint)."""

    @classmethod
    def setUpClass(cls):
        import benchmark
        cls.b = benchmark
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2_bench_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_generated_code_gets_no_stdin(self):
        """F020: the jailed program reads /dev/null, not the rest of the wrapper script piped to `bash -s` (which
        holds the clean-up lines), and the unsafe local runner gives its child no stdin either."""
        b = self.b
        for tool, end in (("bwrap", "python3 -s prog.py </dev/null"), ("unshare", "\"$d\" </dev/null")):
            script = b.WSLSandbox("Ubuntu", tool).script("input()\n", 5)
            jail = [line for line in script.splitlines() if "prlimit" in line]
            self.assertEqual(len(jail), 1, script)
            self.assertTrue(jail[0].endswith(end), jail[0])
            self.assertTrue(script.endswith("rc=$?\nrm -rf \"$d\"\nexit $rc\n"), "the clean-up lines follow the jail")
        with unittest.mock.patch.object(subprocess, "run", side_effect=subprocess.run) as spy:
            res = b.LocalSubprocess().run("input()\n", timeout=5)
        self.assertIs(spy.call_args.kwargs.get("stdin"), subprocess.DEVNULL)
        self.assertEqual((res.passed, res.status), (False, "failed"))
        self.assertIn("EOFError", res.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
