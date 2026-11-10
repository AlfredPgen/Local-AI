"""Staging tests of the 2026-11-10 audit, package A1 (tiny_gpt.py: logging, model, checkpoint files, energy
meter, evaluation, probes, generation). Target class in test_tiny_gpt.py: TestFixes.

    python -m unittest test_a2_a1
"""
import io
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiny_gpt  # noqa: E402
import test_tiny_gpt as base  # noqa: E402 - as a module: a TestCase class imported by name would run twice


def small_model(seed=0, vocab=50):
    return base.TestFixes.small_model(None, seed, vocab)


class Audit2TinyGptA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2_a1_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_tee_survives_a_failed_log_write(self):
        """F363: a log.txt write that fails (disk full, lost drive) must not raise out of print()."""
        class BrokenLog:
            closed = False

            def write(self, text):
                raise OSError(28, "No space left on device")

            def flush(self):
                raise OSError(28, "No space left on device")

        console = io.StringIO()
        tee = tiny_gpt.Tee(console, BrokenLog())
        self.assertEqual(tee.write("step 1\n"), len("step 1\n"))
        self.assertEqual(console.getvalue(), "step 1\n", "the console still gets the text")


if __name__ == "__main__":
    unittest.main()
