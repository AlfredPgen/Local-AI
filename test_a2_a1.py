"""Staging tests of the 2026-11-10 audit, package A1 (tiny_gpt.py: logging, model, checkpoint files, energy
meter, evaluation, probes, generation). Target class in test_tiny_gpt.py: TestFixes.

    python -m unittest test_a2_a1
"""
import inspect
import io
import math
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np
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

    def _evalset(self):
        rng = np.random.default_rng(1)
        return tiny_gpt.EvalSet({"a": rng.integers(0, 50, 200, dtype=np.int64),
                                 "b": rng.integers(0, 50, 90, dtype=np.int64)}, 16, 32)

    def test_evaluate_reports_nan_instead_of_crashing(self):
        """F307: non-finite logits give a nan loss; the calibration bins must not raise."""
        model = small_model(seed=3)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float("nan"))
        got = tiny_gpt.evaluate(model, self._evalset(), torch.device("cpu"), None, 2)
        self.assertTrue(math.isnan(got["loss"]))
        self.assertEqual(sum(got["calibration"]["count"]), got["tokens"])

    def test_evaluate_float64_only_through_to64(self):
        """F306: MPS has no float64, so evaluate() converts through to64 (CPU first on MPS) and nowhere else."""
        lines = [ln.strip() for ln in inspect.getsource(tiny_gpt.evaluate).splitlines() if ".double()" in ln]
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(lines[0].startswith("to64 = "), lines)
        self.assertTrue(lines[1].startswith("count += "), lines)

    @unittest.skipUnless(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
                         "needs an Apple-silicon GPU (MPS)")
    def test_evaluate_on_mps_matches_cpu(self):
        """F306: evaluate() runs on MPS and agrees with the CPU."""
        model, evalset = small_model(seed=3), self._evalset()
        cpu = tiny_gpt.evaluate(model, evalset, torch.device("cpu"), None, 2)
        mps = tiny_gpt.evaluate(model.to("mps"), evalset, torch.device("mps"), None, 2)
        for key in ("loss", "top1", "ece"):
            self.assertAlmostEqual(cpu[key], mps[key], delta=1e-4, msg=key)


if __name__ == "__main__":
    unittest.main()
