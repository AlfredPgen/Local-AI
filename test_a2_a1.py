"""Staging tests of the 2026-11-10 audit, package A1 (tiny_gpt.py: logging, model, checkpoint files, energy
meter, evaluation, probes, generation). Target class in test_tiny_gpt.py: TestFixes.

    python -m unittest test_a2_a1
"""
import dataclasses
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

    def test_read_probes_refuses_a_four_column_line(self):
        """F351: a long-form line without its distractor column is refused with its line number, not read as a
        short-form probe with shifted fields; trailing tabs are not columns."""
        path = os.path.join(self.tmp, "probes4.tsv")
        good = "genetics\teasy\tGenes are made of\tDNA\tsugar|fat|salt\t\t\nOld prompt\tanswer\tx|y\t\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write("# comment\n" + good)
        probes = tiny_gpt.read_probes(path)
        self.assertEqual([(p["category"], p["prompt"], p["answer"], p["distractors"]) for p in probes],
                         [("genetics", "Genes are made of", "DNA", ["sugar", "fat", "salt"]),
                          ("general", "Old prompt", "answer", ["x", "y"])])
        with open(path, "w", encoding="utf-8") as f:
            f.write("# comment\n" + good + "genetics\teasy\tGenes are made of\tDNA\n")
        with self.assertRaises(SystemExit) as ctx:
            tiny_gpt.read_probes(path)
        self.assertIn("probes4.tsv:4:", str(ctx.exception.code))

    def test_document_mask_is_restored_from_the_checkpoint(self):
        """F329: a model built from the checkpoint of a --doc-mask run masks across documents again, so
        evaluate() gives the value training reported; without the setting (or without a BOS id) it does not."""
        trained = small_model(seed=4)  # ctx 16, vocab 50
        tok = base.CharTok()  # bos_id 0

        def checkpoint(**extra):
            return {"format": tiny_gpt.CHECKPOINT_FORMAT, "format_version": tiny_gpt.FORMAT_VERSION,
                    "checkpoint_type": "inference", "model_config": dataclasses.asdict(trained.cfg),
                    "model": {k: v.clone() for k, v in trained.state_dict().items()}, "step": 7, **extra}

        rng = np.random.default_rng(2)
        tokens = rng.integers(1, 50, 200, dtype=np.int64)
        tokens[::7] = tok.bos_id  # several documents in every window
        evalset = tiny_gpt.EvalSet({"a": tokens}, 16, 64)
        cpu = torch.device("cpu")
        plain = tiny_gpt.evaluate(trained, evalset, cpu, None, 2)["loss"]
        trained.doc_sep_id = tok.bos_id  # as _train sets it for --doc-mask on
        masked = tiny_gpt.evaluate(trained, evalset, cpu, None, 2)["loss"]
        self.assertNotAlmostEqual(plain, masked, places=4)
        with unittest.mock.patch.object(tiny_gpt, "tokenizer_from_checkpoint", return_value=tok):
            for assign in (False, True):
                model = tiny_gpt.model_from_checkpoint(checkpoint(train_config={"doc_mask": True}), assign=assign)[0]
                self.assertEqual(model.doc_sep_id, tok.bos_id, f"assign={assign}")
                self.assertEqual(tiny_gpt.evaluate(model, evalset, cpu, None, 2)["loss"], masked)
                for extra in ({"train_config": {"doc_mask": False}}, {"train_config": {}}, {"train_config": None},
                              {}):
                    model = tiny_gpt.model_from_checkpoint(checkpoint(**extra), assign=assign)[0]
                    self.assertIsNone(model.doc_sep_id, extra)
                    self.assertEqual(tiny_gpt.evaluate(model, evalset, cpu, None, 2)["loss"], plain)
        no_bos = base.CharTok()
        no_bos.bos_id = -1
        model = small_model(seed=4)
        tiny_gpt.apply_doc_mask(model, {"train_config": {"doc_mask": True}}, no_bos)
        self.assertIsNone(model.doc_sep_id)
        tiny_gpt.apply_doc_mask(model, {"train_config": {"doc_mask": True}}, tok)
        self.assertEqual(model.doc_sep_id, 0)


if __name__ == "__main__":
    unittest.main()
