"""Staging tests of the 2026-11-10 audit, package H: run_pipeline.py, estimate_dataset.py, gpu_check.py,
tools/*.py and guide/make_figures.py. They move into test_tiny_gpt.py (TestOtherScripts) when the packages
are merged.

    python -m unittest test_a2_h
"""
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
sys.path.insert(0, os.path.join(HERE, "tools"))
import test_tiny_gpt as T  # noqa: E402


def fake_popen(scripts, started, killed):
    """A stand-in for subprocess.Popen: child k answers its wait() calls from scripts[k], where an exception
    class is raised and a number is the exit code (the last entry repeats)."""
    class Fake:
        def __init__(self, cmd, **kwargs):
            self.answers = list(scripts[len(started)])
            started.append(list(cmd))
            self.returncode = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def wait(self, timeout=None):
            answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            if isinstance(answer, type):
                raise answer()
            self.returncode = answer
            return answer

        def poll(self):
            return self.returncode

        def kill(self):
            killed.append(True)
    return Fake


class Audit2PipelineTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2h_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---------------- run_pipeline ----------------
    def test_run_waits_for_the_step_after_ctrl_c(self):
        """F194: Ctrl+C in run_pipeline.run does not kill the child (subprocess.call kills it 0.25 s later on
        macOS/Linux); run waits for it and returns 130 when the child itself exits 0, so no later step starts."""
        import run_pipeline
        started, killed = [], []
        fake = fake_popen([[KeyboardInterrupt, 0]], started, killed)
        with unittest.mock.patch.object(run_pipeline.subprocess, "Popen", fake), \
                redirect_stdout(io.StringIO()) as out:
            try:
                code = run_pipeline.run(["python", "child.py"])
            except KeyboardInterrupt:
                self.fail("the interrupt left run() while the child was still running")
        self.assertEqual((code, killed, started), (130, [], [["python", "child.py"]]))
        self.assertIn("Ctrl+C: waiting", out.getvalue())
        for want in (0, 3):  # no interrupt: the child's own code
            started.clear()
            fake = fake_popen([[want]], started, killed)
            with unittest.mock.patch.object(run_pipeline.subprocess, "Popen", fake), \
                    redirect_stdout(io.StringIO()) as out:
                self.assertEqual(run_pipeline.run(["python", "child.py"]), want)
            self.assertNotIn("Ctrl+C", out.getvalue())
        # an interrupted child that fails keeps its own code
        started.clear()
        fake = fake_popen([[KeyboardInterrupt, 3]], started, killed)
        with unittest.mock.patch.object(run_pipeline.subprocess, "Popen", fake), redirect_stdout(io.StringIO()):
            self.assertEqual(run_pipeline.run(["python", "child.py"]), 3)
        # continue: a build interrupted with Ctrl+C that still exits 0 does not go on to the training
        here = os.path.join(self.tmp, "f194")
        dataset = os.path.join(here, "datasets", "bio_all")
        os.makedirs(dataset, exist_ok=True)
        os.makedirs(os.path.join(here, "data"), exist_ok=True)
        open(os.path.join(dataset, "manifest.json"), "w").close()
        open(os.path.join(here, "R_best.pt"), "w").close()
        started.clear()
        fake = fake_popen([[KeyboardInterrupt, 0], [0]], started, killed)
        with unittest.mock.patch.multiple(run_pipeline, HERE=here, DATASET=dataset, DATA=os.path.join(here, "data")), \
                unittest.mock.patch.object(run_pipeline.subprocess, "Popen", fake), \
                unittest.mock.patch.object(run_pipeline, "gpu_busy", return_value=False), \
                redirect_stdout(io.StringIO()):
            try:
                code = run_pipeline.step_continue(1, "R", os.path.join(here, "new_ds"))
            except KeyboardInterrupt:
                self.fail("the interrupt left run() while the build was still running")
        self.assertEqual(code, 1)
        self.assertEqual([c[1] for c in started], ["data_prep.py"], "no training after an interrupted build")
        self.assertEqual(killed, [])

    def test_estimate_step_uses_the_built_datasets_tokenizer(self):
        """F191: `run_pipeline.py estimate` passes the built dataset as --tokenizer (the tokenizer that exists,
        not a new one at today's automatic vocabulary size); without a dataset it passes none."""
        import run_pipeline
        here = os.path.join(self.tmp, "f191")
        dataset = os.path.join(here, "datasets", "bio_all")
        os.makedirs(dataset, exist_ok=True)
        os.makedirs(os.path.join(here, "data"), exist_ok=True)
        calls = []

        def estimate():
            with unittest.mock.patch.multiple(run_pipeline, HERE=here, DATASET=dataset,
                                              DATA=os.path.join(here, "data")), \
                    unittest.mock.patch.object(run_pipeline, "run", side_effect=lambda cmd: calls.append(cmd) or 0), \
                    unittest.mock.patch.object(sys, "argv", ["run_pipeline.py", "estimate"]):
                self.assertEqual(run_pipeline.main(), 0)
            return calls[-1]

        self.assertNotIn("--tokenizer", estimate(), "no dataset yet: the estimate trains its own tokenizer")
        with open(os.path.join(dataset, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"prep_args": {"superbpe": True}, "tokenizer": {"vocab_size": 16384}}, f)
        cmd = estimate()
        self.assertEqual(cmd[1:4], ["estimate_dataset.py", "--tokenizer", dataset])
        with unittest.mock.patch.multiple(run_pipeline, HERE=here, DATASET=dataset, DATA=os.path.join(here, "data")):
            self.assertEqual(cmd[4:], run_pipeline.build_command()[4:])
            self.assertNotIn("--tokenizer", run_pipeline.build_command(), "the build command itself is unchanged")

    # ---------------- estimate_dataset ----------------
    def text_folder(self, name, n=8):
        folder = os.path.join(self.tmp, name)
        os.makedirs(folder, exist_ok=True)
        for i in range(n):
            with open(os.path.join(folder, f"doc_{i}.md"), "w", encoding="utf-8", newline="\n") as f:
                f.write(f"# Lecture note {i}\n\n" + T.paragraphs(900 + i, 30))
        return folder

    def run_estimate(self, argv):
        """estimate_dataset.main in this process; data_prep's worker globals and the process priority (which
        _init_worker lowers) are left as they were."""
        import data_prep as dp
        import estimate_dataset
        with unittest.mock.patch.multiple(dp, _W_KW=dp._W_KW, _W_CFG=dp._W_CFG, _lower_priority=lambda: None), \
                redirect_stdout(io.StringIO()) as out:
            estimate_dataset.main(argv)
        return out.getvalue()

    def test_estimate_with_a_tokenizer_sizes_models_for_its_vocabulary(self):
        """F191: with --tokenizer the model-size table is built for that tokenizer's vocabulary and one line
        names the vocabulary a rebuild would train; a tokenizer that cannot be read falls back to training."""
        import sentencepiece as spm
        import tiny_gpt
        folder = self.text_folder("f191_text")
        work = os.path.join(self.tmp, "f191_tok")
        os.makedirs(work, exist_ok=True)
        corpus = os.path.join(work, "corpus.txt")
        with open(corpus, "w", encoding="utf-8", newline="\n") as f:
            f.write(T.paragraphs(1, 60).replace("\n\n", "\n"))
        spm.SentencePieceTrainer.train(input=corpus, model_prefix=os.path.join(work, "tok"), model_type="bpe",
                                       vocab_size=120, hard_vocab_limit=False, user_defined_symbols=["\n", "\t"],
                                       unk_id=0, bos_id=1, eos_id=2, pad_id=-1, minloglevel=2)
        model = os.path.join(work, "tok.model")
        size = spm.SentencePieceProcessor(model_file=model).get_piece_size()
        self.assertNotEqual(size, 4096)
        argv = ["--md-dir", folder, "--tflops", "1", "--hours", "1"]
        with unittest.mock.patch.object(tiny_gpt, "shape_ladder", wraps=tiny_gpt.shape_ladder) as ladder:
            out = self.run_estimate(["--tokenizer", model] + argv)
        ladder.assert_called_once_with(size)
        self.assertIn("a rebuild from scratch would train a 4,096-piece tokenizer", out)
        self.assertNotIn("training a", out)
        # a dataset folder whose manifest names no readable tokenizer: not an exit, a trained tokenizer
        broken = os.path.join(self.tmp, "f191_broken")
        os.makedirs(broken, exist_ok=True)
        with open(os.path.join(broken, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"prep_args": {}}, f)
        with unittest.mock.patch.object(tiny_gpt, "shape_ladder", wraps=tiny_gpt.shape_ladder) as ladder:
            out = self.run_estimate(["--tokenizer", broken] + argv)
        ladder.assert_called_once_with(4096)
        self.assertIn("cannot be read", out)
        self.assertIn("training a 4,096-piece SentencePiece tokenizer on the kept sample", out)
        self.assertNotIn("a rebuild from scratch", out)


if __name__ == "__main__":
    unittest.main()
