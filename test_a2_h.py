"""Staging tests of the 2026-11-10 audit, package H: run_pipeline.py, estimate_dataset.py, gpu_check.py,
tools/*.py and guide/make_figures.py. They move into test_tiny_gpt.py (TestOtherScripts) when the packages
are merged.

    python -m unittest test_a2_h
"""
import io
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


if __name__ == "__main__":
    unittest.main()
