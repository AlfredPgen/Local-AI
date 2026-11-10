"""Staged tests of audit 2, package G (finetune.py, posttrain.py, calibrate_probes.py, compare_models.py,
detect_text.py).

Fast, CPU, temporary folders, no network, no model downloads. To be folded into test_tiny_gpt.TestOtherScripts.
"""

import dataclasses
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiny_gpt  # noqa: E402
from test_tiny_gpt import CharTok, types_ns  # noqa: E402


def toy_model(seed=0, d_model=32, ctx=48, vocab=50):
    torch.manual_seed(seed)
    cfg = tiny_gpt.ModelConfig(vocab_size=vocab, ctx=ctx, d_model=d_model, n_layers=2, n_heads=4, n_kv_heads=2)
    return tiny_gpt.TinyGPT(cfg).eval()


class Audit2FinetuneTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_a2g_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def sentencepiece(cls):
        """A small SentencePiece tokenizer trained on the fine-tuning examples (made once per class)."""
        if getattr(cls, "_tok", None) is None:
            import data_prep
            import sentencepiece as spm
            corpus = os.path.join(cls.tmp, "spm_corpus.txt")
            with open(corpus, "w", encoding="utf-8", newline="\n") as out:
                for name in ("sft_examples.jsonl", "dpo_examples.jsonl"):
                    with open(os.path.join(HERE, "finetune_examples", name), encoding="utf-8") as f:
                        for line in f:
                            row = json.loads(line)
                            out.write("\n".join(v for v in row.values() if isinstance(v, str)) + "\n")
                out.write("### Question:\n### Answer:\n" * 20)
            prefix = os.path.join(cls.tmp, "spm")
            spm.SentencePieceTrainer.train(
                input=corpus, model_prefix=prefix, model_type="bpe", vocab_size=600, character_coverage=1.0,
                byte_fallback=True, unk_id=0, bos_id=1, eos_id=2, pad_id=-1, user_defined_symbols=["\n", "\t"],
                remove_extra_whitespaces=False, normalization_rule_name="identity", hard_vocab_limit=False,
                num_threads=1, minloglevel=2)
            cls._tok = data_prep.Tokenizer.from_file(prefix + ".model", "lines")
        return cls._tok

    def finetune_base(self, name):
        """A tiny pre-trained checkpoint with a real tokenizer, as finetune.py needs one."""
        tok = self.sentencepiece()
        return self.save_checkpoint(toy_model(5, ctx=64, vocab=tok.vocab_size), name,
                                    tokenizer={"proto": tok.proto, **tok.meta()})

    def save_checkpoint(self, model, name, **extra):
        path = os.path.join(self.tmp, name)
        torch.save({"format": tiny_gpt.CHECKPOINT_FORMAT, "format_version": tiny_gpt.FORMAT_VERSION,
                    "checkpoint_type": "inference", "model_config": dataclasses.asdict(model.cfg),
                    "model": model.state_dict(), "step": 7, **extra}, path)
        return path

    # ---- calibrate_probes.py ----------------------------------------------------------------------------
    def calibrate(self, tag, extra=(), low_ram=True, bad_sanity=None):
        """calibrate_probes.main() on three toy reference models 'org/a', 'org/b', 'org/c' (c the largest) with
        its own probe file and score cache. low_ram: free RAM is reported so that only c fails the 0.7 rule;
        bad_sanity: the model whose sanity loss is 9. Returns (probe file, its bytes before, exit message or
        None, printed text, calls of free_device_memory)."""
        import calibrate_probes as cp
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil is not installed")
        folder = os.path.join(self.tmp, tag)
        os.makedirs(folder)
        probes = os.path.join(folder, "probes.tsv")
        with open(probes, "w", encoding="utf-8", newline="\n") as f:
            f.write("# toy probes\n" + "".join(f"genetics\tunrated\tGene number {i} is made of\tDNA\tRNA|fat|salt\n"
                                               for i in range(6)))
        with open(probes, "rb") as f:
            before = f.read()
        models = {"org/a": toy_model(1, 16), "org/b": toy_model(2, 32), "org/c": toy_model(3, 64)}
        sizes = {r: sum(p.numel() for p in m.parameters()) for r, m in models.items()}
        self.assertLess(sizes["org/b"], sizes["org/c"])
        # float32 bytes of c exceed 0.7 x free, those of b do not
        free = 4 * (sizes["org/b"] + sizes["org/c"]) / 2 / 0.7 if low_ram else 2 ** 40
        names = {id(m): r for r, m in models.items()}
        threads = torch.get_num_threads()
        argv = ["calibrate_probes.py", "--probes", probes, "--no-record", "--threads", str(threads), "--models",
                *models, *extra]
        message = None
        try:
            with unittest.mock.patch.object(cp, "load_llama", lambda repo, ctx=512: (models[repo], CharTok(),
                                                                                    models[repo].cfg)), \
                    unittest.mock.patch.object(cp, "sanity_loss",
                                               lambda model, tok, device: 9.0 if names[id(model)] == bad_sanity
                                               else 2.0), \
                    unittest.mock.patch.object(cp, "check_packages", lambda: None), \
                    unittest.mock.patch.object(cp, "lower_priority", lambda: None), \
                    unittest.mock.patch.object(cp, "CACHE_DIR", os.path.join(folder, "cache")), \
                    unittest.mock.patch.object(psutil, "virtual_memory", lambda: types_ns(available=free)), \
                    unittest.mock.patch.object(tiny_gpt, "free_device_memory",
                                               wraps=tiny_gpt.free_device_memory) as freed, \
                    unittest.mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()) as out:
                try:
                    cp.main()
                except SystemExit as exc:
                    message = str(exc.code)
        finally:
            torch.set_num_threads(threads)
        return probes, before, message, out.getvalue(), freed.call_count

    def test_calibration_refuses_a_partial_reference_set(self):
        """F023: a requested reference model that was skipped (too little RAM here) stops the run before anything
        is written: the probe file keeps its bytes and no report appears; the message names the model."""
        probes, before, message, out, _ = self.calibrate("partial")
        self.assertIn("org/c: SKIPPED", out)
        self.assertIsNotNone(message, "a skipped model must end the run")
        self.assertIn("org/c", message)
        self.assertIn("--allow-partial", message)
        with open(probes, "rb") as f:
            self.assertEqual(f.read(), before, "labels were rewritten from a partial reference set")
        base = os.path.splitext(probes)[0]
        self.assertFalse(os.path.exists(base + "_calibration.md"))
        self.assertFalse(os.path.exists(base + "_calibration.csv"))
        # --report-only is refused in the same way: its report would replace the full set's
        probes, before, message, out, _ = self.calibrate("partial_report", ["--report-only"])
        self.assertIn("org/c", message or "")
        self.assertFalse(os.path.exists(os.path.splitext(probes)[0] + "_calibration.md"))

    def test_calibration_writes_labels_from_a_complete_or_allowed_set(self):
        """F023: every requested model scored -> labels rewritten; --allow-partial rewrites from the models that
        were scored and the report names the missing one."""
        probes, before, message, out, _ = self.calibrate("complete", low_ram=False)
        self.assertIsNone(message)
        with open(probes, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("unrated", text)
        self.assertEqual(len(tiny_gpt.read_probes(probes)), 6)
        with open(os.path.splitext(probes)[0] + "_calibration.md", encoding="utf-8") as f:
            self.assertNotIn("not scored", f.read())
        probes, before, message, out, _ = self.calibrate("allowed", ["--allow-partial"])
        self.assertIsNone(message)
        with open(probes, encoding="utf-8") as f:
            self.assertNotIn("unrated", f.read())
        with open(os.path.splitext(probes)[0] + "_calibration.md", encoding="utf-8") as f:
            report = f.read()
        self.assertIn("org/c", report)
        self.assertIn("not scored", report)
        with open(os.path.splitext(probes)[0] + "_calibration.csv", encoding="utf-8") as f:
            self.assertFalse(f.readline().startswith("#"), "no comment line in the CSV")

    def test_calibration_releases_a_model_that_fails_the_sanity_check(self):
        """F022 (with F023): a model skipped for its sanity loss is released like a scored one, and counts as
        skipped."""
        probes, before, message, out, freed = self.calibrate("sanity", low_ram=False, bad_sanity="org/b")
        self.assertIn("org/b: SKIPPED", out)
        self.assertEqual(freed, 3, "free_device_memory once per model, the skipped one included")
        self.assertIn("org/b", message or "")
        with open(probes, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_calibration_cache_per_device_and_float32_rope_tables(self):
        """F021: scores made in BF16 on a GPU and in FP32 on the CPU are cached apart, and the cast to bfloat16
        leaves the RoPE tables in float32 (model.to(dtype) alone rounds them)."""
        import calibrate_probes as cp
        model = toy_model(4)
        cp.to_bfloat16(model, model.cfg)
        self.assertEqual(model.token_embedding.weight.dtype, torch.bfloat16)
        self.assertEqual(model.rope.cos.dtype, torch.float32)
        self.assertEqual(model.rope.sin.dtype, torch.float32)
        self.assertEqual(model.rope.cos.shape[-2], model.cfg.ctx)
        cpu, cuda = (cp.cache_file("org/a", "0123", kind, False) for kind in ("cpu", "cuda"))
        self.assertNotEqual(cpu, cuda)
        self.assertEqual(os.path.dirname(cpu), cp.CACHE_DIR)
        self.assertNotEqual(cp.cache_file("org/a", "0123", "cpu", True), cpu)

    # ---- finetune.py ------------------------------------------------------------------------------------
    def finetune(self, folder, mode, base, name, epochs=2, extra=()):
        import finetune
        data = os.path.join(HERE, "finetune_examples", f"{mode}_examples.jsonl")
        torch.manual_seed(0)
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
            saved = finetune.main([mode, "--base", base, "--data", data, "--name", name, "--out-dir", folder,
                                   "--epochs", str(epochs), "--batch", "4", "--device", "cpu", "--max-new-tokens",
                                   "4", "--log-file", os.path.join(folder, "log.txt"), *extra])
        return saved, out.getvalue()

    def test_finetune_keeps_each_finished_epoch(self):
        """F163: an interrupted fine-tune leaves <name>_partial.pt with the finished epochs (the final name is
        never written unfinished); the same command then runs again without --overwrite, and a finished run
        leaves no partial file."""
        import finetune
        folder = os.path.join(self.tmp, "partial_ft")
        os.makedirs(folder)
        base = self.finetune_base("ft_base.pt")
        real = finetune.length_grouped_batches
        calls = []

        def interrupt_in_epoch_2(*a, **k):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real(*a, **k)
        with unittest.mock.patch.object(finetune, "length_grouped_batches", interrupt_in_epoch_2), \
                self.assertRaises(KeyboardInterrupt):
            self.finetune(folder, "sft", base, "X")
        partial = os.path.join(folder, "X_partial.pt")
        self.assertFalse(os.path.exists(os.path.join(folder, "X.pt")))
        self.assertTrue(os.path.isfile(partial), "the finished epoch must be on disk")
        _, _, _, obj = tiny_gpt.load_for_inference(partial, "cpu")
        self.assertEqual(obj["finetune"]["epochs_done"], 1)
        self.assertEqual(obj["finetune"]["epochs"], 2)
        self.assertEqual(len(obj["finetune"]["history"]), 1)
        saved, out = self.finetune(folder, "sft", base, "X")
        self.assertEqual(saved, os.path.join(folder, "X.pt"))
        self.assertIn("epoch 1 of 2", out)
        self.assertFalse(os.path.exists(partial), "a finished run removes its partial file")
        ck = torch.load(saved, map_location="cpu", weights_only=True)
        self.assertNotIn("epochs_done", ck["finetune"])
        self.assertEqual(len(ck["finetune"]["history"]), 2)

    def test_finetune_epoch_saves_do_not_move_training(self):
        """F163 (guard): the per-epoch save changes nothing in training: the final weights equal those of a run
        whose partial saves are skipped (the code before the change)."""
        import finetune
        base = self.finetune_base("ft_base_guard.pt")
        finals = []
        for skip in (False, True):
            folder = os.path.join(self.tmp, f"guard_ft_{int(skip)}")
            os.makedirs(folder)
            real = finetune.save

            def save(model, tok, cfg, base_obj, path, info, history):
                if skip and path.endswith("_partial.pt"):
                    return path
                return real(model, tok, cfg, base_obj, path, info, history)
            with unittest.mock.patch.object(finetune, "save", save):
                saved, _ = self.finetune(folder, "sft", base, "G")
            finals.append(torch.load(saved, map_location="cpu", weights_only=True)["model"])
        self.assertEqual(finals[0].keys(), finals[1].keys())
        for key in finals[0]:
            self.assertTrue(torch.equal(finals[0][key], finals[1][key]), key)

    def test_dpo_checkpoint_keeps_the_sft_record(self):
        """F159: a DPO checkpoint made from an SFT checkpoint carries the SFT stage's record as
        finetune['previous']; a fine-tune of a pre-trained checkpoint has no such key."""
        folder = os.path.join(self.tmp, "previous_ft")
        os.makedirs(folder)
        base = self.finetune_base("ft_base_previous.pt")
        sft, _ = self.finetune(folder, "sft", base, "P_sft", epochs=1)
        dpo, _ = self.finetune(folder, "dpo", sft, "P_dpo", epochs=1)
        first = torch.load(sft, map_location="cpu", weights_only=True)["finetune"]
        self.assertNotIn("previous", first)
        second = torch.load(dpo, map_location="cpu", weights_only=True)["finetune"]
        self.assertEqual(second["method"], "dpo")
        self.assertEqual(second["previous"]["method"], "sft")
        self.assertEqual(second["previous"]["history"], first["history"])
        self.assertEqual(os.path.normcase(second["previous"]["base"]), os.path.normcase(os.path.abspath(base)))

    # ---- posttrain.py -----------------------------------------------------------------------------------
    def posttrain_data(self, tag):
        folder = os.path.join(self.tmp, tag)
        os.makedirs(folder)
        for f in ("sft.jsonl", "dpo.jsonl", "eval_mcq.jsonl"):
            with open(os.path.join(folder, f), "w", encoding="utf-8") as handle:
                handle.write("{}\n")
        return folder

    def test_posttrain_reuse_sft(self):
        """F163: posttrain run --reuse-sft skips SFT when <name>_sft.pt is an SFT made from this --base, and DPO
        starts from it; with another --base it exits; without the flag SFT runs as before."""
        import posttrain
        data = self.posttrain_data("reuse_sft")
        out_dir = os.path.join(data, "out")
        os.makedirs(out_dir)
        base = self.save_checkpoint(toy_model(6), "pt_base.pt")
        other = self.save_checkpoint(toy_model(7), "pt_other.pt")
        sft = self.save_checkpoint(toy_model(8), os.path.join("reuse_sft", "out", "R_sft.pt"),
                                   finetune={"method": "sft", "base": os.path.abspath(base), "data": "sft.jsonl",
                                             "examples": 20, "epochs": 2}, created="2026-11-10T10:00:00+00:00")
        calls = []

        def fake_main(argv):
            calls.append(list(argv))
            return os.path.join(out_dir, argv[argv.index("--name") + 1] + ".pt")
        def fake_evaluate(checkpoints, args):
            calls.append(checkpoints)
        common = ["run", "--name", "R", "--out-dir", out_dir, "--data-dir", data, "--device", "cpu"]
        with unittest.mock.patch("finetune.main", fake_main), \
                unittest.mock.patch.object(posttrain, "evaluate", fake_evaluate), \
                redirect_stdout(io.StringIO()) as out:
            posttrain.main(common + ["--base", base, "--reuse-sft"])
            self.assertEqual([c[0] for c in calls[:-1]], ["dpo"], "SFT is not run again")
            self.assertEqual(calls[0][calls[0].index("--base") + 1], sft)
            self.assertEqual(calls[-1], [base, sft, os.path.join(out_dir, "R_dpo.pt")])
            self.assertIn("examples", out.getvalue())
            calls.clear()
            with self.assertRaises(SystemExit) as caught:
                posttrain.main(common + ["--base", other, "--reuse-sft"])
            self.assertIn("R_sft.pt", str(caught.exception.code))
            self.assertEqual(calls, [])
            posttrain.main(common + ["--base", base])
            self.assertEqual([c[0] for c in calls[:-1]], ["sft", "dpo"], "without --reuse-sft: as before")


if __name__ == "__main__":
    unittest.main()
