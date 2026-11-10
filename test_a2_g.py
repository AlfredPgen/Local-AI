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

    def test_read_jsonl_names_the_line_that_is_not_an_object(self):
        """F158: a JSON line that is valid but not an object ends with the file:line message, not an
        AttributeError."""
        import finetune
        path = os.path.join(self.tmp, "not_objects.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write('{"prompt": "p", "response": "r"}\n["prompt", "response"]\n{"prompt": "q", "response": "s"}\n')
        with self.assertRaises(SystemExit) as caught:
            finetune.read_jsonl(path, ["prompt", "response"])
        self.assertIn(":2:", str(caught.exception.code))
        self.assertIn("JSON object", str(caught.exception.code))

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

    # ---- compare_models.py ------------------------------------------------------------------------------
    def test_bootstrap_in_blocks_draws_the_same_resamples(self):
        """F025 (guard): the bootstrap resamples in blocks of 250 and gives exactly the numbers of one
        reps x documents index matrix (the code before the change, copied here), also when reps is no multiple
        of 250."""
        import math
        import numpy as np
        import compare_models

        def whole_matrix(nll_a, nll_b, bytes_a, bytes_b, reps, seed):
            rng = np.random.default_rng(seed)
            n = len(nll_a)
            ln2 = math.log(2)
            point = nll_a.sum() / (ln2 * bytes_a.sum()) - nll_b.sum() / (ln2 * bytes_b.sum())
            idx = rng.integers(0, n, (reps, n))
            stats = nll_a[idx].sum(1) / (ln2 * bytes_a[idx].sum(1)) - nll_b[idx].sum(1) / (ln2 * bytes_b[idx].sum(1))
            low, high = np.percentile(stats, [2.5, 97.5])
            p = min(1.0, 2 * min((stats <= 0).mean(), (stats >= 0).mean()))
            return point, low, high, p
        data = np.random.default_rng(11)
        for n, reps, seed in ((777, 2001, 0), (33, 100, 5), (40, 250, 1), (5, 613, 2)):
            bytes_a = data.integers(200, 20_000, n).astype(float)
            bytes_b = bytes_a * data.uniform(0.98, 1.0, n)
            nll_a = bytes_a * data.uniform(0.8, 1.6, n)
            nll_b = nll_a * data.uniform(0.97, 1.04, n)
            self.assertEqual(compare_models.bootstrap(nll_a, nll_b, bytes_a, bytes_b, reps, seed),
                             whole_matrix(nll_a, nll_b, bytes_a, bytes_b, reps, seed), (n, reps))

    def test_compare_models_takes_the_dataset_of_either_model(self):
        """F284 (compare_models half): without --dataset the validation documents come from A's dataset, and
        from B's when A's checkpoint records none (an export folder made before exports kept it)."""
        import compare_models

        class Stop(Exception):
            pass
        a, b = (os.path.join(self.tmp, name) for name in ("cmp_A.pt", "cmp_B.pt"))
        for path in (a, b):
            open(path, "wb").close()
        entries = {a: None, b: {"path": "datasets/bio", "fingerprint": "f1"}}
        asked = []

        def resolve(obj, override=None):
            asked.append(obj)
            raise Stop
        for case in ("B's", "A's"):
            with unittest.mock.patch.object(tiny_gpt, "load_for_inference",
                                            lambda path, device, trust=False: (None, None, None,
                                                                               {"dataset": entries[path]})), \
                    unittest.mock.patch.object(tiny_gpt, "resolve_dataset", resolve), \
                    unittest.mock.patch.object(sys, "argv", ["compare_models.py", a, b, "--device", "cpu"]), \
                    self.assertRaises(Stop):
                compare_models.main()
            self.assertEqual(asked, [{"dataset": entries[b] if case == "B's" else entries[a]}], case)
            entries[a] = {"path": "datasets/other", "fingerprint": "f2"}
            asked.clear()

    def test_validation_documents_by_parity_are_disjoint(self):
        """F381 (compare_models half): validation_documents(parity=0) and (parity=1) take documents with an
        even / odd index within their source, so two tools never share one whatever number each asks for; a
        source without an odd document is left out; without parity the selection is exactly the old one."""
        import numpy as np
        import compare_models
        import data_prep
        sizes = {"big": 41, "mid": 7, "one": 1, "two": 2}
        doc = [0] + [1 + i % 40 for i in range(250)]  # a start token, then 250 characters of CharTok text
        arrays = {"val": {s: np.array(doc * n, dtype=np.uint16) for s, n in sizes.items()}}
        manifest = {"tokenizer": {"file": "tokenizer.model", "encode_mode": "lines"}}

        def picked(per_source, **kw):
            with unittest.mock.patch.object(data_prep, "open_token_files",
                                            lambda folder, verify=True: (manifest, arrays)), \
                    unittest.mock.patch.object(data_prep.Tokenizer, "from_file", lambda *a, **k: CharTok()):
                return [(s, k) for s, k, _ in compare_models.validation_documents("ds", per_source, 4000, **kw)]
        for per_source in (1, 4, 30, 352):  # the selection before the change, copied here
            old = [(s, int(k)) for s, n in sorted(sizes.items())
                   for k in np.unique(np.linspace(0, n - 1, min(per_source, n)).astype(int))]
            self.assertEqual(picked(per_source), old, per_source)
        for n_even, n_odd in ((352, 256), (5, 3), (3, 5), (1, 1)):
            even, odd = picked(n_even, parity=0), picked(n_odd, parity=1)
            self.assertFalse(set(even) & set(odd))
            self.assertTrue(all(k % 2 == 0 for _, k in even) and all(k % 2 == 1 for _, k in odd))
            self.assertEqual({s for s, _ in even}, set(sizes))
            self.assertEqual({s for s, _ in odd}, {"big", "mid", "two"}, "a source with one document has no odd one")
        self.assertEqual([k for s, k in picked(352, parity=0) if s == "big"], list(range(0, 41, 2)))
        self.assertEqual([k for s, k in picked(3, parity=1) if s == "big"], [1, 19, 39])
        layout = {}
        picked(4, parity=1, layout=layout)
        self.assertEqual(layout, {s: (n, 0) for s, n in sizes.items()})

    # ---- posttrain.py -----------------------------------------------------------------------------------
    def posttrain_data(self, tag):
        folder = os.path.join(self.tmp, tag)
        os.makedirs(folder)
        for f in ("sft.jsonl", "dpo.jsonl", "eval_mcq.jsonl"):
            with open(os.path.join(folder, f), "w", encoding="utf-8") as handle:
                handle.write("{}\n")
        return folder

    def test_dpo_pairs_never_reject_the_chosen_answer(self):
        """F181: a SciQ distractor or a MedMCQA option with the same text as the right answer is never the
        rejected answer of a preference pair; a question with no other option gives no pair."""
        import random
        import posttrain
        sciq_rows = [{"question": f"Which molecule stores genes, case {i}?", "correct_answer": "DNA",
                      "distractor1": "dna", "distractor2": " DNA ", "distractor3": "Dna", "support": ""}
                     for i in range(3)]
        sciq_rows += [{"question": f"Which organelle makes ATP, case {i}?", "correct_answer": "mitochondria",
                       "distractor1": "Mitochondria", "distractor2": "nucleus", "distractor3": "ribosome",
                       "support": "Mitochondria make ATP."} for i in range(20)]
        med = {"choice_type": "single", "subject_name": "Anatomy", "cop": 0, "exp": ""}
        med_rows = [dict(med, question="Which bone is the thigh bone?", opa="Femur", opb="Femur", opc="Femur",
                         opd="Femur")]
        med_rows += [dict(med, question=f"Which bone is the thigh bone, case {i}?", opa="Femur", opb="femur ",
                          opc="Tibia", opd="Ulna") for i in range(20)]
        streams = {"allenai/sciq": sciq_rows, "openlifescienceai/medmcqa": med_rows}
        with unittest.mock.patch.object(posttrain, "_stream", lambda name, config=None, split="train": streams[name]):
            s_sft, s_dpo = posttrain.sciq(100, random.Random(0))
            m_sft, m_dpo, _ = posttrain.medmcqa(0, 100, random.Random(0))
        self.assertEqual(len(s_sft), 23, "the question-answer pairs are not touched")
        self.assertEqual(len(s_dpo), 20, "no pair from a question whose distractors all equal the answer")
        for pair in s_dpo:
            self.assertNotEqual(pair["chosen"].lower(), pair["rejected"].lower(), pair)
        self.assertEqual({p["rejected"] for p in s_dpo}, {"Nucleus.", "Ribosome."})
        self.assertEqual(len(m_dpo), 20, "no pair from a question whose options are all the same")
        for pair in m_dpo:
            self.assertEqual(pair["chosen"], "A. Femur")
            self.assertNotEqual(pair["chosen"][3:].lower(), pair["rejected"][3:].lower(), pair)
        self.assertEqual({p["rejected"] for p in m_dpo}, {"C. Tibia", "D. Ulna"})

    def test_posttrain_eval_before_data_names_the_missing_file(self):
        """F178: `posttrain.py eval` without eval_mcq.jsonl ends with the 'run data first' message, not with a
        FileNotFoundError traceback."""
        import posttrain
        empty = os.path.join(self.tmp, "no_posttrain_data")
        os.makedirs(empty)
        with self.assertRaises(SystemExit) as caught:
            posttrain.main(["eval", os.path.join(self.tmp, "x.pt"), "--data-dir", empty, "--device", "cpu"])
        message = str(caught.exception.code)
        self.assertIn(os.path.join(empty, "eval_mcq.jsonl"), message)
        self.assertIn("posttrain.py data", message)

    def posttrain_run(self, tag, extra, busy=lambda out_dir=None: False, device="cpu"):
        """posttrain.main(['run', ...]) with finetune.main and evaluate replaced by recorders, the busy check by
        `busy`, and `device` as the device that is selected. Returns (calls, out_dir, exit message or None)."""
        import finetune
        import posttrain
        data = self.posttrain_data(tag)
        out_dir = os.path.join(data, "out")
        os.makedirs(out_dir)
        base = self.save_checkpoint(toy_model(6), tag + "_base.pt")
        calls, message = [], None

        def fake_main(argv):
            calls.append(list(argv))
            return os.path.join(out_dir, argv[argv.index("--name") + 1] + ".pt")
        with unittest.mock.patch.object(finetune, "main", fake_main), \
                unittest.mock.patch.object(finetune, "gpu_busy", busy), \
                unittest.mock.patch.object(tiny_gpt, "gpu_busy", lambda *dirs: False), \
                unittest.mock.patch.object(tiny_gpt, "select_device",
                                           lambda preference="auto": torch.device(device)), \
                unittest.mock.patch.object(posttrain, "evaluate", lambda checkpoints, args: None), \
                redirect_stdout(io.StringIO()):
            try:
                posttrain.main(["run", "--base", base, "--name", "R", "--out-dir", out_dir, "--data-dir", data,
                                *extra])
            except SystemExit as exc:
                message = str(exc.code)
        return calls, out_dir, message

    def test_posttrain_run_checks_the_out_dir_for_another_job(self):
        """F180: posttrain run asks finetune.gpu_busy(out_dir), so a training lock in --out-dir stops it before
        SFT starts (its own wrapper looked only beside the script)."""
        seen = []

        def busy(out_dir=None):
            seen.append(out_dir)
            return out_dir is not None and os.path.basename(out_dir) == "out"
        calls, out_dir, message = self.posttrain_run("busy_out_dir", ["--device", "cuda"], busy, device="cuda")
        self.assertEqual(seen, [out_dir])
        self.assertEqual(message, tiny_gpt.GPU_BUSY_MESSAGE)
        self.assertEqual(calls, [], "stopped before SFT")
        calls, _, message = self.posttrain_run("free_out_dir", ["--device", "cuda"], device="cuda")
        self.assertIsNone(message)
        self.assertEqual([c[0] for c in calls], ["sft", "dpo"])

    def test_posttrain_run_passes_the_seed_on(self):
        """F183 (with F182): posttrain run --seed reaches both fine-tunes (the default 0 is finetune's own), and
        eval, which has no use for a seed, no longer accepts one."""
        import posttrain
        calls, _, message = self.posttrain_run("seed_run", ["--device", "cpu", "--seed", "5"])
        self.assertIsNone(message)
        self.assertEqual([c[0] for c in calls], ["sft", "dpo"])
        for argv in calls:
            self.assertEqual(argv[argv.index("--seed") + 1], "5", argv)
        calls, _, _ = self.posttrain_run("seed_default", ["--device", "cpu"])
        for argv in calls:
            self.assertEqual(argv[argv.index("--seed") + 1], "0", argv)
        self.assertEqual(posttrain.parse_args(["data", "--seed", "3"]).seed, 3)
        with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            posttrain.parse_args(["eval", "x.pt", "--seed", "3"])
        self.assertIn("--seed", err.getvalue())

    def test_posttrain_busy_check_uses_the_selected_device(self):
        """F184: the busy check looks at the device that was selected, not at the --device text: with no usable
        GPU the default --device auto runs on the CPU and is not refused as 'GPU is busy'."""
        import finetune
        import posttrain
        data = self.posttrain_data("busy_selected")
        seen = []

        def evaluate_on(device, argv):
            with unittest.mock.patch.object(tiny_gpt, "select_device",
                                            lambda preference="auto": torch.device(device)), \
                    unittest.mock.patch.object(finetune, "gpu_busy", lambda out_dir=None: True), \
                    unittest.mock.patch.object(posttrain, "evaluate", lambda checkpoints, args: seen.append(device)):
                posttrain.main(["eval", "x.pt", "--data-dir", data, *argv])
        evaluate_on("cpu", [])
        self.assertEqual(seen, ["cpu"], "--device auto without a usable GPU is a CPU job")
        with self.assertRaises(SystemExit) as caught:
            evaluate_on("cuda", [])
        self.assertIn(tiny_gpt.GPU_BUSY_MESSAGE, str(caught.exception.code))
        evaluate_on("cuda", ["--force"])
        self.assertEqual(seen, ["cpu", "cuda"])
        busy = lambda out_dir=None: True  # noqa: E731
        calls, _, message = self.posttrain_run("busy_selected_run", [], busy)
        self.assertIsNone(message)
        self.assertEqual([c[0] for c in calls], ["sft", "dpo"])
        calls, _, message = self.posttrain_run("busy_selected_gpu", [], busy, device="cuda")
        self.assertEqual(message, tiny_gpt.GPU_BUSY_MESSAGE)
        self.assertEqual(calls, [])

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
