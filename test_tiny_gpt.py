"""Smoke and integration tests for data_prep.py, tiny_gpt.py and view_pt.py.

Runs on CPU in temporary folders (nothing in the project folder is written):
    python -m unittest -v test_tiny_gpt.py
The integration tests that launch several training runs as child processes (90 % of the suite's time) run
only with TINYGPT_SLOW_TESTS=1 set; without it the fast tests take a few minutes on the CPU. Checks that need
the GPU also need TINYGPT_GPU_TESTS=1 (never while a training run occupies it).
"""

import dataclasses
import glob
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_prep  # noqa: E402
import tiny_gpt  # noqa: E402

PY = sys.executable
# the integration tests (child training processes, minutes each) are opt-in, so everyday edits get a fast suite;
# they launch real processes on purpose (Ctrl+C, the run lock and the fault injection below need them)
SLOW = unittest.skipUnless(os.environ.get("TINYGPT_SLOW_TESTS"),
                           "integration test: set TINYGPT_SLOW_TESTS=1 to run it (the full suite takes about 15-20 min)")
PATCHED_RUN = r'''
import gc
import signal
import sys

import torch

sys.path.insert(0, sys.argv[1])
import tiny_gpt  # noqa: E402

mode, n = sys.argv[2], int(sys.argv[3])
calls = {"next": 0, "clip": 0, "save": 0, "gen": 0}
real_next, real_clip = tiny_gpt.TokenBatcher.next, torch.nn.utils.clip_grad_norm_
real_save, real_gen = tiny_gpt.atomic_save, tiny_gpt.generate
before = {}


def next_batch(self):
    out = real_next(self)  # drawn first: a retry must rewind the sampler
    calls["next"] += 1
    if mode == "interrupt" and calls["next"] == n:
        raise KeyboardInterrupt
    if mode == "oom" and calls["next"] <= n:
        raise RuntimeError("CUDA out of memory (simulated)")
    if mode == "oom_at_step" and calls["next"] == n:  # once, in the middle of the run
        raise RuntimeError("CUDA out of memory (simulated)")
    return out


def clip(params, *a, **k):
    params = list(params)
    calls["clip"] += 1
    if mode == "sigint" and calls["clip"] == n:
        signal.raise_signal(signal.SIGINT)
    if mode == "nangrad" and calls["clip"] == n:  # one infinite gradient element in the n-th update
        params[0].grad.view(-1)[0] = float("inf")
        before.update({i: p.detach().clone() for i, p in enumerate(params)})
    if mode == "nangrad" and calls["clip"] == n + 1:
        changed = sum(int(not torch.equal(before[i], p.detach())) for i, p in enumerate(params))
        print(f"FAULT CHECK: {changed} parameters changed by the update with the non-finite gradient")
    if mode == "diverge" and calls["clip"] >= n:  # every update from the n-th on: the run has diverged
        params[0].grad.view(-1)[0] = float("inf")
    if mode == "gcscan" and calls["clip"] == n:  # the loaded checkpoint dicts must be gone by now
        alive = [o for o in gc.get_objects() if isinstance(o, dict) and "optimizer" in o and "model" in o]
        print(f"FAULT CHECK: {len(alive)} checkpoint dicts alive after setup")
    return real_clip(params, *a, **k)


def save(obj, path):
    if obj.get("checkpoint_type") == "training" and not path.endswith("_crash.pt"):
        calls["save"] += 1
        # savefail: the first n training-state saves fail; saveflaky: of the first n, two in every three fail
        if (mode == "savefail" and calls["save"] <= n) or (
                mode == "saveflaky" and calls["save"] <= n and calls["save"] % 3 != 0):
            raise OSError(28, "No space left on device (simulated)")
    return real_save(obj, path)


def gen(*a, **k):
    calls["gen"] += 1
    if mode == "interrupt_sample" and calls["gen"] == n:  # Ctrl+C right after the metrics line was printed
        raise KeyboardInterrupt
    return real_gen(*a, **k)


tiny_gpt.TokenBatcher.next = next_batch
# a fixed memory budget: the real one is 40% of free RAM, and on a busy machine the planner would switch
# on memory savings in some runs and not in others
tiny_gpt.device_memory_budget = lambda device: (8 * 2 ** 30, "8.0 GiB (test)")
torch.nn.utils.clip_grad_norm_ = clip
tiny_gpt.atomic_save = save
tiny_gpt.generate = gen
tiny_gpt.main(sys.argv[4:])
'''
BIO = ("Genetic drift changes allele frequencies in small populations. Natural selection acts on phenotypes, "
       "and heritability measures additive genetic variance. Proteins are built from amino acids; DNA is "
       "transcribed into RNA. Genome-wide association studies test variants for disease.")
OFF = ("The football season began in autumn. The striker scored twice and the stadium cheered. "
       "Tickets for the next match sold out within an hour.")


def run(args, cwd=HERE, timeout=900):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([PY] + args, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=env)
    return proc.returncode, proc.stdout + proc.stderr


def paragraphs(seed, n=40):
    rng = np.random.default_rng(seed)
    words = ("gene allele protein cell membrane enzyme variance selection drift mutation population heritability "
             "regression likelihood matrix eigenvalue probability cohort biobank pathway receptor ligand dose "
             "pharmacology histology tissue sequencing chromosome locus haplotype linkage").split()
    out = []
    for _ in range(n):
        out.append(" ".join(rng.choice(words, size=int(rng.integers(12, 30)))).capitalize() + ".")
    return "\n\n".join(out)


class Corpus:
    """A small but messy corpus exercising every cleaning / filtering path."""

    def __init__(self, root):
        self.md = os.path.join(root, "md")
        os.makedirs(self.md)
        boiler = "Status: generated by the production pipeline for internal review only"
        for i in range(24):
            text = f"# Topic {i}\n\n{boiler}\n\n{BIO}\n\n{paragraphs(i)}\n\n```python\ndef f(x):\n    return x ** 2\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
            with open(os.path.join(self.md, f"topic_{i:02d}.md"), "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        shutil.copy(os.path.join(self.md, "topic_03.md"), os.path.join(self.md, "copy_of_topic_03.md"))
        with open(os.path.join(self.md, "topic_05.md"), encoding="utf-8") as f:
            near = f.read().replace("Topic 5", "Topic five")
        with open(os.path.join(self.md, "near_topic_05.md"), "w", encoding="utf-8") as f:
            f.write(near + "\nOne extra closing sentence.\n")
        with open(os.path.join(self.md, "windows.md"), "wb") as f:
            f.write(b"\xef\xbb\xbf" + (BIO + "\r\n\r\n" + paragraphs(99)).replace("\n", "\r\n").encode())
        open(os.path.join(self.md, "empty.md"), "w").close()
        with open(os.path.join(self.md, "garbled.md"), "w", encoding="utf-8") as f:
            f.write(BIO + "\n\n" + "��� �� index ��\n" + paragraphs(7) + "\x0c\x01")
        with open(os.path.join(self.md, "long_book.md"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(paragraphs(1000 + k) for k in range(12)))
        with open(os.path.join(self.md, "narration.lt.md"), "w", encoding="utf-8") as f:
            f.write("Lietuviškas tekstas. " * 50)
        # Hugging Face save_to_disk folder standing in for Wikipedia
        from datasets import Dataset
        rows = {"id": [], "url": [], "title": [], "text": []}
        for i in range(60):
            bio = i % 3 != 0
            rows["id"].append(str(i))
            rows["url"].append(f"https://example.org/{i}")
            rows["title"].append(f"{'Genetics' if bio else 'Football'} article {i}")
            rows["text"].append((BIO + "\n\n" + paragraphs(200 + i, 6)) if bio else (OFF + "\n\n" + OFF))
        self.wiki = os.path.join(root, "wiki")
        Dataset.from_dict(rows).save_to_disk(self.wiki)
        # Parquet shard standing in for FineWeb-Edu
        import pyarrow as pa
        import pyarrow.parquet as pq
        n = 30
        table = pa.table({"id": [f"w{i}" for i in range(n)],
                          "text": [(BIO + " " + paragraphs(300 + i, 4)) if i % 2 else OFF * 3 for i in range(n)],
                          "int_score": [i % 5 for i in range(n)]})
        self.parquet = os.path.join(root, "web.parquet")
        pq.write_table(table, self.parquet)
        # gzipped JSON Lines standing in for a peS2o shard
        import gzip
        self.pes2o = os.path.join(root, "pes2o.json.gz")
        with gzip.open(self.pes2o, "wt", encoding="utf-8") as f:
            for i in range(40):
                text = (BIO + " " + paragraphs(700 + i, 5)) if i % 2 else (OFF + " " + OFF + " " + OFF)
                f.write(json.dumps({"id": f"p{i}", "text": text, "source": "s2orc"}) + "\n")
        # a second, named text source (e.g. converted slides)
        self.notes = os.path.join(root, "notes")
        os.makedirs(self.notes)
        for i in range(8):
            with open(os.path.join(self.notes, f"note_{i}.md"), "w", encoding="utf-8") as f:
                f.write(f"# Lecture note {i}\n\n" + paragraphs(500 + i, 12))


def make_office_files(folder):
    """Minimal .pptx / .xlsx / GWAS / .html fixtures without extra packages."""
    import gzip
    import zipfile
    a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    p = "http://schemas.openxmlformats.org/presentationml/2006/main"
    s = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(os.path.join(folder, "talk.pptx"), "w") as z:
        z.writestr("ppt/slides/slide1.xml", f'<p:sld xmlns:a="{a}" xmlns:p="{p}"><p:cSld><p:spTree><p:sp><p:nvSpPr>'
                   f'<p:cNvPr id="1" name="t"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody>'
                   f'<a:p><a:r><a:t>Genetic drift</a:t></a:r></a:p></p:txBody></p:sp><p:sp><p:txBody><a:p><a:r>'
                   f'<a:t>Chance changes allele frequencies</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>')
    with zipfile.ZipFile(os.path.join(folder, "traits.xlsx"), "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{s}" xmlns:r="{r}"><sheets><sheet name="h2" sheetId="1" '
                   f'r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/'
                   '2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/></Relationships>')
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{s}"><si><t>trait</t></si><si><t>height</t></si></sst>')
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{s}"><sheetData><row r="1"><c r="A1" t="s"><v>0</v>'
                   f'</c></row><row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>0.8</v></c></row></sheetData></worksheet>')
    with gzip.open(os.path.join(folder, "gwas.tsv.gz"), "wt", encoding="utf-8") as f:
        f.write("SNP\tCHR\tBP\tBETA\tP\n")
        for i in range(500):
            f.write(f"rs{i}\t{1 + i % 22}\t{1000 * i}\t0.1\t{1e-10 if i in (5, 6) else 0.5}\n")
    with open(os.path.join(folder, "page.html"), "w", encoding="utf-8") as f:
        f.write("<html><script>var x=1;</script><body><h1>Heritability</h1><p>Twin studies.</p></body></html>")
    open(os.path.join(folder, "figure.png"), "wb").close()


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_test_")
        cls.corpus = Corpus(cls.tmp)
        cls.ds = os.path.join(cls.tmp, "dataset")
        with redirect_stdout(io.StringIO()) as buf:
            data_prep.prepare(["--out", cls.ds, "--md-dir", cls.corpus.md, "--md-dir", f"notes={cls.corpus.notes}",
                               "--tokenizer-weights", "md=3", "--wiki-dir", cls.corpus.wiki,
                               "--jsonl", cls.corpus.pes2o, "--jsonl-name", "pes2o",
                               "--wiki-max-docs", "30", "--parquet", cls.corpus.parquet, "--parquet-name", "web",
                               "--min-score", "2", "--include-keywords", "genetic*,allele*,protein,DNA,RNA,genome*",
                               "--keyword-min-distinct", "2", "--vocab-size", "700", "--val-fraction", "0.2",
                               "--max-doc-chars", "4000", "--boilerplate-min-docs", "20", "--threads", "2"])
        cls.prep_log = buf.getvalue()
        cls.runs = os.path.join(cls.tmp, "runs")
        os.makedirs(cls.runs)
        cls.common = ["tiny_gpt.py", "--dataset", cls.ds, "--out-dir", cls.runs, "--log-file",
                      os.path.join(cls.runs, "log.txt"), "--device", "cpu", "--d-model", "64", "--layers", "2",
                      "--heads", "4", "--kv-heads", "2", "--ctx", "64", "--batch-size", "8", "--grad-accum", "1",
                      "--eval-every", "8", "--eval-tokens", "2048", "--save-every", "1000", "--sample-tokens", "8",
                      "--dropout", "0.1", "--tariff", "25"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---------------- data preparation ----------------
    def read_docs(self):
        with open(os.path.join(self.ds, "docs.tsv"), encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f]

    def test_manifest_and_files(self):
        manifest, arrays = data_prep.open_token_files(self.ds, verify=True)
        self.assertEqual(manifest["format"], "tiny_gpt_dataset")
        self.assertEqual(set(manifest["sources"]), {"md", "notes", "wiki", "web", "pes2o"})
        pes2o = manifest["sources"]["pes2o"]
        self.assertEqual(pes2o["kept_docs"], 20, "the 20 biology records pass the keyword filter")
        self.assertEqual(pes2o["skipped"].get("keyword_min_hits"), 20)
        self.assertEqual(manifest["tokenizer"]["sample_weights"]["md"], 3.0)
        self.assertEqual(manifest["sources"]["notes"]["kind"], "markdown")
        for split in ("train", "val"):
            for name, arr in arrays[split].items():
                self.assertEqual(len(arr), manifest["sources"][name][split]["tokens"])
        self.assertTrue(os.path.isfile(os.path.join(self.ds, "report.md")))
        self.assertFalse(os.path.exists(os.path.join(self.ds, "_work")), "work spool must be removed")
        # token distribution: printed, in report.md, and every piece counted in token_counts.tsv
        self.assertIn("Token use: ", self.prep_log)
        self.assertGreater(os.path.getsize(os.path.join(self.ds, "token_summary.png")), 20_000)
        report = open(os.path.join(self.ds, "report.md"), encoding="utf-8").read()
        self.assertIn("### Token use on the training data", report)
        with open(os.path.join(self.ds, "token_counts.tsv"), encoding="utf-8") as handle:
            rows = handle.read().splitlines()[1:]
        self.assertEqual(len(rows), manifest["tokenizer"]["vocab_size"])
        train_total = sum(len(arr) for arr in arrays["train"].values())
        self.assertEqual(sum(int(r.split("	")[3]) for r in rows), train_total)

    def test_cleaning_and_skips_are_recorded(self):
        docs = self.read_docs()
        by_id = {(d["source"], d["doc_id"]): d for d in docs}
        self.assertEqual(by_id[("md", "empty.md")]["reason"], "empty")
        self.assertEqual(by_id[("md", "narration.lt.md")]["reason"], "not_english")
        pair = [by_id[("md", "copy_of_topic_03.md")]["reason"], by_id[("md", "topic_03.md")]["reason"]]
        self.assertEqual(sorted(pair), ["", "exact_duplicate"], "exactly one copy kept, the later one skipped")
        self.assertIn(by_id[("md", "garbled.md")]["status"], ("train", "val"))
        m = json.load(open(os.path.join(self.ds, "manifest.json"), encoding="utf-8"))
        cleaning = m["sources"]["md"]["cleaning"]
        self.assertGreaterEqual(cleaning.get("replacement_lines_removed", 0), 1)
        self.assertGreaterEqual(cleaning.get("bom_removed", 0), 1)
        self.assertGreaterEqual(cleaning.get("newlines_normalised_docs", 0), 1)
        self.assertGreaterEqual(cleaning.get("oversized_docs_split", 0), 1)
        with open(os.path.join(self.ds, "boilerplate.tsv"), encoding="utf-8") as f:
            self.assertIn("generated by the production pipeline", f.read())

    def test_keyword_and_score_filters(self):
        m = json.load(open(os.path.join(self.ds, "manifest.json"), encoding="utf-8"))
        wiki = m["sources"]["wiki"]
        # 40 genetics articles pass the filter, 20 football ones do not; --wiki-max-docs 30 keeps the 30
        # best-ranked and records the other 10
        self.assertEqual(wiki["skipped"].get("keyword_min_hits", 0), 20)
        self.assertEqual(wiki["kept_docs"], 30)
        self.assertEqual(wiki["skipped"].get("over_max_docs_by_rank", 0), 10)
        web = m["sources"]["web"]
        self.assertGreater(web["skipped"].get("score_below_min", 0), 0)
        kept_titles = [d["title"] for d in self.read_docs() if d["source"] == "wiki" and d["status"] in ("train", "val")]
        self.assertTrue(all(t.startswith("Genetics") for t in kept_titles))

    def test_resume_after_pass2_gives_the_same_dataset(self):
        """A build that fails after pass 2 keeps _work/state.pkl and the spool; --resume
        continues from them and produces the same manifest (fingerprint) and docs.tsv
        as an uninterrupted build."""
        base = ["--md-dir", self.corpus.md, "--md-dir", f"notes={self.corpus.notes}", "--vocab-size", "600",
                "--val-fraction", "0.2", "--threads", "2"]
        plain = os.path.join(self.tmp, "resume_plain")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", plain] + base)
        broken = os.path.join(self.tmp, "resume_broken")
        real = data_prep._train_tokenizer

        def fail_once(*a, **k):
            data_prep._train_tokenizer = real
            raise OSError("disk full (simulated)")

        data_prep._train_tokenizer = fail_once
        try:
            with redirect_stdout(io.StringIO()) as buf, self.assertRaises(OSError):
                data_prep.prepare(["--out", broken] + base)
        finally:
            data_prep._train_tokenizer = real
        self.assertIn("--resume", buf.getvalue())
        work = os.path.join(broken, "_work")
        self.assertTrue(os.path.isfile(os.path.join(work, data_prep.STATE_NAME)), "state kept for --resume")
        self.assertTrue(glob.glob(os.path.join(work, "spool2.jsonl*")), "spool kept for --resume")
        self.assertFalse(data_prep._is_incomplete_build(broken), "a resumable build is never deleted")
        with self.assertRaises(SystemExit):  # without --resume the folder is refused with a hint, not rebuilt
            with redirect_stdout(io.StringIO()):
                data_prep.prepare(["--out", broken] + base)
        with redirect_stdout(io.StringIO()) as buf:
            data_prep.prepare(["--out", broken, "--resume", "--threads", "2"])
        self.assertIn("Resuming", buf.getvalue())
        a, b = data_prep.read_manifest(plain), data_prep.read_manifest(broken)
        self.assertEqual(a["fingerprint"], b["fingerprint"])
        self.assertEqual(a["totals"], b["totals"])
        self.assertIn("(resumed)", b["command"])
        with open(os.path.join(plain, "docs.tsv"), encoding="utf-8") as f1, \
                open(os.path.join(broken, "docs.tsv"), encoding="utf-8") as f2:
            self.assertEqual(f1.read(), f2.read())
        self.assertFalse(os.path.exists(work), "work files removed after the resumed build")
        # a failure in pass 3 (state 'tokenizer', tokenizer.model written); the file lost too: --resume still
        # produces a complete dataset, with the tokenizer file (the manifest names it)
        later = os.path.join(self.tmp, "resume_pass3")
        real_split, calls = data_prep.split_long_text, []

        def fail_in_pass3(text, max_chars):
            calls.append(1)
            if len(calls) == 3:
                raise OSError("disk full in pass 3 (simulated)")
            return real_split(text, max_chars)

        data_prep.split_long_text = fail_in_pass3
        try:
            with redirect_stdout(io.StringIO()), self.assertRaises(OSError):
                data_prep.prepare(["--out", later] + base)
        finally:
            data_prep.split_long_text = real_split
        self.assertEqual(data_prep._load_state(os.path.join(later, "_work", data_prep.STATE_NAME))["stage"],
                         "tokenizer")
        os.remove(os.path.join(later, "tokenizer.model"))
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", later, "--resume", "--threads", "2"])
        self.assertTrue(os.path.isfile(os.path.join(later, "tokenizer.model")))
        self.assertEqual(data_prep.read_manifest(later)["fingerprint"], a["fingerprint"])
        data_prep.open_token_files(later, verify=True)

    def test_split_by_group_no_leakage(self):
        groups = {}
        for d in self.read_docs():
            if d["status"] in ("train", "val"):
                groups.setdefault(d["group"], set()).add(d["status"])
        self.assertTrue(all(len(s) == 1 for s in groups.values()), "a group straddles train/val")
        near = [d for d in self.read_docs() if d["doc_id"] in ("topic_05.md", "near_topic_05.md")]
        self.assertEqual(len({d["group"] for d in near}), 1, "near duplicates must share a group")
        self.assertEqual(len({d["status"] for d in near}), 1)

    def test_tokenizer_roundtrip(self):
        manifest = data_prep.read_manifest(self.ds)
        tok = data_prep.Tokenizer.from_file(os.path.join(self.ds, "tokenizer.model"), "lines")
        self.assertEqual(tok.vocab_size, manifest["tokenizer"]["vocab_size"])
        self.assertGreaterEqual(tok.bos_id, 0)
        self.assertGreaterEqual(tok.eos_id, 0)
        for text in ("def f(x):\n    return x ** 2\n", "| a |    b |\n|---|---|", "Gene\n\nDrift  twice",
                     "h² = 0.5 ± 0.1; 家", "  leading spaces"):
            self.assertEqual(tok.decode(tok.encode(text)), text)
        reloaded = data_prep.Tokenizer(tok.proto, "lines")
        self.assertEqual(reloaded.encode("Genetic drift"), tok.encode("Genetic drift"))

    def test_incomplete_build_is_rebuilt_not_refused(self):
        out = os.path.join(self.tmp, "partial")
        os.makedirs(os.path.join(out, "_work"))
        open(os.path.join(out, "train.md.bin"), "wb").close()
        self.assertTrue(data_prep._is_incomplete_build(out))
        open(os.path.join(out, "my_notes.txt"), "w").close()
        self.assertFalse(data_prep._is_incomplete_build(out), "foreign files must never be deleted")
        os.remove(os.path.join(out, "my_notes.txt"))
        open(os.path.join(out, "report.md"), "w").close()
        self.assertFalse(data_prep._is_incomplete_build(out), "a scan-only report is kept")

    # ---------------- model ----------------
    def test_model_forward_backward_and_gqa_paths(self):
        cfg = tiny_gpt.ModelConfig(vocab_size=100, ctx=16, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2)
        torch.manual_seed(0)
        model = tiny_gpt.TinyGPT(cfg)
        x = torch.randint(0, 100, (3, 16))
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (3, 16, 100))
        logits.float().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        if tiny_gpt._enable_gqa_supported():
            model.eval()
            model.set_attention_impl("repeat_kv")
            a = model(x)
            model.set_attention_impl("gqa_native")
            b = model(x)
            self.assertTrue(torch.allclose(a, b, atol=1e-5), "GQA implementations must agree")
        w = model.blocks[0].attn.attention_weights(model.blocks[0].attn_norm(model.token_embedding(x)), model.rope)
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5))
        self.assertEqual(float(w[0, 0, 0, 1:].sum()), 0.0, "attention must be causal")
        total = model.param_counts()[0]
        self.assertEqual(total, tiny_gpt.count_params(cfg)[0])
        self.assertEqual(model.param_counts(), tiny_gpt.count_params(cfg), "dense: active equals total")

    def test_invalid_model_settings_fail_clearly(self):
        with self.assertRaises(SystemExit) as ctx:
            tiny_gpt.ModelConfig(vocab_size=100, d_model=30, n_heads=4)
        self.assertIn("not divisible", str(ctx.exception.code))
        with self.assertRaises(SystemExit):
            tiny_gpt.ModelConfig(vocab_size=100, d_model=32, n_heads=4, n_kv_heads=3)

    def test_wsd_schedule(self):
        s = tiny_gpt.LRSchedule("wsd", 100, 1e-3, 1e-4, 10, 0.2, "1-sqrt")
        self.assertAlmostEqual(s.lr_at(10), 1e-3)
        self.assertAlmostEqual(s.lr_at(5), 5e-4)
        self.assertAlmostEqual(s.lr_at(80), 1e-3)
        self.assertAlmostEqual(s.lr_at(100), 1e-4)
        self.assertLess(s.lr_at(90), 1e-3)
        self.assertEqual(s.phase(95), "decay")
        # a warmup longer than the run is clamped to total - 1, so the peak is still reached
        c = tiny_gpt.LRSchedule("wsd", 1000, 1e-3, 1e-4, 5000, 0.2, "1-sqrt")
        self.assertEqual(c.warmup, 999)
        self.assertAlmostEqual(c.lr_at(999), 1e-3)
        cos = tiny_gpt.LRSchedule("cosine", 100, 1e-3, 1e-4, 10)
        self.assertAlmostEqual(cos.lr_at(10), 1e-3)
        self.assertAlmostEqual(cos.lr_at(55), (1e-3 + 1e-4) / 2)
        self.assertAlmostEqual(cos.lr_at(100), 1e-4)
        self.assertAlmostEqual(cos.lr_at(200), 1e-4, msg="past the end the schedule stays at min_lr")
        self.assertAlmostEqual(tiny_gpt.LRSchedule("wsd", 100, 1e-3, 1e-4, 10, 0.0).lr_at(100), 1e-3,
                               msg="decay_frac 0: warmup-stable only, never decays")
        self.assertAlmostEqual(tiny_gpt.LRSchedule("wsd", 100, 1e-3, 0.0, 10, 0.2).lr_at(100), 0.0)

    def test_planner_respects_data_budget(self):
        manifest = {"tokenizer": {"vocab_size": 8192},
                    "sources": {"md": {"train": {"tokens": 16_000_000}, "doc_tokens_percentiles": {"90": 3000}},
                                "wiki": {"train": {"tokens": 51_000_000}, "doc_tokens_percentiles": {"90": 2000}}}}
        # a fixed memory budget: the real one is 40% of free RAM, so a busy machine would fail the test
        fixed_budget = (4 * 2 ** 30, "4.0 GiB (test)")
        for extra in ([], ["--steps", "15000"]):
            args = tiny_gpt.parse_args(["--name", "x"] + extra)
            with unittest.mock.patch.object(tiny_gpt, "device_memory_budget", return_value=fixed_budget):
                cfg, settings, lines = tiny_gpt.plan_run(args, manifest, torch.device("cpu"), None)
            n = tiny_gpt.count_params(cfg)[0]
            epochs = settings["steps"] * settings["tokens_per_step"] / 67e6
            self.assertLessEqual(epochs, 4.0 + 1e-6, f"planner exceeded the epoch cap with {extra}")
            self.assertLessEqual(n * 20, 4 * 67e6, f"model too large for the data with {extra}")
            if extra:
                self.assertEqual(settings["steps"], 15000)

    # ---------------- training, resume, generation, logs ----------------
    @SLOW
    def test_train_resume_generate_log_and_viewer(self):
        code, out = run(self.common + ["--name", "A", "--steps", "24", "--ema", "on"])  # the EMA is opt-in
        self.assertEqual(code, 0, out)
        code, out_b = run(self.common + ["--name", "B", "--steps", "24", "--stop-after-steps", "16", "--no-probes"])
        self.assertEqual(code, 0, out_b)
        self.assertIn("Stopped after 16 steps", out_b)
        code, out_r = run(["tiny_gpt.py", "--name", "B", "--out-dir", self.runs, "--log-file",
                           os.path.join(self.runs, "log.txt"), "--device", "cpu", "--resume", "--eval-every", "8",
                           "--eval-tokens", "2048", "--sample-tokens", "8", "--no-probes"])
        self.assertEqual(code, 0, out_r)
        self.assertIn("RESUMED", out_r)
        self.assertIn("val(mix)", out_r)
        # extending a run needs at least one step after its warmup
        warmup = torch.load(os.path.join(self.runs, "B.pt"), map_location="cpu",
                            weights_only=True)["train_config"]["warmup_steps"]
        code, out_w = run(["tiny_gpt.py", "--name", "B", "--out-dir", self.runs, "--log-file",
                           os.path.join(self.runs, "log.txt"), "--device", "cpu", "--resume", "--no-probes",
                           "--steps", str(warmup)])
        self.assertNotEqual(code, 0)
        self.assertIn("is not above this run's warmup", out_w)

        def metrics(text):
            # (train, val, measured-at step, status, val(ema) or None): the EMA is scored beside the raw weights
            # while the learning rate is above 5% of peak
            return {int(m.group(1)): (float(m.group(2)), float(m.group(3)), int(m.group(4)), m.group(6),
                                      float(m.group(5)) if m.group(5) else None) for m in
                    re.finditer(r"step +(\d+)/24 \| train ([\d.]+) \| val ([\d.]+) @(\d+)(?:.*? val\(ema\) ([\d.]+))?"
                                r".*?\| (NEW BEST.*|best .*)", text)}

        a, b = metrics(out), metrics(out_r)
        self.assertEqual(sorted(a), [8, 16, 24])
        for step, (tr, va, at, status, _ema) in a.items():
            self.assertEqual(at, step, "validation must be labelled with the step it was measured at")
        self.assertEqual(len({v[1] for v in a.values()}), 3, "validation must be re-measured at every evaluation")
        self.assertEqual(a[24][:2], b[24][:2], "resumed run must reproduce the uninterrupted run exactly")
        # NEW BEST exactly when validation (the better of the raw and the EMA weights) improved on every earlier value
        best = float("inf")
        for step in sorted(a):
            candidate = a[step][1] if a[step][4] is None else min(a[step][1], a[step][4])
            improved = candidate < best
            self.assertEqual(a[step][3].startswith("NEW BEST"), improved)
            self.assertEqual(a[step][3].endswith("(ema weights)"), improved and a[step][4] is not None
                             and a[step][4] < a[step][1])
            best = min(best, candidate)
        lines = out.splitlines()
        for i, line in enumerate(lines):
            if re.match(r"step +(8|16|24)/24 \| train", line):
                self.assertTrue(lines[i + 1].lstrip().startswith("sample @"), "sample must follow the metrics line")
        # log file mirrors the terminal, with header and footer
        log = open(os.path.join(self.runs, "log.txt"), encoding="utf-8").read()
        for needle in ("RUN ", "model: ", "train: ", "schedule: ", "step     24/24", "completed", "RESUMED"):
            self.assertIn(needle, log)
        # refuse to overwrite an existing run
        code, out_x = run(self.common + ["--name", "A", "--steps", "24"])
        self.assertNotEqual(code, 0)
        self.assertIn("already exists", out_x)
        self.assertIn("already exists", open(os.path.join(self.runs, "log.txt"), encoding="utf-8").read())
        # no training state saved yet (e.g. the PC shut down): --resume continues from the best checkpoint
        code, out_c = run(self.common + ["--name", "C", "--steps", "24", "--stop-after-steps", "8", "--no-probes"])
        self.assertEqual(code, 0, out_c)
        os.remove(os.path.join(self.runs, "C.pt"))
        code, out_c = run(["tiny_gpt.py", "--name", "C", "--out-dir", self.runs, "--log-file",
                           os.path.join(self.runs, "log.txt"), "--device", "cpu", "--resume", "--eval-every", "8",
                           "--eval-tokens", "2048", "--sample-tokens", "8", "--no-probes"])
        self.assertEqual(code, 0, out_c)
        self.assertIn("RESUMED FROM", out_c)
        self.assertIn("step     24/24", out_c)
        c_ck = torch.load(os.path.join(self.runs, "C.pt"), map_location="cpu", weights_only=True)
        self.assertEqual((c_ck["step"], c_ck["checkpoint_type"]), (24, "training"))
        # checkpoints: safe loading, types, tied storage
        train_ck = torch.load(os.path.join(self.runs, "A.pt"), map_location="cpu", weights_only=True)
        best_ck = torch.load(os.path.join(self.runs, "A_best.pt"), map_location="cpu", weights_only=True)
        self.assertEqual(train_ck["checkpoint_type"], "training")
        self.assertIn("optimizer", train_ck)
        self.assertEqual(best_ck["checkpoint_type"], "inference")
        self.assertNotIn("optimizer", best_ck)
        # new runs: QK-norm on; with --ema an EMA of the weights kept in the training state with its update count
        # (24 steps, an update every 10: 2), the best file says which it holds; without --ema no average
        self.assertTrue(train_ck["model_config"]["qk_norm"])
        self.assertEqual((train_ck["ema"]["decay"], train_ck["weights"], train_ck["ema"]["updates"]),
                         (0.999, "raw", 2))
        self.assertEqual(set(train_ck["ema"]["state"]), set(train_ck["model"]))
        self.assertNotIn("ema", torch.load(os.path.join(self.runs, "B.pt"), map_location="cpu", weights_only=True))
        self.assertIn(best_ck["weights"], ("raw", "ema"))
        self.assertNotIn("ema", best_ck, "the inference-only file carries one set of weights")
        self.assertEqual(train_ck["model"]["lm_head.weight"].data_ptr(),
                         train_ck["model"]["token_embedding.weight"].data_ptr())
        self.assertEqual(len(train_ck["metrics"]["history"]), 4, "baseline + 3 evaluations")
        with open(os.path.join(self.runs, "A_metrics.jsonl"), encoding="utf-8") as f:
            self.assertEqual(len(f.readlines()), 4)
        # generation from best and from an explicit checkpoint
        code, out_g = run(["tiny_gpt.py", "--name", "A", "--out-dir", self.runs, "--log-file",
                           os.path.join(self.runs, "log.txt"), "--device", "cpu", "--generate", "Genetic drift",
                           "--length", "12"])
        self.assertEqual(code, 0, out_g)
        self.assertIn("Genetic drift", out_g)
        # viewer on training and inference-only checkpoints
        for name, expect in (("A.pt", "optimizer: AdamW state"), ("A_best.pt", "none stored (inference-only")):
            png = os.path.join(self.runs, name + ".png")
            code, out_v = run(["view_pt.py", os.path.join(self.runs, name), "--plot", "--html", "--out", png,
                               "--dpi", "60"])
            self.assertEqual(code, 0, out_v)
            self.assertIn(expect, out_v)
            self.assertNotIn("panel failed", out_v)
            self.assertGreater(os.path.getsize(png), 50_000)
            self.assertTrue(os.path.isfile(os.path.splitext(png)[0] + "_embeddings.html"))
        # default dashboard name carries the step and a timestamp (never overwrites an earlier one)
        code, out_v = run(["view_pt.py", os.path.join(self.runs, "A_best.pt"), "--plot", "--dpi", "50", "--no-forward"])
        self.assertEqual(code, 0, out_v)
        self.assertTrue(os.path.isfile(os.path.join(self.runs, "A_best_dashboard.png")))
        # fact benchmark with per-category table and per-item CSV
        log = os.path.join(self.runs, "log.txt")
        code, out_b = run(["tiny_gpt.py", "--name", "A", "--out-dir", self.runs, "--log-file", log, "--device", "cpu",
                           "--benchmark"])
        self.assertEqual(code, 0, out_b)
        self.assertIn("overall:", out_b)
        self.assertIn("population genetics", out_b)
        self.assertTrue(glob.glob(os.path.join(self.runs, "A_best_benchmark_step24_*.csv")))
        # safetensors export: read back, generate from it, view it
        export = os.path.join(self.runs, "export")
        code, out_e = run(["tiny_gpt.py", "--name", "A", "--out-dir", self.runs, "--log-file", log, "--export", export])
        self.assertEqual(code, 0, out_e)
        self.assertIn("verified", out_e)
        tensors, meta = tiny_gpt.read_safetensors(os.path.join(export, "model.safetensors"))
        self.assertNotIn("lm_head.weight", tensors, "tied weight stored once")
        self.assertEqual(meta["format"], "tinyGPT")
        code, out_ge = run(["tiny_gpt.py", "--checkpoint", export, "--log-file", log, "--device", "cpu",
                            "--generate", "Genetic drift", "--length", "5"])
        self.assertEqual(code, 0, out_ge)
        code, out_ve = run(["view_pt.py", export, "--no-forward"])
        self.assertEqual(code, 0, out_ve)
        self.assertIn("format tinyGPT", out_ve)
        code, _ = run(["tiny_gpt.py", "--name", "A", "--out-dir", self.runs, "--log-file", log, "--export", export])
        self.assertNotEqual(code, 0, "export must refuse a non-empty folder")
        # bits per byte in the metrics line, and a permanent experiments table
        self.assertRegex(out, r"\| bpb \d+\.\d{3} \|")
        with open(os.path.join(self.runs, "results", "experiments.csv"), encoding="utf-8-sig") as f:
            table = f.read()
        self.assertIn("training run", table)
        self.assertIn("benchmark", table)
        # continued pre-training: a new dataset with the same tokenizer, then --init-from the first model, with a
        # time budget (which must also apply to --init-from runs)
        ds2 = os.path.join(self.tmp, "dataset_more")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", ds2, "--md-dir", self.corpus.md, "--tokenizer-from", self.ds,
                               "--keep-split-from", self.ds, "--val-fraction", "0.3", "--seed", "99"])
        before = {(d["source"], d["doc_id"]): d["status"] for d in self.read_docs() if d["status"] in ("train", "val")}
        with open(os.path.join(ds2, "docs.tsv"), encoding="utf-8") as f:
            head = f.readline().rstrip("\n").split("\t")
            after = [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f]
        shared = [d for d in after if (d["source"], d["doc_id"]) in before and d["status"] in ("train", "val")]
        self.assertGreater(len(shared), 10)
        for d in shared:  # a different seed would split differently: the earlier split must win
            self.assertEqual(d["status"], before[(d["source"], d["doc_id"])], d["doc_id"])
        code, out_i = run(["tiny_gpt.py", "--dataset", ds2, "--init-from", os.path.join(self.runs, "A_best.pt"),
                           "--name", "A_more", "--out-dir", self.runs, "--log-file", log, "--device", "cpu",
                           "--time-budget-hours", "0.01", "--batch-size", "8", "--grad-accum", "1",
                           "--eval-every", "4", "--eval-tokens", "1024", "--sample-tokens", "4", "--no-probes",
                           "--tariff", "25"])
        self.assertEqual(code, 0, out_i)
        self.assertIn("initialised from", out_i)
        self.assertIn("time budget 0.01 h", out_i)
        # energy and carbon: metrics line, run summary, table, and a checkpoint total that adds up across --resume
        self.assertRegex(out, r"\| \d+\.\d{2} kWh, [\d,]+ g CO2e, £\d+\.\d{2} \|")
        self.assertIn("whole run", out)
        self.assertIn("energy_kwh", table.splitlines()[0])
        self.assertIn("Earlier sessions of this run", out_r)
        b_ck = torch.load(os.path.join(self.runs, "B.pt"), map_location="cpu", weights_only=True)
        self.assertGreater(b_ck["energy"]["cpu"], 0)
        # paired document bootstrap between two checkpoints
        code, out_c = run(["compare_models.py", os.path.join(self.runs, "A_best.pt"), os.path.join(self.runs, "B.pt"),
                           "--docs", "4", "--reps", "200", "--device", "cpu"])
        self.assertEqual(code, 0, out_c)
        self.assertIn("Verdict:", out_c)
        self.assertTrue(glob.glob(os.path.join(self.runs, "compare_A_best_vs_B_*.csv")))
        # watermark: detectable with the key, absent without it
        import detect_text
        model, tok, cfg, _ = tiny_gpt.load_for_inference(os.path.join(self.runs, "A_best.pt"))
        marked = tiny_gpt.generate(model, tok, "Genetic drift", 120, torch.device("cpu"),
                                   generator=torch.Generator().manual_seed(1), watermark=("k", 0.25, 4.0))
        plain = tiny_gpt.generate(model, tok, "Genetic drift", 120, torch.device("cpu"),
                                  generator=torch.Generator().manual_seed(1))
        z_marked = detect_text.watermark_z(tok.encode(marked), "k", cfg.vocab_size, 0.25)[0]
        z_plain = detect_text.watermark_z(tok.encode(plain), "k", cfg.vocab_size, 0.25)[0]
        self.assertGreater(z_marked, 4, (z_marked, z_plain))
        self.assertLess(z_plain, 3, (z_marked, z_plain))
        code, out_d = run(["detect_text.py", "Genetic drift is a random process.", "--checkpoint",
                           os.path.join(self.runs, "A_best.pt"), "--watermark-key", "k"])
        self.assertEqual(code, 0, out_d)
        self.assertIn("likelihood score", out_d)
        # supervised fine-tuning, preference tuning, asking
        examples = os.path.join(HERE, "finetune_examples")
        code, out_s = run(["finetune.py", "sft", "--base", os.path.join(self.runs, "A_best.pt"), "--data",
                           os.path.join(examples, "sft_examples.jsonl"), "--name", "A_sft", "--out-dir", self.runs,
                           "--epochs", "1", "--batch", "4", "--device", "cpu", "--log-file", log])
        self.assertEqual(code, 0, out_s)
        self.assertIn("held-out loss", out_s)
        code, out_p = run(["finetune.py", "dpo", "--base", os.path.join(self.runs, "A_sft.pt"), "--data",
                           os.path.join(examples, "dpo_examples.jsonl"), "--name", "A_dpo", "--out-dir", self.runs,
                           "--epochs", "1", "--batch", "4", "--device", "cpu", "--log-file", log])
        self.assertEqual(code, 0, out_p)
        self.assertIn("reward_accuracy", out_p)
        code, out_a = run(["finetune.py", "ask", "What is an allele?", "--checkpoint",
                           os.path.join(self.runs, "A_dpo.pt"), "--device", "cpu", "--log-file", log,
                           "--max-new-tokens", "10"])
        self.assertEqual(code, 0, out_a)
        ck = torch.load(os.path.join(self.runs, "A_dpo.pt"), map_location="cpu", weights_only=True)
        self.assertEqual(ck["finetune"]["method"], "dpo")

    def test_keyword_syntax(self):
        kw = data_prep.KeywordFilter(["=MET", "*mab", "genom*", "Hardy-Weinberg"], [], min_hits=1, min_distinct=1)
        self.assertEqual(kw.evaluate("", "we met at noon")[1], "keyword_min_hits", "=MET must be case-sensitive")
        keep, _, _, terms = kw.evaluate("", "MET signalling, adalimumab, genomics, Hardy–Weinberg")
        self.assertTrue(keep)
        self.assertEqual(set(terms), {"=MET", "*mab", "genom*", "Hardy-Weinberg"})
        import collections
        import pyarrow as pa
        import pyarrow.compute as pc
        # overlapping terms each count (as in the Python check), in any separator, with case-sensitive wildcards
        kw = data_prep.KeywordFilter(["=MET", "*mab", "genom*", "Hardy-Weinberg", "genetic drift", "genetic*",
                                      "drift", "=BRCA*", "β-cell", "Crohn's disease"], [], min_hits=1)
        texts = ["we met", "MET and adalimumab", "genetic drift and genetic drifts", "BRCA1 and brca1",
                 "a β-cell here", "Crohn’s disease", "Hardy–Weinberg / genomics"]
        re2 = sum(np.asarray(pc.count_substring_regex(pa.array(texts), pattern=p, ignore_case=True))
                  for p in kw.include_patterns)
        python = [sum(kw._include.count(t, collections.Counter()).values()) for t in texts]
        self.assertEqual(python, [0, 2, 4, 1, 1, 1, 2])
        self.assertEqual(re2.tolist(), python, "the C++ prefilter must count what the Python filter counts")
        self.assertGreater(len(data_prep.read_terms(os.path.join(HERE, "keywords.txt") + ",extra term")), 500)

    def test_probe_formats_and_statistics(self):
        path = os.path.join(self.tmp, "probes.tsv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# comment\ngenetics\teasy\tGenes are made of\tDNA\tsugar|fat|salt\nOld prompt\tanswer\tx|y\n")
        probes = tiny_gpt.read_probes(path)
        self.assertEqual([p["category"] for p in probes], ["genetics", "general"])
        bench = tiny_gpt.read_probes(tiny_gpt.DEFAULT_PROBES)
        self.assertGreaterEqual(len(bench), 300)
        self.assertEqual(len({p["category"] for p in bench}), 16)
        self.assertTrue(all(len(p["distractors"]) == 3 for p in bench), "three near-miss distractors per item")
        self.assertTrue({p["difficulty"] for p in bench} <= {"easy", "medium", "hard"}, "labels come from calibration")
        low, high = tiny_gpt.wilson_interval(5, 10)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertEqual(tiny_gpt.fmt_dur(35_400), "9h 50m")
        self.assertEqual(tiny_gpt.fmt_dur(125), "2m 05s")

    def test_compat_dataset_names_follow_settings(self):
        a = tiny_gpt.parse_args(["--data", self.corpus.md, "--wiki-articles", "30000"])
        b = tiny_gpt.parse_args(["--data", self.corpus.md, "--wiki-articles", "50000"])
        self.assertNotEqual(tiny_gpt.compat_dataset_dir(a), tiny_gpt.compat_dataset_dir(b))
        self.assertIn("wiki50000", tiny_gpt.compat_dataset_dir(b))

    def test_converter(self):
        src, out = os.path.join(self.tmp, "raw"), os.path.join(self.tmp, "converted")
        os.makedirs(src)
        make_office_files(src)
        with open(os.path.join(src, "script.py"), "w", encoding="utf-8") as f:
            f.write("def heritability(va, vp):\n    return va / vp\n")
        import pymupdf
        pdf = pymupdf.open()
        pdf.new_page().insert_text((72, 72), "Heritability is the share of variance explained by genes.")
        pdf.save(os.path.join(src, "one_page.pdf"))
        pdf.close()
        pdf = pymupdf.open()  # a plot page with thousands of vector paths, then an ordinary page
        page = pdf.new_page()
        page.insert_text((72, 60), "Figure 1. Posterior draws of the heritability.")
        shape = page.new_shape()
        for i in range(12000):
            x, y = 72 + (i * 37) % 450, 100 + (i * 53) % 600
            shape.draw_line((x, y), (x + 1.5, y + 1.5))
        shape.finish(width=0.3)
        shape.commit()
        pdf.new_page().insert_text((72, 72), "Linkage disequilibrium decays with recombination.")
        pdf.save(os.path.join(src, "plot_pages.pdf"))
        pdf.close()
        import docx
        document = docx.Document()
        document.add_heading("Heritability", 1)
        document.add_paragraph("Twin studies compare MZ and DZ twins.")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text, table.cell(1, 0).text = "trait", "height"
        document.save(os.path.join(src, "notes.docx"))
        code, text = run(["convert_to_markdown.py", src, "--out", out, "--by-type", "--workers", "2"])
        self.assertEqual(code, 0, text)
        slides = open(os.path.join(out, "slides", "talk.md"), encoding="utf-8").read()
        self.assertIn("## Slide 1: Genetic drift", slides)
        self.assertIn("- Chance changes allele frequencies", slides)
        self.assertIn("| height | 0.8 |", open(os.path.join(out, "tables", "traits.md"), encoding="utf-8").read())
        gwas = open(os.path.join(out, "tables", "gwas.md"), encoding="utf-8").read()
        self.assertIn("500 genetic variants", gwas)
        self.assertIn("lambda GC", gwas)
        self.assertIn("2 variants reach genome-wide significance", gwas)
        docx_md = open(os.path.join(out, "articles", "notes.md"), encoding="utf-8").read()
        self.assertIn("# Heritability", docx_md)
        self.assertIn("| trait |", docx_md)
        html_md = open(os.path.join(out, "articles", "page.md"), encoding="utf-8").read()
        self.assertIn("Twin studies.", html_md)
        self.assertNotIn("var x", html_md)
        report = open(os.path.join(out, "conversion_report.tsv"), encoding="utf-8").read()
        self.assertIn("image: a text-only model cannot use pixels", report)
        self.assertTrue(os.path.isfile(os.path.join(out, "codes", "script.md")))
        self.assertTrue(os.path.isfile(os.path.join(out, "articles", "one_page.md")), "short PDFs are articles")
        plots = open(os.path.join(out, "articles", "plot_pages.md"), encoding="utf-8").read()
        self.assertIn("Posterior draws of the heritability", plots, "plot-heavy pages are read as plain text")
        self.assertIn("Linkage disequilibrium decays", plots)

    def test_llama_checkpoint_conversion_and_floor(self):
        """A Hugging Face Llama checkpoint (rotate-half RoPE layout) loads into TinyGPT
        with identical outputs; the shuffled-question floor is scored and deterministic."""
        import calibrate_probes
        cfg = tiny_gpt.ModelConfig(vocab_size=97, ctx=32, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
        torch.manual_seed(0)
        ref = tiny_gpt.TinyGPT(cfg).eval()
        sd = ref.state_dict()

        def to_hf(w, heads):  # the permutation Hugging Face's Llama conversion applies
            out, inp = w.shape
            return w.reshape(heads, out // heads // 2, 2, inp).transpose(1, 2).reshape(out, inp)
        raw = {"model.embed_tokens.weight": sd["token_embedding.weight"], "model.norm.weight": sd["norm.weight"]}
        for i in range(cfg.n_layers):
            b, h = f"blocks.{i}.", f"model.layers.{i}."
            raw[h + "self_attn.q_proj.weight"] = to_hf(sd[b + "attn.q_proj.weight"], cfg.n_heads)
            raw[h + "self_attn.k_proj.weight"] = to_hf(sd[b + "attn.k_proj.weight"], cfg.n_kv_heads)
            for src, dst in (("self_attn.v_proj", "attn.v_proj"), ("self_attn.o_proj", "attn.o_proj"),
                             ("mlp.gate_proj", "ffn.gate"), ("mlp.up_proj", "ffn.up"), ("mlp.down_proj", "ffn.down"),
                             ("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")):
                raw[h + src + ".weight"] = sd[b + dst + ".weight"]
        config = {"model_type": "llama", "vocab_size": 97, "hidden_size": 64, "num_hidden_layers": 2,
                  "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": cfg.ffn_hidden,
                  "rope_theta": cfg.rope_base, "rms_norm_eps": cfg.norm_eps, "tie_word_embeddings": True}
        model, _ = calibrate_probes.tinygpt_from_llama(config, raw, ctx=32)
        x = torch.randint(0, 97, (2, 20))
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x), ref(x), atol=1e-5), "converted model must match exactly")
        self.assertEqual(tiny_gpt.shuffled_prompt("a b c d e f", 1), tiny_gpt.shuffled_prompt("a b c d e f", 1))
        self.assertEqual(sorted(tiny_gpt.shuffled_prompt("a b c d e f", 2).split()), list("abcdef"))

    def test_data_prep_limits_ids_and_bytes(self):
        """--*-max-docs fills from documents actually kept, equal file names in two
        folders of one source stay distinct, document parts rejoin exactly, and
        byte_tables count exactly the bytes of the text (bits per byte)."""
        a, b = os.path.join(self.tmp, "fold_a"), os.path.join(self.tmp, "fold_b")
        for k, folder in enumerate((a, b)):
            os.makedirs(folder)
            with open(os.path.join(folder, "same.md"), "w", encoding="utf-8") as f:
                f.write(f"# Folder {k}\n\n" + BIO + "\n\n" + paragraphs(1200 + k, 8))
        out = os.path.join(self.tmp, "limits_dataset")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", out, "--md-dir", f"md={a}", "--md-dir", f"md={b}",
                               "--jsonl", self.corpus.pes2o, "--jsonl-name", "pes2o", "--jsonl-max-docs", "5",
                               "--parquet", self.corpus.parquet, "--parquet-name", "web", "--parquet-max-docs", "4",
                               "--include-keywords", "genetic*,allele*,protein,DNA,RNA,genome*",
                               "--keyword-sources", "PES2O,web", "--vocab-size", "600", "--val-fraction", "0.3"])
        m = data_prep.read_manifest(out)
        self.assertEqual(m["sources"]["pes2o"]["kept_docs"], 5)
        self.assertTrue(m["sources"]["pes2o"]["stopped_early"])
        self.assertEqual(m["sources"]["web"]["kept_docs"], 4)
        with open(os.path.join(out, "docs.tsv"), encoding="utf-8") as f:
            md_ids = sorted(line.split("\t")[1] for line in f if line.startswith("md\t"))
        self.assertEqual(md_ids, ["fold_a/same.md", "fold_b/same.md"], "ids use / on every system")
        tok = data_prep.load_tokenizer_from_any(self.ds)
        size, marked, line_break = tok.byte_tables()
        _, arrays = data_prep.open_token_files(self.ds)
        ids = np.concatenate([np.asarray(x, dtype=np.int64) for split in arrays.values() for x in split.values()])
        starts = np.nonzero(ids == tok.bos_id)[0].tolist() + [len(ids)]
        for s, e in zip(starts[:-1], starts[1:]):
            doc = ids[s:e]
            self.assertEqual(int(size[doc[1:]].sum() - (marked[doc[1:]] & line_break[doc[:-1]]).sum()),
                             len(tok.decode(doc).encode("utf-8")))
        text = "\n\n".join(paragraphs(1300 + k, 6) for k in range(6)) + "\nlast line"
        parts = data_prep.split_long_text(text, 1000)
        self.assertGreater(len(parts), 3)
        self.assertEqual(tok.decode([i for part in tok.encode_batch(parts) for i in part]), text)
        for extra, message in ((["--keyword-sources", "nosuch"], "unknown source"),
                               (["--md-dir", f"Books={a}", "--md-dir", f"books={b}"], "ignoring case")):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                data_prep.prepare(["--out", os.path.join(self.tmp, "refused"), "--md-dir", a] + extra)
            self.assertIn(message, str(ctx.exception.code))

    def test_arxiv_latex_to_markdown(self):
        """arXiv e-print -> Markdown: included files, figure captions, numbered
        citations, back matter dropped, e-mails masked; archive members outside
        the archive and local files named in the LaTeX are never read."""
        import tarfile
        import download_arxiv as da
        self.assertEqual(da.paper_group("cs.LG q-bio.PE"), "biology")
        self.assertEqual(da.paper_group("math.PR stat.TH"), "statistics")
        self.assertIsNone(da.paper_group("hep-th"))
        self.assertEqual(da.number_citations("A [@b; @c] then @d and [@b]; mail a@b.org"),
                         "A [1, 2] then [3] and [1]; mail a@b.org")
        oai = (b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords><record><header/><metadata>'
               b'<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>2101.00001</id><created>2021-01-01</created>'
               b'<title>Drift  and\n selection</title><categories>q-bio.PE</categories><abstract> A b. </abstract>'
               b'</arXiv></metadata></record><resumptionToken>tok</resumptionToken></ListRecords></OAI-PMH>')
        records, token, error = da.parse_oai_page(oai)
        self.assertEqual((records[0]["title"], records[0]["abstract"], token, error),
                         ("Drift and selection", "A b.", "tok", None))
        pandoc = da.pandoc_path()
        if not pandoc:
            self.skipTest("pandoc is not installed")
        secret = os.path.join(self.tmp, "local_secret.txt")
        with open(secret, "w", encoding="utf-8") as f:
            f.write("TOPSECRET")
        main = ("\\documentclass{article}\\begin{document}\\section{Introduction}\n"
                "Genetic drift changes allele frequencies \\citep{kimura1968,ohta1973}, as \\citet{wright1931} "
                "showed. Contact: someone@example.org.\n\\input{methods}\n\\input{../evil}\n"
                f"\\lstinputlisting{{{secret.replace(os.sep, '/')}}}\n"
                "\\begin{figure}\\includegraphics{f.png}\\caption{Allele frequency over time.}\\label{fig:a}"
                "\\end{figure}\n\\section{Acknowledgements}We thank the funders.\n\\end{document}\n")
        blob = io.BytesIO()
        with tarfile.open(fileobj=blob, mode="w:gz") as tar:
            for name, text in (("main.tex", main), ("methods.tex", "\\section{Methods}We use $N_e = 100$."),
                               ("../evil.tex", "EVIL")):
                data = text.encode("utf-8")
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        md, method = da.latex_markdown(blob.getvalue(), "Drift", "An abstract.", pandoc)
        self.assertEqual(method, "latex")
        self.assertTrue(md.startswith("# Drift\n\n## Abstract\n\nAn abstract."))
        self.assertIn("## Methods", md)
        self.assertIn("$N_e = 100$", md)
        self.assertIn("[1, 2]", md)
        self.assertIn("[3]", md)
        self.assertIn("Figure: Allele frequency over time.", md)
        self.assertIn("[email]", md)
        for absent in ("Acknowledgements", "funders", "EVIL", "TOPSECRET", "\\label", "@"):
            self.assertNotIn(absent, md)

    def test_superbpe_tokenizer(self):
        """--superbpe adds tokens spanning up to 4 words (never digits), text stays
        lossless and byte_tables still count exactly the bytes of the text."""
        out = os.path.join(self.tmp, "superbpe_dataset")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", out, "--md-dir", self.corpus.md, "--md-dir", f"notes={self.corpus.notes}",
                               "--vocab-size", "800", "--val-fraction", "0.2", "--superbpe",
                               "--superbpe-fraction", "0.2"])
        sb = data_prep.read_manifest(out)["tokenizer"]["superbpe"]
        self.assertGreater(sb["multiword_pieces"], 0)
        self.assertGreater(sb["token_saving"], 0)
        tok = data_prep.load_tokenizer_from_any(out)
        added = [tok.piece(i) for i in range(sb["stage1_pieces"], tok.vocab_size)]
        self.assertEqual(len(added), sb["added_pieces"])
        for piece in added:
            self.assertLessEqual(data_prep._piece_words(piece), 4, piece)
            self.assertFalse(any(ch.isdigit() for ch in piece), piece)
        size, marked, line_break = tok.byte_tables()
        _, arrays = data_prep.open_token_files(out)
        ids = np.concatenate([np.asarray(x, dtype=np.int64) for split in arrays.values() for x in split.values()])
        self.assertTrue((ids >= sb["stage1_pieces"]).any())  # the multi-word tokens are used
        starts = np.nonzero(ids == tok.bos_id)[0].tolist() + [len(ids)]
        for s, e in zip(starts[:-1], starts[1:]):
            doc = ids[s:e]
            self.assertEqual(int(size[doc[1:]].sum() - (marked[doc[1:]] & line_break[doc[:-1]]).sum()),
                             len(tok.decode(doc).encode("utf-8")))
        text = "Genetic drift is one of the forces of evolution.\n  Two  spaces, 12,345 alleles and\tone tab."
        self.assertEqual(tok.decode(tok.encode(text)), text)
        self.assertIn("SuperBPE:", open(os.path.join(out, "report.md"), encoding="utf-8").read())
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            data_prep.prepare(["--out", os.path.join(self.tmp, "refused_sb"), "--md-dir", self.corpus.md,
                               "--superbpe", "--tokenizer-from", out])

    def test_parallel_workers_give_identical_datasets(self):
        """Documents checked in worker processes (from the first document on) give
        exactly the same dataset as the single-process build."""
        base = ["data_prep.py", "--md-dir", self.corpus.md, "--wiki-dir", self.corpus.wiki, "--wiki-max-docs", "30",
                "--parquet", self.corpus.parquet, "--parquet-name", "web", "--min-score", "2",
                "--parquet-max-docs", "4", "--jsonl", self.corpus.pes2o, "--jsonl-name", "pes2o", "--jsonl-max-docs", "6",
                "--include-keywords", "genetic*,allele*,protein,DNA,RNA,genome*", "--vocab-size", "700",
                "--val-fraction", "0.2", "--max-doc-chars", "4000"]
        prints = []
        for workers in ("1", "2"):
            out = os.path.join(self.tmp, f"parallel_{workers}")
            code, text = run(base + ["--out", out, "--workers", workers, "--worker-warmup", "0"])
            self.assertEqual(code, 0, text)
            if workers == "2":
                self.assertIn("worker processes", text)
            prints.append(data_prep.read_manifest(out)["fingerprint"])
        self.assertEqual(prints[0], prints[1])

    def test_converter_edge_cases(self):
        import pymupdf
        import convert_to_markdown as conv
        src, out = os.path.join(self.tmp, "edge_src"), os.path.join(self.tmp, "edge_out")
        for sub in ("p1", "p2"):
            os.makedirs(os.path.join(src, sub))
        # 1. name clashes: same file name in two folders, and notes.md next to notes.html
        for sub in ("p1", "p2"):
            with open(os.path.join(src, sub, "media.md"), "w", encoding="utf-8") as f:
                f.write(f"# Media {sub}\n\nText from {sub}.\n")
        with open(os.path.join(src, "notes.md"), "w", encoding="utf-8") as f:
            f.write("My own notes about heritability.\n")
        with open(os.path.join(src, "notes.html"), "w", encoding="utf-8") as f:
            f.write("<table><tr><td>height</td><td>0.8</td></tr></table><p>Twin studies of heritability.</p>")
        # 2. prose .txt whose first line has commas: copied whole, not cut to a table
        prose = "\n".join(f"Line {i}: drift, selection, and mutation shape variation." for i in range(400))
        with open(os.path.join(src, "prose.txt"), "w", encoding="utf-8") as f:
            f.write(prose + "\n")
        # 3. GWAS files the fast reader rejects (ragged row, '-' missing value), a -log10 p column, spaces
        rows = ["SNP\tCHR\tBP\tP\tBETA\tA1"] + [f"rs{i}\t1\t{i * 1000}\t0.5\t0.01\tA" for i in range(1, 400)]
        rows += ["rs9001\t2\t5000000\t1e-20\t0.3\tG", "rs9002\t2\t5500000\t1e-12\t0.2\tT", "rs9003\t3\t100\t-\t0.1",
                 "rs9004\t5\t9000000\t1e-400\t0.5\tC"]
        with open(os.path.join(src, "ragged.tsv"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        with open(os.path.join(src, "neglog.tsv"), "w", encoding="utf-8") as f:
            f.write("CHR\tPOS\tLOG10P\n" + "\n".join(f"1\t{i * 10_000}\t0.30103" for i in range(1, 300))
                    + "\n4\t123456\t20\n")
        with open(os.path.join(src, "spaced.txt"), "w", encoding="utf-8") as f:
            f.write("CHR BP P\n" + "\n".join(f"1  {i * 100}  0.5" for i in range(1, 200)) + "\n7 777 3e-9\n")
        # 4. a notebook: markdown and code kept, outputs dropped
        with open(os.path.join(src, "analysis.ipynb"), "w", encoding="utf-8") as f:
            json.dump({"metadata": {"kernelspec": {"language": "python"}}, "cells": [
                {"cell_type": "markdown", "source": ["# Heritability\n", "Twin design."]},
                {"cell_type": "code", "source": "h2 = 2 * (0.8 - 0.5)",
                 "outputs": [{"data": {"image/png": "iVBORw0KGgo" * 50}}]}]}, f)
        code, text = run(["convert_to_markdown.py", src, "--out", out, "--recursive"])
        self.assertEqual(code, 0, text)
        made = sorted(os.listdir(out))
        self.assertIn("media.md", made)
        self.assertIn("media__p2.md", made, "both media.md inputs kept")
        self.assertIn("notes.md", made)
        self.assertIn("notes.html.md", made, "notes.html must not overwrite notes.md")
        self.assertEqual(open(os.path.join(out, "prose.md"), encoding="utf-8").read().strip(), prose)
        ragged = open(os.path.join(out, "ragged.md"), encoding="utf-8").read()
        self.assertIn("402 genetic variants", ragged)
        self.assertIn("3 variants reach genome-wide significance (p < 5e-8), forming 2 distinct loci", ragged,
                      "the two chr2 hits 0.5 Mb apart form one locus")
        self.assertIn("1.00e-20", ragged)
        self.assertIn("1.00e-400", ragged, "p-values below 1e-308 keep their exponent")
        self.assertIn("effect allele", ragged)
        neglog = open(os.path.join(out, "neglog.md"), encoding="utf-8").read()
        self.assertIn("lambda GC is 1.000", neglog)
        self.assertIn("1.00e-20", neglog)
        spaced = open(os.path.join(out, "spaced.md"), encoding="utf-8").read()
        self.assertIn("200 genetic variants", spaced)
        notebook = open(os.path.join(out, "analysis.md"), encoding="utf-8").read()
        self.assertIn("```python\nh2 = 2 * (0.8 - 0.5)\n```", notebook)
        self.assertNotIn("iVBOR", notebook)
        self.assertIn("height | 0.8", open(os.path.join(out, "notes.html.md"), encoding="utf-8").read())
        code, text = run(["convert_to_markdown.py", src, "--out", out, "--recursive"])
        self.assertIn("exists (use --overwrite)", open(os.path.join(out, "conversion_report.tsv"), encoding="utf-8").read())
        # 5. numbered labels at page edges survive; stepping page numbers and footers do not
        labels = [1, 1, 2, 3, 3, 4, 5, 5]  # a table can run over two pages: labels do not step like page numbers
        pages = [f"Supplementary Table {n}\nBody text {i} about linkage.\n\n{101 + i}" for i, n in enumerate(labels)]
        kept = "\n".join(conv.strip_page_furniture(pages))
        self.assertIn("Supplementary Table 5", kept)
        self.assertNotIn("\n105", "\n" + kept)
        # 6. a figure page with many separate shapes is read as plain text
        pdf = pymupdf.open()
        page = pdf.new_page()
        for i in range(800):
            page.draw_rect(pymupdf.Rect(20 + (i % 40) * 13, 40 + (i // 40) * 13, 30 + (i % 40) * 13,
                                        50 + (i // 40) * 13), fill=(0.2, 0.4, 0.8))
        self.assertTrue(conv._heavy_page(page))
        self.assertFalse(conv._heavy_page(pdf.new_page()))
        pdf.close()

    def test_pmc_jats_to_markdown(self):
        """PubMed Central XML becomes clean Markdown: sections, LaTeX equations,
        tables and captions kept; references, declarations and emails dropped."""
        import download_pmc
        xml = b"""<?xml version="1.0"?>
<pmc-articleset><article xml:lang="en" xmlns:mml="http://www.w3.org/1998/Math/MathML"
  xmlns:xlink="http://www.w3.org/1999/xlink">
<front><journal-meta><journal-title-group><journal-title>Genetics</journal-title></journal-title-group></journal-meta>
<article-meta><article-id pub-id-type="pmcid">PMC123</article-id>
<title-group><article-title>Drift in <italic>small</italic> populations</article-title></title-group>
<pub-date><year>2020</year></pub-date>
<permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/"><license-p>CC</license-p></license></permissions>
<abstract><p>Allele frequencies change by chance.</p></abstract></article-meta></front>
<body><sec><title>Introduction</title>
<p>Genetic drift was described by Wright<sup><xref ref-type="bibr" rid="b1">1</xref></sup>. The variance is
<inline-formula><mml:math><mml:msub><mml:mi>p</mml:mi><mml:mi>t</mml:mi></mml:msub></mml:math></inline-formula>, see below.
<disp-formula><alternatives><tex-math>\\documentclass{minimal}\\begin{document}}{}$$V = pq/2N$$\\end{document}</tex-math>
</alternatives></disp-formula> Contact: someone@example.org</p>
<list list-type="bullet"><list-item><p>small N</p></list-item><list-item><p>no selection</p></list-item></list>
<table-wrap><label>Table 1</label><caption><p>Effective sizes.</p></caption><table><thead><tr><th>Species</th><th>Ne</th></tr></thead>
<tbody><tr><td>Drosophila</td><td>10<sup>6</sup></td></tr></tbody></table></table-wrap>
<fig><label>Figure 1</label><caption><title>Fixation.</title><p>Most alleles are lost.</p></caption></fig>
<sec><title>A model</title><p>We model <mml:math><mml:mfrac><mml:mn>1</mml:mn><mml:mrow><mml:mn>2</mml:mn><mml:mi>N</mml:mi></mml:mrow></mml:mfrac></mml:math> per generation.</p></sec></sec>
<sec><title>Competing interests</title><p>None declared.</p></sec></body>
<back><ref-list><ref><mixed-citation>Wright S (1931) Evolution in Mendelian populations.</mixed-citation></ref></ref-list></back>
</article></pmc-articleset>"""
        import xml.etree.ElementTree as ET
        article = ET.fromstring(xml).find("article")
        self.assertEqual(download_pmc.article_pmcid(article), "PMC123")
        info, md = download_pmc.Jats(article).markdown()
        self.assertEqual((info["title"], info["journal"], info["year"], info["licence"]),
                         ("Drift in small populations", "Genetics", "2020", "CC BY 4.0"))
        for expected in ("# Drift in small populations", "## Abstract", "## Introduction", "### A model",
                         "described by Wright [1].", "$p_{t}$, see below.", "$$V = pq/2N$$", "- small N",
                         "Table 1. Effective sizes.", "| Species | Ne |", "| Drosophila | 10^6 |",
                         "Figure 1. Fixation. Most alleles are lost.", "$\\frac{1}{2N}$", "Contact: [email]"):
            self.assertIn(expected, md)
        for dropped in ("Competing interests", "None declared", "Mendelian populations", "someone@", "documentclass"):
            self.assertNotIn(dropped, md)

    def test_calibration_label_writer(self):
        import calibrate_probes
        path = os.path.join(self.tmp, "labels.tsv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# comment\ngenetics\tunrated\tGenes are made of\tDNA\tRNA|fat|salt\n"
                    "Cells are made of\tcells\tatoms|stars|rocks\n")
        calibrate_probes.rewrite_difficulties(path, ["easy", "hard"])
        probes = tiny_gpt.read_probes(path)
        self.assertEqual([(p["category"], p["difficulty"]) for p in probes], [("genetics", "easy"), ("general", "hard")])
        self.assertTrue(open(path, encoding="utf-8").read().startswith("# comment"))

    def test_page_furniture_is_stripped(self):
        import convert_to_markdown as conv
        pages = []
        for i in range(12):
            top = ["This electronic edition is for non-commercial purposes only.", f"{i + 1}",
                   "2. SINGLE-PARAMETER MODELS" if i % 2 else "2.1. ESTIMATING A PROBABILITY"]
            if i in (1, 5, 9):
                top.insert(0, "where")  # a real word that happens to start a page
            body = [f"Body text of page {i} about heritability and genetic drift." for _ in range(8)]
            bottom = [f"Nature | Vol 631 | 18 July 2024 | **{580 + i}**" if i % 2 else f"{580 + i} | Nature | Vol 631 | 18 July 2024"]
            if i in (0, 4, 8):
                bottom.append(f"Chapter {i // 4 + 1}")
            pages.append("\n".join(top + body + bottom))
        lines = "\n".join(conv.strip_page_furniture(pages)).splitlines()
        text = "\n".join(lines)
        for junk in ("non-commercial", "Nature | Vol 631", "SINGLE-PARAMETER MODELS", "ESTIMATING A PROBABILITY"):
            self.assertNotIn(junk, text)
        self.assertFalse([l for l in lines if l.strip().isdigit()], "page numbers are removed")
        self.assertEqual(lines.count("where"), 3, "ordinary words at a page edge are kept")
        for kept in ("Chapter 1", "Chapter 2", "Chapter 3", "Body text of page 7 about heritability and genetic drift."):
            self.assertIn(kept, lines)

    def test_text_root_and_language(self):
        root = os.path.join(self.tmp, "training_root")
        for folder in ("books", "articles", "wikipedia", "_private"):
            os.makedirs(os.path.join(root, folder))
        for folder in ("books", "articles", "_private"):
            for i in range(3):
                with open(os.path.join(root, folder, f"{folder}_{i}.md"), "w", encoding="utf-8") as f:
                    f.write(f"# {folder} {i}\n\n" + BIO + "\n\n" + paragraphs(900 + i, 10))
        out = os.path.join(self.tmp, "root_dataset")
        with redirect_stdout(io.StringIO()):
            data_prep.prepare(["--out", out, "--text-root", root, "--vocab-size", "600", "--val-fraction", "0.3"])
        self.assertEqual(set(data_prep.read_manifest(out)["sources"]), {"books", "articles"})
        self.assertTrue(data_prep.looks_english(BIO * 3))
        self.assertFalse(data_prep.looks_english("Genetinis dreifas yra atsitiktinis alel\u0173 da\u017eni\u0173 kitimas. " * 12))

    def test_unsafe_checkpoint_is_refused(self):
        class Payload:
            def __reduce__(self):
                return (print, ("UNSAFE CODE RAN",))

        path = os.path.join(self.tmp, "evil.pt")
        torch.save({"model": {}, "x": Payload()}, path)
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as ctx:
            tiny_gpt.safe_load(path)
        self.assertIn("Refusing to load", str(ctx.exception.code))
        self.assertNotIn("UNSAFE CODE RAN", buf.getvalue())

    def test_cli_rejects_bad_values(self):
        for bad in (["--name", "..\\evil"], ["--lr", "5"], ["--steps", "0"], ["--resume", "--init-from", "x.pt"],
                    ["--steps", "10", "--warmup-steps", "10"]):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                with open(os.devnull, "w") as devnull:
                    old = sys.stderr
                    sys.stderr = devnull
                    try:
                        tiny_gpt.parse_args(bad)
                    finally:
                        sys.stderr = old
        with redirect_stdout(io.StringIO()) as buf:
            tiny_gpt.parse_args(["--steps", "10", "--warmup-steps", "2", "--decay-frac", "0"])
        self.assertIn("--decay-frac 0 with wsd: the learning rate never decays", buf.getvalue())

    @SLOW
    def test_memory_savings_for_larger_models(self):
        """Activation checkpointing and the chunked loss leave the mathematics unchanged; the planner turns them
        on only when needed; a run with them stops, resumes exactly, and may switch them on resume; fine-tuning
        accumulates micro-batches to the same result as whole batches."""
        torch.manual_seed(0)
        cfg = tiny_gpt.ModelConfig(vocab_size=300, ctx=32, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2)
        model = tiny_gpt.TinyGPT(cfg)
        x = torch.randint(0, 300, (3, 33))

        def loss_and_grads(checkpointing, chunk, moe=False):
            model.zero_grad()
            model.grad_checkpoint = checkpointing
            aux = None
            if chunk:
                hidden = model(x[:, :-1], return_hidden=True, return_aux=True)
                hidden, aux = hidden if moe else (hidden[0], None)
                ce, z, n = tiny_gpt.chunked_lm_loss(hidden, model.lm_head.weight, x[:, 1:], chunk)
                loss = ce / n + 1e-4 * z / n
            else:
                logits, aux = model(x[:, :-1], return_aux=True)
                flat = logits.float().view(-1, 300)
                loss = torch.nn.functional.cross_entropy(flat, x[:, 1:].reshape(-1)) + 1e-4 * torch.logsumexp(flat, -1).pow(2).mean()
            if aux is not None:  # the mixture-of-experts terms: recomputed under checkpointing like everything else
                loss = loss + 0.01 * aux["lb"] + 1e-3 * aux["z"]
            loss.backward()
            counts = aux["counts"].clone() if aux is not None else None
            return loss.item(), torch.cat([p.grad.flatten() for p in model.parameters()]), counts

        base_loss, base_grad, _ = loss_and_grads(False, 0)
        for checkpointing, chunk in ((True, 0), (False, 20), (True, 7)):
            loss, grad, _ = loss_and_grads(checkpointing, chunk)
            self.assertAlmostEqual(loss, base_loss, places=5)
            self.assertLess((grad - base_grad).abs().max().item(), 1e-6)
        # the same with mixture-of-experts blocks (every second block; the sort inside the experts has
        # data-dependent sizes, which the non-reentrant recompute handles): loss, gradients and the routing counts
        # are unchanged by checkpointing and the loss in pieces
        dense_model = model
        torch.manual_seed(0)
        moe_cfg = tiny_gpt.ModelConfig(vocab_size=300, ctx=32, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2,
                                       n_experts=4, experts_per_token=2, moe_every=2, shared_hidden=64)
        model = tiny_gpt.TinyGPT(moe_cfg)
        self.assertEqual([b.moe for b in model.blocks], [False, True, False])
        base_loss, base_grad, base_counts = loss_and_grads(False, 0, moe=True)
        self.assertEqual(tuple(base_counts.shape), (1, 4))
        self.assertEqual(int(base_counts.sum()), 3 * 32 * 2)
        for checkpointing, chunk in ((True, 0), (False, 20), (True, 7)):
            loss, grad, counts = loss_and_grads(checkpointing, chunk, moe=True)
            self.assertAlmostEqual(loss, base_loss, places=5)
            self.assertLess((grad - base_grad).abs().max().item(), 1e-6)
            self.assertTrue(torch.equal(counts, base_counts))
        model = dense_model
        model.grad_checkpoint = False
        y = x[:, 1:].clone()
        y[0, :5] = -100
        with torch.no_grad():
            want = torch.log_softmax(model(x[:, :-1]).float(), -1).gather(-1, y.clamp(min=0)[..., None]).squeeze(-1)
            got = tiny_gpt.target_logprobs(model, x[:, :-1], y, 13)
        self.assertLess(((want * (y != -100)) - got).abs().max().item(), 1e-5)
        # planner: nothing switched on when it fits; savings when it does not
        big = tiny_gpt.ModelConfig(vocab_size=16384, ctx=1024, d_model=1024, n_layers=18, n_heads=16, n_kv_heads=4)
        # the same shape with 8 experts: 687M parameters in total, 220M active; the estimate (weights, gradients,
        # optimizer state over the total) no longer fits this laptop's budget, as the audit's table says
        big_moe = tiny_gpt.ModelConfig(**{**dataclasses.asdict(big), "n_experts": 8, "experts_per_token": 2})
        self.assertEqual(round(tiny_gpt.count_params(big_moe)[0] / 1e6), 687)
        self.assertEqual(round(tiny_gpt.count_params(big_moe)[2] / 1e6), 220)
        self.assertGreater(tiny_gpt.estimate_train_bytes(big_moe, 2, True, True, 4096, "muon"), 6.0 * 2 ** 30)
        self.assertEqual(tiny_gpt.choose_memory_plan(big, 262_144, 64 * 2 ** 30, True)[2:], (False, 0))
        micro, _, ckpt, _ = tiny_gpt.choose_memory_plan(big, 262_144, int(5.6 * 2 ** 30), True)
        self.assertTrue(ckpt)
        self.assertGreaterEqual(micro, 1)
        # Muon's smaller optimizer state: at this card's budget the planner takes micro 2 x 132 without recomputation
        # (about 2k tokens per micro-step already saturate the GEMMs) over 4 x 66 with it
        self.assertEqual(tiny_gpt.choose_memory_plan(big, 270_336, int(6.0 * 2 ** 30), True, "muon"),
                         (2, 132, False, 0))
        with self.assertRaises(SystemExit):
            tiny_gpt.choose_memory_plan(big, 262_144, 2 * 2 ** 30, True)
        # a run with both savings: stopped, resumed (and switched off on resume) like an uninterrupted run
        mem = self.common + ["--activation-checkpointing", "on", "--loss-chunk-tokens", "128", "--no-probes",
                             "--dropout", "0"]
        code, out = run(mem + ["--name", "S", "--steps", "16"])
        self.assertEqual(code, 0, out)
        self.assertIn("activation checkpointing on", out)
        self.assertIn("loss computed 128 tokens at a time", out)
        code, out_s = run(mem + ["--name", "T", "--steps", "16", "--stop-after-steps", "8"])
        self.assertEqual(code, 0, out_s)
        log = os.path.join(self.runs, "log.txt")
        code, out_r = run(["tiny_gpt.py", "--name", "T", "--out-dir", self.runs, "--log-file", log, "--device", "cpu",
                           "--resume", "--eval-every", "8", "--eval-tokens", "2048", "--sample-tokens", "8",
                           "--no-probes"])
        self.assertEqual(code, 0, out_r)

        def val16(text):
            return [float(v) for v in re.findall(r"step +16/16 \| train [\d.]+ \| val ([\d.]+)", text)]

        self.assertTrue(val16(out))
        self.assertEqual(val16(out), val16(out_r), "a resumed run with the memory savings must reproduce the run")
        ck = torch.load(os.path.join(self.runs, "T.pt"), map_location="cpu", weights_only=True)
        self.assertEqual((ck["train_config"]["activation_checkpointing"], ck["train_config"]["loss_chunk_tokens"]),
                         (True, 128))
        code, out_o = run(["tiny_gpt.py", "--name", "T", "--out-dir", self.runs, "--log-file", log, "--device", "cpu",
                           "--resume", "--steps", "20", "--eval-every", "8", "--eval-tokens", "2048",
                           "--sample-tokens", "8", "--no-probes", "--activation-checkpointing", "off",
                           "--loss-chunk-tokens", "0", "--min-lr-ratio", "0.05", "--tie-embeddings", "off"])
        self.assertEqual(code, 0, out_o)
        ck = torch.load(os.path.join(self.runs, "T.pt"), map_location="cpu", weights_only=True)
        self.assertEqual((ck["step"], ck["train_config"]["activation_checkpointing"]), (20, False))
        # step 16 of the new 20 is still in the stable phase, so the end point of the decay may change
        self.assertIn("from the decay phase on", out_o)
        self.assertIn("--tie-embeddings off ignored on --resume", out_o)
        self.assertAlmostEqual(ck["train_config"]["min_lr"], 0.05 * ck["train_config"]["lr"])
        self.assertEqual(ck["train_config"]["mix_alpha"], 1.0)
        # fine-tuning in micro-batches of 1 with recomputation: the same held-out loss as whole batches of 4
        examples = os.path.join(HERE, "finetune_examples")
        losses = []
        for name, extra in (("S_sft4", []), ("S_sft1", ["--micro-batch", "1", "--activation-checkpointing"])):
            code, out_f = run(["finetune.py", "sft", "--base", os.path.join(self.runs, "S_best.pt"), "--data",
                               os.path.join(examples, "sft_examples.jsonl"), "--name", name, "--out-dir", self.runs,
                               "--epochs", "1", "--batch", "4", "--device", "cpu", "--log-file", log] + extra)
            self.assertEqual(code, 0, out_f)
            losses.append(float(re.findall(r"epoch 1 \| train loss [\d.]+ \| held-out loss ([\d.]+)", out_f)[0]))
        self.assertAlmostEqual(losses[0], losses[1], places=3)

    @SLOW
    def test_muon_optimizer(self):
        """--optimizer muon: Muon for the 14 block matrices, AdamW for the embedding and 9 norm gains (5 RMSNorm
        gains and the 4 QK-norm gains new runs have); an
        interrupted run resumes exactly, keeps its optimizer, and the recipe reaches experiments.csv."""
        import csv
        log = os.path.join(self.runs, "log.txt")
        muon = self.common + ["--optimizer", "muon", "--no-probes"]
        code, out = run(muon + ["--name", "M", "--steps", "16"])
        self.assertEqual(code, 0, out)
        self.assertIn("optimizer: Muon + AdamW (Muon for 14 block matrices", out)
        code, out_s = run(muon + ["--name", "N", "--steps", "16", "--stop-after-steps", "8"])
        self.assertEqual(code, 0, out_s)
        code, out_r = run(["tiny_gpt.py", "--name", "N", "--out-dir", self.runs, "--log-file", log, "--device", "cpu",
                           "--resume", "--eval-every", "8", "--eval-tokens", "2048", "--sample-tokens", "8",
                           "--no-probes", "--optimizer", "adamw"])
        self.assertEqual(code, 0, out_r)
        self.assertIn("--optimizer adamw ignored on --resume", out_r)
        self.assertIn("optimizer state 24/24 tensors", out_r)

        def val16(text):
            return re.findall(r"step +16/16 \| train [\d.]+ \| val ([\d.]+)", text)

        self.assertTrue(val16(out))
        self.assertEqual(val16(out), val16(out_r), "a resumed Muon run must reproduce the uninterrupted run")
        ck = torch.load(os.path.join(self.runs, "N.pt"), map_location="cpu", weights_only=True)
        self.assertEqual(ck["train_config"]["optimizer"], "muon")
        self.assertEqual(len(ck["optimizer_param_names"]), 24)
        kinds = [sorted(k for k in s if k != "step") for s in ck["optimizer"]["state"].values()]
        self.assertEqual(sum(k == ["momentum_buffer"] for k in kinds), 14)
        self.assertEqual(sum(k == ["exp_avg", "exp_avg_sq"] for k in kinds), 10)
        with open(os.path.join(self.runs, "results", "experiments.csv"), encoding="utf-8-sig", newline="") as handle:
            row = [r for r in csv.DictReader(handle) if r["name"] == "N" and r["event"] == "training run"][-1]
        self.assertEqual((row["optimizer"], row["schedule"], row["decay_shape"], row["betas"], row["micro_batch"]),
                         ("Muon + AdamW", "WSD", "1-sqrt", "0.9/0.95", "8"))
        # AdamW -> Muon with --init-from: AdamW state only for the parameters AdamW still trains
        code, out_p = run(self.common + ["--name", "P", "--steps", "8", "--no-probes"])
        self.assertEqual(code, 0, out_p)
        code, out_q = run(muon + ["--name", "Q", "--steps", "16", "--init-from", os.path.join(self.runs, "P.pt")])
        self.assertEqual(code, 0, out_q)
        self.assertIn("optimizer state restored for 10/24 tensors", out_q)
        self.assertIn("learning rate re-warmed over 8 steps for the 14 tensors whose optimizer state restarts", out_q)
        self.assertEqual(torch.load(os.path.join(self.runs, "Q.pt"), map_location="cpu",
                                    weights_only=True)["train_config"].get("rewarm_steps"), 8)
        png = os.path.join(self.runs, "N.png")
        code, out_v = run(["view_pt.py", os.path.join(self.runs, "N.pt"), "--plot", "--out", png, "--dpi", "50",
                           "--no-forward"])
        self.assertEqual(code, 0, out_v)
        self.assertIn("optimizer: Muon + AdamW state for 24 tensors", out_v)
        self.assertNotIn("panel failed", out_v)

    def test_moe_run_resume_export_viewer_and_finetune(self):
        """A mixture-of-experts run (2 layers, 4 experts, 2 per token, Muon): the experts' 2-D matrices are on Muon
        and the routers on AdamW; the metrics lines carry the balance figures; a stopped run resumes exactly (with
        the grouped experts where the probe passes); the checkpoint, the experiments table, --export, the viewer,
        finetune --freeze-router and tools/edit_weights.py (refused) all know the architecture."""
        import csv
        log = os.path.join(self.runs, "log.txt")
        moe = self.common + ["--optimizer", "muon", "--no-probes", "--experts", "4"]
        code, out = run(moe + ["--name", "X", "--steps", "16"])
        self.assertEqual(code, 0, out)
        # 2 blocks x (4 attention projections + 4 experts x 2 matrices) on Muon; the embedding and the 2 routers on AdamW
        self.assertIn("Muon for 24 block matrices", out)
        self.assertIn("AdamW for the embedding (3 matrices)", out)
        self.assertIn("experts: 4 per MoE block (2 of 2 blocks), 2 per token, hidden 128", out)
        self.assertRegex(out, r"parameters: [\d,]+ total, [\d,]+ non-embedding, [\d,]+ active per token")
        self.assertRegex(out, r"step +16/16 \| train .*\| moe lb [\d.]+ z [\d.]+ load max [\d.]+ dead \d+ \|")
        code, out_s = run(moe + ["--name", "Y", "--steps", "16", "--stop-after-steps", "8"])
        self.assertEqual(code, 0, out_s)
        code, out_r = run(["tiny_gpt.py", "--name", "Y", "--out-dir", self.runs, "--log-file", log, "--device", "cpu",
                           "--resume", "--eval-every", "8", "--eval-tokens", "2048", "--sample-tokens", "8",
                           "--no-probes", "--moe-impl", "grouped", "--experts", "8", "--balance", "loss"])
        self.assertEqual(code, 0, out_r)
        self.assertIn("optimizer state 36/36 tensors", out_r)
        self.assertIn("--experts 8 (the architecture is the checkpoint's)", out_r)
        self.assertIn("--balance loss (the checkpoint's is bias)", out_r)
        if tiny_gpt.grouped_mm_available(torch.device("cpu"))[0]:
            self.assertIn("experts run as grouped", out_r)

        def val16(text):
            return re.findall(r"step +16/16 \| train ([\d.]+) \| val ([\d.]+)", text)

        self.assertTrue(val16(out))
        self.assertEqual(val16(out), val16(out_r), "a resumed MoE run must reproduce the uninterrupted run")
        ck = torch.load(os.path.join(self.runs, "Y.pt"), map_location="cpu", weights_only=True)
        self.assertEqual(ck["format_version"], tiny_gpt.FORMAT_VERSION)
        self.assertEqual((ck["model_config"]["n_experts"], ck["model_config"]["experts_per_token"],
                          ck["model_config"]["expert_hidden"]), (4, 2, 128))
        self.assertIn("blocks.1.ffn.router.bias", ck["model"])
        self.assertIn("blocks.1.ffn.experts.gate_up.3", ck["model"])
        self.assertEqual(tuple(ck["model"]["blocks.0.ffn.experts.down.0"].shape), (64, 128))
        self.assertEqual((ck["train_config"]["lb_loss"], ck["train_config"]["bias_rate"]), (0.01, 1e-3))
        self.assertFalse(torch.equal(ck["model"]["blocks.0.ffn.router.bias"], torch.zeros(4)),
                         "the balancing bias moved after 16 steps")
        kinds = [sorted(k for k in st if k != "step") for st in ck["optimizer"]["state"].values()]
        self.assertEqual(sum(k == ["momentum_buffer"] for k in kinds), 24)
        with open(os.path.join(self.runs, "Y_metrics.jsonl"), encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual([len(r["moe_load"]) for r in rows if "moe_load" in r], [2, 2], "per-block load for view_pt")
        self.assertTrue(all(r["dead_experts"] == 0 for r in rows if "moe_lb" in r))
        with open(os.path.join(self.runs, "results", "experiments.csv"), encoding="utf-8-sig", newline="") as handle:
            row = [r for r in csv.DictReader(handle) if r["name"] == "Y" and r["event"] == "training run"][-1]
        self.assertEqual((row["moe"], row["balance"], row["lb_loss"]), ("4x2/1", "bias", "0.0"))
        self.assertLess(int(row["active_params"]), int(row["params"]))
        self.assertEqual(int(row["params"]), tiny_gpt.count_params(tiny_gpt.ModelConfig(**ck["model_config"]))[0])
        # safetensors export (router bias buffers included), generation from it, the viewer, fine-tuning
        export = os.path.join(self.runs, "export_moe")
        code, out_e = run(["tiny_gpt.py", "--name", "Y", "--out-dir", self.runs, "--log-file", log, "--export", export])
        self.assertEqual(code, 0, out_e)
        tensors, meta = tiny_gpt.read_safetensors(os.path.join(export, "model.safetensors"))
        self.assertEqual(meta.get("n_experts"), "4")
        self.assertIn("blocks.0.ffn.router.bias", tensors)
        with open(os.path.join(export, "config.json"), encoding="utf-8") as handle:
            self.assertIn("mixture of experts (4 SwiGLU experts", json.load(handle)["architecture"])
        code, out_g = run(["tiny_gpt.py", "--checkpoint", export, "--log-file", log, "--device", "cpu",
                           "--generate", "Genetic drift", "--length", "5"])
        self.assertEqual(code, 0, out_g)
        png = os.path.join(self.runs, "Y.png")
        code, out_v = run(["view_pt.py", os.path.join(self.runs, "Y.pt"), "--plot", "--out", png, "--dpi", "50",
                           "--dataset", self.ds])
        self.assertEqual(code, 0, out_v)
        self.assertIn("experts 4 x hidden 128, 2 per token in 2 of 2 blocks", out_v)
        self.assertRegex(out_v, r"parameters [\d,]+ total, [\d,]+ active per token")
        self.assertNotIn("panel failed", out_v)
        self.assertGreater(os.path.getsize(png), 50_000)
        code, out_f = run(["finetune.py", "sft", "--base", os.path.join(self.runs, "X_best.pt"), "--data",
                           os.path.join(HERE, "finetune_examples", "sft_examples.jsonl"), "--name", "X_sft",
                           "--out-dir", self.runs, "--epochs", "1", "--batch", "4", "--device", "cpu", "--log-file",
                           log, "--freeze-router"])
        self.assertEqual(code, 0, out_f)
        self.assertIn("router frozen: 2 routing matrices", out_f)
        sft = torch.load(os.path.join(self.runs, "X_sft.pt"), map_location="cpu", weights_only=True)
        best = torch.load(os.path.join(self.runs, "X_best.pt"), map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(sft["model"]["blocks.0.ffn.router.weight"],
                                    best["model"]["blocks.0.ffn.router.weight"]))
        self.assertFalse(torch.equal(sft["model"]["blocks.0.ffn.experts.down.0"],
                                     best["model"]["blocks.0.ffn.experts.down.0"]))
        code, out_w = run([os.path.join("tools", "edit_weights.py"), os.path.join(self.runs, "X_best.pt"),
                           os.path.join(self.runs, "X_edit.pt"), "--device", "cpu"])
        self.assertNotEqual(code, 0)
        self.assertIn("this checkpoint has mixture-of-experts blocks", out_w)

    # ---------------- interrupted steps, first-step out-of-memory, resume refusals ----------------
    def patched_run(self, mode, n, args):
        """tiny_gpt.main(args) in a child process with a fixed memory budget and one fault injected:
        'interrupt' raises KeyboardInterrupt in the n-th batch read (inside a step), 'sigint' sends SIGINT
        inside the n-th optimizer update, 'oom' makes the first n batch reads raise a CUDA out-of-memory
        error, 'nangrad' puts an infinite gradient element into the n-th update and 'diverge' into every update
        from the n-th on ('none': no fault)."""
        script = os.path.join(self.tmp, "patched_run.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(PATCHED_RUN)
        return run([script, HERE, mode, str(n)] + args)

    def resume_args(self, name):
        return ["--name", name, "--out-dir", self.runs, "--log-file", os.path.join(self.runs, "log.txt"),
                "--device", "cpu", "--resume", "--eval-every", "8", "--eval-tokens", "2048", "--sample-tokens", "8",
                "--no-probes"]

    @SLOW
    def test_interrupted_step_resumes_exactly(self):
        """Ctrl+C inside a step (grad-accum 2, between evaluations) rewinds the sampler, dropout RNG and loss
        sums; Ctrl+C inside the optimizer update is held until the step is complete; both resume exactly.
        --resume refuses a newer fallback training state and carries on past an unreadable best file."""
        base = list(self.common[1:])
        base[base.index("--batch-size") + 1], base[base.index("--grad-accum") + 1] = "4", "2"
        base += ["--steps", "16", "--no-probes"]

        def at16(text):
            return re.findall(r"step +16/16 \| train ([\d.]+) \| val ([\d.]+)", text)

        code, out = self.patched_run("none", 0, base + ["--name", "K0"])
        self.assertEqual(code, 0, out)
        self.assertTrue(at16(out))
        # 1. KeyboardInterrupt in the second micro-batch of step 12 (batch read 24): step 11 is saved
        code, out_k = self.patched_run("interrupt", 24, base + ["--name", "K1"])
        self.assertEqual(code, 0, out_k)
        self.assertIn("Paused after step 11 ", out_k)
        ck = os.path.join(self.runs, "K1.pt")
        # a fallback training state newer than K1.pt (saved while K1.pt was locked): --resume refuses
        late = os.path.join(self.runs, "K1.123456.pt")
        shutil.copy(ck, late)
        os.utime(late, (os.path.getmtime(ck) + 60,) * 2)
        code, out_f = self.patched_run("none", 0, self.resume_args("K1"))
        self.assertNotEqual(code, 0)
        self.assertIn("is a newer training state", out_f)
        os.remove(late)
        # an unreadable best file is noted, not fatal
        with open(os.path.join(self.runs, "K1_best.pt"), "wb") as f:
            f.write(b"not a checkpoint")
        code, out_r = self.patched_run("none", 0, self.resume_args("K1"))
        self.assertEqual(code, 0, out_r)
        self.assertIn("could not read", out_r)
        self.assertEqual(at16(out), at16(out_r), "a run interrupted inside a step must resume exactly")
        # 2. SIGINT inside the optimizer update of step 10: held, so step 10 completes and is saved as done
        if not hasattr(signal, "raise_signal"):
            return
        code, out_s = self.patched_run("sigint", 10, base + ["--name", "K2"])
        self.assertEqual(code, 0, out_s)
        self.assertIn("Paused after step 10 ", out_s)
        self.assertEqual(torch.load(os.path.join(self.runs, "K2.pt"), map_location="cpu",
                                    weights_only=True)["step"], 10)
        code, out_r = self.patched_run("none", 0, self.resume_args("K2"))
        self.assertEqual(code, 0, out_r)
        self.assertEqual(at16(out), at16(out_r), "a run interrupted inside the update must resume exactly")
        self.assertFalse(glob.glob(os.path.join(self.runs, "K*.train.lock")), "the run lock must be released")

    @SLOW
    def test_first_step_out_of_memory_escalation(self):
        """A first-step out-of-memory error switches on the cheapest saving first (loss in pieces, the micro-batch
        halved down to about 2k tokens, then activation checkpointing, then halved further: at ctx 64 a
        micro-batch of 8 is already below 2k tokens, so recomputation comes before the halving), retries on the
        same windows, never switches on a saving the command line turned off, and experiments.csv records the
        settings actually used."""
        import csv
        base = list(self.common[1:]) + ["--steps", "8", "--no-probes"]

        def at8(text):
            return re.findall(r"step +8/8 \| train ([\d.]+) \| val ([\d.]+)", text)

        code, out = self.patched_run("oom", 3, base + ["--name", "O1"])
        self.assertEqual(code, 0, out)
        order = [out.find(s) for s in ("retrying with the loss computed", "retrying with activation checkpointing",
                                       "retrying with micro-batch 4 x 2")]
        self.assertTrue(all(i >= 0 for i in order) and order == sorted(order), out)
        with open(os.path.join(self.runs, "results", "experiments.csv"), encoding="utf-8-sig", newline="") as handle:
            row = [r for r in csv.DictReader(handle) if r["name"] == "O1" and r["event"] == "training run"][-1]
        self.assertEqual((row["activation_checkpointing"], row["loss_chunk_tokens"], row["micro_batch"]),
                         ("on", str(tiny_gpt.LOSS_CHUNK_TOKENS), "4"))
        # one retry trains on the same windows as a run that had the loss in pieces from the start
        code, out_1 = self.patched_run("oom", 1, base + ["--name", "O2"])
        self.assertEqual(code, 0, out_1)
        code, out_2 = self.patched_run("none", 0, base + ["--name", "O3", "--loss-chunk-tokens",
                                                           str(tiny_gpt.LOSS_CHUNK_TOKENS)])
        self.assertEqual(code, 0, out_2)
        self.assertTrue(at8(out_1))
        self.assertEqual(at8(out_1), at8(out_2), out_1 + out_2)
        # savings turned off on the command line stay off: nothing is left at micro-batch 1
        forced = list(base)
        forced[forced.index("--batch-size") + 1] = "2"
        code, out_x = self.patched_run("oom", 1000, forced + ["--name", "O4", "--loss-chunk-tokens", "0",
                                                              "--activation-checkpointing", "off"])
        self.assertNotEqual(code, 0)
        self.assertIn("--loss-chunk-tokens 0 and --activation-checkpointing off", out_x)
        self.assertNotIn("retrying with the loss computed", out_x)
        # --init-from a training state without --steps: the planned steps come after its step
        code, out_p = self.patched_run("none", 0, list(self.common[1:]) + [
            "--name", "O5", "--no-probes", "--stop-after-steps", "1", "--init-from", os.path.join(self.runs, "O3.pt"),
            "--ctx", "96"])
        self.assertEqual(code, 0, out_p)
        self.assertIn("the schedule continues at the --init-from checkpoint's step 8", out_p)
        self.assertIn("model was trained at ctx 64; continuing at 96", out_p)
        self.assertEqual(torch.load(os.path.join(self.runs, "O5.pt"), map_location="cpu",
                                    weights_only=True)["model_config"]["ctx"], 96)
        self.assertEqual(torch.load(os.path.join(self.runs, "O5.pt"), map_location="cpu", weights_only=True)["step"], 9)

    def test_resume_with_a_new_micro_batch(self):
        """--resume may trade micro-batch for accumulation at the same tokens per step (2 x 132 instead of 4 x 66
        with recomputation): the run continues to its step count with the same tokens seen, the metrics row and
        the saved settings carry the new pair and the peak memory, and a pair that changes the tokens per step
        is refused."""
        base = self.common + ["--steps", "8", "--no-probes"]  # 8 x 64 tokens x 1 = 512 tokens per step
        code, out = run(base + ["--name", "MB", "--stop-after-steps", "4"])
        self.assertEqual(code, 0, out)
        code, out_bad = run(["tiny_gpt.py"] + self.resume_args("MB") + ["--batch-size", "4", "--grad-accum", "3"])
        self.assertNotEqual(code, 0)
        self.assertIn("tokens per step must stay 512", out_bad)
        self.assertIn("--batch-size B --grad-accum 512//(B*64)", out_bad)
        # the accumulation may be left out: 512 // (4 x 64) = 2
        code, out_r = run(["tiny_gpt.py"] + self.resume_args("MB") + ["--batch-size", "4"])
        self.assertEqual(code, 0, out_r)
        self.assertIn("micro-batch 8 x 1 -> 4 x 2 (same 512 tokens per step", out_r)
        self.assertNotIn("--batch-size ignored", out_r)
        ck = torch.load(os.path.join(self.runs, "MB.pt"), map_location="cpu", weights_only=True)
        tc = ck["train_config"]
        self.assertEqual((ck["step"], ck["tokens_seen"], tc["micro_batch"], tc["grad_accum"], tc["tokens_per_step"]),
                         (8, 8 * 512, 4, 2, 512))
        metrics = glob.glob(os.path.join(self.runs, "MB*metrics*.jsonl"))
        self.assertEqual(len(metrics), 1, metrics)
        with open(metrics[0], encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        last = [r for r in rows if r.get("step") == 8]
        self.assertEqual(len(last), 1, rows)
        self.assertGreater(last[0]["peak_mem_gib"], 0.0)
        self.assertEqual(last[0]["tokens"], 8 * 512)
        # the same pair again is not a change (no note), and a different accumulation alone is refused
        code, out_same = run(["tiny_gpt.py"] + self.resume_args("MB") + ["--steps", "9", "--batch-size", "4",
                                                                         "--grad-accum", "2"])
        self.assertEqual(code, 0, out_same)
        self.assertNotIn("micro-batch 4 x 2 ->", out_same)
        code, out_acc = run(["tiny_gpt.py"] + self.resume_args("MB") + ["--steps", "10", "--grad-accum", "4"])
        self.assertNotEqual(code, 0)
        self.assertIn("tokens per step must stay 512", out_acc)

    def test_mid_run_faults_recover(self):
        """Faults after the first step: an out-of-memory error replays the step on the same windows; a
        non-finite gradient norm skips the update and changes no weight; a failed periodic save is retried
        (three in a row end the run, with the error recorded in experiments.csv); Ctrl+C right after a
        metrics line does not fold that interval into the next row; the loaded checkpoint is freed after
        setup."""
        import csv
        base = list(self.common[1:]) + ["--steps", "8", "--no-probes"]

        def lines(text, total):
            return re.findall(r"step +(\d+)/%d \| train ([\d.]+) \| val ([\d.]+)" % total, text)

        code, out = self.patched_run("none", 0, base + ["--name", "F0"])
        self.assertEqual(code, 0, out)
        self.assertTrue(lines(out, 8))
        # out of memory at step 3 (batch read 3): replayed on the same windows, so the run matches an unfaulted one
        code, out_o = self.patched_run("oom_at_step", 3, base + ["--name", "F1"])
        self.assertEqual(code, 0, out_o)
        self.assertIn("Out of memory at step 3; retrying the same step (1/2).", out_o)
        self.assertEqual(lines(out, 8), lines(out_o, 8), out_o)
        # an infinite gradient element at step 4: the update is skipped, every weight stays, the run completes;
        # the step-4 train value is the mean over steps 1-3 alone (the skipped step is in neither sum nor count)
        code, out_n = self.patched_run("nangrad", 4, base + ["--name", "F2", "--eval-every", "4", "--log-every", "1"])
        self.assertEqual(code, 0, out_n)
        self.assertIn("step 4: non-finite gradient norm inf; update skipped (1/20)", out_n)
        self.assertIn("FAULT CHECK: 0 parameters changed", out_n)
        self.assertTrue(lines(out_n, 8), "the metrics line must still carry finite numbers")
        progress = {int(m.group(1)): float(m.group(2)) for m in
                    re.finditer(r"^step +(\d+)/8 \| +[\d.]+% \| train ([\d.]+)", out_n, re.M)}
        self.assertEqual(sorted(k for k in progress if k < 4), [1, 2, 3], out_n)
        train_at_4 = float([m for m in lines(out_n, 8) if m[0] == "4"][0][1])
        self.assertAlmostEqual(train_at_4, sum(progress[k] for k in (1, 2, 3)) / 3, delta=0.0015, msg=out_n)
        self.assertEqual(torch.load(os.path.join(self.runs, "F2.pt"), map_location="cpu",
                                    weights_only=True)["step"], 8)
        # saves that fail: training goes on and the counter resets after a success ...
        code, out_s = self.patched_run("saveflaky", 6, base + ["--name", "F3", "--save-every", "1"])
        self.assertEqual(code, 0, out_s)
        self.assertEqual(out_s.count("WARNING: could not save"), 4, out_s)
        self.assertIn("(2/3)", out_s)
        self.assertNotIn("(3/3)", out_s)
        self.assertEqual(torch.load(os.path.join(self.runs, "F3.pt"), map_location="cpu",
                                    weights_only=True)["step"], 8)
        # ... but three failures in a row end the run, with a crash file and the error in experiments.csv
        code, out_f = self.patched_run("savefail", 3, base + ["--name", "F4", "--save-every", "1"])
        self.assertNotEqual(code, 0)
        self.assertIn("(3/3)", out_f)
        self.assertTrue(os.path.isfile(os.path.join(self.runs, "F4_crash.pt")))
        with open(os.path.join(self.runs, "results", "experiments.csv"), encoding="utf-8-sig", newline="") as handle:
            row = [r for r in csv.DictReader(handle) if r["name"] == "F4" and r["event"] == "training run"][-1]
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["notes"].startswith("OSError"), row["notes"])
        # Ctrl+C inside the sample after the step-8 metrics line: the interval was already recorded, so the
        # resumed run's step-16 train value covers steps 9-16 only, as in an uninterrupted run
        base16 = list(self.common[1:]) + ["--steps", "16", "--no-probes"]
        code, out_u = self.patched_run("none", 0, base16 + ["--name", "F5"])
        self.assertEqual(code, 0, out_u)
        code, out_i = self.patched_run("interrupt_sample", 1, base16 + ["--name", "F6"])
        self.assertEqual(code, 0, out_i)
        self.assertIn("Paused after step 8 ", out_i)
        code, out_r = self.patched_run("gcscan", 1, self.resume_args("F6"))
        self.assertEqual(code, 0, out_r)
        self.assertIn("FAULT CHECK: 0 checkpoint dicts alive after setup", out_r)
        self.assertEqual([m for m in lines(out_u, 16) if m[0] == "16"], [m for m in lines(out_r, 16) if m[0] == "16"])

    def test_keep_split_from_and_build_lock_refusals(self):
        """--keep-split-from needs a finished earlier build (manifest.json, a docs.tsv ending in a newline);
        a live _build.lock in --out refuses the build; a stale one is replaced."""
        src = os.path.join(self.tmp, "cut_dataset")
        os.makedirs(src, exist_ok=True)
        with open(os.path.join(self.ds, "docs.tsv"), "rb") as f:
            ledger = f.read()
        with open(os.path.join(src, "docs.tsv"), "wb") as f:
            f.write(ledger.rstrip(b"\n"))
        argv = ["--out", os.path.join(self.tmp, "never_built"), "--md-dir", self.corpus.notes, "--keep-split-from", src]
        with open(os.devnull, "w") as devnull, unittest.mock.patch("sys.stderr", devnull):
            with self.assertRaises(SystemExit):  # no manifest.json
                data_prep.prepare(argv)
            shutil.copy(os.path.join(self.ds, "manifest.json"), src)
            with self.assertRaises(SystemExit):  # docs.tsv cut short
                data_prep.prepare(argv)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "never_built")))
        out = os.path.join(self.tmp, "locked_out")
        os.makedirs(out)
        lock = os.path.join(out, data_prep.LOCK_NAME)
        with open(lock, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getppid(), "started": None}, f)  # a live process that is not this one
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            data_prep.prepare(["--out", out, "--md-dir", self.corpus.notes, "--vocab-size", "600"])
        self.assertIn("being built by another", str(ctx.exception.code))
        self.assertTrue(os.path.isfile(lock), "another build's lock is never removed")
        dead = subprocess.run([PY, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
        with open(lock, "w", encoding="utf-8") as f:
            json.dump({"pid": int(dead.stdout), "started": None}, f)
        self.assertFalse(data_prep._lock_owner_alive(lock))
        self.assertEqual(data_prep._acquire_lock(lock), lock)
        data_prep._release_lock(lock)
        self.assertFalse(os.path.exists(lock))

    def test_compare_models_leaves_out_documents_the_other_model_trained_on(self):
        import compare_models
        layout = {}
        docs = compare_models.validation_documents(self.ds, 4, 2000, layout)
        self.assertTrue(docs)
        fingerprint = data_prep.read_manifest(self.ds)["fingerprint"]
        with open(os.path.join(self.ds, "docs.tsv"), encoding="utf-8", newline="\n") as f:
            lines = f.read().split("\n")
        head = lines[0].split("\t")
        status, doc_id = head.index("status"), head.index("doc_id")
        self.assertIn("content_hash", head)

        def other(name, rename):
            folder = os.path.join(self.tmp, name)
            os.makedirs(folder, exist_ok=True)
            rows = [lines[0]]
            for line in lines[1:]:
                cells = line.split("\t")
                if len(cells) == len(head):
                    if cells[status] == "val":
                        cells[status] = "train"
                    if rename:  # moved or renamed files: matched by their text
                        cells[doc_id] = "moved/" + cells[doc_id]
                rows.append("\t".join(cells))
            with open(os.path.join(folder, "docs.tsv"), "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(rows))
            return folder

        with redirect_stdout(io.StringIO()):
            for rename in (False, True):
                folder = other(f"other_ds_{rename}", rename)
                drop, unchecked, unknown = compare_models.training_overlap(
                    docs, self.ds, layout, {"A": {"path": self.ds, "fingerprint": fingerprint},
                                            "B": {"path": folder, "fingerprint": "different"}})
                self.assertEqual(drop, set(range(len(docs))), f"rename={rename}")
                self.assertEqual((unchecked, unknown), ({}, {}))
            drop, unchecked, unknown = compare_models.training_overlap(
                docs, self.ds, layout, {"B": {"path": os.path.join(self.tmp, "missing_ds"), "fingerprint": "x"},
                                        "C": {}})
        self.assertEqual(drop, set())
        self.assertEqual((list(unchecked), list(unknown)), (["B"], ["C"]))

    @SLOW
    def test_divergence_never_overwrites_the_good_files(self):
        """A run whose gradients stay infinite (20 skipped updates in a row) stops with an error and never writes
        a bad state: <name>.pt still holds the last good weights (the ones the step-4 best was saved from), no
        evaluation after the divergence becomes a NEW BEST, and the diverged state goes to <name>_crash.pt.
        --overwrite of the earlier run first renames its .pt, _best.pt and _metrics.jsonl to *.bak-<time>,
        byte for byte."""
        base = list(self.common[1:]) + ["--no-probes", "--eval-every", "4", "--save-every", "4", "--name", "DV"]
        code, out = self.patched_run("none", 0, base + ["--steps", "8"])
        self.assertEqual(code, 0, out)
        files = {p: os.path.join(self.runs, "DV" + p) for p in (".pt", "_best.pt", "_metrics.jsonl")}
        before = {p: data_prep.sha256_file(f) for p, f in files.items()}
        # every update from step 5 on has an infinite gradient element: steps 5-24 are skipped, then the run stops
        code, out_d = self.patched_run("diverge", 5, base + ["--steps", "40", "--overwrite"])
        self.assertNotEqual(code, 0, out_d)
        self.assertIn("update skipped (20/20)", out_d)
        self.assertIn("was not overwritten", out_d)
        for p, f in files.items():
            backups = glob.glob(glob.escape(f) + ".bak-*")
            self.assertEqual(len(backups), 1, (p, backups))
            self.assertEqual(data_prep.sha256_file(backups[0]), before[p], p)
        statuses = re.findall(r"^step +(\d+)/40 \| train .*\| (NEW BEST.*|best [\d.]+ @\d+)$", out_d, re.M)
        self.assertEqual([s for s, status in statuses if status.startswith("NEW BEST")], ["4"], out_d)
        self.assertTrue(os.path.isfile(os.path.join(self.runs, "DV_crash.pt")))
        last = torch.load(files[".pt"], map_location="cpu", weights_only=True)
        best = torch.load(files["_best.pt"], map_location="cpu", weights_only=True)
        self.assertEqual(last["step"], 20, "the periodic saves before the stop hold the unchanged good weights")
        self.assertEqual(best["step"], 4)
        for name, tensor in last["model"].items():
            self.assertTrue(bool(torch.isfinite(tensor.float()).all()), name)
            self.assertTrue(torch.equal(tensor.to(best["model"][name].dtype), best["model"][name]), name)

    def test_legacy_v3_checkpoint_read_only(self):
        """A checkpoint in the original script's v3 layout (settings CTX/D/LAYERS/HEADS, unrounded 8d/3 FFN,
        a SentencePiece proto used in 'plain' mode, model_version 3), built here so the legacy loader runs on
        every machine: it is classified, rebuilt with the same logits, generates, and the file is not changed."""
        with open(os.path.join(self.ds, "tokenizer.model"), "rb") as handle:
            proto = handle.read()
        vocab = data_prep.Tokenizer(proto, "plain").vocab_size
        torch.manual_seed(4)
        cfg = tiny_gpt.ModelConfig(vocab_size=vocab, ctx=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4,
                                   ffn_hidden=(8 * 64) // 3)
        original = tiny_gpt.TinyGPT(cfg).eval()
        path = os.path.join(self.tmp, "legacy_v3_best.pt")
        torch.save({"model_version": 3, "settings": {"CTX": 64, "D": 64, "LAYERS": 2, "HEADS": 4},
                    "model": original.state_dict(), "tokenizer_proto": proto, "step": 1000, "best_val": 3.21}, path)
        before = data_prep.sha256_file(path)
        model, tok, loaded_cfg, obj = tiny_gpt.load_for_inference(path)
        self.assertEqual(tiny_gpt.checkpoint_info(obj), ("legacy-v3", "inference"))
        self.assertEqual(tok.encode_mode, "plain")
        self.assertEqual(loaded_cfg.ffn_hidden, (8 * loaded_cfg.d_model) // 3)
        self.assertEqual((loaded_cfg.vocab_size, loaded_cfg.ctx, loaded_cfg.n_kv_heads), (vocab, 64, 4))
        self.assertTrue(loaded_cfg.tie_embeddings)
        x = torch.randint(0, vocab, (2, 20), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            torch.testing.assert_close(model(x), original(x), rtol=1e-5, atol=1e-5)
        text = tiny_gpt.generate(model, tok, "Genetic drift", 5, torch.device("cpu"),
                                 generator=torch.Generator().manual_seed(0))
        self.assertTrue(text.startswith("Genetic drift"), text)
        self.assertEqual(before, data_prep.sha256_file(path))
        self.assertEqual(tiny_gpt.checkpoint_info({"model_version": 3, "settings": {}, "optimizer": {}}),
                         ("legacy-v3", "training"))


class CharTok:
    """A tiny stand-in tokenizer for model-level tests."""
    bos_id, eos_id = 0, -1

    def encode(self, text):
        return [1 + ord(c) % 40 for c in text]

    def decode(self, ids):
        return "".join(chr(64 + i % 60) for i in ids)


class TestFixes(unittest.TestCase):
    """Fast unit-level regression tests for the 2026-10 audit fixes (no dataset needed)."""

    def test_vocab_size_grows_with_data(self):
        """4,096 below 20M characters, then doubling for every 10x more text, capped at 65,536 (uint16 token files)."""
        cases = [(5e6, 4096), (2e7, 8192), (1.99e8, 8192), (2e8, 16384), (1.5e9, 16384), (1.5e10, 32768),
                 (2e10, 65536), (2e11, 65536), (2e13, 65536)]
        for chars, want in cases:
            self.assertEqual(data_prep.auto_vocab_size(chars), want, chars)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_fixes_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_kv_cache_matches_full_recompute(self):
        """KV-cached decoding gives the logits of a full forward pass: a prompt, one token at a
        time, several tokens after a cached prefix, a batch, and a restart past the context."""
        model = self.small_model(seed=3, vocab=50)  # ctx 16
        gen = torch.Generator().manual_seed(0)
        seq = torch.randint(0, 50, (2, 16), generator=gen)
        with torch.no_grad():
            full = model(seq)
            cache = model.new_cache()
            parts = [model(seq[:, :5], cache=cache)]  # prompt
            parts += [model(seq[:, i:i + 1], cache=cache) for i in range(5, 9)]  # one at a time
            parts.append(model(seq[:, 9:16], cache=cache))  # several new tokens after a prefix
            cached = torch.cat(parts, dim=1)
            self.assertTrue(torch.allclose(full, cached, atol=1e-4), (full - cached).abs().max())
            with self.assertRaises(ValueError):  # the cache never grows past the context
                model(seq[:, :1], cache=cache)
        ids = seq[0].tolist()[:3]
        dec = tiny_gpt.IncrementalDecoder(model, torch.device("cpu"))
        for _ in range(40):  # 43 tokens with ctx 16: the cache restarts twice
            want = model(torch.tensor([ids[-16:]]))[0, -1] if len(ids) - dec.base <= 16 and dec.cache else None
            got = dec.next_logits(ids)
            if want is not None and dec.base == max(0, len(ids) - 16):
                self.assertTrue(torch.allclose(want, got, atol=1e-4))
            ids.append(int(got.argmax()))
        self.assertEqual(len(ids), 43)

    def test_markdown_gzip_inputs(self):
        """data_prep reads a.md.gz like a.md (same id and title); a damaged .gz is skipped and counted."""
        import collections
        import gzip
        folder = os.path.join(self.tmp, "gz_md")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "plain.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("# Plain\n\nGenetic drift.")
        with gzip.open(os.path.join(folder, "packed.md.gz"), "wt", encoding="utf-8", newline="\n") as f:
            f.write("# Packed\n\nAllele frequencies.")
        with open(os.path.join(folder, "broken.txt.gz"), "wb") as f:
            f.write(b"not gzip")
        stats = {"scanned": 0, "skipped": collections.Counter(), "cleaning": collections.Counter()}
        docs = list(data_prep.iter_markdown([folder], list(data_prep.MD_PATTERNS_DEFAULT), [], False, stats, []))
        self.assertEqual([(d[0], d[1], d[2]) for d in docs],
                         [("packed.md", "packed", "# Packed\n\nAllele frequencies."),
                          ("plain.md", "plain", "# Plain\n\nGenetic drift.")])
        self.assertEqual(stats["skipped"]["bad_gzip"], 1)

    def test_keyword_covered_terms(self):
        """tools/kw_covered.py: a term whose every match is another term's match only double-counts."""
        sys.path.insert(0, os.path.join(HERE, "tools"))
        import kw_covered
        terms = ["polymer*", "polymerase", "spatial transcriptomic*", "spatial transcriptomics", "CITE-seq",
                 "=CITE-seq", "=MET", "=Met", "regression", "linear regression", "*mab", "adalimumab",
                 "China Kadoorie", "China Kadoorie Biobank"]
        self.assertEqual(kw_covered.find(terms, nested=False),
                         {"polymerase": "polymer*", "spatial transcriptomics": "spatial transcriptomic*",
                          "=CITE-seq": "CITE-seq"})
        nested = kw_covered.find(terms, nested=True)
        self.assertEqual(nested.get("China Kadoorie Biobank"), "China Kadoorie")
        for kept in ("linear regression", "adalimumab", "=Met", "=MET", "polymer*"):
            self.assertNotIn(kept, nested)

    def test_download_docs_helpers(self):
        """download_docs.py: unsafe archive names, bad links, crawl scope, robots wildcards, decoding, Rd code."""
        import urllib.robotparser
        import download_docs as d
        for bad in ("/etc/passwd", "C:/x.md", "../up.md", "a/../../b.md", "a\x00b"):
            self.assertIsNone(d.safe_member(bad), bad)
        self.assertEqual(d.safe_member("repo-main\\docs\\./a.md"), "repo-main/docs/a.md")
        for bad in ("http://localhost:PORT/", "http://x:99999/", "https://[hostname]/"):
            self.assertIsNone(d.normalise_url(bad), bad)
        scope = d.Scope("https://a.org/docs/")
        self.assertIn("https://a.org/docs/x", scope)
        self.assertNotIn("https://a.org/other/x", scope)
        self.assertNotIn("https://a.org:8443/docs/x", scope)
        rp = urllib.robotparser.RobotFileParser()
        d.parse_robots(rp, "User-agent: *\nDisallow: /*?\nDisallow: /private/\nCrawl-delay: 2.5\n")
        self.assertTrue(rp.can_fetch("tinyGPT-docs", "https://a.org/docs/manual.html"))
        self.assertFalse(rp.can_fetch("tinyGPT-docs", "https://a.org/docs/manual.html?x=1"))
        self.assertFalse(rp.can_fetch("tinyGPT-docs", "https://a.org/private/x.html"))
        self.assertEqual(rp.crawl_delay("tinyGPT-docs"), 3)
        self.assertEqual(d.decode_text("don’t – café".encode("cp1252")), "don’t – café")
        md = d.rd_markdown(r"\name{f}\title{F}\usage{f(fun = function(v) { v + 1 })}" "\n"
                           r"\examples{for (i in 1:3) {" "\n  print(i)\n}\n}")
        self.assertIn("f(fun = function(v) { v + 1 })", md)
        self.assertIn("for (i in 1:3) {\n  print(i)\n}", md)

    def small_model(self, seed=0, vocab=50):
        torch.manual_seed(seed)
        cfg = tiny_gpt.ModelConfig(vocab_size=vocab, ctx=16, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2)
        return tiny_gpt.TinyGPT(cfg).eval()

    # ---------------- tiny_gpt ----------------
    def test_split_batch_keeps_the_planned_tokens(self):
        self.assertEqual(tiny_gpt.split_batch(64, 8), (8, 8))
        self.assertEqual(tiny_gpt.split_batch(61, 32), (30, 2))
        for largest in (1, 2, 3, 5, 8, 16, 32):
            for target in range(largest, 400):  # the caller's largest micro-batch never exceeds the target
                m, a = tiny_gpt.split_batch(target, largest)
                self.assertTrue((largest + 1) // 2 <= m <= largest, (target, largest, m))
                self.assertLessEqual(m * a, target, (target, largest))
                self.assertGreaterEqual(m * a, largest * (target // largest), "never worse than the largest")
                if target >= 8 * largest:
                    self.assertGreaterEqual(m * a, 0.95 * target, (target, largest))
        cfg = tiny_gpt.ModelConfig(vocab_size=100, ctx=16, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2)
        self.assertEqual(tiny_gpt.choose_memory_plan(cfg, 48, 2 ** 34, False, micro=2)[:2], (2, 1),
                         "an explicit micro-batch rounds the accumulation down")

    def test_generate_last_position_matches_full_logits(self):
        model = self.small_model()

        class Full(torch.nn.Module):  # not a TinyGPT: generate() takes the full-logits path
            def __init__(self, inner):
                super().__init__()
                self.inner, self.ctx = inner, inner.ctx

            def forward(self, x):
                return self.inner(x)

        tok = CharTok()
        a = tiny_gpt.generate(model, tok, "Genetic drift", 20, torch.device("cpu"), temperature=0)
        b = tiny_gpt.generate(Full(model), tok, "Genetic drift", 20, torch.device("cpu"), temperature=0)
        self.assertEqual(a, b)
        model.train()
        grad_modes = []
        hook = model.register_forward_hook(lambda *_: grad_modes.append(torch.is_grad_enabled()))
        with torch.enable_grad():
            tiny_gpt.generate(model, tok, "x", 3, torch.device("cpu"), temperature=0)
        hook.remove()
        self.assertEqual(grad_modes, [False] * 3, "sampling builds no autograd graph")
        self.assertTrue(model.training, "generate() restores training mode")

    def test_defer_interrupt_holds_ctrl_c_until_the_update_is_done(self):
        if not hasattr(signal, "raise_signal") or signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
            self.skipTest("needs Python's own Ctrl+C handler")
        held, done = tiny_gpt.DeferInterrupt(), []
        with held:
            signal.raise_signal(signal.SIGINT)
            done.append(1)  # still runs: the interrupt is held
        self.assertEqual(done, [1])
        self.assertIs(signal.getsignal(signal.SIGINT), signal.default_int_handler)
        with self.assertRaises(KeyboardInterrupt):
            held.deliver()
        held.deliver()  # delivered once only

    # ---------------- data_prep: 2026-11 audit (WP8) ----------------
    def test_spool_round_trip_zstd_and_plain(self):
        folder = os.path.join(self.tmp, "spools")
        os.makedirs(folder)
        rows = [{"s": "md", "id": f"d{i}", "x": "héllo\nwörld " * 50 + str(i)} for i in range(200)]
        for have in (True, False):
            data_prep._ZSTD = have
            try:
                path = data_prep._spool_path(folder, "spool1.md.jsonl")
                self.assertEqual(path.endswith(".zst"), have)
                with data_prep._open_spool(path, "w") as handle:
                    for r in rows:
                        json.dump(r, handle, ensure_ascii=False)
                        handle.write("\n")
                with data_prep._open_spool(path, "r") as handle:
                    back = [json.loads(line) for line in handle]
                self.assertEqual(back, rows)
                self.assertEqual(data_prep._spool_line_count(path), (200, True))
                cut = path + ".cut"  # a spool cut short by a crash is reported incomplete, not as an exception
                shutil.copy(path, cut)
                with open(cut, "r+b") as handle:
                    handle.truncate(os.path.getsize(cut) // 2)
                lines, complete = data_prep._spool_line_count(cut)
                self.assertFalse(complete)
                self.assertLess(lines, 200)
                if have:
                    self.assertLess(os.path.getsize(path), len(json.dumps(rows)) / 5, "zstd compresses the JSON")
            finally:
                data_prep._ZSTD = None

    def test_prefilter_pattern_cache_hit_and_miss(self):
        terms = ["genetic drift", "allele*", "=MET", "Hardy-Weinberg"]
        cache = os.path.join(self.tmp, "kw.txt.prefilter.json")
        fresh = data_prep.cached_prefilter_patterns(terms, cache)
        self.assertEqual(fresh, data_prep._prefilter_patterns(terms))
        self.assertTrue(os.path.isfile(cache))
        with open(cache, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored["patterns"] = ["(?:sentinel)"]  # a hit returns what the cache holds
        with open(cache, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        self.assertEqual(data_prep.cached_prefilter_patterns(terms, cache), ["(?:sentinel)"])
        # a changed term list (or pattern size) misses and rewrites the cache
        self.assertEqual(data_prep.cached_prefilter_patterns(terms + ["CRISPR"], cache),
                         data_prep._prefilter_patterns(terms + ["CRISPR"]))
        self.assertEqual(data_prep.cached_prefilter_patterns(terms, cache), fresh)
        with open(cache, "w", encoding="utf-8") as handle:
            handle.write("not json")
        self.assertEqual(data_prep.cached_prefilter_patterns(terms, cache), fresh, "a damaged cache is ignored")
        self.assertIsNone(data_prep.prefilter_cache_path("genetic drift,allele*"))
        kwfile = os.path.join(self.tmp, "kw.txt")
        open(kwfile, "w").close()
        self.assertEqual(data_prep.prefilter_cache_path(kwfile), os.path.abspath(kwfile) + ".prefilter.json")

    def test_phrase_index_matches_first_word_matcher(self):
        """The (first word, second word) phrase index counts exactly what the earlier
        first-word matcher counted, on keywords.txt and planted edge cases."""
        import collections

        class FirstWordMatcher(data_prep._TermMatcher):  # the matcher before the audit, as a reference
            def __init__(self, terms):
                super().__init__(terms)
                self.table, self.table_cs = collections.defaultdict(list), collections.defaultdict(list)
                for term in terms:
                    core, cs, _lead, trail = data_prep._term_parts(term)
                    words = data_prep._WORDS_RE.findall(core)
                    if len(words) < 2:
                        continue
                    if not cs:
                        words = [w.lower() for w in words]
                    (self.table_cs if cs else self.table)[words[0]].append((words[1:], term, trail))

            @staticmethod
            def _phrases(seq, present, table, counter, weight):
                n = len(seq)
                for i, w in enumerate(seq):
                    if w not in present:
                        continue
                    for rest, term, trail in table[w]:
                        k = len(rest)
                        if i + k >= n or seq[i + 1:i + k] != rest[:-1]:
                            continue
                        last = seq[i + k]
                        if last.startswith(rest[-1]) if trail else last == rest[-1]:
                            counter[term] += weight

            def count(self, text, counter, weight=1):
                words = data_prep._WORDS_RE.findall(text)
                if not words:
                    return counter
                lower = [w.lower() for w in words]
                freq = collections.Counter(lower)
                for w in freq.keys() & self.single.keys():
                    counter[self.single[w]] += freq[w] * weight
                for affixes in self.affixes:
                    affixes.count(freq, counter, weight)
                present = freq.keys() & self.table.keys()
                if present:
                    self._phrases(lower, present, self.table, counter, weight)
                if self.need_case:
                    freq_cs = collections.Counter(words)
                    for w in freq_cs.keys() & self.single_cs.keys():
                        counter[self.single_cs[w]] += freq_cs[w] * weight
                    for affixes in self.affixes_cs:
                        affixes.count(freq_cs, counter, weight)
                    present = freq_cs.keys() & self.table_cs.keys()
                    if present:
                        self._phrases(words, present, self.table_cs, counter, weight)
                return counter

        terms = data_prep.read_terms(os.path.join(HERE, "keywords.txt")) + [
            "Gallus gallus", "Loa loa", "Johnson & Johnson", "genetic drif*", "linkage disequilibrium map*",
            "=HLA class I", "cell cell adhesion"]
        old, new = FirstWordMatcher(terms), data_prep._TermMatcher(terms)
        texts = [BIO, paragraphs(3), "Gallus gallus gallus is the red junglefowl.", "gallus " * 5, "Loa loa loa loa",
                 "Johnson & Johnson makes genetic tests; Boris Johnson spoke.", "genetic drifting and genetic drift",
                 "linkage disequilibrium mapping, linkage disequilibrium maps, linkage disequilibrium.",
                 "HLA class I and hla class i molecules; HLA class II", "cell cell cell adhesion molecules",
                 "genome-wide association study of heritability in the UK Biobank cohort, Hardy-Weinberg equilibrium",
                 "a lone first word: linkage", "", "   "]
        for text in texts:
            self.assertEqual(new.count(text, collections.Counter()), old.count(text, collections.Counter()), text)
        kw = data_prep.KeywordFilter(["cell cell adhesion", "genetic drif*"], [], min_hits=1, min_distinct=1)
        self.assertEqual(kw.evaluate("", "cell cell cell adhesion")[3], collections.Counter({"cell cell adhesion": 1}))
        self.assertEqual(kw.evaluate("", "genetic drift genetic drifts")[3], collections.Counter({"genetic drif*": 2}))

    def test_structural_line_exemptions(self):
        """Display maths and plain headings before a paragraph are never boilerplate;
        short lines followed by short lines (category runs) still are."""
        sl = data_prep._structural_line
        para = "We fitted a linear mixed model to every trait and estimated the additive variance from the kinship."
        self.assertTrue(len(para) >= 80)
        for line in ("## Materials and Methods", "-----", "MATERIALS AND METHODS",
                     "$$ \\hat{\\beta} = (X^T X)^{-1} X^T y $$", "\\begin{equation} E = mc^2", "x \\end{align}"):
            self.assertTrue(sl(line, "Category: Genetics"), line)
        self.assertTrue(sl("Materials and Methods", para))
        self.assertTrue(sl("Etymology", para))
        self.assertFalse(sl("Materials and Methods", "Category: Genetics"), "a short line after a short line")
        self.assertFalse(sl("Materials and Methods"), "without the next line the heading rule is off")
        self.assertFalse(sl("Materials and Methods", ""))
        self.assertFalse(sl("This item is not available for download.", para), "a sentence is not a heading")
        self.assertFalse(sl("one two three four five six seven eight nine", para), "more than 8 words")
        self.assertFalse(sl("Please sign in to access the full text", "Category: x"))
        lines = ["Materials and Methods", "", para, "Category: Genetics", "", "Category: Biology"]
        self.assertEqual(data_prep._next_nonblank(lines, 0), para)
        self.assertEqual(data_prep._next_nonblank(lines, 3), "Category: Biology")
        self.assertEqual(data_prep._next_nonblank(lines, 5), "")
        # the cleaning fast path gives the same text as the regexes it replaced
        import collections
        for text in ("a  \n b\t\n\n\n\nc \n", "plain\ntext", "x\n\n\n\n\ny", "tabs\t\t\nend  "):
            stats = collections.Counter()
            want = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+(?=\n)", "", text)).strip()
            self.assertEqual(data_prep.clean_text(text, stats), want, repr(text))

    def test_leakage_move_updates_val_chars(self):
        import collections
        docs = [["md", "a", "A", 1000, 0, 0, None, None], ["md", "b", "B", 2000, 0, 0, None, None],
                ["md", "c", "C", 300, 0, 0, None, None], ["md", "d", "D", 400, 0, 0, None, None],
                ["web", "e", "E", 500, 0, 0, None, None]]
        roots = np.asarray([0, 1, 1, 3, 4], dtype=np.int32)  # b and c are a near-duplicate group
        clusters = {1: [1, 2]}
        split_of = np.asarray([1, 2, 2, 2, 2], dtype=np.int8)  # a train; b, c, d, e validation
        dropped = np.zeros(5, dtype=bool)
        sources = {"md": {"val_chars": 2700, "train_chars": 1000}, "web": {"val_chars": 500, "train_chars": 0}}
        forced = collections.Counter()
        rows = [(0.9, "md", "c", "C", 20, 2), (0.6, "md", "d", "D", 12, 3), (0.7, "web", "e", "E", 9, 4),
                (0.1, "md", "b", "B", 30, 1)]
        out = data_prep._move_leaks_to_train(rows, docs, roots, clusters, split_of, dropped, {4}, sources, forced)
        self.assertEqual([r[6] for r in out], [1, 1, 0, 0], "e is fixed as validation by --keep-split-from")
        self.assertEqual(split_of.tolist(), [1, 1, 1, 1, 2], "c's whole group (b, c) moved with it")
        self.assertEqual(sources["md"], {"val_chars": 0, "train_chars": 3700})
        self.assertEqual(sources["web"], {"val_chars": 500, "train_chars": 0})
        self.assertEqual(forced["leaks_into_train"], 3)

    def test_tokenizer_window_sampler_covers_every_source(self):
        """Window sampling gives each source its share in characters; sampling whole
        documents left a one-book source out with probability 0.99."""
        train_chars = {"books": 2_000_000, "web": 200_000_000, "notes": 50_000}
        shares = data_prep.tokenizer_shares(train_chars, {"books": 3.0})
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertAlmostEqual(shares["books"] / shares["web"], 3 * 2 / 200)
        sample = 10_000_000
        probs = {n: min(1.0, sample * shares[n] / train_chars[n]) for n in train_chars}
        self.assertAlmostEqual(probs["books"], sample * 3 / (3 * 2_000_000 + 200_000_000 + 50_000))
        book = "\n".join(paragraphs(k, 60) for k in range(80))  # one long document (~870k characters)
        kept = data_prep.tokenizer_sample_lines(book, "books", "the_book.md", 1234, probs["books"])
        chars = sum(len(p) for p in kept)
        self.assertGreater(chars, 0, "the single book contributes windows")
        self.assertLess(abs(chars - probs["books"] * len(book)) / (probs["books"] * len(book)), 0.25)
        self.assertEqual(kept, data_prep.tokenizer_sample_lines(book, "books", "the_book.md", 1234, probs["books"]))
        self.assertNotEqual(kept, data_prep.tokenizer_sample_lines(book, "books", "other.md", 1234, probs["books"]))
        self.assertEqual(data_prep.tokenizer_sample_lines(book, "books", "x", 1, 1.0),
                         [ln for ln in book.split("\n") if ln.strip()], "probability 1 keeps every non-blank line")
        self.assertEqual(data_prep.tokenizer_sample_lines(book, "books", "x", 1, 0.0), [])
        long_line = "w" * 10_000
        self.assertEqual([len(p) for p in data_prep.tokenizer_sample_lines(long_line, "s", "x", 1, 1.0)],
                         [4000, 4000, 2000], "a long line is cut into window-sized pieces")

    def test_markdown_ids_use_forward_slashes(self):
        import collections
        root = os.path.join(self.tmp, "nested_ids")
        os.makedirs(os.path.join(root, "a", "b"))
        for name in (os.path.join("a", "b", "deep.md"), "top.md"):
            with open(os.path.join(root, name), "w", encoding="utf-8") as handle:
                handle.write(BIO)
        stats = {"scanned": 0, "skipped": collections.Counter(), "cleaning": collections.Counter()}
        ids = [d[0] for d in data_prep.iter_markdown([root], ["*.md"], [], True, stats, [])]
        self.assertEqual(sorted(ids), ["a/b/deep.md", "top.md"])
        second = os.path.join(self.tmp, "nested_ids_2")
        os.makedirs(second)
        with open(os.path.join(second, "top.md"), "w", encoding="utf-8") as handle:
            handle.write(BIO)
        ids = [d[0] for d in data_prep.iter_markdown([root, second], ["*.md"], [], True, stats, [])]
        self.assertEqual(sorted(ids), ["nested_ids/a/b/deep.md", "nested_ids/top.md", "nested_ids_2/top.md"])

    def test_lsh_joins_chains_through_the_previous_member(self):
        """A ~ B ~ C with A !~ C: C shares a bucket with A and B in bands 6-9 only; the
        comparison with the bucket's first member (A) fails, the one with the previous
        member (B) joins it. Signatures are band-major (bands, n, rows)."""
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2 ** 32, 128, dtype=np.uint64).astype(np.uint32)
        c = rng.integers(0, 2 ** 32, 128, dtype=np.uint64).astype(np.uint32)
        c[48:80] = a[48:80]                      # bands 6-9 shared by all three
        b = a.copy()
        b[80:] = c[80:]                          # B = A on 0-79, = C on 80-127 ...
        b[80::8] += 1                            # ... except one value per band, so B and C share no extra bucket
        sims = {k: float((x == y).mean()) for k, (x, y) in {"ab": (a, b), "bc": (b, c), "ac": (a, c)}.items()}
        self.assertAlmostEqual(sims["ab"], 80 / 128)
        self.assertAlmostEqual(sims["bc"], 74 / 128)
        self.assertAlmostEqual(sims["ac"], 32 / 128)
        sig = np.stack([a, b, c]).reshape(3, 16, 8).transpose(1, 0, 2).copy()
        self.assertEqual(data_prep.lsh_clusters(sig, 0.55).tolist(), [0, 0, 0])
        self.assertEqual(data_prep.lsh_clusters(sig, 0.6).tolist(), [0, 0, 2], "B ~ C below the threshold")
        self.assertEqual(data_prep.lsh_clusters(sig, 0.55, active=[0, 2]).tolist(), [0, 1, 2])
        self.assertEqual(data_prep.signature_similarity(sig, np.asarray([1, 2]), np.asarray([0, 0])).tolist(),
                         [80 / 128, 32 / 128])
        path = os.path.join(self.tmp, "signatures.npy")
        mm = data_prep.open_signatures(path, 3)
        for i, s in enumerate((a, b, c)):
            mm[:, i, :] = s.reshape(16, 8)
        self.assertEqual(data_prep.lsh_clusters(mm, 0.55).tolist(), [0, 0, 0])
        del mm
        uf = data_prep.UnionFind(5)
        uf.union(4, 1)
        uf.union(3, 4)
        self.assertEqual(uf.roots().tolist(), [0, 1, 2, 1, 1])
        self.assertEqual(uf.parent.dtype, np.int32)

    def test_rope_rotation_in_fp32_rounded_once(self):
        """RoPE rotates in fp32 and rounds once: a bf16 input gives exactly the fp32 rotation cast to bf16, and an
        fp32 input the fp32 rotation itself."""
        rope = tiny_gpt.RotaryEmbedding(8, 32)
        torch.manual_seed(0)
        q, k = torch.randn(2, 3, 20, 8), torch.randn(2, 3, 20, 8)
        cos, sin = rope.cos[:, :, 4:24], rope.sin[:, :, 4:24]
        want_q = q * cos + tiny_gpt.rotate_half(q) * sin
        want_k = k * cos + tiny_gpt.rotate_half(k) * sin
        got_q, got_k = rope(q, k, pos=4)
        self.assertTrue(torch.equal(got_q, want_q) and torch.equal(got_k, want_k))
        qb, kb = q.to(torch.bfloat16), k.to(torch.bfloat16)
        got_qb, got_kb = rope(qb, kb, pos=4)
        self.assertEqual((got_qb.dtype, got_kb.dtype), (torch.bfloat16, torch.bfloat16))
        self.assertTrue(torch.equal(got_qb, (qb.float() * cos + tiny_gpt.rotate_half(qb.float()) * sin).to(torch.bfloat16)))
        self.assertTrue(torch.equal(got_kb, (kb.float() * cos + tiny_gpt.rotate_half(kb.float()) * sin).to(torch.bfloat16)))
        tiny_gpt.free_device_memory(torch.device("cpu"))  # a no-op on the CPU, never an error

    def test_safetensors_round_trip_all_dtypes(self):
        """write_safetensors/read_safetensors agree for every supported dtype, a non-contiguous view, a 0-d
        tensor and a tensor with no elements."""
        tensors = {}
        for dtype in tiny_gpt._ST_DTYPES:
            if dtype == torch.bool:
                tensors["t_bool"] = torch.tensor([[True, False, True], [False, False, True]])
            elif dtype.is_floating_point:
                tensors[f"t_{str(dtype).split('.')[-1]}"] = torch.randn(3, 5).to(dtype)
            else:
                tensors[f"t_{str(dtype).split('.')[-1]}"] = torch.arange(-6, 9).reshape(3, 5).to(dtype)
        tensors["strided"] = torch.arange(24.0).reshape(4, 6)[:, ::2]
        tensors["scalar"] = torch.tensor(2.5)
        tensors["empty"] = torch.zeros(0, 7)
        path = os.path.join(self.tmp, "all.safetensors")
        tiny_gpt.write_safetensors(tensors, path, {"note": "x"})
        back, meta = tiny_gpt.read_safetensors(path)
        self.assertEqual(meta, {"note": "x"})
        self.assertEqual(sorted(back), sorted(tensors))
        for name, t in tensors.items():
            self.assertEqual((back[name].dtype, tuple(back[name].shape)), (t.dtype, tuple(t.shape)), name)
            self.assertTrue(torch.equal(back[name], t), name)

    def test_evaluate_matches_by_hand(self):
        """evaluate(): per-source loss, balanced and mix-weighted means, bits per byte, top-1 and the expected
        calibration error agree with a by-hand log-softmax."""
        model = self.small_model(seed=3)
        rng = np.random.default_rng(1)
        arrays = {"a": rng.integers(0, 50, 200, dtype=np.int64), "b": rng.integers(0, 50, 90, dtype=np.int64)}
        evalset = tiny_gpt.EvalSet(arrays, 16, 32)
        self.assertEqual({n: w.shape[0] for n, w in evalset.windows.items()}, {"a": 2, "b": 2})
        size = torch.from_numpy(rng.integers(1, 5, 50).astype(np.int64))
        marked = torch.from_numpy(rng.random(50) < 0.5)
        line_break = torch.from_numpy(rng.random(50) < 0.5)
        share = {"a": 0.75, "b": 0.25}
        got = tiny_gpt.evaluate(model, evalset, torch.device("cpu"), None, 1, byte_tables=(size, marked, line_break),
                                share=share)
        loss, bpb, correct, total = {}, {}, 0, 0
        conf_sum, acc_sum = np.zeros(10), np.zeros(10)
        with torch.no_grad():
            for name, windows in evalset.windows.items():
                x, y = windows[:, :-1], windows[:, 1:]
                logp = torch.log_softmax(model(x).float(), -1).double()
                nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
                loss[name] = nll.mean().item()
                n_bytes = int(size[y].sum()) - int((marked[y] & line_break[x]).sum())
                bpb[name] = nll.sum().item() / (math.log(2) * n_bytes)
                conf, pred = logp.exp().max(-1)
                correct += int((pred == y).sum())
                total += y.numel()
                for c, ok in zip(conf.flatten().tolist(), (pred == y).flatten().tolist()):
                    b = int(min(c, 1 - 1e-9) * 10)
                    conf_sum[b] += c
                    acc_sum[b] += float(ok)
        for name in loss:
            self.assertAlmostEqual(got["per_source"][name], loss[name], places=6)
            self.assertAlmostEqual(got["bpb_per_source"][name], bpb[name], places=6)
        self.assertAlmostEqual(got["loss"], (loss["a"] + loss["b"]) / 2, places=6)
        self.assertAlmostEqual(got["loss_mix"], 0.75 * loss["a"] + 0.25 * loss["b"], places=6)
        self.assertAlmostEqual(got["bpb"], (bpb["a"] + bpb["b"]) / 2, places=6)
        self.assertAlmostEqual(got["bpb_mix"], 0.75 * bpb["a"] + 0.25 * bpb["b"], places=6)
        self.assertAlmostEqual(got["top1"], correct / total, places=9)
        self.assertAlmostEqual(got["ece"], float(np.abs(conf_sum - acc_sum).sum() / total), places=6)
        self.assertEqual(got["tokens"], total)
        # without a share there is no mix-weighted number; a share that misses a source gives it weight 0
        self.assertIsNone(tiny_gpt.evaluate(model, evalset, torch.device("cpu"), None, 2)["loss_mix"])
        self.assertAlmostEqual(tiny_gpt.evaluate(model, evalset, torch.device("cpu"), None, 2,
                                                 share={"a": 0.4})["loss_mix"], loss["a"], places=6)

    def test_atomic_save_reports_the_fallback_path(self):
        path = os.path.join(self.tmp, "x.pt")
        self.assertEqual(tiny_gpt.atomic_save({"a": torch.ones(2)}, path), path)
        real, calls = os.replace, []

        def locked(src, dst):
            calls.append(dst)
            if dst == path:
                raise PermissionError("in use")
            return real(src, dst)

        with unittest.mock.patch("os.replace", side_effect=locked), unittest.mock.patch("time.sleep"), \
                redirect_stdout(io.StringIO()) as buf:
            written = tiny_gpt.atomic_save({"a": torch.zeros(2)}, path)
        self.assertNotEqual(written, path)
        self.assertRegex(os.path.basename(written), r"^x\.\d{6}\.pt$")
        self.assertIn("locked", buf.getvalue())
        self.assertTrue(torch.equal(torch.load(written, weights_only=True)["a"], torch.zeros(2)))
        self.assertTrue(torch.equal(torch.load(path, weights_only=True)["a"], torch.ones(2)), "old file intact")
        self.assertFalse(glob.glob(path + ".tmp-*"), "no temporary file left")
        os.utime(written, (os.path.getmtime(path) + 60,) * 2)
        self.assertEqual(tiny_gpt.newer_fallbacks(path), [written])

    def test_train_lock_refuses_a_second_trainer(self):
        import argparse
        args = argparse.Namespace(out_dir=os.path.join(self.tmp, "runs_lock"), name="R")
        lock = tiny_gpt.acquire_run_lock(args)
        self.assertTrue(os.path.isfile(lock))
        self.assertEqual(tiny_gpt.acquire_run_lock(args), lock, "a lock left by this process is stale")
        with open(lock, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getppid(), "started": None}, f)
        with self.assertRaises(SystemExit) as ctx:
            tiny_gpt.acquire_run_lock(args)
        self.assertIn("being trained by another process", str(ctx.exception.code))
        os.remove(lock)

    def test_score_probes_template_wraps_after_shuffling(self):
        model, tok, cpu = self.small_model(), CharTok(), torch.device("cpu")
        probes = [{"prompt": "Genes are made of long chains", "answer": "DNA", "distractors": ["fat", "salt"],
                   "category": "genetics", "difficulty": "easy"},
                  {"prompt": "Cells divide by a process called", "answer": "mitosis", "distractors": ["osmosis"],
                   "category": "cells", "difficulty": "easy"}]
        template = "### Question\n{prompt}\n### Answer\n"
        _, got = tiny_gpt.score_probes(model, tok, probes, cpu, None, floor=True, template=template)
        wrap = lambda text: [dict(p, prompt=template.format(prompt=text(p))) for p in probes]  # noqa: E731
        _, plain = tiny_gpt.score_probes(model, tok, wrap(lambda p: p["prompt"]), cpu, None, floor=False)
        self.assertEqual([r["prompt"] for r in got], [p["prompt"] for p in probes], "results keep the prompt")
        for a, b in zip(got, plain):
            self.assertAlmostEqual(a["correct_score"], b["correct_score"], places=5)
        floors = [0.0] * len(probes)
        for v in range(1, tiny_gpt.FLOOR_SHUFFLES + 1):  # only the question's words shuffled, template intact
            _, rows = tiny_gpt.score_probes(model, tok, wrap(lambda p: tiny_gpt.shuffled_prompt(p["prompt"], v)),
                                            cpu, None, floor=False)
            for i, r in enumerate(rows):
                floors[i] += r["correct"] / tiny_gpt.FLOOR_SHUFFLES
        for r, f in zip(got, floors):
            self.assertAlmostEqual(r["floor_correct"], f)

    # ---------------- data_prep ----------------
    def test_keyword_prefilter_with_self_overlapping_phrases(self):
        import pyarrow as pa
        import pyarrow.compute as pc
        kw = data_prep.KeywordFilter(["Gallus gallus", "Johnson & Johnson", "Loa loa", "genetic*"], [],
                                     min_hits=1, min_distinct=1)

        def re2(text):
            return sum(int(pc.count_substring_regex(pa.array([text]), pattern=p, ignore_case=True)[0].as_py())
                       for p in kw.include_patterns)

        for text, exact in (("Gallus gallus gallus is the red junglefowl.", None),
                            ("gallus " * 5, None), ("Johnson & Johnson makes genetic tests.", None),
                            ("Loa loa loa loa", None),
                            ("Boris Johnson spoke. Later Johnson spoke again, and Johnson left.", 0)):
            python = kw.evaluate("", text)[2]
            self.assertGreaterEqual(re2(text), python, text)
            if exact is not None:
                self.assertEqual((re2(text), python), (exact, exact), "stray words of a phrase are not hits")

    def test_shingles_are_identical_across_windows(self):
        text = (BIO + " " + paragraphs(5, 30) + " Ünïcode wörds ") * 3
        whole = data_prep.ShingleHasher()
        small = data_prep.ShingleHasher()
        small.WINDOW, small.BLOCK = 97, 13
        self.assertTrue(np.array_equal(whole.word_hashes(text), small.word_hashes(text)))
        self.assertTrue(np.array_equal(whole.shingles(text, 5), small.shingles(text, 5)))

    def test_text_cleaning_and_file_listing(self):
        import collections
        stats = collections.Counter()
        self.assertEqual(data_prep.clean_text("a\u2581b \u2581c", stats), "a_b _c")
        self.assertEqual((stats["u2581_replaced_docs"], stats["u2581_replaced_chars"]), (1, 2))
        self.assertEqual(data_prep._SURROGATE_RE.sub("\ufffd", "x\ud83dy"), "x\ufffdy")
        folder = os.path.join(self.tmp, "Books [2024]")
        os.makedirs(os.path.join(folder, "sub"))
        for name in ("Notes.MD", "a.md", "skip.md", os.path.join("sub", "b.md"), "x.pdf", ".hidden.md"):
            open(os.path.join(folder, name), "w").close()
        flat = [os.path.basename(p) for p in data_prep.list_text_files(folder, ["*.md"])]
        self.assertEqual(sorted(flat), ["Notes.MD", "a.md", "skip.md"])
        deep = data_prep.list_text_files(folder, ["*.md"], recursive=True)
        self.assertEqual(len(deep), 4)
        kept = [p for p in deep if not data_prep._excluded(p, os.path.relpath(p, folder), ["skip.md"])]
        self.assertEqual(len(kept), 3)
        self.assertEqual(data_prep.parse_tokenizer_weights("MD=3", ["md", "wiki"]), {"md": 3.0})
        for bad in ("md=x", "books=3", "=2"):
            with self.assertRaises(ValueError):
                data_prep.parse_tokenizer_weights(bad, ["md", "wiki"])
        self.assertEqual(data_prep._split_key("md", "a\\b\tc.md"), data_prep._split_key("md", "a/b c.md"))
        root = os.path.join(self.tmp, "root")
        os.makedirs(os.path.join(root, "lecture notes"))
        os.makedirs(os.path.join(root, "books"))
        found, skipped = data_prep.text_root_folders(root)
        self.assertEqual([n for n, _ in found], ["books"])
        self.assertEqual([n for n, _ in skipped], ["lecture notes"])

    # ---------------- converter ----------------
    def test_converter_text_helpers(self):
        import convert_to_markdown as conv
        greek = "Heritability h² of αβγ traits\n"
        cases = {"bom16.txt": b"\xff\xfe" + greek.encode("utf-16-le"), "le16.txt": greek.encode("utf-16-le"),
                 "stray.txt": greek.encode("utf-8") + b"bad \x96 byte\n"}
        for name, raw in cases.items():
            path = os.path.join(self.tmp, name)
            with open(path, "wb") as f:
                f.write(raw)
            text = conv.read_text(path)
            self.assertTrue(text.startswith(greek.rstrip("\n")), (name, text[:40]))
        self.assertEqual(conv.read_text(os.path.join(self.tmp, "stray.txt")).count("\ufffd"), 1)
        md = ("Intro<br>text\n| a<br>b | c |\n<!-- Start of picture text -->tick 1 2 3<!-- End of picture text -->\n"
              "![](C:/x/paper (1)_images/paper (1).pdf-0001-03.png)\nEnd")
        clean = conv._clean_pdf_markdown(md)
        self.assertIn("Intro\ntext", clean)
        self.assertIn("| a b | c |", clean)
        self.assertNotIn("tick", clean)
        self.assertNotIn("![", clean)
        self.assertNotIn("png", clean)
        linked = conv._clean_pdf_markdown(md, image_dir=os.path.join(self.tmp, "paper (1)_images"))
        self.assertIn("![](<paper (1)_images/paper (1).pdf-0001-03.png>)", linked)
        self.assertEqual(conv.sniff_delimiter("trait;h2;se"), ";")
        self.assertEqual(conv.sniff_delimiter("a,b;c"), ",")
        self.assertEqual(conv.sniff_delimiter("a\tb,c"), "\t")
        self.assertEqual(conv._xlsx_number("45366", (True, False), False), "2024-03-15")
        self.assertEqual(conv._xlsx_number("0.30000000000000004", None, False), "0.3")
        self.assertEqual(conv._neglog10("0"), math.inf)
        self.assertAlmostEqual(conv._neglog10("1e-400"), 400)
        table = conv.md_table([[f"c{i}" for i in range(100)], [str(i) for i in range(100)]], 10, max_cols=30)
        self.assertEqual(table.splitlines()[0].count("|"), 31)
        self.assertIn("70 more columns not shown (limit --max-cols 30)", table)
        import gzip
        import argparse
        vcf = os.path.join(self.tmp, "ieu-a-2.vcf.gz")
        with gzip.open(vcf, "wt") as f:
            f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\n1\t100\trs1\n")
        with self.assertRaises(RuntimeError) as ctx:
            conv.convert_table_or_text(vcf, argparse.Namespace(no_gwas=False, max_rows=200, max_cols=30))
        self.assertIn("data, not text", str(ctx.exception))
        semi = os.path.join(self.tmp, "eu.csv")
        with open(semi, "w", encoding="utf-8") as f:
            f.write("trait;h2;se\nheight;0,8;0,02\n")
        text, kind = conv.convert_table_or_text(semi, argparse.Namespace(no_gwas=False, max_rows=200, max_cols=30))
        self.assertEqual(kind, "table")
        self.assertIn("| height | 0,8 | 0,02 |", text)

    def test_page_furniture_keeps_acronym_headings(self):
        import convert_to_markdown as conv
        romans = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
        pages = [f"## MCMC\nBody text {k} about linkage and sampling.\n{r}" for k, r in enumerate(romans)]
        pages[3] = pages[3].replace("## MCMC", "## LD")
        kept = "\n".join(conv.strip_page_furniture(pages))
        self.assertEqual(kept.count("## MCMC"), 7)
        self.assertIn("## LD", kept)
        self.assertFalse(re.search(r"^(i|ii|iii|iv|v|vi|vii|viii)$", kept, re.M), "stepping roman numerals go")
        two = conv.strip_page_furniture(["Page 1 of 2\nFirst page text here.", "Page 2 of 2\nSecond page text."])
        self.assertNotIn("Page", "\n".join(two))

    # ---------------- downloaders ----------------
    def test_back_matter_headings(self):
        import download_pmc
        drop = ("Funding and competing interests", "Author Contributions Statement", "Acknowledgements",
                "5. References", "Funding/Support", "Role of the Funder/Sponsor", "Additional file 1",
                "Supplementary Material for Section 3", "Data availability statement")
        keep = ("Contributions of rare variants", "Correspondence analysis", "Ethics of germline editing",
                "Consent and data sharing", "Supplementary data analysis", "Contributions",
                "Support vector machines", "Declaration of Helsinki", "2. Methods")
        for heading in drop:
            self.assertTrue(download_pmc.SKIP_SECTION_RE.search(heading), heading)
        for heading in keep:
            self.assertFalse(download_pmc.SKIP_SECTION_RE.search(heading), heading)

    def test_youtube_error_status(self):
        import youtube_transcripts as yt
        self.assertEqual(yt.error_status("Video unavailable. This content isn't available, try again later. The "
                                         "current session has been rate-limited by YouTube"), "blocked")
        self.assertEqual(yt.error_status("ERROR: Requested format is not available"), "error")
        self.assertEqual(yt.error_status("Premieres in 3 hours"), "upcoming")
        self.assertEqual(yt.error_status("Private video. Sign in if you've been granted access"), "unavailable")
        self.assertEqual(yt.error_status("Sign in to confirm your age"), "unavailable")
        report = os.path.join(self.tmp, "report.tsv")
        with open(report, "w", encoding="utf-8") as f:
            f.write("video_id\tstatus\tsource\n"
                    "a\tunavailable\tRequested format is not available\n"
                    "b\tunavailable\tPrivate video\n"
                    "c\tunavailable\tPremieres in 2 days\n")
        self.assertEqual(yt.read_status(report), {"a": "error", "b": "unavailable", "c": "error"})

    def test_arxiv_cache_and_main_file(self):
        import gzip
        import download_arxiv as ax
        cache = os.path.join(self.tmp, "arxiv_cache")
        os.makedirs(cache)
        path = os.path.join(cache, ax._set_file("q-bio") + ".jsonl.gz")
        rec = lambda pid, title: json.dumps({"id": pid, "title": title}) + "\n"  # noqa: E731
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(rec("1", "v1") + rec("2", "only"))
        with gzip.open(path, "at", encoding="utf-8") as f:  # an --update appends the revised paper
            f.write(rec("1", "v2"))
        self.assertEqual([(r["id"], r["title"]) for r in ax.read_cache(cache, ["q-bio"])], [("2", "only"), ("1", "v2")])
        with open(path, "ab") as f:  # a gzip member cut off by a crash
            f.write(gzip.compress(rec("3", "lost").encode() * 50)[:40])
        with redirect_stdout(io.StringIO()) as buf:
            got = sorted(r["id"] for r in ax.read_cache(cache, ["q-bio"]))
        self.assertEqual(got[:2], ["1", "2"], "the records before the damage are used")
        self.assertIn("damaged", buf.getvalue())
        section = "\\documentclass[../main.tex]{subfiles}\n\\begin{document}\n" + "Long section text. " * 500 + \
                  "\n\\end{document}"
        files = {"main.tex": "\\documentclass{article}\n\\begin{document}\n\\subfile{sections/intro}\n\\end{document}",
                 "sections/intro.tex": section,
                 "fig.tex": "\\documentclass{standalone}\n\\begin{document}x\\end{document}",
                 "notes.tex": "% \\begin{document}\n" + "notes " * 2000}
        self.assertEqual(ax.main_tex(files), "main.tex")

    # ---------------- viewer ----------------
    def test_viewer_text_helpers(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import view_pt
        fig = plt.figure()
        for piece in ("$$", "$}$", "$^$", "$x$"):
            label = view_pt.label_text(piece)
            fig.text(0.1, 0.5, label)
            fig.canvas.draw()  # mathtext would raise on these
        plt.close(fig)
        home = os.path.expanduser("~")
        for variant in (home, home.replace("\\", "/"), home.upper(), home.replace("\\", "\\\\")):
            self.assertEqual(view_pt.scrub(f"cannot open {variant}{os.sep}x.pt"), f"cannot open ~{os.sep}x.pt")
        out = os.path.join(self.tmp, "dash.png")

        def write(p):
            with open(p, "wb") as f:
                f.write(b"png")

        self.assertEqual(view_pt.save_replacing(write, out), out)
        real = os.replace

        def locked(src, dst):
            if dst == out:
                raise PermissionError("in use")
            return real(src, dst)

        with unittest.mock.patch("os.replace", side_effect=locked), unittest.mock.patch("time.sleep"), \
                redirect_stdout(io.StringIO()):
            fallback = view_pt.save_replacing(write, out)
        self.assertNotEqual(fallback, out)
        self.assertTrue(os.path.isfile(fallback))
        self.assertFalse(glob.glob(os.path.join(self.tmp, "dash.tmp-*")))

    def test_viewer_history_from_log_names_the_run(self):
        import view_pt
        bar = "=" * 30
        log = os.path.join(self.tmp, "view_log.txt")
        with open(log, "w", encoding="utf-8") as f:
            f.write(f"\n{bar}\nRUN 1\ncheckpoints: C:\\x\\A.pt (best C:\\x\\A_best.pt)\n"
                    "step     8/24 | train 5.1000 | val 5.2000 @8 | lr 1.0e-03\n"
                    f"\n{bar}\nRUN 2\ncheckpoints: C:\\x\\B.pt (best C:\\x\\B_best.pt)\n"
                    "step     8/24 | train 4.0000 | val 4.1000 @8 | lr 1.0e-03\n")
        hist, label = view_pt.history_from_log(log, "/home/u/x/A_best.pt", None, "tinygpt")
        self.assertEqual([h["val_loss"] for h in hist], [5.2])
        self.assertIn("A", label)
        hist, label = view_pt.history_from_log(log, "C.pt", None, "tinygpt")
        self.assertEqual(hist, [])
        self.assertIn("names C.pt", label)
        hist, _ = view_pt.history_from_log(log, "C.pt", None, None)
        self.assertEqual([h["val_loss"] for h in hist], [4.1], "an unknown family falls back to the last run")
        self.assertEqual(view_pt.history_from_log(log, "A.pt", None, "legacy-v3")[0], [])

    def test_viewer_statistics_in_pieces(self):
        import view_pt
        model = self.small_model(vocab=60)
        windows = torch.randint(0, 60, (5, 17), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            big = view_pt.forward_stats(model, windows, 60, torch.device("cpu"))
            small = view_pt.forward_stats(model, windows, 60, torch.device("cpu"), token_budget=16, rows=5)
        self.assertAlmostEqual(big["loss"], small["loss"], places=6)
        self.assertEqual(big["calibration"]["count"], small["calibration"]["count"])
        self.assertLess(np.abs(big["mean_prob"] - small["mean_prob"]).max(), 1e-6)
        with torch.no_grad():
            ref = -torch.log_softmax(model(windows[:, :-1]).float(), -1).gather(
                -1, windows[:, 1:, None]).mean().item()
        self.assertAlmostEqual(big["loss"], ref, places=5)
        a = view_pt.attention_entropy(model, windows[0], torch.device("cpu"))
        b = view_pt.attention_entropy(model, windows[0], torch.device("cpu"), budget=1)
        self.assertLess(np.abs(a - b).max(), 1e-6)

    # ---------------- post-training and orchestration ----------------
    def test_finetune_length_grouping_and_micro_batch_check(self):
        import random
        import finetune
        items = [(list(range(1, 2 + (i * 7) % 23)), None) for i in range(53)]
        batches = finetune.length_grouped_batches(items, 4, random.Random(0), group=3)
        self.assertEqual(len(batches), math.ceil(53 / 4))
        used = sorted(id(item) for b in batches for item in b)
        self.assertEqual(used, sorted(id(item) for item in items), "every example used exactly once")
        data = os.path.join(self.tmp, "sft.jsonl")
        open(data, "w").close()
        log = os.path.join(self.tmp, "ft_log.txt")
        for bad in ("0", "-1", "9"):
            with open(os.devnull, "w") as devnull, unittest.mock.patch("sys.stderr", devnull), \
                    redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                finetune.main(["sft", "--base", data, "--data", data, "--name", "x", "--batch", "8",
                               "--micro-batch", bad, "--log-file", log, "--device", "cpu"])
            self.assertEqual(ctx.exception.code, 2, bad)

    def test_full_loss_shares_one_logsumexp(self):
        """The full-logits loss and its z-loss come from one logsumexp: the same value and gradients as
        F.cross_entropy plus a second logsumexp; the piece-wise (eager) loss agrees with it."""
        torch.manual_seed(0)
        logits = torch.randn(37, 300, requires_grad=True)
        y = torch.randint(0, 300, (37,))
        loss, lse = tiny_gpt.full_lm_loss(logits, y)
        total = loss + 1e-4 * lse.pow(2).mean()
        grad, = torch.autograd.grad(total, logits)
        want = (torch.nn.functional.cross_entropy(logits, y)
                + 1e-4 * torch.logsumexp(logits, -1).pow(2).mean())
        want_grad, = torch.autograd.grad(want, logits)
        self.assertTrue(torch.allclose(total, want, atol=1e-6), (total.item(), want.item()))
        self.assertTrue(torch.allclose(grad, want_grad, atol=1e-6), (grad - want_grad).abs().max().item())
        h, w = torch.randn(37, 16), torch.randn(300, 16)
        ce, z, n = tiny_gpt.chunked_lm_loss(h, w, y, 10)
        full, lse2 = tiny_gpt.full_lm_loss(h @ w.t(), y)
        self.assertEqual(int(n), 37)
        self.assertTrue(torch.allclose(ce / n, full, atol=1e-5))
        self.assertTrue(torch.allclose(z / n, lse2.pow(2).mean(), atol=1e-4))
        self.assertIs(tiny_gpt._chunk_loss_fn, tiny_gpt._chunk_loss, "eager piece-wise loss on the CPU")

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("TINYGPT_GPU_TESTS"),
                         "needs CUDA and TINYGPT_GPU_TESTS=1 (never while a training run occupies the GPU)")
    def test_compiled_chunk_loss_matches_eager(self):
        """The compiled piece-wise loss (used under torch.compile on CUDA) gives the eager values and gradients,
        -100 labels included."""
        dev = torch.device("cuda")
        torch.manual_seed(0)
        h = torch.randn(64, 32, device=dev, requires_grad=True)
        w = torch.randn(500, 32, device=dev, requires_grad=True)
        y = torch.randint(0, 500, (64,), device=dev)
        y[:3] = -100

        def run_loss(compiled):
            tiny_gpt.set_chunk_loss_compiled(compiled)
            try:
                h.grad = w.grad = None
                ce, z, n = tiny_gpt.chunked_lm_loss(h, w, y, 24)
                (ce / n + 1e-4 * z / n).backward()
                return ce.detach().clone(), z.detach().clone(), h.grad.clone(), w.grad.clone()
            finally:
                tiny_gpt.set_chunk_loss_compiled(False)

        for a, b in zip(run_loss(False), run_loss(True)):
            self.assertTrue(torch.allclose(a, b, atol=1e-4, rtol=1e-4), (a - b).abs().max().item())

    def test_speed_probe_flops_and_peak_memory(self):
        """The planner's speed probe times micro-steps and one optimizer update per step with the planned
        optimizer and returns the peak memory; the FLOP count carries the recompute of activation
        checkpointing; peak_memory_bytes is a running maximum."""
        cfg = tiny_gpt.ModelConfig(vocab_size=300, ctx=32, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
        cpu = torch.device("cpu")
        n = tiny_gpt.count_params(cfg)[0]
        self.assertEqual(tiny_gpt.flops_per_token(cfg), 6 * n + 12 * 2 * 32 * 64)
        self.assertEqual(tiny_gpt.flops_per_token(cfg, True), tiny_gpt.flops_per_token(cfg) + 2 * n + 4 * 2 * 32 * 64)
        built = []
        real = tiny_gpt.build_optimizer

        def spy(model, settings, device):
            built.append(settings["optimizer"])
            return real(model, settings, device)

        with unittest.mock.patch.object(tiny_gpt, "build_optimizer", side_effect=spy):
            sec, peak = tiny_gpt.measure_seconds_per_step(cfg, 2, 3, cpu, None, warmup=1, timed=2, settings={
                "optimizer": "muon", "lr": 1e-3, "weight_decay": 0.1, "betas": [0.9, 0.95]})
            sec_ck, peak_ck = tiny_gpt.measure_seconds_per_step(cfg, 2, 3, cpu, None, warmup=1, timed=2,
                                                                checkpointing=True, loss_chunk=16)
        self.assertEqual(built, ["muon", "adamw"])
        self.assertGreater(sec, 0.0)
        self.assertGreater(sec_ck, 0.0)
        self.assertGreater(peak, 0)
        self.assertGreater(peak_ck, 0)
        self.assertEqual(tiny_gpt.peak_memory_bytes(cpu, 10 ** 15), 10 ** 15)
        self.assertGreater(tiny_gpt.peak_memory_bytes(cpu), 0)

    def test_time_budget_ladder_search(self):
        """plan_with_time_budget without an explicit shape: the ladder binary search (mocked speed test, bigger
        is slower) picks the largest shape whose tokens >= r x N fit the budget, with the device's hot factor
        and the compile correction; a hopeless budget falls to the smallest shape with a note; a probe peak above
        the memory budget steps to the next memory option; a recorded sustained speed caps the step time."""
        manifest = {"tokenizer": {"vocab_size": 16384},
                    "sources": {"md": {"train": {"tokens": 1_000_000_000}, "doc_tokens_percentiles": {"90": 1000}}}}
        budget = (64 * 2 ** 30, "64 GiB (test)")
        probes = []
        peak = {"gib": 1.0}

        def fake_measure(cfg, micro, accum, device, amp_dtype, attn_pref="auto", warmup=2, timed=3,
                         checkpointing=False, loss_chunk=0, settings=None):
            probes.append((cfg.d_model, settings["optimizer"]))
            return micro * accum * cfg.ctx * tiny_gpt.count_params(cfg)[0] * 1e-13, int(peak["gib"] * 2 ** 30)

        def plan(argv, device=torch.device("cpu"), speed=(None, "no table (test)"), triton=False):
            args = tiny_gpt.parse_args(["--name", "x"] + argv)
            with unittest.mock.patch.object(tiny_gpt, "device_memory_budget", return_value=budget), \
                    unittest.mock.patch.object(tiny_gpt, "measure_seconds_per_step", side_effect=fake_measure), \
                    unittest.mock.patch.object(tiny_gpt, "sustained_speed", return_value=speed), \
                    unittest.mock.patch.object(tiny_gpt, "have_triton", return_value=triton):
                return tiny_gpt.plan_with_time_budget(args, manifest, device, None)

        ladder = tiny_gpt.shape_ladder(16384)
        # the fake step time is tokens/step x N x 1e-13 s, so the tokens that fit are budget_s / (slow x N x 1e-13)
        # and the condition tokens >= 20 N holds for N^2 <= budget_s / (slow x 20 x 1e-13)
        hours, slow = 1.0, tiny_gpt.SLOW_FACTOR["cpu"]
        bound = hours * 3600 * 0.92 / (slow * 20 * 1e-13)
        want = max((c for c in ladder if tiny_gpt.count_params(c)[0] ** 2 <= bound), key=lambda c: c.d_model)
        del probes[:]
        cfg, settings, lines = plan(["--time-budget-hours", str(hours), "--optimizer", "muon", "--compile", "off"])
        self.assertEqual((cfg.d_model, cfg.n_layers), (want.d_model, want.n_layers))
        self.assertGreater(want.d_model, ladder[0].d_model)
        self.assertLess(want.d_model, 1024)
        self.assertTrue(all(o == "muon" for _, o in probes), probes)
        self.assertLessEqual(len(probes), 6, "a binary search over the ladder, not a scan")
        self.assertTrue(any(f"slowed by {slow:g}x" in ln for ln in lines), lines)
        self.assertTrue(any("probe measured 1.00 GiB" in ln for ln in lines), lines)
        self.assertGreaterEqual(settings["steps"] * settings["tokens_per_step"],
                                20 * tiny_gpt.count_params(cfg)[0] * 0.9)
        self.assertTrue(any("satisfies the data AND the time budget" in ln for ln in lines), lines)
        # a hopeless budget (the smallest allowed, 0.01 h, at 1000 tokens per parameter): the smallest shape,
        # trained on what fits
        cfg0, settings0, lines0 = plan(["--time-budget-hours", "0.01", "--tokens-per-param", "1000", "--compile",
                                        "off"])
        self.assertEqual(cfg0.d_model, ladder[0].d_model)
        self.assertTrue(any("even the smallest shape" in ln for ln in lines0), lines0)
        # compile will be on (CUDA with Triton): the probe's time is divided by COMPILE_SPEEDUP, so a bit more fits
        del probes[:]
        cfg_c, _, lines_c = plan(["--time-budget-hours", str(hours)], device=torch.device("cuda"), triton=True)
        self.assertTrue(any(f"divided by {tiny_gpt.COMPILE_SPEEDUP:g}" in ln for ln in lines_c), lines_c)
        self.assertTrue(any("compiled;" in ln for ln in lines_c), lines_c)
        bound_c = hours * 3600 * 0.92 * tiny_gpt.COMPILE_SPEEDUP / (tiny_gpt.SLOW_FACTOR["cuda"] * 20 * 1e-13)
        want_c = max((c for c in ladder if tiny_gpt.count_params(c)[0] ** 2 <= bound_c), key=lambda c: c.d_model)
        self.assertEqual(cfg_c.d_model, want_c.d_model)
        # a sustained speed from earlier runs: the step time is at least tokens x FLOP/token / speed (here 6x the
        # fake measurement), so the tokens that fit are budget_s x speed / FLOP per token
        speed = 1e13
        cfg_s, settings_s, lines_s = plan(["--time-budget-hours", str(hours), "--compile", "off"],
                                          speed=(speed, "test table"))
        self.assertTrue(any("speed from test table" in ln for ln in lines_s), lines_s)

        def fits_at_speed(c):
            shaped = tiny_gpt.ModelConfig(**dataclasses.asdict(c))
            shaped.ctx = tiny_gpt.default_ctx(c.d_model)  # plan_run's ctx rule
            return hours * 3600 * 0.92 * speed / tiny_gpt.flops_per_token(shaped) >= 20 * tiny_gpt.count_params(c)[0]

        want_s = max((c for c in ladder if fits_at_speed(c)), key=lambda c: c.d_model)
        self.assertEqual(cfg_s.d_model, want_s.d_model)
        self.assertNotEqual(cfg_s.d_model, want.d_model, "the speed path, not the measurement, set the shape")
        fit = hours * 3600 * 0.92 * speed / tiny_gpt.flops_per_token(cfg_s, settings_s["activation_checkpointing"])
        self.assertLessEqual(settings_s["steps"] * settings_s["tokens_per_step"], fit + settings_s["tokens_per_step"])
        # the probe's peak above the memory budget: the next memory option from the start
        peak["gib"] = 100.0
        _, settings_p, lines_p = plan(["--time-budget-hours", str(hours), "--compile", "off"])
        self.assertEqual((settings_p["activation_checkpointing"], settings_p["loss_chunk_tokens"]),
                         (False, tiny_gpt.LOSS_CHUNK_TOKENS))
        self.assertTrue(any("next memory option" in ln for ln in lines_p), lines_p)
        peak["gib"] = 1.0

    def test_sustained_speed_reads_the_device_rows(self):
        """sustained_speed takes the latest run of 12 minutes or more on the same kind of device from
        experiments.csv (rows without a device column count as CUDA), with the recompute of a checkpointed run
        in its FLOP count."""
        import csv
        base = {"event": "training run", "vocab": "16384", "ctx": "1024", "d_model": "1024", "layers": "18",
                "heads": "16", "kv_heads": "4", "params": "216180000"}
        cfg = tiny_gpt.ModelConfig(vocab_size=16384, ctx=1024, d_model=1024, n_layers=18, n_heads=16, n_kv_heads=4)
        rows = [{**base, "name": "gpu_ckpt", "time": "2026-11-01T00:00:00", "device": "cuda", "tokens_seen": "3600000",
                 "train_hours": "0.5", "activation_checkpointing": "on"},
                {**base, "name": "mac", "time": "2026-11-02T00:00:00", "device": "mps", "tokens_seen": "7200000",
                 "train_hours": "1", "activation_checkpointing": "off"},
                {**base, "name": "short", "time": "2026-11-03T00:00:00", "device": "cuda", "tokens_seen": "100000",
                 "train_hours": "0.1", "activation_checkpointing": "off"},
                {**base, "event": "benchmark", "name": "bench", "device": "cuda", "tokens_seen": "", "train_hours": ""}]
        out = os.path.join(self.tmp, "speed_rows")
        os.makedirs(os.path.join(out, "results"))
        with open(os.path.join(out, "results", "experiments.csv"), "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        speed, note = tiny_gpt.sustained_speed(out, torch.device("cuda"))
        self.assertAlmostEqual(speed, 3600000 / 1800 * tiny_gpt.flops_per_token(cfg, True), delta=1.0)
        self.assertIn("gpu_ckpt", note)
        self.assertIn("activation checkpointing", note)
        speed_m, note_m = tiny_gpt.sustained_speed(out, torch.device("mps"))
        self.assertAlmostEqual(speed_m, 7200000 / 3600 * tiny_gpt.flops_per_token(cfg), delta=1.0)
        self.assertIn("mac", note_m)
        self.assertEqual(tiny_gpt.sustained_speed(out, torch.device("cpu"))[0], None)
        self.assertIn("cpu device", tiny_gpt.sustained_speed(out, torch.device("cpu"))[1])
        self.assertEqual(tiny_gpt.sustained_speed(os.path.join(self.tmp, "nowhere"), torch.device("cuda")),
                         (None, "no experiments.csv"))
        # a table from before the device column: its runs were CUDA runs
        legacy = os.path.join(self.tmp, "speed_legacy")
        os.makedirs(os.path.join(legacy, "results"))
        old = {k: v for k, v in rows[1].items() if k not in ("device", "activation_checkpointing")}
        with open(os.path.join(legacy, "results", "experiments.csv"), "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(old))
            writer.writeheader()
            writer.writerow(old)
        self.assertIn("mac", tiny_gpt.sustained_speed(legacy, torch.device("cuda"))[1])
        self.assertEqual(tiny_gpt.sustained_speed(legacy, torch.device("mps"))[0], None)

    def test_experiments_csv_gains_the_peak_memory_column(self):
        """A table written before peak_mem_gib existed is copied aside and rewritten once with the full header:
        the old rows keep their values and the new row carries the new column."""
        import csv
        out = os.path.join(self.tmp, "exp_migration")
        os.makedirs(os.path.join(out, "results"))
        path = os.path.join(out, "results", "experiments.csv")
        old_cols = [c for c in tiny_gpt.EXPERIMENT_COLUMNS if c != "peak_mem_gib"]
        self.assertIn("peak_mem_gib", tiny_gpt.EXPERIMENT_COLUMNS)
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=old_cols)
            writer.writeheader()
            writer.writerow({"time": "2026-11-01T00:00:00", "event": "training run", "name": "old", "tok_s": "123",
                             "device": "cuda"})
        with redirect_stdout(io.StringIO()):
            tiny_gpt.record_experiment(out, {"event": "training run", "name": "new", "peak_mem_gib": "1.23",
                                             "device": "cpu"})
        with open(path, encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(header, tiny_gpt.EXPERIMENT_COLUMNS)
        self.assertEqual([(r["name"], r["peak_mem_gib"], r["tok_s"], r["device"]) for r in rows],
                         [("old", "", "123", "cuda"), ("new", "1.23", "", "cpu")])
        self.assertEqual(len(glob.glob(path + ".before-columns-*")), 1)
        # already current: no second copy
        with redirect_stdout(io.StringIO()):
            tiny_gpt.record_experiment(out, {"event": "training run", "name": "newer", "peak_mem_gib": "2.5"})
        self.assertEqual(len(glob.glob(path + ".before-columns-*")), 1)

    def test_run_pipeline_guards(self):
        import run_pipeline
        here = os.path.join(self.tmp, "pipe")
        dataset = os.path.join(here, "datasets", "bio_all")
        os.makedirs(dataset)
        open(os.path.join(dataset, "manifest.json"), "w").close()
        open(os.path.join(here, "R_best.pt"), "w").close()
        calls = []
        with unittest.mock.patch.multiple(run_pipeline, HERE=here, DATASET=dataset, SUPERBPE=True,
                                          DATA=os.path.join(here, "no_data")), \
                unittest.mock.patch.object(run_pipeline, "run", side_effect=lambda cmd: calls.append(cmd) or 0), \
                unittest.mock.patch.object(run_pipeline, "gpu_busy", return_value=False), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_pipeline.step_continue(1, "R", os.path.join(here, "new_ds")), 0)
        build = calls[0]
        self.assertIn("--tokenizer-from", build)
        self.assertNotIn("--superbpe", build)
        self.assertNotIn("--tokenizer-weights", build, "no books/articles folders: no weights")
        with unittest.mock.patch.multiple(run_pipeline, HERE=here, DATASET=dataset), \
                unittest.mock.patch.object(run_pipeline, "run", side_effect=AssertionError("must not run")), \
                unittest.mock.patch.object(run_pipeline, "gpu_busy", return_value=True), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_pipeline.step_plan(1, "R"), 1)
            with unittest.mock.patch.object(sys, "argv", ["run_pipeline.py", "resume", "--name", "R"]):
                self.assertEqual(run_pipeline.main(), 1)

    def test_gpu_busy_sees_training_locks_on_every_device(self):
        """One gpu_busy for every script: a <name>.train.lock whose process is alive counts on any device (the only
        signal on a Mac, where there is no nvidia-smi); this process's own lock does not; where nvidia-smi exists,
        more than 2500 MiB used or 20 % utilization counts too. The other scripts share it and its message."""
        folder = os.path.join(self.tmp, "busy")
        os.makedirs(folder)
        lock = os.path.join(folder, "other.train.lock")
        no_smi = unittest.mock.patch.object(tiny_gpt.subprocess, "run", side_effect=OSError("no nvidia-smi"))
        with no_smi:
            self.assertFalse(tiny_gpt.gpu_busy(folder))
            with open(lock, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getppid(), "started": None}, f)  # a live process (the test runner's parent)
            self.assertTrue(tiny_gpt.gpu_busy(folder))
            self.assertTrue(tiny_gpt.gpu_busy(os.path.join(self.tmp, "no_such_folder"), folder))
            with open(lock, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "started": None}, f)  # this process's own lock
            self.assertFalse(tiny_gpt.gpu_busy(folder))
            os.remove(lock)
        for text, busy in (("3000, 0\n", True), ("100, 25\n", True), ("100, 5\n", False)):
            fake = subprocess.CompletedProcess([], 0, stdout=text)
            with unittest.mock.patch.object(tiny_gpt.subprocess, "run", return_value=fake):
                self.assertEqual(tiny_gpt.gpu_busy(folder), busy, text)
        import calibrate_probes
        import finetune
        import run_pipeline
        self.assertIs(calibrate_probes.gpu_busy, tiny_gpt.gpu_busy)
        with unittest.mock.patch.object(tiny_gpt, "gpu_busy", return_value=True) as shared:
            self.assertTrue(run_pipeline.gpu_busy())
            shared.assert_called_with(run_pipeline.HERE)
            self.assertTrue(finetune.gpu_busy(folder))
            self.assertEqual(shared.call_args[0], (finetune.HERE, folder))
        self.assertEqual(run_pipeline.GPU_BUSY, tiny_gpt.GPU_BUSY_MESSAGE)
        self.assertIn("unified memory", tiny_gpt.GPU_BUSY_MESSAGE)
        # finetune refuses every accelerator while another job runs, never the CPU, and --force overrides
        import argparse
        with unittest.mock.patch.object(finetune, "gpu_busy", return_value=True):
            for dev in ("cuda", "mps"):
                with self.assertRaises(SystemExit):
                    finetune.check_gpu_free(torch.device(dev), argparse.Namespace(force=False, out_dir=folder))
            finetune.check_gpu_free(torch.device("cpu"), argparse.Namespace(force=False, out_dir=folder))
            finetune.check_gpu_free(torch.device("mps"), argparse.Namespace(force=True, out_dir=folder))

    def test_mps_device_precision_and_budget_selection(self):
        """select_device, choose_precision and device_memory_budget on an emulated Mac and on the CPU: MPS is taken
        in auto mode when CUDA is absent, BF16 when its probe passes (else FP32), FP16 only with loss scaling, and
        the planner's budget is 75 % of Metal's recommended working set (CUDA's 0.8 less the memory-mapped token
        files and the OS that share unified memory)."""
        mps, cpu = torch.device("mps"), torch.device("cpu")
        with unittest.mock.patch.object(torch.cuda, "is_available", return_value=False), \
                unittest.mock.patch.object(torch.backends.mps, "is_available", return_value=True):
            self.assertEqual(tiny_gpt.select_device("auto").type, "mps")
            self.assertEqual(tiny_gpt.select_device("mps").type, "mps")
            self.assertEqual(tiny_gpt.select_device("cpu").type, "cpu")
            with self.assertRaises(SystemExit):
                tiny_gpt.select_device("cuda")
        with unittest.mock.patch.object(torch.cuda, "is_available", return_value=False), \
                unittest.mock.patch.object(torch.backends.mps, "is_available", return_value=False):
            self.assertEqual(tiny_gpt.select_device("auto").type, "cpu")
            with self.assertRaises(SystemExit):
                tiny_gpt.select_device("mps")
        with unittest.mock.patch.object(tiny_gpt, "_mps_bf16_ok", return_value=True):
            self.assertEqual(tiny_gpt.choose_precision(mps, "auto"), (torch.bfloat16, "BF16 autocast", False))
        with unittest.mock.patch.object(tiny_gpt, "_mps_bf16_ok", return_value=False):
            self.assertEqual(tiny_gpt.choose_precision(mps, "auto"), (None, "FP32", False))
        with unittest.mock.patch.object(tiny_gpt, "_mps_scaler_ok", return_value=False):
            with self.assertRaises(SystemExit):
                tiny_gpt.choose_precision(mps, "fp16")
            self.assertEqual(tiny_gpt.choose_precision(mps, "fp16", training=False),
                             (torch.float16, "FP16 autocast", False))
        with unittest.mock.patch.object(tiny_gpt, "_mps_scaler_ok", return_value=True):
            self.assertEqual(tiny_gpt.choose_precision(mps, "fp16"),
                             (torch.float16, "FP16 autocast + loss scaling", True))
        self.assertEqual(tiny_gpt.choose_precision(cpu, "auto"), (None, "FP32", False))
        with self.assertRaises(SystemExit):
            tiny_gpt.choose_precision(cpu, "fp16")
        rec = 72 * 2 ** 30  # Metal's recommended working set on a 96 GB Mac Studio
        with unittest.mock.patch.object(torch.mps, "recommended_max_memory", return_value=rec):
            budget, text = tiny_gpt.device_memory_budget(mps)
        self.assertEqual(budget, int(rec * 0.75))
        self.assertIn("72.0 GiB recommended working set", text)
        self.assertIn("budget 75 % = 54.0 GiB", text)
        with unittest.mock.patch.object(torch.mps, "recommended_max_memory", side_effect=RuntimeError("no MPS")):
            budget_f, _ = tiny_gpt.device_memory_budget(mps)  # falls back to the free RAM
        self.assertGreater(budget_f, 0)
        cpu_budget, cpu_text = tiny_gpt.device_memory_budget(cpu)
        self.assertGreater(cpu_budget, 0)
        self.assertIn("free RAM", cpu_text)
        self.assertEqual(tiny_gpt.parse_args(["--name", "x", "--best-bf16"]).best_bf16, True)
        self.assertEqual(tiny_gpt.parse_args(["--name", "x"]).best_bf16, False)

    def test_mps_labels_environment_notes_and_memory_cap(self):
        """device_label names the chip, cores, recommended memory and macOS (falling back to 'Apple MPS'); the run
        header prints the backend, the MPS allocator variables and the fallback warning; the hard memory cap is set
        once per process, on MPS only, and never raises."""
        mps, cpu = torch.device("mps"), torch.device("cpu")
        with unittest.mock.patch.object(torch.backends.mps, "get_name", return_value="Apple M5 Ultra", create=True), \
                unittest.mock.patch.object(torch.backends.mps, "get_core_count", return_value=80, create=True), \
                unittest.mock.patch.object(torch.mps, "recommended_max_memory", return_value=72 * 2 ** 30), \
                unittest.mock.patch.object(tiny_gpt.platform, "mac_ver", return_value=("26.1", ("", "", ""), "arm64")):
            self.assertEqual(tiny_gpt.device_label(mps),
                             "Apple M5 Ultra (80 GPU cores, 72 GiB recommended, macOS 26.1)")
        with unittest.mock.patch.object(torch.backends.mps, "get_name", side_effect=RuntimeError("no MPS"),
                                        create=True):
            self.assertEqual(tiny_gpt.device_label(mps), "Apple MPS")
        self.assertEqual(tiny_gpt.backend_label(mps), "mps")
        self.assertEqual(tiny_gpt.backend_label(cpu), "cpu")
        self.assertEqual(tiny_gpt.backend_label(torch.device("cuda")), f"cuda {torch.version.cuda}")
        self.assertEqual(tiny_gpt.mps_env_notes({}), [])
        notes = tiny_gpt.mps_env_notes({"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.2", "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
        self.assertEqual(len(notes), 2)
        self.assertIn("PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2", notes[0])
        self.assertIn("10-100x slower", notes[1])
        with unittest.mock.patch.object(tiny_gpt, "_MPS_CAPPED", False), \
                unittest.mock.patch.object(torch.mps, "set_per_process_memory_fraction") as cap:
            self.assertFalse(tiny_gpt.cap_device_memory(cpu))
            self.assertTrue(tiny_gpt.cap_device_memory(mps))
            self.assertFalse(tiny_gpt.cap_device_memory(mps), "once per process")
            cap.assert_called_once_with(1.0)
        with unittest.mock.patch.object(tiny_gpt, "_MPS_CAPPED", False), \
                unittest.mock.patch.object(torch.mps, "set_per_process_memory_fraction",
                                           side_effect=RuntimeError("no MPS")):
            self.assertFalse(tiny_gpt.cap_device_memory(mps))

    def test_device_rng_helpers_and_fork(self):
        """device_rng_state / set_device_rng_state round-trip the MPS generator (None on the CPU, whose state
        torch.get_rng_state holds, and when the backend cannot report one); fork_device_rng leaves the CPU stream as
        it was and asks fork_rng for the device's own generator on MPS."""
        mps, cpu = torch.device("mps"), torch.device("cpu")
        self.assertIsNone(tiny_gpt.device_rng_state(cpu))
        tiny_gpt.set_device_rng_state(cpu, None)  # a no-op, never an error
        fake, stored = torch.arange(8, dtype=torch.uint8), {}
        with unittest.mock.patch.object(torch.mps, "get_rng_state", return_value=fake), \
                unittest.mock.patch.object(torch.mps, "set_rng_state", side_effect=lambda s: stored.update(s=s)):
            state = tiny_gpt.device_rng_state(mps)
            self.assertTrue(torch.equal(state, fake))
            tiny_gpt.set_device_rng_state(mps, state)
        self.assertTrue(torch.equal(stored["s"], fake))
        with unittest.mock.patch.object(torch.mps, "get_rng_state", side_effect=RuntimeError("no MPS")):
            self.assertIsNone(tiny_gpt.device_rng_state(mps))
        if not torch.cuda.is_available():
            self.assertIn(tiny_gpt.device_rng_state(torch.device("cuda")), ([], None))
        torch.manual_seed(5)
        before = torch.get_rng_state()
        with tiny_gpt.fork_device_rng(cpu):
            torch.randn(100)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        with unittest.mock.patch.object(torch.random, "fork_rng") as fork:
            tiny_gpt.fork_device_rng(mps)
        fork.assert_called_once_with(devices=[0], device_type="mps")

    def test_make_adamw_falls_back_cleanly(self):
        """make_adamw asks for the fused kernel on CUDA and MPS and falls back to plain AdamW where the PyTorch
        refuses it; the CPU stays plain, so build_optimizer's CPU arithmetic (the tests' step comparisons) is
        unchanged."""
        p = [torch.nn.Parameter(torch.zeros(3))]
        opt, label = tiny_gpt.make_adamw([{"params": p}], torch.device("cpu"), lr=1e-3)
        self.assertEqual(label, "AdamW")
        self.assertFalse(opt.defaults.get("fused"))
        asked, real = [], torch.optim.AdamW

        def refuse(groups, **kw):
            asked.append(kw.get("fused"))
            if kw.get("fused"):
                raise RuntimeError("fused AdamW not supported here")
            return real(groups, **kw)

        with unittest.mock.patch.object(torch.optim, "AdamW", side_effect=refuse):
            _, label_m = tiny_gpt.make_adamw([{"params": p}], torch.device("mps"), lr=1e-3)
        self.assertEqual((label_m, asked), ("AdamW", [True, None]))
        cfg = tiny_gpt.ModelConfig(vocab_size=50, ctx=8, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1)
        settings = {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.1, "betas": [0.9, 0.95]}
        self.assertEqual(tiny_gpt.build_optimizer(tiny_gpt.TinyGPT(cfg), settings, torch.device("cpu"))[1], "AdamW")

    def test_attention_memory_probe_drives_the_mps_estimate(self):
        """attention_keeps_scores is a measurement (cached per device type; CUDA and CPU: False); when it is true
        the estimate charges n_heads x ctx x 6 bytes per token and layer (one layer with checkpointing), and at
        ctx >= 2048 activation checkpointing becomes the first memory option."""
        mps, cpu = torch.device("mps"), torch.device("cpu")
        cfg = tiny_gpt.ModelConfig(vocab_size=16384, ctx=2048, d_model=1024, n_layers=18, n_heads=16, n_kv_heads=4)
        self.assertFalse(tiny_gpt.attention_keeps_scores(cpu))
        with unittest.mock.patch.dict(tiny_gpt._ATTN_SCORES_CACHE, clear=True), \
                unittest.mock.patch.object(tiny_gpt, "_attention_growth_bytes", return_value=16 * 2 ** 20) as probe:
            self.assertTrue(tiny_gpt.attention_keeps_scores(mps))
            self.assertTrue(tiny_gpt.attention_keeps_scores(mps))
            self.assertEqual(probe.call_count, 1, "measured once per device type")
            plain = tiny_gpt.estimate_train_bytes(cfg, 2, True)
            self.assertEqual(tiny_gpt.estimate_train_bytes(cfg, 2, True, device=mps) - plain,
                             2 * 2048 * 18 * 16 * 2048 * 6)
            self.assertEqual(tiny_gpt.estimate_train_bytes(cfg, 2, True, checkpointing=True, device=mps)
                             - tiny_gpt.estimate_train_bytes(cfg, 2, True, checkpointing=True),
                             2 * 2048 * 16 * 2048 * 6, "with checkpointing only the recomputed block keeps scores")
            self.assertEqual(tiny_gpt.estimate_train_bytes(cfg, 2, True, device=cpu), plain)
            budget = 64 * 2 ** 30
            self.assertTrue(tiny_gpt.choose_memory_plan(cfg, 262_144, budget, True, "muon", device=mps)[2])
            self.assertFalse(tiny_gpt.choose_memory_plan(cfg, 262_144, budget, True, "muon", device=cpu)[2])
            short = tiny_gpt.ModelConfig(**{**dataclasses.asdict(cfg), "ctx": 1024})
            self.assertFalse(tiny_gpt.choose_memory_plan(short, 262_144, budget, True, "muon", device=mps)[2])
            self.assertFalse(tiny_gpt.choose_memory_plan(cfg, 262_144, budget, True, "muon", "off", device=mps)[2],
                             "never an option the command line turned off")
        with unittest.mock.patch.dict(tiny_gpt._ATTN_SCORES_CACHE, clear=True), \
                unittest.mock.patch.object(tiny_gpt, "_attention_growth_bytes", return_value=2 ** 20):
            self.assertFalse(tiny_gpt.attention_keeps_scores(mps), "a fused kernel keeps about 1 MiB")
        with unittest.mock.patch.dict(tiny_gpt._ATTN_SCORES_CACHE, clear=True), \
                unittest.mock.patch.object(tiny_gpt, "_attention_growth_bytes", side_effect=RuntimeError("x")):
            self.assertFalse(tiny_gpt.attention_keeps_scores(mps))

    def test_time_budget_ladder_retries_memory_options_on_oom(self):
        """An out-of-memory error in the ladder's speed test measures the same shape again with activation
        checkpointing on (unless the command line turned it off), then with the micro-batch halved down to 1,
        before the shape counts as not fitting; the plan then carries the option that fit."""
        manifest = {"tokenizer": {"vocab_size": 16384},
                    "sources": {"md": {"train": {"tokens": 1_000_000_000}, "doc_tokens_percentiles": {"90": 1000}}}}
        ladder = tiny_gpt.shape_ladder(16384)
        bound = 3600 * 0.92 / (tiny_gpt.SLOW_FACTOR["cpu"] * 20 * 1e-13)
        want = max((c for c in ladder if tiny_gpt.count_params(c)[0] ** 2 <= bound), key=lambda c: c.d_model)
        below = max(c.d_model for c in ladder if c.d_model < want.d_model)
        seen = []

        def measure(fails):
            def fake(cfg, micro, accum, device, amp_dtype, attn_pref="auto", warmup=2, timed=3,
                     checkpointing=False, loss_chunk=0, settings=None):
                seen.append((cfg.d_model, micro, checkpointing))
                if cfg.d_model == want.d_model and fails(micro, checkpointing):
                    raise RuntimeError("MPS backend out of memory (simulated)")
                return micro * accum * cfg.ctx * tiny_gpt.count_params(cfg)[0] * 1e-13, 2 ** 30
            return fake

        def plan(argv, fails):
            del seen[:]
            args = tiny_gpt.parse_args(["--name", "x", "--time-budget-hours", "1", "--compile", "off"] + argv)
            with unittest.mock.patch.object(tiny_gpt, "device_memory_budget", return_value=(64 * 2 ** 30, "64 GiB")), \
                    unittest.mock.patch.object(tiny_gpt, "measure_seconds_per_step", side_effect=measure(fails)), \
                    unittest.mock.patch.object(tiny_gpt, "sustained_speed", return_value=(None, "no table")), \
                    unittest.mock.patch.object(tiny_gpt, "have_triton", return_value=False):
                return tiny_gpt.plan_with_time_budget(args, manifest, torch.device("cpu"), None)

        cfg, settings, lines = plan([], lambda micro, ck: not ck)
        self.assertEqual(cfg.d_model, want.d_model)
        self.assertTrue(settings["activation_checkpointing"])
        self.assertTrue(any("measured again with activation checkpointing on" in ln for ln in lines), lines)
        self.assertEqual([ck for d, _, ck in seen if d == want.d_model], [False, True])
        cfg2, settings2, lines2 = plan(["--activation-checkpointing", "off"], lambda micro, ck: micro > 1)
        self.assertEqual((cfg2.d_model, settings2["micro_batch"], settings2["activation_checkpointing"]),
                         (want.d_model, 1, False))
        self.assertTrue(any("measured again with 1" in ln for ln in lines2), lines2)
        micros = [m for d, m, _ in seen if d == want.d_model]
        self.assertEqual(micros, sorted(micros, reverse=True))
        self.assertTrue(all(a == 2 * b for a, b in zip(micros, micros[1:])), micros)
        cfg3, _, lines3 = plan([], lambda micro, ck: True)
        self.assertEqual(cfg3.d_model, below)
        self.assertTrue(any("does not fit in memory even at micro-batch 1" in ln for ln in lines3), lines3)

    def test_energy_is_blank_where_the_gpu_is_not_measured(self):
        """On an accelerator without nvidia-smi (MPS) the experiments table leaves energy, CO2e and cost blank and
        says why, and the meter's description says how to measure the chip's power; CPU runs (an estimate by
        design) and measured GPUs keep their figures."""
        mps, cpu = torch.device("mps"), torch.device("cpu")
        meter = tiny_gpt.EnergyMeter(mps, 35.0, 125.0, period=3600)
        try:
            self.assertFalse(meter.gpu_measured)
            self.assertIn("sudo powermetrics", meter.describe())
            cols = tiny_gpt.energy_columns(meter, mps)
            self.assertEqual((cols["energy_kwh"], cols["co2e_kg"], cols["cost_gbp"]), ("", "", ""))
            self.assertEqual(cols["notes"], "energy not measured (no nvidia-smi)")
            self.assertEqual(tiny_gpt.energy_columns(meter, mps, "RuntimeError: x")["notes"],
                             "RuntimeError: x; energy not measured (no nvidia-smi)")
        finally:
            meter.stop()
        meter_cpu = tiny_gpt.EnergyMeter(cpu, 35.0, 125.0, period=3600)
        try:
            self.assertNotIn("powermetrics", meter_cpu.describe())
            cols = tiny_gpt.energy_columns(meter_cpu, cpu, None)
            self.assertIsNone(cols["notes"])
            self.assertGreaterEqual(float(cols["energy_kwh"]), 0.0)
            meter_cpu.gpu_measured = True  # as on a measured GPU
            self.assertEqual(tiny_gpt.energy_columns(meter_cpu, mps, "n")["notes"], "n")
        finally:
            meter_cpu.stop()

    def test_checkpoint_weights_bf16_best_and_mps_path(self):
        """weights_for_checkpoint keeps tied weights as one tensor; --best-bf16 stores half-size weights that both
        load paths read back into a float32 model; on MPS the device tensors themselves are handed to torch.save
        (no bulk CPU copy), on CUDA/CPU a CPU copy as before."""
        cfg = tiny_gpt.ModelConfig(vocab_size=50, ctx=8, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1)
        torch.manual_seed(0)
        model = tiny_gpt.TinyGPT(cfg)
        cpu, mps = torch.device("cpu"), torch.device("mps")
        full = tiny_gpt.weights_for_checkpoint(model, cpu)
        self.assertIs(full["lm_head.weight"], full["token_embedding.weight"])
        self.assertEqual(full["token_embedding.weight"].dtype, torch.float32)
        self.assertNotEqual(full["token_embedding.weight"].data_ptr(), model.token_embedding.weight.data_ptr())
        on_device = tiny_gpt.weights_for_checkpoint(model, mps)  # the model's own tensors, nothing copied
        self.assertEqual(on_device["token_embedding.weight"].data_ptr(), model.token_embedding.weight.data_ptr())
        self.assertIs(on_device["lm_head.weight"], on_device["token_embedding.weight"])
        half = tiny_gpt.weights_for_checkpoint(model, cpu, bf16=True)
        self.assertIs(half["lm_head.weight"], half["token_embedding.weight"])
        self.assertTrue(all(v.dtype == torch.bfloat16 for v in half.values() if v.is_floating_point()))
        self.assertEqual(set(half), set(full))
        want = model.token_embedding.weight.detach().to(torch.bfloat16).float()
        copied = tiny_gpt.TinyGPT(cfg)
        copied.load_state_dict(half, strict=False)
        self.assertEqual(copied.token_embedding.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(copied.token_embedding.weight, want))
        assigned = tiny_gpt._model_assigned({"model": half}, cfg, cpu)
        self.assertIsNotNone(assigned)
        self.assertEqual(assigned.token_embedding.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(assigned.token_embedding.weight, want))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.zeros((1, 4), dtype=torch.long)).sum().backward()
        opt.step()
        state = opt.state_dict()
        self.assertEqual(tiny_gpt.optimizer_state_for_checkpoint(opt, mps)["state"].keys(), state["state"].keys())
        cpu_state = tiny_gpt.optimizer_state_for_checkpoint(opt, cpu)
        self.assertTrue(all(v.device.type == "cpu" for s in cpu_state["state"].values() for v in s.values()
                            if isinstance(v, torch.Tensor)))


    # ---------------- self-scaling planner (2026-11 audit, WP4) ----------------
    def test_planner_rules_keep_the_216m_plan(self):
        """One context rule, a RoPE base that follows the context, the compute-budget batch and the LR batch factor
        are monotone and bounded over the ladder; the current 216M plan (d 1024, L 18, ctx 1024, 262,144 or
        270,336 tokens/step, lr 4.76e-4) comes out unchanged, with the WSD decay now ending at zero."""
        self.assertEqual([tiny_gpt.default_ctx(d) for d in (128, 192, 256, 512, 768, 1024, 1280, 2048, 2560, 3072)],
                         [256, 256, 512, 512, 1024, 1024, 2048, 2048, 4096, 4096])
        self.assertEqual(tiny_gpt.default_ctx(1024), 1024)
        self.assertEqual([tiny_gpt.default_rope_base(c) for c in (512, 1024, 2048, 4096, 8192, 16384)],
                         [10_000.0, 10_000.0, 10_000.0, 40_000.0, 160_000.0, 640_000.0])
        self.assertEqual(tiny_gpt.tokens_per_step_for(216e6, 16.2e9), 262_144)
        self.assertEqual(tiny_gpt.tokens_per_step_for(1.45e9, 29e9), 524_288)
        self.assertEqual(tiny_gpt.tokens_per_step_for(3.2e9, 64e9), 1_048_576)
        self.assertEqual(tiny_gpt.tokens_per_step_for(1e5, 1e6), 2 ** 14, "clamped below")
        self.assertEqual(tiny_gpt.tokens_per_step_for(1e12, 1e14), 2 ** 21, "clamped above")
        self.assertEqual(tiny_gpt.tokens_per_step_legacy(216e6), 262_144)
        self.assertEqual(tiny_gpt.lr_batch_factor(216e6, 262_144), 1.0)
        self.assertEqual(tiny_gpt.lr_batch_factor(216e6, 270_336), 1.0, "accumulation rounding leaves the LR alone")
        self.assertAlmostEqual(tiny_gpt.lr_batch_factor(216e6, 2 * 262_144), 1.4, msg="sqrt(2) clamped to 1.4")
        self.assertAlmostEqual(tiny_gpt.lr_batch_factor(216e6, 262_144 // 2), math.sqrt(0.5))
        self.assertEqual(tiny_gpt.lr_batch_factor(216e6, 262_144 // 16), 0.7)
        ladder = tiny_gpt.shape_ladder(16384)
        prev_tps, prev_lr, prev_ctx = 0, float("inf"), 0
        for c in ladder:
            n = tiny_gpt.count_params(c)[0]
            tps = tiny_gpt.tokens_per_step_for(n, 20 * n)
            self.assertEqual(tps & (tps - 1), 0, f"{tps} is not a power of two")
            self.assertTrue(2 ** 14 <= tps <= 2 ** 21, tps)
            self.assertGreaterEqual(tps, prev_tps, "tokens/step grows with the model")
            self.assertLessEqual(tiny_gpt.heuristic_lr(n), prev_lr, "the heuristic LR falls with the model")
            ctx = tiny_gpt.default_ctx(c.d_model)
            self.assertGreaterEqual(ctx, prev_ctx)
            self.assertGreaterEqual(tiny_gpt.default_rope_base(ctx), 10_000.0)
            shaped = tiny_gpt.ModelConfig(**{**dataclasses.asdict(c), "ctx": ctx})
            self.assertTrue(0.0 < tiny_gpt.attention_flop_share(shaped) < 0.5)
            prev_tps, prev_lr, prev_ctx = tps, tiny_gpt.heuristic_lr(n), ctx
        # the 216M run's plan at the current dataset size (4.05B unique tokens x 4 epochs)
        manifest = {"tokenizer": {"vocab_size": 16384}, "sources": {
            "wiki": {"train": {"tokens": 1_430_000_000}, "doc_tokens_percentiles": {"50": 600}},
            "pmc": {"train": {"tokens": 1_105_000_000}, "doc_tokens_percentiles": {"50": 10_072}},
            "fineweb": {"train": {"tokens": 810_000_000}, "doc_tokens_percentiles": {"50": 500}},
            "pes2o": {"train": {"tokens": 441_000_000}, "doc_tokens_percentiles": {"50": 5_000}},
            "arxiv": {"train": {"tokens": 170_000_000}, "doc_tokens_percentiles": {"50": 12_332}},
            "books": {"train": {"tokens": 20_000_000}, "doc_tokens_percentiles": {"50": 206_767}},
            "notes": {"train": {"tokens": 24_000_000}, "doc_tokens_percentiles": {"50": 800}}}}
        budget = (64 * 2 ** 30, "64 GiB (test)")

        def plan(argv):
            args = tiny_gpt.parse_args(["--name", "x", "--d-model", "1024", "--layers", "18"] + argv)
            with unittest.mock.patch.object(tiny_gpt, "device_memory_budget", return_value=budget):
                return tiny_gpt.plan_run(args, manifest, torch.device("cpu"), None)

        cfg, st, lines = plan(["--optimizer", "muon"])
        n = tiny_gpt.count_params(cfg)[0]
        self.assertEqual((cfg.ctx, cfg.rope_base, cfg.tie_embeddings, cfg.n_heads, cfg.n_kv_heads),
                         (1024, 10_000.0, True, 16, 4))
        self.assertEqual(st["tokens_per_step"], 262_144)
        self.assertEqual(st["lr"], tiny_gpt.heuristic_lr(n), "batch factor exactly 1 at the current shape")
        # the running run's recorded 4.76e-4 (old table) doubled by LR_TABLE_FACTOR; --resume keeps the recorded one
        self.assertAlmostEqual(st["lr"], 2 * 4.76e-4, delta=1e-6)
        self.assertEqual(st["weight_decay"], 0.05, "lr x wd unchanged: the decay timescale below stays 21,020 steps")
        self.assertEqual(st["min_lr"], 0.0, "wsd decays to zero by default")
        text = "\n".join(lines)
        self.assertIn("batch factor 1.00", text)
        self.assertIn("ctx 1024 (default for d_model 1024): attention 15% of the training FLOPs", text)
        self.assertIn("default for wsd: decay to zero", text)
        self.assertIn("decay timescale 1/(lr x wd) = 21,020 steps", text)
        self.assertRegex(text, r"sources: wiki 35\.8% \(4\.00 epochs\), pmc 27\.6% \(4\.00 epochs\)")
        self.assertIn("ctx 1024 is shorter than the median document of pmc (median document 10,072 tokens, 28%", text)
        self.assertNotIn("WARNING", text)
        self.assertNotIn("books (median", text, "books carry 0.5% of the tokens: below the 10% bar")
        # the run's actual batch (4 x 66 x 1024) and its 60,000 steps: the same learning rate
        cfg2, st2, _ = plan(["--optimizer", "muon", "--tokens-per-step", "270336", "--steps", "60000"])
        self.assertEqual((st2["tokens_per_step"], st2["lr"], st2["steps"]), (270_336, st["lr"], 60_000))
        # cosine keeps one tenth; an explicit ratio wins; --lr switches the batch factor off
        _, st_c, lines_c = plan(["--schedule", "cosine"])
        self.assertAlmostEqual(st_c["min_lr"], 0.1 * st_c["lr"])
        self.assertIn("default for cosine: one tenth", "\n".join(lines_c))
        _, st_r, lines_r = plan(["--min-lr-ratio", "0.3", "--lr", "1e-3"])
        self.assertAlmostEqual(st_r["min_lr"], 3e-4)
        self.assertIn("learning rate 1.00e-03 peak (set)", "\n".join(lines_r))
        # a doubled batch raises the LR by the clamped sqrt factor, printed
        _, st_b, lines_b = plan(["--tokens-per-step", "524288"])
        self.assertAlmostEqual(st_b["lr"], 1.4 * tiny_gpt.heuristic_lr(n))
        self.assertIn("batch factor 1.40 for 524,288 tokens/step against the table's 262,144", "\n".join(lines_b))
        # --batch-size and --grad-accum fix the batch far from the planner's target: the factor follows the batch the
        # run will use (8 x 1024 x 1 = 8,192 tokens/step), not the --steps-derived target
        _, st_s, lines_s = plan(["--batch-size", "8", "--grad-accum", "1", "--steps", "6"])
        self.assertEqual(st_s["tokens_per_step"], 8_192)
        self.assertAlmostEqual(st_s["lr"], 0.7 * tiny_gpt.heuristic_lr(n))
        self.assertIn("batch factor 0.70 for 8,192 tokens/step against the table's 262,144", "\n".join(lines_s))
        # --mix-alpha flattens the mix in the plan and warns about sources repeated past --max-epochs
        _, st_a, lines_a = plan(["--mix-alpha", "0.8"])
        text_a = "\n".join(lines_a)
        self.assertIn("--mix-alpha 0.8: shares ~ weight x tokens^alpha", text_a)
        self.assertRegex(text_a, r"WARNING: .*books would be seen 8\.\d times.*above --max-epochs 4")
        # larger shapes: the Mac ladder gets ctx 2048 / 4096 and a RoPE base that follows
        with unittest.mock.patch.object(tiny_gpt, "device_memory_budget", return_value=budget):
            big, st_big, lines_big = tiny_gpt.plan_run(tiny_gpt.parse_args(["--name", "x", "--d-model", "2560"]),
                                                       manifest, torch.device("cpu"), None)
        self.assertEqual((big.ctx, big.rope_base), (4096, 40_000.0))
        self.assertIn("RoPE base 40,000 (default for this ctx)", "\n".join(lines_big))
        # plan_table shares the rules
        with unittest.mock.patch.object(tiny_gpt, "device_memory_budget", return_value=budget):
            rows = tiny_gpt.plan_table(16384, torch.device("cpu"), None, 20.0, 4.0)
        for unique, planned, c, n_row, tps, fits, lr in rows:
            self.assertEqual(c.ctx, tiny_gpt.default_ctx(c.d_model))
            self.assertEqual(tps, tiny_gpt.tokens_per_step_for(n_row, planned))
            self.assertAlmostEqual(lr, tiny_gpt.heuristic_lr(n_row) * tiny_gpt.lr_batch_factor(n_row, tps))

    def test_min_lr_default_and_resume_override(self):
        """--min-lr-ratio: None resolves per schedule; on --resume the end point may change while the run is in
        its warmup or stable phase (nothing moves before decay_start) and is refused inside the decay."""
        self.assertEqual(tiny_gpt.resolve_min_lr_ratio(None, "wsd"), 0.0)
        self.assertEqual(tiny_gpt.resolve_min_lr_ratio(None, "cosine"), 0.1)
        self.assertEqual(tiny_gpt.resolve_min_lr_ratio(0.3, "wsd"), 0.3)
        self.assertIsNone(tiny_gpt.parse_args(["--name", "x"]).min_lr_ratio)
        settings = {"schedule": "wsd", "steps": 100, "lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 10,
                    "decay_frac": 0.2, "decay_shape": "1-sqrt"}

        def schedule():
            return tiny_gpt.LRSchedule(settings["schedule"], settings["steps"], settings["lr"], settings["min_lr"],
                                       settings["warmup_steps"], settings["decay_frac"], settings["decay_shape"])

        before = [schedule().lr_at(s) for s in range(0, 81)]
        applied, note = tiny_gpt.resume_min_lr(settings, 50, 0.0)
        self.assertTrue(applied)
        self.assertIn("from the decay phase on", note)
        self.assertEqual(settings["min_lr"], 0.0)
        self.assertEqual([schedule().lr_at(s) for s in range(0, 81)], before, "nothing changes before the decay")
        self.assertEqual(schedule().lr_at(100), 0.0)
        self.assertLess(schedule().lr_at(99), 1e-4)
        applied, note = tiny_gpt.resume_min_lr(settings, 50, 0.0)
        self.assertEqual((applied, "is already" in note), (False, True))
        applied, note = tiny_gpt.resume_min_lr(settings, 90, 0.5)
        self.assertFalse(applied)
        self.assertIn("ignored (already in the decay phase)", note)
        self.assertEqual(settings["min_lr"], 0.0)
        cosine = dict(settings, schedule="cosine", min_lr=1e-4)
        self.assertFalse(tiny_gpt.resume_min_lr(cosine, 50, 0.0)[0], "cosine decays from the warmup on")
        self.assertTrue(tiny_gpt.resume_min_lr(cosine, 5, 0.0)[0], "but may still change during the warmup")

    def test_mix_alpha_shares(self):
        """--mix-alpha: sampling shares ~ weight x tokens^alpha; alpha 0 ignores the sizes, so weights {a: 3, b: 1}
        give TokenBatcher.share() {0.75, 0.25}; alpha 1 is proportional sampling as before."""
        rng = np.random.default_rng(0)
        arrays = {"a": rng.integers(0, 50, 100, dtype=np.int64), "b": rng.integers(0, 50, 300, dtype=np.int64)}
        sizes = {n: len(a) for n, a in arrays.items()}
        device = torch.device("cpu")
        flat = tiny_gpt.mix_weights("a=3,b=1", arrays, sizes, alpha=0.0)
        share = tiny_gpt.TokenBatcher(arrays, flat, 8, 4, device, 0).share()
        self.assertAlmostEqual(share["a"], 0.75, places=6)
        self.assertAlmostEqual(share["b"], 0.25, places=6)
        self.assertEqual(tiny_gpt.mix_weights("a=3,b=1", arrays, sizes, alpha=1.0), {"a": 3.0, "b": 1.0})
        share1 = tiny_gpt.TokenBatcher(arrays, tiny_gpt.mix_weights(None, arrays, sizes, 1.0), 8, 4, device, 0).share()
        self.assertAlmostEqual(share1["a"], 0.25, places=6)
        half = tiny_gpt.TokenBatcher(arrays, tiny_gpt.mix_weights(None, arrays, sizes, 0.5), 8, 4, device, 0).share()
        self.assertAlmostEqual(half["a"], math.sqrt(100) / (math.sqrt(100) + math.sqrt(300)), places=6)
        self.assertEqual(tiny_gpt.parse_args(["--name", "x", "--mix-alpha", "0.8"]).mix_alpha, 0.8)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            with open(os.devnull, "w") as devnull, unittest.mock.patch.object(sys, "stderr", devnull):
                tiny_gpt.parse_args(["--name", "x", "--mix-alpha", "3"])

    def test_eval_size_and_batch_rules(self):
        """The validation set of a new run follows the tokens trained between evaluations (65,536 today, more from
        ~1M tokens/step); the evaluation batch follows the memory budget, not the training micro-batch."""
        self.assertEqual(tiny_gpt.default_eval_tokens(200, 270_336, 10, 1e9), 65_536)
        self.assertEqual(tiny_gpt.default_eval_tokens(200, 2 ** 20, 10, 1e9), 131_072)
        self.assertEqual(tiny_gpt.default_eval_tokens(200, 2 ** 21, 10, 1e9), 262_144)
        self.assertEqual(tiny_gpt.default_eval_tokens(200, 2 ** 21, 10, 100_000), 100_000, "capped by the data")
        self.assertEqual(tiny_gpt.default_eval_tokens(200, 2 ** 21, 10, 1_000), 65_536, "never below today's size")
        self.assertIsNone(tiny_gpt.parse_args(["--name", "x"]).eval_tokens)
        self.assertEqual(tiny_gpt.default_eval_batch(2, 1024, 16384, 6 * 2 ** 30), 2, "never below the micro-batch")
        self.assertEqual(tiny_gpt.default_eval_batch(2, 1024, 16384, 64 * 2 ** 30), 16, "at most 16 windows")
        self.assertEqual(tiny_gpt.default_eval_batch(4, 1024, 16384, 40 * 2 ** 30), 10)
        self.assertEqual(tiny_gpt.default_eval_batch(1, 4096, 32768, 54 * 2 ** 30), 1)
        big = tiny_gpt.ModelConfig(vocab_size=16384, ctx=1024, d_model=1024, n_layers=18, n_heads=16, n_kv_heads=4)
        attn = 12 * 18 * 1024 * 1024
        self.assertAlmostEqual(tiny_gpt.attention_flop_share(big), attn / (6 * tiny_gpt.count_params(big)[0] + attn))

    def test_init_from_ctx_override_and_untied_embedding(self):
        """config_from_checkpoint(obj, ctx=N): the weights of a model trained at one context load unchanged into a
        model at another (RoPE tables are rebuilt), and agree on the shared prefix. --tie-embeddings off puts
        the input embedding in the no-decay group and keeps the output matrix decayed."""
        model = self.small_model(seed=5)
        obj = {"format": tiny_gpt.CHECKPOINT_FORMAT, "checkpoint_type": "inference",
               "model_config": dataclasses.asdict(model.cfg), "model": model.state_dict()}
        self.assertEqual(tiny_gpt.config_from_checkpoint(obj).ctx, 16)
        cfg = tiny_gpt.config_from_checkpoint(obj, ctx=32)
        self.assertEqual(cfg.ctx, 32)
        self.assertEqual(dataclasses.asdict(cfg), {**dataclasses.asdict(model.cfg), "ctx": 32})
        longer = tiny_gpt.TinyGPT(cfg).eval()
        longer.load_state_dict(obj["model"])
        x = torch.randint(0, 50, (2, 32))
        with torch.no_grad():
            full = longer(x)
            self.assertEqual(tuple(full.shape), (2, 32, 50))
            torch.testing.assert_close(longer(x[:, :16]), model(x[:, :16]), rtol=1e-5, atol=1e-5)
        with self.assertRaises(SystemExit):
            tiny_gpt.config_from_checkpoint(obj, ctx=7)
        settings = {"optimizer": "adamw", "lr": 1e-3, "betas": [0.9, 0.95], "weight_decay": 0.1}
        _, _, names, note = tiny_gpt.build_optimizer(model, settings, torch.device("cpu"))
        self.assertIn("token_embedding.weight", names[0], "tied: the one matrix keeps its decay")
        self.assertNotIn("untied", note)
        untied_cfg = tiny_gpt.ModelConfig(vocab_size=50, ctx=16, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                                          tie_embeddings=False)
        untied = tiny_gpt.TinyGPT(untied_cfg)
        self.assertEqual(untied.param_counts()[0], tiny_gpt.count_params(untied_cfg)[0])
        _, _, names_u, note_u = tiny_gpt.build_optimizer(untied, settings, torch.device("cpu"))
        self.assertIn("token_embedding.weight", names_u[1])
        self.assertIn("lm_head.weight", names_u[0])
        self.assertIn("untied input embedding", note_u)
        _, _, names_m, note_m = tiny_gpt.build_optimizer(untied, dict(settings, optimizer="muon"), torch.device("cpu"))
        self.assertIn("token_embedding.weight", names_m[1])
        self.assertNotIn("token_embedding.weight", names_m[2], "MUON_EXCLUDED keeps the embedding on AdamW")
        manifest = {"tokenizer": {"vocab_size": 700},
                    "sources": {"md": {"train": {"tokens": 5_000_000}, "doc_tokens_percentiles": {"50": 300}}}}
        args = tiny_gpt.parse_args(["--name", "x", "--tie-embeddings", "off", "--d-model", "64"])
        with unittest.mock.patch.object(tiny_gpt, "device_memory_budget", return_value=(4 * 2 ** 30, "4 GiB (test)")):
            cfg_u, _, lines = tiny_gpt.plan_run(args, manifest, torch.device("cpu"), None)
        self.assertFalse(cfg_u.tie_embeddings)
        self.assertIn("excluding the (untied) embedding", "\n".join(lines))
        # --init-from plans with the checkpoint's architecture (fixed_cfg), the tying included, whatever the flags say
        tied_args = tiny_gpt.parse_args(["--name", "x", "--tie-embeddings", "on", "--d-model", "128"])
        with unittest.mock.patch.object(tiny_gpt, "device_memory_budget", return_value=(4 * 2 ** 30, "4 GiB (test)")):
            cfg_f, _, lines_f = tiny_gpt.plan_run(tied_args, manifest, torch.device("cpu"), None, fixed_cfg=untied_cfg)
            cfg_t, _, _ = tiny_gpt.plan_run(args, manifest, torch.device("cpu"), None, fixed_cfg=model.cfg)
        self.assertEqual(dataclasses.asdict(cfg_f), dataclasses.asdict(untied_cfg))
        self.assertIn("excluding the (untied) embedding", "\n".join(lines_f))
        self.assertIn("the architecture of the --init-from checkpoint", "\n".join(lines_f))
        self.assertTrue(cfg_t.tie_embeddings)

    def test_qk_norm_model_cache_config_and_export(self):
        """QK-norm (quality-3): one gain vector per query and key norm per layer, counted by count_params; the
        attention logits are bounded by sqrt(head_dim) x gain^2; the KV cache agrees with a full forward pass;
        a config dict from before the field loads with it off; the gains sit in the no-decay AdamW group and
        never on Muon; a safetensors export round-trips them; --init-from plans with the checkpoint's setting."""
        cfg = tiny_gpt.ModelConfig(vocab_size=50, ctx=16, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                                   qk_norm=True)
        torch.manual_seed(1)
        model = tiny_gpt.TinyGPT(cfg)
        attn = model.blocks[0].attn
        self.assertEqual(tuple(attn.q_norm.weight.shape), (cfg.head_dim,))
        self.assertEqual(model.param_counts()[0], tiny_gpt.count_params(cfg)[0])
        plain = tiny_gpt.ModelConfig(**{**dataclasses.asdict(cfg), "qk_norm": False})
        self.assertEqual(tiny_gpt.count_params(cfg)[0] - tiny_gpt.count_params(plain)[0],
                         cfg.n_layers * 2 * cfg.head_dim)
        self.assertIsNone(tiny_gpt.TinyGPT(plain).blocks[0].attn.q_norm)
        x = torch.randint(0, 50, (2, 16))
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 16, 50))
        logits.float().logsumexp(-1).mean().backward()
        self.assertTrue(all(p.grad is not None for n, p in model.named_parameters() if "_norm" in n))
        model.eval()
        with torch.no_grad():
            full = model(x)
            cache = model.new_cache()
            parts = [model(x[:, :5], cache=cache)] + [model(x[:, i:i + 1], cache=cache) for i in range(5, 16)]
            torch.testing.assert_close(torch.cat(parts, 1), full, rtol=1e-4, atol=1e-4)
            # |q . k| <= |q| |k| = head_dim x (gain^2, 1 at init), so the logits q.k / sqrt(d) are bounded by sqrt(d)
            q, k, _ = attn._qkv(model.blocks[0].attn_norm(model.token_embedding(x)), model.rope)
            scores = (q @ k.repeat_interleave(attn.n_rep, dim=1).transpose(-1, -2)) / math.sqrt(attn.head_dim)
            self.assertLessEqual(scores.abs().max().item(), math.sqrt(attn.head_dim) + 1e-4)
            self.assertAlmostEqual(q.pow(2).mean(-1).max().item(), 1.0, places=3, msg="queries have RMS 1")
        old = {k: v for k, v in dataclasses.asdict(cfg).items() if k != "qk_norm"}
        self.assertFalse(tiny_gpt.ModelConfig(**old).qk_norm, "checkpoints from before the field load with it off")
        obj = {"format": tiny_gpt.CHECKPOINT_FORMAT, "checkpoint_type": "inference",
               "model_config": dataclasses.asdict(cfg), "model": model.state_dict()}
        self.assertTrue(tiny_gpt.config_from_checkpoint(obj).qk_norm)
        settings = {"optimizer": "muon", "lr": 1e-3, "betas": [0.9, 0.95], "weight_decay": 0.1}
        _, _, names, _ = tiny_gpt.build_optimizer(model, settings, torch.device("cpu"))
        self.assertIn("blocks.0.attn.q_norm.weight", names[1], "1-D gains: AdamW without weight decay")
        self.assertNotIn("blocks.0.attn.k_norm.weight", names[0] + names[2])
        path = os.path.join(self.tmp, "qk.safetensors")
        tensors = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
        tiny_gpt.write_safetensors(tensors, path, {"qk_norm": cfg.qk_norm})
        back, meta = tiny_gpt.read_safetensors(path)
        self.assertEqual(meta.get("qk_norm"), "True")
        self.assertIn("blocks.1.attn.k_norm.weight", back)
        again = tiny_gpt.TinyGPT(cfg).eval()
        again.load_state_dict(back, strict=False)
        with torch.no_grad():
            torch.testing.assert_close(again(x), full)
        args = tiny_gpt.parse_args(["--name", "x", "--qk-norm", "off"])
        self.assertEqual(tiny_gpt.parse_args(["--name", "x"]).qk_norm, None,
                         "unset: on for new runs, the checkpoint's on --resume/--init-from")
        manifest = {"tokenizer": {"vocab_size": 50},
                    "sources": {"md": {"train": {"tokens": 5_000_000}, "doc_tokens_percentiles": {"50": 300}}}}
        with unittest.mock.patch.object(tiny_gpt, "device_memory_budget", return_value=(4 * 2 ** 30, "4 GiB (test)")):
            on, _, _ = tiny_gpt.plan_run(args, manifest, torch.device("cpu"), None, fixed_cfg=cfg)
            off, _, _ = tiny_gpt.plan_run(args, manifest, torch.device("cpu"), None, fixed_cfg=plain)
        self.assertTrue(on.qk_norm, "--init-from: the checkpoint's setting, not the flag's")
        self.assertFalse(off.qk_norm)

    def test_quality_flags_in_the_plan(self):
        """--qk-norm (default on), --lr-scale (multiplies the heuristic rate, printed, recorded; no effect with --lr),
        LR_TABLE at twice the published rates with weight decay halved (lr x wd constant; the 2026-11-05 sweep),
        --ema (default off, on = 0.999; resolve_ema keeps a resumed run as it was) and --doc-mask (default off) in
        plan_run and parse_args."""
        manifest = {"tokenizer": {"vocab_size": 700},
                    "sources": {"md": {"train": {"tokens": 5_000_000}, "doc_tokens_percentiles": {"50": 300}}}}

        def plan(argv):
            args = tiny_gpt.parse_args(["--name", "x", "--d-model", "64"] + argv)
            with unittest.mock.patch.object(tiny_gpt, "device_memory_budget",
                                            return_value=(4 * 2 ** 30, "4 GiB (test)")):
                return tiny_gpt.plan_run(args, manifest, torch.device("cpu"), None)

        cfg, st, lines = plan([])
        text = "\n".join(lines)
        self.assertTrue(cfg.qk_norm)
        self.assertEqual((st["lr_scale"], st["ema"], st["doc_mask"], st["weight_decay"]), (1.0, 0.0, False, 0.05))
        self.assertIn(", QK-norm, ctx 256", text)
        self.assertIn("no EMA (opt-in: --ema 0.999)", text)
        self.assertNotIn("--lr-scale", text)
        for flag in ("on", "0.999"):
            _, st_e, lines_e = plan(["--ema", flag])
            self.assertEqual(st_e["ema"], 0.999)
            self.assertIn("weights: EMA with decay 0.999 per step", "\n".join(lines_e))
        cfg2, st2, lines2 = plan(["--lr-scale", "2", "--qk-norm", "off", "--ema", "off", "--doc-mask", "on"])
        text2 = "\n".join(lines2)
        self.assertAlmostEqual(st2["lr"], 2 * st["lr"])
        self.assertEqual((st2["lr_scale"], st2["ema"], st2["doc_mask"], cfg2.qk_norm), (2.0, 0.0, True, False))
        self.assertIn("x --lr-scale 2", text2)
        self.assertIn("no EMA (--ema off)", text2)
        self.assertIn("document mask on", text2)
        self.assertNotIn("QK-norm", text2)
        _, st3, lines3 = plan(["--lr", "1e-3", "--lr-scale", "2", "--ema", "0.99"])
        self.assertEqual((st3["lr"], st3["ema"], st3["lr_scale"]), (1e-3, 0.99, 1.0),
                         "--lr fixes the rate, so the recorded scale is the one the run carries")
        self.assertIn("learning rate 1.00e-03 peak (set)", "\n".join(lines3))
        self.assertIn("--lr-scale 2 has no effect", "\n".join(lines3))
        published = [(1e6, 2.0e-3), (1e7, 1.2e-3), (1e8, 6.0e-4), (1e9, 3.0e-4)]
        self.assertEqual(tiny_gpt.LR_TABLE_FACTOR, 2.0, "the factor both sweep sizes agreed on")
        self.assertEqual(tiny_gpt.LR_TABLE, [(n, tiny_gpt.LR_TABLE_FACTOR * lr) for n, lr in published],
                         "the published rates times one constant: the slope is unchanged")
        for n in (3e6, 2e7, 2.16e8, 5e8):
            self.assertAlmostEqual(tiny_gpt.heuristic_lr(n) * tiny_gpt.WEIGHT_DECAY_DEFAULT,
                                   tiny_gpt.heuristic_lr(n) / tiny_gpt.LR_TABLE_FACTOR * 0.1, places=15,
                                   msg="lr x weight decay as at the published rates with weight decay 0.1")
        self.assertEqual(tiny_gpt.parse_args(["--name", "x"]).weight_decay, 0.05)
        self.assertEqual(tiny_gpt.parse_args(["--name", "x", "--ema", "off"]).ema, 0.0)
        self.assertEqual(tiny_gpt.parse_args(["--name", "x", "--ema", "on"]).ema, 0.999)
        self.assertIsNone(tiny_gpt.parse_args(["--name", "x"]).ema)
        for bad in (["--lr-scale", "0"], ["--ema", "1.5"], ["--ema", "maybe"], ["--qk-norm", "maybe"],
                    ["--doc-mask", "1"]):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                with open(os.devnull, "w") as devnull:
                    old = sys.stderr
                    sys.stderr = devnull
                    try:
                        tiny_gpt.parse_args(["--name", "x"] + bad)
                    finally:
                        sys.stderr = old
        # resolve_ema: a resumed run never gains an average silently
        s = {}
        self.assertEqual((tiny_gpt.resolve_ema(s, None, resumed=True), s["ema"]), (0.0, 0.0))
        self.assertEqual(tiny_gpt.resolve_ema({"ema": 0.999}, None, resumed=True), 0.999)
        self.assertEqual(tiny_gpt.resolve_ema({}, 0.99, resumed=True), 0.99)
        self.assertEqual(tiny_gpt.resolve_ema({"ema": 0.999}, 0.0, resumed=True), 0.0)
        self.assertEqual(tiny_gpt.resolve_ema({}, None, resumed=False), 0.0, "new runs: no average unless asked")
        self.assertEqual(tiny_gpt.resolve_ema({}, 0.999, resumed=False), 0.999)
        self.assertEqual(tiny_gpt.resolve_ema({}, 0.0, resumed=False), 0.0)
        self.assertIn("lr_scale", tiny_gpt.EXPERIMENT_COLUMNS)
        self.assertIn("ema", tiny_gpt.EXPERIMENT_COLUMNS)

    def test_ema_update_arithmetic_and_weight_swap(self):
        """The EMA (quality-7): fp32 CPU copies that keep the tied embedding one tensor; ema <- d ema + (1 - d) w
        per update, compounding; swapped_weights scores the average in place and restores the raw weights, the
        parameter objects unchanged; the bf16 copy for --best-bf16 keeps the tie."""
        model = self.small_model(seed=2)
        ema = tiny_gpt.ema_init(model.state_dict().items())
        self.assertIs(ema["lm_head.weight"], ema["token_embedding.weight"], "tied storage stays one tensor")
        self.assertNotEqual(ema["norm.weight"].data_ptr(), model.norm.weight.data_ptr(), "a copy, not a view")
        self.assertEqual(ema["norm.weight"].dtype, torch.float32)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for k, v in ema.items():
            torch.testing.assert_close(v, before[k])
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        tiny_gpt.ema_update(ema, model, 0.9)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(ema[k], 0.9 * before[k] + 0.1 * v)
        tiny_gpt.ema_update(ema, model, 0.9)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(ema[k], 0.81 * before[k] + 0.19 * v)
        self.assertIs(ema["lm_head.weight"], ema["token_embedding.weight"])
        x = torch.randint(0, 50, (1, 8))
        with torch.no_grad():
            raw_out = model(x)
            norm_param = model.norm.weight
            with tiny_gpt.swapped_weights(model, ema) as swapped:
                self.assertIs(swapped, model)
                for k, v in model.state_dict().items():
                    torch.testing.assert_close(v, ema[k])
                self.assertIs(model.norm.weight, norm_param, "copied in place: the parameter objects stay")
                inside = model(x)
            torch.testing.assert_close(model(x), raw_out)
            self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        self.assertFalse(torch.allclose(inside, raw_out))

        class Hold:  # stands in for DeferInterrupt: both weight copies run inside it
            def __init__(self):
                self.entered, self.inside = 0, False

            def __enter__(self):
                self.entered += 1
                self.inside = True

            def __exit__(self, *exc):
                self.inside = False
                return False

        hold = Hold()
        with torch.no_grad(), tiny_gpt.swapped_weights(model, ema, hold=hold):
            self.assertEqual((hold.entered, hold.inside), (1, False), "held while copying, not while evaluating")
            torch.testing.assert_close(model.norm.weight, ema["norm.weight"])
        self.assertEqual(hold.entered, 2)
        with torch.no_grad():
            torch.testing.assert_close(model(x), raw_out)
        half = tiny_gpt.ema_weights_for_checkpoint(ema, bf16=True)
        self.assertEqual(half["norm.weight"].dtype, torch.bfloat16)
        self.assertIs(half["lm_head.weight"], half["token_embedding.weight"])
        self.assertIs(tiny_gpt.ema_weights_for_checkpoint(ema), ema)
        self.assertEqual(tiny_gpt.EMA_EVERY, 10)
        self.assertAlmostEqual(tiny_gpt.EMA_DECAY_DEFAULT ** tiny_gpt.EMA_EVERY, 0.990, places=3)

    def test_ema_is_debiased_from_its_first_update(self):
        """ema_decay_at: the recursion with b_n = d (1 - d^(n-1)) / (1 - d^n) gives exactly the zero-started
        average divided by its weight sum, sum_i (1 - d) d^(n-i) w_i / (1 - d^n), so the starting copy (the
        initial weights) carries no weight after the first update; n None keeps the plain decay (an average from
        a checkpoint without the count) and b_n reaches d once d^n is negligible."""
        d = 0.999 ** tiny_gpt.EMA_EVERY
        self.assertEqual(tiny_gpt.ema_decay_at(d, 1), 0.0)
        self.assertAlmostEqual(tiny_gpt.ema_decay_at(d, 2), d / (1 + d), places=15)
        self.assertEqual(tiny_gpt.ema_decay_at(d, None), d)
        self.assertAlmostEqual(tiny_gpt.ema_decay_at(d, 5000), d, places=15)
        torch.manual_seed(0)
        snapshots = [torch.randn(4, dtype=torch.float64) for _ in range(40)]
        start = torch.full((4,), 1e6, dtype=torch.float64)  # the placeholder: must vanish at the first update
        avg = start.clone()
        for n, w in enumerate(snapshots, 1):
            b = tiny_gpt.ema_decay_at(d, n)
            avg = b * avg + (1 - b) * w
            weights = torch.tensor([(1 - d) * d ** (n - i) for i in range(1, n + 1)], dtype=torch.float64)
            want = (weights[:, None] * torch.stack(snapshots[:n])).sum(0) / (1 - d ** n)
            torch.testing.assert_close(avg, want, rtol=1e-12, atol=1e-9)
        # through ema_update on a model: the first update copies the weights, the second averages the two
        model = self.small_model(seed=3)
        ema = tiny_gpt.ema_init(model.state_dict().items())
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        first = {k: v.detach().clone() for k, v in model.state_dict().items()}
        tiny_gpt.ema_update(ema, model, tiny_gpt.ema_decay_at(d, 1))
        for k, v in first.items():
            torch.testing.assert_close(ema[k], v.float())
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        tiny_gpt.ema_update(ema, model, tiny_gpt.ema_decay_at(d, 2))
        b2 = d / (1 + d)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(ema[k], (b2 * first[k] + (1 - b2) * v).float())

    def test_document_mask_blocks_and_single_document_logits(self):
        """The document mask (quality-6): a by-hand block-diagonal causal mask on a two-document window; a
        single-document window gives the unmasked logits; with the mask the second document of a window is scored
        as if it stood alone (RoPE is relative); the recomputed (checkpointed) blocks see the same mask; the
        memory estimate adds B x T x T bytes."""
        model = self.small_model(seed=4)  # ctx 16, vocab 50
        sep = 1
        x = torch.tensor([[sep, 10, 11, 12, sep, 20, 21, 22]])
        mask = tiny_gpt.document_mask(x, sep)
        self.assertEqual((tuple(mask.shape), mask.dtype), ((1, 1, 8, 8), torch.bool))
        want = torch.zeros(8, 8, dtype=torch.bool)
        for i in range(8):
            for j in range(i + 1):
                want[i, j] = (i < 4) == (j < 4)
        self.assertTrue(torch.equal(mask[0, 0], want))
        single = torch.tensor([[sep, 10, 11, 12, 13, 20, 21, 22]])
        self.assertTrue(torch.equal(tiny_gpt.document_mask(single, sep)[0, 0],
                                    torch.ones(8, 8, dtype=torch.bool).tril()))
        with torch.no_grad():
            plain = model(single)
            model.doc_sep_id = sep
            torch.testing.assert_close(model(single), plain, rtol=1e-5, atol=1e-5)
            two = model(x)
            alone = model(x[:, 4:])
            torch.testing.assert_close(two[:, 4:], alone, rtol=1e-4, atol=1e-4)
            if tiny_gpt._enable_gqa_supported():
                model.set_attention_impl("gqa_native")
                torch.testing.assert_close(model(x), two, rtol=1e-5, atol=1e-5)
                model.set_attention_impl("repeat_kv")
            model.doc_sep_id = None
            self.assertFalse(torch.allclose(model(x)[:, 4:], alone, atol=1e-3), "unmasked, the second document "
                                                                                   "sees the first")
        model.doc_sep_id = sep
        model.train()  # dropout 0: the recomputed blocks give the eval-mode logits
        model.grad_checkpoint = True
        out = model(x)
        torch.testing.assert_close(out, two, rtol=1e-5, atol=1e-5)
        out.sum().backward()
        self.assertIsNotNone(model.blocks[0].attn.q_proj.weight.grad)
        cfg = model.cfg
        self.assertEqual(tiny_gpt.estimate_train_bytes(cfg, 3, True, doc_mask=True)
                         - tiny_gpt.estimate_train_bytes(cfg, 3, True), 3 * cfg.ctx * cfg.ctx)
        self.assertEqual(tiny_gpt.estimate_train_bytes(cfg, 3, True, checkpointing=True, doc_mask=True)
                         - tiny_gpt.estimate_train_bytes(cfg, 3, True, checkpointing=True), 3 * cfg.ctx * cfg.ctx)

    # ---------------- mixture of experts ----------------
    def moe_model(self, seed=0, vocab=50, **fields):
        torch.manual_seed(seed)
        cfg = tiny_gpt.ModelConfig(**{"vocab_size": vocab, "ctx": 16, "d_model": 32, "n_layers": 2, "n_heads": 4,
                                      "n_kv_heads": 2, "n_experts": 4, "experts_per_token": 2, **fields})
        return tiny_gpt.TinyGPT(cfg)

    def test_moe_ffn_matches_masked_reference(self):
        """MoEFFN (moe-1, moe-3, moe-6): the sorted per-expert loop equals every expert run on every token and
        masked by the routing weights, in fp32 and under bf16 autocast; the grouped kernel gives the loop's
        gradients where the probe passes; param_counts and count_params agree on all four numbers; the module names
        and shapes are the documented ones; the dense model's forward is untouched."""
        import torch.nn.functional as F
        model = self.moe_model(seed=1).eval()
        ffn = model.blocks[0].ffn
        x = torch.randn(40, 32)

        def reference(x2):
            w, ids, _, _ = ffn.router(x2)
            mix = torch.zeros(x2.size(0), 4).scatter(1, ids, w)
            out = torch.zeros_like(x2)
            for e in range(4):
                gate, up = F.linear(x2, ffn.experts.gate_up[e]).chunk(2, dim=-1)
                out = out + mix[:, e:e + 1] * F.linear(F.silu(gate) * up, ffn.experts.down[e])
            return out

        with torch.no_grad():
            ref = reference(x)
            got, (lb, z, counts) = ffn(x)
            torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-7)
            self.assertEqual(int(counts.sum()), 40 * 2)
            self.assertEqual(tuple(counts.shape), (4,))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                got_bf, (lb_bf, _, counts_bf) = ffn(x)
            self.assertEqual(got_bf.dtype, torch.float32, "accumulated into the residual stream's dtype")
            self.assertLess((got_bf - ref).abs().max().item(), 0.03 * ref.abs().max().item())
            self.assertTrue(torch.equal(counts_bf, counts), "the router decides in fp32 under autocast")
            self.assertAlmostEqual(lb_bf.item(), lb.item(), places=5)
        ok, why = tiny_gpt.grouped_mm_available(torch.device("cpu"))
        if ok:
            grads = {}
            for impl in ("grouped", "loop"):
                ffn.experts.impl = impl
                ffn.zero_grad()
                out, (lb, z, _) = ffn(x)
                (out.pow(2).sum() * 1e3 + lb + z).backward()
                grads[impl] = torch.cat([p.grad.flatten() for p in ffn.parameters()])
            torch.testing.assert_close(grads["grouped"], grads["loop"], rtol=1e-4, atol=1e-7)
            with torch.no_grad():  # the cached probe answers inside no_grad as well (model build may sit in one)
                self.assertTrue(tiny_gpt.grouped_mm_available(torch.device("cpu"))[0])
        else:
            self.assertIn("using the per-expert loop", why)
        cfg = model.cfg
        self.assertEqual(cfg.expert_hidden, 64, "SwiGLU 128 / 2 experts per token, rounded to 64")
        self.assertEqual(model.param_counts(), tiny_gpt.count_params(cfg))
        total, non_embed, active, non_embed_active = tiny_gpt.count_params(cfg)
        self.assertEqual(total - active, 2 * 3 * 32 * 64 * (4 - 2))
        self.assertEqual(total - non_embed, 50 * 32)
        self.assertEqual(non_embed - non_embed_active, total - active)
        for fields in ({"moe_every": 2}, {"shared_hidden": 32}, {"tie_embeddings": False, "n_experts": 3,
                                                                  "experts_per_token": 1}):
            m = self.moe_model(**fields)
            self.assertEqual(m.param_counts(), tiny_gpt.count_params(m.cfg), fields)
        self.assertEqual([b.moe for b in self.moe_model(moe_every=2).blocks], [False, True])
        names = dict(model.named_parameters())
        self.assertEqual(tuple(names["blocks.0.ffn.router.weight"].shape), (4, 32))
        self.assertEqual(tuple(names["blocks.0.ffn.experts.gate_up.3"].shape), (128, 32))
        self.assertEqual(tuple(names["blocks.1.ffn.experts.down.0"].shape), (32, 64))
        self.assertIn("blocks.0.ffn.router.bias", model.state_dict(), "the bias is a persistent buffer")
        self.assertNotIn("blocks.0.ffn.router.bias", names, "never a parameter")
        self.assertEqual(tiny_gpt.ModelConfig(**dataclasses.asdict(cfg)).n_experts, 4)
        tokens = torch.randint(0, 50, (2, 16))
        with torch.no_grad():
            logits, aux = model(tokens, return_aux=True)
            self.assertEqual(tuple(logits.shape), (2, 16, 50))
            self.assertEqual(tuple(aux["counts"].shape), (2, 4))
            self.assertTrue(torch.is_tensor(model(tokens)), "without return_aux: the single tensor it always was")
            dense = self.small_model(seed=1)
            self.assertIsNone(dense(tokens, return_aux=True)[1])
            torch.testing.assert_close(dense(tokens, return_aux=True)[0], dense(tokens))
        # the experts' down matrices and the shared SwiGLU's down get the residual scaling at initialisation
        torch.manual_seed(0)
        fresh = self.moe_model(shared_hidden=32)
        self.assertLess(fresh.blocks[0].ffn.experts.down[0].std().item(), 0.015)
        self.assertLess(fresh.blocks[0].ffn.shared.down.weight.std().item(), 0.015)
        self.assertGreater(fresh.blocks[0].ffn.experts.gate_up[0].std().item(), 0.015)

    def test_moe_balance_terms_and_bias_update(self):
        """The aux terms (moe-5, moe-4): the Switch loss is exactly 1.0 at uniform routing and above 1 when the
        router collapses; the counts sum to tokens x k from a static-shape scatter; the router z-loss is
        (log E)^2 at zero logits; update_router_bias lowers the overloaded expert's bias and raises the others, and
        the bias changes which experts are chosen but not the mixing weights."""
        model = self.moe_model(seed=2).eval()
        ffn = model.blocks[0].ffn
        x = torch.randn(64, 32)
        with torch.no_grad():
            ffn.router.weight.zero_()  # every token: uniform probabilities over the 4 experts
            _, (lb, z, counts) = ffn(x)
            self.assertAlmostEqual(lb.item(), 1.0, places=6)
            self.assertAlmostEqual(z.item(), math.log(4) ** 2, places=5)
            self.assertEqual(int(counts.sum()), 64 * 2)
            # collapse: logits (10, 5, 0, 0) for every token (a constant input feature times fixed weights)
            x[:, 0] = 1.0
            ffn.router.weight[0, 0], ffn.router.weight[1, 0] = 10.0, 5.0
            _, (lb_c, _, counts_c) = ffn(x)
            self.assertEqual(counts_c.tolist(), [64, 64, 0, 0])
            self.assertGreater(lb_c.item(), 1.9)
            # the bias: expert 0 overloaded in the step's counts -> its bias falls by the rate, the others rise
            model.update_router_bias(torch.tensor([[100, 10, 10, 8], [1, 1, 1, 1]]), 0.1)
            torch.testing.assert_close(ffn.router.bias, torch.tensor([-0.1, 0.1, 0.1, 0.1]))
            torch.testing.assert_close(model.blocks[1].ffn.router.bias, torch.zeros(4), msg="no expert above the mean")
            # a bias large enough reroutes the tokens away from expert 0; the mixing weights still come from the
            # scores alone (renormalised over the chosen experts)
            ffn.router.bias[0] = -100.0
            w, ids, logits, counts_b = ffn.router(x)
            self.assertEqual(counts_b[0].item(), 0)
            self.assertEqual(ids[0].tolist(), [1, 2])
            probs = logits.softmax(-1)[0]
            torch.testing.assert_close(w[0], probs[[1, 2]] / probs[[1, 2]].sum())
            ffn.router.bias.zero_()
            # the sigmoid router
            sig = self.moe_model(seed=2, router="sigmoid").eval()
            out, (lb_s, _, counts_s) = sig.blocks[0].ffn(x)
            self.assertEqual(int(counts_s.sum()), 128)
            self.assertGreater(lb_s.item(), 0.0)
        # the dense model has no router: update_router_bias is a no-op and the balance settings are plain fields
        dense = self.small_model()
        dense.update_router_bias(torch.zeros(0, 4), 0.1)
        self.assertEqual(dense.cfg.balance, "bias")

    def test_moe_planner_memory_counts_and_validation(self):
        """The planner (moe-6, moe-7, moe-10, moe-15): dense by default (E = 0); --experts N sizes the experts
        from the SwiGLU hidden, records the balance terms and plans the learning rate and batch from the ACTIVE
        count while the memory estimate grows with the number of experts (the total); flops_per_token uses the
        active count; the validation refuses bad expert settings; a checkpoint written with unknown config fields
        or by a newer script is refused with a clear message; the router stays off Muon."""
        manifest = {"tokenizer": {"vocab_size": 700},
                    "sources": {"md": {"train": {"tokens": 20_000_000}, "doc_tokens_percentiles": {"50": 300}}}}
        budget = (4 * 2 ** 30, "4 GiB (test)")

        def plan(argv):
            args = tiny_gpt.parse_args(["--name", "x", "--d-model", "128", "--layers", "4"] + argv)
            with unittest.mock.patch.object(tiny_gpt, "device_memory_budget", return_value=budget):
                return tiny_gpt.plan_run(args, manifest, torch.device("cpu"), None)

        dense_cfg, dense_st, dense_lines = plan([])
        self.assertEqual((dense_cfg.n_experts, dense_cfg.expert_hidden), (0, 0), "dense is the default")
        self.assertNotIn("lb_loss", dense_st)
        self.assertNotIn("MoE", "\n".join(dense_lines))
        cfg, st, lines = plan(["--experts", "8", "--balance", "both", "--lb-loss", "0.02", "--tokens-per-total-param",
                               "50"])
        text = "\n".join(lines)
        self.assertEqual((cfg.n_experts, cfg.experts_per_token, cfg.expert_hidden, cfg.balance),
                         (8, 2, tiny_gpt.expert_hidden_for(cfg.ffn_hidden, 2), "both"))
        self.assertEqual((st["lb_loss"], st["router_z_loss"], st["bias_rate"]), (0.02, 1e-3, 1e-3))
        self.assertIn("MoE: 8 experts of hidden", text)
        self.assertRegex(text, r"parameters [\d,]+ total, [\d,]+ active per token")
        self.assertIn("tokens per TOTAL parameter is below the --tokens-per-total-param floor 50", text)
        self.assertNotIn("floor", "\n".join(plan(["--experts", "8", "--tokens-per-total-param", "20"])[2]))
        total, _, active, _ = tiny_gpt.count_params(cfg)
        self.assertLess(active, total)
        self.assertAlmostEqual(st["lr"], tiny_gpt.heuristic_lr(active) * tiny_gpt.lr_batch_factor(
            active, st["tokens_per_step"]), places=12, msg="the learning rate follows the active count")
        self.assertEqual(tiny_gpt.flops_per_token(cfg), 6 * active + 12 * cfg.n_layers * cfg.ctx * cfg.d_model)
        est = [tiny_gpt.estimate_train_bytes(tiny_gpt.ModelConfig(**{**dataclasses.asdict(cfg), "n_experts": e,
                                                                     "expert_hidden": cfg.expert_hidden}), 4, True,
                                             optimizer="muon") for e in (0, 4, 8, 16)]
        self.assertEqual(est, sorted(est), "memory grows with the number of experts")
        self.assertLess(est[0], est[1])
        # the experiments-table label round-trips into a configuration
        self.assertEqual(cfg.moe_label(), "8x2/1")
        self.assertEqual(tiny_gpt.parse_moe_label("8x2/1"), {"n_experts": 8, "experts_per_token": 2, "moe_every": 1})
        self.assertEqual(tiny_gpt.parse_moe_label(""), {})
        self.assertEqual(tiny_gpt.expert_hidden_for(2816, 2), 1408)
        self.assertEqual(tiny_gpt.expert_hidden_for(192, 2), 128)
        for bad in ({"n_experts": 1}, {"n_experts": 100}, {"n_experts": 4, "experts_per_token": 5},
                    {"n_experts": 4, "moe_every": 0}, {"n_experts": 4, "moe_every": 1000},
                    {"n_experts": 4, "router": "argmax"},
                    {"n_experts": 4, "balance": "none"}, {"n_experts": 4, "expert_hidden": -1}):
            with self.assertRaises(SystemExit, msg=bad):
                tiny_gpt.ModelConfig(vocab_size=50, d_model=32, n_heads=4, n_kv_heads=2, **bad)
        for argv in (["--experts", "1"], ["--experts", "4", "--experts-per-token", "5"], ["--moe-every", "0"]):
            with self.assertRaises(SystemExit, msg=argv), redirect_stdout(io.StringIO()):
                with unittest.mock.patch("sys.stderr", io.StringIO()):
                    tiny_gpt.parse_args(["--name", "x"] + argv)
        # checkpoints: every old one loads (the fields default to dense); unknown fields are a clear refusal
        self.assertEqual(tiny_gpt.FORMAT_VERSION, 5)
        model = self.moe_model(seed=3)
        obj = {"format": tiny_gpt.CHECKPOINT_FORMAT, "checkpoint_type": "inference", "format_version": 5,
               "model_config": dataclasses.asdict(model.cfg), "model": model.state_dict()}
        self.assertEqual(tiny_gpt.config_from_checkpoint(obj).n_experts, 4)
        old = {k: v for k, v in obj["model_config"].items() if k not in ("n_experts", "experts_per_token",
                                                                          "expert_hidden", "shared_hidden",
                                                                          "moe_every", "router", "balance")}
        self.assertEqual(tiny_gpt.config_from_checkpoint({**obj, "model_config": old}).n_experts, 0)
        with self.assertRaises(SystemExit) as ctx:
            tiny_gpt.config_from_checkpoint({**obj, "model_config": {**old, "n_galaxies": 3}})
        self.assertIn("['n_galaxies']", str(ctx.exception.code))
        self.assertNotIn("newer", str(ctx.exception.code))
        with self.assertRaises(SystemExit) as ctx:
            tiny_gpt.config_from_checkpoint({**obj, "format_version": 99, "model_config": {**old, "n_galaxies": 3}})
        self.assertIn("written by a newer tiny_gpt.py", str(ctx.exception.code))
        again = tiny_gpt.TinyGPT(tiny_gpt.config_from_checkpoint(obj)).eval()
        self.assertEqual(again.load_state_dict(obj["model"], strict=False).unexpected_keys, [])
        x = torch.randint(0, 50, (1, 16))
        with torch.no_grad():
            torch.testing.assert_close(again(x), model.eval()(x))
        # optimizer groups: the experts' matrices on Muon, the router on AdamW with decay, the bias nowhere
        settings = {"optimizer": "muon", "lr": 1e-3, "betas": [0.9, 0.95], "weight_decay": 0.1}
        _, _, names, note = tiny_gpt.build_optimizer(model, settings, torch.device("cpu"))
        self.assertIn("blocks.0.ffn.router.weight", names[0])
        self.assertNotIn("blocks.0.ffn.router.weight", names[2])
        self.assertIn("blocks.1.ffn.experts.gate_up.3", names[2])
        self.assertIn("blocks.1.ffn.experts.down.0", names[2])
        self.assertEqual(len(names[2]), 2 * (4 + 4 * 2))
        self.assertIn("Muon for 24 block matrices", note)
        # the dense regression (moe-14 item 7): a dense Muon run's groups and log line are what they were
        _, _, dense_names, dense_note = tiny_gpt.build_optimizer(self.small_model(), settings, torch.device("cpu"))
        self.assertEqual(len(dense_names[2]), 14)
        self.assertEqual(dense_names[0], ["token_embedding.weight"])
        self.assertIn("Muon for 14 block matrices, its step scaled to AdamW's update size; AdamW for the embedding "
                      "(1 matrix) and 5 norm gains; weight decay on all matrices", dense_note)
        self.assertEqual(sum(len(g) for g in dense_names), 20, "optimizer state 20/20 tensors on resume")

    def test_moe_kv_cache_matches_full_recompute(self):
        """KV-cached decoding of a mixture-of-experts model (moe-16): the experts act per token, so the cache
        needs nothing; a prompt, one token at a time and several tokens after a prefix give the full logits; the
        decode steps skip the experts nobody chose."""
        model = self.moe_model(seed=3).eval()
        gen = torch.Generator().manual_seed(0)
        seq = torch.randint(0, 50, (2, 16), generator=gen)
        with torch.no_grad():
            full = model(seq)
            cache = model.new_cache()
            parts = [model(seq[:, :5], cache=cache)]
            parts += [model(seq[:, i:i + 1], cache=cache) for i in range(5, 9)]
            parts.append(model(seq[:, 9:16], cache=cache))
            torch.testing.assert_close(torch.cat(parts, dim=1), full, rtol=1e-4, atol=1e-5)
            with self.assertRaises(ValueError):
                model(seq[:, :1], cache=cache)
            _, aux = model(seq[:, :1], return_aux=True)
            self.assertEqual(int(aux["counts"].sum()), 2 * 2 * 2, "2 blocks x 2 sequences of one token x 2 experts")
        dec = tiny_gpt.IncrementalDecoder(model, torch.device("cpu"))
        ids = seq[0].tolist()[:3]
        for _ in range(20):
            got = dec.next_logits(ids)
            if dec.base == max(0, len(ids) - 16):
                want = model(torch.tensor([ids[-16:]]))[0, -1]
                torch.testing.assert_close(want, got, rtol=1e-4, atol=1e-5)
            ids.append(int(got.argmax()))
        self.assertEqual(len(ids), 23)

    # ---------------- WP9: numerics, sampling and planner helpers without a dataset ----------------
    def test_grad_scaler_skips_the_combined_optimizer_on_inf(self):
        """GradScaler (the FP16 path) with the Muon + AdamW CombinedOptimizer: an infinite gradient element makes
        scaler.step skip the update of BOTH optimizers (no weight and no optimizer state changes) and halves the
        loss scale; a finite step afterwards updates every parameter. GradScaler('cpu') runs the same bookkeeping
        as on CUDA, so the skip semantics are checked without a GPU."""
        model = self.small_model(seed=7).train()
        settings = {"optimizer": "muon", "lr": 1e-3, "betas": [0.9, 0.95], "weight_decay": 0.1}
        opt = tiny_gpt.build_optimizer(model, settings, torch.device("cpu"))[0]
        self.assertIsInstance(opt, tiny_gpt.CombinedOptimizer)
        params = [p for g in opt.param_groups for p in g["params"]]
        scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 10)
        x = torch.randint(0, 50, (2, 16), generator=torch.Generator().manual_seed(0))

        def backward(poison=False):
            loss = torch.nn.functional.cross_entropy(model(x[:, :-1]).reshape(-1, 50), x[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            if poison:
                params[0].grad.view(-1)[0] = float("inf")
            scaler.unscale_(opt)

        def snapshot():
            sd = opt.state_dict()["state"]
            return ([p.detach().clone() for p in params],
                    {i: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()} for i, st in sd.items()})

        backward()
        scaler.step(opt)  # a normal step first, so there is optimizer state to leave untouched
        scaler.update()
        opt.zero_grad(set_to_none=True)
        self.assertEqual(scaler.get_scale(), 2.0 ** 10)
        weights, state = snapshot()
        self.assertEqual(len(state), len(params), "both optimizers hold a state per parameter after one step")
        backward(poison=True)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        changed = sum(int(not torch.equal(w, p.detach())) for w, p in zip(weights, params))
        self.assertEqual(changed, 0, f"{changed}/{len(params)} parameters changed by the skipped update")
        after = opt.state_dict()["state"]
        for i, st in state.items():
            for k, v in st.items():
                same = torch.equal(v, after[i][k]) if torch.is_tensor(v) else v == after[i][k]
                self.assertTrue(same, f"optimizer state {i}/{k} changed by the skipped update")
        self.assertEqual(scaler.get_scale(), 2.0 ** 9, "the scale halves after an overflow")
        backward()
        scaler.step(opt)
        scaler.update()
        changed = sum(int(not torch.equal(w, p.detach())) for w, p in zip(weights, params))
        self.assertEqual(changed, len(params), "a finite step updates every parameter")

    def test_kv_cache_under_bf16_autocast(self):
        """Generation in bfloat16 (amp_dtype as on the GPU and the Mac): the KV-cached decoder agrees with a full
        bf16 forward pass within bf16 rounding (atol 0.05) and picks the same greedy token as the FP32 model;
        generate() under autocast gives the greedy FP32 continuation."""
        torch.manual_seed(11)
        cfg = tiny_gpt.ModelConfig(vocab_size=50, ctx=16, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2)
        model = tiny_gpt.TinyGPT(cfg).eval()
        with torch.no_grad():  # a larger output scale, so the greedy choice is clear of bf16 rounding
            model.lm_head.weight.mul_(8.0)
        seq = torch.randint(0, 50, (1, 16), generator=torch.Generator().manual_seed(2)).tolist()[0]
        bf16 = torch.bfloat16
        with torch.no_grad():
            full32 = model(torch.tensor([seq]))[0].float()
            with torch.autocast("cpu", dtype=bf16):
                full16 = model(torch.tensor([seq]))[0].float()
        dec = tiny_gpt.IncrementalDecoder(model, torch.device("cpu"), bf16)
        for i in range(4, 17):  # the prompt (4 tokens), then one token at a time
            got = dec.next_logits(seq[:i])
            self.assertEqual(got.dtype, torch.float32)
            self.assertTrue(torch.allclose(got, full16[i - 1], atol=0.05), (got - full16[i - 1]).abs().max())
            self.assertEqual(int(got.argmax()), int(full32[i - 1].argmax()), f"greedy token at position {i}")
        tok = CharTok()
        a = tiny_gpt.generate(model, tok, "drift", 10, torch.device("cpu"), bf16, temperature=0)
        b = tiny_gpt.generate(model, tok, "drift", 10, torch.device("cpu"), None, temperature=0)
        self.assertEqual(a, b)

    def test_planner_helpers_are_finite_and_bounded(self):
        """swiglu_hidden is 8d/3 rounded up to a multiple of 64 (2752 at d 1024, the 216M run's FFN);
        count_params equals the built model's parameter count, tied or untied, MHA or GQA, with or without
        QK-norm; heuristic_lr stays finite inside its table and never rises with the model; tokens_per_step_for is
        a power of two in [16,384, 2,097,152] for any size and budget; default_ctx and the RoPE base rule are
        bounded and monotone."""
        self.assertEqual(tiny_gpt.swiglu_hidden(1024), 2752)
        for d in range(64, 4097, 64):
            h = tiny_gpt.swiglu_hidden(d)
            self.assertEqual(h % 64, 0)
            self.assertTrue(8 * d / 3 <= h < 8 * d / 3 + 64, (d, h))
        for tie in (True, False):
            for kv in (4, 2, 1):
                for qk in (False, True):
                    cfg = tiny_gpt.ModelConfig(vocab_size=70, ctx=16, d_model=32, n_layers=2, n_heads=4,
                                               n_kv_heads=kv, tie_embeddings=tie, qk_norm=qk)
                    model = tiny_gpt.TinyGPT(cfg)
                    unique = sum(p.numel() for p in model.parameters())  # parameters() lists a tied matrix once
                    self.assertEqual(tiny_gpt.count_params(cfg)[0], unique, (tie, kv, qk))
                    self.assertEqual(tuple(tiny_gpt.count_params(cfg)), tuple(model.param_counts()), (tie, kv, qk))
                    embed = 70 * 32 * (1 if tie else 2)
                    self.assertEqual(tiny_gpt.count_params(cfg)[1], unique - embed)
        lrs = [lr for _, lr in tiny_gpt.LR_TABLE]
        prev = float("inf")
        for n in np.logspace(2, 13, 89):
            lr = tiny_gpt.heuristic_lr(float(n))
            self.assertTrue(math.isfinite(lr) and min(lrs) - 1e-15 <= lr <= max(lrs) + 1e-15, (n, lr))
            self.assertLessEqual(lr, prev + 1e-15, f"the heuristic LR rose at N = {n:.3g}")
            prev = lr
            for per_param in (1, 20, 200):
                tps = tiny_gpt.tokens_per_step_for(float(n), per_param * float(n))
                self.assertEqual(tps & (tps - 1), 0, tps)
                self.assertTrue(2 ** 14 <= tps <= 2 ** 21, (n, per_param, tps))
        self.assertAlmostEqual(tiny_gpt.heuristic_lr(0), max(lrs), places=15, msg="no parameters: the first row")
        self.assertEqual(tiny_gpt.tokens_per_step_for(0, 0), 2 ** 14)
        prev_ctx = 0
        for d in range(64, 8193, 64):
            ctx = tiny_gpt.default_ctx(d)
            self.assertIn(ctx, (256, 512, 1024, 2048, 4096))
            self.assertGreaterEqual(ctx, prev_ctx)
            prev_ctx = ctx
        bases = [tiny_gpt.default_rope_base(2 ** k) for k in range(6, 18)]
        self.assertEqual(bases, sorted(bases))
        self.assertEqual(min(bases), 10_000.0)

    class _FixedLogits(torch.nn.Module):
        """A stand-in model whose next-token logits are always the same (not a TinyGPT, so generate() calls it
        on the whole window, as for wrappers)."""
        ctx = 32

        def __init__(self, logits):
            super().__init__()
            self.logits = logits

        def forward(self, x):
            return self.logits.expand(x.shape[0], x.shape[1], -1)

    class _IdTok:
        """Encodes to fixed ids and decodes to the id list itself, so a test can read the sampled ids."""
        bos_id = -1

        def __init__(self, eos_id=-1):
            self.eos_id = eos_id

        def encode(self, text):
            return [0]

        def decode(self, ids):
            return list(ids)

    def test_sampling_constraints(self):
        """generate(): top_k keeps only the k most likely tokens, top_p the smallest set whose mass reaches p,
        a temperature near 0 (or 0) is greedy, and the end-of-document token stops generation without being
        emitted."""
        probs = torch.tensor([0.02, 0.05, 0.08, 0.40, 0.03, 0.10, 0.02, 0.30])  # most likely: 3, then 7, then 5
        model = self._FixedLogits(probs.log())
        cpu = torch.device("cpu")

        def sample(n=400, eos=-1, **kw):
            gen = torch.Generator().manual_seed(0)
            return tiny_gpt.generate(model, self._IdTok(eos), "x", n, cpu, generator=gen, **kw)[1:]

        top2 = sample(temperature=1.0, top_k=2, top_p=1.0)
        self.assertEqual(set(top2), {3, 7}, "top_k 2: only the two most likely tokens")
        self.assertAlmostEqual(top2.count(3) / len(top2), 0.4 / 0.7, delta=0.08)
        nucleus = sample(temperature=1.0, top_k=0, top_p=0.5)
        self.assertEqual(set(nucleus), {3, 7}, "top_p 0.5: 0.40 + 0.30 is the smallest set reaching half the mass")
        nucleus75 = sample(temperature=1.0, top_k=0, top_p=0.75)
        self.assertEqual(set(nucleus75), {3, 7, 5})
        free = sample(temperature=1.0, top_k=0, top_p=1.0)
        self.assertEqual(set(free), set(range(8)), "no limits: every token can be drawn")
        self.assertEqual(sample(50, temperature=1e-6), [3] * 50, "a temperature near 0 is greedy")
        self.assertEqual(sample(50, temperature=0), [3] * 50)
        self.assertEqual(sample(50, eos=3, temperature=0), [], "end-of-document first: nothing is emitted")
        stopped = sample(400, eos=7, temperature=1.0, top_k=2, top_p=1.0)
        self.assertNotIn(7, stopped)
        self.assertLess(len(stopped), 400, "generation stops at the end-of-document token")

    def test_parse_mix_and_sampled_shares(self):
        """--mix: parse_mix accepts name=0 and rejects unknown names, missing '=', non-numbers, negative and
        all-zero weights; TokenBatcher's drawn windows follow share() (2,000 windows within 3 points), and a source
        with weight 0 is never drawn."""
        sources = {"a": 1, "b": 1, "c": 1}
        self.assertEqual(tiny_gpt.parse_mix("a=3, b=0", sources), {"a": 3.0, "b": 0.0, "c": 1.0})
        self.assertEqual(tiny_gpt.parse_mix(None, sources), {"a": 1.0, "b": 1.0, "c": 1.0})
        for bad in ("d=1", "a", "a=x", "a=-1", "a=0,b=0,c=0"):
            with self.assertRaises(SystemExit, msg=bad):
                tiny_gpt.parse_mix(bad, sources)
        arrays = {"a": np.arange(0, 25, dtype=np.int64).repeat(40), "b": np.arange(25, 50, dtype=np.int64).repeat(40),
                  "c": np.arange(50, 75, dtype=np.int64).repeat(40)}  # the token value tells the source
        batcher = tiny_gpt.TokenBatcher(arrays, tiny_gpt.parse_mix("a=3,b=1,c=0", arrays), 8, 100,
                                        torch.device("cpu"), 0)
        share = batcher.share()
        self.assertEqual(sorted(share), ["a", "b"])
        self.assertAlmostEqual(share["a"], 0.75, places=6)
        drawn = torch.cat([batcher.next()[0][:, 0] for _ in range(20)])
        self.assertEqual(drawn.numel(), 2000)
        self.assertEqual(int((drawn >= 50).sum()), 0, "weight 0: never drawn")
        self.assertAlmostEqual(float((drawn < 25).float().mean()), 0.75, delta=0.03)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("TINYGPT_GPU_TESTS"),
                         "needs CUDA and TINYGPT_GPU_TESTS=1 (never while a training run occupies the GPU)")
    def test_training_peak_memory_within_the_estimate(self):
        """estimate_train_bytes against the CUDA allocator: three real training micro-steps (bf16 autocast, AdamW
        and Muon, with and without checkpointing) never allocate more than 1.3 x the estimate the planner uses."""
        dev = torch.device("cuda")
        cfg = tiny_gpt.ModelConfig(vocab_size=8192, ctx=256, d_model=256, n_layers=4, n_heads=8, n_kv_heads=2)
        for optimizer in ("adamw", "muon"):
            for checkpointing in (False, True):
                settings = {"optimizer": optimizer, "lr": 1e-4, "weight_decay": 0.0, "betas": [0.9, 0.95]}
                _, peak = tiny_gpt.measure_seconds_per_step(cfg, 4, 1, dev, torch.bfloat16, warmup=1, timed=2,
                                                            checkpointing=checkpointing, settings=settings)
                estimate = tiny_gpt.estimate_train_bytes(cfg, 4, True, checkpointing, 0, optimizer, dev)
                self.assertLessEqual(peak, 1.3 * estimate, (optimizer, checkpointing, peak, estimate))


class TestOtherScripts(unittest.TestCase):
    """run_pipeline, compare_models, tools/edit_weights.py, finetune, benchmark's batched decoding, view_pt's
    window budget, estimate_dataset and load_for_inference / resolve_dataset (fast, CPU, temporary folders)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_scripts_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def model(seed=0, ctx=48, vocab=50, varied=True, **moe):
        """A tiny model; varied=True draws the matrices wider than the training init, so greedy decoding does
        not just repeat one token (a fresh model's outputs are nearly flat)."""
        torch.manual_seed(seed)
        cfg = tiny_gpt.ModelConfig(vocab_size=vocab, ctx=ctx, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, **moe)
        model = tiny_gpt.TinyGPT(cfg).eval()
        if varied:
            with torch.no_grad():
                for p in model.parameters():
                    if p.dim() == 2:
                        p.normal_(0, 0.4)
        return model

    def save_checkpoint(self, model, name, **extra):
        path = os.path.join(self.tmp, name)
        torch.save({"format": tiny_gpt.CHECKPOINT_FORMAT, "format_version": tiny_gpt.FORMAT_VERSION,
                    "checkpoint_type": "inference", "model_config": dataclasses.asdict(model.cfg),
                    "model": model.state_dict(), "step": 7, **extra}, path)
        return path

    # ---------------- run_pipeline (other-scripts-3, -19) ----------------
    def test_run_pipeline_commands_follow_the_manifest_and_parse(self):
        """build_command takes superbpe, tokenizer weights (of folders that exist), wiki_max_docs and
        near_dup_max_docs from a built dataset's manifest (the retired 3,000,000 default is not repeated), and
        its options and train_command's parse with data_prep's and tiny_gpt's own parsers; continue trains with
        the base checkpoint's optimizer, read with weights_only=True."""
        import run_pipeline
        here = os.path.join(self.tmp, "pipe")
        data = os.path.join(here, "data")
        for sub in ("books", "notes", "wikipedia"):
            os.makedirs(os.path.join(data, sub), exist_ok=True)
        dataset = os.path.join(here, "datasets", "bio_all")
        os.makedirs(dataset, exist_ok=True)
        prep = {"superbpe": True, "tokenizer_weights": "books=3,articles=3", "wiki_max_docs": 1234,
                "near_dup_max_docs": 3_000_000}
        with open(os.path.join(dataset, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"prep_args": prep}, f)
        with unittest.mock.patch.multiple(run_pipeline, HERE=here, DATASET=dataset, DATA=data, SUPERBPE=False):
            cmd = run_pipeline.build_command()
            self.assertIn("--superbpe", cmd, "the manifest's SuperBPE wins over the constant")
            self.assertEqual(cmd[cmd.index("--tokenizer-weights") + 1], "books=3", "no articles folder here")
            with open(os.path.join(dataset, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"prep_args": {**prep, "tokenizer_weights": "books=3,wiki=2,fineweb=0.5"}}, f)
            weighted = run_pipeline.build_command()
            self.assertEqual(weighted[weighted.index("--tokenizer-weights") + 1], "books=3,wiki=2",
                             "a named source the command reads keeps its weight; one it does not read is dropped")
            with open(os.path.join(dataset, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"prep_args": prep}, f)
            self.assertEqual(cmd[cmd.index("--wiki-max-docs") + 1], "1234")
            self.assertNotIn("--near-dup-max-docs", cmd, "the old default is not a choice to repeat")
            args = data_prep.parse_args(cmd[2:])
            self.assertTrue(args.superbpe)
            self.assertEqual((args.wiki_max_docs, args.near_dup_max_docs, args.near_dup), (1234, 0, "drop"))
            with open(os.path.join(dataset, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"prep_args": {**prep, "near_dup_max_docs": 5_000_000, "superbpe": False}}, f)
            cmd = run_pipeline.build_command()
            self.assertEqual(cmd[cmd.index("--near-dup-max-docs") + 1], "5000000")
            self.assertNotIn("--superbpe", cmd)
            self.assertEqual(data_prep.parse_args(cmd[2:]).near_dup_max_docs, 5_000_000)
            os.remove(os.path.join(dataset, "manifest.json"))
            cmd = run_pipeline.build_command()  # no manifest: the constants
            self.assertNotIn("--superbpe", cmd)
            self.assertEqual(cmd[cmd.index("--wiki-max-docs") + 1], str(run_pipeline.WIKI_MAX_DOCS))
            train = run_pipeline.train_command(12, "R", plan=True)
            targs = tiny_gpt.parse_args(train[2:])
            self.assertEqual((targs.name, targs.optimizer, targs.time_budget_hours, targs.plan),
                             ("R", run_pipeline.OPTIMIZER, 12.0, True))
        self.assertEqual((run_pipeline.RUN_NAME, run_pipeline.OPTIMIZER, run_pipeline.SUPERBPE),
                         ("tinyGPT_muon", "muon", True))
        # continue: the optimizer comes from the base checkpoint's train_config
        base = self.save_checkpoint(self.model(varied=False), "R_best.pt", train_config={"optimizer": "adamw"})
        self.assertEqual(run_pipeline.checkpoint_optimizer(base), "adamw")
        self.assertEqual(run_pipeline.checkpoint_optimizer(os.path.join(self.tmp, "missing.pt")),
                         run_pipeline.OPTIMIZER)
        with open(os.path.join(dataset, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"prep_args": prep}, f)
        os.makedirs(os.path.join(here, "new_ds"), exist_ok=True)
        open(os.path.join(here, "new_ds", "manifest.json"), "w").close()
        shutil.copy(base, os.path.join(here, "R_best.pt"))
        calls = []
        with unittest.mock.patch.multiple(run_pipeline, HERE=here, DATASET=dataset, DATA=data), \
                unittest.mock.patch.object(run_pipeline, "run", side_effect=lambda cmd: calls.append(cmd) or 0), \
                unittest.mock.patch.object(run_pipeline, "gpu_busy", return_value=False), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_pipeline.step_continue(1, "R", os.path.join(here, "new_ds")), 0)
        self.assertEqual(calls[-1][calls[-1].index("--optimizer") + 1], "adamw")
        self.assertEqual(calls[-1][calls[-1].index("--init-from") + 1], os.path.join(here, "R_best.pt"))

    # ---------------- load_for_inference, resolve_dataset (other-scripts-10, -13) ----------------
    def test_load_for_inference_drops_the_second_copy(self):
        """The model holds the checkpoint's own tensors and the returned dict has no "model" (one copy of the
        weights); everything else is still there and generation equals the old copying path."""
        path = self.save_checkpoint(self.model(seed=4), "L.pt", metrics={"best_val": 2.5},
                                    dataset={"path": "x", "fingerprint": "f"})
        with unittest.mock.patch.object(tiny_gpt, "tokenizer_from_checkpoint", return_value=CharTok()):
            model, tok, cfg, obj = tiny_gpt.load_for_inference(path)
            old_model = tiny_gpt.model_from_checkpoint(tiny_gpt.safe_load(path))[0]
        self.assertNotIn("model", obj)
        self.assertEqual((obj["step"], obj["metrics"]["best_val"], obj["dataset"]["fingerprint"]), (7, 2.5, "f"))
        self.assertTrue(all(p.dtype == torch.float32 and not p.is_meta for p in model.parameters()))
        gen = torch.Generator()
        a = tiny_gpt.generate(model, tok, "gene drift", 20, torch.device("cpu"), generator=gen.manual_seed(1))
        b = tiny_gpt.generate(old_model, tok, "gene drift", 20, torch.device("cpu"), generator=gen.manual_seed(1))
        self.assertEqual(a, b)

    def test_resolve_dataset_by_path_then_by_name_with_the_same_fingerprint(self):
        home = os.path.join(self.tmp, "repo")
        stored = os.path.join(home, "datasets", "bio_x")
        os.makedirs(stored, exist_ok=True)
        manifest = {"format": data_prep.FORMAT, "format_version": 1, "fingerprint": "abc"}
        with open(os.path.join(stored, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with unittest.mock.patch.object(tiny_gpt, "HERE", home):
            obj = {"dataset": {"path": stored, "fingerprint": "abc"}}
            self.assertEqual(tiny_gpt.resolve_dataset(obj), stored)
            self.assertEqual(tiny_gpt.resolve_dataset(obj, "other"), "other", "--dataset wins as given")
            moved = {"dataset": {"path": r"C:\Users\someone\proj\datasets\bio_x", "fingerprint": "abc"}}
            self.assertEqual(os.path.normpath(tiny_gpt.resolve_dataset(moved)), os.path.normpath(stored))
            self.assertEqual(os.path.normpath(tiny_gpt.resolve_dataset({"dataset": {"path": "datasets/bio_x",
                                                                                     "fingerprint": "abc"}})),
                             os.path.normpath(stored), "a relative path from the project folder")
            with self.assertRaises(SystemExit) as ctx:
                tiny_gpt.resolve_dataset({"dataset": {"path": stored, "fingerprint": "zzz"}})
            self.assertIn("different fingerprint", str(ctx.exception.code))
            with self.assertRaises(SystemExit) as ctx:
                tiny_gpt.resolve_dataset({"dataset": {"path": "/Users/x/datasets/elsewhere", "fingerprint": "abc"}})
            self.assertIn("--dataset", str(ctx.exception.code))
            with self.assertRaises(SystemExit):
                tiny_gpt.resolve_dataset({})
            # an older checkpoint without a fingerprint: its own path is used, a same-named folder is not
            self.assertEqual(tiny_gpt.resolve_dataset({"dataset": {"path": stored}}), stored)
            with self.assertRaises(SystemExit) as ctx:
                tiny_gpt.resolve_dataset({"dataset": {"path": r"C:\Users\someone\proj\datasets\bio_x"}})
            self.assertIn("no dataset fingerprint", str(ctx.exception.code))

    # ---------------- compare_models (other-scripts-2) ----------------
    def test_compare_models_autocast_only_off_the_cpu(self):
        """The CPU scores in float32 (choose_precision gives no autocast type there and document_nll leaves
        autocast disabled); an autocast type, when given, is used."""
        import compare_models
        cpu = torch.device("cpu")
        amp, _, _ = tiny_gpt.choose_precision(cpu, "auto", training=False)
        self.assertIsNone(amp)
        class Tok(CharTok):
            def is_special(self, t):
                return t == self.bos_id
        model, tok = self.model(seed=2), Tok()
        seen = []
        real = torch.autocast

        def spy(*a, **k):
            seen.append((k.get("dtype"), k.get("enabled")))
            return real(*a, **k)
        with unittest.mock.patch.object(torch, "autocast", spy):
            nll, nbytes = compare_models.document_nll(model, tok, "allele frequency drift " * 4, cpu, amp)
            compare_models.document_nll(model, tok, "allele frequency drift " * 4, cpu, torch.bfloat16)
        self.assertTrue(seen and seen[0][1] is False, seen)
        self.assertIn((torch.bfloat16, True), seen)
        self.assertGreater(nll, 0)
        self.assertGreater(nbytes, 0)

    # ---------------- tools/edit_weights.py (other-scripts-1, -8, -19) ----------------
    def test_edit_weights_edits_undo_and_metrics(self):
        sys.path.insert(0, os.path.join(HERE, "tools"))
        import edit_weights as ew
        model = self.model(seed=6)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for edit in ({"kind": "neurons", "layer": 1, "neurons": [0, 3, 5]},
                     {"kind": "direction", "layer": 0, "alpha": 0.5,
                      "direction": torch.nn.functional.normalize(torch.randn(32), dim=0)}):
            undo = ew.apply_edit(model, edit)
            changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
            if edit["kind"] == "neurons":
                self.assertEqual(changed, ["blocks.1.ffn.down.weight"])
                self.assertTrue((model.blocks[1].ffn.down.weight[:, [0, 3, 5]] == 0).all())
            else:
                self.assertEqual(sorted(changed), ["blocks.0.attn.o_proj.weight", "blocks.0.ffn.down.weight"])
            undo()
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()), edit["kind"])
        last = {"loss": 2.0, "step": 7, "probes": {"accuracy": 0.5}}
        m = ew.edited_metrics({"history": [1], "best_val": 1.9, "last_eval": last})
        self.assertEqual((m["history"], m["best_val"], m["last_eval"], m["base_last_eval"]), ([1], 1.9, None, last))
        self.assertEqual(ew.edited_metrics(m)["base_last_eval"], last, "an edit of an edit keeps the original's")
        # band statistics: float64 sums on the CPU from float32 per-batch sums, equal to a direct float64 sum
        stats = ew.BandStats(2, 3, 4)
        gen = torch.Generator().manual_seed(0)
        feats, labels, bands = torch.randn(50, 4, generator=gen), torch.randint(0, 2, (50,), generator=gen), \
            torch.randint(0, 3, (50,), generator=gen)
        stats.add(1, feats[:20], labels[:20], bands[:20])
        stats.add(1, feats[20:], labels[20:], bands[20:])
        self.assertEqual(stats.s[0].dtype, torch.float64)
        self.assertEqual(stats.s[0].device.type, "cpu")
        for v in (0, 1):
            sel = labels == v
            want = torch.zeros(3, 4, dtype=torch.float64).index_add_(0, bands[sel], feats[sel].double())
            torch.testing.assert_close(stats.s[v][1], want, rtol=1e-6, atol=1e-6)
            self.assertEqual(stats.n[v, 1].tolist(), torch.bincount(bands[sel], minlength=3).double().tolist())
        self.assertTrue((stats.s[0][0] == 0).all(), "layer 0 untouched")

    def test_viewer_does_not_reuse_an_edited_files_stored_evaluation(self):
        import view_pt
        sha = data_prep.sha256_file(tiny_gpt.DEFAULT_PROBES)
        stored = {"file_sha256": sha, "accuracy": 0.5}
        args = types_ns(probes=tiny_gpt.DEFAULT_PROBES)
        ck = types_ns(metrics={"last_eval": {"step": 7, "probes": stored}}, obj={"step": 7}, finetune=None, edit=None)
        self.assertIs(view_pt.probe_report(ck, None, None, args), stored)
        ck.edit = {"chosen": {"kind": "neurons"}}
        self.assertIsNone(view_pt.probe_report(ck, None, None, args), "edited weights are scored, not looked up")
        # the text summary still shows the original's last evaluation, labelled as such
        sys.path.insert(0, os.path.join(HERE, "tools"))
        import edit_weights as ew
        path = self.save_checkpoint(self.model(varied=False), "E_edited.pt", edit={"chosen": {"kind": "neurons"}},
                                    metrics=ew.edited_metrics({"last_eval": {"loss": 2.5, "step": 7}}))
        with unittest.mock.patch.object(tiny_gpt, "tokenizer_from_checkpoint", return_value=None),                 redirect_stdout(io.StringIO()) as buf:
            view_pt.print_summary(view_pt.Checkpoint(path))
        self.assertIn("latest validation (original model, before the edit): loss 2.5", buf.getvalue())

    # ---------------- finetune (other-scripts-6, -7) ----------------
    def test_finetune_learning_rate_and_batched_tokenisation(self):
        import finetune
        self.assertEqual([finetune.default_lr(n, "sft") for n in (20e6, 216e6, 500e6, 1e9, 3e9)],
                         [1e-4, 1e-4, 5e-5, 5e-5, 2e-5])
        self.assertAlmostEqual(finetune.default_lr(216e6, "dpo"), 1e-5)

        class BatchTok(CharTok):
            eos_id, calls = 45, 0

            def encode_batch(self, texts, threads=1):
                BatchTok.calls += 1
                return [self.encode(t) for t in texts]
        for tok in (CharTok(), BatchTok()):
            for prompt, answer, ctx in (("What is drift?", "Random change.", 64), ("Q" * 50, "A" * 30, 32)):
                question = tok.encode(finetune.TEMPLATE.format(prompt=prompt.strip()))
                self.assertEqual(finetune.pair_from_ids(tok, question, tok.encode(answer), ctx),
                                 finetune.encode_pair(tok, prompt, answer, ctx))
            self.assertEqual(finetune.encode_texts(tok, ["ab", "c"]), [tok.encode("ab"), tok.encode("c")])
        self.assertEqual(BatchTok.calls, 1)

    # ---------------- benchmark (other-scripts-5) ----------------
    def batched_matches_single(self, model):
        import benchmark
        import random
        cpu, tok = torch.device("cpu"), CharTok()
        rng = random.Random(0)
        # 20 prompts of 1-40 characters: some groups mix lengths (left padding), the longest need the one-at-a-time
        # path (prompt + 12 new tokens > ctx 48)
        prompts = ["".join(rng.choice("abcdefghij klmnop") for _ in range(rng.randint(1, 40))) for _ in range(20)]
        free = [benchmark.greedy_generate(model, tok, p, 12, cpu) for p in prompts]
        self.assertGreater(len({c for f in free for c in f}), 5, "the model must not just repeat one token")

        def stop_for(text):  # a character that first appears after a few tokens: the row stops early
            return next(((c,) for j, c in enumerate(text) if j >= 2 and c not in text[:j]), ())
        stops = [stop_for(f) if i % 2 == 0 else () for i, f in enumerate(free)]
        want = [benchmark.greedy_generate(model, tok, p, 12, cpu, stops=s) for p, s in zip(prompts, stops)]
        got = benchmark.batched_greedy(model, tok, prompts, 12, cpu, item_stops=stops, batch=6)
        self.assertEqual(got, want)
        self.assertLess(min(len(w) for w in want), 12, "some rows finish early and leave the batch")
        self.assertEqual(benchmark.batched_greedy(model, tok, prompts[:5], 12, cpu, stops=("Q",), batch=3),
                         [benchmark.greedy_generate(model, tok, p, 12, cpu, stops=("Q",)) for p in prompts[:5]])

    def test_batched_greedy_equals_one_at_a_time(self):
        """benchmark.batched_greedy (left-padded rows, key-padding mask in the KV cache) gives greedy_generate's
        texts token for token on 20 items, with per-row stop strings and rows leaving the batch."""
        self.batched_matches_single(self.model(seed=1))

    def test_batched_greedy_equals_one_at_a_time_with_experts(self):
        """The same with a 2-layer mixture-of-experts model (4 experts, 2 per token): routing is per token, so the
        padding rows change nothing for the others."""
        self.batched_matches_single(self.model(seed=1, n_experts=4, experts_per_token=2, moe_every=1))

    def test_key_padding_mask_in_the_kv_cache(self):
        """A left-padded row decoded with a key_valid mask gives the logits of the same row alone."""
        model = self.model(seed=8, varied=False)
        row = torch.randint(1, 50, (1, 9))
        with torch.no_grad():
            alone = model.lm_head(model(row, return_hidden=True, cache=model.new_cache())[:, -1])
            padded = torch.cat([torch.zeros(1, 4, dtype=torch.long), row], dim=1)
            other = torch.randint(1, 50, (1, 13))
            cache = model.new_cache()
            valid = torch.tensor([[False] * 4 + [True] * 9 + [True], [True] * 14])
            for layer in cache:
                layer["key_valid"] = valid
            h = model(torch.cat([padded, other]), return_hidden=True, cache=cache)
            self.assertTrue(torch.isfinite(h).all(), "padding queries attend to themselves: no NaN")
            torch.testing.assert_close(model.lm_head(h[:1, -1]), alone, rtol=1e-4, atol=1e-4)
            nxt = torch.tensor([[7], [7]])
            step = model.lm_head(model(nxt, return_hidden=True, cache=cache)[:, -1])
            ref = model.lm_head(model(torch.cat([row, nxt[:1]], dim=1), return_hidden=True)[:, -1])
            torch.testing.assert_close(step[:1], ref, rtol=1e-4, atol=1e-4)

    # ---------------- view_pt (other-scripts-16) ----------------
    def test_dashboard_window_budget_scales_with_the_model(self):
        import view_pt

        def cfg(d, layers):
            return tiny_gpt.ModelConfig(vocab_size=16384, ctx=1024, d_model=d, n_layers=layers, n_heads=d // 64,
                                        n_kv_heads=max(1, d // 256))
        self.assertEqual(view_pt.window_budget(cfg(256, 4)), 16)
        self.assertEqual(view_pt.window_budget(cfg(1024, 16)), 16, "the 216M model keeps 16 windows")
        self.assertEqual(view_pt.window_budget(cfg(2048, 24)), 4, "1B+: 4 windows")
        big = types_ns(cfg=cfg(4096, 2), weights={}, embedding=None)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        with unittest.mock.patch.object(torch.linalg, "svdvals", side_effect=AssertionError("no SVD")):
            view_pt.panel_spectra(ax, big)
        plt.close(fig)

    # ---------------- estimate_dataset (other-scripts-17) ----------------
    def test_estimate_dataset_shares_the_pipeline_and_device_speed(self):
        import estimate_dataset
        import run_pipeline
        self.assertEqual(estimate_dataset.default_prep_args(), run_pipeline.build_command()[4:])
        out = os.path.join(self.tmp, "speed")
        os.makedirs(os.path.join(out, "results"), exist_ok=True)
        with open(os.path.join(out, "results", "experiments.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv_writer(f)
            w.writerow(["time", "event", "name", "params", "d_model", "layers", "heads", "kv_heads", "ctx", "vocab",
                        "tokens_seen", "train_hours", "device"])
            w.writerow(["2026-11-01T00:00", "training run", "gpu_run", "20000000", "384", "6", "6", "2", "1024",
                        "16384", "1000000000", "1.0", "cuda"])
        flops, how = estimate_dataset.measured_speed(out, torch.device("cpu"))
        self.assertIsNone(flops)
        self.assertIn("--tflops", how)
        flops, how = estimate_dataset.measured_speed(out, torch.device("cuda"))
        self.assertGreater(flops, 0)
        self.assertIn("gpu_run", how)

    # ---------------- posttrain (other-scripts-19) ----------------
    def test_posttrain_evaluate_on_ten_questions(self):
        import posttrain
        data = os.path.join(self.tmp, "posttrain")
        os.makedirs(data, exist_ok=True)
        with open(os.path.join(data, "eval_mcq.jsonl"), "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({"category": f"cat{i % 2}:x", "prompt": f"Which gene number {i}?",
                                    "answer": "alpha", "distractors": ["beta", "gamma", "delta"]}) + "\n")
        path = self.save_checkpoint(self.model(seed=9, ctx=64), "P.pt")
        args = types_ns(device="cpu", data_dir=data, probes=os.path.join(self.tmp, "no_probes.tsv"),
                        trust_checkpoint=False, max_new_tokens=8)
        with unittest.mock.patch.object(tiny_gpt, "tokenizer_from_checkpoint", return_value=CharTok()), \
                redirect_stdout(io.StringIO()) as out:
            posttrain.evaluate([path], args)
        with open(os.path.join(data, "posttrain_report.md"), encoding="utf-8") as handle:
            report = handle.read()
        self.assertIn("| P.pt |", report)
        self.assertIn("cat0", report)
        self.assertIn("multiple choice", out.getvalue())


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


def csv_writer(handle):
    import csv
    return csv.writer(handle)


class TestRxiv(unittest.TestCase):
    """download_rxiv.py (bioRxiv / medRxiv), offline: API pages, keyword list, JATS, report logic, bot checks."""

    JATS = (b'<article xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">'
            b'<front><article-meta><title-group><article-title>Allele frequency drift</article-title></title-group>'
            b'<abstract><p>We model drift. Contact x.y@uni.edu.</p></abstract></article-meta></front><body>'
            b'<sec><title>Introduction</title><p>Genetic drift changes allele frequencies '
            b'<xref ref-type="bibr" rid="r1">(1)</xref>, with variance '
            b'<inline-formula><mml:math><mml:mfrac><mml:mi>p</mml:mi><mml:mi>N</mml:mi></mml:mfrac></mml:math>'
            b'</inline-formula>. ' + b'Population genetics text. ' * 200 + b'</p>'
            b'<fig id="f1"><label>Figure 1</label><caption><p>Drift paths.</p></caption>'
            b'<graphic xlink:href="f1.tif"/></fig></sec>'
            b'<sec><title>Acknowledgements</title><p>We thank our funders.</p></sec></body>'
            b'<back><ack><p>Thanks.</p></ack><ref-list><ref id="r1"><mixed-citation>Wright S. 1931.'
            b'</mixed-citation></ref></ref-list></back></article>')

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_rxiv_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def page(records, total):
        return json.dumps({"messages": [{"status": "ok", "count": len(records), "total": str(total)}],
                           "collection": records}).encode()

    @staticmethod
    def rec(doi, version, title="t", abstract="a", date="2025-01-15"):
        return {"doi": doi, "version": str(version), "title": title, "abstract": abstract, "date": date,
                "category": "Genetics", "license": "cc_by", "authors": "A, B.", "published": "NA",
                "jatsxml": f"https://www.biorxiv.org/content/early/x/{doi[8:]}.source.xml", "server": "biorxiv"}

    def test_harvest_pages_windows_and_latest_version(self):
        import datetime
        import download_rxiv as rx
        recs, total, size = rx.parse_details_page(self.page([self.rec("10.1101/1", 1), {"title": "no doi"}], 9))
        self.assertEqual((len(recs), total, size, recs[0]["version"], recs[0]["category"]), (1, 9, 2, 1, "genetics"))
        empty = json.dumps({"messages": [{"status": "no posts found"}], "collection": []}).encode()
        self.assertEqual(rx.parse_details_page(empty), ([], 0, 0))
        with self.assertRaises(rx.Blocked):
            rx.parse_details_page(b"<!DOCTYPE html><html><title>Just a moment...</title></html>")
        start = (datetime.date.today() - datetime.timedelta(days=40)).isoformat()
        calls = []

        def fake_get(url, limit, tries=4, timeout=60):
            calls.append(url)
            first, _end, cursor = url.rstrip("/").split("/")[-3:]
            if first != start:
                return empty
            if cursor == "0":  # the window holds 3 records: two pages; v2 of DOI 1 comes BEFORE v1
                return self.page([self.rec("10.1101/1", 2, "new"), self.rec("10.1101/2", 1)], 3)
            return self.page([self.rec("10.1101/1", 1, "old")], 3)

        cache = os.path.join(self.tmp, "rx_cache")
        os.makedirs(cache)
        with unittest.mock.patch.object(rx, "get", side_effect=fake_get), redirect_stdout(io.StringIO()):
            self.assertEqual(rx.harvest("biorxiv", cache, start=start), 3)
            self.assertEqual(rx.harvest("biorxiv", cache), 3, "a finished harvest is not asked again")
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[1].endswith("/2"), "the cursor moves by the records on the page")
        with open(os.path.join(cache, "biorxiv.state.json"), encoding="utf-8") as f:
            state = json.load(f)
        self.assertTrue(state["complete"])
        got = {r["doi"]: (r["version"], r["title"]) for r in rx.read_cache(os.path.join(cache, "biorxiv.jsonl.gz"))}
        self.assertEqual(got, {"10.1101/1": (2, "new"), "10.1101/2": (1, "t")})
        with unittest.mock.patch.object(rx, "get", side_effect=fake_get), redirect_stdout(io.StringIO()):
            rx.harvest("biorxiv", cache, update=True)
        with open(os.path.join(cache, "biorxiv.state.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["harvested_until"], datetime.date.today().isoformat())
        self.assertIn((datetime.date.today() - datetime.timedelta(days=3)).isoformat(), calls[3],
                      "an update starts 3 days before the last harvest")

    def test_keyword_list(self):
        import gzip
        import download_arxiv as ax
        import download_rxiv as rx
        meta = os.path.join(self.tmp, "rx_list")
        os.makedirs(meta)
        cache = os.path.join(meta, "biorxiv.jsonl.gz")
        with gzip.open(cache, "wt", encoding="utf-8") as f:
            for doi, title, abstract in (
                    ("10.1101/a", "Allele frequency under drift", "genetic drift and selection " + "word " * 300),
                    ("10.1101/b", "Allele drift selection", "allele drift selection"),
                    ("10.1101/c", "Protein folding", "folding of a protein")):
                f.write(json.dumps({"doi": doi, "version": 1, "date": "2024-02-01", "title": title,
                                    "abstract": abstract, "category": "genetics", "license": "cc_no",
                                    "jatsxml": ""}) + "\n")
        kw = os.path.join(meta, "kw.txt")
        with open(kw, "w", encoding="utf-8") as f:
            f.write("# terms\nallele\ndrift\nselection\n")
        with redirect_stdout(io.StringIO()):
            rows = rx.build_list(cache, kw, 3, 3, os.path.join(meta, "list.tsv"))
        self.assertEqual([r["id"] for r in rows], ["10.1101/b", "10.1101/a"], "denser match first, c left out")
        back = ax.read_list(os.path.join(meta, "list.tsv"))
        self.assertEqual([(r["id"], r["licence"], r["year"]) for r in back],
                         [("10.1101/b", "cc_no", "2024"), ("10.1101/a", "cc_no", "2024")])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rx.build_list(cache, kw, 3, 4, os.path.join(meta, "list.tsv")), [],
                             "--min-distinct applies")

    def test_jats_markdown(self):
        import download_rxiv as rx
        md, body = rx.jats_markdown(self.JATS, "ignored", "ignored")
        self.assertTrue(md.startswith("# Allele frequency drift\n\n## Abstract\n\nWe model drift."), md[:200])
        self.assertIn("[email]", md)
        self.assertNotIn("uni.edu", md)
        self.assertIn("\\frac{p}{N}", md)
        self.assertIn("(1)", md)
        self.assertIn("Figure 1. Drift paths.", md)
        for gone in ("Acknowledgements", "funders", "Thanks", "Wright"):
            self.assertNotIn(gone, md)
        self.assertGreater(body, 3000)
        front = (b'<article><front><article-meta><title-group><article-title>T</article-title></title-group>'
                 b'</article-meta></front></article>')
        md, body = rx.jats_markdown(front, "Meta title", "Meta abstract")
        self.assertEqual(body, 0)
        self.assertEqual(md, "# T\n\n## Abstract\n\nMeta abstract\n\n")
        with self.assertRaises(ValueError):
            rx.jats_markdown(b"<html><body>x</body></html>", "", "")
        evil = (b'<?xml version="1.0"?><!DOCTYPE article [<!ENTITY x SYSTEM "file:///c:/windows/win.ini">]>'
                b'<article><body><p>&x;</p></body></article>')
        try:
            md, _ = rx.jats_markdown(evil, "t", "")
        except Exception:  # noqa: BLE001 - refusing the entity is fine too
            md = ""
        self.assertNotIn("[fonts]", md.lower(), "an external entity is never read")

    def test_bot_check_detection(self):
        import download_rxiv as rx
        page = (b"<!DOCTYPE html><html lang='en-US'><head><title>Just a moment...</title></head>"
                b"<body><script src='/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1'></script></body></html>")
        self.assertTrue(rx.is_challenge(page))
        self.assertTrue(rx.is_challenge(b"", {"cf-mitigated": "challenge"}))
        self.assertFalse(rx.is_challenge(self.JATS))
        self.assertFalse(rx.is_challenge(b"%PDF-1.4 ... just a moment"))
        self.assertFalse(rx.is_challenge(b"<!DOCTYPE html><html><body>Page not found</body></html>"))
        self.assertTrue(rx.looks_like_html(b"\n <!doctype HTML><html>"))
        self.assertFalse(rx.looks_like_html(b'<?xml version="1.0"?><article/>'))
        self.assertTrue(rx._own_site("https://www.medrxiv.org/content/x.source.xml", "medrxiv"))
        for bad in ("http://www.medrxiv.org/x", "https://evil.org/www.medrxiv.org", "https://medrxiv.org.evil.com/x"):
            self.assertFalse(rx._own_site(bad, "medrxiv"), bad)

    def test_fetch_routes_and_report(self):
        import urllib.error
        import download_arxiv as ax
        import download_rxiv as rx
        out = os.path.join(self.tmp, "rx_out")
        os.makedirs(out)
        row = {"id": "10.1101/2025.01.14.000001", "version": "2", "category": "genetics", "year": "2025",
               "licence": "cc_by", "title": "Allele: drift?", "abstract": "abs",
               "jatsxml": "https://www.biorxiv.org/content/early/2025/01/15/2025.01.14.000001.source.xml"}
        challenge = b"<!DOCTYPE html><html><title>Just a moment...</title></html>"
        search = json.dumps({"resultList": {"result": [
            {"id": "PPR1", "source": "PPR", "doi": row["id"].upper(), "inEPMC": "Y"}]}}).encode()

        def routes(table, seen=None):
            def fake(url, limit, tries=4, timeout=60):
                if seen is not None:
                    seen.append(url)
                for key, answer in table.items():
                    if key in url:
                        if isinstance(answer, Exception):
                            raise answer
                        return answer
                raise AssertionError(url)
            return fake

        # the server's JATS is a bot-check page: Europe PMC's copy is used
        table = {"source.xml": rx.Blocked("bot-check page"), "search": search, "PPR1/fullTextXML": self.JATS}
        with unittest.mock.patch.object(rx, "get", side_effect=routes(table)):
            r = rx.fetch_paper(row, "biorxiv", out, 3000)
        self.assertEqual((r["status"], r["method"]), ("ok", "europepmc"))
        self.assertEqual(r["file"], "10.1101_2025.01.14.000001__Allele drift.md")
        self.assertTrue(os.path.isfile(os.path.join(out, r["file"])))
        self.assertFalse(glob.glob(os.path.join(out, "*.part")))
        # everything blocked: an error (retried), flagged as blocked
        err403 = urllib.error.HTTPError("u", 403, "Forbidden", {}, None)
        table = {"source.xml": rx.Blocked("403"), "search": err403, "full.pdf": rx.Blocked("403")}
        with unittest.mock.patch.object(rx, "get", side_effect=routes(table)), \
                unittest.mock.patch.object(ax, "check_pdf_route", return_value=None):
            r = rx.fetch_paper(row, "biorxiv", out, 3000)
        self.assertTrue(r["status"].startswith("error: jats blocked"), r["status"])
        self.assertTrue(r["blocked"])
        # nothing anywhere: no_text (final); the PDF URL names the version
        seen = []
        table = {"source.xml": None, "search": json.dumps({"resultList": {"result": []}}).encode(),
                 "v2.full.pdf": None}
        with unittest.mock.patch.object(rx, "get", side_effect=routes(table, seen)):
            r = rx.fetch_paper(row, "biorxiv", out, 3000)
        self.assertEqual(r["status"], "no_text")
        self.assertEqual(seen[-1], "https://www.biorxiv.org/content/10.1101/2025.01.14.000001v2.full.pdf")
        # JATS without a body and no other route: too_short (final)
        front = (b"<article><front><article-meta><title-group><article-title>T</article-title></title-group>"
                 b"<abstract><p>Short.</p></abstract></article-meta></front></article>")
        with unittest.mock.patch.object(rx, "get", side_effect=routes({"source.xml": front})):
            r = rx.fetch_paper(row, "biorxiv", out, 3000, europepmc=False, pdf=False)
        self.assertEqual(r["status"], "too_short")
        self.assertTrue(r["method"].startswith("jats"))
        # a challenge page that comes back as 200 is caught by get() itself
        with unittest.mock.patch.object(ax, "http_get", return_value=challenge), self.assertRaises(rx.Blocked):
            rx.get("https://www.biorxiv.org/x", 100)
        # report: final statuses are skipped, errors retried, PDF time-outs only with a larger limit
        report = os.path.join(self.tmp, "rx_report.tsv")
        with open(report, "w", encoding="utf-8") as f:
            f.write("\t".join(rx.REPORT_COLUMNS) + "\n")
            for pid, status, method in (("d/1", "ok", "jats"), ("d/2", "error: jats blocked (403)", ""),
                                        ("d/3", "no_text", "jats missing"),
                                        ("d/4", "pdf_timeout", "pdf took over 180 s"),
                                        ("d/5", "too_short", "pdf"), ("d/6", "error: x", ""), ("d/6", "ok", "pdf")):
                f.write("\t".join([pid, status, "0", method, "", "", "", ""]) + "\n")
        limits = {}
        done = ax.read_report(report, limits)
        rows = [{"id": f"d/{i}"} for i in range(1, 9)]
        self.assertEqual([r["id"] for r in rx.todo_rows(rows, done, limits, {"d_7"}, 180)], ["d/2", "d/8"])
        self.assertEqual([r["id"] for r in rx.todo_rows(rows, done, limits, set(), 600)], ["d/2", "d/4", "d/7", "d/8"])

    def test_args_defaults(self):
        import download_rxiv as rx
        args = rx.parse_args(["--server", "medrxiv"])
        self.assertEqual(args.out, os.path.abspath(os.path.join(os.path.expanduser("~"), "ai_training_data",
                                                                "medrxiv")))
        self.assertEqual(os.path.basename(args.meta), "_medrxiv_meta")
        self.assertEqual(args.wait, (5.0, 15.0))
        with unittest.mock.patch.dict(os.environ, {"RXTEST": self.tmp}):
            args = rx.parse_args(["--out", "%RXTEST%\\bio" if sys.platform == "win32" else "$RXTEST/bio"])
        self.assertEqual(args.out, os.path.join(os.path.abspath(self.tmp), "bio"))


class TestBenchmarkSuite(unittest.TestCase):
    """benchmark.py: offline only (no downloads, no sandbox, no real checkpoint)."""

    @classmethod
    def setUpClass(cls):
        import benchmark
        cls.b = benchmark
        cls.tmp = tempfile.mkdtemp(prefix="tiny_gpt_bench_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_item_normalisation(self):
        b = self.b
        mmlu = b.norm_mmlu([{"question": "Q1?", "subject": "medical_genetics", "choices": ["a", "b", "c", "d"],
                             "answer": 2},
                            {"question": "Q2?", "subject": "astronomy", "choices": np.array(["w", "x", "y", "z"]),
                             "answer": "0"}])
        self.assertEqual(mmlu[0]["answer"], "c")
        self.assertEqual(mmlu[0]["tags"], ["biology", "genetics", "medicine"])
        self.assertEqual(mmlu[1]["answer"], "w")
        self.assertNotIn("tags", mmlu[1])
        pro = b.norm_mmlu_pro([{"question_id": 7, "question": "Q", "options": ["p", "N/A", "q", "r"],
                                "answer_index": 2, "category": "health"}])[0]
        self.assertEqual((pro["options"], pro["answer"], pro["tags"]), (["p", "q", "r"], "q", ["health"]))
        arc = b.norm_arc([{"id": "x1", "question": "Which?", "choices": {"text": ["one", "two"], "label": ["1", "2"]},
                           "answerKey": "2"}], "ARC-Easy")[0]
        self.assertEqual(arc["answer"], "two")
        medqa = b.norm_medqa([{"question": "Q", "options": "{'A': 'x', 'B': 'y'}", "answer_idx": "B",
                               "meta_info": "step1"}])[0]
        self.assertEqual((medqa["options"], medqa["answer"]), (["x", "y"], "y"))
        self.assertEqual(b._list("['T' 'P' 'Ala']"), ["T", "P", "Ala"], "numpy's printed list")
        lab = b.norm_labbench([{"id": "1", "question": "Q?", "ideal": "His", "distractors": "['T' 'P' 'Ala']",
                                "protocol": "Step 1", "subtask": "s"}], "ProtocolQA")[0]
        self.assertTrue(lab["question"].startswith("Step 1"))
        self.assertEqual(lab["options"], ["His", "T", "P", "Ala"])
        calc = b.norm_medcalc([{"Row Number": "3", "Question": "Score?", "Category": "risk", "Patient Note": "n",
                                "Output Type": "integer", "Ground Truth Answer": "4", "Lower Limit": "4",
                                "Upper Limit": "4", "Calculator Name": "c"}])[0]
        self.assertEqual((calc["kind"], calc["options"]), ("numeric", None))
        for item in mmlu + [pro, arc, medqa, lab, calc]:
            self.assertIsNone(b.valid_item(item), item)
            self.assertTrue({"id", "question", "options", "answer", "category", "kind"} <= set(item))
        self.assertEqual(b.valid_item(dict(arc, answer="three")), "answer not among the options")
        self.assertEqual(b.valid_item(dict(arc, options=["two", "two"])), "answer duplicated among the options")
        self.assertEqual(b.valid_item(dict(calc, options=["1"])), "options must be null for non-MCQ items")

    def test_mcq_scoring_on_a_tiny_model(self):
        b = self.b
        folder = os.path.join(self.tmp, "bench", "toy")
        os.makedirs(folder)
        items = [b.mcq_item(f"t{i}", f"Which gene number {i}?", ["alpha", "beta", "gamma"], ["alpha", "beta",
                            "gamma"][i % 3], "genes" if i % 2 else "cells", ["biology"] if i < 3 else None)
                 for i in range(6)]
        b.write_items(os.path.join(folder, "items.jsonl"), items)
        with unittest.mock.patch.object(b, "BENCH_DIR", os.path.join(self.tmp, "bench")):
            back = b.read_items("toy")
        self.assertEqual(back, items)
        torch.manual_seed(0)
        cfg = tiny_gpt.ModelConfig(vocab_size=50, ctx=16, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2)
        model = tiny_gpt.TinyGPT(cfg).eval()
        rows = b.score_mcq(model, CharTok(), back, torch.device("cpu"), None, floor=True, batch_size=4)
        self.assertEqual(len(rows), 6)
        for r in rows:
            self.assertAlmostEqual(r["chance"], 1 / 3)
            self.assertIn(r["floor_correct"], (0, 1 / 3, 2 / 3, 1))
            self.assertEqual(r["correct"], r["correct_score"] > r["best_other_score"])
        summary = {s["category"]: s for s in b.summarise(rows)}
        self.assertEqual(summary["all"]["n"], 6)
        self.assertEqual(summary["[biology]"]["n"], 3)
        self.assertEqual(summary["genes"]["n"] + summary["cells"]["n"], 6)
        self.assertLessEqual(summary["all"]["low"], summary["all"]["accuracy"])
        self.assertEqual(b.sample_items(back, 4, "toy"), b.sample_items(list(back), 4, "toy"))
        self.assertEqual(len(b.sample_items(back, 4, "toy")), 4)
        self.assertEqual(b.sample_items(back, 0, "toy"), back)

    def test_numeric_parsing(self):
        b = self.b
        self.assertEqual(b.parse_first_number("about 1,234.5 mL/min, not 7"), 1234.5)
        self.assertEqual(b.parse_first_number("score: -3"), -3.0)
        self.assertIsNone(b.parse_first_number("no number here"))
        dec = {"answer": "25.2381", "output_type": "decimal", "lower": "23.97619", "upper": "26.50001"}
        self.assertTrue(b.numeric_correct(dec, " 26.1 mL/min")[0])
        self.assertFalse(b.numeric_correct(dec, " 27")[0])
        self.assertFalse(b.numeric_correct(dec, " unknown")[0])
        integer = {"answer": "4", "output_type": "integer", "lower": "4", "upper": "4"}
        self.assertTrue(b.numeric_correct(integer, "4 points")[0])
        self.assertFalse(b.numeric_correct(integer, "5")[0])
        date = {"answer": "09/11/2014", "output_type": "date", "lower": "09/11/2014", "upper": "09/11/2014"}
        self.assertTrue(b.numeric_correct(date, " 9/11/2014.")[0])
        self.assertFalse(b.numeric_correct(date, " 09/12/2014")[0])
        tup = {"answer": "(4 weeks, 2 days)", "output_type": "integer"}
        self.assertTrue(b.numeric_correct(tup, "4 weeks and 2 days")[0])
        self.assertFalse(b.numeric_correct(tup, "4 weeks and 3 days")[0])

    def test_sandbox_selector_never_runs_code_without_a_sandbox(self):
        b = self.b
        with unittest.mock.patch.object(b, "docker_available", return_value=False), \
                unittest.mock.patch.object(b, "wsl_distros", return_value=[]):
            self.assertIsNone(b.select_sandbox(log=lambda *a: None))
            self.assertIsInstance(b.select_sandbox(unsafe=True, log=lambda *a: None), b.LocalSubprocess)
        with unittest.mock.patch.object(b, "docker_available", return_value=True):
            self.assertIsInstance(b.select_sandbox(log=lambda *a: None), b.DockerSandbox)
        with unittest.mock.patch.object(b, "docker_available", return_value=False), \
                unittest.mock.patch.object(b, "wsl_distros", return_value=["Debian", "Ubuntu"]), \
                unittest.mock.patch.object(b, "wsl_probe", side_effect=lambda d: ("bwrap", False) if d == "Ubuntu"
                                           else None):
            sb = b.select_sandbox(log=lambda *a: None)
            self.assertIsInstance(sb, b.WSLSandbox)
            self.assertEqual(sb.distro, "Ubuntu")
            script = sb.script("print(1)\n", 5)
            self.assertIn("--unshare-all", script)
            self.assertIn("timeout -k 2 5", script)
            self.assertNotIn("print(1)", script, "the program travels base64-encoded")
        # a run with a coding benchmark and no sandbox: refused, and nothing is executed anywhere
        bench = os.path.join(self.tmp, "nosandbox")
        os.makedirs(os.path.join(bench, "humaneval"))
        b.write_items(os.path.join(bench, "humaneval", "items.jsonl"),
                      [{"id": "h0", "question": "def f():\n", "options": None, "answer": "    return 1\n",
                        "category": "python", "kind": "code", "style": "humaneval", "test": "def check(c): pass",
                        "entry_point": "f"}])
        with open(os.path.join(bench, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"benchmarks": {"humaneval": {"items": 1}}}, handle)
        boom = unittest.mock.Mock(side_effect=AssertionError("nothing may be executed"))
        args = b.parse_args(["run", "--checkpoint", "none.pt", "--benchmarks", "humaneval", "--device", "cpu"])
        out = io.StringIO()
        with unittest.mock.patch.object(b, "BENCH_DIR", bench), \
                unittest.mock.patch.object(b, "MANIFEST", os.path.join(bench, "manifest.json")), \
                unittest.mock.patch.object(b, "select_sandbox", return_value=None), \
                unittest.mock.patch.object(b, "_run", boom), \
                unittest.mock.patch.object(subprocess, "run", boom), \
                unittest.mock.patch.object(subprocess, "Popen", boom), \
                unittest.mock.patch.object(b.LocalSubprocess, "run", boom), \
                unittest.mock.patch.object(b, "greedy_generate", boom), \
                redirect_stdout(out):
            self.assertEqual(b.cmd_run(args), 1)
        self.assertIn("REFUSING", out.getvalue())
        boom.assert_not_called()

    def test_results_csv_and_plot(self):
        b = self.b
        path = os.path.join(self.tmp, "results", "benchmarks.csv")
        home_ck = os.path.join(os.path.expanduser("~"), "models", "tinyGPT_best.pt")
        rows = []
        for ck, step, shift in ((home_ck, 1000, 0.0), (home_ck.replace("best", "late"), 2000, 0.05)):
            for bench, cat, n, acc, chance, floor in (
                    ("mmlu", "all", 50, 0.28, 0.25, 0.26), ("mmlu", "[biology]", 9, 0.33, 0.25, 0.3),
                    ("mmlu", "anatomy", 5, 0.4, 0.25, 0.2), ("mmlu", "virology", 4, 0.25, 0.25, 0.25),
                    ("medqa", "all", 50, 0.3, 0.25, 0.27), ("humaneval", "all", 20, 0.0, 0.0, None),
                    ("medcalc", "all", 30, 0.03, 0.0, None)):
                low, high = tiny_gpt.wilson_interval(round((acc + shift) * n), n)
                rows.append({"time": "t", "checkpoint": ck, "step": step, "params": 1, "benchmark": bench,
                             "category": cat, "n": n, "accuracy": acc + shift, "low": low, "high": high,
                             "chance": chance, "floor": "" if floor is None else floor, "notes": f"from {ck}"})
        b.append_results(path, rows[:7])
        b.append_results(path, rows[7:])
        with open(path, encoding="utf-8-sig") as handle:
            text = handle.read()
        self.assertEqual(text.count("time,checkpoint,step"), 1, "one header")
        self.assertEqual(text.splitlines()[0].split(","), b.RESULT_COLUMNS)
        self.assertNotIn(os.path.expanduser("~"), text)
        self.assertIn("~", text)
        self.assertEqual(len(b.read_results(path)), 14)
        self.assertEqual(len(b.latest_rows(b.read_results(path), "tinyGPT_best.pt")), 7)
        # the same file scored again later, at step 3000 and on medqa only: that step's rows, never a mix of steps
        read = b.read_results(path)
        two_steps = read + [dict(read[4], step="3000", accuracy="0.5")]
        latest = b.latest_rows(two_steps, "tinyGPT_best.pt")
        self.assertEqual([(r["benchmark"], r["category"], r["step"]) for r in latest], [("medqa", "all", "3000")])
        self.assertEqual({r["step"] for r in b.latest_rows(two_steps[:7] + two_steps[-1:])}, {"3000"})
        said = []
        b.latest_rows(two_steps, "tinyGPT_best.pt", log=said.append)
        self.assertEqual(len(said), 1)
        self.assertIn("step 3000", said[0])
        self.assertIn("humaneval, medcalc, mmlu", said[0])
        b.latest_rows(read, "tinyGPT_best.pt", log=said.append)
        self.assertEqual(len(said), 1, "one step: nothing to say")
        item_csv = os.path.join(self.tmp, "items.csv")
        b.write_item_csv(item_csv, [{"id": "a", "correct": True, "tags": ["x", "y"]}, {"id": "b", "extra": 1}])
        with open(item_csv, encoding="utf-8-sig") as handle:
            self.assertEqual(handle.readline().strip(), "id,correct,tags,extra")
        out_dir = os.path.join(self.tmp, "plots")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(b.main(["plot", "--results", path, "--out-dir", out_dir, "--dpi", "40"]), 0)
        for name in ("tinyGPT_late_benchmarks.png", "tinyGPT_late_categories.png", "progress.png"):
            p = os.path.join(out_dir, name)
            self.assertTrue(os.path.isfile(p) and os.path.getsize(p) > 1000, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
