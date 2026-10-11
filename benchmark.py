r"""Benchmark suite for tinyGPT: public knowledge, medicine, biology, statistics,
coding and 3D-modelling benchmarks, downloaded once and scored on any checkpoint.

  download   fetch every freely downloadable benchmark into benchmarks\<name>\
             (raw files + a normalised items.jsonl) and write benchmarks\manifest.json
             (source, revision, licence, item count, sha256, date). Hugging Face
             files are fetched directly (parquet/json/csv); no dataset scripts, no
             trust_remote_code, nothing gated or registration-only.
  run        score a checkpoint:
               mcq      option log-likelihood per character (as the fact benchmark),
                        with chance, the shuffled-question floor and a 95% interval
               numeric  greedy answer, first number parsed, the benchmark's tolerance
               code     greedy completion (pass@1), tests run ONLY in a sandbox:
                        Docker (no network, 1 GB, 1 CPU) if its daemon runs, else WSL
                        with bubblewrap (no network, no Windows drives), else refused
                        unless --unsafe-local-execution
               open / vision-graded sets are listed but not scored
             Results: results\benchmarks.csv (appended), per-item CSVs in
             benchmarks\runs\<checkpoint>_<time>\, one row in experiments.csv.
  plot       PNGs in benchmarks\plots\: accuracy per benchmark (95% interval,
             chance, floor), MMLU / MMLU-Pro categories, progress over checkpoints.
  sandbox-setup
             install numpy (and with --datasci pandas, scipy, scikit-learn,
             matplotlib) into a folder inside WSL that the sandbox mounts read-only;
             HumanEval+ and MBPP+ test harnesses need numpy, DS-1000 needs the rest.

Examples
--------
    python benchmark.py download
    python benchmark.py download --only mmlu,humaneval
    python benchmark.py run --checkpoint tinyGPT_best.pt --limit 200
    python benchmark.py run --checkpoint tinyGPT_best.pt --benchmarks mmlu,medqa,humaneval --device cpu
    python benchmark.py plot
    python benchmark.py sandbox-setup

Benchmark data carries canary strings (LAB-Bench) and must never be used for
training: benchmarks\ is git-ignored and outside the training-data folders.
"""

import argparse
import base64
import csv
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BENCH_DIR = os.path.join(HERE, "benchmarks")
MANIFEST = os.path.join(BENCH_DIR, "manifest.json")
RUNS_DIR = os.path.join(BENCH_DIR, "runs")
PLOTS_DIR = os.path.join(BENCH_DIR, "plots")
RESULTS_CSV = os.path.join(HERE, "results", "benchmarks.csv")
RESULT_COLUMNS = ["time", "checkpoint", "step", "params", "benchmark", "category", "n", "accuracy", "low", "high",
                  "chance", "floor", "notes"]
MCQ_TEMPLATE = "{prompt}\nAnswer:"
LETTERS = "ABCDEFGHIJKLMNOP"
GROUPS = ["knowledge", "medicine", "biology", "statistics", "code", "3d"]
SANDBOX_SITE = "$HOME/.cache/tinygpt-sandbox/site"  # inside WSL; mounted read-only as /site

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _flat(value):
    return " ".join(str(value if value is not None else "").split())


def _list(value):
    """A list from a parquet/numpy array, a list, or a printed list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            if re.search(r"""['"]\s+['"]""", text):  # numpy's printed form: literal_eval would join the strings
                return [a or b for a, b in re.findall(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", text)]
            try:
                import ast
                return list(ast.literal_eval(text))
            except (ValueError, SyntaxError):  # numpy's printed form: ['a' 'b'] without commas
                return [a or b for a, b in re.findall(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", text)]
        return [value]
    return list(value)


def _dict(value):
    if value is None:
        return {}
    if isinstance(value, str):
        import ast
        try:
            return dict(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            return json.loads(value)
    return dict(value)


def stable_key(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def home_short(text):
    """Home folder shown as ~ (for anything written under results\\)."""
    text = str(text)
    home = os.path.expanduser("~")
    for variant in {home, home.replace("\\", "/")}:
        text = re.sub(re.escape(variant), "~", text, flags=re.IGNORECASE)
    return text


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def read_table(path):
    """Rows (dicts, NaN -> None) of a parquet, csv, jsonl or json file."""
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            obj = json.load(handle)
        return obj if isinstance(obj, list) else list(obj.values())
    import pandas as pd
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.astype(object).where(df.notna(), None)
    return df.to_dict("records")


def mcq_item(iid, question, options, answer, category, tags=None, **extra):
    options = [_flat(o) for o in options]
    answer = _flat(answer)
    item = {"id": str(iid), "question": str(question).strip(), "options": options, "answer": answer,
            "category": category, "kind": "mcq"}
    if tags:
        item["tags"] = sorted(set(tags))
    item.update(extra)
    return item


def valid_item(item):
    """Why an item cannot be used (None if it can)."""
    for key in ("id", "question", "kind", "category"):
        if not item.get(key):
            return f"missing {key}"
    if item["kind"] not in ("mcq", "code", "numeric", "open"):
        return f"unknown kind {item['kind']}"
    if item["kind"] == "mcq":
        opts = item.get("options") or []
        if len(opts) < 2:
            return "fewer than two options"
        if any(not o for o in opts):
            return "empty option"
        if item.get("answer") not in opts:
            return "answer not among the options"
        if opts.count(item["answer"]) > 1:
            return "answer duplicated among the options"
    elif item.get("options") is not None:
        return "options must be null for non-MCQ items"
    return None


# ---------------------------------------------------------------------------
# normalisers: raw rows -> items (pure functions, tested offline)
# ---------------------------------------------------------------------------
MMLU_TAGS = {
    "biology": {"anatomy", "college_biology", "high_school_biology", "medical_genetics", "virology"},
    "medicine": {"anatomy", "clinical_knowledge", "college_medicine", "professional_medicine", "medical_genetics",
                 "nutrition", "virology", "human_aging"},
    "genetics": {"medical_genetics"},
    "statistics": {"high_school_statistics", "econometrics"},
}


def norm_mmlu(rows):
    out = []
    for i, r in enumerate(rows):
        choices = [_flat(c) for c in _list(r["choices"])]
        subject = r["subject"]
        tags = [t for t, subjects in MMLU_TAGS.items() if subject in subjects]
        out.append(mcq_item(f"mmlu-{i}", r["question"], choices, choices[int(r["answer"])], subject, tags))
    return out


def norm_mmlu_pro(rows):
    out = []
    for r in rows:
        options = [_flat(o) for o in _list(r["options"])]
        k = int(r["answer_index"])
        answer = options[k]
        options = [o for j, o in enumerate(options) if (o and o != "N/A") or j == k]
        tag = {"biology": "biology", "health": "health"}.get(r["category"])
        out.append(mcq_item(f"mmlupro-{r['question_id']}", r["question"], options, answer, r["category"],
                            [tag] if tag else None, src=r.get("src")))
    return out


def norm_medqa(rows):
    out = []
    for i, r in enumerate(rows):
        opts = _dict(r["options"])
        keys = sorted(opts)
        out.append(mcq_item(f"medqa-{i}", r["question"], [opts[k] for k in keys], opts[r["answer_idx"]],
                            r.get("meta_info") or "usmle"))
    return out


def norm_medmcqa(rows):
    import posttrain
    base, _ = posttrain._medmcqa_index([r for r in rows if r.get("exp")][:2000])
    out = []
    for r in rows:
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        out.append(mcq_item(f"medmcqa-{r['id']}", r["question"], opts, opts[int(r["cop"]) - base],
                            r.get("subject_name") or "unknown", choice_type=r.get("choice_type")))
    return out


def norm_pubmedqa(rows):
    out = []
    for r in rows:
        ctx = _dict(r["context"])
        abstract = " ".join(_flat(c) for c in _list(ctx.get("contexts")))
        d = (r.get("final_decision") or "").strip().lower()
        out.append(mcq_item(f"pubmedqa-{r['pubid']}", r["question"], ["yes", "no", "maybe"], d, "expert-labelled",
                            context=abstract))
    return out


def norm_medxpertqa(rows):
    out = []
    for r in rows:
        opts = _dict(r["options"])
        keys = sorted(opts)
        question = re.split(r"\n\s*Answer Choices:", r["question"])[0]
        out.append(mcq_item(r["id"], question, [opts[k] for k in keys], opts[r["label"]],
                            r.get("medical_task") or "unknown", body_system=r.get("body_system"),
                            question_type=r.get("question_type")))
    return out


def norm_sciq(rows):
    out = []
    for i, r in enumerate(rows):
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        opts.sort(key=lambda o: stable_key("sciq", i, o))  # a fixed order, not always the answer first
        out.append(mcq_item(f"sciq-{i}", r["question"], opts, r["correct_answer"], "science"))
    return out


def norm_arc(rows, category):
    out = []
    for r in rows:
        ch = _dict(r["choices"])
        texts, labels = _list(ch["text"]), [str(x) for x in _list(ch["label"])]
        out.append(mcq_item(r["id"], r["question"], texts, texts[labels.index(str(r["answerKey"]))], category))
    return out


def norm_labbench(rows, subset):
    out = []
    for r in rows:
        question = r["question"].strip()
        if subset == "ProtocolQA" and r.get("protocol"):
            question = r["protocol"].strip() + "\n\n" + question
        out.append(mcq_item(f"{subset}-{r['id']}", question, [r["ideal"]] + _list(r["distractors"]), r["ideal"],
                            subset, subtask=r.get("subtask")))
    return out


def norm_geneturing(rows):
    return [{"id": f"geneturing-{r['idx']}", "question": r["question"], "options": None,
             "answer": r["goldstandard"], "category": r["module"], "kind": "open"} for r in rows]


def norm_medcalc(rows):
    out = []
    for r in rows:
        out.append({"id": f"medcalc-{r['Row Number']}", "question": r["Question"], "options": None,
                    "answer": str(r["Ground Truth Answer"]), "category": r["Category"], "kind": "numeric",
                    "context": r["Patient Note"], "output_type": r["Output Type"],
                    "lower": r["Lower Limit"], "upper": r["Upper Limit"], "calculator": r["Calculator Name"]})
    return out


def norm_healthbench(rows):
    out = []
    for r in rows:
        convo = "\n\n".join(f"{m['role']}: {m['content']}" for m in r["prompt"])
        themes = [t.split(":", 1)[1] for t in r.get("example_tags", []) if t.startswith("theme:")]
        out.append({"id": r["prompt_id"], "question": convo, "options": None, "answer": None,
                    "category": themes[0] if themes else "unknown", "kind": "open", "rubrics": r["rubrics"]})
    return out


def norm_humaneval(rows, plus=False):
    return [{"id": r["task_id"], "question": r["prompt"], "options": None, "answer": r["canonical_solution"],
             "category": "python", "kind": "code", "test": r["test"], "entry_point": r["entry_point"],
             "style": "humaneval", "requires": ["numpy"] if plus else []} for r in rows]


def norm_mbpp(rows, plus=False):
    out = []
    for r in rows:
        tests = [str(t) for t in _list(r["test_list"])]
        item = {"id": f"mbpp{'plus' if plus else ''}-{r['task_id']}", "question": r["prompt"], "options": None,
                "answer": r["code"], "category": "python", "kind": "code", "style": "mbpp",
                "test_list": tests, "test_imports": [str(t) for t in _list(r.get("test_imports"))],
                "requires": ["numpy"] if plus else []}
        if plus:
            item["test"] = r["test"]
        out.append(item)
    return out


DS1000_MODULES = {"Pandas": "pandas", "Numpy": "numpy", "Matplotlib": "matplotlib", "Sklearn": "sklearn",
                  "Scipy": "scipy", "Pytorch": "torch", "Tensorflow": "tensorflow"}


def norm_ds1000(rows):
    out = []
    for r in rows:
        meta = _dict(r["metadata"])
        lib = meta.get("library", "unknown")
        mods = {"numpy", DS1000_MODULES.get(lib, lib.lower())}
        out.append({"id": f"ds1000-{meta.get('problem_id')}", "question": r["prompt"], "options": None,
                    "answer": r["reference_code"], "category": lib, "kind": "code", "style": "ds1000",
                    "code_context": r["code_context"], "requires": sorted(mods)})
    return out


def norm_biocoder(rows):
    return [{"id": f"biocoder-{r['test_case_id'][:16]}", "question": r["promptSummaryOnly"], "options": None,
             "answer": r["goldenCode"], "category": r.get("language", "unknown"), "kind": "code",
             "style": "biocoder", "repository": r.get("repository"), "signature": r.get("signature")} for r in rows]


def norm_bigcodebench(rows):
    return [{"id": r["task_id"], "question": r["complete_prompt"], "options": None, "answer": r["canonical_solution"],
             "category": "python", "kind": "code", "style": "bigcodebench", "test": r["test"],
             "entry_point": r["entry_point"], "requires": sorted(set(_list(r["libs"])))} for r in rows]


def norm_cadbench(rows):
    return [{"id": f"cadbench-{i}-{r['id'][:8]}", "question": r["instruction"], "options": None, "answer": None,
             "category": r.get("type") or "unknown", "kind": "code", "style": "blender", "name": r.get("name"),
             "criteria": r.get("criteria")} for i, r in enumerate(rows)]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def _spec(name, title, group, repo, files, licence, normalise, kind="mcq", grading="loglik", scored=True,
          notes="", **extra):
    return {"name": name, "title": title, "group": group, "repo": repo, "files": files, "licence": licence,
            "normalise": normalise, "kind": kind, "grading": grading, "scored": scored, "notes": notes,
            "url": f"https://huggingface.co/datasets/{repo}", **extra}


def _one(fn):
    return lambda raw: fn(read_table(raw[0]))


LABBENCH_SUBSETS = ["DbQA", "SeqQA", "ProtocolQA", "CloningScenarios"]

BENCHMARKS = [
    _spec("mmlu", "MMLU (57 subjects, test)", "knowledge", "cais/mmlu", ["all/test-00000-of-00001.parquet"], "MIT",
          _one(norm_mmlu), notes="tags: biology, medicine, genetics, statistics subjects"),
    _spec("mmlu_pro", "MMLU-Pro (test)", "knowledge", "TIGER-Lab/MMLU-Pro", ["data/test-00000-of-00001.parquet"],
          "MIT", _one(norm_mmlu_pro), notes="up to 10 options; tags: biology, health"),
    _spec("medqa", "MedQA USMLE 4-option (test)", "medicine", "GBaker/MedQA-USMLE-4-options",
          ["phrases_no_exclude_test.jsonl"], "CC BY 4.0", _one(norm_medqa),
          notes="licence as declared on the Hugging Face card of this copy"),
    _spec("medmcqa", "MedMCQA (validation)", "medicine", "openlifescienceai/medmcqa",
          ["data/validation-00000-of-00001.parquet"], "Apache-2.0", _one(norm_medmcqa),
          notes="test answers are not public; the validation split is the usual evaluation set"),
    _spec("pubmedqa", "PubMedQA expert-labelled (pqa_labeled)", "medicine", "qiaojin/PubMedQA",
          ["pqa_labeled/train-00000-of-00001.parquet"], "MIT", _one(norm_pubmedqa),
          notes="reasoning-required setting: abstract (without its conclusion) + question; yes/no/maybe"),
    _spec("medxpertqa", "MedXpertQA Text (test)", "medicine", "TsinghuaC3I/MedXpertQA", ["Text/test.jsonl"], "MIT",
          _one(norm_medxpertqa), notes="10 options; the multimodal (MM) part needs images and is not downloaded"),
    _spec("sciq", "SciQ (test)", "knowledge", "allenai/sciq", ["data/test-00000-of-00001.parquet"],
          "CC BY-NC 3.0 (non-commercial)", _one(norm_sciq)),
    _spec("arc_easy", "ARC-Easy (test)", "knowledge", "allenai/ai2_arc", ["ARC-Easy/test-00000-of-00001.parquet"],
          "CC BY-SA 4.0", lambda raw: norm_arc(read_table(raw[0]), "ARC-Easy")),
    _spec("arc_challenge", "ARC-Challenge (test)", "knowledge", "allenai/ai2_arc",
          ["ARC-Challenge/test-00000-of-00001.parquet"], "CC BY-SA 4.0",
          lambda raw: norm_arc(read_table(raw[0]), "ARC-Challenge")),
    _spec("lab_bench", "LAB-Bench (DbQA, SeqQA, ProtocolQA, CloningScenarios)", "biology", "futurehouse/lab-bench",
          [f"{s}/train-00000-of-00001.parquet" for s in LABBENCH_SUBSETS] + ["LICENSE"], "CC BY-SA 4.0",
          lambda raw: [it for s, p in zip(LABBENCH_SUBSETS, raw) for it in norm_labbench(read_table(p), s)],
          notes="self-contained subsets only; carries a canary string: never train on it"),
    _spec("geneturing", "GeneTuring", "biology", "vladimire/geneturing", ["all/geneturing.csv"],
          "Apache-2.0 (as declared by this Hugging Face copy)", _one(norm_geneturing), kind="open", grading="none",
          scored=False, notes="third-party copy of GeneTuring (Hou & Ji 2023); short free-text answers, not scored"),
    _spec("medcalc", "MedCalc-Bench v1.2 (test)", "medicine", "ncbi/MedCalc-Bench-v1.2",
          ["test_data_11_18_final.csv"], "CC BY-SA 4.0", _one(norm_medcalc), kind="numeric", grading="numeric",
          notes="v1.0 is gated on Hugging Face; v1.2 is open. Tolerance: decimal answers within the given "
                "lower/upper limits (5%), integers exact, dates exact"),
    _spec("healthbench", "HealthBench (oss_eval)", "medicine", "openai/healthbench",
          ["2025-05-07-06-14-12_oss_eval.jsonl"], "MIT", _one(norm_healthbench), kind="open", grading="rubric",
          scored=False, notes="rubric-graded conversations (needs a grader model); not scored"),
    _spec("humaneval", "HumanEval", "code", "openai/openai_humaneval",
          ["openai_humaneval/test-00000-of-00001.parquet"], "MIT", _one(norm_humaneval), kind="code",
          grading="exec"),
    _spec("humanevalplus", "HumanEval+ (EvalPlus)", "code", "evalplus/humanevalplus",
          ["data/test-00000-of-00001-5973903632b82d40.parquet"], "Apache-2.0",
          lambda raw: norm_humaneval(read_table(raw[0]), plus=True), kind="code", grading="exec",
          notes="test harness needs numpy in the sandbox (sandbox-setup)"),
    _spec("mbpp", "MBPP sanitized (test)", "code", "google-research-datasets/mbpp",
          ["sanitized/test-00000-of-00001.parquet"], "CC BY 4.0", _one(norm_mbpp), kind="code", grading="exec"),
    _spec("mbppplus", "MBPP+ (EvalPlus)", "code", "evalplus/mbppplus",
          ["data/test-00000-of-00001-d5781c9c51e02795.parquet"], "Apache-2.0",
          lambda raw: norm_mbpp(read_table(raw[0]), plus=True), kind="code", grading="exec",
          notes="test harness needs numpy in the sandbox (sandbox-setup)"),
    _spec("ds1000", "DS-1000", "code", "xlangai/DS-1000", ["test.jsonl"], "CC BY-SA 4.0", _one(norm_ds1000),
          kind="code", grading="exec", notes="each item needs its library in the sandbox (pandas, numpy, scipy, "
                                             "sklearn, matplotlib, torch, tensorflow); items without it are skipped"),
    _spec("biocoder", "BioCoder (public set)", "code", "lilbillbiscuit/biocoder_public", ["biocoder.json"],
          "CC BY 4.0", _one(norm_biocoder), kind="code", grading="none", scored=False,
          notes="bioinformatics functions from real repositories; the public release has no test harness "
                "(BioCoder runs hidden tests in its own Docker images), so not scored"),
    _spec("bigcodebench", "BigCodeBench v0.1.4", "code", "bigcode/bigcodebench",
          ["data/v0.1.4-00000-of-00001.parquet"], "Apache-2.0", _one(norm_bigcodebench), kind="code",
          grading="exec-heavy", scored=False,
          notes="data only: its tests need ~70 libraries; run with BigCodeBench's own Docker environment"),
    _spec("cadbench", "CADBench (BlenderLLM)", "3d", "FreedomIntelligence/CADBench", ["CADBench.jsonl"],
          "Apache-2.0", _one(norm_cadbench), kind="code", grading="vision", scored=False,
          notes="Blender Python scripts from text; graded by rendering + a vision model against per-item criteria. "
                "Only an automatic 'script runs and makes geometry' check is possible here, and only when the "
                "sandbox has headless Blender"),
]
BY_NAME = {b["name"]: b for b in BENCHMARKS}

SKIPPED = [
    ("GPQA", "gated (agreement required)"),
    ("Humanity's Last Exam", "gated"),
    ("BioASQ", "registration required"),
    ("NEJM / JAMA clinical challenges", "copyrighted, not freely redistributed"),
    ("MIMIC-based tasks", "credentialed access (PhysioNet)"),
    ("Virology Capabilities Test (VCT)", "restricted"),
    ("MedCalc-Bench v1.0", "gated on Hugging Face; the open v1.2 is used instead"),
    ("LAB-Bench LitQA2", "needs literature retrieval; closed-book it measures recall of 2024 papers"),
    ("LAB-Bench SuppQA", "needs the paper's supplementary material (not included)"),
    ("LAB-Bench FigQA / TableQA", "need images (333 MB)"),
    ("MedXpertQA MM", "needs images (517 MB)"),
    ("BioCoder hidden set (lilbillbiscuit/biocoder_hidden)", "no licence declared; training-format pairs, no tests"),
    ("BlenderGym (richard-guyunqi/BlenderGym_subset)", "1.3 GB and no licence declared"),
    ("CADPrompt (AI4Manufacturing/cadprompt)", "gated and no licence declared"),
    ("Text2CAD (SadilKhan/Text2CAD)", "gated, CC BY-NC-SA 4.0, hundreds of GB"),
    ("3DCodeBench (3dcodebench/3dcodebench)", "1.9 GB archive"),
    ("BlenderBench (DietCoke4671/BlenderBench)", "19 GB of .blend files"),
    ("Text2CAD-Bench (AICAD/Text2CAD-Bench)", "no licence declared"),
]


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def load_manifest():
    if os.path.isfile(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as handle:
            return json.load(handle)
    return {"benchmarks": {}, "skipped": []}


def save_manifest(man):
    os.makedirs(BENCH_DIR, exist_ok=True)
    with open(MANIFEST + ".part", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(man, handle, indent=2, ensure_ascii=False)
    os.replace(MANIFEST + ".part", MANIFEST)


def write_items(path, items):
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(path + ".part", path)


def read_items(name):
    path = os.path.join(BENCH_DIR, name, "items.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def folder_bytes(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def download_one(spec, man, force=False):
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(spec["repo"])
    if info.gated:
        raise RuntimeError(f"{spec['repo']} is gated ({info.gated}); not downloaded")
    card = info.card_data.to_dict() if info.card_data else {}
    folder = os.path.join(BENCH_DIR, spec["name"])
    raw_dir = os.path.join(folder, "raw")
    old = man["benchmarks"].get(spec["name"], {})
    raw_paths = [os.path.join(raw_dir, *f.split("/")) for f in spec["files"]]
    if (not force and old.get("revision") == info.sha and all(os.path.isfile(p) for p in raw_paths)
            and os.path.isfile(os.path.join(folder, "items.jsonl"))):
        return old, "up to date"
    os.makedirs(raw_dir, exist_ok=True)
    for f in spec["files"]:
        hf_hub_download(spec["repo"], f, repo_type="dataset", revision=info.sha, local_dir=raw_dir)
    items = spec["normalise"](raw_paths)
    bad = [(it["id"], why) for it in items if (why := valid_item(it))]
    items = [it for it in items if not valid_item(it)]
    items_path = os.path.join(folder, "items.jsonl")
    write_items(items_path, items)
    entry = {"title": spec["title"], "group": spec["group"], "kind": spec["kind"], "grading": spec["grading"],
             "scored": spec["scored"], "source": spec["url"], "repo_id": spec["repo"], "revision": info.sha,
             "licence": spec["licence"], "card_licence": card.get("license"),
             "files": [{"path": f, "bytes": os.path.getsize(p), "sha256": file_sha256(p)}
                       for f, p in zip(spec["files"], raw_paths)],
             "items": len(items), "dropped_items": len(bad), "dropped_examples": bad[:5],
             "items_sha256": file_sha256(items_path), "categories": len({it["category"] for it in items}),
             "bytes": folder_bytes(folder), "downloaded": now_iso(), "notes": spec["notes"]}
    return entry, "downloaded"


def cmd_download(args):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    names = parse_names(args.only, BY_NAME)
    man = load_manifest()
    print(f"{'benchmark':<15} {'items':>7} {'size':>9}  {'licence':<32} status")
    failures = []
    for name in names:
        spec = BY_NAME[name]
        try:
            entry, status = download_one(spec, man, args.force)
        except Exception as exc:  # noqa: BLE001 - one failed source must not stop the rest
            failures.append(name)
            print(f"{name:<15} {'-':>7} {'-':>9}  {spec['licence'][:32]:<32} FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:160]}", flush=True)
            continue
        man["benchmarks"][name] = entry
        save_manifest(man)
        print(f"{name:<15} {entry['items']:>7,} {entry['bytes'] / 1e6:>7.1f}MB  {entry['licence'][:32]:<32} {status}"
              + (f" ({entry['dropped_items']} malformed items dropped)" if entry["dropped_items"] else ""),
              flush=True)
    man["skipped"] = [{"name": n, "reason": r} for n, r in SKIPPED]
    man["updated"] = now_iso()
    save_manifest(man)
    total = sum(e["bytes"] for e in man["benchmarks"].values())
    print(f"\n{len(man['benchmarks'])} benchmarks, {total / 1e6:,.1f} MB in {BENCH_DIR}; manifest: {MANIFEST}")
    print("Not downloaded: " + "; ".join(f"{n} ({r})" for n, r in SKIPPED))
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    return 0


def parse_names(text, known):
    if not text:
        return list(known)
    names = [n.strip().lower().replace("-", "_") for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"Unknown benchmark(s): {', '.join(unknown)}. Known: {', '.join(known)}")
    return names


# ---------------------------------------------------------------------------
# sandbox for generated code
# ---------------------------------------------------------------------------
class SandboxResult:
    def __init__(self, passed, status, output=""):
        self.passed, self.status, self.output = passed, status, output


def _run(cmd, timeout, stdin=None):
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout, creationflags=flags)


def docker_available():
    if not shutil.which("docker"):
        return False
    try:
        return _run(["docker", "info", "--format", "{{.ServerVersion}}"], 20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def wsl_distros():
    if not shutil.which("wsl"):
        return []
    try:
        out = _run(["wsl", "-l", "-q"], 30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    text = out.decode("utf-16-le", errors="ignore") if b"\x00" in out else out.decode(errors="ignore")
    return [d.strip() for d in text.replace("\x00", "").splitlines()
            if d.strip() and not d.strip().lower().startswith("docker-desktop")]


WSL_PROBE = r"""
command -v python3 >/dev/null || { echo NO_PYTHON; exit 3; }
command -v timeout >/dev/null || { echo NO_TIMEOUT; exit 3; }
if command -v bwrap >/dev/null && bwrap --ro-bind / / --unshare-all --die-with-parent true 2>/dev/null; then
  echo TOOL=bwrap
elif command -v unshare >/dev/null && unshare -rmn true 2>/dev/null; then
  echo TOOL=unshare
else
  echo NO_ISOLATION; exit 3
fi
command -v blender >/dev/null && echo BLENDER=yes
exit 0
"""


def wsl_probe(distro):
    """(tool, has_blender) if `distro` can isolate code (bubblewrap or unshare), else None."""
    try:
        proc = _run(["wsl", "-d", distro, "-e", "bash", "-s"], 120, WSL_PROBE.encode())
    except (OSError, subprocess.SubprocessError):
        return None
    text = proc.stdout.decode(errors="replace")
    m = re.search(r"TOOL=(\w+)", text)
    if proc.returncode != 0 or not m:
        return None
    return m.group(1), "BLENDER=yes" in text


class DockerSandbox:
    name = "docker"

    def __init__(self, image="python:3.12-slim", memory="1g", cpus="1"):
        self.image, self.memory, self.cpus = image, memory, cpus
        self.has_blender = False

    def describe(self):
        return f"Docker {self.image} (--network none, --memory {self.memory}, --cpus {self.cpus}, read-only root)"

    def run(self, program, timeout=20, runner="python"):
        exe = ["python", "-"] if runner == "python" else ["blender", "-b", "--factory-startup",
                                                          "--python-exit-code", "1", "--python", "/dev/stdin"]
        cmd = ["docker", "run", "--rm", "-i", "--network", "none", "--memory", self.memory, "--cpus", self.cpus,
               "--pids-limit", "256", "--read-only", "--tmpfs", "/tmp", "-e", "MPLBACKEND=Agg", "-w", "/tmp",
               self.image, "timeout", "-k", "2", str(int(timeout))] + exe
        return _finish(cmd, program, timeout)


class WSLSandbox:
    name = "wsl"

    def __init__(self, distro, tool="bwrap", has_blender=False, memory_mb=1024):
        self.distro, self.tool, self.has_blender, self.memory_mb = distro, tool, has_blender, memory_mb

    def describe(self):
        how = ("bubblewrap: no network, only /usr visible read-only, no Windows drives" if self.tool == "bwrap"
               else "unshare: no network, /mnt and /home hidden")
        return f"WSL {self.distro} ({how}; {self.memory_mb} MB address space, timeout)"

    def script(self, program, timeout, runner="python"):
        b64 = base64.b64encode(program.encode("utf-8")).decode("ascii")
        lines = [b64[i:i + 76] for i in range(0, len(b64), 76)]
        mem = self.memory_mb * 1024 * 1024
        if runner == "python":
            exe = "python3 -s prog.py"
        else:
            exe = "blender -b --factory-startup --python-exit-code 1 --python prog.py"
        if self.tool == "bwrap":
            jail = ("args='--ro-bind /usr /usr --proc /proc --dev /dev --tmpfs /tmp --bind \"$d\" /work --chdir /work "
                    "--unshare-all --die-with-parent --new-session --clearenv --setenv PATH /usr/bin:/bin "
                    "--setenv HOME /work --setenv MPLBACKEND Agg --setenv LANG C.UTF-8'\n"
                    "for l in bin lib lib64 sbin; do [ -L /$l ] && args=\"$args --symlink usr/$l /$l\"; done\n"
                    "[ -d /etc/alternatives ] && args=\"$args --ro-bind /etc/alternatives /etc/alternatives\"\n"
                    f"[ -d {SANDBOX_SITE} ] && "
                    f"args=\"$args --ro-bind {SANDBOX_SITE} /site --setenv PYTHONPATH /site\"\n"
                    f"eval timeout -k 2 {int(timeout)} prlimit --as={mem} bwrap $args {exe}\n")
        else:
            jail = (f"cd \"$d\"; timeout -k 2 {int(timeout)} prlimit --as={mem} unshare -rmn sh -c "
                    f"'mount -t tmpfs none /mnt && mount -t tmpfs none /home && cd \"$0\" && MPLBACKEND=Agg "
                    f"exec {exe}' \"$d\"\n")
        return ("d=$(mktemp -d /tmp/tgbench.XXXXXX)\n"
                "base64 -d > \"$d/prog.py\" <<'__TINYGPT_B64__'\n" + "\n".join(lines) + "\n__TINYGPT_B64__\n"
                + jail + "rc=$?\nrm -rf \"$d\"\nexit $rc\n")

    def run(self, program, timeout=20, runner="python"):
        return _finish(["wsl", "-d", self.distro, "-e", "bash", "-s"], self.script(program, timeout, runner),
                       timeout)


class LocalSubprocess:
    """--unsafe-local-execution only: a separate Python process in a temporary folder (not isolated)."""
    name = "local-unsafe"
    has_blender = False

    def describe(self):
        return "UNSAFE local subprocess (no isolation: generated code can read and change your files)"

    def run(self, program, timeout=20, runner="python"):
        if runner != "python":
            return SandboxResult(False, "unsupported")
        with tempfile.TemporaryDirectory(prefix="tgbench_") as d:
            path = os.path.join(d, "prog.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(program)
            try:
                proc = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, timeout=timeout + 5)
            except subprocess.TimeoutExpired:
                return SandboxResult(False, "timeout")
        out = (proc.stdout + proc.stderr).decode(errors="replace")[-2000:]
        return SandboxResult(proc.returncode == 0, "passed" if proc.returncode == 0 else "failed", out)


def _finish(cmd, stdin_text, timeout):
    try:
        proc = _run(cmd, timeout + 60, stdin_text.encode("utf-8"))
    except subprocess.TimeoutExpired:
        return SandboxResult(False, "timeout")
    out = (proc.stdout + proc.stderr).decode(errors="replace")[-2000:]
    if proc.returncode == 0:
        return SandboxResult(True, "passed", out)
    if proc.returncode in (124, 137):
        return SandboxResult(False, "timeout", out)
    return SandboxResult(False, "failed", out)


def select_sandbox(prefer="auto", unsafe=False, distro=None, docker_image="python:3.12-slim", log=print):
    """The safest available place to run generated code, or None (then code is never run)."""
    if prefer in ("auto", "docker"):
        if docker_available():
            return DockerSandbox(docker_image)
        log("Docker: daemon not running (or docker not installed).")
    if prefer in ("auto", "wsl"):
        candidates = [distro] if distro else sorted(wsl_distros(),
                                                    key=lambda d: (not d.lower().startswith("ubuntu"), d))
        for d in candidates:
            probe = wsl_probe(d)
            if probe:
                return WSLSandbox(d, probe[0], probe[1])
        log(f"WSL: no distribution with python3 and bubblewrap/unshare ({', '.join(candidates) or 'none found'}).")
    if unsafe:
        return LocalSubprocess()
    return None


_MODULE_CACHE = {}


def sandbox_modules(sandbox, modules):
    """{module: importable?} inside the sandbox."""
    key = (sandbox.name, getattr(sandbox, "distro", getattr(sandbox, "image", "")))
    cache = _MODULE_CACHE.setdefault(key, {})
    todo = [m for m in modules if m not in cache]
    if todo:
        prog = ("import importlib.util, json\nprint('MODS=' + json.dumps({m: importlib.util.find_spec(m) is not None "
                f"for m in {todo!r}}}))\n")
        res = sandbox.run(prog, timeout=30)
        m = re.search(r"MODS=(\{.*\})", res.output)
        found = json.loads(m.group(1)) if m else {}
        for mod in todo:
            cache[mod] = bool(found.get(mod))
    return {m: cache[m] for m in modules}


# ---------------------------------------------------------------------------
# prompts and checks for generated code
# ---------------------------------------------------------------------------
CODE_STOPS = {
    "humaneval": ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert"],
    "mbpp": ['\n"""', "\nassert", "\nif __name__", "\nprint(", "\n# Test", "\n```"],
    "ds1000": ["</code>", "# SOLUTION END", "END SOLUTION", "\n```"],
    "blender": ["\n\n\n", "\n```", "\n# End"],
}
BLENDER_EPILOGUE = """

import bpy as _bpy, sys as _sys
_meshes = [o for o in _bpy.data.objects if o.type == 'MESH' and len(o.data.vertices) > 0]
print('MESH_OBJECTS=%d' % len(_meshes))
_sys.exit(0 if _meshes else 1)
"""


def code_prompt(item):
    style = item.get("style")
    if style == "humaneval":
        return item["question"]
    if style == "mbpp":
        example = item["test_list"][0] if item.get("test_list") else ""
        return f'"""\n{item["question"].strip()}\n{example}\n"""\n'
    if style == "ds1000":
        return item["question"]
    if style == "blender":
        return f"# Blender Python script: {item['question'].strip()}\nimport bpy\n"
    raise ValueError(f"no prompt style {style}")


def cut_at_stops(text, stops):
    cut = min((i for i in (text.find(s) for s in stops) if i >= 0), default=-1)
    return text[:cut] if cut >= 0 else text


def code_program(item, completion):
    style = item.get("style")
    if style == "humaneval":
        return f"{item['question']}{completion}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
    if style == "mbpp":
        imports = "\n".join(item.get("test_imports") or [])
        tests = item["test"] if item.get("test") else "\n".join(item["test_list"])
        return f"{imports}\n{completion}\n\n{tests}\n"
    if style == "ds1000":
        ctx = item["code_context"]
        return (f"{ctx}\ncode = {completion!r}\ntest_execution(code)\n"
                + ("test_string(code)\n" if "test_string(" in ctx else ""))
    if style == "blender":
        return "import bpy\n" + completion + BLENDER_EPILOGUE
    raise ValueError(f"no program style {style}")


# ---------------------------------------------------------------------------
# numeric answers (MedCalc-Bench)
# ---------------------------------------------------------------------------
NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+")
DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")


def parse_first_number(text):
    m = NUM_RE.search(text or "")
    return float(m.group(0).replace(",", "")) if m else None


def parse_date(text):
    m = DATE_RE.search(text or "")
    if not m:
        return None
    month, day, year = (int(g) for g in m.groups())
    if year < 100:
        year += 2000
    return (year, month, day)


def numeric_correct(item, generated):
    """(correct, parsed) following MedCalc-Bench: dates exact, tuples (weeks, days) exact, integers exact after
    rounding, decimals within the item's lower/upper limits."""
    gt = str(item["answer"]).strip()
    if item.get("output_type") == "date" or DATE_RE.search(gt):
        got, want = parse_date(generated), parse_date(gt)
        if want is not None:
            return got == want, got
    if gt.startswith("("):
        want = [float(x) for x in NUM_RE.findall(gt)]
        got = [float(x.replace(",", "")) for x in NUM_RE.findall(generated or "")][:len(want)]
        return got == want, got
    value = parse_first_number(generated)
    if value is None:
        return False, None
    if item.get("output_type") == "integer":
        return round(value) == round(float(gt)), value
    try:
        low, high = float(item.get("lower")), float(item.get("upper"))
    except (TypeError, ValueError):
        low = high = float(gt)
    return low <= value <= high, value


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def sample_items(items, limit, name):
    """A deterministic sample of `limit` items (by hash of benchmark and id), in file order."""
    if not limit or len(items) <= limit:
        return items
    keep = set(sorted(range(len(items)), key=lambda i: stable_key(name, items[i]["id"]))[:limit])
    return [it for i, it in enumerate(items) if i in keep]


def mcq_prompt(item):
    if item.get("context"):
        return f"{item['context'].strip()}\nQuestion: {item['question'].strip()}"
    return f"Question: {item['question'].strip()}"


def score_mcq(model, tok, items, device, amp_dtype, floor=True, batch_size=32):
    import tiny_gpt
    probes = [{"category": it["category"], "difficulty": "unrated", "prompt": mcq_prompt(it), "answer": it["answer"],
               "distractors": [o for o in it["options"] if o != it["answer"]]} for it in items]
    _, results = tiny_gpt.score_probes(model, tok, probes, device, amp_dtype, batch_size=batch_size, floor=floor,
                                       template=MCQ_TEMPLATE)
    rows = []
    for it, r in zip(items, results):
        rows.append({"id": it["id"], "category": it["category"], "tags": it.get("tags", []),
                     "correct": bool(r["correct"]), "chance": r["chance"], "floor_correct": r.get("floor_correct"),
                     "answer": it["answer"], "correct_score": round(r["correct_score"], 4),
                     "best_other": r["best_distractor"], "best_other_score": round(r["best_distractor_score"], 4)})
    return rows


def greedy_generate(model, tok, prompt, max_new_tokens, device, amp_dtype=None, stops=()):
    """Greedy continuation of `prompt`, stopped at the first stop string."""
    import torch
    import tiny_gpt
    prefix = [tok.bos_id] if tok.bos_id >= 0 else []
    ids = prefix + tok.encode(prompt)
    start, n_prompt = len(prefix), len(ids)
    head = tok.decode(ids[start:])
    text = ""
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-model.ctx:]], device=device)
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                if isinstance(model, tiny_gpt.TinyGPT):
                    logits = model.lm_head(model(x, return_hidden=True)[:, -1])[0]
                else:
                    logits = model(x)[0, -1]
            nxt = int(logits.float().argmax())
            if nxt == tok.eos_id:
                break
            ids.append(nxt)
            full = tok.decode(ids[start:])
            text = full[len(head):] if full.startswith(head) else tok.decode(ids[n_prompt:])
            if stops and any(s in text for s in stops):
                return cut_at_stops(text, stops)
    return cut_at_stops(text, stops) if stops else text


def score_numeric(model, tok, items, device, amp_dtype, max_new_tokens=24):
    rows = []
    for it in items:
        prompt = f"{it.get('context', '').strip()}\n\nQuestion: {it['question'].strip()}\nAnswer:"
        gen = greedy_generate(model, tok, prompt, max_new_tokens, device, amp_dtype, stops=["\n"])
        ok, parsed = numeric_correct(it, gen)
        rows.append({"id": it["id"], "category": it["category"], "correct": bool(ok), "chance": 0.0,
                     "floor_correct": None, "answer": it["answer"], "parsed": parsed, "generated": gen[:200]})
    return rows


def score_code(model, tok, items, device, amp_dtype, sandbox, max_new_tokens=256, timeout=20, log=print):
    rows = []
    for k, it in enumerate(items, 1):
        style = it["style"]
        runner = "blender" if style == "blender" else "python"
        gen = greedy_generate(model, tok, code_prompt(it), max_new_tokens, device, amp_dtype,
                              stops=CODE_STOPS[style])
        res = sandbox.run(code_program(it, gen), timeout=timeout, runner=runner)
        rows.append({"id": it["id"], "category": it["category"], "correct": bool(res.passed), "chance": 0.0,
                     "floor_correct": None, "status": res.status, "generated": gen[:400],
                     "output": res.output[-300:]})
        if k % 10 == 0:
            log(f"    {k}/{len(items)} done, {sum(r['correct'] for r in rows)} passed", flush=True)
    return rows


def summarise(rows, by="category"):
    """Summary dicts: overall ('all'), per category and per tag ('[tag]')."""
    import tiny_gpt

    def one(label, group):
        k, n = sum(r["correct"] for r in group), len(group)
        low, high = tiny_gpt.wilson_interval(k, n)
        floors = [r["floor_correct"] for r in group if r.get("floor_correct") is not None]
        return {"category": label, "n": n, "correct": k, "accuracy": k / n if n else float("nan"), "low": low,
                "high": high, "chance": sum(r["chance"] for r in group) / n if n else float("nan"),
                "floor": sum(floors) / len(floors) if floors else None}

    out = [one("all", rows)]
    tags = sorted({t for r in rows for t in r.get("tags", [])})
    for t in tags:
        out.append(one(f"[{t}]", [r for r in rows if t in r.get("tags", [])]))
    cats = sorted({r[by] for r in rows})
    if len(cats) > 1:
        for c in cats:
            out.append(one(c, [r for r in rows if r[by] == c]))
    return out


def fmt(x, spec=".4f"):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def append_results(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: home_short(v) if isinstance(v, str) else v for k, v in row.items()})


def write_item_csv(path, rows):
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (";".join(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else v)
                             for k, v in r.items()})


def gpu_busy():
    try:
        import run_pipeline
        return run_pipeline.gpu_busy()
    except Exception:  # noqa: BLE001
        return False


def cmd_run(args):
    man = load_manifest()
    available = [n for n in BY_NAME if n in man["benchmarks"]
                 and os.path.isfile(os.path.join(BENCH_DIR, n, "items.jsonl"))]
    names = parse_names(args.benchmarks, BY_NAME) if args.benchmarks else [n for n in available
                                                                           if BY_NAME[n]["scored"]]
    missing = [n for n in names if n not in available]
    if missing:
        raise SystemExit(f"Not downloaded: {', '.join(missing)}. Run `python benchmark.py download --only "
                         f"{','.join(missing)}` first.")
    unscored = [n for n in names if not BY_NAME[n]["scored"]]
    for n in unscored:
        spec = BY_NAME[n]
        why = {"vision": "requires vision grading", "rubric": "requires rubric grading by a model",
               "exec-heavy": "requires its own heavy test environment"}.get(spec["grading"], "no automatic grader")
        print(f"{n}: {why}: not scored ({spec['notes']})")
    names = [n for n in names if BY_NAME[n]["scored"]]
    code_names = [n for n in names if BY_NAME[n]["kind"] == "code"]
    sandbox = None
    if code_names:
        sandbox = select_sandbox(args.sandbox, args.unsafe_local_execution, args.wsl_distro, args.docker_image)
        if sandbox is None:
            print(f"REFUSING to run generated code for {', '.join(code_names)}: no sandbox (start Docker Desktop, "
                  "or install bubblewrap in a WSL distribution with python3). Generated code is never run on the "
                  "Windows host; --unsafe-local-execution runs it in a plain subprocess if you accept the risk.")
            names = [n for n in names if n not in code_names]
        else:
            print(f"Sandbox for generated code: {sandbox.describe()}")
    if not names:
        print("Nothing to score.")
        return 1
    import tiny_gpt
    if args.device != "cpu" and not args.force and gpu_busy():
        raise SystemExit("The GPU is busy (another job?). Use --device cpu, wait, or pass --force.")
    device = tiny_gpt.select_device(args.device)
    amp_dtype, amp_name, _ = tiny_gpt.choose_precision(device, "auto", training=False)
    path = os.path.abspath(args.checkpoint)
    model, tok, cfg, obj = tiny_gpt.load_for_inference(path, device, args.trust_checkpoint)
    model.set_attention_impl(tiny_gpt.choose_attention(device, cfg, amp_dtype, "auto")[0])
    step, params = obj.get("step"), tiny_gpt.count_params(cfg)[0]
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"{os.path.splitext(os.path.basename(path))[0]}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"{os.path.basename(path)}: step {step}, {params / 1e6:.1f}M parameters, {tiny_gpt.device_label(device)} "
          f"({amp_name}); limit {args.limit or 'none'}\n")
    t_all = time.time()
    csv_rows, notes = [], []
    for name in names:
        spec, t0 = BY_NAME[name], time.time()
        items = sample_items(read_items(name), args.limit, name)
        note = f"{len(items)} of {man['benchmarks'][name]['items']} items" + (" (hash sample)" if args.limit else "")
        if spec["kind"] == "code":
            needed = sorted({m for it in items for m in it.get("requires", [])})
            have = sandbox_modules(sandbox, needed) if needed else {}
            lacking = {m for m, ok in have.items() if not ok}
            if lacking:
                before = len(items)
                items = [it for it in items if not set(it.get("requires", [])) & lacking]
                note += f"; {before - len(items)} skipped: sandbox lacks {', '.join(sorted(lacking))}"
                print(f"{name}: sandbox lacks {', '.join(sorted(lacking))} ({before - len(items)} items skipped; "
                      "see `python benchmark.py sandbox-setup`)")
            if not items:
                continue
        print(f"{name}: {len(items)} items ({spec['kind']}) ...", flush=True)
        if spec["kind"] == "mcq":
            rows = score_mcq(model, tok, items, device, amp_dtype, floor=not args.no_floor, batch_size=args.batch)
            note += "; option log-likelihood per character"
        elif spec["kind"] == "numeric":
            rows = score_numeric(model, tok, items, device, amp_dtype)
            note += "; greedy, first number, MedCalc tolerance"
        else:
            rows = score_code(model, tok, items, device, amp_dtype, sandbox, args.max_new_tokens, args.timeout)
            note += f"; greedy pass@1, tests in {sandbox.name} sandbox"
        write_item_csv(os.path.join(run_dir, f"{name}.csv"), rows)
        summaries = summarise(rows)
        for s in summaries:
            csv_rows.append({"time": now_iso(), "checkpoint": path, "step": step, "params": params, "benchmark": name,
                             "category": s["category"], "n": s["n"], "accuracy": fmt(s["accuracy"]),
                             "low": fmt(s["low"]), "high": fmt(s["high"]), "chance": fmt(s["chance"]),
                             "floor": fmt(s["floor"]), "notes": note})
        s = summaries[0]
        floor_txt = f", floor {s['floor']:.1%}" if s["floor"] is not None else ""
        print(f"  {name}: {s['correct']}/{s['n']} = {s['accuracy']:.1%} (95% {s['low']:.1%}-{s['high']:.1%}); "
              f"chance {s['chance']:.1%}{floor_txt}  [{time.time() - t0:,.0f} s]", flush=True)
        for t in (x for x in summaries if x["category"].startswith("[")):
            print(f"    {t['category']:<14} {t['correct']}/{t['n']} = {t['accuracy']:.1%}")
        notes.append(f"{name} {s['accuracy']:.3f} (n={s['n']}, chance {s['chance']:.2f}"
                     + (f", floor {s['floor']:.3f}" if s["floor"] is not None else "") + ")")
        append_results(args.results, csv_rows[-len(summaries):])
    print(f"\nResults appended to {args.results}; per-item CSVs in {run_dir} ({time.time() - t_all:,.0f} s)")
    if args.no_publish:
        tiny_gpt.publish_experiments = lambda *a, **k: print("experiments.csv: not published (--no-publish)")
    tiny_gpt.record_experiment(os.path.dirname(path), {
        "event": "benchmark suite", "name": os.path.basename(path), "step": step, "params": params,
        "d_model": cfg.d_model, "layers": cfg.n_layers, "ctx": cfg.ctx, "vocab": cfg.vocab_size,
        "device": tiny_gpt.device_label(device), "checkpoint": path,
        "notes": "; ".join(notes) + f"; limit {args.limit or 'none'}; items in benchmarks\\runs\\"
                                    f"{os.path.basename(run_dir)}"})
    return 0


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def read_results(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def latest_rows(rows, checkpoint=None):
    """The most recent result of each (benchmark, category) for one checkpoint (default: the last one run)."""
    if checkpoint:
        want = os.path.basename(checkpoint).lower()
        rows = [r for r in rows if os.path.basename(r["checkpoint"].replace("\\", "/")).lower() == want]
    elif rows:
        last = rows[-1]["checkpoint"]
        rows = [r for r in rows if r["checkpoint"] == last]
    best = {}
    for r in rows:
        best[(r["benchmark"], r["category"])] = r
    return list(best.values())


def plot_group(bench, category):
    if category.startswith("["):
        tag = category.strip("[]")
        return {"health": "medicine", "genetics": "biology"}.get(tag, tag if tag in GROUPS else "knowledge")
    return BY_NAME[bench]["group"] if bench in BY_NAME else "knowledge"


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import view_pt as vp

    rows = read_results(args.results)
    if not rows:
        raise SystemExit(f"No results in {args.results}; run `python benchmark.py run` first.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.labelsize": 8.5, "axes.labelcolor": vp.INK2, "text.color": vp.INK,
                         "axes.titlecolor": vp.INK, "figure.facecolor": vp.SURFACE, "legend.fontsize": 7,
                         "legend.frameon": False})
    os.makedirs(args.out_dir, exist_ok=True)
    latest = latest_rows(rows, args.checkpoint)
    if not latest:
        raise SystemExit(f"No results for {args.checkpoint} in {args.results}.")
    ck = os.path.basename(latest[0]["checkpoint"].replace("\\", "/"))
    stem = os.path.splitext(ck)[0]
    step = latest[0]["step"]
    written = []
    colour = {g: vp.SERIES[i] for i, g in enumerate(GROUPS)}

    # (a) accuracy per benchmark, grouped
    bars = [r for r in latest if r["category"] == "all" or (r["category"].startswith("[") and r["benchmark"]
                                                            in ("mmlu", "mmlu_pro"))]
    bars.sort(key=lambda r: (GROUPS.index(plot_group(r["benchmark"], r["category"])), r["benchmark"],
                             r["category"] != "all", r["category"]))
    labels = [(BY_NAME[r["benchmark"]]["title"].split(" (")[0] if r["benchmark"] in BY_NAME else r["benchmark"])
              + ("" if r["category"] == "all" else f" {r['category']}") + f" (n={r['n']})" for r in bars]
    acc = np.array([_f(r["accuracy"]) for r in bars])
    low = np.array([_f(r["low"]) for r in bars])
    high = np.array([_f(r["high"]) for r in bars])
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.36 * len(bars)), layout="constrained")
    vp.style_axes(ax)
    y = np.arange(len(bars))[::-1]
    groups = [plot_group(r["benchmark"], r["category"]) for r in bars]
    ax.barh(y, acc, color=[colour[g] for g in groups], height=0.62, alpha=0.9)
    ax.errorbar(acc, y, xerr=[acc - low, high - acc], fmt="none", ecolor=vp.INK2, elinewidth=0.8, capsize=2)
    for yi, r in zip(y, bars):
        c, f = _f(r["chance"]), _f(r["floor"])
        if c is not None and c > 0:
            ax.plot([c, c], [yi - 0.38, yi + 0.38], color=vp.MUTED, lw=1.6, ls=":")
        if f is not None:
            ax.plot([f, f], [yi - 0.38, yi + 0.38], color=vp.INK, lw=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy (MCQ: answer preferred by log-likelihood per character; numeric/code: exact / pass@1); "
                  "lines: 95% interval")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=colour[g], label=g) for g in GROUPS if g in groups]
    handles += [Line2D([], [], color=vp.MUTED, ls=":", lw=1.6, label="chance"),
                Line2D([], [], color=vp.INK, lw=2, label="floor (question words shuffled)")]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title(f"Benchmarks: {ck} (step {step})", loc="left")
    out = os.path.join(args.out_dir, f"{stem}_benchmarks.png")
    fig.savefig(out, dpi=args.dpi, facecolor=vp.SURFACE)
    plt.close(fig)
    written.append(out)

    # (b) categories of MMLU and MMLU-Pro
    panels = []
    for bench in ("mmlu", "mmlu_pro"):
        cats = [r for r in latest if r["benchmark"] == bench and r["category"] != "all"
                and not r["category"].startswith("[")]
        if cats:
            panels.append((bench, sorted(cats, key=lambda r: _f(r["accuracy"]))))
    if panels:
        height = 1.5 + 0.2 * max(len(c) for _, c in panels)
        fig, axs = plt.subplots(1, len(panels), figsize=(7 * len(panels), height), layout="constrained",
                                squeeze=False)
        for ax, (bench, cats) in zip(axs[0], panels):
            vp.style_axes(ax)
            a = np.array([_f(r["accuracy"]) for r in cats])
            lo = np.array([_f(r["low"]) for r in cats])
            hi = np.array([_f(r["high"]) for r in cats])
            fl = [_f(r["floor"]) for r in cats]
            yy = np.arange(len(cats))
            ax.barh(yy, a, color=[vp.GOOD if f is not None and v > f else vp.CRITICAL for v, f in zip(a, fl)],
                    height=0.62, alpha=0.9)
            ax.errorbar(a, yy, xerr=[a - lo, hi - a], fmt="none", ecolor=vp.INK2, elinewidth=0.7, capsize=1.5)
            ax.scatter([f if f is not None else np.nan for f in fl], yy, marker="|", s=80, color=vp.INK, lw=1.6,
                       zorder=4, label="floor (question words shuffled)")
            chance = np.mean([_f(r["chance"]) for r in cats])
            ax.axvline(chance, color=vp.MUTED, lw=0.8, ls=":", label=f"chance (mean {chance:.0%})")
            ax.set_yticks(yy)
            ax.set_yticklabels([f"{r['category']} (n={r['n']})" for r in cats], fontsize=6.5)
            ax.set_xlim(0, 1)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("accuracy; green above floor, red at or below")
            ax.set_title(f"{BY_NAME[bench]['title']} by category", loc="left")
            ax.legend(loc="lower right")
        out = os.path.join(args.out_dir, f"{stem}_categories.png")
        fig.savefig(out, dpi=args.dpi, facecolor=vp.SURFACE)
        plt.close(fig)
        written.append(out)

    # (c) progress over checkpoints / steps
    overall = [r for r in rows if r["category"] == "all"]
    runs = sorted({(r["checkpoint"], r["step"]) for r in overall}, key=lambda cs: (_f(cs[1]) or 0, cs[0]))
    if len(runs) > 1:
        fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
        vp.style_axes(ax)
        steps_known = all(_f(s) is not None for _, s in runs) and len({s for _, s in runs}) == len(runs)
        xpos = {cs: (_f(cs[1]) if steps_known else i) for i, cs in enumerate(runs)}
        benches = sorted({r["benchmark"] for r in overall})
        for i, b in enumerate(benches):
            pts = {}
            for r in overall:
                if r["benchmark"] == b:
                    pts[xpos[(r["checkpoint"], r["step"])]] = _f(r["accuracy"])
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], color=vp.SERIES[i % len(vp.SERIES)], lw=1.8, marker="o", ms=3.5,
                    label=b)
        if not steps_known:
            ax.set_xticks(range(len(runs)))
            ax.set_xticklabels([f"{os.path.basename(c.replace(chr(92), '/'))}\nstep {s}" for c, s in runs],
                               fontsize=7)
        ax.set_xlabel("training step" if steps_known else "checkpoint")
        ax.set_ylabel("accuracy (latest run of each checkpoint)")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", ncols=3)
        ax.set_title("Benchmark progress over checkpoints", loc="left")
        out = os.path.join(args.out_dir, "progress.png")
        fig.savefig(out, dpi=args.dpi, facecolor=vp.SURFACE)
        plt.close(fig)
        written.append(out)
    else:
        print("Progress plot: only one checkpoint/step in the results so far (needs two).")
    for w in written:
        print(w)
    return 0


# ---------------------------------------------------------------------------
# sandbox setup
# ---------------------------------------------------------------------------
def cmd_sandbox_setup(args):
    distros = [args.wsl_distro] if args.wsl_distro else sorted(
        wsl_distros(), key=lambda d: (not d.lower().startswith("ubuntu"), d))
    distro = next((d for d in distros if wsl_probe(d)), None)
    if not distro:
        raise SystemExit("No usable WSL distribution (python3 + bubblewrap/unshare). For Docker, build an image "
                         "FROM python:3.12-slim with `pip install numpy` and pass --docker-image.")
    pkgs = ["numpy"] + (["pandas", "scipy", "scikit-learn", "matplotlib"] if args.datasci else [])
    script = (f"mkdir -p {SANDBOX_SITE} && python3 -m pip install --quiet --disable-pip-version-check "
              f"--break-system-packages --upgrade --target {SANDBOX_SITE} {' '.join(pkgs)} && echo SETUP_OK")
    print(f"Installing {', '.join(pkgs)} into {SANDBOX_SITE} in WSL {distro} (mounted read-only as /site) ...")
    proc = _run(["wsl", "-d", distro, "-e", "bash", "-s"], 1800, script.encode())
    out = (proc.stdout + proc.stderr).decode(errors="replace")
    print(out[-1500:])
    return 0 if "SETUP_OK" in out else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download and normalise the benchmarks")
    d.add_argument("--only", help="comma-separated names (default: all): " + ",".join(BY_NAME))
    d.add_argument("--force", action="store_true", help="download again even if the revision is unchanged")
    r = sub.add_parser("run", help="score a checkpoint")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--benchmarks", help="comma-separated names (default: every downloaded, scorable benchmark)")
    r.add_argument("--limit", type=int, default=0, help="deterministic sample of at most N items per benchmark")
    r.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    r.add_argument("--force", action="store_true", help="use the GPU even if another job seems to be using it")
    r.add_argument("--no-floor", action="store_true", help="skip the shuffled-question floor (4x faster MCQ)")
    r.add_argument("--batch", type=int, default=32, help="sequences per batch for MCQ scoring")
    r.add_argument("--max-new-tokens", type=int, default=256, help="generated tokens per coding item")
    r.add_argument("--timeout", type=int, default=20, help="seconds per test run in the sandbox")
    r.add_argument("--sandbox", choices=("auto", "docker", "wsl"), default="auto")
    r.add_argument("--wsl-distro", help="WSL distribution for the sandbox (default: the first usable, Ubuntu first)")
    r.add_argument("--docker-image", default="python:3.12-slim")
    r.add_argument("--unsafe-local-execution", action="store_true",
                   help="without Docker or WSL, run generated code in a plain local subprocess (NOT isolated)")
    r.add_argument("--no-publish", action="store_true", help="record in experiments.csv but do not push it")
    r.add_argument("--trust-checkpoint", action="store_true")
    r.add_argument("--results", default=RESULTS_CSV)
    pl = sub.add_parser("plot", help="draw result PNGs")
    pl.add_argument("--checkpoint", help="checkpoint file name to plot (default: the last one run)")
    pl.add_argument("--results", default=RESULTS_CSV)
    pl.add_argument("--out-dir", default=PLOTS_DIR)
    pl.add_argument("--dpi", type=int, default=150)
    s = sub.add_parser("sandbox-setup", help="install numpy (and --datasci libraries) for the WSL sandbox")
    s.add_argument("--datasci", action="store_true", help="also pandas, scipy, scikit-learn, matplotlib (DS-1000)")
    s.add_argument("--wsl-distro")
    return p.parse_args(argv)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_args(argv)
    return {"download": cmd_download, "run": cmd_run, "plot": cmd_plot,
            "sandbox-setup": cmd_sandbox_setup}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
