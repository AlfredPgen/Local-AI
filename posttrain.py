r"""Post-training for tinyGPT: answering biology questions, from open datasets.

Pre-training (tiny_gpt.py) teaches language and facts; post-training teaches the
model to answer. This script prepares the data, runs the two methods in
finetune.py one after the other, and measures what each step changed.

  data   build the files in --data-dir (default posttrain_data\):
           sft.jsonl       question -> answer pairs (supervised fine-tuning)
           dpo.jsonl       question, preferred answer, worse answer (DPO)
           eval_mcq.jsonl  held-out multiple-choice questions, never trained on
           sources.md      what came from where, how many, and each licence
         Sources (downloaded with the Hugging Face `datasets` library, streamed,
         so only what is used is read):
           PubMedQA (MIT)             research questions answered by the abstract's conclusion
           MedMCQA (Apache-2.0)       medical-school questions with explanations; biology subjects
           SciQ (CC BY-NC 3.0)        science questions with a supporting paragraph
           SmolTalk (see its card)    assistant conversations that match the keyword list
           UltraFeedback (MIT)        rated answer pairs that match the keyword list (DPO)
           MMLU (MIT), PubMedQA expert-labelled set: evaluation only
  run    SFT from a pre-trained checkpoint, DPO from the SFT model, then `eval` of
         all three (base, SFT, DPO).
  eval   for any checkpoints: multiple-choice accuracy on the held-out questions
         against its shuffled-question floor (as in the fact benchmark), the fact
         benchmark itself, and answers to fixed example questions; written to
         <data-dir>\posttrain_report.md.

Examples
--------
    python posttrain.py data
    python posttrain.py run --base tinyGPT_best.pt --name tinyGPT
    python posttrain.py eval tinyGPT_best.pt tinyGPT_sft.pt tinyGPT_dpo.pt

The GPU must be free for `run` and `eval` (one GPU job at a time).
"""

import argparse
import datetime
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_DATA = os.path.join(HERE, "posttrain_data")
LICENCES = {
    "pubmedqa": ("qiaojin/PubMedQA", "MIT"),
    "medmcqa": ("openlifescienceai/medmcqa", "Apache-2.0"),
    "sciq": ("allenai/sciq", "CC BY-NC 3.0 (non-commercial)"),
    "smoltalk": ("HuggingFaceTB/smoltalk", "see the dataset card"),
    "ultrafeedback": ("HuggingFaceH4/ultrafeedback_binarized", "MIT"),
    "mmlu": ("cais/mmlu", "MIT"),
}
MEDMCQA_SUBJECTS = {"Anatomy", "Biochemistry", "Microbiology", "Pathology", "Pharmacology", "Physiology",
                    "Medicine", "Social & Preventive Medicine"}
MMLU_SUBJECTS = ["anatomy", "clinical_knowledge", "college_biology", "college_medicine", "high_school_biology",
                 "medical_genetics", "nutrition", "virology", "high_school_statistics"]
EXAMPLE_QUESTIONS = [
    "What does genetic drift do to allele frequencies in a small population?",
    "What is linkage disequilibrium?",
    "Explain narrow-sense heritability in one or two sentences.",
    "What does a genome-wide association study test at each variant?",
    "Why does inbreeding increase the frequency of recessive disorders?",
    "What is the difference between a p-value and a false discovery rate?",
    "How does natural selection differ from genetic drift?",
    "What is a polygenic score?",
]
LETTERS = "ABCD"


def _text(value):
    """Normalised text: Windows line ends, no trailing spaces, at most one blank line."""
    lines = [line.rstrip() for line in str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out, blank = [], False
    for line in lines:
        if not line.strip():
            if out and not blank:
                out.append("")
            blank = True
            continue
        out.append(line)
        blank = False
    return "\n".join(out).strip()


def _flat(value):
    return " ".join(str(value or "").split())


def _stream(name, config=None, split="train"):
    from datasets import load_dataset
    return load_dataset(name, config, split=split, streaming=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def pubmedqa_sft(cap, exclude_ids):
    for r in _stream("qiaojin/PubMedQA", "pqa_artificial"):
        if cap <= 0:
            return
        if str(r.get("pubid")) in exclude_ids:
            continue
        q, a = _flat(r.get("question")), _text(r.get("long_answer"))
        if len(q) > 15 and len(a) > 40:
            cap -= 1
            yield {"prompt": q, "response": a, "source": "pubmedqa"}


def _medmcqa_index(sample):
    """MedMCQA's 'cop' field: 0-based or 1-based? Decided from the data (the
    explanation names the right option far more often than a wrong one)."""
    score = {0: 0, 1: 0}
    for r in sample:
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        exp = (r.get("exp") or "").lower()
        for base in (0, 1):
            k = int(r["cop"]) - base
            if 0 <= k < 4 and len(opts[k]) > 3 and opts[k].lower() in exp:
                score[base] += 1
    return max(score, key=score.get), score


def medmcqa(cap_sft, cap_dpo, rng):
    rows = []
    for r in _stream("openlifescienceai/medmcqa"):
        if r.get("choice_type") != "single" or r.get("subject_name") not in MEDMCQA_SUBJECTS:
            continue
        rows.append(r)
        if len(rows) >= (cap_sft + cap_dpo) * 2:
            break
    base, score = _medmcqa_index(rows[:2000])
    sft, dpo = [], []
    for r in rows:
        opts = [_flat(r[k]) for k in ("opa", "opb", "opc", "opd")]
        k = int(r["cop"]) - base
        exp = _text(r.get("exp"))
        if not (0 <= k < 4) or not all(opts):
            continue
        prompt = _flat(r["question"]) + "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        if len(sft) < cap_sft and len(exp) >= 40:
            sft.append({"prompt": prompt, "response": f"{LETTERS[k]}. {opts[k]}. {exp[:1500]}", "source": "medmcqa"})
        elif len(dpo) < cap_dpo:
            wrong = rng.choice([i for i in range(4) if i != k])
            dpo.append({"prompt": prompt, "chosen": f"{LETTERS[k]}. {opts[k]}",
                        "rejected": f"{LETTERS[wrong]}. {opts[wrong]}", "source": "medmcqa"})
        if len(sft) >= cap_sft and len(dpo) >= cap_dpo:
            break
    return sft, dpo, {"cop_base": base, "cop_check": score}


def sciq(cap_dpo, rng):
    sft, dpo = [], []
    for r in _stream("allenai/sciq"):
        q, a = _flat(r["question"]), _flat(r["correct_answer"])
        wrong = [_flat(r[f"distractor{i}"]) for i in (1, 2, 3) if _flat(r[f"distractor{i}"])]
        if not q or not a:
            continue
        support = _text(r.get("support"))
        answer = a[0].upper() + a[1:] + "."
        sft.append({"prompt": q, "response": answer + (" " + support if support else ""), "source": "sciq"})
        if len(dpo) < cap_dpo and wrong:
            w = rng.choice(wrong)
            dpo.append({"prompt": q, "chosen": answer, "rejected": w[0].upper() + w[1:] + ".", "source": "sciq"})
    return sft, dpo


def _messages(value):
    if isinstance(value, str):
        value = json.loads(value)
    return value or []


def smoltalk(cap, kw, max_chars=4000, scan_limit=400_000):
    out, seen = [], 0
    for r in _stream("HuggingFaceTB/smoltalk", "smol-magpie-ultra"):
        seen += 1
        if len(out) >= cap or seen > scan_limit:
            break
        msgs = _messages(r.get("messages"))
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        reply = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        if not user or not reply or len(user) + len(reply) > max_chars:
            continue
        if kw.evaluate(None, user + "\n" + reply)[0]:
            out.append({"prompt": _text(user), "response": _text(reply), "source": "smoltalk"})
    return out, seen


def ultrafeedback(cap, kw, max_chars=4000):
    out, seen = [], 0
    for r in _stream("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs"):
        seen += 1
        if len(out) >= cap:
            break
        chosen, rejected = _messages(r.get("chosen")), _messages(r.get("rejected"))
        if len(chosen) != 2 or len(rejected) != 2:
            continue
        prompt, good, bad = r.get("prompt") or "", chosen[-1]["content"], rejected[-1]["content"]
        if (not good or not bad or good == bad or float(r.get("score_chosen", 0)) <= float(r.get("score_rejected", 0))
                or len(prompt) + len(good) + len(bad) > max_chars):
            continue
        if kw.evaluate(None, prompt + "\n" + good)[0]:
            out.append({"prompt": _text(prompt), "chosen": _text(good), "rejected": _text(bad),
                        "source": "ultrafeedback"})
    return out, seen


def eval_questions():
    rows = []
    for subject in MMLU_SUBJECTS:
        for r in _stream("cais/mmlu", subject, "test"):
            choices = [_flat(c) for c in r["choices"]]
            k = int(r["answer"])
            if all(choices) and len(set(choices)) == len(choices):
                rows.append({"category": f"mmlu:{subject}", "prompt": _flat(r["question"]), "answer": choices[k],
                             "distractors": [c for i, c in enumerate(choices) if i != k]})
    labelled_ids = set()
    for r in _stream("qiaojin/PubMedQA", "pqa_labeled"):
        labelled_ids.add(str(r.get("pubid")))
        d = (r.get("final_decision") or "").strip().lower()
        if d in ("yes", "no", "maybe"):
            rows.append({"category": "pubmedqa-labelled", "prompt": _flat(r["question"]), "answer": d,
                         "distractors": [x for x in ("yes", "no", "maybe") if x != d]})
    return rows, labelled_ids


def _write_jsonl(path, rows):
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(path + ".part", path)


def build_data(args):
    import data_prep as dp
    os.makedirs(args.data_dir, exist_ok=True)
    rng = random.Random(args.seed)
    kw = dp.KeywordFilter(dp.read_terms(args.keywords), [], min_hits=3, min_distinct=3)
    t0 = time.time()
    print("evaluation questions (MMLU biology, medicine, genetics and statistics; PubMedQA expert-labelled) ...",
          flush=True)
    evals, labelled_ids = eval_questions()
    print(f"  {len(evals):,} questions ({time.time() - t0:,.0f} s)\nPubMedQA ...", flush=True)
    sft = list(pubmedqa_sft(args.pubmedqa, labelled_ids))
    print(f"  {len(sft):,} pairs ({time.time() - t0:,.0f} s)\nMedMCQA ...", flush=True)
    m_sft, m_dpo, m_info = medmcqa(args.medmcqa, args.mcq_pairs, rng)
    print(f"  {len(m_sft):,} pairs, {len(m_dpo):,} preference pairs; answer index is "
          f"{m_info['cop_base']}-based (explanation check {m_info['cop_check']}) ({time.time() - t0:,.0f} s)\nSciQ ...",
          flush=True)
    s_sft, s_dpo = sciq(args.mcq_pairs, rng)
    print(f"  {len(s_sft):,} pairs, {len(s_dpo):,} preference pairs ({time.time() - t0:,.0f} s)\n"
          "SmolTalk (keyword-matched) ...", flush=True)
    t_sft, t_seen = smoltalk(args.smoltalk, kw)
    print(f"  {len(t_sft):,} of {t_seen:,} conversations ({time.time() - t0:,.0f} s)\n"
          "UltraFeedback (keyword-matched) ...", flush=True)
    u_dpo, u_seen = ultrafeedback(args.ultrafeedback, kw)
    print(f"  {len(u_dpo):,} of {u_seen:,} pairs ({time.time() - t0:,.0f} s)", flush=True)
    eval_prompts = {r["prompt"].lower() for r in evals}
    sft_all = [r for r in sft + m_sft + s_sft + t_sft if r["prompt"].split("\n")[0].lower() not in eval_prompts]
    dpo_all = [r for r in m_dpo + s_dpo + u_dpo if r["prompt"].split("\n")[0].lower() not in eval_prompts]
    rng.shuffle(sft_all)
    rng.shuffle(dpo_all)
    _write_jsonl(os.path.join(args.data_dir, "sft.jsonl"), sft_all)
    _write_jsonl(os.path.join(args.data_dir, "dpo.jsonl"), dpo_all)
    _write_jsonl(os.path.join(args.data_dir, "eval_mcq.jsonl"), evals)
    count = lambda rows, key: sum(1 for r in rows if r.get("source") == key)  # noqa: E731
    lines = ["# Post-training data", "", f"Built {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} "
             f"in {time.time() - t0:,.0f} s. Keyword filter: {os.path.basename(args.keywords)} (3 matches from 3 "
             "different terms) for SmolTalk and UltraFeedback.", "",
             "| Source | Dataset | Licence | SFT pairs | DPO pairs |", "|---|---|---|---|---|"]
    for key in ("pubmedqa", "medmcqa", "sciq", "smoltalk", "ultrafeedback"):
        name, lic = LICENCES[key]
        lines.append(f"| {key} | {name} | {lic} | {count(sft_all, key):,} | {count(dpo_all, key):,} |")
    lines += ["", f"Evaluation only ({len(evals):,} questions, never trained on): MMLU test ({LICENCES['mmlu'][1]}; "
              + ", ".join(MMLU_SUBJECTS) + ") and PubMedQA's expert-labelled set (MIT). Training questions that "
              "match an evaluation question are removed.", "",
              "Check the licences before sharing a model trained on this data (SciQ is non-commercial)."]
    with open(os.path.join(args.data_dir, "sources.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\n{len(sft_all):,} SFT pairs, {len(dpo_all):,} DPO pairs, {len(evals):,} evaluation questions in "
          f"{args.data_dir} ({time.time() - t0:,.0f} s); see sources.md")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def evaluate(checkpoints, args):
    import torch
    import finetune
    import tiny_gpt
    device = tiny_gpt.select_device(args.device)
    amp_dtype, amp_name, _ = tiny_gpt.choose_precision(device, "auto")
    evals = [json.loads(line) for line in open(os.path.join(args.data_dir, "eval_mcq.jsonl"), encoding="utf-8")]
    mcq = [{"category": r["category"].split(":")[0], "difficulty": "unrated",
            "prompt": finetune.TEMPLATE.format(prompt=r["prompt"]), "answer": r["answer"],
            "distractors": r["distractors"]} for r in evals]
    facts = tiny_gpt.read_probes(args.probes) if os.path.isfile(args.probes) else []
    rows, answers = [], {}
    for path in checkpoints:
        t0 = time.time()
        model, tok, cfg, _ = tiny_gpt.load_for_inference(path, device, args.trust_checkpoint)
        model.eval()
        acc, results = tiny_gpt.score_probes(model, tok, mcq, device, amp_dtype)
        floor = tiny_gpt.probe_floor(results)
        by_cat = {k: v["accuracy"] for k, v in tiny_gpt.probe_summary(results).items()}
        fact_acc = fact_floor = None
        if facts:
            fact_acc, fact_results = tiny_gpt.score_probes(model, tok, facts, device, amp_dtype)
            fact_floor = tiny_gpt.probe_floor(fact_results)
        answers[path] = [finetune.answer_question(model, tok, q, device, amp_dtype, args.max_new_tokens)
                         for q in EXAMPLE_QUESTIONS]
        rows.append((path, acc, floor, by_cat, fact_acc, fact_floor))
        print(f"{os.path.basename(path)}: multiple choice {acc:.3f} (floor {floor:.3f})"
              + (f" | fact benchmark {fact_acc:.3f} (floor {fact_floor:.3f})" if facts else "")
              + f" | {time.time() - t0:,.0f} s", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cats = sorted({c for r in rows for c in r[3]})
    lines = ["# Post-training report", "",
             f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')} on {tiny_gpt.device_label(device)} "
             f"({amp_name}). Multiple choice: {len(mcq):,} held-out questions, scored like the fact benchmark "
             "(the answer's log-probability per character against the wrong options; the floor is the accuracy "
             "with the question's words shuffled). Chance is 25% for four options, 33% for PubMedQA.", "",
             "| Checkpoint | Multiple choice | Floor | " + " | ".join(cats) + " | Fact benchmark (floor) |",
             "|---" * (4 + len(cats)) + "|"]
    for path, acc, floor, by_cat, fact_acc, fact_floor in rows:
        lines.append(f"| {os.path.basename(path)} | {acc:.3f} | {floor:.3f} | "
                     + " | ".join(f"{by_cat.get(c, float('nan')):.3f}" for c in cats) + " | "
                     + (f"{fact_acc:.3f} ({fact_floor:.3f})" if fact_acc is not None else "-") + " |")
    lines += ["", "## Example answers", ""]
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        lines.append(f"**{q}**\n")
        for path in checkpoints:
            lines.append(f"- *{os.path.basename(path)}*: {_flat(answers[path][i])}")
        lines.append("")
    report = os.path.join(args.data_dir, "posttrain_report.md")
    with open(report, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"report: {report}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def gpu_busy():
    try:
        import run_pipeline
        return run_pipeline.gpu_busy()
    except Exception:  # noqa: BLE001
        return False


def run(args):
    import finetune
    for f in ("sft.jsonl", "dpo.jsonl", "eval_mcq.jsonl"):
        if not os.path.isfile(os.path.join(args.data_dir, f)):
            raise SystemExit(f"{os.path.join(args.data_dir, f)} is missing: run `python posttrain.py data` first.")
    if args.device != "cpu" and gpu_busy():
        raise SystemExit("The GPU is busy (another training run?). Two GPU jobs at once can crash the graphics "
                         "driver; wait for the other job or stop it first.")
    common = ["--out-dir", args.out_dir, "--device", args.device] + (["--overwrite"] if args.overwrite else []) \
        + (["--trust-checkpoint"] if args.trust_checkpoint else [])
    sft_path = os.path.join(args.out_dir, args.name + "_sft.pt")
    dpo_path = os.path.join(args.out_dir, args.name + "_dpo.pt")
    finetune.main(["sft", "--base", args.base, "--data", os.path.join(args.data_dir, "sft.jsonl"),
                   "--name", args.name + "_sft", "--epochs", str(args.sft_epochs), "--batch", str(args.batch),
                   "--val-fraction", "0.02"] + common)
    finetune.main(["dpo", "--base", sft_path, "--data", os.path.join(args.data_dir, "dpo.jsonl"),
                   "--name", args.name + "_dpo", "--epochs", str(args.dpo_epochs), "--batch", str(args.batch),
                   "--val-fraction", "0.02"] + common)
    evaluate([args.base, sft_path, dpo_path], args)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data", help="download and convert the post-training data")
    d.add_argument("--keywords", default=os.path.join(HERE, "keywords_biology.txt"))
    d.add_argument("--pubmedqa", type=int, default=40_000, help="PubMedQA question-answer pairs")
    d.add_argument("--medmcqa", type=int, default=30_000, help="MedMCQA questions with explanations")
    d.add_argument("--mcq-pairs", type=int, default=10_000, help="preference pairs from MedMCQA and from SciQ each")
    d.add_argument("--smoltalk", type=int, default=20_000, help="keyword-matched SmolTalk conversations")
    d.add_argument("--ultrafeedback", type=int, default=10_000, help="keyword-matched UltraFeedback pairs")
    r = sub.add_parser("run", help="SFT, then DPO, then evaluation")
    r.add_argument("--base", required=True, help="pre-trained checkpoint, e.g. tinyGPT_best.pt")
    r.add_argument("--name", required=True, help="outputs <name>_sft.pt and <name>_dpo.pt")
    r.add_argument("--out-dir", default=HERE)
    r.add_argument("--sft-epochs", type=int, default=2)
    r.add_argument("--dpo-epochs", type=int, default=1)
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--overwrite", action="store_true")
    e = sub.add_parser("eval", help="evaluate checkpoints")
    e.add_argument("checkpoints", nargs="+")
    for s in (d, r, e):
        s.add_argument("--data-dir", default=DEFAULT_DATA)
        s.add_argument("--seed", type=int, default=0)
    for s in (r, e):
        s.add_argument("--probes", default=os.path.join(HERE, "probes_biology.tsv"))
        s.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
        s.add_argument("--max-new-tokens", type=int, default=120)
        s.add_argument("--trust-checkpoint", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_args(argv)
    args.data_dir = os.path.abspath(args.data_dir)
    if args.cmd == "data":
        build_data(args)
    elif args.cmd == "run":
        run(args)
    else:
        evaluate(args.checkpoints, args)


if __name__ == "__main__":
    main()
