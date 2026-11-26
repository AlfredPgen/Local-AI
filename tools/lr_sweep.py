"""Learning-rate sweep for SFT: short fine-tunes at several peak learning rates on
the same examples, compared on held-out loss, held-out multiple choice and the
fact benchmark.

    python tools/lr_sweep.py --base tinyGPT_muon_best.pt --lrs 1e-5,2e-5,4e-5,7e-5,1e-4

1. Data: --train examples and --val held-out examples drawn (seed 0) from
   posttrain_data/sft.jsonl into one file; every run uses that file with the same
   seed, so all runs see the same examples in the same order and share the held-out set.
2. Each learning rate: finetune.py sft, 1 epoch, the same schedule shape as the full
   run (cosine, 10 % warm-up, floor 0.1 x peak), held-out loss read at the end.
3. Each resulting model, and the base once: accuracy on posttrain_data/eval_mcq.jsonl
   (2,815 held-out questions, never trained on; answer template as posttrain.py
   eval) and on the fact probes. No shuffled-question floor (the same for all runs).
4. Writes <out>/sweep.json and <out>/sweep.md. The swept checkpoints are deleted
   after scoring (rerunning the sweep recreates them); --keep keeps them.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def score(path, device, mcq, facts):
    import finetune
    import tiny_gpt
    amp_dtype, _, _ = tiny_gpt.choose_precision(device, "auto", training=False)
    model, tok, _, _ = tiny_gpt.load_for_inference(path, device, False)
    model.eval()
    mcq_acc, _ = tiny_gpt.score_probes(model, tok, mcq, device, amp_dtype, floor=False, template=finetune.TEMPLATE)
    fact_acc, _ = tiny_gpt.score_probes(model, tok, facts, device, amp_dtype, floor=False)
    del model
    tiny_gpt.free_device_memory(device)
    return mcq_acc, fact_acc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--lrs", default="1e-5,2e-5,4e-5,7e-5,1e-4")
    p.add_argument("--train", type=int, default=12800)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--data-dir", default=os.path.join(HERE, "posttrain_data"))
    p.add_argument("--out", default=os.path.join(HERE, "logs", "lr_sweep_" + time.strftime("%Y-%m-%d")))
    p.add_argument("--keep", action="store_true", help="keep the swept checkpoints")
    args = p.parse_args()
    lrs = [float(x) for x in args.lrs.split(",")]
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.data_dir, "sft.jsonl"), encoding="utf-8") as f:
        rows = f.readlines()
    random.Random(0).shuffle(rows)
    n = args.train + args.val
    data = os.path.join(args.out, "sweep_data.jsonl")
    with open(data, "w", encoding="utf-8") as f:
        f.writelines(rows[:n])
    frac = args.val / n  # finetune.py holds out round(frac * n) = --val examples

    import torch
    import tiny_gpt
    device = tiny_gpt.select_device("auto")
    with open(os.path.join(args.data_dir, "eval_mcq.jsonl"), encoding="utf-8") as f:
        evals = [json.loads(line) for line in f]
    mcq = [{"category": r["category"].split(":")[0], "difficulty": "unrated", "prompt": r["prompt"],
            "answer": r["answer"], "distractors": r["distractors"]} for r in evals]
    facts = tiny_gpt.read_probes(tiny_gpt.DEFAULT_PROBES)

    results_path = os.path.join(args.out, "sweep.json")
    results = json.load(open(results_path, encoding="utf-8")) if os.path.exists(results_path) else {}

    def save_results():
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)

    if "base" not in results:
        t0 = time.time()
        m, fa = score(args.base, device, mcq, facts)
        results["base"] = {"lr": 0.0, "val_loss_start": None, "val_loss": None, "mcq": m, "facts": fa}
        save_results()
        print(f"base: multiple choice {m:.3f} | facts {fa:.3f} | {time.time() - t0:,.0f} s", flush=True)

    for lr in lrs:
        key = f"{lr:g}"
        if key in results:
            print(f"lr {key}: done before, skipped", flush=True)
            continue
        name = f"sweep_lr{key}"
        log = os.path.join(args.out, name + ".log")
        cmd = [sys.executable, os.path.join(HERE, "finetune.py"), "sft", "--base", args.base, "--data", data,
               "--name", name, "--epochs", "1", "--batch", str(args.batch), "--val-fraction", f"{frac:.6f}",
               "--lr", key, "--eval-every", "0", "--out-dir", args.out, "--log-file", log, "--force",
               "--overwrite", "--seed", "0"]
        print(f"lr {key}: fine-tuning ...", flush=True)
        t0 = time.time()
        # the child's lines are shown as they come (progress) and kept (the losses are read from them)
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        kept = []
        for line in proc.stdout:
            kept.append(line)
            if not line.startswith(("held-out question", "answer:")):
                print("    " + line.rstrip(), flush=True)
        code = proc.wait()
        text = "".join(kept)
        if code != 0:
            raise SystemExit(f"lr {key}: finetune.py failed (code {code})")
        start = re.search(r"epoch 0 \| held-out loss ([0-9.]+)", text)
        end = re.search(r"epoch 1 \| train loss ([0-9.]+) \| held-out loss ([0-9.]+)", text)
        ckpt = os.path.join(args.out, name + ".pt")
        m, fa = score(ckpt, device, mcq, facts)
        results[key] = {"lr": lr, "val_loss_start": float(start.group(1)), "train_loss": float(end.group(1)),
                        "val_loss": float(end.group(2)), "mcq": m, "facts": fa,
                        "minutes": round((time.time() - t0) / 60, 1)}
        save_results()
        print(f"lr {key}: held-out loss {start.group(1)} -> {end.group(2)} | multiple choice {m:.3f} | "
              f"facts {fa:.3f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        if not args.keep:
            for f in os.listdir(args.out):
                if f.startswith(name) and f.endswith(".pt"):
                    os.remove(os.path.join(args.out, f))
        torch.cuda.empty_cache() if device.type == "cuda" else None

    n_mcq, n_facts = len(mcq), len(facts)
    lines = ["# SFT learning-rate sweep", "",
             f"Base {os.path.basename(args.base)}; {args.train:,} training and {args.val:,} held-out examples from "
             f"sft.jsonl (seed 0), batch {args.batch}, 1 epoch ({args.train // args.batch} steps), cosine schedule "
             f"with 10 % warm-up. Multiple choice: {n_mcq:,} held-out questions (eval_mcq.jsonl); facts: "
             f"{n_facts} probes. One 95 % interval is about +-{1.96 * (0.25 / n_mcq) ** 0.5:.3f} (multiple "
             f"choice) and +-{1.96 * (0.25 / n_facts) ** 0.5:.3f} (facts) for a single accuracy near 50 %.", "",
             "| peak lr | held-out loss (start -> end) | multiple choice | facts |", "|---|---|---|---|"]
    for key, r in sorted(results.items(), key=lambda kv: kv[1]["lr"]):
        loss = "-" if r["val_loss"] is None else f"{r['val_loss_start']:.4f} -> **{r['val_loss']:.4f}**"
        lines.append(f"| {'base (no SFT)' if key == 'base' else key} | {loss} | {r['mcq']:.3f} | {r['facts']:.3f} |")
    with open(os.path.join(args.out, "sweep.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
