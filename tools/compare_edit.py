r"""Original vs edited checkpoint (from tools/edit_weights.py), on prompts the edit
never saw: repetition loops, validation bits per byte and the fact benchmark,
with a picture.

    python tools/compare_edit.py tinyGPT_muon_best.pt tinyGPT_muon_best_noloop.pt

1. Loops: both models continue the same held-out prompts (the odd-numbered
   validation documents; edit_weights.py used the even ones) with the same random
   seeds; per text, the share of generated tokens that repeat an earlier 4-gram.
   A paired bootstrap over prompts gives a 95% interval for the change.
2. Bits per byte on validation documents, per source and overall: paired over
   documents (compare_models.py's method), 95% interval.
3. Fact benchmark (probes_biology.tsv): accuracy of each model and a paired
   bootstrap interval over questions.
Writes <edited>_vs_original.png and .json next to the edited checkpoint, and a few
sample continuations side by side in a .md file.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_models as cm  # noqa: E402
import edit_weights as ew  # noqa: E402
import tiny_gpt as tg  # noqa: E402


def paired_ci(a, b, reps=2000, seed=0):
    """mean(b - a) with a percentile 95% interval from resampling pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (reps, len(a)))
    diffs = (b[idx] - a[idx]).mean(1)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float((b - a).mean()), float(low), float(high)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("original")
    p.add_argument("edited")
    p.add_argument("--prompts", type=int, default=256, help="held-out prompts for the loop test")
    p.add_argument("--prompt-tokens", type=int, default=32)
    p.add_argument("--new-tokens", type=int, default=160)
    p.add_argument("--docs", type=int, default=40, help="validation documents per source for bits per byte")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true", help="use the GPU even if another job seems to be using it")
    p.add_argument("--dataset", default=None,
                   help="dataset for the held-out prompts and documents (default: the original checkpoint's, found "
                        "by its stored path or by name in datasets/ with the same fingerprint)")
    args = p.parse_args(argv)

    device = tg.select_device(args.device)
    if device.type != "cpu" and not args.force and tg.gpu_busy(HERE):
        raise SystemExit(tg.GPU_BUSY_MESSAGE + " (Or use --device cpu, or pass --force.)")
    amp, amp_name, _ = tg.choose_precision(device, "auto", training=False)
    probes = tg.read_probes(tg.DEFAULT_PROBES)
    res = {}
    seqs, models_info = {}, {}
    docs = None
    for label, path in (("original", args.original), ("edited", args.edited)):
        model, tok, cfg, obj = tg.load_for_inference(path, device)
        models_info[label] = {"path": tg._home_short(os.path.abspath(path)), "step": obj.get("step"),
                              "edit": ((obj.get("edit") or {}).get("chosen") or {})}
        room = model.ctx - args.prompt_tokens - 1  # prompt (+ start token) and continuation must fit the context
        if args.new_tokens > room:
            print(f"Context is {model.ctx} tokens: generating {room} new tokens instead of {args.new_tokens}.")
            args.new_tokens = max(room, 1)
        if docs is None:
            override = args.dataset and (args.dataset if os.path.isdir(args.dataset)
                                         else os.path.join(HERE, args.dataset))
            dataset = tg.resolve_dataset(obj, override)
            prompts = ew.prompts_from_validation(dataset, tok, args.prompts, args.prompt_tokens, "test", args.seed)
            start = len(prompts[0])
            docs = cm.validation_documents(dataset, args.docs, 20_000)
        print(f"{label}: {path} ({tg.device_label(device)}, {amp_name}) - generating", flush=True)
        seqs[label] = ew.generate_all(model, prompts, args.new_tokens, device, amp, args.batch, args.seed)
        print(f"{label}: bits per byte on {len(docs)} validation documents", flush=True)
        nll, nbytes = [], []
        for _, _, text in docs:
            n, b = cm.document_nll(model, tok, text, device, amp)
            nll.append(n)
            nbytes.append(b)
        print(f"{label}: fact benchmark ({len(probes)} questions)", flush=True)
        acc, results = tg.score_probes(model, tok, probes, device, amp, floor=False)
        res[label] = {"rates": ew.per_text_rates(seqs[label], start, tok.eos_id), "nll": np.array(nll),
                      "bytes": np.array(nbytes), "probe": np.array([r["correct"] for r in results], float),
                      "probe_acc": acc}
        del model
        tg.free_device_memory(device)
    o, e = res["original"], res["edited"]

    loop_diff = paired_ci(o["rates"], e["rates"])
    looping = (float((o["rates"] > 0.25).mean()), float((e["rates"] > 0.25).mean()))
    loop_flag_diff = paired_ci((o["rates"] > 0.25).astype(float), (e["rates"] > 0.25).astype(float))
    ln2 = math.log(2)
    bpb = {k: float(r["nll"].sum() / (ln2 * r["bytes"].sum())) for k, r in res.items()}
    point, low, high, pval = cm.bootstrap(e["nll"], o["nll"], e["bytes"], o["bytes"], 2000, 0)
    sources = sorted({s for s, _, _ in docs})
    by_source = {}
    for s in sources:
        idx = [i for i, d in enumerate(docs) if d[0] == s]
        pt, lo, hi, _ = cm.bootstrap(e["nll"][idx], o["nll"][idx], e["bytes"][idx], o["bytes"][idx], 2000, 0)
        by_source[s] = (pt, lo, hi)
    probe_diff = paired_ci(o["probe"], e["probe"])
    summary = {
        "models": models_info,
        "loop_prompts": len(o["rates"]), "new_tokens": args.new_tokens,
        "repeat_token_share": {"original": float(o["rates"].mean()), "edited": float(e["rates"].mean()),
                               "change": loop_diff},
        "looping_texts": {"original": looping[0], "edited": looping[1], "change": loop_flag_diff},
        "bits_per_byte": {"original": bpb["original"], "edited": bpb["edited"],
                          "change": [point, low, high], "p": pval, "documents": len(docs), "by_source": by_source},
        "fact_benchmark": {"original": o["probe_acc"], "edited": e["probe_acc"], "change": probe_diff,
                           "questions": len(o["probe"])},
    }
    base = os.path.splitext(os.path.abspath(args.edited))[0] + "_vs_original"
    with open(base + ".json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)

    # side-by-side samples: the prompts where the original looped most (tok from the loop: an edited copy keeps
    # the original's tokenizer)
    order = np.argsort(-o["rates"])[:6]
    with open(base + "_samples.md", "w", encoding="utf-8") as handle:
        handle.write("# Same prompt, same seed: original vs edited\n\n")
        for i in order:
            prompt = tok.decode([t for t in seqs["original"][i][:start].tolist() if not tok.is_special(t)])
            handle.write(f"## Prompt {i}\n\n> {prompt}\n\n")
            for label in ("original", "edited"):
                cont = tok.decode([t for t in seqs[label][i][start:].tolist() if not tok.is_special(t)])
                handle.write(f"**{label}** (repeated 4-grams {res[label]['rates'][i]:.0%}):\n\n{cont}\n\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, grey = "#3b6ea8", "#d9822b", "#888888"
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), dpi=130)
    ed = models_info["edited"]["edit"]
    what = (f"{len(ed.get('neurons', []))} neuron(s) zeroed" if ed.get("kind") == "neurons"
            else f"{ed.get('alpha', 0):.2f} x loop direction removed")
    fig.suptitle(f"Edited (layer {ed.get('layer')}: {what}) vs original, "
                 f"step {models_info['original']['step']:,} - held-out prompts and validation text", fontsize=12)
    a = ax[0, 0]
    bins = np.linspace(0, max(o["rates"].max(), e["rates"].max(), 0.05), 30)
    a.hist(o["rates"], bins, alpha=0.6, color=grey, label=f"original (mean {o['rates'].mean():.1%})")
    a.hist(e["rates"], bins, alpha=0.6, color=blue, label=f"edited (mean {e['rates'].mean():.1%})")
    a.axvline(0.25, color="black", lw=0.8, ls=":")
    a.set(title=f"Repetition per generated text ({len(o['rates'])} prompts, {args.new_tokens} tokens)",
          xlabel="share of tokens repeating an earlier 4-gram", ylabel="texts")
    a.legend(frameon=False)
    a = ax[0, 1]
    names = ["repeated tokens", "looping texts (>25%)"]
    vals_o = [o["rates"].mean(), looping[0]]
    vals_e = [e["rates"].mean(), looping[1]]
    errs = [loop_diff, loop_flag_diff]
    x = np.arange(2)
    a.bar(x - 0.18, vals_o, 0.36, color=grey, label="original")
    a.bar(x + 0.18, vals_e, 0.36, color=blue, label="edited",
          yerr=[[v - (vo + d[1]) for v, vo, d in zip(vals_e, vals_o, errs)],
                [(vo + d[2]) - v for v, vo, d in zip(vals_e, vals_o, errs)]], capsize=4)
    for xi, vo, ve, d in zip(x, vals_o, vals_e, errs):
        a.text(xi, max(vo, ve) * 1.05 + 0.002, f"{d[0]:+.1%}\n[{d[1]:+.1%}, {d[2]:+.1%}]", ha="center", fontsize=8)
    a.set_xticks(x, names)
    a.set(title="Loops: edited minus original, 95% interval", ylabel="share")
    a.legend(frameon=False)
    a = ax[1, 0]
    ys = np.arange(len(sources) + 1)
    pts = [by_source[s] for s in sources] + [(point, low, high)]
    labels = sources + ["ALL"]
    for yi, (pt, lo, hi) in zip(ys, pts):
        colour = blue if hi < 0 else orange if lo > 0 else grey
        a.errorbar(pt, yi, xerr=[[pt - lo], [hi - pt]], fmt="o", color=colour, capsize=3)
    a.axvline(0, color="black", lw=0.8)
    a.set_yticks(ys, labels)
    a.set(title=f"Bits per byte, edited minus original (left = better); overall {bpb['original']:.4f} -> "
                f"{bpb['edited']:.4f}", xlabel="difference in bits per byte (95% interval)")
    a = ax[1, 1]
    a.bar([0, 1], [o["probe_acc"], e["probe_acc"]], color=[grey, blue], width=0.5)
    a.set_xticks([0, 1], ["original", "edited"])
    a.text(0.5, max(o["probe_acc"], e["probe_acc"]) * 1.03,
           f"{probe_diff[0]:+.1%} [{probe_diff[1]:+.1%}, {probe_diff[2]:+.1%}]", ha="center")
    a.set(title=f"Fact benchmark accuracy ({len(o['probe'])} questions)", ylabel="accuracy",
          ylim=(0, max(o["probe_acc"], e["probe_acc"]) * 1.2))
    fig.tight_layout()
    fig.savefig(base + ".png")
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=1, default=float))
    print(f"Picture: {base}.png | numbers: {base}.json | samples: {base}_samples.md")


if __name__ == "__main__":
    main()
