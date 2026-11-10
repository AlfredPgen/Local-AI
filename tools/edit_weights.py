r"""Look for weights in tinyGPT that drive repetition loops, edit them, and save the
result as a NEW checkpoint. The input checkpoint is only read, never written.

    python tools/edit_weights.py tinyGPT_muon_best.pt tinyGPT_muon_best_noloop.pt

1. Prompts: the first --prompt-tokens tokens of validation documents (the even-
   numbered ones; compare_edit.py tests on the odd-numbered ones, never seen here).
2. The model continues each prompt (sampling as tiny_gpt.py --generate does).
3. Every generated position is labelled "loop" if the 4 tokens ending at the next
   token already appeared earlier in the text, else "fresh". Loops happen late in a
   text, so loop and fresh positions are compared within bands of 16 positions
   (otherwise "loop" would partly mean "late in the text").
4. Two kinds of edit are searched, per layer:
   - neurons: the MLP neurons most active at loop positions (effect size); their
     output columns in MLP "down" are zeroed (k = 1, 2, 4, 8, 16 neurons);
   - direction: r = mean(hidden | loop) - mean(hidden | fresh) at that layer's output;
     a fraction alpha of r is projected out of that layer's two residual writers
     (attention o_proj and MLP down: W <- W - alpha r r^T W).
5. Every candidate generates again on separate prompts ("choice" prompts) and is
   scored on loop rate and on bits per byte of a few validation documents; a
   candidate that raises bits per byte by more than --max-bpb-loss is rejected. A
   random edit of the same size is the control.
6. The best candidate (lowest loop rate) is saved: a copy of the input checkpoint
   with the edited weights and an "edit" record; a JSON report goes next to it. If no
   candidate beats the original, nothing is saved and the report says so.
"""
import argparse
import datetime
import json
import math
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import compare_models as cm  # noqa: E402
import tiny_gpt as tg  # noqa: E402

NGRAM = 4
BAND = 16


def prompts_from_validation(dataset, tok, n, prompt_tokens, split, seed):
    """Token tuples: the first prompt_tokens tokens of validation documents (even
    indices for split 'find', odd for 'test'), mixed across sources, seeded."""
    docs = cm.validation_documents(dataset, per_source=max(8, n), max_chars=4000)
    docs = [d for i, d in enumerate(docs) if i % 2 == (0 if split == "find" else 1)]
    random.Random(seed).shuffle(docs)
    out = []
    for source, k, text in docs:
        ids = tok.encode(text)
        if len(ids) >= prompt_tokens:
            out.append(((tok.bos_id,) if tok.bos_id >= 0 else ()) + tuple(ids[:prompt_tokens]))
        if len(out) == n:
            break
    return out


def sample_next(logits, generator, temperature, top_k, top_p):
    logits = logits.float() / max(temperature, 1e-6)
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(-1)
    if 0 < top_p < 1:
        sp, order = probs.sort(dim=-1, descending=True)
        sp = sp.masked_fill(sp.cumsum(-1) - sp > top_p, 0)
        probs = torch.zeros_like(probs).scatter(-1, order, sp)
        probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate_batch(model, prompts, new_tokens, device, amp, seed, temperature=0.8, top_k=50, top_p=0.95):
    """Continuations of equal-length prompts; (B, P + new) tokens on CPU. KV-cached:
    the prompts are processed once, then one token per step (P + new <= context)."""
    x = torch.tensor(prompts, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    cache, new = model.new_cache(), x
    for _ in range(new_tokens):
        with torch.autocast(device.type, dtype=amp or torch.float32, enabled=amp is not None):
            logits = model.lm_head(model(new, return_hidden=True, cache=cache)[:, -1])
        new = sample_next(logits, gen, temperature, top_k, top_p).unsqueeze(1)
        x = torch.cat([x, new], dim=1)
    return x.cpu()


def generate_all(model, prompts, new_tokens, device, amp, batch, seed):
    return torch.cat([generate_batch(model, prompts[b:b + batch], new_tokens, device, amp, seed + b)
                      for b in range(0, len(prompts), batch)])


def loop_labels(seq, start, eos):
    """{position p: True if the 4-gram ending at token p+1 occurred earlier in seq}, for
    the generated part; positions after an end-of-text token are dropped."""
    seen, labels = set(), {}
    toks = seq.tolist()
    for i in range(NGRAM - 1, len(toks)):
        gram = tuple(toks[i - NGRAM + 1:i + 1])
        if i >= start:
            labels[i - 1] = gram in seen
        seen.add(gram)
    if eos >= 0 and eos in toks[start:]:
        cut = toks.index(eos, start)
        labels = {p: v for p, v in labels.items() if p + 1 < cut}
    return labels


def per_text_rates(seqs, start, eos, keep_empty=True):
    """Per text, the share of generated tokens repeating an earlier 4-gram. A text with
    no generated token before its end-of-text token counts as 0.0 (keep_empty, so the
    array stays aligned with seqs) or is left out."""
    out = []
    for s in seqs:
        lab = loop_labels(s, start, eos)
        if lab or keep_empty:
            out.append(sum(lab.values()) / len(lab) if lab else 0.0)
    return np.array(out)


def loop_stats(seqs, start, eos):
    """Mean share of generated tokens repeating an earlier 4-gram, and share of texts
    where more than a quarter do ('looping' texts)."""
    rates = per_text_rates(seqs, start, eos, keep_empty=False)
    if not len(rates):
        rates = np.array([0.0])
    return {"repeat_token_share": float(rates.mean()), "looping_texts": float((rates > 0.25).mean()),
            "texts": len(rates)}


class BandStats:
    """Per layer and position band: sums and sums of squares of a feature vector at
    loop and at fresh positions, for a band-matched difference of means.

    The running sums are float64 on the CPU (Apple's MPS has no float64, and the
    variance sum(x^2)/n - mean^2 needs it over hundreds of batches); each batch is
    reduced on the compute device in float32 into (n_bands, dim) sums, and only
    those n_bands x dim numbers cross to the CPU per layer and batch (a batch adds
    at most batch x 16 values per band, well within float32)."""

    def __init__(self, n_layers, n_bands, dim):
        z = lambda: torch.zeros(n_layers, n_bands, dim, dtype=torch.float64)  # noqa: E731
        self.s = [z(), z()]
        self.q = [z(), z()]
        self.n = torch.zeros(2, n_layers, n_bands, dtype=torch.float64)
        self.n_bands = n_bands

    def add(self, li, feats, labels, bands):
        for v in (0, 1):
            sel = labels == v
            if sel.any():
                f, b = feats[sel].float(), bands[sel]
                s = torch.zeros(self.n_bands, f.shape[1], dtype=torch.float32, device=f.device)
                q = torch.zeros_like(s)
                s.index_add_(0, b, f)
                q.index_add_(0, b, f * f)
                self.s[v][li] += s.cpu().double()
                self.q[v][li] += q.cpu().double()
                self.n[v, li] += torch.bincount(b, minlength=self.n_bands).cpu().double()

    def effect(self, li):
        """(band-matched mean difference loop - fresh, effect size per feature)."""
        n1, n0 = self.n[1, li], self.n[0, li]
        ok = (n1 >= 20) & (n0 >= 20)
        w = torch.where(ok, torch.minimum(n1, n0), torch.zeros_like(n1))
        w = w / w.sum().clamp(min=1)
        mu1 = self.s[1][li] / n1.clamp(min=1)[:, None]
        mu0 = self.s[0][li] / n0.clamp(min=1)[:, None]
        diff = (w[:, None] * (mu1 - mu0)).sum(0)
        tot = n1.sum() + n0.sum()
        mean = (self.s[1][li].sum(0) + self.s[0][li].sum(0)) / tot
        var = (self.q[1][li].sum(0) + self.q[0][li].sum(0)) / tot - mean * mean
        return diff, diff / var.clamp(min=1e-12).sqrt()


@torch.no_grad()
def collect(model, seqs, start, eos, device, amp, batch):
    """Band-matched loop-vs-fresh statistics of each layer's output (residual stream)
    and of each layer's MLP neurons (the input of MLP down)."""
    n_layers, d, hdim = len(model.blocks), model.cfg.d_model, model.cfg.ffn_hidden
    n_bands = math.ceil(seqs.shape[1] / BAND)
    resid, neur = BandStats(n_layers, n_bands, d), BandStats(n_layers, n_bands, hdim)
    cap_r, cap_n = {}, {}
    hooks = [blk.register_forward_hook(lambda m, i, o, li=li: cap_r.__setitem__(li, o))
             for li, blk in enumerate(model.blocks)]
    hooks += [blk.ffn.down.register_forward_pre_hook(lambda m, i, li=li: cap_n.__setitem__(li, i[0]))
              for li, blk in enumerate(model.blocks)]
    counts = [0, 0]
    try:
        for b in range(0, len(seqs), batch):
            chunk = seqs[b:b + batch]
            lab = torch.full(chunk.shape, -1, dtype=torch.long)
            for r, s in enumerate(chunk):
                for p, v in loop_labels(s, start, eos).items():
                    lab[r, p] = int(v)
            counts[0] += int((lab == 0).sum())
            counts[1] += int((lab == 1).sum())
            with torch.autocast(device.type, dtype=amp or torch.float32, enabled=amp is not None):
                model(chunk.to(device), return_hidden=True)
            keep = lab.reshape(-1) >= 0
            labels = lab.reshape(-1)[keep].to(device)
            bands = (torch.arange(chunk.shape[1]).repeat(chunk.shape[0]) // BAND)[keep].to(device)
            for li in range(n_layers):
                resid.add(li, cap_r[li].reshape(-1, d)[keep.to(device)], labels, bands)
                neur.add(li, cap_n[li].reshape(-1, hdim)[keep.to(device)], labels, bands)
    finally:
        for hk in hooks:
            hk.remove()
    return resid, neur, {"loop_positions": counts[1], "fresh_positions": counts[0]}


def require_dense(model):
    """The edits address the dense SwiGLU's down matrix of a block; a mixture-of-experts
    block has several experts and a router instead, and no single 'loop neuron'."""
    if any(not hasattr(b.ffn, "down") for b in model.blocks):
        raise SystemExit("edit_weights works on dense SwiGLU blocks; this checkpoint has mixture-of-experts blocks")


def apply_edit(model, edit):
    """Change the model in place; returns an undo function."""
    blk = model.blocks[edit["layer"]]
    if edit["kind"] == "neurons":
        w = blk.ffn.down.weight
        cols = torch.tensor(edit["neurons"], device=w.device)
        old = w[:, cols].clone()
        with torch.no_grad():
            w[:, cols] = 0

        def undo():
            with torch.no_grad():
                w[:, cols] = old
        return undo
    r = edit["direction"].to(blk.attn.o_proj.weight.device, torch.float32)
    olds = []
    with torch.no_grad():
        for w in (blk.attn.o_proj.weight, blk.ffn.down.weight):
            olds.append((w, w.clone()))
            w.sub_((edit["alpha"] * torch.outer(r, r @ w.float())).to(w.dtype))

    def undo():
        with torch.no_grad():
            for w, o in olds:
                w.copy_(o)
    return undo


def quick_bpb(model, tok, docs, device, amp=None):
    nll = nbytes = 0.0
    for _, _, text in docs:
        n, b = cm.document_nll(model, tok, text, device, amp=amp)
        nll, nbytes = nll + n, nbytes + b
    return nll / (math.log(2) * max(nbytes, 1))


def edited_metrics(metrics):
    """The metrics an edited copy carries: the history and best values stay (they
    are the original's pre-training), but the stored last evaluation (validation
    loss, fact benchmark, calibration) was measured on the original weights, so it
    becomes base_last_eval and last_eval is None, as in a fine-tuned file; the
    viewer then scores the edited weights themselves."""
    m = dict(metrics or {})
    return {**m, "last_eval": None, "base_last_eval": m.get("last_eval") or m.get("base_last_eval")}


def describe(e):
    if e["kind"] == "neurons":
        return f"layer {e['layer']}: zero {len(e['neurons'])} neuron(s) {e['neurons'][:8]}"
    return f"layer {e['layer']}: remove {e['alpha']:.2f} x loop direction"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="input checkpoint (read only)")
    p.add_argument("out", help="new checkpoint to write (must differ from the input)")
    p.add_argument("--dataset", default=None, help="dataset for validation prompts (default: the checkpoint's)")
    p.add_argument("--prompts", type=int, default=256, help="prompts for finding loop neurons and directions")
    p.add_argument("--dev-prompts", type=int, default=96, help="other prompts for choosing among candidates")
    p.add_argument("--prompt-tokens", type=int, default=32)
    p.add_argument("--new-tokens", type=int, default=160)
    p.add_argument("--layers", type=int, default=4, help="layers with the strongest loop neuron to try edits in")
    p.add_argument("--bpb-docs", type=int, default=3, help="validation documents per source for the safety check")
    p.add_argument("--max-bpb-loss", type=float, default=0.002, help="largest bits-per-byte increase accepted")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true", help="use the GPU even if another job seems to be using it")
    p.add_argument("--overwrite", action="store_true", help="replace an existing OUT (never the input)")
    args = p.parse_args(argv)

    src, dst = os.path.abspath(args.checkpoint), os.path.abspath(args.out)
    if os.path.normcase(src) == os.path.normcase(dst):
        raise SystemExit("OUT must be a new file: the input checkpoint is never overwritten.")
    if os.path.exists(dst) and not args.overwrite:
        raise SystemExit(f"{dst} exists; pass --overwrite to replace it.")
    device = tg.select_device(args.device)
    if device.type != "cpu" and not args.force and tg.gpu_busy(HERE):
        raise SystemExit(tg.GPU_BUSY_MESSAGE + " (Or use --device cpu, or pass --force.)")
    amp, amp_name, _ = tg.choose_precision(device, "auto", training=False)
    model, tok, cfg, obj = tg.load_for_inference(src, device)
    require_dense(model)
    room = model.ctx - args.prompt_tokens - 1  # prompt (+ start token) and continuation must fit the context
    if args.new_tokens > room:
        print(f"Context is {model.ctx} tokens: generating {room} new tokens instead of {args.new_tokens}.")
        args.new_tokens = max(room, 1)
    # --dataset as given or from the project folder; else the checkpoint's own dataset on this machine
    override = args.dataset and (args.dataset if os.path.isdir(args.dataset) else os.path.join(HERE, args.dataset))
    dataset = tg.resolve_dataset(obj, override)
    print(f"{src} | step {obj.get('step')} | {tg.device_label(device)} | {amp_name}", flush=True)

    allp = prompts_from_validation(dataset, tok, args.prompts + args.dev_prompts, args.prompt_tokens, "find",
                                   args.seed)
    find, dev = allp[:args.prompts], allp[args.prompts:]
    start = len(find[0])
    bpb_docs = [d for i, d in enumerate(cm.validation_documents(dataset, 2 * args.bpb_docs, 6000)) if i % 2 == 0]
    print(f"{len(find)} prompts to find candidates, {len(dev)} to choose; {args.new_tokens} new tokens each; "
          f"safety check on {len(bpb_docs)} validation documents", flush=True)

    seqs = generate_all(model, find, args.new_tokens, device, amp, args.batch, args.seed)
    base_find = loop_stats(seqs, start, tok.eos_id)
    print(f"original, find prompts: {base_find}", flush=True)
    resid, neur, counts = collect(model, seqs, start, tok.eos_id, device, amp, args.batch)
    del seqs
    layer_info = []
    for li in range(len(model.blocks)):
        diff, eff = resid.effect(li)
        r = diff / diff.norm()
        _, neff = neur.effect(li)
        top = torch.argsort(neff, descending=True)[:16]
        layer_info.append({"layer": li, "direction": r.float().cpu(), "direction_effect": float(diff.norm()),
                           "neurons": top.tolist(), "neuron_effects": [float(neff[i]) for i in top],
                           "max_neuron_effect": float(neff[top[0]])})
    print("strongest loop neuron per layer (effect size): " +
          ", ".join(f"{d['layer']}:{d['max_neuron_effect']:.2f}" for d in layer_info), flush=True)
    chosen_layers = [d["layer"] for d in sorted(layer_info, key=lambda d: -d["max_neuron_effect"])[:args.layers]]

    base_dev = loop_stats(generate_all(model, dev, args.new_tokens, device, amp, args.batch, args.seed + 7), start,
                          tok.eos_id)
    base_bpb = quick_bpb(model, tok, bpb_docs, device, amp)
    print(f"original, choice prompts: {base_dev}, bits per byte {base_bpb:.4f}", flush=True)

    rng = torch.Generator().manual_seed(args.seed)
    candidates = []
    for li in chosen_layers:
        info = layer_info[li]
        for k in (1, 2, 4, 8, 16):
            candidates.append({"kind": "neurons", "layer": li, "neurons": info["neurons"][:k]})
        for alpha in (0.25, 0.5, 1.0):
            candidates.append({"kind": "direction", "layer": li, "alpha": alpha, "direction": info["direction"]})
    li0 = chosen_layers[0]
    rand_neurons = torch.randperm(cfg.ffn_hidden, generator=rng)[:16].tolist()
    rand_dir = torch.randn(cfg.d_model, generator=rng)
    controls = [{"kind": "neurons", "layer": li0, "neurons": rand_neurons, "control": True},
                {"kind": "direction", "layer": li0, "alpha": 1.0, "direction": rand_dir / rand_dir.norm(),
                 "control": True}]
    trials = []
    for cand in candidates + controls:
        undo = apply_edit(model, cand)
        st = loop_stats(generate_all(model, dev, args.new_tokens, device, amp, args.batch, args.seed + 7), start,
                        tok.eos_id)
        bpb = quick_bpb(model, tok, bpb_docs, device, amp)
        undo()
        t = {**{k: v for k, v in cand.items() if k != "direction"}, **st, "bpb": bpb, "bpb_change": bpb - base_bpb}
        trials.append(t)
        print(f"{'CONTROL ' if cand.get('control') else ''}{describe(cand)}: loop share "
              f"{st['repeat_token_share']:.3f} (original {base_dev['repeat_token_share']:.3f}), looping texts "
              f"{st['looping_texts']:.2f}, bits per byte {bpb - base_bpb:+.4f}", flush=True)

    ok = [(c, t) for c, t in zip(candidates, trials) if t["bpb_change"] <= args.max_bpb_loss]
    best = min(ok, key=lambda ct: ct[1]["repeat_token_share"], default=None)
    record = {"source_checkpoint": tg._home_short(src), "created": datetime.datetime.now().isoformat(timespec="seconds"),
              "ngram": NGRAM, "position_band": BAND, "positions": counts, "original_find": base_find,
              "original_choice": base_dev, "original_bpb_check": base_bpb, "trials": trials}
    report = os.path.splitext(dst)[0] + "_edit.json"
    if best is None or best[1]["repeat_token_share"] >= base_dev["repeat_token_share"]:
        record["result"] = "no candidate reduced loops without raising bits per byte; nothing saved"
        with open(report, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1)
        print(f"No candidate beat the original. Report: {report}")
        return
    cand, t = best
    apply_edit(model, cand)
    record.update({"chosen": {k: v for k, v in cand.items() if k != "direction"}, "chosen_trial": t})
    if cand["kind"] == "direction":
        record["direction"] = cand["direction"].tolist()
    out = dict(obj)
    out["model"] = tg.cpu_state_dict(model)
    out["edit"] = record
    out["metrics"] = edited_metrics(obj.get("metrics"))
    out["checkpoint_type"] = "inference"
    for key in ("optimizer", "optimizer_param_names", "scaler", "rng", "batcher", "interval", "ema"):
        out.pop(key, None)  # an edited copy is for evaluation only, never for --resume
    written = tg.atomic_save(out, dst)
    with open(report, "w", encoding="utf-8") as handle:
        json.dump({k: v for k, v in record.items() if k != "direction"}, handle, indent=1)
    print(f"Chose {describe(cand)}: loop share {base_dev['repeat_token_share']:.3f} -> "
          f"{t['repeat_token_share']:.3f}, bits per byte {t['bpb_change']:+.4f} on the choice set. "
          f"Saved {written} and {report}.")


if __name__ == "__main__":
    main()
