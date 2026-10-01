r"""Inspect a tinyGPT checkpoint (training or inference-only; legacy files from the original script too) or
any PyTorch state dict, and draw a diagnostic dashboard.

Examples
--------
    python view_pt.py tinyGPT_best.pt                      # text summary + tensor table
    python view_pt.py tinyGPT.pt --plot                    # 4x4 dashboard PNG next to the file (tinyGPT_dashboard.png,
                                                           # replaced each run; --out keeps a copy)
    python view_pt.py tinyGPT.pt --plot --html             # + hoverable embedding explorer
    python view_pt.py tiny_gpt_bpe_best.pt --plot --history-log log.txt --dataset datasets\bio_v1
    python view_pt.py tinyGPT.pt --tensor blocks.0.attn.q_proj.weight
    python view_pt.py tinyGPT.pt --similar genetic

Files are opened with torch.load(weights_only=True), which refuses arbitrary
Python objects. On PyTorch older than 2.6 this protection has a known bypass
(CVE-2025-32434): open .pt files only if you made them, and exchange models as
--export folders (safetensors, never unpickled). --trust-checkpoint lifts the
check entirely, for your own old files only.

Dashboard rows
  1  training history   loss curves, validation perplexity, learning rate,
                        confidence / accuracy / fact probes ("how opinionated")
  2  tokens             embedding PCA by token class, clustered cosine
                        similarity, embedding norms, output preference
  3  weights            per-block component scale, RMSNorm gains, singular
                        spectra, per-head projection scale
  4  behaviour          calibration, cloze fact probes, attention entropy,
                        Adam update size
Panels that need data the checkpoint does not have say so instead of
plotting invented values.
"""

import argparse
import datetime
import html
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_prep  # noqa: E402
import tiny_gpt  # noqa: E402

# Palette: validated categorical order (dataviz reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#9a9994"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f3b4b0", "#e34948", "#9c2a2a"]
TOKEN_CLASSES = ["word start", "word piece", "number", "punctuation", "symbol/math", "non-Latin letters",
                 "byte fallback", "special/whitespace"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="checkpoint or state-dict file")
    p.add_argument("--tensor", help="print one tensor by its state-dict name")
    p.add_argument("--similar", metavar="TOKEN", help="nearest tokens by embedding cosine similarity")
    p.add_argument("--top", type=int, default=10, help="neighbours shown by --similar")
    p.add_argument("--plot", action="store_true", help="write the dashboard PNG")
    p.add_argument("--html", action="store_true", help="write a hoverable token-embedding explorer (HTML)")
    p.add_argument("--out", help="dashboard path (default <checkpoint>_dashboard.png, replaced each run)")
    p.add_argument("--no-push", action="store_true",
                   help="do not commit and push the dashboard to GitHub (done automatically for checkpoints in "
                        "this repository's folder)")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--dataset", help="dataset folder for forward-pass panels (default: the one in the checkpoint)")
    p.add_argument("--eval-text", help="plain-text file for forward-pass panels when no dataset is available")
    p.add_argument("--probes", default=os.path.join(HERE, "probes_biology.tsv"), help="cloze fact probes TSV")
    p.add_argument("--no-forward", action="store_true", help="skip panels that run the model")
    p.add_argument("--history-log", help="log.txt to read loss curves from (for checkpoints without history)")
    p.add_argument("--device", choices=("cpu", "auto"), default="cpu", help="device for forward-pass panels")
    p.add_argument("--trust-checkpoint", action="store_true")
    args = p.parse_args()
    exported = os.path.isdir(args.file) and os.path.isfile(os.path.join(args.file, "model.safetensors"))
    if not (os.path.isfile(args.file) or exported):
        p.error(f"checkpoint does not exist (or is not an --export folder): {args.file}")
    if not 50 <= args.dpi <= 600:
        p.error("--dpi must be between 50 and 600")
    for name in ("dataset", "eval_text", "history_log"):
        value = getattr(args, name)
        if value and not os.path.exists(value):
            p.error(f"--{name.replace('_', '-')} not found: {value}")
    return args


# ---------------------------------------------------------------------------
# Loading and describing
# ---------------------------------------------------------------------------
class Checkpoint:
    """Everything the viewer needs, normalised across checkpoint families."""

    def __init__(self, path, trust=False):
        self.path = path
        self.obj = tiny_gpt.load_any(path, trust)
        self.family, self.kind = tiny_gpt.checkpoint_info(self.obj)
        self.weights = self._find_weights(self.obj)
        self.cfg = self.tok = None
        self.problems = []
        if self.family != "unknown":
            try:
                self.cfg = tiny_gpt.config_from_checkpoint(self.obj)
            except SystemExit as exc:
                self.problems.append(str(exc))
            try:
                self.tok = tiny_gpt.tokenizer_from_checkpoint(self.obj)
            except (SystemExit, ValueError) as exc:
                self.problems.append(f"tokenizer: {exc}")
        self.embedding = self.weights.get("token_embedding.weight", self.weights.get("tok.weight"))
        self.metrics = self._metrics()
        self.history = self.metrics.get("history") or []
        self.history_source = "checkpoint" if self.history else None

    @staticmethod
    def _find_weights(obj):
        if isinstance(obj, torch.Tensor):
            return {"tensor": obj}
        if isinstance(obj, dict):
            for key in ("model", "state_dict"):
                cand = obj.get(key)
                if isinstance(cand, dict):
                    tensors = {k: v for k, v in cand.items() if isinstance(v, torch.Tensor)}
                    if tensors:
                        return tensors
            return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        return {}

    def _metrics(self):
        o = self.obj if isinstance(self.obj, dict) else {}
        if self.family == "tinygpt":
            return dict(o.get("metrics") or {})
        m = {}
        if "best_val" in o:
            m["best_val"], m["best_step"] = o.get("best_val"), o.get("best_val_step")
        if o.get("last_val") is not None:
            m["last_eval"] = {"loss": o["last_val"], "step": o.get("last_val_step")}
        return m

    @property
    def model_name(self):
        if self.cfg is None:
            return "character-level tiny_gpt (v1)" if "tok.weight" in self.weights else "generic PyTorch weights"
        gqa = f"GQA {self.cfg.n_heads}/{self.cfg.n_kv_heads}" if self.cfg.n_kv_heads < self.cfg.n_heads else "MHA"
        tied = "tied embeddings" if self.cfg.tie_embeddings else "untied embeddings"
        return f"decoder-only Transformer: RoPE, RMSNorm, SwiGLU, {gqa}, {tied}"

    def pieces(self):
        if self.tok is not None:
            return [self.tok.piece(i) for i in range(self.tok.vocab_size)]
        vocab = self.obj.get("vocab") if isinstance(self.obj, dict) else None
        return list(vocab) if isinstance(vocab, (list, tuple)) else None

    def adam_names(self):
        """Parameter name for each optimizer state index."""
        o = self.obj
        if o.get("optimizer_param_names"):
            return list(o["optimizer_param_names"])
        names, seen = [], set()
        for name, tensor in self.weights.items():
            ptr = (tensor.data_ptr(), tuple(tensor.shape))
            if ptr in seen or (name == "lm_head.weight" and self.cfg is not None and self.cfg.tie_embeddings):
                continue
            seen.add(ptr)
            names.append(name)
        return names


def family_label(family):
    if family == "tinygpt":
        return "tinyGPT"
    if family.startswith("legacy-v"):
        return f"legacy (original script, format {family[len('legacy-v'):]})"
    return family


def pretty_name(name, tied=True):
    """blocks.0.attn.q_proj.weight -> 'block 0 / attn.q_proj'."""
    n = re.sub(r"\.weight$", "", name)
    m = re.match(r"blocks\.(\d+)\.(.+)", n)
    if m:
        return f"block {m.group(1)} / {m.group(2)}"
    head = "output head (tied)" if tied else "output head"
    return {"token_embedding": "token embedding", "norm": "final norm",
            "lm_head": head if name == "lm_head.weight" else n}.get(n, n)


def fmt(v, spec=".4f"):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def output_name(ck, kind, ext):
    """<checkpoint>_<kind><ext>: the same name every time, so a new dashboard replaces the
    previous one (the step and the time it was drawn are in the picture's title)."""
    base = os.path.normpath(ck.path)
    stem = base if os.path.isdir(base) else os.path.splitext(base)[0]  # export folders may contain dots
    return f"{stem}_{kind}{ext}"


def print_summary(ck):
    o = ck.obj if isinstance(ck.obj, dict) else {}
    size = (sum(os.path.getsize(os.path.join(ck.path, f)) for f in os.listdir(ck.path))
            if os.path.isdir(ck.path) else os.path.getsize(ck.path))
    print(f"{ck.path}: {size / 1e6:.2f} MB | format {family_label(ck.family)} | type {ck.kind}"
          + (f" | created {o['created']}" if o.get("created") else ""))
    print(f"model: {ck.model_name}")
    for problem in ck.problems:
        print(f"  note: {problem}")
    if ck.cfg is not None:
        c = ck.cfg
        n_total = sum(t.numel() for t in {t.data_ptr(): t for t in ck.weights.values()}.values())
        print(f"architecture: vocab {c.vocab_size:,} | ctx {c.ctx} | d_model {c.d_model} | layers {c.n_layers} | "
              f"query heads {c.n_heads} / key-value heads {c.n_kv_heads} (head size {c.head_dim}) | SwiGLU hidden "
              f"{c.ffn_hidden} | RoPE base {c.rope_base:g} | dropout {c.dropout} | parameters {n_total:,}")
    if ck.tok is not None:
        comp = {}
        for i in range(ck.tok.vocab_size):
            k = token_class(ck.tok, i)
            comp[k] = comp.get(k, 0) + 1
        print(f"tokenizer: SentencePiece BPE ({ck.tok.encode_mode} mode) | {ck.tok.vocab_size:,} pieces | BOS "
              f"{ck.tok.bos_id} EOS {ck.tok.eos_id} newline {ck.tok.newline_id} | "
              + ", ".join(f"{k} {v:,}" for k, v in sorted(comp.items(), key=lambda x: -x[1])))
    if "step" in o:
        total = (o.get("train_config") or {}).get("steps", o.get("total_steps"))
        extra = ""
        if o.get("tokens_seen"):
            extra += f" | tokens seen {o['tokens_seen']:,}"
        if o.get("train_seconds"):
            extra += f" | training time {tiny_gpt.fmt_hms(o['train_seconds'])}"
        print(f"step: {o['step']:,} / {total if total is not None else '?'}{extra}")
    m = ck.metrics
    last = m.get("last_eval")
    if last:
        per = last.get("per_source") or {}
        per_text = (" [" + ", ".join(f"{k} {v:.4f}" for k, v in per.items()) + "]") if len(per) > 1 else ""
        ppl = math.exp(min(last["loss"], 50)) if last.get("loss") is not None else None
        print(f"latest validation: loss {fmt(last.get('loss'))} measured at step {last.get('step')}{per_text} | "
              f"perplexity {fmt(ppl, ',.2f')}"
              + (f" | entropy {last['entropy']:.3f} nats | mean confidence {last['confidence']:.3f} | top-1 "
                 f"accuracy {last['top1']:.3f} | ECE {last['ece']:.3f}" if "entropy" in last else ""))
    if m.get("best_val") is not None and math.isfinite(float(m["best_val"])):
        print(f"best validation: loss {float(m['best_val']):.4f} at step {m.get('best_step')} | perplexity "
              f"{math.exp(min(float(m['best_val']), 50)):,.2f}")
    if ck.family.startswith("legacy"):
        print("  note: v3 logs printed the latest validation value at every report step, so values between "
              "evaluations (every --eval-every steps, default 1000) were not fresh measurements.")
    if ck.history:
        print(f"history: {len(ck.history)} evaluations stored (steps {ck.history[0]['step']}-{ck.history[-1]['step']})")
    ds = o.get("dataset")
    if isinstance(ds, dict):
        exists = os.path.isdir(ds.get("path", ""))
        print(f"dataset: {ds.get('path')} ({'found' if exists else 'NOT FOUND'}) | fingerprint "
              f"{str(ds.get('fingerprint'))[:12]} | train tokens {ds.get('train_tokens')}")
    tc = o.get("train_config")
    if tc:
        print("training settings: " + ", ".join(f"{k}={v}" for k, v in tc.items()))
    opt = o.get("optimizer")
    if isinstance(opt, dict) and opt.get("state"):
        steps = [float(s["step"]) for s in opt["state"].values() if isinstance(s, dict) and "step" in s]
        lrs = [g.get("lr") for g in opt.get("param_groups", [])]
        print(f"optimizer: AdamW state for {len(opt['state'])} tensors | updates {int(max(steps)) if steps else 0:,} "
              f"| group learning rates {', '.join(f'{x:.2e}' for x in lrs if x is not None)}")
    elif ck.family != "unknown":
        print("optimizer: none stored (inference-only checkpoint)")
    print()


def print_tensors(ck):
    if not ck.weights:
        keys = ", ".join(ck.obj.keys()) if isinstance(ck.obj, dict) else type(ck.obj).__name__
        print(f"No tensors found. Top-level keys: {keys}")
        return
    groups = {}
    for name, t in ck.weights.items():
        groups.setdefault((t.data_ptr(), tuple(t.shape)), []).append(name)
    shared = {n: g for g in groups.values() if len(g) > 1 for n in g}
    total = sum(ck.weights[g[0]].numel() for g in groups.values())
    print(f"{len(ck.weights)} named tensors | {total:,} unique values")
    print(f"{'tensor':34s} {'shape':>16s} {'count':>11s} {'mean':>10s} {'rms':>10s} {'min':>10s} {'max':>10s}")
    for name, t in ck.weights.items():
        v = t.float()
        stats = (v.mean().item(), v.square().mean().sqrt().item(), v.min().item(), v.max().item()) if v.numel() \
            else (float("nan"),) * 4
        label = pretty_name(name, tied=ck.cfg is None or ck.cfg.tie_embeddings) + (" *" if name in shared else "")
        print(f"{label[:34]:34s} {str(tuple(t.shape)):>16s} {t.numel():>11,} " + " ".join(f"{s:10.4g}" for s in stats))
    if shared:
        print("* shares storage with another entry (tied input/output embedding): counted once.")


def token_class(tok, i):
    c = data_prep.token_category(tok, i)
    if c in ("special", "whitespace/newline"):
        return "special/whitespace"
    return c if c in TOKEN_CLASSES else "other"


# ---------------------------------------------------------------------------
# Forward-pass data
# ---------------------------------------------------------------------------
def eval_windows(ck, args, max_windows=48):
    """Validation token windows for the checkpoint's tokenizer, plus where they came from."""
    if ck.tok is None or ck.cfg is None:
        return None, "no tokenizer/architecture in this checkpoint"
    ctx = ck.cfg.ctx
    ds_path = args.dataset or ((ck.obj.get("dataset") or {}).get("path") if isinstance(ck.obj, dict) else None)
    if ds_path and os.path.isdir(ds_path):
        manifest, arrays = data_prep.open_token_files(ds_path, verify=False)
        meta = manifest["tokenizer"]
        ds_tok = data_prep.Tokenizer.from_file(os.path.join(ds_path, meta["file"]), meta.get("encode_mode", "lines"))
        same = ds_tok.sha256 == ck.tok.sha256 and ds_tok.encode_mode == ck.tok.encode_mode
        rows = []
        per = max(1, max_windows // max(len(arrays["val"]), 1))
        for name, arr in sorted(arrays["val"].items()):
            span = ctx + 1 if same else ctx * 3
            if len(arr) <= span:
                continue
            for s in np.linspace(0, len(arr) - span - 1, per).astype(np.int64):
                ids = np.asarray(arr[s:s + span], dtype=np.int64)
                if not same:
                    ids = np.asarray(ck.tok.encode(ds_tok.decode(ids.tolist())), dtype=np.int64)
                if len(ids) >= ctx + 1:
                    rows.append(ids[:ctx + 1])
        if rows:
            note = "validation windows of " + os.path.basename(os.path.normpath(ds_path))
            return torch.from_numpy(np.stack(rows)), note + ("" if same else " (re-tokenized for this checkpoint)")
    if args.eval_text:
        with open(args.eval_text, encoding="utf-8", errors="replace") as handle:
            ids = np.asarray(ck.tok.encode(handle.read()), dtype=np.int64)
        n = min(max_windows, (len(ids) - 1) // ctx)
        if n >= 1:
            starts = np.linspace(0, len(ids) - ctx - 1, n).astype(np.int64)
            return torch.from_numpy(np.stack([ids[s:s + ctx + 1] for s in starts])), os.path.basename(args.eval_text)
        return None, f"{args.eval_text} is shorter than one context window ({ctx + 1} tokens)"
    return None, "no evaluation text: the checkpoint's dataset is not available; pass --dataset or --eval-text"


@torch.no_grad()
def forward_stats(model, windows, vocab, device, bins=10):
    mean_prob = torch.zeros(vocab, dtype=torch.float64)
    freq = torch.zeros(vocab, dtype=torch.float64)
    conf_sum = torch.zeros(bins, dtype=torch.float64)
    acc_sum = torch.zeros(bins, dtype=torch.float64)
    count = torch.zeros(bins, dtype=torch.float64)
    nll, n = 0.0, 0
    for start in range(0, windows.shape[0], 8):
        batch = windows[start:start + 8].to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        logp = F.log_softmax(model(x).float(), dim=-1)
        probs = logp.exp()
        mean_prob += probs.sum((0, 1)).double().cpu()
        freq += torch.bincount(y.flatten().cpu(), minlength=vocab).double()
        conf, pred = probs.max(-1)
        c = conf.double().flatten().cpu()
        ok = torch.isfinite(c)  # NaN weights (a diverged run) give NaN confidences; skip them
        right = (pred == y).double().flatten().cpu()[ok]
        c = c[ok]
        which = (c.clamp(0, 1 - 1e-9) * bins).long().clamp(0, bins - 1)  # float64: 1.0 stays in the top bin
        conf_sum += torch.bincount(which, c, bins)
        acc_sum += torch.bincount(which, right, bins)
        count += torch.bincount(which, minlength=bins).double()
        nll += -logp.gather(-1, y.unsqueeze(-1)).sum().item()
        n += y.numel()
    total = max(n, 1)
    nz = count > 0
    return {"mean_prob": (mean_prob / total).numpy(), "freq": (freq / total).numpy(), "loss": nll / total,
            "tokens": n, "calibration": {"confidence": (conf_sum / count.clamp(min=1)).tolist(),
                                          "accuracy": (acc_sum / count.clamp(min=1)).tolist(),
                                          "count": count.tolist()},
            "ece": float((conf_sum[nz] - acc_sum[nz]).abs().sum() / total)}


@torch.no_grad()
def attention_entropy(model, window, device):
    """Mean entropy (bits) of each head's attention distribution over one window."""
    x = window[None, :-1].to(device)
    h = model.token_embedding(x)
    out = []
    t = x.shape[1]
    denom = torch.log2(torch.arange(1, t + 1, device=device).float()).clamp(min=1e-9)
    for block in model.blocks:
        w = block.attn.attention_weights(block.attn_norm(h), model.rope)[0]  # (H, T, T)
        ent = -(w * torch.log2(w.clamp(min=1e-12))).sum(-1)  # (H, T)
        out.append((ent[:, 1:] / denom[1:]).mean(-1).cpu().numpy())
        h = block(h, model.rope)
    return np.stack(out)  # (layers, heads), 0 = attends to one token, 1 = uniform over the prefix


# ---------------------------------------------------------------------------
# History from log files (for checkpoints that do not store it)
# ---------------------------------------------------------------------------
NEW_LINE = re.compile(r"step\s+(\d+)/(\d+) \| train ([\d.]+) \| val ([\d.]+) @(\d+).*?\| lr ([\d.e+-]+)")
OLD_LINE = re.compile(r"step\s+(\d+) \| training loss ([\d.]+) \| validation loss ([\d.]+) \| perplexity "
                     r"([\d.,]+) \| lr ([\d.e+-]+)")


def history_from_log(path, ck_path=None, max_step=None):
    """History of the viewed checkpoint's run from log.txt. tinyGPT sessions are
    matched by the checkpoint path in their header ("checkpoints: ...<name>.pt"),
    and all sessions of that run (resumes included) are merged, the latest value
    per step winning, up to the checkpoint's step. Without a match the last run is
    used, and the source label says so. v3 lines repeat a stale validation value
    between evaluations, so for v3 only evaluation steps keep a val point."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    blocks = re.split(r"\n(?=--- run started |={20,}\nRUN )", text)
    stem = re.sub(r"_best$", "", os.path.splitext(os.path.basename(os.path.normpath(ck_path)))[0]) if ck_path else None
    tiny = [b for b in blocks if NEW_LINE.search(b)]
    mine = [b for b in tiny if stem and re.search(r"checkpoints: .*?[\\/]" + re.escape(stem) + r"\.pt ", b)]
    if mine or tiny:
        merged = {}
        for block in (mine or tiny[-1:]):
            for s, _, tr, va, vs, lr in NEW_LINE.findall(block):
                if int(vs) == int(s) and (max_step is None or int(s) <= int(max_step)):
                    merged[int(s)] = {"step": int(s), "train_loss": float(tr), "val_loss": float(va), "lr": float(lr),
                                      "ppl": math.exp(min(float(va), 50))}
        if merged:
            label = (f"log ({len(mine)} session(s) of {stem})" if mine
                     else "log (last tinyGPT run; no session names this checkpoint)")
            return [merged[s] for s in sorted(merged)], label
    for block in reversed(blocks):
        v3 = OLD_LINE.findall(block)
        if v3:
            cmd = re.search(r"Command: (.*)", block)
            command = cmd.group(1) if cmd else ""
            every = re.search(r"--eval-every[ =](\d+)", command)
            every = int(every.group(1)) if every else 1000
            final = re.search(r"--steps[ =](\d+)", command)
            final = int(final.group(1)) if final else 8000
            resumed = "--resume" in command
            out, first = [], True
            for s, tr, va, ppl, lr in v3:
                s = int(s)
                entry = {"step": s, "train_loss": float(tr), "lr": float(lr), "val_loss": None, "ppl": None}
                if (first and not resumed) or s % every == 0 or s == final:
                    entry["val_loss"], entry["ppl"] = float(va), float(ppl.replace(",", ""))
                first = False
                out.append(entry)
            start = re.search(r"--- run started (\S+)", block)
            return out, f"log (v3 run {start.group(1) if start else '?'}, val kept at eval steps only)"
    return [], "no step lines found in the log"


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def label_text(piece, limit=12):
    """Make a token piece printable in the plot font."""
    s = piece.replace("\n", "\\n").replace("\t", "\\t")
    out = []
    for ch in s:
        out.append(ch if _renderable(ch) else f"\\u{ord(ch):04x}")
    s = "".join(out)
    return s if len(s) <= limit else s[:limit - 1] + "\u2026"


_GLYPHS = None


def _renderable(ch):
    global _GLYPHS
    if _GLYPHS is None:
        try:
            from matplotlib import font_manager
            from matplotlib.ft2font import FT2Font
            _GLYPHS = set(FT2Font(font_manager.findfont("DejaVu Sans")).get_charmap())
        except Exception:  # noqa: BLE001
            _GLYPHS = set(range(32, 0x250))
    return ord(ch) in _GLYPHS


def titled(ax, main, sub=None):
    """Bold panel title plus a wrapped grey subtitle (keeps titles inside their column)."""
    lines = textwrap.wrap(sub, 92) if sub else []
    ax.set_title(main, loc="left", fontsize=10.5, pad=6 + 10.5 * len(lines))
    if lines:
        ax.text(0, 1.012, "\n".join(lines), transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom",
                ha="left", linespacing=1.25)


def message(ax, text, title):
    titled(ax, title)
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=9, color=INK2,
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)


def style_axes(ax, grid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.6)
    ax.grid(grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def heatmap(ax, fig, matrix, rows, cols, cmap, label, vmin=None, vmax=None, annotate=True, fmt_spec=".3g"):
    ax.grid(False)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7, rotation=0 if len(cols) <= 8 else 90)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(label, fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    if annotate and matrix.size <= 120:
        lo, hi = np.nanmin(matrix), np.nanmax(matrix)
        mid = (lo + hi) / 2
        for (i, j), v in np.ndenumerate(matrix):
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt_spec), ha="center", va="center", fontsize=6,
                        color="#ffffff" if (v > mid if cmap.name.startswith("seq") else abs(v) > 0.6) else INK)
    return im


def create_dashboard(ck, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    warnings.filterwarnings("ignore", message="Glyph .* missing")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.labelsize": 8.5, "axes.labelcolor": INK2, "text.color": INK,
                         "axes.titlecolor": INK, "figure.facecolor": SURFACE, "legend.fontsize": 7,
                         "legend.frameon": False})
    seq = LinearSegmentedColormap.from_list("seq_blue", BLUES)
    div = LinearSegmentedColormap.from_list("div_blue_red", DIVERGING)

    # One sub-figure per row: panels align within a row only, so a colorbar or
    # long tick labels in one panel do not shrink the whole column.
    fig = plt.figure(figsize=(26, 23), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.1, h_pad=0.12)
    axs = []
    for row in fig.subfigures(4, 1, hspace=0.015):
        row.set_facecolor(SURFACE)
        axs.extend(row.subplots(1, 4, gridspec_kw={"wspace": 0.08}))
    for ax in axs:
        style_axes(ax)

    history = ck.history
    if not history and args.history_log:
        history, ck.history_source = history_from_log(args.history_log, ck.path,
                                                      (ck.obj or {}).get("step") if isinstance(ck.obj, dict) else None)
    pieces = ck.pieces()
    classes = [token_class(ck.tok, i) for i in range(ck.tok.vocab_size)] if ck.tok is not None else None

    # forward-pass material, computed once
    model = fstats = windows = None
    fnote = "forward-pass panels disabled (--no-forward)" if args.no_forward else ""
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if not args.no_forward and ck.cfg is not None and ck.tok is not None:
        try:
            model, _, _ = tiny_gpt.model_from_checkpoint(ck.obj, device)
            windows, fnote = eval_windows(ck, args)
            if windows is not None:
                fstats = forward_stats(model, windows, ck.cfg.vocab_size, device)
        except SystemExit as exc:
            fnote = str(exc)
        except Exception as exc:  # noqa: BLE001 - a failed forward pass only disables the forward panels
            model = fstats = windows = None
            fnote = f"forward pass failed: {type(exc).__name__}: {str(exc)[:120]}"
    elif not args.no_forward:
        fnote = "no tiny_gpt architecture/tokenizer in this file"

    panels = [
        ("Loss by step", lambda ax: panel_loss(ax, history, ck)),
        ("Validation perplexity", lambda ax: panel_ppl(ax, history, ck)),
        ("Learning rate", lambda ax: panel_lr(ax, history, ck)),
        ("How opinionated the model is", lambda ax: panel_confidence(ax, history, ck)),
        ("Token embeddings (PCA)", lambda ax: panel_pca(ax, ck, pieces, classes, fstats)),
        ("Token similarity (clustered)", lambda ax: panel_similarity(ax, fig, ck, pieces, classes, fstats, div)),
        ("Token embedding norms", lambda ax: panel_norms(ax, ck, classes)),
        ("Output preference", lambda ax: panel_preference(ax, ck, pieces, classes, fstats, fnote)),
        ("Weight scale per block", lambda ax: panel_components(ax, fig, ck, seq)),
        ("RMSNorm gains", lambda ax: panel_norm_gains(ax, ck)),
        ("Singular value spectra", lambda ax: panel_spectra(ax, ck)),
        ("Per-head projection scale", lambda ax: panel_heads(ax, fig, ck, seq)),
        ("Calibration", lambda ax: panel_calibration(ax, ck, fstats, fnote)),
        ("Cloze fact probes", lambda ax: panel_probes(ax, ck, model, device, args)),
        ("Attention entropy", lambda ax: panel_attention(ax, fig, model, windows, device, seq, fnote)),
        ("Adam update size", lambda ax: panel_adam(ax, fig, ck, seq)),
    ]
    for ax, (title, draw) in zip(axs, panels):
        try:
            draw(ax)
            if not ax.get_title(loc="left"):
                ax.set_title(title, loc="left")
        except Exception as exc:  # noqa: BLE001 - one broken panel must not kill the dashboard
            ax.cla()
            text = str(exc).replace(os.path.expanduser("~"), "~")  # the dashboard may be published
            message(ax, f"panel failed: {type(exc).__name__}: {text[:120]}", title)
            print(f"WARNING: dashboard panel '{title}' failed: {type(exc).__name__}: {exc}")

    o = ck.obj if isinstance(ck.obj, dict) else {}
    last = ck.metrics.get("last_eval") or {}
    best = ck.metrics.get("best_val")
    sub = [f"{os.path.basename(ck.path)} | {family_label(ck.family)} {ck.kind} checkpoint | step {o.get('step', '?')} "
           f"| drawn {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    if last.get("loss") is not None:
        sub.append(f"latest val {last['loss']:.4f} @{last.get('step')}")
    if best is not None and math.isfinite(float(best)):
        sub.append(f"best val {float(best):.4f} @{ck.metrics.get('best_step')}")
    if ck.history_source:
        sub.append(f"history from {ck.history_source}")
    if fnote and fstats is None and not args.no_forward:
        sub.append(f"forward panels: {fnote}")
    elif fstats is not None:
        sub.append(f"forward panels: {fstats['tokens']:,} tokens from {fnote} (loss {fstats['loss']:.3f})")
    fig.suptitle(ck.model_name + chr(10) + " | ".join(sub), fontsize=12, color=INK, x=0.01, ha="left",
                 linespacing=1.5)
    out = args.out or output_name(ck, "dashboard", ".png")
    finalize_legends(fig)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved dashboard: {out}")
    return out


# ---- row 1: training history -------------------------------------------------
def _no_history(ax, ck, title):
    if ck.family == "tinygpt":
        why = "this checkpoint stores no evaluation history yet"
    elif ck.family.startswith("legacy"):
        why = ("v3 checkpoints store only the latest and best validation loss.\n"
               "Pass --history-log log.txt to read the curve from the log.")
    else:
        why = "not a tiny_gpt training checkpoint"
    message(ax, why, title)


def panel_loss(ax, history, ck):
    title = "Loss by step"
    if not history:
        return _no_history(ax, ck, title)
    tr = [(h["step"], h["train_loss"]) for h in history if h.get("train_loss") is not None]
    va = [(h["step"], h["val_loss"]) for h in history if h.get("val_loss") is not None]
    legacy = (ck.history_source or "").startswith("log (v3")
    if tr:
        ax.plot(*zip(*tr), color=SERIES[0], lw=2,
                label="training (one batch, v3 log)" if legacy else "training (mean since previous line)")
    if va:
        ax.plot(*zip(*va), color=SERIES[1], lw=2, marker="o", ms=3.5, label="validation (mean over sources)")
    per = {}
    for h in history:
        for k, v in (h.get("val_per_source") or {}).items():
            per.setdefault(k, []).append((h["step"], v))
    if len(per) > 1:
        for i, (k, pts) in enumerate(sorted(per.items())):
            ax.plot(*zip(*pts), color=SERIES[2 + i % 6], lw=1, alpha=0.9, label=f"validation: {k}")
    best = min(va, key=lambda p: p[1]) if va else None
    if best:
        ax.scatter([best[0]], [best[1]], s=90, marker="*", color=SERIES[1], edgecolor=SURFACE, lw=1.5, zorder=5)
        ax.annotate(f"best {best[1]:.3f} @{best[0]}", best, xytext=(6, 8), textcoords="offset points",
                    fontsize=7.5, color=INK2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (nats per token)")
    place_legend(ax)
    titled(ax, title, "training = one batch at the plotted step; validation = random validation batches, fresh "
                      "only at evaluation steps (v3 log)" if legacy else
                      "training = mean loss since the previous metrics line; validation = fixed windows, "
                      "measured at the plotted step")


def panel_ppl(ax, history, ck):
    title = "Validation perplexity"
    va = [(h["step"], h.get("ppl") or math.exp(min(h["val_loss"], 50))) for h in history or []
          if h.get("val_loss") is not None]
    if not va:
        return _no_history(ax, ck, title)
    ax.plot(*zip(*va), color=SERIES[1], lw=2, marker="o", ms=3.5)
    ax.set_yscale("log")
    s, p = va[-1]
    ax.annotate(f"{p:,.1f} @{s}", (s, p), xytext=(-4, 8), textcoords="offset points", ha="right", fontsize=7.5,
                color=INK2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("exp(validation loss)")
    titled(ax, title, "exp(mean validation loss), log scale; lower is better")


def panel_lr(ax, history, ck):
    title = "Learning rate"
    tc = (ck.obj.get("train_config") if isinstance(ck.obj, dict) else None) or {}
    drew = False
    if tc.get("steps"):
        sched = tiny_gpt.LRSchedule(tc["schedule"], tc["steps"], tc["lr"], tc["min_lr"], tc["warmup_steps"],
                                    tc.get("decay_frac", 0.2), tc.get("decay_shape", "1-sqrt"))
        xs = np.unique(np.linspace(1, tc["steps"], 400).astype(int))
        ax.plot(xs, [sched.lr_at(int(x)) for x in xs], color=MUTED, lw=1.2, label="planned schedule")
        ax.axvline(ck.obj.get("step", 0), color=INK2, lw=0.8)
        ax.annotate(f"now: step {ck.obj.get('step', 0):,}", (ck.obj.get("step", 0), sched.peak), xytext=(4, -10),
                    textcoords="offset points", fontsize=7.5, color=INK2)
        drew = True
    pts = [(h["step"], h["lr"]) for h in history or [] if h.get("lr")]
    if pts:
        ax.plot(*zip(*pts), color=SERIES[0], lw=0, marker="o", ms=3.5, label="at evaluations")
        drew = True
    if not drew:
        return _no_history(ax, ck, title)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("learning rate")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1e}"))
    place_legend(ax)
    titled(ax, title, sched.describe() if tc.get("steps") else "values recorded at evaluations")


def panel_confidence(ax, history, ck):
    title = "How opinionated the model is"
    rows = [h for h in history or [] if h.get("confidence") is not None]
    if not rows:
        return message(ax, "Confidence is recorded at each evaluation by tinyGPT.\n"
                            "This checkpoint has no such history.", title)
    steps = [h["step"] for h in rows]
    ax.plot(steps, [h["confidence"] for h in rows], color=SERIES[0], lw=2, label="mean top-1 probability (confidence)")
    ax.plot(steps, [h["top1"] for h in rows], color=SERIES[1], lw=2, label="top-1 accuracy")
    probes = [(h["step"], h["probe_acc"]) for h in rows if h.get("probe_acc") is not None]
    if probes:
        ax.plot(*zip(*probes), color=SERIES[2], lw=1.5, marker="o", ms=3, label="fact-probe accuracy")
    floors = [(h["step"], h["probe_floor"]) for h in rows if h.get("probe_floor") is not None]
    if floors:
        ax.plot(*zip(*floors), color=SERIES[2], lw=1.2, ls="--", label="fact-probe floor (shuffled questions)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("proportion")
    last = rows[-1]
    place_legend(ax)
    titled(ax, title, f"confidence above accuracy = overconfident; mean prediction entropy "
                      f"{last['entropy']:.2f} nats at step {last['step']}")


# ---- row 2: tokens ------------------------------------------------------------
def _token_order(ck, classes, fstats, want):
    """Most frequent tokens of the wanted classes (eval-text frequency, else low
    merge ID, which in BPE means an early/frequent merge)."""
    ids = [i for i, c in enumerate(classes) if c in want]
    if fstats is not None:
        ids.sort(key=lambda i: -fstats["freq"][i])
    return ids


def place_labels(ax, points, texts, fontsize=6.5, max_labels=None):
    """Annotate points without overlapping labels: try a few offsets around each
    point and skip the label if every position collides with one already placed."""
    root = ax.figure
    while getattr(root, "figure", root) is not root:
        root = root.figure
    renderer = root.canvas.get_renderer()
    ax.autoscale_view()
    placed, shown = [], 0
    offsets = [(4, 3), (4, -9), (-4, 3), (-4, -9), (6, 11), (6, -16), (-6, 11), (-6, -16)]
    for (x, y), text in zip(points, texts):
        if max_labels and shown >= max_labels:
            break
        for dx, dy in offsets:
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                              color=INK, ha="left" if dx > 0 else "right", va="bottom")
            box = ann.get_window_extent(renderer).expanded(1.04, 1.15)
            if not any(box.overlaps(other) for other in placed):
                placed.append(box)
                shown += 1
                break
            ann.remove()
    return shown


_LEGENDS = []  # (axes, legend options), placed by finalize_legends() once the layout is final
LEGEND_SPOTS = ("upper right", "upper left", "lower right", "lower left", "upper center", "lower center",
                "center right", "center left")


def place_legend(ax, **options):
    """Add a legend; finalize_legends() later moves it to a spot that covers no data."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", **options)
        _LEGENDS.append((ax, options))


def _root_figure(fig):
    while getattr(fig, "figure", fig) is not fig:
        fig = fig.figure
    return fig


def _obstacles(ax, renderer):
    """Everything drawn in the axes, in axes coordinates: line vertices (with extra
    points along each segment, so a line crossing a legend counts), scatter points,
    and the boxes of text labels and bars."""
    to_axes = ax.transAxes.inverted()
    pts = []
    for line in ax.get_lines():
        xy = np.asarray(line.get_xydata(), float)
        if not len(xy):
            continue
        with np.errstate(all="ignore"):
            disp = line.get_transform().transform(xy)
        disp = disp[np.isfinite(disp).all(1)]
        if len(disp) > 1 and line.get_linestyle() not in ("None", "", " ") and line.get_linewidth() > 0:
            t = np.linspace(0, 1, 16)[:, None, None]
            disp = (disp[:-1][None] * (1 - t) + disp[1:][None] * t).reshape(-1, 2)
        pts.append(to_axes.transform(disp))
    for coll in ax.collections:
        offsets = np.asarray(coll.get_offsets(), float)
        if len(offsets) and offsets.ndim == 2:
            with np.errstate(all="ignore"):
                disp = coll.get_offset_transform().transform(offsets)
            pts.append(to_axes.transform(disp[np.isfinite(disp).all(1)]))
    boxes = [artist.get_window_extent(renderer).transformed(to_axes) for artist in list(ax.texts) + list(ax.patches)]
    return (np.concatenate(pts) if pts else np.zeros((0, 2))), boxes


def _legend_cost(ax, legend, renderer, pts, boxes):
    bb = legend.get_window_extent(renderer).transformed(ax.transAxes.inverted())
    inside = ((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum()
    return int(inside) + 50 * sum(1 for b in boxes if b.overlaps(bb))


def finalize_legends(fig):
    """Put each legend where it covers no data (lines, points, labels, bars). If no
    such spot exists, add empty space above the data and put the legend there."""
    if not _LEGENDS:
        return
    root = _root_figure(fig)
    root.draw_without_rendering()
    renderer = root.canvas.get_renderer()
    for ax, options in _LEGENDS:
        pts, boxes = _obstacles(ax, renderer)
        best = None
        for loc in LEGEND_SPOTS:
            cost = _legend_cost(ax, ax.legend(loc=loc, **options), renderer, pts, boxes)
            if best is None or cost < best[0]:
                best = (cost, loc)
            if cost == 0:
                break
        if best[0] == 0:
            ax.legend(loc=best[1], **options)
            continue
        y0, y1 = ax.get_ylim()
        n = len(ax.get_legend_handles_labels()[1])
        for head in (0.25, 0.4, 0.6, 0.9):
            if ax.get_yscale() == "log" and y0 > 0:
                ax.set_ylim(y0, y1 * (y1 / y0) ** head)
            else:
                ax.set_ylim(y0, y1 + (y1 - y0) * head)
            legend = ax.legend(loc="upper center", ncol=min(3, n), **options)
            pts, boxes = _obstacles(ax, renderer)
            if _legend_cost(ax, legend, renderer, pts, boxes) == 0:
                break
    _LEGENDS.clear()


def panel_pca(ax, ck, pieces, classes, fstats):
    title = "Token embeddings (PCA)"
    if ck.embedding is None or classes is None:
        return message(ax, "no token embedding or tokenizer in this file", title)
    e = ck.embedding.float()[:len(classes)]
    centered = e - e.mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    coords = (centered @ vh[:2].T).numpy()
    var = (s ** 2 / (s ** 2).sum()).numpy()
    counts = {}
    for k, cls in enumerate(TOKEN_CLASSES + ["other"]):
        idx = [i for i, c in enumerate(classes) if c == cls]
        if not idx:
            continue
        color = SERIES[k] if k < len(SERIES) else OTHER
        ax.scatter(coords[idx, 0], coords[idx, 1], s=9, marker=MARKERS[k] if k < len(MARKERS) else ".",
                   color=color, alpha=0.65, linewidths=0, label=f"{cls} ({len(idx):,})")
        counts[cls] = len(idx)
    top = _token_order(ck, classes, fstats, {"word start", "word piece", "number", "punctuation"})[:40]
    shown = place_labels(ax, [coords[i] for i in top], [label_text(pieces[i]) for i in top], max_labels=22)
    ax.set_xlabel(f"PC 1 ({var[0]:.1%} of variance)")
    ax.set_ylabel(f"PC 2 ({var[1]:.1%} of variance)")
    place_legend(ax, markerscale=1.6, fontsize=6.5)
    which = "most frequent (evaluation text)" if fstats is not None else "earliest-merged"
    titled(ax, title, f"each dot is one vocabulary token's embedding row; \u2581 marks a word start; "
                      f"labels: {shown} of the {which} tokens that fit without overlapping")


def panel_similarity(ax, fig, ck, pieces, classes, fstats, cmap):
    title = "Token similarity (clustered)"
    if ck.embedding is None or classes is None:
        return message(ax, "no token embedding or tokenizer in this file", title)
    ids = [i for i in _token_order(ck, classes, fstats, {"word start", "word piece"})
           if len(pieces[i].replace("\u2581", "")) >= 3][:36]
    if len(ids) < 4:
        return message(ax, "fewer than 4 word tokens", title)
    e = F.normalize(ck.embedding.float()[ids], dim=1)
    sims = (e @ e.T).numpy()
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        dist = np.clip(1 - sims, 0, 2)
        np.fill_diagonal(dist, 0)
        order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
        how = "average-linkage clustering on 1 - cosine"
    except ImportError:
        order = np.argsort(np.linalg.eigh(sims)[1][:, -2])
        how = "ordered by the leading eigenvector"
    sims = sims[np.ix_(order, order)]
    labels = [label_text(pieces[ids[k]], 10) for k in order]
    heatmap(ax, fig, sims, labels, labels, cmap, "cosine similarity", -1, 1, annotate=False)
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("token (same clustered order as rows)")
    ax.set_ylabel("token")
    titled(ax, title, f"36 frequent word tokens; rows and columns share one order ({how})")


def panel_norms(ax, ck, classes):
    title = "Token embedding norms"
    if ck.embedding is None:
        return message(ax, "no token embedding in this file", title)
    norms = ck.embedding.float().norm(dim=1).numpy()
    if classes is None:
        ax.plot(norms, color=SERIES[0], lw=0.8)
    else:
        for k, cls in enumerate(TOKEN_CLASSES + ["other"]):
            idx = [i for i, c in enumerate(classes) if c == cls]
            if idx:
                ax.scatter(idx, norms[idx], s=4, color=SERIES[k] if k < len(SERIES) else OTHER,
                           marker=MARKERS[k] if k < len(MARKERS) else ".", linewidths=0, alpha=0.7, label=cls)
        place_legend(ax, markerscale=2.5, fontsize=6.5)
    ax.set_xlabel("token ID (BPE merge order: low = merged early)")
    ax.set_ylabel("L2 norm of embedding row")
    tied = ck.cfg is not None and ck.cfg.tie_embeddings
    titled(ax, title, "tied embeddings: each row is also that token's output weight vector" if tied else None)


def panel_preference(ax, ck, pieces, classes, fstats, fnote):
    title = "Output preference"
    if fstats is None:
        return message(ax, "Needs a forward pass: " + fnote + "\n\nThis model has no output-bias tensor, so its\n"
                            "'preferences' are measured from predictions.", title)
    mean_p, freq = fstats["mean_prob"], fstats["freq"]
    seen = freq > 0
    lo = max(min(mean_p[seen].min(), freq[seen].min()), 1e-8)
    for k, cls in enumerate(TOKEN_CLASSES + ["other"]):
        idx = [i for i, c in enumerate(classes) if c == cls and seen[i]]
        if idx:
            ax.scatter(freq[idx], mean_p[idx], s=10, color=SERIES[k] if k < len(SERIES) else OTHER,
                       marker=MARKERS[k] if k < len(MARKERS) else ".", linewidths=0, alpha=0.7, label=cls)
    ax.plot([lo, 1], [lo, 1], color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ratio = np.where(seen, np.log(np.maximum(mean_p, 1e-12) / np.maximum(freq, 1e-12)), -np.inf)
    over = [i for i in np.argsort(-ratio)[:20] if seen[i] and freq[i] * fstats["tokens"] >= 3][:8]
    place_labels(ax, [(freq[i], mean_p[i]) for i in over], [label_text(pieces[i]) for i in over], max_labels=6)
    ax.set_xlabel("observed frequency of the token as next token")
    ax.set_ylabel("mean predicted probability")
    place_legend(ax, fontsize=6.5, markerscale=1.8)
    titled(ax, title, "above the diagonal = predicted more often than it occurs; the model has no output-bias "
                      "tensor, so preference is measured from its predictions (labels: most over-predicted)")


# ---- row 3: weights -----------------------------------------------------------
COMPONENTS = [("attn.q_proj", "attn.q_proj.weight"), ("attn.k_proj", "attn.k_proj.weight"),
              ("attn.v_proj", "attn.v_proj.weight"), ("attn.o_proj", "attn.o_proj.weight"),
              ("ffn.gate", "ffn.gate.weight"), ("ffn.up", "ffn.up.weight"), ("ffn.down", "ffn.down.weight")]


def _layers(ck):
    blocks = {int(m.group(1)) for n in ck.weights for m in [re.match(r"blocks\.(\d+)\.", n)] if m}
    return sorted(blocks)


def panel_components(ax, fig, ck, cmap):
    title = "Weight scale per block"
    layers = _layers(ck)
    if not layers:
        return message(ax, "no Transformer blocks found", title)
    m = np.full((len(COMPONENTS), len(layers)), np.nan)
    for j, b in enumerate(layers):
        for i, (_, rel) in enumerate(COMPONENTS):
            t = ck.weights.get(f"blocks.{b}.{rel}")
            if t is not None:
                m[i, j] = t.float().square().mean().sqrt().item()
    heatmap(ax, fig, m, [c for c, _ in COMPONENTS], [f"block {b}" for b in layers], cmap, "RMS of weights")
    titled(ax, title, "RMS of each weight matrix; initialised at 0.02 (attn.o_proj and ffn.down at "
                      "0.02 / sqrt(2 x layers))")


def panel_norm_gains(ax, ck):
    title = "RMSNorm gains"
    items = []
    for name, t in ck.weights.items():
        m = re.match(r"blocks\.(\d+)\.(attn_norm|ffn_norm)\.weight$", name)
        if m:
            items.append((f"block {m.group(1)}\n{m.group(2)}", t))
    if "norm.weight" in ck.weights:
        items.append(("final\nnorm", ck.weights["norm.weight"]))
    if not items:
        return message(ax, "no RMSNorm gains found", title)
    bp = ax.boxplot([t.float().numpy() for _, t in items], showfliers=False, patch_artist=True, widths=0.6)
    for k, box in enumerate(bp["boxes"]):
        box.set(facecolor=SERIES[0] if "attn" in items[k][0] else (SERIES[1] if "ffn" in items[k][0] else OTHER),
                alpha=0.55, edgecolor=INK2, linewidth=0.6)
    for part in ("whiskers", "caps", "medians"):
        for line in bp[part]:
            line.set(color=INK2, linewidth=0.8)
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_xticks(range(1, len(items) + 1))
    ax.set_xticklabels([n for n, _ in items], fontsize=6.5, rotation=0 if len(items) <= 9 else 90)
    ax.set_ylabel("gain per channel (1.0 at initialisation)")
    ax.grid(axis="x", visible=False)
    titled(ax, title, "per-channel gains: attn_norm blue, ffn_norm orange, final grey; 1.0 at initialisation")


def panel_spectra(ax, ck):
    title = "Singular value spectra"
    layers = _layers(ck)
    mid = layers[len(layers) // 2] if layers else None
    targets = [("token embedding", ck.embedding)] + [
        (f"block {mid} / {c}", ck.weights.get(f"blocks.{mid}.{rel}")) for c, rel in COMPONENTS] if mid is not None \
        else [("token embedding", ck.embedding)]
    drew = 0
    for k, (label, m) in enumerate(targets):
        if m is None or m.ndim != 2:
            continue
        sv = torch.linalg.svdvals(m.float())
        if sv.numel() == 0 or sv[0] <= 0:
            continue
        x = np.arange(1, sv.numel() + 1) / sv.numel()
        ax.plot(x, (sv / sv[0]).numpy(), color=SERIES[k % len(SERIES)], lw=1.6, label=label)
        drew += 1
    if not drew:
        return message(ax, "no 2-D weight matrices", title)
    ax.set_yscale("log")
    ax.set_xlabel("singular value rank / matrix rank")
    ax.set_ylabel("\u03c3\u1d62 / \u03c3\u2098\u2090\u2093")
    place_legend(ax, fontsize=6.5)
    titled(ax, title, "token embedding and the middle block's matrices; a fast drop means low effective rank")


def panel_heads(ax, fig, ck, cmap):
    title = "Per-head projection scale"
    if ck.cfg is None:
        return message(ax, "needs the architecture settings (head counts)", title)
    c, layers = ck.cfg, _layers(ck)
    hd = c.head_dim
    rows, data = [], []
    for proj, n, axis in (("q", c.n_heads, 0), ("k", c.n_kv_heads, 0), ("v", c.n_kv_heads, 0), ("o", c.n_heads, 1)):
        for h in range(n):
            vals = []
            for b in layers:
                w = ck.weights.get(f"blocks.{b}.attn.{proj}_proj.weight")
                if w is None:
                    vals.append(np.nan)
                    continue
                part = w[h * hd:(h + 1) * hd] if axis == 0 else w[:, h * hd:(h + 1) * hd]
                vals.append(part.float().square().mean().sqrt().item())
            rows.append(f"{proj}_proj head {h}")
            data.append(vals)
    heatmap(ax, fig, np.array(data), rows, [f"block {b}" for b in layers], cmap, "RMS of head slice",
            annotate=len(rows) * len(layers) <= 120)
    ax.set_yticklabels(rows, fontsize=5.5 if len(rows) > 24 else 7)
    share = c.n_heads // c.n_kv_heads
    note = f"GQA: query heads {share}k..{share}k+{share - 1} share key/value head k" if share > 1 else "MHA"
    titled(ax, title, note)


# ---- row 4: behaviour -----------------------------------------------------------
def panel_calibration(ax, ck, fstats, fnote):
    title = "Calibration of the top prediction"
    last = ck.metrics.get("last_eval") or {}
    cal, ece, src = None, None, ""
    if fstats is not None:
        cal, ece, src = fstats["calibration"], fstats["ece"], "forward pass now"
    elif last.get("calibration"):
        cal, ece, src = last["calibration"], last.get("ece"), f"stored at step {last.get('step')}"
    if cal is None:
        return message(ax, "No calibration data: " + fnote, title)
    conf, acc, cnt = (np.array(cal[k]) for k in ("confidence", "accuracy", "count"))
    keep = cnt > 0
    centers = (np.arange(len(cnt)) + 0.5) / len(cnt)
    ax.bar(centers[keep], acc[keep], width=0.9 / len(cnt), color=SERIES[0], alpha=0.85,
           label="accuracy in confidence bin")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, label="perfect calibration")
    ax.scatter(conf[keep], acc[keep], s=np.clip(cnt[keep] / cnt.max() * 120, 8, 120), color=SERIES[1], zorder=4,
               edgecolor=SURFACE, linewidths=1.2, label="mean confidence (size = tokens)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence (probability of the top prediction)")
    ax.set_ylabel("fraction of top predictions that were right")
    place_legend(ax)
    titled(ax, title, f"expected calibration error {ece:.3f} ({src}); bars below the diagonal = overconfident")


def panel_probes(ax, ck, model, device, args):
    title = "Fact benchmark"
    if model is None:
        return message(ax, "needs a tinyGPT model with a tokenizer", title)
    if not args.probes or not os.path.isfile(args.probes):
        return message(ax, f"probe file not found: {args.probes}", title)
    probes = tiny_gpt.read_probes(args.probes)
    acc, results = tiny_gpt.score_probes(model, ck.tok, probes, device, None)
    summary = sorted(tiny_gpt.probe_summary(results).items(), key=lambda kv: kv[1]["accuracy"])
    names = [f"{name} (n={s['n']})" for name, s in summary]
    accs = np.array([s["accuracy"] for _, s in summary])
    lows = accs - np.array([s["low"] for _, s in summary])
    highs = np.array([s["high"] for _, s in summary]) - accs
    chance = sum(r["chance"] for r in results) / len(results)
    floors = np.array([s["floor"] for _, s in summary])
    floor = tiny_gpt.probe_floor(results)
    y = np.arange(len(summary))
    ax.barh(y, accs, color=[GOOD if a > f else CRITICAL for a, f in zip(accs, floors)], height=0.62, alpha=0.9)
    ax.errorbar(accs, y, xerr=[lows, highs], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    ax.scatter(floors, y, marker="|", s=140, color=INK, lw=2, zorder=4, label="floor (question words shuffled)")
    ax.axvline(chance, color=MUTED, lw=0.8, ls=":")
    ax.axvline(floor, color=INK, lw=1, ls="--")
    ax.annotate(f"floor {floor:.0%}", (floor, len(summary) - 0.4), xytext=(3, 0), textcoords="offset points",
                fontsize=7, color=INK, va="bottom")
    ax.annotate(f"chance {chance:.0%}", (chance, -0.6), xytext=(-3, 0), textcoords="offset points",
                fontsize=7, color=MUTED, va="top", ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy (correct continuation preferred over all distractors); lines: 95% interval")
    k, n = sum(r["correct"] for r in results), len(results)
    low, high = tiny_gpt.wilson_interval(k, n)
    by_diff = tiny_gpt.probe_summary(results, "difficulty")
    diff_text = ", ".join(f"{d} {by_diff[d]['accuracy']:.0%}" for d in ("easy", "medium", "hard") if d in by_diff)
    place_legend(ax, fontsize=6.5)
    titled(ax, f"{title}: {k}/{n} correct ({acc:.0%}, 95% interval {low:.0%}-{high:.0%}), floor {floor:.0%}",
           f"per category, {os.path.basename(args.probes)}; {diff_text}. Floor = accuracy with each question's "
           "words shuffled (topic words only); green = above its floor, i.e. the sentence itself helped. "
           "Per-item results: tiny_gpt.py --benchmark")


def panel_attention(ax, fig, model, windows, device, cmap, fnote):
    title = "Attention entropy"
    if model is None or windows is None:
        return message(ax, "Needs a forward pass: " + (fnote or "no model"), title)
    ent = attention_entropy(model, windows[0], device)
    heatmap(ax, fig, ent, [f"block {b}" for b in range(ent.shape[0])], [f"head {h}" for h in range(ent.shape[1])],
            cmap, "normalised entropy", 0, 1, fmt_spec=".2f")
    ax.set_xlabel("query head")
    titled(ax, title, "one validation window, entropy normalised by the prefix length: 0 = attends to one "
                      "token, 1 = spread evenly")


def panel_adam(ax, fig, ck, cmap):
    title = "Adam update size"
    opt = ck.obj.get("optimizer") if isinstance(ck.obj, dict) else None
    if not isinstance(opt, dict) or not opt.get("state"):
        kind = "inference-only checkpoint" if ck.kind in ("inference", "inference_only") else "no optimizer state"
        return message(ax, f"{kind}: optimizer moments are only saved in training checkpoints (<name>.pt)", title)
    names = ck.adam_names()
    size = {}
    for idx, st in opt["state"].items():
        idx = int(idx)
        if idx < len(names) and isinstance(st, dict) and "exp_avg" in st and "exp_avg_sq" in st:
            m, v = st["exp_avg"].float(), st["exp_avg_sq"].float()
            size[names[idx]] = (m.abs() / (v.sqrt() + 1e-8)).mean().item()
    layers = _layers(ck)
    if not size or not layers:
        return message(ax, "optimizer state could not be matched to parameter names", title)
    grid = np.full((len(COMPONENTS), len(layers)), np.nan)
    for j, b in enumerate(layers):
        for i, (_, rel) in enumerate(COMPONENTS):
            grid[i, j] = size.get(f"blocks.{b}.{rel}", np.nan)
    heatmap(ax, fig, grid, [c for c, _ in COMPONENTS], [f"block {b}" for b in layers], cmap,
            "mean |m| / (sqrt(v) + eps)")
    extra = ", ".join(f"{pretty_name(n)} {size[n]:.3f}" for n in ("token_embedding.weight", "norm.weight") if n in size)
    titled(ax, title, "mean |m| / (sqrt(v) + eps) per matrix; near 1 = gradient sign consistent across steps"
                      + (f"; {extra}" if extra else ""))


# ---------------------------------------------------------------------------
# HTML embedding explorer (hover labels)
# ---------------------------------------------------------------------------
def _git(*args):
    return subprocess.run(["git", "-C", HERE, *args], capture_output=True, text=True, timeout=300)


def push_dashboard(png, ck):
    """Copy the dashboard to dashboards/ in this repository, commit only that file
    and push it, for checkpoints in this repository's folder (test runs and other
    folders are left alone). A git problem is reported, never fatal; a commit that
    could not be pushed goes up with the next push."""
    try:
        top = _git("rev-parse", "--show-toplevel")
        if top.returncode != 0:
            return
        root = os.path.normcase(os.path.abspath(top.stdout.strip()))
        ck_dir = os.path.normcase(os.path.dirname(os.path.abspath(ck.path)))
        if os.path.commonpath([root, ck_dir]) != root:
            return
        dest_dir = os.path.join(top.stdout.strip(), "dashboards")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(png))
        if os.path.abspath(dest) != os.path.abspath(png):
            shutil.copy2(png, dest)
        rel = os.path.relpath(dest, top.stdout.strip()).replace("\\", "/")
        _git("add", "--", rel)
        if _git("diff", "--cached", "--quiet", "--", rel).returncode == 0:
            print(f"Dashboard unchanged since the last push: {rel}")
            return
        step = ck.obj.get("step", "?") if isinstance(ck.obj, dict) else "?"
        commit = _git("commit", "-q", "-m", f"Dashboard: {os.path.basename(ck.path)} at step {step}", "--", rel)
        if commit.returncode != 0:
            print(f"WARNING: could not commit {rel}: {(commit.stderr or commit.stdout).strip()[:200]}")
            return
        push = _git("push", "-q", "origin", "HEAD")
        if push.returncode != 0:
            print(f"WARNING: committed {rel} but could not push (it goes up with the next push): "
                  f"{push.stderr.strip()[:200]}")
            return
        print(f"Pushed dashboard to GitHub: {rel}")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:  # no git, timeout, other drive...
        print(f"WARNING: dashboard not pushed ({type(exc).__name__}: {exc})")


def write_embedding_html(ck, out_path):
    if ck.embedding is None or ck.tok is None:
        print("--html needs token embeddings and a tokenizer.")
        return None
    pieces = ck.pieces()
    classes = [token_class(ck.tok, i) for i in range(ck.tok.vocab_size)]
    e = ck.embedding.float()[:len(pieces)]
    c = e - e.mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(c, full_matrices=False)
    xy = (c @ vh[:2].T).numpy()
    norms = e.norm(dim=1).numpy()
    w, h, pad = 900, 640, 40
    lo, hi = xy.min(0), xy.max(0)
    sx = lambda v: pad + (v - lo[0]) / max(hi[0] - lo[0], 1e-9) * (w - 2 * pad)  # noqa: E731
    sy = lambda v: h - pad - (v - lo[1]) / max(hi[1] - lo[1], 1e-9) * (h - 2 * pad)  # noqa: E731
    order = TOKEN_CLASSES + ["other"]
    dots = []
    for i, (x, y) in enumerate(xy):
        k = order.index(classes[i])
        tip = html.escape(f"id {i}  {pieces[i]!r}  ({classes[i]})  norm {norms[i]:.3f}")
        dots.append(f'<circle class="c{k}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2"><title>{tip}</title></circle>')
    legend = "".join(f'<li><span class="sw c{k}"></span>{html.escape(cls)} ({classes.count(cls):,})</li>'
                     for k, cls in enumerate(order) if cls in classes)
    colors = "".join(f".c{k}{{fill:{col};background:{col}}}" for k, col in enumerate(SERIES + [OTHER]))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Token embeddings</title>
<style>
:root{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e1}}
@media (prefers-color-scheme:dark){{:root{{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#383835}}}}
body{{margin:0;padding:16px;background:var(--surface);color:var(--ink);font:14px system-ui,sans-serif}}
h1{{font-size:18px;margin:0 0 4px}} p{{color:var(--ink2);margin:0 0 12px;max-width:70ch}}
svg{{width:100%;max-width:{w}px;height:auto;border:1px solid var(--grid);border-radius:6px}}
circle{{opacity:.7}} circle:hover{{opacity:1;stroke:var(--ink);stroke-width:1.5}}
ul{{list-style:none;padding:0;display:flex;flex-wrap:wrap;gap:6px 16px;color:var(--ink2)}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}} {colors}
</style></head><body><h1>Token embeddings: {html.escape(os.path.basename(ck.path))}</h1>
<p>Each dot is one vocabulary token's embedding row projected on its first two principal components
(PC 1 {float((s[0] ** 2 / (s ** 2).sum())):.1%}, PC 2 {float((s[1] ** 2 / (s ** 2).sum())):.1%} of variance).
Hover a dot for the token, its class and embedding norm.</p><ul>{legend}</ul>
<svg viewBox="0 0 {w} {h}" role="img" aria-label="PCA scatter of token embeddings">{''.join(dots)}</svg>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(page)
    print(f"Saved embedding explorer: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ck = Checkpoint(args.file, args.trust_checkpoint)
    print_summary(ck)
    print_tensors(ck)
    if args.tensor:
        if args.tensor not in ck.weights:
            sys.exit(f"\nNo tensor named {args.tensor!r}. Use one of the names listed above.")
        torch.set_printoptions(precision=4, edgeitems=4, linewidth=120, sci_mode=False)
        print(f"\n{args.tensor} ({pretty_name(args.tensor)}):\n{ck.weights[args.tensor]}")
    if args.similar:
        pieces = ck.pieces()
        if ck.embedding is None or pieces is None:
            sys.exit("\n--similar needs token embeddings and a tokenizer.")
        if ck.tok is not None:
            ids = [i for i in ck.tok.encode(args.similar) if i != ck.tok.newline_id]
            if not ids:
                sys.exit(f"\nCould not tokenize {args.similar!r}.")
            token_id = ids[0]
            if len(ids) > 1:
                print(f"\nNote: {args.similar!r} is {len(ids)} tokens "
                      f"({', '.join(repr(pieces[i]) for i in ids)}); comparing the first.")
        else:
            if args.similar not in pieces:
                sys.exit(f"\n{args.similar!r} is not in the vocabulary.")
            token_id = pieces.index(args.similar)
        vec = F.normalize(ck.embedding[:len(pieces)].float(), dim=1)
        sims = vec @ vec[token_id]
        order = [int(i) for i in sims.argsort(descending=True) if int(i) != token_id][:args.top]
        print(f"\nTokens closest to {pieces[token_id]!r} (id {token_id}) by cosine similarity of embedding rows:")
        for i in order:
            print(f"  {i:6d} {pieces[i]!r:28.28s} {sims[i].item():.3f}")
    if args.plot:
        out = create_dashboard(ck, args)
        if out and not args.no_push:
            push_dashboard(out, ck)
    if args.html:
        write_embedding_html(ck, os.path.splitext(args.out)[0] + "_embeddings.html" if args.out
                             else output_name(ck, "embeddings", ".html"))


if __name__ == "__main__":
    main()
