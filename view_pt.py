r"""Inspect a tinyGPT checkpoint (training or inference-only; legacy files from the original script too) or
any PyTorch state dict, and draw a diagnostic dashboard.

Examples
--------
    python view_pt.py tinyGPT_best.pt                      # text summary + tensor table
    python view_pt.py tinyGPT.pt --plot                    # dashboard PNG next to the file (tinyGPT_dashboard.png,
                                                           # replaced each run; --out writes it elsewhere)
    python view_pt.py tinyGPT.pt --plot --panels all       # + the weight and behaviour panels (runs the model)
    python view_pt.py tinyGPT.pt --plot-tokens             # the token panels (tinyGPT_tokens.png)
    python view_pt.py tinyGPT.pt --plot --html             # + hoverable embedding explorer
    python view_pt.py tinyGPT_muon_best.pt --plot --panels all --history-log log.txt --dataset datasets\bio_v1
    python view_pt.py tinyGPT.pt --tensor blocks.0.attn.q_proj.weight
    python view_pt.py tinyGPT.pt --similar genetic

Files are opened with torch.load(weights_only=True), which refuses arbitrary
Python objects. On PyTorch older than 2.6 this protection has a known bypass
(CVE-2025-32434): open .pt files only if you made them, and exchange models as
--export folders (safetensors, never unpickled). --trust-checkpoint lifts the
check entirely, for your own old files only.

Dashboard (--plot): nine panels from the stored evaluations; the model is not run
  is the run healthy             loss, validation minus training, gradient norm
  is it learning what it is for  validation by source, biology facts over
                                 time and by subject
  is it worth the hours          improvement per 1,000 steps, speed, plan
  mixture of experts only        expert balance, load per block, router bias
A training or best checkpoint's curves go on with the evaluations its run has
since appended to <name>_metrics.jsonl beside it.
--panels all adds the learning rate, calibration, weight scale per block,
RMSNorm gains by depth, singular spectra, per-head projection scale and
attention entropy, and shows the fact benchmark of the file's own weights
(scored now unless training stored it for this step).
--plot-tokens writes a second picture: embedding PCA by token class, clustered
cosine similarity, embedding norms, output preference.
Panels that need data the checkpoint does not have say so instead of
plotting invented values.
"""

import argparse
import bisect
import datetime
import html
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
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
# File types matplotlib's savefig writes without another program (the extensions --plot --out accepts).
PICTURE_TYPES = ("png", "pdf", "svg", "svgz", "jpg", "jpeg", "webp", "tif", "tiff", "eps", "ps", "raw", "rgba")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="checkpoint or state-dict file")
    p.add_argument("--tensor", help="print one tensor by its state-dict name")
    p.add_argument("--similar", metavar="TOKEN", help="nearest tokens by embedding cosine similarity")
    p.add_argument("--top", type=int, default=10, help="neighbours shown by --similar")
    p.add_argument("--plot", action="store_true", help="write the dashboard PNG")
    p.add_argument("--panels", choices=("history", "all"), default="history",
                   help="what --plot draws. history (default): nine panels from the stored evaluations (is the run "
                        "healthy, is it learning what it is for, is it worth the hours); the model is not run. "
                        "all: those and the weight and behaviour panels, which build the model and run it on "
                        "validation windows (slower)")
    p.add_argument("--plot-tokens", action="store_true",
                   help="write a second picture with the four token panels (embedding PCA, similarity, norms, output "
                        "preference): <checkpoint>_tokens.png, or X_tokens.png with --out X.png. Never pushed")
    p.add_argument("--html", action="store_true", help="write a hoverable token-embedding explorer (HTML)")
    p.add_argument("--out", help="dashboard path (default <checkpoint>_dashboard.png, replaced each run)")
    p.add_argument("--no-push", action="store_true",
                   help="do not commit and push the dashboard to GitHub (done automatically for checkpoints in "
                        "this repository's folder)")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--pca-labels", type=int, default=50,
                   help="--plot-tokens: most frequent tokens labelled in the 'Token embeddings (PCA)' panel "
                        "(without a forward pass: the lowest token IDs)")
    p.add_argument("--sim-tokens", type=int, default=100,
                   help="--plot-tokens: most frequent word tokens in the 'Token similarity (clustered)' panel "
                        "(without a forward pass: the lowest token IDs)")
    p.add_argument("--dataset", help="dataset folder for forward-pass panels (default: the one in the checkpoint)")
    p.add_argument("--eval-text", help="plain-text file for forward-pass panels when no dataset is available")
    p.add_argument("--probes", default=os.path.join(HERE, "probes_biology.tsv"), help="cloze fact probes TSV")
    p.add_argument("--no-forward", action="store_true",
                   help="--panels all, --plot-tokens: do not run the model (its panels say so)")
    p.add_argument("--full", action="store_true",
                   help="also the slow parts for big models (singular value spectra above d_model 2048)")
    p.add_argument("--history-log", help="log.txt to read loss curves from (for checkpoints without history)")
    p.add_argument("--device", choices=("cpu", "auto", "cuda", "mps"), default="cpu",
                   help="device for forward-pass panels (auto: CUDA, else Apple MPS, else CPU)")
    p.add_argument("--trust-checkpoint", action="store_true")
    args = p.parse_args()
    exported = os.path.isdir(args.file) and os.path.isfile(os.path.join(args.file, "model.safetensors"))
    if not (os.path.isfile(args.file) or exported):
        p.error(f"checkpoint does not exist (or is not an --export folder): {args.file}")
    if not 50 <= args.dpi <= 600:
        p.error("--dpi must be between 50 and 600")
    if not 0 <= args.pca_labels <= 300:
        p.error("--pca-labels must be between 0 and 300")
    if not 4 <= args.sim_tokens <= 300:
        p.error("--sim-tokens must be between 4 and 300")
    for name in ("dataset", "eval_text", "history_log"):
        value = getattr(args, name)
        if value and not os.path.exists(value):
            p.error(f"--{name.replace('_', '-')} not found: {value}")
    # checked here, not when the finished picture is saved; without an output that uses it --out is ignored
    if args.out and (args.plot or args.plot_tokens or args.html):
        folder = os.path.dirname(os.path.abspath(args.out))
        if not os.path.isdir(folder):
            p.error(f"--out: the folder does not exist: {folder}")
        ext = os.path.splitext(args.out)[1].lstrip(".").lower()  # none means PNG; --html and --plot-tokens use the stem
        if args.plot and ext and ext not in PICTURE_TYPES:
            p.error(f"--out: cannot write a .{ext} picture (use one of: {', '.join(PICTURE_TYPES)})")
    return args


# ---------------------------------------------------------------------------
# Loading and describing
# ---------------------------------------------------------------------------
class Checkpoint:
    """Everything the viewer needs, normalised across checkpoint families."""

    def __init__(self, path, trust=False):
        self.path = path
        self.obj = load_checkpoint(path, trust)
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
        # SFT/DPO files (finetune.py): fine-tuned weights, but step, metrics and train_config are the base
        # model's pre-training; the fine-tuning curve is in finetune["history"].
        ft = self.obj.get("finetune") if isinstance(self.obj, dict) else None
        self.finetune = ft if isinstance(ft, dict) else None
        # tools/edit_weights.py files: edited weights, so the base model's stored evaluation does not describe them
        ed = self.obj.get("edit") if isinstance(self.obj, dict) else None
        self.edit = ed if isinstance(ed, dict) else None

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
        qk = ", QK-norm" if getattr(self.cfg, "qk_norm", False) else ""
        moe = (f", MoE {self.cfg.n_experts} experts x {self.cfg.experts_per_token} per token"
               if getattr(self.cfg, "n_experts", 0) else "")
        return f"decoder-only Transformer: RoPE, RMSNorm, SwiGLU{moe}, {gqa}{qk}, {tied}"

    @property
    def moe(self):
        """Whether the architecture has mixture-of-experts blocks."""
        return bool(self.cfg is not None and getattr(self.cfg, "n_experts", 0))

    def pieces(self):
        if self.tok is not None:
            return [self.tok.piece(i) for i in range(self.tok.vocab_size)]
        vocab = self.obj.get("vocab") if isinstance(self.obj, dict) else None
        return list(vocab) if isinstance(vocab, (list, tuple)) else None


def load_checkpoint(path, trust=False):
    """tiny_gpt.load_any, memory-mapped where that is safe: tensors (the optimizer
    state of a training checkpoint included) are then read from disk only when a
    panel uses them. Not on Windows, where a mapped file cannot be replaced, so
    training could not save over a checkpoint while its dashboard is drawn."""
    return tiny_gpt.load_any(path, trust, mmap=True)


def build_model(ck, device):
    """The checkpoint's model in eval mode, holding the checkpoint's own tensors
    (tiny_gpt.model_from_checkpoint with assign=True): no random initialisation
    and no second copy of the weights in memory."""
    return tiny_gpt.model_from_checkpoint(ck.obj, device, assign=True)[0]


def scrub(text):
    """Text drawn on the dashboard with the home folder shown as ~ (the PNG may be
    pushed to GitHub): plain, forward-slash and doubled-backslash (repr) forms."""
    home = os.path.expanduser("~")
    for variant in sorted({home, home.replace("\\", "/"), home.replace("\\", "\\\\")}, key=len, reverse=True):
        text = re.sub(re.escape(variant), "~", str(text), flags=re.IGNORECASE)
    return text


def path_name(path):
    """Last part of a path written on any system (a Windows path read on the Mac too)."""
    return re.split(r"[\\/]", str(path or "?").rstrip("\\/"))[-1]


def finetune_label(ft):
    return str(ft.get("method", "fine-tuning")).upper()


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
    previous one (the picture's header names the run, its step and when it was drawn)."""
    base = os.path.normpath(ck.path)
    stem = base if os.path.isdir(base) else os.path.splitext(base)[0]  # export folders may contain dots
    return f"{stem}_{kind}{ext}"


def print_summary(ck):
    o = ck.obj if isinstance(ck.obj, dict) else {}
    size = (sum(os.path.getsize(os.path.join(ck.path, f)) for f in os.listdir(ck.path))
            if os.path.isdir(ck.path) else os.path.getsize(ck.path))
    # a best file may hold the EMA of the weights instead of the trained ones (training keeps both and saves the
    # better); its "weights" field says which, so the label follows it everywhere the loss is quoted
    ema_note = (f" (EMA weights, decay {o.get('ema_decay'):g})" if o.get("weights") == "ema" and o.get("ema_decay")
                else " (EMA weights)" if o.get("weights") == "ema" else "")
    print(f"{ck.path}: {size / 1e6:.2f} MB | format {family_label(ck.family)} | type {ck.kind}{ema_note}"
          + (f" | created {o['created']}" if o.get("created") else ""))
    print(f"model: {ck.model_name}")
    for problem in ck.problems:
        print(f"  note: {problem}")
    if ck.cfg is not None:
        c = ck.cfg
        n_total = sum(t.numel() for t in {t.data_ptr(): t for t in ck.weights.values()}.values())
        moe = ""
        if ck.moe:
            # the parameter tensors include the router bias buffers (E per MoE block), hence the count from the
            # architecture; active = what one token's forward pass touches (count_params)
            n_total, _, n_active, _ = tiny_gpt.count_params(c)
            moe = (f" | experts {c.n_experts} x hidden {c.expert_hidden}, {c.experts_per_token} per token in "
                   f"{c.n_moe_blocks} of {c.n_layers} blocks, {c.router} router, balance {c.balance}"
                   + (f", shared SwiGLU {c.shared_hidden}" if c.shared_hidden else "")
                   + f" | parameters {n_total:,} total, {n_active:,} active per token")
        print(f"architecture: vocab {c.vocab_size:,} | ctx {c.ctx} | d_model {c.d_model} | layers {c.n_layers} | "
              f"query heads {c.n_heads} / key-value heads {c.n_kv_heads} (head size {c.head_dim}) | SwiGLU hidden "
              f"{c.ffn_hidden} | RoPE base {c.rope_base:g} | dropout {c.dropout}"
              + (moe if moe else f" | parameters {n_total:,}"))
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
        print(f"step: {o['step']:,} / {total if total is not None else '?'}{extra}"
              + (" (base model's pre-training)" if ck.finetune else ""))
    ft = ck.finetune
    if ft:
        hist = ft.get("history") or []
        held = (f" | held-out loss {hist[-1]['val_loss']:.4f} after epoch {hist[-1].get('epoch', len(hist))}"
                if hist and hist[-1].get("val_loss") is not None else "")
        base = path_name(ft.get("base"))
        print(f"fine-tuned: {finetune_label(ft)} from {base}, {ft.get('examples', '?')} examples, "
              f"{ft.get('epochs', '?')} epochs, lr {ft.get('lr', '?')}{held}")
    base_note = derived_note(ck)  # whose stored numbers these are, for a fine-tuned or an edited file
    m = ck.metrics
    # fine-tuned and edited files keep the base model's last evaluation (newer ones as base_last_eval)
    last = m.get("last_eval") or (m.get("base_last_eval") if ft or ck.edit else None)
    last_note = base_note if ft or not m.get("last_eval") else ""
    if not last_note and holds_ema(ck):  # training evaluates the raw weights; the average only when it picks it
        last_note = " (raw weights; this file holds the EMA average)"
    if last:
        per = last.get("per_source") or {}
        per_text = (" [" + ", ".join(f"{k} {v:.4f}" for k, v in per.items()) + "]") if len(per) > 1 else ""
        ppl = math.exp(min(last["loss"], 50)) if last.get("loss") is not None else None
        print(f"latest validation{last_note}: loss {fmt(last.get('loss'))} measured at step {last.get('step')}"
              f"{per_text} | perplexity {fmt(ppl, ',.2f')}"
              + (f" | entropy {last['entropy']:.3f} nats | mean confidence {last['confidence']:.3f} | top-1 "
                 f"accuracy {last['top1']:.3f} | ECE {last['ece']:.3f}" if "entropy" in last else ""))
    if m.get("best_val") is not None and math.isfinite(float(m["best_val"])):
        print(f"best{' (ema)' if ema_note else ''} validation{base_note}: loss {float(m['best_val']):.4f} at step "
              f"{m.get('best_step')} | perplexity {math.exp(min(float(m['best_val']), 50)):,.2f}")
    if ck.family.startswith("legacy"):
        print("  note: v3 logs printed the latest validation value at every report step, so values between "
              "evaluations (every --eval-every steps, default 1000) were not fresh measurements.")
    if ck.history:
        newer = newer_rows(ck)
        print(f"history{base_note}: {len(ck.history)} evaluations stored "
              f"(steps {ck.history[0]['step']}-{ck.history[-1]['step']})"
              + (f" + {len(newer)} newer in {os.path.basename(metrics_file(ck))} (to step {newer[-1]['step']})"
                 if newer else ""))
    ds = o.get("dataset")
    if isinstance(ds, dict):
        try:  # at its stored path or in datasets/ with the stored fingerprint: an export folder stores the name only
            tiny_gpt.resolve_dataset(o)
            exists = True
        except SystemExit:  # not on this machine, or rebuilt since
            exists = False
        tokens = f" | train tokens {ds['train_tokens']}" if ds.get("train_tokens") is not None else ""
        print(f"dataset: {ds.get('path')} ({'found' if exists else 'NOT FOUND'}) | fingerprint "
              f"{str(ds.get('fingerprint'))[:12]}{tokens}")
    tc = o.get("train_config")
    if tc:
        print("training settings: " + ", ".join(f"{k}={v}" for k, v in tc.items()))
    opt = o.get("optimizer")
    if isinstance(opt, dict) and opt.get("state"):
        steps = [float(s["step"]) for s in opt["state"].values() if isinstance(s, dict) and "step" in s]
        lrs = [g.get("lr") for g in opt.get("param_groups", [])]
        kind = tiny_gpt.OPTIMIZER_LABELS.get((tc or {}).get("optimizer", "adamw"), "AdamW")
        print(f"optimizer: {kind} state for {len(opt['state'])} tensors | updates {int(max(steps)) if steps else 0:,} "
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
def window_budget(cfg):
    """How many validation windows the forward-pass panels run: 16 up to the
    ~220M models trained so far, fewer for bigger ones (4 from about 800M), since
    the CPU forward pass costs about 2 x active parameters per token:
    total = min(16, max(4, round(16 x 2.2e8 / active parameters)))."""
    try:
        n_params = tiny_gpt.count_params(cfg)[2]
    except Exception:  # noqa: BLE001 - an unusual config only loses the scaling
        return 16
    return int(min(16, max(4, round(16 * 2.2e8 / max(n_params, 1)))))


def eval_windows(ck, args, max_windows=None):
    """Validation token windows for the checkpoint's tokenizer, plus where they came from.

    max_windows is the TOTAL number of windows (default window_budget: 16, fewer
    for models above ~220M parameters), handed out round-robin over the
    validation sources, biggest first: with 12 sources a per-source share would
    be 1 for any total up to 23 and could not shrink with the model."""
    if ck.tok is None or ck.cfg is None:
        return None, "no tokenizer/architecture in this checkpoint"
    ctx = ck.cfg.ctx
    total = max(1, int(max_windows or window_budget(ck.cfg)))
    ds_path, refused = args.dataset, ""
    if not ds_path:
        try:
            ds_path = tiny_gpt.resolve_dataset(ck.obj if isinstance(ck.obj, dict) else {})
        except SystemExit as exc:  # not here, or rebuilt since (its validation text may be training text)
            ds_path, refused = None, str(exc)
    if ds_path and os.path.isdir(ds_path):
        manifest, arrays = data_prep.open_token_files(ds_path, verify=False)
        meta = manifest["tokenizer"]
        ds_tok = data_prep.Tokenizer.from_file(os.path.join(ds_path, meta["file"]), meta.get("encode_mode", "lines"))
        same = ds_tok.sha256 == ck.tok.sha256 and ds_tok.encode_mode == ck.tok.encode_mode
        span = ctx + 1 if same else ctx * 3
        sources = sorted(((name, arr) for name, arr in arrays["val"].items() if len(arr) > span),
                         key=lambda item: (-len(item[1]), item[0]))
        rows = []
        for k, (name, arr) in enumerate(sources):
            count = total // len(sources) + (1 if k < total % len(sources) else 0)  # round-robin share
            if count < 1:
                continue
            for s in np.linspace(0, len(arr) - span - 1, count).astype(np.int64):
                ids = np.asarray(arr[s:s + span], dtype=np.int64)
                if not same:
                    ids = np.asarray(ck.tok.encode(ds_tok.decode(ids.tolist())), dtype=np.int64)
                if len(ids) >= ctx + 1:
                    rows.append(ids[:ctx + 1])
        if rows:
            stored = (ck.obj.get("dataset") or {}) if isinstance(ck.obj, dict) else {}
            stored_path = stored.get("path") if isinstance(stored, dict) else None
            found = ("" if args.dataset or not stored_path or os.path.normcase(os.path.abspath(ds_path)) in {
                os.path.normcase(os.path.abspath(stored_path)),
                os.path.normcase(os.path.abspath(os.path.join(HERE, stored_path)))}
                else " (found by name in datasets/)")
            note = "validation windows of " + os.path.basename(os.path.normpath(ds_path)) + found
            return torch.from_numpy(np.stack(rows)), note + ("" if same else " (re-tokenized for this checkpoint)")
    if args.eval_text:
        with open(args.eval_text, encoding="utf-8", errors="replace") as handle:
            ids = np.asarray(ck.tok.encode(handle.read()), dtype=np.int64)
        n = min(total, (len(ids) - 1) // ctx)
        if n >= 1:
            starts = np.linspace(0, len(ids) - ctx - 1, n).astype(np.int64)
            return torch.from_numpy(np.stack([ids[s:s + ctx + 1] for s in starts])), os.path.basename(args.eval_text)
        return None, f"{args.eval_text} is shorter than one context window ({ctx + 1} tokens)"
    if refused:
        return None, refused + " Or pass --eval-text."
    return None, "no evaluation text: the checkpoint's dataset is not available; pass --dataset or --eval-text"


@torch.no_grad()
def forward_stats(model, windows, vocab, device, bins=10, token_budget=4096, rows=1024):
    """Output-preference and calibration statistics over the windows. The model runs
    on about token_budget tokens at a time and the output layer on `rows` positions
    at a time, so the (tokens x vocabulary) table never exists in full."""
    mean_prob = torch.zeros(vocab, dtype=torch.float64)
    freq = torch.zeros(vocab, dtype=torch.float64)
    conf_sum = torch.zeros(bins, dtype=torch.float64)
    acc_sum = torch.zeros(bins, dtype=torch.float64)
    count = torch.zeros(bins, dtype=torch.float64)
    nll, n = 0.0, 0
    per = max(1, token_budget // max(windows.shape[1] - 1, 1))
    for start in range(0, windows.shape[0], per):
        batch = windows[start:start + per].to(device)
        x, y = batch[:, :-1], batch[:, 1:].reshape(-1)
        hidden = model(x, return_hidden=True)
        hidden = hidden.reshape(-1, hidden.size(-1))
        for s in range(0, y.numel(), rows):
            ys = y[s:s + rows]
            logp = model.lm_head(hidden[s:s + rows]).float()
            logp -= torch.logsumexp(logp, dim=-1, keepdim=True)  # log-softmax in place
            nll += -logp.gather(-1, ys.unsqueeze(-1)).sum().item()
            top, pred = logp.max(-1)
            mean_prob += logp.exp_().sum(0).cpu().double()  # probabilities, in place (MPS has no float64)
            freq += torch.bincount(ys.cpu(), minlength=vocab).double()
            c = top.exp().cpu().double()
            ok = torch.isfinite(c)  # NaN weights (a diverged run) give NaN confidences; skip them
            right = (pred == ys).cpu().double()[ok]
            c = c[ok]
            which = (c.clamp(0, 1 - 1e-9) * bins).long().clamp(0, bins - 1)  # float64: 1.0 stays in the top bin
            conf_sum += torch.bincount(which, c, bins)
            acc_sum += torch.bincount(which, right, bins)
            count += torch.bincount(which, minlength=bins).double()
        n += y.numel()
    total = max(n, 1)
    nz = count > 0
    return {"mean_prob": (mean_prob / total).numpy(), "freq": (freq / total).numpy(), "loss": nll / total,
            "tokens": n, "calibration": {"confidence": (conf_sum / count.clamp(min=1)).tolist(),
                                          "accuracy": (acc_sum / count.clamp(min=1)).tolist(),
                                          "count": count.tolist()},
            "ece": float((conf_sum[nz] - acc_sum[nz]).abs().sum() / total)}


@torch.no_grad()
def attention_entropy(model, window, device, budget=2 ** 25, with_rms=False):
    """Mean entropy (bits) of each head's attention distribution over one window,
    computed for a few heads at a time (about `budget` values per heads x T x T table).
    with_rms=True also returns the RMS of the residual stream after the embedding and
    after every block (layers + 1 values) from the same pass."""
    x = window[None, :-1].to(device)
    h = model.token_embedding(x)
    out = []
    rms = [h.float().pow(2).mean().sqrt().item()]
    t = x.shape[1]
    denom = torch.log2(torch.arange(1, t + 1, device=device).float()).clamp(min=1e-9)
    mask = torch.ones(t, t, dtype=torch.bool, device=device).triu(1)
    group = max(1, budget // (t * t))
    for block in model.blocks:
        attn = block.attn
        q, k, _ = attn._qkv(block.attn_norm(h), model.rope)  # causal softmax(QK^T/sqrt(d))
        k = k.repeat_interleave(attn.n_rep, dim=1)
        ent = []
        for h0 in range(0, q.size(1), group):
            scores = (q[0, h0:h0 + group].float() @ k[0, h0:h0 + group].float().transpose(-1, -2))
            scores.div_(math.sqrt(attn.head_dim))
            logw = scores.masked_fill_(mask, float("-inf")).log_softmax(-1)
            del scores
            w = logw.exp()
            ent.append(-(w.mul_(logw.clamp_(min=-1e4))).sum(-1) / math.log(2))  # 0 * log 0 counts as 0
            del logw, w
        ent = torch.cat(ent)  # (H, T)
        out.append((ent[:, 1:] / denom[1:]).mean(-1).cpu().numpy())
        h = block(h, model.rope)
        if isinstance(h, tuple):  # a mixture-of-experts block returns (residual stream, aux terms)
            h = h[0]
        rms.append(h.float().pow(2).mean().sqrt().item())
    entropy = np.stack(out)  # (layers, heads), 0 = attends to one token, 1 = uniform over the prefix
    return (entropy, np.array(rms)) if with_rms else entropy


# ---------------------------------------------------------------------------
# History from log files (for checkpoints that do not store it)
# ---------------------------------------------------------------------------
NEW_LINE = re.compile(r"step\s+(\d+)/(\d+) \| train ([\d.]+) \| val ([\d.]+) @(\d+).*?\| lr ([\d.e+-]+)")
OLD_LINE = re.compile(r"step\s+(\d+) \| training loss ([\d.]+) \| validation loss ([\d.]+) \| perplexity "
                     r"([\d.,]+) \| lr ([\d.e+-]+)")


def history_from_log(path, ck_path=None, max_step=None, family=None):
    """History of the viewed checkpoint's run from log.txt. tinyGPT sessions are
    matched by the checkpoint path in their header ("checkpoints: ...<name>.pt"),
    and all sessions of that run (resumes included) are merged, the latest value
    per step winning, up to the checkpoint's step; sessions before the last fresh
    start under that name are another run and are left out. A tinyGPT checkpoint that no
    session names gets no curve (never another run's); a legacy (original script)
    checkpoint is read from the last v3 run only; a file of unknown family falls
    back to the last run, and the source label says so. v3 lines repeat a stale
    validation value between evaluations, so for v3 only evaluation steps keep a
    val point."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    blocks = re.split(r"\n(?=--- run started |={20,}\nRUN )", text)
    # path_name, not os.path.basename: an export's exported_from may be a Windows path read on the Mac
    stem = re.sub(r"_best$", "", os.path.splitext(path_name(ck_path))[0]) if ck_path else None
    legacy = bool(family) and family.startswith("legacy")
    tiny = [] if legacy else [b for b in blocks if NEW_LINE.search(b)]
    named = [b for b in blocks if stem and not legacy
             and re.search(r"checkpoints: .*?[\\/]" + re.escape(stem) + r"\.pt ", b)]
    # a fresh start under the same name (--overwrite, --init-from) begins a new curve, also when that session
    # ended before its first evaluation; a block without such a RUN line (older logs) is merged as before
    fresh = [i for i, b in enumerate(named) for run_line in [re.search(r"^RUN .*$", b, re.MULTILINE)]
             if run_line and re.search(r"\| (fresh run|initialised from )", run_line.group(0))]
    mine = [b for b in named[fresh[-1] if fresh else 0:] if NEW_LINE.search(b)]
    if fresh and not mine:
        return [], f"no evaluation in {os.path.basename(path)} since the last fresh start of {stem}.pt"
    if family == "tinygpt" and not mine:
        return [], f"no tinyGPT session in {os.path.basename(path)} names {stem}.pt"
    if mine or tiny:
        merged, sessions = {}, 0
        for block in (mine or tiny[-1:]):
            sessions += 1
            for s, _, tr, va, vs, lr in NEW_LINE.findall(block):
                if int(vs) == int(s) and (max_step is None or int(s) <= int(max_step)):
                    merged[int(s)] = {"step": int(s), "train_loss": float(tr), "val_loss": float(va), "lr": float(lr),
                                      "ppl": math.exp(min(float(va), 50))}
        if merged:
            label = (f"log ({sessions} session(s) of {stem})" if mine
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
                if max_step is not None and s > int(max_step):
                    break
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
    s = s if len(s) <= limit else s[:limit - 1] + "\u2026"
    return s.replace("$", r"\$")  # an even number of $ would be read as matplotlib mathtext


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


# The dashboard's matplotlib settings, at the sizes of --panels all (_draw replaces the sizes per mode).
BASE_STYLE = {"font.family": "DejaVu Sans", "axes.titlesize": 10, "axes.titleweight": "bold",
              "axes.labelsize": 8.5, "axes.labelcolor": INK2, "text.color": INK,
              "axes.titlecolor": INK, "figure.facecolor": SURFACE, "legend.fontsize": 7,
              "legend.frameon": False}
# Font sizes (points) per mode, read by titled, message, heatmap and the panels' notes. history: the default
# picture, legible when its 3,840 x 2,160 px fill a screen (nothing below 9.5 pt); all: the dense 26 in wide
# picture meant for zooming, also used for the token picture. wrap: characters per subtitle line.
SIZES = {
    "history": {"title": 14, "label": 11, "sub": 10, "tick": 10, "legend": 10, "note": 10, "message": 11, "heat": 10,
                "cell": 9.5, "head": 15, "head2": 10, "wrap": 58},
    "all": {"title": 10.5, "label": 8.5, "sub": 7.5, "tick": 8, "legend": 7, "note": 7.5, "message": 9, "heat": 7,
            "cell": 6, "head": 12, "head2": 8.5, "wrap": 92},
}
_SIZE = dict(SIZES["all"])  # the table of the picture being drawn (set by _draw)


def titled(ax, main, sub=None):
    """Bold panel title plus a wrapped grey subtitle: whole lines, never cut; the title moves up to make room."""
    lines = textwrap.wrap(scrub(sub), _SIZE["wrap"]) if sub else []
    ax.set_title(scrub(main), loc="left", fontsize=_SIZE["title"], pad=6 + 1.25 * _SIZE["sub"] * len(lines))
    if lines:
        ax.text(0, 1.012, "\n".join(lines), transform=ax.transAxes, fontsize=_SIZE["sub"], color=INK2, va="bottom",
                ha="left", linespacing=1.25)


def message(ax, text, title):
    titled(ax, title)
    ax.text(0.5, 0.5, scrub(text), ha="center", va="center", wrap=True, fontsize=_SIZE["message"], color=INK2,
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)


def style_axes(ax, grid=True, ticksize=8):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=ticksize, length=3, width=0.6)
    ax.grid(grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def heatmap(ax, matrix, rows, cols, cmap, label, vmin=None, vmax=None, annotate=True, fmt_spec=".3g"):
    ax.grid(False)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=_SIZE["heat"])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=_SIZE["heat"], rotation=0 if len(cols) <= 8 else 90)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(label, fontsize=_SIZE["tick"], color=INK2)
    cb.ax.tick_params(labelsize=_SIZE["heat"], colors=INK2)
    cb.outline.set_visible(False)
    if annotate and matrix.size <= 120:
        for (i, j), v in np.ndenumerate(matrix):
            if np.isfinite(v):
                pos = float(im.norm(v))  # where the cell's colour lies on the drawn scale (vmin and vmax included)
                dark = pos > 0.5 if cmap.name.startswith("seq") else abs(pos - 0.5) > 0.3
                ax.text(j, i, format(v, fmt_spec), ha="center", va="center", fontsize=_SIZE["cell"],
                        color="#ffffff" if dark else INK)
    return im


def train_config(ck):
    """The checkpoint's stored training settings ({} when it has none)."""
    return (ck.obj.get("train_config") if isinstance(ck.obj, dict) else None) or {}


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def rows_after(history, metrics_path):
    """The evaluations in <name>_metrics.jsonl that follow the last one of `history` (a checkpoint's stored
    rows), so a best file's curves reach the run's latest evaluation. The file is read once; lines that do not
    parse (a half-written last line) are skipped; rows are taken in file order and, when a row's step is not
    above the last kept one, the kept rows from that step on are dropped (a resumed session's rows replace the
    abandoned tail). [] unless some row has the history's last step with the same val_loss, which is what shows
    that the file continues this history (and not another run of the same name)."""
    if not history or not os.path.isfile(metrics_path):
        return []
    last = history[-1]
    try:
        with open(metrics_path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return []
    kept, anchored = [], False
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or not _finite(row.get("step")):
            continue
        if row["step"] == last.get("step") and row.get("val_loss") == last.get("val_loss"):
            anchored = True
        while kept and kept[-1]["step"] >= row["step"]:
            kept.pop()
        kept.append(row)
    return [row for row in kept if row["step"] > last["step"]] if anchored and _finite(last.get("step")) else []


def metrics_file(ck):
    """<folder>/<run>_metrics.jsonl beside a checkpoint file, the run's name being the file's without _best."""
    stem = re.sub(r"_best$", "", os.path.splitext(os.path.basename(ck.path))[0])
    return os.path.join(os.path.dirname(os.path.abspath(ck.path)), stem + "_metrics.jsonl")


def newer_rows(ck):
    """rows_after for the viewed file: only for a tinyGPT training or best checkpoint (a fine-tuned, edited or
    exported model has left its run). Read once per file."""
    cached = getattr(ck, "_newer", None)
    if cached is None:
        o = ck.obj if isinstance(ck.obj, dict) else {}
        own = (ck.family == "tinygpt" and os.path.isfile(ck.path) and not ck.finetune and not ck.edit
               and not o.get("exported_from"))
        cached = ck._newer = rows_after(ck.history, metrics_file(ck)) if own else []
    return cached


def dashboard_history(ck, args):
    """The evaluations the history panels draw: the checkpoint's stored history, else the --history-log curve,
    then the newer evaluations of the run's metrics file (newer_rows)."""
    history = ck.history
    if not history and args.history_log:
        # an --export folder is named in no log header: match the checkpoint it was exported from
        src = (ck.obj.get("exported_from") if isinstance(ck.obj, dict) else None) or ck.path
        history, ck.history_source = history_from_log(args.history_log, src,
                                                      ck.obj.get("step") if isinstance(ck.obj, dict) else None,
                                                      ck.family)
    return list(history) + newer_rows(ck)


def dashboard_header(ck, history=None, now=None):
    """The two header lines of a picture. Line 1: run, step, validation loss, bits per byte, fact score and when
    it was drawn, each only when the file has it (a file with none of them: its model name alone). Line 2:
    architecture and parameter count, whose history a fine-tuned or edited file shows, and how far the curves
    go when the metrics file added evaluations after the checkpoint's."""
    o = ck.obj if isinstance(ck.obj, dict) else {}
    history = ck.history if history is None else history
    last = history[-1] if history else {}
    stem = path_name(ck.path) if os.path.isdir(ck.path) else os.path.splitext(path_name(ck.path))[0]
    parts = []
    step, total = o.get("step"), train_config(ck).get("steps")
    beyond = _finite(step) and _finite(last.get("step")) and last["step"] > step
    shown = last["step"] if beyond else step
    if _finite(shown):
        parts.append(f"step {shown:,}"
                     + (f" of {total:,} ({shown / total:.0%})" if _finite(total) and total > 0 else "")
                     + (" (base model's pre-training)" if ck.finetune else derived_note(ck)))
    best, best_step = ck.metrics.get("best_val"), ck.metrics.get("best_step")
    best_text = f"best {float(best):.3f} @{best_step:,}" if _finite(best) and _finite(best_step) else ""
    if _finite(last.get("val_loss")):
        parts.append(f"val {last['val_loss']:.3f}" + (f" ({best_text})" if best_text else ""))
    elif best_text:
        parts.append(best_text.replace("best", "best val", 1))
    if _finite(last.get("bpb")):
        parts.append(f"{last['bpb']:.3f} bits/byte")
    if _finite(last.get("probe_acc")):
        parts.append(f"facts {last['probe_acc']:.0%}"
                     + (f" (floor {last['probe_floor']:.0%})" if _finite(last.get("probe_floor")) else ""))
    if not parts:
        return scrub(ck.model_name), ""
    drawn = time.strftime("drawn %Y-%m-%d %H:%M", now or time.localtime())
    line1 = " | ".join([re.sub(r"_best$", "", stem)] + parts + [drawn])
    line2 = ck.model_name + (f", {tiny_gpt.count_params(ck.cfg)[0]:,} parameters" if ck.cfg is not None else "")
    if ck.finetune:
        line2 += (f" | fine-tuned ({finetune_label(ck.finetune)}) from {path_name(ck.finetune.get('base'))}: "
                  "the pre-training history panels show the base model before fine-tuning")
    elif ck.edit:
        line2 += " | edited weights: the history panels show the" + derived_note(ck).replace(" (", " ").rstrip(")")
    if beyond:
        line2 += f" | weights: {'best ' if stem.endswith('_best') else ''}@{step:,}; curves to step {last['step']:,}"
    return scrub(line1), scrub(line2)


def create_dashboard(ck, args):
    """Draw what the command line asks for: the dashboard (--plot; nine history panels, or with --panels all
    the weight and behaviour panels too) and the token picture (--plot-tokens). The model is built and run only
    for --panels all and --plot-tokens, once for both. Returns the dashboard's path when it was written under
    its own name (the one picture that is pushed), else None."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import LinearSegmentedColormap

    warnings.filterwarnings("ignore", message="Glyph .* missing")
    seq = LinearSegmentedColormap.from_list("seq_blue", BLUES)
    div = LinearSegmentedColormap.from_list("div_blue_red", DIVERGING)
    history = dashboard_history(ck, args)
    everything = args.plot and args.panels == "all"
    wants_model = everything or args.plot_tokens

    # forward-pass material, computed once; the model is kept when only the windows or statistics fail,
    # so the fact benchmark (which needs much less memory) can still be scored
    model = fstats = windows = None
    fnote = ("add --panels all" if not wants_model
             else "forward-pass panels disabled (--no-forward)" if args.no_forward else "")
    device = torch.device("cpu")
    if wants_model and not args.no_forward and ck.cfg is not None and ck.tok is not None:
        try:
            device = torch.device("cpu") if args.device == "cpu" else tiny_gpt.select_device(args.device)
            model = build_model(ck, device)
        except SystemExit as exc:
            fnote = str(exc)
        except Exception as exc:  # noqa: BLE001 - a model that cannot be built only disables the forward panels
            fnote = f"model could not be built: {type(exc).__name__}: {str(exc)[:120]}"
        if model is not None:
            try:
                windows, fnote = eval_windows(ck, args)
                if windows is not None:
                    fstats = forward_stats(model, windows, ck.cfg.vocab_size, device)
            except SystemExit as exc:
                fstats = windows = None
                fnote = str(exc)
            except Exception as exc:  # noqa: BLE001 - a failed forward pass only disables the forward panels
                fstats = windows = None
                fnote = f"forward pass failed: {type(exc).__name__}: {str(exc)[:120]}"
    elif wants_model and not args.no_forward:
        fnote = "no tiny_gpt architecture/tokenizer in this file"

    panels = [
        ("Loss", lambda ax: panel_loss(ax, history, ck)),
        ("Validation minus training", lambda ax: panel_gap(ax, history, ck)),
        ("Gradient norm", lambda ax: panel_grad_norm(ax, history, ck)),
        ("Validation by source", lambda ax: panel_sources(ax, history, ck)),
        ("Biology facts", lambda ax: panel_facts(ax, history, ck, args)),
        ("Fact benchmark", lambda ax: panel_probes(ax, ck, model, device, args, fnote)) if everything else
        ("Biology facts by subject", lambda ax: panel_facts_subjects(ax, history, ck, args, fnote)),
        ("Improvement per 1,000 steps", lambda ax: panel_improvement(ax, history, ck)),
        ("Speed", lambda ax: panel_speed(ax, history, ck)),
        ("Plan", lambda ax: panel_plan(ax, history, ck)),
    ]
    if everything:
        panels += [
            ("Learning rate", lambda ax: panel_lr(ax, history, ck)),
            ("Calibration of the top prediction", lambda ax: panel_calibration(ax, ck, fstats, fnote)),
            ("Weight scale per block", lambda ax: panel_components(ax, ck, seq)),
            ("RMSNorm gains by depth", lambda ax: panel_norm_gains(ax, ck)),
            ("Singular value spectra", lambda ax: panel_spectra(ax, ck, args.full)),
            ("Per-head projection scale", lambda ax: panel_heads(ax, ck, seq)),
            ("Attention entropy", lambda ax: panel_attention(ax, model, windows, device, seq, fnote)),
        ]
    if ck.moe:
        panels += [
            ("Expert balance", lambda ax: panel_moe_balance(ax, history, ck)),
            ("Expert load per block", lambda ax: panel_moe_load(ax, ck, history, seq)),
            ("Router bias", lambda ax: panel_router_bias(ax, ck, div)),
        ]
    header = dashboard_header(ck, history)
    out = None
    if args.plot:
        if not everything and (ck.finetune or ck.edit or not history):
            print("weight and behaviour panels: add --panels all")
        # history: 3 columns on 25.6 x 14.4 in (3,840 x 2,160 px at the default 150 dpi); all: 4 columns, 26 in
        # wide; a mixture-of-experts file has one more row of the same height
        cols, width, row_height = (4, 26, 23 / 4) if everything else (3, 25.6, 14.4 / 3)
        target = args.out or output_name(ck, "dashboard", ".png")
        written = _draw(panels, cols, (width, row_height * math.ceil(len(panels) / cols)), SIZES[args.panels], target,
                        header, args.dpi)
        print(f"Saved dashboard: {written}")
        out = written if written == target else None  # a fallback name is not pushed
    if args.plot_tokens:
        pieces = ck.pieces()
        classes = [token_class(ck.tok, i) for i in range(ck.tok.vocab_size)] if ck.tok is not None else None
        token_panels = [
            ("Token embeddings (PCA)", lambda ax: panel_pca(ax, ck, pieces, classes, fstats, args.pca_labels)),
            ("Token similarity (clustered)",
             lambda ax: panel_similarity(ax, ck, pieces, classes, fstats, div, args.sim_tokens)),
            ("Token embedding norms", lambda ax: panel_norms(ax, ck, classes)),
            ("Output preference", lambda ax: panel_preference(ax, pieces, classes, fstats, fnote)),
        ]
        target = os.path.splitext(args.out)[0] + "_tokens.png" if args.out else output_name(ck, "tokens", ".png")
        print(f"Saved token panels: {_draw(token_panels, 4, (26, 6.4), SIZES['all'], target, header, args.dpi)}")
    return out


def _draw(panels, cols, figsize, sizes, out, header, dpi):
    """One picture: a two-line header, then the panels in rows of `cols`. One sub-figure per row: panels align
    within a row only, so a colour bar or long tick labels in one panel do not shrink a whole column. `sizes`
    is the mode's font table. A panel that fails says so in its place. Returns the path written (a time-stamped
    one when `out` is locked by another program)."""
    import matplotlib.pyplot as plt
    _SIZE.update(sizes)
    plt.rcParams.update({**BASE_STYLE, "axes.titlesize": sizes["title"], "axes.labelsize": sizes["label"],
                         "legend.fontsize": sizes["legend"]})
    n_rows = math.ceil(len(panels) / cols)
    head_height = 2.1 * (sizes["head"] + sizes["head2"]) / 72  # inches
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.1, h_pad=0.12)
    head, *rows = fig.subfigures(n_rows + 1, 1, hspace=0.015, squeeze=False, height_ratios=[
        head_height] + [(figsize[1] - head_height) / n_rows] * n_rows)[:, 0]
    axs = []
    for row in rows:
        row.set_facecolor(SURFACE)
        axs.extend(row.subplots(1, cols, squeeze=False, gridspec_kw={"wspace": 0.08})[0])
    for ax in axs:
        style_axes(ax, ticksize=sizes["tick"])
    for ax in axs[len(panels):]:
        ax.set_axis_off()
    for ax, (title, draw) in zip(axs, panels):
        try:
            draw(ax)
            if not ax.get_title(loc="left"):
                ax.set_title(title, loc="left")
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - one broken panel must not kill the dashboard
            ax.cla()
            message(ax, f"panel failed: {type(exc).__name__}: {scrub(exc)[:120]}", title)  # it may be published
            print(f"WARNING: dashboard panel '{title}' failed: {type(exc).__name__}: {exc}")
    head.set_facecolor(SURFACE)
    head.text(0.005, 0.5, header[0], fontsize=sizes["head"], fontweight="bold", color=INK, ha="left", va="bottom")
    head.text(0.005, 0.42, header[1], fontsize=sizes["head2"], color=INK2, ha="left", va="top")
    finalize_legends(fig)
    fmt_name = os.path.splitext(out)[1].lstrip(".").lower() or "png"
    written = save_replacing(lambda tmp: fig.savefig(tmp, dpi=dpi, facecolor=SURFACE, format=fmt_name), out)
    plt.close(fig)
    return written


def save_replacing(write, out):
    """write(tmp) to a temporary file next to `out`, then rename it over `out`, so
    nobody (an image viewer, a second viewer, the push) ever sees a half-written
    file. Retries briefly while Windows reports `out` as in use; if it stays
    locked, the file is kept under a time-stamped name, which is returned."""
    root, ext = os.path.splitext(out)
    tmp = f"{root}.tmp-{os.getpid()}{ext}"
    try:
        write(tmp)
        for attempt in range(6):
            try:
                os.replace(tmp, out)
                return out
            except PermissionError:
                time.sleep(0.5 * (attempt + 1))
        fallback = f"{root}.{time.strftime('%H%M%S')}{ext}"
        os.replace(tmp, fallback)
        print(f"WARNING: {out} is locked by another program; saved to {fallback} instead.")
        return fallback
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---- history panels (the default picture) -------------------------------------
def derived_note(ck):
    """Label for the numbers a fine-tuned or edited file carries over from the model it was made from."""
    if ck.finetune:
        return " (base model, before fine-tuning)"
    return " (original model, before the edit)" if ck.edit else ""


def _history_sub(ck, sub=None):
    """Subtitle plus where the curve came from (a log file) and, for a fine-tuned or
    edited file, that it is the base model's pre-training."""
    parts = [sub] if sub else []
    if ck.history_source and ck.history_source != "checkpoint":
        parts.append("curve from " + ck.history_source)
    if ck.finetune:
        parts.append(f"base model pre-training (before {finetune_label(ck.finetune)})")
    elif ck.edit:
        parts.append(derived_note(ck).strip(" ()"))
    return "; ".join(parts) or None


def _no_history(ax, ck, title):
    if ck.history_source and ck.history_source != "checkpoint":
        why = f"no history: {ck.history_source}"
    elif ck.family == "tinygpt":
        why = "this checkpoint stores no evaluation history yet"
    elif ck.family.startswith("legacy"):
        why = ("v3 checkpoints store only the latest and best validation loss.\n"
               "Pass --history-log log.txt to read the curve from the log.")
    else:
        why = "not a tiny_gpt training checkpoint"
    message(ax, why, title)


def _series(history, key):
    """(step, value) of the evaluations whose `key` is a finite number."""
    return [(h["step"], h[key]) for h in history or [] if _finite(h.get(key))]


def _log_y(ax):
    ax.set_yscale("log")
    ax.tick_params(which="minor", labelsize=_SIZE["tick"], colors=INK2)


def source_shares(ck):
    """The share of the training windows drawn from each source (TokenBatcher.share(): weight x tokens, from
    the stored --mix, --mix-alpha and the dataset's train-token counts), or None when the file does not say."""
    o = ck.obj if isinstance(ck.obj, dict) else {}
    counts = o["dataset"].get("train_tokens") if isinstance(o.get("dataset"), dict) else None
    if not isinstance(counts, dict) or not counts:
        return None
    tc = o.get("train_config") or {}
    try:
        weights = tiny_gpt.mix_weights(tc.get("mix"), counts, counts, float(tc.get("mix_alpha", 1.0)))
        raw = {name: max(float(weights.get(name, 0.0)), 0.0) * float(n) for name, n in counts.items()}
    except (SystemExit, TypeError, ValueError):  # a --mix naming a source this dataset does not have
        return None
    total = sum(raw.values())
    return {name: v / total for name, v in raw.items()} if total > 0 else None


def val_mix_of(row, ck):
    """A history row's validation loss weighted like the training mix: the stored val_mix, else the same mean
    from the row's per-source losses and the file's sampling shares (as evaluate's mix_mean: a source the
    shares do not know counts 0). None when the row or the file lacks what it takes."""
    if _finite(row.get("val_mix")):
        return float(row["val_mix"])
    per = row.get("val_per_source")
    share = source_shares(ck) if isinstance(per, dict) and per else None
    if not share:
        return None
    weights = {name: share.get(name, 0.0) for name in per}
    total = sum(weights.values())
    return float(sum(per[name] * weights[name] for name in per) / total) if total > 0 else None


def corpus_tokens(ck):
    """Training tokens of the sources the run samples (one 'epoch' of tokens sampled), or None."""
    share = source_shares(ck)
    if not share:
        return None
    counts = ck.obj["dataset"]["train_tokens"]
    return sum(n for name, n in counts.items() if share.get(name, 0) > 0)


def panel_finetune_loss(ax, ck):
    """The SFT/DPO curve stored in the fine-tuned file (finetune["history"], one point per epoch)."""
    ft = ck.finetune
    method = finetune_label(ft)
    rows = ft["history"]
    xs = [h.get("step", h.get("epoch")) for h in rows]
    tr = [(x, h["train_loss"]) for x, h in zip(xs, rows) if h.get("train_loss") is not None]
    va = [(x, h["val_loss"]) for x, h in zip(xs, rows) if h.get("val_loss") is not None]
    if tr:
        ax.plot(*zip(*tr), color=SERIES[0], lw=2, marker="o", ms=3.5, label="training (mean over the epoch)")
    if va:
        ax.plot(*zip(*va), color=SERIES[1], lw=2, marker="o", ms=3.5, label="held-out (after each epoch)")
        ax.annotate(f"{va[-1][1]:.3f}", va[-1], xytext=(-4, 8), textcoords="offset points", ha="right",
                    fontsize=_SIZE["note"], color=INK2)
    ax.set_xlabel(f"{method} optimizer step" if any(h.get("step") is not None for h in rows) else "epoch")
    ax.set_ylabel("cross-entropy of the answer tokens (nats)" if method == "SFT" else f"{method} loss")
    place_legend(ax)
    titled(ax, f"Fine-tuning loss ({method})",
           f"{len(rows)} epochs from {path_name(ft.get('base'))}; the other history panels show its pre-training")


# upper limit of the loss axis once five evaluations lie below it (the first ones start near 10) and while the
# newest one does
LOSS_TOP = 4.0


def panel_loss(ax, history, ck):
    """Training loss, validation at the training mix (val_mix_of) and the balanced validation mean that picks
    the best checkpoint. x is the tokens seen on a log axis when every row has them over more than a 20-fold
    range (then with marks at one corpus of tokens and at the planned end), else the step."""
    title = "Loss"
    if ck.finetune and ck.finetune.get("history"):
        return panel_finetune_loss(ax, ck)
    if not history:
        return _no_history(ax, ck, title)
    legacy = (ck.history_source or "").startswith("log (v3")
    tokens = [h.get("tokens") for h in history]
    positive = [t for t in tokens if _finite(t) and t > 0]
    by_tokens = all(_finite(t) for t in tokens) and len(positive) > 1 and max(positive) > 20 * min(positive)
    xs = [t / 1e9 for t in tokens] if by_tokens else [h["step"] for h in history]

    def points(values):  # a log axis has no place for the baseline's 0 tokens
        return [(x, v) for x, v in zip(xs, values) if _finite(v) and (x > 0 or not by_tokens)]

    mix = [val_mix_of(h, ck) for h in history]
    tr, vm = points([h.get("train_loss") for h in history]), points(mix)
    va = points([h.get("val_loss") for h in history])
    drawn = [p for series in (tr, vm, va) for p in series]
    if not drawn:
        return _no_history(ax, ck, title)
    few = {"marker": "o", "ms": 4} if len(history) < 3 else {}
    if tr:
        ax.plot(*zip(*tr), color=SERIES[0], lw=2, **few, label="training (one batch, v3 log)" if legacy
                else "training (mean since the previous evaluation)")
    if vm:
        ax.plot(*zip(*vm), color=SERIES[1], lw=2, **few, label="validation, training mix")
    if va:
        ax.plot(*zip(*va), color=SERIES[2], lw=2, **few, label="validation, sources equal; picks the best")
    notes = []
    if va:
        best = min(va, key=lambda p: p[1])
        ax.scatter([best[0]], [best[1]], s=90, marker="*", color=SERIES[2], edgecolor=SURFACE, lw=1.5, zorder=5)
        ax.annotate(f"best {best[1]:.3f}", best, xytext=(6, 8), textcoords="offset points",
                    fontsize=_SIZE["note"], color=INK2)
        notes.append(f"validation now {va[-1][1]:.3f} (perplexity {math.exp(min(va[-1][1], 50)):,.1f})")
    left = min(x for x, _ in drawn)
    # the cut is for the first descent: a run whose newest evaluation is above it again (unstable, diverging) is
    # drawn whole, so every curve's last point is always inside the axes
    newest = max(series[-1][1] for series in (tr, vm, va) if series)
    if (len([v for _, v in va if v <= LOSS_TOP]) >= 5 and max(v for _, v in drawn) > LOSS_TOP
            and newest <= LOSS_TOP):
        low = min(v for _, v in drawn if v <= LOSS_TOP)
        ax.set_ylim(low - 0.04 * (LOSS_TOP - low), LOSS_TOP)
        left = min(x for x, v in drawn if v <= LOSS_TOP)  # the axis starts at the first point it can show
        above = sum(1 for _, v in va if v > LOSS_TOP)
        notes.append(f"{above} evaluation{'s' if above != 1 else ''} above {LOSS_TOP:g} not shown")
    right = max(x for x, _ in drawn)
    if by_tokens:
        ax.set_xscale("log")
        ax.tick_params(which="minor", labelsize=_SIZE["tick"], colors=INK2)
        tc, last = train_config(ck), history[-1]
        marks = []
        if _finite(tc.get("steps")) and _finite(tc.get("tokens_per_step")) and tc["steps"] > last["step"]:
            marks.append((xs[-1] + (tc["steps"] - last["step"]) * tc["tokens_per_step"] / 1e9, "planned end"))
            right = max(right, marks[-1][0])
        corpus = corpus_tokens(ck)
        if corpus and left <= corpus / 1e9 <= right:
            marks.append((corpus / 1e9, "tokens sampled = corpus size"))
        for x, text in marks:  # vlines, not axvline: the panel's lines stay its three curves
            ax.vlines(x, 0, 1, transform=ax.get_xaxis_transform(), color=MUTED, lw=1, ls=":")
            ax.text(x, 0.98, text + " ", transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="top",
                    fontsize=_SIZE["note"], color=MUTED)
        ax.set_xlim(left / 1.12, right * 1.12)
        ax.set_xlabel("training tokens (billions, log)")
    else:
        if left > min(x for x, _ in drawn):
            ax.set_xlim(left=left - 0.02 * (right - left))
        ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (nats per token)")
    place_legend(ax)
    if legacy:
        notes.append("training: one batch per step; validation: random batches")
    if vm and any(m is not None and not _finite(h.get("val_mix")) for m, h in zip(mix, history)):
        notes.append("training mix: recomputed with today's sampling shares where a row stores none")
    titled(ax, title, _history_sub(ck, "; ".join(notes)))


def gap_points(history, ck):
    """(step, validation at the training mix minus training loss) of the evaluations that have both."""
    out = []
    for h in history or []:
        mix = val_mix_of(h, ck)
        if mix is not None and _finite(h.get("train_loss")):
            out.append((h["step"], mix - h["train_loss"]))
    return out


def panel_gap(ax, history, ck):
    title = "Validation minus training"
    if not history:
        return _no_history(ax, ck, title)
    pts = gap_points(history, ck)
    if len(pts) < 2:
        return message(ax, "Needs two evaluations with a training loss and a validation loss at the training mix\n"
                           "(per-source losses and the dataset's token counts).", title)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.plot(*zip(*pts), color=SERIES[0], lw=2, **({"marker": "o", "ms": 4} if len(pts) < 3 else {}))
    warm = train_config(ck).get("warmup_steps") or 0
    ys = [g for s, g in pts if s >= warm] or [g for _, g in pts]  # the first steps fall too fast to compare
    low, high = min(ys + [0.0]), max(ys + [0.0])
    pad = 0.08 * (high - low) or 0.01
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("nats per token")
    titled(ax, title, _history_sub(ck, f"now {pts[-1][1]:+.3f}; same source mix on both sides; above 0 = worse on "
                                       "held-out text"))


def panel_grad_norm(ax, history, ck):
    title = "Gradient norm"
    if not history:
        return _no_history(ax, ck, title)
    pts = [(s, v) for s, v in _series(history, "grad_norm") if v > 0]
    if not pts:
        return message(ax, "this history has no gradient norms", title)
    ax.plot(*zip(*pts), color=SERIES[0], lw=1.6, marker="o", ms=3)
    _log_y(ax)
    clip = train_config(ck).get("grad_clip")
    if _finite(clip) and clip > 0:
        ax.axhline(clip, color=INK2, lw=0.9, ls="--")
        ax.annotate(f"clipped at {clip:g}", (pts[0][0], clip), xytext=(2, 3), textcoords="offset points",
                    fontsize=_SIZE["note"], color=INK2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("L2 norm of the gradient (log)")
    titled(ax, title, _history_sub(ck, f"now {pts[-1][1]:.2f}; one step before each evaluation, before clipping"))


def panel_sources(ax, history, ck):
    """The last evaluation by source: bits per byte (nats per token when the row has no bytes), sorted, with
    the source's share of the training windows and the change over the last 20 evaluations."""
    title = "Validation by source"
    if not history:
        return _no_history(ax, ck, title)
    last = history[-1]
    key, unit = next(((k, u) for k, u in (("bpb_per_source", "bits per byte"), ("val_per_source", "nats per token"))
                      if isinstance(last.get(k), dict) and last[k]), (None, None))
    if key is None:
        return message(ax, "the last evaluation has no per-source values", title)
    now = {name: v for name, v in last[key].items() if _finite(v)}
    if len(now) < 2:
        return message(ax, "one source: its loss is the Loss panel's", title)
    back = min(20, len(history) - 1)
    then = history[-1 - back].get(key) if back else None
    then = then if isinstance(then, dict) else {}
    share = source_shares(ck) or {}
    names = sorted(now, key=now.get, reverse=True)  # drawn bottom to top: the lowest value on top
    ax.barh(range(len(names)), [now[n] for n in names], color=SERIES[0], height=0.62)
    for i, name in enumerate(names):
        change = f" ({round(now[name] - then[name], 3) + 0.0:+.3f})" if _finite(then.get(name)) else ""
        ax.annotate(f"{now[name]:.3f}{change}", (now[name], i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=_SIZE["note"], color=INK2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} ({'<0.1%' if 0 < share[n] < 0.001 else format(share[n], '.1%')} of training)"
                        if n in share else n for n in names], fontsize=_SIZE["tick"])
    ax.set_xlim(0, max(now.values()) * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"{unit} (lower is better)")
    titled(ax, title, _history_sub(ck, f"at step {last['step']:,}" + (
        f"; in brackets after a value: change over the last {back} evaluation{'s' if back != 1 else ''}"
        if then else "")))


def trailing(values, k, mean=np.mean):
    """For each value the mean (or `mean`) of it and the up to k - 1 values before it."""
    return [float(mean(values[max(0, i - k + 1):i + 1])) for i in range(len(values))]


def stored_probes(ck, args, raw_weights=False):
    """The fact-benchmark report training stored for these weights (probe_report without a model), or None.
    raw_weights: for a panel that draws the history's scores, as probe_report."""
    try:
        return (probe_report(ck, None, None, args, raw_weights) if args.probes and os.path.isfile(args.probes)
                else None)
    except (OSError, SystemExit):
        return None


def floor_colour(group):
    """Bar colour of one group of probes: green or red only when its 95 % interval excludes its floor
    (accuracy with the question words shuffled), grey when the interval includes it or is not known."""
    if not group or not all(_finite(group.get(k)) for k in ("low", "high", "floor")):
        return OTHER
    return GOOD if group["low"] > group["floor"] else CRITICAL if group["high"] < group["floor"] else OTHER


def panel_facts(ax, history, ck, args):
    """Fact-probe accuracy and its floor at every evaluation: faint points, and the mean of the last five as
    the line (one evaluation of a few hundred probes moves by chance)."""
    from matplotlib.ticker import FuncFormatter
    title = "Biology facts"
    acc, floor = _series(history, "probe_acc"), _series(history, "probe_floor")
    report = stored_probes(ck, args, raw_weights=bool(acc))  # without a curve it would be shown as this file's score
    derived = ck.finetune or ck.edit
    if not acc:
        if report:
            low, high = tiny_gpt.wilson_interval(report["correct"], report["n"])
            return message(ax, f"{report['correct'] / max(report['n'], 1):.0%} correct ({low:.0%}-{high:.0%}) of "
                               f"{report['n']} probes at step {ck.obj.get('step')}"
                               + (f", floor {report['floor']:.0%}" if _finite(report.get("floor")) else "")
                               + ".\nThis file has no score per evaluation to draw a curve from.", title)
        return message(ax, "No fact-benchmark scores in this file's history.\n--panels all scores the probes now.",
                       title)
    k, now = 5, []
    for pts, color, label in ((acc, SERIES[0], "correct answer preferred"),
                              (floor, SERIES[1], "floor: question words shuffled")):
        if not pts:
            continue
        xs, ys = zip(*pts)
        if len(ys) >= k:
            ax.scatter(xs, ys, s=12, color=color, alpha=0.35, linewidths=0)
            line = trailing(ys, k)
            ax.plot(xs, line, color=color, lw=2, label=f"{label} (mean of {k})")
        else:
            line = list(ys)
            ax.plot(xs, line, color=color, lw=2, marker="o", ms=3.5, label=label)
        now.append(line[-1])
    values = [v for pts in (acc, floor) for _, v in pts]
    if report and _finite(report.get("chance")):
        ax.axhline(report["chance"], color=MUTED, lw=0.8, ls=":")
        ax.annotate(f"chance {report['chance']:.0%}", (acc[0][0], report["chance"]), xytext=(2, 3),
                    textcoords="offset points", fontsize=_SIZE["note"], color=MUTED)
        values.append(report["chance"])
    pad = 0.12 * (max(values) - min(values)) or 0.02
    ax.set_ylim(max(0.0, min(values) - pad), min(0.999, max(values) + pad))
    # as many decimals as the tick has (15%, 15.5%): whole percents would repeat when the scores lie close together
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{round(100 * v, 2):g}%"))
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("probes answered correctly")
    place_legend(ax)
    notes = []
    steps = np.diff([v for _, v in acc])
    if len(steps) >= 2:
        notes.append(f"one evaluation moves about ±{100 * float(np.std(steps, ddof=1)) / math.sqrt(2):.1f} "
                     "points by chance")
    if report:
        by = report.get("difficulty") or {}
        notes.append(", ".join(f"{d} {by[d]['accuracy']:.0%}" for d in ("easy", "medium", "hard") if d in by))
    if derived:
        notes.append("these weights: --panels all")
    titled(ax, f"{title}: {now[0]:.0%} right" + (f", floor {now[1]:.0%}" if len(now) > 1 else ""),
           _history_sub(ck, "; ".join(n for n in notes if n)))


def panel_facts_subjects(ax, history, ck, args, fnote=""):
    """Fact-probe accuracy per subject, the mean of the last five evaluations; a bar is green or red only when
    the stored report of the last evaluation puts its 95 % interval clear of the subject's floor."""
    from matplotlib.ticker import PercentFormatter
    title = "Biology facts by subject"
    rows = [h["probe_by_category"] for h in history or [] if isinstance(h.get("probe_by_category"), dict)
            and h["probe_by_category"]][-5:]
    if not rows:
        return panel_probes(ax, ck, None, None, args, fnote)  # the stored report's bars, or what it takes
    means = {}
    for name in sorted({n for r in rows for n in r}):
        got = [r[name] for r in rows if _finite(r.get(name))]
        if got:
            means[name] = float(np.mean(got))
    report = stored_probes(ck, args, raw_weights=True)
    groups = report["category"] if report and set(report.get("category") or {}) == set(means) else {}
    names = sorted(means, key=means.get)
    y = np.arange(len(names))
    ax.barh(y, [means[n] for n in names], color=[floor_colour(groups.get(n)) for n in names], height=0.62, alpha=0.9)
    for i, name in enumerate(names):
        ax.annotate(f"{means[name]:.0%}", (means[name], i), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=_SIZE["note"], color=INK2)
    floors = [(groups[n]["floor"], i) for i, n in enumerate(names) if n in groups and _finite(groups[n].get("floor"))]
    if floors:
        ax.scatter(*zip(*floors), marker="|", s=140, color=INK, lw=2, zorder=4,
                   label="floor (question words shuffled)")
        place_legend(ax)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} (n={groups[n]['n']})" if n in groups else n for n in names], fontsize=_SIZE["tick"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("probes of the subject answered correctly")
    colours = ("; green or red: the last evaluation's 95% interval is clear of the subject's floor" if groups
               else "")
    titled(ax, title, _history_sub(ck, f"mean of the last {len(rows)} evaluation{'s' if len(rows) != 1 else ''}"
                                       + colours))


def _rate(v):
    """A loss change per 1,000 steps as text: four decimals, or powers of ten below that."""
    return f"{v:.4f}" if abs(v) >= 1e-4 else f"{v:.1e}"


def improvement_width(points, window=1000):
    """Steps in each of the two groups of evaluations that improvement_points compares: `window`, or the usual
    distance between evaluations when that is larger (--eval-every above 1,000: one evaluation a group)."""
    gaps = np.diff(sorted({s for s, _ in points}))
    return max(window, int(np.median(gaps))) if len(gaps) else window


def improvement_points(points, start=0, window=1000):
    """(step s, fall of the loss per `window` steps) at the evaluations with s >= start + 2 x width, width from
    improvement_width: the mean over the evaluations in (s - 2 x width, s - width] minus the mean over those
    in (s - width, s], over the distance between the two groups' mean steps (`window` steps when the
    evaluations divide it evenly; any other --eval-every would give the fall over more steps than it says)."""
    width = improvement_width(points, window)
    ordered = sorted(points, key=lambda point: point[0])  # by step: each group is one slice, found by bisection
    steps = [t for t, _ in ordered]
    out = []
    for s, _ in points:
        if s < start + 2 * width:
            continue
        first, middle, last = (bisect.bisect_right(steps, s - k * width) for k in (2, 1, 0))
        before, recent = ordered[first:middle], ordered[middle:last]
        if before and recent:
            apart = np.mean([t for t, _ in recent]) - np.mean([t for t, _ in before])
            fall = np.mean([v for _, v in before]) - np.mean([v for _, v in recent])
            out.append((s, float(fall * (window / apart))))
    return out


def panel_improvement(ax, history, ck):
    title = "Improvement per 1,000 steps"
    if not history:
        return _no_history(ax, ck, title)
    warm = int(train_config(ck).get("warmup_steps") or 0)
    mix = [(h["step"], m) for h in history for m in [val_mix_of(h, ck)] if m is not None]
    hidden = drew = 0
    for label, color, pts in (("validation, training mix", SERIES[1], mix),
                              ("validation, sources equal", SERIES[2], _series(history, "val_loss"))):
        rate = improvement_points(pts, warm)
        if len(rate) < 4:
            continue
        hidden += sum(1 for _, v in rate if v <= 0)
        ax.plot([s for s, _ in rate], [v if v > 0 else np.nan for _, v in rate], color=color, lw=2, label=label)
        if rate[-1][1] > 0:
            ax.annotate(_rate(rate[-1][1]), rate[-1], xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=_SIZE["note"], color=INK2)
        drew += 1
    width = improvement_width(_series(history, "val_loss"))  # 1,000, or more when the evaluations are further apart
    if not drew:
        return message(ax, f"Drawn once four evaluations lie at or after step {2 * width + warm:,}:\nthe fall of "
                           f"the validation loss from one {width:,} steps to the next.", title)
    _log_y(ax)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("nats per 1,000 steps (log)")
    place_legend(ax)
    titled(ax, title, _history_sub(ck, f"mean validation loss of the {width:,} steps before, minus that of the last "
                                       f"{width:,}" + (", per 1,000 steps" if width != 1000 else "")
                                       + (f"; {hidden} value{'s' if hidden != 1 else ''} at or below 0 not drawn"
                                          if hidden else "")))


def panel_speed(ax, history, ck):
    title = "Speed"
    if not history:
        return _no_history(ax, ck, title)
    pts = [(s, v) for s, v in _series(history, "tok_s") if v > 0]
    if not pts:
        return message(ax, "this history has no training speed (tok_s)", title)
    xs, ys = zip(*pts)
    for a, b in zip(history, history[1:]):  # a stop and resume (only the metrics file's rows carry the run id)
        if a.get("run_id") and b.get("run_id") and a["run_id"] != b["run_id"]:
            ax.vlines(b["step"], 0, 1, transform=ax.get_xaxis_transform(), color=GRID, lw=1.2, zorder=0)
    ax.scatter(xs, ys, s=14, color=SERIES[0], alpha=0.5, linewidths=0)
    median = trailing(ys, 10, np.median)
    ax.plot(xs, median, color=SERIES[0], lw=2)
    ax.set_ylim(0, max(ys) * 1.1)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("tokens per second")
    last = history[-1]
    notes = [f"median of the last {min(10, len(ys))}: {median[-1]:,.0f} tok/s between evaluations"]
    if _finite(last.get("peak_mem_gib")):
        notes.append(f"peak memory {last['peak_mem_gib']:.1f} GiB")
    notes += _used(last)
    titled(ax, title, _history_sub(ck, "; ".join(notes)))


def _used(row):
    """['12.3 kWh', '£3.07'] of a history row, each only when the row has it."""
    out = []
    energy = row.get("energy_wh")
    if isinstance(energy, dict) and energy and all(_finite(v) for v in energy.values()):
        out.append(f"{sum(energy.values()) / 1000:.1f} kWh")
    if _finite(row.get("cost_gbp")):
        out.append(f"£{row['cost_gbp']:.2f}")
    return out


def _hours_per_day(history):
    """(training hours per day over the last 7 days of the history, time of the last evaluation), or None
    unless the rows' time stamps span 7 days."""
    stamped = []
    for h in history:
        try:
            stamped.append((datetime.datetime.fromisoformat(h["time"]), float(h["train_seconds"])))
        except (KeyError, TypeError, ValueError):
            continue
    try:
        if len(stamped) < 2 or (stamped[-1][0] - stamped[0][0]).total_seconds() < 7 * 86400:
            return None
        week = [(t, s) for t, s in stamped if (stamped[-1][0] - t).total_seconds() <= 7 * 86400]
        days = (week[-1][0] - week[0][0]).total_seconds() / 86400
    except TypeError:  # time stamps with and without a time zone
        return None
    hours = (week[-1][1] - week[0][1]) / 3600
    return (hours / days, stamped[-1][0]) if days > 0 and hours > 0 else None


def plan_lines(history, ck):
    """The Plan tile as (headline, detail) pairs; a pair, or a part of one, whose inputs the file lacks is left
    out: progress, the losses, the rate of improvement, the time left, energy and cost, the learning rate."""
    tc, last = train_config(ck), history[-1]
    step, total, per_step = last.get("step"), tc.get("steps"), tc.get("tokens_per_step")
    planned = _finite(step) and _finite(total) and total > 0
    out = []
    if _finite(step):
        corpus, tokens = corpus_tokens(ck), last.get("tokens")
        detail = ""
        if corpus and _finite(tokens):
            ahead = (f" of {(tokens + (total - step) * per_step) / corpus:.2f}"
                     if planned and _finite(per_step) and total >= step else "")
            detail = f"epoch {tokens / corpus:.2f}{ahead} (tokens sampled / corpus size)"
        out.append((f"step {step:,}" + (f" of {total:,} ({step / total:.1%})" if planned else ""), detail))
    if _finite(last.get("val_loss")):
        best, best_step = ck.metrics.get("best_val"), ck.metrics.get("best_step")
        mix = val_mix_of(last, ck)
        out.append((f"validation {last['val_loss']:.3f}"
                    + (f" (best {float(best):.3f} @{best_step:,})" if _finite(best) and _finite(best_step) else ""),
                    " | ".join(([f"at the training mix {mix:.3f}"] if mix is not None else [])
                               + ([f"training {last['train_loss']:.3f}"] if _finite(last.get("train_loss")) else []))))
    scores = _series(history, "val_loss")
    rate = improvement_points(scores, int(tc.get("warmup_steps") or 0))
    if rate:
        gain, detail, width = rate[-1][1], "", improvement_width(scores)
        timed = [(h["step"], h["train_seconds"]) for h in history
                 if _finite(h.get("train_seconds")) and _finite(step) and h["step"] >= step - width]
        if len(timed) > 1 and timed[-1][1] > timed[0][1]:
            per_day = gain / 1000 * (timed[-1][0] - timed[0][0]) / (timed[-1][1] - timed[0][1]) * 86400
            detail = f"{per_day:.3f} per 24 h of training"
        out.append((f"improving {_rate(gain)} per 1,000 steps" if gain > 0
                    else f"not improving (validation up {_rate(-gain)} per 1,000 steps)" if gain < 0
                    else "not improving (validation flat: 0 per 1,000 steps)", detail))
    speeds = [v for _, v in _series(history, "tok_s") if v > 0][-10:]
    if speeds and planned and _finite(per_step) and total > step:
        hours = (total - step) * per_step / float(np.median(speeds)) / 3600
        detail = f"at {float(np.median(speeds)):,.0f} tok/s (median of the last {len(speeds)} evaluations)"
        duty = _hours_per_day(history)
        if duty:
            done = duty[1] + datetime.timedelta(days=hours / duty[0])
            detail += f"; at {duty[0]:.1f} h a day (last 7 days) done about {done:%Y-%m-%d}"
        out.append((f"about {hours:,.0f} h of training left", detail))
    if _used(last):
        out.append((", ".join(_used(last)) + " used so far", ""))
    if _finite(last.get("lr")):
        detail = ""
        if planned and all(k in tc for k in ("schedule", "lr", "min_lr", "warmup_steps")):
            start = tiny_gpt.LRSchedule(tc["schedule"], total, tc["lr"], tc["min_lr"], tc["warmup_steps"],
                                        tc.get("decay_frac", 0.2), tc.get("decay_shape", "1-sqrt")).decay_start
            detail = f"decay starts at step {start:,}" if step < start else f"decaying since step {start:,}"
        out.append((f"learning rate {last['lr']:.2e}", detail))
    return out


def panel_plan(ax, history, ck):
    """A text tile: where the run is, what it has reached, how fast it still improves and what is left."""
    title = "Plan"
    if not history:
        return _no_history(ax, ck, title)
    ax.set_axis_off()
    lines = plan_lines(history, ck)
    for i, (headline, detail) in enumerate(lines):
        y = 0.97 - i / max(len(lines), 6)
        ax.annotate(scrub(headline), (0, y), xycoords="axes fraction", va="top", fontsize=_SIZE["title"], color=INK)
        if detail:
            ax.annotate(scrub(detail), (0, y), xycoords="axes fraction", xytext=(0, -1.5 * _SIZE["title"]),
                        textcoords="offset points", va="top", fontsize=_SIZE["sub"], color=INK2)
    titled(ax, title, _history_sub(ck))


def panel_lr(ax, history, ck):
    title = "Learning rate"
    tc = train_config(ck)
    drew = False
    if tc.get("steps"):
        sched = tiny_gpt.LRSchedule(tc["schedule"], tc["steps"], tc["lr"], tc["min_lr"], tc["warmup_steps"],
                                    tc.get("decay_frac", 0.2), tc.get("decay_shape", "1-sqrt"))
        xs = np.unique(np.linspace(1, tc["steps"], 400).astype(int))
        ax.plot(xs, [sched.lr_at(int(x)) for x in xs], color=MUTED, lw=1.2, label="planned schedule")
        # the last evaluation drawn: later than the file's step when the metrics file added rows
        now = max([ck.obj.get("step", 0)] + [h["step"] for h in (history or [])[-1:]])
        ax.axvline(now, color=INK2, lw=0.8)
        ax.annotate(f"now: step {now:,}", (now, sched.peak), xytext=(4, -10),
                    textcoords="offset points", fontsize=_SIZE["note"], color=INK2)
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
    titled(ax, title, _history_sub(ck))


# ---- token panels (--plot-tokens) ------------------------------------------------
def _token_order(classes, fstats, want):
    """Most frequent tokens of the wanted classes (eval-text frequency, else low
    merge ID, which in BPE means an early/frequent merge)."""
    ids = [i for i, c in enumerate(classes) if c in want]
    if fstats is not None:
        ids.sort(key=lambda i: -fstats["freq"][i])
    return ids


def place_labels(ax, points, texts, fontsize=6.5, max_labels=None):
    """Annotate points without overlapping labels: try a few offsets around each
    point and skip the label if every position collides with one already placed."""
    renderer = _root_figure(ax.figure).canvas.get_renderer()
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


def embedding_pca(ck, n):
    """(coordinates on the first two principal components, their shares of the
    variance) of the first n embedding rows, from the top eigenvectors of the
    d x d covariance: no (vocabulary x d) SVD. Computed once (the --html explorer
    uses the same); each axis points to its largest-magnitude loading."""
    cached = getattr(ck, "_pca", None)
    if cached is not None and cached[0] == n:
        return cached[1], cached[2]
    e = ck.embedding.float()[:n]
    centered = e - e.mean(0, keepdim=True)
    cov = (centered.T @ centered).double()
    vals, vecs = torch.linalg.eigh(cov)  # ascending
    top = vecs[:, -2:].flip(-1)
    top = top * torch.sign(top.gather(0, top.abs().argmax(0, keepdim=True)))
    coords = (centered @ top.float()).numpy()
    total = float(cov.diagonal().sum())  # total variance = squared Frobenius norm of the centred rows
    var = (vals[-2:].flip(0).clamp(min=0) / max(total, 1e-30)).numpy()
    ck._pca = (n, coords, var)
    return coords, var


def panel_pca(ax, ck, pieces, classes, fstats, n_labels=50):
    title = "Token embeddings (PCA)"
    if ck.embedding is None or classes is None:
        return message(ax, "no token embedding or tokenizer in this file", title)
    coords, var = embedding_pca(ck, len(classes))
    counts = {}
    for k, cls in enumerate(TOKEN_CLASSES + ["other"]):
        idx = [i for i, c in enumerate(classes) if c == cls]
        if not idx:
            continue
        color = SERIES[k] if k < len(SERIES) else OTHER
        ax.scatter(coords[idx, 0], coords[idx, 1], s=9, marker=MARKERS[k] if k < len(MARKERS) else ".",
                   color=color, alpha=0.65, linewidths=0, label=f"{cls} ({len(idx):,})")
        counts[cls] = len(idx)
    # candidates beyond n_labels, because labels that would overlap one already placed are skipped
    top = _token_order(classes, fstats, {"word start", "word piece", "number", "punctuation"})[:3 * n_labels + 40]
    shown = place_labels(ax, [coords[i] for i in top], [label_text(pieces[i]) for i in top],
                         fontsize=6.5 if n_labels <= 25 else 5.5, max_labels=n_labels) if n_labels else 0
    ax.set_xlabel(f"PC 1 ({var[0]:.1%} of variance)")
    ax.set_ylabel(f"PC 2 ({var[1]:.1%} of variance)")
    place_legend(ax, markerscale=1.6, fontsize=6.5)
    which = ("most frequent tokens (in the evaluation windows)" if fstats is not None
             else "earliest-merged tokens (lowest IDs; no forward pass)")
    titled(ax, title, f"one dot per token; {shown} {which} labelled; \u2581 marks a word start")


def panel_similarity(ax, ck, pieces, classes, fstats, cmap, n_tokens=100):
    title = "Token similarity (clustered)"
    if ck.embedding is None or classes is None:
        return message(ax, "no token embedding or tokenizer in this file", title)
    ids = [i for i in _token_order(classes, fstats, {"word start", "word piece"})
           if len(pieces[i].replace("\u2581", "")) >= 3][:n_tokens]
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
    except ImportError:
        order = np.argsort(np.linalg.eigh(sims)[1][:, -2])
    sims = sims[np.ix_(order, order)]
    labels = [label_text(pieces[ids[k]], 10) for k in order]
    heatmap(ax, sims, labels, labels, cmap, "cosine similarity", -1, 1, annotate=False)
    tick_size = min(4.5, 300 / len(ids))   # 4.5 pt up to ~65 tokens, then smaller so 100 rows still fit the panel
    ax.set_xticklabels(labels, fontsize=tick_size, rotation=90)
    ax.set_yticklabels(labels, fontsize=tick_size)
    ax.set_xlabel("token (same clustered order as rows)")
    ax.set_ylabel("token")
    which = ("most frequent word tokens (in the evaluation windows)" if fstats is not None
             else "earliest-merged word tokens (lowest IDs; no forward pass)")
    titled(ax, title, f"{len(ids)} {which}, clustered by cosine similarity")


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
    titled(ax, title, "tied: each row is also the token's output weights" if tied else None)


def panel_preference(ax, pieces, classes, fstats, fnote):
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
    titled(ax, title, "above the diagonal = predicted more often than it occurs")


# ---- weights (--panels all) ----------------------------------------------------
COMPONENTS = [("attn.q_proj", "attn.q_proj.weight"), ("attn.k_proj", "attn.k_proj.weight"),
              ("attn.v_proj", "attn.v_proj.weight"), ("attn.o_proj", "attn.o_proj.weight"),
              ("ffn.gate", "ffn.gate.weight"), ("ffn.up", "ffn.up.weight"), ("ffn.down", "ffn.down.weight")]


def _layers(ck):
    blocks = {int(m.group(1)) for n in ck.weights for m in [re.match(r"blocks\.(\d+)\.", n)] if m}
    return sorted(blocks)


def components(ck):
    """The rows of the per-block panels as (label, [tensor names relative to
    blocks.N.]): the dense set (COMPONENTS), or for a mixture-of-experts checkpoint
    the attention projections, the router and one row per expert (its gate_up and
    down matrices together), the shared SwiGLU if there is one, and the dense FFN
    rows when only every m-th block has experts. Rows that no block has are left
    out, so the heat maps draw no all-NaN rows."""
    rows = [(c, [rel]) for c, rel in COMPONENTS]
    cfg = ck.cfg
    if ck.moe:
        rows = rows[:4] + [("ffn.router", ["ffn.router.weight"])]
        rows += [(f"expert {e}", [f"ffn.experts.gate_up.{e}", f"ffn.experts.down.{e}"]) for e in range(cfg.n_experts)]
        if cfg.shared_hidden:
            rows.append(("ffn.shared", ["ffn.shared.gate.weight", "ffn.shared.up.weight", "ffn.shared.down.weight"]))
        if cfg.n_moe_blocks < cfg.n_layers:
            rows += [(c, [rel]) for c, rel in COMPONENTS[4:]]
    layers = _layers(ck)
    return [(c, rels) for c, rels in rows
            if any(f"blocks.{b}.{r}" in ck.weights for b in layers for r in rels)]


def _rms_of(ck, b, rels):
    """RMS over all of block b's tensors named in rels (NaN when none is present)."""
    sq, n = 0.0, 0
    for r in rels:
        t = ck.weights.get(f"blocks.{b}.{r}")
        if t is not None:
            sq += t.float().square().sum().item()
            n += t.numel()
    return math.sqrt(sq / n) if n else np.nan


def panel_components(ax, ck, cmap):
    title = "Weight scale per block"
    layers = _layers(ck)
    if not layers:
        return message(ax, "no Transformer blocks found", title)
    comps = components(ck)
    m = np.full((len(comps), len(layers)), np.nan)
    for j, b in enumerate(layers):
        for i, (_, rels) in enumerate(comps):
            m[i, j] = _rms_of(ck, b, rels)
    heatmap(ax, m, [c for c, _ in comps], [f"block {b}" for b in layers], cmap, "RMS of weights")
    titled(ax, title, "RMS of each weight matrix" + ("; an expert row is its gate_up and down together" if ck.moe
                                                    else ""))


NORM_KINDS = ("attn_norm", "ffn_norm", "q_norm", "k_norm")


def norm_gain_profiles(ck):
    """{kind: [(block, 10th percentile, median, 90th percentile of the gain vector)]} by depth, for the kinds of
    RMSNorm the file has (q_norm and k_norm only with QK-norm)."""
    out = {}
    for name, t in ck.weights.items():
        m = re.match(r"blocks\.(\d+)\.(?:attn\.)?(attn_norm|ffn_norm|q_norm|k_norm)\.weight$", name)
        if m:
            low, mid, high = np.quantile(t.float().numpy(), [0.1, 0.5, 0.9])
            out.setdefault(m.group(2), []).append((int(m.group(1)), float(low), float(mid), float(high)))
    return {kind: sorted(rows) for kind, rows in out.items()}


def panel_norm_gains(ax, ck):
    from matplotlib.ticker import MaxNLocator
    title = "RMSNorm gains by depth"
    profiles = norm_gain_profiles(ck)
    final = ck.weights.get("norm.weight")
    if not profiles and final is None:
        return message(ax, "no RMSNorm gains found", title)
    ax.axhline(1.0, color=MUTED, lw=0.8)
    for k, kind in enumerate(NORM_KINDS):
        if kind in profiles:
            blocks, low, mid, high = zip(*profiles[kind])
            ax.fill_between(blocks, low, high, color=SERIES[k], alpha=0.18, linewidth=0)
            ax.plot(blocks, mid, color=SERIES[k], lw=1.8, marker="o", ms=3, label=kind)
    if final is not None:
        x = max((rows[-1][0] for rows in profiles.values()), default=-1) + 1
        y = float(final.float().median())
        ax.scatter([x], [y], s=36, color=INK2, zorder=4)
        ax.annotate("final norm", (x, y), xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=_SIZE["note"], color=INK2)
    low, high = ax.get_ylim()
    low = min(0.0, low)  # from zero: a gain's size is read off the axis, not only its spread
    ax.set_ylim(low, high + 0.1 * (high - low))  # room above the highest gain for the final norm's label
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax.set_xlabel("block")
    ax.set_ylabel("gain per channel (1.0 at initialisation)")
    place_legend(ax)
    titled(ax, title, "line: median over the channels; band: 10th to 90th percentile; q_norm and k_norm (QK-norm) "
                      "set how sharp attention is")


SPECTRA_MAX_D = 2048  # wider models: the full SVDs take minutes on the CPU, so only with --full


def panel_spectra(ax, ck, full=False):
    title = "Singular value spectra"
    if ck.cfg is not None and ck.cfg.d_model > SPECTRA_MAX_D and not full:
        return message(ax, f"skipped for d_model {ck.cfg.d_model:,} > {SPECTRA_MAX_D:,} (slow); pass --full", title)
    layers = _layers(ck)
    mid = layers[len(layers) // 2] if layers else None
    targets = [("token embedding", ck.embedding)] + [
        (f"block {mid} / {c}", ck.weights.get(f"blocks.{mid}.{rels[0]}")) for c, rels in components(ck)] \
        if mid is not None else [("token embedding", ck.embedding)]
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
    titled(ax, title, "a fast drop means low effective rank")


def panel_heads(ax, ck, cmap):
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
    heatmap(ax, np.array(data), rows, [f"block {b}" for b in layers], cmap, "RMS of head slice",
            annotate=len(rows) * len(layers) <= 120)
    ax.set_yticklabels(rows, fontsize=5.5 if len(rows) > 24 else 7)
    share = c.n_heads // c.n_kv_heads
    note = f"GQA: query heads {share}k..{share}k+{share - 1} share key/value head k" if share > 1 else "MHA"
    titled(ax, title, note)


# ---- behaviour (--panels all) ----------------------------------------------------
def holds_ema(ck):
    """Whether the file's weights are the EMA average: its stored evaluation (calibration, fact benchmark,
    latest validation) measured the raw weights, which the file does not hold."""
    return isinstance(ck.obj, dict) and ck.obj.get("weights") == "ema"


def panel_calibration(ax, ck, fstats, fnote):
    """The reliability diagram: from the run's own evaluation when it measured these weights at this step (all
    of --eval-tokens, sources equal), else from the forward pass over the dashboard's windows, else the stored
    evaluation labelled as another model's."""
    title = "Calibration of the top prediction"
    derived = ck.finetune or ck.edit  # weights changed after the stored evaluation (fine-tuned or edited)
    last = ck.metrics.get("last_eval") or (ck.metrics.get("base_last_eval") if derived else None) or {}
    step = ck.obj.get("step") if isinstance(ck.obj, dict) else None
    cal, ece, source = None, None, ""
    if last.get("calibration") and last.get("step") == step and not derived and not holds_ema(ck):
        cal, ece = last["calibration"], last.get("ece")
        source = f"from the run's evaluation at step {step:,} (sources equal)"
    elif fstats is not None:
        cal, ece = fstats["calibration"], fstats["ece"]
        source = f"from {fstats['tokens']:,} tokens of this forward pass (biggest sources first)"
    elif last.get("calibration"):
        cal, ece = last["calibration"], last.get("ece")
        source = f"from the stored evaluation at step {last.get('step')}"
        title += derived_note(ck) or (" (raw weights; this file holds the EMA average)" if holds_ema(ck) else "")
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
    titled(ax, title, f"expected calibration error {fmt(ece, '.3f')}; bars below the diagonal = overconfident; "
                      f"{source}")


def probe_report(ck, model, device, args, raw_weights=False):
    """The checkpoint's own probe results when training scored this probe file
    at this step (no model run); otherwise score the probes now. raw_weights: the
    caller shows the history's per-evaluation scores, which are the raw weights'
    like the stored report, so a best file that holds the EMA average may use it."""
    sha = tiny_gpt.file_sha256(args.probes)
    last = ck.metrics.get("last_eval") or {}
    stored = last.get("probes")
    step = ck.obj.get("step") if isinstance(ck.obj, dict) else None
    # a fine-tuned or edited file carries the base model's last_eval (and step), and a best file that holds the
    # EMA average the raw weights': the stored report is not these weights'
    derived = ck.finetune or ck.edit or (holds_ema(ck) and not raw_weights)
    if stored and stored.get("file_sha256") == sha and last.get("step") == step and not derived:
        return stored
    if model is None:
        return None
    _, results = tiny_gpt.score_probes(model, ck.tok, tiny_gpt.read_probes(args.probes), device, None)
    return tiny_gpt.probe_report(results, sha)


def panel_probes(ax, ck, model, device, args, fnote=""):
    title = "Fact benchmark"
    if not args.probes or not os.path.isfile(args.probes):
        return message(ax, f"probe file not found: {args.probes}", title)
    report = probe_report(ck, model, device, args)
    if report is None:
        return message(ax, ("Needs the model: " + fnote) if fnote else "needs a tinyGPT model with a tokenizer", title)
    acc = report["correct"] / max(report["n"], 1)
    summary = sorted(report["category"].items(), key=lambda kv: kv[1]["accuracy"])
    names = [f"{name} (n={s['n']})" for name, s in summary]
    accs = np.array([s["accuracy"] for _, s in summary])
    lows = accs - np.array([s["low"] for _, s in summary])
    highs = np.array([s["high"] for _, s in summary]) - accs
    chance = report["chance"]
    floors = np.array([s["floor"] for _, s in summary])
    floor = report["floor"]
    y = np.arange(len(summary))
    ax.barh(y, accs, color=[floor_colour(s) for _, s in summary], height=0.62, alpha=0.9)
    ax.scatter([], [], marker="s", s=40, color=OTHER, label="grey: the 95% interval includes the floor")
    ax.errorbar(accs, y, xerr=[lows, highs], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    ax.scatter(floors, y, marker="|", s=140, color=INK, lw=2, zorder=4, label="floor (question words shuffled)")
    ax.axvline(chance, color=MUTED, lw=0.8, ls=":")
    ax.axvline(floor, color=INK, lw=1, ls="--")
    ax.annotate(f"floor {floor:.0%}", (floor, len(summary) - 0.4), xytext=(3, 0), textcoords="offset points",
                fontsize=_SIZE["heat"], color=INK, va="bottom")
    ax.annotate(f"chance {chance:.0%}", (chance, -0.6), xytext=(-3, 0), textcoords="offset points",
                fontsize=_SIZE["heat"], color=MUTED, va="top", ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=_SIZE["heat"])
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy (correct continuation preferred over all distractors); lines: 95% interval")
    low, high = tiny_gpt.wilson_interval(report["correct"], report["n"])
    by_diff = report["difficulty"]
    diff_text = ", ".join(f"{d} {by_diff[d]['accuracy']:.0%}" for d in ("easy", "medium", "hard") if d in by_diff)
    place_legend(ax)
    titled(ax, f"{title}: {acc:.0%} correct ({low:.0%}-{high:.0%}), floor {floor:.0%}",
           f"{diff_text}; green or red: the 95% interval is clear of the subject's floor (question words shuffled)")


def panel_attention(ax, model, windows, device, cmap, fnote):
    title = "Attention entropy"
    if model is None or windows is None:
        return message(ax, "Needs a forward pass: " + (fnote or "no model"), title)
    ent, rms = attention_entropy(model, windows[0], device, with_rms=True)
    heatmap(ax, ent, [f"block {b}" for b in range(ent.shape[0])], [f"head {h}" for h in range(ent.shape[1])],
            cmap, "normalised entropy", 0, 1, fmt_spec=".2f")
    ax.set_xlabel("query head")
    titled(ax, title, f"0 = attends to one token, 1 = spread evenly; residual stream RMS {rms[0]:.2f} after the "
                      f"embedding, {rms[-1]:.2f} after the last block")


# ---- mixture of experts only (both modes) ------------------------------------------
def _moe_blocks(ck):
    return [i for i in range(ck.cfg.n_layers) if ck.cfg.is_moe_block(i)]


def panel_moe_balance(ax, history, ck):
    """moe_lb (the Switch load-balancing loss, 1.0 = even routing), the busiest
    expert's load (tokens / mean tokens of its block) and the dead experts, at every
    evaluation (from the training history)."""
    title = "Expert balance"
    rows = [h for h in history or [] if h.get("moe_lb") is not None]
    if not rows:
        return _no_history(ax, ck, title) if not history else message(
            ax, "this history has no mixture-of-experts figures (moe_lb)", title)
    steps = [h["step"] for h in rows]
    ax.plot(steps, [h["moe_lb"] for h in rows], color=SERIES[0], lw=1.6, marker="o", ms=3, label="load-balancing loss")
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("E x sum f_e P_e (1.0 = even)")
    ax2 = ax.twinx()
    style_axes(ax2, grid=False, ticksize=_SIZE["tick"])
    ax2.plot(steps, [h.get("moe_load_max") for h in rows], color=SERIES[1], lw=1.4, marker="s", ms=3,
             label="busiest expert / mean")
    dead = [(s, h.get("dead_experts")) for s, h in zip(steps, rows) if h.get("dead_experts")]
    if dead:
        ax2.plot(*zip(*dead), color=CRITICAL, lw=0, marker="x", ms=6, label="dead experts (count)")
    ax2.set_ylabel("load of the busiest expert (1.0 = even)")
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper right")
    titled(ax, title, _history_sub(ck, "blue: Switch loss; orange: busiest expert's tokens / mean; red x: experts "
                                       "that saw no token"))


def panel_moe_load(ax, ck, history, cmap):
    """Tokens per expert relative to the block's mean since the last evaluation, per MoE block (last history row)."""
    title = "Expert load per block"
    rows = [h for h in history or [] if h.get("moe_load")]
    if not rows:
        return message(ax, "no per-block expert load in the history yet", title)
    load = np.array(rows[-1]["moe_load"], dtype=float)
    blocks = _moe_blocks(ck)
    cols = [f"block {b}" for b in blocks[:load.shape[0]]]
    heatmap(ax, load.T, [f"expert {e}" for e in range(load.shape[1])], cols, cmap, "tokens / mean tokens",
            vmin=0.0)
    titled(ax, title, f"at step {rows[-1]['step']:,}: 1.0 = even; 0 = a dead expert")


def panel_router_bias(ax, ck, cmap):
    """DeepSeek-V3's balancing bias per expert and MoE block (a buffer in the weights)."""
    title = "Router bias"
    items = [(b, ck.weights.get(f"blocks.{b}.ffn.router.bias")) for b in _moe_blocks(ck)]
    items = [(b, t) for b, t in items if t is not None]
    if not items:
        return message(ax, "no router bias buffers in this file", title)
    grid = np.stack([t.float().numpy() for _, t in items], axis=1)  # (E, blocks)
    lim = max(float(np.abs(grid).max()), 1e-6)
    heatmap(ax, grid, [f"expert {e}" for e in range(grid.shape[0])], [f"block {b}" for b, _ in items], cmap,
            "bias added to the routing score", vmin=-lim, vmax=lim, fmt_spec=".2g")
    titled(ax, title, "negative = the expert was overloaded and is now chosen less (auxiliary-loss-free balancing)")


# ---------------------------------------------------------------------------
# GitHub push
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
        # realpath on both folders: git names the real folder, the checkpoint may be reached through a link to
        # it. The file's own name is not resolved: a checkpoint that is a link belongs to the folder it is in.
        root = os.path.normcase(os.path.realpath(top.stdout.strip()))
        ck_dir = os.path.normcase(os.path.realpath(os.path.dirname(os.path.abspath(ck.path))))
        if os.path.commonpath([root, ck_dir]) != root:
            return
        rel = "dashboards/" + os.path.basename(png)
        step = ck.obj.get("step", "?") if isinstance(ck.obj, dict) else "?"
        status = tiny_gpt.publish_to_github(rel, f"Dashboard: {os.path.basename(ck.path)} at step {step}", src=png)
        print(f"Dashboard on GitHub ({rel}): {status}")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:  # no git, timeout, other drive...
        print(f"WARNING: dashboard not pushed ({type(exc).__name__}: {exc})")


# ---------------------------------------------------------------------------
# HTML embedding explorer (hover labels)
# ---------------------------------------------------------------------------
def write_embedding_html(ck, out_path):
    if ck.embedding is None or ck.tok is None:
        print("--html needs token embeddings and a tokenizer.")
        return None
    pieces = ck.pieces()
    classes = [token_class(ck.tok, i) for i in range(ck.tok.vocab_size)]
    xy, var = embedding_pca(ck, len(pieces))
    norms = ck.embedding.float()[:len(pieces)].norm(dim=1).numpy()
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
(PC 1 {float(var[0]):.1%}, PC 2 {float(var[1]):.1%} of variance).
Hover a dot for the token, its class and embedding norm.</p><ul>{legend}</ul>
<svg viewBox="0 0 {w} {h}" role="img" aria-label="PCA scatter of token embeddings">{''.join(dots)}</svg>
</body></html>"""
    def write(tmp):
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(page)
    out_path = save_replacing(write, out_path)
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
    # the tensor table is for the plain call and for looking a name up, not for every picture the trainer draws
    if not (args.plot or args.html) or args.tensor or args.similar:
        print_tensors(ck)
    if args.tensor:
        if args.tensor not in ck.weights:
            sys.exit(f"\nNo tensor named {args.tensor!r}. Use one of the names listed above.")
        torch.set_printoptions(precision=4, edgeitems=4, linewidth=120, sci_mode=False)
        label = pretty_name(args.tensor, tied=ck.cfg is None or ck.cfg.tie_embeddings)
        print(f"\n{args.tensor} ({label}):\n{ck.weights[args.tensor]}")
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
    if args.plot or args.plot_tokens:
        out = create_dashboard(ck, args)
        if out and not args.no_push:
            push_dashboard(out, ck)
    if args.html:
        write_embedding_html(ck, os.path.splitext(args.out)[0] + "_embeddings.html" if args.out
                             else output_name(ck, "embeddings", ".html"))


if __name__ == "__main__":
    main()
