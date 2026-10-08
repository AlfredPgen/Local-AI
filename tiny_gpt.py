r"""Train and sample a compact decoder-only language model.

Architecture: SentencePiece BPE tokens, pre-norm Transformer blocks with RoPE,
RMSNorm, SwiGLU, grouped-query attention (GQA), tied input/output embeddings
and PyTorch scaled-dot-product attention (SDPA). Every size, the context
length, batch and learning rate are configurable; unset values are chosen by
a planner from the prepared dataset's measured token counts and the device's
free memory (see --plan).

Workflow
--------
1. Build a dataset once (text cleaning, keyword filters, dedup, split, tokenizer):
       python data_prep.py --out datasets\bio_v1 --md-dir %USERPROFILE%\ai_training_data ^
           --wiki-dir %USERPROFILE%\ai_training_data\wikipedia --wiki-max-docs 100000 ^
           --include-keywords keywords.txt --keyword-min-distinct 3
2. Look at the automatic plan, then train (the time budget measures the real
   speed on this device and picks the largest model the data AND the time allow):
       python tiny_gpt.py --dataset datasets\bio_v1 --plan --time-budget-hours 8
       python tiny_gpt.py --dataset datasets\bio_v1 --name bio_v1 --time-budget-hours 8
       python tiny_gpt.py --dataset datasets\bio_v1 --name bio_small --d-model 256 --layers 6 --steps 5000
3. Resume after Ctrl+C (same name; the dataset path is stored in the checkpoint):
       python tiny_gpt.py --name bio_v1 --resume
       python tiny_gpt.py --name bio_v1 --resume --steps 30000     # extend the run
4. Generate:
       python tiny_gpt.py --name bio_v1 --generate "Genetic drift is" --length 200
       python tiny_gpt.py --checkpoint tiny_gpt_bpe_best.pt --generate "Genetic drift"   (checkpoints from the original script work too)

Output: <name>.pt (training checkpoint), <name>_best.pt (inference-only, best
validation loss), <name>_metrics.jsonl, and log.txt (everything printed to the
terminal, including errors, is appended to it).

Portability: CUDA (BF16/FP16 autocast, fused AdamW), Apple MPS (BF16 when
supported) and CPU all work in eager mode. torch.compile is used only where
Triton exists (--compile auto); Windows never needs it.
"""

import argparse
import atexit
import dataclasses
import datetime
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import data_prep

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FORMAT = "tiny_gpt_checkpoint"
FORMAT_VERSION = 4
DEFAULT_NAME = "tinyGPT"
DEFAULT_PROBES = os.path.join(HERE, "probes_biology.tsv")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")


# ---------------------------------------------------------------------------
# Logging: mirror stdout/stderr into log.txt
# ---------------------------------------------------------------------------
class Tee:
    """File-like object that writes to the console and the run log."""

    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        try:
            self.console.write(text)
            self.console.flush()
        except (OSError, ValueError):
            pass
        if not self.log.closed:
            self.log.write(text)
            if "\n" in text:
                self.log.flush()
        return len(text)

    def flush(self):
        for stream in (self.console, self.log):
            try:
                if not getattr(stream, "closed", False):
                    stream.flush()
            except (OSError, ValueError):
                pass

    def isatty(self):
        return getattr(self.console, "isatty", lambda: False)()

    def fileno(self):
        return self.console.fileno()

    @property
    def encoding(self):
        return "utf-8"

    @property
    def errors(self):
        return "replace"


_LOG_STATE = {}


def start_log(path):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise SystemExit(f"Log folder does not exist: {directory}")
    handle = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    _LOG_STATE.update(handle=handle, out=sys.stdout, err=sys.stderr, path=path)
    sys.stdout = Tee(_LOG_STATE["out"], handle)
    sys.stderr = Tee(_LOG_STATE["err"], handle)
    warnings.showwarning = _show_warning
    atexit.register(stop_log)
    return handle


def _show_warning(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))


def stop_log():
    handle = _LOG_STATE.pop("handle", None)
    if handle is None:
        return
    sys.stdout, sys.stderr = _LOG_STATE.pop("out"), _LOG_STATE.pop("err")
    handle.close()


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def fmt_hms(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def fmt_dur(seconds):
    """Duration as '9h 50m' / '12m 03s' (never confused with a clock time)."""
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def file_sha256(path):
    try:
        return data_prep.sha256_file(path)
    except OSError:
        return "?"


# ---------------------------------------------------------------------------
# Device, precision, attention kernel
# ---------------------------------------------------------------------------
def select_device(preference="auto"):
    cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if preference == "cuda" or (preference == "auto" and cuda_ok):
        if not cuda_ok:
            raise SystemExit("--device cuda requested but CUDA is not available in this PyTorch build.")
        return torch.device("cuda")
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if preference == "mps" or (preference == "auto" and mps_ok):
        if not mps_ok:
            raise SystemExit("--device mps requested but Apple MPS is not available.")
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device):
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"{props.name} ({props.total_memory / 2 ** 30:.1f} GiB, sm_{props.major}{props.minor})"
    if device.type == "mps":
        return "Apple MPS"
    return f"CPU ({os.cpu_count()} threads)"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _mps_bf16_ok():
    try:
        a = torch.ones(8, 8, device="mps")
        with torch.autocast("mps", dtype=torch.bfloat16):
            return bool(torch.isfinite((a @ a).sum()).item())
    except Exception:  # noqa: BLE001 - any failure means "not supported"
        return False


def choose_precision(device, preference="auto"):
    """Return (autocast dtype or None, label, needs GradScaler)."""
    if preference == "auto":
        if device.type == "cuda":
            preference = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        elif device.type == "mps":
            preference = "bf16" if _mps_bf16_ok() else "fp32"
        else:
            preference = "fp32"
    if preference == "fp32":
        return None, "FP32", False
    if preference == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise SystemExit("This GPU does not support BF16; use --precision fp16 or auto.")
        return torch.bfloat16, "BF16 autocast", False
    if preference == "fp16":
        if device.type == "cpu":
            raise SystemExit("--precision fp16 is not supported on CPU; use fp32 or bf16.")
        return torch.float16, "FP16 autocast", device.type == "cuda"
    raise SystemExit(f"Unknown precision {preference!r}")


def _enable_gqa_supported():
    try:
        q = torch.zeros(1, 4, 2, 8)
        k = torch.zeros(1, 2, 2, 8)
        F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
        return True
    except (TypeError, RuntimeError):
        return False


_ATTN_CACHE = {}


def choose_attention(device, cfg, amp_dtype, preference="auto"):
    """Pick how K/V heads are shared with query heads.

    'gqa_native' passes enable_gqa=True to SDPA; 'repeat_kv' expands K/V to all
    query heads (a small copy) first. Both compute the same attention. Which is
    faster depends on which SDPA kernels this PyTorch build dispatches to: on
    PyTorch 2.5.1 for Windows, enable_gqa falls back to the MATH kernel and was
    measured ~13x slower per attention call (RTX 3070 Laptop GPU). So on CUDA
    the two are simply timed on the model's real head shape and the faster wins.
    """
    if cfg.n_kv_heads == cfg.n_heads:
        return "repeat_kv", "multi-head attention (no K/V sharing)"
    native = _enable_gqa_supported()
    if preference == "gqa_native":
        if not native:
            raise SystemExit("--attn gqa_native needs PyTorch >= 2.5 (enable_gqa).")
        return "gqa_native", "forced by --attn"
    if preference == "repeat_kv":
        return "repeat_kv", "forced by --attn"
    if not native:
        return "repeat_kv", "this PyTorch has no enable_gqa"
    if device.type != "cuda":
        return "repeat_kv", f"{device.type}: no fused GQA kernel to gain from"
    key = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, min(cfg.ctx, 512), str(amp_dtype))
    if key in _ATTN_CACHE:
        return _ATTN_CACHE[key]
    dtype = amp_dtype or torch.float32
    t = min(cfg.ctx, 512)
    rep = cfg.n_heads // cfg.n_kv_heads
    q = torch.randn(8, cfg.n_heads, t, cfg.head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(8, cfg.n_kv_heads, t, cfg.head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(8, cfg.n_kv_heads, t, cfg.head_dim, device=device, dtype=dtype, requires_grad=True)
    calls = {
        "gqa_native": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True),
        "repeat_kv": lambda: F.scaled_dot_product_attention(
            q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1), is_causal=True),
    }
    times = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fn in calls.items():
            for _ in range(2):
                fn().float().sum().backward()
            synchronize(device)
            t0 = time.perf_counter()
            for _ in range(5):
                fn().float().sum().backward()
            synchronize(device)
            times[name] = (time.perf_counter() - t0) / 5
    del q, k, v
    best = min(times, key=times.get)
    other = "repeat_kv" if best == "gqa_native" else "gqa_native"
    result = (best, f"timed on this GPU: {times[best] * 1e3:.1f} ms vs {times[other] * 1e3:.1f} ms for {other}")
    _ATTN_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ModelConfig:
    vocab_size: int
    ctx: int = 512
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_hidden: int = 0
    rope_base: float = 10_000.0
    dropout: float = 0.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self):
        if not self.ffn_hidden:
            self.ffn_hidden = swiglu_hidden(self.d_model)
        self.validate()

    def validate(self):
        problems = []
        if self.vocab_size < 16:
            problems.append("vocab_size must be >= 16")
        if self.d_model % self.n_heads:
            problems.append(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")
        elif (self.d_model // self.n_heads) % 2:
            problems.append("head size (d_model / n_heads) must be even for RoPE")
        if self.n_kv_heads < 1 or self.n_heads % self.n_kv_heads:
            problems.append(f"n_heads {self.n_heads} must be a multiple of n_kv_heads {self.n_kv_heads}")
        if not 8 <= self.ctx <= 65_536:
            problems.append("ctx must be between 8 and 65536")
        if not 1 <= self.n_layers <= 256:
            problems.append("n_layers must be between 1 and 256")
        if not 0.0 <= self.dropout < 0.9:
            problems.append("dropout must be in [0, 0.9)")
        if problems:
            raise SystemExit("Invalid model settings: " + "; ".join(problems))

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


def swiglu_hidden(d_model, multiple=64):
    """8/3 * d rounded up to a multiple of 64 (keeps parameters ~ a 4x GELU MLP
    and gives tensor-core friendly shapes)."""
    return int(multiple * math.ceil((8 * d_model / 3) / multiple))


class RMSNorm(nn.Module):
    """RMS normalization with FP32 variance accumulation."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


def rotate_half(x):
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Interleaved-pair RoPE (same layout as the original script), cached to ctx."""

    def __init__(self, head_dim, max_seq_len, base=10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.repeat_interleave(torch.outer(torch.arange(max_seq_len).float(), inv_freq), 2, dim=-1)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def forward(self, q, k):
        t = q.size(-2)
        cos = self.cos[:, :, :t].to(dtype=q.dtype)
        sin = self.sin[:, :, :t].to(dtype=q.dtype)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class GroupedQueryAttention(nn.Module):
    """n_heads query heads share n_kv_heads key/value heads (GQA)."""

    def __init__(self, dim, n_heads, n_kv_heads, dropout):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.impl = "repeat_kv"
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def _qkv(self, x, rope):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = rope(q, k)
        return q, k, v

    def forward(self, x, rope):
        b, t, _ = x.shape
        q, k, v = self._qkv(x, rope)
        p = self.dropout if self.training else 0.0
        if self.impl == "gqa_native" and self.n_rep > 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True, enable_gqa=True)
        else:
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, -1))

    @torch.no_grad()
    def attention_weights(self, x, rope):
        """Explicit causal softmax(QK^T/sqrt(d)) per head, shape (B, H, T, T)."""
        q, k, _ = self._qkv(x, rope)
        k = k.repeat_interleave(self.n_rep, dim=1)
        t = q.size(-2)
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        return scores.masked_fill(mask, float("-inf")).softmax(-1)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.dropout)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, rope):
        x = x + self.resid_dropout(self.attn(self.attn_norm(x), rope))
        return x + self.ffn(self.ffn_norm(x))


class TinyGPT(nn.Module):
    """Module names match the original script, so its (legacy) weights load unchanged."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ctx = cfg.ctx
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.ctx, cfg.rope_base)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=resid_std)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        # Activation checkpointing: keep only each block's input during the forward pass and recompute the
        # block in the backward pass (about a third more compute, a fraction of the activation memory).
        self.grad_checkpoint = False

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def set_attention_impl(self, impl):
        for block in self.blocks:
            block.attn.impl = impl

    def forward(self, tokens, return_hidden=False):
        """Logits, or (return_hidden=True) the final normalised hidden states, from
        which chunked_lm_loss computes the loss without all logits at once."""
        x = self.embedding_dropout(self.token_embedding(tokens))
        recompute = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if recompute:
                x = torch.utils.checkpoint.checkpoint(block, x, self.rope, use_reentrant=False)
            else:
                x = block(x, self.rope)
        x = self.norm(x)
        return x if return_hidden else self.lm_head(x)

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        embed = self.token_embedding.weight.numel()
        if not self.cfg.tie_embeddings:
            embed += self.lm_head.weight.numel()
        return total, total - embed


LOSS_CHUNK_TOKENS = 4096  # tokens per piece when the loss is computed in pieces


def _chunk_loss(h, weight, y):
    logits = F.linear(h, weight).float()
    lse = torch.logsumexp(logits, dim=-1)
    valid = y != -100
    target = logits.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return ((lse - target) * valid).sum(), (lse.pow(2) * valid).sum()


def _chunk_logp(h, weight, y):
    logits = F.linear(h, weight).float()
    target = logits.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (target - torch.logsumexp(logits, dim=-1)) * (y != -100)


def chunked_lm_loss(hidden, weight, targets, chunk_tokens=LOSS_CHUNK_TOKENS):
    """(sum of cross-entropy, sum of log Z^2, number of labelled tokens) of the
    output layer `weight` applied to `hidden`, computed `chunk_tokens` tokens at a
    time; each piece's logits are recomputed in the backward pass, so the
    (tokens x vocabulary) table never exists in full. Labels of -100 are ignored."""
    h = hidden.reshape(-1, hidden.size(-1))
    y = targets.reshape(-1)
    ce = z = None
    for s in range(0, y.numel(), chunk_tokens):
        c, zc = torch.utils.checkpoint.checkpoint(_chunk_loss, h[s:s + chunk_tokens], weight, y[s:s + chunk_tokens],
                                                  use_reentrant=False)
        ce, z = (c, zc) if ce is None else (ce + c, z + zc)
    return ce, z, (y != -100).sum()


def target_logprobs(model, x, y, chunk_tokens=LOSS_CHUNK_TOKENS):
    """log p(y_t | x_<=t) at every position (0 where y is -100), shaped like y,
    without the full (tokens x vocabulary) table: the output layer is applied
    in pieces (recomputed in the backward pass when gradients are needed)."""
    h = model(x, return_hidden=True)
    flat_h, flat_y = h.reshape(-1, h.size(-1)), y.reshape(-1)
    pieces = []
    for s in range(0, flat_y.numel(), chunk_tokens):
        args = (flat_h[s:s + chunk_tokens], model.lm_head.weight, flat_y[s:s + chunk_tokens])
        pieces.append(torch.utils.checkpoint.checkpoint(_chunk_logp, *args, use_reentrant=False)
                      if torch.is_grad_enabled() else _chunk_logp(*args))
    return torch.cat(pieces).view_as(y)


def count_params(cfg):
    """Parameter count without building the model (unique, tied counted once)."""
    d, h = cfg.d_model, cfg.ffn_hidden
    kv = cfg.n_kv_heads * cfg.head_dim
    per_layer = d * d * 2 + d * kv * 2 + 3 * d * h + 2 * d
    embed = cfg.vocab_size * d * (1 if cfg.tie_embeddings else 2)
    return per_layer * cfg.n_layers + d + embed, per_layer * cfg.n_layers + d


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------
class LRSchedule:
    """Warmup-Stable-Decay (default) or warmup + cosine.

    WSD: linear warmup, constant peak, then a decay over the last `decay_frac`
    of steps with shape '1-sqrt' (Hagele et al. 2024), 'cosine' or 'linear'.
    Stateless: lr_at(step) depends only on the step, so resume is exact.
    """

    def __init__(self, kind, total_steps, peak_lr, min_lr, warmup_steps, decay_frac=0.2, shape="1-sqrt"):
        self.kind, self.total, self.peak, self.min_lr = kind, int(total_steps), float(peak_lr), float(min_lr)
        self.warmup = int(min(max(warmup_steps, 0), max(self.total - 1, 0)))
        self.shape = shape
        if kind == "wsd":
            self.decay_start = self.total - int(round(decay_frac * self.total))
            self.decay_start = max(self.decay_start, self.warmup)
        else:
            self.decay_start = self.warmup

    def lr_at(self, step):
        if step <= self.warmup:
            return self.peak * step / max(self.warmup, 1)
        if step <= self.decay_start:
            return self.peak
        t = min(1.0, (step - self.decay_start) / max(self.total - self.decay_start, 1))
        shape = "cosine" if self.kind == "cosine" else self.shape
        if shape == "cosine":
            f = 0.5 * (1 + math.cos(math.pi * t))
        elif shape == "linear":
            f = 1 - t
        else:
            f = 1 - math.sqrt(t)
        return self.min_lr + (self.peak - self.min_lr) * f

    def phase(self, step):
        if step <= self.warmup:
            return "warmup"
        return "stable" if step <= self.decay_start else "decay"

    def describe(self):
        if self.kind == "wsd":
            return (f"WSD: warmup 1-{self.warmup} (linear), stable to {self.decay_start} at {self.peak:.2e}, "
                    f"decay {self.decay_start + 1}-{self.total} ({self.shape}) to {self.min_lr:.2e}")
        return f"cosine: warmup 1-{self.warmup}, cosine to {self.min_lr:.2e} at {self.total}"


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def safe_load(path, trust=False):
    """torch.load with weights_only=True (no arbitrary Python objects)."""
    if not os.path.isfile(path):
        raise SystemExit(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else type(exc).__name__
        if trust:
            print(f"WARNING: {path} needs full unpickling ({first}); loading it because --trust-checkpoint "
                  "was given. Only do this for files you created yourself.")
            return torch.load(path, map_location="cpu", weights_only=False)
        raise SystemExit(
            f"Refusing to load {path}: it is unreadable or contains arbitrary Python objects "
            f"({type(exc).__name__}: {first}). tiny_gpt checkpoints load safely; if you created this "
            "file yourself and trust it, pass --trust-checkpoint.")


def checkpoint_info(obj):
    """Classify a loaded checkpoint: (family, type), family 'tinygpt' (this script)
    or 'legacy-v3' / 'legacy-v2' (the original script), type 'training' or 'inference'."""
    if not isinstance(obj, dict):
        return "unknown", "unknown"
    if obj.get("format") == CHECKPOINT_FORMAT:
        return "tinygpt", obj.get("checkpoint_type", "unknown")
    if obj.get("model_version") in (2, 3) and "settings" in obj:
        kind = obj.get("checkpoint_type") or ("training" if "optimizer" in obj else "inference_only")
        return f"legacy-v{obj['model_version']}", "training" if kind == "training" else "inference"
    return "unknown", "unknown"


def config_from_checkpoint(obj):
    family, _ = checkpoint_info(obj)
    if family == "tinygpt":
        return ModelConfig(**obj["model_config"])
    if family in ("legacy-v3", "legacy-v2"):
        s = obj["settings"]
        weights = obj["model"]
        tied = ("lm_head.weight" not in weights or
                torch.equal(weights["lm_head.weight"], weights["token_embedding.weight"]))
        return ModelConfig(vocab_size=int(obj.get("vocab_size") or weights["token_embedding.weight"].shape[0]),
                           ctx=int(s["CTX"]), d_model=int(s["D"]), n_layers=int(s["LAYERS"]),
                           n_heads=int(s["HEADS"]), n_kv_heads=int(s.get("KV_HEADS", s["HEADS"])),
                           ffn_hidden=(8 * int(s["D"])) // 3, rope_base=float(s.get("ROPE_BASE", 10_000.0)),
                           dropout=float(s.get("DROPOUT", 0.0)), tie_embeddings=tied)
    raise SystemExit("Unrecognised checkpoint layout (neither tinyGPT nor the original script). "
                     "view_pt.py can still list its tensors.")


def tokenizer_from_checkpoint(obj):
    family, _ = checkpoint_info(obj)
    if family == "tinygpt":
        meta = obj["tokenizer"]
        return data_prep.Tokenizer(meta["proto"], meta.get("encode_mode", "lines"))
    if obj.get("tokenizer_proto"):
        return data_prep.Tokenizer(obj["tokenizer_proto"], "plain")
    raise SystemExit("This checkpoint has no tokenizer; it cannot be used for generation.")


def model_from_checkpoint(obj, device="cpu"):
    """(model in eval mode, tokenizer, config) from an already loaded checkpoint (any format)."""
    cfg = config_from_checkpoint(obj)
    cfg.dropout = 0.0
    model = TinyGPT(cfg)
    missing, unexpected = model.load_state_dict(obj["model"], strict=False)
    missing = [m for m in missing if not (m == "lm_head.weight" and cfg.tie_embeddings)]
    if missing or unexpected:
        raise SystemExit(f"Checkpoint weights do not match the architecture "
                         f"(missing {missing[:4]}, unexpected {unexpected[:4]}).")
    model.to(device).eval()
    return model, tokenizer_from_checkpoint(obj), cfg


def load_for_inference(path, device="cpu", trust=False):
    """(model, tokenizer, config, raw checkpoint dict) for a tinyGPT or legacy
    checkpoint file, or an --export folder."""
    obj = load_any(path, trust)
    model, tok, cfg = model_from_checkpoint(obj, device)
    return model, tok, cfg, obj


def cpu_state_dict(model):
    """State dict on CPU; tensors that share storage (tied embeddings) stay shared."""
    out, seen = {}, {}
    for key, value in model.state_dict().items():
        ident = (value.data_ptr(), tuple(value.shape), value.dtype)
        if ident not in seen:
            seen[ident] = value.detach().to("cpu", copy=True)
        out[key] = seen[ident]
    return out


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def atomic_save(obj, path):
    """Write to a temp file in the same folder, then rename over the target.
    Retries briefly if Windows reports the target as in use (e.g. open in the viewer)."""
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(obj, tmp)
        for attempt in range(6):
            try:
                os.replace(tmp, path)
                return path
            except PermissionError:
                time.sleep(0.5 * (attempt + 1))
        fallback = f"{os.path.splitext(path)[0]}.{datetime.datetime.now():%H%M%S}.pt"
        os.replace(tmp, fallback)
        print(f"WARNING: {path} is locked by another program; saved to {fallback} instead.")
        return fallback
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def launch_dashboard(path, args, state):
    """Draw view_pt.py's dashboard for a new best checkpoint in the background
    (CPU, idle priority; training does not wait). Skipped while the previous
    one is still drawing. view_pt.py pushes it to GitHub (for checkpoints in this
    repository's folder) at most every --dashboard-push-hours."""
    if state["proc"] is not None and state["proc"].poll() is None:
        return
    now = time.time()
    push = args.dashboard_push_hours <= 0 or now - state["pushed"] >= args.dashboard_push_hours * 3600
    cmd = [sys.executable, os.path.join(HERE, "view_pt.py"), path, "--plot"] + ([] if push else ["--no-push"])
    log_dir = os.path.join(args.out_dir, "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "dashboard.log"), "a", encoding="utf-8") as log:
            log.write(f"\n--- {now_iso()} | {' '.join(cmd[1:])}\n")
            log.flush()
            if sys.platform == "win32":
                flags = subprocess.IDLE_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
                state["proc"] = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, creationflags=flags)
            else:
                state["proc"] = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                                                 preexec_fn=lambda: os.nice(19))
        if push:
            state["pushed"] = now
    except OSError as exc:
        print(f"note: dashboard not started ({type(exc).__name__}: {exc})")


class CombinedOptimizer:
    """Several optimizers used as one (Muon for the block matrices, AdamW for the
    rest): one list of parameter groups, one step, and one state dict whose
    indices run through the optimizers in order, so a checkpoint holds a single
    flat list of parameter states, as with AdamW alone. GradScaler works with it
    through param_groups and step()."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def step(self):
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        state, groups, offset = {}, [], 0
        for o in self.optimizers:
            sd = o.state_dict()
            state.update({offset + int(k): v for k, v in sd["state"].items()})
            groups += [{**g, "params": [offset + i for i in g["params"]]} for g in sd["param_groups"]]
            offset += sum(len(g["params"]) for g in o.param_groups)
        return {"state": state, "param_groups": groups}

    def load_state_dict(self, sd):
        offset, first = 0, 0
        for o in self.optimizers:
            n = sum(len(g["params"]) for g in o.param_groups)
            groups = sd["param_groups"][first:first + len(o.param_groups)]
            o.load_state_dict({"state": {int(k) - offset: v for k, v in sd["state"].items()
                                         if offset <= int(k) < offset + n},
                               "param_groups": [{**g, "params": [i - offset for i in g["params"]]} for g in groups]})
            offset += n
            first += len(o.param_groups)


OPTIMIZER_LABELS = {"adamw": "AdamW", "muon": "Muon + AdamW"}
MUON_EXCLUDED = ("token_embedding.weight", "lm_head.weight")  # embedding and output layer stay on AdamW
STATE_KEYS = {"AdamW": "exp_avg", "Muon": "momentum_buffer"}  # the state each optimizer keeps per parameter


def build_optimizer(model, settings, device):
    """AdamW for every parameter, or (optimizer 'muon') Muon for the matrices
    inside the blocks and AdamW for the embedding/output matrix and the norm
    gains, the split Muon is designed for (Jordan et al. 2024). Muon's step is
    scaled to AdamW's typical update size (Liu et al. 2025; PyTorch's
    adjust_lr_fn='match_rms_adamw'), so one learning rate, schedule and weight
    decay drive both. Returns (optimizer, name, parameter names per group,
    description)."""
    named = list(model.named_parameters())
    use_muon = settings.get("optimizer", "adamw") == "muon"
    muon = [(n, p) for n, p in named if use_muon and p.dim() == 2 and n not in MUON_EXCLUDED]
    taken = {n for n, _ in muon}
    decay = [(n, p) for n, p in named if p.dim() >= 2 and n not in taken]
    no_decay = [(n, p) for n, p in named if p.dim() < 2]
    groups = [{"params": [p for _, p in decay], "weight_decay": settings["weight_decay"]},
              {"params": [p for _, p in no_decay], "weight_decay": 0.0}]
    opt_kwargs = dict(lr=settings["lr"], betas=tuple(settings["betas"]))
    adamw, name = None, "AdamW"
    if device.type == "cuda":
        try:
            adamw, name = torch.optim.AdamW(groups, fused=True, **opt_kwargs), "fused AdamW"
        except (TypeError, RuntimeError, ValueError):
            adamw = None
    if adamw is None:
        adamw = torch.optim.AdamW(groups, **opt_kwargs)
    names = [[n for n, _ in decay], [n for n, _ in no_decay]]
    if not use_muon:
        return adamw, name, names, f"weight decay on {len(decay)} matrices, not on {len(no_decay)} norm gains"
    if not hasattr(torch.optim, "Muon"):
        raise SystemExit(f"--optimizer muon needs PyTorch 2.9 or newer (this is {torch.__version__}).")
    muon_opt = torch.optim.Muon([{"params": [p for _, p in muon], "weight_decay": settings["weight_decay"]}],
                                lr=settings["lr"], momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")
    return (CombinedOptimizer([adamw, muon_opt]), f"Muon + {name}", names + [[n for n, _ in muon]],
            f"Muon for {len(muon)} block matrices, its step scaled to AdamW's update size; {name} for the embedding "
            f"({len(decay)} matri{'x' if len(decay) == 1 else 'ces'}) and {len(no_decay)} norm gains; weight decay "
            "on all matrices")


def _state_keys(opt):
    """Per parameter (flat order), the state entry its optimizer keeps (None: unknown optimizer)."""
    subs = opt.optimizers if isinstance(opt, CombinedOptimizer) else [opt]
    return [STATE_KEYS.get(type(o).__name__) for o in subs for g in o.param_groups for _ in g["params"]]


def load_optimizer_by_name(opt, saved, saved_names, current_names):
    """Restore optimizer state (AdamW moments, Muon momentum) by parameter name,
    independent of group layout (legacy checkpoints used one group; tinyGPT uses
    decay / no-decay groups, plus a Muon group). A saved state is used only if
    the same kind of optimizer made it and it has the parameter's shape, so
    --init-from an AdamW run into a Muon run restarts Muon's momentum. The
    remapped state goes through opt.load_state_dict, which puts every tensor on
    the right device and dtype (fused AdamW keeps its step counters on the GPU)."""
    flat_current = [n for group in current_names for n in group]
    params = [p for g in opt.param_groups for p in g["params"]]
    keys = _state_keys(opt)
    saved_state = {int(k): v for k, v in saved.get("state", {}).items()}
    by_name = {name: saved_state.get(i) for i, name in enumerate(saved_names)}
    current = opt.state_dict()
    new_state = {}
    for index, (name, param, key) in enumerate(zip(flat_current, params, keys)):
        state = by_name.get(name)
        if isinstance(state, dict) and (key is None or (key in state and state[key].shape == param.shape)):
            new_state[index] = state
    opt.load_state_dict({"state": new_state, "param_groups": current["param_groups"]})
    return len(new_state), len(params)


def legacy_optimizer_names(model):
    """Parameter names in model.parameters() order (the legacy optimizer's single group)."""
    return [name for name, _ in model.named_parameters()]


# ---------------------------------------------------------------------------
# Data: batches, fixed validation windows, probes
# ---------------------------------------------------------------------------
def parse_mix(text, sources):
    weights = {s: 1.0 for s in sources}
    if not text:
        return weights
    for item in text.split(","):
        if "=" not in item:
            raise SystemExit(f"--mix expects name=weight pairs, got {item!r}")
        name, value = (x.strip() for x in item.split("=", 1))
        if name not in sources:
            raise SystemExit(f"--mix: unknown source {name!r}; dataset sources are {sorted(sources)}")
        try:
            weights[name] = float(value)
        except ValueError:
            raise SystemExit(f"--mix: weight for {name} is not a number: {value!r}")
        if weights[name] < 0:
            raise SystemExit("--mix weights must be >= 0")
    if sum(weights.values()) <= 0:
        raise SystemExit("--mix: at least one weight must be positive")
    return weights


GRID_G_PER_KWH = 125.0  # GB grid, mean of Oct 2025 - Sep 2026 (National Grid ESO Carbon Intensity API)
RAM_W_PER_GB = 0.375    # memory power rule of thumb (3 W per 8 GB, as CodeCarbon uses)
OCTOPUS_RATES = ("https://api.octopus.energy/v1/products/{p}/electricity-tariffs/E-1R-{p}-{r}/"
                 "standard-unit-rates/")
FALLBACK_PENCE = 27.0   # Flexible Octopus, region H, September 2026: used until real rates are known


class Tariff:
    """Electricity unit rates in pence per kWh, VAT included; the standing charge
    is paid anyway and not counted. A number is a fixed price. Otherwise it is an
    Octopus Energy product code, whose published rates for the region are read
    from Octopus's public API (no account or personal data), so Agile's
    half-hourly prices and Go's night rate apply to the half-hour in which the
    electricity was used."""

    def __init__(self, spec, region):
        self.region = (region or "H").upper()
        try:
            self.fixed = float(spec)
            self.name = f"{self.fixed:g} p/kWh"
        except (TypeError, ValueError):
            self.fixed = None
            self.product = str(spec).upper()
            self.name = f"Octopus {self.product}, region {self.region}"
        self.rates = {}  # half-hour start (epoch seconds) -> pence; filled on demand
        self.periods = []  # (start, end, pence) as published
        self.last_fetch = 0.0

    def rate_at(self, slot):
        if self.fixed is not None:
            return self.fixed
        if slot not in self.rates:
            for start, end, pence in self.periods:
                if start <= slot < end:
                    self.rates[slot] = pence
                    break
        return self.rates.get(slot)

    def fetch(self, t0, t1):
        """Load the published rates covering [t0, t1]; network errors leave them unknown."""
        if self.fixed is not None or time.time() - self.last_fetch < 600:
            return
        self.last_fetch = time.time()
        import urllib.parse
        import urllib.request

        def iso(t):
            return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        def epoch(s):
            return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

        url = OCTOPUS_RATES.format(p=self.product, r=self.region) + "?" + urllib.parse.urlencode(
            {"period_from": iso(t0), "period_to": iso(t1 + 1800), "page_size": 1500})
        try:
            while url:
                request = urllib.request.Request(url, headers={"User-Agent": "tinyGPT energy meter"})
                with urllib.request.urlopen(request, timeout=15) as response:
                    data = json.load(response)
                for r in data.get("results", []):
                    if r.get("payment_method") in (None, "DIRECT_DEBIT"):
                        end = epoch(r["valid_to"]) if r.get("valid_to") else float("inf")
                        self.periods.append((epoch(r["valid_from"]), end, float(r["value_inc_vat"])))
                url = data.get("next")
        except Exception:  # noqa: BLE001 - offline: priced later, or at the fallback rate
            pass
        self.periods = sorted(set(self.periods))
        self.rates.clear()


def _slot(t=None):
    t = time.time() if t is None else t
    return int(t // 1800 * 1800)


class EnergyMeter:
    """Electricity used by a run and its carbon footprint.

    GPU (NVIDIA): measured. nvidia-smi reports the board's power draw every 5 s
    and the energy is its time integral (the whole GPU, so the display's small
    share is included). CPU and memory: estimated, as the CPU's rated power
    (--cpu-watts) times this process's share of all CPU time, plus 0.375 W per
    GB of this process's memory. Not included: screen, fans, charger losses,
    building the dataset, and the carbon of making the hardware. Carbon is
    energy x grid intensity (--grid-intensity, grams CO2e per kWh); cost is the
    energy of each half-hour times that half-hour's unit rate (Tariff). The
    totals are stored in the checkpoint, so they add up across --resume sessions."""

    def __init__(self, device, cpu_watts, grid_g_per_kwh, state=None, period=5.0, tariff=None):
        state = state or {}
        self.wh = {k: float(state.get(k, 0.0)) for k in ("gpu", "cpu", "ram")}
        self.tariff = tariff
        self.cost_p = float(state.get("cost_p", 0.0))  # pence for half-hours already priced
        self.pending = {int(k): float(v) for k, v in (state.get("pending_wh") or {}).items()}  # half-hour -> Wh
        self._samples = 0
        self.cpu_watts, self.grid, self.period = float(cpu_watts), float(grid_g_per_kwh), period
        self.gpu_index, self.gpu_measured = None, False
        self._lock, self._sampling, self._stop = threading.Lock(), threading.Lock(), threading.Event()
        if device.type == "cuda":
            visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
            index = device.index or 0
            self.gpu_index = visible[index] if index < len(visible) else str(index)
            self.gpu_measured = self._gpu_watts() is not None
        try:
            import psutil
            self._proc = psutil.Process()
            self._ncpu = psutil.cpu_count() or 1
            self._cpu_time = sum(self._proc.cpu_times()[:2])
        except Exception:  # noqa: BLE001 - no psutil: CPU and memory are not estimated
            self._proc = None
        self._last_t, self._last_gpu_w = time.perf_counter(), None
        self._thread = threading.Thread(target=self._run, name="energy-meter", daemon=True)
        self._thread.start()

    def _gpu_watts(self):
        try:
            out = subprocess.run(["nvidia-smi", f"--id={self.gpu_index}", "--query-gpu=power.draw",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            return float(out.stdout.strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def _sample(self):
        with self._sampling:  # the background thread and snapshot() both sample
            self._sample_locked()

    def _sample_locked(self):
        now = time.perf_counter()
        hours = (now - self._last_t) / 3600
        gpu_w = self._gpu_watts() if self.gpu_measured else None
        cpu_w = ram_w = 0.0
        if self._proc is not None:
            try:
                cpu_time = sum(self._proc.cpu_times()[:2])
                share = (cpu_time - self._cpu_time) / max((now - self._last_t) * self._ncpu, 1e-9)
                self._cpu_time = cpu_time
                cpu_w = self.cpu_watts * min(max(share, 0.0), 1.0)
                ram_w = RAM_W_PER_GB * self._proc.memory_info().rss / 2 ** 30
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            added = 0.0
            if gpu_w is not None:  # trapezoid between samples
                previous = self._last_gpu_w if self._last_gpu_w is not None else gpu_w
                added += (previous + gpu_w) / 2 * hours
                self.wh["gpu"] += (previous + gpu_w) / 2 * hours
                self._last_gpu_w = gpu_w
            self.wh["cpu"] += cpu_w * hours
            self.wh["ram"] += ram_w * hours
            added += (cpu_w + ram_w) * hours
            slot = _slot()
            self.pending[slot] = self.pending.get(slot, 0.0) + added
            self._last_t = now

    def _price(self):
        """Move the energy of half-hours whose unit rate is known into cost_p."""
        if self.tariff is None:
            return
        with self._lock:
            slots = sorted(self.pending)
        if not slots:
            return
        if any(self.tariff.rate_at(s) is None for s in slots):
            self.tariff.fetch(slots[0], slots[-1])  # at most every 10 min, in this background thread
        with self._lock:
            for s in list(self.pending):
                rate = self.tariff.rate_at(s)
                if rate is not None:
                    self.cost_p += self.pending.pop(s) / 1000 * rate

    def _run(self):
        while not self._stop.wait(self.period):
            self._sample()
            self._samples += 1
            if self._samples % 60 == 1:  # about every 5 minutes
                self._price()

    def stop(self):
        if not self._stop.is_set():
            self._stop.set()
            self._thread.join(timeout=15)
            self._sample()
            self._price()

    def snapshot(self):
        """Totals up to now (a fresh reading, so a checkpoint saved seconds after the start is not 0)."""
        if not self._stop.is_set():
            self._sample()
        with self._lock:
            return {k: round(v, 4) for k, v in self.wh.items()}

    def state(self):
        """Everything a checkpoint keeps, so energy and cost add up across --resume."""
        wh = self.snapshot()
        with self._lock:
            return {**wh, "cost_p": round(self.cost_p, 4), "pending_wh": {str(k): v for k, v in self.pending.items()}}

    def cost_gbp(self):
        """Priced half-hours, plus the rest at the latest known rate (or the fallback)."""
        with self._lock:
            pending = sum(self.pending.values())
            cost = self.cost_p
        if self.tariff is None:
            return None
        rate = self.tariff.rate_at(_slot())
        return (cost + pending / 1000 * (rate if rate is not None else FALLBACK_PENCE)) / 100

    def kwh(self):
        with self._lock:
            return sum(self.wh.values()) / 1000

    def co2e_g(self):
        return self.kwh() * self.grid

    def short(self):
        cost = self.cost_gbp()
        return f"{self.kwh():.2f} kWh, {self.co2e_g():,.0f} g CO2e" + (f", £{cost:.2f}" if cost is not None else "")

    def describe(self):
        gpu = (f"GPU {self.gpu_index} measured with nvidia-smi every {self.period:g} s" if self.gpu_measured
               else "GPU not measured on this device")
        host = (f"CPU estimated as {self.cpu_watts:g} W x this run's share of CPU time (--cpu-watts), memory as "
                f"{RAM_W_PER_GB} W/GB" if self._proc is not None else "CPU and memory not estimated (no psutil)")
        cost = (f"; cost at {self.tariff.name} unit rates incl. VAT (--tariff, --region; standing charge not counted)"
                if self.tariff is not None else "")
        return (f"energy: {gpu}; {host}; CO2e at {self.grid:g} g/kWh (--grid-intensity; default: GB grid average "
                f"Oct 2025-Sep 2026){cost}. Excludes screen, charger losses, data preparation and hardware "
                f"manufacture.")


class TokenBatcher:
    """Random fixed-length windows from memory-mapped token files.

    Each sequence picks a source with probability proportional to
    weight x tokens (so --mix md=3 over-samples Markdown 3x per token), then a
    uniformly random start. Only the batch is materialised; the corpus stays on
    disk. The torch.Generator state is saved in checkpoints for exact resume.
    """

    def __init__(self, arrays, weights, ctx, batch_size, device, seed):
        self.names = sorted(n for n in arrays if weights.get(n, 0) > 0)
        self.arrays, self.ctx, self.batch_size, self.device = arrays, ctx, batch_size, device
        self.lens = [len(arrays[n]) for n in self.names]
        short = [n for n, length in zip(self.names, self.lens) if length <= ctx + 1]
        if short:
            raise SystemExit(f"Training source(s) {short} have fewer than ctx+2 = {ctx + 2} tokens; "
                             "lower --ctx or exclude them with --mix name=0.")
        raw = torch.tensor([weights[n] * length for n, length in zip(self.names, self.lens)], dtype=torch.float64)
        self.probs = (raw / raw.sum()).float()
        self.gen = torch.Generator().manual_seed(int(seed))
        self.offsets = np.arange(ctx + 1, dtype=np.int64)
        self.pin = device.type == "cuda"

    def share(self):
        return {n: float(p) for n, p in zip(self.names, self.probs)}

    def next(self):
        b, ctx = self.batch_size, self.ctx
        pick = torch.multinomial(self.probs, b, replacement=True, generator=self.gen)
        buf = np.empty((b, ctx + 1), dtype=np.int64)
        for j, name in enumerate(self.names):
            rows = (pick == j).nonzero().flatten()
            if len(rows) == 0:
                continue
            starts = torch.randint(0, self.lens[j] - ctx, (len(rows),), generator=self.gen).numpy()
            buf[rows.numpy()] = self.arrays[name][starts[:, None] + self.offsets[None, :]]
        t = torch.from_numpy(buf)
        if self.pin:
            t = t.pin_memory()
        t = t.to(self.device, non_blocking=True)
        return t[:, :-1], t[:, 1:]

    def state_dict(self):
        return {"generator": self.gen.get_state()}

    def load_state_dict(self, state):
        if state and "generator" in state:
            self.gen.set_state(state["generator"].cpu() if isinstance(state["generator"], torch.Tensor)
                               else state["generator"])


class EvalSet:
    """Fixed, evenly spaced validation windows per source.

    The same windows are scored at every evaluation, so successive validation
    losses differ only because the model changed (no sampling noise), and
    NEW BEST decisions compare like with like.
    """

    def __init__(self, arrays, ctx, tokens_per_source):
        self.windows, self.skipped = {}, []
        for name, array in sorted(arrays.items()):
            n = min(max(1, tokens_per_source // ctx), (len(array) - 1) // ctx)
            if n < 1:
                self.skipped.append(name)
                continue
            starts = np.linspace(0, len(array) - ctx - 1, n).astype(np.int64)
            self.windows[name] = torch.from_numpy(
                np.asarray(array[starts[:, None] + np.arange(ctx + 1)[None, :]], dtype=np.int64))
        if not self.windows:
            raise SystemExit(f"No validation source has more than ctx+1 = {ctx + 1} tokens; "
                             "rebuild the dataset with a larger --val-fraction or lower --ctx.")

    def describe(self):
        return ", ".join(f"{n} {w.shape[0]} x {w.shape[1] - 1}" for n, w in self.windows.items())


@torch.no_grad()
def evaluate(model, evalset, device, amp_dtype, batch_windows, bins=10, byte_tables=None):
    """Validation loss per source plus how 'opinionated' the model is.

    Returns mean loss per source; 'loss' is the mean over sources. Entropy is
    the mean entropy (nats) of the predicted next-token distribution,
    confidence the mean top-1 probability, top1 the accuracy of the top
    prediction, and ece the expected calibration error of that confidence.
    bpb (bits per byte) = total NLL / (ln 2 x UTF-8 bytes of the predicted
    tokens, counted exactly with Tokenizer.byte_tables): unlike loss per token
    it does not depend on the tokenizer, so models with different vocabularies
    can be compared.
    """
    was_training = model.training
    model.eval()
    per_source = {}
    sums = {"entropy": 0.0, "confidence": 0.0, "top1": 0.0, "tokens": 0}
    conf_sum = torch.zeros(bins, dtype=torch.float64)
    acc_sum = torch.zeros(bins, dtype=torch.float64)
    count = torch.zeros(bins, dtype=torch.float64)
    bpb_source = {}
    for name, windows in evalset.windows.items():
        nll_total, tokens, n_bytes = 0.0, 0, 0
        for start in range(0, windows.shape[0], batch_windows):
            batch = windows[start:start + batch_windows].to(device, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits = model(x)
            logp = F.log_softmax(logits.float(), dim=-1)
            nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            probs = logp.exp()
            entropy = -(probs * logp).sum(-1)
            conf, pred = probs.max(-1)
            correct = (pred == y).double()
            nll_total += nll.double().sum().item()
            tokens += y.numel()
            if byte_tables is not None:
                size, marked, line_break = byte_tables
                yc, xc = y.cpu(), x.cpu()
                n_bytes += int(size[yc].sum()) - int((marked[yc] & line_break[xc]).sum())
            sums["entropy"] += entropy.double().sum().item()
            sums["confidence"] += conf.double().sum().item()
            sums["top1"] += correct.sum().item()
            which = (conf.double().clamp(0, 1 - 1e-9) * bins).long().flatten().cpu()
            conf_sum += torch.bincount(which, conf.double().flatten().cpu(), bins)
            acc_sum += torch.bincount(which, correct.flatten().cpu(), bins)
            count += torch.bincount(which, minlength=bins).double()
        per_source[name] = nll_total / max(tokens, 1)
        if n_bytes:
            bpb_source[name] = nll_total / (math.log(2) * n_bytes)
        sums["tokens"] += tokens
    if was_training:
        model.train()
    n = max(sums["tokens"], 1)
    nz = count > 0
    ece = float(((conf_sum[nz] - acc_sum[nz]).abs()).sum() / n)
    return {
        "loss": float(np.mean(list(per_source.values()))),
        "per_source": per_source,
        "bpb": float(np.mean(list(bpb_source.values()))) if bpb_source else None,
        "bpb_per_source": bpb_source,
        "tokens": sums["tokens"],
        "entropy": sums["entropy"] / n,
        "confidence": sums["confidence"] / n,
        "top1": sums["top1"] / n,
        "ece": ece,
        "calibration": {"confidence": (conf_sum / count.clamp(min=1)).tolist(),
                        "accuracy": (acc_sum / count.clamp(min=1)).tolist(),
                        "count": count.tolist()},
    }


def read_probes(path):
    """Probe TSV. Five columns: category, difficulty, prompt, correct continuation,
    distractor|distractor|...; the older three-column form (prompt, answer,
    distractors) is also accepted (category 'general')."""
    probes = []
    with open(path, encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [p.strip() for p in line.rstrip("\n").split("\t")]
            if len(parts) >= 5:
                category, difficulty, prompt, answer, options = parts[:5]
            elif len(parts) >= 3:
                category, difficulty, (prompt, answer, options) = "general", "unrated", parts[:3]
            else:
                raise SystemExit(f"{path}:{line_no}: expected category<TAB>difficulty<TAB>prompt<TAB>answer<TAB>"
                                 "distractor|distractor (or prompt<TAB>answer<TAB>distractors)")
            distractors = [d.strip() for d in options.split("|") if d.strip()]
            if not prompt or not answer or not distractors:
                raise SystemExit(f"{path}:{line_no}: empty prompt, answer or distractor list")
            probes.append({"category": category, "difficulty": difficulty, "prompt": prompt, "answer": answer,
                           "distractors": distractors})
    return probes


FLOOR_SHUFFLES = 3  # word orders averaged for the floor


def shuffled_prompt(prompt, k):
    """The prompt's words in the k-th fixed pseudo-random order: the topic words
    stay, the sentence is destroyed."""
    words = prompt.split()
    random.Random(int.from_bytes(hashlib.sha256(f"{k}|{prompt}".encode()).digest()[:8], "little")).shuffle(words)
    return " ".join(words)


@torch.no_grad()
def score_probes(model, tok, probes, device, amp_dtype, batch_size=64, floor=True):
    """Cloze fact probes: is the correct continuation more probable (per
    character) than every distractor? A proxy for factual reliability, not a
    hallucination rate. All (probe, option) sequences are scored in right-padded
    batches; causal attention means padding never affects the scored positions.

    With floor=True each probe is also scored with its question's words shuffled
    (FLOOR_SHUFFLES orders): what topic words alone achieve. That share, not
    chance, is the realistic floor; accuracy above it needs the sentence itself."""
    was_training = model.training
    model.eval()
    prefix = [tok.bos_id] if tok.bos_id >= 0 else []
    versions = 1 + (FLOOR_SHUFFLES if floor else 0)
    # tokenizers whose words carry a leading space (Hugging Face byte-level BPE) supply encode_continuation
    encode_cont = getattr(tok, "encode_continuation", tok.encode)
    seqs = []
    for i, probe in enumerate(probes):
        options = [encode_cont(o) for o in [probe["answer"]] + probe["distractors"]]  # once, not per version
        chars = [max(len(o), 1) for o in [probe["answer"]] + probe["distractors"]]
        for v in range(versions):
            context = prefix + tok.encode(probe["prompt"] if v == 0 else shuffled_prompt(probe["prompt"], v))
            for j, cont in enumerate(options):
                ids = (context + cont)[-model.ctx - 1:]
                seqs.append((i, v, j, ids, len(cont), chars[j]))
    scores = [[[None] * (1 + len(p["distractors"])) for _ in range(versions)] for p in probes]
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start:start + batch_size]
        width = max(len(s[3]) for s in chunk) - 1
        x = torch.zeros((len(chunk), width), dtype=torch.long)
        rows, cols, targets, owner = [], [], [], []
        for b, (_, _, _, ids, n_cont, _) in enumerate(chunk):
            x[b, :len(ids) - 1] = torch.tensor(ids[:-1])
            rows += [b] * n_cont
            cols += range(len(ids) - 1 - n_cont, len(ids) - 1)
            targets += ids[len(ids) - n_cont:]
            owner += [b] * n_cont
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            if isinstance(model, TinyGPT):  # the output layer only at the scored positions, not everywhere
                hidden = model(x.to(device), return_hidden=True)
                picked = model.lm_head(hidden[torch.tensor(rows, device=device), torch.tensor(cols, device=device)])
            else:
                logits = model(x.to(device))
                picked = logits[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]
        picked = picked.float()
        target_logp = (picked.gather(-1, torch.tensor(targets, device=device)[:, None]).squeeze(-1)
                       - torch.logsumexp(picked, dim=-1))  # only the scored positions, not the whole vocabulary
        sums = torch.zeros(len(chunk), device=device).index_add_(0, torch.tensor(owner, device=device),
                                                                 target_logp).cpu()
        for b, (i, v, j, _, _, n_chars) in enumerate(chunk):
            scores[i][v][j] = float(sums[b]) / n_chars
    results = []
    for probe, per_version in zip(probes, scores):
        s = per_version[0]
        row = {"category": probe["category"], "difficulty": probe["difficulty"], "prompt": probe["prompt"],
               "answer": probe["answer"], "correct_score": s[0], "best_distractor_score": max(s[1:]),
               "best_distractor": probe["distractors"][int(np.argmax(s[1:]))],
               "chance": 1.0 / len(s), "correct": bool(s[0] > max(s[1:]))}
        if floor:
            row["floor_correct"] = sum(sv[0] > max(sv[1:]) for sv in per_version[1:]) / FLOOR_SHUFFLES
        results.append(row)
    if was_training:
        model.train()
    acc = sum(r["correct"] for r in results) / max(len(results), 1)
    return acc, results


def probe_report(results, probes_sha256):
    """Probe results small enough to keep in a checkpoint: what the dashboard's
    fact-benchmark panel draws, so it need not run the model again."""
    n = len(results)
    return {"file_sha256": probes_sha256, "n": n, "correct": sum(r["correct"] for r in results),
            "chance": sum(r["chance"] for r in results) / max(n, 1), "floor": probe_floor(results),
            "category": probe_summary(results), "difficulty": probe_summary(results, "difficulty")}


def probe_floor(results):
    """Mean accuracy with shuffled questions (None if not scored)."""
    vals = [r["floor_correct"] for r in results if "floor_correct" in r]
    return sum(vals) / len(vals) if vals else None


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def probe_summary(results, key="category"):
    """{group: {n, correct, accuracy, chance, low, high}} for category or difficulty."""
    groups = {}
    for r in results:
        groups.setdefault(r[key], []).append(r)
    out = {}
    for name, rows in groups.items():
        k, n = sum(r["correct"] for r in rows), len(rows)
        low, high = wilson_interval(k, n)
        out[name] = {"n": n, "correct": k, "accuracy": k / n, "chance": sum(r["chance"] for r in rows) / n,
                     "low": low, "high": high, "floor": probe_floor(rows)}
    return out


@torch.no_grad()
def watermark_green(key, prev_token, vocab_size, gamma=0.25, _cache={}):
    """Green list for a watermark (Kirchenbauer et al. 2023): a pseudo-random
    fraction gamma of the vocabulary, seeded by a secret key and the previous
    token. Anyone with the key can recompute it; nobody else can."""
    k = (key, int(prev_token), vocab_size, gamma)
    if k not in _cache:
        seed = int.from_bytes(hashlib.sha256(f"{key}|{int(prev_token)}".encode()).digest()[:8], "little")
        perm = torch.randperm(vocab_size, generator=torch.Generator().manual_seed(seed))
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        mask[perm[: int(gamma * vocab_size)]] = True
        if len(_cache) > 50_000:
            _cache.clear()
        _cache[k] = mask
    return _cache[k]


def generate(model, tok, prompt, max_new_tokens, device, amp_dtype=None, temperature=0.8, top_k=50,
             top_p=0.95, generator=None, watermark=None):
    """Sample a continuation. Sampling runs on CPU with its own generator, so
    it never disturbs the training random stream. watermark=(key, gamma, delta)
    adds delta to the logits of the key's green list at every step, leaving a
    statistical signature that detect_text.py can test for."""
    was_training = model.training
    model.eval()
    prefix = [tok.bos_id] if tok.bos_id >= 0 else []
    ids = prefix + (tok.encode(prompt) if prompt else ([] if prefix else tok.encode("\n")))
    start = len(prefix)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.ctx:]], device=device)
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            logits = model(x)[0, -1].float().cpu()
        if watermark:
            key, gamma, delta = watermark
            logits = logits + delta * watermark_green(key, ids[-1], logits.numel(), gamma).float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k and top_k < logits.numel():
                kth = torch.topk(logits, top_k).values[-1]
                logits[logits < kth] = float("-inf")
            probs = logits.softmax(-1)
            if 0 < top_p < 1:
                sorted_p, order = probs.sort(descending=True)
                drop = sorted_p.cumsum(0) - sorted_p > top_p
                sorted_p[drop] = 0
                probs = torch.zeros_like(probs).scatter(0, order, sorted_p)
                probs /= probs.sum()
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        if nxt == tok.eos_id:
            break
        ids.append(nxt)
    if was_training:
        model.train()
    return tok.decode(ids[start:])


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
LR_TABLE = [(1e6, 2.0e-3), (1e7, 1.2e-3), (1e8, 6.0e-4), (1e9, 3.0e-4)]


def heuristic_lr(n_params):
    """Log-log interpolation of peak AdamW learning rates used for published
    decoder models (GPT-3 125M 6e-4, Pythia-70M 1e-3, 160M 6e-4, 1B 3e-4),
    extended downward for small models. A starting point, not an optimum."""
    x = math.log10(max(n_params, 1))
    pts = [(math.log10(n), math.log10(lr)) for n, lr in LR_TABLE]
    if x <= pts[0][0]:
        return 10 ** pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return 10 ** (y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return 10 ** pts[-1][1]


def default_layers(d_model):
    return int(min(max(round(d_model / 64) + 2, 4), 32))


def default_head_dim(d_model):
    return 32 if d_model <= 256 else (64 if d_model <= 1024 else 128)


def default_kv_heads(n_heads):
    """GQA with 4 query heads per K/V head once there are >= 8 heads."""
    return n_heads // 4 if n_heads >= 8 and n_heads % 4 == 0 else n_heads


def shape_ladder(vocab_size, ctx=512):
    """Candidate shapes, depth growing with width (small models favour depth)."""
    out = []
    for d in (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 2048, 2560, 3072):
        heads = d // default_head_dim(d)
        out.append(ModelConfig(vocab_size=vocab_size, ctx=ctx, d_model=d, n_layers=default_layers(d),
                               n_heads=heads, n_kv_heads=default_kv_heads(heads)))
    return out


def tokens_per_step_for(n_params):
    """~16 * sqrt(N) tokens per optimizer step, rounded to a power of two,
    between 16k and 512k (small models reach their critical batch early)."""
    raw = 16 * math.sqrt(n_params)
    return int(min(max(2 ** round(math.log2(raw)), 16_384), 524_288))


def device_memory_budget(device):
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(free * 0.80), f"{free / 2 ** 30:.1f} GiB free on the GPU"
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except ImportError:
        avail = 8 * 2 ** 30
    if device.type == "mps":
        rec = getattr(torch.mps, "recommended_max_memory", None)
        budget = rec() if rec else avail
        return int(min(budget, avail) * 0.6), f"{min(budget, avail) / 2 ** 30:.1f} GiB usable unified memory"
    return int(avail * 0.4), f"{avail / 2 ** 30:.1f} GiB free RAM"


def estimate_train_bytes(cfg, micro_batch, amp=True, checkpointing=False, loss_chunk=0, optimizer="adamw"):
    """Training memory: FP32 weights and gradients (4 + 4 bytes per parameter),
    optimizer state (AdamW 8; Muon 4 for the block matrices), activations kept
    for the backward pass (only each block's input with activation
    checkpointing, plus one block being recomputed) and the output-layer logits
    (one piece of loss_chunk tokens when the loss is computed in pieces)."""
    n_total, n_nonembed = count_params(cfg)
    t = micro_batch * cfg.ctx
    b = 2 if amp else 4
    kv = cfg.n_kv_heads * cfg.head_dim
    per_token_layer = 4 * cfg.d_model + b * (7 * cfg.d_model + 4 * kv + 4 * cfg.ffn_hidden) + 3 * cfg.n_heads
    if checkpointing:
        activations = t * cfg.n_layers * b * cfg.d_model + t * per_token_layer
    else:
        activations = t * cfg.n_layers * per_token_layer
    logits = (min(t, loss_chunk) if loss_chunk else t) * cfg.vocab_size * (b + 4 + 4)
    if loss_chunk:
        logits += t * cfg.d_model * (b + 4)  # the hidden states and their gradient, kept whole
    state = 8 * n_total if optimizer != "muon" else 8 * (n_total - n_nonembed) + 4 * n_nonembed
    return int((activations + logits) * 1.3 + n_total * 8 + state + 400 * 2 ** 20)


def memory_options(checkpointing="auto", loss_chunk=None):
    """(activation checkpointing, loss chunk tokens) choices allowed by the
    command line, cheapest first: the chunked loss costs a few per cent more
    compute, activation checkpointing about a third."""
    ck = [False, True] if checkpointing == "auto" else [checkpointing == "on"]
    lc = [0, LOSS_CHUNK_TOKENS] if loss_chunk is None else [loss_chunk]
    return sorted(((c, l) for c in ck for l in lc), key=lambda o: (o[0], o[1] != 0))


def choose_memory_plan(cfg, tokens_per_step, budget_bytes, amp, optimizer="adamw", checkpointing="auto",
                       loss_chunk=None, micro=None):
    """(micro-batch, accumulation, activation checkpointing, loss chunk tokens):
    the cheapest memory option that reaches a micro-batch of min(target, 4)
    sequences, else the cheapest that fits at all. With `micro` given (an
    explicit --batch-size), the cheapest option that fits that batch."""
    target = max(1, tokens_per_step // cfg.ctx)
    options = memory_options(checkpointing, loss_chunk)
    fits = lambda m, o: estimate_train_bytes(cfg, m, amp, o[0], o[1], optimizer) <= budget_bytes  # noqa: E731
    if micro is not None:
        chosen = next((o for o in options if fits(micro, o)), options[-1])
        return micro, max(1, target // micro), chosen[0], chosen[1]
    best = {}
    for o in options:
        if fits(1, o):
            m = 1
            while m * 2 <= target and fits(m * 2, o):
                m *= 2
            best[o] = m
    if not best:
        raise SystemExit(f"The model does not fit in memory even at batch 1 (ctx {cfg.ctx}) with activation "
                         "checkpointing and the loss in pieces; choose a smaller --d-model/--layers/--ctx.")
    want = min(target, 4)
    o = next((o for o in options if best.get(o, 0) >= want), next(o for o in options if o in best))
    return best[o], max(1, target // best[o]), o[0], o[1]


def choose_micro_batch(cfg, tokens_per_step, budget_bytes, amp):
    micro, accum, _, _ = choose_memory_plan(cfg, tokens_per_step, budget_bytes, amp)
    return micro, accum


def plan_run(args, manifest, device, amp_dtype, token_cap=None):
    """Fill every unset hyper-parameter from the dataset and device.

    token_cap limits the training-token budget (used by --time-budget-hours).
    Returns (ModelConfig, train settings dict, explanation lines)."""
    vocab = int(manifest["tokenizer"]["vocab_size"])
    train_by_source = {n: int(s["train"]["tokens"]) for n, s in manifest["sources"].items() if s.get("train")}
    unique = sum(train_by_source.values())
    lines = []
    budget_bytes, budget_text = device_memory_budget(device)
    r = args.tokens_per_param
    max_epochs = args.max_epochs

    # --- training-token budget D ---
    # D is fixed by the data (or by an explicit token count). --steps alone decides
    # how many updates share D; if --steps x the batch exceeds D, a note says so.
    if args.steps and args.tokens_per_step:
        planned = int(args.steps * args.tokens_per_step)
        why = "--steps x --tokens-per-step"
    elif args.train_tokens:
        planned = int(args.train_tokens)
        why = "--train-tokens"
    elif args.epochs:
        planned = int(args.epochs * unique)
        why = f"--epochs {args.epochs}"
    else:
        planned = int(max_epochs * unique)
        why = f"--max-epochs {max_epochs} x {unique:,} unique training tokens"
    if token_cap is not None and token_cap < planned:
        planned = int(token_cap)
        why += f"; capped to {planned:,} by --time-budget-hours"

    # --- model shape ---
    # Largest ladder shape whose parameter count N satisfies D >= r * N.
    p90 = max(int(s.get("doc_tokens_percentiles", {}).get("90", 512)) for s in manifest["sources"].values())
    ladder = shape_ladder(vocab)
    fitting = [c for c in ladder if count_params(c)[0] * r <= planned]
    target = planned / r
    explicit_shape = any(v is not None for v in (args.d_model, args.layers, args.heads, args.kv_heads))

    def shaped(base):
        cfg = ModelConfig(**dataclasses.asdict(base))
        if explicit_shape:
            if args.d_model is not None:
                cfg.d_model = args.d_model
                cfg.n_layers = default_layers(args.d_model)
                cfg.n_heads = max(1, args.d_model // default_head_dim(args.d_model))
            if args.layers is not None:
                cfg.n_layers = args.layers
            if args.heads is not None:
                cfg.n_heads = args.heads
            cfg.n_kv_heads = args.kv_heads if args.kv_heads is not None else default_kv_heads(cfg.n_heads)
        if args.ctx:
            cfg.ctx = args.ctx
        else:
            cap = 256 if cfg.d_model <= 192 else (512 if cfg.d_model <= 512 else 1024)
            want = 2 ** math.ceil(math.log2(max(p90, 256)))
            cfg.ctx = int(min(cap, want))
        cfg.ffn_hidden = args.ffn_hidden or swiglu_hidden(cfg.d_model)
        if args.rope_base:
            cfg.rope_base = args.rope_base
        cfg.validate()
        return cfg

    def batch_for(cfg):
        n = count_params(cfg)[0]
        if args.tokens_per_step:
            tps = args.tokens_per_step
        elif args.steps:
            tps = max(cfg.ctx, planned // args.steps)
        else:
            tps = tokens_per_step_for(n)
        amp, opt_kind = amp_dtype is not None, args.optimizer or "adamw"
        if args.batch_size:
            micro, accum, ck, lc = choose_memory_plan(cfg, tps, budget_bytes, amp, opt_kind,
                                                      args.activation_checkpointing, args.loss_chunk_tokens,
                                                      micro=args.batch_size)
            accum = args.grad_accum or max(1, tps // (micro * cfg.ctx))
            if estimate_train_bytes(cfg, micro, amp, ck, lc, opt_kind) > budget_bytes:
                lines.append(f"WARNING: batch {micro} x ctx {cfg.ctx} may not fit ({budget_text}); an "
                             "out-of-memory error at step 1 switches on memory savings or halves the batch.")
            return micro, accum, ck, lc
        micro, accum, ck, lc = choose_memory_plan(cfg, tps, budget_bytes, amp, opt_kind,
                                                  args.activation_checkpointing, args.loss_chunk_tokens)
        return micro, (args.grad_accum or accum), ck, lc

    if explicit_shape:
        cfg = shaped(fitting[-1] if fitting else ladder[0])  # options given override the data-chosen shape
        shape_why = "command-line values, ladder defaults for the rest"
        micro, accum, ckpt, chunk = batch_for(cfg)
    else:
        shape_why = (f"largest ladder shape with <= D/{r:g} = {target / 1e6:,.2f}M parameters"
                     if fitting else f"smallest ladder shape (the data supports only {target / 1e6:,.2f}M)")
        candidates = (fitting or ladder[:1])[::-1]  # the data-chosen shape first, then smaller ones
        for k, base in enumerate(candidates):
            cfg = shaped(base)
            try:
                micro, accum, ckpt, chunk = batch_for(cfg)
            except SystemExit:
                if k == len(candidates) - 1:
                    raise
                continue
            if k:
                shape_why += (f"; the larger {count_params(shaped(candidates[0]))[0] / 1e6:,.1f}M shape does not fit "
                              f"in memory, so the largest that fits")
            break
    n_total, n_nonembed = count_params(cfg)
    tokens_per_step = micro * accum * cfg.ctx

    steps = args.steps or max(1, planned // tokens_per_step)  # round down: never exceed the epoch cap
    train_tokens = steps * tokens_per_step
    epochs = train_tokens / max(unique, 1)
    dropout = args.dropout if args.dropout is not None else (0.0 if epochs <= 4 else 0.1)
    cfg.dropout = dropout
    lr = args.lr or heuristic_lr(n_total)
    warmup = args.warmup_steps if args.warmup_steps is not None else int(min(max(round(0.02 * steps), 50), 2000, max(1, steps // 10)))
    settings = {
        "steps": steps, "micro_batch": micro, "grad_accum": accum, "tokens_per_step": tokens_per_step,
        "lr": lr, "min_lr": lr * args.min_lr_ratio, "warmup_steps": warmup, "schedule": args.schedule,
        "decay_frac": args.decay_frac, "decay_shape": args.decay_shape, "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2], "grad_clip": args.grad_clip, "planned_train_tokens": train_tokens,
        "optimizer": args.optimizer or "adamw", "activation_checkpointing": ckpt, "loss_chunk_tokens": chunk,
    }
    lines += [
        f"unique training tokens U = {unique:,} ({', '.join(f'{k} {v:,}' for k, v in train_by_source.items())}); "
        f"vocabulary {vocab:,}; {budget_text}",
        f"token budget D = {planned:,} ({why}); shape: {shape_why}",
        f"model: d_model {cfg.d_model}, layers {cfg.n_layers}, heads {cfg.n_heads}/{cfg.n_kv_heads} kv "
        f"(head {cfg.head_dim}), SwiGLU {cfg.ffn_hidden}, ctx {cfg.ctx}; parameters {n_total:,} total, "
        f"{n_nonembed:,} excluding the (tied) embedding",
        f"batch: {micro} x {cfg.ctx} tokens x {accum} accumulation = {tokens_per_step:,} tokens/step; "
        f"{steps:,} steps = {train_tokens:,} training tokens",
        f"ratios: {train_tokens / n_total:,.1f} training tokens per parameter "
        f"({train_tokens / max(n_nonembed, 1):,.1f} per non-embedding parameter); "
        f"unique data {unique / n_total:,.1f} tokens per parameter; {epochs:,.2f} epochs",
        f"learning rate {lr:.2e} peak (heuristic for {n_total / 1e6:,.1f}M parameters), min {lr * args.min_lr_ratio:.2e}; "
        f"dropout {dropout} ({'set' if args.dropout is not None else 'auto: 0 for <= 4 epochs, else 0.1'}); "
        f"weight decay {args.weight_decay} on matrices only; optimizer {OPTIMIZER_LABELS[args.optimizer or 'adamw']}",
        "memory: " + ("activation checkpointing on (blocks recomputed in the backward pass)" if ckpt
                      else "activation checkpointing off") + "; "
        + (f"loss computed {chunk:,} tokens at a time" if chunk else "loss computed on the whole micro-batch")
        + f" (estimated {estimate_train_bytes(cfg, micro, amp_dtype is not None, ckpt, chunk, args.optimizer or 'adamw') / 2 ** 30:.1f} GiB "
        f"of {budget_text})",
    ]
    if args.steps and train_tokens > max_epochs * unique:
        lines.append(f"note: --steps {args.steps:,} trains {epochs:.2f} epochs, more than --max-epochs {max_epochs:g}.")
    if epochs > 4:
        lines.append(f"note: {epochs:.1f} epochs repeats data beyond the ~4 epochs that Muennighoff et al. (2023) "
                     "found nearly as good as fresh data; expect diminishing returns and watch the val-train gap.")
    return cfg, settings, lines


def measure_seconds_per_step(cfg, micro, accum, device, amp_dtype, attn_pref="auto", warmup=2, timed=3,
                             checkpointing=False, loss_chunk=0):
    """Time real training micro-steps of this shape on this device (random tokens),
    with the planned memory options. Runs inside fork_rng so the caller's random
    state (model initialisation) is untouched."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(0)
        return _measure(cfg, micro, accum, device, amp_dtype, attn_pref, warmup, timed, checkpointing, loss_chunk)


def _measure(cfg, micro, accum, device, amp_dtype, attn_pref, warmup, timed, checkpointing=False, loss_chunk=0):
    probe_cfg = ModelConfig(**dataclasses.asdict(cfg))
    model = TinyGPT(probe_cfg).to(device)
    model.set_attention_impl(choose_attention(device, probe_cfg, amp_dtype, attn_pref)[0])
    model.grad_checkpoint = checkpointing
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, **({"fused": True} if device.type == "cuda" else {}))
    x = torch.randint(0, cfg.vocab_size, (micro, cfg.ctx + 1), device=device)

    def micro_step():
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            if loss_chunk:
                ce, _, n = chunked_lm_loss(model(x[:, :-1], return_hidden=True), model.lm_head.weight, x[:, 1:],
                                           loss_chunk)
                loss = ce / n
            else:
                loss = F.cross_entropy(model(x[:, :-1]).float().view(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        loss.backward()

    try:
        for _ in range(warmup):
            micro_step()
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(timed):
            micro_step()
        opt.step()
        synchronize(device)
        return (time.perf_counter() - t0) / timed * accum
    finally:
        del model, opt, x
        if device.type == "cuda":
            torch.cuda.empty_cache()


def flops_per_token(cfg):
    """Training floating-point operations per token: 6 per parameter (forward and
    backward) plus the attention scores, 12 x layers x context x width."""
    return 6 * count_params(cfg)[0] + 12 * cfg.n_layers * cfg.ctx * cfg.d_model


def sustained_speed(out_dir, device):
    """(FLOP/s, description) that this computer sustained in its last training run
    of at least 12 minutes (experiments.csv), or (None, reason). A short benchmark
    on a cool laptop GPU runs at boost clocks; over hours the power and heat limits
    slow it down (1.7x on this RTX 3070), so a real run is the better guide."""
    import csv
    if device.type != "cuda":
        return None, "not a CUDA device"
    try:
        with open(os.path.join(out_dir, "experiments.csv"), encoding="utf-8-sig", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("event") == "training run"
                    and (r.get("device") or "cuda").startswith("cuda")]
    except OSError:
        return None, "no experiments.csv"
    for r in reversed(rows):
        try:
            cfg = ModelConfig(vocab_size=int(r["vocab"]), ctx=int(r["ctx"]), d_model=int(r["d_model"]),
                              n_layers=int(r["layers"]), n_heads=int(r["heads"]), n_kv_heads=int(r["kv_heads"]))
            tokens, hours = float(r["tokens_seen"]), float(r["train_hours"])
        except (KeyError, TypeError, ValueError):
            continue
        if hours >= 0.2 and tokens > 0:
            speed = tokens / (hours * 3600) * flops_per_token(cfg)
            return speed, (f"{speed / 1e12:.1f} TFLOP/s sustained by run '{r.get('name')}' of {r.get('time', '')[:10]} "
                           f"({int(r['params']) / 1e6:.1f}M parameters, {tokens / hours / 3600:,.0f} tokens/s)")
    return None, "no training run of 12 minutes or more yet"


def plan_with_time_budget(args, manifest, device, amp_dtype):
    """plan_run, then (with --time-budget-hours) choose the largest ladder shape
    whose parameter count N still satisfies tokens >= r * N when tokens are
    limited by BOTH the data cap and what fits in the time budget at the speed
    measured on this device. Binary search over the ladder (bigger = slower, so
    the condition is monotone); evaluation/sampling/checkpoint overhead 8%."""
    cfg, settings, lines = plan_run(args, manifest, device, amp_dtype)
    hours = getattr(args, "time_budget_hours", None)
    if not hours:
        return cfg, settings, lines
    budget_s = hours * 3600 * 0.92
    r = args.tokens_per_param
    speed, speed_note = sustained_speed(args.out_dir, device)
    notes = [f"time budget: speed from {speed_note}" if speed else
             f"time budget: {speed_note}, so short measurements are slowed by 1.6x (the gap between a cool and a "
             "hot laptop GPU); later plans use the speed of your runs"]

    def trial(shape_args):
        c, st, _ = plan_run(shape_args, manifest, device, amp_dtype)
        measured = measure_seconds_per_step(c, st["micro_batch"], st["grad_accum"], device, amp_dtype, args.attn,
                                            checkpointing=st["activation_checkpointing"],
                                            loss_chunk=st["loss_chunk_tokens"])
        sec = max(measured, st["tokens_per_step"] * flops_per_token(c) / speed) if speed else measured * 1.6
        fit_tokens = int(budget_s / sec) * st["tokens_per_step"]
        tokens = min(fit_tokens, st["planned_train_tokens"])
        n = count_params(c)[0]
        notes.append(f"time budget {hours:g} h: {n / 1e6:,.1f}M parameters, {sec:.2f} s/step expected over hours "
                     f"({measured:.2f} measured in a short test; {st['tokens_per_step']:,} tokens/step) -> "
                     f"{fit_tokens:,} tokens fit in time")
        return n * r <= tokens, tokens

    explicit = any(v is not None for v in (args.d_model, args.layers, args.heads, args.kv_heads))
    if explicit:
        _, tokens = trial(args)
        chosen_args = args
    else:
        ladder = shape_ladder(int(manifest["tokenizer"]["vocab_size"]))
        top = max(i for i, c in enumerate(ladder) if c.d_model <= cfg.d_model)

        def shape_args(i):
            c = ladder[i]
            return argparse.Namespace(**{**vars(args), "d_model": c.d_model, "layers": c.n_layers,
                                         "heads": c.n_heads, "kv_heads": c.n_kv_heads})

        results = {}

        def ok(i):
            if i not in results:
                try:
                    results[i] = trial(shape_args(i))
                except SystemExit:
                    results[i] = (False, 0)
                except RuntimeError as exc:  # torch.cuda.OutOfMemoryError is a RuntimeError
                    if "out of memory" not in str(exc).lower():
                        raise
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    results[i] = (False, 0)
            return results[i][0]

        if ok(top):
            best = top
        else:
            lo, hi, best = 0, top - 1, None
            while lo <= hi:
                mid = (lo + hi) // 2
                if ok(mid):
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            if best is None:
                best = 0
                ok(0)
                notes.append("time budget: even the smallest shape cannot reach the tokens-per-parameter target "
                             "in time; it trains on what fits")
        tokens = results[best][1]
        chosen_args = shape_args(best)
    cfg, settings, lines = plan_run(chosen_args, manifest, device, amp_dtype, token_cap=tokens)
    lines[1] = lines[1].replace("command-line values, ladder defaults for the rest",
                                "largest ladder shape that satisfies the data AND the time budget"
                                if not explicit else "command-line values, tokens capped by the time budget")
    return cfg, settings, lines + notes


def plan_table(vocab, device, amp_dtype, r, max_epochs):
    """What the planner would choose for other corpus sizes (same vocabulary)."""
    rows = []
    budget, _ = device_memory_budget(device)
    for unique in (5e6, 2e7, 1e8, 3e8, 1e9, 1e10):
        planned = max_epochs * unique  # the planner's default: up to --max-epochs passes over the data
        ladder = shape_ladder(vocab)
        fitting = [c for c in ladder if count_params(c)[0] <= planned / r]
        cfg = fitting[-1] if fitting else ladder[0]
        cfg.ctx = 256 if cfg.d_model <= 192 else (512 if cfg.d_model <= 512 else 1024)
        n, _ = count_params(cfg)
        tps = tokens_per_step_for(n)
        try:
            micro, accum = choose_micro_batch(cfg, tps, budget, amp_dtype is not None)
            fits = f"{micro}x{accum}"
        except SystemExit:
            fits = "does not fit"
        rows.append((unique, planned, cfg, n, tps, fits, heuristic_lr(n)))
    return rows


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("mode")
    g.add_argument("--generate", metavar="PROMPT", help="generate text from a checkpoint and exit")
    g.add_argument("--plan", action="store_true", help="print the automatic plan for --dataset and exit")
    g.add_argument("--export", metavar="FOLDER",
                   help="write the checkpoint as model.safetensors + config.json + tokenizer.model and exit")
    g.add_argument("--benchmark", action="store_true",
                   help="score the fact benchmark (--probes) on a checkpoint, per category, and exit")
    g = p.add_argument_group("data")
    g.add_argument("--dataset", help="folder built by data_prep.py")
    g.add_argument("--mix", help="source weights per token, e.g. md=3,wiki=1 (default: proportional)")
    g.add_argument("--data", help="(compat) folder of .md files: builds datasets/<name> with data_prep defaults")
    g.add_argument("--wiki-dir", default=os.path.join(os.path.expanduser("~"), "ai_training_data", "wikipedia"),
                   help="(compat)")
    g.add_argument("--wiki-articles", type=int, default=0, help="(compat) Wikipedia articles to add")
    g.add_argument("--include-keywords", help="(compat) keyword filter passed to data_prep")
    g.add_argument("--tokenizer-vocab-size", type=int, default=0, help="(compat) vocabulary for auto-prep")
    g = p.add_argument_group("run")
    g.add_argument("--name", default=DEFAULT_NAME, help="checkpoint base name")
    g.add_argument("--out-dir", default=HERE, help="folder for checkpoints and metrics")
    g.add_argument("--resume", action="store_true", help="continue <name>.pt exactly where it stopped")
    g.add_argument("--init-from", help="start a new run from another checkpoint's weights (tinyGPT or legacy)")
    g.add_argument("--overwrite", action="store_true",
                   help="allow a fresh run over an existing name (old files are renamed *.bak-<time>)")
    g.add_argument("--log-file", default=os.path.join(HERE, "log.txt"))
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--trust-checkpoint", action="store_true",
                   help="allow full unpickling of a checkpoint that fails the safe loader (only your own files)")
    g = p.add_argument_group("model (unset = planned from data)")
    g.add_argument("--d-model", type=int)
    g.add_argument("--layers", type=int)
    g.add_argument("--heads", type=int)
    g.add_argument("--kv-heads", type=int)
    g.add_argument("--ctx", type=int)
    g.add_argument("--ffn-hidden", type=int)
    g.add_argument("--rope-base", type=float)
    g.add_argument("--dropout", type=float)
    g = p.add_argument_group("budget")
    g.add_argument("--steps", type=int, help="optimizer steps (on --resume: new total)")
    g.add_argument("--train-tokens", type=float, help="training tokens instead of --steps")
    g.add_argument("--epochs", type=float, help="passes over the unique training tokens")
    g.add_argument("--tokens-per-param", type=float, default=20.0,
                   help="planner target of training tokens per parameter (Chinchilla-style heuristic)")
    g.add_argument("--max-epochs", type=float, default=4.0, help="planner cap on repeated passes over the data")
    g.add_argument("--time-budget-hours", type=float,
                   help="measure speed on this device and shrink tokens/model so training fits in this many hours")
    g = p.add_argument_group("batch and optimizer")
    g.add_argument("--batch-size", type=int, help="sequences per micro-batch")
    g.add_argument("--grad-accum", type=int, help="micro-batches per optimizer step")
    g.add_argument("--tokens-per-step", type=int)
    g.add_argument("--lr", type=float, help="peak learning rate")
    g.add_argument("--min-lr-ratio", type=float, default=0.1)
    g.add_argument("--warmup-steps", type=int)
    g.add_argument("--schedule", choices=("wsd", "cosine"), default="wsd")
    g.add_argument("--decay-frac", type=float, default=0.2, help="WSD: share of steps in the final decay")
    g.add_argument("--decay-shape", choices=("1-sqrt", "cosine", "linear"), default="1-sqrt")
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.95)
    g.add_argument("--grad-clip", type=float, default=1.0, help="clip the gradient norm to this (0 = off)")
    g.add_argument("--activation-checkpointing", choices=("auto", "on", "off"), default="auto",
                   help="recompute each block in the backward pass instead of storing its activations: much less "
                        "memory, about a third more compute (auto: only when the model would not otherwise fit "
                        "with a micro-batch of 4); may be changed on --resume")
    g.add_argument("--loss-chunk-tokens", type=int,
                   help="compute the output layer and loss this many tokens at a time, so the full tokens x "
                        "vocabulary table never exists (0 = off; default: auto, %d when it is needed to fit); may be "
                        "changed on --resume" % LOSS_CHUNK_TOKENS)
    g.add_argument("--optimizer", choices=("adamw", "muon"),
                   help="adamw (default), or muon: Muon for the matrices inside the blocks and AdamW for the embedding "
                        "and norm gains, both driven by --lr and the schedule (Muon's step is scaled to AdamW's "
                        "update size); resumed runs keep their optimizer")
    g.add_argument("--z-loss", type=float, default=1e-4,
                   help="weight of the z-loss stability term (0 disables; resumed runs keep their setting)")
    g = p.add_argument_group("evaluation and logging")
    g.add_argument("--eval-every", type=int, default=200, help="steps between validation + metrics line")
    g.add_argument("--eval-tokens", type=int, default=65_536, help="fixed validation tokens per source")
    g.add_argument("--heartbeat-seconds", type=int, default=120,
                   help="print a short training-only progress line when nothing was printed for this long (0 = off)")
    g.add_argument("--log-every", "--report-every", type=int, default=0,
                   help="extra training-only progress lines between evaluations (0 = none)")
    g.add_argument("--dashboard", choices=("auto", "on", "off"), default="auto",
                   help="redraw view_pt.py's dashboard in the background after each new best checkpoint (auto: "
                        "for runs saved in this script's folder; test runs elsewhere are skipped)")
    g.add_argument("--dashboard-push-hours", type=float, default=3.0,
                   help="push the dashboard to GitHub at most this often (0 = every time; default 3, since every "
                        "1.6 MB version stays in the repository's history)")
    g.add_argument("--sample-every", type=int,
                   help="write a sample at evaluations whose step is a multiple of N (default: every evaluation; 0 = off)")
    g.add_argument("--sample-tokens", type=int, default=48)
    g.add_argument("--sample-prompt", default="Genetic drift is")
    g.add_argument("--save-every", type=int, default=1000, help="steps between training-state saves")
    g.add_argument("--save-minutes", type=float, default=30.0,
                   help="also save the training state after this many minutes (0 = off), so a shutdown loses little")
    g.add_argument("--stop-after-steps", type=int,
                   help="end this session cleanly after N more steps (checkpoint saved; continue with --resume)")
    g.add_argument("--probes", default=DEFAULT_PROBES, help="cloze fact probes TSV scored at each evaluation")
    g.add_argument("--no-probes", action="store_true")
    g = p.add_argument_group("system")
    g.add_argument("--cpu-watts", type=float, default=35.0,
                   help="rated power of the CPU, for the energy estimate (i7-11370H: 35 W)")
    g.add_argument("--grid-intensity", type=float, default=GRID_G_PER_KWH,
                   help="grams CO2e per kWh of electricity (default: GB grid average, Oct 2025-Sep 2026)")
    g.add_argument("--tariff", default="VAR-22-11-01",
                   help="electricity price for the cost estimate: an Octopus product code (VAR-22-11-01 Flexible "
                        "Octopus, AGILE-24-10-01 Agile, GO-VAR-22-10-14 Go, COSY-22-12-08 Cosy, or a fixed tariff's "
                        "code) or a price in pence per kWh, e.g. 24.5")
    g.add_argument("--region", default="H",
                   help="Octopus region letter of your supply (A to P, as on your bill; H = Southern Electric)")
    g.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    g.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    g.add_argument("--compile", nargs="?", const="on", choices=("auto", "on", "off"), default="auto")
    g.add_argument("--no-compile", dest="compile", action="store_const", const="off")
    g.add_argument("--attn", choices=("auto", "gqa_native", "repeat_kv"), default="auto")
    g.add_argument("--threads", type=int, help="CPU threads for PyTorch")
    g.add_argument("--suppress", action="store_true", help=argparse.SUPPRESS)  # old script's option: accepted, no effect
    g = p.add_argument_group("generation")
    g.add_argument("--checkpoint",
                   help="checkpoint for --generate/--benchmark/--export (default <name>_best.pt, else <name>.pt)")
    g.add_argument("--length", type=int, default=200, help="new tokens to generate")
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--watermark-key", help="secret key: embed a detectable watermark in generated text")
    g.add_argument("--watermark-gamma", type=float, default=0.25, help="green-list share of the vocabulary")
    g.add_argument("--watermark-delta", type=float, default=2.0, help="logit boost for green tokens")
    args = p.parse_args(argv)
    validate_args(args, p)
    return args


def validate_args(args, parser):
    if not NAME_RE.fullmatch(args.name):
        parser.error("--name may contain letters, digits, '.', '_' and '-' only (no folders)")
    positive = ["steps", "d_model", "layers", "heads", "kv_heads", "ctx", "ffn_hidden", "batch_size", "grad_accum",
                "tokens_per_step", "threads", "eval_every", "eval_tokens", "save_every", "sample_tokens", "length"]
    for name in positive:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name, lo, hi in (("lr", 1e-7, 1.0), ("min_lr_ratio", 0.0, 1.0), ("decay_frac", 0.0, 0.9),
                         ("weight_decay", 0.0, 1.0), ("beta1", 0.0, 0.9999), ("beta2", 0.0, 0.99999),
                         ("grad_clip", 0.0, 1e3), ("dropout", 0.0, 0.9), ("top_p", 0.0, 1.0),
                         ("tokens_per_param", 0.1, 1e6), ("max_epochs", 0.01, 1e4), ("temperature", 0.0, 10.0)):
        value = getattr(args, name)
        if value is not None and not lo <= value <= hi:
            parser.error(f"--{name.replace('_', '-')} must be between {lo} and {hi}")
    if not 0 < args.watermark_gamma < 1:
        parser.error("--watermark-gamma must be between 0 and 1")
    if args.time_budget_hours is not None and args.steps is not None:
        parser.error("give either --steps or --time-budget-hours (the budget chooses the number of steps)")
    if args.time_budget_hours is not None and not 0.01 <= args.time_budget_hours <= 24 * 90:
        parser.error("--time-budget-hours must be between 0.01 and 2160")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.log_every < 0 or (args.sample_every is not None and args.sample_every < 0):
        parser.error("--log-every/--sample-every must be >= 0")
    if args.resume and args.init_from:
        parser.error("use either --resume or --init-from, not both")
    if args.train_tokens is not None and args.train_tokens <= 0 or args.epochs is not None and args.epochs <= 0:
        parser.error("--train-tokens/--epochs must be positive")
    if sum(x is not None for x in (args.steps, args.train_tokens, args.epochs)) > 1:
        parser.error("give only one of --steps, --train-tokens, --epochs")
    args.out_dir = os.path.abspath(args.out_dir)
    if args.loss_chunk_tokens is not None and not (args.loss_chunk_tokens == 0 or 64 <= args.loss_chunk_tokens):
        parser.error("--loss-chunk-tokens must be 0 (off) or at least 64")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        parser.error("--stop-after-steps must be positive")
    if not os.path.isdir(args.out_dir):
        parser.error(f"--out-dir does not exist: {args.out_dir}")
    if args.dataset and not os.path.isdir(args.dataset):
        parser.error(f"--dataset folder not found: {args.dataset}")
    if args.data and not os.path.isdir(args.data):
        parser.error(f"--data folder not found: {args.data}")
    if args.sample_every is None:
        args.sample_every = args.eval_every
    if args.no_probes:
        args.probes = None
    elif args.probes and not os.path.isfile(args.probes):
        if args.probes != DEFAULT_PROBES:
            parser.error(f"--probes file not found: {args.probes}")
        args.probes = None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def checkpoint_paths(args):
    base = os.path.join(args.out_dir, args.name)
    return base + ".pt", base + "_best.pt", base + "_metrics.jsonl"


_ST_DTYPES = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64",
              torch.int32: "I32", torch.int16: "I16", torch.uint8: "U8", torch.bool: "BOOL"}


def write_safetensors(tensors, path, metadata=None):
    """Minimal writer for the safetensors format (8-byte header length, JSON
    header, raw little-endian data). No pickle, so loading can never run code."""
    import struct
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        t = tensors[name].detach().cpu().contiguous()
        if t.dtype not in _ST_DTYPES:
            raise ValueError(f"unsupported dtype {t.dtype} for {name}")
        raw = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()
        header[name] = {"dtype": _ST_DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    header["__metadata__"] = {str(k): str(v) for k, v in (metadata or {}).items()}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for raw in blobs:
            handle.write(raw)
    os.replace(tmp, path)


def read_safetensors(path):
    import struct
    by_name = {v: k for k, v in _ST_DTYPES.items()}
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 100 * 2 ** 20:
            raise SystemExit(f"{path}: implausible safetensors header ({size:,} bytes)")
        header = json.loads(handle.read(size))
        data = handle.read()
    metadata = header.pop("__metadata__", {})
    tensors = {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        if not 0 <= begin <= end <= len(data) or info["dtype"] not in by_name:
            raise SystemExit(f"{path}: corrupt entry {name}")
        dtype = by_name[info["dtype"]]
        buf = torch.frombuffer(bytearray(data[begin:end]), dtype=torch.int16 if dtype == torch.bfloat16 else dtype)
        tensors[name] = (buf.view(torch.bfloat16) if dtype == torch.bfloat16 else buf).reshape(info["shape"])
    return tensors, metadata


def load_exported(folder):
    """An --export folder as a checkpoint-like dict (inference only)."""
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    tensors, _ = read_safetensors(os.path.join(folder, "model.safetensors"))
    if config["model_config"].get("tie_embeddings", True) and "lm_head.weight" not in tensors:
        tensors["lm_head.weight"] = tensors["token_embedding.weight"]
    with open(os.path.join(folder, config["tokenizer"]["file"]), "rb") as handle:
        proto = handle.read()
    return {"format": CHECKPOINT_FORMAT, "format_version": FORMAT_VERSION, "checkpoint_type": "inference",
            "model_config": config["model_config"], "model": tensors,
            "tokenizer": {**config["tokenizer"], "proto": proto}, "step": config.get("step"),
            "metrics": config.get("metrics", {}), "exported_from": config.get("source_checkpoint")}


def load_any(path, trust=False):
    """A .pt checkpoint (safe loader) or an --export folder."""
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "model.safetensors")):
        return load_exported(path)
    return safe_load(path, trust)


def cmd_export(args):
    """Write <folder>/model.safetensors + config.json + tokenizer.model."""
    ckpt_path, best_path, _ = checkpoint_paths(args)
    path = args.checkpoint or (best_path if os.path.exists(best_path) else ckpt_path)
    if not os.path.exists(path):
        raise SystemExit(f"No checkpoint {path}; pass --checkpoint.{_resume_hint(args)}")
    out = os.path.abspath(args.export)
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty; choose a new folder.")
    obj = safe_load(path, args.trust_checkpoint)
    model, tok, cfg = model_from_checkpoint(obj)
    tensors = {k: v for k, v in model.state_dict().items() if not (cfg.tie_embeddings and k == "lm_head.weight")}
    os.makedirs(out, exist_ok=True)
    write_safetensors(tensors, os.path.join(out, "model.safetensors"),
                      {"format": "tinyGPT", "tie_embeddings": cfg.tie_embeddings})
    with open(os.path.join(out, "tokenizer.model"), "wb") as handle:
        handle.write(tok.proto)
    metrics = obj.get("metrics", {}) if isinstance(obj.get("metrics"), dict) else {}
    config = {"architecture": "tinyGPT decoder-only Transformer: interleaved-pair RoPE, RMSNorm, SwiGLU, "
                              "grouped-query attention, tied input/output embeddings, no biases",
              "model_config": dataclasses.asdict(cfg), "tokenizer": {**tok.meta(), "file": "tokenizer.model"},
              "source_checkpoint": os.path.abspath(path), "step": obj.get("step"),
              "metrics": {"best_val": metrics.get("best_val", obj.get("best_val")),
                          "best_step": metrics.get("best_step", obj.get("best_val_step"))},
              "exported": now_iso()}
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    back, _ = read_safetensors(os.path.join(out, "model.safetensors"))
    same = all(torch.equal(back[k], v.cpu()) for k, v in tensors.items())
    if not same:
        raise SystemExit("Export verification failed: tensors differ after reading back.")
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"Exported {path} -> {out} (model.safetensors, config.json, tokenizer.model; {size / 1e6:.1f} MB); "
          f"verified {len(tensors)} tensors by reading them back.\n"
          f"Use it with: python tiny_gpt.py --checkpoint {out} --generate \"Genetic drift is\"")


def cmd_benchmark(args):
    """Score the cloze fact benchmark on a checkpoint; print a per-category table
    and write every item's result to a CSV next to the checkpoint."""
    ckpt_path, best_path, _ = checkpoint_paths(args)
    path = args.checkpoint or (best_path if os.path.exists(best_path) else ckpt_path)
    if not os.path.exists(path):
        raise SystemExit(f"No checkpoint {path}; pass --checkpoint.{_resume_hint(args)}")
    probes_file = args.probes or DEFAULT_PROBES
    if not os.path.isfile(probes_file):
        raise SystemExit(f"Probe file not found: {probes_file}")
    device = select_device(args.device)
    amp_dtype, amp_name, _ = choose_precision(device, args.precision)
    model, tok, cfg, obj = load_for_inference(path, device, args.trust_checkpoint)
    model.set_attention_impl(choose_attention(device, cfg, amp_dtype, args.attn)[0])
    probes = read_probes(probes_file)
    acc, results = score_probes(model, tok, probes, device, amp_dtype)
    k, n = sum(r["correct"] for r in results), len(results)
    low, high = wilson_interval(k, n)
    chance = sum(r["chance"] for r in results) / n
    print(f"Benchmark {os.path.basename(probes_file)} on {path} (step {obj.get('step', '?')}, {device_label(device)})")
    floor = probe_floor(results)
    print(f"overall: {k}/{n} = {acc:.1%} (95% interval {low:.1%}-{high:.1%}); floor {floor:.1%} "
          f"(question words shuffled); chance {chance:.1%}")
    for key in ("category", "difficulty"):
        print(f"\n{key:<34} {'n':>4} {'correct':>8} {'accuracy':>9} {'95% interval':>15} {'floor':>6}")
        for name, s in sorted(probe_summary(results, key).items(), key=lambda kv: -kv[1]["accuracy"]):
            print(f"{name:<34} {s['n']:>4} {s['correct']:>8} {s['accuracy']:>9.1%} "
                  f"{s['low']:>7.0%}-{s['high']:<6.0%} {s['floor']:>6.0%}")
    print("\nFloor = accuracy when each question's words are shuffled (topic words kept, sentence destroyed); only "
          "accuracy above it reflects knowledge. With about 20 items per category, category differences smaller than "
          "about 30 percentage points are noise; track the overall accuracy across checkpoints instead.")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out = f"{os.path.splitext(path)[0]}_benchmark_step{obj.get('step', 0)}_{stamp}.csv"
    import csv
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "difficulty", "prompt", "answer", "correct", "correct_score",
                         "best_distractor_score", "floor_correct"])
        for r in results:
            writer.writerow([r["category"], r["difficulty"], r["prompt"], r["answer"], int(r["correct"]),
                             f"{r['correct_score']:.4f}", f"{r['best_distractor_score']:.4f}",
                             f"{r['floor_correct']:.3f}"])
    print(f"Per-item results: {out}")
    record_experiment(os.path.dirname(os.path.abspath(path)), {
        "event": "benchmark", "name": os.path.basename(path), "step": obj.get("step"), "params": cfg and
        count_params(cfg)[0], "d_model": cfg.d_model, "layers": cfg.n_layers, "ctx": cfg.ctx, "vocab": cfg.vocab_size,
        "benchmark_accuracy": f"{acc:.3f}", "benchmark_floor": f"{floor:.3f}", "benchmark_items": n,
        "checkpoint": path, "notes": f"95% interval {low:.3f}-{high:.3f}; chance {chance:.3f}; "
                                    f"{os.path.basename(probes_file)}; items in {os.path.basename(out)}"})


def cmd_generate(args):
    ckpt_path, best_path, _ = checkpoint_paths(args)
    path = args.checkpoint or (best_path if os.path.exists(best_path) else ckpt_path)
    if not os.path.exists(path):
        hint = ""
        if not args.checkpoint and os.path.exists(os.path.join(args.out_dir, "tiny_gpt_bpe_best.pt")):
            hint = " Your legacy model is still available: --checkpoint tiny_gpt_bpe_best.pt"
        raise SystemExit(f"No checkpoint {path}; train first or pass --checkpoint.{hint}")
    device = select_device(args.device)
    amp_dtype, amp_name, _ = choose_precision(device, args.precision)
    model, tok, cfg, obj = load_for_inference(path, device, args.trust_checkpoint)
    impl, _ = choose_attention(device, cfg, amp_dtype, args.attn)
    model.set_attention_impl(impl)
    family, kind = checkpoint_info(obj)
    metrics = obj.get("metrics", {}) if family == "tinygpt" else {}
    best = metrics.get("best_val", obj.get("best_val"))
    best_step = metrics.get("best_step", obj.get("best_val_step"))
    print(f"Checkpoint {path} ({family}, {kind}) | step {obj.get('step', '?'):,} | best validation loss "
          f"{best:.4f} @{best_step} | {device_label(device)} | {amp_name}" if best is not None else
          f"Checkpoint {path} ({family}, {kind}) | step {obj.get('step', '?')}")
    if family != "tinygpt":
        print("Note: legacy checkpoints use the old tokenizer (runs of spaces collapse; no end-of-document token).")
    gen = torch.Generator().manual_seed(args.seed)
    watermark = (args.watermark_key, args.watermark_gamma, args.watermark_delta) if args.watermark_key else None
    text = generate(model, tok, args.generate, args.length, device, amp_dtype, args.temperature, args.top_k,
                    args.top_p, gen, watermark)
    print(f"\n{args.generate}{text}" if args.generate and not text.startswith(args.generate) else f"\n{text}")


def compat_dataset_dir(args):
    """datasets/<md folder>_wiki<N>[_kw]_<hash of all settings>. The name encodes
    the settings, so a different --wiki-articles, keyword file or vocabulary
    builds a new dataset instead of silently reusing an old one (the bug that
    trained a "--wiki-articles 50000" run on 30,000 articles)."""
    canon = lambda p: os.path.normcase(os.path.realpath(p))  # noqa: E731 - C:\X and c:\x are one folder
    options = {"md_dir": canon(args.data) if args.data else None,
               "wiki_dir": canon(args.wiki_dir) if args.wiki_articles else None,
               "wiki_max_docs": args.wiki_articles or 0, "include_keywords": args.include_keywords,
               "vocab_size": args.tokenizer_vocab_size or 0}
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:6]
    parts = [os.path.basename(os.path.normpath(args.data)) if args.data else "nomd"]
    if args.wiki_articles:
        parts.append(f"wiki{args.wiki_articles}")
    if args.include_keywords:
        parts.append("kw")
    return os.path.join(HERE, "datasets", "_".join(parts + [digest]))


def ensure_dataset(args):
    """--dataset, or the dataset path stored in a checkpoint, or (compat) build one from --data."""
    if args.dataset:
        return os.path.abspath(args.dataset)
    if args.data or args.wiki_articles:
        path = compat_dataset_dir(args)
        if os.path.exists(os.path.join(path, "manifest.json")):
            manifest = data_prep.read_manifest(path)
            kept = {n: s.get("kept_docs") for n, s in manifest["sources"].items()}
            print(f"Reusing dataset {path}, built earlier with the same --data/--wiki-articles/"
                  f"--include-keywords/--tokenizer-vocab-size (documents kept: {kept}).")
            return path
        prep = ["--out", path]
        if args.data:
            has_subfolders = any(os.path.isdir(os.path.join(args.data, d)) and not d.startswith((".", "_"))
                                 and d.lower() not in ("wikipedia", "web", "images") for d in os.listdir(args.data))
            prep += ["--text-root", args.data] if has_subfolders else ["--md-dir", args.data]
        if args.wiki_articles:
            prep += ["--wiki-dir", args.wiki_dir, "--wiki-max-docs", str(args.wiki_articles)]
            if not args.include_keywords:
                prep += ["--wiki-select", "first"]
        if args.include_keywords:
            prep += ["--include-keywords", args.include_keywords]
        if args.tokenizer_vocab_size:
            prep += ["--vocab-size", str(args.tokenizer_vocab_size)]
        print("Building dataset: python data_prep.py " + " ".join(f'"{x}"' if " " in x else x for x in prep))
        # its own process: data_prep's worker processes would otherwise re-import this script (and torch)
        code = subprocess.run([sys.executable, os.path.join(HERE, "data_prep.py"), *prep]).returncode
        if code:
            raise SystemExit(f"data_prep.py failed (exit code {code}); see its messages above.")
        return path
    return None


def _resume_hint(args):
    """Name other training checkpoints in the folder (e.g. runs made before the rename to tinyGPT)."""
    found = []
    for name in sorted(os.listdir(args.out_dir)):
        if name.endswith(".pt") and not name.endswith("_best.pt") and "_crash" not in name:
            found.append(name[:-3])
    return (" Training checkpoints in this folder: " + ", ".join(found) +
            ". Pass the right one with --name, e.g. --name " + found[0] + " --resume") if found else ""


def cmd_train(args):
    ckpt_path, best_path, metrics_path = checkpoint_paths(args)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    t_run = time.perf_counter()
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_dtype, amp_name, need_scaler = choose_precision(device, args.precision)

    # ---------------- resume / init / fresh ----------------
    resume_obj = init_obj = None
    from_best = False  # resuming from <name>_best.pt because no training state was ever saved
    if args.resume:
        if os.path.exists(ckpt_path):
            resume_path = ckpt_path
        elif os.path.exists(best_path):
            resume_path, from_best = best_path, True
        else:
            raise SystemExit(f"Nothing to resume: {ckpt_path} does not exist.{_resume_hint(args)}")
        resume_obj = safe_load(resume_path, args.trust_checkpoint)
        family, kind = checkpoint_info(resume_obj)
        wanted = "inference" if from_best else "training"
        if family != "tinygpt" or kind != wanted or "train_config" not in resume_obj:
            raise SystemExit(f"{resume_path} is a {family} {kind} checkpoint; --resume needs a tinyGPT checkpoint "
                             "written by a training run. Start a new run from its weights with --init-from instead.")
        dataset_dir = args.dataset or resume_obj["dataset"]["path"]
    else:
        existing = [p for p in (ckpt_path, best_path) if os.path.exists(p)]
        if existing and not args.overwrite:
            raise SystemExit(f"{existing[0]} already exists. Use --resume to continue it, a new --name, "
                             "or --overwrite (existing files are renamed, not deleted).")
        dataset_dir = ensure_dataset(args)
        if args.init_from:
            init_obj = safe_load(args.init_from, args.trust_checkpoint)
    if not dataset_dir:
        raise SystemExit("No training data: pass --dataset <folder from data_prep.py> (or --data <md folder>).")
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"Dataset folder not found: {dataset_dir} (moved? pass --dataset).")
    manifest, arrays = data_prep.open_token_files(dataset_dir, verify=True)
    tok_meta = manifest["tokenizer"]
    tok = data_prep.Tokenizer.from_file(os.path.join(dataset_dir, tok_meta["file"]), tok_meta.get("encode_mode", "lines"))
    if resume_obj is not None:
        if resume_obj["dataset"]["fingerprint"] != manifest["fingerprint"]:
            raise SystemExit(f"The dataset at {dataset_dir} differs from the one this run was trained on "
                             f"(fingerprint {manifest['fingerprint'][:12]} vs "
                             f"{resume_obj['dataset']['fingerprint'][:12]}). Resume with the original dataset, "
                             "or start a new run with --init-from.")
    if init_obj is not None:
        init_tok = tokenizer_from_checkpoint(init_obj)
        if init_tok.sha256 != tok.sha256 or init_tok.encode_mode != tok.encode_mode:
            raise SystemExit("--init-from checkpoint uses a different tokenizer than the dataset. Rebuild the "
                             f"dataset with: python data_prep.py ... --tokenizer-from {args.init_from}")

    # ---------------- configuration ----------------
    plan_lines = []
    if resume_obj is not None:
        cfg = ModelConfig(**resume_obj["model_config"])
        settings = dict(resume_obj["train_config"])
        settings.setdefault("optimizer", "adamw")  # runs from before --optimizer used AdamW
        # memory options do not change the mathematics, so a resumed run may switch them (e.g. on another machine)
        settings.setdefault("activation_checkpointing", False)
        settings.setdefault("loss_chunk_tokens", 0)
        if args.activation_checkpointing != "auto":
            settings["activation_checkpointing"] = args.activation_checkpointing == "on"
        if args.loss_chunk_tokens is not None:
            settings["loss_chunk_tokens"] = args.loss_chunk_tokens
        if args.optimizer and args.optimizer != settings["optimizer"]:
            print(f"Note: --optimizer {args.optimizer} ignored on --resume: this run uses {settings['optimizer']} "
                  "(start a new run, or --init-from this one, to change it).")
        ignored = [f"--{k.replace('_', '-')}" for k in ("d_model", "layers", "heads", "kv_heads", "ctx", "lr",
                                                        "batch_size", "grad_accum", "dropout", "warmup_steps",
                                                        "epochs", "train_tokens", "time_budget_hours",
                                                        "tokens_per_step")
                   if getattr(args, k) is not None]
        if ignored:
            print(f"Note: {', '.join(ignored)} ignored on --resume (the checkpoint's settings are kept).")
        if args.steps and args.steps != settings["steps"]:
            old = LRSchedule(settings["schedule"], settings["steps"], settings["lr"], settings["min_lr"],
                             settings["warmup_steps"], settings["decay_frac"], settings["decay_shape"])
            new = LRSchedule(settings["schedule"], args.steps, settings["lr"], settings["min_lr"],
                             settings["warmup_steps"], settings["decay_frac"], settings["decay_shape"])
            if old.phase(int(resume_obj["step"])) == "decay":
                print(f"WARNING: the run is already in its LR decay phase; changing --steps from "
                      f"{settings['steps']:,} to {args.steps:,} moves the decay and re-raises the LR.")
            elif new.phase(int(resume_obj["step"])) == "decay":
                print(f"WARNING: with --steps {args.steps:,} step {int(resume_obj['step']):,} is already inside the "
                      f"decay phase: the learning rate drops at once to {new.lr_at(int(resume_obj['step'])):.2e}.")
            settings["steps"] = args.steps
            settings["planned_train_tokens"] = args.steps * settings["tokens_per_step"]
    elif init_obj is not None:
        cfg = config_from_checkpoint(init_obj)  # --time-budget-hours then sets how much of the new data is read
        _, settings, plan_lines = plan_with_time_budget(_args_with_shape(args, cfg), manifest, device, amp_dtype)
        cfg.dropout = args.dropout if args.dropout is not None else 0.0
        plan_lines.append(f"architecture taken from --init-from {args.init_from}; dropout {cfg.dropout} "
                          f"({'set' if args.dropout is not None else 'default 0 when continuing from weights'}, "
                          f"not the planned value above)")
    else:
        cfg, settings, plan_lines = plan_with_time_budget(args, manifest, device, amp_dtype)
    if cfg.vocab_size != tok.vocab_size:
        raise SystemExit(f"Model vocabulary {cfg.vocab_size} != dataset tokenizer {tok.vocab_size}.")
    total_steps = int(settings["steps"])
    micro, accum = int(settings["micro_batch"]), int(settings["grad_accum"])
    mix = parse_mix(args.mix if resume_obj is None else (args.mix or settings.get("mix")), arrays["train"])
    byte_tables = tuple(torch.from_numpy(t) for t in tok.byte_tables())
    z_loss = float(settings.setdefault("z_loss", args.z_loss if resume_obj is None else 0.0))
    eval_tokens = int(settings.setdefault("eval_tokens", args.eval_tokens))  # fixed windows stay fixed on resume
    if resume_obj is not None and eval_tokens != args.eval_tokens and args.eval_tokens != 65_536:
        print(f"Note: --eval-tokens ignored on --resume; this run keeps {eval_tokens:,} (its validation windows).")
    probes_sha = file_sha256(args.probes)[:16] if args.probes else None
    if resume_obj is not None and settings.get("probes_sha") and probes_sha and settings["probes_sha"] != probes_sha:
        print(f"Note: {os.path.basename(args.probes)} has changed since this run started: fact-probe accuracies "
              "from now on are not comparable with the earlier ones.")
    settings["probes_sha"] = probes_sha
    settings["mix"] = args.mix if resume_obj is None else (args.mix or settings.get("mix"))

    # ---------------- model & optimizer ----------------
    torch.manual_seed(args.seed)  # planning and attention timing must not change the initial weights
    model = TinyGPT(cfg).to(device)
    attn_impl, attn_why = choose_attention(device, cfg, amp_dtype, args.attn)
    model.set_attention_impl(attn_impl)
    source_obj = resume_obj or init_obj
    if source_obj is not None:
        missing, unexpected = model.load_state_dict({k: v for k, v in source_obj["model"].items()}, strict=False)
        missing = [m for m in missing if not (m == "lm_head.weight" and cfg.tie_embeddings)]
        if missing or unexpected:
            raise SystemExit(f"Checkpoint weights do not fit the model (missing {missing[:3]}, "
                             f"unexpected {unexpected[:3]}).")
    opt, opt_name, param_names, opt_note = build_optimizer(model, settings, device)
    model.grad_checkpoint = bool(settings.get("activation_checkpointing"))
    loss_chunk = int(settings.get("loss_chunk_tokens") or 0)
    scaler = torch.amp.GradScaler(device.type) if need_scaler else None
    schedule = LRSchedule(settings["schedule"], total_steps, settings["lr"], settings["min_lr"],
                          settings["warmup_steps"], settings["decay_frac"], settings["decay_shape"])

    batcher = TokenBatcher(arrays["train"], mix, cfg.ctx, micro, device, args.seed)
    evalset = EvalSet(arrays["val"], cfg.ctx, eval_tokens)
    eval_batch = max(1, min(micro, int(3e8 // (cfg.ctx * cfg.vocab_size * 4))))
    probes = read_probes(args.probes) if args.probes else []
    probes_sha = file_sha256(args.probes) if probes else None

    start_step, tokens_seen, train_seconds = 0, 0, 0.0
    run_seconds_before = 0.0  # wall-clock time of earlier sessions of this run (before --resume)
    rewarm_steps = 0
    interval_state = None
    history, best_val, best_step, last_eval = [], float("inf"), None, None
    resume_note = "fresh run"
    if resume_obj is not None:
        start_step = int(resume_obj["step"])
        tokens_seen = int(resume_obj.get("tokens_seen", 0))
        train_seconds = float(resume_obj.get("train_seconds", 0.0))
        # checkpoints from before this field existed: training time is the closest record
        run_seconds_before = float(resume_obj.get("run_seconds", train_seconds))
        m = resume_obj.get("metrics", {})
        history = list(m.get("history", []))
        best_val = float(m.get("best_val", float("inf")))
        best_step = m.get("best_step")
        last_eval = m.get("last_eval")
        if from_best:
            # The best checkpoint has the weights, step, schedule and history, but not the optimizer's running averages
            # or the sampler state: the averages restart (they rebuild within ~50 steps, so the learning rate is
            # re-warmed over those steps) and the sampler is re-seeded so the first windows are not replayed.
            rewarm_steps = min(50, max(total_steps - start_step, 1))
            batcher.gen.manual_seed(int(args.seed) + start_step)
            resume_note = (f"RESUMED FROM {best_path} at step {start_step:,}: no training state had been saved "
                           f"({ckpt_path} missing), so the optimizer's averages restart and the learning rate is re-warmed "
                           f"over {rewarm_steps} steps; weights, schedule position and history are kept")
        else:
            restored, total_p = load_optimizer_by_name(opt, resume_obj["optimizer"],
                                                       resume_obj["optimizer_param_names"], param_names)
            if scaler is not None and resume_obj.get("scaler"):
                scaler.load_state_dict(resume_obj["scaler"])
            rng = resume_obj.get("rng", {})
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"].cpu())
            if device.type == "cuda" and rng.get("cuda"):
                try:
                    torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
                except (RuntimeError, TypeError) as exc:
                    print(f"CUDA RNG state not restored ({exc}); only dropout masks differ.")
            batcher.load_state_dict(resume_obj.get("batcher"))
            interval_state = resume_obj.get("interval")
            if os.path.exists(best_path):
                try:
                    newer = safe_load(best_path, args.trust_checkpoint)
                    nm = newer.get("metrics", {})
                    if (newer.get("dataset", {}).get("fingerprint") == manifest["fingerprint"]
                            and float(nm.get("best_val", float("inf"))) < best_val):
                        print(f"Note: {best_path} holds a better model (val {float(nm['best_val']):.4f} "
                              f"@{nm.get('best_step')}) than this training state records; it is only replaced "
                              "by something better.")
                        best_val, best_step = float(nm["best_val"]), nm.get("best_step")
                    del newer
                except Exception as exc:  # noqa: BLE001
                    print(f"Note: could not read {best_path} ({exc}); it may be replaced by the next best.")
            resume_note = (f"RESUMED {ckpt_path} at step {start_step:,} (optimizer state {restored}/{total_p} "
                           f"tensors; best {best_val:.4f} @{best_step}; last eval "
                           f"{last_eval['loss']:.4f} @{last_eval['step']})" if last_eval else
                           f"RESUMED {ckpt_path} at step {start_step:,}")
    elif init_obj is not None:
        family, kind = checkpoint_info(init_obj)
        note = f"initialised from {args.init_from} ({family} {kind}, step {init_obj.get('step', '?')})"
        if init_obj.get("optimizer") and kind == "training":
            flat = init_obj.get("optimizer_param_names") or legacy_optimizer_names(model)
            restored, total_p = load_optimizer_by_name(opt, init_obj["optimizer"], flat, param_names)
            start_step = int(init_obj.get("step", 0))
            note += (f"; optimizer state restored for {restored}/{total_p} tensors; schedule continues at step "
                     f"{start_step:,}")
        resume_note = note + "; best/last validation reset because the validation set is new"
        if start_step >= total_steps:
            raise SystemExit(f"--init-from checkpoint is at step {start_step:,}; set --steps above that.")

    train_model = maybe_compile(model, args.compile, device)
    n_total, n_nonembed = model.param_counts()
    unique = sum(len(a) for a in arrays["train"].values())

    # ---------------- run header ----------------
    print(f"\n{'=' * 100}\nRUN {run_id} started {now_iso()} | {resume_note}")
    print(f"command: {' '.join(sys.argv)}")
    print(f"python {platform.python_version()} | torch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"{platform.platform()} | tiny_gpt.py sha256 {file_sha256(__file__)[:12]}")
    print(f"device: {device_label(device)} | precision: {amp_name} | attention: SDPA {attn_impl} ({attn_why}) | "
          f"optimizer: {opt_name} ({opt_note}) | "
          f"compile: {'on' if train_model is not model else 'off (eager)'}")
    print(f"dataset: {dataset_dir} | fingerprint {manifest['fingerprint'][:12]} | tokenizer "
          f"{tok.vocab_size:,} pieces ({tok.encode_mode}) | train "
          + ", ".join(f"{n} {len(a):,}" for n, a in arrays["train"].items())
          + " | val " + ", ".join(f"{n} {len(a):,}" for n, a in arrays["val"].items()))
    print("sampling share per source: " + ", ".join(f"{n} {p:.1%}" for n, p in batcher.share().items())
          + f" | validation windows (fixed): {evalset.describe()}"
          + (f" | skipped (too small): {evalset.skipped}" if evalset.skipped else ""))
    print("model: " + json.dumps(dataclasses.asdict(cfg)))
    print(f"parameters: {n_total:,} total, {n_nonembed:,} non-embedding")
    print("train: " + json.dumps({k: v for k, v in settings.items()}))
    print(f"schedule: {schedule.describe()}")
    for line in plan_lines:
        print(f"plan: {line}")
    train_tokens_total = total_steps * settings["tokens_per_step"]
    print(f"budget: {total_steps:,} steps x {settings['tokens_per_step']:,} tokens = {train_tokens_total:,} tokens "
          f"= {train_tokens_total / n_total:,.1f} tokens/parameter, {train_tokens_total / max(unique, 1):,.2f} epochs")
    print(f"every {args.eval_every} steps: metrics line (train = mean training loss since the previous line; "
          f"val = mean of per-source validation losses on fixed windows, val X @S was measured at step S; "
          f"ppl = exp(val)), then a sample"
          + (f"; fact probes: {len(probes)} from {os.path.basename(args.probes)}" if probes else "")
          + "; lines starting with '...' in between are progress only (train loss since the previous line, "
          "no validation)")
    meter = EnergyMeter(device, args.cpu_watts, args.grid_intensity,
                        resume_obj.get("energy") if resume_obj is not None else None,
                        tariff=Tariff(args.tariff, args.region))
    print(meter.describe() + (f" Earlier sessions of this run: {meter.short()}." if meter.kwh() else ""))
    save_when = f"every {args.save_every} steps" + (f" and every {args.save_minutes:g} min" if args.save_minutes > 0
                                                     else "")
    print(f"checkpoints: {ckpt_path} {save_when} (training state), {best_path} on NEW BEST "
          f"(inference only); metrics: {metrics_path}\n{'=' * 100}")
    if start_step >= total_steps:
        print(f"Checkpoint already reached step {start_step:,} of {total_steps:,}; nothing to train. "
              "Extend with --resume --steps N.")
        meter.stop()
        return

    sample_gen_seed = args.seed + 7

    def payload(step, inference_only):
        base = {
            "format": CHECKPOINT_FORMAT, "format_version": FORMAT_VERSION,
            "checkpoint_type": "inference" if inference_only else "training",
            "created": now_iso(), "run_id": run_id, "torch_version": str(torch.__version__),
            "model_config": dataclasses.asdict(cfg), "model": cpu_state_dict(model),
            "tokenizer": {"proto": tok.proto, **tok.meta()},
            "dataset": {"path": dataset_dir, "fingerprint": manifest["fingerprint"],
                        "train_tokens": {n: len(a) for n, a in arrays["train"].items()},
                        "val_tokens": {n: len(a) for n, a in arrays["val"].items()}},
            "train_config": dict(settings), "step": step, "tokens_seen": tokens_seen,
            "train_seconds": train_seconds, "run_seconds": run_seconds(), "energy": meter.state(),
            "metrics": {"best_val": best_val, "best_step": best_step, "last_eval": last_eval,
                        "history": history},
        }
        if not inference_only:
            base.update({
                "optimizer": to_cpu(opt.state_dict()), "optimizer_param_names": [n for g in param_names for n in g],
                "scaler": scaler.state_dict() if scaler else None,
                "rng": {"torch": torch.get_rng_state(),
                        "cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()] if device.type == "cuda" else None},
                "batcher": batcher.state_dict(),
                "interval": {"loss_sum": float(interval_loss), "micro": interval_micro},
            })
        return base

    def write_metrics(entry):
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"run_id": run_id, **entry}) + "\n")

    if args.overwrite and resume_obj is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for path in (ckpt_path, best_path, metrics_path):
            if os.path.exists(path):
                os.replace(path, f"{path}.bak-{stamp}")
                print(f"renamed existing {path} -> {path}.bak-{stamp}")

    # ---------------- baseline evaluation ----------------
    if resume_obj is None and not history:
        print(f"measuring the untrained model on the fixed validation windows ({evalset.describe()}) ...")
        ev = evaluate(model, evalset, device, amp_dtype, eval_batch, byte_tables=byte_tables)
        history.append({"step": start_step, "tokens": tokens_seen, "train_loss": None, "val_loss": ev["loss"],
                        "bpb": ev["bpb"], "bpb_per_source": ev["bpb_per_source"],
                        "val_per_source": ev["per_source"], "ppl": math.exp(min(ev["loss"], 50)), "lr": 0.0,
                        "tok_s": None, "elapsed_s": 0.0, "train_seconds": train_seconds,
                        "entropy": ev["entropy"], "confidence": ev["confidence"], "top1": ev["top1"],
                        "ece": ev["ece"], "probe_acc": None, "grad_norm": None, "is_best": False,
                        "baseline": True, "time": now_iso()})
        write_metrics(history[-1])
        print(f"step {start_step:>6}/{total_steps} | baseline before training | val {ev['loss']:.3f} @{start_step}"
              + _per_source(ev) + f" | ppl {math.exp(min(ev['loss'], 50)):,.1f} | bpb {ev['bpb']:.3f} "
              f"| uniform-guess loss would be {math.log(cfg.vocab_size):.3f} (not counted as best)")

    def run_seconds():
        """Wall-clock time of the whole run so far: earlier sessions plus this one."""
        return run_seconds_before + (time.perf_counter() - t_run)

    def elapsed_text(now=None):
        session = (now or time.perf_counter()) - t_run
        if not run_seconds_before:
            return f"elapsed {fmt_hms(session)}"
        return f"elapsed {fmt_hms(session)} (whole run {fmt_hms(run_seconds_before + session)})"

    dashboard = {"proc": None, "pushed": 0.0}
    dashboard_on = args.dashboard == "on" or (
        args.dashboard == "auto" and os.path.normcase(os.path.abspath(args.out_dir)) == os.path.normcase(HERE))

    # ---------------- training loop ----------------
    model.train()
    status, step = "completed", start_step
    interval_loss = torch.zeros((), device=device)
    interval_micro, interval_tokens, interval_train_time = 0, 0, 0.0
    progress_loss = torch.zeros((), device=device)
    progress_micro = 0
    last_gnorm = torch.zeros((), device=device)
    seg = time.perf_counter()
    interval_wall, interval_start_step = time.perf_counter(), start_step
    last_print = time.perf_counter()
    first_step = True
    done_step = start_step  # last step whose optimizer update finished
    last_save = time.perf_counter()
    if interval_state:  # resumed between evaluations: the next "train" value covers the whole interval
        interval_loss += float(interval_state.get("loss_sum", 0.0))
        interval_micro = int(interval_state.get("micro", 0))
    try:
        step = start_step
        while step < total_steps:
            step += 1
            sampler_snapshot = (batcher.gen.get_state(),
                                torch.get_rng_state() if cfg.dropout > 0 else None,
                                torch.cuda.get_rng_state_all() if cfg.dropout > 0 and device.type == "cuda" else None)
            lr = schedule.lr_at(step)
            if step - start_step <= rewarm_steps:  # only after resuming from a best checkpoint
                lr *= (step - start_step) / rewarm_steps
            for group in opt.param_groups:
                group["lr"] = lr
            loss_before = (interval_loss.clone(), progress_loss.clone())  # restored if this step is retried
            try:
                for _ in range(accum):
                    x, y = batcher.next()
                    with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                        if loss_chunk:
                            ce, zsum, n_tok = chunked_lm_loss(train_model(x, return_hidden=True),
                                                              model.lm_head.weight, y, loss_chunk)
                            flat = None
                        else:
                            logits = train_model(x)
                    if loss_chunk:
                        loss = ce / n_tok
                    else:
                        flat = logits.float().view(-1, cfg.vocab_size)
                        loss = F.cross_entropy(flat, y.reshape(-1))
                    interval_loss += loss.detach()
                    progress_loss += loss.detach()
                    if z_loss:
                        # PaLM z-loss: keeps log Z (the softmax normaliser) near 0; reported loss excludes it
                        loss = loss + z_loss * (zsum / n_tok if loss_chunk
                                                else torch.logsumexp(flat, dim=-1).pow(2).mean())
                    if scaler is not None:
                        scaler.scale(loss / accum).backward()
                    else:
                        (loss / accum).backward()
                if scaler is not None:
                    scaler.unscale_(opt)
                last_gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"] or float("inf"))
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            except RuntimeError as exc:  # torch.cuda.OutOfMemoryError / MPS OOM are RuntimeErrors
                saving_left = not loss_chunk or not model.grad_checkpoint
                if not first_step or (micro == 1 and not saving_left) or "out of memory" not in str(exc).lower():
                    raise
                opt.zero_grad(set_to_none=True)
                x = y = logits = flat = loss = ce = zsum = None  # drop the failed batch and its graph before retrying
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                interval_loss.copy_(loss_before[0])
                progress_loss.copy_(loss_before[1])
                # cheapest remedy first: the loss in pieces, then a smaller micro-batch, then recomputation
                if not loss_chunk:
                    loss_chunk = settings["loss_chunk_tokens"] = LOSS_CHUNK_TOKENS
                    print(f"Out of memory at the first step; retrying with the loss computed {loss_chunk:,} tokens "
                          "at a time.")
                elif micro > 1:
                    old_tps = settings["tokens_per_step"]
                    micro, accum = micro // 2, math.ceil(micro * accum / (micro // 2))
                    settings["micro_batch"], settings["grad_accum"] = micro, accum
                    settings["tokens_per_step"] = micro * accum * cfg.ctx
                    batcher.batch_size = micro
                    eval_batch = max(1, min(eval_batch, micro))
                    print(f"Out of memory at the first step; retrying with micro-batch {micro} x {accum} "
                          "accumulation " + ("(same tokens per step)." if settings["tokens_per_step"] == old_tps
                                             else f"({settings['tokens_per_step']:,} tokens per step instead of "
                                                  f"{old_tps:,})."))
                else:
                    model.grad_checkpoint = settings["activation_checkpointing"] = True
                    print("Out of memory at the first step with micro-batch 1; retrying with activation "
                          "checkpointing (blocks recomputed in the backward pass).")
                step -= 1
                continue
            first_step = False
            done_step = step
            interval_micro += accum
            progress_micro += accum
            interval_tokens += settings["tokens_per_step"]
            tokens_seen += settings["tokens_per_step"]

            do_eval = step % args.eval_every == 0 or step == total_steps
            do_progress = args.log_every and step % args.log_every == 0 and not do_eval
            stop_now = bool(args.stop_after_steps) and step - start_step >= args.stop_after_steps
            do_save = (step % args.save_every == 0 or step == total_steps or stop_now
                       or (args.save_minutes > 0 and time.perf_counter() - last_save >= args.save_minutes * 60))
            heartbeat = (args.heartbeat_seconds > 0 and not (do_eval or do_progress)
                         and time.perf_counter() - last_print >= args.heartbeat_seconds)
            if not (do_eval or do_progress or do_save or heartbeat):
                continue

            synchronize(device)
            now = time.perf_counter()
            interval_train_time += now - seg
            train_seconds += now - seg

            if do_progress or heartbeat:
                ptrain = (progress_loss / max(progress_micro, 1)).item()
                next_eval = min(total_steps, (step // args.eval_every + 1) * args.eval_every)
                per_step = interval_train_time / max(step - interval_start_step, 1)
                print(f"{'  ... ' if heartbeat else ''}step {step:>6}/{total_steps} | train {ptrain:.3f} | "
                      f"lr {lr:.2e} | {interval_tokens / max(interval_train_time, 1e-9):,.0f} tok/s | "
                      f"{per_step:.2f} s/step | {elapsed_text(now)} | "
                      f"next val @{next_eval} in {fmt_dur(per_step * (next_eval - step))}")
                progress_loss.zero_()
                progress_micro = 0
                last_print = time.perf_counter()

            if do_eval:
                train_loss = (interval_loss / max(interval_micro, 1)).item()
                if not math.isfinite(train_loss):
                    raise FloatingPointError(
                        f"training loss became {train_loss} by step {step}. The last good checkpoint is "
                        f"{ckpt_path}; resume it with a lower --lr (start a new --name with --init-from).")
                tok_s = interval_tokens / max(interval_train_time, 1e-9)
                ev = evaluate(model, evalset, device, amp_dtype, eval_batch, byte_tables=byte_tables)
                ev["step"] = step
                probe_acc, probe_by_category, probe_floor_acc = None, None, None
                if probes:
                    probe_acc, probe_results = score_probes(model, tok, probes, device, amp_dtype)
                    probe_floor_acc = probe_floor(probe_results)
                    probe_by_category = {k: round(v["accuracy"], 4) for k, v in probe_summary(probe_results).items()}
                    ev["probes"] = probe_report(probe_results, probes_sha)  # kept in the checkpoint for view_pt
                prev_best, prev_step = best_val, best_step
                is_best = ev["loss"] < best_val
                if is_best:
                    best_val, best_step = ev["loss"], step
                last_eval = {k: v for k, v in ev.items()}
                elapsed = time.perf_counter() - t_run
                per_step = (time.perf_counter() - interval_wall) / max(step - interval_start_step, 1)
                eta = per_step * (total_steps - step)
                entry = {"step": step, "tokens": tokens_seen, "train_loss": train_loss, "val_loss": ev["loss"],
                         "val_per_source": ev["per_source"], "ppl": math.exp(min(ev["loss"], 50)), "lr": lr,
                         "bpb": ev["bpb"], "bpb_per_source": ev["bpb_per_source"],
                         "tok_s": tok_s, "elapsed_s": elapsed, "train_seconds": train_seconds,
                         "run_seconds": run_seconds(),
                         "entropy": ev["entropy"], "confidence": ev["confidence"], "top1": ev["top1"],
                         "ece": ev["ece"], "probe_acc": probe_acc, "probe_floor": probe_floor_acc,
                         "probe_by_category": probe_by_category,
                         "grad_norm": float(last_gnorm), "energy_wh": meter.snapshot(),
                         "co2e_g": round(meter.co2e_g(), 1), "cost_gbp": round(meter.cost_gbp() or 0.0, 4),
                         "is_best": is_best, "time": now_iso()}
                history.append(entry)
                write_metrics(entry)
                if is_best:  # before printing or sampling, so Ctrl+C can never leave an announced best unsaved
                    atomic_save(payload(step, inference_only=True), best_path)
                    if dashboard_on:
                        launch_dashboard(best_path, args, dashboard)
                if is_best:
                    status_text = (f"NEW BEST (prev {prev_best:.3f} @{prev_step})" if prev_step is not None
                                   else "NEW BEST (first evaluation)")
                else:
                    status_text = f"best {best_val:.3f} @{best_step}"
                print(f"step {step:>6}/{total_steps} | train {train_loss:.3f} | val {ev['loss']:.3f} @{step}"
                      + _per_source(ev) + f" | ppl {entry['ppl']:,.2f} | bpb {ev['bpb']:.3f} | lr {lr:.2e} | {tok_s:,.0f} tok/s | "
                      f"{meter.short()} | {elapsed_text()} | ETA {fmt_dur(eta)} | {status_text}")
                last_print = time.perf_counter()
                if args.sample_every and (step % args.sample_every == 0 or step == total_steps):
                    gen = torch.Generator().manual_seed(sample_gen_seed)
                    text = generate(model, tok, args.sample_prompt, args.sample_tokens, device, amp_dtype,
                                    0.8, 50, 0.95, gen)
                    shown = text.replace("\n", " \u23ce ").strip()
                    print(f"    sample @{step}: \"{shown}\"")
                interval_loss.zero_()
                interval_micro, interval_tokens, interval_train_time = 0, 0, 0.0
                progress_loss.zero_()
                progress_micro = 0
                interval_wall, interval_start_step = time.perf_counter(), step

            if do_save:
                if not bool(torch.isfinite(last_gnorm)):
                    raise FloatingPointError(
                        f"gradients became {float(last_gnorm)} by step {step}; {ckpt_path} was not overwritten and "
                        "still holds the last good training state. Resume it with a lower --lr (a new --name with "
                        "--init-from).")
                atomic_save(payload(step, inference_only=False), ckpt_path)
                last_save = time.perf_counter()
            if stop_now and step < total_steps:
                status = "stopped"
                print(f"Stopped after {args.stop_after_steps:,} steps this session (--stop-after-steps); saved "
                      f"{ckpt_path}.\nContinue with: python tiny_gpt.py --name {args.name} --resume")
                break
            seg = time.perf_counter()
    except KeyboardInterrupt:
        status = "interrupted"
        train_seconds += time.perf_counter() - seg
        if done_step < step:
            # Ctrl+C arrived inside a step: its update never happened. Rewind the
            # sampler so --resume replays exactly that step, and save the last
            # completed step (the original script saved the unfinished step number).
            batcher.gen.set_state(sampler_snapshot[0])
            if sampler_snapshot[1] is not None:  # dropout masks of the replayed step
                torch.set_rng_state(sampler_snapshot[1])
            if sampler_snapshot[2] is not None:
                torch.cuda.set_rng_state_all(sampler_snapshot[2])
            opt.zero_grad(set_to_none=True)
            step = done_step
        if bool(torch.isfinite(last_gnorm)):
            atomic_save(payload(step, inference_only=False), ckpt_path)
            print(f"\nPaused after step {step:,} (last completed update); saved {ckpt_path}.\n"
                  f"Resume with: python tiny_gpt.py --name {args.name} --resume")
        else:
            crash = os.path.join(args.out_dir, f"{args.name}_crash.pt")
            atomic_save(payload(step, inference_only=False), crash)
            print(f"\nPaused after step {step:,}, but the gradients were not finite: the state went to {crash}; "
                  f"{ckpt_path} still holds the last good training state.")
    except BaseException:
        status = "failed"
        crash = os.path.join(args.out_dir, f"{args.name}_crash.pt")
        try:
            atomic_save(payload(step, inference_only=False), crash)
            print(f"\nState at step {step:,} saved to {crash} for inspection (not used by --resume).")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not save crash state: {exc}")
        raise
    finally:
        meter.stop()
        n_total, n_nonembed = model.param_counts()
        last_hist = history[-1] if history else {}
        record_experiment(args.out_dir, {
            "event": "training run", "name": args.name, "status": status, "step": step, "total_steps": total_steps,
            "params": n_total, "non_embedding_params": n_nonembed, "d_model": cfg.d_model, "layers": cfg.n_layers,
            "heads": cfg.n_heads, "kv_heads": cfg.n_kv_heads, "ctx": cfg.ctx, "vocab": cfg.vocab_size,
            "tokens_seen": tokens_seen, "tokens_per_step": settings["tokens_per_step"],
            "micro_batch": settings["micro_batch"], "grad_accum": settings["grad_accum"],
            "optimizer": OPTIMIZER_LABELS.get(settings.get("optimizer", "adamw")), "lr": f"{settings['lr']:.3g}",
            "min_lr": f"{settings['min_lr']:.3g}", "schedule": {"wsd": "WSD"}.get(settings["schedule"],
                                                                                 settings["schedule"]),
            "warmup_steps": settings["warmup_steps"],
            "decay_frac": settings["decay_frac"] if settings["schedule"] == "wsd" else None,
            "decay_shape": settings["decay_shape"] if settings["schedule"] == "wsd" else None,
            "weight_decay": settings["weight_decay"], "betas": "/".join(f"{b:g}" for b in settings["betas"]),
            "grad_clip": settings["grad_clip"], "z_loss": z_loss, "dropout": cfg.dropout, "precision": amp_name,
            "dataset": os.path.basename(os.path.normpath(dataset_dir)),
            "best_val": f"{best_val:.4f}" if math.isfinite(best_val) else None, "best_step": best_step,
            "last_val": f"{last_eval['loss']:.4f}" if last_eval else None,
            "last_bpb": f"{last_eval['bpb']:.4f}" if last_eval and last_eval.get("bpb") else None,
            "last_ppl": f"{math.exp(min(last_eval['loss'], 50)):.2f}" if last_eval else None,
            "benchmark_accuracy": f"{last_hist['probe_acc']:.3f}" if last_hist.get("probe_acc") is not None else None,
            "benchmark_floor": f"{last_hist['probe_floor']:.3f}" if last_hist.get("probe_floor") is not None else None,
            "tok_s": f"{last_hist['tok_s']:.0f}" if last_hist.get("tok_s") else None, "device": device.type,
            "train_hours": f"{train_seconds / 3600:.2f}", "run_hours": f"{run_seconds() / 3600:.2f}",
            "energy_kwh": f"{meter.kwh():.3f}",
            "co2e_kg": f"{meter.co2e_g() / 1000:.4f}", "cost_gbp": f"{meter.cost_gbp() or 0.0:.2f}",
            "checkpoint": ckpt_path})
        print(f"RUN {run_id} {status} at {now_iso()} | step {step:,}/{total_steps:,} | best val "
              f"{best_val:.4f} @{best_step} | last val "
              + (f"{last_eval['loss']:.4f} @{last_eval['step']}" if last_eval else "none")
              + f" | this session {fmt_hms(time.perf_counter() - t_run)} | whole run: "
              f"{fmt_hms(run_seconds())} in total, {fmt_hms(train_seconds)} of it training, {meter.short()} "
              f"(GPU {meter.wh['gpu'] / 1000:.2f}, CPU {meter.wh['cpu'] / 1000:.2f}, memory "
              f"{meter.wh['ram'] / 1000:.3f} kWh)")


EXPERIMENT_COLUMNS = ["time", "event", "name", "status", "step", "total_steps", "params", "non_embedding_params",
                      "d_model", "layers", "heads", "kv_heads", "ctx", "vocab", "tokens_seen", "tokens_per_step",
                      "micro_batch", "grad_accum", "optimizer", "lr", "min_lr", "schedule", "warmup_steps",
                      "decay_frac", "decay_shape", "weight_decay", "betas", "grad_clip", "z_loss", "dropout",
                      "precision", "dataset", "best_val", "best_step", "last_val", "last_bpb", "last_ppl",
                      "benchmark_accuracy", "benchmark_floor", "benchmark_items", "tok_s", "device", "train_hours",
                      "run_hours", "energy_kwh", "co2e_kg", "cost_gbp", "checkpoint", "notes"]


def publish_to_github(rel, message, content=None, src=None, wait=True):
    """Put one file into this repository at `rel` (text `content`, or a copy of
    `src`), commit only that file and push. wait=False pushes in the background,
    so a run can end at once. Returns a short status; never raises."""
    try:
        top = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                             timeout=60)
        if top.returncode != 0:
            return "not in a git repository"
        root = top.stdout.strip()
        dest = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if content is not None:
            with open(dest + ".tmp", "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            os.replace(dest + ".tmp", dest)
        elif os.path.abspath(src) != os.path.abspath(dest):
            import shutil
            shutil.copy2(src, dest)

        def git(*args):
            return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=300)

        git("add", "--", rel)
        if git("diff", "--cached", "--quiet", "--", rel).returncode == 0:
            return "unchanged since the last push"
        commit = git("commit", "-q", "-m", message, "--", rel)
        if commit.returncode != 0:
            return "not committed: " + (commit.stderr or commit.stdout).strip()[:200]
        if not wait:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            subprocess.Popen(["git", "-C", root, "push", "-q", "origin", "HEAD"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=flags)
            return "committed; pushing in the background"
        push = git("push", "-q", "origin", "HEAD")
        if push.returncode != 0:
            return "committed but not pushed (it goes up with the next push): " + push.stderr.strip()[:200]
        return "pushed"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return f"not published ({type(exc).__name__}: {exc})"


def publish_experiments(path, row):
    """The experiments table of this folder's runs, on GitHub as results/experiments.csv,
    with the home folder shown as ~ (the local file keeps full paths)."""
    with open(path, encoding="utf-8-sig") as handle:
        text = handle.read()
    home = os.path.expanduser("~")
    for variant in {home, home.replace("\\", "/")}:
        text = re.sub(re.escape(variant), "~", text, flags=re.IGNORECASE)
    what = " ".join(str(row.get(k)) for k in ("event", "name", "status") if row.get(k))
    status = publish_to_github("results/experiments.csv", f"Experiments: {what}", content=text, wait=False)
    print(f"experiments.csv on GitHub (results/experiments.csv): {status}")


def record_experiment(out_dir, row):
    """Append one row to <out_dir>/experiments.csv (opens in Excel): a
    permanent record of every run, benchmark and comparison."""
    import csv
    path = os.path.join(out_dir, "experiments.csv")
    new = not os.path.exists(path)
    if not new:
        try:
            with open(path, encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), [])
            with open(path, encoding="utf-8-sig", newline="") as handle:
                old_rows = list(csv.DictReader(handle))
            # known columns in the standard order; columns this version does not know are kept at the end
            target = (EXPERIMENT_COLUMNS if set(header) <= set(EXPERIMENT_COLUMNS)
                      else header + [c for c in EXPERIMENT_COLUMNS if c not in header])
            if header != target:
                # a newer version added columns: keep a copy, rewrite once with the full header, lose nothing
                stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                import shutil
                shutil.copy2(path, f"{path}.before-columns-{stamp}")
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=target)
                    writer.writeheader()
                    writer.writerows(old_rows)
                os.replace(tmp, path)
            columns = target
        except PermissionError:
            columns = EXPERIMENT_COLUMNS  # the append below reports the locked file
    else:
        columns = EXPERIMENT_COLUMNS
    try:
        with open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow({"time": now_iso(), **{k: v for k, v in row.items() if v is not None}})
    except PermissionError:
        print(f"Note: {path} is open in another program (Excel?); this row was not recorded: {row}")
        return
    if os.path.normcase(os.path.abspath(out_dir)) == os.path.normcase(HERE):  # this folder's runs, not test runs
        publish_experiments(path, row)


def _per_source(ev):
    if len(ev["per_source"]) < 2:
        return ""
    return " [" + " ".join(f"{n} {v:.3f}" for n, v in ev["per_source"].items()) + "]"


def _args_with_shape(args, cfg):
    clone = argparse.Namespace(**vars(args))
    clone.d_model, clone.layers, clone.heads, clone.kv_heads = cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads
    clone.ctx, clone.ffn_hidden, clone.rope_base = cfg.ctx, cfg.ffn_hidden, cfg.rope_base
    return clone


def maybe_compile(model, mode, device):
    if mode == "off":
        return model
    try:
        import triton  # noqa: F401
        have_triton = True
    except ImportError:
        have_triton = False
    # auto: compile on CUDA whenever Triton is importable, Windows included (the triton-windows package; measured
    # 1.45x faster training on an RTX 3070 laptop for 36.6M and 70M models). The first steps then take ~10-30 s longer.
    if mode == "auto" and not (device.type == "cuda" and have_triton):
        return model
    if not have_triton and device.type == "cuda":
        print("torch.compile skipped: Triton is not installed (on Windows: python -m pip install triton-windows, the "
              "version matching PyTorch); using eager mode.")
        return model
    try:
        compiled = torch.compile(model)
        x = torch.zeros((1, min(model.ctx, 64)), dtype=torch.long, device=device)
        compiled(x).float().sum().backward()
        model.zero_grad(set_to_none=True)
        return compiled
    except Exception as exc:  # noqa: BLE001
        print(f"torch.compile failed ({type(exc).__name__}: {str(exc)[:120]}); using eager mode.")
        model.zero_grad(set_to_none=True)
        return model


def cmd_plan(args):
    dataset_dir = ensure_dataset(args)
    if not dataset_dir:
        raise SystemExit("--plan needs --dataset.")
    manifest = data_prep.read_manifest(dataset_dir)
    device = select_device(args.device)
    amp_dtype, amp_name, _ = choose_precision(device, args.precision)
    cfg, settings, lines = plan_with_time_budget(args, manifest, device, amp_dtype)
    print(f"\nPlan for {dataset_dir} on {device_label(device)} ({amp_name}) at {now_iso()}")
    for line in lines:
        print("  " + line)
    print("  schedule: " + LRSchedule(settings["schedule"], settings["steps"], settings["lr"], settings["min_lr"],
                                      settings["warmup_steps"], settings["decay_frac"],
                                      settings["decay_shape"]).describe())
    print(f"\nAssumptions: the planner sizes the model so training tokens ~= {args.tokens_per_param:g} x parameters "
          "(Hoffmann et al. 2022 found ~20 compute-optimal for large models trained once on fresh data; it is a "
          "heuristic, not a law, and small models deployed for inference are routinely trained far beyond it). "
          f"Repetition is capped at {args.max_epochs:g} epochs (Muennighoff et al. 2023: up to ~4 epochs of repeated "
          "data is almost as good as unique data; returns fade after that). Parameters include the tied embedding. "
          "Learning rate and tokens/step are heuristics; confirm with a short LR sweep on your data.")
    print("\nWhat the planner would pick for other corpus sizes (same vocabulary; 'batch' = micro x accumulation "
          "that fits this device now):")
    print(f"  {'unique tokens':>14} {'train tokens':>14} {'d':>5} {'L':>3} {'H/KV':>6} {'ctx':>5} {'params':>11} "
          f"{'tok/param':>9} {'tok/step':>9} {'batch':>12} {'lr':>8}")
    for unique, planned, c, n, tps, fits, lr in plan_table(int(manifest["tokenizer"]["vocab_size"]), device,
                                                           amp_dtype, args.tokens_per_param, args.max_epochs):
        print(f"  {unique:>14,.0f} {planned:>14,.0f} {c.d_model:>5} {c.n_layers:>3} {c.n_heads:>3}/{c.n_kv_heads:<2} "
              f"{c.ctx:>5} {n:>11,} {planned / n:>9.1f} {tps:>9,} {fits:>12} {lr:>8.1e}")


def _log_path_from(argv):
    """--log-file is read before full parsing so argument errors are logged too."""
    for i, item in enumerate(argv):
        if item == "--log-file" and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith("--log-file="):
            return item.split("=", 1)[1]
    return os.path.join(HERE, "log.txt")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    start_log(_log_path_from(argv))
    print(f"\n--- tinyGPT {now_iso()} | log {os.path.abspath(_log_path_from(argv))} ---")
    try:
        args = parse_args(argv)
        if args.generate is not None:
            cmd_generate(args)
        elif args.benchmark:
            cmd_benchmark(args)
        elif args.export:
            cmd_export(args)
        elif args.plan:
            cmd_plan(args)
        else:
            cmd_train(args)
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"ERROR: {exc.code}" if isinstance(exc.code, str) else f"exit code {exc.code}", file=sys.stderr)
            raise SystemExit(1)
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        stop_log()


if __name__ == "__main__":
    main()
