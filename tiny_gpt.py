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
       python run_pipeline.py build        (the data_prep.py command it runs: HOW_TO_RUN.md, Step 3)
2. Look at the automatic plan, then train (the time budget measures the real
   speed on this device and picks the largest model the data AND the time allow):
       python tiny_gpt.py --dataset datasets\bio_all --plan --time-budget-hours 48
       python tiny_gpt.py --dataset datasets\bio_all --name bio_48h --time-budget-hours 48
       python tiny_gpt.py --dataset datasets\bio_all --name bio_small --d-model 256 --layers 6 --steps 5000 --tokens-per-step 65536
3. Resume after Ctrl+C (same name; the dataset path is stored in the checkpoint):
       python tiny_gpt.py --name tinyGPT_muon --resume
       python tiny_gpt.py --name tinyGPT_muon --resume --steps 70000     # extend the run
4. Generate (the default --name is tinyGPT; give the run's name or a --checkpoint):
       python tiny_gpt.py --name tinyGPT_muon --generate "Genetic drift is" --length 200
       python tiny_gpt.py --checkpoint tinyGPT_muon_best.pt --generate "Genetic drift"

Output: <name>.pt (training checkpoint), <name>_best.pt (inference-only, best
validation loss), <name>_metrics.jsonl, and log.txt (everything printed to the
terminal, including errors, is appended to it).

Defaults of a new run (a resumed run keeps what its checkpoint recorded):
AdamW (--optimizer muon for Muon), QK-norm on, a WSD schedule decaying to 0,
the planner's learning-rate table (heuristic_lr) with weight decay 0.05, no
weight average (--ema on), no document mask (--doc-mask on), and a dense
SwiGLU in every block (--experts N for a mixture of experts).

Portability: CUDA (BF16/FP16 autocast, fused AdamW), Apple MPS (BF16 autocast
and fused AdamW when this PyTorch supports them; the run caps itself at Metal's
recommended working set) and CPU all work in eager mode. torch.compile is used
in auto mode where Triton exists (CUDA; on Windows via the triton-windows
package, about 1.45x faster); on MPS it is tried only with --compile on and
falls back to eager when the graph fails.
"""

import argparse
import atexit
import contextlib
import dataclasses
import datetime
import gc
import glob
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import signal
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
FORMAT_VERSION = 5  # 5: mixture-of-experts fields in model_config and router.bias buffers
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
        try:
            if not self.log.closed:
                self.log.write(text)
                if "\n" in text:
                    self.log.flush()
        except (OSError, ValueError):  # a full or lost disk: the line is missing from the log, the run goes on
            pass
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


def log_only(text):
    """A line for log.txt only: detail that makes a run reproducible but that nobody reads on screen.
    Without an open log file it is printed, so nothing is ever lost."""
    handle = _LOG_STATE.get("handle")
    if handle is None or handle.closed:
        print(text)
        return
    try:
        handle.write(text + "\n")
        handle.flush()
    except (OSError, ValueError):
        pass


def clock():
    """'Tue 12:41': when a line was printed."""
    return time.strftime("%a %H:%M")


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def fmt_hms(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def fmt_dur(seconds):
    """Duration as '17d 04h' / '9h 50m' / '12m 03s' (never confused with a clock time)."""
    seconds = int(max(seconds, 0))
    if seconds >= 172_800:  # from two days on: '17d 04h'
        return f"{seconds // 86_400}d {seconds % 86_400 // 3600:02d}h"
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def file_sha256(path):
    try:
        return data_prep.sha256_file(path)
    except OSError:
        return "?"


def source_sha256(path):
    """sha256 of a source file with its CRLF line ends read as LF, so the same code has the same hash in a
    Windows checkout and elsewhere (the run header's hash of this script); '?' when it cannot be read.
    file_sha256 hashes the bytes as they are: dataset and probe files, and the stored probes_sha, use that."""
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
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
        try:  # the chip, its GPU core count and Metal's recommended working set (torch >= 2.4)
            return (f"{torch.backends.mps.get_name()} ({torch.backends.mps.get_core_count()} GPU cores, "
                    f"{torch.mps.recommended_max_memory() / 2 ** 30:.0f} GiB recommended, "
                    f"macOS {platform.mac_ver()[0] or '?'})")
        except Exception:  # noqa: BLE001 - an older PyTorch without these calls
            return "Apple MPS"
    return f"CPU ({os.cpu_count()} threads)"


def backend_label(device):
    """'cuda 12.6', 'mps' or 'cpu' for the run header (torch.version.cuda is None off CUDA)."""
    return f"cuda {torch.version.cuda}" if device.type == "cuda" else device.type


_MPS_CAPPED = False


def cap_device_memory(device):
    """On MPS, cap this process at Metal's recommended working set (fraction 1.0 of
    it; PyTorch's default high-watermark ratio 1.7 lets allocations go beyond it,
    and unified memory then pages to the SSD for hours instead of failing). With
    the cap an over-budget plan raises the "MPS backend out of memory" RuntimeError
    that the first-step handler and the planner catch. Done once per process;
    nothing on CUDA (its allocator fails at the card's memory) or the CPU."""
    global _MPS_CAPPED
    if device.type != "mps" or _MPS_CAPPED:
        return False
    _MPS_CAPPED = True
    fn = getattr(torch.mps, "set_per_process_memory_fraction", None)
    if fn is None:
        return False
    try:
        fn(1.0)
        return True
    except (RuntimeError, TypeError, ValueError):
        return False


def device_rng_state(device):
    """The device's random generator state(s), on the CPU: the list of every CUDA
    device's state, the MPS generator's state, None on the CPU (torch.get_rng_state
    holds that one) or when the backend cannot report it."""
    try:
        if device.type == "cuda":
            return [s.cpu() for s in torch.cuda.get_rng_state_all()]
        if device.type == "mps":
            return torch.mps.get_rng_state().cpu()
    except (RuntimeError, AttributeError):
        return None
    return None


def set_device_rng_state(device, state):
    """Undo of device_rng_state: nothing for None (another device, or an older checkpoint)."""
    if state is None:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([s.cpu() for s in state])
    elif device.type == "mps":
        torch.mps.set_rng_state(state.cpu())


def fork_device_rng(device):
    """torch.random.fork_rng over the CPU generator and this device's (CUDA or MPS),
    so the dropout masks of a speed probe or compile warm-up do not consume the
    run's random stream. On the CPU only its own generator is forked."""
    if device.type in ("cuda", "mps"):
        return torch.random.fork_rng(devices=[device.index or 0], device_type=device.type)
    return torch.random.fork_rng(devices=[])


def mps_env_notes(environ=None):
    """Lines for the run header about the MPS allocator environment variables that
    change memory behaviour or speed (printed only on MPS)."""
    env = os.environ if environ is None else environ
    notes = []
    for key in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"):
        if env.get(key):
            notes.append(f"{key}={env[key]} (allocator watermark; the default high ratio is 1.7, low 1.4; the run "
                         "caps itself at the recommended working set with set_per_process_memory_fraction(1.0))")
    if env.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        notes.append("WARNING: PYTORCH_ENABLE_MPS_FALLBACK=1: operations MPS has no kernel for run on the CPU; expect "
                     "10-100x slower steps (set it only to diagnose a missing operation)")
    return notes


GPU_BUSY_MESSAGE = ("The GPU is busy (another training run?): two training jobs share one GPU / unified memory, so "
                    "the second run slows both or swaps (and two CUDA jobs on a throttled laptop GPU can reset the "
                    "driver). Wait for the other job or stop it first.")


def gpu_busy(*out_dirs, smi=True):
    """True when another training job is running, on any device: a <name>.train.lock
    whose process is alive in one of `out_dirs` (default: this script's folder), or,
    where nvidia-smi exists, more than 2500 MiB of GPU memory in use or a utilization
    of 20 % or more (a training run holds gigabytes even while it saves). Without
    nvidia-smi (macOS) the lock files are the only signal, so every trainer in this
    project takes one. smi=False asks the lock files alone: nvidia-smi's figures are
    also that high on desktop load, which is too weak a reason to warn every run."""
    for folder in out_dirs or (HERE,):
        try:
            if any(data_prep._lock_owner_alive(lock) for lock in glob.glob(os.path.join(folder, "*.train.lock"))):
                return True
        except OSError:
            pass
    if not smi:
        return False
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
        rows = [[float(v) for v in line.split(",")] for line in out.strip().splitlines()]
        return any(used > 2500 or util >= 20 for used, util in rows)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return False


def make_adamw(groups, device, **kwargs):
    """(AdamW optimizer, label): the fused kernel where there is one (CUDA, MPS: one
    launch per step instead of about ten per parameter tensor), plain AdamW where
    this PyTorch refuses it. The CPU deliberately stays plain, so step-for-step
    comparisons on the CPU (the tests) keep their exact arithmetic."""
    if device.type in ("cuda", "mps"):
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs), "fused AdamW"
        except (TypeError, RuntimeError, ValueError):
            pass
    return torch.optim.AdamW(groups, **kwargs), "AdamW"


def free_device_memory(device):
    """Return freed tensors to the device's allocator (and the allocator's cache to
    the driver) after an out-of-memory error or a failed save: gc first, so that
    tensors kept alive only by a reference cycle are released as well."""
    gc.collect()
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "empty_cache"):
        try:
            backend.empty_cache()
        except RuntimeError:  # a backend that is present but not initialised
            pass


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


def _mps_scaler_ok():
    """Whether this PyTorch can scale FP16 gradients on MPS (GradScaler('mps'))."""
    try:
        p = torch.nn.Parameter(torch.ones(4, device="mps"))
        opt = torch.optim.SGD([p], lr=0.1)
        scaler = torch.amp.GradScaler("mps")
        scaler.scale(p.sum()).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        return bool(torch.isfinite(p).all().item())
    except Exception:  # noqa: BLE001 - any failure means "not supported"
        return False


def choose_precision(device, preference="auto", training=True):
    """Return (autocast dtype or None, label, needs GradScaler). FP16 training
    needs loss scaling (GradScaler) so small gradients do not underflow; on MPS
    it is used when this PyTorch supports it, otherwise FP16 training is refused
    (training=False: inference only, no gradients, so no scaler is needed)."""
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
        if device.type == "mps" and training:
            if not _mps_scaler_ok():
                raise SystemExit("--precision fp16 on Apple MPS needs gradient (loss) scaling, which this PyTorch does "
                                 "not support on MPS; use --precision bf16 (or auto) or fp32.")
            return torch.float16, "FP16 autocast + loss scaling", True
        return torch.float16, "FP16 autocast", device.type == "cuda" and training
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
    faster depends on which SDPA kernels this PyTorch build dispatches to: some
    builds (measured on Windows CUDA) send enable_gqa to the slow MATH kernel, so
    on CUDA the two are timed on the model's real head shape and the faster wins;
    the run header prints the timing.
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
    q = k = v = None  # freed (not del: the timing lambdas share these names, which pyflakes flags)
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
    # QK-norm (an RMSNorm over each query and key head before RoPE). False is the default every checkpoint
    # written before the field existed loads with; the planner turns it on for new runs.
    qk_norm: bool = False
    # Mixture of experts (n_experts 0 = the dense SwiGLU every checkpoint written before these fields existed
    # has). In an MoE block the SwiGLU is replaced by n_experts smaller SwiGLUs of expert_hidden (0 = auto:
    # ffn_hidden / experts_per_token rounded to 64, so the FLOPs per token equal the dense FFN) and a router that
    # sends each token to experts_per_token of them; shared_hidden > 0 adds a SwiGLU every token goes through.
    # moe_every m makes every m-th block an MoE block (1: all). router: "softmax" (Mixtral) or "sigmoid"
    # (DeepSeek-V3) scores. balance: "loss" (the Switch load-balancing loss), "bias" (DeepSeek-V3's
    # auxiliary-loss-free bias on the routing scores, updated after each step) or "both".
    n_experts: int = 0
    experts_per_token: int = 2
    expert_hidden: int = 0
    shared_hidden: int = 0
    moe_every: int = 1
    router: str = "softmax"
    balance: str = "bias"

    def __post_init__(self):
        if not self.ffn_hidden:
            self.ffn_hidden = swiglu_hidden(self.d_model)
        if self.n_experts and not self.expert_hidden:
            self.expert_hidden = expert_hidden_for(self.ffn_hidden, self.experts_per_token)
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
        if self.n_experts and not 2 <= self.n_experts <= 64:
            problems.append(f"n_experts must be 0 (dense) or between 2 and 64, not {self.n_experts}")
        if self.n_experts and not 1 <= self.experts_per_token <= self.n_experts:
            problems.append(f"experts_per_token {self.experts_per_token} must be between 1 and n_experts "
                            f"{self.n_experts}")
        if self.n_experts and self.expert_hidden < 1:
            problems.append("expert_hidden must be positive")
        if self.n_experts and self.moe_every < 1:
            problems.append("moe_every must be at least 1")
        if self.n_experts and self.moe_every > self.n_layers:
            # (index + 1) % moe_every == 0 never holds: an "MoE" run with no MoE block (and no load figures to report)
            problems.append(f"moe_every {self.moe_every} leaves no mixture-of-experts block in {self.n_layers} layers")
        if self.shared_hidden < 0:
            problems.append("shared_hidden must be >= 0")
        if self.router not in ("softmax", "sigmoid"):
            problems.append(f"router must be softmax or sigmoid, not {self.router!r}")
        if self.balance not in ("loss", "bias", "both"):
            problems.append(f"balance must be loss, bias or both, not {self.balance!r}")
        if problems:
            raise SystemExit("Invalid model settings: " + "; ".join(problems))

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    def is_moe_block(self, index):
        """Whether block `index` (0-based) is a mixture-of-experts block."""
        return bool(self.n_experts) and (index + 1) % self.moe_every == 0

    @property
    def n_moe_blocks(self):
        return sum(self.is_moe_block(i) for i in range(self.n_layers))

    def moe_label(self):
        """'8x2/1' (experts x per token / every), the experiments-table form; '' for a dense model."""
        return f"{self.n_experts}x{self.experts_per_token}/{self.moe_every}" if self.n_experts else ""


def swiglu_hidden(d_model, multiple=64):
    """8/3 * d rounded up to a multiple of 64 (keeps parameters ~ a 4x GELU MLP
    and gives tensor-core friendly shapes)."""
    return int(multiple * math.ceil((8 * d_model / 3) / multiple))


def expert_hidden_for(ffn_hidden, experts_per_token, multiple=64):
    """Hidden size of one expert so that the experts_per_token experts a token
    goes through do the dense FFN's work: ffn_hidden / k rounded to a multiple of
    64 (at least 64)."""
    return int(max(multiple, multiple * round(ffn_hidden / max(experts_per_token, 1) / multiple)))


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

    def forward(self, q, k, pos=0):
        """pos: the absolute position of the first token in q and k (KV-cached decoding)."""
        t = q.size(-2)
        cos, sin = self.cos[:, :, pos:pos + t], self.sin[:, :, pos:pos + t]
        # The rotation is done in fp32 and rounded once at the end: cos/sin stay exact, so the products are
        # rounded to bf16 once instead of twice. torch.compile fuses the casts, so under
        # autocast this costs nothing; in eager mode it is about 1% more memory traffic.
        qf, kf = q.float(), k.float()
        return ((qf * cos + rotate_half(qf) * sin).to(q.dtype), (kf * cos + rotate_half(kf) * sin).to(k.dtype))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Router(nn.Module):
    """Chooses experts_per_token experts for every token. `weight` (E, d) is the
    trained routing matrix; `bias` (E,) is DeepSeek-V3's auxiliary-loss-free
    balancing term (Wang et al. 2024): a buffer, never a parameter, added to the
    scores for the choice of experts only (not to the weights the chosen
    experts' outputs are mixed with), moved by TinyGPT.update_router_bias after
    each optimizer step so that overloaded experts are chosen less. The logits
    are computed in fp32 whatever autocast says: routing decisions and the
    load-balancing loss are too sensitive to bf16 rounding."""

    def __init__(self, dim, n_experts, experts_per_token, kind="softmax"):
        super().__init__()
        self.n_experts, self.k, self.kind = n_experts, experts_per_token, kind
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.register_buffer("bias", torch.zeros(n_experts))

    def forward(self, x2):
        """x2 (T, d) -> normalised top-k weights (T, k), expert ids (T, k), fp32 logits (T, E) and the number of
        tokens per expert (E,). The count is a scatter_add_ of static shape (bincount's data-dependent output
        shape would break a torch.compile graph)."""
        with torch.autocast(device_type=x2.device.type, enabled=False):
            logits = F.linear(x2.float(), self.weight.float())
            scores = logits.softmax(-1) if self.kind == "softmax" else torch.sigmoid(logits)
            _, ids = (scores + self.bias).topk(self.k, dim=-1)
            w = scores.gather(-1, ids)
            w = w / w.sum(-1, keepdim=True)
        flat = ids.reshape(-1)
        counts = torch.zeros(self.n_experts, dtype=torch.long, device=x2.device).scatter_add_(
            0, flat, torch.ones_like(flat))
        return w, ids, logits, counts


class Experts(nn.Module):
    """n_experts SwiGLUs of hidden size h, each a 2-D gate_up (2h, d) and down
    (d, h) matrix. Kept as separate 2-D parameters (an nn.ParameterList) rather
    than one (E, 2h, d) tensor, so torch.optim.Muon (2-D only), build_optimizer's
    p.dim() == 2 rule and the by-name optimizer-state reload all work unchanged.
    impl "loop": the tokens are sorted by expert and each expert with work runs
    one matmul (the general implementation, every device); "grouped": the same
    sort feeding torch._grouped_mm (one kernel for all experts; a gain only for a
    compiled CUDA graph, and used only when the probe at model build passed)."""

    def __init__(self, dim, hidden, n_experts, experts_per_token):
        super().__init__()
        self.dim, self.hidden, self.k = dim, hidden, experts_per_token
        self.impl = "loop"
        self.gate_up = nn.ParameterList([nn.Parameter(torch.empty(2 * hidden, dim)) for _ in range(n_experts)])
        self.down = nn.ParameterList([nn.Parameter(torch.empty(dim, hidden)) for _ in range(n_experts)])
        for p in list(self.gate_up) + list(self.down):
            nn.init.normal_(p, mean=0.0, std=0.02)  # down is rescaled by TinyGPT like every residual projection

    def forward(self, x2, w, ids, counts):
        """x2 (T, d), the router's weights, ids and counts -> (T, d): each token's chosen experts' outputs, mixed
        with its routing weights and summed."""
        flat = ids.reshape(-1)
        order = flat.argsort(stable=True)  # slots sorted by expert; stable, so a token's slots keep their order
        token_of = order // self.k
        xs = x2[token_of]
        ys = self._grouped(xs, counts) if self.impl == "grouped" else self._loop(xs, counts)
        ys = ys * w.reshape(-1)[order].unsqueeze(-1).to(ys.dtype)
        return torch.zeros_like(x2).index_add_(0, token_of, ys.to(x2.dtype))

    def _loop(self, xs, counts):
        outs, start = [], 0
        for e, n in enumerate(counts.tolist()):
            if n == 0:  # an expert nobody chose (common when decoding one token) launches nothing
                continue
            xe = xs[start:start + n]
            gate, up = F.linear(xe, self.gate_up[e]).chunk(2, dim=-1)
            outs.append(F.linear(F.silu(gate) * up, self.down[e]))
            start += n
        return torch.cat(outs) if outs else xs[:0]

    def _grouped(self, xs, counts):
        # autocast does not cover _grouped_mm, so the inputs are cast to its dtype by hand; offsets are the
        # int32 cumulative counts the kernel expects
        kind = xs.device.type
        dtype = torch.get_autocast_dtype(kind) if torch.is_autocast_enabled(kind) else xs.dtype
        offs = counts.cumsum(0).to(torch.int32)
        gate_up = torch.stack(list(self.gate_up)).transpose(1, 2).to(dtype)  # (E, d, 2h)
        down = torch.stack(list(self.down)).transpose(1, 2).to(dtype)  # (E, h, d)
        gate, up = torch._grouped_mm(xs.to(dtype), gate_up, offs=offs).chunk(2, dim=-1)
        return torch._grouped_mm(F.silu(gate) * up, down, offs=offs)


_GROUPED_MM_PROBE = {}


def grouped_mm_available(device):
    """Whether torch._grouped_mm runs forward AND backward on this device: a
    3-expert toy problem with a strided loss (y.pow(2).mean(); a .sum() hands the
    backward a stride-0 gradient some kernels reject, which is not a failure of the
    real training loss). Returns (ok, reason); cached per device type."""
    key = device.type
    if key not in _GROUPED_MM_PROBE:
        try:
            if not hasattr(torch, "_grouped_mm"):
                raise RuntimeError("this PyTorch has no torch._grouped_mm")
            dtype = torch.bfloat16 if key in ("cuda", "mps") else torch.float32
            with torch.enable_grad():  # the probe may be called from inside a no_grad block
                # fixed inputs (no random draw, so the run's RNG state is untouched)
                x = (torch.arange(96, device=device, dtype=dtype).reshape(12, 8) / 96 - 0.5).requires_grad_()
                w = (torch.arange(384, device=device, dtype=dtype).reshape(3, 8, 16) / 384 - 0.5).requires_grad_()
                offs = torch.tensor([5, 7, 12], dtype=torch.int32, device=device)
                y = torch._grouped_mm(x, w, offs=offs)
                y.pow(2).mean().backward()
            if x.grad is None or w.grad is None or not torch.isfinite(w.grad).all():
                raise RuntimeError("the backward pass produced no gradient")
            _GROUPED_MM_PROBE[key] = (True, f"torch._grouped_mm probe passed on {key}")
        except Exception as exc:  # noqa: BLE001 - any failure means the per-expert loop is used
            _GROUPED_MM_PROBE[key] = (False, f"torch._grouped_mm probe failed on {key} ({type(exc).__name__}: "
                                             f"{str(exc)[:100]}); using the per-expert loop")
    return _GROUPED_MM_PROBE[key]


class MoEFFN(nn.Module):
    """A mixture-of-experts feed-forward layer: Router + Experts (+ an optional
    shared SwiGLU every token goes through). forward returns (output, aux) with
    aux = (lb, z, counts): the Switch load-balancing loss lb = E x sum_e f_e P_e
    (Fedus et al. 2022; f_e = share of the routed token-slots that went to expert
    e, no gradient; P_e = mean routing probability of e; exactly 1.0 at uniform
    routing, larger when the router collapses onto few experts), the router
    z-loss z = mean over tokens of (log sum_e exp logit_e)^2 (Zoph et al. 2022,
    keeps the routing logits from growing without bound), and the tokens per
    expert (E,). The terms are returned, never accumulated on the module: under
    non-reentrant activation checkpointing the recompute may run fully, partly or
    not at all, and torch.compile treats attribute writes as side effects."""

    def __init__(self, cfg):
        super().__init__()
        self.n_experts, self.k = cfg.n_experts, cfg.experts_per_token
        self.router = Router(cfg.d_model, cfg.n_experts, cfg.experts_per_token, cfg.router)
        self.experts = Experts(cfg.d_model, cfg.expert_hidden, cfg.n_experts, cfg.experts_per_token)
        self.shared = SwiGLU(cfg.d_model, cfg.shared_hidden, 0.0) if cfg.shared_hidden else None
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        w, ids, logits, counts = self.router(x2)
        out = self.experts(x2, w, ids, counts)
        if self.shared is not None:
            out = out + self.shared(x2).to(out.dtype)
        f = counts.to(logits.dtype) / (x2.size(0) * self.k)
        p = logits.softmax(-1).mean(0)
        lb = self.n_experts * (f * p).sum()
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return self.dropout(out).view(shape), (lb, z, counts)


class GroupedQueryAttention(nn.Module):
    """n_heads query heads share n_kv_heads key/value heads (GQA)."""

    def __init__(self, dim, n_heads, n_kv_heads, dropout, qk_norm=False, norm_eps=1e-6):
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
        # QK-norm (Dehghani et al. 2023; Wortsman et al. 2023): an RMSNorm over every query and every key head,
        # applied after the projection and before RoPE (the Gemma 3 / Qwen3 / OLMo 2 order), one gain vector of
        # head_dim values shared by all heads. Each normalised vector has RMS 1 (times its gain), so the attention
        # logits q.k / sqrt(d) are bounded by sqrt(d) x gain^2 whatever the learning rate: the logit-growth
        # instability at high rates disappears and the loss-against-LR curve flattens. Off, the layer is exactly
        # the plain GQA it always was.
        self.q_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else None

    def _qkv(self, x, rope, pos=0):
        """Rotated queries and keys plus values, (B, heads, T, head_dim). The single path for training,
        KV-cached decoding and the viewer's attention entropy, so QK-norm reaches all three alike."""
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q, k, pos)
        return q, k, v

    def forward(self, x, rope, cache=None, mask=None):
        """cache: None (training, scoring), or this layer's dict for KV-cached decoding:
        the keys and values of the tokens seen so far are kept in it, so each new token
        costs one position of work instead of the whole sequence. mask: None (plain
        causal attention, the fast kernels) or a boolean (B, 1, T, T) document mask from
        document_mask(), causal and confined to each document."""
        if cache is not None:
            return self._forward_cached(x, rope, cache)
        b, t, _ = x.shape
        q, k, v = self._qkv(x, rope)
        p = self.dropout if self.training else 0.0
        if self.impl == "gqa_native" and self.n_rep > 1 and mask is None:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True, enable_gqa=True)
        else:
            # a document mask always takes this path: SDPA's enable_gqa runs only on the flash and math kernels,
            # and flash takes no mask, so enable_gqa with a mask would fall to the math kernel (every head's full
            # T x T scores kept, many times slower); with the K/V heads repeated the masked call can use the
            # memory-efficient kernel
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            if mask is None:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, -1))

    def _forward_cached(self, x, rope, cache, key_valid=None):
        """KV-cached attention. key_valid: optional boolean (B, past + t) key-padding
        mask, True at real tokens, for a batch of left-padded rows decoded together
        (benchmark.batched_greedy); by default it is read from cache["key_valid"],
        which the caller sets in every layer's dict, so TinyGPT.forward needs no new
        argument. It is combined with the causal mask on the prompt and passed as the
        attention mask on later steps. A padding query has no real key before it, so
        it attends to itself (a row of -inf would be NaN); its output is never read.
        RoPE scores depend only on relative positions, so left padding leaves every
        real token's attention as it is without padding."""
        b, t, _ = x.shape
        past = cache["k"].size(-2) if "k" in cache else 0
        if key_valid is None:
            key_valid = cache.get("key_valid")
        q, k, v = self._qkv(x, rope, past)
        if past:
            k = torch.cat([cache["k"], k], dim=-2)
            v = torch.cat([cache["v"], v], dim=-2)
        cache["k"], cache["v"] = k, v  # kept with the n_kv_heads heads (GQA), before repeating
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        if key_valid is not None:
            keys = key_valid[:, None, None, :past + t].to(device=x.device, dtype=torch.bool)  # (B, 1, 1, past + t)
            if t == 1:  # one new (real) token sees every real token before it
                allowed = keys
            else:  # causal, real keys only, and every query its own position (no all-masked row)
                causal = torch.ones(t, past + t, dtype=torch.bool, device=x.device).tril(past)
                own = torch.zeros(t, past + t, dtype=torch.bool, device=x.device)
                own[torch.arange(t), past + torch.arange(t)] = True
                allowed = (causal & keys) | own
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        elif past == 0:  # the prompt: ordinary causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif t == 1:  # one new token sees everything before it
            y = F.scaled_dot_product_attention(q, k, v)
        else:  # several new tokens after a cached prefix
            allowed = torch.ones(t, past + t, dtype=torch.bool, device=x.device).tril(past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, -1))


class TransformerBlock(nn.Module):
    """One pre-norm block. A dense block returns the residual stream; a
    mixture-of-experts block (self.moe) returns (residual stream, aux) with the
    MoEFFN's (lb, z, counts) terms."""

    def __init__(self, cfg, index=0):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.dropout, cfg.qk_norm,
                                          cfg.norm_eps)
        self.moe = cfg.is_moe_block(index)
        self.ffn = MoEFFN(cfg) if self.moe else SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, rope, cache=None, mask=None):
        x = x + self.resid_dropout(self.attn(self.attn_norm(x), rope, cache, mask))
        if self.moe:
            y, aux = self.ffn(self.ffn_norm(x))
            return x + y, aux
        return x + self.ffn(self.ffn_norm(x))


def document_mask(tokens, sep_id):
    """Boolean (B, 1, T, T) attention mask for the windows `tokens` (B, T): position i
    may attend to position j when j <= i (causal) and both lie in the same document.
    Documents are delimited by `sep_id`, the BOS token data_prep puts at the start of
    every document, so doc_id = cumsum(tokens == sep_id) and the BOS belongs to the
    document it opens. A window without a separator is one document and the mask is
    the plain causal one. Memory: B x T x T bytes."""
    doc = (tokens == sep_id).cumsum(dim=1)
    same = doc.unsqueeze(-1) == doc.unsqueeze(-2)
    causal = torch.ones(tokens.size(1), tokens.size(1), dtype=torch.bool, device=tokens.device).tril()
    return (same & causal).unsqueeze(1)


class TinyGPT(nn.Module):
    """Module names match the original script, so its (legacy) weights load unchanged."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ctx = cfg.ctx
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.ctx, cfg.rope_base)
        self.blocks = nn.ModuleList(TransformerBlock(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            # every projection that writes into the residual stream: o_proj, the dense (and shared) SwiGLU's down,
            # and each expert's down ("...experts.down.<e>", a ParameterList entry)
            if name.endswith("o_proj.weight") or name.endswith("down.weight") or ".down." in name:
                nn.init.normal_(p, mean=0.0, std=resid_std)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        # Activation checkpointing: keep only each block's input during the forward pass and recompute the
        # block in the backward pass (about a third more compute, a fraction of the activation memory).
        self.grad_checkpoint = False
        # Document mask (--doc-mask on): the id of the token that starts every document (BOS); attention then
        # stays within each document of a window (document_mask). None: plain causal attention over the window.
        # A training-objective setting kept in train_config, not part of the architecture, so a checkpoint's
        # weights load whichever way it was trained.
        self.doc_sep_id = None

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def set_attention_impl(self, impl):
        for block in self.blocks:
            block.attn.impl = impl

    def set_moe_impl(self, impl):
        """How the experts run: "loop" or "grouped" (see Experts)."""
        for block in self.blocks:
            if block.moe:
                block.ffn.experts.impl = impl

    def moe_blocks(self):
        return [block for block in self.blocks if block.moe]

    @torch.no_grad()
    def update_router_bias(self, counts, rate):
        """DeepSeek-V3's auxiliary-loss-free balancing (Wang et al. 2024): after each
        optimizer step, every expert that received more than the mean number of tokens
        in a layer has its routing bias lowered by `rate`, every expert below the mean
        has it raised: bias -= rate x sign(c - mean(c)). counts: (n_moe_blocks, E), the
        tokens per expert over the step."""
        for block, c in zip(self.moe_blocks(), counts):
            c = c.to(block.ffn.router.bias.dtype)
            block.ffn.router.bias.sub_(rate * torch.sign(c - c.mean()))

    def new_cache(self):
        """An empty KV cache (one dict per layer) for forward(..., cache=...)."""
        return [{} for _ in self.blocks]

    @staticmethod
    def cache_length(cache):
        return cache[0]["k"].size(-2) if cache and "k" in cache[0] else 0

    def forward(self, tokens, return_hidden=False, cache=None, return_aux=False):
        """Logits, or (return_hidden=True) the final normalised hidden states, from
        which chunked_lm_loss computes the loss without all logits at once.
        cache (from new_cache): KV-cached decoding; `tokens` are then only the new
        tokens, after those already in the cache (at most ctx in total).
        return_aux=True returns (output, aux) instead, aux the mixture-of-experts
        terms summed over the MoE blocks, {"lb", "z", "counts" (n_moe_blocks, E)},
        or None for a dense model; without it the output is the single tensor it
        always was (the experts act per token, so the KV cache needs nothing)."""
        if cache is not None and self.cache_length(cache) + tokens.size(1) > self.ctx:
            raise ValueError(f"KV cache full: {self.cache_length(cache)} + {tokens.size(1)} tokens > ctx {self.ctx}")
        x = self.embedding_dropout(self.token_embedding(tokens))
        recompute = self.grad_checkpoint and self.training and torch.is_grad_enabled() and cache is None
        # the document mask applies to whole windows (training, evaluation, scoring); KV-cached decoding
        # continues one text and keeps plain causal attention
        mask = document_mask(tokens, self.doc_sep_id) if self.doc_sep_id is not None and cache is None else None
        lb = z = None
        counts = []
        for i, block in enumerate(self.blocks):
            if recompute:
                x = torch.utils.checkpoint.checkpoint(block, x, self.rope, None, mask, use_reentrant=False)
            elif cache is not None:
                x = block(x, self.rope, cache[i])
            else:
                x = block(x, self.rope, None, mask)
            if block.moe:
                x, (b_lb, b_z, b_counts) = x
                lb = b_lb if lb is None else lb + b_lb
                z = b_z if z is None else z + b_z
                counts.append(b_counts)
        x = self.norm(x)
        out = x if return_hidden else self.lm_head(x)
        if not return_aux:
            return out
        return out, ({"lb": lb, "z": z, "counts": torch.stack(counts)} if counts else None)

    def param_counts(self):
        """(total, non-embedding total, active per token, non-embedding active): the
        same four numbers as count_params(cfg), from the built model."""
        total = sum(p.numel() for p in self.parameters())
        embed = self.token_embedding.weight.numel()
        if not self.cfg.tie_embeddings:
            embed += self.lm_head.weight.numel()
        counted = count_params(self.cfg)
        idle = counted[0] - counted[2]  # the experts a token does not use (0 for a dense model)
        return total, total - embed, total - idle, total - idle - embed


LOSS_CHUNK_TOKENS = 4096  # tokens per piece when the loss is computed in pieces
# mixture-of-experts training terms (see MoEFFN and the training loop): the Switch load-balancing loss weight
# (Fedus et al. 2022 use 0.01), the router z-loss weight (Zoph et al. 2022 use 1e-3) and the per-step movement of
# DeepSeek-V3's balancing bias (Wang et al. 2024 use 1e-3)
LB_LOSS_DEFAULT = 0.01
ROUTER_Z_LOSS_DEFAULT = 1e-3
BIAS_RATE_DEFAULT = 1e-3
# per-step decay of the weight average (--ema 0.999, or --ema on). Opt-in: the 2026-11-05 A/B (d 384, 500 steps)
# found no gain from it, and new runs train without one unless --ema is given
EMA_DECAY_DEFAULT = 0.999
EMA_EVERY = 10  # steps between EMA updates; the per-step decay is compounded over them (0.999^10 ~ 0.990)
# FP16 updates the GradScaler may skip in a row (overflowing gradients) before the run counts as diverged; a fresh
# run's scale (65,536) settles within a handful of skips, so 20 in a row means no update is getting through
MAX_OVERFLOW_SKIPS = 20


def _chunk_loss(h, weight, y):
    logits = F.linear(h, weight).float()
    lse = torch.logsumexp(logits, dim=-1)
    valid = y != -100
    target = logits.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return ((lse - target) * valid).sum(), (lse.pow(2) * valid).sum()


def full_lm_loss(logits, y):
    """(mean cross-entropy, log Z per position) of the full FP32 logits table
    `logits` (tokens x vocabulary) against the labels `y` (tokens). One logsumexp
    serves both the loss and the z-loss: cross-entropy is log Z - logit[target]
    (what F.cross_entropy computes), so a second logsumexp for the z-loss term
    would redo the most expensive reduction of the step. Every label counts
    (training windows never carry -100)."""
    lse = torch.logsumexp(logits, dim=-1)
    target = logits.gather(-1, y.reshape(-1, 1)).squeeze(-1)
    return (lse - target).mean(), lse


def _chunk_logp(h, weight, y):
    logits = F.linear(h, weight).float()
    target = logits.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (target - torch.logsumexp(logits, dim=-1)) * (y != -100)


# The piece-wise loss function training calls: _chunk_loss itself (eager), or its compiled form once maybe_compile
# has compiled the model on CUDA (the compiled piece fuses the FP32 logits, logsumexp and gather, which lowers the
# peak memory by 0.45-1.25 GiB at the 216M shape at no cost in time). Eager stays the fallback everywhere else
# (compile off or failed, CPU, MPS until tested there).
_chunk_loss_fn = _chunk_loss


def set_chunk_loss_compiled(on):
    """Switch the piece-wise loss between its compiled and its eager form."""
    global _chunk_loss_fn
    _chunk_loss_fn = torch.compile(_chunk_loss, dynamic=False) if on else _chunk_loss
    return _chunk_loss_fn


def chunked_lm_loss(hidden, weight, targets, chunk_tokens=LOSS_CHUNK_TOKENS):
    """(sum of cross-entropy, sum of log Z^2, number of labelled tokens) of the
    output layer `weight` applied to `hidden`, computed `chunk_tokens` tokens at a
    time; each piece's logits are recomputed in the backward pass, so the
    (tokens x vocabulary) table never exists in full. Labels of -100 are ignored."""
    h = hidden.reshape(-1, hidden.size(-1))
    y = targets.reshape(-1)
    ce = z = None
    for s in range(0, y.numel(), chunk_tokens):
        c, zc = torch.utils.checkpoint.checkpoint(_chunk_loss_fn, h[s:s + chunk_tokens], weight,
                                                  y[s:s + chunk_tokens], use_reentrant=False)
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
    """(total, non-embedding total, active per token, non-embedding active)
    without building the model (unique, tied counted once). Active = the
    parameters a token's forward pass touches: in a mixture-of-experts block
    experts_per_token of the n_experts experts, so active = total - n_moe_blocks x
    3 d h_e (E - k); for a dense model active equals total. [0] is the total, as
    every caller that indexes the result expects. The planner's compute rules
    (FLOPs, tokens per step, learning rate, the tokens-per-parameter fit) take the
    active count; memory and the parameter column take the total."""
    d, h = cfg.d_model, cfg.ffn_hidden
    kv = cfg.n_kv_heads * cfg.head_dim
    per_layer = d * d * 2 + d * kv * 2 + 2 * d  # attention and the two norms
    if cfg.qk_norm:
        per_layer += 2 * cfg.head_dim  # the query and key norm gains, shared across heads
    n_moe = cfg.n_moe_blocks
    e, k, h_e, shared = cfg.n_experts, cfg.experts_per_token, cfg.expert_hidden, cfg.shared_hidden
    moe_ffn = e * 3 * d * h_e + e * d + 3 * d * shared if n_moe else 0  # experts, router, shared SwiGLU
    body = per_layer * cfg.n_layers + 3 * d * h * (cfg.n_layers - n_moe) + moe_ffn * n_moe + d
    embed = cfg.vocab_size * d * (1 if cfg.tie_embeddings else 2)
    idle = n_moe * 3 * d * h_e * (e - k) if n_moe else 0
    return body + embed, body, body + embed - idle, body - idle


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
            warmup = "no warmup" if self.warmup == 0 else f"warmup 1-{self.warmup} (linear)"
            decay = ("no decay" if self.decay_start >= self.total
                     else f"decay {self.decay_start + 1}-{self.total} ({self.shape}) to {self.min_lr:.2e}")
            return f"WSD: {warmup}, stable to {self.decay_start} at {self.peak:.2e}, {decay}"
        warmup = "no warmup" if self.warmup == 0 else f"warmup 1-{self.warmup}"
        return f"cosine: {warmup}, cosine to {self.min_lr:.2e} at {self.total}"


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def safe_load(path, trust=False, mmap=False):
    """torch.load with weights_only=True (no arbitrary Python objects).

    mmap=True maps the file instead of reading it: tensors are read from disk only
    when used. Ignored on Windows, where a mapped file cannot be replaced, so a
    training run's atomic_save could not overwrite a checkpoint held open here;
    files that cannot be mapped (the old non-zip format) are read normally."""
    if not os.path.isfile(path):
        raise SystemExit(f"Checkpoint not found: {path}")
    if mmap and os.name != "nt":
        try:
            return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except Exception:  # noqa: BLE001 - old (non-zip) files, or ones that need --trust-checkpoint: below
            pass
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


def config_from_checkpoint(obj, ctx=None):
    """The model's configuration; ctx replaces the stored context length (the
    two-stage long-context recipe: --init-from <best> --ctx 4096). No weight
    depends on ctx: the RoPE tables are non-persistent buffers rebuilt from
    cfg.ctx, so the checkpoint's tensors load unchanged."""
    family, _ = checkpoint_info(obj)
    if family == "tinygpt":
        try:
            cfg = ModelConfig(**obj["model_config"])
        except TypeError as exc:  # a field this version of the script has never heard of
            known = {f.name for f in dataclasses.fields(ModelConfig)}
            unknown = sorted(set(obj["model_config"]) - known)
            version = int(obj.get("format_version", 0) or 0)
            raise SystemExit(
                f"This checkpoint's model settings use fields this tiny_gpt.py does not know: {unknown} "
                f"(checkpoint format_version {version}, this script reads up to {FORMAT_VERSION}"
                + ("; written by a newer tiny_gpt.py" if version > FORMAT_VERSION else "")
                + "). Update tiny_gpt.py to open it.") from exc
    elif family in ("legacy-v3", "legacy-v2"):
        s = obj["settings"]
        weights = obj["model"]
        tied = ("lm_head.weight" not in weights or
                torch.equal(weights["lm_head.weight"], weights["token_embedding.weight"]))
        cfg = ModelConfig(vocab_size=int(obj.get("vocab_size") or weights["token_embedding.weight"].shape[0]),
                          ctx=int(s["CTX"]), d_model=int(s["D"]), n_layers=int(s["LAYERS"]),
                          n_heads=int(s["HEADS"]), n_kv_heads=int(s.get("KV_HEADS", s["HEADS"])),
                          ffn_hidden=(8 * int(s["D"])) // 3, rope_base=float(s.get("ROPE_BASE", 10_000.0)),
                          dropout=float(s.get("DROPOUT", 0.0)), tie_embeddings=tied)
    else:
        raise SystemExit("Unrecognised checkpoint layout (neither tinyGPT nor the original script). "
                         "view_pt.py can still list its tensors.")
    if ctx:
        cfg.ctx = int(ctx)
        cfg.validate()
    return cfg


def tokenizer_from_checkpoint(obj):
    family, _ = checkpoint_info(obj)
    if family == "tinygpt":
        meta = obj["tokenizer"]
        return data_prep.Tokenizer(meta["proto"], meta.get("encode_mode", "lines"))
    if obj.get("tokenizer_proto"):
        return data_prep.Tokenizer(obj["tokenizer_proto"], "plain")
    raise SystemExit("This checkpoint has no tokenizer; it cannot be used for generation.")


def _model_assigned(obj, cfg, device):
    """The model built on the meta device and given the checkpoint's own tensors
    (load_state_dict(assign=True)): no random initialisation and no second copy of
    the weights. None when this PyTorch cannot do it or something stays unset."""
    try:
        with torch.device("meta"):
            model = TinyGPT(cfg)
        missing, unexpected = model.load_state_dict(obj["model"], strict=False, assign=True)
    except (TypeError, RuntimeError, NotImplementedError):  # older PyTorch
        return None
    missing = [m for m in missing if not (m == "lm_head.weight" and cfg.tie_embeddings)]
    if missing or unexpected:
        raise SystemExit(f"Checkpoint weights do not match the architecture "
                         f"(missing {missing[:4]}, unexpected {unexpected[:4]}).")
    if cfg.tie_embeddings:
        model.lm_head.weight = model.token_embedding.weight  # assign=True replaced the shared parameter
    model.rope = RotaryEmbedding(cfg.head_dim, cfg.ctx, cfg.rope_base)  # its buffers are not in the file
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        return None
    return model.float().to(device).eval()  # float32 as below (no copy if already float32 on the CPU)


def model_from_checkpoint(obj, device="cpu", assign=False):
    """(model in eval mode, tokenizer, config) from an already loaded checkpoint (any format).

    assign=True: the model holds the checkpoint's own tensors instead of copies
    (less memory, no random initialisation); changing the model then changes
    obj["model"] too, so use it only when obj is not saved again."""
    cfg = config_from_checkpoint(obj)
    cfg.dropout = 0.0
    if assign:
        model = _model_assigned(obj, cfg, device)
        if model is not None:
            tok = tokenizer_from_checkpoint(obj)
            apply_doc_mask(model, obj, tok)
            return model, tok, cfg
    model = TinyGPT(cfg)
    missing, unexpected = model.load_state_dict(obj["model"], strict=False)
    missing = [m for m in missing if not (m == "lm_head.weight" and cfg.tie_embeddings)]
    if missing or unexpected:
        raise SystemExit(f"Checkpoint weights do not match the architecture "
                         f"(missing {missing[:4]}, unexpected {unexpected[:4]}).")
    model.to(device).eval()
    tok = tokenizer_from_checkpoint(obj)
    apply_doc_mask(model, obj, tok)
    return model, tok, cfg


def apply_doc_mask(model, obj, tok):
    """Give a model built from checkpoint `obj` the document mask its run trained with
    (train_config "doc_mask", --doc-mask on): attention again stays within each document
    of a window, so scoring outside training measures what training measured. Off for
    every other checkpoint and for a tokenizer without a BOS id. KV-cached decoding
    takes no mask, so generation is unchanged."""
    bos = getattr(tok, "bos_id", None) if (obj.get("train_config") or {}).get("doc_mask") else None
    model.doc_sep_id = bos if bos is not None and bos >= 0 else None


def load_for_inference(path, device="cpu", trust=False):
    """(model, tokenizer, config, raw checkpoint dict) for a tinyGPT or legacy
    checkpoint file, or an --export folder.

    The model is given the checkpoint's own tensors (assign=True) and the
    returned dict no longer holds the weights ("model" is removed): one copy of
    the weights in memory instead of two (at 216M 825 MiB and 2.6 s per load
    instead of 1,649 MiB and 4.6 s; at 3B 12 GB instead of 24). Everything else
    in the dict (metrics, dataset, train_config, tokenizer...) is still there;
    a caller that needs the weights saves model.state_dict()."""
    obj = load_any(path, trust)
    model, tok, cfg = model_from_checkpoint(obj, device, assign=True)
    obj.pop("model", None)
    return model, tok, cfg, obj


def dataset_fingerprint(ds_path):
    """The fingerprint in a dataset folder's manifest (None if it has no readable manifest)."""
    try:
        return data_prep.read_manifest(ds_path).get("fingerprint")
    except (SystemExit, OSError, ValueError, AttributeError):
        return None


def resolve_dataset(obj, override=None):
    """The dataset folder a checkpoint was trained on, on this machine.

    override (a --dataset option) wins as given. Otherwise the path stored in
    the checkpoint, when that folder exists and its manifest has the stored
    fingerprint (a relative path is tried from the current folder and from this
    script's folder); else datasets/<same name> in this repository with the same
    fingerprint (the checkpoint came from another machine, where the path was
    C:\\... or /Users/...). A folder with another fingerprint is never used: a
    rebuilt dataset's validation split may hold text the model trained on.
    SystemExit asking for --dataset when nothing matches."""
    if override:
        return override
    stored = obj.get("dataset") if isinstance(obj, dict) else None
    stored = stored if isinstance(stored, dict) else {}
    path, want = stored.get("path"), stored.get("fingerprint")
    if not path:
        raise SystemExit("This checkpoint records no dataset: pass --dataset <folder>.")
    name = re.split(r"[\\/]", str(path).rstrip("\\/"))[-1]
    candidates = [path] + ([] if os.path.isabs(path) else [os.path.join(HERE, path)])
    candidates.append(os.path.join(HERE, "datasets", name))
    rebuilt = None
    for i, cand in enumerate(candidates):
        by_name = i == len(candidates) - 1
        if not os.path.isdir(cand):
            continue
        found = dataset_fingerprint(cand)
        if found is None:  # not a finished dataset
            continue
        if found == want or (not want and not by_name):
            return cand
        if not want:
            # an older checkpoint without a stored fingerprint is trusted only at its own path: a folder found by
            # name alone cannot be checked, and a rebuild's validation split may hold this model's training text
            raise SystemExit(f"The checkpoint records no dataset fingerprint, so datasets/{name} cannot be checked "
                             "to be the one it was trained on: pass --dataset to use it.")
        rebuilt = rebuilt or cand
    if rebuilt:
        # names only, no folders: view_pt shows this message on dashboards that may be published
        raise SystemExit(f"The dataset now at {os.path.basename(os.path.normpath(rebuilt))} is not the one this model "
                         "was trained on (different fingerprint), so its validation text may contain training text; "
                         "pass --dataset to use it anyway.")
    raise SystemExit(f"The checkpoint's dataset {name} is not on this machine (neither at its stored path nor in "
                     "datasets/): pass --dataset <folder>.")


def cpu_state_dict(model):
    """State dict on CPU; tensors that share storage (tied embeddings) stay shared."""
    out, seen = {}, {}
    for key, value in model.state_dict().items():
        ident = (value.data_ptr(), tuple(value.shape), value.dtype)
        if ident not in seen:
            seen[ident] = value.detach().to("cpu", copy=True)
        out[key] = seen[ident]
    return out


def resolve_ema(settings, requested, resumed):
    """The EMA decay this session trains with (0.0 = off), written into the settings.
    A new run keeps no average unless --ema asks for one (--ema 0.999 or --ema on):
    the 2026-11-05 A/B at d 384 (500 steps) found no gain (the raw weights won every
    evaluation), so the average is an opt-in for long runs, where its stable-phase
    gain is still to be measured. A resumed run keeps what its checkpoint says, and
    a checkpoint from before the key existed says off: a run never gains an average
    silently. --ema on a resume overrides: a decay replaces the run's decay and its
    saved average continues (a run without one starts it from the current weights);
    off removes it."""
    if resumed:
        settings.setdefault("ema", 0.0)
        if requested is not None:
            settings["ema"] = float(requested)
    else:
        settings["ema"] = 0.0 if requested is None else float(requested)
    return float(settings["ema"] or 0.0)


def ema_init(items):
    """Fresh fp32 CPU copies of the tensors in `items` ((name, tensor) pairs, a
    state_dict's or a saved EMA's), tensors that share storage (tied embeddings)
    staying one tensor, so the average costs 4 bytes per unique parameter."""
    out, seen = {}, {}
    for key, value in items:
        ident = (value.data_ptr(), tuple(value.shape), value.dtype)
        if ident not in seen:
            v = value.detach()
            seen[ident] = v.to(device="cpu", dtype=torch.float32 if v.is_floating_point() else v.dtype, copy=True)
        out[key] = seen[ident]
    return out


def ema_decay_at(decay, n):
    """The decay of the n-th update (n = 1, 2, ...) of an average that starts at
    zero and is divided by its weight sum, as Adam debiases its moments (Kingma and
    Ba 2015): with m_n = d m_(n-1) + (1 - d) w_n and m_0 = 0, the debiased average
    a_n = m_n / (1 - d^n) obeys a_n = b_n a_(n-1) + (1 - b_n) w_n with
    b_n = d (1 - d^(n-1)) / (1 - d^n). So b_1 = 0 (the first update copies the
    weights, whatever the average held before), b_2 = d / (1 + d) (two snapshots
    nearly equal), and b_n -> d once d^n is small (after ~460 updates at d = 0.990).
    n None: the plain decay d (an average restored from a checkpoint written before
    the update count was saved, which continues exactly as it did)."""
    if n is None:
        return decay
    if n <= 1:
        return 0.0
    return decay * (1.0 - decay ** (n - 1)) / (1.0 - decay ** n)


@torch.no_grad()
def ema_update(ema, model, decay):
    """ema <- decay x ema + (1 - decay) x w for every unique tensor (torch.lerp with
    weight 1 - decay), the weights read from the device once per update. Called every
    EMA_EVERY steps with the per-step decay compounded: d_update = d_step ** EMA_EVERY."""
    done = set()
    for key, value in model.state_dict().items():
        e = ema[key]
        if id(e) in done:
            continue
        done.add(id(e))
        w = value.detach().to(device="cpu", dtype=e.dtype)
        if e.is_floating_point():
            e.lerp_(w, 1.0 - decay)
        else:
            e.copy_(w)


def ema_weights_for_checkpoint(ema, bf16=False):
    """The EMA tensors as an inference-only checkpoint stores them: as they are (fp32,
    tied storage shared), or cast to bfloat16 for --best-bf16."""
    if not bf16:
        return ema
    out, seen = {}, {}
    for key, value in ema.items():
        if id(value) not in seen:
            seen[id(value)] = value.to(torch.bfloat16) if value.is_floating_point() else value
        out[key] = seen[id(value)]
    return out


class swapped_weights:
    """with swapped_weights(model, weights): the model carries `weights` (a state_dict
    such as the EMA) inside the block and its own weights again afterwards, copied in
    place (load_state_dict), so the parameter tensors the optimizer and a compiled
    graph hold stay the same objects. The raw weights wait on the CPU meanwhile.
    `hold` (a DeferInterrupt) keeps Ctrl+C out of the two copies: an interrupt halfway
    through one would leave a model that is part raw and part average, and the pause
    that follows would save it as the run's weights; the caller delivers it afterwards."""

    def __init__(self, model, weights, hold=None):
        self.model, self.weights, self.raw = model, weights, None
        self.hold = hold if hold is not None else contextlib.nullcontext()

    def __enter__(self):
        with self.hold:
            self.raw = cpu_state_dict(self.model)
            self.model.load_state_dict(self.weights)
        return self.model

    def __exit__(self, *exc):
        with self.hold:
            self.model.load_state_dict(self.raw)
        self.raw = None
        return False


def weights_for_checkpoint(model, device, bf16=False):
    """The model's weights as a checkpoint stores them. On CUDA and the CPU a CPU
    copy (cpu_state_dict, tied storage shared); on MPS the device tensors themselves:
    torch.save moves one storage at a time to the CPU and writes shared storage once,
    so unified memory never holds a second full copy of the state (26-39 GB at 3B).
    bf16=True (--best-bf16, inference-only copies): half-size bfloat16 weights, cast
    on the CPU for a CUDA model and on the device otherwise, tied weights still one tensor."""
    if not bf16 and device.type != "mps":
        return cpu_state_dict(model)
    out, seen = {}, {}
    for key, value in model.state_dict().items():
        ident = (value.data_ptr(), tuple(value.shape), value.dtype)
        if ident not in seen:
            v = value.detach()
            if bf16 and v.is_floating_point():
                # on CUDA straight to a CPU copy: a bf16 copy on the GPU would need memory the run may not have
                v = v.to("cpu" if device.type == "cuda" else v.device, torch.bfloat16)
            seen[ident] = v
        out[key] = seen[ident]
    return out


def optimizer_state_for_checkpoint(opt, device):
    """The optimizer's state dict as a checkpoint stores it: on MPS the device tensors
    (see weights_for_checkpoint), elsewhere a CPU copy."""
    state = opt.state_dict()
    return state if device.type == "mps" else to_cpu(state)


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def atomic_save(obj, path):
    """Write to a temp file in the same folder, flush it to the disk, then rename
    over the target (so a power cut right after a save cannot leave a renamed but
    unwritten file in place of the previous good one). Retries briefly if Windows
    reports the target as in use (e.g. open in the viewer); returns the path
    actually written, which differs from `path` only if it stayed locked."""
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "wb") as handle:
            torch.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
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


def last_dashboard_push(png_name):
    """Unix time of the last commit of dashboards/<png_name> in this script's repository; 0.0 when there is
    none (or no git, or no repository). A session starts its --dashboard-push-hours clock from it, so that a
    --resume does not push again at its first new best."""
    try:
        out = subprocess.run(["git", "-C", HERE, "log", "-1", "--format=%ct", "--", f"dashboards/{png_name}"],
                             capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else 0.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def launch_dashboard(path, args, state):
    """Draw view_pt.py's dashboard of the best checkpoint in the background (CPU,
    idle priority; training does not wait): after every evaluation, since
    view_pt.py adds the evaluations after the best file's last step from
    <name>_metrics.jsonl. Skipped while the previous one is still drawing.
    view_pt.py pushes it to GitHub (for checkpoints in this repository's
    folder) at most every --dashboard-push-hours."""
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
                                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, creationflags=flags)
            else:
                # through the nice command, not preexec_fn: code run between fork and exec can deadlock the
                # child of a process that has threads (the energy meter, torch)
                niced = (["nice", "-n", "19"] + cmd) if shutil.which("nice") else cmd
                state["proc"] = subprocess.Popen(niced, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
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
# name suffixes that stay on AdamW in a Muon run: the embedding and output layer (the split Muon is designed for)
# and the mixture-of-experts routers (a Muon update would orthogonalise the routing matrix; the router z-loss is
# the one control on its logit scale, so it trains like any other AdamW matrix, decayed)
MUON_EXCLUDED = ("token_embedding.weight", "lm_head.weight", "router.weight")
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
    muon = [(n, p) for n, p in named if use_muon and p.dim() == 2 and not n.endswith(MUON_EXCLUDED)]
    taken = {n for n, _ in muon}
    # an untied input embedding is a lookup table (each row trained only when its token occurs), so like the norm
    # gains it is not decayed; tied, the one matrix is also the output layer and keeps its decay
    tied = getattr(getattr(model, "cfg", None), "tie_embeddings", True)
    undecayed = set() if tied else {"token_embedding.weight"}
    decay = [(n, p) for n, p in named if p.dim() >= 2 and n not in taken and n not in undecayed]
    no_decay = [(n, p) for n, p in named if p.dim() < 2 or n in undecayed]
    groups = [{"params": [p for _, p in decay], "weight_decay": settings["weight_decay"]},
              {"params": [p for _, p in no_decay], "weight_decay": 0.0}]
    adamw, name = make_adamw(groups, device, lr=settings["lr"], betas=tuple(settings["betas"]))
    names = [[n for n, _ in decay], [n for n, _ in no_decay]]
    gains = f"{len(no_decay) - len(undecayed)} norm gains" + ("" if tied else " and the untied input embedding")
    if not use_muon:
        return adamw, name, names, f"weight decay on {len(decay)} matrices, not on {gains}"
    if not hasattr(torch.optim, "Muon"):
        raise SystemExit(f"--optimizer muon needs PyTorch 2.9 or newer (this is {torch.__version__}).")
    muon_opt = torch.optim.Muon([{"params": [p for _, p in muon], "weight_decay": settings["weight_decay"]}],
                                lr=settings["lr"], momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")
    return (CombinedOptimizer([adamw, muon_opt]), f"Muon + {name}", names + [[n for n, _ in muon]],
            f"Muon for {len(muon)} block matrices, its step scaled to AdamW's update size; {name} for the embedding "
            f"({len(decay)} matri{'x' if len(decay) == 1 else 'ces'}) and {gains}; weight decay "
            "on all matrices" + ("" if tied else " but the untied input embedding"))


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
        if not math.isfinite(weights[name]) or weights[name] < 0:
            raise SystemExit("--mix weights must be finite numbers >= 0")
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
        self.gpu_index, self.gpu_measured, self.device_type = None, False, device.type
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
        dt = now - self._last_t
        # a gap far longer than the period means the machine slept with the run alive: no power was drawn
        # then, so at most three periods (plus 15 s of slack) are counted. Known limit: this thread also asks
        # for the tariff (_price, at most one request per 10 minutes); a request that hangs for longer than
        # that with no evaluation or save sampling meanwhile loses the excess, although power was drawn
        hours = min(dt, 3 * self.period + 15) / 3600
        gpu_w = self._gpu_watts() if self.gpu_measured else None
        cpu_w = ram_w = 0.0
        if self._proc is not None:
            try:
                cpu_time = sum(self._proc.cpu_times()[:2])
                share = (cpu_time - self._cpu_time) / max(dt * self._ncpu, 1e-9)
                self._cpu_time = cpu_time
                cpu_w = self.cpu_watts * min(max(share, 0.0), 1.0)
                ram_w = RAM_W_PER_GB * self._proc.memory_info().rss / 2 ** 30
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            added = 0.0
            if gpu_w is not None:  # trapezoid between samples
                previous = self._last_gpu_w if self._last_gpu_w is not None else gpu_w
                gpu_wh = (previous + gpu_w) / 2 * hours
                added += gpu_wh
                self.wh["gpu"] += gpu_wh
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

    def short(self, co2e=True):
        """'11.25 kWh, 1,407 g CO2e, £2.98'; co2e=False leaves out the carbon (the energy x a fixed intensity)."""
        cost = self.cost_gbp()
        return (f"{self.kwh():.2f} kWh" + (f", {self.co2e_g():,.0f} g CO2e" if co2e else "")
                + (f", £{cost:.2f}" if cost is not None else ""))

    def describe(self):
        gpu = (f"GPU {self.gpu_index} measured with nvidia-smi every {self.period:g} s" if self.gpu_measured
               else "GPU not measured on this device")
        if not self.gpu_measured and self.device_type == "mps":
            gpu += (" (the chip draws most of the power; measure once with sudo powermetrics --samplers "
                    "cpu_power,gpu_power -i 1000 -n 10 and pass --cpu-watts <package W> for an estimate)")
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
        return input_and_target(t)

    def state_dict(self):
        return {"generator": self.gen.get_state()}

    def load_state_dict(self, state):
        if state and "generator" in state:
            self.gen.set_state(state["generator"].cpu() if isinstance(state["generator"], torch.Tensor)
                               else state["generator"])


def input_and_target(windows):
    """(x, y) of a (batch, ctx + 1) tensor of token windows: the first and the last ctx columns. Both are views,
    so their strides are (ctx + 1, 1). torch.compile guards on the strides of its input: the warm-up of
    maybe_compile cuts its batch here too, or the first training step would compile the model once more."""
    return windows[:, :-1], windows[:, 1:]


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
            raise SystemExit(f"No validation source has at least ctx+1 = {ctx + 1} tokens; "
                             "rebuild the dataset with a larger --val-fraction or lower --ctx.")


@torch.no_grad()
def evaluate(model, evalset, device, amp_dtype, batch_windows, bins=10, byte_tables=None, share=None):
    """Validation loss per source plus how 'opinionated' the model is.

    Returns mean loss per source; 'loss' is the mean over sources (balanced, so
    the small sources count as much as the large ones; it drives NEW BEST).
    With `share` (sampling probability per source, TokenBatcher.share()),
    'loss_mix' and 'bpb_mix' are the same means weighted as training samples
    the sources: that is the number to compare with the training loss, since
    the balanced mean differs from it by composition alone. Entropy is
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
    # MPS has no float64: there the tensors go to the CPU first
    to64 = (lambda t: t.cpu().double()) if device.type == "mps" else (lambda t: t.double())  # noqa: E731
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
            correct = to64(pred == y)
            nll_total += to64(nll).sum().item()
            tokens += y.numel()
            if byte_tables is not None:
                size, marked, line_break = byte_tables
                yc, xc = y.cpu(), x.cpu()
                n_bytes += int(size[yc].sum()) - int((marked[yc] & line_break[xc]).sum())
            sums["entropy"] += to64(entropy).sum().item()
            c = to64(conf)
            sums["confidence"] += c.sum().item()
            sums["top1"] += correct.sum().item()
            # a non-finite logit (NaN confidence) lands in bin 0: the loss then reads nan instead of bincount raising
            which = (c.nan_to_num(0.0).clamp(0, 1 - 1e-9) * bins).long().clamp(0, bins - 1).flatten().cpu()
            conf_sum += torch.bincount(which, c.flatten().cpu(), bins)
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

    def mix_mean(values):
        # sum_s w_s x_s / sum_s w_s over the sources present, w_s = training sampling share of source s
        if share is None or not values:
            return None
        w = {name: float(share.get(name, 0.0)) for name in values}
        return float(sum(values[name] * w[name] for name in values) / max(sum(w.values()), 1e-12))

    return {
        "loss": float(np.mean(list(per_source.values()))),
        "loss_mix": mix_mean(per_source),
        "per_source": per_source,
        "bpb": float(np.mean(list(bpb_source.values()))) if bpb_source else None,
        "bpb_mix": mix_mean(bpb_source),
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
            while parts and not parts[-1]:  # trailing tabs are not columns
                parts.pop()
            if len(parts) >= 5:
                category, difficulty, prompt, answer, options = parts[:5]
            elif len(parts) == 3:  # exactly three: four columns are a long-form line without its distractors
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
def score_probes(model, tok, probes, device, amp_dtype, batch_size=64, floor=True, template=None):
    """Cloze fact probes: is the correct continuation more probable (per
    character) than every distractor? A proxy for factual reliability, not a
    hallucination rate. All (probe, option) sequences are scored in right-padded
    batches; causal attention means padding never affects the scored positions.

    With floor=True each probe is also scored with its question's words shuffled
    (FLOOR_SHUFFLES orders): what topic words alone achieve. That share, not
    chance, is the realistic floor; accuracy above it needs the sentence itself.

    template (a format string with {prompt}, e.g. finetune.TEMPLATE) wraps each
    prompt after any shuffling, so only the question's words are shuffled and
    the template stays intact; results keep the unwrapped prompt."""
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
            text = probe["prompt"] if v == 0 else shuffled_prompt(probe["prompt"], v)
            context = prefix + tok.encode(template.format(prompt=text) if template else text)
            for j, cont in enumerate(options):
                ids = (context + cont)[-model.ctx - 1:]
                seqs.append((i, v, j, ids, len(cont), chars[j]))
    scores = [[[None] * (1 + len(p["distractors"])) for _ in range(versions)] for p in probes]
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start:start + batch_size]
        width = max(len(s[3]) for s in chunk) - 1
        x = torch.zeros((len(chunk), width), dtype=torch.long)
        rows, cols, targets = [], [], []
        for b, (_, _, _, ids, n_cont, _) in enumerate(chunk):
            x[b, :len(ids) - 1] = torch.tensor(ids[:-1])
            rows += [b] * n_cont
            cols += range(len(ids) - 1 - n_cont, len(ids) - 1)
            targets += ids[len(ids) - n_cont:]
        rows_t = torch.tensor(rows, device=device)  # the batch row of each scored position
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            if isinstance(model, TinyGPT):  # the output layer only at the scored positions, not everywhere
                hidden = model(x.to(device), return_hidden=True)
                picked = model.lm_head(hidden[rows_t, torch.tensor(cols, device=device)])
            else:
                logits = model(x.to(device))
                picked = logits[rows_t, torch.tensor(cols, device=device)]
        picked = picked.float()
        target_logp = (picked.gather(-1, torch.tensor(targets, device=device)[:, None]).squeeze(-1)
                       - torch.logsumexp(picked, dim=-1))  # only the scored positions, not the whole vocabulary
        sums = torch.zeros(len(chunk), device=device).index_add_(0, rows_t, target_logp).cpu()
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
        if len(_cache) >= max(256, 64 * 2 ** 20 // vocab_size):  # at most about 64 MiB of masks (1 byte per token)
            _cache.clear()
        _cache[k] = mask
    return _cache[k]


class IncrementalDecoder:
    """Next-token logits for a growing token list, with a KV cache for TinyGPT models:
    the prompt is processed once, then one token per step. Texts longer than the
    context are approximated by restarting the cache from the last 7/8 window;
    non-TinyGPT models (wrappers, legacy) are recomputed in full each step."""

    def __init__(self, model, device, amp_dtype=None):
        self.model, self.device, self.amp_dtype = model, device, amp_dtype
        self.cached = isinstance(model, TinyGPT)
        self.cache, self.base, self.n = None, 0, 0

    @torch.no_grad()
    def next_logits(self, ids):
        """Logits (float, CPU) for the token after `ids`; ids must only ever grow by appending."""
        model, ctx = self.model, self.model.ctx
        with torch.autocast(self.device.type, dtype=self.amp_dtype or torch.float32,
                            enabled=self.amp_dtype is not None):
            if not self.cached:
                return model(torch.tensor([ids[-ctx:]], device=self.device))[0, -1].float().cpu()
            if self.cache is None or len(ids) - self.base > ctx or len(ids) < self.n:
                keep = ctx if self.cache is None else ctx - max(1, ctx // 8)
                self.base, self.cache = max(0, len(ids) - keep), model.new_cache()
                new = ids[self.base:]
            else:
                new = ids[self.n:]
            self.n = len(ids)
            h = model(torch.tensor([new], device=self.device), return_hidden=True, cache=self.cache)
            return model.lm_head(h[:, -1])[0].float().cpu()


def sample_next(logits_2d, generator, temperature, top_k, top_p):
    """One sampled token id per row of logits_2d (B, V): the logits divided by the temperature, cut to the
    top_k largest (0 = no cut) and then to the smallest set whose probability reaches top_p (nucleus sampling;
    outside 0..1 = no cut), and one draw per row from `generator`. A temperature below 1e-6 counts as 1e-6
    (as good as greedy), so the division cannot overflow; generate() decodes greedily at 0 and below."""
    logits = logits_2d.float() / max(temperature, 1e-6)
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
    decoder = IncrementalDecoder(model, device, amp_dtype)
    for _ in range(max_new_tokens):
        logits = decoder.next_logits(ids)
        if watermark:
            key, gamma, delta = watermark
            logits = logits + delta * watermark_green(key, ids[-1], logits.numel(), gamma).float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            nxt = int(sample_next(logits[None], generator, temperature, top_k, top_p)[0])
        if nxt == tok.eos_id:
            break
        ids.append(nxt)
    if was_training:
        model.train()
    return tok.decode(ids[start:])


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
# LR_TABLE_FACTOR x the published rates: the sweep behind the factor is in heuristic_lr's docstring
LR_TABLE_FACTOR = 2.0
LR_TABLE = [(1e6, 4.0e-3), (1e7, 2.4e-3), (1e8, 1.2e-3), (1e9, 6.0e-4)]
# the published 0.1 divided by the same factor, so lr x weight decay (the shrink per step) stays as published
WEIGHT_DECAY_DEFAULT = 0.1 / LR_TABLE_FACTOR


def heuristic_lr(n_params):
    """Log-log interpolation of peak learning rates: LR_TABLE_FACTOR x the AdamW
    rates used for published decoder models (GPT-3 125M 6e-4, Pythia-70M 1e-3,
    160M 6e-4, Pythia-410M 3e-4, GPT-3 760M 2.5e-4), extended downward for small
    models; the slope is the published one. The table assumes the 0.5-2M tokens
    per step those runs used; lr_batch_factor moves the rate when the planner's
    batch differs from the old 16 sqrt(N) rule. A starting point, not an optimum.

    The factor 2: a sweep on 2026-11-05 (Muon, QK-norm, 65,536 tokens per step,
    seed 1, WSD with 20 % decay to 0; bits per byte on the balanced validation
    set after the decay). d 384 (20.5M parameters, 500 steps): published rates x1
    1.5735, x2 1.5138 (with weight decay 0.05), x4 1.4950 (0.025). d 512 (36.6M,
    400 steps): x2 1.5804, x4 1.5337. Both sizes put the short-run optimum at or
    above x4; a 500-step optimum over-states the rate of a run 60x longer by about
    1.7x (DeepSeek LLM's C^-0.125 compute term), 4 / 1.7 = 2.35, so the measured
    arm below that, 2, is applied. x1 was not run at d 512 (time cap) and AdamW
    alone was not swept. Runs planned before the sweep keep their recorded rate
    (--resume never re-plans the rate)."""
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


def default_ctx(d_model):
    """Context length by width: the one rule plan_run, plan_table and
    estimate_dataset share. 256 up to d 192, 512 up to d 512, 1024 up to d 1024
    (the laptop plan), 2048 up to d 2048 (the 1.45B Mac shape), 4096 above.
    Attention is 12 x layers x ctx x d FLOPs per token on top of 6 N, so a longer
    context costs a growing share (attention_flop_share); --ctx overrides."""
    if d_model <= 192:
        return 256
    if d_model <= 512:
        return 512
    if d_model <= 1024:
        return 1024
    return 2048 if d_model <= 2048 else 4096


def default_rope_base(ctx):
    """RoPE base for a context: 10,000 up to ctx 2048, then growing as ctx^2
    (10,000 x (ctx / 2048)^2 rounded to thousands: 4k 40k, 8k 160k, 16k 640k).
    Against the lower bound of Men et al. 2024 ("Base of RoPE bounds context
    length", Table 2: 1k 4.3e3, 2k 1.6e4, 4k 2.7e4, 8k 8.4e4, 16k 3.1e5) the
    default is at or above it at every length except ctx 2048, where 10,000 is
    below the 1.6e4 bound. Below the bound the model keeps its perplexity but
    loses long-range retrieval."""
    if ctx <= 2048:
        return 10_000.0
    return float(round(10_000 * (ctx / 2048) ** 2, -3))


def attention_flop_share(cfg):
    """Share of the training FLOPs per token spent on attention scores:
    12 L ctx d / (6 N + 12 L ctx d) (see flops_per_token)."""
    return 12 * cfg.n_layers * cfg.ctx * cfg.d_model / flops_per_token(cfg)


def default_head_dim(d_model):
    return 32 if d_model <= 256 else (64 if d_model <= 1024 else 128)


def default_kv_heads(n_heads):
    """GQA with 4 query heads per K/V head once there are >= 8 heads, when the
    head count is divisible by 4 (otherwise ordinary multi-head attention)."""
    return n_heads // 4 if n_heads >= 8 and n_heads % 4 == 0 else n_heads


def shape_ladder(vocab_size, ctx=512):
    """Candidate shapes, depth growing with width (small models favour depth)."""
    out = []
    for d in (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 2048, 2560, 3072):
        heads = d // default_head_dim(d)
        out.append(ModelConfig(vocab_size=vocab_size, ctx=ctx, d_model=d, n_layers=default_layers(d),
                               n_heads=heads, n_kv_heads=default_kv_heads(heads)))
    return out


def tokens_per_step_legacy(n_params):
    """The reference batch the learning-rate table was calibrated at (the old
    16 sqrt(N) rule): ~16 sqrt(N) tokens per optimizer step, rounded to a power
    of two between 16k and 512k (lr_batch_factor)."""
    raw = 16 * math.sqrt(n_params)
    return int(min(max(2 ** round(math.log2(raw)), 16_384), 524_288))


def tokens_per_step_for(n_params, planned_tokens):
    """Tokens per optimizer step from the compute budget C = 6 N D:
    half of DeepSeek's (Bi et al. 2024) fitted batch B_opt = 0.292 C^0.327
    (the 0.5 keeps the batch below B_opt, where per-token efficiency is best),
    rounded to a power of two between 16k and 2M. 262,144 at 216M / 16.2B
    (the old 16 sqrt(N) rule's value), 524,288 at 1.45B / 29B, 1,048,576 at
    3.2B / 64B."""
    compute = 6.0 * max(n_params, 1) * max(planned_tokens, 1)
    raw = 0.5 * 0.292 * compute ** 0.327
    return int(min(max(2 ** round(math.log2(raw)), 2 ** 14), 2 ** 21))


def lr_batch_factor(n_params, tokens_per_step):
    """Factor on heuristic_lr when the planned batch differs from the batch its
    table was calibrated against (tokens_per_step_legacy): sqrt of the ratio
    (the square-root rule of Malladi et al. 2022 for Adam), clamped to
    [0.7, 1.4]. The ratio is rounded to a power of two first, so the
    accumulation rounding of a batch (270,336 against 262,144) or a small
    --tokens-per-step tweak leaves the rate exactly where it was; only a
    doubled or halved batch moves it. Muon follows through match_rms_adamw."""
    ratio = tokens_per_step / tokens_per_step_legacy(n_params)
    rounded = 2.0 ** round(math.log2(ratio))
    return min(max(math.sqrt(rounded), 0.7), 1.4)


def resolve_min_lr_ratio(ratio, schedule):
    """--min-lr-ratio unset: WSD decays to zero (Hagele et al. 2024 and MiniCPM end
    their decay at ~0; stopping at 0.1 x peak leaves 0.005-0.03 nats on the
    table), cosine keeps the customary one tenth of the peak."""
    if ratio is not None:
        return float(ratio)
    return 0.0 if schedule == "wsd" else 0.1


def resume_min_lr(settings, step, ratio):
    """--min-lr-ratio on --resume, mirroring the --steps override: before the
    decay phase the end point of the schedule may still change (lr_at() does
    not depend on min_lr until decay_start, so nothing moves until then);
    inside it the change would bend the running decay, so it is refused.
    Returns (applied, note); applied=True means settings['min_lr'] changed."""
    schedule = LRSchedule(settings["schedule"], settings["steps"], settings["lr"], settings["min_lr"],
                          settings["warmup_steps"], settings["decay_frac"], settings["decay_shape"])
    new_min = float(settings["lr"]) * float(ratio)
    if new_min == float(settings["min_lr"]):
        return False, f"Note: --min-lr-ratio {ratio:g}: this run's min_lr is already {new_min:.2e}."
    if schedule.phase(int(step)) == "decay":
        return False, (f"Note: --min-lr-ratio {ratio:g} ignored (already in the decay phase); the schedule keeps "
                       f"min_lr {settings['min_lr']:.2e}.")
    settings["min_lr"] = new_min
    return True, (f"Note: --min-lr-ratio {ratio:g}: min_lr {new_min:.2e} from the decay phase on; nothing changes "
                  "before it.")


def default_eval_tokens(eval_every, tokens_per_step, n_sources, available):
    """Fixed validation tokens per source for a new run: about 0.5 % of the
    tokens trained between evaluations, shared by the sources and rounded to
    a power of two, at least 65,536 (today's value up to ~1M tokens/step) and
    at most the tokens the largest validation source has (EvalSet caps the
    smaller ones itself)."""
    raw = 0.005 * eval_every * tokens_per_step / max(n_sources, 1)
    want = 2 ** round(math.log2(max(raw, 1)))
    return int(max(65_536, min(want, max(int(available), 65_536))))


def default_eval_batch(micro, ctx, vocab_size, budget_bytes):
    """Windows per evaluation forward pass, independent of the training
    micro-batch (no gradients are kept): evaluate() holds three fp32 copies of
    the logits (logits.float(), log-probabilities, probabilities), so 5 % of the
    memory budget pays for budget x 0.05 / (ctx x vocab x 4 x 3) windows, at most
    16 and never fewer than the micro-batch (which is known to fit with its
    gradients)."""
    fits = int(0.05 * budget_bytes) // (ctx * vocab_size * 4 * 3)
    return max(int(micro), min(16, fits))


def mix_weights(text, sources, sizes, alpha=1.0):
    """TokenBatcher weights from --mix and --mix-alpha. TokenBatcher samples a
    source with probability p_i ~ w_i x n_i (tokens), so passing w_i x n_i^(alpha-1)
    gives p_i ~ w_i x n_i^alpha: alpha 1 (default) is proportional sampling,
    alpha 0 ignores the sizes, values in between flatten a mix one source
    dominates (temperature sampling, as in multilingual training)."""
    weights = parse_mix(text, sources)
    if alpha == 1.0:
        return weights
    scaled = {}
    for name, w in weights.items():
        n = max(int(sizes.get(name, 0)), 1)
        scaled[name] = w * n ** (alpha - 1.0)
    return scaled


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
        # Metal's recommended working set (about 75 % of the unified memory on Apple silicon), and 75 % of that as
        # the planner's budget rather than CUDA's 80 %: the same memory also holds the memory-mapped token files
        # (8 GB now, 25-40 GB for the next dataset) and the OS. cap_device_memory makes it a hard cap, so an
        # overshoot fails fast instead of paging to the SSD. Raise to 0.8 once a Mac run's peak_mem_gib confirms
        # the estimate.
        rec = None
        fn = getattr(torch.mps, "recommended_max_memory", None)
        if fn is not None:
            try:
                rec = int(fn())
            except RuntimeError:
                rec = None
        if not rec:
            rec = avail
        return int(rec * 0.75), (f"{rec / 2 ** 30:.1f} GiB recommended working set (Metal), budget 75 % = "
                                 f"{rec * 0.75 / 2 ** 30:.1f} GiB")
    return int(avail * 0.4), f"{avail / 2 ** 30:.1f} GiB free RAM"


_PSUTIL_PROC = None


def peak_memory_bytes(device, running=0):
    """The larger of `running` and this process's peak memory on `device` so far:
    the CUDA allocator's high-water mark of allocated tensors (since the last
    torch.cuda.reset_peak_memory_stats), the MPS driver's current allocation (an
    upper bound that includes the allocator's cache; the caller keeps the running
    maximum by sampling after each step), or the resident set size on the CPU."""
    global _PSUTIL_PROC
    if device.type == "cuda":
        return max(running, int(torch.cuda.max_memory_allocated(device)))
    if device.type == "mps":
        fn = getattr(torch.mps, "driver_allocated_memory", None)
        return max(running, int(fn()) if fn else 0)
    try:
        if _PSUTIL_PROC is None:
            import psutil
            _PSUTIL_PROC = psutil.Process()
        return max(running, int(_PSUTIL_PROC.memory_info().rss))
    except ImportError:
        return running


_ATTN_SCORES_CACHE = {}


def _attention_growth_bytes(device):
    """Bytes the autograd graph of one causal SDPA call keeps on `device` for the
    backward pass: q, k, v of shape (1, 4, 1024, 64) in bfloat16 with gradients,
    measured as the device's allocated memory after the forward minus before it
    (the saved tensors are freed again by the backward, so it is measured here)."""
    allocated = torch.mps.current_allocated_memory if device.type == "mps" else torch.cuda.memory_allocated
    q, k, v = (torch.randn(1, 4, 1024, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3))
    synchronize(device)
    before = int(allocated())
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    synchronize(device)
    growth = int(allocated()) - before
    y.square().mean().backward()  # the path training takes must work as well
    return growth


def attention_keeps_scores(device):
    """Whether training attention on this device keeps the (B, H, T, T) score and
    softmax tensors for the backward pass (the composite "math" path) instead of a
    fused kernel that recomputes them (flash attention on CUDA, where it keeps only
    the output and a logsumexp per row). Decided by a measurement, not by the device
    type: on MPS the fused kernel is used only without gradients, so the probe sees
    the saved (1, 4, 1024, 1024) softmax (8 MiB in bf16, 16 MiB in fp32) against the
    1 MiB of output and logsumexp a fused kernel keeps; the threshold is 4 MiB. CUDA
    and CPU: False (their planner estimates are calibrated as they are). Cached per
    device type."""
    if device.type != "mps":
        return False
    if device.type not in _ATTN_SCORES_CACHE:
        try:
            _ATTN_SCORES_CACHE[device.type] = _attention_growth_bytes(device) > 4 * 2 ** 20
        except Exception:  # noqa: BLE001 - a backend that cannot run the probe: assume the cheaper kernel
            _ATTN_SCORES_CACHE[device.type] = False
        free_device_memory(device)
    return _ATTN_SCORES_CACHE[device.type]


def estimate_train_bytes(cfg, micro_batch, amp=True, checkpointing=False, loss_chunk=0, optimizer="adamw",
                         device=None, doc_mask=False):
    """Training memory: FP32 weights and gradients (4 + 4 bytes per parameter),
    the half-precision weight copies autocast keeps during a step (2), optimizer
    state (AdamW 8; Muon 4 for the block matrices), activations kept for the
    backward pass (only each block's FP32 input with activation checkpointing,
    plus one block being recomputed) and the output-layer logits (one piece of
    loss_chunk tokens when the loss is computed in pieces). The reference is the
    CUDA allocator's ALLOCATED peak (peak_memory_bytes); the 0.8 budget factor is
    the margin and the first-step out-of-memory handler the safety net, so no
    further factor is applied here. With `device` given and
    attention_keeps_scores(device) true (MPS), each layer also keeps its causal
    scores: n_heads x ctx x 6 bytes per token (the bf16 cast copy, 2, and the fp32
    softmax output, 4); with checkpointing the `+ t * per_token_layer` term already
    charges them for the one block being recomputed. doc_mask adds the boolean
    (B, 1, T, T) document mask, one byte per entry. In a mixture-of-experts block
    the FFN keeps activations for the experts_per_token experts a token visits
    (k x h_e + the shared hidden instead of ffn_hidden) plus the routing: the
    gathered inputs and the expert outputs (2 k d b), the fp32 scores, logits and
    probabilities (3 E x 4) and the int64 ids and sort order (2 k x 8); weights,
    gradients, the half-precision copies and the optimizer state count every
    expert (the TOTAL parameter count), the experts in the Muon body. Checked on
    the RTX 3070 (2026-11-05, allocated peak / estimate): d 768 E 8 k 2 with
    checkpointing at micro 4, 5.02 / 5.3 GiB; dense at the same shape 4.37 / 4.1
    (micro 4) and 2.74 / 2.6 (with checkpointing); the 216M shape at micro 2
    without checkpointing 4.88 / 5.0. All inside the 0.7-1.3 band, so the terms
    stand as they are."""
    n_total, n_nonembed = count_params(cfg)[:2]
    t = micro_batch * cfg.ctx
    b = 2 if amp else 4
    kv = cfg.n_kv_heads * cfg.head_dim
    per_token_attn = 4 * cfg.d_model + b * (7 * cfg.d_model + 4 * kv) + 3 * cfg.n_heads
    if device is not None and attention_keeps_scores(device):
        per_token_attn += cfg.n_heads * cfg.ctx * 6
    per_token_dense = per_token_attn + b * 4 * cfg.ffn_hidden
    n_moe = cfg.n_moe_blocks
    k, e = cfg.experts_per_token, cfg.n_experts
    per_token_moe = (per_token_attn + b * 4 * (k * cfg.expert_hidden + cfg.shared_hidden)
                     + 2 * k * cfg.d_model * b + 3 * e * 4 + 2 * k * 8) if n_moe else per_token_dense
    per_token_layer = max(per_token_dense, per_token_moe)  # the one block being recomputed under checkpointing
    if checkpointing:
        activations = t * cfg.n_layers * 4 * cfg.d_model + t * per_token_layer  # the residual stream is FP32
    else:
        activations = t * ((cfg.n_layers - n_moe) * per_token_dense + n_moe * per_token_moe)
    if doc_mask:
        activations += t * cfg.ctx  # B x T x T bytes
    logits = (min(t, loss_chunk) if loss_chunk else t) * cfg.vocab_size * (b + 4 + 4)
    if loss_chunk:
        logits += t * cfg.d_model * (4 + 4)  # the (FP32) hidden states and their gradient, kept whole
    state = 8 * n_total if optimizer != "muon" else 8 * (n_total - n_nonembed) + 4 * n_nonembed
    weight_cache = (2 if amp else 0) * n_total
    return int(activations + logits + n_total * 8 + weight_cache + state + 400 * 2 ** 20)


def memory_options(checkpointing="auto", loss_chunk=None):
    """(activation checkpointing, loss chunk tokens) choices allowed by the
    command line, cheapest first: the chunked loss costs 1-8 % of step time
    (nothing measurable when compiled), activation checkpointing 21-32 %
    (measured on the RTX 3070 at the 216M shape)."""
    ck = [False, True] if checkpointing == "auto" else [checkpointing == "on"]
    lc = [0, LOSS_CHUNK_TOKENS] if loss_chunk is None else [loss_chunk]
    # cheapest first (the comprehension already yields that order; the key states the rule)
    return sorted(((c, l) for c in ck for l in lc), key=lambda o: (o[0], o[1] != 0))


def choose_memory_plan(cfg, tokens_per_step, budget_bytes, amp, optimizer="adamw", checkpointing="auto",
                       loss_chunk=None, micro=None, device=None, doc_mask=False):
    """(micro-batch, accumulation, activation checkpointing, loss chunk tokens):
    the cheapest memory option that reaches a micro-batch of about 2k tokens
    (min(target, 2048 // ctx) sequences: 2 at ctx 1024, 1 at ctx >= 2048), else
    the cheapest that fits at all. About 2k tokens per micro-step already
    saturate the GEMMs of a >= 1024-wide model (micro 2 vs 4: -7 % compiled),
    so a 4-sequence micro-batch is never worth the +21-32 % recompute of
    activation checkpointing. With `micro` given (an explicit --batch-size),
    the cheapest option that fits that batch. On a device whose attention keeps
    the (B, H, T, T) scores (attention_keeps_scores: MPS) activation checkpointing
    comes first at ctx >= 2048, where the quadratic score memory outweighs its
    recompute."""
    target = max(1, tokens_per_step // cfg.ctx)
    options = memory_options(checkpointing, loss_chunk)
    if device is not None and cfg.ctx >= 2048 and attention_keeps_scores(device):
        options.sort(key=lambda o: (not o[0], o[1] != 0))
    fits = lambda m, o: estimate_train_bytes(cfg, m, amp, o[0], o[1], optimizer, device, doc_mask) <= budget_bytes  # noqa: E731
    if micro is not None:
        chosen = next((o for o in options if fits(micro, o)), options[-1])
        # rounded down like split_batch: never more tokens per step than asked for (the caller may round otherwise)
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
    want = min(target, max(1, 2048 // cfg.ctx))
    o = next((o for o in options if best.get(o, 0) >= want), next(o for o in options if o in best))
    m, accum = split_batch(target, best[o], want)
    return m, accum, o[0], o[1]


def split_batch(target, largest, smallest=1):
    """(micro-batch, accumulation) whose product is closest to `target` sequences
    without exceeding it, the micro-batch between max(largest / 2, smallest) and
    `largest` (the largest that fits); ties go to the larger micro-batch. Powers of
    two divide exactly, so they keep `largest`; other targets (from --steps) no
    longer lose up to half their tokens to rounding (61 = 32 x 1 becomes 30 x 2)."""
    best = None
    for m in range(largest, max((largest + 1) // 2, min(smallest, largest), 1) - 1, -1):
        accum = max(1, target // m)
        if best is None or m * accum > best[0] * best[1]:
            best = (m, accum)
    return best


def plan_run(args, manifest, device, amp_dtype, token_cap=None, fixed_cfg=None, verbose=True):
    """Fill every unset hyper-parameter from the dataset and device.

    token_cap limits the training-token budget (used by --time-budget-hours).
    verbose=False: the explanation lines for a run header, without what the
    header prints itself (the per-source token list, the batch, the weights and
    source shares that are all equal); warnings and notes are always kept.
    fixed_cfg: an architecture that is not up for planning (--init-from: the
    checkpoint's ModelConfig, experts included); memory, batch, steps and the
    learning rate are then sized for exactly it. The compute rules (tokens per
    step, learning rate, FLOPs, the tokens-per-parameter fit) use the ACTIVE
    parameter count, memory the TOTAL (count_params).
    Returns (ModelConfig, train settings dict, explanation lines)."""
    vocab = int(manifest["tokenizer"]["vocab_size"])
    train_by_source = {n: int(s["train"]["tokens"]) for n, s in manifest["sources"].items() if s.get("train")}
    unique = sum(train_by_source.values())
    lines = []
    budget_bytes, budget_text = device_memory_budget(device)
    r = args.tokens_per_param
    max_epochs = args.max_epochs
    # the document mask stays opt-in at every context: at ctx 512 it changed bits per byte by -0.0016 (noise) for
    # ~8 % more forward+backward time (2026-11-05 A/B); ctx >= 2048 is unmeasured, so no planner default yet
    doc_mask = args.doc_mask == "on"

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
    ladder = shape_ladder(vocab)
    fitting = [c for c in ladder if count_params(c)[2] * r <= planned]
    target = planned / r
    explicit_shape = any(v is not None for v in (args.d_model, args.layers, args.heads, args.kv_heads))
    n_experts = int(args.experts)

    def shaped(base):
        if fixed_cfg is not None:
            return ModelConfig(**dataclasses.asdict(fixed_cfg))
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
        cfg.ctx = args.ctx if args.ctx else default_ctx(cfg.d_model)
        cfg.ffn_hidden = args.ffn_hidden or swiglu_hidden(cfg.d_model)
        # the RoPE base follows the context unless given (10,000 up to ctx 2048, then ctx^2)
        cfg.rope_base = args.rope_base if args.rope_base else default_rope_base(cfg.ctx)
        cfg.tie_embeddings = args.tie_embeddings != "off"
        # new runs get QK-norm unless told otherwise: -0.106 bits per byte against the same run without it at the
        # same rate (d 384, 500 steps, 2026-11-05 A/B), at no step time that the clock noise could resolve
        cfg.qk_norm = args.qk_norm != "off"
        # mixture of experts only when asked (--experts N); dense stays the default
        cfg.n_experts = n_experts
        cfg.experts_per_token = int(args.experts_per_token)
        cfg.expert_hidden = (int(args.expert_hidden)
                             or expert_hidden_for(cfg.ffn_hidden, cfg.experts_per_token)) if n_experts else 0
        cfg.shared_hidden = int(args.shared_hidden) if n_experts else 0
        cfg.moe_every = int(args.moe_every)
        cfg.router = args.router or "softmax"
        cfg.balance = args.balance or "bias"
        cfg.validate()
        return cfg

    def target_tps(cfg):
        """Tokens per step before the micro-batch rounding: the command line, or the compute-budget rule."""
        if args.tokens_per_step:
            return args.tokens_per_step
        if args.steps:
            return max(cfg.ctx, planned // args.steps)
        return tokens_per_step_for(count_params(cfg)[2], planned)

    def batch_for(cfg):
        tps = target_tps(cfg)
        amp, opt_kind = amp_dtype is not None, args.optimizer or "adamw"
        if args.batch_size:
            if fixed_cfg is None and not explicit_shape:
                # a ladder shape must fit at batch 1 with some memory option whatever --batch-size says: the
                # SystemExit of this call sends the candidate loop below to the next smaller shape
                choose_memory_plan(cfg, tps, budget_bytes, amp, opt_kind, args.activation_checkpointing,
                                   args.loss_chunk_tokens, device=device, doc_mask=doc_mask)
            micro, accum, ck, lc = choose_memory_plan(cfg, tps, budget_bytes, amp, opt_kind,
                                                      args.activation_checkpointing, args.loss_chunk_tokens,
                                                      micro=args.batch_size, device=device, doc_mask=doc_mask)
            if args.steps:  # steps fixed: rounding up would train past the token budget (and the epoch cap)
                accum = args.grad_accum or max(1, tps // (micro * cfg.ctx))
            else:  # steps = budget // tokens per step below, so the nearest accumulation stays within the budget
                accum = args.grad_accum or max(1, round(tps / (micro * cfg.ctx)))
            if estimate_train_bytes(cfg, micro, amp, ck, lc, opt_kind, device, doc_mask) > budget_bytes:
                lines.append(f"WARNING: batch {micro} x ctx {cfg.ctx} may not fit ({budget_text}); an "
                             "out-of-memory error at step 1 switches on memory savings or halves the batch.")
            return micro, accum, ck, lc
        micro, accum, ck, lc = choose_memory_plan(cfg, tps, budget_bytes, amp, opt_kind,
                                                  args.activation_checkpointing, args.loss_chunk_tokens,
                                                  device=device, doc_mask=doc_mask)
        return micro, (args.grad_accum or accum), ck, lc

    if fixed_cfg is not None:
        cfg = shaped(None)
        shape_why = "the architecture of the --init-from checkpoint"
        micro, accum, ckpt, chunk = batch_for(cfg)
    elif explicit_shape:
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
    n_total, n_nonembed, n_active, _ = count_params(cfg)
    tokens_per_step = micro * accum * cfg.ctx

    steps = args.steps or max(1, planned // tokens_per_step)  # round down: never exceed the epoch cap
    train_tokens = steps * tokens_per_step
    epochs = train_tokens / max(unique, 1)
    dropout = args.dropout if args.dropout is not None else (0.0 if epochs <= 4 else 0.1)
    cfg.dropout = dropout
    # lr_batch_factor on the batch the run will use (micro x accumulation x ctx): about x1.4 / x0.7 for double / half
    lr_factor = 1.0 if args.lr else lr_batch_factor(n_active, tokens_per_step)
    # --lr-scale: the sweep knob on the heuristic rate
    lr_scale = float(args.lr_scale)
    if args.lr and lr_scale != 1.0:  # recorded as the factor the run's rate actually carries
        lines.append(f"--lr-scale {lr_scale:g} has no effect: --lr sets the peak learning rate (recorded as 1)")
        lr_scale = 1.0
    lr = args.lr or heuristic_lr(n_active) * lr_factor * lr_scale
    min_lr_ratio = resolve_min_lr_ratio(args.min_lr_ratio, args.schedule)
    warmup = args.warmup_steps if args.warmup_steps is not None else int(min(max(round(0.02 * steps), 50), 2000, max(1, steps // 10)))
    if args.warmup_steps is not None and args.warmup_steps >= steps:  # validate_args can only compare with --steps
        raise SystemExit(f"--warmup-steps {args.warmup_steps:,} must be smaller than the {steps:,} planned steps.")
    mix_alpha = 1.0 if args.mix_alpha is None else float(args.mix_alpha)
    weights = mix_weights(args.mix, train_by_source, train_by_source, mix_alpha)
    raw_share = {s: weights[s] * n for s, n in train_by_source.items()}
    share = {s: v / max(sum(raw_share.values()), 1e-12) for s, v in raw_share.items()}
    source_epochs = {s: train_tokens * share[s] / max(n, 1) for s, n in train_by_source.items()}
    settings = {
        "steps": steps, "micro_batch": micro, "grad_accum": accum, "tokens_per_step": tokens_per_step,
        "lr": lr, "min_lr": lr * min_lr_ratio, "warmup_steps": warmup, "schedule": args.schedule,
        "decay_frac": args.decay_frac, "decay_shape": args.decay_shape, "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2], "grad_clip": args.grad_clip, "planned_train_tokens": train_tokens,
        "optimizer": args.optimizer or "adamw", "activation_checkpointing": ckpt, "loss_chunk_tokens": chunk,
        "lr_scale": lr_scale, "doc_mask": doc_mask,
    }
    if cfg.n_experts:  # the mixture-of-experts training terms (see the training loop)
        settings["lb_loss"] = float(args.lb_loss)
        settings["router_z_loss"] = float(args.router_z_loss)
        settings["bias_rate"] = float(args.bias_rate)
    resolve_ema(settings, args.ema, resumed=False)
    moe_text = ""
    if cfg.n_experts:
        moe_text = (f", MoE: {cfg.n_experts} experts of hidden {cfg.expert_hidden}, {cfg.experts_per_token} per "
                    f"token" + (f" in every {cfg.moe_every}. block" if cfg.moe_every > 1 else "")
                    + (f", shared SwiGLU {cfg.shared_hidden}" if cfg.shared_hidden else "")
                    + f", {cfg.router} router, balance {cfg.balance}")
    ctx_why = ("--ctx" if args.ctx else "from the checkpoint" if fixed_cfg is not None
               else f"default for d_model {cfg.d_model}")
    attention = f"attention {attention_flop_share(cfg):.0%} of the training FLOPs per token"
    block = [
        f"unique training tokens U = {unique:,}"
        + (f" ({', '.join(f'{k} {v:,}' for k, v in train_by_source.items())})" if verbose else "")
        + f"; vocabulary {vocab:,}; {budget_text}",
        f"token budget D = {planned:,} ({why}); shape: {shape_why}",
        f"model: d_model {cfg.d_model}, layers {cfg.n_layers}, heads {cfg.n_heads}/{cfg.n_kv_heads} kv "
        f"(head {cfg.head_dim}), SwiGLU {cfg.ffn_hidden}{', QK-norm' if cfg.qk_norm else ''}{moe_text}, ctx {cfg.ctx}"
        f"{'' if verbose else f' ({ctx_why}; {attention})'}, RoPE base {cfg.rope_base:,.0f}"
        f"{'' if args.rope_base or fixed_cfg is not None else ' (default for this ctx)'}; parameters {n_total:,} total, "
        + (f"{n_active:,} active per token, " if cfg.n_experts else "")
        + f"{n_nonembed:,} excluding the ({'tied' if cfg.tie_embeddings else 'untied'}) embedding",
        f"ctx {cfg.ctx} ({ctx_why}): {attention}",
        f"batch: {micro} x {cfg.ctx} tokens x {accum} accumulation = {tokens_per_step:,} tokens/step; "
        f"{steps:,} steps = {train_tokens:,} training tokens",
        f"ratios: {train_tokens / n_active:,.1f} training tokens per {'active ' if cfg.n_experts else ''}parameter "
        + (f"({train_tokens / n_total:,.1f} per total parameter, {train_tokens / max(n_nonembed, 1):,.1f} per "
           f"non-embedding parameter); " if cfg.n_experts else
           f"({train_tokens / max(n_nonembed, 1):,.1f} per non-embedding parameter); ")
        + f"unique data {unique / n_total:,.1f} tokens per parameter; {epochs:,.2f} epochs",
        f"learning rate {lr:.2e} peak ("
        + ("set" if args.lr else f"heuristic for {n_active / 1e6:,.1f}M {'active ' if cfg.n_experts else ''}parameters x batch factor {lr_factor:.2f}"
           + ("" if lr_factor == 1.0 else f" for {tokens_per_step:,} tokens/step against the table's "
                                          f"{tokens_per_step_legacy(n_active):,}")
           + ("" if lr_scale == 1.0 else f" x --lr-scale {lr_scale:g}"))
        + f"), min {lr * min_lr_ratio:.2e} ({min_lr_ratio:g} of peak"
        + ("" if args.min_lr_ratio is not None else
           f", default for {args.schedule}: {'decay to zero' if min_lr_ratio == 0 else 'one tenth'}") + "); "
        f"dropout {dropout} ({'set' if args.dropout is not None else 'auto: 0 for <= 4 epochs, else 0.1'}); "
        f"weight decay {args.weight_decay} on matrices only; optimizer {OPTIMIZER_LABELS[args.optimizer or 'adamw']}",
        # the EMA timescale of AdamW's weight decay, 1 / (lr x wd) steps (Wang and Aitchison 2024): how far back the
        # weights remember; a timescale far above the run means the decay hardly acts, far below means forgetting
        ("decay timescale 1/(lr x wd) = " + (f"{1 / (lr * args.weight_decay):,.0f} steps "
                                             f"({1 / (lr * args.weight_decay) / steps:.0%} of the run)"
                                             if args.weight_decay > 0 else "infinite (no weight decay)")),
        "sources: " + ", ".join(f"{s} {share[s]:.1%} ({source_epochs[s]:.2f} epochs)" for s in train_by_source)
        + (f"; --mix-alpha {mix_alpha:g}: shares ~ weight x tokens^alpha" if mix_alpha != 1.0 else ""),
        "memory: " + ("activation checkpointing on (blocks recomputed in the backward pass)" if ckpt
                      else "activation checkpointing off") + "; "
        + (f"loss computed {chunk:,} tokens at a time" if chunk else "loss computed on the whole micro-batch")
        + f" (estimated {estimate_train_bytes(cfg, micro, amp_dtype is not None, ckpt, chunk, args.optimizer or 'adamw', device, doc_mask) / 2 ** 30:.1f} GiB "
        f"of {budget_text})",
        "weights: " + (f"EMA with decay {settings['ema']:g} per step (updated every {EMA_EVERY} steps), scored beside "
                       "the raw weights at each evaluation; the better of the two is saved as the best model"
                       if settings["ema"] else "no EMA (--ema off)" if args.ema is not None
                       else "no EMA (opt-in: --ema 0.999)")
        + ("; document mask on: attention stays within each document of a window (B x T x T bytes of mask)"
           if doc_mask else ""),
    ]
    if not verbose:
        # the run header has the batch and the weights; the ratios of a dense model follow from its budget line, and
        # source shares are news only when the sources are not all read the same number of times
        hidden = ("ctx ", "batch:", "weights:") + (() if cfg.n_experts else ("ratios:",))
        if len({f"{e:.2f}" for e in source_epochs.values()}) <= 1:
            hidden += ("sources:",)
        block = [line for line in block if not line.startswith(hidden)]
    lines += block
    if args.steps and train_tokens > max_epochs * unique:
        lines.append(f"note: --steps {args.steps:,} trains {epochs:.2f} epochs, more than --max-epochs {max_epochs:g}.")
    usual_tps = tokens_per_step_for(n_active, planned)
    if args.steps and not args.tokens_per_step and tokens_per_step > 4 * usual_tps:
        # --steps alone shares the whole budget D among the steps: nothing bounds the step (target_tps)
        lines.append(f"WARNING: --steps {args.steps:,} alone divides the token budget D by the steps: "
                     f"{tokens_per_step:,} tokens per step, {tokens_per_step / usual_tps:,.0f}x the {usual_tps:,} the "
                     "planner would choose for this model and budget. For a short run add --tokens-per-step (the "
                     "budget is then --steps x --tokens-per-step), or give --train-tokens instead of --steps.")
    floor = float(args.tokens_per_total_param)
    if cfg.n_experts and floor > 0 and train_tokens / n_total < floor:
        lines.append(f"WARNING: {train_tokens / n_total:,.1f} training tokens per TOTAL parameter is below the "
                     f"--tokens-per-total-param floor {floor:g}: every published mixture-of-experts gain was "
                     "measured at 22-700; with this little data the experts are unlikely to beat the dense model "
                     "(fewer --experts, --moe-every 2, or more data).")
    if cfg.n_experts >= 32:
        lines.append(f"note: {cfg.n_experts} experts on one device run as {cfg.n_experts} small matmuls per layer "
                     "(or one grouped kernel with --moe-impl grouped on CUDA); fine-grained experts pay off with a "
                     "fused grouped GEMM (SM90 CUDA), not here. 8-16 experts, 2 per token, is the default range.")
    over = [s for s, e in source_epochs.items() if e > max_epochs + 1e-9]
    if over:
        lines.append("WARNING: " + ", ".join(f"{s} would be seen {source_epochs[s]:.1f} times" for s in over)
                     + f", above --max-epochs {max_epochs:g} (repeats beyond ~4 fade, Muennighoff et al. 2023); "
                     "lower its --mix weight or bring --mix-alpha closer to 1.")
    # a context shorter than the typical document of a source with a real share of the training tokens means that
    # source is mostly read in pieces (random windows cross document boundaries)
    long_docs = []
    for name, s in manifest["sources"].items():
        p50 = int(s.get("doc_tokens_percentiles", {}).get("50", 0) or 0)
        if share.get(name, 0.0) > 0.10 and p50 > cfg.ctx:
            long_docs.append(f"{name} (median document {p50:,} tokens, {share[name]:.0%} of training tokens)")
    if long_docs:
        lines.append(f"note: ctx {cfg.ctx} is shorter than the median document of " + ", ".join(long_docs)
                     + "; these are read in windows that cross documents.")
    if epochs > 4:
        lines.append(f"note: {epochs:.1f} epochs repeats data beyond the ~4 epochs that Muennighoff et al. (2023) "
                     "found nearly as good as fresh data; expect diminishing returns and watch val(mix) against train.")
    return cfg, settings, lines


def measure_seconds_per_step(cfg, micro, accum, device, amp_dtype, attn_pref="auto", warmup=2, timed=3,
                             checkpointing=False, loss_chunk=0, settings=None):
    """(seconds per optimizer step, peak memory in bytes) of real training
    micro-steps of this shape on this device (random tokens), with the planned
    memory options and the planned optimizer (`settings`: optimizer, lr,
    weight_decay, betas; AdamW when None). Runs inside fork_rng so the caller's
    random state (model initialisation) is untouched."""
    with fork_device_rng(device):
        torch.manual_seed(0)
        return _measure(cfg, micro, accum, device, amp_dtype, attn_pref, warmup, timed, checkpointing, loss_chunk,
                        settings)


def _measure(cfg, micro, accum, device, amp_dtype, attn_pref, warmup, timed, checkpointing=False, loss_chunk=0,
             settings=None):
    cap_device_memory(device)  # MPS: an over-budget probe fails with out-of-memory instead of swapping
    probe_cfg = ModelConfig(**dataclasses.asdict(cfg))
    model = TinyGPT(probe_cfg).to(device)
    model.set_attention_impl(choose_attention(device, probe_cfg, amp_dtype, attn_pref)[0])
    model.grad_checkpoint = checkpointing
    # the optimizer the run will use (Muon's orthogonalisation adds a few hundred ms per update at 216M), so the
    # update is timed as it will run: once per step, not scaled up with the micro-steps
    opt_settings = {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 0.0, "betas": [0.9, 0.95], **(settings or {})}
    opt = build_optimizer(model, opt_settings, device)[0]
    x = torch.randint(0, cfg.vocab_size, (micro, cfg.ctx + 1), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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
        opt.step()  # the first update allocates the optimizer state: part of the peak, not of the timing
        opt.zero_grad(set_to_none=True)
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(timed):
            micro_step()
        synchronize(device)
        t1 = time.perf_counter()
        opt.step()
        opt.zero_grad(set_to_none=True)
        synchronize(device)
        t2 = time.perf_counter()
        # a step is `accum` micro-steps (forward and backward) and ONE optimizer update
        return (t1 - t0) / timed * accum + (t2 - t1), peak_memory_bytes(device)
    finally:
        model = opt = x = None  # freed before empty_cache (not del: micro_step shares these names)
        free_device_memory(device)


def flops_per_token(cfg, checkpointing=False):
    """Training floating-point operations per token: 6 per parameter (2 in the
    forward pass, 4 in the backward) plus the attention scores, 12 x layers x
    context x width (QK^T and AV: 4 per layer forward, 8 backward). Activation
    checkpointing runs the forward pass a second time: 8 per parameter and 16 x
    layers x context x width. The parameters are the ACTIVE ones: in a
    mixture-of-experts block only the experts a token visits do work."""
    n = count_params(cfg)[2]
    return (8 if checkpointing else 6) * n + (16 if checkpointing else 12) * cfg.n_layers * cfg.ctx * cfg.d_model


def sustained_speed(out_dir, device):
    """(FLOP/s, description) that this computer sustained in its last training run
    of at least 12 minutes on this kind of device (experiments.csv), or (None,
    reason). A short benchmark on a cool laptop GPU runs at boost clocks; over
    hours the power and heat limits slow it down (about 1.6x on this RTX 3070), so a
    real run is the better guide. Rows from before the device column are CUDA.
    The rows of a resumed run are cumulative: when an earlier row of the same
    run exists (same name, device and shape, written less than the run's whole
    training time before this one, so that it cannot be an older run of that
    name) the speed is that of the last session alone, the difference between
    the two rows, if it lasted 12 minutes or more."""
    import csv
    try:
        path = experiments_path(out_dir)
        if not os.path.isfile(path):  # a table from before results/ was its only home
            path = os.path.join(out_dir, "experiments.csv")
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("event") == "training run"
                    and (r.get("device") or "cuda").startswith(device.type)]
    except OSError:
        return None, "no experiments.csv"
    for i in range(len(rows) - 1, -1, -1):
        r = rows[i]
        try:
            moe = parse_moe_label(r.get("moe"))  # "8x2/1" from the moe column; blank or absent = dense
            cfg = ModelConfig(vocab_size=int(r["vocab"]), ctx=int(r["ctx"]), d_model=int(r["d_model"]),
                              n_layers=int(r["layers"]), n_heads=int(r["heads"]), n_kv_heads=int(r["kv_heads"]),
                              **moe)
            tokens, hours = float(r["tokens_seen"]), float(r["train_hours"])
        except (KeyError, TypeError, ValueError, SystemExit):
            continue
        if hours >= 0.2 and tokens > 0:
            # the run's own FLOP count: a checkpointed run did more work per token than its tokens/s suggest
            recompute = (r.get("activation_checkpointing") or "off") == "on"
            session = ""
            same_run = ("name", "device", "params", "d_model", "layers", "ctx", "vocab")
            earlier = next((p for p in reversed(rows[:i]) if all(p.get(k) == r.get(k) for k in same_run)), None)
            if earlier is not None:
                try:
                    more_tokens = tokens - float(earlier["tokens_seen"])
                    more_hours = hours - float(earlier["train_hours"])
                    between = (datetime.datetime.fromisoformat(r["time"])
                               - datetime.datetime.fromisoformat(earlier["time"])).total_seconds() / 3600
                except (KeyError, TypeError, ValueError):  # a row without a usable time: the whole run
                    more_tokens = more_hours = between = 0.0
                # the table has no run id, and a run started again from scratch under its name and shape can pass
                # the earlier row in tokens and hours. The clock tells the two apart: all the training of a new run
                # fits between the two rows; of a resumed run only the last session does (0.01 h: the rounding)
                if more_tokens > 0 and 0.2 <= more_hours <= between + 0.01 < hours:
                    tokens, hours, session = more_tokens, more_hours, ", last session"
            speed = tokens / (hours * 3600) * flops_per_token(cfg, recompute)
            return speed, (f"{speed / 1e12:.1f} TFLOP/s sustained by run '{r.get('name')}' of {r.get('time', '')[:10]} "
                           f"({int(r['params']) / 1e6:.1f}M parameters, {tokens / hours / 3600:,.0f} tokens/s"
                           f"{', activation checkpointing' if recompute else ''}{session})")
    return None, f"no training run of 12 minutes or more on a {device.type} device yet"


def parse_moe_label(text):
    """ModelConfig fields from an experiments-table moe label "ExK/M" (experts x
    per token / every); {} for a blank (dense) one."""
    text = (text or "").strip()
    if not text:
        return {}
    m = re.fullmatch(r"(\d+)x(\d+)/(\d+)", text)
    if not m:
        raise ValueError(f"not a moe label: {text!r}")
    return {"n_experts": int(m.group(1)), "experts_per_token": int(m.group(2)), "moe_every": int(m.group(3))}


# compiled / eager training speed, measured on the RTX 3070 (1.31x at 6 layers, 1.45x at the 216M shape)
COMPILE_SPEEDUP = 1.35
# share of a time budget that goes to training steps: evaluation, sampling and checkpoints take about 8 %
TIME_BUDGET_EFFICIENCY = 0.92
# how much slower hours of training run than a short test on a cool device: a laptop GPU's power and heat limits
# (sm clock 430-1290 MHz across runs on this RTX 3070); a desktop-class Mac or a CPU hardly throttles
SLOW_FACTOR = {"cuda": 1.6, "mps": 1.1, "cpu": 1.1}


def have_triton():
    """Whether Triton, torch.compile's GPU backend, is installed (only that matters)."""
    try:
        import triton
        del triton
        return True
    except ImportError:
        return False


def plan_with_time_budget(args, manifest, device, amp_dtype, fixed_cfg=None, verbose=True):
    """plan_run, then (with --time-budget-hours) choose the largest ladder shape
    whose parameter count N still satisfies tokens >= r * N when tokens are
    limited by BOTH the data cap and what fits in the time budget at the speed
    measured on this device. Binary search over the ladder (bigger = slower, so
    the condition is monotone); evaluation/sampling/checkpoint overhead 8%.
    fixed_cfg (--init-from): the architecture is given, so only its tokens are
    capped by the budget (N is its active parameter count). verbose: as in
    plan_run, for the lines that are returned."""
    cfg, settings, lines = plan_run(args, manifest, device, amp_dtype, fixed_cfg=fixed_cfg, verbose=verbose)
    hours = args.time_budget_hours
    if not hours:
        return cfg, settings, lines
    budget_s = hours * 3600 * TIME_BUDGET_EFFICIENCY
    r = args.tokens_per_param
    speed, speed_note = sustained_speed(args.out_dir, device)
    slow = SLOW_FACTOR.get(device.type, SLOW_FACTOR["cuda"])
    # the probe runs eager; when the run will be compiled its time is divided by the measured compiled/eager ratio
    compiled = args.compile != "off" and device.type == "cuda" and have_triton()
    budget_bytes = device_memory_budget(device)[0]
    notes = [f"time budget: speed from {speed_note}" if speed else
             f"time budget: {speed_note}, so short measurements are slowed by {slow:g}x (the gap between a cool and a "
             f"hot {'laptop GPU' if device.type == 'cuda' else device.type + ' device'}); later plans use the speed "
             "of your runs"]
    if compiled:
        notes.append(f"time budget: the speed test runs eager and torch.compile will be on, so its times are divided "
                     f"by {COMPILE_SPEEDUP:g} (compiled/eager measured on this GPU)")

    def trial(shape_args):
        """(reaches tokens >= r x N, tokens that fit, peak memory of the speed test in bytes, whether the time
        and not the data limits the tokens) for this shape."""
        c, st, _ = plan_run(shape_args, manifest, device, amp_dtype, fixed_cfg=fixed_cfg)
        raw, peak = measure_seconds_per_step(c, st["micro_batch"], st["grad_accum"], device, amp_dtype, args.attn,
                                             checkpointing=st["activation_checkpointing"],
                                             loss_chunk=st["loss_chunk_tokens"], settings=st)
        measured = raw / COMPILE_SPEEDUP if compiled else raw
        step_flops = st["tokens_per_step"] * flops_per_token(c, st["activation_checkpointing"])
        sec = max(measured, step_flops / speed) if speed else measured * slow
        fit_tokens = int(budget_s / sec) * st["tokens_per_step"]
        tokens = min(fit_tokens, st["planned_train_tokens"])
        n = count_params(c)[2]
        est = estimate_train_bytes(c, st["micro_batch"], amp_dtype is not None, st["activation_checkpointing"],
                                   st["loss_chunk_tokens"], st["optimizer"], device, st.get("doc_mask", False))
        notes.append(f"time budget {hours:g} h: {n / 1e6:,.1f}M parameters, {sec:.2f} s/step expected over hours "
                     f"({raw:.2f} measured in a short test" + (f", {measured:.2f} compiled" if compiled else "")
                     + f"; {st['tokens_per_step']:,} tokens/step) -> {fit_tokens:,} tokens fit in time; memory: "
                     f"estimated {est / 2 ** 30:.2f} GiB, probe measured {peak / 2 ** 30:.2f} GiB "
                     f"(ratio {peak / max(est, 1):.2f})")
        return n * r <= tokens, tokens, peak, fit_tokens < st["planned_train_tokens"]

    def is_oom(exc):  # torch.cuda.OutOfMemoryError and the MPS "out of memory" error are RuntimeErrors
        if "out of memory" not in str(exc).lower():
            return False
        free_device_memory(device)
        return True

    def halved(a, micro):
        return argparse.Namespace(**{**vars(a), "batch_size": micro // 2,
                                     "grad_accum": a.grad_accum * 2 if a.grad_accum else None})

    explicit = fixed_cfg is not None or any(v is not None for v in (args.d_model, args.layers, args.heads,
                                                                     args.kv_heads))

    def measured(a):
        """(reaches the target, tokens, peak, time binds, args): trial(a), and when the speed test runs out of
        memory the same shape again with activation checkpointing on (unless the command line turned it off),
        then with the micro-batch halved down to 1, before the shape counts as not fitting (tokens None). Without
        the retries an out-of-memory error would silently pick a smaller MODEL, or refuse a given shape, where a
        memory option would do. A given shape (explicit) first halves its micro-batch down to 1, which costs
        no compute: a plan that halving alone makes fit stays the plan it always was and keeps all its tokens
        (checkpointing costs about a quarter of them in the same time). Only where that used to be the end,
        checkpointing follows, with the batch options as given, and its own halving."""
        given = a
        while True:
            try:
                return trial(a) + (a,)
            except SystemExit:
                if explicit:  # a given shape the planner refuses: its own message
                    raise
                return False, None, 0, True, a
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise
                c, st, _ = plan_run(a, manifest, device, amp_dtype, fixed_cfg=fixed_cfg)
                halve_first = explicit and st["micro_batch"] > 1
                if not st["activation_checkpointing"] and a.activation_checkpointing != "off" and not halve_first:
                    notes.append(f"time budget: d_model {c.d_model} ran out of memory in the speed test; "
                                 "measured again with activation checkpointing on")
                    a = argparse.Namespace(**{**vars(a), "activation_checkpointing": "on",
                                              "batch_size": given.batch_size, "grad_accum": given.grad_accum})
                elif st["micro_batch"] > 1:
                    notes.append(f"time budget: d_model {c.d_model} ran out of memory in the speed test at "
                                 f"micro-batch {st['micro_batch']}; measured again with {st['micro_batch'] // 2}")
                    a = halved(a, st["micro_batch"])
                else:
                    notes.append(f"time budget: d_model {c.d_model} does not fit in memory even at micro-batch "
                                 "1 with activation checkpointing; a smaller shape is used")
                    return False, None, 0, True, a

    if explicit:
        _, tokens, peak, time_binds, chosen_args = measured(args)  # the given shape must be measured
        if tokens is None:
            raise SystemExit("time budget: this model shape runs out of memory on this device even at "
                             "micro-batch 1; choose a smaller --d-model/--layers/--ctx.")
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
                results[i] = measured(shape_args(i))
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
                             + ("in time; it trains on what fits" if results[0][3] else
                                "on this data; it trains on what the data supports"))
        _, tokens, peak, time_binds, chosen_args = results[best]  # with the memory options the retries settled on
    # the cap is passed (and named in the plan) only when the time, not the data, limits the tokens; without it the
    # plan is the one the speed test measured (tokens None: the smallest ladder shape did not fit, nothing to train)
    cfg, settings, lines = plan_run(chosen_args, manifest, device, amp_dtype,
                                    token_cap=(tokens or 0) if time_binds else None, fixed_cfg=fixed_cfg,
                                    verbose=verbose)
    # the shape note sits in the 'token budget D' line, which a warning or note printed before it can shift
    shape_note = ("largest ladder shape that satisfies the data AND the time budget" if not explicit
                  else "command-line values, tokens capped by the time budget" if time_binds
                  else "command-line values, ladder defaults for the rest")
    lines = [line.replace("command-line values, ladder defaults for the rest", shape_note)
             if line.startswith("token budget D") else line for line in lines]
    if peak > budget_bytes:
        # the estimate was too low for this shape: step to the next memory option now, instead of leaving the
        # first step to run out of memory and find it
        options = memory_options(args.activation_checkpointing, args.loss_chunk_tokens)
        current = (bool(settings["activation_checkpointing"]), int(settings["loss_chunk_tokens"]))
        later = options[options.index(current) + 1:] if current in options else []
        if later:
            settings["activation_checkpointing"], settings["loss_chunk_tokens"] = later[0]
            notes.append(f"time budget: the speed test's peak memory {peak / 2 ** 30:.2f} GiB is above the budget "
                         f"{budget_bytes / 2 ** 30:.2f} GiB, so the next memory option is used from the start: "
                         + ("activation checkpointing on" if later[0][0] else "activation checkpointing off") + ", "
                         + (f"loss computed {later[0][1]:,} tokens at a time" if later[0][1]
                            else "loss computed on the whole micro-batch"))
    return cfg, settings, lines + notes


def plan_table(vocab, device, amp_dtype, r, max_epochs, optimizer="adamw"):
    """What the planner would choose for other corpus sizes (same vocabulary): the
    batch is the memory plan of `optimizer` on `device` (choose_memory_plan), the
    learning rate follows the batch that plan reaches."""
    rows = []
    budget, _ = device_memory_budget(device)
    for unique in (5e6, 2e7, 1e8, 3e8, 1e9, 1e10):
        planned = max_epochs * unique  # the planner's default: up to --max-epochs passes over the data
        ladder = shape_ladder(vocab)
        fitting = [c for c in ladder if count_params(c)[0] <= planned / r]
        cfg = fitting[-1] if fitting else ladder[0]
        cfg.ctx = default_ctx(cfg.d_model)
        cfg.rope_base = default_rope_base(cfg.ctx)
        n = count_params(cfg)[0]
        tps = tokens_per_step_for(n, planned)
        reached = tps  # the target when nothing fits
        try:
            micro, accum, _, _ = choose_memory_plan(cfg, tps, budget, amp_dtype is not None, optimizer, device=device)
            fits = f"{micro}x{accum}"
            reached = micro * accum * cfg.ctx
        except SystemExit:
            fits = "does not fit"
        rows.append((unique, planned, cfg, n, tps, fits, heuristic_lr(n) * lr_batch_factor(n, reached)))
    return rows


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
# The defaults of the options a run fixes when it starts and --resume does not take. parse_args reads them from
# here, and a resumed run compares a value with them: one that is neither the default nor the run's own was typed.
FIXED_OPTION_DEFAULTS = {"schedule": "wsd", "decay_frac": 0.2, "decay_shape": "1-sqrt",
                         "weight_decay": WEIGHT_DECAY_DEFAULT, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0}


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
    g.add_argument("--mix-alpha", type=float,
                   help="sampling share of a source ~ its --mix weight x its tokens^alpha (default 1: proportional "
                        "to size; below 1 flattens a mix one source dominates; honoured on --resume like --mix)")
    g = p.add_argument_group("run")
    g.add_argument("--name", default=DEFAULT_NAME, help="checkpoint base name")
    g.add_argument("--out-dir", default=HERE, help="folder for checkpoints and metrics")
    g.add_argument("--resume", action="store_true", help="continue <name>.pt: same windows, schedule and optimizer state (GPU kernels are not "
                        "bit-deterministic, so a replayed step differs at about 1e-5 in the loss)")
    g.add_argument("--init-from", help="start a new run from another checkpoint's weights (tinyGPT or legacy)")
    g.add_argument("--overwrite", action="store_true",
                   help="allow a fresh run over an existing name (old files are renamed *.bak-<time>)")
    g.add_argument("--log-file", default=os.path.join(HERE, "log.txt"))
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--trust-checkpoint", action="store_true",
                   help="allow full unpickling of a checkpoint that fails the safe loader (only your own files)")
    g = p.add_argument_group("model (unset = planned from data)")
    g.add_argument("--d-model", type=int, help="model width (default: the largest ladder shape the data supports)")
    g.add_argument("--layers", type=int, help="transformer blocks (default: d_model / 64 + 2, between 4 and 32)")
    g.add_argument("--heads", type=int, help="query heads (default: heads of width 32, 64 or 128 by d_model)")
    g.add_argument("--kv-heads", type=int,
                   help="key/value heads (default: heads / 4 from 8 heads on when the head count is divisible by 4, "
                        "grouped-query attention; otherwise ordinary multi-head attention); equal to --heads gives "
                        "ordinary multi-head attention, about 10 %% slower for about 10 %% more parameters")
    g.add_argument("--ctx", type=int,
                   help="context length in tokens (default by width: 256 up to d 192, 512 up to d 512, 1024 up to "
                        "d 1024, 2048 up to d 2048, 4096 above)")
    g.add_argument("--ffn-hidden", type=int, help="SwiGLU hidden width (default: 8 d_model / 3 rounded up to 64)")
    g.add_argument("--rope-base", type=float, help="default: 10,000 up to ctx 2048, then 10,000 x (ctx/2048)^2")
    g.add_argument("--tie-embeddings", choices=("on", "off"), default="on",
                   help="share the input embedding with the output layer (default on, as Llama 3.2 and Gemma do "
                        "(Qwen2.5 in its small models)); off trains a separate output matrix")
    g.add_argument("--qk-norm", choices=("on", "off"),
                   help="RMSNorm over each query and key head before RoPE (QK-norm, as in Gemma 3, Qwen3 and OLMo 2; "
                        "keeps high learning rates stable): default on for new runs; a checkpoint keeps its own "
                        "setting on --resume and --init-from")
    g.add_argument("--dropout", type=float, help="default: 0 up to 4 epochs, else 0.1")
    g = p.add_argument_group("mixture of experts (default: none, a dense SwiGLU in every block)")
    g.add_argument("--experts", type=int, default=0,
                   help="replace the SwiGLU of every block (or every --moe-every-th) by this many smaller experts "
                        "with a router (0 = dense; 2-64; 8-16 is the useful range on one device). The model then "
                        "has more parameters at the same FLOPs per token; whether that beats dense depends on the "
                        "data (unproven below ~20 tokens per TOTAL parameter: A/B it first)")
    g.add_argument("--experts-per-token", type=int, default=2, help="experts each token is routed to (default 2)")
    g.add_argument("--expert-hidden", type=int, default=0,
                   help="hidden size of one expert (default: the SwiGLU hidden / experts per token, rounded to 64, "
                        "so the active FLOPs equal the dense FFN)")
    g.add_argument("--shared-hidden", type=int, default=0,
                   help="hidden size of a SwiGLU every token also goes through (DeepSeek-style shared expert; 0 = none)")
    g.add_argument("--moe-every", type=int, default=1, help="make every N-th block an MoE block (1 = all)")
    g.add_argument("--router", choices=("softmax", "sigmoid"),
                   help="routing scores: softmax over the experts (Mixtral) or a sigmoid per expert (DeepSeek-V3); "
                        "default softmax")
    g.add_argument("--balance", choices=("loss", "bias", "both"),
                   help="how the experts are kept evenly loaded: the Switch load-balancing loss (--lb-loss), "
                        "DeepSeek-V3's auxiliary-loss-free routing bias updated after every step (--bias-rate), "
                        "or both (default bias)")
    g.add_argument("--lb-loss", type=float, default=LB_LOSS_DEFAULT,
                   help="weight of the load-balancing loss E x sum f_e P_e when --balance is loss or both "
                        "(default %g; the term is reported as moe_lb whatever the balance)" % LB_LOSS_DEFAULT)
    g.add_argument("--router-z-loss", type=float, default=ROUTER_Z_LOSS_DEFAULT,
                   help="weight of the router z-loss (mean (log sum exp logits)^2; keeps the routing logits "
                        "bounded; default %g)" % ROUTER_Z_LOSS_DEFAULT)
    g.add_argument("--bias-rate", type=float, default=BIAS_RATE_DEFAULT,
                   help="step of the routing bias per optimizer step when --balance is bias or both (default %g)"
                        % BIAS_RATE_DEFAULT)
    g.add_argument("--moe-impl", choices=("loop", "grouped"), default="loop",
                   help="how the experts run: one matmul per expert (loop, default, every device) or one "
                        "torch._grouped_mm kernel for all (grouped; a gain only on compiled CUDA, used only when "
                        "a forward+backward probe passes at model build, else the loop)")
    g.add_argument("--tokens-per-total-param", type=float, default=0.0,
                   help="warn when an MoE run plans fewer training tokens per TOTAL parameter than this "
                        "(0 = off)")
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
    g.add_argument("--tokens-per-step", type=int,
                   help="tokens per optimizer step (default: half of DeepSeek's B_opt = 0.292 C^0.327 for C = 6 N D, "
                        "a power of two between 16,384 and 2,097,152; with --steps, the planned tokens / --steps)")
    g.add_argument("--lr", type=float, help="peak learning rate")
    g.add_argument("--lr-scale", type=float, default=1.0,
                   help="multiply the planner's heuristic peak learning rate by this (for LR sweeps; no effect with "
                        "--lr; ignored on --resume)")
    g.add_argument("--min-lr-ratio", type=float,
                   help="learning rate at the end of the run as a share of the peak (default: 0 for wsd, 0.1 for "
                        "cosine); on --resume it may still be changed before the decay phase begins")
    g.add_argument("--warmup-steps", type=int,
                   help="default: 2 %% of the steps, between 50 and 2,000 and at most a tenth of the steps")
    g.add_argument("--schedule", choices=("wsd", "cosine"), default=FIXED_OPTION_DEFAULTS["schedule"])
    g.add_argument("--decay-frac", type=float, default=FIXED_OPTION_DEFAULTS["decay_frac"],
                   help="WSD: share of steps in the final decay")
    g.add_argument("--decay-shape", choices=("1-sqrt", "cosine", "linear"),
                   default=FIXED_OPTION_DEFAULTS["decay_shape"])
    g.add_argument("--weight-decay", type=float, default=FIXED_OPTION_DEFAULTS["weight_decay"],
                   help="decoupled weight decay on the matrices (default %g: lr x weight decay as at the published "
                        "rates, the table rate being %g times those); resumed runs keep their setting"
                        % (WEIGHT_DECAY_DEFAULT, LR_TABLE_FACTOR))
    g.add_argument("--beta1", type=float, default=FIXED_OPTION_DEFAULTS["beta1"])
    g.add_argument("--beta2", type=float, default=FIXED_OPTION_DEFAULTS["beta2"])
    g.add_argument("--grad-clip", type=float, default=FIXED_OPTION_DEFAULTS["grad_clip"],
                   help="clip the gradient norm to this (0 = off)")
    g.add_argument("--activation-checkpointing", choices=("auto", "on", "off"), default="auto",
                   help="recompute each block in the backward pass instead of storing its activations: much less "
                        "memory, 21-32 %% more step time (auto: only when the model would not otherwise fit with a "
                        "micro-batch of about 2,048 tokens, 2 sequences at ctx 1024; on Apple GPUs (MPS) at ctx >= "
                        "2048 auto prefers it, since the attention scores dominate memory); may be changed on "
                        "--resume")
    g.add_argument("--loss-chunk-tokens", type=int,
                   help="compute the output layer and loss this many tokens at a time, so the full tokens x "
                        "vocabulary table never exists (0 = off; default: auto, %d when it is needed to fit); may be "
                        "changed on --resume" % LOSS_CHUNK_TOKENS)
    g.add_argument("--optimizer", choices=("adamw", "muon"),
                   help="adamw (default), or muon: Muon for the matrices inside the blocks and AdamW for the embedding "
                        "and norm gains, both driven by --lr and the schedule (Muon's step is scaled to AdamW's "
                        "update size); resumed runs keep their optimizer")
    g.add_argument("--z-loss", type=float,
                   help="weight of the z-loss stability term (0 disables; resumed runs keep their setting)")
    g.add_argument("--ema", metavar="DECAY|on|off",
                   help="exponential moving average of the weights (fp32, on the CPU, debiased so it never holds the "
                        "initial weights), scored beside the raw weights at each evaluation and saved as "
                        "<name>_best.pt when it is the better of the two (default off; on = %g per step; a resumed "
                        "run keeps its setting unless this is given: a decay replaces the run's decay and its saved "
                        "average continues, a run without one starts it from the current weights; off removes it)"
                        % EMA_DECAY_DEFAULT)
    g.add_argument("--doc-mask", choices=("on", "off"),
                   help="confine attention to each document of a training window (documents start with BOS): default "
                        "off; it is the training objective, so fixed for a run and ignored on --resume")
    g = p.add_argument_group("evaluation and logging")
    g.add_argument("--eval-every", type=int, default=200, help="steps between validation + metrics line")
    g.add_argument("--eval-tokens", type=int,
                   help="fixed validation tokens per source (default: 65,536, more at >= 1M tokens/step: about 0.5%% "
                        "of the tokens trained between evaluations; fixed for the run, so ignored on --resume)")
    g.add_argument("--heartbeat-seconds", type=int, default=120,
                   help="print a short training-only progress line when nothing was printed for this long (0 = off)")
    g.add_argument("--log-every", "--report-every", type=int, default=0,
                   help="extra training-only progress lines between evaluations (0 = none)")
    g.add_argument("--dashboard", choices=("auto", "on", "off"), default="auto",
                   help="redraw view_pt.py's dashboard of the best checkpoint in the background after each "
                        "evaluation (auto: for runs saved in this script's folder; test runs elsewhere are skipped)")
    g.add_argument("--dashboard-push-hours", type=float, default=12.0,
                   help="push the dashboard to GitHub at most this often (0 = every time; default 12, since every "
                        "1.6 MB version stays in the repository's history)")
    g.add_argument("--sample-every", type=int,
                   help="write a sample at evaluations whose step is a multiple of N (default: every evaluation; 0 = off)")
    g.add_argument("--sample-tokens", type=int, default=48)
    g.add_argument("--sample-prompt", default="Genetic drift is")
    g.add_argument("--save-every", type=int, default=1000, help="steps between training-state saves")
    g.add_argument("--save-minutes", type=float, default=30.0,
                   help="also save the training state after this many minutes (0 = off), so a shutdown loses little")
    g.add_argument("--best-bf16", action="store_true",
                   help="store the inference-only <name>_best.pt weights in bfloat16 (half the file; the training "
                        "state stays float32): for models of about 1B parameters and more")
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
    g.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                   help="auto: CUDA, else Apple MPS, else the CPU")
    g.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto",
                   help="auto: BF16 autocast on CUDA (FP16 with gradient scaling on a GPU without BF16) and on MPS "
                        "where this PyTorch supports it, FP32 on the CPU")
    g.add_argument("--compile", nargs="?", const="on", choices=("auto", "on", "off"), default="auto",
                   help="torch.compile (auto: on CUDA when Triton is installed, about 1.45x faster; on MPS only with "
                        "--compile on, falling back to eager when the graph fails)")
    g.add_argument("--no-compile", dest="compile", action="store_const", const="off", help="same as --compile off")
    g.add_argument("--attn", choices=("auto", "gqa_native", "repeat_kv"), default="auto",
                   help="grouped-query attention path (auto: on CUDA the faster of the two, timed once per head shape; "
                        "repeat_kv elsewhere)")
    g.add_argument("--threads", type=int, help="CPU threads for PyTorch")
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
    argv = sys.argv[1:] if argv is None else list(argv)
    # --data went with the options of the original script; argparse would now read it as an abbreviation of
    # --dataset and take an old '--data <md folder>' for a prepared dataset, so it is refused like the others
    removed = [item for item in argv if item == "--data" or item.startswith("--data=")]
    if removed:
        p.error("unrecognized arguments: " + " ".join(removed))
    args = p.parse_args(argv)
    validate_args(args, p)
    return args


def validate_args(args, parser):
    if not NAME_RE.fullmatch(args.name):
        parser.error("--name may contain letters, digits, '.', '_' and '-' only (no folders)")
    positive = ["steps", "d_model", "layers", "heads", "kv_heads", "ctx", "ffn_hidden", "batch_size", "grad_accum",
                "tokens_per_step", "threads", "eval_every", "eval_tokens", "save_every", "sample_tokens", "length",
                "experts_per_token", "moe_every", "stop_after_steps"]
    for name in positive:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name, lo, hi in (("lr", 1e-7, 1.0), ("min_lr_ratio", 0.0, 1.0), ("decay_frac", 0.0, 0.9),
                         ("weight_decay", 0.0, 1.0), ("beta1", 0.0, 0.9999), ("beta2", 0.0, 0.99999),
                         ("grad_clip", 0.0, 1e3), ("dropout", 0.0, 0.9), ("top_p", 0.0, 1.0),
                         ("tokens_per_param", 0.1, 1e6), ("max_epochs", 0.01, 1e4), ("temperature", 0.0, 10.0),
                         ("mix_alpha", 0.0, 2.0), ("lr_scale", 0.01, 100.0), ("lb_loss", 0.0, 10.0),
                         ("router_z_loss", 0.0, 10.0), ("bias_rate", 0.0, 1.0), ("tokens_per_total_param", 0.0, 1e6),
                         ("expert_hidden", 0, 1e6), ("shared_hidden", 0, 1e6), ("rope_base", 1, 1e9),
                         ("z_loss", 0, 10)):
        value = getattr(args, name)
        if value is not None and not lo <= value <= hi:
            parser.error(f"--{name.replace('_', '-')} must be between {lo} and {hi}")
    if args.experts and not 2 <= args.experts <= 64:
        parser.error("--experts must be 0 (dense) or between 2 and 64")
    if args.experts and args.experts_per_token > args.experts:
        parser.error("--experts-per-token must not exceed --experts")
    if not 0 < args.watermark_gamma < 1:
        parser.error("--watermark-gamma must be between 0 and 1")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0 (0 = no cut)")
    if args.ema is not None:  # "off", "on" or a per-step decay; None = the default for the run (see resolve_ema)
        if str(args.ema).lower() == "off":
            args.ema = 0.0
        elif str(args.ema).lower() == "on":
            args.ema = EMA_DECAY_DEFAULT
        else:
            try:
                args.ema = float(args.ema)
            except ValueError:
                parser.error("--ema must be a decay such as 0.999, on or off")
            if not 0.5 <= args.ema <= 0.99999:
                parser.error("--ema must be a decay between 0.5 and 0.99999, or off")
    if args.time_budget_hours is not None and args.steps is not None:
        parser.error("give either --steps or --time-budget-hours (the budget chooses the number of steps)")
    if args.time_budget_hours is not None and not 0.01 <= args.time_budget_hours <= 24 * 90:
        parser.error("--time-budget-hours must be between 0.01 and 2160")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.warmup_steps is not None and args.steps is not None and args.warmup_steps >= args.steps:
        parser.error("--warmup-steps must be smaller than --steps")
    if args.schedule == "wsd" and args.decay_frac == 0 and not args.resume:  # on --resume the flag is ignored
        print("WARNING: --decay-frac 0 with wsd: the learning rate never decays (warmup-stable only)")
    if args.log_every < 0 or (args.sample_every is not None and args.sample_every < 0):
        parser.error("--log-every/--sample-every must be >= 0")
    if args.resume and args.init_from:
        parser.error("use either --resume or --init-from, not both")
    if any(value is not None and not math.isfinite(value) for value in (args.train_tokens, args.epochs)):
        parser.error("--train-tokens/--epochs must be finite numbers")
    if args.train_tokens is not None and args.train_tokens <= 0 or args.epochs is not None and args.epochs <= 0:
        parser.error("--train-tokens/--epochs must be positive")
    if sum(x is not None for x in (args.steps, args.train_tokens, args.epochs)) > 1:
        parser.error("give only one of --steps, --train-tokens, --epochs")
    args.out_dir = os.path.abspath(args.out_dir)
    if args.loss_chunk_tokens is not None and not (args.loss_chunk_tokens == 0 or 64 <= args.loss_chunk_tokens):
        parser.error("--loss-chunk-tokens must be 0 (off) or at least 64")
    if not os.path.isdir(args.out_dir):
        parser.error(f"--out-dir does not exist: {args.out_dir}")
    if args.dataset and not os.path.isdir(args.dataset):
        parser.error(f"--dataset folder not found: {args.dataset}")
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
    header, kept, offset = {}, [], 0
    for name in sorted(tensors):
        t = tensors[name].detach().cpu().contiguous()
        if t.dtype not in _ST_DTYPES:
            raise ValueError(f"unsupported dtype {t.dtype} for {name}")
        size = t.numel() * t.element_size()
        header[name] = {"dtype": _ST_DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + size]}
        kept.append(t)
        offset += size
    header["__metadata__"] = {str(k): str(v) for k, v in (metadata or {}).items()}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for t in kept:  # each tensor's own memory, without a bytes copy of the model (numpy has no bfloat16)
            flat = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).reshape(-1)
            handle.write(memoryview(flat.numpy()).cast("B"))
    os.replace(tmp, path)


def read_safetensors(path):
    import struct
    by_name = {v: k for k, v in _ST_DTYPES.items()}
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 100 * 2 ** 20:
            raise SystemExit(f"{path}: implausible safetensors header ({size:,} bytes)")
        header = json.loads(handle.read(size))
        metadata = header.pop("__metadata__", {})
        base = 8 + size
        file_len = os.fstat(handle.fileno()).st_size
        tensors = {}
        for name, info in header.items():
            begin, end = info["data_offsets"]
            if not 0 <= begin <= end <= file_len - base or info["dtype"] not in by_name:
                raise SystemExit(f"{path}: corrupt entry {name}")
            dtype = by_name[info["dtype"]]
            if begin == end:  # torch.frombuffer refuses an empty buffer
                tensors[name] = torch.empty(info["shape"], dtype=dtype)
                continue
            raw = bytearray(end - begin)  # read into the tensor's own memory, without holding the whole file too
            handle.seek(base + begin)
            if handle.readinto(raw) != end - begin:
                raise SystemExit(f"{path}: corrupt entry {name}")
            buf = torch.frombuffer(raw, dtype=torch.int16 if dtype == torch.bfloat16 else dtype)
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
            "metrics": config.get("metrics", {}), "exported_from": config.get("source_checkpoint"),
            "dataset": config.get("dataset"), "train_config": config.get("train_config"),
            "weights": config.get("weights", "raw")}


def load_any(path, trust=False, mmap=False):
    """A .pt checkpoint (safe loader; mmap as in safe_load) or an --export folder."""
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "model.safetensors")):
        return load_exported(path)
    return safe_load(path, trust, mmap)


def _home_short(path):
    """`path` with the home folder shown as ~, so a file meant for sharing does not carry the user name.
    For one path: only a home folder at its start, as a whole folder (_hide_home, for the text of the experiments
    table, replaces the home string wherever it stands)."""
    home = os.path.expanduser("~")
    if os.path.normcase(path).startswith(os.path.normcase(home) + os.sep):
        return "~" + path[len(home):]
    return path


def resolve_checkpoint(args, hint):
    """The checkpoint --export, --benchmark and --generate work on: --checkpoint, else this run's best file,
    else its training checkpoint, as a normalised path (an export folder typed with a trailing separator is the
    folder itself). When it does not exist: SystemExit with `hint` and the other runs of the folder."""
    ckpt_path, best_path, _ = checkpoint_paths(args)
    path = os.path.normpath(args.checkpoint or (best_path if os.path.exists(best_path) else ckpt_path))
    if not os.path.exists(path):
        raise SystemExit(f"No checkpoint {path}; {hint}.{_resume_hint(args, flag='')}")
    return path


def cmd_export(args):
    """Write <folder>/model.safetensors + config.json + tokenizer.model."""
    path = resolve_checkpoint(args, "pass --checkpoint")
    out = os.path.abspath(args.export)
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty; choose a new folder.")
    model, tok, cfg, obj = load_for_inference(path, "cpu", args.trust_checkpoint)  # one copy of the weights
    obj.pop("optimizer", None)
    tensors = {k: v for k, v in model.state_dict().items() if not (cfg.tie_embeddings and k == "lm_head.weight")}
    os.makedirs(out, exist_ok=True)
    write_safetensors(tensors, os.path.join(out, "model.safetensors"),
                      {"format": "tinyGPT", "tie_embeddings": cfg.tie_embeddings, "qk_norm": cfg.qk_norm,
                       "n_experts": cfg.n_experts})
    with open(os.path.join(out, "tokenizer.model"), "wb") as handle:
        handle.write(tok.proto)
    metrics = obj.get("metrics", {}) if isinstance(obj.get("metrics"), dict) else {}
    config = {"architecture": "tinyGPT decoder-only Transformer: interleaved-pair RoPE, RMSNorm, SwiGLU, "
                              + (f"grouped-query attention ({cfg.n_heads} query / {cfg.n_kv_heads} key-value heads)"
                                 if cfg.n_kv_heads < cfg.n_heads else "multi-head attention")
                              + (", QK-norm (RMSNorm with a shared gain on each query and key head, before RoPE)"
                                 if cfg.qk_norm else "")
                              + (f", mixture of experts ({cfg.n_experts} SwiGLU experts of hidden "
                                 f"{cfg.expert_hidden} per MoE block, {cfg.experts_per_token} per token chosen by a "
                                 f"{cfg.router} router with a balancing bias"
                                 + (f", every {cfg.moe_every}. block" if cfg.moe_every > 1 else "")
                                 + (f", plus a shared SwiGLU of hidden {cfg.shared_hidden}" if cfg.shared_hidden
                                    else "") + ")" if cfg.n_experts else "")
                              + (", tied input/output embeddings" if cfg.tie_embeddings else ", untied output layer")
                              + ", no biases",
              "model_config": dataclasses.asdict(cfg), "tokenizer": {**tok.meta(), "file": "tokenizer.model"},
              "source_checkpoint": _home_short(os.path.abspath(path)), "step": obj.get("step"),
              "weights": obj.get("weights", "raw"),  # "ema": the averaged weights a best file may hold
              "metrics": {"best_val": metrics.get("best_val", obj.get("best_val")),
                          "best_step": metrics.get("best_step", obj.get("best_val_step"))},
              "exported": now_iso()}
    dataset = obj.get("dataset")
    if isinstance(dataset, dict) and dataset.get("path"):
        # the folder's name and the fingerprint find the dataset again (resolve_dataset); no folder of this machine.
        # The name by resolve_dataset's rule: the path may have been stored on the other kind of system
        config["dataset"] = {"path": re.split(r"[\\/]", str(dataset["path"]).rstrip("\\/"))[-1],
                             "fingerprint": dataset.get("fingerprint")}
    if obj.get("train_config"):
        config["train_config"] = obj["train_config"]
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
    path = resolve_checkpoint(args, "pass --checkpoint")
    probes_file = args.probes or DEFAULT_PROBES
    if not os.path.isfile(probes_file):
        raise SystemExit(f"Probe file not found: {probes_file}")
    device = select_device(args.device)
    amp_dtype, _, _ = choose_precision(device, args.precision, training=False)
    model, tok, cfg, obj = load_for_inference(path, device, args.trust_checkpoint)
    model.set_attention_impl(choose_attention(device, cfg, amp_dtype, args.attn)[0])
    probes = read_probes(probes_file)
    if not probes:
        raise SystemExit(f"{probes_file}: no probes found")
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
        "event": "benchmark", "name": os.path.basename(path), "step": obj.get("step"),
        "params": count_params(cfg)[0], "d_model": cfg.d_model, "layers": cfg.n_layers, "ctx": cfg.ctx,
        "vocab": cfg.vocab_size,
        "benchmark_accuracy": f"{acc:.3f}", "benchmark_floor": f"{floor:.3f}", "benchmark_items": n,
        "checkpoint": path, "notes": f"95% interval {low:.3f}-{high:.3f}; chance {chance:.3f}; "
                                    f"{os.path.basename(probes_file)}; items in {os.path.basename(out)}"})


def cmd_generate(args):
    path = resolve_checkpoint(args, "train first or pass --checkpoint")
    device = select_device(args.device)
    amp_dtype, amp_name, _ = choose_precision(device, args.precision, training=False)
    model, tok, cfg, obj = load_for_inference(path, device, args.trust_checkpoint)
    impl, _ = choose_attention(device, cfg, amp_dtype, args.attn)
    model.set_attention_impl(impl)
    family, kind = checkpoint_info(obj)
    metrics = obj.get("metrics", {}) if family == "tinygpt" else {}
    best = metrics.get("best_val", obj.get("best_val"))
    best_step = metrics.get("best_step", obj.get("best_val_step"))
    step = obj.get("step")
    step_s = f"{step:,}" if isinstance(step, int) else "?"
    print(f"Checkpoint {path} ({family}, {kind}) | step {step_s} | best validation loss "
          f"{best:.4f} @{best_step} | {device_label(device)} | {amp_name}" if best is not None else
          f"Checkpoint {path} ({family}, {kind}) | step {step_s}")
    if family != "tinygpt":
        print("Note: legacy checkpoints use the old tokenizer (runs of spaces collapse; no end-of-document token).")
    gen = torch.Generator().manual_seed(args.seed)
    watermark = (args.watermark_key, args.watermark_gamma, args.watermark_delta) if args.watermark_key else None
    text = generate(model, tok, args.generate, args.length, device, amp_dtype, args.temperature, args.top_k,
                    args.top_p, gen, watermark)
    print(f"\n{args.generate}{text}" if args.generate and not text.startswith(args.generate) else f"\n{text}")


def ensure_dataset(args):
    """--dataset as an absolute path, or None (a resumed run takes the path from its checkpoint in _train)."""
    return os.path.abspath(args.dataset) if args.dataset else None


def resume_dataset_dir(given, obj):
    """The dataset folder of a resumed run as an absolute path: --dataset when it is given, else the folder
    resolve_dataset finds for the checkpoint (its stored path, tried from this script's folder too, or
    datasets/<name> with the same fingerprint). The checkpoint stores what is returned here, so a relative
    --dataset does not make the run depend on the folder it was resumed from. A dataset that is found nowhere
    keeps the stored text: _train's own messages then explain a missing or a rebuilt dataset."""
    if given:
        return os.path.abspath(given)
    try:
        return os.path.abspath(resolve_dataset(obj))
    except SystemExit:
        return obj["dataset"]["path"]


def _resume_hint(args, flag=" --resume"):
    """Name other training checkpoints in the folder (e.g. runs made before the rename to tinyGPT). `flag` ends
    the example: " --resume" after a failed --resume, nothing for a command that only reads a checkpoint."""
    found = []
    for name in sorted(os.listdir(args.out_dir)):
        if name.endswith(".pt") and not name.endswith("_best.pt") and "_crash" not in name:
            found.append(name[:-3])
    return (" Training checkpoints in this folder: " + ", ".join(found) +
            ". Pass the right one with --name, e.g. --name " + found[0] + flag) if found else ""


def newer_fallbacks(path):
    """Files that atomic_save wrote instead of `path` while it was locked by another
    program (<base>.HHMMSS.pt) and that are newer than `path`, oldest first."""
    folder, name = os.path.split(path)
    pattern = re.compile(re.escape(os.path.splitext(name)[0]) + r"\.\d{6}\.pt")
    since = os.path.getmtime(path) if os.path.exists(path) else 0.0
    found = [os.path.join(folder, f) for f in os.listdir(folder or ".") if pattern.fullmatch(f)]
    return sorted((f for f in found if os.path.getmtime(f) > since), key=os.path.getmtime)


# Signals that pause a training run like Ctrl+C (cmd_train gives them Python's Ctrl+C handler): kill, a closed
# terminal, logout. Ctrl+Break (SIGBREAK) is not among them: it stays the one unconditional stop on Windows.
PAUSE_SIGNALS = ("SIGTERM", "SIGHUP")


class DeferInterrupt:
    """Holds Ctrl+C (and the PAUSE_SIGNALS that cmd_train treats like it) while the
    optimizer update runs, so a step is applied in full or not at all (an interrupt
    halfway through Muon's or AdamW's update would save a partly updated model under
    the previous step number); deliver() raises the held KeyboardInterrupt afterwards.
    Only possible in the main thread, and only for a signal that has Python's own
    Ctrl+C handler; otherwise it does nothing."""

    def __init__(self):
        self.hit, self.old = False, {}

    def _hold(self, signum, frame):
        self.hit = True

    def __enter__(self):
        self.old = {}
        try:
            for name in ("SIGINT",) + PAUSE_SIGNALS:
                number = getattr(signal, name, None)
                if number is not None and signal.getsignal(number) is signal.default_int_handler:
                    self.old[number] = signal.signal(number, self._hold)
        except ValueError:  # not the main thread
            pass
        return self

    def __exit__(self, *exc):
        for number, handler in self.old.items():
            signal.signal(number, handler)
        self.old = {}
        return False

    def deliver(self):
        if self.hit:
            self.hit = False
            raise KeyboardInterrupt


def acquire_run_lock(args):
    """Mark <out_dir>/<name> as being trained by this process (an exclusive
    <name>.train.lock holding the process id), so two trainers never write the
    same run's checkpoints. A lock whose process has ended is stale and replaced."""
    os.makedirs(args.out_dir, exist_ok=True)
    lock = os.path.join(args.out_dir, f"{args.name}.train.lock")
    for attempt in (1, 2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if attempt == 2 or data_prep._lock_owner_alive(lock):
                raise SystemExit(f"Run {args.name!r} in {args.out_dir} is being trained by another process (see "
                                 f"{lock}). Wait for it to finish or stop it first; delete the lock file only if "
                                 "no training is running.")
            os.remove(lock)  # stale: its run ended without removing it
    started = None
    try:
        import psutil
        started = psutil.Process().create_time()
    except Exception:  # noqa: BLE001 - without psutil only the process id is checked
        pass
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started": started}, handle)
    return lock


def release_run_lock(lock):
    """Remove a lock acquire_run_lock returned (only if it is still this process's)."""
    data_prep._release_lock(lock)


def cmd_train(args):
    for name in PAUSE_SIGNALS:  # kill, a closed terminal and logout pause the run (KeyboardInterrupt) like Ctrl+C
        if hasattr(signal, name):
            try:
                # a signal the run was started to ignore stays ignored: nohup leaves SIGHUP so, to keep the run
                # going when its terminal closes
                if signal.getsignal(getattr(signal, name)) is not signal.SIG_IGN:
                    signal.signal(getattr(signal, name), signal.default_int_handler)
            except (ValueError, OSError):  # not the main thread, or the platform refuses the signal
                pass
    lock = acquire_run_lock(args)
    try:
        # another run (another --name) training here: said, not refused, and from the lock files alone, so that a
        # plain --resume is never stopped or warned by what else the desktop is doing
        if args.device != "cpu" and gpu_busy(args.out_dir, HERE, smi=False):
            print("WARNING: " + GPU_BUSY_MESSAGE)
        return _train(args)
    finally:
        release_run_lock(lock)


def _train(args):
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
    cap_device_memory(device)  # MPS: fail fast with out-of-memory instead of paging to the SSD
    amp_dtype, amp_name, need_scaler = choose_precision(device, args.precision)

    # ---------------- resume / init / fresh ----------------
    resume_obj = init_obj = None
    from_best = False  # resuming from <name>_best.pt because no training state was ever saved
    if args.resume:
        late = newer_fallbacks(ckpt_path)
        if late:
            raise SystemExit(f"{late[-1]} is a newer training state than {ckpt_path}: it was saved under that name "
                             f"because {os.path.basename(ckpt_path)} was locked by another program. Rename it to "
                             f"{ckpt_path} (replacing the older file) and resume again, or delete it to resume from "
                             "the older state.")
        for late_best in newer_fallbacks(best_path):
            print(f"Note: {late_best} is a newer best model than {best_path} (saved there while the file was "
                  "locked); rename it over the best file to use it.")
        if os.path.exists(ckpt_path):
            resume_path = ckpt_path
        elif os.path.exists(best_path):
            resume_path, from_best = best_path, True
        else:
            raise SystemExit(f"Nothing to resume: {ckpt_path} does not exist.{_resume_hint(args)}")
        resume_obj = safe_load(resume_path, args.trust_checkpoint, mmap=True)
        family, kind = checkpoint_info(resume_obj)
        wanted = "inference" if from_best else "training"
        if family != "tinygpt" or kind != wanted or "train_config" not in resume_obj:
            raise SystemExit(f"{resume_path} is a {family} {kind} checkpoint; --resume needs a tinyGPT checkpoint "
                             "written by a training run. Start a new run from its weights with --init-from instead.")
        dataset_dir = resume_dataset_dir(args.dataset, resume_obj)
    else:
        existing = [p for p in (ckpt_path, best_path) if os.path.exists(p)]
        if existing and not args.overwrite:
            raise SystemExit(f"{existing[0]} already exists. Use --resume to continue it, a new --name, "
                             "or --overwrite (existing files are renamed, not deleted).")
        dataset_dir = ensure_dataset(args)
        if args.init_from:
            init_obj = safe_load(args.init_from, args.trust_checkpoint, mmap=True)
    if not dataset_dir:
        raise SystemExit("No training data: pass --dataset <folder from data_prep.py>.")
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"Dataset folder not found: {dataset_dir} (moved? pass --dataset).")
    # said before the silent part (hashing the token files, building and compiling the model), and before the
    # header's bar: view_pt splits the sessions of a log at the bar followed by 'RUN '
    source_path = resume_path if args.resume else args.init_from
    print((f"loading {os.path.basename(os.path.normpath(source_path))}, " if source_path else "")
          + "checking the dataset and compiling the model ...")
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
        cfg = config_from_checkpoint(resume_obj)  # a clear error for a file written by a newer tiny_gpt.py
        settings = dict(resume_obj["train_config"])
        settings.setdefault("optimizer", "adamw")  # runs from before --optimizer used AdamW
        if cfg.n_experts:  # the mixture-of-experts terms; fixed for a run like z_loss
            settings.setdefault("lb_loss", args.lb_loss)
            settings.setdefault("router_z_loss", args.router_z_loss)
            settings.setdefault("bias_rate", args.bias_rate)
        # memory options do not change the mathematics, so a resumed run may switch them (e.g. on another machine)
        settings.setdefault("activation_checkpointing", False)
        settings.setdefault("loss_chunk_tokens", 0)
        settings.setdefault("lr_scale", 1.0)  # runs planned before --lr-scale used the table rate as it stands
        settings.setdefault("doc_mask", False)  # runs from before the mask attended across documents
        resolve_ema(settings, args.ema, resumed=True)  # off unless the checkpoint or --ema says otherwise
        if args.activation_checkpointing != "auto":
            settings["activation_checkpointing"] = args.activation_checkpointing == "on"
        if args.loss_chunk_tokens is not None:
            settings["loss_chunk_tokens"] = args.loss_chunk_tokens
        if args.optimizer and args.optimizer != settings["optimizer"]:
            print(f"Note: --optimizer {args.optimizer} ignored on --resume: this run uses {settings['optimizer']} "
                  "(start a new run, or --init-from this one, to change it).")
        if args.batch_size or args.grad_accum:
            # the micro-batch is a memory choice too: a resumed run may trade micro-batch for accumulation (2 x 132
            # instead of 4 x 66 with recomputation) as long as every step still sees the same number of tokens
            tps = int(settings["tokens_per_step"])
            old_micro, old_accum = int(settings["micro_batch"]), int(settings["grad_accum"])
            new_micro = int(args.batch_size or old_micro)
            new_accum = int(args.grad_accum or tps // (new_micro * cfg.ctx))
            if new_micro * new_accum * cfg.ctx != tps:
                raise SystemExit(f"tokens per step must stay {tps:,} on --resume: use --batch-size B --grad-accum "
                                 f"{tps:,}//(B*{cfg.ctx}) (this run: {old_micro} x {old_accum}; asked: "
                                 f"{new_micro} x {new_accum} = {new_micro * new_accum * cfg.ctx:,} tokens).")
            if (new_micro, new_accum) != (old_micro, old_accum):
                settings["micro_batch"], settings["grad_accum"] = new_micro, new_accum
                print(f"Note: micro-batch {old_micro} x {old_accum} -> {new_micro} x {new_accum} (same {tps:,} tokens "
                      "per step; the random windows drawn after this resume differ from an uninterrupted run, "
                      "same mixture).")
        ignored = [f"--{k.replace('_', '-')}" for k in ("d_model", "layers", "heads", "kv_heads", "ctx",
                                                        "ffn_hidden", "rope_base", "lr",
                                                        "dropout", "warmup_steps",
                                                        "epochs", "train_tokens", "time_budget_hours",
                                                        "tokens_per_step")
                   if getattr(args, k) is not None]
        # options with a default: named when the value is neither that default (not typed, as far as can be told)
        # nor the run's own, so a plain --resume of a run started under an older default says nothing
        betas = list(settings.get("betas") or (None, None))
        kept = {**{k: settings.get(k) for k in FIXED_OPTION_DEFAULTS}, "beta1": betas[0], "beta2": betas[1]}
        ignored += [f"--{k.replace('_', '-')}" for k, default in FIXED_OPTION_DEFAULTS.items()
                    if getattr(args, k) != default and getattr(args, k) != kept[k]]
        if args.tie_embeddings == "off" and cfg.tie_embeddings:
            ignored.append("--tie-embeddings off")
        if args.qk_norm is not None and (args.qk_norm == "on") != cfg.qk_norm:
            ignored.append(f"--qk-norm {args.qk_norm}")
        if args.experts and args.experts != cfg.n_experts:
            ignored.append(f"--experts {args.experts} (the architecture is the checkpoint's)")
        for flag, value, have in (("--router", args.router, cfg.router), ("--balance", args.balance, cfg.balance)):
            if value is not None and value != have:  # model fields like the experts: fixed for the run
                ignored.append(f"{flag} {value} (the checkpoint's is {have})")
        if args.lr_scale != 1.0:
            ignored.append(f"--lr-scale {args.lr_scale:g}")
        if args.doc_mask is not None and (args.doc_mask == "on") != bool(settings["doc_mask"]):
            ignored.append(f"--doc-mask {args.doc_mask} (the objective is fixed for a run)")
        if ignored:
            print(f"Note: {', '.join(ignored)} ignored on --resume (the checkpoint's settings are kept).")
        if args.steps and args.steps != settings["steps"]:
            if args.steps <= int(settings["warmup_steps"]):
                raise SystemExit(f"--steps {args.steps:,} is not above this run's warmup of "
                                 f"{int(settings['warmup_steps']):,} steps; the schedule needs at least one step "
                                 "after the warmup.")
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
        if args.min_lr_ratio is not None:
            applied, note = resume_min_lr(settings, int(resume_obj["step"]), args.min_lr_ratio)
            print(note)
            if applied:
                print("schedule now: " + LRSchedule(settings["schedule"], settings["steps"], settings["lr"],
                                                    settings["min_lr"], settings["warmup_steps"],
                                                    settings["decay_frac"], settings["decay_shape"]).describe())
    elif init_obj is not None:
        # --ctx may change the context (the weights do not depend on it); --time-budget-hours then sets how much of
        # the new data is read
        stored_ctx = config_from_checkpoint(init_obj).ctx
        cfg = config_from_checkpoint(init_obj, ctx=args.ctx)
        # the planner sizes memory, batch, steps and learning rate for exactly this architecture (experts included)
        # verbose=False here and for a fresh run: the plan lines without what the run header says itself
        _, settings, plan_lines = plan_with_time_budget(args, manifest, device, amp_dtype, fixed_cfg=cfg,
                                                        verbose=False)
        if args.experts and args.experts != cfg.n_experts:
            plan_lines.append(f"--experts {args.experts} ignored: the architecture is the checkpoint's "
                              f"({cfg.n_experts or 'no'} experts); start a fresh run to change it")
        for flag, value, have in (("--router", args.router, cfg.router), ("--balance", args.balance, cfg.balance)):
            if value is not None and value != have:
                plan_lines.append(f"{flag} {value} ignored: the checkpoint's model keeps {have}")
        for flag, value, have in (("--d-model", args.d_model, cfg.d_model), ("--layers", args.layers, cfg.n_layers),
                                  ("--heads", args.heads, cfg.n_heads), ("--kv-heads", args.kv_heads, cfg.n_kv_heads),
                                  ("--ffn-hidden", args.ffn_hidden, cfg.ffn_hidden),
                                  ("--rope-base", args.rope_base, cfg.rope_base)):
            if value is not None and value != have:
                plan_lines.append(f"{flag} {value:g} ignored: the checkpoint's model keeps {have:g}")
        if args.tie_embeddings == "off" and cfg.tie_embeddings:
            plan_lines.append("--tie-embeddings off ignored: the checkpoint's model keeps its tied embedding")
        if cfg.ctx != stored_ctx:
            plan_lines.append(f"model was trained at ctx {stored_ctx}; continuing at {cfg.ctx} (RoPE tables rebuilt "
                              "for the new length, weights unchanged)")
        if cfg.rope_base < default_rope_base(cfg.ctx):
            # the weights were trained with this base, so it is not changed; a new run at this context would get more
            plan_lines.append(f"RoPE base stays at the checkpoint's {cfg.rope_base:,.0f}, below the rule for ctx "
                              f"{cfg.ctx} ({default_rope_base(cfg.ctx):,.0f}): long-range retrieval may be weaker "
                              "than in a run started at this context")
        cfg.dropout = args.dropout if args.dropout is not None else 0.0
        if args.qk_norm is not None and (args.qk_norm == "on") != cfg.qk_norm:
            plan_lines.append(f"--qk-norm {args.qk_norm} ignored: the architecture is the checkpoint's (QK-norm "
                              f"{'on' if cfg.qk_norm else 'off'}); start a fresh run to change it")
        plan_lines.append(f"architecture taken from --init-from {args.init_from}; dropout {cfg.dropout} "
                          f"({'set' if args.dropout is not None else 'default 0 when continuing from weights'}, "
                          f"not the planned value above)")
        if init_obj.get("optimizer") and checkpoint_info(init_obj)[1] == "training" and not args.steps:
            # a training state continues its schedule at its own step, so the planned steps come after that step
            init_step = int(init_obj.get("step", 0) or 0)
            settings["steps"] += init_step
            plan_lines.append(f"steps: the schedule continues at the --init-from checkpoint's step {init_step:,}, "
                              f"so the planned steps run from there: {settings['steps']:,} in total")
    else:
        cfg, settings, plan_lines = plan_with_time_budget(args, manifest, device, amp_dtype, verbose=False)
    if cfg.vocab_size != tok.vocab_size:
        raise SystemExit(f"Model vocabulary {cfg.vocab_size} != dataset tokenizer {tok.vocab_size}.")
    total_steps = int(settings["steps"])
    micro, accum = int(settings["micro_batch"]), int(settings["grad_accum"])
    mix_text = args.mix if resume_obj is None else (args.mix or settings.get("mix"))
    if args.mix_alpha is not None:
        mix_alpha = float(args.mix_alpha)
    else:  # runs from before --mix-alpha sampled in proportion to size (alpha 1)
        mix_alpha = float(settings.get("mix_alpha", 1.0)) if resume_obj is not None else 1.0
    mix = mix_weights(mix_text, arrays["train"], {n: len(a) for n, a in arrays["train"].items()}, mix_alpha)
    byte_tables = tuple(torch.from_numpy(t) for t in tok.byte_tables())
    # a new run: 1e-4 unless --z-loss says otherwise; a resumed run from before the setting existed had none
    z_loss = float(settings.setdefault("z_loss", (1e-4 if args.z_loss is None else args.z_loss)
                                       if resume_obj is None else 0.0))
    if resume_obj is not None and args.z_loss is not None and args.z_loss != settings["z_loss"]:
        print(f"Note: --z-loss ignored on --resume; this run keeps z_loss {settings['z_loss']:g}.")
    if resume_obj is None and args.eval_tokens is None:  # new run: the size follows the tokens between evaluations
        largest_val = max((len(a) - 1 for a in arrays["val"].values()), default=0)
        args.eval_tokens = default_eval_tokens(args.eval_every, int(settings["tokens_per_step"]), len(arrays["val"]),
                                               largest_val)
    # fixed windows stay fixed on resume; runs from before the key existed used the old default of 65,536
    eval_tokens = int(settings.setdefault("eval_tokens",
                                          65_536 if resume_obj is not None else args.eval_tokens or 65_536))
    if resume_obj is not None and args.eval_tokens is not None and eval_tokens != args.eval_tokens:
        print(f"Note: --eval-tokens ignored on --resume; this run keeps {eval_tokens:,} (its validation windows).")
    # hashed once: the settings keep the first 16 characters, the probe report of each evaluation the whole digest
    probes_full = file_sha256(args.probes) if args.probes else None
    probes_sha = probes_full[:16] if probes_full else None
    if resume_obj is not None and settings.get("probes_sha") and probes_sha and settings["probes_sha"] != probes_sha:
        print(f"Note: {os.path.basename(args.probes)} has changed since this run started: fact-probe accuracies "
              "from now on are not comparable with the earlier ones.")
    settings["probes_sha"] = probes_sha
    settings["mix"] = mix_text
    settings["mix_alpha"] = mix_alpha
    ema_decay = float(settings.get("ema") or 0.0)
    if settings.get("doc_mask") and tok.bos_id < 0:
        print("WARNING: --doc-mask needs a tokenizer with a BOS token (documents are delimited by it); this one has "
              "none, so attention spans the whole window.")
        settings["doc_mask"] = False

    # ---------------- model & optimizer ----------------
    torch.manual_seed(args.seed)  # planning and attention timing must not change the initial weights
    model = TinyGPT(cfg).to(device)
    attn_impl, attn_why = choose_attention(device, cfg, amp_dtype, args.attn)
    model.set_attention_impl(attn_impl)
    moe = bool(cfg.n_experts)
    moe_impl, moe_impl_why = "loop", "one matmul per expert"
    if moe and args.moe_impl == "grouped":
        ok, moe_impl_why = grouped_mm_available(device)
        moe_impl = "grouped" if ok else "loop"
    model.set_moe_impl(moe_impl)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)  # peak_mem_gib on the metrics lines counts from the built model
    source_obj = resume_obj or init_obj
    if source_obj is not None:
        missing, unexpected = model.load_state_dict(source_obj["model"], strict=False)
        missing = [m for m in missing if not (m == "lm_head.weight" and cfg.tie_embeddings)]
        if missing or unexpected:
            raise SystemExit(f"Checkpoint weights do not fit the model (missing {missing[:3]}, "
                             f"unexpected {unexpected[:3]}).")
    opt, opt_name, param_names, opt_note = build_optimizer(model, settings, device)
    model.grad_checkpoint = bool(settings.get("activation_checkpointing"))
    model.doc_sep_id = tok.bos_id if settings.get("doc_mask") else None
    loss_chunk = int(settings.get("loss_chunk_tokens") or 0)
    # mixture-of-experts terms added to the loss that is backpropagated (the reported train loss stays pure
    # cross-entropy): the Switch load-balancing loss when --balance is loss or both, the router z-loss always, and
    # DeepSeek-V3's bias update after every step when --balance is bias or both (TinyGPT.update_router_bias)
    lb_loss = float(settings.get("lb_loss", LB_LOSS_DEFAULT)) if moe else 0.0
    lb_weight = lb_loss if moe and cfg.balance in ("loss", "both") else 0.0
    router_z_loss = float(settings.get("router_z_loss", ROUTER_Z_LOSS_DEFAULT)) if moe else 0.0
    bias_rate = float(settings.get("bias_rate", BIAS_RATE_DEFAULT)) if moe and cfg.balance in ("bias", "both") else 0.0
    scaler = torch.amp.GradScaler(device.type) if need_scaler else None
    schedule = LRSchedule(settings["schedule"], total_steps, settings["lr"], settings["min_lr"],
                          settings["warmup_steps"], settings["decay_frac"], settings["decay_shape"])

    batcher = TokenBatcher(arrays["train"], mix, cfg.ctx, micro, device, args.seed)
    evalset = EvalSet(arrays["val"], cfg.ctx, eval_tokens)
    # evaluation keeps no gradients, so its batch is set by the memory budget, not by the training micro-batch
    eval_batch = default_eval_batch(micro, cfg.ctx, cfg.vocab_size, device_memory_budget(device)[0])

    def evaluate_retrying():
        """evaluate() with the training step's halve-on-out-of-memory retry: the evaluation batch may be larger
        than the micro-batch, so the first evaluation can be the first time this many windows meet the model."""
        nonlocal eval_batch
        was_training = model.training
        try:
            while True:
                try:
                    return evaluate(model, evalset, device, amp_dtype, eval_batch, byte_tables=byte_tables,
                                    share=batcher.share())
                except RuntimeError as exc:  # torch.cuda.OutOfMemoryError / MPS OOM are RuntimeErrors
                    if "out of memory" not in str(exc).lower() or eval_batch <= 1:
                        raise
                    # the traceback holds the frames of the failed forward pass, and they its activations: cleared
                    # first, or emptying the allocator returns nothing of them
                    traceback.clear_frames(exc.__traceback__)
                    free_device_memory(device)
                    eval_batch = max(1, eval_batch // 2)
                    print(f"Out of memory during evaluation; retrying with {eval_batch} windows at a time.")
        finally:
            model.train(was_training)
    probes = read_probes(args.probes) if args.probes else []

    start_step, tokens_seen, train_seconds = 0, 0, 0.0
    run_seconds_before = 0.0  # wall-clock time of earlier sessions of this run (before --resume)
    # learning-rate ramp after a resume from a best checkpoint, or an --init-from whose optimizer state partly
    # restarts
    rewarm_start, rewarm_steps = 0, 0
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
            rewarm_start, rewarm_steps = start_step, min(50, max(total_steps - start_step, 1))
            # stored, so a pause during the ramp resumes inside it instead of at the full learning rate
            settings["rewarm_start"], settings["rewarm_steps"] = rewarm_start, rewarm_steps
            batcher.gen.manual_seed(int(args.seed) + start_step)
            resume_note = (f"RESUMED FROM {best_path} at step {start_step:,}: no training state had been saved "
                           f"({ckpt_path} missing), so the optimizer's averages restart and the learning rate is re-warmed "
                           f"over {rewarm_steps} steps; weights, schedule position and history are kept")
        else:
            rewarm_start, rewarm_steps = int(settings.get("rewarm_start", 0)), int(settings.get("rewarm_steps", 0))
            restored, total_p = load_optimizer_by_name(opt, resume_obj["optimizer"],
                                                       resume_obj["optimizer_param_names"], param_names)
            if scaler is not None and resume_obj.get("scaler"):
                scaler.load_state_dict(resume_obj["scaler"])
            rng = resume_obj.get("rng", {})
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"].cpu())
            saved_state = rng.get(device.type) if device.type in ("cuda", "mps") else None
            if saved_state is not None and (device.type != "cuda" or len(saved_state)):
                try:
                    set_device_rng_state(device, saved_state)
                except (RuntimeError, TypeError) as exc:
                    print(f"{device.type.upper()} RNG state not restored ({exc}); only dropout masks differ.")
            batcher.load_state_dict(resume_obj.get("batcher"))
            interval_state = resume_obj.get("interval")
            if os.path.exists(best_path):
                try:
                    newer = safe_load(best_path, args.trust_checkpoint, mmap=True)
                    nm = newer.get("metrics", {})
                    if (newer.get("dataset", {}).get("fingerprint") == manifest["fingerprint"]
                            and float(nm.get("best_val", float("inf"))) < best_val):
                        print(f"Note: {best_path} holds a better model (val {float(nm['best_val']):.4f} "
                              f"@{nm.get('best_step')}) than this training state records; it is only replaced "
                              "by something better.")
                        best_val, best_step = float(nm["best_val"]), nm.get("best_step")
                    del newer
                except (Exception, SystemExit) as exc:  # noqa: BLE001 - safe_load reports a bad file as SystemExit
                    print(f"Note: could not read {best_path} ({exc}); it may be replaced by the next best.")
            # no best: every save of the best file so far failed and was tolerated
            best_text = f"{best_val:.4f} @{best_step}" if best_step is not None else "none"
            resume_note = (f"RESUMED {ckpt_path} at step {start_step:,} (optimizer state {restored}/{total_p} "
                           f"tensors; best {best_text}; last eval "
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
            if restored < total_p:
                # the tensors without state (a changed optimizer) start their running averages from zero at the
                # schedule's current learning rate: the same 50-step re-warm as a resume from a best checkpoint,
                # stored in the settings so a pause inside the ramp resumes inside it
                rewarm_start, rewarm_steps = start_step, min(50, max(total_steps - start_step, 1))
                settings["rewarm_start"], settings["rewarm_steps"] = rewarm_start, rewarm_steps
                note += (f"; learning rate re-warmed over {rewarm_steps} steps for the {total_p - restored} tensors "
                         "whose optimizer state restarts")
        resume_note = note + "; best/last validation reset because the validation set is new"
        if start_step >= total_steps:
            raise SystemExit(f"--init-from checkpoint is at step {start_step:,}; set --steps above that.")
        if start_step and schedule.phase(start_step + 1) == "decay":
            # the schedule of a training state is continued over the old and the new steps together, so its decay
            # (a share of that total) can begin before the first step of this run
            plan_lines.append(f"WARNING: this run starts at step {start_step + 1:,}, inside the decay of the "
                              f"schedule it continues (steps {schedule.decay_start + 1:,}-{total_steps:,}): the "
                              f"learning rate begins at {schedule.lr_at(start_step + 1):.2e}, below the peak "
                              f"{float(settings['lr']):.2e}, and only falls; pass --steps (the total, counted from "
                              "step 0) to place the decay")

    # The EMA of the weights (fp32 on the CPU, never trained, never in the optimizer): the saved average when the
    # training checkpoint has one for exactly these tensors, else a fresh one starting from the current weights (a
    # new run, --init-from, a resume from the best file, or --ema given on a run that had none).
    # ema_updates counts the updates the average has absorbed (ema_decay_at debiases with it); an average restored
    # from a checkpoint written before the count was saved has None and continues with the plain decay, as it did
    ema, ema_note, ema_updates = None, "no EMA of the weights", 0
    if ema_decay:
        saved = resume_obj.get("ema") if resume_obj is not None else None
        state = saved.get("state") if isinstance(saved, dict) else None
        if state and set(state) == set(model.state_dict()):
            ema = ema_init(state.items())
            ema_updates = saved.get("updates")
            ema_note = f"EMA of the weights restored from the checkpoint (decay {ema_decay:g} per step)"
        else:
            ema = ema_init(model.state_dict().items())  # a placeholder: the first update replaces it (debiased)
            ema_note = (f"EMA of the weights started from the step {start_step:,} weights (decay {ema_decay:g} per "
                        "step)" if start_step else f"EMA of the weights (decay {ema_decay:g} per step)")
        state = saved = None

    # Everything the checkpoints held has been copied into the model, optimizer, scaler, RNG and sampler: drop
    # them now (1.8 GB of RAM for the 216M run) instead of keeping them alive for the whole session.
    resumed = resume_obj is not None
    energy_before = resume_obj.get("energy") if resumed else None
    resume_obj = init_obj = source_obj = None

    train_model = maybe_compile(model, args.compile, device, return_hidden=bool(loss_chunk), amp_dtype=amp_dtype,
                                micro=micro, return_aux=moe)
    n_total, n_nonembed, n_active, _ = model.param_counts()
    unique = sum(len(a) for a in arrays["train"].values())

    # ---------------- run header ----------------
    print(f"\n{'=' * 100}\nRUN {run_id} started {now_iso()} | {resume_note}")
    # print: the screen and log.txt. log_only: what makes the run reproducible but nobody reads on screen.
    log_only(f"command: {' '.join(sys.argv)}")
    log_only(f"python {platform.python_version()} | torch {torch.__version__} | backend {backend_label(device)} | "
             f"{platform.platform()} | tiny_gpt.py sha256 {source_sha256(__file__)[:12]}")
    print(f"device: {device_label(device)} | {amp_name} | {opt_name} | "
          f"compile: {'on' if train_model is not model else 'off (eager)'}")
    log_only(f"device detail: attention: SDPA {attn_impl} ({attn_why}) | optimizer: {opt_name} ({opt_note})")
    if device.type == "mps":
        for note in mps_env_notes():
            print(f"environment: {note}")
    share = batcher.share()
    n_windows = sum(w.shape[0] for w in evalset.windows.values())
    print(f"data: {dataset_dir} ({manifest['fingerprint'][:12]}) | {unique:,} training tokens in "
          f"{len(arrays['train'])} sources | validation: {n_windows:,} fixed windows of {cfg.ctx:,} tokens from "
          f"{len(evalset.windows)} sources"
          + (f" | skipped (too small): {evalset.skipped}" if evalset.skipped else ""))
    log_only(f"dataset: {dataset_dir} | fingerprint {manifest['fingerprint'][:12]} | tokenizer "
             f"{tok.vocab_size:,} pieces ({tok.encode_mode}) | per source (training tokens, sampling share; "
             "validation tokens, fixed windows): "
             + "; ".join(f"{n} {len(a):,} {share.get(n, 0.0):.1%}"
                         + (f", {len(arrays['val'][n]):,}" if n in arrays["val"] else ", no validation")
                         + (f" {evalset.windows[n].shape[0]} x {cfg.ctx}" if n in evalset.windows else "")
                         for n, a in sorted(arrays["train"].items())))
    print(f"model: d_model {cfg.d_model}, {cfg.n_layers} layers, {cfg.n_heads} heads / {cfg.n_kv_heads} KV, ctx "
          f"{cfg.ctx:,}, vocabulary {cfg.vocab_size:,}")
    log_only("model: " + json.dumps(dataclasses.asdict(cfg)))
    print(f"parameters: {n_total:,} total, {n_nonembed:,} non-embedding"
          + (f", {n_active:,} active per token" if moe else ""))
    log_only("train: " + json.dumps(settings))
    # a resumed run: the whole run. After --init-from of a training state: the steps of this run alone (the
    # schedule goes on at the checkpoint's step, whose tokens were another run's)
    budget_steps = total_steps if resumed else max(total_steps - start_step, 0)
    train_tokens_total = budget_steps * settings["tokens_per_step"]
    print(f"budget: {budget_steps:,} steps"
          + (f" ({start_step + 1:,}-{total_steps:,})" if budget_steps != total_steps else "")
          + f" x {settings['tokens_per_step']:,} tokens ({micro} x {accum} x {cfg.ctx}) "
          f"= {train_tokens_total:,} tokens = {train_tokens_total / n_total:,.1f} tokens/parameter, "
          f"{train_tokens_total / max(unique, 1):,.2f} epochs")
    print(f"schedule: {schedule.describe()}")
    if ema is not None or model.doc_sep_id is not None:
        print(f"weights: {ema_note}"
              + (f"; updated every {EMA_EVERY} steps with decay {ema_decay ** EMA_EVERY:.4f} per update, scored "
                 "beside the raw weights at each evaluation while the learning rate is above 5% of peak; the better "
                 f"of the two is saved as {os.path.basename(best_path)} (its 'weights' field says which)"
                 if ema is not None else "")
              + ("; document mask on: attention stays within each document of a window"
                 if model.doc_sep_id is not None else ""))
    if moe:
        print(f"experts: {cfg.n_experts} per MoE block ({cfg.n_moe_blocks} of {cfg.n_layers} blocks), "
              f"{cfg.experts_per_token} per token, hidden {cfg.expert_hidden}, {cfg.router} router; balance "
              f"{cfg.balance}: load-balancing loss weight {lb_weight:g}, router z-loss {router_z_loss:g}, bias rate "
              f"{bias_rate:g} per step; experts run as {moe_impl} ({moe_impl_why}); metrics lines report moe_lb (1.0 "
              "= even routing), moe_z, moe_load_max (busiest expert / mean) and dead_experts")
    for line in plan_lines:
        print(f"plan: {line}")
    meter = EnergyMeter(device, args.cpu_watts, args.grid_intensity, energy_before,
                        tariff=Tariff(args.tariff, args.region))
    if meter.kwh():
        print(f"Earlier sessions of this run: {fmt_hms(run_seconds_before)}, {meter.short()}")
    log_only(meter.describe())
    log_only(f"evaluation: every {args.eval_every} steps on the fixed validation windows"
             + (f"; fact probes: {len(probes)} from {os.path.basename(args.probes)}" if probes else ""))
    save_when = f"every {args.save_every} steps" + (f" and every {args.save_minutes:g} min" if args.save_minutes > 0
                                                     else "")
    print(f"output: validation every {args.eval_every} steps"
          + (f" with {len(probes)} fact probes" if probes else "")
          + f" | {os.path.basename(ckpt_path)} {save_when}, {os.path.basename(best_path)} on NEW BEST, in "
          f"{os.path.dirname(os.path.abspath(ckpt_path))}"
          + (f" | full settings: {os.path.basename(_LOG_STATE['path'])}" if _LOG_STATE.get("path") else ""))
    log_only(f"checkpoints: {ckpt_path} {save_when} (training state), {best_path} on NEW BEST "
             f"(inference only); metrics: {metrics_path}")
    print("=" * 100)
    if start_step >= total_steps:
        print(f"Checkpoint already reached step {start_step:,} of {total_steps:,}; nothing to train. "
              "Extend with --resume --steps N.")
        meter.stop()
        return

    sample_gen_seed = args.seed + 7

    def payload(step, inference_only, weights="raw"):
        """The checkpoint dict: weights "raw" (the trained weights) or "ema" (the average, for a best file)."""
        base = {
            "format": CHECKPOINT_FORMAT, "format_version": FORMAT_VERSION,
            "checkpoint_type": "inference" if inference_only else "training",
            "created": now_iso(), "run_id": run_id, "torch_version": str(torch.__version__),
            "model_config": dataclasses.asdict(cfg),
            "model": (weights_for_checkpoint(model, device, bf16=inference_only and args.best_bf16) if weights == "raw"
                      else ema_weights_for_checkpoint(ema, bf16=inference_only and args.best_bf16)),
            "tokenizer": {"proto": tok.proto, **tok.meta()},
            "dataset": {"path": dataset_dir, "fingerprint": manifest["fingerprint"],
                        "train_tokens": {n: len(a) for n, a in arrays["train"].items()},
                        "val_tokens": {n: len(a) for n, a in arrays["val"].items()}},
            "train_config": dict(settings), "step": step, "tokens_seen": tokens_seen,
            "train_seconds": train_seconds, "run_seconds": run_seconds(), "energy": meter.state(),
            "metrics": {"best_val": best_val, "best_step": best_step, "last_eval": last_eval,
                        "history": history},
            "weights": weights,
        }
        if weights == "ema":
            base["ema_decay"] = ema_decay
        if not inference_only:
            if ema is not None:  # optional key: readers use .get("ema")
                base["ema"] = {"decay": ema_decay, "every": EMA_EVERY, "state": ema, "updates": ema_updates}
            base.update({
                "optimizer": optimizer_state_for_checkpoint(opt, device),
                "optimizer_param_names": [n for g in param_names for n in g],
                "scaler": scaler.state_dict() if scaler else None,
                # "cuda" keeps its name and shape (older readers); "mps" is the Mac's generator (None elsewhere)
                "rng": {"torch": torch.get_rng_state(),
                        "cuda": device_rng_state(device) if device.type == "cuda" else None,
                        "mps": device_rng_state(device) if device.type == "mps" else None},
                "batcher": batcher.state_dict(),
                # a mixture of experts adds the sums of its balance figures (optional keys: a dense run has none)
                "interval": {"loss_sum": float(interval_loss), "micro": interval_micro,
                             **({"lb_sum": float(interval_lb), "z_sum": float(interval_z),
                                 "counts": interval_counts.tolist()} if moe else {})},
            })
        return base

    def write_metrics(entry):
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"run_id": run_id, **entry}) + "\n")

    if args.overwrite and not resumed:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for path in (ckpt_path, best_path, metrics_path):
            if os.path.exists(path):
                os.replace(path, f"{path}.bak-{stamp}")
                print(f"renamed existing {path} -> {path}.bak-{stamp}")

    # ---------------- baseline evaluation ----------------
    if not resumed and not history:
        print("measuring the untrained model on the fixed validation windows ...")
        ev = evaluate_retrying()
        history.append({"step": start_step, "tokens": tokens_seen, "train_loss": None, "val_loss": ev["loss"],
                        "val_mix": ev["loss_mix"], "bpb": ev["bpb"], "bpb_mix": ev["bpb_mix"],
                        "bpb_per_source": ev["bpb_per_source"],
                        "val_per_source": ev["per_source"], "ppl": math.exp(min(ev["loss"], 50)), "lr": 0.0,
                        "tok_s": None, "elapsed_s": 0.0, "train_seconds": train_seconds,
                        "entropy": ev["entropy"], "confidence": ev["confidence"], "top1": ev["top1"],
                        "ece": ev["ece"], "probe_acc": None, "grad_norm": None, "is_best": False,
                        "baseline": True, "time": now_iso()})
        try:
            write_metrics(history[-1])
        except OSError as exc:  # the row is in the history that the checkpoints carry
            print(f"WARNING: could not write {metrics_path} ({type(exc).__name__}: {str(exc)[:160]}); the baseline "
                  "evaluation is missing from it")
        print(f"step {start_step:>6}/{total_steps} | baseline before training | val {ev['loss']:.4f} @{start_step}"
              f" | bpb {ev['bpb']:.3f} | uniform-guess loss would be {math.log(cfg.vocab_size):.3f} "
              "(not counted as best)")

    def run_seconds():
        """Wall-clock time of the whole run so far: earlier sessions plus this one."""
        return run_seconds_before + (time.perf_counter() - t_run)

    dashboard_on = args.dashboard == "on" or (
        args.dashboard == "auto" and os.path.normcase(os.path.abspath(args.out_dir)) == os.path.normcase(HERE))
    best_png = os.path.splitext(os.path.basename(best_path))[0] + "_dashboard.png"  # view_pt.py's name for it
    dashboard = {"proc": None, "pushed": last_dashboard_push(best_png) if dashboard_on else 0.0}

    # ---------------- training loop ----------------
    model.train()
    status, step = "completed", start_step
    interval_loss = torch.zeros((), device=device)
    interval_micro, interval_tokens, interval_train_time = 0, 0, 0.0
    progress_loss = torch.zeros((), device=device)
    progress_micro = 0
    last_gnorm = torch.zeros((), device=device)
    overflow_run = 0  # FP16 updates skipped in a row by the GradScaler (gradients overflowed)
    nonfinite_run = 0  # updates skipped in a row because the gradient norm was not finite (BF16/FP32 path)
    # mixture of experts: tokens per (MoE block, expert) of the current step (the bias update's input) and since
    # the last metrics line (the load figures), and the balancing-term sums reported there
    step_counts = torch.zeros((cfg.n_moe_blocks, cfg.n_experts), dtype=torch.long, device=device) if moe else None
    interval_counts = torch.zeros_like(step_counts) if moe else None
    interval_lb = torch.zeros((), device=device)
    interval_z = torch.zeros((), device=device)
    aux = None
    save_failures = 0  # periodic saves failed in a row (disk full, locked file, no memory for the copy)
    best_failures = 0  # saves of the best file failed in a row
    oom_retries = 0  # out-of-memory retries of the current step (after the first step)
    peak_mem = 0  # bytes: the process's peak memory on the device so far, sampled after every step (peak_mem_gib)
    fail_note = None  # the exception that ended the run, for the experiments table
    seg = time.perf_counter()
    interval_wall, interval_start_step = time.perf_counter(), start_step
    last_print = time.perf_counter()
    first_step = True
    done_step = start_step  # last step whose optimizer update finished
    last_save = time.perf_counter()
    if interval_state:  # resumed between evaluations: the next "train" value covers the whole interval
        interval_loss += float(interval_state.get("loss_sum", 0.0))
        interval_micro = int(interval_state.get("micro", 0))
        if moe:
            # the balance sums of the interval so far; a checkpoint from before they were saved resumes with them
            # at zero, as it always did (its first moe figures after the pause are then too low)
            if interval_state.get("lb_sum") is not None:
                interval_lb.fill_(float(interval_state["lb_sum"]))
            if interval_state.get("z_sum") is not None:
                interval_z.fill_(float(interval_state["z_sum"]))
            if interval_state.get("counts") is not None:
                saved_counts = torch.tensor(interval_state["counts"], dtype=torch.long)
                if saved_counts.shape == interval_counts.shape:
                    interval_counts.copy_(saved_counts)
    held = DeferInterrupt()
    # first-step out-of-memory: halve the micro-batch down to about 2k tokens (the planner's preference: micro 2 x
    # 132 without checkpointing beats 4 x 66 with it), then recompute blocks, then halve further
    micro_floor = min(max(1, 2048 // cfg.ctx), micro)
    fallbacks = {}  # file atomic_save wrote instead of a locked checkpoint in this session (deleted once superseded)

    def save(path, inference_only, at_step, weights="raw"):
        written = atomic_save(payload(at_step, inference_only, weights), path)
        old = fallbacks.pop(path, None)
        if old and old != written:
            try:
                os.remove(old)
            except OSError:
                pass
        if written != path:
            fallbacks[path] = written
        return written

    def snapshot():
        """Everything an unfinished step changes before its update: sampler, dropout RNG, loss sums."""
        return (batcher.gen.get_state(),
                torch.get_rng_state() if cfg.dropout > 0 else None,
                torch.cuda.get_rng_state_all() if cfg.dropout > 0 and device.type == "cuda" else None,
                interval_loss.clone(), progress_loss.clone(),
                device_rng_state(device) if cfg.dropout > 0 and device.type == "mps" else None,
                interval_lb.clone(), interval_z.clone())

    def rewind(snap):
        """Undo an unfinished step, so it is replayed exactly (same windows, dropout masks and loss sums)."""
        batcher.gen.set_state(snap[0])
        if snap[1] is not None:
            torch.set_rng_state(snap[1])
        if snap[2] is not None:
            torch.cuda.set_rng_state_all(snap[2])
        if snap[5] is not None:
            set_device_rng_state(device, snap[5])
        interval_loss.copy_(snap[3])
        progress_loss.copy_(snap[4])
        interval_lb.copy_(snap[6])
        interval_z.copy_(snap[7])
        if moe:
            step_counts.zero_()

    try:
        step = start_step
        while step < total_steps:
            step_start = snapshot()  # taken before the step number moves on, so it always belongs to this step
            scaler_state = scaler.state_dict() if scaler is not None else None  # for a retry of this step
            step += 1
            lr = schedule.lr_at(step)
            # after a resume from a best checkpoint, or an --init-from whose optimizer state partly restarts
            if rewarm_steps and step - rewarm_start <= rewarm_steps:
                lr *= (step - rewarm_start) / rewarm_steps
            for group in opt.param_groups:
                group["lr"] = lr
            try:
                for _ in range(accum):
                    x, y = batcher.next()
                    with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                        if moe:  # the experts' balancing terms come back with the output
                            output, aux = train_model(x, return_hidden=bool(loss_chunk), return_aux=True)
                        else:
                            output = train_model(x, return_hidden=True) if loss_chunk else train_model(x)
                        if loss_chunk:
                            ce, zsum, n_tok = chunked_lm_loss(output, model.lm_head.weight, y, loss_chunk)
                            flat = None
                        else:
                            logits = output
                        output = None
                    if loss_chunk:
                        loss = ce / n_tok
                    else:
                        flat = logits.float().view(-1, cfg.vocab_size)
                        loss, lse = full_lm_loss(flat, y.reshape(-1))
                    interval_loss += loss.detach()
                    progress_loss += loss.detach()
                    if z_loss:
                        # PaLM z-loss: keeps log Z (the softmax normaliser) near 0; reported loss excludes it
                        loss = loss + z_loss * (zsum / n_tok if loss_chunk else lse.pow(2).mean())
                    if aux is not None:
                        # mixture of experts: the load-balancing loss (when --balance is loss or both) and the router
                        # z-loss join the backpropagated loss; the tokens per expert feed the bias update and the load
                        # figures. Nothing is kept on the modules (see MoEFFN).
                        interval_lb += aux["lb"].detach()
                        interval_z += aux["z"].detach()
                        step_counts += aux["counts"]
                        if lb_weight or router_z_loss:
                            loss = loss + lb_weight * aux["lb"] + router_z_loss * aux["z"]
                        aux = None
                    if scaler is not None:
                        scaler.scale(loss / accum).backward()
                    else:
                        (loss / accum).backward()
                with held:  # Ctrl+C waits until the update and its bookkeeping are complete
                    applied = True  # False when the update is dropped for a non-finite gradient norm
                    if scaler is not None:
                        scaler.unscale_(opt)
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"] or float("inf"))
                    if scaler is not None:
                        scale = scaler.get_scale()
                        scaler.step(opt)
                        scaler.update()
                        # a lower scale means the FP16 gradients overflowed: the scaler skipped this update (normal
                        # now and then), so its infinite norm is no sign of divergence and the last norm stands.
                        # Many skips in a row (or a scale below 1) mean every update is being dropped: that run
                        # has diverged, and the infinite norm is kept so the save and Ctrl+C checks stop it.
                        new_scale = scaler.get_scale()
                        if new_scale >= scale:
                            last_gnorm, overflow_run = gnorm, 0
                        else:
                            overflow_run += 1
                            if overflow_run >= MAX_OVERFLOW_SKIPS or new_scale < 1.0:
                                last_gnorm = gnorm
                    elif not bool(torch.isfinite(gnorm)):
                        # Without a GradScaler nothing skips a bad step for us: one inf or NaN gradient element
                        # would poison every weight through the optimizer's moments. Drop the update and the
                        # batch's share of the loss sums; last_gnorm is not touched, so the save and Ctrl+C
                        # checks still see the last good norm and keep the still-good weights.
                        opt.zero_grad(set_to_none=True)
                        interval_loss.copy_(step_start[3])
                        progress_loss.copy_(step_start[4])
                        interval_lb.copy_(step_start[6])
                        interval_z.copy_(step_start[7])
                        applied = False
                        nonfinite_run += 1
                        print(f"step {step}: non-finite gradient norm {float(gnorm)}; update skipped "
                              f"({nonfinite_run}/{MAX_OVERFLOW_SKIPS})")
                        if nonfinite_run >= MAX_OVERFLOW_SKIPS:
                            last_gnorm = gnorm
                            raise FloatingPointError(
                                f"gradients became {float(gnorm)} by step {step}; {ckpt_path} was not overwritten "
                                "and still holds the last good training state. Resume it with a lower --lr (a new "
                                "--name with --init-from).")
                    else:
                        opt.step()
                        last_gnorm, nonfinite_run = gnorm, 0
                    opt.zero_grad(set_to_none=True)
                    if moe:
                        if applied and bias_rate:
                            # DeepSeek-V3 (Wang et al. 2024): bias -= rate x sign(tokens - mean tokens) per expert,
                            # so the experts the step overloaded are chosen less in the next
                            model.update_router_bias(step_counts, bias_rate)
                        if applied:
                            interval_counts += step_counts
                        step_counts.zero_()
                    if applied:  # a skipped step's loss left the sums above, so it leaves their counts too
                        interval_micro += accum
                        progress_micro += accum
                    interval_tokens += settings["tokens_per_step"]
                    tokens_seen += settings["tokens_per_step"]
                    done_step = step
            except RuntimeError as exc:  # torch.cuda.OutOfMemoryError / MPS OOM are RuntimeErrors
                oom = "out of memory" in str(exc).lower()
                if not first_step:
                    if not oom:
                        raise
                    # Out of memory later in the run (fragmentation, another program on the GPU): the step is
                    # replayed on the same windows up to twice, after the allocator has been emptied, rather than
                    # ending a run that trained fine until now.
                    oom_retries += 1
                    if oom_retries > 2:
                        raise
                    traceback.clear_frames(exc.__traceback__)  # the failed pass's frames hold its activations
                    opt.zero_grad(set_to_none=True)
                    x = y = logits = flat = lse = loss = ce = zsum = output = aux = None
                    free_device_memory(device)
                    rewind(step_start)
                    if scaler is not None:  # a new scaler with the same scale: forgets an unscale_ already done
                        scaler = torch.amp.GradScaler(device.type)
                        scaler.load_state_dict(scaler_state)
                    time.sleep(5)
                    print(f"Out of memory at step {step}; retrying the same step ({oom_retries}/2).")
                    step -= 1
                    held.deliver()
                    continue
                # a compiled model that fails at the first step for another reason (a graph recompiled after an
                # earlier out-of-memory retry) is retried once in eager mode; an eager failure is raised as it is
                if not (oom or train_model is not model):
                    raise
                traceback.clear_frames(exc.__traceback__)  # the failed pass's frames hold its activations
                opt.zero_grad(set_to_none=True)
                x = y = logits = flat = lse = loss = ce = zsum = output = aux = None  # drop the failed batch and its graph before retrying
                free_device_memory(device)
                rewind(step_start)  # the retry trains on the same windows, as an uninterrupted run would
                if scaler is not None:  # a new scaler with the same scale: forgets an unscale_ already done
                    scaler = torch.amp.GradScaler(device.type)
                    scaler.load_state_dict(scaler_state)
                # cheapest remedy first: the loss in pieces, then a smaller micro-batch down to about 2,048 tokens
                # (micro_floor), then recomputation, then smaller still; never one that the command line switched off
                chunk_ok = not loss_chunk and args.loss_chunk_tokens != 0
                ckpt_ok = not model.grad_checkpoint and args.activation_checkpointing != "off"
                if not oom:
                    train_model = model
                    set_chunk_loss_compiled(False)
                    print(f"torch.compile failed at the first step ({type(exc).__name__}: {str(exc)[:120]}); "
                          "retrying in eager mode.")
                elif chunk_ok:
                    loss_chunk = settings["loss_chunk_tokens"] = LOSS_CHUNK_TOKENS
                    print(f"Out of memory at the first step; retrying with the loss computed {loss_chunk:,} tokens "
                          "at a time.")
                elif micro > 1 and (micro > micro_floor or not ckpt_ok):
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
                    if settings["tokens_per_step"] != old_tps:  # the header's budget line is stale: say it again
                        train_tokens_total = total_steps * settings["tokens_per_step"]
                        settings["planned_train_tokens"] = train_tokens_total
                        print(f"budget now: {total_steps:,} steps x {settings['tokens_per_step']:,} tokens = "
                              f"{train_tokens_total:,} tokens = {train_tokens_total / n_total:,.1f} tokens/parameter, "
                              f"{train_tokens_total / max(unique, 1):,.2f} epochs")
                elif ckpt_ok:
                    model.grad_checkpoint = settings["activation_checkpointing"] = True
                    print(f"Out of memory at the first step with micro-batch {micro}; retrying with activation "
                          "checkpointing (blocks recomputed in the backward pass).")
                else:
                    forced = [o for o, off in (("--loss-chunk-tokens 0", args.loss_chunk_tokens == 0),
                                               ("--activation-checkpointing off",
                                                args.activation_checkpointing == "off")) if off]
                    raise SystemExit(
                        "Out of memory at the first step even at micro-batch 1"
                        + (f"; {' and '.join(forced)} kept those memory savings off: drop "
                           f"{'them' if len(forced) > 1 else 'it'} or" if forced
                           else " with the loss in pieces and activation checkpointing;")
                        + " choose a smaller --d-model/--layers/--ctx.") from exc
                step -= 1
                held.deliver()  # a Ctrl+C held during the failed update
                continue
            first_step = False
            oom_retries = 0
            if ema is not None and step % EMA_EVERY == 0:
                with held:  # the average is updated whole or not at all
                    # ema <- b ema + (1 - b) w with b = d^k debiased (ema_decay_at), k = EMA_EVERY: the per-step
                    # decay compounded over the steps between updates (0.999^10 ~ 0.990, a horizon of ~1,000 steps)
                    if ema_updates is not None:
                        ema_updates += 1
                    ema_update(ema, model, ema_decay_at(ema_decay ** EMA_EVERY, ema_updates))
            held.deliver()  # a Ctrl+C held during the update: the step is complete, so it is saved as done
            peak_mem = peak_memory_bytes(device, peak_mem)

            do_eval = step % args.eval_every == 0 or step == total_steps
            do_progress = args.log_every and step % args.log_every == 0 and not do_eval
            stop_now = bool(args.stop_after_steps) and step - start_step >= args.stop_after_steps
            do_save = (step % args.save_every == 0 or step == total_steps or stop_now
                       or (args.save_minutes > 0 and time.perf_counter() - last_save >= args.save_minutes * 60))
            # one after the first completed step of a session (the run is seen to train), then one whenever
            # nothing was printed for --heartbeat-seconds
            heartbeat = (args.heartbeat_seconds > 0 and not (do_eval or do_progress)
                         and (step == start_step + 1
                              or time.perf_counter() - last_print >= args.heartbeat_seconds))
            if not (do_eval or do_progress or do_save or heartbeat):
                continue

            synchronize(device)
            now = time.perf_counter()
            interval_train_time += now - seg
            train_seconds += now - seg
            seg = None  # counted: evaluation, sampling and saving below are not training time

            if do_progress or heartbeat:
                ptrain = (progress_loss / max(progress_micro, 1)).item()
                per_step = interval_train_time / max(step - interval_start_step, 1)
                next_eval = min(total_steps, (step // args.eval_every + 1) * args.eval_every)
                # progress only: the share done, the training loss since the previous line, the speed and the clock
                # time; validation, learning rate, energy and ETA are on the evaluation lines
                print(f"{'  ... ' if heartbeat else ''}step {step:>6}/{total_steps} | {step / total_steps:5.1%} | "
                      f"train {ptrain:.3f} | {interval_tokens / max(interval_train_time, 1e-9):,.0f} tok/s | "
                      f"{clock()} | next val in {fmt_dur(per_step * (next_eval - step))}")
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
                ev = evaluate_retrying()
                ev["step"] = step
                # the EMA weights are scored as well (one more evaluation) while the learning rate is above 5 % of
                # the peak; in the tail of the decay the raw weights settle and the two agree, so that pass is saved
                ev_ema = None
                # ema_updates 0: no update has replaced the starting copy yet, so there is nothing to score
                if ema is not None and ema_updates != 0 and lr >= 0.05 * float(settings["lr"]):
                    with swapped_weights(model, ema, hold=held):  # the weight copies are whole or not at all
                        ev_ema = evaluate_retrying()
                    held.deliver()  # a Ctrl+C held during a copy: the raw weights are back in place
                probe_acc, probe_by_category, probe_floor_acc = None, None, None
                if probes:
                    probe_acc, probe_results = score_probes(model, tok, probes, device, amp_dtype)
                    probe_floor_acc = probe_floor(probe_results)
                    probe_by_category = {k: round(v["accuracy"], 4) for k, v in probe_summary(probe_results).items()}
                    ev["probes"] = probe_report(probe_results, probes_full)  # kept in the checkpoint for view_pt
                prev_best, prev_step = best_val, best_step
                # the best model is whichever weights score lower, the raw ones or the average; the best file
                # records which it holds ("weights": "raw" | "ema")
                best_source = "ema" if ev_ema is not None and ev_ema["loss"] < ev["loss"] else "raw"
                candidate = ev_ema["loss"] if best_source == "ema" else ev["loss"]
                is_best = candidate < best_val
                if is_best:
                    best_val, best_step = candidate, step
                last_eval = ev
                elapsed = time.perf_counter() - t_run
                per_step = (time.perf_counter() - interval_wall) / max(step - interval_start_step, 1)
                eta = per_step * (total_steps - step)
                entry = {"step": step, "tokens": tokens_seen, "train_loss": train_loss, "val_loss": ev["loss"],
                         "val_mix": ev["loss_mix"],
                         "val_per_source": ev["per_source"], "ppl": math.exp(min(ev["loss"], 50)), "lr": lr,
                         "bpb": ev["bpb"], "bpb_mix": ev["bpb_mix"], "bpb_per_source": ev["bpb_per_source"],
                         "tok_s": tok_s, "elapsed_s": elapsed, "train_seconds": train_seconds,
                         "run_seconds": run_seconds(),
                         "entropy": ev["entropy"], "confidence": ev["confidence"], "top1": ev["top1"],
                         "ece": ev["ece"], "probe_acc": probe_acc, "probe_floor": probe_floor_acc,
                         "probe_by_category": probe_by_category,
                         "grad_norm": float(last_gnorm), "peak_mem_gib": round(peak_mem / 2 ** 30, 3),
                         "energy_wh": meter.snapshot(),
                         "co2e_g": round(meter.co2e_g(), 1), "cost_gbp": round(meter.cost_gbp() or 0.0, 4),
                         "is_best": is_best, "best_weights": best_source if is_best else None,
                         "val_ema": ev_ema["loss"] if ev_ema is not None else None,
                         "bpb_ema": ev_ema["bpb"] if ev_ema is not None else None, "time": now_iso()}
                moe_text = ""
                if moe:
                    # per MoE block the mean over its micro-batches of the balancing terms (lb 1.0 = even routing),
                    # and from the tokens per expert since the last line: load = tokens / mean tokens of the block
                    # (moe_load_max is the busiest expert anywhere), dead_experts = experts that saw no token
                    per = max(interval_micro, 1) * cfg.n_moe_blocks
                    load = interval_counts.float() / interval_counts.float().mean(dim=1, keepdim=True).clamp(min=1e-9)
                    entry.update({"moe_lb": (interval_lb / per).item(), "moe_z": (interval_z / per).item(),
                                  "moe_load_max": load.max().item(),
                                  "dead_experts": int((interval_counts == 0).sum().item()),
                                  "moe_load": [[round(v, 3) for v in row] for row in load.tolist()]})
                    moe_text = (f" | moe lb {entry['moe_lb']:.3f} z {entry['moe_z']:.2f} load max "
                                f"{entry['moe_load_max']:.2f} dead {entry['dead_experts']}")
                    interval_counts.zero_()
                history.append(entry)
                # the interval is recorded: zero its sums now, so a Ctrl+C during the best save, the print or the
                # sample does not fold these steps into the next row as well
                interval_loss.zero_()
                interval_lb.zero_()
                interval_z.zero_()
                interval_micro, interval_tokens, interval_train_time = 0, 0, 0.0
                if is_best:  # before printing or sampling, so Ctrl+C can never leave an announced best unsaved
                    try:
                        save(best_path, True, step, best_source)
                        best_failures = 0
                    except BaseException as exc:
                        # not written (Ctrl+C, full disk, no memory for the copy): record no best that no file holds
                        best_val, best_step = prev_best, prev_step
                        entry["is_best"], entry["best_weights"] = False, None
                        best_failures += 1
                        if not isinstance(exc, Exception) or best_failures >= 3:
                            # the evaluation stays in history (and the saved training state): keep the file in step
                            try:
                                write_metrics(entry)
                            except Exception:  # noqa: BLE001 - the disk may be the problem; the first error matters
                                pass
                            raise
                        # like a failed periodic save, this must not end a run that is training fine: the evaluation
                        # counts as no best, so the next one that beats the best file on disk is saved in its place
                        is_best = False
                        print(f"WARNING: could not save {best_path} ({type(exc).__name__}: {str(exc)[:160]}); "
                              f"training continues, the next better evaluation tries again ({best_failures}/3)")
                        free_device_memory(device)
                try:
                    write_metrics(entry)
                except OSError as exc:  # the row is in the history that the checkpoints carry
                    print(f"WARNING: could not write {metrics_path} ({type(exc).__name__}: {str(exc)[:160]}); "
                          f"the evaluation of step {step} is missing from it")
                # after the row is written (view_pt.py adds the rows after the best file's last step), and at every
                # evaluation: the curves go on when the run stops improving
                if dashboard_on and os.path.exists(best_path):
                    launch_dashboard(best_path, args, dashboard)
                if is_best:
                    status_text = (f"NEW BEST (prev {prev_best:.4f} @{prev_step})" if prev_step is not None
                                   else "NEW BEST (first evaluation)") + (" (ema weights)" if best_source == "ema"
                                                                           else "")
                elif best_step is None:
                    status_text = "no best saved yet"  # only after a failed first save of the best file
                else:
                    status_text = f"best {best_val:.4f} @{best_step}"
                print(f"step {step:>6}/{total_steps} | train {train_loss:.3f} | val {ev['loss']:.4f} @{step}"
                      + (f" | val(mix) {ev['loss_mix']:.3f}" if ev.get("loss_mix") is not None else "")
                      + (f" | val(ema) {ev_ema['loss']:.4f}" if ev_ema is not None else "")
                      + f" | bpb {ev['bpb']:.3f}"
                      + (f" | probes {probe_acc:.1%}" if probe_acc is not None else "")
                      + f" | lr {lr:.2e} | {tok_s:,.0f} tok/s | "
                      f"{meter.short(co2e=False)} | {clock()} | ETA {fmt_dur(eta)}{moe_text} | {status_text}")
                last_print = time.perf_counter()
                if args.sample_every and (step % args.sample_every == 0 or step == total_steps):
                    gen = torch.Generator().manual_seed(sample_gen_seed)
                    text = generate(model, tok, args.sample_prompt, args.sample_tokens, device, amp_dtype,
                                    0.8, 50, 0.95, gen)
                    shown = text.replace("\n", " \u23ce ").strip()
                    print(f"    sample @{step}: \"{shown}\"")
                progress_loss.zero_()
                progress_micro = 0
                interval_wall, interval_start_step = time.perf_counter(), step

            if do_save:
                if not bool(torch.isfinite(last_gnorm)):
                    raise FloatingPointError(
                        f"gradients became {float(last_gnorm)} by step {step}; {ckpt_path} was not overwritten and "
                        "still holds the last good training state. Resume it with a lower --lr (a new --name with "
                        "--init-from).")
                try:
                    saved = save(ckpt_path, False, step)
                    save_failures = 0
                except Exception as exc:  # noqa: BLE001 - not BaseException: Ctrl+C keeps its own path below
                    # A save that fails (disk full, file locked, no RAM for the CPU copy) must not end a run that
                    # is training fine: the loop goes on and tries again in about a minute (with --save-minutes 0 at
                    # the next multiple of --save-every), three times in a row at most. A save that the session
                    # needs now (its last step, --stop-after-steps) is raised.
                    save_failures += 1
                    next_try = ("in about a minute" if args.save_minutes > 0 else "at step "
                                f"{min(total_steps, (step // args.save_every + 1) * args.save_every):,}")
                    print(f"WARNING: could not save {ckpt_path} ({type(exc).__name__}: {str(exc)[:160]}); "
                          f"training continues, next attempt {next_try} ({save_failures}/3)")
                    x = y = logits = flat = lse = loss = ce = zsum = output = aux = None
                    free_device_memory(device)
                    last_save = time.perf_counter() - args.save_minutes * 60 + 60
                    if save_failures >= 3 or stop_now or step == total_steps:
                        raise
                else:
                    last_save = time.perf_counter()
            if stop_now and step < total_steps:
                status = "stopped"
                print(f"Stopped after {args.stop_after_steps:,} steps this session (--stop-after-steps); saved "
                      f"{saved}.\nContinue with: python tiny_gpt.py --name {args.name} --resume")
                break
            seg = time.perf_counter()
    except KeyboardInterrupt:
        status = "interrupted"
        # A second Ctrl+C is held until the state of this pause is written: it used to abort the save, and the
        # checkpoint kept the last periodic state (up to --save-minutes of training lost).
        with held:
            if seg is not None:  # the time since the last counted point was training
                train_seconds += time.perf_counter() - seg
            if done_step < step:
                # Ctrl+C arrived inside a step, before its update (which Ctrl+C never
                # splits). Rewind the sampler, dropout RNG and loss sums so --resume
                # replays exactly that step, and save the last completed step.
                rewind(step_start)
                opt.zero_grad(set_to_none=True)
                step = done_step
            print(f"\nCtrl+C: saving the training state at step {step:,}, please wait ...")
            if bool(torch.isfinite(last_gnorm)):
                saved = save(ckpt_path, False, step)
                print(f"Paused after step {step:,} (last completed update); saved {saved}.\n"
                      f"Resume with: python tiny_gpt.py --name {args.name} --resume")
            else:
                crash = os.path.join(args.out_dir, f"{args.name}_crash.pt")
                atomic_save(payload(step, inference_only=False), crash)
                print(f"Paused after step {step:,}, but the gradients were not finite: the state went to {crash}; "
                      f"{ckpt_path} still holds the last good training state.")
        held.hit = False  # a Ctrl+C held during the save asked for the pause that has just been made
    except BaseException as exc:
        status = "failed"
        fail_note = f"{type(exc).__name__}: {str(exc)[:200]}"
        crash = os.path.join(args.out_dir, f"{args.name}_crash.pt")
        # the step's tensors and graph are dropped first: after an out-of-memory error the full state (optimizer
        # included) needs the room, and if even that fails the weights alone are still worth keeping
        x = y = logits = flat = lse = loss = ce = zsum = output = aux = None
        traceback.clear_frames(exc.__traceback__)  # and what the frames of the failed call still hold
        gc.collect()
        try:
            try:
                atomic_save(payload(step, inference_only=False), crash)
                print(f"\nState at step {step:,} saved to {crash} for inspection (not used by --resume).")
            except Exception as save_exc:  # noqa: BLE001
                print(f"Could not save the full crash state ({type(save_exc).__name__}: {str(save_exc)[:160]}); "
                      "saving the weights only.")
                atomic_save(payload(step, inference_only=True), crash)
                print(f"\nWeights at step {step:,} saved to {crash} for inspection (not used by --resume).")
        except Exception as save_exc:  # noqa: BLE001
            print(f"Could not save crash state: {save_exc}")
        raise
    finally:
        meter.stop()
        last_hist = history[-1] if history else {}
        record_experiment(args.out_dir, {
            "event": "training run", "name": args.name, "status": status, "step": step, "total_steps": total_steps,
            "params": n_total, "non_embedding_params": n_nonembed, "active_params": n_active,
            "moe": cfg.moe_label() or None, "lb_loss": lb_weight if moe else None,
            "router_z_loss": router_z_loss if moe else None, "balance": cfg.balance if moe else None,
            "d_model": cfg.d_model, "layers": cfg.n_layers,
            "heads": cfg.n_heads, "kv_heads": cfg.n_kv_heads, "ctx": cfg.ctx, "vocab": cfg.vocab_size,
            "tokens_seen": tokens_seen, "tokens_per_step": settings["tokens_per_step"],
            "micro_batch": settings["micro_batch"], "grad_accum": settings["grad_accum"],
            "activation_checkpointing": "on" if settings.get("activation_checkpointing") else "off",
            "loss_chunk_tokens": int(settings.get("loss_chunk_tokens") or 0),
            "optimizer": OPTIMIZER_LABELS.get(settings.get("optimizer", "adamw")), "lr": f"{settings['lr']:.3g}",
            "min_lr": f"{settings['min_lr']:.3g}", "schedule": {"wsd": "WSD"}.get(settings["schedule"],
                                                                                 settings["schedule"]),
            "warmup_steps": settings["warmup_steps"],
            "decay_frac": settings["decay_frac"] if settings["schedule"] == "wsd" else None,
            "decay_shape": settings["decay_shape"] if settings["schedule"] == "wsd" else None,
            "weight_decay": settings["weight_decay"], "betas": "/".join(f"{b:g}" for b in settings["betas"]),
            "grad_clip": settings["grad_clip"], "z_loss": z_loss, "ema": f"{ema_decay:g}" if ema_decay else "off",
            "lr_scale": f"{float(settings.get('lr_scale', 1.0)):g}", "dropout": cfg.dropout, "precision": amp_name,
            "dataset": os.path.basename(os.path.normpath(dataset_dir)),
            "best_val": f"{best_val:.4f}" if math.isfinite(best_val) else None, "best_step": best_step,
            "last_val": f"{last_eval['loss']:.4f}" if last_eval else None,
            "last_val_mix": f"{last_eval['loss_mix']:.4f}" if last_eval and last_eval.get("loss_mix") is not None
            else None,
            "last_bpb": f"{last_eval['bpb']:.4f}" if last_eval and last_eval.get("bpb") else None,
            "last_ppl": f"{math.exp(min(last_eval['loss'], 50)):.2f}" if last_eval else None,
            "benchmark_accuracy": f"{last_hist['probe_acc']:.3f}" if last_hist.get("probe_acc") is not None else None,
            "benchmark_floor": f"{last_hist['probe_floor']:.3f}" if last_hist.get("probe_floor") is not None else None,
            "tok_s": f"{last_hist['tok_s']:.0f}" if last_hist.get("tok_s") else None,
            "peak_mem_gib": f"{peak_mem / 2 ** 30:.2f}" if peak_mem else None, "device": device.type,
            "train_hours": f"{train_seconds / 3600:.2f}", "run_hours": f"{run_seconds() / 3600:.2f}",
            **energy_columns(meter, device, fail_note), "checkpoint": ckpt_path})
        print(f"RUN {run_id} {status} at {now_iso()} | step {step:,}/{total_steps:,} | best val "
              + (f"{best_val:.4f} @{best_step}" if best_step is not None else "none") + " | last val "
              + (f"{last_eval['loss']:.4f} @{last_eval['step']}" if last_eval else "none")
              + f" | this session {fmt_hms(time.perf_counter() - t_run)} | whole run: "
              f"{fmt_hms(run_seconds())} in total, {fmt_hms(train_seconds)} of it training, {meter.short()} "
              f"(GPU {meter.wh['gpu'] / 1000:.2f}, CPU {meter.wh['cpu'] / 1000:.2f}, memory "
              f"{meter.wh['ram'] / 1000:.3f} kWh)")


def energy_columns(meter, device, note=None):
    """The experiments-table energy, carbon and cost of a run, and its notes: real
    figures only when the GPU was measured or the run was on the CPU (whose power
    is estimated). On an unmeasured GPU (MPS: no nvidia-smi) the three columns are
    left blank and the notes say so, instead of a CPU-only figure posing as the
    run's energy."""
    if meter.gpu_measured or device.type == "cpu":
        return {"energy_kwh": f"{meter.kwh():.3f}", "co2e_kg": f"{meter.co2e_g() / 1000:.4f}",
                "cost_gbp": f"{meter.cost_gbp() or 0.0:.2f}", "notes": note}
    return {"energy_kwh": "", "co2e_kg": "", "cost_gbp": "",
            "notes": "; ".join(n for n in (note, "energy not measured (no nvidia-smi)") if n)}


EXPERIMENT_COLUMNS = ["time", "event", "name", "status", "step", "total_steps", "params", "non_embedding_params",
                      "active_params", "moe",
                      "d_model", "layers", "heads", "kv_heads", "ctx", "vocab", "tokens_seen", "tokens_per_step",
                      "micro_batch", "grad_accum", "activation_checkpointing", "loss_chunk_tokens", "optimizer", "lr",
                      "lr_scale", "min_lr", "schedule", "warmup_steps", "decay_frac", "decay_shape", "weight_decay", "betas", "grad_clip", "z_loss", "ema", "dropout",
                      "lb_loss", "router_z_loss", "balance",
                      "precision", "dataset", "best_val", "best_step", "last_val", "last_val_mix", "last_bpb", "last_ppl",
                      "benchmark_accuracy", "benchmark_floor", "benchmark_items", "tok_s", "peak_mem_gib", "device",
                      "train_hours",
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


def experiments_path(out_dir):
    """The one experiments table: <out_dir>/results/experiments.csv (for this
    folder's runs, the file in the repository that is pushed to GitHub)."""
    return os.path.join(out_dir, "results", "experiments.csv")


def _hide_home(value):
    """Text with the home folder shown as ~ (the table is published as it is)."""
    if not isinstance(value, str):
        return value
    home = os.path.expanduser("~")
    for variant in {home, home.replace("\\", "/")}:
        value = re.sub(re.escape(variant), "~", value, flags=re.IGNORECASE)
    return value


def publish_experiments(path, row):
    """Push the experiments table (results/experiments.csv) to GitHub."""
    with open(path, encoding="utf-8-sig") as handle:
        text = _hide_home(handle.read())  # rows are written with ~ already; older rows are covered too
    what = " ".join(str(row.get(k)) for k in ("event", "name", "status") if row.get(k))
    # with the byte-order mark the table was created with (read above without it): Excel takes the file for UTF-8
    status = publish_to_github("results/experiments.csv", f"Experiments: {what}", content="\ufeff" + text,
                               wait=False)
    print(f"experiments.csv on GitHub (results/experiments.csv): {status}")


def record_experiment(out_dir, row):
    """Append one row to <out_dir>/results/experiments.csv (opens in Excel), the
    single permanent record of every run, benchmark and comparison; paths show
    the home folder as ~. A table left by older versions directly in <out_dir>
    is moved there first (lose nothing). A table that cannot be read or written
    (another encoding, a damaged file, a full disk) costs this row and one note:
    a run calls this as it ends, and must still reach its last line."""
    import csv
    path = experiments_path(out_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    legacy = os.path.join(out_dir, "experiments.csv")
    try:
        if os.path.isfile(legacy) and not os.path.exists(path):
            with open(legacy, encoding="utf-8-sig") as handle:
                text = _hide_home(handle.read())
            with open(path, "w", encoding="utf-8-sig", newline="") as handle:
                handle.write(text)
            os.remove(legacy)
        new = not os.path.exists(path)
        if not new:
            try:
                with open(path, encoding="utf-8-sig", newline="") as handle:
                    header = next(csv.reader(handle), [])
                with open(path, encoding="utf-8-sig", newline="") as handle:
                    # restkey: a row with more cells than the header has columns keeps them under a name that is
                    # no column (None, the default, is refused by the writer)
                    old_rows = list(csv.DictReader(handle, restkey="_extra"))
                # known columns in the standard order; columns this version does not know are kept at the end
                target = (EXPERIMENT_COLUMNS if set(header) <= set(EXPERIMENT_COLUMNS)
                          else header + [c for c in EXPERIMENT_COLUMNS if c not in header])
                if header != target:
                    # a newer version added columns: keep a copy, rewrite once with the full header, lose nothing
                    # (cells beyond the header stay in the copy only)
                    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    import shutil
                    shutil.copy2(path, f"{path}.before-columns-{stamp}")
                    tmp = path + ".tmp"
                    with open(tmp, "w", encoding="utf-8-sig", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=target, extrasaction="ignore")
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
                writer.writerow({"time": now_iso(), **{k: _hide_home(v) for k, v in row.items() if v is not None}})
        except PermissionError:
            print(f"Note: {path} is open in another program (Excel?); this row was not recorded: {row}")
            return
    except (OSError, ValueError, csv.Error) as exc:  # UnicodeDecodeError is a ValueError
        try:
            os.remove(path + ".tmp")
        except OSError:
            pass
        print(f"Note: {path} could not be read or written ({type(exc).__name__}: {str(exc)[:160]}); this row was not "
              f"recorded: {row}")
        return
    if os.path.normcase(os.path.abspath(out_dir)) == os.path.normcase(HERE):  # this folder's runs, not test runs
        publish_experiments(path, row)


# Whether the user chose the folder of torch.compile's cache. Asked when this module is imported, because torch
# writes its own choice (a folder in the temp folder) into the variable when its compiler is first imported, and
# building an optimizer imports it: a training run does that before maybe_compile is reached.
COMPILE_CACHE_GIVEN = "TORCHINDUCTOR_CACHE_DIR" in os.environ


def maybe_compile(model, mode, device, return_hidden=False, amp_dtype=None, micro=1, return_aux=False):
    """The model compiled with torch.compile, warmed up once with exactly the call
    training makes (a (micro, ctx) batch with the layout of TokenBatcher.next()'s,
    under the same autocast; return_hidden: the chunked loss), so the training
    graph is compiled here and a compile failure
    falls back to eager instead of failing step 1; or the model itself (eager)
    when compiling is off, unavailable or fails. Running out of memory in the
    warm-up is not a compile failure: the compiled model is returned and the
    first step's out-of-memory handling chooses the memory savings."""
    if mode == "off":
        return model
    triton_ok = have_triton()
    # auto: compile on CUDA whenever Triton is importable, Windows included (the triton-windows package; measured
    # 1.45x faster training on an RTX 3070 laptop for 36.6M and 70M models). The first steps then take ~10-30 s longer.
    if mode == "auto" and not (device.type == "cuda" and triton_ok):
        if device.type == "mps":
            print("torch.compile not tried in auto mode on MPS (prototype backend); test once with --compile on and "
                  "compare tok/s.")
        return model
    if not triton_ok and device.type == "cuda":
        print("torch.compile skipped: Triton is not installed (on Windows: python -m pip install triton-windows, the "
              "version matching PyTorch); using eager mode.")
        return model
    compiled, failure = None, None
    try:
        # the compiled graphs are kept beside the script, not in the temp folder (which gets emptied: every graph
        # is then compiled again at the next start); a TORCHINDUCTOR_CACHE_DIR the user set wins. Assigned, not
        # setdefault: by now torch has put its temp folder into the variable (see COMPILE_CACHE_GIVEN), and it
        # reads the variable each time it needs the folder
        if not COMPILE_CACHE_GIVEN:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(HERE, ".compile_cache")
        # dynamic=False: a new shape (a halved micro-batch after an out-of-memory error) gets its own static graph
        # instead of turning the training graph into a slower dynamic-shape one
        compiled = torch.compile(model, dynamic=False)
        # the shape, the strides and the autocast state training uses: Dynamo guards on all three, so this graph is
        # the one step 1 reuses (a batch is the first ctx columns of a (micro, ctx + 1) tensor: input_and_target)
        x, _ = input_and_target(torch.zeros((max(1, int(micro)), model.ctx + 1), dtype=torch.long, device=device))
        # fork_rng: the warm-up's dropout masks must not consume the (restored) random state of the run
        with fork_device_rng(device):
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                if return_aux:  # a mixture of experts: training asks for the aux terms as well
                    out, aux = compiled(x, return_hidden=return_hidden, return_aux=True)
                    out = out.float().sum() + aux["lb"] + aux["z"]
                else:
                    out = compiled(x, return_hidden=return_hidden)
            out.float().sum().backward()
        del out
    except Exception as exc:  # noqa: BLE001
        failure = (type(exc).__name__, str(exc))  # not the exception: its traceback would keep the warm-up's tensors
    model.zero_grad(set_to_none=True)
    if failure is None:
        if device.type == "cuda":  # the loss in pieces (now, or after a first-step out-of-memory) is compiled too
            set_chunk_loss_compiled(True)
        return compiled
    free_device_memory(device)
    if compiled is not None and "out of memory" in failure[1].lower():
        print("torch.compile warm-up ran out of memory; keeping the compiled model (the first step's "
              "out-of-memory handling picks the memory savings).")
        if device.type == "cuda":  # the loss in pieces that handling is likely to switch on runs compiled as well
            set_chunk_loss_compiled(True)
        return compiled
    print(f"torch.compile failed ({failure[0]}: {failure[1][:120]}); using eager mode.")
    return model


def cmd_plan(args):
    dataset_dir = ensure_dataset(args)
    if not dataset_dir:
        raise SystemExit("--plan needs --dataset.")
    manifest = data_prep.read_manifest(dataset_dir)
    device = select_device(args.device)
    amp_dtype, amp_name, _ = choose_precision(device, args.precision)
    if args.resume or args.init_from:
        # the plan never opens a checkpoint (so it cannot change one); a resumed run keeps its own settings
        print(f"Note: --plan describes a new run on this dataset; {'--resume' if args.resume else '--init-from'} is "
              "ignored here (a resumed run keeps its checkpoint's settings, printed in its run header).")
    if args.time_budget_hours and args.device != "cpu" and gpu_busy(args.out_dir, HERE, smi=False):
        print("WARNING: " + GPU_BUSY_MESSAGE)  # the time budget is planned from training steps timed now
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
                                                           amp_dtype, args.tokens_per_param, args.max_epochs,
                                                           args.optimizer or "adamw"):
        print(f"  {unique:>14,.0f} {planned:>14,.0f} {c.d_model:>5} {c.n_layers:>3} {c.n_heads:>3}/{c.n_kv_heads:<2} "
              f"{c.ctx:>5} {n:>11,} {planned / n:>9.1f} {tps:>9,} {fits:>12} {lr:>8.1e}")


def _log_path_from(argv):
    """--log-file is read before full parsing so argument errors are logged too: by a parser that knows this one
    option, so that an abbreviation ('--log-f X') means here what it means to parse_args. '--l', '--lo', '--log'
    and '--log-' are not shown to it: parse_args refuses them as ambiguous (--log-every, --lr, ...), so the
    value after a mistyped '--log 50' is no log file."""
    default = os.path.join(HERE, "log.txt")
    pre = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    pre.add_argument("--log-file", default=default)
    try:
        return pre.parse_known_args([item for item in argv if item.split("=", 1)[0] not in
                                     ("--l", "--lo", "--log", "--log-")])[0].log_file
    except argparse.ArgumentError:  # no value after it: parse_args reports that, into the default log
        return default


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if any(item in ("-h", "--help") for item in argv):
        parse_args(argv)  # prints the options and exits: the help text is no part of a run's log
        return
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
