r"""Set the fact benchmark's difficulty labels from how reference language models actually score.

    python calibrate_probes.py                    # score, write difficulty labels, write a report
    python calibrate_probes.py --report-only      # score and report; leave the probe file unchanged
    python calibrate_probes.py --device cuda      # only when no training run is using the GPU

Reference models are small open models with the Llama design, the same as tinyGPT's
(RoPE, RMSNorm, SwiGLU, grouped-query attention). They are downloaded once from Hugging
Face and run by tinyGPT's own model code; reading them needs only the huggingface_hub
and tokenizers packages (pip install huggingface_hub "tokenizers>=0.15"). Each model
answers every item the way tinyGPT does: the correct continuation must beat every
distractor in log-probability per character. An item is then labelled
    easy    if at least 3/4 of the reference models answer it correctly,
    hard    if at most 1/4 do,
    medium  otherwise.
tinyGPT itself is never used for the labels, so its scores per difficulty stay unbiased.

Items that every reference model gets wrong, all preferring the same distractor, are
listed for a manual check: that distractor may also be correct, or the item ambiguous.
"""

import argparse
import csv
import datetime
import hashlib
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiny_gpt  # noqa: E402

DEFAULT_MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "HuggingFaceTB/SmolLM2-1.7B",
]
SANITY_TEXT = ("Genetic drift is the change in the frequency of an existing gene variant in a population due to "
               "random chance. It may cause gene variants to disappear completely and thereby reduce genetic "
               "variation. It can also cause initially rare alleles to become much more frequent and even fixed. "
               "When few copies of an allele exist, the effect of genetic drift is more notable, and when many "
               "copies exist, the effect is less notable.")


# ---------------------------------------------------------------------------
# reading Hugging Face Llama checkpoints into tinyGPT's model
# ---------------------------------------------------------------------------
def download(repo):
    """config, tokenizer and weights from the Hugging Face cache (downloaded once)."""
    from huggingface_hub import hf_hub_download
    files = {name: hf_hub_download(repo, name) for name in ("config.json", "tokenizer.json")}
    try:
        files["weights"] = [hf_hub_download(repo, "model.safetensors")]
    except Exception:  # noqa: BLE001 - sharded checkpoints list their parts in an index
        index = json.load(open(hf_hub_download(repo, "model.safetensors.index.json"), encoding="utf-8"))
        files["weights"] = [hf_hub_download(repo, part) for part in sorted(set(index["weight_map"].values()))]
    return files


def mmap_safetensors(path):
    """Tensors backed by a copy-on-write memory map: pages are read from disk only when
    used and never written back, so a 4 GB file does not need 4 GB of free RAM."""
    import struct
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 100 * 2 ** 20:
            raise SystemExit(f"{path}: implausible safetensors header")
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    data = np.memmap(path, dtype=np.uint8, mode="c", offset=8 + size)
    kinds = {"F32": (np.float32, None), "F16": (np.float16, None), "BF16": (np.int16, torch.bfloat16)}
    out = {}
    for name, info in header.items():
        if info["dtype"] not in kinds:
            raise SystemExit(f"{path}: unsupported dtype {info['dtype']} for {name}")
        np_type, view = kinds[info["dtype"]]
        begin, end = info["data_offsets"]
        t = torch.from_numpy(data[begin:end].view(np_type))
        out[name] = (t.view(view) if view is not None else t).reshape(info["shape"])
    return out


def interleave_rope_rows(w, n_heads):
    """Hugging Face Llama checkpoints order each head's rows as [first halves | second
    halves] (rotate-half RoPE); tinyGPT rotates adjacent pairs. Reorder the rows."""
    out_dim, in_dim = w.shape
    head = out_dim // n_heads
    return w.reshape(n_heads, 2, head // 2, in_dim).transpose(1, 2).reshape(out_dim, in_dim).contiguous()


def load_llama(repo, ctx=512):
    files = download(repo)
    config = json.load(open(files["config.json"], encoding="utf-8"))
    raw = {}
    for part in files["weights"]:
        raw.update(mmap_safetensors(part))
    model, cfg = tinygpt_from_llama(config, raw, ctx, repo)
    return model, HFTokenizer(files["tokenizer.json"]), cfg


def tinygpt_from_llama(config, raw, ctx=512, name="model"):
    """A TinyGPT holding a Hugging Face Llama checkpoint (config dict + tensors)."""
    problems = []
    if config.get("model_type") != "llama":
        problems.append(f"model_type {config.get('model_type')} (need llama)")
    if config.get("attention_bias") or config.get("mlp_bias"):
        problems.append("bias terms")
    if config.get("rope_scaling"):
        problems.append("rope_scaling")
    if config.get("hidden_act", "silu") != "silu":
        problems.append(f"activation {config.get('hidden_act')}")
    heads = config["num_attention_heads"]
    if config.get("head_dim", config["hidden_size"] // heads) != config["hidden_size"] // heads:
        problems.append("head_dim differs from hidden_size / heads")
    if problems:
        raise SystemExit(f"{name} cannot run on tinyGPT's model code: " + ", ".join(problems))
    kv = config.get("num_key_value_heads") or heads
    cfg = tiny_gpt.ModelConfig(
        vocab_size=config["vocab_size"], ctx=ctx, d_model=config["hidden_size"], n_layers=config["num_hidden_layers"],
        n_heads=heads, n_kv_heads=kv, ffn_hidden=config["intermediate_size"],
        rope_base=float(config.get("rope_theta", 10000.0)), norm_eps=float(config.get("rms_norm_eps", 1e-6)),
        tie_embeddings=bool(config.get("tie_word_embeddings", False)))
    state = {"token_embedding.weight": raw["model.embed_tokens.weight"], "norm.weight": raw["model.norm.weight"]}
    for i in range(cfg.n_layers):
        src, dst = f"model.layers.{i}.", f"blocks.{i}."
        state[dst + "attn.q_proj.weight"] = interleave_rope_rows(raw[src + "self_attn.q_proj.weight"], heads)
        state[dst + "attn.k_proj.weight"] = interleave_rope_rows(raw[src + "self_attn.k_proj.weight"], kv)
        state[dst + "attn.v_proj.weight"] = raw[src + "self_attn.v_proj.weight"]
        state[dst + "attn.o_proj.weight"] = raw[src + "self_attn.o_proj.weight"]
        state[dst + "ffn.gate.weight"] = raw[src + "mlp.gate_proj.weight"]
        state[dst + "ffn.up.weight"] = raw[src + "mlp.up_proj.weight"]
        state[dst + "ffn.down.weight"] = raw[src + "mlp.down_proj.weight"]
        state[dst + "attn_norm.weight"] = raw[src + "input_layernorm.weight"]
        state[dst + "ffn_norm.weight"] = raw[src + "post_attention_layernorm.weight"]
    if not cfg.tie_embeddings:
        state["lm_head.weight"] = raw["lm_head.weight"]
    with torch.device("meta"):  # no memory for the random initial weights
        model = tiny_gpt.TinyGPT(cfg)
    model.load_state_dict(state, strict=not cfg.tie_embeddings, assign=True)
    if cfg.tie_embeddings:
        model.lm_head.weight = model.token_embedding.weight
    dtype = model.token_embedding.weight.dtype
    model.rope = tiny_gpt.RotaryEmbedding(cfg.head_dim, cfg.ctx, cfg.rope_base)
    for module in model.modules():  # plain float32 norm gains are fine; match the weights' dtype elsewhere
        if isinstance(module, tiny_gpt.RMSNorm):
            module.weight.data = module.weight.data.to(dtype)
    model.eval()
    return model, cfg


class HFTokenizer:
    """Hugging Face tokenizer.json with the interface score_probes uses."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(path)
        with_special = self.tk.encode("a", add_special_tokens=True).ids
        plain = self._enc("a")
        # a start token is a special token placed BEFORE the text (some tokenizers only append an end token)
        prepends = len(with_special) > len(plain) and with_special[1:1 + len(plain)] == plain
        self.bos_id = with_special[0] if prepends else -1
        joint, head = self._enc("The allele frequency"), self._enc("The allele")
        # byte-level BPE (GPT-2 style) marks a word start with a leading space; SentencePiece adds it itself
        self.space = head + self._enc(" frequency") == joint or head + self._enc("frequency") != joint

    def _enc(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def encode(self, text):
        return self._enc(text)

    def encode_continuation(self, text):
        return self._enc(" " + text if self.space else text)


@torch.no_grad()
def sanity_loss(model, tok, device):
    ids = ([tok.bos_id] if tok.bos_id >= 0 else []) + tok.encode(SANITY_TEXT)
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor(ids[1:], device=device)
    logits = model(x)[0].float()
    return float(torch.nn.functional.cross_entropy(logits, y))


# ---------------------------------------------------------------------------
def gpu_busy():
    """True if another job holds GPU memory (a training run uses several GB, even
    while it saves a checkpoint and the utilization drops) or keeps the GPU busy."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        rows = [[float(v) for v in line.split(",")] for line in out.strip().splitlines()]
        return any(used > 2500 or util >= 20 for used, util in rows)
    except Exception:  # noqa: BLE001
        return False


def model_code():
    """Source of every model class in tiny_gpt (TinyGPT, its blocks, attention,
    feed-forward, RotaryEmbedding, RMSNorm ...): a change there changes the scores."""
    classes = [c for c in vars(tiny_gpt).values() if isinstance(c, type) and issubclass(c, torch.nn.Module)
               and c.__module__ == tiny_gpt.__name__]
    return "".join(inspect.getsource(c) for c in sorted(classes, key=lambda c: c.__name__))


def check_packages():
    """Reference models need two Hugging Face packages; say so before any download."""
    missing = []
    for module, package in (("huggingface_hub", "huggingface_hub"), ("tokenizers", "tokenizers>=0.15")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit("calibrate_probes.py needs " + " and ".join(missing) + ": pip install "
                         + " ".join(f'"{m}"' for m in missing))


def lower_priority():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001
        pass


def label(share):
    return "easy" if share >= 0.75 else "hard" if share <= 0.25 else "medium"


def rewrite_difficulties(path, labels):
    """Replace the difficulty column of each item line, keeping comments and order.
    Old three-column lines (prompt, answer, distractors) become five-column lines
    (category 'general'), the same items read_probes sees."""
    lines = open(path, encoding="utf-8-sig").read().splitlines()
    out, k = [], 0
    for line in lines:
        parts = line.split("\t")
        if line.strip() and not line.lstrip().startswith("#") and len(parts) >= 3:
            if len(parts) < 5:
                parts = ["general", "unrated"] + parts[:3]
            parts[1] = labels[k]
            k += 1
            line = "\t".join(parts)
        out.append(line)
    if k != len(labels):
        raise SystemExit(f"{path}: found {k} items but have {len(labels)} labels; file not changed")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(out) + "\n")
    os.replace(tmp, path)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", default=tiny_gpt.DEFAULT_PROBES)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Hugging Face repos with the Llama design")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (leave some for other work)")
    ap.add_argument("--batch", type=int, default=16, help="sequences scored together")
    ap.add_argument("--floor", action="store_true", help="also score shuffled questions (4x slower)")
    ap.add_argument("--report-only", action="store_true", help="do not rewrite the difficulty labels")
    ap.add_argument("--force", action="store_true", help="use the GPU even if it looks busy")
    ap.add_argument("--no-record", action="store_true", help="do not add rows to experiments.csv")
    args = ap.parse_args()
    lower_priority()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and gpu_busy() and not args.force:
        raise SystemExit("The GPU looks busy (a training run?). Two GPU jobs at once crashed the driver before; "
                         "use --device cpu, or wait.")
    device = torch.device(args.device)
    probes = tiny_gpt.read_probes(args.probes)
    base = os.path.splitext(args.probes)[0]
    print(f"{len(probes)} items from {args.probes}; {len(args.models)} reference models on {device}")

    per_model = {}
    scoring = "".join(inspect.getsource(f) for f in (tiny_gpt.score_probes, tiny_gpt.shuffled_prompt, HFTokenizer,
                                                    tinygpt_from_llama, interleave_rope_rows)) + model_code()
    items_key = hashlib.sha256((json.dumps([[p["prompt"], p["answer"], p["distractors"]] for p in probes])
                                + scoring).encode()).hexdigest()[:16]  # new items or new scoring code: rescore
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "tinygpt_calibration")
    def cache_path(repo):
        return os.path.join(cache_dir, f"{repo.replace('/', '__')}_{items_key}{'_floor' if args.floor else ''}.json")

    if not all(os.path.isfile(cache_path(repo)) for repo in args.models):
        check_packages()
    for repo in args.models:
        started = datetime.datetime.now()
        cache = cache_path(repo)
        if os.path.isfile(cache):  # this model already scored exactly these items with this code
            try:
                with open(cache, encoding="utf-8") as handle:
                    m = json.load(handle)
                if len(m["results"]) != len(probes):
                    raise ValueError("wrong number of items")
                for result, probe in zip(m["results"], probes):  # labels may have been renamed since
                    result["category"], result["difficulty"] = probe["category"], probe["difficulty"]
                per_model[repo] = m
                print(f"{repo}: {m['params'] / 1e6:,.0f}M parameters | accuracy {m['acc']:.1%} (cached)")
                continue
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"{repo}: the cached scores in {cache} are unreadable ({exc}); scoring again")
                check_packages()
        model, tok, cfg = load_llama(repo)
        n_params = sum(p.numel() for p in model.parameters())
        if device.type == "cpu":
            # float32 on the CPU (bfloat16 arithmetic is emulated and slow), if it fits in memory
            try:
                import psutil
                free = psutil.virtual_memory().available
            except ImportError:
                free = float("inf")
            if n_params * 4 > 0.7 * free:
                print(f"{repo}: SKIPPED on the CPU - needs {n_params * 4 / 2 ** 30:.1f} GiB in float32, "
                      f"{free / 2 ** 30:.1f} GiB free. Rerun later, or with --device cuda when no training is running.")
                del model
                continue
            model.float()
        elif model.token_embedding.weight.dtype == torch.float32:
            model.to(torch.bfloat16)  # GPU: half the memory, fast tensor cores
        model.to(device)
        loss = sanity_loss(model, tok, device)
        if loss > 5.0:
            print(f"{repo}: SKIPPED - reads ordinary English at {loss:.2f} nats/token (expected about 1-3.5), "
                  "so the conversion is wrong for this model")
            continue
        acc, results = tiny_gpt.score_probes(model, tok, probes, device, None, batch_size=args.batch,
                                             floor=args.floor)
        floor = tiny_gpt.probe_floor(results)
        secs = (datetime.datetime.now() - started).total_seconds()
        print(f"{repo}: {n_params / 1e6:,.0f}M parameters | sanity loss {loss:.2f} nats/token | accuracy {acc:.1%}"
              + (f" | floor {floor:.1%}" if floor is not None else "") + f" | {secs:.0f} s")
        per_model[repo] = {"params": n_params, "acc": acc, "floor": floor, "results": results, "loss": loss}
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache + ".tmp", "w", encoding="utf-8") as handle:  # then renamed: a cut-off write leaves no cache
            json.dump(per_model[repo], handle)
        os.replace(cache + ".tmp", cache)
        if not args.no_record:
            tiny_gpt.record_experiment(HERE, {
                "event": "reference benchmark", "name": repo, "params": n_params, "d_model": cfg.d_model,
                "layers": cfg.n_layers, "heads": cfg.n_heads, "kv_heads": cfg.n_kv_heads, "vocab": cfg.vocab_size,
                "benchmark_accuracy": f"{acc:.3f}", "benchmark_floor": f"{floor:.3f}" if floor is not None else None,
                "benchmark_items": len(probes), "notes": f"{os.path.basename(args.probes)}; calibrate_probes.py"})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(per_model) < 2:
        raise SystemExit("Fewer than 2 reference models scored; difficulty labels not changed.")

    repos = list(per_model)
    strongest = max(repos, key=lambda r: per_model[r]["acc"])
    rows, labels, flagged = [], [], []
    for j, probe in enumerate(probes):
        right = [per_model[r]["results"][j]["correct"] for r in repos]
        share = sum(right) / len(right)
        margins = [per_model[r]["results"][j]["correct_score"] - per_model[r]["results"][j]["best_distractor_score"]
                   for r in repos]
        winners = {per_model[r]["results"][j]["best_distractor"] for r in repos}
        labels.append(label(share))
        if not any(right) and len(winners) == 1:
            flagged.append((probe, next(iter(winners)), float(np.mean(margins))))
        rows.append([probe["category"], labels[-1], f"{share:.2f}", f"{np.mean(margins):.3f}", probe["prompt"],
                     probe["answer"], " | ".join(probe["distractors"])]
                    + [int(r) for r in right])

    with open(base + "_calibration.csv", "w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["category", "difficulty", "share_correct", "mean_margin", "prompt", "answer", "distractors"]
                   + [r.split("/")[-1] for r in repos])
        w.writerows(rows)
    counts = {d: labels.count(d) for d in ("easy", "medium", "hard")}
    lines = [f"# Difficulty calibration of {os.path.basename(args.probes)}", "",
             f"{datetime.datetime.now():%Y-%m-%d %H:%M}; {len(probes)} items; labels: easy if at least 3/4 of the "
             "reference models are right, hard if at most 1/4, medium otherwise.", "",
             "| Reference model | Parameters | Accuracy | Floor | Sanity loss |", "|---|---|---|---|---|"]
    for r in repos:
        m = per_model[r]
        lines.append(f"| {r} | {m['params'] / 1e6:,.0f}M | {m['acc']:.1%} | "
                     + (f"{m['floor']:.1%}" if m["floor"] is not None else "-") + f" | {m['loss']:.2f} |")
    lines += ["", f"Labels: easy {counts['easy']}, medium {counts['medium']}, hard {counts['hard']}.", "",
              "## Accuracy by category", "", "| Category | " + " | ".join(r.split("/")[-1] for r in repos) + " |",
              "|---|" + "---|" * len(repos)]
    for cat in sorted({p["category"] for p in probes}):
        cells = []
        for r in repos:
            rs = [x for x in per_model[r]["results"] if x["category"] == cat]
            cells.append(f"{sum(x['correct'] for x in rs) / len(rs):.0%}" if rs else "-")
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines += ["", f"## Items every reference model got wrong, all choosing the same distractor ({len(flagged)})", "",
              "Check each: the distractor may also be correct, or the wording ambiguous.", ""]
    for probe, winner, margin in flagged:
        lines.append(f"- [{probe['category']}] {probe['prompt']} **{probe['answer']}** "
                     f"(all chose: *{winner}*; mean margin {margin:.2f})")
    with open(base + "_calibration.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nlabels: {counts} | strongest reference: {strongest} ({per_model[strongest]['acc']:.1%}) | "
          f"flagged for checking: {len(flagged)}")
    print(f"report: {base}_calibration.md | per item: {base}_calibration.csv")
    if not args.report_only:
        rewrite_difficulties(args.probes, labels)
        print(f"difficulty labels written into {args.probes}")


if __name__ == "__main__":
    main()
