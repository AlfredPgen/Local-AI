r"""Supervised fine-tuning (SFT) and preference tuning (DPO) for tinyGPT.

Pre-training (tiny_gpt.py) teaches the model language and facts by predicting
the next token of raw text. Post-training teaches it to *answer*:

  sft   learn from question -> answer pairs. The loss is computed only on the
        answer tokens, so the model learns to respond rather than to write
        questions. Data: JSON Lines {"prompt": "...", "response": "..."}.
  dpo   learn which of two answers is preferred (Direct Preference
        Optimisation, Rafailov et al. 2023). The starting model is the
        reference (its log-probabilities of every pair are computed once,
        before the first update); the loss pushes the log-probability ratio of
        the chosen answer above that of the rejected one:
            L = -log sigmoid( beta * [(log pi(c) - log ref(c)) - (log pi(r) - log ref(r))] )
        Data: {"prompt": "...", "chosen": "...", "rejected": "..."}.
  ask   answer a question with a fine-tuned checkpoint (uses the same template).

Reinforcement learning (e.g. GRPO with automatically checkable answers) is
described in the learning guide; it needs a reward signal and a stronger base
model than tinyGPT is today, so it is not implemented here.

Examples
--------
    python finetune.py sft --base tinyGPT_best.pt --data finetune_examples\sft_examples.jsonl --name tinyGPT_sft
    python finetune.py dpo --base tinyGPT_sft.pt --data finetune_examples\dpo_examples.jsonl --name tinyGPT_dpo
    python finetune.py ask --checkpoint tinyGPT_sft.pt "What does genetic drift do to small populations?"

All printed output is appended to log.txt, like tiny_gpt.py. Use --device cpu
while another training run is using the GPU (the GPU is refused then; --force
overrides). Batches hold examples of similar length, so little compute goes into
padding (--no-length-grouping: plain shuffled batches).
"""

import argparse
import dataclasses
import gc
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiny_gpt  # noqa: E402


TEMPLATE = "### Question:\n{prompt}\n\n### Answer:\n"


def read_jsonl(path, fields):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON ({exc})")
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: expected a JSON object")
            missing = [f for f in fields if not isinstance(row.get(f), str) or not row[f].strip()]
            if missing:
                raise SystemExit(f"{path}:{line_no}: missing text field(s) {missing}")
            rows.append(row)
    if len(rows) < 2:
        raise SystemExit(f"{path}: need at least 2 examples")
    return rows


def encode_texts(tok, texts):
    """Token ids of many texts in one call, on every CPU thread (tokenizer's
    encode_batch: ~15x faster than one encode per text); one at a time for a
    tokenizer without it."""
    texts = list(texts)
    batch = getattr(tok, "encode_batch", None)
    if batch is None or not texts:
        return [tok.encode(t) for t in texts]
    return batch(texts, threads=os.cpu_count() or 1)


def pair_from_ids(tok, question, response_ids, ctx):
    """(ids, number of prompt tokens, truncated?) from already tokenised parts:
    `question` is the template with the prompt, `response_ids` the answer without
    its end token. The response ends with EOS so the model learns to stop. If the
    pair is longer than the context, the question is cut from the left (keeping
    the template's '### Answer:' next to the answer) and then the end of the
    answer, so training always covers answer tokens."""
    bos = [tok.bos_id] if tok.bos_id >= 0 else []
    question = list(question)
    answer = list(response_ids) + ([tok.eos_id] if tok.eos_id >= 0 else [])
    room = ctx + 1 - len(bos)
    truncated = len(question) + len(answer) > room
    if truncated:
        answer = answer[: max(1, room - min(len(question), room // 2))]
        question = question[-(room - len(answer)):]
    prefix = bos + question
    ids = prefix + answer
    return ids, len(prefix), truncated


def batch_tensors(items, device):
    """Right-padded inputs and labels; labels are -100 on prompt tokens and padding."""
    width = max(len(item[0]) for item in items) - 1
    x = torch.zeros((len(items), width), dtype=torch.long)
    y = torch.full((len(items), width), -100, dtype=torch.long)
    for i, (ids, n_prompt, *_) in enumerate(items):
        seq = torch.tensor(ids)
        x[i, :len(ids) - 1] = seq[:-1]
        target = seq[1:].clone()
        target[: n_prompt - 1] = -100
        y[i, :len(ids) - 1] = target
    return x.to(device), y.to(device)


def sequence_logp(model, x, y, amp_dtype, device):
    """Sum of log-probabilities of the labelled (answer) tokens, per sequence
    (the output layer applied in pieces: no full tokens x vocabulary table)."""
    with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        return tiny_gpt.target_logprobs(model, x, y).sum(-1)


def pieces(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def is_oom(exc):
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# Hand freed blocks back after an out-of-memory error (call it outside the `except` block: the error's traceback
# still holds the failed step's tensors); the same helper the training script uses, on every device
free_memory = tiny_gpt.free_device_memory


def item_length(item):
    """Padded width an SFT example or DPO pair needs (the longer answer for DPO)."""
    return len(item[0]) if isinstance(item[0][0], int) else max(len(item[0][0]), len(item[1][0]))


def length_grouped_batches(items, batch, rng, group=50):
    """Batches of similar length, so little compute goes into padding (as Hugging
    Face's group_by_length): shuffle, sort each mega-batch of `group` x `batch`
    examples by length, cut it into batches, then shuffle the batch order. There
    are as many batches as with plain slicing."""
    items = list(items)
    rng.shuffle(items)
    out = []
    for i in range(0, len(items), group * batch):
        mega = sorted(items[i:i + group * batch], key=item_length)
        out += pieces(mega, batch)
    rng.shuffle(out)
    return out


def default_lr(n_params, mode):
    """Fine-tuning learning rate by base model size: 1e-4 below 300M parameters,
    5e-5 up to 1B, 2e-5 above (bigger models move further per step at the same
    rate); DPO one tenth of that (its loss is a small preference margin on a
    model that already answers)."""
    lr = 1e-4 if n_params < 300e6 else 5e-5 if n_params <= 1e9 else 2e-5
    return lr / 10 if mode == "dpo" else lr


def gpu_busy(out_dir=None):
    """Another training job on this machine (tiny_gpt.gpu_busy: live training locks here and in out_dir, or an
    occupied NVIDIA GPU)."""
    try:
        dirs = (HERE,) + ((out_dir,) if out_dir and os.path.abspath(out_dir) != os.path.abspath(HERE) else ())
        return tiny_gpt.gpu_busy(*dirs)
    except Exception:  # noqa: BLE001
        return False


def check_gpu_free(device, args):
    # every accelerator, not only CUDA: on a Mac a second trainer competes for the unified memory
    if device.type != "cpu" and not args.force and gpu_busy(getattr(args, "out_dir", None)):
        raise SystemExit(tiny_gpt.GPU_BUSY_MESSAGE + " (Or use --device cpu, or pass --force.)")


def save(model, tok, cfg, base_obj, path, info, history):
    """The fine-tune curve lives in finetune['history']; metrics stay the base
    model's pre-training history, so dashboards label every curve correctly.
    The base model's last evaluation (validation loss, fact benchmark,
    calibration) is kept as base_last_eval, not last_eval: it was measured on
    the base weights, so the viewer scores this checkpoint itself. Returns the
    path written (atomic_save falls back to another name if the file is locked)."""
    base_metrics = base_obj.get("metrics") or {}
    payload = {
        "format": tiny_gpt.CHECKPOINT_FORMAT, "format_version": tiny_gpt.FORMAT_VERSION,
        "checkpoint_type": "inference", "created": tiny_gpt.now_iso(), "torch_version": str(torch.__version__),
        "model_config": dataclasses.asdict(cfg), "model": tiny_gpt.cpu_state_dict(model),
        "tokenizer": {"proto": tok.proto, **tok.meta()}, "dataset": base_obj.get("dataset"),
        "step": base_obj.get("step"), "train_config": base_obj.get("train_config"),
        "finetune": {**info, "history": history,
                     **({"previous": base_obj["finetune"]} if base_obj.get("finetune") else {})},
        "metrics": {"history": base_metrics.get("history", []), "best_val": base_metrics.get("best_val"),
                    "best_step": base_metrics.get("best_step"), "last_eval": None,
                    "base_last_eval": base_metrics.get("last_eval") or base_metrics.get("base_last_eval")},
    }
    return tiny_gpt.atomic_save(payload, path)


def train(args):
    device = tiny_gpt.select_device(args.device)
    check_gpu_free(device, args)
    amp_dtype, amp_name, need_scaler = tiny_gpt.choose_precision(device, args.precision)
    scaler = torch.amp.GradScaler(device.type) if need_scaler else None  # fp16 needs loss scaling
    model, tok, cfg, base_obj = tiny_gpt.load_for_inference(args.base, device, args.trust_checkpoint)
    # keep only what save() needs: the raw weights (and any optimizer state) would sit in RAM until the end
    base_obj = {k: base_obj.get(k) for k in ("dataset", "step", "train_config", "metrics", "finetune")}
    gc.collect()
    model.set_attention_impl(tiny_gpt.choose_attention(device, cfg, amp_dtype, "auto")[0])
    out_path = os.path.join(args.out_dir, args.name + ".pt")
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"{out_path} exists; choose a new --name or pass --overwrite.")
    fields = ["prompt", "response"] if args.mode == "sft" else ["prompt", "chosen", "rejected"]
    rows = read_jsonl(args.data, fields)
    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(round(args.val_fraction * len(rows))))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    if args.lr is None:
        args.lr = default_lr(tiny_gpt.count_params(cfg)[0], args.mode)
        lr_note = " (default for this model size; --lr sets it)"
    else:
        lr_note = ""
    model.train()
    model.grad_checkpoint = args.activation_checkpointing
    mem = {"micro": min(args.micro_batch or args.batch, args.batch)}  # halved, or checkpointing added, on OOM
    # --freeze-router: a mixture-of-experts model keeps its pre-trained routing (the router matrices get no
    # gradient; the balancing bias buffers are never updated here in any case). Off by default: OLMoE's released
    # SFT trained the router along with the experts, without a balance loss.
    frozen = 0
    if getattr(args, "freeze_router", False):
        for name, p in model.named_parameters():
            if name.endswith("router.weight"):
                p.requires_grad_(False)
                frozen += 1
        print(f"router frozen: {frozen} routing matri{'x' if frozen == 1 else 'ces'} keep their pre-trained weights"
              if frozen else "note: --freeze-router has no effect, this model has no mixture-of-experts router")
    decay = [p for p in model.parameters() if p.dim() >= 2 and p.requires_grad]
    other = [p for p in model.parameters() if p.dim() < 2 and p.requires_grad]
    opt, _ = tiny_gpt.make_adamw([{"params": decay, "weight_decay": args.weight_decay},
                                  {"params": other, "weight_decay": 0.0}], device, lr=args.lr, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train_rows) / args.batch)
    total = steps_per_epoch * args.epochs
    schedule = tiny_gpt.LRSchedule("cosine", total, args.lr, args.lr * 0.1, max(1, total // 10))
    print(f"{args.mode.upper()} from {args.base} on {tiny_gpt.device_label(device)} ({amp_name}) | "
          f"{len(train_rows)} training and {len(val_rows)} held-out examples | {args.epochs} epochs x "
          f"{steps_per_epoch} steps | lr {args.lr:.1e}{lr_note}" + (f" | beta {args.beta}" if args.mode == "dpo"
                                                                     else ""))

    def encode_rows(rows_):
        # two tokenizer calls for all rows (the questions, then every answer) instead of two or four per row
        questions = encode_texts(tok, [TEMPLATE.format(prompt=r["prompt"].strip()) for r in rows_])
        keys = ("response",) if args.mode == "sft" else ("chosen", "rejected")
        answers = encode_texts(tok, [r[k].strip() for r in rows_ for k in keys])
        pairs = [pair_from_ids(tok, questions[i], answers[i * len(keys) + j], cfg.ctx)
                 for i in range(len(rows_)) for j in range(len(keys))]
        if args.mode == "sft":
            return pairs
        return [(pairs[2 * i], pairs[2 * i + 1]) for i in range(len(rows_))]

    train_items, val_items = encode_rows(train_rows), encode_rows(val_rows)
    cut = sum(it[2] if args.mode == "sft" else (it[0][2] or it[1][2]) for it in train_items + val_items)
    if cut:
        print(f"note: {cut} example(s) longer than the context ({cfg.ctx} tokens) were shortened: the question from "
              "the left, then the end of the answer")

    if args.mode == "dpo":
        # the reference is the starting model; score every pair once before the first update
        model.eval()
        with torch.no_grad():
            for items in (train_items, val_items):
                i = 0
                while i < len(items):
                    chunk = items[i:i + mem["micro"]]
                    oom = False
                    try:
                        xc, yc = batch_tensors([c for c, *_ in chunk], device)
                        xr, yr = batch_tensors([r for _, r, *_ in chunk], device)
                        rc = sequence_logp(model, xc, yc, amp_dtype, device).tolist()
                        rr = sequence_logp(model, xr, yr, amp_dtype, device).tolist()
                    except RuntimeError as exc:
                        if not is_oom(exc) or mem["micro"] == 1:
                            raise
                        oom = True
                    if oom:  # outside the except block, so the failed piece's tensors can be freed
                        xc = yc = xr = yr = None
                        mem["micro"] //= 2
                        free_memory(device)
                        continue
                    items[i:i + len(chunk)] = [(c, r, a, b) for (c, r, *_), a, b in zip(chunk, rc, rr)]
                    i += len(chunk)
        model.train()
        free_memory(device)
    val_items.sort(key=item_length)  # held-out batches of similar length: less padding, same averages

    def step_loss(items):
        """(mean loss, sum of per-unit losses, number of units, extra sums); units are answer tokens
        for SFT and pairs for DPO, so held-out averages weight every unit equally."""
        if args.mode == "sft":
            x, y = batch_tensors(items, device)
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                total, _, n = tiny_gpt.chunked_lm_loss(model(x, return_hidden=True), model.lm_head.weight, y)
            n = int(n)
            return total / max(n, 1), total.item(), n, {}
        xc, yc = batch_tensors([c for c, *_ in items], device)
        xr, yr = batch_tensors([r for _, r, *_ in items], device)
        pc, pr = sequence_logp(model, xc, yc, amp_dtype, device), sequence_logp(model, xr, yr, amp_dtype, device)
        rc = torch.tensor([it[2] for it in items], device=device)
        rr = torch.tensor([it[3] for it in items], device=device)
        margin = args.beta * ((pc - rc) - (pr - rr))
        losses = -F.logsigmoid(margin)
        return losses.mean(), losses.sum().item(), len(items), {"reward_accuracy": (margin > 0).float().sum().item(),
                                                                 "margin": margin.sum().item()}

    def units_of(items):
        return sum(len(ids) - n_prompt for ids, n_prompt, *_ in items) if args.mode == "sft" else len(items)

    def train_batch(items):
        """Gradients of one optimizer step's batch, accumulated over micro-batches of
        mem['micro'] examples (each weighted by its share of the batch's units); on
        out-of-memory the micro-batch is halved, then activation checkpointing is
        switched on, and the batch starts again. Returns the batch's mean loss."""
        items = sorted(items, key=item_length)  # pieces of similar length: less padding, the same gradient
        while True:
            opt.zero_grad(set_to_none=True)
            total_units = units_of(items)
            loss_sum, loss = 0.0, None
            try:
                for part in pieces(items, mem["micro"]):
                    loss, part_sum, n, _ = step_loss(part)
                    weight = n / max(total_units, 1)
                    if scaler is not None:
                        scaler.scale(loss * weight).backward()
                    else:
                        (loss * weight).backward()
                    loss_sum += part_sum
                return loss_sum / max(total_units, 1)
            except RuntimeError as exc:
                if not is_oom(exc) or (mem["micro"] == 1 and model.grad_checkpoint):
                    raise
            # out of memory: drop the failed piece's graph (held by `loss` and, inside the except block, by the
            # error's traceback) and its gradients before the retry, or the retry runs out of memory too
            loss = None
            opt.zero_grad(set_to_none=True)
            free_memory(device)
            if mem["micro"] > 1:
                mem["micro"] //= 2
                print(f"out of memory: micro-batch {mem['micro']} (gradients accumulated to {args.batch})")
            else:
                model.grad_checkpoint = True
                print("out of memory at micro-batch 1: activation checkpointing on")

    def evaluate_items(items):
        model.eval()
        total, units, extra = 0.0, 0, {}
        with torch.no_grad():
            for i in range(0, len(items), mem["micro"]):
                _, loss_sum, n, info = step_loss(items[i:i + mem["micro"]])
                total, units = total + loss_sum, units + n
                for k, v in info.items():
                    extra[k] = extra.get(k, 0.0) + v
        model.train()
        pairs = len(items)
        return total / max(units, 1), {k: v / max(pairs, 1) for k, v in extra.items()}

    info = {"method": args.mode, "base": os.path.abspath(args.base), "data": os.path.abspath(args.data),
            "examples": len(rows), "epochs": args.epochs, "lr": args.lr, "template": TEMPLATE,
            **({"beta": args.beta} if args.mode == "dpo" else {})}
    # each finished epoch but the last is kept in <name>_partial.pt, so an interruption loses at most one epoch;
    # <name>.pt is written only by a finished run
    partial_path = os.path.join(args.out_dir, args.name + "_partial.pt")
    history, step, t0 = [], 0, time.time()
    val_loss, val_info = evaluate_items(val_items)
    print(f"epoch 0 | held-out loss {val_loss:.4f}" + "".join(f" | {k} {v:.3f}" for k, v in val_info.items()))
    t_train = last_beat = time.time()  # the rate excludes the initial evaluation
    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        if args.no_length_grouping:
            rng.shuffle(train_items)
            batches = pieces(train_items, args.batch)
        else:
            batches = length_grouped_batches(train_items, args.batch, rng)
        running = []
        for batch in batches:
            step += 1
            for group in opt.param_groups:
                group["lr"] = schedule.lr_at(step)
            loss = train_batch(batch)
            if scaler is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            running.append(loss)
            now = time.time()
            if now - last_beat >= 60 or step == total:  # a progress line about once a minute
                rate = (now - t_train) / max(step, 1)
                recent = running[-50:]
                print(f"  ... step {step:,}/{total:,} | epoch {epoch} | train loss {sum(recent) / len(recent):.4f} "
                      f"(last {len(recent)} steps) | lr {schedule.lr_at(step):.2e} | {rate:.2f} s/step | "
                      f"elapsed {tiny_gpt.fmt_hms(now - t0)} | left {tiny_gpt.fmt_hms(rate * (total - step))}",
                      flush=True)
                last_beat = now
        val_loss, val_info = evaluate_items(val_items)
        history.append({"epoch": epoch, "step": step, "train_loss": sum(running) / len(running),
                        "val_loss": val_loss, **val_info})
        print(f"epoch {epoch} | train loss {history[-1]['train_loss']:.4f} | held-out loss {val_loss:.4f}"
              + "".join(f" | {k} {v:.3f}" for k, v in val_info.items()) + f" | {time.time() - t0:,.0f} s")
        if epoch < args.epochs:
            kept = save(model, tok, cfg, base_obj, partial_path, {**info, "epochs_done": epoch}, history)
            print(f"saved {kept} (epoch {epoch} of {args.epochs}; a rerun starts again from the base)")
    prompt = val_rows[0]["prompt"]
    answer = answer_question(model, tok, prompt, device, amp_dtype, args.max_new_tokens)
    print(f"held-out question: {prompt}\nanswer: {answer}")
    saved = save(model, tok, cfg, base_obj, out_path, info, history)
    print(f"saved {saved}")
    try:
        os.remove(partial_path)
    except OSError:
        pass
    tiny_gpt.record_experiment(args.out_dir, {
        "event": f"fine-tune ({args.mode})", "name": args.name, "status": "completed", "checkpoint": saved,
        "notes": f"base {os.path.basename(args.base)}; {len(rows)} examples; held-out loss {val_loss:.4f}"
                 + "".join(f"; {k} {v:.3f}" for k, v in val_info.items())})
    return saved


def answer_question(model, tok, prompt, device, amp_dtype, max_new_tokens, seed=0):
    text = tiny_gpt.generate(model, tok, TEMPLATE.format(prompt=prompt.strip()), max_new_tokens, device, amp_dtype,
                             0.7, 40, 0.9, torch.Generator().manual_seed(seed))
    return text.split("### Answer:", 1)[-1].strip()


def ask(args):
    device = tiny_gpt.select_device(args.device)
    check_gpu_free(device, args)
    amp_dtype, _, _ = tiny_gpt.choose_precision(device, args.precision, training=False)  # generation only
    model, tok, cfg, _ = tiny_gpt.load_for_inference(args.checkpoint, device, args.trust_checkpoint)
    print(answer_question(model, tok, args.question, device, amp_dtype, args.max_new_tokens, args.seed))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    log = tiny_gpt._log_path_from(argv)  # understands both "--log-file PATH" and "--log-file=PATH"
    # --help is no run: its usage text is printed, not written to the log (argument errors are still logged)
    if not any(a in ("-h", "--help") for a in argv):
        tiny_gpt.start_log(log)
        print(f"\n--- finetune {tiny_gpt.now_iso()} | {' '.join(argv)} ---")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for mode in ("sft", "dpo"):
        s = sub.add_parser(mode)
        s.add_argument("--base", required=True, help="checkpoint to start from")
        s.add_argument("--data", required=True, help="JSON Lines file")
        s.add_argument("--name", required=True, help="output checkpoint name")
        s.add_argument("--out-dir", default=HERE)
        s.add_argument("--epochs", type=int, default=3)
        s.add_argument("--batch", type=int, default=8, help="examples per optimizer step")
        s.add_argument("--micro-batch", type=int,
                       help="examples per forward pass (default: --batch); gradients are accumulated to --batch, "
                            "and an out-of-memory error halves it automatically")
        s.add_argument("--activation-checkpointing", action="store_true",
                       help="recompute blocks in the backward pass (less memory, slower; switched on "
                            "automatically after an out-of-memory error at micro-batch 1)")
        s.add_argument("--lr", type=float, default=None,
                       help="learning rate (default by model size: 1e-4 below 300M parameters, 5e-5 up to 1B, 2e-5 "
                            "above; DPO one tenth)")
        s.add_argument("--beta", type=float, default=0.1, help="DPO: strength of the preference")
        s.add_argument("--weight-decay", type=float, default=0.0)
        s.add_argument("--val-fraction", type=float, default=0.1)
        s.add_argument("--max-new-tokens", type=int, default=80)
        s.add_argument("--no-length-grouping", action="store_true",
                       help="plain shuffled batches (default: batches of examples of similar length, so less "
                            "compute goes into padding)")
        s.add_argument("--freeze-router", action="store_true",
                       help="mixture-of-experts models: keep the routers as pre-trained (default: they train too)")
        s.add_argument("--overwrite", action="store_true")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--max-new-tokens", type=int, default=120)
    for s in (sub.choices["sft"], sub.choices["dpo"], a):
        s.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
        s.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--trust-checkpoint", action="store_true")
        s.add_argument("--force", action="store_true", help="use the GPU even if another job seems to be using it")
        s.add_argument("--log-file", default=log)
    try:
        args = p.parse_args(argv)
        if args.mode == "ask":
            ask(args)
            return
        if not tiny_gpt.NAME_RE.fullmatch(args.name):
            p.error("--name may contain letters, digits, '.', '_' and '-' only")
        for path in (args.base, args.data):
            if not os.path.exists(path):
                p.error(f"not found: {path}")
        if not (1 <= args.epochs <= 100 and 1 <= args.batch <= 1024 and (args.lr is None or 0 < args.lr < 1)
                and 0 < args.val_fraction < 0.5
                and (args.micro_batch is None or 1 <= args.micro_batch <= args.batch)):
            p.error("check --epochs (1-100), --batch (1-1024), --micro-batch (1 to --batch), --lr (0-1) and "
                    "--val-fraction (0-0.5)")
        # a fine-tune takes the same <name>.train.lock as a training run, so gpu_busy() sees it on every device
        lock = tiny_gpt.acquire_run_lock(args)
        try:
            return train(args)
        finally:
            tiny_gpt.release_run_lock(lock)
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"ERROR: {exc.code}" if isinstance(exc.code, str) else f"exit code {exc.code}", file=sys.stderr)
        raise
    finally:
        tiny_gpt.stop_log()


if __name__ == "__main__":
    main()
