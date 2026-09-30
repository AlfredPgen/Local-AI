r"""GPU telemetry and throughput benchmarks for tiny_gpt (read-only: changes no settings).

    python gpu_check.py                    # snapshot: clocks, power limit, throttle reasons, power plan
    python gpu_check.py --load 20          # 20 s matrix-multiply load while sampling clocks each second
    python gpu_check.py --train-bench      # tiny_gpt training-step benchmark (synthetic tokens)
    python gpu_check.py --train-bench --d-model 320 --layers 7 --heads 5 --kv-heads 5 --ctx 512 --batch 32

Idle GPUs sit in P8 at ~210 MHz; that is normal. What matters is the state
*under load*: if the GPU stays in P8 / low clocks while 100% busy, a power
or thermal policy is holding it back. The script names the active reason
reported by the NVIDIA driver and the Windows power plan, and leaves every
setting for you to change.
"""

import argparse
import ctypes
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

QUERY = ["pstate", "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem", "power.draw",
         "enforced.power.limit", "power.default_limit", "temperature.gpu", "utilization.gpu",
         "memory.used", "memory.total", "clocks_event_reasons.gpu_idle", "clocks_event_reasons.sw_power_cap",
         "clocks_event_reasons.sw_thermal_slowdown", "clocks_event_reasons.hw_slowdown",
         "clocks_event_reasons.hw_thermal_slowdown", "clocks_event_reasons.hw_power_brake_slowdown"]


def nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=" + ",".join(QUERY), "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    values = [v.strip() for v in out[0].split(",")]
    return dict(zip(QUERY, values))


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def windows_power():
    info = {}
    if sys.platform != "win32":
        return info

    class Status(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    status = Status()
    if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        info["ac"] = {0: "on battery", 1: "on AC power"}.get(status.ACLineStatus, "unknown")
        info["battery"] = status.BatteryLifePercent if status.BatteryLifePercent <= 100 else None
        info["saver"] = bool(status.SystemStatusFlag)
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10).stdout
        info["scheme"] = out.split("(")[-1].split(")")[0].strip() if "(" in out else out.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return info


def diagnose(sample, loaded, power):
    notes = []
    if sample is None:
        return ["nvidia-smi is not available (not an NVIDIA system?); no driver telemetry."]
    util = num(sample["utilization.gpu"])
    sm, sm_max = num(sample["clocks.sm"]), num(sample["clocks.max.sm"])
    limit, default = num(sample["enforced.power.limit"]), num(sample["power.default_limit"])
    temp = num(sample["temperature.gpu"])
    active = lambda k: sample.get(f"clocks_event_reasons.{k}", "").lower() == "active"  # noqa: E731
    if not loaded and util < 20:
        notes.append(f"GPU idle ({util:.0f}% busy): {sample['pstate']} at {sm:.0f} MHz is the normal idle state. "
                     "Measure under load with --load or during training.")
    if loaded and (sample["pstate"] in ("P5", "P8", "P12") or sm < 0.3 * sm_max):
        notes.append(f"Under load the GPU stays in {sample['pstate']} at {sm:.0f} of {sm_max:.0f} MHz: it is being "
                     "held in a low performance state. This is a platform/driver power policy, not the "
                     "training code.")
    if limit < default * 0.9:
        notes.append(f"Enforced power limit {limit:.0f} W vs {default:.0f} W default. On ASUS laptops this is set "
                     "by the Armoury Crate operating mode (Silent/Eco lower it), by running on battery, or by a "
                     "low-wattage USB-C charger instead of the barrel charger.")
    if active("sw_power_cap"):
        notes.append("Driver reports 'SW power cap' active: clocks are limited to stay under that power limit.")
    if active("sw_thermal_slowdown"):
        hot = temp >= 85
        notes.append(f"'SW thermal slowdown' active at {temp:.0f} C: " + (
            "the GPU is hot; check vents, fan mode and ambient temperature." if hot else
            "below any overheating point, so this is a software temperature target (fan/noise profile), "
            "typically part of a Silent mode."))
    if active("hw_thermal_slowdown") or active("hw_slowdown"):
        notes.append("Hardware slowdown active: the GPU protects itself (overheating or power delivery).")
    if active("hw_power_brake_slowdown"):
        notes.append("Power-brake slowdown: the system asked the GPU to cut power (charger/battery limit).")
    if power.get("scheme"):
        notes.append(f"Windows power plan: {power['scheme']} ({power.get('ac', '?')}"
                     + (f", battery {power['battery']}%" if power.get("battery") is not None else "")
                     + (", battery saver ON" if power.get("saver") else "") + ").")
        if "silent" in power["scheme"].lower() or "eco" in power["scheme"].lower():
            notes.append("Suggestion (your choice, nothing is changed here): switch Armoury Crate to Performance "
                         "or Turbo, use the original charger, then rerun: python gpu_check.py --load 20")
    return notes


def print_sample(label, s):
    print(f"{label}: {s['pstate']} | SM {s['clocks.sm']} / {s['clocks.max.sm']} MHz | memory {s['clocks.mem']} / "
          f"{s['clocks.max.mem']} MHz | power {s['power.draw']} W of {s['enforced.power.limit']} W enforced "
          f"({s['power.default_limit']} W default) | {s['temperature.gpu']} C | busy {s['utilization.gpu']}% | "
          f"VRAM {s['memory.used']} / {s['memory.total']} MiB")
    reasons = [k.split(".")[1] for k in QUERY if k.startswith("clocks_event") and s.get(k, "").lower() == "active"]
    print(f"   clock-limit reasons active: {', '.join(reasons) or 'none'}")


def load_test(seconds):
    if not torch.cuda.is_available():
        print("CUDA not available; skipping the load test.")
        return
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            s = nvidia_smi()
            if s:
                samples.append(s)
            stop.wait(1.0)

    thread = threading.Thread(target=sampler, daemon=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    a = torch.randn(4096, 4096, device="cuda", dtype=dtype)
    b = torch.randn(4096, 4096, device="cuda", dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    thread.start()
    rates, t_end = [], time.time() + seconds
    while time.time() < t_end:
        t0 = time.perf_counter()
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()
        rates.append(20 * 2 * 4096 ** 3 / (time.perf_counter() - t0) / 1e12)
    stop.set()
    thread.join()
    busy = [s for s in samples if num(s["utilization.gpu"]) >= 50] or samples
    print(f"\nLoad test: {seconds} s of {str(dtype).split('.')[-1]} 4096x4096 matmuls: median "
          f"{statistics.median(rates):.2f} TFLOP/s (min {min(rates):.2f}, max {max(rates):.2f})")
    if busy:
        clocks = [num(s["clocks.sm"]) for s in busy]
        power = [num(s["power.draw"]) for s in busy]
        states = sorted({s["pstate"] for s in busy})
        print(f"   while busy: P-states {', '.join(states)} | SM clock median {statistics.median(clocks):.0f} MHz "
              f"(max {max(clocks):.0f}) | power median {statistics.median(power):.1f} W")
        print_sample("   last busy sample", busy[-1])
        return busy[-1]
    return None


def train_bench(args):
    import tiny_gpt
    device = tiny_gpt.select_device(args.device)
    amp_dtype, amp_name, _ = tiny_gpt.choose_precision(device, args.precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    variants = []
    base = dict(vocab_size=args.vocab, ctx=args.ctx, d_model=args.d_model, n_layers=args.layers,
                n_heads=args.heads, n_kv_heads=args.kv_heads)
    impls = ["gqa_native", "repeat_kv"] if args.kv_heads < args.heads else ["repeat_kv"]
    if args.kv_heads < args.heads and not tiny_gpt._enable_gqa_supported():
        impls = ["repeat_kv"]
    for impl in impls:
        for dropout in sorted({0.0, args.dropout}):
            variants.append((impl, dropout))
    print(f"\nTraining-step benchmark on {tiny_gpt.device_label(device)} | {amp_name} | vocab {args.vocab} | "
          f"ctx {args.ctx} | batch {args.batch} | d_model {args.d_model} | layers {args.layers} | heads "
          f"{args.heads}/{args.kv_heads} | {args.warmup} warm-up steps excluded, {args.steps} timed")
    results = []
    for impl, dropout in variants:
        cfg = tiny_gpt.ModelConfig(dropout=dropout, **base)
        torch.manual_seed(0)
        model = tiny_gpt.TinyGPT(cfg).to(device)
        model.set_attention_impl(impl)
        kwargs = {"fused": True} if device.type == "cuda" else {}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, **kwargs)
        x = torch.randint(0, args.vocab, (args.batch, args.ctx + 1), device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        def step():
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, args.vocab), x[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        for _ in range(args.warmup):
            step()
        tiny_gpt.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            step()
        tiny_gpt.synchronize(device)
        dt = (time.perf_counter() - t0) / args.steps
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else float("nan")
        tok_s = args.batch * args.ctx / dt
        results.append((impl, dropout, dt, tok_s, peak))
        print(f"  attention {impl:10s} dropout {dropout:<4} : {dt * 1000:8.1f} ms/step | {tok_s:>10,.0f} tok/s | "
              f"peak memory {peak:.2f} GiB")
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(results) > 1:
        slow = max(results, key=lambda r: r[2])
        fast = min(results, key=lambda r: r[2])
        print(f"  fastest: {fast[0]} dropout {fast[1]} ({slow[2] / fast[2]:.1f}x faster than {slow[0]} dropout {slow[1]})")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load", type=int, default=0, metavar="SECONDS", help="run a GPU load test (max 120 s)")
    p.add_argument("--train-bench", action="store_true")
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15, help="also benchmarked against dropout 0")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--force", action="store_true", help="benchmark even if the GPU looks busy")
    args = p.parse_args()
    if not 0 <= args.load <= 120:
        p.error("--load must be between 0 and 120 seconds")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    print(f"python {platform.python_version()} | torch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"{platform.platform()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 2 ** 30:.1f} GiB, {props.multi_processor_count} SMs, "
              f"BF16 {'yes' if torch.cuda.is_bf16_supported() else 'no'}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("GPU: Apple MPS (no nvidia-smi telemetry on macOS; use Activity Monitor or `sudo powermetrics`).")
    power = windows_power()
    snapshot = nvidia_smi()
    if snapshot:
        print_sample("now", snapshot)
    busy = bool(snapshot) and (num(snapshot["utilization.gpu"]) > 30 or num(snapshot["memory.used"]) > 2500)
    warning = ("\nThe GPU is busy (another job is running). Two CUDA jobs on a throttled laptop GPU can trip the "
               "Windows 2-second GPU watchdog (driver reset, both jobs killed). Stop the other job or pass --force.")
    loaded = None
    if args.load:
        if busy and not args.force:
            print(warning)
            return
        loaded = load_test(args.load)
    print("\nDiagnosis:")
    for note in diagnose(loaded or snapshot, loaded is not None, power):
        print(f" - {note}")
    if args.train_bench:
        uses_gpu = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        if busy and uses_gpu and not args.force:  # a CPU benchmark cannot collide with a GPU job
            print(warning)
            return
        train_bench(args)


if __name__ == "__main__":
    main()
