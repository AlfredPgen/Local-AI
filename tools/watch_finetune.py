"""Live plot of a running (or finished) finetune.py run, read from its log file:
training loss, held-out loss, held-out accuracy (when the run measures it) and
the learning rate, redrawn every --every seconds. Reads only the log, so it
never touches the GPU or the run.

    python tools/watch_finetune.py                 # log.txt, the newest fine-tune in it
    python tools/watch_finetune.py --log-file X --png out.png --once
"""
import argparse
import os
import re
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e8e7e2", "#fbfaf7"

HEADER = re.compile(r"^--- finetune (\S+) \| (\w+) .*--name (\S+)")
PLAN = re.compile(r"(\d+) epochs x (\d+) steps")
PROGRESS = re.compile(r"step ([\d,]+)/[\d,]+ \| epoch \d+ \| train loss ([\d.]+) .*?\| lr ([\d.e+-]+)")
EVAL = re.compile(r"^(?:step ([\d,]+)|epoch (\d+))(?: \| train loss [\d.]+)? \| held-out loss ([\d.]+)(.*)")
EXTRA = re.compile(r"\| (multiple choice|facts|reward_accuracy) ([\d.]+)")  # reward_accuracy: DPO pairs won


def parse(path):
    """The newest fine-tune in the log: {'title', 'train': [(step, loss)], 'val': [(step, loss)],
    'lr': [(step, lr)], 'acc': {name: [(step, value)]}, 'total': steps}."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    start = max((i for i, ln in enumerate(lines) if HEADER.match(ln)), default=None)
    if start is None:
        return None
    when, mode, name = HEADER.match(lines[start]).groups()
    run = {"title": f"{mode.upper()} {name} (started {when[:16].replace('T', ' ')} UTC)",
           "train": [], "val": [], "lr": [], "acc": {}, "total": None, "per_epoch": None}
    for ln in lines[start + 1:]:
        ln = ln.strip()
        if (m := PLAN.search(ln)) and run["per_epoch"] is None:
            run["per_epoch"] = int(m.group(2))
            run["total"] = int(m.group(1)) * run["per_epoch"]
        elif m := PROGRESS.search(ln):
            step = int(m.group(1).replace(",", ""))
            run["train"].append((step, float(m.group(2))))
            run["lr"].append((step, float(m.group(3))))
        elif m := EVAL.match(ln):
            step = int(m.group(1).replace(",", "")) if m.group(1) else int(m.group(2)) * (run["per_epoch"] or 0)
            run["val"].append((step, float(m.group(3))))
            for key, value in EXTRA.findall(m.group(4)):
                run["acc"].setdefault(key, []).append((step, float(value)))
    for key in ("val",):
        run[key] = sorted(dict(run[key]).items())  # an epoch end and a mid-epoch reading at the same step: one point
    return run


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def draw(fig, run):
    import matplotlib.ticker as mt
    fig.clear()
    fig.patch.set_facecolor("white")
    axes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [2.2, 1.6, 1]})
    for ax in axes:
        style(ax)
        ax.set_xlabel("optimizer step", color=INK2, fontsize=8)
        if run["total"]:
            ax.set_xlim(0, run["total"])
        ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    loss, acc, lr = axes
    if run["train"]:
        loss.plot(*zip(*run["train"]), color=SERIES[0], lw=1, alpha=0.55, label="training (last 50 steps)")
    if run["val"]:
        loss.plot(*zip(*run["val"]), color=SERIES[1], lw=2, marker="o", ms=3.5, label="held-out")
        step, best = min(run["val"], key=lambda p: p[1])
        loss.annotate(f"{run['val'][-1][1]:.4f}", run["val"][-1], textcoords="offset points", xytext=(4, 6),
                      fontsize=8, color=SERIES[1])
        what = ("DPO loss (0.693 = no preference learned)" if run["title"].startswith("DPO")
                else "Loss per answer token")
        loss.set_title(f"{what}\nlowest held-out {best:.4f} at step {step:,}", loc="left", fontsize=9.5, color=INK)
    loss.set_ylim(bottom=0)
    loss.legend(frameon=False, fontsize=8, loc="lower left")
    if run["acc"]:
        for i, (key, pts) in enumerate(sorted(run["acc"].items())):
            label = "pairs where the chosen answer wins" if key == "reward_accuracy" else key
            acc.plot(*zip(*pts), color=SERIES[2 + i], lw=2, marker="o", ms=3.5, label=label)
            acc.annotate(f"{pts[-1][1]:.3f}", pts[-1], textcoords="offset points", xytext=(4, 6), fontsize=8,
                         color=SERIES[2 + i])
        acc.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=0))
        acc.legend(frameon=False, fontsize=8)
        acc.set_title("Held-out accuracy", loc="left", fontsize=9.5, color=INK)
    else:
        acc.set_title("Held-out accuracy", loc="left", fontsize=9.5, color=INK)
        acc.text(0.5, 0.5, "not measured during this run\n(posttrain's eval scores it at the end)",
                 transform=acc.transAxes, ha="center", va="center", fontsize=8.5, color=MUTED)
    if run["lr"]:
        lr.plot(*zip(*run["lr"]), color=SERIES[0], lw=1.5)
        lr.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.1e}"))
    lr.set_title("Learning rate", loc="left", fontsize=9.5, color=INK)
    done = run["train"][-1][0] if run["train"] else 0
    fig.suptitle(f"{run['title']}  |  step {done:,} of {run['total'] or '?':,}", x=0.01, ha="left", fontsize=11,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-file", default=os.path.join(HERE, "log.txt"))
    p.add_argument("--png", default=os.path.join(HERE, "logs", "finetune_live.png"))
    p.add_argument("--every", type=float, default=20, help="seconds between redraws")
    p.add_argument("--once", action="store_true", help="write the PNG once and exit (no window)")
    args = p.parse_args()
    import matplotlib
    if args.once:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 4.6))
    while True:
        run = parse(args.log_file)
        if run:
            draw(fig, run)
            os.makedirs(os.path.dirname(os.path.abspath(args.png)), exist_ok=True)
            fig.savefig(args.png, dpi=110)
        if args.once:
            return
        plt.pause(args.every)
        if not plt.fignum_exists(fig.number):  # window closed
            return


if __name__ == "__main__":
    main()
