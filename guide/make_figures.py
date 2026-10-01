"""Figures for the tinyGPT learning guide (Nature-style schematics)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _count_terms():
    """Keyword terms exactly as data_prep.py reads them."""
    try:
        import sys
        sys.path.insert(0, PROJECT)
        import data_prep
        return len(data_prep.read_terms(os.path.join(PROJECT, "keywords.txt")))
    except Exception:  # noqa: BLE001
        return 0


def _count_lines(name):
    """Items in a project list file, so the figure never shows stale counts."""
    try:
        with open(os.path.join(PROJECT, name), encoding="utf-8-sig") as handle:
            return sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))
    except OSError:
        return 0
INK, MUTED = "#1f1f1f", "#5b5b5b"
FILL = {"data": "#e8f0fa", "prep": "#fdf1e3", "train": "#e7f5ee", "post": "#f6ebf3", "use": "#f1f1ee"}
EDGE = {"data": "#2a78d6", "prep": "#d9822b", "train": "#1b9e6b", "post": "#a8528c", "use": "#7a7a74"}
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "text.color": INK, "mathtext.fontset": "dejavusans"})


def box(ax, x, y, w, h, text, kind, bold=False, size=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=0.9, edgecolor=EDGE[kind], facecolor=FILL[kind]))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, linewidth=0.9,
                                 color=color, shrinkA=0, shrinkB=0))


def line(ax, xs, ys):
    ax.plot(xs, ys, color=MUTED, linewidth=0.9, solid_capstyle="butt")


def panel(ax, x, y, letter, title):
    ax.text(x, y, letter, fontsize=10, fontweight="bold", va="bottom")
    ax.text(x + 0.022, y + 0.002, title, fontsize=8, fontweight="bold", va="bottom", color=INK)


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def pipeline():
    fig, ax = canvas(7.1, 5.4)
    # a: sources, joined by a bus into the first preparation step
    panel(ax, 0.01, 0.93, "a", "Sources")
    items = ["Your Markdown\nbooks · articles · scripts · notes",
             "PDF, DOCX, PPTX, XLSX, GWAS\nvia convert_to_markdown.py",
             "Wikipedia\n6.4 M English articles",
             "FineWeb-Edu · peS2o\neducational web · open papers"]
    centres = []
    for i, text in enumerate(items):
        y = 0.78 - i * 0.12
        box(ax, 0.01, y, 0.2, 0.1, text, "data", size=6.8)
        centres.append(y + 0.05)
        line(ax, [0.21, 0.235], [y + 0.05, y + 0.05])
    line(ax, [0.235, 0.235], [min(centres), 0.878])
    arrow(ax, 0.235, 0.878, 0.26, 0.878)
    # b: data preparation
    panel(ax, 0.26, 0.93, "b", "data_prep.py")
    steps = ["Clean: UTF-8, NFC, control chars", "English only", f"Keyword filter ({_count_terms()} terms)",
             "Exact + near-duplicates", "Boilerplate lines removed", "Split by document",
             "BPE tokenizer (8,192 pieces)", "Token shards + report"]
    for i, text in enumerate(steps):
        box(ax, 0.26, 0.85 - i * 0.068, 0.22, 0.056, text, "prep", size=6.8)
        if i:
            arrow(ax, 0.37, 0.85 - (i - 1) * 0.068, 0.37, 0.85 - i * 0.068 + 0.056)
    # c: pre-training loop
    panel(ax, 0.53, 0.93, "c", "tiny_gpt.py  pre-training")
    loop = [("Random windows\nfrom token shards", 0.81), ("Transformer\nforward pass", 0.695),
            ("Cross-entropy\n+ z-loss", 0.58), ("Back-propagation\n(gradients)", 0.465),
            ("AdamW update\nWSD learning rate", 0.35)]
    for text, y in loop:
        box(ax, 0.55, y, 0.17, 0.085, text, "train")
    for (_, y0), (_, y1) in zip(loop, loop[1:]):
        arrow(ax, 0.635, y0, 0.635, y1 + 0.085)
    line(ax, [0.48, 0.505, 0.505], [0.378, 0.378, 0.852])
    arrow(ax, 0.505, 0.852, 0.55, 0.852)
    line(ax, [0.72, 0.738, 0.738], [0.392, 0.392, 0.852])
    arrow(ax, 0.738, 0.852, 0.72, 0.852)
    ax.text(0.749, 0.47, "repeat every step", rotation=90, ha="center", va="center", fontsize=6.2, color=MUTED)
    box(ax, 0.765, 0.60, 0.225, 0.27,
        "Every 200 steps\n\nvalidation loss · perplexity\nbits per byte · per source\n"
        f"fact benchmark ({_count_lines('probes_biology.tsv')} items)\nsample text · checkpoint\nexperiments.csv", "use", size=6.8)
    arrow(ax, 0.738, 0.735, 0.765, 0.735)
    # d: post-training
    panel(ax, 0.53, 0.26, "d", "finetune.py  post-training")
    post = [("Supervised\nfine-tuning (SFT)", 0.55), ("Preference\ntuning (DPO)", 0.705),
            ("Reinforcement\nlearning (RL)*", 0.86)]
    for text, x in post:
        box(ax, x, 0.13, 0.135, 0.1, text, "post")
    arrow(ax, 0.685, 0.18, 0.705, 0.18)
    arrow(ax, 0.84, 0.18, 0.86, 0.18)
    line(ax, [0.6175, 0.6175, 0.515, 0.515], [0.35, 0.31, 0.31, 0.18])
    arrow(ax, 0.515, 0.18, 0.55, 0.18)
    ax.text(0.627, 0.31, "after pre-training", fontsize=6.2, color=MUTED, va="center")
    # e: evaluation and use
    panel(ax, 0.01, 0.26, "e", "Evaluate and use")
    uses = ["view_pt.py\ndashboard", "compare_models.py\npaired bootstrap", "--benchmark\nper category",
            "--export\nsafetensors", "--generate\n(+ watermark)", "detect_text.py\nauthorship tests"]
    for i, text in enumerate(uses):
        box(ax, 0.01 + (i % 3) * 0.165, 0.13 - (i // 3) * 0.115, 0.15, 0.095, text, "use", size=6.6)
    ax.text(0.99, 0.01, "* described in the guide; needs a stronger base model", ha="right", fontsize=6,
            color=MUTED)
    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), dpi=300)
    plt.close(fig)


def architecture():
    fig, ax = canvas(7.1, 4.6)
    panel(ax, 0.01, 0.94, "a", "One tinyGPT forward pass")
    col = 0.08
    for text, y, kind in (("token IDs  (512 per window)", 0.86, "data"), ("token embedding  (8,192 × d)", 0.77, "prep")):
        box(ax, col, y, 0.34, 0.06, text, kind)
    arrow(ax, col + 0.17, 0.86, col + 0.17, 0.83)
    ax.add_patch(FancyBboxPatch((col - 0.02, 0.27), 0.38, 0.47, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=0.9, edgecolor=EDGE["train"], facecolor="none", linestyle=(0, (3, 2))))
    ax.text(col + 0.36, 0.72, "× L blocks", ha="right", fontsize=7, color=EDGE["train"], fontweight="bold")
    for text, y, h in (("RMSNorm", 0.64, 0.055), ("grouped-query attention\nwith RoPE positions", 0.52, 0.075),
                       ("RMSNorm", 0.43, 0.055), ("SwiGLU feed-forward", 0.32, 0.055)):
        box(ax, col + 0.03, y, 0.26, h, text, "train")
    arrow(ax, col + 0.17, 0.77, col + 0.17, 0.695)
    arrow(ax, col + 0.16, 0.64, col + 0.16, 0.595)
    arrow(ax, col + 0.16, 0.52, col + 0.16, 0.485)
    arrow(ax, col + 0.16, 0.43, col + 0.16, 0.375)
    for y0, y1 in ((0.705, 0.505), (0.495, 0.30)):
        line(ax, [col + 0.30, col + 0.33, col + 0.33, col + 0.30], [y0, y0, y1, y1])
        ax.text(col + 0.345, (y0 + y1) / 2, "+ residual", rotation=90, fontsize=6, color=MUTED, va="center",
                ha="center")
    arrow(ax, col + 0.17, 0.27, col + 0.17, 0.25)
    for text, y, kind in (("final RMSNorm", 0.19, "train"), ("output layer = embedding (tied)", 0.10, "prep"),
                          ("softmax → p(next token)", 0.01, "use")):
        box(ax, col, y, 0.34, 0.06, text, kind)
    arrow(ax, col + 0.17, 0.19, col + 0.17, 0.16)
    arrow(ax, col + 0.17, 0.10, col + 0.17, 0.07)

    panel(ax, 0.50, 0.94, "b", "Causal attention in one head")
    lines = ["for position $t$: query $q_t$;  for $s \\leq t$: keys $k_s$, values $v_s$",
             "",
             "weights  $a_{ts} = \\mathrm{softmax}_s\\,(q_t \\cdot k_s / \\sqrt{d_h})$",
             "output   $o_t = \\sum_{s \\leq t} a_{ts}\\, v_s$",
             "",
             "RoPE rotates $q$ and $k$ by angles proportional to position,",
             "so $q_t \\cdot k_s$ depends only on the distance $t - s$.",
             "GQA: several query heads share one key/value head."]
    box(ax, 0.51, 0.55, 0.47, 0.35, "\n".join(lines), "use", size=7)

    panel(ax, 0.50, 0.44, "c", "Where the parameters are")
    text = ("attention (q, k, v, o):  $2d^2 + 2\\,d\\,d_{kv}$\n"
            "feed-forward (gate, up, down):  $3\\,d\\,h$,   $h \\approx 8d/3$\n"
            "norms:  $2d$ per block;   embedding:  $V d$ (shared with output)\n\n"
            "your run:  $d$ = 384,  $L$ = 8,  $V$ = 8,192\n"
            "blocks  8 × 1,770,240 + 384 = 14,162,304\n"
            "embedding  8,192 × 384 = 3,145,728\n"
            "total  17,308,032 parameters")
    box(ax, 0.51, 0.06, 0.47, 0.33, text, "prep", size=7)
    fig.savefig(os.path.join(OUT, "fig2_architecture.png"), dpi=300)
    plt.close(fig)


def stages():
    fig, ax = canvas(7.1, 2.5)
    rows = [("1  Pre-training", "raw text; predict the next token",
             "$10^8$ tokens (tinyGPT) to $10^{13}$ (frontier)", "train"),
            ("2  Supervised fine-tuning", "(question, ideal answer); loss on the answer only",
             "$10^3$ to $10^6$ pairs", "post"),
            ("3  Preference tuning", "(prompt, better, worse); DPO or RLHF", "$10^3$ to $10^6$ comparisons", "post"),
            ("4  Reinforcement learning", "checkable problems; reward correct reasoning",
             "$10^4$ to $10^6$ problems", "post")]
    for i, (name, what, size, kind) in enumerate(rows):
        y = 0.77 - i * 0.235
        box(ax, 0.01, y, 0.25, 0.18, name, kind, bold=True, size=8)
        box(ax, 0.28, y, 0.43, 0.18, what, "use", size=7.5)
        box(ax, 0.73, y, 0.26, 0.18, size, "use", size=7.5)
        if i:
            arrow(ax, 0.135, y + 0.235, 0.135, y + 0.18)
    fig.savefig(os.path.join(OUT, "fig3_stages.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pipeline()
    architecture()
    stages()
    print("figures written to", OUT)
