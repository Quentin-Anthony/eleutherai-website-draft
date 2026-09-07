"""Homepage focus-area figures for eleuther.ai, drawn with matplotlib.

Paste into a notebook or run as a script. Each function returns (fig, ax) so
you can tweak placement, then call save(fig, "name") to write the SVG that
data/home.yaml points at. Colors are the site's CSS tokens.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

# --- site palette ----------------------------------------------------------
BG = "#10151d"        # --panel
TEXT = "#ffffff"      # maximum contrast on the panel
MUTED = "#c4c9d1"     # --body, for secondary labels
LINE = "#3a4656"      # axis lines, a step lighter than --line for legibility
GREEN = "#62d77a"
BLUE = "#78b8ee"
GOLD = "#d8b44a"
VIOLET = "#ad8df2"
CORAL = "#ee7b79"

FIGSIZE = (10, 7)     # 10:7 keeps the four homepage panels the same shape
FONT = {"family": ["Inter", "DejaVu Sans", "sans-serif"]}
plt.rcParams.update({
    "font.family": FONT["family"],
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": LINE,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "font.size": 18,
    "svg.fonttype": "none",   # keep text as text in the SVG so it stays crisp and editable
})


def blank_axes(ax, spines=("left", "bottom")):
    """No grid, no ticks; keep only the named spines."""
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in spines)
        ax.spines[side].set_linewidth(2)


def save(fig, name, outdir="static/images/research/homepage"):
    path = f"{outdir}/{name}.svg"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    # matplotlib leaves trailing spaces in path data, which fails `git diff --check`
    with open(path) as handle:
        lines = [line.rstrip() + "\n" for line in handle]
    with open(path, "w") as handle:
        handle.writelines(lines)


def neural_network(ax, x0, y0, width, height, layers=(4, 6, 6, 3), color=VIOLET,
                   node_ms=13, edge_alpha=0.35, edge_lw=1.2, zorder=6):
    """Draw a fully connected feed-forward network inside the box (x0, y0, width, height).

    Nodes are plot markers (sized in points), so the drawing looks the same
    whatever the axes' aspect ratio."""
    xs = np.linspace(x0, x0 + width, len(layers))
    coords = []
    for x, n in zip(xs, layers):
        ys = np.linspace(y0 + height, y0, n + 2)[1:-1] if n > 1 else [y0 + height / 2]
        coords.append([(x, y) for y in ys])
    for a, b in zip(coords[:-1], coords[1:]):
        for (xa, ya) in a:
            for (xb, yb) in b:
                ax.plot([xa, xb], [ya, yb], color=color, alpha=edge_alpha, lw=edge_lw, zorder=zorder)
    for layer in coords:
        xs_, ys_ = zip(*layer)
        ax.plot(xs_, ys_, "o", ms=node_ms, mfc=color, mec=BG, mew=1.5, zorder=zorder + 1)
    return coords


def document(ax, x, y, w=0.9, h=1.1, color=VIOLET, lines=4):
    """A training document: rounded page with text lines inside."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=BG, edgecolor=color, lw=2, zorder=2))
    for i in range(lines):
        ly = y + h - 0.22 - i * (h - 0.35) / (lines - 1)
        lw_frac = 0.7 if i < lines - 1 else 0.45
        ax.plot([x + 0.15, x + 0.15 + (w - 0.3) * lw_frac], [ly, ly], color=color, lw=2.5,
                solid_capstyle="round", zorder=3)


# ---------------------------------------------------------------------------
# 1. Evaluation: benchmark saturation
# ---------------------------------------------------------------------------
def evaluation_figure():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    blank_axes(ax)
    x = np.linspace(0, 1, 400)
    logistic = lambda center, k, top=0.97: top / (1 + np.exp(-k * (x - center)))

    # Older benchmarks: saturate early, pinned against the ceiling on the right.
    for center, k, alpha in [(0.14, 22, 1.0), (0.24, 18, 0.75), (0.34, 15, 0.5)]:
        ax.plot(x, logistic(center, k), color=GOLD, lw=3.5, alpha=alpha, solid_capstyle="round")
        ax.plot(1, logistic(center, k)[-1], "o", ms=10, mfc=BG, mec=GOLD, mew=2.5)
    # Newer benchmarks: still climbing.
    for center, k, alpha in [(0.85, 8, 1.0), (1.05, 8, 0.65)]:
        ax.plot(x, logistic(center, k), color=BLUE, lw=3.5, alpha=alpha, solid_capstyle="round")
        ax.plot(1, logistic(center, k)[-1], "o", ms=10, mfc=BG, mec=BLUE, mew=2.5)

    ax.axhline(0.97, color=GOLD, lw=2, ls=(0, (6, 6)))
    ax.text(0.995, 0.995, "Ceiling", color=GOLD, fontsize=18, ha="right", va="bottom")

    # Annotations sit in the empty regions; move freely.
    ax.text(0.50, 0.76, "Saturated: No Longer\nSeparates Models", color=TEXT, fontsize=19,
            ha="left", va="top", linespacing=1.3)
    ax.text(0.40, 0.30, "Still Informative", color=BLUE, fontsize=19, ha="left", va="bottom")

    # Legend as plain text in the upper-left gap.
    ax.plot([0.04, 0.10], [0.62, 0.62], color=GOLD, lw=4)
    ax.text(0.12, 0.62, "Older Benchmarks", color=TEXT, fontsize=18, va="center")
    ax.plot([0.04, 0.10], [0.54, 0.54], color=BLUE, lw=4)
    ax.text(0.12, 0.54, "Newer Benchmarks", color=TEXT, fontsize=18, va="center")

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Time Since Release  →", fontsize=19, loc="right", labelpad=12)
    ax.set_ylabel("Score", fontsize=19, loc="top", rotation=0, labelpad=-30)
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Open-weight safety: filtering pretraining data
# ---------------------------------------------------------------------------
def open_weight_safety_figure(show_attack=True):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    blank_axes(ax, spines=())
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.8)
    ax.set_aspect("equal")

    # Pretraining documents flowing right; coral ones are hazardous.
    docs_y = np.linspace(7.4, 1.4, 6)
    hazardous = {1, 4}
    ax.text(1.2, 9.0, "Pretraining Data", color=TEXT, fontsize=19, fontweight="bold", ha="center")
    for i, y in enumerate(docs_y):
        color = CORAL if i in hazardous else VIOLET
        document(ax, 0.75, y - 0.55, color=color)
        # kept docs continue through the filter to the model; hazardous ones stop at the filter
        end = 3.0 if i in hazardous else 6.1
        ax.plot([1.85, end], [y, y], color=color, lw=2.5, alpha=0.9, zorder=1)

    # Filter bar.
    ax.add_patch(FancyBboxPatch((3.0, 0.7), 0.35, 7.6, boxstyle="round,pad=0.02,rounding_size=0.17",
                                facecolor=BG, edgecolor=VIOLET, lw=2.5, zorder=4))
    for y in np.linspace(1.2, 7.8, 12):
        ax.plot([3.08, 3.27], [y, y], color=VIOLET, lw=2, zorder=5)
    ax.text(3.17, 0.15, "Filter", color=TEXT, fontsize=19, fontweight="bold", ha="center")

    # Hazardous docs are diverted downward.
    for i in hazardous:
        y = docs_y[i]
        ax.add_patch(FancyArrowPatch((3.35, y), (4.6, 0.55), connectionstyle="arc3,rad=-0.35",
                                     arrowstyle="-|>", mutation_scale=22, color=CORAL, lw=2.5, zorder=3))
    ax.text(4.75, 0.35, "Removed", color=CORAL, fontsize=18, ha="left", va="center")

    # Model: a real network.
    ax.add_patch(FancyBboxPatch((6.1, 0.9), 3.9, 7.2, boxstyle="round,pad=0.02,rounding_size=0.25",
                                facecolor="#0b1018", edgecolor=VIOLET, lw=2.5, zorder=2))
    neural_network(ax, 6.6, 1.3, 2.9, 5.4, layers=(5, 7, 7, 4), color=VIOLET, node_ms=12)
    ax.text(8.05, 8.55, "Open-Weight Model", color=TEXT, fontsize=19, fontweight="bold", ha="center", zorder=8)

    # Optional: adversarial fine-tuning hitting a model that never learned the content.
    if show_attack:
        for y in (6.2, 4.5, 2.8):
            ax.add_patch(FancyArrowPatch((13.2, y), (10.45, y), arrowstyle="-|>", mutation_scale=22,
                                         color=CORAL, lw=2.5, zorder=3))
        ax.plot([10.25, 10.25], [1.4, 7.6], color=VIOLET, lw=4, solid_capstyle="round", zorder=4)
        ax.text(12.3, 7.5, "Adversarial\nFine-Tuning", color=TEXT, fontsize=18, fontweight="bold",
                ha="center", va="center", linespacing=1.2)
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Interpretability over time: checkpoints along the loss curve
# ---------------------------------------------------------------------------
def interpretability_figure():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    blank_axes(ax)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.12)

    x = np.linspace(0, 1, 400)
    loss = 0.12 + 0.88 * np.exp(-4.2 * x)
    ax.plot(x, loss, color=GREEN, lw=3.5, solid_capstyle="round", zorder=2)

    # Saved checkpoints, each with a vertical line to the axis.
    ckpt_x = np.array([0.0, 0.08, 0.17, 0.28, 0.42, 0.56, 0.7, 0.85, 1.0])
    ckpt_y = 0.12 + 0.88 * np.exp(-4.2 * ckpt_x)
    for cx, cy in zip(ckpt_x, ckpt_y):
        ax.plot([cx, cx], [0, cy], color=GREEN, lw=1.5, ls=(0, (3, 4)), alpha=0.8, zorder=1)
    ax.plot(ckpt_x, ckpt_y, "o", ms=11, mfc=BG, mec=GREEN, mew=2.5, zorder=3)

    ax.set_xlabel("Training Steps  →", fontsize=19, loc="right", labelpad=12)
    ax.set_ylabel("Loss", fontsize=19, loc="top", rotation=0, labelpad=-30)
    ax.text(0.5, -0.09, "Saved Checkpoints", color=TEXT, fontsize=18, ha="center", va="top")
    ax.plot([0.0, 1.0], [-0.045, -0.045], color=GREEN, lw=1.5, alpha=0.8)  # bracket line under the ticks
    ax.plot([0.0, 0.0], [-0.045, -0.025], color=GREEN, lw=1.5, alpha=0.8)
    ax.plot([1.0, 1.0], [-0.045, -0.025], color=GREEN, lw=1.5, alpha=0.8)

    # Lens over one checkpoint: the network inside is what we study.
    # The axes are not square, so an Ellipse in data units is what renders as a circle.
    aspect = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]) * FIGSIZE[0] / FIGSIZE[1]
    lens_ckpt = 5
    lx, ly, r = 0.76, 0.52, 0.16
    ax.plot([ckpt_x[lens_ckpt], lx], [ckpt_y[lens_ckpt] + 0.02, ly - r * aspect], color=GREEN, lw=1.8,
            ls=(0, (3, 4)), zorder=2)
    ax.add_patch(Ellipse((lx, ly), 2 * r, 2 * r * aspect, facecolor="#0b1018", edgecolor=GREEN, lw=2.5, zorder=4))
    neural_network(ax, lx - 0.12, ly - 0.10 * aspect, 0.24, 0.20 * aspect, layers=(3, 5, 5, 2), color=GREEN,
                   node_ms=9, edge_lw=1.0)

    # What we do at every checkpoint.
    todo = ["Probe Representations", "Trace Behavior to Training Data", "Compare Across Checkpoints"]
    ax.text(0.30, 1.08, "At Every Checkpoint", color=GREEN, fontsize=18, fontweight="bold", ha="left", va="center")
    for i, label in enumerate(todo):
        ax.text(0.30, 1.005 - i * 0.075, "•  " + label, color=TEXT, fontsize=18, ha="left", va="center")
    return fig, ax


if __name__ == "__main__":
    fig, _ = evaluation_figure(); save(fig, "evaluation")
    fig, _ = open_weight_safety_figure(); save(fig, "open-weight-safety")
    fig, _ = interpretability_figure(); save(fig, "interpretability-over-time")
    print("wrote 3 SVGs")
