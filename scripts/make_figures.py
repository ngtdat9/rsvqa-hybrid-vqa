"""Charts for the README and the progress report.

    python scripts/make_figures.py

Writes PNGs (white background, 2x) to reports/figures/readme/ for the README, and SVGs to
docs/report/charts/ for the PDF. The numbers are copied from the reports in reports/*.md; the training
curves are read from the full run's log, reports/training_log_hr_full_20261031-090319.jsonl.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PNG_DIR = ROOT / "reports" / "figures" / "readme"
SVG_DIR = ROOT / "docs" / "report" / "charts"
LOG = ROOT / "reports" / "training_log_hr_full_20261031-090319.jsonl"

# Okabe-Ito (colour-blind safe), fixed order: ours, CLIP, trivial / random
BLUE, ORANGE, GREY, VERMILLION, GREEN, SKY = "#0072B2", "#E69F00", "#9A9A9A", "#D55E00", "#009E73", "#56B4E9"
INK, MUTED, GRID = "#1f2328", "#59636e", "#e5e7eb"

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "svg.fonttype": "none", "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left",
})
TYPES = ["Presence", "Comparison", "Area", "Count"]
PNG_DIR.mkdir(parents=True, exist_ok=True)
SVG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(SVG_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PNG_DIR / f"{name}.png", bbox_inches="tight", dpi=200, facecolor="white", pad_inches=0.12)
    plt.close(fig)


def bars(ax, groups, series, colors, labels, fmt="{:.0f}", ylim=None, ylabel=None):
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (values, color, label) in enumerate(zip(series, colors, labels)):
        pos = x - 0.4 + width * (i + 0.5)
        rects = ax.bar(pos, values, width - 0.04, color=color, label=label, zorder=2)
        for r, v in zip(rects, values):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(v), ha="center", va="bottom",
                    fontsize=7 if n > 2 or len(groups) > 3 else 7.5, color=INK)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel)


# 1. full run training curves (three panels, one axis each)
rows = [json.loads(line) for line in LOG.read_text().splitlines()]
epochs = [r["epoch"] + 1 for r in rows]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
ax = axes[0]
ax.plot(epochs, [r["train_loss"] for r in rows], color=ORANGE, lw=2, marker="o", ms=3.5, label="train")
ax.plot(epochs, [r["val_loss"] for r in rows], color=BLUE, lw=2, marker="o", ms=3.5, label="validation")
ax.set_title("Loss")
ax.legend(fontsize=8)
ax = axes[1]
ax.plot(epochs, [100 * r["val_class_acc"] for r in rows], color=BLUE, lw=2, marker="o", ms=3.5)
ax.set_title("Validation accuracy (%)")
ax.annotate(f"{100 * rows[-1]['val_class_acc']:.1f}%", (epochs[-1], 100 * rows[-1]["val_class_acc"]),
            xytext=(-4, -14), textcoords="offset points", ha="right", fontsize=8, color=INK)
ax = axes[2]
ax.plot(epochs, [r["val_count_mae"] for r in rows], color=BLUE, lw=2, marker="o", ms=3.5, label="model")
ax.axhline(rows[0]["val_count_mae_baseline"], color=GREY, lw=1.5, ls="--", label="always the median")
ax.set_ylim(0, 2.5)
ax.set_title("Validation count MAE")
ax.legend(fontsize=8, loc="center right")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.set_xticks(range(1, 11))
fig.tight_layout()
save(fig, "training_curves")

# 2. test vs test_phili, per type
fig, ax = plt.subplots(figsize=(5.4, 2.4))
bars(ax, ["Overall", "Presence", "Comparison", "Area"], [[90.2, 91.7, 90.9, 86.1], [76.9, 80.2, 79.6, 64.8]],
     [BLUE, SKY], ["test (same region as training)", "test_phili (Philadelphia, unseen city)"],
     fmt="{:.1f}", ylim=(0, 105), ylabel="accuracy (%)")
ax.legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))
save(fig, "test_vs_phili")

# 3. ours vs CLIP vs trivial
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
data = {
    "test": ([91.7, 90.9, 86.1], [53.2, 49.6, 42.7], [60.8, 66.6, 62.4]),
    "test_phili": ([80.2, 79.6, 64.8], [55.8, 57.4, 48.0], [57.3, 66.6, 57.4]),
}
for ax, (name, series) in zip(axes, data.items()):
    bars(ax, TYPES[:3], series, [BLUE, ORANGE, GREY], ["our model", "zero-shot CLIP", "most common answer"],
         ylim=(0, 105))
    ax.set_title(name)
axes[0].set_ylabel("accuracy (%)")
axes[0].legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(1.05, 1.28))
fig.tight_layout()
save(fig, "clip_comparison")

# 4. occlusion vs random deletion effect
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
occ = {"test": ([0.143, 0.191, 0.159, 0.312], [0.028, 0.038, 0.034, 0.175]),
       "test_phili": ([0.139, 0.142, 0.141, 0.135], [0.017, 0.004, 0.028, 0.081])}
for ax, (name, (o, r)) in zip(axes, occ.items()):
    bars(ax, ["Presence", "Compar.", "Area", "Count"], [o, r], [BLUE, GREY],
         ["blank the occlusion map's top cells", "blank random cells"], fmt="{:.2f}", ylim=(0, 0.36))
    ax.set_title(name)
axes[0].set_ylabel("change in the answer")
axes[0].legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(1.05, 1.28))
fig.tight_layout()
save(fig, "occlusion_deletion")

# 5. retrieval AUROC: neighbours vs random (dot plot with connector)
fig, ax = plt.subplots(figsize=(5.0, 1.9))
ret = [0.747, 0.665, 0.790, 0.846]
rnd = [0.491, 0.530, 0.632, 0.778]
y = np.arange(4)[::-1]
for yi, a, b in zip(y, ret, rnd):
    ax.plot([b, a], [yi, yi], color=GRID, lw=3, zorder=1)
ax.scatter(rnd, y, s=55, color=GREY, zorder=2, label="5 random training questions")
ax.scatter(ret, y, s=55, color=BLUE, zorder=3, label="5 nearest neighbours (retrieval)")
for yi, a in zip(y, ret):
    ax.text(a + 0.012, yi, f"{a:.3f}", va="center", fontsize=8, color=INK)
ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
ax.set_yticks(y, TYPES)
ax.set_xlim(0.4, 0.92)
ax.set_xlabel("AUROC for flagging the model's wrong answers")
ax.grid(axis="y", visible=False)
ax.legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.25))
save(fig, "retrieval_auroc")

# 6. warning: error rate when warned vs not
fig, ax = plt.subplots(figsize=(5.0, 2.1))
bars(ax, [f"{t}\n({w}% warned)" for t, w in zip(TYPES, [5.0, 4.5, 11.5, 1.1])],
     [[43.3, 33.6, 60.9, 85.7], [11.3, 11.4, 15.1, 35.2]], [VERMILLION, GREY],
     ["warned", "not warned"], fmt="{:.0f}%", ylim=(0, 100), ylabel="model wrong (%)")
ax.tick_params(axis="x", labelsize=7.5)
ax.legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))
save(fig, "warning_rates")

# 7. tuning: gradient norms small vs wide model (log scale is one axis)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
bars(ax, ["small model\n(0.9M params)", "wide model\n(3.3M params)"], [[4, 88], [42, 12000]], [BLUE, VERMILLION],
     ["mean", "max"], fmt="{:,.0f}")
ax.set_yscale("log")
ax.set_ylim(1, 60000)
ax.set_ylabel("gradient norm (log scale)")
ax.legend(fontsize=8, loc="upper left")
save(fig, "grad_norms")
print("charts:", sorted(p.stem for p in SVG_DIR.glob("*.svg")))
