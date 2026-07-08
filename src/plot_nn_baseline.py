"""Plot the nearest-neighbour baseline accuracies from results/nn_baseline_accuracy.csv.

Figure 1: two rows (exp1, exp2), shared y-axis; paired top-1/top-3 boxplots over
          the 10 replicates at each consecutive transition, replicate points overlaid.
Figure 2: the 0->2 comparison (direct vs chained-via-exp1 vs chained-via-exp2), top-1.

Palette: Okabe-Ito (colour-blind safe).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Okabe-Ito
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#4d4d4d"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150,
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.8,
})

RNG = np.random.default_rng(0)  # jitter only; not part of the experiment


def box_group(ax, x, vals, color, width=0.30, label=None):
    """A single thin box with its 10 replicate points jittered on top."""
    bp = ax.boxplot(
        vals, positions=[x], widths=width, patch_artist=True,
        showcaps=False, showfliers=False, medianprops=dict(color=GREY, lw=1.6),
        whiskerprops=dict(color=color, lw=1.2), boxprops=dict(color=color, lw=1.2),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.16)
    jit = (RNG.random(len(vals)) - 0.5) * width * 0.7
    ax.scatter(np.full(len(vals), x) + jit, vals, s=14, color=color,
               alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3,
               label=label)


def figure1(df):
    exps = [("exp1", "Experiment 1  (Δt = 1)"),
            ("exp2", "Experiment 2  (Δt = 0.5)")]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharey=True)

    for ax, (exp, title) in zip(axes, exps):
        sub = df[(df.figure == 1) & (df.panel == exp)]
        # keep transitions in chronological order
        transitions = sorted(sub.transition.unique(),
                             key=lambda s: float(s.split("→")[0]))
        for xi, tr in enumerate(transitions):
            for off, (metric, color) in zip((-0.19, 0.19),
                                            [("top-1", BLUE), ("top-3", ORANGE)]):
                vals = sub[(sub.transition == tr) & (sub.metric == metric)]\
                    .sort_values("replicate").accuracy.to_numpy()
                box_group(ax, xi + off, vals, color,
                          label=metric if xi == 0 else None)
        ax.set_xticks(range(len(transitions)))
        ax.set_xticklabels(transitions)
        ax.set_ylim(0, 1.03)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color="#dddddd", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel("ancestor-recovery accuracy")
        ax.set_title(title, loc="left")
        ax.margins(x=0.08)
    axes[0].legend(title="metric", frameon=False, loc="lower left")
    axes[1].set_xlabel("snapshot transition")
    fig.suptitle("Nearest-neighbour ancestry recovery: consecutive transitions",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = RESULTS / "fig1_nn_consecutive.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure2(df):
    sub = df[df.figure == 2]
    methods = [("direct", "direct\n(1 jump)", BLUE),
               ("chained via exp1", "chained via exp1\n(2 hops)", ORANGE),
               ("chained via exp2", "chained via exp2\n(4 hops)", GREEN)]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for xi, (m, label, color) in enumerate(methods):
        vals = sub[sub.method == m].sort_values("replicate").accuracy.to_numpy()
        box_group(ax, xi, vals, color, width=0.34)
        ax.text(xi, vals.mean(), f"  mean {vals.mean():.2f}", va="center",
                ha="left", fontsize=9, color=GREY)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m[1] for m in methods])
    ax.set_ylim(0.45, 0.75)
    ax.grid(axis="y", color="#dddddd", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("0→2 ancestor-recovery accuracy (top-1)")
    ax.set_title("Estimating the 0→2 ancestry: one jump vs. chaining short steps",
                 loc="left", fontsize=12, fontweight="bold")
    ax.margins(x=0.12)
    fig.tight_layout()
    out = RESULTS / "fig2_nn_0to2.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    df = pd.read_csv(RESULTS / "nn_baseline_accuracy.csv")
    print("wrote", figure1(df))
    print("wrote", figure2(df))


if __name__ == "__main__":
    main()
