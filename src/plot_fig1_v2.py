"""Fig1 v2: NN baselines (top-1, top-3) plus the four OT methods (top-1),
per consecutive transition, two rows (exp1, exp2), over the 10 replicates.

Reads results/nn_baseline_accuracy.csv and results/ot_accuracy.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# NN baselines in greys; OT methods in Okabe-Ito hues (all CVD-safe).
METHOD_ORDER = [
    ("NN top-1", "#444444"),
    ("NN top-3", "#999999"),
    ("entropic-kl", "#0072B2"),
    ("mm-kl", "#E69F00"),
    ("mm-l2", "#009E73"),
    ("partial", "#D55E00"),
]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150,
    "font.size": 10, "axes.titlesize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.8,
})
RNG = np.random.default_rng(0)


def combined():
    nn = pd.read_csv(RESULTS / "nn_baseline_accuracy.csv")
    nn = nn[nn.figure == 1].copy()
    nn["display"] = "NN " + nn["metric"]
    ot = pd.read_csv(RESULTS / "ot_accuracy.csv").copy()
    ot["display"] = ot["method"]
    cols = ["panel", "transition", "display", "replicate", "accuracy"]
    return pd.concat([nn[cols], ot[cols]], ignore_index=True)


def box(ax, x, vals, color, width, label=None):
    bp = ax.boxplot(vals, positions=[x], widths=width, patch_artist=True,
                    showcaps=False, showfliers=False,
                    medianprops=dict(color="#333333", lw=1.3),
                    whiskerprops=dict(color=color, lw=1.0),
                    boxprops=dict(color=color, lw=1.0))
    for p in bp["boxes"]:
        p.set_facecolor(color); p.set_alpha(0.16)
    jit = (RNG.random(len(vals)) - 0.5) * width * 0.6
    ax.scatter(np.full(len(vals), x) + jit, vals, s=9, color=color,
               alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3, label=label)


def main():
    df = combined()
    offs = np.linspace(-0.36, 0.36, len(METHOD_ORDER))
    width = (offs[1] - offs[0]) * 0.8

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.4), sharey=True)
    for ax, (exp, title) in zip(axes, [("exp1", "Experiment 1  (Δt = 1)"),
                                       ("exp2", "Experiment 2  (Δt = 0.5)")]):
        sub = df[df.panel == exp]
        transitions = sorted(sub.transition.unique(),
                             key=lambda s: float(s.split("→")[0]))
        for xi, tr in enumerate(transitions):
            for off, (disp, color) in zip(offs, METHOD_ORDER):
                vals = sub[(sub.transition == tr) & (sub.display == disp)] \
                    .sort_values("replicate").accuracy.to_numpy()
                if len(vals):
                    box(ax, xi + off, vals, color, width,
                        label=disp if xi == 0 else None)
        ax.set_xticks(range(len(transitions)))
        ax.set_xticklabels(transitions)
        ax.set_ylim(0, 1.03)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color="#dddddd", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylabel("ancestor-recovery accuracy")
        ax.set_title(title, loc="left")
        ax.margins(x=0.03)
    axes[0].legend(ncol=6, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, 1.06), columnspacing=1.2, handletextpad=0.3)
    axes[1].set_xlabel("snapshot transition")
    fig.suptitle("Ancestry recovery: NN baselines vs. OT methods (top-1)",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RESULTS / "fig1_v2_nn_vs_ot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
