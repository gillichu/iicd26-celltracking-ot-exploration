"""Visualise a single replicate.

1. Static figure: t=0 and t=1 points (Experiment 1) with the ground-truth ancestry
   drawn as light links from each t=1 cell to its true t=0 ancestor.
2. Animated GIF: the *true* continuous process over [0, 2]. The replicate's master
   trajectory is regenerated from the seed (deterministic), and every cell is drawn
   at its real Brownian position on the fine grid — cells appear at birth, diffuse,
   and are replaced by two daughters at division. No interpolation: the snapshots
   on disk are exact subsamples of this same trajectory.

Usage: python src/plot_replicate.py [replicate]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from simulate import simulate_master

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#888888"

plt.rcParams.update({
    "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.8,
})


def snap(exp, rep, k):
    return pd.read_csv(DATA / exp / f"rep{rep:02d}" / "snapshots" / f"t{k}.csv")[["x", "y"]].to_numpy()


def relmat(exp, rep, i, j):
    return np.load(DATA / exp / f"rep{rep:02d}" / "relations" / f"{i}-{j}.npy")


def regenerate_trajectory(rep):
    """Reproduce the replicate's master process exactly from the seed."""
    union = sorted({float(t) for e in PARAMS["experiments"].values()
                    for t in e["snapshot_times"]})
    seeds = np.random.SeedSequence(PARAMS["seed"]).spawn(PARAMS["n_replicates"])
    rng = np.random.default_rng(seeds[rep])
    cells, _, _, traj = simulate_master(
        rng, n_founders=PARAMS["n_founders"], lam=PARAMS["lambda_div"],
        sigma=PARAMS["sigma"], box=PARAMS["founder_box"], T=PARAMS["horizon_T"],
        snapshot_times=union, sim_dt=PARAMS["sim_grid_dt"], return_trajectory=True)
    return cells, traj


def static_t0_t1(rep):
    P0, P1 = snap("exp1", rep, 0), snap("exp1", rep, 1)
    anc = relmat("exp1", rep, 0, 1).argmax(1)

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    for d, a in enumerate(anc):
        ax.plot([P0[a, 0], P1[d, 0]], [P0[a, 1], P1[d, 1]],
                color=GREY, lw=0.5, alpha=0.35, zorder=1)
    ax.scatter(P0[:, 0], P0[:, 1], s=42, color=BLUE, edgecolor="white",
               linewidth=0.5, zorder=3, label=f"t = 0  ({len(P0)} cells)")
    ax.scatter(P1[:, 0], P1[:, 1], s=26, color=ORANGE, edgecolor="white",
               linewidth=0.5, zorder=2, label=f"t = 1  ({len(P1)} cells)")
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"Replicate {rep}: t=0 → t=1 with ground-truth ancestry",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = RESULTS / f"rep{rep:02d}_t0_t1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def animate(rep, frame_stride=2, fps=20, flash_frames=3):
    """GIF of the true process; a frame every ``frame_stride`` fine-grid steps.

    Each division is highlighted: the two newborn daughters flash orange (and
    enlarge) for ``flash_frames`` frames right after they appear.
    """
    cells, traj = regenerate_trajectory(rep)
    dt = PARAMS["sim_grid_dt"]
    T = PARAMS["horizon_T"]
    grid = np.round(np.arange(0.0, T + dt / 2, dt), 10)
    anim_times = grid[::frame_stride]

    # For each cell, its exact grid position at each time it exists.
    posdict = {cid: {round(float(t), 10): p for t, p in zip(ts, ps)}
               for cid, (ts, ps) in traj.items()}
    # Point cloud at each animation time, keeping cell ids to align colours.
    clouds, cloud_ids = [], []
    for t in anim_times:
        key = round(float(t), 10)
        ids = [cid for cid, d in posdict.items() if key in d]
        clouds.append(np.array([posdict[cid][key] for cid in ids]))
        cloud_ids.append(ids)

    # A division = a daughter's first appearance; flash it for a few frames.
    first_frame = {}
    for cid in posdict:
        present = [i for i, t in enumerate(anim_times)
                   if round(float(t), 10) in posdict[cid]]
        if present:
            first_frame[cid] = present[0]
    flash = [set() for _ in anim_times]
    for cid, c in cells.items():
        if c.parent_id != -1 and cid in first_frame:   # daughters only, not founders
            f0 = first_frame[cid]
            for j in range(f0, min(f0 + flash_frames, len(anim_times))):
                flash[j].add(cid)

    allpts = np.vstack([c for c in clouds if len(c)])
    pad = 1.5
    xlim = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
    ylim = (allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)

    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    scat = ax.scatter([], [], s=22, edgecolor="white", linewidth=0.35)
    title = ax.set_title("", loc="left", fontsize=12, fontweight="bold")

    def update(i):
        scat.set_offsets(clouds[i])
        fl = flash[i]
        scat.set_facecolors([ORANGE if c in fl else BLUE for c in cloud_ids[i]])
        scat.set_sizes([48 if c in fl else 22 for c in cloud_ids[i]])
        n_div = len(fl)
        title.set_text(f"Replicate {rep}   t = {anim_times[i]:.2f}   "
                       f"({len(clouds[i])} cells,  {n_div} dividing)")
        return scat, title

    anim = FuncAnimation(fig, update, frames=len(anim_times), blit=False)
    out = RESULTS / f"rep{rep:02d}_animation.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return out


def main():
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    RESULTS.mkdir(exist_ok=True)
    print("wrote", static_t0_t1(rep))
    print("wrote", animate(rep))


if __name__ == "__main__":
    main()
