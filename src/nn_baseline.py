"""Nearest-neighbour baseline for ancestry recovery.

For each cell at a later snapshot, predict its ancestor as the nearest cell at an
earlier snapshot (Euclidean). Score against the ground-truth ancestry matrices.

Produces a tidy long-form CSV of per-replicate accuracies:
    results/nn_baseline_accuracy.csv
with columns: figure, panel, transition, method, metric, replicate, accuracy

- figure 1: consecutive-transition accuracy, top-1 and top-3, per experiment.
- figure 2: the 0->2 comparison — direct (one jump) vs chained via exp1 (through
  t=1) vs chained via exp2 (through all five timepoints); top-1 only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
N_REP = 10

EXP_TIMES = {
    "exp1": [0.0, 1.0, 2.0],
    "exp2": [0.0, 0.5, 1.0, 1.5, 2.0],
}


def snap(exp, rep, k):
    df = pd.read_csv(DATA / exp / f"rep{rep:02d}" / "snapshots" / f"t{k}.csv")
    return df[["x", "y"]].to_numpy()


def relmat(exp, rep, i, j):
    return np.load(DATA / exp / f"rep{rep:02d}" / "relations" / f"{i}-{j}.npy")


def fmt_t(t):
    return f"{t:g}"


def nn_map(Xi, Xj):
    """For each cell in Xj (later), index of nearest cell in Xi (earlier)."""
    return cKDTree(Xi).query(Xj, k=1)[1]


def topk_acc(exp, rep, i, j, k):
    """Fraction of later-cells whose true ancestor is among their k nearest."""
    Xi, Xj = snap(exp, rep, i), snap(exp, rep, j)
    true_anc = relmat(exp, rep, i, j).argmax(1)
    _, idx = cKDTree(Xi).query(Xj, k=k)
    idx = idx.reshape(len(Xj), k)
    return np.mean([true_anc[m] in idx[m] for m in range(len(Xj))])


def chained_top1_acc(exp, rep):
    """Compose consecutive NN maps to predict each final cell's t=0 ancestor."""
    times = EXP_TIMES[exp]
    nsteps = len(times) - 1
    maps = [nn_map(snap(exp, rep, k), snap(exp, rep, k + 1)) for k in range(nsteps)]
    anc = np.arange(len(snap(exp, rep, nsteps)))
    for k in range(nsteps - 1, -1, -1):
        anc = maps[k][anc]
    true_anc = relmat(exp, rep, 0, nsteps).argmax(1)
    return np.mean(anc == true_anc)


def direct_02_top1_acc(rep):
    """One-jump NN from t=0 to t=2 (endpoints only; identical across experiments)."""
    Xi, Xj = snap("exp1", rep, 0), snap("exp1", rep, 2)
    true_anc = relmat("exp1", rep, 0, 2).argmax(1)
    return np.mean(nn_map(Xi, Xj) == true_anc)


def main():
    rows = []

    # Figure 1: consecutive transitions, top-1 and top-3.
    for exp, times in EXP_TIMES.items():
        for k in range(len(times) - 1):
            transition = f"{fmt_t(times[k])}→{fmt_t(times[k + 1])}"
            for metric, kk in [("top-1", 1), ("top-3", 3)]:
                for rep in range(N_REP):
                    rows.append(dict(
                        figure=1, panel=exp, transition=transition,
                        method="NN", metric=metric, replicate=rep,
                        accuracy=topk_acc(exp, rep, k, k + 1, kk),
                    ))

    # Figure 2: the 0->2 comparison, top-1 only.
    for method, fn in [
        ("direct", direct_02_top1_acc),
        ("chained via exp1", lambda r: chained_top1_acc("exp1", r)),
        ("chained via exp2", lambda r: chained_top1_acc("exp2", r)),
    ]:
        for rep in range(N_REP):
            rows.append(dict(
                figure=2, panel="0→2", transition="0→2",
                method=method, metric="top-1", replicate=rep,
                accuracy=fn(rep),
            ))

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "nn_baseline_accuracy.csv"
    df.to_csv(out, index=False)

    # Console summary (mean over replicates).
    summ = (df.groupby(["figure", "panel", "transition", "method", "metric"])
              ["accuracy"].agg(["mean", "std"]).reset_index())
    print(f"Wrote {out}\n")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
