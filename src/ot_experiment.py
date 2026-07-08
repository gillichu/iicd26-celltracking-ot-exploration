"""Optimal-transport ancestry recovery on the consecutive snapshot transitions.

Adapted from a collaborator's single-replicate script. Runs four unbalanced /
partial OT couplings (POT) on every consecutive transition of both experiments,
all 10 replicates, and scores top-1 ancestor recovery against ground truth.

Fix vs. the original: the snapshot CSV is `obs_id, x, y`; we use ONLY x, y as
spatial coordinates (the original fed obs_id in as a third coordinate).

Outputs:
  results/ot/<exp>/rep<NN>/<i>-<j>/<method>.npy   full coupling (ns, nt)
  results/ot/<exp>/rep<NN>/<i>-<j>/<method>.csv   sparse: source_obs_id, dest_obs_id, mass
  results/ot_summary.csv                          per case: accuracy, transport cost, mass
  results/ot_accuracy.csv                         long form (NN-compatible) for fig1 v2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import ot

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
N_REP = 10

EXP_TIMES = {"exp1": [0.0, 1.0, 2.0], "exp2": [0.0, 0.5, 1.0, 1.5, 2.0]}

# Collaborator's hyperparameters (kept as-is).
REG = 0.005          # entropic regularisation (on cost normalised to max 1)
REG_M_KL = 0.05      # marginal relaxation, KL
REG_M_L2 = 5.0       # marginal relaxation, L2
MASS = 0.7           # transported mass for partial OT

METHODS = ["entropic-kl", "mm-kl", "mm-l2", "partial"]
SPARSE_TOL = 1e-6    # keep coupling entries above SPARSE_TOL * max in the CSV


def fmt_t(t):
    return f"{t:g}"


def load_xy(exp, rep, k):
    df = pd.read_csv(DATA / exp / f"rep{rep:02d}" / "snapshots" / f"t{k}.csv")
    return df[["x", "y"]].to_numpy()          # obs_id is the row order; drop it as a coord


def relmat(exp, rep, i, j):
    return np.load(DATA / exp / f"rep{rep:02d}" / "relations" / f"{i}-{j}.npy")


def compute_couplings(xs, xt):
    """Return {method: coupling} and the normalised cost matrix M (ns, nt)."""
    a = np.ones(len(xs)) / len(xs)
    b = np.ones(len(xt)) / len(xt)
    M = ot.dist(xs, xt)          # squared Euclidean by default
    M = M / M.max()

    couplings = {
        "entropic-kl": ot.unbalanced.sinkhorn_unbalanced(a, b, M, REG, REG_M_KL),
        "mm-kl": ot.unbalanced.mm_unbalanced(a, b, M, REG_M_KL, div="kl"),
        "mm-l2": ot.unbalanced.mm_unbalanced(a, b, M, REG_M_L2, div="l2"),
        "partial": ot.partial.partial_wasserstein(a, b, M, m=MASS),
    }
    return couplings, M


def save_coupling(dest_dir, method, P):
    dest_dir.mkdir(parents=True, exist_ok=True)
    np.save(dest_dir / f"{method}.npy", P)
    ii, jj = np.where(P > SPARSE_TOL * P.max())
    pd.DataFrame({"source_obs_id": ii, "dest_obs_id": jj, "mass": P[ii, jj]}) \
        .to_csv(dest_dir / f"{method}.csv", index=False)


def main():
    summary, acc_long = [], []
    for exp, times in EXP_TIMES.items():
        for rep in range(N_REP):
            for i in range(len(times) - 1):
                j = i + 1
                transition = f"{fmt_t(times[i])}→{fmt_t(times[j])}"
                xs, xt = load_xy(exp, rep, i), load_xy(exp, rep, j)
                true_anc = relmat(exp, rep, i, j).argmax(1)   # source per dest
                couplings, M = compute_couplings(xs, xt)

                dest_dir = RESULTS / "ot" / exp / f"rep{rep:02d}" / f"{i}-{j}"
                for method, P in couplings.items():
                    save_coupling(dest_dir, method, P)
                    pred = P.argmax(0)                        # predicted source per dest
                    acc = float(np.mean(pred == true_anc))
                    summary.append(dict(
                        exp=exp, replicate=rep, i=i, j=j, transition=transition,
                        method=method, accuracy_top1=acc,
                        transport_cost=float(np.sum(P * M)), mass=float(P.sum())))
                    acc_long.append(dict(
                        figure=1, panel=exp, transition=transition, method=method,
                        metric="top-1", replicate=rep, accuracy=acc))

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(RESULTS / "ot_summary.csv", index=False)
    pd.DataFrame(acc_long).to_csv(RESULTS / "ot_accuracy.csv", index=False)

    # Console recap: mean top-1 accuracy per method x transition.
    s = pd.DataFrame(summary)
    recap = s.groupby(["exp", "transition", "method"])["accuracy_top1"].mean().reset_index()
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(recap.to_string(index=False))


if __name__ == "__main__":
    main()
