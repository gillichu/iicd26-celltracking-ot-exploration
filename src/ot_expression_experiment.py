"""Optimal-transport ancestry recovery using MOSTA-derived gene-expression PCA
embeddings only (no spatial position).

Adapted from ot_experiment.py (which costs on x, y). Here the transport cost is
squared-Euclidean distance between each observation's PC1..PC30 embedding instead.
The link from an observation to its expression profile goes through the hidden
key/tk.csv (obs_id -> true_cell_id): this is used only to look up that single
observation's own PC vector from pca_embedding.csv (indexed by cell_id, covering
the whole simulated lineage) -- the same role key/tk.csv already plays for x, y in
generate.py. It is never used to compare identity *across* snapshots, so no
ground-truth correspondence is leaked into the transport cost; the true ancestor
labels (relations/i-j.npy) are used only for scoring, exactly as in
ot_experiment.py.

Outputs:
  results/ot_expr/<exp>/rep<NN>/<i>-<j>/<method>.npy   full coupling (ns, nt)
  results/ot_expr/<exp>/rep<NN>/<i>-<j>/<method>.csv   sparse: source_obs_id, dest_obs_id, mass
  results/ot_expr_summary.csv    per case: accuracy, transport cost, mass
  results/ot_expr_accuracy.csv   long form (NN-compatible) for plotting
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import ot
from scipy.optimize import linprog
from scipy.sparse import eye, kron

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
N_REP = 10

EXP_TIMES = {"exp1": [0.0, 1.0, 2.0], "exp2": [0.0, 0.5, 1.0, 1.5, 2.0]}

# Same OT hyperparameters as ot_experiment.py, kept identical for comparability.
REG = 0.005          # entropic regularisation (on cost normalised to max 1)
REG_M_KL = [0.01, 0.5 ]  # marginal relaxation, KL: (a small / source loose, b large / target strict)
REG_M_L2 = [1.0, 50.0 ]  # marginal relaxation, L2: (a small / source loose, b large / target strict)
MASS = 0.7           # transported mass for partial OT

METHODS = ["entropic-kl", "mm-kl", "mm-l2", "partial", "covered-lp"]
SPARSE_TOL = 1e-6    # keep coupling entries above SPARSE_TOL * max in the CSV


def fmt_t(t):
    return f"{t:g}"


def covered_lp(M):
    """
    M: (n_anc, n_desc) cost matrix, e.g. normalized squared Euclidean
       distance between ancestor and descendant PC embeddings.

    Returns P: (n_anc, n_desc) optimal transport plan.

    Constraints:
      - sum_a P[a, d] == 1   for every d   (each descendant fully "covered"
                                              by exactly one unit of mass,
                                              split across ancestors)
      - sum_d P[a, d] >= 1   for every a   (birth-only: every ancestor must
                                              have at least one descendant,
                                              no orphaned ancestors)
      - 0 <= P[a, d] <= 1
    """
    n_anc, n_desc = M.shape
    if n_desc < n_anc:
        raise ValueError(
            f"infeasible: {n_anc} ancestors each need >=1 descendant, "
            f"but only {n_desc} descendants exist to cover exactly once"
        )

    c = M.ravel()  # flatten row-major: index(a, d) = a * n_desc + d

    # sum_a P[a,d] == 1  -->  A_eq @ x == b_eq
    A_eq = kron(np.ones((1, n_anc)), eye(n_desc))       # (n_desc, n_anc*n_desc)
    b_eq = np.ones(n_desc)

    # sum_d P[a,d] >= 1  -->  -sum_d P[a,d] <= -1
    A_ub = -kron(eye(n_anc), np.ones((1, n_desc)))       # (n_anc, n_anc*n_desc)
    b_ub = -np.ones(n_anc)

    bounds = [(0, 1)] * (n_anc * n_desc)

    res = linprog(
        c,
        A_ub=A_ub.tocsr(), b_ub=b_ub,
        A_eq=A_eq.tocsr(), b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    return res.x.reshape(n_anc, n_desc)


def load_expr(exp, rep, k, pca):
    """PC1..PC30 for each observation at snapshot k, in obs_id row order.

    key/tk.csv is written in the same obs_id order as snapshots/tk.csv (both come
    from the same enumerate() loop in generate.py), so a straight row lookup by
    true_cell_id lines up with that obs_id order without any extra join logic.
    """
    key = pd.read_csv(DATA / exp / f"rep{rep:02d}" / "key" / f"t{k}.csv")
    return pca.loc[key["true_cell_id"]].to_numpy()


def relmat(exp, rep, i, j):
    return np.load(DATA / exp / f"rep{rep:02d}" / "relations" / f"{i}-{j}.npy")


def compute_couplings(es, et):
    """Return {method: coupling} and the normalised cost matrix M (ns, nt)."""
    a = np.ones(len(es)) / len(es)
    b = np.ones(len(et)) / len(et)
    M = ot.dist(es, et)          # squared Euclidean by default
    M = M / M.max()

    couplings = {
        "entropic-kl": ot.unbalanced.sinkhorn_unbalanced(a, b, M, REG, REG_M_KL),
        "mm-kl": ot.unbalanced.mm_unbalanced(a, b, M, REG_M_KL, div="kl"),
        "mm-l2": ot.unbalanced.mm_unbalanced(a, b, M, REG_M_L2, div="l2"),
        "partial": ot.partial.partial_wasserstein(a, b, M, m=MASS),
        "covered-lp": covered_lp(M),
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
            rep_dir = DATA / exp / f"rep{rep:02d}"
            pca = pd.read_csv(rep_dir / "pca_embedding.csv").set_index("cell_id")

            for i in range(len(times) - 1):
                j = i + 1
                transition = f"{fmt_t(times[i])}→{fmt_t(times[j])}"
                es, et = load_expr(exp, rep, i, pca), load_expr(exp, rep, j, pca)
                true_anc = relmat(exp, rep, i, j).argmax(1)   # source per dest
                couplings, M = compute_couplings(es, et)

                dest_dir = RESULTS / "ot_expr" / exp / f"rep{rep:02d}" / f"{i}-{j}"
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
    pd.DataFrame(summary).to_csv(RESULTS / "ot_expr_summary.csv", index=False)
    pd.DataFrame(acc_long).to_csv(RESULTS / "ot_expr_accuracy.csv", index=False)

    # Console recap: mean top-1 accuracy per method x transition.
    s = pd.DataFrame(summary)
    recap = s.groupby(["exp", "transition", "method"])["accuracy_top1"].mean().reset_index()
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(recap.to_string(index=False))


if __name__ == "__main__":
    main()
