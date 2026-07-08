"""Generate the cell-tracking datasets from params.yaml.

For each replicate we simulate one master process, then write both experiments'
observations, ground-truth lineage, and ancestry-relation matrices.

Output layout (per experiment E in {exp1, exp2}, per replicate NN):
    data/E/repNN/snapshots/t{k}.csv        obs_id, x, y      (what a tracker sees)
    data/E/repNN/key/t{k}.csv              obs_id, true_cell_id   (hidden key)
    data/E/repNN/lineage.csv               cell_id,parent_id,founder_id,birth_time,div_time
    data/E/repNN/relations/{i}-{j}.npy     binary (n_j, n_i) ancestry matrix
    data/E/repNN/relations/{i}-{j}.csv     descendant_obs_id, ancestor_obs_id (edge list)
    data/manifest.json                     seeds, versions, git commit, summary
"""

from __future__ import annotations

import itertools
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import yaml

from simulate import simulate_master, observed_cells, ancestor_in

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"
DATA_DIR = ROOT / "data"


def git_commit():
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def write_snapshot(rep_dir, k, obs_ids_cells, positions, s):
    """Write observation and hidden-key files for one snapshot."""
    snap_dir = rep_dir / "snapshots"
    key_dir = rep_dir / "key"
    snap_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    with open(snap_dir / f"t{k}.csv", "w") as f_obs, \
         open(key_dir / f"t{k}.csv", "w") as f_key:
        f_obs.write("obs_id,x,y\n")
        f_key.write("obs_id,true_cell_id\n")
        for obs_id, cid in enumerate(obs_ids_cells):
            x, y = positions[cid][s]
            f_obs.write(f"{obs_id},{x:.6f},{y:.6f}\n")
            f_key.write(f"{obs_id},{cid}\n")


def write_lineage(rep_dir, cells, div_pos):
    with open(rep_dir / "lineage.csv", "w") as f:
        f.write("cell_id,parent_id,founder_id,birth_time,div_time,div_x,div_y\n")
        for cid in sorted(cells):
            c = cells[cid]
            if c.div_time is None:
                div, dx, dy = "", "", ""
            else:
                x, y = div_pos[cid]
                div, dx, dy = f"{c.div_time:.6f}", f"{x:.6f}", f"{y:.6f}"
            f.write(f"{cid},{c.parent_id},{c.founder_id},"
                    f"{c.birth_time:.6f},{div},{dx},{dy}\n")


def write_relation(rel_dir, i, j, cells, obs_i, obs_j):
    """Binary (n_j, n_i) ancestry matrix + edge-list CSV for snapshot pair i<j."""
    rel_dir.mkdir(parents=True, exist_ok=True)
    idx_i = {cid: a for a, cid in enumerate(obs_i)}
    M = np.zeros((len(obs_j), len(obs_i)), dtype=np.int8)
    edges = []
    target = set(obs_i)
    for d_idx, d_cid in enumerate(obs_j):
        a_cid = ancestor_in(cells, d_cid, target)
        a_idx = idx_i[a_cid]
        M[d_idx, a_idx] = 1
        edges.append((d_idx, a_idx))

    np.save(rel_dir / f"{i}-{j}.npy", M)
    with open(rel_dir / f"{i}-{j}.csv", "w") as f:
        f.write("descendant_obs_id,ancestor_obs_id\n")
        for d_idx, a_idx in edges:
            f.write(f"{d_idx},{a_idx}\n")


def main():
    params = yaml.safe_load(PARAMS_PATH.read_text())
    n_reps = params["n_replicates"]
    T = params["horizon_T"]
    sim_dt = params["sim_grid_dt"]
    experiments = params["experiments"]

    # Union of all snapshot times across experiments -> the master query grid.
    union_times = sorted(set(
        float(t) for exp in experiments.values() for t in exp["snapshot_times"]
    ))

    master_ss = np.random.SeedSequence(params["seed"])
    rep_seeds = master_ss.spawn(n_reps)

    summary = {"replicates": []}
    for rep in range(n_reps):
        rng = np.random.default_rng(rep_seeds[rep])
        cells, positions, div_pos, _ = simulate_master(
            rng,
            n_founders=params["n_founders"],
            lam=params["lambda_div"],
            sigma=params["sigma"],
            box=params["founder_box"],
            T=T,
            snapshot_times=union_times,
            sim_dt=sim_dt,
        )

        rep_summary = {"replicate": rep, "n_cells_total": len(cells), "experiments": {}}
        for exp_name, exp in experiments.items():
            times = [float(t) for t in exp["snapshot_times"]]
            rep_dir = DATA_DIR / exp_name / f"rep{rep:02d}"

            # Precompute the observed-cell cut at each snapshot.
            obs_by_time = {s: observed_cells(positions, s) for s in times}

            for k, s in enumerate(times):
                write_snapshot(rep_dir, k, obs_by_time[s], positions, s)
            write_lineage(rep_dir, cells, div_pos)

            rel_dir = rep_dir / "relations"
            for i, j in itertools.combinations(range(len(times)), 2):
                write_relation(rel_dir, i, j, cells,
                               obs_by_time[times[i]], obs_by_time[times[j]])

            rep_summary["experiments"][exp_name] = {
                "snapshot_times": times,
                "counts": [len(obs_by_time[s]) for s in times],
            }
        summary["replicates"].append(rep_summary)

    manifest = {
        "params": params,
        "master_seed": params["seed"],
        "replicate_seeds": [
            {"entropy": s.entropy, "spawn_key": list(s.spawn_key)} for s in rep_seeds
        ],
        "union_snapshot_times": union_times,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
        },
        "git_commit": git_commit(),
        "summary": summary,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Console recap.
    print(f"Generated {n_reps} replicates into {DATA_DIR}")
    for rs in summary["replicates"]:
        parts = ", ".join(
            f"{name} {e['counts']}" for name, e in rs["experiments"].items()
        )
        print(f"  rep{rs['replicate']:02d}  total_cells={rs['n_cells_total']:4d}  {parts}")


if __name__ == "__main__":
    main()
