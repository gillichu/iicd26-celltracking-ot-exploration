"""Generate V2 datasets with more spread-out cell movement.

Same simulation as generate.py, but with a larger Brownian-motion sigma (4x the
original 0.5 -> 2.0), so cell tracks are more spread out. Writes to new
exp1_V2 / exp2_V2 folders under data/, and a separate manifest_v2.json, so the
original exp1/exp2 data and manifest.json are never touched.

Output layout (per experiment E in {exp1_V2, exp2_V2}, per replicate NN):
    data/E/repNN/snapshots/t{k}.csv        obs_id, x, y      (what a tracker sees)
    data/E/repNN/key/t{k}.csv              obs_id, true_cell_id   (hidden key)
    data/E/repNN/lineage.csv               cell_id,parent_id,founder_id,birth_time,div_time
    data/E/repNN/relations/{i}-{j}.npy     binary (n_j, n_i) ancestry matrix
    data/E/repNN/relations/{i}-{j}.csv     descendant_obs_id, ancestor_obs_id (edge list)
    data/manifest_v2.json                  seeds, versions, git commit, summary
"""

from __future__ import annotations

import itertools
import json
import platform
from pathlib import Path

import numpy as np
import yaml

from simulate import simulate_master, observed_cells
from generate import git_commit, write_snapshot, write_lineage, write_relation

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"
DATA_DIR = ROOT / "data"

SIGMA_V2 = 2.0          # 4x the original sigma (0.5) -> cells spread out more
EXPERIMENT_SUFFIX = "_V2"


def main():
    params = yaml.safe_load(PARAMS_PATH.read_text())
    params["sigma"] = SIGMA_V2

    n_reps = params["n_replicates"]
    T = params["horizon_T"]
    sim_dt = params["sim_grid_dt"]
    experiments = params["experiments"]

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
            out_name = f"{exp_name}{EXPERIMENT_SUFFIX}"
            times = [float(t) for t in exp["snapshot_times"]]
            rep_dir = DATA_DIR / out_name / f"rep{rep:02d}"

            obs_by_time = {s: observed_cells(positions, s) for s in times}

            for k, s in enumerate(times):
                write_snapshot(rep_dir, k, obs_by_time[s], positions, s)
            write_lineage(rep_dir, cells, div_pos)

            rel_dir = rep_dir / "relations"
            for i, j in itertools.combinations(range(len(times)), 2):
                write_relation(rel_dir, i, j, cells,
                               obs_by_time[times[i]], obs_by_time[times[j]])

            rep_summary["experiments"][out_name] = {
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
    (DATA_DIR / "manifest_v2.json").write_text(json.dumps(manifest, indent=2))

    out_names = [f"{name}{EXPERIMENT_SUFFIX}" for name in experiments]
    print(f"Generated {n_reps} replicates (sigma={SIGMA_V2}) into {DATA_DIR} "
          f"[{', '.join(out_names)}]")
    for rs in summary["replicates"]:
        parts = ", ".join(
            f"{name} {e['counts']}" for name, e in rs["experiments"].items()
        )
        print(f"  rep{rs['replicate']:02d}  total_cells={rs['n_cells_total']:4d}  {parts}")


if __name__ == "__main__":
    main()
