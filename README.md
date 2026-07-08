# iicd26-celltracking-ot-exploration

Synthetic cell-tracking data for exploring optimal-transport (OT) approaches to
lineage reconstruction.

We simulate a **birth-only (pure-birth / Yule) branching process** in continuous
time, with each cell performing independent **2D Brownian motion**. We observe the
population as spatial snapshots at a set of timepoints and provide the
ground-truth ancestry between snapshots — i.e. the true transport coupling an OT
method should recover.

## The process

- Start with `N = 100` founder cells at `t = 0`, placed uniformly in a `[0, 20]²` box.
- Each living cell divides after an `Exp(λ)` waiting time (`λ = ln 2`, one doubling
  per unit time); division is binary (parent retires, two fresh cells appear); no death.
- Each cell diffuses as Brownian motion, `dx, dy ~ N(0, σ² dt)` with `σ = 0.5`,
  sampled on a fine grid (`sim_grid_dt = 0.01`). Daughters start at the parent's
  exact position at the division instant.
- Horizon `T = 2`, giving ~4× growth (≈390 cells by `t = 2`).
- 10 replicates, each a fresh master process with a reproducible spawned seed.

The snapshots are **exact subsamples of each cell's true Brownian trajectory** — no
interpolation. The full trajectory is not stored (it is deterministically
regenerable from the seed via `simulate_master(..., return_trajectory=True)`), but
every division's spatial position is recorded in `lineage.csv`.

All parameters live in [`params.yaml`](params.yaml).

## Two experiments, one process

Both experiments **sample the same master trajectories per replicate**, differing
only in snapshot density — so they are directly comparable.

| Experiment | Snapshot times            |
|------------|---------------------------|
| `exp1`     | `0, 1, 2`                 |
| `exp2`     | `0, 0.5, 1, 1.5, 2`       |

## Data layout

```
data/<exp>/rep<NN>/
  snapshots/t<k>.csv     # obs_id, x, y      — the observation a tracker sees
  key/t<k>.csv           # obs_id, true_cell_id   — hidden lineage key (for evaluation/debug)
  lineage.csv            # cell_id, parent_id, founder_id, birth_time, div_time, div_x, div_y
  relations/<i>-<j>.npy  # binary (n_j, n_i) ancestry matrix, all forward pairs i<j
  relations/<i>-<j>.csv  # descendant_obs_id, ancestor_obs_id   (same info, edge list)
data/manifest.json       # seeds, library versions, git commit, per-replicate counts
```

**Observations** expose only `obs_id` (a per-snapshot row index), never a persistent
cell id — cross-time identity is not leaked to the tracker. The hidden `key/` files
map each `obs_id` back to its true `cell_id` for scoring.

**Ancestry matrix** `M` for a snapshot pair `(t_i, t_j)`, `i < j`, has shape
`(n_j, n_i)`: `M[d, a] = 1` iff observation `a` at `t_i` is the ancestor of
observation `d` at `t_j`. Every row sums to exactly 1 (unique ancestor); columns
sum to ≥ 1 (a cell may have several descendants). This is the ground-truth 0/1
transport coupling to score an OT plan against.

## Reproducing

```bash
pip install -r requirements.txt
python src/generate.py
```

Generation is deterministic from `seed` in `params.yaml`; regenerating produces
byte-identical files. To change scale/rates/noise, edit `params.yaml` and rerun.

## Layout

- `src/simulate.py` — the master process (lineage forest + Brownian positions).
- `src/generate.py` — runs all replicates × experiments and writes `data/`.
