"""Core simulation: one continuous-time pure-birth (Yule) branching process
with independent 2D Brownian motion per cell.

A single master process is simulated per replicate over the full horizon; both
experiments then sample its trajectories at different snapshot times, so the two
experiments share identical underlying biology per replicate.

Conventions
-----------
- Binary division: a dividing cell retires and produces two fresh cells (new ids).
  A cell that does not divide keeps its id across snapshots, but observation files
  do NOT expose cell ids, so cross-time identity is never leaked to a tracker.
- No death (pure birth).
- A cell is "present" at snapshot time s iff it has been born and has not yet
  divided: birth_time <= s and (div_time is None or s < div_time). A cell that
  never divides is present through the horizon T, so it appears in the terminal
  snapshot. Division is a continuous random time, so a division landing exactly
  on a snapshot has probability zero; the strict < is only a boundary tie-break.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass
class Cell:
    cell_id: int
    parent_id: int      # -1 for founders
    founder_id: int     # id of the founder at the root of this lineage
    birth_time: float
    div_time: float | None   # None if the cell is still alive at the horizon T


def _build_tree(rng, n_founders, lam, T):
    """Simulate the lineage forest (times only, no space)."""
    cells: dict[int, Cell] = {}
    heap: list[tuple[float, int]] = []
    next_id = 0

    for _ in range(n_founders):
        cid = next_id
        next_id += 1
        div = rng.exponential(1.0 / lam)          # birth at t=0
        div_time = div if div <= T else None
        cells[cid] = Cell(cid, -1, cid, 0.0, div_time)
        if div_time is not None:
            heapq.heappush(heap, (div_time, cid))

    while heap:
        div_time, cid = heapq.heappop(heap)
        for _ in range(2):
            kid = next_id
            next_id += 1
            d = div_time + rng.exponential(1.0 / lam)
            kdiv = d if d <= T else None
            cells[kid] = Cell(kid, cid, cells[cid].founder_id, div_time, kdiv)
            if kdiv is not None:
                heapq.heappush(heap, (kdiv, kid))

    return cells


def _simulate_positions(rng, cells, sigma, box, T, snapshot_times, sim_dt,
                        return_trajectory):
    """Walk a continuous Brownian path along every lineage on a fine time grid.

    Each cell's path is sampled on a global grid of step ``sim_dt`` restricted to
    its lifespan, plus its exact birth and division instants. Snapshot positions
    are read off this same path, so a snapshot is an *exact subsample* of the true
    trajectory (no interpolation anywhere). Division positions are the path value
    at the (continuous) division time, and seed the daughters' starting positions.

    Returns
    -------
    positions : dict[int, dict[float, (x, y)]]   snapshot positions per cell
    div_pos   : dict[int, (x, y)]                position at each division
    trajectory: dict[int, (times[N], xy[N, 2])]  full path per cell (or None)
    """
    grid = np.round(np.arange(0.0, T + sim_dt / 2, sim_dt), 10)

    positions: dict[int, dict[float, np.ndarray]] = {}
    div_pos: dict[int, np.ndarray] = {}
    trajectory: dict | None = {} if return_trajectory else None

    # Ascending id guarantees parents are processed before their children.
    for cid in sorted(cells):
        c = cells[cid]
        start = rng.uniform(0.0, box, size=2) if c.parent_id == -1 \
            else div_pos[c.parent_id]

        end = c.div_time if c.div_time is not None else float(T)
        # Interior grid points strictly inside the lifespan, plus the exact
        # birth (start) and end (division or horizon) instants.
        interior = grid[(grid > c.birth_time) & (grid < end)]
        times = np.concatenate(([c.birth_time], interior, [end]))

        gaps = np.diff(times)
        steps = rng.normal(0.0, 1.0, size=(len(gaps), 2)) * (sigma * np.sqrt(gaps))[:, None]
        path = np.empty((len(times), 2))
        path[0] = start
        path[1:] = start + np.cumsum(steps, axis=0)

        row_of = {round(float(t), 10): k for k, t in enumerate(times)}

        # Present at snapshot s iff born by s and not yet divided at s. A cell that
        # never divides is present through the horizon, so it appears at t = T.
        pm = {}
        for s in snapshot_times:
            if c.birth_time <= s and (c.div_time is None or s < c.div_time):
                pm[s] = path[row_of[round(float(s), 10)]]
        positions[cid] = pm

        if c.div_time is not None:
            div_pos[cid] = path[-1]
        if return_trajectory:
            trajectory[cid] = (times, path)

    return positions, div_pos, trajectory


def simulate_master(rng, *, n_founders, lam, sigma, box, T, snapshot_times,
                    sim_dt, return_trajectory=False):
    """Simulate one master process.

    snapshot_times should be the union of every snapshot time any experiment
    needs; each must fall on the ``sim_dt`` grid so snapshots subsample the path.

    Returns
    -------
    cells      : dict[int, Cell]                 the full lineage forest
    positions  : dict[int, dict[float, (x, y)]]  snapshot positions
    div_pos    : dict[int, (x, y)]               position at each division
    trajectory : dict[int, (times, xy)] or None  full paths if requested
    """
    snapshot_times = sorted(set(float(t) for t in snapshot_times))
    cells = _build_tree(rng, n_founders, lam, T)
    positions, div_pos, trajectory = _simulate_positions(
        rng, cells, sigma, box, T, snapshot_times, sim_dt, return_trajectory)
    return cells, positions, div_pos, trajectory


def observed_cells(positions, s):
    """Sorted list of cell ids present at snapshot time s (deterministic order)."""
    return sorted(cid for cid, pm in positions.items() if s in pm)


def ancestor_in(cells, cid, target_ids):
    """Walk up the lineage from `cid` until reaching a cell in `target_ids`."""
    x = cid
    while x not in target_ids:
        x = cells[x].parent_id
        if x == -1:
            return None
    return x
