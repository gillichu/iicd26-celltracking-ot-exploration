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
import math
from dataclasses import dataclass


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


def _lay_down_positions(rng, cells, sigma, box, T, query_times):
    """Assign 2D Brownian-motion positions along the tree.

    Returns
    -------
    positions : dict[int, dict[float, (x, y)]]
        For each cell, its position at every query time it is present for.
    """
    positions: dict[int, dict[float, tuple[float, float]]] = {}
    div_pos: dict[int, tuple[float, float]] = {}

    # Ascending id guarantees parents are processed before their children.
    for cid in sorted(cells):
        c = cells[cid]
        if c.parent_id == -1:
            start = rng.uniform(0.0, box, size=2)
        else:
            start = div_pos[c.parent_id]

        # Present at snapshot s iff born by s and not yet divided at s. A cell that
        # never divides (div_time is None) is present through the horizon T, so the
        # terminal snapshot at T is included.
        present = [t for t in query_times
                   if c.birth_time <= t and (c.div_time is None or t < c.div_time)]

        # Times we must evaluate: snapshot times the cell is present for, plus its
        # own division time (needed to seed the daughters' starting positions).
        targets = sorted(set(present + ([c.div_time] if c.div_time is not None else [])))

        cur_t = c.birth_time
        cur_p = start.copy()
        pos_map: dict[float, tuple[float, float]] = {}
        for t in targets:
            dt = t - cur_t
            if dt > 0:
                cur_p = cur_p + rng.normal(0.0, sigma * math.sqrt(dt), size=2)
            cur_t = t
            pos_map[t] = cur_p.copy()

        positions[cid] = {t: pos_map[t] for t in present}
        if c.div_time is not None:
            div_pos[cid] = pos_map[c.div_time]

    return positions


def simulate_master(rng, *, n_founders, lam, sigma, box, T, query_times):
    """Simulate one master process.

    query_times should be the union of every snapshot time any experiment needs.

    Returns
    -------
    cells : dict[int, Cell]        the full lineage forest
    positions : dict[int, dict[float, (x, y)]]
    """
    query_times = sorted(set(float(t) for t in query_times))
    cells = _build_tree(rng, n_founders, lam, T)
    positions = _lay_down_positions(rng, cells, sigma, box, T, query_times)
    return cells, positions


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
