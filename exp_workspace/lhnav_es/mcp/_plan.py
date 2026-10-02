"""Grid path planning for the SLAM instrument tools.

Pure-grid helpers for preview_path (a read-only route overlay): A* over
FREE cells of one floor's occupancy grid, nearest-free snapping for targets
that land on unknown/obstacle cells, and metre-spaced waypoint downsampling. No habitat, no SLAM — numpy only, so the
module unit-tests offline.

Cells are (zi, xi) into the shared occupancy geometry (see _mapping.py);
8-connected moves cost cell_size (straight) / cell_size*sqrt(2) (diagonal).
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from ._mapping import FREE

_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1),
          (-1, -1), (-1, 1), (1, -1), (1, 1)]


def nearest_free(grid: np.ndarray, cell: tuple, max_r_cells: int) -> tuple | None:
    """Nearest FREE cell to ``cell`` within ``max_r_cells`` (euclidean), or
    None. Returns ``cell`` itself when it is already free."""
    zi, xi = cell
    n = grid.shape[0]
    if 0 <= zi < n and 0 <= xi < n and grid[zi, xi] == FREE:
        return cell
    z0, z1 = max(0, zi - max_r_cells), min(n, zi + max_r_cells + 1)
    x0, x1 = max(0, xi - max_r_cells), min(n, xi + max_r_cells + 1)
    cand = np.argwhere(grid[z0:z1, x0:x1] == FREE)
    if cand.size == 0:
        return None
    d2 = (cand[:, 0] + z0 - zi) ** 2 + (cand[:, 1] + x0 - xi) ** 2
    best = cand[int(np.argmin(d2))]
    if float(d2.min()) > max_r_cells ** 2:
        return None
    return int(best[0]) + z0, int(best[1]) + x0


def astar_free(grid: np.ndarray, start: tuple, goal: tuple) -> list | None:
    """Shortest 8-connected path over FREE cells, start/goal inclusive.
    Returns [(zi, xi), ...] or None when goal is unreachable."""
    if start == goal:
        return [start]
    n = grid.shape[0]
    gz, gx = goal
    open_q = [(0.0, start)]
    g_cost = {start: 0.0}
    came: dict = {}
    while open_q:
        _, cur = heapq.heappop(open_q)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        cz, cx = cur
        base = g_cost[cur]
        for dz, dx in _STEPS:
            nz, nx = cz + dz, cx + dx
            if not (0 <= nz < n and 0 <= nx < n) or grid[nz, nx] != FREE:
                continue
            ng = base + (1.4142135623730951 if dz and dx else 1.0)
            nxt = (nz, nx)
            if ng < g_cost.get(nxt, math.inf):
                g_cost[nxt] = ng
                came[nxt] = cur
                h = math.hypot(gz - nz, gx - nx)
                heapq.heappush(open_q, (ng + h, nxt))
    return None


def path_length_m(path: list, cell_size: float) -> float:
    total = 0.0
    for (z0, x0), (z1, x1) in zip(path, path[1:]):
        total += math.hypot(z1 - z0, x1 - x0) * cell_size
    return total


def downsample_waypoints(path: list, cell_size: float, every_m: float = 1.0) -> list:
    """Cells spaced ~``every_m`` along the path, final cell always kept."""
    if len(path) <= 1:
        return list(path)
    out = []
    acc = every_m  # emit the first step's waypoint once a metre has passed
    for prev, cur in zip(path, path[1:]):
        acc += math.hypot(cur[0] - prev[0], cur[1] - prev[1]) * cell_size
        if acc >= every_m:
            out.append(cur)
            acc = 0.0
    if not out or out[-1] != path[-1]:
        out.append(path[-1])
    return out
