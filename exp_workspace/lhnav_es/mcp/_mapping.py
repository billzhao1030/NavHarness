"""2D occupancy mapping from depth + camera pose (+ map rendering).

Frame convention: poses are 4x4 world-from-camera in OpenCV camera axes
(x right, y DOWN, z forward), with the world anchored at the FIRST camera
frame — exactly what ORB-SLAM3 emits, and what _slam_env converts GT poses
to. The camera is level at ``camera_height`` above the floor, so in this
world frame the floor sits near y = +camera_height and the ground plane is
x-z.

Grid: fixed-size square, ``map_size_m`` across at ``cell_size`` resolution,
centered on the world origin (= the start position). Ternary cells:
0 unknown / 1 free / 2 obstacle, with obstacle-wins overwrite and free
carved by per-ray Bresenham from the agent cell to each measured endpoint.
Depth is read on the closed band [depth_min, depth_max], and an obstacle is
never overwritten by free — not by a ray, and not by the body's own cell.
(Map v3, ``EvidenceMap`` below, reads the same band but keeps evidence
rather than a verdict, so a later look can take a cell back.)
"""

from __future__ import annotations

import numpy as np

UNKNOWN, FREE, OBSTACLE = 0, 1, 2


class OccupancyMap:
    def __init__(
        self,
        cell_size: float = 0.10,
        map_size_m: float = 48.0,
        camera_height: float = 1.25,
        obstacle_h_min: float = 0.15,
        obstacle_h_max: float = 1.4,
        depth_min: float = 0.3,
        depth_max: float = 4.5,
        px_stride: int = 4,
    ) -> None:
        self.cell_size = cell_size
        self.n = round(map_size_m / cell_size)
        self.origin = self.n // 2  # world (0,0) -> cell (origin, origin)
        self.grid = np.zeros((self.n, self.n), dtype=np.uint8)  # [zi, xi]
        self.camera_height = camera_height
        self.obstacle_h_min = obstacle_h_min
        self.obstacle_h_max = obstacle_h_max
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.px_stride = px_stride

    # ── coordinate helpers ──

    def world_to_cell(self, x: float, z: float) -> tuple:
        xi = int(np.floor(x / self.cell_size)) + self.origin
        zi = int(np.floor(z / self.cell_size)) + self.origin
        return zi, xi

    def cell_to_world(self, zi: int, xi: int) -> tuple:
        x = (xi - self.origin + 0.5) * self.cell_size
        z = (zi - self.origin + 0.5) * self.cell_size
        return x, z

    def in_bounds(self, zi: int, xi: int) -> bool:
        return 0 <= zi < self.n and 0 <= xi < self.n

    # ── integration ──

    def integrate(self, pose_wc: np.ndarray, depth_m: np.ndarray, intrinsics: dict,
                  floor_ref_y: float | None = None, weight: float = 1.0) -> None:
        """Project one depth frame through the pose into the grid.

        ``weight`` is map v3's confidence knob and is ignored here: a verdict
        map has no way to believe something a little.

        ``floor_ref_y`` (map v2): camera world-y to use as the height
        reference, so obstacle/floor classification is relative to the floor
        UNDER the camera. Default None keeps the v1 behaviour (reference =
        the start floor, world y 0).

        Every ray is rasterised at once (``_bresenham_many``), which gives
        the same grid a per-ray loop would at a fraction of the cost.

        The scatter is safe because the frame-level result is order
        independent: a cell that is an obstacle endpoint this frame ends
        OBSTACLE whichever ray is visited first (the free write is guarded by
        ``!= OBSTACLE``, the endpoint write is not), and every other touched
        cell ends FREE unless it was already OBSTACLE. The body's own cell
        follows the same rule: standing in it says it is free, but an
        obstacle already there is kept.
        """
        got = self._rays(pose_wc, depth_m, intrinsics, floor_ref_y)
        if got is None:
            return
        free_cells, obst, agent = got
        flat_grid = self.grid.reshape(-1)
        if agent is not None and flat_grid[agent] != OBSTACLE:
            flat_grid[agent] = FREE
        is_obst = np.zeros(self.n * self.n, bool)
        is_obst[obst] = True
        writable = (~is_obst[free_cells]) & (flat_grid[free_cells] != OBSTACLE)
        flat_grid[free_cells[writable]] = FREE
        flat_grid[obst] = OBSTACLE

    def _rays(self, pose_wc: np.ndarray, depth_m: np.ndarray, intrinsics: dict,
              floor_ref_y: float | None = None):
        """What one frame touches: (free cells, obstacle cells, agent cell).

        Flat indices (zi * n + xi), deduplicated, obstacle-wins-within-the-
        frame, ray endpoints excluded from the free set. Shared by
        integrate() and map v3, so v3 writes EVIDENCE where v1/v2 write a
        verdict, from exactly the same geometry.
        """
        s = self.px_stride
        h, w = depth_m.shape
        fx = intrinsics["fx"] * w / intrinsics["width"]
        fy = intrinsics["fy"] * h / intrinsics["height"]
        cx = intrinsics["cx"] * w / intrinsics["width"]
        cy = intrinsics["cy"] * h / intrinsics["height"]

        vs, us = np.mgrid[0:h:s, 0:w:s]
        z = depth_m[::s, ::s].astype(np.float64)
        valid = (z >= self.depth_min) & (z <= self.depth_max)     # the closed band
        us, vs, z = us[valid], vs[valid], z[valid]
        if z.size == 0:
            return None
        x_cam = (us - cx) / fx * z
        y_cam = (vs - cy) / fy * z
        pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=0)
        pts_w = pose_wc @ pts_cam  # 4xN

        ref_y = 0.0 if floor_ref_y is None else floor_ref_y
        height_above_floor = self.camera_height - (pts_w[1] - ref_y)
        is_obstacle = (height_above_floor > self.obstacle_h_min) & \
                      (height_above_floor < self.obstacle_h_max)
        is_floor = (height_above_floor <= self.obstacle_h_min) & \
                   (height_above_floor > -0.5)
        keep = is_obstacle | is_floor
        if not keep.any():
            return None
        xw, zw = pts_w[0][keep], pts_w[2][keep]
        obs_flag = is_obstacle[keep]

        a_zi, a_xi = self.world_to_cell(float(pose_wc[0, 3]), float(pose_wc[2, 3]))
        agent = a_zi * self.n + a_xi if self.in_bounds(a_zi, a_xi) else None

        # Endpoint cells, deduplicated (obstacle wins within the frame)
        zi = np.floor(zw / self.cell_size).astype(int) + self.origin
        xi = np.floor(xw / self.cell_size).astype(int) + self.origin
        ib = (zi >= 0) & (zi < self.n) & (xi >= 0) & (xi < self.n)
        zi, xi, obs_flag = zi[ib], xi[ib], obs_flag[ib]
        if zi.size == 0:
            return None

        seen_flat = zi * self.n + xi
        order = np.argsort(seen_flat, kind="stable")
        flat_s = seen_flat[order]
        obs_s = obs_flag[order]
        uniq, first = np.unique(flat_s, return_index=True)
        obs_u = np.maximum.reduceat(obs_s.astype(np.int8), first).astype(bool)
        ez = uniq // self.n
        ex = uniq % self.n

        zz, xx, ray = _bresenham_many(a_zi, a_xi, ez, ex)
        last = np.zeros(zz.size, bool)
        last[np.cumsum(np.bincount(ray, minlength=uniq.size)) - 1] = True
        inner = ~last

        obst = uniq[obs_u]
        # A ray starts at the AGENT's cell, which need not be on the grid: a
        # tour can walk (or a drifting pose wander) past the 48 m square. Its
        # inner cells are then partly off the grid too, and a flat index built
        # from them either overruns it (an IndexError) or, for a negative row,
        # wraps silently onto the opposite edge and paints free space across a wall there. Keep only
        # the cells that exist.
        on_grid = inner & (zz >= 0) & (zz < self.n) & (xx >= 0) & (xx < self.n)
        free_cells = np.concatenate([zz[on_grid] * self.n + xx[on_grid], uniq[~obs_u]])
        return free_cells, obst, agent

    def mark_obstacle_ahead(self, pose_wc: np.ndarray, dist_m: float = 0.3) -> None:
        """On collision: stamp an obstacle just ahead of the camera. A
        pitched camera has its forward projected onto the ground plane
        first, so the stamp lands ``dist_m`` ahead of the BODY; a level
        camera's forward is used as is."""
        fwd = pose_wc[:3, 2].copy()  # camera z axis in world
        if abs(float(fwd[1])) > 1e-6:
            fwd[1] = 0.0
            norm = float(np.linalg.norm(fwd))
            if norm < 1e-9:  # looking straight up/down — no meaningful "ahead"
                return
            fwd = fwd / norm
        p = pose_wc[:3, 3] + fwd * dist_m
        zi, xi = self.world_to_cell(float(p[0]), float(p[2]))
        if self.in_bounds(zi, xi):
            self.grid[zi, xi] = OBSTACLE

    # ── stats ──

    def explored_area_m2(self) -> float:
        return float((self.grid != UNKNOWN).sum()) * self.cell_size ** 2

    def free_area_m2(self) -> float:
        return float((self.grid == FREE).sum()) * self.cell_size ** 2


class LayeredOccupancyMap:
    """Per-floor occupancy layers (map v2).

    One OccupancyMap per registered floor. The camera's world-y picks the
    layer each frame: within ``floor_merge_m`` of a registered floor it
    joins that floor, otherwise a new floor is registered. Heights inside a
    frame are classified relative to the CURRENT camera height, so the
    plane under the agent is always the reference — v1 anchors everything
    to the start floor and so misclassifies other storeys.

    ``.grid`` exposes the current floor's grid, so single-grid callers
    (render, frontier, evaluate) work unchanged. Geometry constants
    (cell_size / n / origin) are shared by construction.
    """

    def __init__(
        self,
        cell_size: float = 0.10,
        map_size_m: float = 48.0,
        camera_height: float = 1.25,
        floor_merge_m: float = 1.0,
        dwell_feeds: int = 8,
        dwell_span_m: float = 0.2,
        layer_cls: type = None,
        floor_policy: str = "height",
        stair_m: float = 1.8,             # the shortest credible storey
        window_feeds: int = 700,          # ~3 min of micro-steps: longer than a climb, far
                                          # shorter than the day drift needs to fake one
    ) -> None:
        # map v3 is this same registry over EvidenceMap layers — the floor
        # logic is unchanged, only what a layer does with a frame.
        self.layer_cls = layer_cls or OccupancyMap
        self._layer_kw = {"cell_size": cell_size, "map_size_m": map_size_m,
                          "camera_height": camera_height}
        self.floor_merge_m = floor_merge_m
        # Floor switches need DWELL: mid-stair heights would otherwise
        # register a fencepost "floor" every merge-threshold of climb. Only when the
        # camera height has been stable — span < dwell_span_m over the last
        # dwell_feeds frames — may the registry switch or register.
        self._dwell_feeds = dwell_feeds
        self._dwell_span_m = dwell_span_m
        self._y_hist: list = []
        # floor_policy:
        #   "height"  a settled camera height is matched against the registered
        #             floor heights, and a new floor is born if it is more than
        #             floor_merge_m from all of them. It reads an INTEGRAL:
        #             every metre of accumulated vertical drift mints a storey.
        #   "stairs"  a level index that moves by +-1 when the robot TRAVERSES
        #             something, and never otherwise. It reads a DERIVATIVE over
        #             a short window, which is the half that survives drift:
        #             within one window the believed height drifts far less than
        #             a storey (2.2-2.6 m in these houses), and nothing accumulates.
        self.floor_policy = str(floor_policy)
        self.stair_m = float(stair_m)
        self._win_feeds = int(window_feeds)
        self._y_win: list = []      # the height of the last level we stood still on
        self.level = 0                            # topological: +1 up a staircase, -1 down
        self._level_layer: dict = {0: 0}          # level -> index into self.layers
        self.transitions: list = []               # [{from, to, dy, feed}] — the paper trail
        # world is anchored at the start camera, so floor 0 sits at y = 0
        self.layers: list = [self.layer_cls(**self._layer_kw)]
        self.floor_ys: list = [0.0]
        self.current_floor = 0
        proto = self.layers[0]
        self.cell_size = proto.cell_size
        self.n = proto.n
        self.origin = proto.origin

    # ── floor registry ──

    def _update_floor(self, pose_y: float) -> None:
        self._y_hist.append(pose_y)
        if len(self._y_hist) > self._dwell_feeds:
            self._y_hist.pop(0)
        if len(self._y_hist) < self._dwell_feeds:
            return  # not enough history yet — stay on the current floor
        if self.floor_policy == "stairs":
            # NOT gated on the dwell test: a staircase taken in 0.05 m steps
            # passes it on every window, so the whole climb would vanish into
            # "drift".
            self._update_level(pose_y)
            return
        if max(self._y_hist) - min(self._y_hist) >= self._dwell_span_m:
            return  # climbing/descending — no switches mid-transit
        y_stable = sorted(self._y_hist)[len(self._y_hist) // 2]
        d, fid = min((abs(y_stable - fy), i) for i, fy in enumerate(self.floor_ys))
        if d <= self.floor_merge_m:
            self.current_floor = fid
            return
        self.floor_ys.append(y_stable)
        self.layers.append(self.layer_cls(**self._layer_kw))
        self.current_floor = len(self.layers) - 1

    # ── the stairs policy ─────────────────────────────────────────────

    def _update_level(self, y_now: float) -> None:
        """Move a level when the height has MOVED A STOREY WITHIN A WINDOW.

        The statistic is |y(t) - y(t-W)| over a window of a few hundred
        feeds. Drift within one window stays far below one storey (2.2-2.6 m
        in these houses), and it cannot accumulate into the statistic because
        the window slides.

        Two simpler designs do not work:

        * comparing the last two SETTLED heights. On a staircase climbed in
          0.05 m micro-steps the rise per 8-frame dwell window is under 0.2 m,
          so the whole climb reads as a sequence of settles and is absorbed as
          drift, 0.1 m at a time.
        * requiring the change to be QUICK (a max gap in feeds): a gap short
          enough to matter is shorter than an actual climb, so it blocks the
          event the policy exists for.
        """
        self._y_win.append(float(y_now))
        if len(self._y_win) > self._win_feeds:
            self._y_win.pop(0)
        if len(self._y_win) < self._win_feeds:
            return
        dy = self._y_win[-1] - self._y_win[0]
        if abs(dy) < self.stair_m:
            return
        # the map frame is OpenCV: y points DOWN, so climbing makes y smaller.
        # One traversal is one level, whatever its height: a level index is
        # topological, and a robot that climbs two flights without pausing is
        # rarer than a tracker that mismeasures one.
        self.set_level(self.level + (-1 if dy > 0 else 1), self._y_win[-1], dy=dy)
        self._y_win.clear()          # one climb fires once

    def set_level(self, level: int, y_hint: float | None = None, dy: float | None = None) -> int:
        """Put the robot on a level by NAME, creating the layer if it is new.

        The other way a floor changes is that somebody carries the robot
        (a wake-up after a move). A traversal counter cannot see that, so the
        wake-up hands the level in from the place it recognised, and this is
        the door.
        """
        level = int(level)
        if level not in self._level_layer:
            self.layers.append(self.layer_cls(**self._layer_kw))
            self.floor_ys.append(float(y_hint if y_hint is not None else 0.0))
            self._level_layer[level] = len(self.layers) - 1
        if level != self.level:
            self.transitions.append({"from": self.level, "to": level,
                                     "dy": (None if dy is None else round(float(dy), 3)),
                                     "layer": self._level_layer[level]})
        self.level = level
        self.current_floor = self._level_layer[level]
        if y_hint is not None:
            self._y_win.clear()
        return self.current_floor

    @property
    def grid(self) -> np.ndarray:
        return self.layers[self.current_floor].grid

    @property
    def current_floor_y(self) -> float:
        return self.floor_ys[self.current_floor]

    # ── integration (mirrors OccupancyMap's surface) ──

    def integrate(self, pose_wc: np.ndarray, depth_m: np.ndarray, intrinsics: dict,
                  ref_y: float | None = None, weight: float = 1.0) -> None:
        """``ref_y`` (lever-arm camera): the LEVEL eye height to use as the
        floor reference instead of the camera's own y — a pitched camera on
        the lever drops below eye height, which would otherwise misread the
        floor as a 0.2 m obstacle. None = the camera's own y."""
        pose_y = float(pose_wc[1, 3]) if ref_y is None else float(ref_y)
        self._update_floor(pose_y)
        self.layers[self.current_floor].integrate(
            pose_wc, depth_m, intrinsics, floor_ref_y=pose_y, weight=weight)

    def mark_obstacle_ahead(self, pose_wc: np.ndarray, dist_m: float = 0.3) -> None:
        self.layers[self.current_floor].mark_obstacle_ahead(pose_wc, dist_m)

    # ── coordinate helpers (pure geometry, layer-independent) ──

    def world_to_cell(self, x: float, z: float) -> tuple:
        return self.layers[0].world_to_cell(x, z)

    def cell_to_world(self, zi: int, xi: int) -> tuple:
        return self.layers[0].cell_to_world(zi, xi)

    def in_bounds(self, zi: int, xi: int) -> bool:
        return self.layers[0].in_bounds(zi, xi)

    # ── stats (all floors) ──

    def explored_area_m2(self) -> float:
        return float(sum(layer.explored_area_m2() for layer in self.layers))

    def free_area_m2(self) -> float:
        return float(sum(layer.free_area_m2() for layer in self.layers))

    def decay(self, gamma: float = 0.5) -> dict:
        """Forget proportionally on every floor (map v3 layers only)."""
        out = [lay.decay(gamma) for lay in self.layers if hasattr(lay, "decay")]
        return {"layers": len(out), "known_before": sum(o["known_before"] for o in out),
                "known_after": sum(o["known_after"] for o in out), "decayed": float(gamma)}



# ── map v3: evidence, not a verdict ───────────────────────────────────
#
# Quarter-units of log-odds, so a confidence weight of 0.25 is still a whole
# step. The thresholds are set so ONE full-weight observation classifies a
# cell exactly as v1/v2 would: the change is not that the map believes less,
# it is that it can be talked out of what it believes.
L_OCC, L_FREE, L_CLAMP = 8, 4, 80        # +2.0 / -1.0 / +-20 log-odds
T_OCC, T_FREE = 5, -3                    # >= obstacle, <= free, between = unknown
L_BUMP = 24                              # a collision: felt, not seen


class EvidenceMap(OccupancyMap):
    """Occupancy the robot can change its mind about (map v3).

    v1 and v2 write a verdict. The first frame that calls a cell OBSTACLE
    makes it an obstacle for the rest of the run, and nothing erases it. Over
    one episode that is nearly harmless; over a whole day it is the wrong
    invariant twice over:

      * a wall painted while the pose was drifting stays after the pose
        recovers, so the map is the scar tissue of every bad moment rather
        than the current best estimate;
      * there is nowhere for a correction to land. ORB-SLAM3 revises its own
        past on a loop closure; a verdict map cannot receive that.

    Here a frame leaves EVIDENCE: cells a ray passes through lose L_FREE,
    its endpoint gains L_OCC, the sum is clamped to +-L_CLAMP, and ``grid``
    is the classification of that sum. One clean look still decides an
    unknown cell, so a fresh map looks exactly like v2's; a wall that three
    later passes walk straight through goes back to free. ``weight`` scales
    the step, so evidence gathered while the pose is not trusted (the frames
    after a re-anchor) cannot on its own build a wall — it can only agree
    with one.

    ``grid`` stays a materialised uint8 array, updated in place on the cells
    a frame touched, so every consumer — render, frontier, A*, the spatial
    dump — reads it exactly as it reads a v1/v2 grid.
    """

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.evidence = np.zeros((self.n, self.n), dtype=np.int16)

    def _classify(self, idx: np.ndarray) -> None:
        flat_e = self.evidence.reshape(-1)
        flat_g = self.grid.reshape(-1)
        e = flat_e[idx]
        out = np.full(e.shape, UNKNOWN, dtype=np.uint8)
        out[e >= T_OCC] = OBSTACLE
        out[e <= T_FREE] = FREE
        flat_g[idx] = out

    def integrate(self, pose_wc: np.ndarray, depth_m: np.ndarray, intrinsics: dict,
                  floor_ref_y: float | None = None, weight: float = 1.0) -> None:
        got = self._rays(pose_wc, depth_m, intrinsics, floor_ref_y)
        if got is None:
            return
        free_cells, obst, agent = got
        w = float(max(0.0, min(1.0, weight)))
        if w <= 0.0:
            return
        d_free = max(1, int(round(L_FREE * w)))
        d_occ = max(1, int(round(L_OCC * w)))
        flat_e = self.evidence.reshape(-1)
        # One update per cell per FRAME, not per ray: a cell crossed by two
        # hundred rays of the same frame is one observation, not two hundred.
        is_obst = np.zeros(self.n * self.n, bool)
        is_obst[obst] = True
        free_u = np.unique(free_cells)
        free_u = free_u[~is_obst[free_u]]
        if agent is not None:
            free_u = np.unique(np.append(free_u, agent))
        # Plain fancy indexing, not np.add.at: both index sets are the output
        # of np.unique and disjoint by construction, so there is nothing to
        # accumulate and the unbuffered scatter would be far slower for nothing.
        flat_e[free_u] -= d_free
        flat_e[obst] += d_occ
        touched = np.unique(np.concatenate([free_u, obst]))
        np.clip(flat_e, -L_CLAMP, L_CLAMP, out=flat_e)
        self._classify(touched)

    def mark_obstacle_ahead(self, pose_wc: np.ndarray, dist_m: float = 0.3) -> None:
        """A collision is felt, not seen — it outvotes a few rays but is not
        the last word either."""
        fwd = pose_wc[:3, 2]
        x = float(pose_wc[0, 3]) + float(fwd[0]) * dist_m
        z = float(pose_wc[2, 3]) + float(fwd[2]) * dist_m
        zi, xi = self.world_to_cell(x, z)
        if not self.in_bounds(zi, xi):
            return
        idx = np.array([zi * self.n + xi])
        flat_e = self.evidence.reshape(-1)
        flat_e[idx] = np.clip(flat_e[idx] + L_BUMP, -L_CLAMP, L_CLAMP)
        self._classify(idx)

    def decay(self, gamma: float = 0.5) -> dict:
        """Forget proportionally — what to do when the frame the map was
        painted in is known to have moved (a loop closure, a re-anchor that
        the tracker later contradicted). Everything keeps its sign and its
        ordering; the map simply becomes re-earnable. Nothing is deleted, so
        a corridor observed a hundred times survives a decay that erases a
        wall seen once.
        """
        before = int((self.grid != UNKNOWN).sum())
        self.evidence = (self.evidence.astype(np.float32) * float(gamma)).astype(np.int16)
        self._classify(np.arange(self.n * self.n))
        return {"decayed": float(gamma), "known_before": before,
                "known_after": int((self.grid != UNKNOWN).sum())}

def _bresenham(z0: int, x0: int, z1: int, x1: int) -> list:
    """Integer line from (z0,x0) to (z1,x1), endpoints inclusive."""
    dz, dx = abs(z1 - z0), abs(x1 - x0)
    sz = 1 if z1 >= z0 else -1
    sx = 1 if x1 >= x0 else -1
    cells = []
    if dx >= dz:
        err = dx // 2
        z = z0
        for x in range(x0, x1 + sx, sx):
            cells.append((z, x))
            err -= dz
            if err < 0:
                z += sz
                err += dx
    else:
        err = dz // 2
        x = x0
        for z in range(z0, z1 + sz, sz):
            cells.append((z, x))
            err -= dx
            if err < 0:
                x += sx
                err += dz
    return cells



def _bresenham_many(z0: int, x0: int, ez: np.ndarray, ex: np.ndarray) -> tuple:
    """Every ray from (z0, x0) to each (ez[i], ex[i]), rasterised at once.

    Emits exactly the cells ``_bresenham`` would, in the same per-ray order.

    Derivation for the |dx| >= |dz| branch. ``err`` starts at ``dx//2`` and per
    step does ``err -= dz; if err < 0: z += sz; err += dx``, so after t steps
    and k increments the invariant ``0 <= dx//2 - dz*t + dx*k < dx`` holds,
    giving the unique ``k(t) = ceil((dz*t - dx//2) / dx)`` — integer-only as
    ``-((dx//2 - dz*t) // dx)``. The minor/major axes swap in the other branch.
    """
    ez = np.asarray(ez, np.int64)
    ex = np.asarray(ex, np.int64)
    dz = np.abs(ez - z0)
    dx = np.abs(ex - x0)
    sz = np.where(ez >= z0, 1, -1)
    sx = np.where(ex >= x0, 1, -1)
    xmajor = dx >= dz
    n = np.where(xmajor, dx, dz)                  # steps; cells per ray = n+1
    total = int((n + 1).sum())
    if total == 0:
        e = np.empty(0, np.int64)
        return e, e, e
    ray = np.repeat(np.arange(len(ez)), n + 1)
    starts = np.concatenate(([0], np.cumsum(n + 1)[:-1]))
    t = np.arange(total) - np.repeat(starts, n + 1)
    major_is_x = xmajor[ray]
    maj = np.where(major_is_x, dx[ray], dz[ray])
    mnr = np.where(major_is_x, dz[ray], dx[ray])
    safe = np.maximum(maj, 1)                     # n == 0 rays: k stays 0
    k = -((safe // 2 - mnr * t) // safe)
    zz = np.where(major_is_x, z0 + sz[ray] * k, z0 + sz[ray] * t)
    xx = np.where(major_is_x, x0 + sx[ray] * t, x0 + sx[ray] * k)
    return zz, xx, ray

def render_map(grid: np.ndarray, fmask: np.ndarray | None,
               agent_cell: tuple | None,
               plan: list | None) -> np.ndarray:
    img = np.full((*grid.shape, 3), 128, dtype=np.uint8)  # unknown: gray
    img[grid == FREE] = (255, 255, 255)
    img[grid == OBSTACLE] = (0, 0, 0)
    if fmask is not None:
        img[fmask] = (220, 40, 40)
    if plan:
        for zi, xi in plan:
            img[zi, xi] = (40, 180, 60)
    if agent_cell is not None:
        zi, xi = agent_cell
        img[max(0, zi - 2):zi + 3, max(0, xi - 2):xi + 3] = (40, 80, 255)
    return img[::-1]  # flip so +z points up


def save_map_png(path: str, grid: np.ndarray, fmask: np.ndarray | None,
                 agent_cell: tuple | None,
                 plan: list | None) -> None:
    from PIL import Image

    Image.fromarray(render_map(grid, fmask, agent_cell, plan)).save(path)
