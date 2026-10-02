"""The atlas: one grid per ORB-SLAM3 map, and a merge that takes the glue back.

When tracking dies and relocalisation fails, ORB-SLAM3 starts a NEW map rather
than corrupting the one it had, and merges the two later if place recognition
reconnects them. Gluing the new map on at its first tracked frame
(``anchor = last_pose @ inv(raw)``) and writing everything into one grid
would commit permanently to the frame of a pose that is in doubt, after which
no correction could land.

These tests pin the atlas behaviour, with no simulator and no container:

  1. a new map gets its OWN grid — the good map is not written into by a guess;
  2. the agent still sees the whole house (``view`` composites the live epochs),
     because a robot that goes blind after a tracking glitch is worse, not
     safer;
  3. when the atlas rewrites the past, the affected epoch's cells MOVE by the
     transform ORB-SLAM3 implies, and the epoch stops being provisional;
  4. a fold with no correction is exact — compositing must not smear the map;
  5. when the atlas is discarded (new episode), stragglers collapse rather than
     stranding what they mapped;
  6. every keyframe is claimed by exactly one epoch, so a merge is attributed
     to the map that actually moved.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sen = importlib.import_module("exp_workspace.lhnav_es.mcp._slam_env")
mapping = importlib.import_module("exp_workspace.lhnav_es.mcp._mapping")

OBSTACLE, FREE, UNKNOWN = mapping.OBSTACLE, mapping.FREE, mapping.UNKNOWN


class FakeSlam:
    """Keyframes with poses we can rewrite, the way a map merge rewrites them."""

    def __init__(self) -> None:
        self.kf: dict[float, np.ndarray] = {}

    def add(self, t: float, xyz=(0.0, 0.0, 0.0)) -> None:
        m = np.eye(4)
        m[:3, 3] = xyz
        self.kf[round(float(t), 4)] = m

    def shift(self, ts, G) -> None:
        """What a merge does: the same keyframes, reported in a new frame."""
        for t in ts:
            self.kf[round(float(t), 4)] = G @ self.kf[round(float(t), 4)]

    def keyframes(self) -> list:
        return [{"t": t, "matrix": m.tolist()} for t, m in sorted(self.kf.items())]


def make(map_mode: str = "v3", cell: float = 0.10, size: float = 12.0):
    """A SlamEnv with only the atlas wired — no body, no simulator."""
    se = object.__new__(sen.SlamEnv)
    se.env = type("B", (), {"camera_height": 1.25})()
    se.map_mode = map_mode
    se._map_kw = dict(cell_size=cell, map_size_m=size, mapper={}, map_mode=map_mode)
    se.occ = se._new_map()
    se.epochs = [sen.MapEpoch(0, np.eye(4), se.occ, provisional=False, tick=0)]
    se.epoch = 0
    se._anchor = np.eye(4)
    se.last_pose = np.eye(4)
    se.tick = 0
    se.slam = FakeSlam()
    se._claimed_kf = set()
    se._resync_every = 400
    se._last_resync = -10_000
    se.merges = []
    se._view = None
    se._view_key = None
    se.reanchors = 0
    se.reanchor_log = []
    se._lost_run = 0
    se._conf_hold = 0
    return se


def paint(occ, x: float, z: float, value=OBSTACLE, r: int = 2) -> None:
    """Put a small blob at a world point, on whatever layer is current."""
    layer = occ.layers[occ.current_floor] if hasattr(occ, "layers") else occ
    zi, xi = layer.world_to_cell(x, z)
    layer.grid[zi - r:zi + r + 1, xi - r:xi + r + 1] = value
    if hasattr(layer, "evidence"):
        ev = mapping.L_CLAMP if value == OBSTACLE else -mapping.L_CLAMP
        layer.evidence[zi - r:zi + r + 1, xi - r:xi + r + 1] = ev


def blob_center(occ, value=OBSTACLE):
    layer = occ.layers[occ.current_floor] if hasattr(occ, "layers") else occ
    zi, xi = np.nonzero(layer.grid == value)
    assert zi.size, "nothing painted"
    return layer.cell_to_world(int(round(zi.mean())), int(round(xi.mean())))


def translation(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, yaw: float = 0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    G = np.eye(4)
    G[0, 0], G[0, 2] = c, s
    G[2, 0], G[2, 2] = -s, c
    G[:3, 3] = (dx, dy, dz)
    return G


# ── 1. a new map does not write into the old one ──────────────────────────

def test_new_epoch_gets_its_own_grid():
    se = make()
    paint(se.occ, 2.0, 0.0)
    good = se.occ
    good_cells = int((good.layers[0].grid == OBSTACLE).sum())

    se.last_pose = np.eye(4)
    se._open_epoch(np.eye(4), state=2)

    assert len(se.epochs) == 2
    assert se.occ is not good, "the new map must not be the old map"
    assert se.epochs[1].provisional, "a guessed anchor is provisional"
    assert not se.epochs[0].provisional
    # whatever the new map records, the good one is untouched
    paint(se.occ, -3.0, 1.0)
    assert int((good.layers[0].grid == OBSTACLE).sum()) == good_cells


# ── 2. the agent does not go blind ────────────────────────────────────────

def test_view_composites_live_epochs():
    se = make()
    paint(se.occ, 2.0, 0.0)
    se._open_epoch(np.eye(4), state=2)
    paint(se.occ, -3.0, 1.0)

    vm = se.view()
    assert vm is not se.occ, "more than one live epoch -> a composite"
    layer = vm.layers[vm.current_floor]
    for x, z in ((2.0, 0.0), (-3.0, 1.0)):
        zi, xi = layer.world_to_cell(x, z)
        assert layer.grid[zi, xi] == OBSTACLE, f"({x}, {z}) missing from the view"
    # one epoch: no copy, no cost
    se2 = make()
    assert se2.view() is se2.occ


def test_view_is_cached_until_the_map_moves():
    se = make()
    paint(se.occ, 2.0, 0.0)
    se._open_epoch(np.eye(4), state=2)
    first = se.view()
    assert se.view() is first
    se.tick += 1
    assert se.view() is not first


# ── 3. a merge moves the cells ────────────────────────────────────────────

def test_resync_folds_the_old_map_by_the_transform_orb_implies():
    se = make()
    for t in (0.1, 0.2, 0.3, 0.4):
        se.slam.add(t)
    paint(se.occ, 2.0, 0.0)
    old = se.epochs[0]

    se._open_epoch(np.eye(4), state=2)          # claims 0.1..0.4 for epoch 0
    assert len(old.samples) == 4

    assert se.resync()["merged"] == 0, "nothing has moved yet"

    G = translation(dx=1.0)                      # the past is 1 m further along x
    se.slam.shift([0.1, 0.2, 0.3, 0.4], G)
    out = se.resync(force=True)

    assert out["merged"] == 1
    rec = out["merges"][0]
    assert rec["epoch"] == 0 and rec["into"] == 1
    assert rec["dx"] == pytest.approx(1.0, abs=0.02)
    assert old.merged_into == 1 and not old.provisional
    assert not se.epochs[1].provisional, "the current map is tied to a known one now"

    x, z = blob_center(se.occ)
    assert x == pytest.approx(3.0, abs=0.11), f"blob landed at x={x}, wanted 3.0"
    assert z == pytest.approx(0.0, abs=0.11)


def test_resync_ignores_a_past_that_sits_still():
    se = make()
    for t in (0.1, 0.2, 0.3, 0.4):
        se.slam.add(t)
    se._open_epoch(np.eye(4), state=2)
    se.slam.shift([0.1], translation(dx=5.0))    # one outlier, not a merge
    out = se.resync(force=True)
    assert out["merged"] == 0
    assert se.epochs[0].merged_into is None


def test_resync_is_rate_limited():
    se = make()
    se.slam.add(0.1)
    se._open_epoch(np.eye(4), state=2)
    se._last_resync = se.tick
    assert se.resync()["reason"] == "too soon"


def test_resync_survives_a_dead_client():
    se = make()
    se.slam.add(0.1)
    se._open_epoch(np.eye(4), state=2)

    class Dead:
        def keyframes(self):
            raise RuntimeError("container gone")

    se.slam = Dead()
    assert "error" in se.resync(force=True)      # reported, never raised


# ── 4. no correction must mean no smearing ────────────────────────────────

def test_identity_fold_is_exact():
    se = make()
    paint(se.occ, 2.0, 0.0)
    before = se.occ.layers[0].grid.copy()
    dst = se._new_map()
    se._fold_into(se.occ, dst, np.eye(4))
    assert np.array_equal(dst.layers[0].grid, before)


# ── 5. a discarded atlas must not strand what it mapped ───────────────────

def test_collapse_keeps_the_cells_of_epochs_that_can_never_merge():
    se = make()
    paint(se.occ, 2.0, 0.0)
    se._open_epoch(np.eye(4), state=2)
    paint(se.occ, -3.0, 1.0)

    assert se._collapse() == 1
    layer = se.occ.layers[se.occ.current_floor]
    for x, z in ((2.0, 0.0), (-3.0, 1.0)):
        zi, xi = layer.world_to_cell(x, z)
        assert layer.grid[zi, xi] == OBSTACLE
    assert se.view() is se.occ, "one live epoch again"


# ── 6. attribution ────────────────────────────────────────────────────────

def test_each_keyframe_is_claimed_once():
    se = make()
    se.slam.add(0.1); se.slam.add(0.2)
    se._open_epoch(np.eye(4), state=2)           # epoch 0 claims both
    se.slam.add(0.3)
    se._open_epoch(np.eye(4), state=2)           # epoch 1 claims only 0.3
    assert [t for t, _ in se.epochs[0].samples] == [0.1, 0.2]
    assert [t for t, _ in se.epochs[1].samples] == [0.3]
    assert se.epochs[2].samples == []


def test_a_new_epoch_opens_on_the_level_it_left():
    se = make()
    se.occ.set_level(2)
    se._open_epoch(np.eye(4), state=2)
    assert se.occ.level == 2, "level numbers must stay comparable across epochs"


# ── 7. starting up is not the same as losing the map ──────────────────────

def test_rebase_moves_epoch_zero_instead_of_opening_a_new_one():
    """A state-1 before the first successful track is SLAM finding its feet."""
    se = make()
    se._fresh_atlas = True
    se.last_pose = None
    lever = translation(dx=0.5, dy=1.2)          # a lever-arm start frame
    se._anchor = lever.copy()
    se.epochs[0].anchor = lever.copy()

    raw = translation(dz=0.3)
    se._rebase_epoch0(raw)

    assert len(se.epochs) == 1, "start-up must not fork the atlas"
    assert np.allclose(se._anchor, lever @ np.linalg.inv(raw))
    assert np.allclose(se.epochs[0].anchor, se._anchor)
    assert not se.epochs[0].provisional
    # the start frame survives: raw = the first tracked pose still maps to it
    assert np.allclose(se._anchor @ raw, lever)


def test_a_dead_map_still_forks_the_atlas():
    se = make()
    se._fresh_atlas = False
    se.last_pose = translation(dx=2.0)
    se._open_epoch(translation(), state=2)
    assert len(se.epochs) == 2 and se.epochs[1].provisional
