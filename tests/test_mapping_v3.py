"""map v3 — evidence occupancy: does it still look like v2, and can it forget?

Three questions, no simulator and no env:

  1. the split of ``integrate`` into ``_rays`` + write left v1/v2 alone;
  2. one clean look classifies a cell exactly as v2 does, so a fresh v3 map
     is not a different map;
  3. a wall painted from a drifted pose is erased by later passes that walk
     through it — the thing v2 structurally cannot do, so a v2 map is worse
     than the pose error behind it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
REL = "exp_workspace/lhnav_es/mcp/_mapping.py"
INTR = {"fx": 128.0, "fy": 128.0, "cx": 128.0, "cy": 128.0, "width": 256, "height": 256}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mapping = _load(REPO / REL, "mapping_v3")


def pose_at(x: float, z: float, yaw: float = 0.0) -> np.ndarray:
    """A level camera at (x, z) looking along yaw, in the mapping frame
    (OpenCV axes: +z forward, +x right, y down)."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    T[:3, 3] = [x, 0.0, z]
    return T


def wall_at(dist_m: float, shape=(64, 64)) -> np.ndarray:
    """A depth frame of a flat wall square-on at ``dist_m``.

    Keep every distance inside [depth_min, depth_max] = [0.3, 4.5] m — a wall
    past the cut-off is not a faint wall, it is no wall at all, and a test
    written against one passes by integrating nothing.
    """
    assert 0.3 <= dist_m <= 4.5, "outside the mapper's depth window"
    return np.full(shape, float(dist_m), dtype=np.float32)


# ── 1. the refactor changed nothing ──


def test_rays_split_is_behaviour_preserving() -> None:
    if subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"HEAD:{REL}"],
                      capture_output=True).returncode:
        pytest.skip("no git history for this file — the refactor it pins predates this repo")
    head = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{REL}"],
                          capture_output=True, text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "before.py"
    tmp.write_text(head)
    before = _load(tmp, "mapping_before")
    rng = np.random.default_rng(11)
    a = before.OccupancyMap(cell_size=0.10, map_size_m=48.0, camera_height=1.25)
    b = mapping.OccupancyMap(cell_size=0.10, map_size_m=48.0, camera_height=1.25)
    for _ in range(20):
        p = pose_at(rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-np.pi, np.pi))
        d = rng.uniform(0.4, 5.5, size=(96, 96)).astype(np.float32)
        d[rng.random((96, 96)) < 0.2] = 0.0
        a.integrate(p, d, INTR)
        b.integrate(p, d, INTR)
    assert int((a.grid != b.grid).sum()) == 0


# ── 2. one clean look is the same map ──


def test_first_look_matches_v2() -> None:
    v2 = mapping.OccupancyMap(cell_size=0.10, map_size_m=48.0, camera_height=1.25)
    v3 = mapping.EvidenceMap(cell_size=0.10, map_size_m=48.0, camera_height=1.25)
    p, d = pose_at(0.0, 0.0), wall_at(3.0)
    v2.integrate(p, d, INTR)
    v3.integrate(p, d, INTR)
    assert int((v2.grid != v3.grid).sum()) == 0
    assert v3.free_area_m2() == pytest.approx(v2.free_area_m2())


# ── 3. the one thing v2 cannot do ──


def _ghost_then_walk_through(mapper) -> tuple[int, int]:
    """Paint a wall that is not there, then walk through where it was.

    A pose 1.5 m short of the truth paints a wall across z ~ +1.5. The robot
    then does what a robot does: walks the corridor from z = -1 to z = +3,
    looking ahead, so every later frame's rays cross the ghost. Returns the
    ghost cells before and after.
    """
    mapper.integrate(pose_at(0.0, -1.5), wall_at(3.0), INTR)     # the ghost
    band = slice(mapper.origin + 10, mapper.origin + 21)          # z in [1.0, 2.1) m
    before = int((mapper.grid[band, :] == mapping.OBSTACLE).sum())
    # The real wall is at 4 m ahead of each pose, i.e. z >= 3 — always past
    # the ghost band, so anything left in that band is the ghost and not a
    # new endpoint landing in the same row.
    for i in range(9):
        mapper.integrate(pose_at(0.0, -1.0 + 0.25 * i), wall_at(4.0), INTR)
    after = int((mapper.grid[band, :] == mapping.OBSTACLE).sum())
    seen = slice(mapper.origin - 8, mapper.origin + 9)            # x in [-0.8, 0.9)
    centre = int((mapper.grid[band, seen] == mapping.OBSTACLE).sum())
    return before, after, centre


def test_v2_keeps_a_ghost_wall_forever() -> None:
    before, after, centre = _ghost_then_walk_through(
        mapping.OccupancyMap(cell_size=0.10, map_size_m=48.0, camera_height=1.25))
    assert before > 0
    assert after == before, (
        f"v2 is a verdict map: walking through a wall cannot erase it "
        f"({before} -> {after})")
    assert centre > 0, "not even the cells it drove straight through"


def test_v3_walks_the_ghost_wall_out() -> None:
    before, after, centre = _ghost_then_walk_through(
        mapping.EvidenceMap(cell_size=0.10, map_size_m=48.0, camera_height=1.25))
    assert before > 0
    # Everything it looked at again is gone; the survivors are the arc's outer
    # tips, which lie outside every later 90 deg fan. An evidence map erases
    # what it re-observes, not what it never sees twice.
    assert centre == 0, f"{centre} ghost cells left where the robot drove through"
    assert after <= before // 3, f"ghost barely reduced ({before} -> {after})"


def test_the_true_wall_survives_the_same_passes() -> None:
    m = mapping.EvidenceMap(cell_size=0.10, map_size_m=48.0, camera_height=1.25)
    for _ in range(10):
        m.integrate(pose_at(0.0, 0.0), wall_at(4.0), INTR)
    band = slice(m.origin + 35, m.origin + 46)              # z in [3.5, 4.6) m
    assert int((m.grid[band, :] == mapping.OBSTACLE).sum()) > 0


# ── weight and decay ──


def test_low_confidence_evidence_cannot_build_a_wall_alone() -> None:
    m = mapping.EvidenceMap(cell_size=0.10, map_size_m=48.0, camera_height=1.25)
    m.integrate(pose_at(0.0, 0.0), wall_at(3.0), INTR, weight=0.25)
    assert int((m.grid == mapping.OBSTACLE).sum()) == 0, "a quarter-trusted look asserts nothing"
    for _ in range(3):
        m.integrate(pose_at(0.0, 0.0), wall_at(3.0), INTR, weight=0.25)
    assert int((m.grid == mapping.OBSTACLE).sum()) > 0, "but four of them agree"


def test_decay_keeps_what_was_seen_often() -> None:
    m = mapping.EvidenceMap(cell_size=0.10, map_size_m=48.0, camera_height=1.25)
    for _ in range(12):
        m.integrate(pose_at(0.0, 0.0), wall_at(4.0), INTR)
    m.integrate(pose_at(0.0, 0.0, np.pi / 2), wall_at(2.0), INTR)   # seen once
    solid = int((m.grid == mapping.OBSTACLE).sum())
    m.decay(0.25)
    left = int((m.grid == mapping.OBSTACLE).sum())
    assert 0 < left < solid, f"decay should thin the map, not empty it ({left} of {solid})"


def test_layered_v3_registers_floors_and_forgets() -> None:
    lay = mapping.LayeredOccupancyMap(cell_size=0.10, map_size_m=48.0,
                                      camera_height=1.25, layer_cls=mapping.EvidenceMap)
    for _ in range(12):
        lay.integrate(pose_at(0.0, 0.0), wall_at(3.5), INTR)
    assert isinstance(lay.layers[0], mapping.EvidenceMap)
    assert lay.free_area_m2() > 0
    out = lay.decay(0.0)
    assert out["known_after"] == 0 and out["layers"] == 1


if __name__ == "__main__":
    ok, bad = 0, []
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001 — a report, not a crash
            bad.append((name, exc))
            print(f"  FAIL {name}: {exc}")
        else:
            ok += 1
            print(f"  PASS {name}")
    print(f"{ok} passed, {len(bad)} failed")
    sys.exit(1 if bad else 0)
