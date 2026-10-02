"""Which floor am I on: a height to match, or a traversal to count?

`floor_policy="height"` compares the settled camera height against the
heights of the floors already registered. That reads an INTEGRAL, and
vertical drift integrates: given a long enough run, slow drift alone mints
storeys that do not exist.

`floor_policy="stairs"` moves a level index by +-1 when the settled height
changes by a lot IN A SHORT TIME, and never otherwise. That reads a
DERIVATIVE, which drift does not accumulate into.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mapping_floor", REPO / "exp_workspace/lhnav_es/mcp/_mapping.py")
mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapping)

STOREY = 2.6            # typical metres between floors
SETTLE = 12             # feeds of standing still
WIN = 60                # the detector's window, shrunk so a unit test is a unit test
                        # (the shipped default is 700 feeds — see _update_level)


def build(policy: str, **kw):
    return mapping.LayeredOccupancyMap(cell_size=0.10, map_size_m=48.0, camera_height=1.25,
                                       floor_policy=policy, window_feeds=WIN, **kw)


def feed(occ, ys):
    for y in ys:
        occ._update_floor(float(y))


def still(y, n=SETTLE):
    return [y] * n


def ramp(y0, y1, n):
    return list(np.linspace(y0, y1, n))


# ── drift: storeys minted by integration ──


def drift_trace():
    """Several metres of slow vertical drift: about 0.18 mm per feed.

    A staircase in the same units is on the order of 5 mm per feed, over
    twenty times faster. The synthetic drift has to be slow or the test is
    not testing the thing that separates them.
    """
    per_feed = 6.89 / 39360
    return list(-per_feed * np.arange(39360))    # y is DOWN, so this creeps upward


def test_height_policy_turns_drift_into_storeys() -> None:
    occ = mapping.LayeredOccupancyMap(cell_size=0.10, map_size_m=48.0,
                                      camera_height=1.25, floor_policy="height")
    feed(occ, drift_trace())
    assert len(occ.layers) >= 5, (
        f"expected drift to mint storeys, got {len(occ.layers)}")


def test_stairs_policy_ignores_drift_entirely() -> None:
    occ = build("stairs")
    feed(occ, drift_trace())
    assert len(occ.layers) == 1, f"drift minted {len(occ.layers)} levels"
    assert occ.level == 0 and occ.transitions == []


# ── a real staircase ──


def test_a_staircase_moves_one_level_up_and_back() -> None:
    occ = build("stairs")
    feed(occ, still(0.0, WIN) + ramp(0.0, -STOREY, WIN) + still(-STOREY, WIN))
    assert occ.level == 1, f"climbing should reach level 1, got {occ.level}"
    assert len(occ.layers) == 2
    feed(occ, ramp(-STOREY, 0.0, WIN) + still(0.0, WIN))
    assert occ.level == 0, "coming back down returns to the level already there"
    assert len(occ.layers) == 2, "and does not mint a third"


def test_two_flights_are_two_levels() -> None:
    occ = build("stairs")
    feed(occ, still(0.0, WIN))
    for k in (1, 2):
        feed(occ, ramp(-STOREY * (k - 1), -STOREY * k, WIN) + still(-STOREY * k, WIN))
    assert occ.level == 2 and len(occ.layers) == 3


def test_a_slow_climb_is_not_counted() -> None:
    """A metre and a half that takes an hour is drift, not a staircase: the
    transit has to be quick as well as large."""
    occ = build("stairs")
    feed(occ, still(0.0, WIN))
    for i in range(1, 60):
        feed(occ, still(-0.02 * i, n=WIN))
    assert occ.level == 0, f"a slow creep reached level {occ.level}"


# ── being carried: the half a traversal counter cannot see ──


def test_a_teleport_is_not_a_traversal_and_set_level_is_the_door() -> None:
    """A wake-up relocation can move the robot between storeys without any
    traversal. The counter must not guess at it; the wake-up hands the level
    in from the place it recognised."""
    occ = build("stairs")
    feed(occ, still(0.0, WIN))
    occ._y_hist.clear(); occ._y_win.clear()         # what the wake-up freeze does
    layer = occ.set_level(2, y_hint=-2 * STOREY)
    assert occ.level == 2 and occ.current_floor == layer and len(occ.layers) == 2
    feed(occ, still(-2 * STOREY, WIN))
    assert occ.level == 2, "and it stays there"
    back = occ.set_level(0)
    assert back == 0 and len(occ.layers) == 2, "a level already seen is reused"


def test_levels_can_go_below_the_start() -> None:
    occ = build("stairs")
    feed(occ, still(0.0, WIN) + ramp(0.0, STOREY, WIN) + still(STOREY, WIN))
    assert occ.level == -1, "downstairs from the start is level -1"


def test_grid_follows_the_level() -> None:
    occ = build("stairs")
    feed(occ, still(0.0, WIN))
    g0 = occ.grid
    feed(occ, ramp(0.0, -STOREY, WIN) + still(-STOREY, WIN))
    assert occ.grid is not g0, "a new level must hand out a different grid"


if __name__ == "__main__":
    ok, bad = 0, []
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
        except Exception as exc:                    # noqa: BLE001
            bad.append(name); print(f"  FAIL {name}: {exc}")
        else:
            ok += 1; print(f"  PASS {name}")
    print(f"{ok} passed, {len(bad)} failed")
    raise SystemExit(1 if bad else 0)
