"""The occupancy rules and the env's readings for the recovery controller, without a simulator.

Two things the paper states about the map — depth is read on the closed band
[0.3, 4.5] m, and an obstacle is never overwritten by free — and two readings
the controller's rules take from every hop: newly explored area (ΔA), carried
on every motion reply so it is measured without a ``get_map`` call, and a
floor mismatch (the body is off the floor layer the map plans on), which
escalates a recovery to ``floor`` and lifts the progress guard.
"""

from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

mapping = importlib.import_module("exp_workspace.lhnav_es.mcp._mapping")
sen = importlib.import_module("exp_workspace.lhnav_es.mcp._slam_env")
stack = importlib.import_module("exp_workspace.lhnav_es.mcp.stack")
envmod = importlib.import_module("exp_workspace.lhnav_es.mcp.env")

OBSTACLE, FREE, UNKNOWN = mapping.OBSTACLE, mapping.FREE, mapping.UNKNOWN
INTR = {"fx": 128.0, "fy": 128.0, "cx": 128.0, "cy": 128.0, "width": 256, "height": 256}


def wall(dist_m: float, shape=(64, 64)) -> np.ndarray:
    return np.full(shape, float(dist_m), dtype=np.float32)


def at(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


# ── the occupancy rules (v1 / v2 layers) ─────────────────────────────────


def test_depth_is_read_on_the_closed_band() -> None:
    """A wall at exactly depth_max is a wall; so is one at exactly depth_min.
    (4.5 and 0.5 are exact in float32, so the bound itself is what is tested.)"""
    far = mapping.OccupancyMap(cell_size=0.10, map_size_m=24.0, camera_height=1.25)
    far.integrate(at(), wall(4.5), INTR)
    assert (far.grid == OBSTACLE).any(), "a wall at depth_max is in the band"
    beyond = mapping.OccupancyMap(cell_size=0.10, map_size_m=24.0, camera_height=1.25)
    beyond.integrate(at(), wall(4.75), INTR)
    assert not (beyond.grid != UNKNOWN).any(), "past depth_max nothing is written"
    near = mapping.OccupancyMap(cell_size=0.10, map_size_m=24.0, camera_height=1.25, depth_min=0.5)
    near.integrate(at(), wall(0.5), INTR)
    assert (near.grid == OBSTACLE).any(), "a wall at depth_min is in the band"


def test_an_obstacle_in_the_bodys_own_cell_is_not_overwritten_by_free() -> None:
    for occ in (mapping.OccupancyMap(cell_size=0.10, map_size_m=24.0, camera_height=1.25),
                mapping.LayeredOccupancyMap(cell_size=0.10, map_size_m=24.0, camera_height=1.25)):
        layer = occ.layers[0] if hasattr(occ, "layers") else occ
        zi, xi = layer.world_to_cell(0.0, 0.0)
        layer.grid[zi, xi] = OBSTACLE                    # e.g. a collision stamped there
        occ.integrate(at(), wall(3.0), INTR)
        assert layer.grid[zi, xi] == OBSTACLE, type(occ).__name__
        assert (layer.grid == FREE).any(), "the rays still carved free space"


def test_a_free_body_cell_is_still_marked_free() -> None:
    occ = mapping.OccupancyMap(cell_size=0.10, map_size_m=24.0, camera_height=1.25)
    occ.integrate(at(), wall(3.0), INTR)
    zi, xi = occ.world_to_cell(0.0, 0.0)
    assert occ.grid[zi, xi] == FREE


# ── the readings: floor mismatch and the whole map's explored area ───────


def make(map_mode: str = "v2", policy: str = "height"):
    """A SlamEnv with only the map wired — no body, no tracker."""
    se = object.__new__(sen.SlamEnv)
    se.env = SimpleNamespace(camera_height=1.25)
    se.map_mode = map_mode
    se._map_kw = dict(cell_size=0.10, map_size_m=12.0,
                      mapper={"floor_policy": policy} if map_mode != "v1" else {}, map_mode=map_mode)
    se.occ = se._new_map()
    se.epochs = [sen.MapEpoch(0, np.eye(4), se.occ, provisional=False, tick=0)]
    se.epoch = 0
    se.last_pose = at()
    se.tick = 0
    se.merges = []
    se._view = None
    se._view_key = None
    se.status = "ok"
    return se


def test_floor_mismatch_is_the_body_off_the_planning_layer() -> None:
    se = make("v2")
    assert se.floor_mismatch() is False
    se.last_pose = at(y=-1.6)                      # 1.6 m up a staircase (y points down)
    assert se.floor_mismatch() is True, "routes are still planned on the floor below"
    for _ in range(10):                            # it settles: a new floor layer opens there
        se.occ._update_floor(-2.6)
    se.last_pose = at(y=-2.6)
    assert se.occ.current_floor == 1 and se.floor_mismatch() is False
    assert make("v1").floor_mismatch() is False, "one layer: nothing to mismatch"
    stairs = make("v2", policy="stairs")
    stairs.last_pose = at(y=-1.6)
    assert stairs.floor_mismatch() is False, "the stairs policy names its level by traversal"


def test_body_facts_measure_the_whole_map() -> None:
    se = make("v2")
    se.occ.layers[0].grid[:10, :10] = FREE                         # 100 cells on floor 1
    se.occ._update_floor(0.0)
    for _ in range(10):
        se.occ._update_floor(-2.6)
    se.occ.layers[1].grid[:5, :10] = OBSTACLE                      # 50 cells on floor 2
    se.last_pose = at(y=-2.6)
    facts = se.body_facts()
    assert facts == {"explored_m2": 1.5, "current_floor": 1, "floor_mismatch": False}


def test_every_step_carries_the_readings_on_the_instrumented_arm_only() -> None:
    se = make("v2")
    se.reanchors = 0
    se.env = SimpleNamespace(camera_height=1.25, collisions=0)
    se.occ.layers[0].grid[:4, :5] = FREE
    micro = object.__new__(stack.SlamMicroStep)
    micro.slam, micro.instruments = se, 2
    facts = micro._slam_facts()
    assert facts["explored_m2"] == 0.2 and facts["floor_mismatch"] is False and facts["slam"] == "ok"
    micro.instruments = 0
    assert "explored_m2" not in micro._slam_facts(), "a bare run's map is drawn from the simulator's pose"


def _manager(micro, info: dict):
    """The env's manager over a stub stack: one step's info, one observation."""
    m = object.__new__(envmod.SlamEsEnvManager)
    m._lock = threading.RLock()
    m._line = "goat"
    m._micro = micro
    m._episode_done = False
    m._step_count = 0
    m._info = {}
    m._last_action = None
    m._current_obs = {"rgb": np.zeros((2, 2, 3), np.uint8), "depth": None}
    m._ensure_placed_unlocked = lambda: None
    m._step_budget_unlocked = lambda: 500
    m._metrics_unlocked = lambda *a: {}
    episode = SimpleNamespace(episode_id="e1", instruction="walk to the sofa", goal=SimpleNamespace(kind="none"))
    base = SimpleNamespace(actions=(0, 1, 2, 3), episode=episode, current_goal=lambda: None,
                           world=SimpleNamespace(pose=lambda: ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])))
    m._env = SimpleNamespace(unwrapped=base, step=lambda a: (m._current_obs, 0.0, False, False, dict(info)))
    return m


def test_the_step_and_observe_replies_carry_explored_area_and_the_mismatch() -> None:
    se = make("v2")
    se.occ.layers[0].grid[:4, :5] = FREE
    se.last_pose = at(y=-1.6)
    se.env = SimpleNamespace(camera_height=1.25, collisions=0, intrinsics=lambda: INTR)
    micro = object.__new__(stack.SlamMicroStep)
    micro.slam, micro.instruments = se, 2
    m = _manager(micro, {"slam": "ok", "tick": 5, **micro.body_facts()})
    info = m.step(1)["info"]
    assert info["explored_m2"] == 0.2 and info["floor_mismatch"] is True and info["current_floor"] == 0
    seen = m.observe()["info"]
    assert seen == {"explored_m2": 0.2, "current_floor": 0, "floor_mismatch": True, "slam": "ok"}
    micro.instruments = 0
    assert m.observe()["info"] == {}


def test_the_map_reply_names_the_planning_layer_for_the_place_graph() -> None:
    from core.harnesses.navh.env.bench import map_floor

    se = make("v2")
    se.est_track, se.ep_track0, se.task_track0, se.markers, se.map_window = [], 0, 0, [], None
    se.get_pose = lambda: {"x": 0.0, "z": 0.0, "yaw_deg": 0.0}
    m = _manager(None, {})
    m._slam_cfg = {}
    out = m._get_map_v2_unlocked(se)
    assert out["current_floor"] == 0 and out["floor_mismatch"] is False
    assert map_floor({"info": out}) == 0, "the orchestrator reads the layer a marker was dropped on"


def test_a_step_cap_truncation_leaves_the_look_around_where_the_body_stopped() -> None:
    """A task its step budget ended is certified with the four views at its
    stopping point, taken after the env has truncated the episode; a STOP
    still ends looking."""
    se = SimpleNamespace(look_around=lambda: {"views": ["f", "r", "b", "l"], "pitch_deg": 0.0})
    m = _manager(SimpleNamespace(slam=se), {})
    m._placed, m._instruments = True, 2
    base = m._env.unwrapped
    m._env = SimpleNamespace(unwrapped=base, step=lambda a: (m._current_obs, 0.0, False, True, {}))
    assert m.step(1)["truncated"] is True and m._episode_done
    assert m.look_around()["views"] == ["f", "r", "b", "l"]
    m._episode_done = False
    m._env = SimpleNamespace(unwrapped=base,
                             step=lambda a: (m._current_obs, 0.0, True, False, {"stop_called": True}))
    assert m.step(0)["terminated"] is True
    assert m.look_around() == {"error": "episode already done"}
