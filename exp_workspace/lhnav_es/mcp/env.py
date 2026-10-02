"""LhNavEsEnv — the lhnav_es arm's world: the SLAM stack on EmbodiedScore-envs
plus what a lifelong / long-horizon chain needs on top — GOAT's sub-goal cursor
on the step reply, a ``goal_spec`` verb (the goal in force, the image goal
rendered), ``reset_spatial`` (the episodic setting: SLAM session, map and
markers rebuilt in place), IVLN-CE tours as one unit, and checkpoints.

    env:
      _target_: exp_workspace.lhnav_es.mcp.env.LhNavEsEnv
      python: habitat
      port: 9275
      env_fields: [instruments]        # the arm switch, pushed with configure()

The environment half — episodes, bodies, task semantics, metrics, budgets
— is EmbodiedScore-envs (``stack.build``: the STANDARD body at micro
quanta under the line's own metric wrapper); the instrument half is the
SLAM middleware (``_slam_env`` / ``_mapping`` / ``_frontier`` / ``_plan`` /
``_map_render`` / ``_slam`` — ORB-SLAM3 in its container).

Verbs: ``configure(line, split, instruments)`` · ``place(index)`` (the task
facts, per family: instruction / question + choices / object_category) ·
``observe`` · ``step`` · ``evaluate(pred_letter)`` · ``stop_truth`` (what a
STOP here would score, nothing issued) — plus the six read-only
instruments ``get_pose`` / ``get_map`` / ``get_trajectory`` / ``look_around``
/ ``mark`` / ``preview_path``, refused unless ``instruments >= 1`` (0 bare,
1 SLAM + map v1, 2 SLAM + map v2). Body: STANDARD on every line. Episode
ends: STOP or the coarse-step cap on the nav lines; on the question lines the
cap only ends MOVEMENT and the answer rides ``evaluate(pred_letter)``.
Episode order: file order, except the lines in ``_SHUFFLED_LINES``, which
keep habitat's seed-100 shuffle.

``SlamEsEnvManager`` holds the one live stack; ``LhNavEsEnv`` is its verb
surface for ``core.envserver``.
"""

from __future__ import annotations

import base64
import concurrent.futures
import logging
import math
import os
import pathlib
import random
import threading
from pathlib import Path
from typing import Any

import numpy as np

from core.envserver import ndarray_wire, png_b64

from . import stack
from ._frontier import frontier_mask
from ._map_render import render_annotated_map, render_annotated_map_v2
from ._plan import astar_free, nearest_free, path_length_m
from ._slam_env import SlamEnv

log = logging.getLogger("lhnav_es.mcp")

HABITAT_SHUFFLE_SEED = 100
_SHUFFLED_LINES = frozenset(
    {"objectnav-hm3d-v1", "objectnav-hm3d-v2", "objectnav-mp3d-v1", "ovon", "goat"}
)
_DEFAULT_LINE = os.environ.get("SLAMES_LINE", "vlnce-r2r")
_DEFAULT_SPLIT = os.environ.get("SLAMES_SPLIT", "")
_DEFAULT_GPU_ID = int(os.environ.get("SLAMES_GPU_ID", "0"))
_ACTION_STOP = 0
_ACTION_SUBTASK_STOP = 6
_ACTION_NAMES = ("STOP", "FORWARD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN", "SUBTASK_STOP")
_INSTRUMENT_LEVELS = (0, 1, 2)
_LETTERS = "ABCD"

# Pose/map frame facts (verified against habitat): world anchored at
# the episode start pose, up on the map = +z = starting heading, right = +x,
# yaw_deg increases turning RIGHT, 0 = starting heading.
_POSE_CONVENTION = (
    "x right / z forward of your START pose (metres); "
    "yaw_deg: 0 = your starting heading, increases turning "
    "right; on the map, up = +z (your starting heading)"
)


def _repo_root() -> str:
    # __file__ lives at exp_workspace/lhnav_es/mcp/env.py — three parents.
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def _data_roots() -> tuple[str, str]:
    root = _repo_root()
    return (
        os.environ.get("EMBODIEDSCORE_DATA_ROOT")
        or os.path.join(root, "data", "embodiedscore", "datasets"),
        os.environ.get("EMBODIEDSCORE_SCENE_ROOT")
        or os.path.join(root, "data", "embodiedscore", "scenes"),
    )


def _legacy_scene_id(scene_file: str) -> str:
    _, scene_root = _data_roots()
    scene_root = os.path.abspath(scene_root)
    f = os.path.abspath(scene_file)
    rel = os.path.relpath(f, scene_root) if f.startswith(scene_root) else os.path.basename(f)
    return os.path.join("data", "scene_datasets", rel)


def habitat_shuffle_order(n: int, seed: int = HABITAT_SHUFFLE_SEED) -> list[int]:
    order = list(range(n))
    rng = random.Random()
    rng.seed(seed)
    rng.shuffle(order)
    return order


def _disk_splits(line: str, data_root: str) -> list[str]:
    """Splits present on disk beyond the package's declared ones."""
    root = Path(data_root)
    found: list[str] = []
    if line in ("vlnce-r2r", "vlnce-rxr"):
        base = (
            root
            / "vlnce"
            / ("R2R_VLNCE_v1-3_preprocessed" if line == "vlnce-r2r" else "RxR_VLNCE_v0")
        )
        suffix = ".json.gz" if line == "vlnce-r2r" else "_guide.json.gz"
        if base.is_dir():
            found = sorted(
                d.name for d in base.iterdir() if d.is_dir() and (d / f"{d.name}{suffix}").is_file()
            )
    elif line.startswith("objectnav-"):
        base = root / "objectnav" / line[len("objectnav-") :].replace("-", "/")
        if base.is_dir():
            found = sorted(
                d.name for d in base.iterdir() if d.is_dir() and (d / f"{d.name}.json.gz").is_file()
            )
    elif line in ("hmeqa", "mthm3d"):
        base = root / ("hmeqa" if line == "hmeqa" else "mt_hm3d")
        found = sorted(
            p.name[len("questions_") : -len(".csv")] for p in base.glob("questions_*.csv")
        )
    elif line == "express":
        found = sorted(
            p.name[len("express-bench_") : -len(".json")]
            for p in (root / "express_bench").glob("express-bench_*.json")
        )
    return found


def _default_split(splits: list[str]) -> str:
    for name in ("val_unseen", "val", "val_seen"):
        if name in splits:
            return name
    return splits[0]


def _finite_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite_or_none(value)
    return value


def _goal_dict(goal: Any) -> dict[str, Any]:
    if goal is None:
        return {"kind": "none"}
    kind = getattr(goal, "kind", "")
    if kind == "point":
        return {"kind": kind, "position": list(goal.position), "radius": float(goal.radius)}
    if kind == "object":
        return {"kind": kind, "category": goal.category, "n_instances": len(goal.instances)}
    if kind == "image":
        return {"kind": kind, "category": goal.instance.category}
    if kind == "description":
        return {"kind": kind, "category": goal.instance.category, "description": goal.description}
    if kind == "question":
        return {
            "kind": kind,
            "text": goal.text,
            "answer": goal.answer,
            "choices": list(goal.choices) if goal.choices else None,
        }
    if kind == "sequence":
        return {"kind": kind, "n_goals": len(goal.goals), "kinds": [g.kind for g in goal.goals]}
    return {"kind": kind}


def _task_text(episode: Any, goal: Any) -> str:
    """The task statement the dataset gives — never a prompt of our own."""
    if episode.instruction:
        return str(episode.instruction)
    kind = getattr(goal, "kind", "")
    if kind == "question":
        return str(episode.info.get("question_formatted") or goal.text)
    if kind == "object":
        return str(goal.category)
    if kind == "description":
        return str(goal.description)
    if kind == "image":
        return str(goal.instance.category)
    return ""


def score_answer(pred_letter: str, gt_letter: str) -> dict:
    """pred == gt -> success 1.0; an empty / invalid
    prediction is the benchmark's own zero (an episode that never answered)."""
    pred = str(pred_letter or "").strip().strip("\"'.()").upper()
    gt = str(gt_letter or "").strip().upper()
    answered = len(pred) == 1 and pred in _LETTERS
    return {
        "success": float(answered and pred == gt),
        "answered": float(answered),
        "pred_letter": pred if answered else "",
        "gt_letter": gt,
    }


# ══════════════════════════════════════════════════════════════════════
# SlamEsEnvManager — singleton runtime
# ══════════════════════════════════════════════════════════════════════


class SlamEsEnvManager:
    """One SLAM stack at a time (line, split, instruments).

    All public methods are blocking — call via
    ``asyncio.get_running_loop().run_in_executor(mgr.executor, fn)``.
    The stack (episodes) is built lazily on the first call that needs it;
    the simulator opens on the first placement and persists across
    episodes; the SLAM session is per episode (``SlamMicroStep.reset``).
    """

    _instance: SlamEsEnvManager | None = None

    def __init__(self) -> None:
        self._env: Any = None
        self._micro: Any = None  # the SlamMicroStep wrapper inside the stack
        self._map_dir: Any = None   # set_map_capture: where map renders are kept
        self._map_every = 1
        self._map_shots = 0
        self._benchmark: Any = None
        self._order: list[int] = []
        self._current_obs: dict | None = None
        self._info: dict = {}
        self._placed = False
        self._tour: dict | None = None   # the tour position last placed (IVLN-CE)
        self._transit: dict | None = None  # what the walk into it cost the map
        self._tour_eps: list = []        # the tour in force, as episode indices
        self._tour_pos = 0               # which of them is seated
        self._tour_index = -1            # which tour of the split
        self._tour_rows: list = []       # each finished episode's metrics
        self._episode_done = False
        self._truncated = False          # the episode ended on its step cap, not on a STOP
        self._step_count = 0
        self._last_action: int | None = None
        self._answer: str | None = None
        self._lock = threading.RLock()
        self._line: str | None = None
        self._split: str | None = None
        self._live: tuple[str, str] | None = None
        self._instruments = 0
        self._episode_index = 0
        self._gpu_id = _DEFAULT_GPU_ID
        self._slam_cfg: dict[str, Any] = {}  # the env block's ``slam`` (stack.SLAM_KEYS)
        self._goal_image_cache: dict[tuple[int, int], str] = {}  # (episode, sub-goal) -> png b64
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="slam_es"
        )

    @classmethod
    def get(cls) -> SlamEsEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def initialized(self) -> bool:
        return self._line is not None

    @property
    def line(self) -> str | None:
        return self._line

    @property
    def split(self) -> str | None:
        return self._split

    @property
    def instruments(self) -> int:
        return self._instruments

    @property
    def _base(self) -> Any:
        return self._env.unwrapped

    @property
    def _se(self) -> SlamEnv | None:
        return None if self._micro is None else self._micro.slam

    @property
    def slam_cfg(self) -> dict[str, Any]:
        return dict(self._slam_cfg)

    @property
    def _render(self) -> dict[str, Any]:
        """render_annotated_map(_v2) kwargs from env.slam.render (out_px /
        margin_m / min_window_m / grid_every_m); {} = the renderer's defaults."""
        return dict(self._slam_cfg.get("render") or {})

    @property
    def tour(self) -> dict | None:
        """Where the episode in force sits in its tour (IVLN-CE)."""
        return dict(self._tour) if self._tour else None

    @property
    def transit(self) -> dict | None:
        """What the oracle walk into this episode cost the map, if there was one."""
        return dict(self._transit) if getattr(self, "_transit", None) else None

    @property
    def episodes(self) -> list:
        eps = self._env.unwrapped.episodes
        return [eps[i] for i in self._order]

    # ── Lifecycle ──

    def initialize(
        self,
        line: str = _DEFAULT_LINE,
        split: str = _DEFAULT_SPLIT,
        instruments: int = 0,
        gpu_id: int = _DEFAULT_GPU_ID,
        slam: dict[str, Any] | None = None,
    ) -> dict:
        with self._lock:
            if self._line is not None:
                log.warning("SlamEsEnvManager already initialized — skipping")
                return self.get_episode_info()
            self._gpu_id = int(gpu_id)
            self._slam_cfg = dict(slam or {})
            self.configure(line=line, split=split or None, instruments=instruments)
            self._ensure_built_unlocked()
            return self.get_episode_info()

    def configure(
        self, line: str | None = None, split: str | None = None, instruments: int | None = None
    ) -> dict:
        """Update the desired (line, split, instruments); no build. A line
        change with no split named takes the line's validation split."""
        with self._lock:
            new_line = line or self._line or _DEFAULT_LINE
            if split is None:
                split = (
                    self._split
                    if (self._split and self._line == new_line)
                    else _default_split(self.list_splits(new_line))
                )
            if instruments is not None:
                level = int(instruments)
                if level not in _INSTRUMENT_LEVELS:
                    raise ValueError(
                        f"instruments must be one of {_INSTRUMENT_LEVELS}, got {level}"
                    )
                if level != self._instruments:
                    self._instruments = level
                    self._placed = (
                        False  # the next placement rebuilds the SLAM session for the new level
                    )
            if (new_line, split) != (self._line, self._split):
                self._episode_index = 0
            self._line, self._split = new_line, split
            return {"line": self._line, "split": self._split, "instruments": self._instruments}

    def _ensure_built_unlocked(self) -> None:
        if self._line is None:
            raise RuntimeError("Environment not initialized")
        if self._env is not None and self._live == (self._line, self._split):
            return
        import embodiedscore_envs as es

        if self._env is not None:
            self._env.close()
            self._env = None
            self._micro = None
        if self._split not in self.list_splits(self._line):
            raise ValueError(
                f"{self._line}: unknown split {self._split!r} (known: {self.list_splits(self._line)})"
            )
        data_root, scene_root = _data_roots()
        log.info(
            "Building the SLAM stack %s split=%s instruments=%d gpu=%d",
            self._line,
            self._split,
            self._instruments,
            self._gpu_id,
        )
        self._env = stack.build(
            self._line,
            self._split,
            instruments=self._instruments,
            data_root=data_root,
            scene_root=scene_root,
            gpu_id=self._gpu_id,
            slam=self._slam_cfg,
        )
        self._micro = self._find_micro(self._env)
        self._benchmark = es.benchmark(self._line, "standard")
        n = len(self._env.unwrapped.episodes)
        self._order = (
            habitat_shuffle_order(n) if self._benchmark.line in _SHUFFLED_LINES else list(range(n))
        )
        self._live = (self._line, self._split)
        self._placed = False
        self._tour = None
        self._transit = None
        self._episode_done = False
        self._current_obs, self._info = None, {}
        log.info("Stack ready — %d episodes", n)

    @staticmethod
    def _find_micro(env: Any) -> Any:
        e = env
        while e is not None:
            if isinstance(e, stack.SlamMicroStep):
                return e
            e = getattr(e, "env", None)
        raise RuntimeError("SlamMicroStep not found in the stack")

    def _ensure_placed_unlocked(self) -> None:
        self._ensure_built_unlocked()
        if not self._placed:
            self._place_unlocked(self._episode_index)

    def shutdown(self) -> None:
        with self._lock:
            if self._env is not None:
                log.info("Shutting down the SLAM stack")
                self._env.close()
                self._env = None
                self._micro = None
            self._live = None
            self._placed = False
            self._tour = None
            self._transit = None

    # ── Catalogue (env panel) ──

    def list_lines(self) -> list[str]:
        from embodiedscore_envs.benchmarks import BENCHMARKS

        return sorted({b.line for b in BENCHMARKS.values()})

    def list_splits(self, line: str | None = None) -> list[str]:
        import embodiedscore_envs as es

        line = line or self._line or _DEFAULT_LINE
        declared = list(es.benchmark(line).splits)
        data_root, _ = _data_roots()
        return declared + [s for s in _disk_splits(line, data_root) if s not in declared]

    def body(self) -> dict:
        """The arm's body in the env panel's ``action_space`` key set (what
        the driver's action-space guard reads): STANDARD at coarse quanta."""
        f = stack.FINE
        return {
            "turn_angle_deg": float(stack.COARSE_TURN_DEG),
            "forward_step_m": float(stack.COARSE_STEP_M),
            "tilt_angle_deg": float(stack.COARSE_TILT_DEG),
            "pitch_limit_deg": float(stack.PITCH_LIMIT_DEG),
            "start_pitch_deg": 0.0,
            "camera_lever_arm": bool(f.pitch_moves_camera),
            "rgb_hfov_deg": float(f.rgb.hfov_deg),
            "rgb_resolution": [int(f.rgb.height), int(f.rgb.width)],
            "camera_height_m": float(f.rgb.position[1]),
            "agent_height_m": float(f.agent_height_m),
            "agent_radius_m": float(f.agent_radius_m),
            "allow_sliding": bool(f.allow_sliding),
            "micro_per_action": int(stack.MICRO_PER_ACTION),
            "task_config": "exp_workspace/lhnav_es/mcp/stack.py:FINE (presets.bodies.STANDARD at micro quanta)",
        }

    def get_total_episodes(self) -> int:
        with self._lock:
            if self._line is None:
                return 0
            self._ensure_built_unlocked()
            return len(self._order)

    def get_episodes_list(self, offset: int = 0, limit: int = 50) -> dict:
        with self._lock:
            if self._line is None:
                return {"episodes": [], "total": 0}
            self._ensure_built_unlocked()
            eps = self.episodes
            out = []
            for i, ep in enumerate(eps[offset : offset + limit]):
                goal = ep.goal
                task = (
                    f"{len(goal.goals)} subtasks"
                    if goal.kind == "sequence"
                    else _task_text(ep, goal)
                )
                out.append(
                    {
                        "index": offset + i,
                        "episode_id": str(ep.episode_id),
                        "scene_id": _legacy_scene_id(ep.scene.scene_file),
                        "goal_kind": goal.kind,
                        "task": task,
                    }
                )
            return {"episodes": out, "total": len(eps)}

    # ── Episode control (env panel + reset) ──

    def _place_unlocked(self, index: int, keep_map: bool = False) -> None:
        self._micro.instruments = self._instruments
        tour = self._continues_tour(index)
        self._micro.keep_map_next = tour or (keep_map and self._same_scene(index))
        try:
            # An IR2R-CE transit is driven, the tracker fed at every oracle step;
            # the next tour of a deployment day is a placement, re-found by a
            # turn in place (so is a transit the oracle had to end by placing).
            walked = self._walk_transit() if tour else None
            obs, info = self._env.reset(options={"episode": self._order[index]})
            transit = None
            if self._micro.keep_map_next:
                spin = walked is None or bool(walked.get("teleported"))
                transit = self._micro.continue_tour(spin=spin, walked=walked)
        finally:
            self._micro.keep_map_next = False
        self._current_obs, self._info = obs, info
        self._tour = dict(info.get("tour") or {}) or None
        # what the transit cost, if this episode was walked to rather than
        # placed: the tracker's status after the turn in place, and (for
        # evaluation only) how far its pose is from the body's. Recorded per
        # episode so a tour's map can be audited afterwards.
        self._transit = dict(transit) if transit else None
        self._placed = True
        self._episode_done = False
        self._step_count = 0
        self._last_action = None
        self._answer = None
        self._episode_index = index

    def _tour_wrapper(self):
        """The package's TourWrapper in this stack (IR2R-CE), or None."""
        e = self._env
        while e is not None:
            if hasattr(e, "transit") and hasattr(e, "tour_of"):
                return e
            e = getattr(e, "env", None)
        return None

    def _walk_transit(self) -> dict | None:
        """Walk the pending tour transit one oracle step at a time, feeding the
        tracker at each (``SlamMicroStep.walk_transit``)."""
        tw = self._tour_wrapper()
        if tw is None:
            return None
        return self._micro.walk_transit(lambda: tw.transit(run_to_completion=False))

    def _continues_tour(self, index: int) -> bool:
        """Is the episode about to be placed the next one of the tour already
        running here? (IVLN-CE; every other line has tours of one.)

        Only then does the package walk the body over with its own oracle
        instead of teleporting it, and only then is keeping the map the honest
        thing to do. A tour picked up mid-way on a fresh server — a resume, a
        re-run of one episode — is teleported by the package and gets a fresh
        map, which is what the record should say happened.
        """
        prev = getattr(self, "_tour", None)
        if not prev or self._micro is None or self._micro.slam is None:
            return False
        nxt = self._tour_of(index)
        keep = bool(nxt and nxt.get("tour_id") == prev.get("tour_id")
                    and int(nxt.get("tour_index", 0)) == int(prev.get("tour_index", -1)) + 1)
        return keep

    def _same_scene(self, index: int) -> bool:
        """Is episode ``index`` in the scene already running here, with a live
        SLAM session? Only then can a placement keep the map: a GOAT deployment
        chains several episodes of one house, and the body is put down at the
        next episode's start with the map, the markers and the world frame it
        already has (``place_next``)."""
        if not self._placed or self._micro is None or self._micro.slam is None:
            return False
        eps = self._env.unwrapped.episodes
        try:
            nxt = eps[self._order[index]].scene.scene_file
            cur = self._base.episode.scene.scene_file
        except (IndexError, AttributeError):
            return False
        return os.path.abspath(nxt) == os.path.abspath(cur)

    def _tour_of(self, index: int) -> dict | None:
        """Which tour position episode ``index`` sits at, without placing it."""
        env = self._env
        while env is not None and not hasattr(env, "tour_of"):
            env = getattr(env, "env", None)
        if env is None:
            return None
        tour_id, pos, size = env.tour_of(self._order[index])
        return {"tour_id": tour_id, "tour_index": pos, "tour_size": size}

    def set_episode_by_index(self, index: int, keep_map: bool = False) -> dict:
        """Place + arm episode ``index``; idempotent on a still-fresh identical
        placement (the driver's field-push + play + reset sequence costs ONE
        build — the SLAM container + 360° bootstrap are ~40 s). ``keep_map``:
        the next episode of the same house, placed with the map kept
        (``place_next``)."""
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_built_unlocked()
            if index < 0 or index >= len(self._order):
                return {"error": f"Index {index} out of range (0-{len(self._order) - 1})"}
            fresh = (
                self._placed
                and self._episode_index == index
                and self._step_count == 0
                and not self._episode_done
            )
            if keep_map and not self._same_scene(index):
                return {"error": f"episode {index} is not in the scene running here; "
                                 "a map can only be kept within one house"}
            if not fresh:
                self._place_unlocked(index, keep_map=keep_map)
            return self._get_episode_info_unlocked()

    def ensure_live(self) -> dict:
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            if not self._episode_done:
                return self._get_episode_info_unlocked()
            return self.set_episode_by_index(max(self._episode_index, 0))

    # ── Transition ──

    def _is_question(self) -> bool:
        return self._base.episode.goal.kind == "question"

    def _stop_called(self) -> bool:
        return bool(self._info.get("stop_called", False))

    def step(self, action: int) -> dict:
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            if self._episode_done:
                return {"error": "Episode already done", "terminated": True}
            action = int(action)
            if action >= len(self._base.actions):
                return {
                    "error": f"action {action} not in this line's table {[_ACTION_NAMES[int(a)] for a in self._base.actions]}"
                }
            # the distance the sub-goal about to be closed is
            # scored at (SequenceNavMetrics scores prev_dtg on SUBTASK_STOP)
            prev_dtg = _finite_or_none(self._info.get("distance_to_goal"))
            obs, _r, terminated, truncated, info = self._env.step(action)
            self._current_obs, self._info = obs, info
            self._step_count += 1
            self._last_action = action
            self._episode_done = bool(terminated or truncated)
            self._truncated = bool(truncated) and not bool(terminated)
            out: dict[str, Any] = {
                "action": action,
                "action_name": _ACTION_NAMES[action],
                "step_count": self._step_count,
                "collision": bool(info.get("collision", False)),
                "collisions": int(info.get("collisions", 0)),
                "collided": bool(info.get("collided", False)),
                "micro_steps_done": int(info.get("micro_steps_done", 0)),
                "slam": str(info.get("slam", "")),
                "reanchors": int(info.get("reanchors", 0)),
                "tick": int(info.get("tick", 0)),
                # the recovery controller's readings (SlamMicroStep.body_facts):
                # newly explored area and a floor mismatch, without a get_map call
                **{k: info[k] for k in ("explored_m2", "current_floor", "floor_mismatch") if k in info},
                "distance_to_goal": _finite_or_none(info.get("distance_to_goal")),
                "goal_index": int(info.get("goal_index", 0)),
                "step_budget": self._step_budget_unlocked(),
                "episode_id": str(self._base.episode.episode_id),
            }
            if "pitch_deg" in info:
                # the sensor read carries float32 noise (-59.9999992); report exact multiples
                out["pitch_deg"] = round(float(info["pitch_deg"]), 3)
                if info.get("tilt_clamped"):
                    out["tilt_clamped"] = True
            out.update(self._get_agent_state_unlocked())
            # GOAT's cursor — which sub-goal just closed, and
            # its own score off the package's per-sub-goal lists
            if info.get("subtask_closed"):
                out["subtask_closed"] = True
                if self._se is not None:          # the next sub-goal draws its own path
                    self._se.begin_task(int(info.get("goal_index", 0)))
                m = _json_safe(dict(info.get("metrics") or {}))
                i = int(info.get("goal_index", 0)) - 1
                succ, spl = m.get("subtask_success") or [], m.get("spl_by_subtask") or []
                out["subtask_result"] = {
                    "index": i,
                    "success": float(succ[i]) if 0 <= i < len(succ) else None,
                    "spl": float(spl[i]) if 0 <= i < len(spl) else None,
                    "distance_to_goal": prev_dtg,
                    "subtasks_completed": m.get("subtasks_completed"),
                    "n_subtasks": m.get("n_subtasks"),
                }
            if self._episode_done:
                out["metrics"] = self._metrics_unlocked()
            return {
                "reward": 0.0,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": out,
            }

    # ── Perception (pull) ──

    def observe(self) -> dict:
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            obs = self._current_obs or {}
            rgb = obs.get("rgb")
            depth = obs.get("depth")
            ep = self._base.episode
            return {
                "rgb": np.asarray(rgb, dtype=np.uint8) if rgb is not None else None,
                "depth": np.asarray(depth, dtype=np.float32).squeeze()
                if depth is not None
                else None,  # METRIC metres
                "pose": self._get_agent_state_unlocked(),
                "intrinsics": self._se.env.intrinsics() if self._se is not None else None,
                "instruction_text": _task_text(ep, self._base.current_goal()),
                "info": self._body_facts_unlocked(),
            }

    def _body_facts_unlocked(self) -> dict:
        """The map's readings for the recovery controller, as a step reply
        carries them (``SlamMicroStep.body_facts``); {} on a bare run."""
        if self._micro is None:
            return {}
        facts = self._micro.body_facts()
        if facts and self._se is not None:
            facts["slam"] = self._se.status
        return facts

    def _get_agent_state_unlocked(self) -> dict:
        pos, rot = self._base.world.pose()
        return {
            "position": np.asarray(pos, dtype=float).tolist(),
            "orientation": [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])],
        }

    # ── Instrument reads (SLAM estimates, not ground truth) ──

    def _instrumented_unlocked(self, tool: str) -> dict | None:
        if self._line is None or self._se is None or not self._placed:
            return {"error": "no live episode — set an episode first"}
        if not self._instruments:
            return {"error": f"{tool} requires the instrumented arm (instruments >= 1)"}
        return None

    def get_pose(self) -> dict:
        with self._lock:
            err = self._instrumented_unlocked("get_pose")
            if err:
                return err
            pose = self._se.get_pose()
            pose["convention"] = _POSE_CONVENTION
            return pose

    def get_map(self, floor: int | None = None, view: str = "global") -> dict:
        with self._lock:
            err = self._instrumented_unlocked("get_map")
            if err:
                return err
            se = self._se
            # v3 is v2's floor registry with evidence layers underneath:
            # same layers, same render window, same picture.
            if se.map_mode in ("v2", "v3"):
                out = self._get_map_v2_unlocked(se, floor=floor, view=view)
                self._keep_map(out, view)
                return out
            m = se.get_map()
            vm = se.view()
            m.pop("png", None)
            frontier_cells = [(int(f["id"][1:]), tuple(f.pop("cell"))) for f in m["frontiers"]]
            agent_xz = None
            yaw = None
            if se.last_pose is not None:
                agent_xz = (float(se.last_pose[0, 3]), float(se.last_pose[2, 3]))
                yaw = math.radians(se.get_pose()["yaw_deg"])
            img, window = render_annotated_map(
                vm.grid,
                frontier_mask(vm.grid),
                vm.cell_size,
                vm.origin,
                agent_xz=agent_xz,
                agent_yaw=yaw,
                trajectory=se.drawn_track(),
                frontier_cells=frontier_cells,
                **self._render,
            )
            m["map"] = img
            p = se.get_pose()
            if "x" in p:
                m["pose"] = {"x": p["x"], "z": p["z"], "yaw_deg": p["yaw_deg"]}
            m["map_window_m"] = round(window, 1)
            m["orientation"] = "up = +z (your starting heading); right = +x"
            m["current_floor"] = int(getattr(se.occ, "current_floor", 0))
            m["floor_mismatch"] = se.floor_mismatch()
            return m

    def _get_map_v2_unlocked(
        self,
        se: SlamEnv,
        floor: int | None = None,
        view: str = "global",
        path: list | None = None,
        store_window: bool = True,
    ) -> dict:
        """Map v2: no frontier layer; per-floor render; crop window
        snapped to 2 m multiples and grow-only per episode; ``floor`` views
        another storey; ``view="local"`` adds a fixed 8 m agent-centred crop;
        ``path`` overlays a preview polyline; markers render on their floor."""
        vm = se.view()
        n_floors = len(vm.layers)
        if view == "local" or floor is None:
            fidx = vm.current_floor
        else:
            fidx = int(floor) - 1
            if not 0 <= fidx < n_floors:
                return {"error": f"floor {floor} out of range (1..{n_floors})"}
        layer = vm.layers[fidx]
        floor_y = vm.floor_ys[fidx]
        on_floor = fidx == vm.current_floor
        agent_xz = None
        yaw = None
        if se.last_pose is not None and on_floor:
            agent_xz = (float(se.last_pose[0, 3]), float(se.last_pose[2, 3]))
            yaw = math.radians(se.get_pose()["yaw_deg"])
        # this sub-goal's path, inside this episode's — see SlamEnv.drawn_track
        traj = [p for p in se.drawn_track() if abs(p[1] - floor_y) <= vm.floor_merge_m]
        floor_label = f"floor {fidx + 1}/{n_floors}"
        # where this EPISODE began, independent of how much of the path is drawn —
        # the slice above would otherwise hide the start marker from sub-goal 2 on
        start_here = (len(se.est_track) > se.ep_track0
                      and abs(se.est_track[se.ep_track0][1] - floor_y) <= vm.floor_merge_m)
        markers = [(m["label"], m["x"], m["z"]) for m in se.markers if m["floor"] == fidx]
        img, window, new_window = render_annotated_map_v2(
            layer.grid,
            vm.cell_size,
            vm.origin,
            agent_xz=agent_xz,
            agent_yaw=yaw,
            trajectory=traj,
            window_cells=se.map_window,
            floor_label=floor_label,
            mark_start=start_here,
            markers=markers,
            path=path,
            **self._render,
        )
        if store_window:
            se.map_window = new_window
        out = {
            "map": img,
            "floor": floor_label,
            "view": view,
            "free_m2": layer.free_area_m2(),
            "explored_m2": layer.explored_area_m2(),
            "markers": [{"label": lb, "x": mx, "z": mz} for lb, mx, mz in markers],
            "slam": se.status,
            "tick": se.tick,
            "map_window_m": round(window, 1),
            "orientation": "up = +z (your starting heading); right = +x",
            # the layer routes are planned on (0-based), and whether the body is
            # on it: what the place graph and the recovery controller read
            "current_floor": int(getattr(se.occ, "current_floor", 0)),
            "floor_mismatch": se.floor_mismatch(),
        }
        if view == "local":
            img_l, window_l, _ = render_annotated_map_v2(
                layer.grid,
                vm.cell_size,
                vm.origin,
                agent_xz=agent_xz,
                agent_yaw=yaw,
                trajectory=traj,
                floor_label=floor_label,
                mark_start=start_here,
                markers=markers,
                path=path,
                fixed_window_cells=self._local_window_cells(se),
                **self._render,
            )
            out["map_local"] = img_l
            out["local_window_m"] = round(window_l, 1)
        p = se.get_pose()
        if "x" in p:
            out["pose"] = {"x": p["x"], "z": p["z"], "yaw_deg": p["yaw_deg"]}
        return out

    @staticmethod
    def _local_window_cells(se: SlamEnv, span_m: float = 8.0) -> tuple:
        vm = se.view()
        cs = vm.cell_size
        n = vm.n
        origin = vm.origin
        snap = max(1, round(2.0 / cs))
        side = -(-round(span_m / cs) // snap) * snap
        ax = az = 0.0
        if se.last_pose is not None:
            ax, az = float(se.last_pose[0, 3]), float(se.last_pose[2, 3])
        azi = math.floor(az / cs) + origin
        axi = math.floor(ax / cs) + origin
        zi0 = origin + ((azi - side // 2 - origin) // snap) * snap
        xi0 = origin + ((axi - side // 2 - origin) // snap) * snap
        zi0 = max(0, min(zi0, n - side))
        xi0 = max(0, min(xi0, n - side))
        return (zi0, zi0 + side, xi0, xi0 + side)

    def _plan_unlocked(self, se: SlamEnv, x: float, z: float, goal_snap_m: float = 1.0) -> dict:
        vm = se.view()
        if se.last_pose is None:
            return {"error": "no pose estimate yet"}
        grid = vm.grid
        cs = vm.cell_size
        start = vm.world_to_cell(float(se.last_pose[0, 3]), float(se.last_pose[2, 3]))
        goal = vm.world_to_cell(float(x), float(z))
        s2 = nearest_free(grid, start, max(1, int(0.5 / cs)))
        if s2 is None:
            return {"error": "no free map cells near the agent yet"}
        g2 = nearest_free(grid, goal, max(1, int(goal_snap_m / cs)))
        if g2 is None:
            return {
                "reachable": False,
                "reason": (
                    f"target has no walked-free map cell within {goal_snap_m:g} m — "
                    "the map only knows space you have already seen"
                ),
            }
        cells = astar_free(grid, s2, g2)
        if cells is None:
            return {"reachable": False, "reason": "no free-space route on this floor's map"}
        world = [vm.cell_to_world(zi, xi) for zi, xi in cells]
        return {
            "reachable": True,
            "cells": cells,
            "world": world,
            "distance_m": round(path_length_m(cells, cs), 2),
            "snapped_target": {"x": round(world[-1][0], 2), "z": round(world[-1][1], 2)},
            "snapped": g2 != goal,
        }

    def mark(self, label: str) -> dict:
        with self._lock:
            err = self._instrumented_unlocked("mark")
            if err:
                return err
            return self._se.mark(label)

    def preview_path(self, x: float, z: float) -> dict:
        with self._lock:
            err = self._instrumented_unlocked("preview_path")
            if err:
                return err
            se = self._se
            plan = self._plan_unlocked(se, x, z)
            if "error" in plan or not plan.get("reachable"):
                return plan
            out = self._get_map_v2_unlocked(se, path=plan["world"], store_window=False)
            out.update(
                {
                    "reachable": True,
                    "distance_m": plan["distance_m"],
                    "snapped_target": plan["snapped_target"],
                    "snapped": plan["snapped"],
                }
            )
            return out

    def get_trajectory(self) -> dict:
        with self._lock:
            err = self._instrumented_unlocked("get_trajectory")
            if err:
                return err
            return self._se.get_trajectory()

    def look_around(self) -> dict:
        """360° panorama (4 views 90° apart), SLAM-silent, free. A step-cap
        truncation only ends MOVEMENT — the look stays available where the
        body stopped: on the question lines until the answer, and on every
        line for the stop views a harness takes of a task its budget ended."""
        with self._lock:
            err = self._instrumented_unlocked("look_around")
            if err:
                return err
            if self._episode_done and not (self._truncated
                                           or (self._is_question() and not self._stop_called())):
                return {"error": "episode already done"}
            result = self._se.look_around()
            result["pitch_deg"] = round(float(result.get("pitch_deg", 0.0)), 3)
            result["step_count"] = self._step_count
            return result

    # ── the goal in force ──

    def goal_spec(self) -> dict:
        """The goal in force now (a GOAT sequence's cursor, else the episode
        goal) — what a chained harness reads at every task boundary. An image
        sub-goal carries its photo, rendered from the dataset's stored camera
        pose (the goat-bench recipe; cached per (episode, sub-goal))."""
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            ep = self._base.episode
            goal = self._base.current_goal()
            seq = ep.goal if ep.goal is not None and ep.goal.kind == "sequence" else None
            spec: dict[str, Any] = {
                "task": _task_text(ep, goal),
                "goal": _goal_dict(goal),
                "goal_kind": getattr(goal, "kind", "none"),
                "goal_index": int(self._base.goal_index),
                "n_goals": len(seq.goals) if seq is not None else 1,
                "episode_over": bool(self._episode_done),
                "episode_id": str(ep.episode_id),
                "scene_id": _legacy_scene_id(ep.scene.scene_file),
                "step_count": self._step_count,
                "step_budget": self._step_budget_unlocked(),
            }
            if getattr(goal, "kind", "") == "image":
                spec["goal_image"] = self._goal_image_unlocked(goal)
            if self._tour_eps:
                # a tour is seated: where this episode sits in it. A chain that
                # IS the tour needs the slot to number its tasks — without it
                # every task of a single-goal line reports sub-goal 0 and the
                # rows collapse onto one index.
                spec["tour_slot"] = int(self._tour_pos)
                spec["tour_size"] = len(self._tour_eps)
            return spec

    def _goal_image_unlocked(self, goal: Any) -> str:
        key = (int(self._base.episode.index), int(self._base.goal_index))
        cached = self._goal_image_cache.get(key)
        if cached is None:
            from embodiedscore_envs.benchmarks.env import CameraSpec

            cam = CameraSpec(goal.image_size[1], goal.image_size[0], goal.hfov_deg)
            rgb = self._base.world.render_at(goal.camera_position, goal.camera_rotation, cam)["rgb"]
            cached = png_b64(np.asarray(rgb, dtype=np.uint8))
            self._goal_image_cache[key] = cached
        return cached

    # ── tours (IVLN-CE): a whole tour as ONE unit ──
    #
    # A tour is the point of IVLN: its episodes are walked, not teleported, and
    # what carries between them — the map, the markers, what the agent learned
    # about the house — is what the benchmark is asking about. Scoring each
    # episode as its own record answers a different, shorter question. So these
    # four verbs let a driver treat the tour as the unit: seat it, walk to its
    # next episode, and read the tour's own ruler at the end. An episode-major
    # driver (ES batch) never calls them and is unaffected.

    def tour_plan(self) -> dict:
        """The split's tours, as lists of episode indices in walking order."""
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_built_unlocked()
            tours: dict[str, list[int]] = {}
            for i in range(len(self._order)):
                t = self._tour_of(i)
                if t is None:
                    return {"error": f"{self._line} has no tours (this is an IVLN-CE verb)"}
                tours.setdefault(str(t["tour_id"]), []).append(i)
            order = sorted(tours, key=lambda k: tours[k][0])
            return {"tours": [{"tour_id": k, "episodes": tours[k], "size": len(tours[k])}
                              for k in order], "n_tours": len(order)}

    def place_tour(self, index: int) -> dict:
        """Seat tour ``index`` at its first episode, with a fresh map.

        A tour STARTS clean — that is what makes the carrying over inside it
        mean something — so this is an ordinary placement, teleport and all."""
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_built_unlocked()
            plan = self._tour_plan_unlocked()
            if "error" in plan:
                return plan
            if index < 0 or index >= len(plan["tours"]):
                return {"error": f"tour {index} out of range (0-{len(plan['tours']) - 1})"}
            t = plan["tours"][index]
            self._tour_rows = []
            self._tour_index = index
            self._tour_eps = list(t["episodes"])
            self._tour_pos = 0
            self._tour = None            # a new tour: never continue the last one's map
            self._place_unlocked(self._tour_eps[0])
            return {**self._place_facts_unlocked(),
                    "tour_id": t["tour_id"], "tour_index": index, "tour_size": t["size"],
                    "tour_slot": 0}

    def advance_tour(self) -> dict:
        """Walk to the next episode of the tour in force.

        The package's oracle does the walking (the body is not teleported) and
        the map, the markers and the world frame carry over; what the transit
        cost the tracker is reported, never hidden."""
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            if not getattr(self, "_tour_eps", None):
                return {"error": "no tour seated — place_tour first"}
            self._tour_rows.append(self._tour_row_unlocked())
            if self._tour_pos + 1 >= len(self._tour_eps):
                return {"tour_over": True, "tour_slot": self._tour_pos,
                        "tour_size": len(self._tour_eps)}
            self._tour_pos += 1
            self._place_unlocked(self._tour_eps[self._tour_pos])
            return {**self._place_facts_unlocked(), "tour_over": False,
                    "tour_index": self._tour_index, "tour_slot": self._tour_pos,
                    "tour_size": len(self._tour_eps)}

    def tour_report(self) -> dict:
        """The tour's own ruler: every episode's metrics, their means, and the
        package's tour-level t-nDTW (the one number an episode cannot give)."""
        with self._lock:
            if not getattr(self, "_tour_eps", None):
                return {"error": "no tour seated"}
            rows = list(self._tour_rows)
            if len(rows) < self._tour_pos + 1:
                rows.append(self._tour_row_unlocked())
            keys = ("success", "spl", "ndtw", "oracle_success", "distance_to_goal",
                    "steps_taken", "explored_m2")
            means = {}
            for k in keys:
                vals = [float(r["metrics"][k]) for r in rows
                        if isinstance((r.get("metrics") or {}).get(k), (int, float))]
                if vals:
                    means[k] = sum(vals) / len(vals)
            tour = dict(self._tour or {})
            return {"tour_index": getattr(self, "_tour_index", -1),
                    "tour_id": tour.get("tour_id"), "episodes": rows,
                    "n_episodes": len(rows), **means,
                    "t_ndtw": tour.get("tour_t_ndtw"),
                    "tour_protocol_ok": tour.get("tour_protocol_ok")}

    def _tour_plan_unlocked(self) -> dict:
        tours: dict[str, list[int]] = {}
        for i in range(len(self._order)):
            t = self._tour_of(i)
            if t is None:
                return {"error": f"{self._line} has no tours (this is an IVLN-CE verb)"}
            tours.setdefault(str(t["tour_id"]), []).append(i)
        order = sorted(tours, key=lambda k: tours[k][0])
        return {"tours": [{"tour_id": k, "episodes": tours[k], "size": len(tours[k])}
                          for k in order]}

    def _tour_row_unlocked(self) -> dict:
        ep = self._base.episode
        return {"slot": self._tour_pos, "episode_index": self._episode_index,
                "episode_id": str(ep.episode_id), "metrics": self._metrics_unlocked(),
                "transit": dict(self._transit) if getattr(self, "_transit", None) else None}

    def place_facts(self) -> dict:
        with self._lock:
            return self._place_facts_unlocked()

    def _place_facts_unlocked(self) -> dict:
        """The task facts a placement hands back, as ``place`` words them."""
        meta = self._get_episode_info_unlocked()
        return {"task": str(meta.get("task", "")), "instruction": str(meta.get("task", "")),
                "episode_id": str(meta.get("episode_id", "")),
                "scene_id": str(meta.get("scene_id", "")),
                "question": str(meta.get("question", "")),
                "choices": list(meta.get("choices") or []),
                "object_category": str(meta.get("object_category", "")),
                "goal": meta.get("goal"), "line": str(meta.get("line", "")),
                "step_budget": meta.get("step_budget"),
                "goal_index": meta.get("goal_index", 0), "n_goals": meta.get("n_goals", 1),
                "tour": self.tour, "transit": self.transit}

    # ── checkpoints, resume and replay (the NavHarness seats) ──
    #
    # ES batch runs an episode once, start to finish, so it needs none of this.
    # NavHarness does: a chain is many tasks long, a wall clock can cut one
    # short, and the question worth asking afterwards is the counterfactual —
    # would task 4 have gone differently with another note, another rung — which
    # is only answerable if task 4 can be started again from the state it
    # actually started from. These five verbs are what the harness asks the
    # world for to make that possible.

    def get_body_pose(self) -> dict:
        """Where the body is, in the simulator's own words.

        Ground truth, and deliberately so: it is written into a checkpoint file
        and read only by a replay. It never enters a prompt, a note or the
        ledger — exactly as the map file's geometry does not. A checkpoint is
        infrastructure, not something the agent is told."""
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            pos, rot = self._base.world.pose()
            return {"pose": {"position": np.asarray(pos, dtype=float).tolist(),
                             "rotation_xyzw": [float(r) for r in np.asarray(rot).ravel()[:4]]}}

    def set_body_pose(self, position: list, rotation_xyzw: list | None = None) -> dict:
        """Put the body back where a checkpoint recorded it, and re-anchor.

        Order matters and the harness gets it right: ``load_spatial`` first
        (the map and markers, moving nothing), then this. The SLAM session is
        opened here so that it continues from the checkpoint's estimate rather
        than calling the restored spot the origin.

        The package's own path counter is re-based on the restored position
        too. Without that the teleport itself counts as distance walked, and
        every SPL from a replayed task is wrong by the length of the jump —
        the same re-basing the sub-goal boundary does, for the same reason."""
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            pos = np.asarray(position, dtype=float)
            rot = (np.asarray(rotation_xyzw, dtype=float) if rotation_xyzw is not None
                   else self._base.world.pose()[1])
            self._base.world.place(pos, rot)
            self._current_obs = self._base.world.observe()
            self._episode_done = False
            restored = self._se.restore_body() if self._se is not None else {}
            rebased = self._rebase_metrics_unlocked()
            info = dict(self._info)
            info.update(self._get_agent_state_unlocked())
            info["distance_to_goal"] = rebased.get("distance_to_goal")
            self._info = info
            return {"placed": {"position": pos.tolist(),
                               "rotation_xyzw": [float(r) for r in np.asarray(rot).ravel()[:4]]},
                    **restored, **rebased}

    def _rebase_metrics_unlocked(self) -> dict:
        """Start the package's distance-walked counters again from here.

        ``SequenceNavMetrics`` measures each sub-goal's SPL from where the agent
        stood when the previous one closed (``_se``, ``_walked``, ``_prev``);
        ``NavMetrics`` measures a single-goal episode from its start (``_d0``,
        ``_L``, ``_prev``). A restored body is a new such starting point, so the
        same three fields are set the same way — otherwise the jump is counted
        as walking."""
        pos, _ = self._base.world.pose()
        pos = np.asarray(pos, dtype=float).tolist()
        dtg = self._base._distance_to_goal(np.asarray(pos, dtype=float))
        out: dict[str, Any] = {"distance_to_goal": _finite_or_none(dtg), "rebased": []}
        env = self._env
        while env is not None:
            name = type(env).__name__
            if name == "SequenceNavMetrics":
                env._prev = list(pos)
                env._walked = 0.0
                env._se = float(dtg)
                env._prev_dtg = float(dtg)
                env._dtg = float(dtg)
                out["rebased"].append(name)
            elif name in ("NavMetrics", "VLNVerseMetrics"):
                env._prev = list(pos)
                env._L = 0.0
                env._d0 = float(dtg)
                out["rebased"].append(name)
            env = getattr(env, "env", None)
        return out

    def dump_spatial(self, path: str) -> dict:
        """Freeze the map, the markers, the world frame and the last estimate."""
        with self._lock:
            if self._se is None:
                return {"error": "no SLAM session — dump_spatial needs the instrumented arm"}
            return self._se.dump_spatial(path)

    def load_spatial(self, path: str) -> dict:
        """Read a frozen boundary back. Moves nothing: ``set_body_pose`` next."""
        with self._lock:
            if self._se is None:
                return {"error": "no SLAM session — load_spatial needs the instrumented arm"}
            return self._se.load_spatial(path)

    def subtask_cursor(self) -> dict:
        """Which sub-goal is in force and how much budget is gone — the third
        thing a checkpoint holds, beside the map and the body."""
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            ep = self._base.episode
            seq = ep.goal if ep.goal is not None and ep.goal.kind == "sequence" else None
            return {"cursor": {"goal_index": int(self._base.goal_index),
                               "n_goals": len(seq.goals) if seq is not None else 1,
                               "episode_id": str(ep.episode_id),
                               "episode_index": self._episode_index,
                               "step_count": self._step_count,
                               "step_budget": self._step_budget_unlocked()}}

    def skip_subtask(self) -> dict:
        """Walk the cursor past a sub-goal the harness has already recorded.

        A resume replays the ledger of a run a wall clock cut short and then
        carries on; the env is fresh, so its cursor has to be moved to where the
        ledger stopped. The only way this package advances a GOAT sequence is
        SUBTASK_STOP, which also SCORES the sub-goal — at the start pose, so it
        scores a failure. That is honest rather than convenient: the env's own
        metrics for a resumed chain are void for the skipped sub-goals and the
        harness's ledger is the record of them. ``voided`` says so in the reply
        rather than leaving it to be discovered in a results table."""
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            before = int(self._base.goal_index)
            if _ACTION_SUBTASK_STOP >= len(self._base.actions):
                return {"error": "this line has no SUBTASK_STOP; there is nothing to skip"}
            obs, _r, terminated, truncated, info = self._env.step(_ACTION_SUBTASK_STOP)
            self._current_obs, self._info = obs, info
            self._step_count += 1
            self._episode_done = bool(terminated or truncated)
            self._truncated = bool(truncated) and not bool(terminated)
            return {"subtask_index": int(info.get("goal_index", before + 1)),
                    "was": before, "episode_over": self._episode_done,
                    "voided": "the package scored this sub-goal at the start pose; "
                              "the harness ledger is its record"}

    def set_map_capture(self, directory: str = "", every: int = 1) -> dict:
        """Keep a copy of every map render this env hands out.

        A navigation run that can be read in words and camera frames but never
        shows its MAP is missing the one artefact it is about, and the grid
        itself dies with the env server. The renders are a few KB each and
        land beside the frames the monitor already serves, under
        `map_<n>_<view>.png`.
        """
        d = str(directory or "")
        self._map_dir = pathlib.Path(d) if d else None
        self._map_every = max(1, int(every))
        if self._map_dir is not None:
            self._map_dir.mkdir(parents=True, exist_ok=True)
        return {"map_capture": str(self._map_dir) if self._map_dir else None,
                "every": self._map_every}

    def _keep_map(self, out: dict, view: str) -> None:
        d = getattr(self, "_map_dir", None)
        if d is None or not isinstance(out, dict):
            return
        self._map_shots = n = int(getattr(self, "_map_shots", 0)) + 1
        if n % getattr(self, "_map_every", 1):
            return
        for key, tag in (("map", view), ("map_local", "closeup")):
            b64 = out.get(key)
            if not isinstance(b64, str) or not b64:
                continue
            try:
                (d / f"map_{n:05d}_{tag}.png").write_bytes(base64.b64decode(b64))
            except Exception:                       # noqa: BLE001 — never lose a task over a record
                return

    def set_floor(self, level: int | None = None, marker: str = "", why: str = "") -> dict:
        """Hand the map the level it is on (navharness wake-up -> place graph)."""
        with self._lock:
            se = self._se
            if se is None:
                return {"error": "no live episode"}
            return se.set_floor(level=level, marker=marker, why=why)

    def reset_spatial(self) -> dict:
        """The episodic setting (NavHarness ``persist_map=False``): rebuild
        the SLAM session, the occupancy map and the markers where the body
        stands — the episode, its cursor and its budget do not move."""
        with self._lock:
            if self._line is None or self._micro is None or not self._placed:
                return {"error": "no live episode — set an episode first"}
            if self._episode_done:
                return {"error": "episode already done"}
            boot = self._micro.reset_slam()
            return {"reset": True, "instruments": self._instruments, "bootstrap": boot}

    # ── Metric sink ──

    def _step_budget_unlocked(self) -> int | None:
        static = self._benchmark.max_episode_steps
        return int(static) if static is not None else None

    def _metrics_unlocked(self, pred_letter: str = "") -> dict:
        """The line's own metric keys (package wrapper), plus the arm's
        bookkeeping and — instrumented — the SLAM-quality block. On a
        question line ``success`` is the answer letter against the dataset's
        (HM-EQA semantics; unanswered = 0)."""
        m = _json_safe(dict(self._info.get("metrics") or {}))
        se = self._se
        m.update(
            {
                "steps_taken": float(self._step_count),
                "stop_called": float(self._stop_called()),
                "collisions": float(se.env.collisions) if se is not None else 0.0,
                "instruments": float(self._instruments),
            }
        )
        goal = self._base.episode.goal
        if goal.kind == "question":
            letter = pred_letter or (self._answer or "")
            scored = score_answer(letter, str(goal.answer))
            m.update(scored)
            m["label"] = str(self._base.episode.info.get("label", ""))
            m["final_pitch_deg"] = round(float(se.env.pitch_deg), 3) if se is not None else 0.0
        if self._instruments and se is not None:
            slam = se.metrics()
            ate = slam.get("ate_rmse_m")
            m.update(
                {
                    "ate_rmse_m": None if ate is None or math.isnan(ate) else round(float(ate), 4),
                    "tracking_ratio": round(float(slam["tracking_ratio"]), 4),
                    "lost_ticks": float(slam["lost_ticks"]),
                    "reanchors": float(slam["reanchors"]),
                    # The atlas, as three numbers: how many maps ORB-SLAM3
                    # ended up with, how many it reconnected, and how many are
                    # still sitting on a guessed anchor nobody ever checked.
                    "map_epochs": float(len(slam["epochs"])),
                    "map_merges": float(len(slam["merges"])),
                    "map_provisional": float(sum(e["provisional"] for e in slam["epochs"])),
                    "explored_m2": round(float(slam["explored_m2"]), 2),
                    "micro_ticks": float(slam["ticks"]),
                }
            )
        return m

    def evaluate(self, pred_letter: str = "") -> dict:
        """``pred_letter`` is the question lines' answer (A-D; "" = never
        answered); it is recorded so a later call scores the same."""
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            if pred_letter:
                self._answer = str(pred_letter)
            return self._metrics_unlocked(pred_letter)

    def stop_truth(self) -> dict:
        """What the closing move would score if it were issued here, without
        issuing it: the distance to the goal in force — the body's own fact,
        the one the line's metric wrapper scores a STOP / SUBTASK_STOP
        against — and whether it is inside the line's success radius
        (``success`` 1.0 / 0.0). Read by a harness that refuses a close
        request, so a refused stop can be scored as if it had gone through;
        scoring only. A question line is scored on its answer, not on where
        it stops, and says so."""
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            if self._episode_done:
                return {"error": "Episode already done"}
            if self._is_question():
                return {"error": "a question line is scored on its answer, not on where it stops"}
            dtg = _finite_or_none(self._info.get("distance_to_goal"))
            radius = self._success_distance_unlocked()
            return {
                "success": None if dtg is None or radius is None else float(dtg < radius),
                "distance_to_goal": dtg,
                "success_distance": radius,
                "goal_index": int(self._base.goal_index),
                "step_count": self._step_count,
            }

    def _success_distance_unlocked(self) -> float | None:
        """The radius the line's metric wrapper scores success within (every
        one of them scores ``distance_to_goal < radius``)."""
        env = self._env
        while env is not None:
            if type(env).__name__ in ("SequenceNavMetrics", "NavMetrics", "VLNVerseMetrics"):
                radius = getattr(env, "_sd", None)
                return float(radius) if radius is not None else None
            env = getattr(env, "env", None)
        return None

    # ── Queries ──

    def get_episode_info(self) -> dict:
        with self._lock:
            if self._line is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            return self._get_episode_info_unlocked()

    def _get_episode_info_unlocked(self) -> dict:
        ep = self._base.episode
        goal = self._base.current_goal()
        info: dict[str, Any] = {
            "episode_id": str(ep.episode_id),
            "scene_id": _legacy_scene_id(ep.scene.scene_file),
            "episode_index": self._episode_index,
            "line": self._line,
            "split": self._split,
            "instruments": self._instruments,
            "benchmark": self._benchmark.name,
            "step_count": self._step_count,
            "done": self._episode_done,
            "step_budget": self._step_budget_unlocked(),
            "task": _task_text(ep, goal),
            "goal": _goal_dict(goal),
            "goal_kind": getattr(goal, "kind", "none"),
            "instruction": ep.instruction or "",
            "question": "",
            "choices": [],
            "object_category": "",
            "goal_index": int(self._base.goal_index),
            "n_goals": len(ep.goal.goals) if ep.goal is not None and ep.goal.kind == "sequence" else 1,
            "bootstrap": dict(self._micro.boot)
            if self._micro is not None and self._micro.boot
            else None,
        }
        if goal is not None and goal.kind == "question":
            info["question"] = goal.text
            info["choices"] = list(goal.choices) if goal.choices else []
        elif goal is not None and goal.kind == "object":
            info["object_category"] = goal.category
        return info


# ══════════════════════════════════════════════════════════════════════
# The verb surface
# ══════════════════════════════════════════════════════════════════════


def _image_fields(out: dict, keys: tuple[str, ...]) -> dict:
    """uint8 images → base64 PNG on the wire (what the bridge decodes)."""
    for k in keys:
        v = out.get(k)
        if v is not None and not isinstance(v, str):
            out[k] = png_b64(np.asarray(v, dtype=np.uint8))
    return out


class LhNavEsEnv:
    name = "LhNavEsEnv"

    def __init__(self, gpu_id: int = _DEFAULT_GPU_ID, slam: dict[str, Any] | None = None,
                 python: str | None = None, port: int | None = None,
                 shapes: Any = None, env_fields: Any = None) -> None:
        # python / port / shapes / env_fields are the launcher's and the
        # runner's keys of the same block. ``slam``: the stack's knobs
        # (stack.SLAM_KEYS: cell_size_m / map_size_m / max_coarse_steps /
        # frontier_match_m / start_pitch_deg / map_mode / mapper / render).
        self._gpu_id = int(gpu_id)
        self._slam: dict[str, Any] = dict(slam or {})
        unknown = set(self._slam) - stack.SLAM_KEYS
        if unknown:
            raise ValueError(f"env.slam: unknown key(s) {sorted(unknown)} (known: {sorted(stack.SLAM_KEYS)})")
        self._mgr = SlamEsEnvManager.get()

    # ── the run ──

    def configure(self, line: str, split: str, instruments: int = 0, **_ignored: Any) -> dict:
        m = self._mgr
        if not m.initialized:
            return m.initialize(line=line, split=split, instruments=int(instruments), gpu_id=self._gpu_id,
                                slam=self._slam)
        m.configure(line=line, split=split, instruments=int(instruments))
        return m.get_episode_info()

    def info(self) -> dict:
        m = self._mgr
        return {"name": self.name, "line": m.line, "split": m.split, "instruments": m.instruments,
                "slam": m.slam_cfg,
                "episodes": m.get_total_episodes() if m.initialized else 0}

    # ── per episode ──

    def place(self, index: int) -> dict:
        """Reset episode ``index``; the task facts per family, as the
        nav / objnav / eqa shapes read them."""
        meta = self._mgr.set_episode_by_index(int(index))
        if "error" in meta:
            return meta
        # `task` is what core.shapes.es reads (this folder's shapes are the es.*
        # ones); `instruction` is the nav/objnav/eqa shapes' name for the same
        # sentence, so either reads the same episode.
        return self._mgr.place_facts()

    def place_next(self, index: int) -> dict:
        """Place episode ``index`` of the SAME house with the map kept: the next
        tour of a GOAT deployment day. The package puts the body down at the
        episode's prescribed start (a relocation, not a walk); the SLAM session,
        the occupancy map, the markers and the world frame carry over, and the
        tracker re-finds the body by turning once in place (``continue_tour``).
        ``transit`` reports what that cost — whether the tracker came through
        or the map had to be anchored where the body was put down."""
        meta = self._mgr.set_episode_by_index(int(index), keep_map=True)
        if "error" in meta:
            return meta
        return self._mgr.place_facts()

    # ── tours (IVLN-CE): a whole tour as one unit; the navh seats' shape ──

    def tour_plan(self) -> dict:
        """The split's tours, as lists of episode indices in walking order."""
        return self._mgr.tour_plan()

    def place_tour(self, index: int) -> dict:
        """Seat tour ``index`` at its first episode, with a fresh map."""
        return self._mgr.place_tour(int(index))

    def advance_tour(self) -> dict:
        """Walk to the next episode of the tour in force; the map carries over."""
        return self._mgr.advance_tour()

    def tour_report(self) -> dict:
        """Every episode's metrics, their means, and the package's t-nDTW."""
        return self._mgr.tour_report()

    def goal_spec(self) -> dict:
        """The goal in force: text, kind, cursor, photo for an image sub-goal."""
        return self._mgr.goal_spec()

    def reset_spatial(self) -> dict:
        """Rebuild SLAM / map / markers in place (the episodic setting)."""
        return self._mgr.reset_spatial()

    def set_floor(self, level: int | None = None, marker: str = "", why: str = "") -> dict:
        """Which level the body is on, told rather than inferred (wake-up)."""
        return self._mgr.set_floor(level=level, marker=marker, why=why)

    def set_map_capture(self, directory: str = "", every: int = 1) -> dict:
        """Keep every map render this env hands out, where the monitor looks."""
        return self._mgr.set_map_capture(directory=directory, every=every)

    # ── checkpoints, resume and replay (the NavHarness seats; ES batch uses none) ──

    def get_body_pose(self) -> dict:
        """Where the body is (simulator truth; written to a checkpoint, read only by a replay)."""
        return self._mgr.get_body_pose()

    def set_body_pose(self, position: list, rotation_xyzw: list | None = None) -> dict:
        """Put the body back where a checkpoint recorded it and re-anchor SLAM there."""
        return self._mgr.set_body_pose(position, rotation_xyzw)

    def dump_spatial(self, path: str) -> dict:
        """Freeze the map, the markers, the world frame and the last estimate to a file."""
        return self._mgr.dump_spatial(str(path))

    def load_spatial(self, path: str) -> dict:
        """Read one back (moves nothing — ``set_body_pose`` is what moves the body)."""
        return self._mgr.load_spatial(str(path))

    def subtask_cursor(self) -> dict:
        """Which sub-goal is in force, and how much of the budget is gone."""
        return self._mgr.subtask_cursor()

    def skip_subtask(self) -> dict:
        """Walk the cursor past a sub-goal the harness's ledger already holds (resume)."""
        return self._mgr.skip_subtask()

    def observe(self) -> dict:
        out = self._mgr.observe()
        if "error" in out:
            return out
        depth = out.pop("depth", None)
        out = _image_fields(out, ("rgb",))
        out["depth"] = ndarray_wire(np.asarray(depth, dtype=np.float32)) if depth is not None else None
        return out

    def step(self, action: int) -> dict:
        return self._mgr.step(int(action))

    def evaluate(self, pred_letter: str = "", **_ignored: Any) -> dict:
        return self._mgr.evaluate(str(pred_letter or ""))

    def stop_truth(self) -> dict:
        """Would the closing move succeed at this pose: distance to the goal in
        force and the success radius, nothing issued (scoring only)."""
        return self._mgr.stop_truth()

    # ── instruments (free; refused below instruments >= 1) ──

    def get_pose(self) -> dict:
        return {"pose": self._mgr.get_pose()}

    def get_trajectory(self) -> dict:
        return {"trajectory": self._mgr.get_trajectory()}

    def get_map(self, floor: int | None = None, view: str = "global") -> dict:
        result = self._mgr.get_map(int(floor) if floor not in (None, "") else None, str(view or "global"))
        if "error" in result:
            return {"map": None, "map_local": None, "frontiers": [], "info": result}
        frontiers = result.pop("frontiers", [])
        out = _image_fields({"map": result.pop("map", None), "map_local": result.pop("map_local", None)},
                            ("map", "map_local"))
        out.update(frontiers=frontiers, info=result)
        return out

    def look_around(self) -> dict:
        result = self._mgr.look_around()
        if "error" in result:
            return {"view_front": None, "view_right": None, "view_back": None, "view_left": None, "info": result}
        views = result.pop("views")
        out = {"view_front": views[0], "view_right": views[1], "view_back": views[2], "view_left": views[3]}
        out = _image_fields(out, ("view_front", "view_right", "view_back", "view_left"))
        out["info"] = result
        return out

    def mark(self, label: str) -> dict:
        return {"result": self._mgr.mark(str(label))}

    def preview_path(self, x: float, z: float) -> dict:
        result = self._mgr.preview_path(float(x), float(z))
        if "error" in result:
            return {"map": None, "result": result}
        out = _image_fields({"map": result.pop("map", None)}, ("map",))
        out["result"] = result
        return out

    def close(self) -> None:
        self._mgr.shutdown()
