"""IVLN-CE / IR2R-CE (Krantz et al., CVPR 2023) — R2R-CE episodes walked as
*tours*: inside a tour the agent is never teleported; between two episodes
an oracle walks it from where it stopped to the finished episode's goal
(``oracle_goal``) and on to the next episode's start (``oracle_start``),
without touching the per-episode metrics. The tour-level metric is t-nDTW.

Data (under ``EMBODIEDSCORE_DATA_ROOT``):
    vlnce/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz (+ _gt)   the R2R-CE episodes, reused
    ivlnce/tours.json      {split: {scene: [[episode_id, ...], ...]}}; tour_id = flat index, scene-by-scene
                           in file order, tour-by-tour within a scene (the numbering gt_ndtw.json keys use)
    ivlnce/gt_ndtw.json    {split: {tour_id: [{episode_id, position, phase}, ...]}} — a recorded oracle
                           roll-out of the whole tour, NOT reconstructible from the _gt files (only tour-first
                           episodes coincide); loaded on first use (137 MB, ~5 s)
Episodes are ordered tour-major, tours.json order inside a tour; splits
train / val_seen / val_unseen (envdrop has no GT and is not offered).

Body / actions / per-episode metrics = VLN-CE (0.25 m, 15°, 1.5 / 0.1 agent,
sliding on, success 3.0 m; upstream renders RGB 224², the default here is the
VLN-CE std 512² rig — pixels feed no metric). The tour machinery is
:class:`TourWrapper`, transcribed from upstream ``VLNCEIterativeEnv`` (jacobkrantz/IVLN-CE @ 14417fc):

* follower = habitat-sim's GreedyGeodesicFollower, goal radius 0.25 m
  (= FORWARD_STEP_SIZE); once within radius, turn toward the start heading
  while off by >= 7.5° (TURN_ANGLE / 2)
* a GreedyFollowerError or 1000 oracle steps in a phase resolve the phase to
  STOP — oracle_start additionally teleports onto the start pose then
* PRECISE_EPISODE_START False: a successful oracle_start walk is not snapped
  (the next episode begins up to 0.25 m / 7.5° off its start)
* order between two episodes: oracle_goal walk (old episode loaded) ->
  inner reset of the next episode (metrics anchored at ITS canonical start)
  -> agent put back at its previous end pose -> oracle_start walk
* the tour path records ``{position, phase, episode_id}`` BEFORE every action,
  agent and oracle alike, with the episode currently loaded (so oracle_start
  entries carry the next episode's id); phase-entry probes do not append
* t-nDTW (vendored verbatim from upstream ``tour_ndtw.py`` bar the dtw-python
  call, re-implemented locally): agent path de-duplicated, GT raw, agent
  phases only, both truncated to the episodes whose agent phase finished,
  windowed symmetric1 DTW pinned at every episode boundary,
  ``exp(-D / (len(gt) * 3.0)) * 100`` rounded to 4 dp; per-tour only — the
  split-level weighting (|T| vs |T|-1) is the harness's call
* a tour must stay on one worker: the path is wrapper state
"""

from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
import quaternion  # noqa: F401  (numpy-quaternion: registers np.quaternion)
import scipy.spatial.distance

from .env import (VLN_KEYS, Act, Benchmark, Episode, NavMetrics, PointGoal, SceneRef, data_root,
                  scene_root)
from .presets import actions, bodies, depth

SUCCESS_DISTANCE = 3.0
ORACLE_GOAL_RADIUS = 0.25          # = FORWARD_STEP_SIZE
ORACLE_STEP_ERROR_LIMIT = 1000     # per oracle phase
PRECISE_EPISODE_START = False
_R2R_DIR = "R2R_VLNCE_v1-3_preprocessed"



# ---- data ----------------------------------------------------------------------

def _load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_tours(split: str, data_root_=None) -> list[list[str]]:
    """tours.json[split] flattened scene-by-scene, tour-by-tour: index = tour_id."""
    with open(data_root(data_root_) / "ivlnce" / "tours.json") as f:
        raw = json.load(f)[split]
    return [[str(e) for e in tour] for scene_tours in raw.values() for tour in scene_tours]


def load_episodes(split: str, data_root_=None, scene_root_=None) -> list[Episode]:
    """R2R-CE episodes of ``split`` regrouped tour-major; episodes absent from a
    tour are dropped, empty tours are dropped (upstream IterativeVLNCEDataset)."""
    root = data_root(data_root_)
    d = root / "vlnce" / _R2R_DIR / split
    raw = _load_json_gz(d / f"{split}.json.gz")["episodes"]
    gt_file = d / f"{split}_gt.json.gz"
    gt_locs = {str(k): v["locations"] for k, v in _load_json_gz(gt_file).items()} if gt_file.is_file() else {}
    by_id = {str(e["episode_id"]): e for e in raw}
    sroot = scene_root(scene_root_)
    tours = load_tours(split, data_root_)
    gt_ndtw_file = str(root / "ivlnce" / "gt_ndtw.json")
    out: list[Episode] = []
    for tour_id, tour in enumerate(tours):
        members = [by_id[eid] for eid in tour if eid in by_id]
        if not members:
            continue
        for pos, e in enumerate(members):
            ins = e["instruction"]
            goal = e["goals"][0]
            eid = str(e["episode_id"])
            info = dict(e.get("info") or {})
            info.update({"trajectory_id": str(e.get("trajectory_id", "")), "scene_id": e["scene_id"], "split": split,
                         "tour_id": str(tour_id), "tour_index": pos, "tour_size": len(members),
                         "gt_ndtw_file": gt_ndtw_file})
            out.append(Episode(
                index=len(out), episode_id=eid,
                scene=SceneRef(scene_file=str(sroot / e["scene_id"])),
                start_position=tuple(float(x) for x in e["start_position"]),
                start_rotation=tuple(float(x) for x in e["start_rotation"]),
                goal=PointGoal(tuple(float(x) for x in goal["position"]), float(goal.get("radius") or SUCCESS_DISTANCE)),
                instruction=ins["instruction_text"],
                reference_path=tuple(tuple(float(x) for x in p) for p in e.get("reference_path", [])),
                gt_path=tuple(tuple(float(x) for x in p) for p in gt_locs[eid]) if eid in gt_locs else None,
                info=info,
            ))
    return out


# ---- t-nDTW (upstream tour_ndtw.py; only the dtw-python call is local) ------------

def _symmetric1_distance(lm: np.ndarray, wm: np.ndarray) -> float:
    n, m = lm.shape
    cm = np.full((n, m), np.nan, dtype=np.double)
    cm[0, 0] = lm[0, 0]
    for d in range(1, n + m - 1):
        i = np.arange(max(0, d - (m - 1)), min(n - 1, d) + 1)
        j = d - i
        cand = np.full((3, i.size), np.nan, dtype=np.double)
        sel = (i > 0) & (j > 0)
        cand[0, sel] = cm[i[sel] - 1, j[sel] - 1]
        sel = j > 0
        cand[1, sel] = cm[i[sel], j[sel] - 1]
        sel = i > 0
        cand[2, sel] = cm[i[sel] - 1, j[sel]]
        best = np.fmin(np.fmin(cand[0], cand[1]), cand[2])
        cm[i, j] = np.where(wm[i, j], lm[i, j] + best, np.nan)
    distance = float(cm[n - 1, m - 1])
    if np.isnan(distance):
        raise ValueError("No warping path found compatible with the local constraints")
    return distance


def _local_cost_matrix(x, y) -> np.ndarray:
    x2 = np.array(x, dtype=np.double)
    y2 = np.array(y, dtype=np.double)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    if y2.ndim == 1:
        y2 = y2.reshape(-1, 1)
    return scipy.spatial.distance.cdist(x2, y2, metric="euclidean")


def compute_episodes_per_tour(tours: dict[str, list]) -> dict[str, int]:
    eps_per_tour: dict[str, int] = defaultdict(int)
    for tour_id, path in tours.items():
        for i in range(1, len(path)):
            if path[i]["episode_id"] != path[i - 1]["episode_id"]:
                eps_per_tour[tour_id] += 1
    return eps_per_tour


def window_align_func(iw, jw, query_size, reference_size, alignments):
    window = np.ones((query_size, reference_size), dtype=bool)
    for (i, j) in alignments:
        window[:, j] = False
        window[i, j] = True
    return window


def extract_ep_order(path):
    eps = [p["episode_id"] for p in path]
    eps_single = []
    for i in range(1, len(eps)):
        if eps[i - 1] != eps[i]:
            eps_single.append(eps[i - 1])
    eps_single.append(eps[-1])
    return eps_single


def alignments_from_paths(agent_path, gt_path):
    gt_path = [p for p in gt_path if p["phase"] == "agent"]
    agent_path = [p for p in agent_path if p["phase"] == "agent"]
    assert extract_ep_order(gt_path) == extract_ep_order(agent_path), "agent and GT episode orders do not match."
    agent_alignment_points = []
    for i in range(1, len(agent_path)):
        if agent_path[i]["episode_id"] != agent_path[i - 1]["episode_id"]:
            agent_alignment_points.append(i - 1)
            agent_alignment_points.append(i)
    gt_alignment_points = []
    for i in range(1, len(gt_path)):
        if gt_path[i]["episode_id"] != gt_path[i - 1]["episode_id"]:
            gt_alignment_points.append(i - 1)
            gt_alignment_points.append(i)
    assert len(agent_alignment_points) == len(gt_alignment_points), "mismatch in number of alignment points."
    return list(zip(agent_alignment_points, gt_alignment_points))


def novel_only(path):
    if len(path) == 0:
        return path
    new_path = [path[0]]
    if len(path) == 1:
        return path
    for i in range(1, len(path)):
        if path[i - 1] != path[i]:
            new_path.append(path[i])
    return new_path


def compute_tour_ndtw_scores(agent_paths: dict[str, list], gt_paths: dict[str, list],
                             success_distance: float = SUCCESS_DISTANCE) -> dict[str, float]:
    """Per-tour t-nDTW: upstream ``compute_tour_ndtw``'s arithmetic without the
    split-level fold (the GT path is deliberately NOT de-duplicated — upstream)."""
    if not set(gt_paths.keys()) == set(agent_paths.keys()):
        raise ValueError("tours are different")
    t_ndtws: dict[str, float] = {}
    for tour_id, agent_path in agent_paths.items():
        agent_path = novel_only(agent_path)
        gt_path = gt_paths[tour_id]
        alignments = alignments_from_paths(agent_path, gt_path)
        ap = [p["position"] for p in agent_path if p["phase"] == "agent"]
        gtp = [p["position"] for p in gt_path if p["phase"] == "agent"]
        lm = _local_cost_matrix(ap, gtp)
        wm = window_align_func(None, None, lm.shape[0], lm.shape[1], alignments)
        dtw_dist = _symmetric1_distance(lm, wm)
        t_ndtws[tour_id] = float(np.exp(-dtw_dist / (len(gtp) * success_distance)))
    return t_ndtws


# ---- the tour machine ----------------------------------------------------------------

def _yaw_deg_of(q) -> float:
    """Yaw in [0, 360), 0 = facing -z (upstream _heading_from_quaternion)."""
    v = (q * np.quaternion(0, 0, 0, -1) * q.inverse()).imag
    return math.degrees(math.atan2(-v[0], -v[2]) % (2 * math.pi))


def _quat_xyzw(c) -> Any:
    return np.quaternion(float(c[3]), float(c[0]), float(c[1]), float(c[2]))


class TourWrapper(gym.Wrapper):
    """Tours over a VLN-CE stack. Sits OUTSIDE the per-episode metrics wrapper
    and moves the body directly during oracle transits, so transit steps never
    reach the metrics. ``reset(options={"episode": i})`` continues the tour
    (oracle transit) when ``i`` is the next episode of the current tour and the
    current episode's agent phase is over; anything else places the agent at
    the episode's own start (a jump inside a tour clears the path and marks
    ``tour_protocol_ok`` False). ``info["tour"]`` carries tour_id / tour_index /
    tour_size / tour_episodes_done / transit_steps / tour_protocol_ok /
    tour_t_ndtw (×100, 4 dp; None until an episode of the tour has finished).
    """

    def __init__(self, env: gym.Env, success_distance: float = SUCCESS_DISTANCE) -> None:
        super().__init__(env)
        self._sd = float(success_distance)
        self._eps = env.unwrapped.episodes
        self._world = env.unwrapped.world
        self._by_id = {e.episode_id: e.index for e in self._eps}
        self._tours: dict[str, list[int]] = defaultdict(list)
        for e in self._eps:
            self._tours[e.info["tour_id"]].append(e.index)
        self._gt: dict[str, list] | None = None
        self._cur: int | None = None
        self._done = False
        self._phase = "agent"           # agent | oracle_goal | oracle_start | transit_done
        self._path: list[dict] = []
        self._path_tour: str | None = None
        self._completed: list[str] = []
        self._transit_buffer: list[dict] = []
        self._last_transit: list[dict] = []
        self._teleported = False
        self._protocol_ok = True
        self._target_index: int | None = None
        self._oracle_steps_in_phase = 0
        self._oracle_steps_total = 0
        self._t_ndtw: float | None = None

    # ---- lookups ----------------------------------------------------------------
    def tour_of(self, index: int) -> tuple[str, int, int]:
        e = self._eps[index]
        return e.info["tour_id"], int(e.info["tour_index"]), int(e.info["tour_size"])

    @property
    def tour_path(self) -> list[dict]:
        return list(self._path)

    @property
    def phase(self) -> str:
        return self._phase

    def gt_tour_path(self, tour_id: str) -> list | None:
        if self._gt is None:
            split = self._eps[0].info["split"]
            with open(self._eps[0].info["gt_ndtw_file"]) as f:
                self._gt = json.load(f).get(split) or {}
        return self._gt.get(str(tour_id))

    def tour_t_ndtw(self) -> float | None:
        """t-nDTW ×100 (4 dp) of the current tour's completed prefix, or None."""
        if self._cur is None or not self._completed:
            return None
        tour_id = self.tour_of(self._cur)[0]
        gt_path = self.gt_tour_path(tour_id)
        if not gt_path:
            return None
        done = set(self._completed)
        gt_trunc = [p for p in gt_path if str(p["episode_id"]) in done]
        agent_trunc = [p for p in self._path if str(p["episode_id"]) in done]
        if not gt_trunc or not agent_trunc:
            return None
        try:
            scores = compute_tour_ndtw_scores({tour_id: agent_trunc}, {tour_id: gt_trunc}, self._sd)
        except (AssertionError, ValueError):
            return None
        return round(100.0 * float(scores[tour_id]), 4)

    # ---- gym API ----------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = dict(options or {})
        if "episode" in options:
            index = int(options["episode"])
        elif "episode_id" in options:
            index = self._by_id[str(options["episode_id"])]
        else:
            self.env.unwrapped.reset(seed=seed)          # seed the RNG the standard way
            index = int(self.env.unwrapped.np_random.integers(len(self._eps)))
        options["episode"] = index
        options.pop("episode_id", None)

        cur = self._cur
        if cur is None:
            return self._force_place(index, seed, options, protocol_break=False)
        cur_tour = self.tour_of(cur)[0]
        new_tour = self.tour_of(index)[0]
        same_tour = cur_tour == new_tour
        if same_tour and index == cur + 1 and self._agent_phase_over():
            return self._continue_tour(index, seed, options)
        if index == cur:
            # re-arm in place: drop this episode's entries so the path stays a clean prefix
            eid = self._eps[index].episode_id
            self._path = [p for p in self._path if str(p["episode_id"]) != eid]
            self._completed = [e for e in self._completed if e != eid]
            return self._force_place(index, seed, options, protocol_break=False, keep_path=True)
        return self._force_place(index, seed, options, protocol_break=same_tour)

    def step(self, action):
        if self._phase != "agent":
            raise RuntimeError(f"TourWrapper: oracle transit in progress (phase {self._phase}); reset() first")
        if self._done:
            raise RuntimeError("TourWrapper: episode is over; reset() first")
        self._append_path("agent")
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self._done = True
            eid = self.env.unwrapped.episode.episode_id
            if eid not in self._completed:
                self._completed.append(eid)
            self._t_ndtw = self.tour_t_ndtw()
        info["tour"] = self._tour_info()
        return obs, reward, terminated, truncated, info

    # ---- explicit transit (upstream step_oracle) ------------------------------------
    def transit(self, run_to_completion: bool = True) -> dict[str, Any]:
        """Walk the pending oracle transit toward the next episode of the tour
        (one step, or all of it). A following ``reset`` to that episode then
        re-walks nothing."""
        nxt = self._next_tour_index()
        if nxt is None or not self._agent_phase_over():
            return {"phase": "agent", "transit_done": False, "oracle_steps_total": self._oracle_steps_total,
                    "next_episode_index": nxt, "teleported": False, "last_action": None, "distance_to_target": None,
                    "pending": False}
        last: dict[str, Any] | None = None
        if self._phase != "transit_done":
            if self._phase == "agent":
                self._begin_transit(nxt)
            if run_to_completion:
                guard = 2 * ORACLE_STEP_ERROR_LIMIT + 16
                while self._phase in ("oracle_goal", "oracle_start") and guard > 0:
                    last = self._oracle_advance_once(nxt)
                    guard -= 1
                if self._phase != "transit_done":
                    self._finish_oracle_start()
            elif self._phase in ("oracle_goal", "oracle_start"):
                last = self._oracle_advance_once(nxt)
        dist = None
        if self._phase in ("oracle_goal", "oracle_start"):
            target = self._phase_target(nxt)[0]
            d = self._world.geodesic(self._world.pose()[0], target)
            dist = round(float(d), 4) if math.isfinite(d) else None
        return {"phase": self._phase, "transit_done": self._phase == "transit_done",
                "oracle_steps_total": self._oracle_steps_total, "teleported": self._teleported,
                "next_episode_index": nxt, "pending": True,
                "last_action": (int(last["action"]) if last and last.get("stepped") else None),
                "distance_to_target": dist}

    # ---- placement ----------------------------------------------------------------------
    def _force_place(self, index: int, seed, options, *, protocol_break: bool, keep_path: bool = False):
        prev_tour = self.tour_of(self._cur)[0] if self._cur is not None else None
        new_tour, new_pos, _ = self.tour_of(index)
        obs, info = self.env.reset(seed=seed, options=options)
        self._cur = index
        self._done = False
        self._phase = "agent"
        self._last_transit = []
        self._teleported = False
        self._target_index = None
        self._transit_buffer = []
        clear = not keep_path and (new_tour != prev_tour or protocol_break)
        if clear:
            self._path = []
            self._completed = []
            self._path_tour = new_tour
            self._t_ndtw = None
        if protocol_break:
            self._protocol_ok = False
        elif new_tour != prev_tour:
            self._protocol_ok = new_pos == 0
        info["tour"] = self._tour_info()
        return obs, info

    def _continue_tour(self, index: int, seed, options):
        if self._phase != "transit_done" or self._target_index != index:
            self._run_transit(index)
        delivered_pos, delivered_rot = self._world.pose()
        # inner reset for the target episode (metrics anchored at ITS canonical start —
        # already done by _enter_oracle_start; repeated here so the explicit and implicit
        # transit paths converge), then the delivered pose is restored
        obs, info = self.env.reset(seed=seed, options=options)
        self._world.place(delivered_pos, delivered_rot)
        obs = self._world.observe()
        base = self.env.unwrapped
        pos, rot = self._world.pose()
        info.update({"position": pos.tolist(), "rotation": rot.tolist(), "heading": self._world.heading(),
                     "pitch": self._world.pitch(), "distance_to_goal": base._distance_to_goal(pos)})
        self._cur = index
        self._done = False
        self._last_transit = list(self._transit_buffer)
        self._phase = "agent"
        self._oracle_steps_in_phase = 0
        self._target_index = None
        self._transit_buffer = []
        info["tour"] = self._tour_info()
        return obs, info

    def _tour_info(self) -> dict[str, Any]:
        tour_id, pos, size = self.tour_of(self._cur)
        return {"tour_id": tour_id, "tour_index": pos, "tour_size": size,
                "tour_episodes_done": len(self._completed), "transit_steps": len(self._last_transit),
                "transit_teleported": self._teleported, "tour_protocol_ok": self._protocol_ok,
                "tour_t_ndtw": self._t_ndtw, "phase": self._phase}

    def _agent_phase_over(self) -> bool:
        return self._done or self._phase != "agent"

    def _next_tour_index(self) -> int | None:
        if self._cur is None:
            return None
        k = self._cur + 1
        if k >= len(self._eps) or self.tour_of(k)[0] != self.tour_of(self._cur)[0]:
            return None
        return k

    # ---- the oracle ----------------------------------------------------------------------
    def _append_path(self, phase: str) -> None:
        pos, _ = self._world.pose()
        entry = {"position": pos.tolist(), "phase": phase, "episode_id": str(self.env.unwrapped.episode.episode_id)}
        self._path.append(entry)
        if phase != "agent":
            self._transit_buffer.append({"position": entry["position"], "phase": phase})

    def _next_action(self, position_to, heading_to) -> int | None:
        """upstream _get_next_action: follower action, or the turn toward
        heading_to once within the goal radius; None = STOP."""
        action = self._world.follower(ORACLE_GOAL_RADIUS).next_action(position_to)
        if action is None and heading_to is not None:
            start_rot = _yaw_deg_of(_quat_xyzw(heading_to))
            current_rot = math.degrees(self._world.heading() % (2 * math.pi))
            delta = ((((start_rot - current_rot) % 360) + 540) % 360) - 180
            if abs(delta) >= self._world.body.turn_deg / 2:
                action = Act.RIGHT if delta < 0 else Act.LEFT
        return action

    def _next_action_safe(self, position_to, heading_to, teleport_on_failure: bool) -> tuple[int | None, bool]:
        from .env import FollowerError   # habitat_sim's; resolved here so importing the line needs no simulator
        try:
            action = self._next_action(position_to, heading_to)
            if not (self._oracle_steps_in_phase < ORACLE_STEP_ERROR_LIMIT):
                raise AssertionError("Too many oracle steps.")
            return action, True
        except (FollowerError, AssertionError):
            if teleport_on_failure:
                if heading_to is None:
                    _, rot = self._world.pose()
                    self._world.place(position_to, rot)
                else:
                    self._world.place(position_to, heading_to)
                self._teleported = True
            return None, False

    def _phase_target(self, next_index: int):
        if self._phase == "oracle_goal":
            return self.env.unwrapped.episode.goal.position, None, False
        nxt = self._eps[next_index]
        return nxt.start_position, nxt.start_rotation, True

    def _begin_transit(self, next_index: int) -> None:
        self._phase = "oracle_goal"
        self._oracle_steps_in_phase = 0
        self._oracle_steps_total = 0
        self._transit_buffer = []
        self._teleported = False
        self._target_index = next_index
        goal = self.env.unwrapped.episode.goal.position
        action, _ = self._next_action_safe(goal, None, False)
        if action is None:
            self._enter_oracle_start(next_index)

    def _enter_oracle_start(self, next_index: int) -> None:
        prev_pos, prev_rot = self._world.pose()
        self.env.reset(options={"episode": next_index})       # metrics anchor at the canonical start
        self._world.place(prev_pos, prev_rot)                 # agent back where it finished
        self._phase = "oracle_start"
        self._oracle_steps_in_phase = 0
        nxt = self._eps[next_index]
        action, _ = self._next_action_safe(nxt.start_position, nxt.start_rotation, True)
        if action is None:
            self._finish_oracle_start()

    def _finish_oracle_start(self) -> None:
        if PRECISE_EPISODE_START:   # pragma: no cover — upstream default False
            ep = self.env.unwrapped.episode
            self._world.place(ep.start_position, ep.start_rotation)
        self._phase = "transit_done"
        self._oracle_steps_in_phase = 0

    def _oracle_advance_once(self, next_index: int) -> dict[str, Any]:
        phase = self._phase
        self._append_path(phase)
        position_to, heading_to, teleport = self._phase_target(next_index)
        action, ok = self._next_action_safe(position_to, heading_to, teleport)
        stepped = False
        if action is not None:
            self._world.move(int(action))
            stepped = True
            self._oracle_steps_in_phase += 1
            self._oracle_steps_total += 1
            position_to, heading_to, teleport = self._phase_target(next_index)
            action_next, ok = self._next_action_safe(position_to, heading_to, teleport)
        else:
            action_next = action
        if action_next is None:
            if phase == "oracle_goal":
                self._enter_oracle_start(next_index)
            else:
                self._finish_oracle_start()
        return {"phase_before": phase, "action": int(action) if stepped else 0, "stepped": stepped, "follower_ok": bool(ok)}

    def _run_transit(self, next_index: int) -> None:
        if self._phase == "agent":
            self._begin_transit(next_index)
        guard = 2 * ORACLE_STEP_ERROR_LIMIT + 16
        while self._phase in ("oracle_goal", "oracle_start") and guard > 0:
            self._oracle_advance_once(next_index)
            guard -= 1
        if self._phase != "transit_done":
            self._finish_oracle_start()


def _decl(upstream: bool) -> Benchmark:
    return Benchmark(
        name="ivlnce-upstream" if upstream else "ivlnce",
        gym_id="EmbodiedScore/IVLNCE-Upstream-v0" if upstream else "EmbodiedScore/IVLNCE-v0",
        body=bodies.VLNCE if upstream else bodies.STANDARD, actions=actions.NAV if upstream else actions.STANDARD,
        splits=("train", "val_seen", "val_unseen"),
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(split, data_root, scene_root),
        metrics=lambda env, **o: TourWrapper(NavMetrics(env, success_distance=o.get("success_distance", SUCCESS_DISTANCE), keys=VLN_KEYS), o.get("success_distance", SUCCESS_DISTANCE)),
        depth=depth.STANDARD, max_episode_steps=500, variant="upstream" if upstream else "standard",
        description="IVLN-CE (IR2R-CE) tours " + ("on the VLN-CE rig (224²)" if upstream else "on the STANDARD body") + "; t-nDTW per tour",
    )


BENCHMARKS = (_decl(False), _decl(True))
