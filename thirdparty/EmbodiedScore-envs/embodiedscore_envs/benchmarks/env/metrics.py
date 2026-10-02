"""Metric wrappers (layer L1). They read only the facts the body puts in
``info`` — never the simulator — and write ``info["metrics"]`` after reset and
after every step.

``NavMetrics`` is the one formula set shared by every point / object goal
benchmark, transcribed from habitat-lab (``habitat/tasks/nav/nav.py``:
DistanceToGoal, Success, SPL, SoftSPL — identical in 0.1.7 and 0.2.4) and
VLN-CE (``habitat_extensions/measures.py``: NDTW, PathLength, OracleSuccess,
StepsTaken), quirks included:

* ``distance_to_goal``  the body's fact (geodesic to the nearest target), every step
* ``success``           stop_called and distance_to_goal < success_distance
* ``spl``               success * d0 / max(d0, L); d0 = distance_to_goal at reset
                        (from the simulator, not the dataset), L = sum of *euclidean*
                        step lengths
* ``soft_spl``          max(0, 1 - dtg / d0) * d0 / max(d0, L), every step
* ``path_length``       L (upstream's docstring says geodesic; the code is euclidean and
                        the code is what upstream scores with; the norm is float32 like
                        habitat's); ``path_length_from="euclid64"`` is the same in float64
                        (EXPRESS's own accounting), ``"step_geodesic"`` sums the pose env's
                        per-step geodesics instead (EXPRESS pose protocol)
* ``ndtw``              fastdtw(agent locations, episode.gt_path, euclidean);
                        exp(-dtw / (len(gt) * success_distance)); the location list starts
                        with the start position and skips a step whose position equals
                        the previous one
* ``oracle_success``    1.0 once distance_to_goal < success_distance at any point, reset included
* ``steps_taken``       number of step() calls, STOP included

A benchmark picks the keys it reports. ``SequenceNavMetrics`` is the GOAT-Bench
accounting for a GoalSequence (per-sub-goal restart of d0 and L), transcribed
from goat-bench's ``measures.py`` (GoatSuccess / GoatSPL / GoatSoftSPL).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

NAV_KEYS = ("distance_to_goal", "success", "spl", "soft_spl", "ndtw", "path_length",
            "oracle_success", "steps_taken")
VLN_KEYS = ("distance_to_goal", "success", "spl", "ndtw", "path_length", "oracle_success", "steps_taken")


def _euclid(a, b) -> float:
    """habitat's ``_euclidean_distance`` on float32 agent positions (the norm is
    taken in float32, the running sum in float64 — mirrored exactly)."""
    return float(np.linalg.norm(np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32), ord=2))


def _euclid64(a, b) -> float:
    return float(np.linalg.norm(np.array(b) - np.array(a), ord=2))


def _div(a: float, b: float) -> float:
    """a / b with IEEE semantics (inf / inf = nan, x / 0 = inf|nan) instead of raising."""
    with np.errstate(all="ignore"):
        return float(np.float64(a) / np.float64(b))


def _ratio(d0: float, L: float) -> float:
    """habitat's ``start_end / max(start_end, agent_distance)`` — argument order kept,
    so an inf d0 gives nan and a nan d0 stays nan."""
    return _div(d0, max(d0, L))


def _fastdtw():
    import fastdtw as pkg
    from fastdtw import fastdtw
    if not pkg.fastdtw.__module__.endswith("_fastdtw"):   # pure-Python fallback in use
        warnings.warn(
            "fastdtw is running its pure-Python fallback; nDTW can differ from the habitat-lab "
            "0.1.7 numbers by ~1e-4 (upstream uses the Cython build). Install it with: "
            "pip install cython && pip install --force-reinstall --no-deps --no-cache-dir --no-binary fastdtw --no-build-isolation fastdtw",
            RuntimeWarning, stacklevel=3)
    return fastdtw


class NavMetrics(gym.Wrapper):
    def __init__(self, env: gym.Env, success_distance: float = 3.0, keys: Sequence[str] = NAV_KEYS,
                 path_length_from: str = "euclid") -> None:
        super().__init__(env)
        self._sd = float(success_distance)
        self._keys = tuple(keys)
        unknown = set(self._keys) - set(NAV_KEYS)
        if unknown:
            raise ValueError(f"unknown metric keys {sorted(unknown)}")
        if path_length_from not in ("euclid", "euclid64", "step_geodesic"):
            raise ValueError("path_length_from must be 'euclid', 'euclid64' or 'step_geodesic'")
        self._pl_from = path_length_from
        self._dtw = _fastdtw() if "ndtw" in self._keys else None
        self._m: dict[str, float] = {}
        self._d0 = math.nan
        self._prev: list[float] = []
        self._L = 0.0
        self._steps = 0.0
        self._oracle = 0.0
        self._locations: list[list[float]] = []
        self._gt: list[list[float]] = []

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._m)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pos = list(info["position"])
        dtg = info.get("distance_to_goal")
        self._d0 = math.nan if dtg is None else float(dtg)
        self._prev = pos
        self._L = 0.0
        self._steps = 0.0
        self._oracle = float(self._d0 < self._sd) if not math.isnan(self._d0) else 0.0
        self._locations = []
        self._gt = list(info["episode"].get("gt_path") or []) if "ndtw" in self._keys else []
        self._m = self._compute(pos, self._d0, stop=False)
        return obs, self._annotate(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = list(info["position"])
        dtg = info.get("distance_to_goal")
        dtg = math.nan if dtg is None else float(dtg)
        if self._pl_from == "euclid":
            self._L += _euclid(self._prev, pos)
        elif self._pl_from == "euclid64":
            self._L += _euclid64(self._prev, pos)
        else:
            g = float(info.get("step_geodesic", math.inf))
            if math.isfinite(g):
                self._L += g
        self._prev = pos
        self._steps += 1.0
        if dtg < self._sd:
            self._oracle = 1.0
        self._m = self._compute(pos, dtg, stop=bool(info.get("stop_called", False)))
        return obs, reward, terminated, truncated, self._annotate(info)

    # ---- formulas ------------------------------------------------------------
    def _compute(self, pos: list[float], dtg: float, *, stop: bool) -> dict[str, float]:
        d0, L = self._d0, self._L
        out: dict[str, float] = {}
        success = float(stop and dtg < self._sd)
        for k in self._keys:
            if k == "distance_to_goal":
                out[k] = dtg
            elif k == "success":
                out[k] = success
            elif k == "spl":
                out[k] = success * _ratio(d0, L)                      # habitat: success * d0 / max(d0, L)
            elif k == "soft_spl":
                out[k] = max(0.0, 1.0 - _div(dtg, d0)) * _ratio(d0, L)
            elif k == "ndtw":
                out[k] = self._ndtw_update(pos)
            elif k == "path_length":
                out[k] = L
            elif k == "oracle_success":
                out[k] = self._oracle
            elif k == "steps_taken":
                out[k] = self._steps
        return out

    def _ndtw_update(self, pos: list[float]) -> float:
        if self._locations and pos == self._locations[-1]:
            return self._m.get("ndtw", 0.0)
        self._locations.append(pos)
        if not self._gt:
            return math.nan
        dtw_distance = self._dtw(self._locations, self._gt, dist=_euclid64)[0]
        return float(np.exp(-dtw_distance / (len(self._gt) * self._sd)))

    def _annotate(self, info: dict[str, Any]) -> dict[str, Any]:
        info["metrics"] = dict(self._m)
        info["metrics_valid"] = math.isfinite(self._d0)
        return info


_MODALITY = {"object": "object", "image": "image", "description": "description"}


class SequenceNavMetrics(gym.Wrapper):
    """GOAT-Bench accounting over a GoalSequence.

    Per sub-goal *i* (closed by SUBTASK_STOP): ``subtask_success[i] =
    prev_dtg < success_distance`` where prev_dtg is the distance to the sub-goal
    being closed at the stopping pose (the fact of the step *before* the
    switch); ``spl_by_subtask[i] = subtask_success[i] * se / max(se, walked)``
    with ``se`` the distance to that sub-goal from where it started and
    ``walked`` the euclidean path since — both restart at every switch.
    ``success`` = partial_success (successes / sub-goals), ``spl`` =
    composite_spl (mean of per-sub-goal SPL), ``soft_spl`` = composite_softspl
    (accumulated only on closing steps). ``task_success`` additionally needs
    STOP; ``{object,image,description}_success`` are per-modality rates,
    short-circuited to 0.0 when that modality has no success (upstream quirk:
    an absent modality reads the same as a failed one — the ``n_*_subtasks``
    counts tell them apart).
    """

    def __init__(self, env: gym.Env, success_distance: float = 0.25) -> None:
        super().__init__(env)
        self._sd = float(success_distance)
        self._m: dict[str, Any] = {}

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._m)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        goals = info["episode"]["goal"]["goals"]
        self._mods = [_MODALITY[g["kind"]] for g in goals]
        self._n = len(goals)
        self._counts = {m: self._mods.count(m) for m in _MODALITY.values()}
        self._succ = [0.0] * self._n
        self._spl = [0.0] * self._n
        self._succ_by = {m: 0.0 for m in _MODALITY.values()}
        self._spl_by = {m: 0.0 for m in _MODALITY.values()}
        self._soft_by = {m: 0.0 for m in _MODALITY.values()}
        dtg = info.get("distance_to_goal")
        self._dtg = math.nan if dtg is None else float(dtg)
        self._prev_dtg = self._dtg
        self._se = self._dtg
        self._walked = 0.0
        self._prev = list(info["position"])
        self._idx = 0
        self._steps = 0.0
        self._stop = False
        self._m = self._compute()
        return obs, self._annotate(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = list(info["position"])
        self._walked += _euclid(self._prev, pos)
        self._prev = pos
        self._steps += 1.0
        self._stop = bool(info.get("stop_called", False))
        dtg = info.get("distance_to_goal")
        new_dtg = self._dtg if dtg is None else float(dtg)
        if info.get("subtask_closed") and self._idx < self._n:
            i, mod = self._idx, self._mods[self._idx]
            ok = float(self._prev_dtg < self._sd)
            self._succ[i] = ok
            self._succ_by[mod] += ok
            se, walked = self._se, self._walked
            spl = ok * _ratio(se, walked)
            self._spl[i] = spl
            self._spl_by[mod] += spl
            soft = max(0.0, 1.0 - _div(self._prev_dtg, se)) * _ratio(se, walked)
            self._soft_by[mod] += soft
            self._idx += 1
            self._se = new_dtg          # the next sub-goal, measured from where the agent stands
            self._walked = 0.0
        self._prev_dtg = new_dtg
        self._dtg = new_dtg
        self._m = self._compute()
        return obs, reward, terminated, truncated, self._annotate(info)

    def _rate(self, num: dict[str, float], mod: str) -> float:
        return num[mod] / self._counts[mod] if num[mod] else 0.0

    def _compute(self) -> dict[str, Any]:
        n = float(self._n)
        all_ok = sum(self._succ) == self._n
        return {
            "success": sum(self._succ_by.values()) / n,
            "spl": sum(self._spl_by.values()) / n,
            "soft_spl": sum(self._soft_by.values()) / n,
            "composite_success": float(all_ok),
            "task_success": float(all_ok and self._stop),
            "subtask_success": list(self._succ),
            "spl_by_subtask": list(self._spl),
            "object_success": self._rate(self._succ_by, "object"),
            "image_success": self._rate(self._succ_by, "image"),
            "description_success": self._rate(self._succ_by, "description"),
            "n_object_subtasks": self._counts["object"],
            "n_image_subtasks": self._counts["image"],
            "n_description_subtasks": self._counts["description"],
            "distance_to_target": self._dtg if math.isfinite(self._dtg) else None,
            "n_subtasks": self._n,
            "subtasks_completed": self._idx,
            "steps_taken": self._steps,
        }

    def _annotate(self, info: dict[str, Any]) -> dict[str, Any]:
        info["metrics"] = dict(self._m)
        info["metrics_valid"] = True
        return info

