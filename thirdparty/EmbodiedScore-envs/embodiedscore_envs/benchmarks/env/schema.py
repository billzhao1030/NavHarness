"""The vocabulary every benchmark is written in (layer L1, pure data).

* :class:`Act` — the global action ids. A benchmark's action table is a prefix
  of this sequence, so the integer an agent sends means the same thing in every
  environment of the package.
* Goals — what an episode asks the agent to reach, as typed data. The body
  derives ``distance_to_goal`` from :func:`targets_of`; the metric wrappers
  never look inside a goal beyond its ``kind``.
* :class:`Episode` — one evaluation unit: where it starts, what its goal is,
  what the dataset says about it. Poses are in the engine's own frame — the
  frame the split files store them in: y up with ``(x, y, z, w)`` quaternions.
* :class:`Benchmark` — a declaration (no code of its own) that ``make()`` turns
  into a Gymnasium stack; ``engine`` names the simulator it runs on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

from .sim import Body, SceneRef

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]     # habitat: x, y, z, w

ENGINES = ("habitat",)


class Act(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5
    SUBTASK_STOP = 6


def action_prefix(n: int) -> tuple[Act, ...]:
    """The first ``n`` actions: (STOP,) (STOP, FORWARD, LEFT, RIGHT), ..."""
    return tuple(Act(i) for i in range(n))


# ---- goals -------------------------------------------------------------------

@dataclass(frozen=True)
class PointGoal:
    position: Vec3
    radius: float = 3.0
    kind: str = field(default="point", init=False)


@dataclass(frozen=True)
class ObjectInstance:
    object_id: str
    category: str
    position: Vec3
    view_points: tuple[Vec3, ...]        # agent positions from which the object counts as reached


@dataclass(frozen=True)
class ObjectGoal:
    """Reach any instance of ``category`` in the scene (GOAT-object)."""
    category: str
    instances: tuple[ObjectInstance, ...]
    kind: str = field(default="object", init=False)


@dataclass(frozen=True)
class ImageGoal:
    """Reach the instance shown from a stored camera pose (GOAT-image)."""
    instance: ObjectInstance
    camera_position: Vec3
    camera_rotation: Quat
    hfov_deg: float
    image_size: tuple[int, int]          # (height, width)
    kind: str = field(default="image", init=False)


@dataclass(frozen=True)
class TextGoal:
    """Reach the instance a description refers to (GOAT-description)."""
    instance: ObjectInstance
    description: str
    kind: str = field(default="description", init=False)


@dataclass(frozen=True)
class GoalSequence:
    """Ordered sub-goals, reached one at a time; on GOAT the agent closes
    each with SUBTASK_STOP."""
    goals: tuple["Goal", ...]
    kind: str = field(default="sequence", init=False)


Goal = Union[PointGoal, ObjectGoal, ImageGoal, TextGoal, GoalSequence]


def targets_of(goal: Goal | None, dtype: Any = np.float32) -> np.ndarray | None:
    """The (N, 3) point set ``distance_to_goal`` is measured to (None for no
    goal). float32 is habitat's working precision."""
    if goal is None:
        return None
    if isinstance(goal, PointGoal):
        return np.asarray([goal.position], dtype=dtype)
    if isinstance(goal, ObjectGoal):
        pts = [vp for inst in goal.instances for vp in inst.view_points]
        return np.asarray(pts, dtype=dtype).reshape(-1, 3)
    if isinstance(goal, (ImageGoal, TextGoal)):
        return np.asarray(goal.instance.view_points, dtype=dtype).reshape(-1, 3)
    if isinstance(goal, GoalSequence):
        raise TypeError("targets_of(GoalSequence): pass the current sub-goal")
    raise TypeError(f"not a goal: {goal!r}")


def to_dict(obj: Any) -> Any:
    """dataclasses -> plain dicts (goals keep their ``kind``); tuples -> lists."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---- episode -----------------------------------------------------------------

@dataclass(frozen=True)
class Episode:
    index: int                           # position in the loaded list
    episode_id: str
    scene: SceneRef
    start_position: Vec3 | None          # the engine's frame (habitat y up)
    start_rotation: Quat | None          # habitat x, y, z, w
    goal: Goal | None
    instruction: str | None = None       # VLN: what to follow
    reference_path: tuple[Vec3, ...] | None = None   # VLN: waypoints from the dataset
    gt_path: tuple[Vec3, ...] | None = None          # VLN: dense oracle locations (nDTW reference)
    info: dict[str, Any] = field(default_factory=dict)   # everything else the dataset says

    def as_dict(self) -> dict[str, Any]:
        return to_dict(self)


# ---- benchmark declaration ---------------------------------------------------

@dataclass(frozen=True)
class DepthSpec:
    """How depth leaves the stack: clip to [min, max] metres, optionally scaled to [0, 1]."""
    min_m: float
    max_m: float
    normalize: bool


@dataclass(frozen=True)
class Benchmark:
    name: str                                    # make() key, e.g. "vlnce-r2r" / "vlnce-r2r-upstream"
    gym_id: str                                  # e.g. "EmbodiedScore/VLNCE-R2R-v0"
    # the engine's Body type, checked against ``engine`` in __post_init__
    body: Body
    actions: tuple[Act, ...]
    splits: tuple[str, ...]
    episodes: Callable[..., list[Episode]]      # (split, data_root=None, scene_root=None, **kw) -> episodes
    metrics: Callable[..., Any] | None = None    # (env, **overrides) -> wrapped env
    depth: DepthSpec | None = None
    max_episode_steps: int | None = None         # static budget -> gym TimeLimit
    budget: Callable[[Any, Episode], int] | None = None   # (world, episode) -> per-episode budget
    dtg_policy: str = "when_moved"               # "when_moved" (habitat-lab stock) | "every_step" (goat-bench)
    truncate_at_budget: bool = False             # DynamicTimeLimit on info["step_budget"]
    engine: str = "habitat"                      # "habitat" (habitat-sim, in process)
    description: str = ""
    line: str = ""                               # the benchmark line both variants belong to, e.g. "vlnce-r2r"
    variant: str = "standard"                    # "standard" (EmbodiedScore's shared body) | "upstream" (the line's own evaluator)

    def __post_init__(self) -> None:
        if self.variant not in ("standard", "upstream"):
            raise ValueError(f"{self.name}: variant must be 'standard' or 'upstream'")
        if self.engine not in ENGINES:
            raise ValueError(f"{self.name}: engine must be one of {ENGINES}")
        want = {"habitat": Body}[self.engine]
        if not isinstance(self.body, want):
            raise TypeError(f"{self.name}: engine {self.engine!r} needs a {want.__name__}, got {type(self.body).__name__}")
        if not self.line:
            object.__setattr__(self, "line", self.name[: -len("-upstream")] if self.name.endswith("-upstream") else self.name)


# ---- data roots ----------------------------------------------------------------

def data_root(explicit: str | os.PathLike | None = None) -> Path:
    p = explicit or os.environ.get("EMBODIEDSCORE_DATA_ROOT")
    if not p:
        raise ValueError("data_root not given and EMBODIEDSCORE_DATA_ROOT unset")
    return Path(p)


def scene_root(explicit: str | os.PathLike | None = None) -> Path:
    p = explicit or os.environ.get("EMBODIEDSCORE_SCENE_ROOT")
    if not p:
        raise ValueError("scene_root not given and EMBODIEDSCORE_SCENE_ROOT unset")
    return Path(p)
