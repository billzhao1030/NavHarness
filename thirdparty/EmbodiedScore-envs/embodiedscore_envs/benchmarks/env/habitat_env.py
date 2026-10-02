"""HabitatEnv — the environment body of the habitat engine (layer L1), a plain
``gymnasium.Env``.

What it does: hold an episode list and a Body, put the agent at an episode's
start, execute one action per ``step`` (a movement, STOP, or SUBTASK_STOP),
and report facts — observation + position / rotation / heading / pitch /
collided / distance_to_goal / stop_called / goal_index.

What it deliberately does not do: count steps or enforce a budget
(``TimeLimit`` / ``DynamicTimeLimit``), compute any metric (``NavMetrics`` and
friends), clip or normalise depth (``DepthClip``), keep a "next episode"
cursor (episodes are chosen through ``reset(options=...)`` or sampled with the
env's RNG), know which benchmark it is running, or put the goal into the
observation (goals are episode data and travel in ``info``).

``reward`` is always 0.0: these are evaluation environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .schema import Act, Episode, GoalSequence, targets_of, to_dict
from .sim import Body

if TYPE_CHECKING:
    from .sim.habitat.world import SimWorld

_DISCRETE_DEFAULT = (Act.STOP, Act.FORWARD, Act.LEFT, Act.RIGHT)


class HabitatEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        episodes: Sequence[Episode],
        body: Body,
        actions: Sequence[Act] = _DISCRETE_DEFAULT,
        budget: Callable[[SimWorld, Episode], int] | None = None,
        dtg_policy: str = "when_moved",
        gpu_id: int = 0,
        sim_seed: int = 42,
        render_mode: str | None = None,
    ) -> None:
        """``dtg_policy`` says when ``distance_to_goal`` is re-queried: ``"when_moved"``
        (habitat-lab's DistanceToGoal: only when the position changed by more than
        1e-4, else the last value stands) or ``"every_step"`` (goat-bench's). This
        matters: habitat-sim's MultiGoalShortestPath prunes with bounds carried
        over from the previous query, so the value depends on the query history
        and the legacy call pattern has to be reproduced exactly."""
        if dtg_policy not in ("when_moved", "every_step"):
            raise ValueError("dtg_policy must be 'when_moved' or 'every_step'")
        self._dtg_policy = dtg_policy
        self._dtg_last: tuple[np.ndarray, int, float | None] | None = None   # (position, goal_index, value)
        if not episodes:
            raise ValueError("HabitatEnv: no episodes")
        self.episodes: tuple[Episode, ...] = tuple(episodes)
        self.body = body
        self.actions: tuple[Act, ...] = tuple(Act(a) for a in actions)
        if self.actions != tuple(Act(i) for i in range(len(self.actions))):
            raise ValueError(f"actions must be a prefix of the Act order, got {self.actions}")
        if (Act.LOOK_UP in self.actions or Act.LOOK_DOWN in self.actions) and not body.has_tilt:
            raise ValueError("LOOK actions need a Body with tilt_deg")
        self._budget = budget
        self.render_mode = render_mode
        self._by_id = {e.episode_id: e for e in self.episodes}
        from .sim.habitat.world import SimWorld   # the habitat_sim importer, loaded only for a habitat body
        self._world = SimWorld(body, gpu_id=gpu_id, seed=sim_seed)   # simulator is created on first reset
        self._episode: Episode | None = None
        self._goal_index = 0
        self._stop_called = False
        self._done = False
        self._last_rgb: np.ndarray | None = None

        obs = {"rgb": spaces.Box(0, 255, shape=(body.rgb.height, body.rgb.width, 3), dtype=np.uint8)}
        if body.depth is not None:
            obs["depth"] = spaces.Box(0.0, np.inf, shape=(body.depth.height, body.depth.width, 1), dtype=np.float32)
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Discrete(len(self.actions))

    # ---- episode / goal --------------------------------------------------------
    @property
    def episode(self) -> Episode:
        if self._episode is None:
            raise RuntimeError("HabitatEnv: call reset() first")
        return self._episode

    @property
    def goal_index(self) -> int:
        return self._goal_index

    def current_goal(self):
        """The goal in force now: the sequence's cursor for a GoalSequence,
        else the episode goal; None once a sequence is exhausted."""
        g = self.episode.goal
        if isinstance(g, GoalSequence):
            return g.goals[self._goal_index] if self._goal_index < len(g.goals) else None
        return g

    def _select(self, options: dict[str, Any] | None) -> Episode:
        options = options or {}
        if "episode" in options:
            return self.episodes[int(options["episode"])]
        if "episode_id" in options:
            return self._by_id[str(options["episode_id"])]
        return self.episodes[int(self.np_random.integers(len(self.episodes)))]

    # ---- gym API -------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        ep = self._select(options)
        self._world.load_scene(ep.scene)
        self._episode = ep
        self._goal_index = 0
        self._stop_called = False
        self._done = False
        self._dtg_last = None
        self._place_start()
        obs = self._observe()
        info = self._facts(collided=False)
        info["episode"] = ep.as_dict()
        info["goal"] = to_dict(self.current_goal())
        if self._budget is not None:
            info["step_budget"] = int(self._budget(self._world, ep))
        return obs, info

    def _place_start(self) -> None:
        self._world.place(self.episode.start_position, self.episode.start_rotation)

    def step(self, action: int):
        if self._done:
            raise RuntimeError("HabitatEnv: episode is over; reset() first")
        a = Act(int(action))
        if a not in self.actions:
            raise ValueError(f"action {a!r} not in this environment's table {self.actions}")
        collided = False
        subtask_closed = False
        if a == Act.STOP:
            self._stop_called = True
            self._done = True
        elif a == Act.SUBTASK_STOP:
            if not isinstance(self.episode.goal, GoalSequence):
                raise ValueError("SUBTASK_STOP needs a GoalSequence goal")
            self._goal_index += 1
            subtask_closed = True
            if self._goal_index >= len(self.episode.goal.goals):
                self._done = True
        else:
            collided = self._world.move(int(a))
        obs = self._observe()
        info = self._facts(collided=collided)
        if subtask_closed:
            info["subtask_closed"] = True
            info["goal"] = to_dict(self.current_goal())
        return obs, 0.0, self._done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return None if self._last_rgb is None else self._last_rgb.copy()
        return None

    def close(self) -> None:
        self._world.close()

    # ---- facts ---------------------------------------------------------------------
    def _observe(self) -> dict[str, np.ndarray]:
        obs = self._world.observe()
        self._last_rgb = obs["rgb"]
        return obs

    def _distance_to_goal(self, pos: np.ndarray) -> float | None:
        last = self._dtg_last
        if (self._dtg_policy == "when_moved" and last is not None and last[1] == self._goal_index
                and np.allclose(last[0], pos, atol=1e-4)):
            return last[2]
        targets = targets_of(self.current_goal())
        value = None if targets is None or len(targets) == 0 else \
            self._world.geodesic(pos, targets, key=(self.episode.index, self._goal_index))
        self._dtg_last = (np.array(pos, dtype=np.float32), self._goal_index, value)
        return value

    def _facts(self, *, collided: bool) -> dict[str, Any]:
        pos, rot = self._world.pose()
        return {
            "position": pos.tolist(),
            "rotation": rot.tolist(),
            "heading": self._world.heading(),
            "pitch": self._world.pitch(),
            "collided": bool(collided),
            "distance_to_goal": self._distance_to_goal(pos),
            "stop_called": self._stop_called,
            "goal_index": self._goal_index,
        }

    @property
    def world(self) -> SimWorld:
        """The engine facade — for wrappers and tools that need geodesics, the
        pathfinder, extra renders, or (IVLN-CE transits) to move the body
        without stepping the environment."""
        return self._world

