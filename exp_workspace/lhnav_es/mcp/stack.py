"""stack — the SLAM environment stack on EmbodiedScore-envs.

    base    = build_env(line, split, body=FINE, actions=STANDARD)   # the package body at micro quanta
    slam    = SlamMicroStep(base, instruments=...)                   # coarse action -> MICRO_PER_ACTION fine steps, each fed
    metrics = <the line's metric wrapper>(slam)                      # counts COARSE steps
    env     = TimeLimit(metrics, 500)                                # coarse budget

FINE is the STANDARD preset with its three quanta divided by
``MICRO_PER_ACTION`` (0.25 m -> 0.05 m, 15° -> 3°, 30° -> 6°) and a depth
camera at the RGB resolution (ORB-SLAM3 RGBD wants an aligned pair); depth
stays METRIC (no DepthClip — the SLAM feed and the agent's frame are the
same array). The clamp (±60°), agent cylinder, sliding, navmesh file and
camera mast are the STANDARD body's, so a coarse action lands exactly
where the bare STANDARD step would (turns compose exactly; a forward is
five collinear 0.05 m tries).

What the wrapper adds to ``info`` per coarse step: ``action``,
``micro_steps_done``, ``collision`` (a forward that did not
move — the remaining micro-steps are aborted and the map marks an obstacle
ahead), ``collisions`` (episode count), ``slam`` (tracking status),
``reanchors``, ``tick`` (micro clock), ``pitch_deg`` and ``tilt_clamped`` on
a LOOK action. STOP / SUBTASK_STOP pass through as ONE base step.

The SLAM session (``SlamEnv``) is rebuilt at every reset — the occupancy
grid and the ORB-SLAM3 anchor belong to the episode — and bootstrapped with
the disclosed 360° scan when instruments are on. The simulator itself is
the package's and persists across episodes.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import embodiedscore_envs as es
import gymnasium as gym
from embodiedscore_envs.benchmarks import build_env
from embodiedscore_envs.benchmarks.env import Act, CameraSpec
from embodiedscore_envs.benchmarks.presets import actions, bodies
from gymnasium.wrappers import TimeLimit

from ._body import WorldBody
from ._slam_env import MICRO_PER_ACTION, SlamEnv

COARSE_STEP_M = bodies.STANDARD.forward_step_m       # 0.25
COARSE_TURN_DEG = bodies.STANDARD.turn_deg           # 15
COARSE_TILT_DEG = bodies.STANDARD.tilt_deg           # 30
PITCH_LIMIT_DEG = bodies.STANDARD.tilt_limit_deg     # 60
MAX_COARSE_STEPS = 500
CELL_SIZE_M = 0.10
MAP_SIZE_M = 48.0

FINE = dataclasses.replace(
    bodies.STANDARD,
    forward_step_m=COARSE_STEP_M / MICRO_PER_ACTION,
    turn_deg=COARSE_TURN_DEG / MICRO_PER_ACTION,
    tilt_deg=COARSE_TILT_DEG / MICRO_PER_ACTION,
    depth=CameraSpec(bodies.STANDARD.rgb.width, bodies.STANDARD.rgb.height,
                     bodies.STANDARD.rgb.hfov_deg, bodies.STANDARD.rgb.position),
)


# the keys an experiment file may set under env.slam (configs/*.yaml); the
# module constants above are their defaults where they have one
SLAM_KEYS = frozenset({
    "cell_size_m", "map_size_m", "max_coarse_steps", "frontier_match_m",
    "start_pitch_deg", "map_mode", "mapper", "render",
    "success_distance",
})


class SlamMicroStep(gym.Wrapper):
    """Coarse actions in, MICRO_PER_ACTION fine steps + auto-fed SLAM inside."""

    def __init__(self, env: gym.Env, instruments: int = 0, slam: dict[str, Any] | None = None) -> None:
        super().__init__(env)
        self.instruments = int(instruments)
        self.slam_cfg: dict[str, Any] = dict(slam or {})
        unknown = set(self.slam_cfg) - SLAM_KEYS
        if unknown:
            raise ValueError(f"env.slam: unknown key(s) {sorted(unknown)} (known: {sorted(SLAM_KEYS)})")
        self.slam: SlamEnv | None = None
        self.boot: dict | None = None
        self.keep_map_next = False        # the next reset continues a tour (see reset)

    def _slam_kwargs(self, base: Any) -> dict[str, Any]:
        """SlamEnv's kwargs: the stack's quanta + the env block's ``slam``
        keys. An absent key = the module constant (cell_size_m /
        map_size_m) or SlamEnv's own default (frontier_match_m /
        start_pitch_deg / mapper); ``map_mode`` null = v2 with
        instruments=2, v1 otherwise. ``v3`` is v2's floor layers over evidence
        rather than verdicts, selected only by an explicit ``map_mode``."""
        s = self.slam_cfg
        kw: dict[str, Any] = dict(
            coarse_step_m=COARSE_STEP_M, coarse_turn_deg=COARSE_TURN_DEG,
            cell_size=float(s.get("cell_size_m") or CELL_SIZE_M),
            map_size_m=float(s.get("map_size_m") or MAP_SIZE_M),
            pose_source="slam" if self.instruments else "gt",
            map_mode=str(s.get("map_mode") or ("v2" if self.instruments == 2 else "v1")),
            coarse_tilt_deg=COARSE_TILT_DEG if base.body.has_tilt else None,
            pitch_limit_deg=PITCH_LIMIT_DEG,
        )
        for key in ("frontier_match_m", "start_pitch_deg"):
            if s.get(key) is not None:
                kw[key] = float(s[key])
        if s.get("mapper"):
            kw["mapper"] = dict(s["mapper"])
        return kw

    # ── episode ──

    def reset(self, **kwargs: Any):
        """Place the package episode and give it a SLAM session.

        ``keep_map_next`` is the tour case (IVLN-CE): the package
        walks the body to the next episode's start with its own oracle instead
        of teleporting it, so the map, the markers and the world frame are what
        carries — rebuilding them here would leave a tour that is a tour only in
        the metrics. It is a flag rather than an argument because gymnasium's
        TimeLimit takes no kwargs of its own, and because the tour wrapper
        resets this one AGAIN from inside its own reset (it walks the oracle
        around it), so the decision has to survive being passed through twice.
        All this method then does is not tear the session down; re-finding the
        body waits for ``continue_tour``, once the body has actually arrived.
        """
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        if self.keep_map_next and self.slam is not None:
            info.update(self._slam_facts())
            return obs, info
        base = self.env.unwrapped
        if self.slam is not None:
            self.slam.close()          # ends the ORB-SLAM3 container; the simulator stays
        self.slam = SlamEnv(WorldBody(base), **self._slam_kwargs(base))
        self.slam.reset()
        self.boot = self.slam.bootstrap() if self.instruments else None
        info.update(self._slam_facts())
        if self.boot is not None:
            info["bootstrap"] = dict(self.boot)
        return obs, info

    def walk_transit(self, advance) -> dict | None:
        """Drive the IR2R-CE transit with the tracker fed at every step
        (``SlamEnv.walk_transit``); None when there is no SLAM session."""
        if self.slam is None or not self.instruments:
            return None
        return self.slam.walk_transit(advance)

    def continue_tour(self, spin: bool = True, walked: dict | None = None) -> dict | None:
        """Re-find the body after a transit or a relocation, the placement now settled.

        Kept out of ``reset`` on purpose: at the inner reset the body may not
        be at the new episode's start yet, and a relocalisation turn taken
        there would measure the wrong place."""
        self.keep_map_next = False
        if self.slam is None or not self.instruments:
            self.boot = None
            return None
        self.boot = self.slam.continue_tour(spin=spin, walked=walked)
        return dict(self.boot)

    def reset_slam(self) -> dict | None:
        """The episodic setting (``persist_map: false``): a fresh SLAM session
        (occupancy map, ORB-SLAM3 anchor, markers) where the body stands, the
        package episode untouched. Same construction as ``reset``, minus the
        env reset; the bootstrap scan is the disclosed 360° in-place spin."""
        if self.slam is not None:
            self.slam.close()
        base = self.env.unwrapped
        self.slam = SlamEnv(WorldBody(base), **self._slam_kwargs(base))
        self.slam.reset()
        self.boot = self.slam.bootstrap() if self.instruments else None
        return dict(self.boot) if self.boot else None

    def close(self) -> None:
        if self.slam is not None:
            self.slam.close()
            self.slam = None
        self.env.close()

    # ── the only way the world moves ──

    def step(self, action: int):
        a = int(action)
        se = self.slam
        if se is None:
            raise RuntimeError("SlamMicroStep: reset() first")
        if a in (Act.STOP, Act.SUBTASK_STOP):
            obs, reward, terminated, truncated, info = self.env.step(a)
            info = dict(info)
            info.update({"action": a, "micro_steps_done": 0, "collision": False})
            info.update(self._slam_facts())
            return obs, reward, terminated, truncated, info
        # a movement or LOOK action: SlamEnv drives the micro-steps through
        # WorldBody.step == HabitatEnv.step on the FINE body, feeding SLAM
        # after each; the last micro-step's gym tuple is what goes outward
        act = se.act(a)
        if se.env.last is None:        # only a LOOK clamped to a no-op before any other step could get here
            obs, reward, terminated, truncated, info = self.env.step(a)
        else:
            # a clamped LOOK that did nothing (micro_steps_done 0) is charged
            # like any coarse action and re-emits the previous frame/facts
            obs, reward, terminated, truncated, info = se.env.last
        info = dict(info)
        info.update({
            "action": a,
            "micro_steps_done": int(act.get("micro_steps_done", 0)),
            "collision": bool(act.get("collision", False)),
        })
        if "pitch_deg" in act:
            info["pitch_deg"] = float(act["pitch_deg"])
            info["tilt_clamped"] = bool(act.get("tilt_clamped", False))
        info.update(self._slam_facts())
        return obs, reward, terminated, truncated, info

    def body_facts(self) -> dict:
        """The map's readings for the recovery controller (``SlamEnv.body_facts``):
        explored area, planning layer, floor mismatch. Only on the instrumented
        arm — a bare run's map is drawn from the simulator's pose, and nothing
        read off it may steer the harness."""
        if self.slam is None or not self.instruments:
            return {}
        return self.slam.body_facts()

    def _slam_facts(self) -> dict:
        se = self.slam
        if se is None:
            return {}
        return {"slam": se.status, "reanchors": int(se.reanchors), "tick": int(se.tick),
                "collisions": int(se.env.collisions), **self.body_facts()}


def build(line: str, split: str, *, instruments: int = 0, data_root: str | None = None,
          scene_root: str | None = None, gpu_id: int = 0, max_coarse_steps: int | None = None,
          slam: dict[str, Any] | None = None, success_distance: float | None = None) -> gym.Env:
    """The SLAM stack for a benchmark line (standard task semantics,
    FINE body): TimeLimit(<line metrics>(SlamMicroStep(build_env(...)))).
    ``slam`` = the env block's ``slam`` dict (SLAM_KEYS); ``max_coarse_steps``
    None = its ``max_coarse_steps`` or MAX_COARSE_STEPS.

    ``success_distance`` is the radius a sub-goal is scored on, in metres: the
    package's GOAT declares **0.25 m** to a view point, while GOAT-Bench's own
    threshold is **1.0 m** — four times apart. None = the package's own
    default, so nothing changes silently; the GOAT experiment file sets 1.0 to
    score as GOAT-Bench does."""
    slam = dict(slam or {})
    # It rides in the env block's `slam` dict like the other stack knobs, and
    # is popped here because SlamMicroStep has no business with a scoring
    # radius.
    success_distance = slam.pop("success_distance", success_distance)
    if max_coarse_steps is None:
        max_coarse_steps = int(slam.get("max_coarse_steps") or MAX_COARSE_STEPS)
    b = es.benchmark(line, "standard")
    # the line's OWN action table (GOAT = STANDARD + SUBTASK_STOP); actions.STANDARD
    # on every single-goal line
    table = b.actions if len(b.actions) > len(actions.STANDARD) else actions.STANDARD
    base = build_env(b.name, split, data_root=data_root, scene_root=scene_root, body=FINE,
                     actions=table, gpu_id=gpu_id)
    env: gym.Env = SlamMicroStep(base, instruments=instruments, slam=slam)
    if b.metrics is not None:
        env = (b.metrics(env) if success_distance is None
               else b.metrics(env, success_distance=float(success_distance)))
    return TimeLimit(env, max_episode_steps=int(max_coarse_steps))


__all__ = [
    "COARSE_STEP_M",
    "COARSE_TILT_DEG",
    "COARSE_TURN_DEG",
    "FINE",
    "MAX_COARSE_STEPS",
    "PITCH_LIMIT_DEG",
    "SLAM_KEYS",
    "SlamMicroStep",
    "build",
]
