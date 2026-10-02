"""SimWorld — the engine facade (layer L0).

Owns exactly one ``habitat_sim.Simulator`` and exposes the primitives every
benchmark is built from: load_scene / place / teleport / move / pose /
heading / pitch / observe / render_at / geodesic / snap /
random_navigable_near / bounds / follower / close. It has no notion of
episode, goal, success or metrics; those live above it. Every number that
defines the embodiment comes from a :class:`Body`.

Coordinate conventions are habitat-sim's: metres, y up, the agent faces -z.
Rotations cross this boundary as ``[x, y, z, w]`` quaternion coefficients
(the order the datasets store ``start_rotation`` in). The agent node carries
yaw only; camera pitch (the initial ``Body.camera_pitch_deg`` and the LOOK
actions) lives on the sensor nodes — the same composition habitat-lab's
look_up / look_down use, and the same camera pose explore-eqa gets by
pitching the agent.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .body import Body, CameraSpec, NavMesh
from .scene import SceneRef

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs, quat_to_coeffs
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "embodiedscore_envs needs habitat-sim 0.3.3; build it from "
        "https://github.com/Embodied-Agent-Squad/EmbodiedScore-habitat (BUILD.md)"
    ) from e

# Movement action keys (= the global Act ids; STOP / SUBTASK_STOP never reach SimWorld).
FORWARD, LEFT, RIGHT, LOOK_UP, LOOK_DOWN = 1, 2, 3, 4, 5
_ACTION_NAMES = {FORWARD: "move_forward", LEFT: "turn_left", RIGHT: "turn_right",
                 LOOK_UP: "look_up", LOOK_DOWN: "look_down"}

FollowerError = habitat_sim.errors.GreedyFollowerError


def _swing(position: Any, pitch_deg: float) -> list[float]:
    """Where a camera mounted at ``position`` on a body pitched by ``pitch_deg``
    about x ends up (explore-eqa's agent-level pitch)."""
    p = math.radians(pitch_deg)
    x, y, z = position
    return [float(x), float(y * math.cos(p) - z * math.sin(p)), float(y * math.sin(p) + z * math.cos(p))]


def _camera(uuid: str, kind: Any, spec: CameraSpec, pitch_deg: float, swing: bool) -> Any:
    s = habitat_sim.CameraSensorSpec()
    s.uuid = uuid
    s.sensor_type = kind
    s.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    s.resolution = [spec.height, spec.width]
    s.hfov = spec.hfov_deg
    s.position = _swing(spec.position, pitch_deg) if swing else list(spec.position)
    s.orientation = [math.radians(pitch_deg), 0.0, 0.0]
    return s


def _vec3(p: Any) -> np.ndarray:
    return np.asarray(p, dtype=np.float32).reshape(3)


def _yaw_of(q: Any) -> float:
    """Yaw (habitat-lab HeadingSensor convention: 0 facing -z, left positive) of a quaternion."""
    v = (q * np.quaternion(0, 0, 0, -1) * q.inverse()).imag
    return float(math.atan2(-v[0], -v[2]))


class Follower:
    """A greedy geodesic follower (habitat-sim's C++ GreedyGeodesicFollower —
    what habitat-lab's ShortestPathFollower wraps). ``next_action(goal)``
    returns FORWARD / LEFT / RIGHT, ``None`` once within ``goal_radius``, and
    raises :class:`FollowerError` when it cannot make progress."""

    def __init__(self, sim: Any, goal_radius: float) -> None:
        self._impl = sim.make_greedy_follower(
            0, float(goal_radius), stop_key=0, forward_key=FORWARD, left_key=LEFT, right_key=RIGHT)
        self.goal_radius = float(goal_radius)

    def next_action(self, goal: Any) -> int | None:
        a = self._impl.next_action_along(_vec3(goal))
        return None if a in (0, None) else int(a)


class SimWorld:
    def __init__(self, body: Body, gpu_id: int = 0, seed: int = 42) -> None:
        self.body = body
        self.gpu_id = int(gpu_id)
        self.seed = int(seed)
        self._sim: Any = None
        self._agent: Any = None
        self._scene: SceneRef | None = None
        self._last_obs: dict[str, Any] | None = None
        self._path_cache: tuple[Any, Any] | None = None   # (key, MultiGoalShortestPath)
        self._followers: dict[float, Follower] = {}
        # teleport locomotion keeps its own float64 pose, as explore-eqa does
        self._tp_pos: np.ndarray | None = None
        self._tp_yaw = 0.0
        self._tp_pitch = 0.0

    # ---- lifecycle ---------------------------------------------------------
    def _configuration(self, scene: SceneRef) -> Any:
        b = self.body
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene.scene_file
        if scene.dataset_config:
            sim_cfg.scene_dataset_config_file = scene.dataset_config
        sim_cfg.gpu_device_id = self.gpu_id
        sim_cfg.allow_sliding = b.allow_sliding
        sim_cfg.enable_physics = False
        sim_cfg.random_seed = self.seed
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = b.agent_height_m
        agent_cfg.radius = b.agent_radius_m
        sensors = [_camera("rgb", habitat_sim.SensorType.COLOR, b.rgb, b.camera_pitch_deg, b.pitch_moves_camera)]
        if b.depth is not None:
            sensors.append(_camera("depth", habitat_sim.SensorType.DEPTH, b.depth, b.camera_pitch_deg, b.pitch_moves_camera))
        agent_cfg.sensor_specifications = sensors
        act, amt = habitat_sim.agent.ActionSpec, habitat_sim.agent.ActuationSpec
        space = {
            FORWARD: act("move_forward", amt(amount=b.forward_step_m)),
            LEFT: act("turn_left", amt(amount=b.turn_deg)),
            RIGHT: act("turn_right", amt(amount=b.turn_deg)),
        }
        if b.has_tilt:
            space[LOOK_UP] = act("look_up", amt(amount=b.tilt_deg, constraint=b.tilt_limit_deg))
            space[LOOK_DOWN] = act("look_down", amt(amount=b.tilt_deg, constraint=b.tilt_limit_deg))
        agent_cfg.action_space = space
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def load_scene(self, scene: SceneRef) -> None:
        """Make ``scene`` the active scene (no-op if it already is) and obtain
        its navmesh the way the Body asks (shipped file or recompute)."""
        if self._sim is None:
            self._sim = habitat_sim.Simulator(self._configuration(scene))
        elif scene != self._scene:
            # release the old scene's assets first (habitat-lab's should_close_on_new_scene):
            # a bare reconfigure keeps them and the process grows by ~100 MB per scene
            self._sim.close(destroy=False)
            self._sim.reconfigure(self._configuration(scene))
        else:
            return
        self._scene = scene
        self._agent = self._sim.initialize_agent(0)
        self._sim.pathfinder.seed(self.seed)
        self._apply_navmesh(scene, self.body.navmesh)
        self._last_obs = None
        self._path_cache = None
        self._followers = {}

    def _apply_navmesh(self, scene: SceneRef, spec: NavMesh) -> None:
        import os
        pf = self._sim.pathfinder
        if spec.kind == "file":
            if os.path.isfile(scene.navmesh):
                if not pf.load_nav_mesh(scene.navmesh):
                    raise RuntimeError(f"could not load navmesh: {scene.navmesh}")
                return
            if spec.fallback is None:
                raise RuntimeError(f"navmesh file missing: {scene.navmesh}")
            spec = spec.fallback
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = float(spec.agent_radius)
        settings.agent_height = float(spec.agent_height)
        settings.agent_max_climb = float(spec.agent_max_climb)
        settings.cell_height = float(spec.cell_height)
        settings.include_static_objects = False
        if not self._sim.recompute_navmesh(pf, settings):
            raise RuntimeError(f"navmesh recompute failed for {scene.scene_file}")

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
        self._sim = self._agent = self._scene = self._last_obs = self._path_cache = None
        self._followers = {}

    def __enter__(self) -> "SimWorld":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def scene(self) -> SceneRef | None:
        return self._scene

    @property
    def pathfinder(self) -> Any:
        self._require()
        return self._sim.pathfinder

    def _require(self) -> None:
        if self._sim is None:
            raise RuntimeError("SimWorld: call load_scene() first")

    # ---- placement -----------------------------------------------------------
    def place(self, position: Any, rotation_xyzw: Any) -> None:
        """Put the agent at a pose; cameras go back to the Body's initial pitch."""
        self._require()
        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = quat_from_coeffs(np.asarray(rotation_xyzw, dtype=np.float64))
        self._agent.set_state(state, reset_sensors=True)
        self._last_obs = None
        if self.body.locomotion == "teleport":
            self._tp_yaw = _yaw_of(state.rotation)
            self._tp_pos = np.asarray(position, dtype=np.float64)
            self._tp_pitch = float(self.body.camera_pitch_deg)
            self._tp_apply()

    def teleport(self, position: Any, yaw_rad: float) -> None:
        """Move the agent to ``position`` facing ``yaw_rad`` (about +y, 0 = -z),
        keeping the cameras' current pitch."""
        self._require()
        if self.body.locomotion == "teleport":
            self._tp_pos = np.asarray(position, dtype=np.float64)
            self._tp_yaw = float(yaw_rad)
            self._tp_apply()
            return
        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = np.quaternion(math.cos(yaw_rad / 2), 0.0, math.sin(yaw_rad / 2), 0.0)
        self._agent.set_state(state, reset_sensors=False)
        self._last_obs = None

    # ---- teleport locomotion (explore-eqa arithmetic) --------------------------
    def _tp_apply(self) -> None:
        """agent = (pos, yaw about +y) exactly as explore-eqa's set_state; the
        cameras carry the pitch (rotation + swing) that upstream put on the agent."""
        import magnum as mn
        state = habitat_sim.AgentState()
        state.position = np.asarray(self._tp_pos, dtype=np.float64)
        state.rotation = habitat_sim.utils.common.quat_from_angle_axis(self._tp_yaw, np.array([0.0, 1.0, 0.0]))
        self._agent.set_state(state, reset_sensors=True)
        p = self._tp_pitch
        for uuid, spec in (("rgb", self.body.rgb), ("depth", self.body.depth)):
            if spec is None or uuid not in self._agent._sensors:
                continue
            node = self._agent._sensors[uuid].node
            node.rotation = mn.Quaternion.rotation(mn.Rad(math.radians(p)), mn.Vector3(1.0, 0.0, 0.0))
            node.translation = mn.Vector3(*(_swing(spec.position, p) if self.body.pitch_moves_camera else spec.position))
        self._last_obs = None

    def _tp_move(self, action: int) -> bool:
        b = self.body
        collided = False
        if action == FORWARD:
            fwd = np.array([-math.sin(self._tp_yaw), 0.0, -math.cos(self._tp_yaw)], dtype=np.float64)
            target = self._tp_pos + b.forward_step_m * fwd
            pf = self._sim.pathfinder
            step = pf.try_step if b.allow_sliding else pf.try_step_no_sliding
            new = np.asarray(step(self._tp_pos.astype(np.float32), target.astype(np.float32)), dtype=np.float64)
            collided = float(np.linalg.norm(new - self._tp_pos)) < b.forward_step_m * 0.9
            self._tp_pos = new
        elif action == LEFT:
            self._tp_yaw += math.radians(b.turn_deg)
        elif action == RIGHT:
            self._tp_yaw -= math.radians(b.turn_deg)
        else:   # LOOK_UP / LOOK_DOWN: additive tilt clamped to ±limit (a clamped look is a no-op)
            want = self._tp_pitch + (b.tilt_deg if action == LOOK_UP else -b.tilt_deg)
            lim = b.tilt_limit_deg
            self._tp_pitch = want if lim is None else max(-lim, min(lim, want))
        self._tp_apply()
        self._last_obs = self._sim.get_sensor_observations()
        return collided

    # ---- motion --------------------------------------------------------------
    def move(self, action: int) -> bool:
        """Execute one discrete action (FORWARD / LEFT / RIGHT / LOOK_UP / LOOK_DOWN);
        return whether the navmesh cut the motion short. The frame rendered by
        habitat's step is cached for ``observe()``."""
        self._require()
        if action not in _ACTION_NAMES or (action in (LOOK_UP, LOOK_DOWN) and not self.body.has_tilt):
            raise ValueError(f"SimWorld.move: unsupported action {action!r} for this body")
        if self.body.locomotion == "teleport":
            return self._tp_move(action)
        obs = self._sim.step(action)
        if action in (LOOK_UP, LOOK_DOWN) and self.body.pitch_moves_camera:
            self._reseat_cameras()
            obs = self._sim.get_sensor_observations()
            obs["collided"] = False
        self._last_obs = obs
        return bool(obs["collided"])

    def _reseat_cameras(self) -> None:
        """After a LOOK action on a body whose pitch swings the rig, move each
        camera to where the pitched body would carry it (rotation is already
        applied by habitat's look action)."""
        import magnum as mn
        p = self.pitch()
        for uuid, spec in (("rgb", self.body.rgb), ("depth", self.body.depth)):
            if spec is None or uuid not in self._agent._sensors:
                continue
            self._agent._sensors[uuid].node.translation = mn.Vector3(*_swing(spec.position, p))

    # ---- state ---------------------------------------------------------------
    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(position[3] float32, rotation[x, y, z, w] float64) of the agent."""
        self._require()
        st = self._agent.get_state()
        return np.asarray(st.position, dtype=np.float32), np.asarray(quat_to_coeffs(st.rotation), dtype=np.float64)

    def heading(self) -> float:
        """Yaw in radians with habitat-lab's HeadingSensor convention: 0 when
        facing -z, positive for left (counter-clockwise from above) turns."""
        self._require()
        if self.body.locomotion == "teleport" and self._tp_pos is not None:
            return float(self._tp_yaw)
        return _yaw_of(self._agent.get_state().rotation)

    def pitch(self) -> float:
        """Camera pitch in degrees relative to the body (negative looks down)."""
        self._require()
        if self.body.locomotion == "teleport" and self._tp_pos is not None:
            return float(self._tp_pitch)
        st = self._agent.get_state()
        cam = st.sensor_states["rgb"].rotation
        local = st.rotation.inverse() * cam
        v = (local * np.quaternion(0, 0, 0, -1) * local.inverse()).imag   # camera forward in body frame
        return float(math.degrees(math.atan2(v[1], -v[2])))

    def camera_pose(self, uuid: str = "rgb") -> tuple[np.ndarray, np.ndarray]:
        """World pose of a camera: (position[3] float32, rotation[x, y, z, w])."""
        self._require()
        s = self._agent.get_state().sensor_states[uuid]
        return np.asarray(s.position, dtype=np.float32), np.asarray(quat_to_coeffs(s.rotation), dtype=np.float64)

    # ---- pixels --------------------------------------------------------------
    def observe(self) -> dict[str, np.ndarray]:
        """{"rgb": uint8 HxWx3[, "depth": float32 HxWx1 metres]} at the current pose."""
        self._require()
        obs = self._last_obs if self._last_obs is not None else self._sim.get_sensor_observations()
        self._last_obs = obs
        return self._unpack(obs)

    def _unpack(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        out = {"rgb": np.ascontiguousarray(obs["rgb"][..., :3], dtype=np.uint8)}
        if "depth" in obs:
            d = np.asarray(obs["depth"], dtype=np.float32)
            out["depth"] = d[..., None] if d.ndim == 2 else d
        return out

    def render_at(self, position: Any, rotation_xyzw: Any, camera: CameraSpec | None = None) -> dict[str, np.ndarray]:
        """Render at a pose without disturbing the agent. With ``camera=None``
        the body's own rig is used (cameras keep their current pitch; = habitat-lab
        ``get_observations_at``). With a :class:`CameraSpec` a temporary colour
        sensor is placed *exactly* at (position, rotation) — the instance-image /
        GOAT image-goal recipe — and only ``{"rgb"}`` is returned."""
        self._require()
        saved = self._agent.get_state()
        try:
            if camera is None:
                st = habitat_sim.AgentState()
                st.position = np.asarray(position, dtype=np.float32)
                st.rotation = quat_from_coeffs(np.asarray(rotation_xyzw, dtype=np.float64))
                self._agent.set_state(st, reset_sensors=False)
                return self._unpack(self._sim.get_sensor_observations())
            return {"rgb": self._render_extra_camera(position, rotation_xyzw, camera)}
        finally:
            self._agent.set_state(saved, reset_sensors=False)   # body back; sensors keep their local transforms
            self._last_obs = None

    def _render_extra_camera(self, position: Any, rotation_xyzw: Any, camera: CameraSpec) -> np.ndarray:
        from habitat_sim import bindings as hsim
        from habitat_sim.agent.agent import AgentState, SixDOFPose
        uuid = "_extra_camera"
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [camera.height, camera.width]
        spec.hfov = camera.hfov_deg
        self._sim.add_sensor(spec)
        try:
            st = self._agent.get_state()
            # only the temporary sensor is given a world pose; rgb / depth keep their local transforms
            self._agent.set_state(AgentState(
                position=st.position, rotation=st.rotation,
                sensor_states={uuid: SixDOFPose(
                    position=np.asarray(position, dtype=np.float32),
                    rotation=quat_from_coeffs(np.asarray(rotation_xyzw, dtype=np.float64)))},
            ), reset_sensors=False, infer_sensor_states=False)
            self._sim._sensors[uuid].draw_observation()
            img = self._sim._sensors[uuid].get_observation()[:, :, :3]
            return np.ascontiguousarray(img, dtype=np.uint8)
        finally:
            del self._sim._sensors[uuid]
            hsim.SensorFactory.delete_subtree_sensor(self._agent.scene_node, uuid)
            del self._agent._sensors[uuid]
            self._agent.agent_config.sensor_specifications = [
                s for s in self._agent.agent_config.sensor_specifications if s.uuid != uuid]

    # ---- navmesh queries -----------------------------------------------------
    def geodesic(self, start: Any, targets: Any, key: Any = None) -> float:
        """Geodesic distance from ``start`` to the nearest of ``targets`` (one
        point or an (N, 3) set); ``inf`` when unreachable. Always a
        MultiGoalShortestPath — habitat-lab does the same for a single goal.
        ``key`` names the target set so it is projected once and reused
        across steps (habitat-lab's ``_shortest_path_cache``)."""
        self._require()
        ends = np.asarray(targets, dtype=np.float32).reshape(-1, 3)
        if key is not None and self._path_cache is not None and self._path_cache[0] == key:
            path = self._path_cache[1]
        else:
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_ends = ends
            if key is not None:
                self._path_cache = (key, path)
        path.requested_start = _vec3(start)
        self._sim.pathfinder.find_path(path)
        return float(path.geodesic_distance)

    def snap(self, point: Any) -> np.ndarray:
        """Nearest navigable point (NaN vector when none)."""
        return np.asarray(self.pathfinder.snap_point(_vec3(point)), dtype=np.float64)

    def random_navigable_near(self, point: Any, radius: float) -> np.ndarray:
        return np.asarray(self.pathfinder.get_random_navigable_point_near(_vec3(point), float(radius)), dtype=np.float64)

    def is_navigable(self, point: Any) -> bool:
        return bool(self.pathfinder.is_navigable(_vec3(point)))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.pathfinder.get_bounds()
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    def follower(self, goal_radius: float, fresh: bool = False) -> Follower:
        """A greedy geodesic follower for the current scene. Cached per radius
        and rebuilt on scene change (its anti-thrashing memory persists across
        calls, as habitat-lab's long-lived ShortestPathFollower); ``fresh=True``
        returns a new one (a per-call ShortestPathFollower)."""
        self._require()
        r = float(goal_radius)
        if fresh:
            return Follower(self._sim, r)
        if r not in self._followers:
            self._followers[r] = Follower(self._sim, r)
        return self._followers[r]
