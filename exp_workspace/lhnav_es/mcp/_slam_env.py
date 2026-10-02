"""SlamEnv — perception middleware between habitat and an agentic VLN agent.

``env`` is a ``WorldBody`` adapter over the EmbodiedScore-envs body built
by ``stack.build`` (FINE quanta); the simulator is the package's.

Design:
- The agent keeps the coarse VLN-CE action space (FORWARD 0.25 m, LEFT/RIGHT
  15 deg, STOP) but every coarse action is decomposed into 5 micro-steps
  (0.05 m or 3 deg) executed in habitat. Each micro-step's RGB-D frame is
  auto-fed to ORB-SLAM3 and the occupancy grid as a side effect of motion —
  the agent has no code path to touch, skip, or reorder frames.
- The agent's only relationship with SLAM is read-only queries:
  get_pose / get_map / get_trajectory.
- The middleware never moves the robot on its own. Tracking loss is
  bookkeeping (status flag + re-anchor on recovery), never an action.
  The one disclosed exception is bootstrap(): a 360 deg scan used as
  episode initialization before the agent takes over. look_around()
  runs AT THE AGENT'S REQUEST but feeds nothing — a SLAM-silent panorama
  rendered in place (``render_headings``, net-zero motion), so both the
  invariant and tracking continuity stand.

Micro quantum 0.05 m / 3 deg sits well inside ORB-SLAM3's stable tracking
envelope (0.125 m / 6 deg per frame at 512 px).

Camera pitch: a body may start the camera pitched (the HM-EQA body: 30 deg
down) and expose tilt actions 4/5 (30 deg, clamped +-60).
Tilt is a coarse action like any other — MICRO_PER_ACTION sensor-scoped
look_up/look_down quanta, each auto-fed — and the map frame stays LEVEL:
a pitched-start episode anchors its world at the de-pitched start pose
(HabitatEnv.gt_level_pose_cv) and the SLAM anchor absorbs the start
pitch (_anchor = T0_inv @ cam0), so poses carry the pitch as rotation
while the ground plane stays x-z. Poses in the mapping stack are full
4x4 world-from-camera, so occupancy integration needs no pitch special-
casing; yaw_of ignores the pitched forward's y component. Pitch-0
episodes (R2R/RxR) take the plain anchoring path.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path

import numpy as np

from ._body import WorldBody
from ._frontier import cluster_frontiers, cluster_representative, frontier_mask
from ._geom import angle_diff, mount_transform_cv, relative_cv_pose, yaw_of
from ._mapping import (FREE as FREE_, L_CLAMP as L_CLAMP_, L_FREE as L_FREE_, L_OCC as L_OCC_,
                       OBSTACLE as OBSTACLE_, UNKNOWN as UNKNOWN_,
                       EvidenceMap, LayeredOccupancyMap, OccupancyMap,
                       render_map, save_map_png)
from ._slam import TRACKING_OK, OrbSlam3Client, SlamDied, SlamError

ACTIONS = {"forward": 1, "turn_left": 2, "turn_right": 3,
           "look_up": 4, "look_down": 5}

# How long a pose is treated as a guess after ORB-SLAM3 opens a new map.
#
# A re-anchor pins the new map to wherever the tracker last BELIEVED it was.
# That belief is exactly what was in doubt, so the minutes after a re-anchor
# are the minutes most likely to paint a wall that is not there. Map v3 can
# say so: evidence gathered then counts for a quarter, which is enough to
# agree with a later, confident look and not enough to assert a wall alone.
# v1/v2 ignore the weight — they have no way to believe something a little.
RECOVER_FEEDS = 60          # ~15 s of micro-steps at 0.05 m / 3 deg
RECOVER_WEIGHT = 0.25
MICRO_PER_ACTION = 5
# The tracking envelope: at most 6 deg of rotation per fed frame.
# bootstrap() derives its feed stride from it, so a body with a larger turn
# quantum (30 deg -> 6 deg micro) feeds every micro-turn.
FEED_DEG_PER_FRAME = 6.0



class MapEpoch:
    """One ORB-SLAM3 map, our frame for it, and our grids of it.

    ORB-SLAM3 keeps an ATLAS. When tracking is lost and relocalisation fails it
    does not corrupt the map it had — it opens a NEW one, and later, when place
    recognition matches across them, it MERGES them and rewrites the older
    poses into the survivor's frame. That is exactly the "lose it, start again,
    reconnect when you come back downstairs" behaviour a lifelong robot wants.

    A single re-anchor defeats it: ``anchor = last_pose @ inv(raw)`` on the
    first tracked frame of a new map glues that map onto the pose that was in
    doubt, permanently, in the same coordinate frame as the good map. After
    that no correction can land: the two maps are already one grid, and a
    merge ORB-SLAM3 performs later is invisible on this side because every
    pose is read through the hand-made anchor.

    So each map gets an epoch: its own anchor, its own occupancy layers, and a
    note of whether the anchor is a GUESS. The guess still happens (a robot
    that stops mapping when tracking drops is useless), but it is provisional,
    its evidence is weighted down, and ``resync()`` replaces it with the real
    transform the moment ORB-SLAM3's own keyframes say the two frames have
    become one.
    """

    __slots__ = ("index", "anchor", "provisional", "occ", "samples", "opened_tick", "merged_into")

    def __init__(self, index: int, anchor, occ, *, provisional: bool, tick: int) -> None:
        self.index = int(index)
        self.anchor = anchor
        self.occ = occ
        self.provisional = bool(provisional)
        self.samples: list = []          # [(t, raw pose as ORB reported it THEN)]
        self.opened_tick = int(tick)
        self.merged_into: int | None = None

    def as_record(self) -> dict:
        return {"epoch": self.index, "provisional": self.provisional,
                "opened_tick": self.opened_tick, "samples": len(self.samples),
                "merged_into": self.merged_into,
                "free_m2": round(self.occ.free_area_m2(), 1)}

class SlamEnv:
    """Coarse actions in, auto-fed SLAM inside, read-only queries out."""

    def __init__(
        self,
        body: WorldBody,
        coarse_step_m: float = 0.25,
        coarse_turn_deg: float = 15.0,
        cell_size: float = 0.10,
        map_size_m: float = 48.0,
        pose_source: str = "slam",
        frontier_match_m: float = 1.5,
        map_mode: str = "v1",
        start_pitch_deg: float = 0.0,
        coarse_tilt_deg: float | None = None,
        pitch_limit_deg: float | None = 60.0,
        mapper: dict | None = None,
    ) -> None:
        self.micro_step_m = coarse_step_m / MICRO_PER_ACTION      # 0.05 m
        self.micro_turn_deg = coarse_turn_deg / MICRO_PER_ACTION  # 3 deg
        # camera tilt (HM-EQA body): coarse 30 deg -> 5 x 6 deg micro-tilts,
        # the same quantum rule as turns; None = body without tilt actions
        self.coarse_tilt_deg = coarse_tilt_deg
        self.micro_tilt_deg = (None if coarse_tilt_deg is None
                               else coarse_tilt_deg / MICRO_PER_ACTION)
        self.start_pitch_deg = float(start_pitch_deg)
        # None = no clamp (the ObjectNav body: habitat-lab registers its
        # look actions with constraint=None)
        self.pitch_limit_deg = (None if pitch_limit_deg is None
                                else float(pitch_limit_deg))
        if self.start_pitch_deg and self.micro_tilt_deg is None:
            raise ValueError("start_pitch_deg needs coarse_tilt_deg (tilt actions)")
        # Lever-arm camera: SLAM tracks
        # the CAMERA, which swings with pitch; everything the agent reads
        # (get_pose / trajectory / markers / map reference) is the level
        # EYE pose (camera position at pitch 0), converted through the
        # mount kinematics — so a tilt never moves the agent's dot.
        # the body is the package env's (FINE quanta = the micro quanta above);
        # the lever arm is the Body's own ``pitch_moves_camera``
        self.env = body
        self._lever = bool(body.lever_arm)
        self._last_cam_pose: np.ndarray | None = None
        if body.tilt_angle_deg is not None and self.micro_tilt_deg is not None \
                and abs(float(body.tilt_angle_deg) - self.micro_tilt_deg) > 1e-9:
            raise ValueError(f"body micro-tilt {body.tilt_angle_deg} != coarse_tilt_deg / {MICRO_PER_ACTION}")
        # map v2: per-floor layers, height reference = current
        # camera; no frontier surface; grow-only snapped render window.
        self.map_mode = map_mode
        # ``mapper``: the map class's own kwargs beyond the three below
        # (v1 OccupancyMap: obstacle_h_min / obstacle_h_max / depth_min /
        # depth_max / px_stride; v2 LayeredOccupancyMap: floor_merge_m /
        # dwell_feeds / dwell_span_m) — a key the class lacks is a TypeError
        mk = dict(mapper or {})
        if map_mode == "v3":
            # v3 = v2's floor registry over evidence layers. Same geometry,
            # same floors, same render; the layers accumulate evidence
            # instead of writing verdicts, so the map can be walked back.
            self.occ = LayeredOccupancyMap(cell_size=cell_size,
                                           map_size_m=map_size_m,
                                           camera_height=self.env.camera_height,
                                           layer_cls=EvidenceMap, **mk)
        elif map_mode == "v2":
            self.occ = LayeredOccupancyMap(cell_size=cell_size,
                                           map_size_m=map_size_m,
                                           camera_height=self.env.camera_height, **mk)
        else:
            self.occ = OccupancyMap(cell_size=cell_size, map_size_m=map_size_m,
                                    camera_height=self.env.camera_height, **mk)
        self.map_window: tuple | None = None  # v2 render crop, grow-only
        # The atlas on our side: one MapEpoch per ORB-SLAM3 map. `self.occ`
        # always points at the current epoch's grids, so every consumer reads
        # one grid however many epochs there are.
        self._map_kw = dict(cell_size=cell_size, map_size_m=map_size_m, mapper=mk,
                            map_mode=map_mode)
        self.epochs: list = []
        self.epoch = 0
        self._claimed_kf: set = set()      # keyframe t -> already tagged to an epoch
        self._resync_every = 400           # feeds between merge checks
        self._last_resync = 0
        self.merges: list = []
        self._view = None                  # composite of the live epochs
        self._view_key: tuple | None = None
        self._fresh_atlas = True           # no map has tracked yet this episode
        self.slam_restarts = 0             # sessions replaced after the container died
        self.slam_restart_log: list = []
        self._restore: dict | None = None  # a loaded checkpoint waiting for its body (load_spatial)
        self._map_freeze = False           # the grid takes nothing while a pose is unsettled
        self.pose_source = pose_source
        self.slam: OrbSlam3Client | None = None
        self.tick = 0                      # micro-step counter = global clock
        self.status = "ok"
        self.last_pose: np.ndarray | None = None
        self.est_track: list = []
        self.gt_track: list = []
        self.lost_ticks = 0
        self.reanchors = 0
        self._anchor = np.eye(4)
        self._need_reanchor = False
        # Pose confidence (map v3). `_conf_hold` counts down the feeds that
        # follow a re-anchor; `reanchor_log` is the paper trail — when the
        # tracker gave up, how long it had been lost, and what the map was
        # told to believe.
        self._conf_hold = 0
        self._lost_run = 0
        # Where the CURRENT sub-goal's path starts in est_track. A whole-visit
        # track soon covers every room and stops saying anything about what the
        # robot has just been doing. The map keeps the geometry; the line is a
        # working note, and it is this task's.
        self.task_track0 = 0
        # Where THIS EPISODE's path starts when the track is carried across
        # episodes (a tour keeps one map, and `load_spatial` restores the whole
        # visit's track with it). Earlier episodes stay on record — they are
        # what the map was built from — they are simply not redrawn.
        self.ep_track0 = 0
        self.reanchor_log: list = []
        self._T0_inv: np.ndarray | None = None
        # Stable frontier ids: clusters are recomputed per
        # get_map, but an opening keeps its id across calls — each current
        # representative is greedily matched to the nearest registered one
        # within ``frontier_match_m``; unmatched clusters draw fresh ids from
        # a monotonic counter (retired ids are never reused).
        self._frontier_match_m = frontier_match_m
        self._frontier_registry: list = []  # [(id, (wx, wz))]
        self._frontier_next_id = 1
        # Agent-dropped named markers (mark()):
        # [{"label", "x", "z", "floor"}]; same label re-marks (moves) it.
        self.markers: list = []

    # ── episode control ──

    def reset(self) -> None:
        """Anchor on the episode the package env has just placed (start
        pitch applied here; the body is already at the start pose)."""
        self.env.reset(pitch_deg=self.start_pitch_deg)
        if self.start_pitch_deg or self._lever:
            # Pitched / lever-arm start (HM-EQA): the map frame is the LEVEL
            # start EYE pose, so its ground plane is x-z; the first camera
            # frame — where ORB-SLAM3 puts its own origin — sits at the
            # mount transform from it, which the SLAM anchor absorbs
            # (camera poses = anchor @ raw).
            self._T0_inv = np.linalg.inv(self.env.gt_level_pose_cv())
            self._anchor = self._T0_inv @ self.env.gt_camera_pose_cv()
            self._anchor0 = self._anchor.copy()
        else:
            self._T0_inv = np.linalg.inv(self.env.gt_camera_pose_cv())
        # A new episode throws the atlas away, so any epoch still waiting for a
        # merge will never get one: its guessed anchor is all it will ever have.
        # Collapse them into one grid and start the next atlas with a single
        # epoch, anchored on whatever `reset` just worked out above.
        self._collapse()
        self.epochs = [MapEpoch(0, self._anchor.copy(), self.occ,
                                provisional=False, tick=self.tick)]
        self.epoch = 0
        self._claimed_kf.clear()
        self._view = None
        self._fresh_atlas = True
        self.ep_track0 = len(self.est_track)
        self.task_track0 = self.ep_track0
        if self.pose_source == "slam":
            self.slam = OrbSlam3Client()
            self.slam.init(self.env.intrinsics(), fps=10.0)
        self._feed()  # anchor SLAM on the first frame

    def walk_transit(self, advance) -> dict:
        """An IR2R-CE transit, driven: the benchmark's oracle moves the body to
        the next episode's start one fine step at a time, and every step is fed
        to the tracker and the map like any other motion, so tracking stays
        continuous across the tour and the corridors crossed are mapped.

        ``advance()`` takes one oracle step and returns the package's transit
        report (``TourWrapper.transit(run_to_completion=False)``). If the oracle
        cannot reach the start it puts the body there instead (``teleported``);
        that frame is NOT fed, and the caller treats the arrival as a relocation
        (``continue_tour`` with the turn in place)."""
        if self._T0_inv is None:
            raise RuntimeError("walk_transit before reset(): there is no tour to continue")
        lost_before, tick0, fed = self.lost_ticks, self.tick, 0
        teleported = False
        for _ in range(4096):                      # the package's own guard is 2 x 1000 steps
            r = advance() or {}
            if r.get("teleported"):
                teleported = True                  # the body jumped: do not feed the jump
                break
            if r.get("last_action") is not None:
                self._feed()
                fed += 1
            if r.get("transit_done") or not r.get("pending"):
                break
        return {"walked": True, "teleported": teleported, "transit_feeds": fed,
                "transit_ticks": self.tick - tick0,
                "transit_lost_ticks": self.lost_ticks - lost_before}

    def continue_tour(self, steps_per_feed: int | None = None, spin: bool = True,
                      walked: dict | None = None) -> dict:
        """After the body has been moved, keep the map and re-find the body.

        Two ways in. An IR2R-CE transit is driven (``walk_transit``): the
        tracker has followed the body all the way, so there is nothing to
        re-find and ``spin`` is False. A relocation — the next tour of a
        deployment day, or a transit the oracle had to finish by placing the
        body — is a jump the tracker did not see: the body turns once in place
        while feeding the tracker, which seeks relocalisation against its
        retained map. Nothing here reads the simulator's pose: if tracking is
        lost and the tracker opens a new map, that map is re-anchored to the
        last trusted pose, as for any tracking loss.

        The report's ``reloc_error_m`` compares the tracker's pose with the
        simulator's on arrival. It is an evaluation figure like SPL:
        recorded, never shown to the agent, never used by the map.
        """
        if self._T0_inv is None:
            raise RuntimeError("continue_tour before reset(): there is no tour to continue")
        if steps_per_feed is None:
            steps_per_feed = max(1, round(FEED_DEG_PER_FRAME / self.micro_turn_deg))
        pose_before = None if self.last_pose is None else self.last_pose.copy()
        self.env.reset(pitch_deg=self.start_pitch_deg)   # re-level the camera; the body stays put
        lost_before = self.lost_ticks
        self._fresh_atlas = False        # the atlas is the tour's, not a new session's
        self.ep_track0 = len(self.est_track)   # per-episode metrics start here
        self.task_track0 = self.ep_track0
        tick0 = self.tick
        n = round(360.0 / (self.micro_turn_deg * steps_per_feed)) if spin else 0
        # one turn in place, for the tracker: nothing is drawn from a pose that
        # is still being settled
        self._map_freeze = True
        try:
            for _ in range(n):
                for _ in range(steps_per_feed):
                    self.env.step(2)
                self._feed()
        finally:
            self._map_freeze = False
        err, jump = self._transit_error(pose_before)
        return {**(walked or {"walked": False}),
                "slam": self.status, "pose_source": self.pose_source,
                "reloc_error_m": (round(err, 3) if err is not None else None),
                "pose_jump_m": (round(jump, 3) if jump is not None else None),
                "lost_ticks": self.lost_ticks - lost_before,
                "spin_ticks": self.tick - tick0, "feeds": n,
                "epochs": len(self.epochs), "level": self.floor_level(),
                "explored_m2": round(self.view().explored_area_m2(), 2),
                "markers_kept": len(self.markers)}

    def _transit_error(self, pose_before):
        """How far the tracker's pose is from where the body actually is, and
        how far its own belief moved. Both are metrics like SPL: recorded,
        never shown to the agent and never fed to the map."""
        gt_pose = relative_cv_pose(self.env.gt_camera_pose_cv(), self._T0_inv)
        jump = err = None
        if pose_before is not None and self.last_pose is not None:
            jump = float(np.linalg.norm(self.last_pose[:3, 3] - pose_before[:3, 3]))
        if self.last_pose is not None and self.status == "ok":
            err = float(np.linalg.norm(self.last_pose[:3, 3] - gt_pose[:3, 3]))
        return err, jump

    def bootstrap(self, steps_per_feed: int | None = None) -> dict:
        """Disclosed episode init: one full 360 deg in-place scan.

        Feeds SLAM every ``steps_per_feed`` micro-turns (default: whatever
        makes FEED_DEG_PER_FRAME — 2 on the std body's 3 deg micro-turn ->
        6 deg per frame, 60 frames; 1 on a 6 deg micro-turn). Spin-induced
        yaw drift varies run to run (RANSAC stochasticity) — downstream
        consumers get an internally consistent (pose, map) pair either way.
        Frames render only on observe(), so intermediate un-observed steps
        skip nothing.
        """
        if steps_per_feed is None:
            steps_per_feed = max(1, round(FEED_DEG_PER_FRAME / self.micro_turn_deg))
        n = round(360.0 / (self.micro_turn_deg * steps_per_feed))
        for _ in range(n):
            for _ in range(steps_per_feed):
                self.env.step(2)
            self._feed()
        return {"feeds": n, "deg_per_feed": self.micro_turn_deg * steps_per_feed,
                "slam": self.status, "tick": self.tick}

    def look_around(self) -> dict:
        """Agent-invoked 360 deg in-place panorama.

        SLAM-SILENT: it feeds NOTHING — no slam.track, no map integration, no
        trajectory points. Pure renders, like observe(). This is safe because
        the panorama is a closed in-place turn: it ends at the exact entry
        pose, so from SLAM's perspective look_around never happened (the next
        fed frame matches the last one) — which also avoids the yaw drift a
        fed spin adds. The cost: the occupancy map does NOT gain a 360 deg
        scan from this tool.

        Four RGB views 90 deg apart — hfov is 90, so they tile the circle:
        [0, +90, 180, -90] deg relative to the entry heading (positive =
        right, matching get_pose), ending back at the entry heading.
        In-place and free: 0 path length, no coarse-step charge.
        """
        # rendered at the four headings without moving the body
        # (the closed spin's exact output, minus the 120 micro-turns)
        views = self.env.render_headings([0, 90, 180, -90])
        return {"views": views, "headings_deg": [0, 90, 180, -90],
                "pitch_deg": self.env.pitch_deg,
                "slam": self.status, "tick": self.tick}

    def close(self) -> None:
        if self.slam is not None:
            self.slam.close()
        self.env.close()

    # ── the only way the world moves ──

    def act(self, action: int | str, feed_every: int = 1) -> dict:
        """Execute one coarse action as 5 sim micro-steps.

        Default feeds SLAM on every micro-step (0.05 m / 3 deg per frame).
        ``feed_every`` sparsifies feeding (always feeds the last / colliding
        step); dense feeding is the default.
        """
        a = ACTIONS.get(action, action) if isinstance(action, str) else int(action)
        if a in (4, 5):
            return self._tilt(a, feed_every=feed_every)
        if a not in (1, 2, 3):
            raise ValueError(f"unknown action {action!r}")
        done = 0
        collision = False
        for i in range(MICRO_PER_ACTION):
            collision = self.env.step(a)
            done += 1
            if (i + 1) % feed_every == 0 or i == MICRO_PER_ACTION - 1 or collision:
                self._feed()
            if collision:  # abort remaining micro-steps, report honestly
                if self.last_pose is not None:
                    self.occ.mark_obstacle_ahead(self.last_pose)
                break
        return {"action": a, "micro_steps_done": done, "collision": collision,
                "slam": self.status, "tick": self.tick}

    def _tilt(self, a: int, feed_every: int = 1) -> dict:
        """One coarse camera tilt (4 = up, 5 = down) as MICRO_PER_ACTION
        sensor-scoped micro-tilts, each auto-fed like a turn. The +-pitch
        limit is applied at the coarse level (HM-EQA semantics): a tilt
        that would cross it is cut at the limit, and one that cannot move
        at all is a no-op the caller still charges as a step."""
        if self.micro_tilt_deg is None:
            raise ValueError(f"action {a} (camera tilt) is not registered on this body")
        sign = 1.0 if a == 4 else -1.0
        want = self.env.pitch_deg + sign * self.coarse_tilt_deg
        target = (want if self.pitch_limit_deg is None
                  else max(-self.pitch_limit_deg, min(self.pitch_limit_deg, want)))
        n = round(abs(target - self.env.pitch_deg) / self.micro_tilt_deg)
        done = 0
        for i in range(n):
            self.env.step(a)
            done += 1
            if (i + 1) % feed_every == 0 or i == n - 1:
                self._feed()
        return {"action": a, "micro_steps_done": done, "collision": False,
                "pitch_deg": self.env.pitch_deg,
                "tilt_clamped": abs(target - want) > 1e-9,
                "slam": self.status, "tick": self.tick}

    # ── auto-feed (structural invariant: every rendered frame, once, in order) ──

    def _feed(self) -> None:
        if self._lever:
            self._feed_lever()
            return
        obs = self.env.observe()
        assert self._T0_inv is not None, "reset() before stepping"
        gt_pose = relative_cv_pose(self.env.gt_camera_pose_cv(), self._T0_inv)
        if self.slam is not None:
            try:
                res = self.slam.track(obs["rgb"], obs["depth"], t=self.tick * 0.1)
            except (SlamDied, SlamError) as e:
                # ORB-SLAM3 died inside its own loop closure, or refused the frame
                # with an exception of its own. Everything this side owns — the
                # grids, the markers, the world frame — is still here, so the honest
                # move is the one the stack already makes for a tracker that lost
                # its map: open a new session, mark the pose a guess, and let the
                # next tracked frame start a new (provisional) epoch. Losing the
                # whole episode instead would be a worse record, not a truer one.
                self._restart_slam(str(e))
                res = {"pose": None, "state": 3, "ok_frames": 0}
            raw = res["pose"] if res["state"] == TRACKING_OK else None
            if res["state"] == 1:  # NOT_INITIALIZED after a reset: origin moved
                self._need_reanchor = True
            if raw is not None and self._need_reanchor:
                # A state-1 BEFORE the first successful track is just SLAM
                # starting up — the atlas has one map and epoch 0 is it. Only a
                # map that dies mid-run and is replaced deserves a new epoch,
                # re-anchored to the last trusted pose.
                if self._fresh_atlas:
                    self._rebase_epoch0(raw)
                else:
                    self._open_epoch(raw, res.get("state"))
                self._need_reanchor = False
            if raw is not None:
                self._fresh_atlas = False
            pose = None if raw is None else self._anchor @ raw
            self.status = "ok" if raw is not None else "slam_lost"
            self._lost_run = 0 if raw is not None else self._lost_run + 1
            if len(self.epochs) > 1:
                self.resync()          # cheap and rate-limited; a no-op until a merge
        else:
            pose = gt_pose
        if pose is not None:
            self.last_pose = pose
            if not self._map_freeze:
                # Frozen while a pose is still being settled (a tour transit):
                # painting from a pose that is about to be corrected writes
                # walls into open floor that only a later pass can erase.
                self.occ.integrate(pose, obs["depth"], self.env.intrinsics(),
                                   weight=self._pose_weight())
            self.est_track.append([float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])])
            self.gt_track.append([float(gt_pose[0, 3]), float(gt_pose[1, 3]),
                                  float(gt_pose[2, 3])])
        else:
            self.lost_ticks += 1
        self.tick += 1

    def _restart_slam(self, why: str) -> None:
        """Replace a dead SLAM session; the map and the frame stay ours."""
        with contextlib.suppress(Exception):
            self.slam.close()
        self.slam = OrbSlam3Client()
        self.slam.init(self.env.intrinsics(), fps=10.0)
        self._need_reanchor = True
        self._fresh_atlas = False        # the next tracked frame opens a new epoch
        self._claimed_kf.clear()
        self.slam_restarts += 1
        self.slam_restart_log.append({"tick": int(self.tick), "why": why[:200],
                                      "at": (None if self.last_pose is None else
                                             [round(float(self.last_pose[i, 3]), 2) for i in (0, 1, 2)])})

    def _feed_lever(self) -> None:
        """_feed for the lever-arm camera: SLAM / GT give the
        CAMERA pose; the eye pose = camera @ inv(T_eye_cam) (mount
        kinematics at the commanded pitch) is what the agent-facing state
        and the floor reference use. Same structural invariant: every
        rendered frame, once, in order."""
        obs = self.env.observe()
        assert self._T0_inv is not None, "reset() before stepping"
        T_eye_cam = mount_transform_cv(self.env.pitch_deg, self.env.camera_height)
        T_cam_eye = np.linalg.inv(T_eye_cam)
        gt_cam = relative_cv_pose(self.env.gt_camera_pose_cv(), self._T0_inv)
        gt_eye = relative_cv_pose(self.env.gt_level_pose_cv(), self._T0_inv)
        if self.slam is not None:
            try:
                res = self.slam.track(obs["rgb"], obs["depth"], t=self.tick * 0.1)
            except (SlamDied, SlamError) as e:
                # ORB-SLAM3 died inside its own loop closure, or refused the frame
                # with an exception of its own. Everything this side owns — the
                # grids, the markers, the world frame — is still here, so the honest
                # move is the one the stack already makes for a tracker that lost
                # its map: open a new session, mark the pose a guess, and let the
                # next tracked frame start a new (provisional) epoch. Losing the
                # whole episode instead would be a worse record, not a truer one.
                self._restart_slam(str(e))
                res = {"pose": None, "state": 3, "ok_frames": 0}
            raw = res["pose"] if res["state"] == TRACKING_OK else None
            if res["state"] == 1:  # NOT_INITIALIZED after a reset: origin moved
                self._need_reanchor = True
            if raw is not None and self._need_reanchor:
                base = (self._last_cam_pose if self._last_cam_pose is not None
                        else self._anchor0)
                self._anchor = base @ np.linalg.inv(raw)
                self._need_reanchor = False
                self.reanchors += 1
                self._note_reanchor(res.get("state"))
            cam = None if raw is None else self._anchor @ raw
            self.status = "ok" if raw is not None else "slam_lost"
            self._lost_run = 0 if raw is not None else self._lost_run + 1
        else:
            cam = gt_cam
        if cam is not None:
            eye = cam @ T_cam_eye
            self._last_cam_pose = cam
            self.last_pose = eye
            if isinstance(self.occ, LayeredOccupancyMap):
                # v2 classifies heights against the CURRENT camera y — hand
                # it the eye height so the lever's drop is not a "floor rise"
                self.occ.integrate(cam, obs["depth"], self.env.intrinsics(),
                                   ref_y=float(eye[1, 3]), weight=self._pose_weight())
            else:
                # v1 references the start floor (world y 0), camera y unused
                self.occ.integrate(cam, obs["depth"], self.env.intrinsics(),
                                   weight=self._pose_weight())
            self.est_track.append([float(eye[0, 3]), float(eye[1, 3]), float(eye[2, 3])])
            self.gt_track.append([float(gt_eye[0, 3]), float(gt_eye[1, 3]),
                                  float(gt_eye[2, 3])])
        else:
            self.lost_ticks += 1
        self.tick += 1

    # ── pose confidence (map v3) ──────────────────────────────────────

    def _note_reanchor(self, state) -> None:
        """Record the one moment this stack has no way to be right about.

        ORB-SLAM3 has opened a new map; there is nothing to relocalise it
        against, so it is pinned to the last pose the old map produced. If
        that pose had drifted, every later pose inherits the drift and no
        mechanism ever takes it back. The least this side can do is write down
        when it happened and how long the tracker had been lost first.
        """
        self.reanchor_log.append({
            "tick": int(self.tick),
            "lost_run": int(self._lost_run),
            "state": None if state is None else int(state),
            "at": (None if self.last_pose is None else
                   [round(float(self.last_pose[i, 3]), 2) for i in (0, 1, 2)]),
        })
        self._conf_hold = RECOVER_FEEDS

    def _pose_weight(self) -> float:
        """How much the next frame is allowed to assert."""
        if self._conf_hold <= 0:
            return 1.0
        self._conf_hold -= 1
        return RECOVER_WEIGHT

    def forget_map(self, gamma: float = 0.5) -> dict:
        """Thin the map's evidence (map v3 only).

        For the moment the frame the map was painted in is known to have
        moved. Nothing is deleted: a corridor seen a hundred times survives a
        decay that erases a wall seen once, so the map becomes re-earnable
        rather than empty.
        """
        if not hasattr(self.occ, "decay"):
            return {"error": f"map_mode {self.map_mode} has no evidence to decay"}
        return self.occ.decay(gamma)

    # ── the atlas: one epoch per ORB-SLAM3 map ────────────────────────

    def _new_map(self):
        """A blank map set shaped like the one this env was built with."""
        kw = self._map_kw
        mk = dict(kw["mapper"] or {})
        if kw["map_mode"] == "v3":
            return LayeredOccupancyMap(cell_size=kw["cell_size"], map_size_m=kw["map_size_m"],
                                       camera_height=self.env.camera_height,
                                       layer_cls=EvidenceMap, **mk)
        if kw["map_mode"] == "v2":
            return LayeredOccupancyMap(cell_size=kw["cell_size"], map_size_m=kw["map_size_m"],
                                       camera_height=self.env.camera_height, **mk)
        return OccupancyMap(cell_size=kw["cell_size"], map_size_m=kw["map_size_m"],
                            camera_height=self.env.camera_height, **mk)

    def _claim_keyframes(self, ep) -> int:
        """Snapshot the atlas's keyframes and tag the new ones to `ep`.

        The pose we save is what ORB-SLAM3 believed at this instant. That is the
        whole trick: a keyframe whose pose later MOVES has been rewritten by a
        loop closure or a map merge, and how far it moved is the transform
        between the frame it was in and the frame it is in now.

        Claiming happens when an epoch closes, and only for timestamps nobody
        has claimed yet, so every keyframe belongs to the epoch that created it.
        Sampling our own tracking poses would not work: those are ordinary-frame
        poses at our synthetic tick times, and the atlas is keyed by keyframe.
        """
        if self.slam is None:
            return 0
        try:
            kfs = self.slam.keyframes()
        except Exception:                       # noqa: BLE001 — never lose a task to bookkeeping
            return 0
        got = 0
        for k in kfs:
            t = round(float(k["t"]), 4)
            if t in self._claimed_kf:
                continue
            self._claimed_kf.add(t)
            ep.samples.append((t, np.asarray(k["matrix"], float)))
            got += 1
        if len(ep.samples) > 400:               # keep the spread, drop the bulk
            ep.samples = ep.samples[::2]
        return got

    def _rebase_epoch0(self, raw) -> None:
        """SLAM took a few frames to find its feet; move epoch 0's anchor.

        ``self._anchor`` already holds the world pose that RAW = I means — the
        start camera pose, which for a lever-arm body is not identity. Using it
        as the fallback keeps that start frame instead of silently replacing it
        with the origin.
        """
        base = self.last_pose if self.last_pose is not None else self._anchor
        self._anchor = base @ np.linalg.inv(raw)
        self.epochs[self.epoch].anchor = self._anchor

    def _open_epoch(self, raw, state) -> None:
        """ORB-SLAM3 opened a new map. Open one here too, re-anchored to the
        last trusted pose: A = T_last · T_first^-1.

        The anchor is a guess — a robot that refuses to map until it is certain
        maps nothing — so it is marked PROVISIONAL: the new grids are SEPARATE,
        a wrong guess cannot reach into the good map, and `resync` can throw it
        away once the atlas says where this map really sits.
        """
        old = self.epochs[self.epoch]
        self._claim_keyframes(old)
        base = self.last_pose if self.last_pose is not None else self._anchor
        guess = base @ np.linalg.inv(raw)
        occ = self._new_map()
        if hasattr(occ, "set_level") and hasattr(old.occ, "level"):
            occ.set_level(old.occ.level)        # level numbers stay comparable across epochs
        ep = MapEpoch(len(self.epochs), guess, occ, provisional=True, tick=self.tick)
        self.epochs.append(ep)
        self.epoch = ep.index
        self.occ = occ
        self._anchor = guess
        self.reanchors += 1
        self._note_reanchor(state)

    def resync(self, tol_m: float = 0.5, force: bool = False) -> dict:
        """Ask ORB-SLAM3 whether it has changed its mind about the past.

            M = raw_now @ inv(raw_then)            per claimed keyframe
            G = (A_current @ M) @ inv(A_epoch)     what that epoch's grid must move by

        If most of an old epoch's keyframes have moved, its map has been merged
        into the current one; fold its grids through G and its anchor stops
        being a guess. This is the only path in the stack by which a past
        mistake is taken back rather than merely stopped.
        """
        if self.slam is None or len(self.epochs) < 2:
            return {"merged": 0, "reason": "nothing to merge"}
        if not force and (self.tick - self._last_resync) < self._resync_every:
            return {"merged": 0, "reason": "too soon"}
        self._last_resync = self.tick
        try:
            kfs = {round(float(k["t"]), 4): np.asarray(k["matrix"], float)
                   for k in self.slam.keyframes()}
        except Exception as exc:                # noqa: BLE001
            return {"merged": 0, "error": str(exc)[:200]}
        if not kfs:
            return {"merged": 0, "reason": "no keyframes yet"}
        cur = self.epochs[self.epoch]
        out = []
        for ep in self.epochs:
            if ep.index == cur.index or ep.merged_into is not None or not ep.samples:
                continue
            seen = [(kfs[t], p) for t, p in ep.samples if t in kfs]
            if not seen:
                continue
            moved = [(now @ np.linalg.inv(then),
                      float(np.linalg.norm(now[:3, 3] - then[:3, 3])))
                     for now, then in seen]
            shifted = [m for m, d in moved if d > tol_m]
            if len(shifted) < max(2, len(moved) // 2):
                continue                        # not a merge: the past sits still
            M = shifted[len(shifted) // 2]
            G = (cur.anchor @ M) @ np.linalg.inv(ep.anchor)
            rec = {"epoch": ep.index, "into": cur.index, "keyframes": len(moved),
                   "moved": len(shifted),
                   "shift_m": round(float(np.median([d for _, d in moved])), 3),
                   "tick": self.tick, **self._fold(ep, G)}
            ep.anchor = cur.anchor @ M
            ep.provisional = False
            ep.merged_into = cur.index
            self.merges.append(rec)
            out.append(rec)
        if out and cur.provisional:
            cur.provisional = False             # it is tied to a known map now
        return {"merged": len(out), "merges": out}

    def view(self):
        """The map as the agent should see it: every live epoch at once.

        Underneath, each ORB map keeps its own grid so that a later correction
        can move one without disturbing the others. But the robot should not go
        blind the moment tracking breaks — the house it mapped before the break
        is still the best thing it knows. So reads composite the live epochs
        into one throwaway map.

        Compositing needs no transform: a grid is written through its epoch's
        anchor, so it is already in the common world frame. The anchor is what
        that frame MEANS for that epoch, and when `resync` corrects one, `_fold`
        moves its cells and retires it. One epoch — the usual case — costs
        nothing: the map itself is returned.
        """
        live = [e for e in self.epochs if e.merged_into is None]
        if len(live) <= 1:
            return self.occ
        key = (self.tick, len(self.epochs), len(self.merges))
        if self._view_key == key and self._view is not None:
            return self._view
        cur = self.epochs[self.epoch]
        comp = self._new_map()
        if hasattr(comp, "set_level") and hasattr(cur.occ, "_level_layer"):
            # Same levels in the same layer ORDER, so a caller holding a floor
            # index from `self.occ` indexes the same storey here.
            for lvl, _ in sorted(cur.occ._level_layer.items(), key=lambda kv: kv[1]):
                comp.set_level(lvl)
            comp.floor_ys[:len(cur.occ.floor_ys)] = list(cur.occ.floor_ys)
            comp.set_level(cur.occ.level)
        eye = np.eye(4)
        for ep in live:
            self._fold_into(ep.occ, comp, eye)
        self._view, self._view_key = comp, key
        return comp

    def _collapse(self) -> int:
        """Fold every live epoch into the current one and keep only that grid.

        Used when the atlas is about to be discarded: a guess that can never be
        checked is simply what happened, and pretending otherwise would only
        strand the cells it mapped.
        """
        live = [e for e in self.epochs if e.merged_into is None and e.index != self.epoch]
        for ep in live:
            self._fold_into(ep.occ, self.occ, np.eye(4))
            ep.merged_into = self.epoch
            ep.provisional = False
        self._view = None
        return len(live)

    def _fold(self, ep, G) -> dict:
        """An epoch's anchor was wrong by G. Move its cells and retire it."""
        folded = self._fold_into(ep.occ, self.occ, G)
        self._view = None
        yaw = math.atan2(float(G[0, 2]), float(G[2, 2]))
        return {"yaw_deg": round(math.degrees(yaw), 1),
                "dx": round(float(G[0, 3]), 2), "dy": round(float(G[1, 3]), 2),
                "dz": round(float(G[2, 3]), 2), **folded}

    def _fold_into(self, src_occ, dst_occ, G) -> dict:
        """Add one map's cells into another, moved by G.

        Only G's horizontal part touches a grid — a 2-D layer has no third axis
        to rotate — and layers pair up by LEVEL NUMBER, which is why a new epoch
        inherits the level it opened on. Evidence adds and clips (agreement
        strengthens, disagreement cancels); a plain grid takes obstacle-wins.
        """
        yaw = math.atan2(float(G[0, 2]), float(G[2, 2]))
        dx, dz = float(G[0, 3]), float(G[2, 3])
        src_by_level = getattr(src_occ, "_level_layer", None)
        dst_by_level = getattr(dst_occ, "_level_layer", None)
        if src_by_level is None or dst_by_level is None:
            pairs = [(src_occ, dst_occ)]
        else:
            keep, pairs = dst_occ.level, []
            for lvl, si in sorted(src_by_level.items()):
                if lvl not in dst_by_level:
                    dst_occ.set_level(lvl)          # creates the layer
                pairs.append((src_occ.layers[si], dst_occ.layers[dst_by_level[lvl]]))
            dst_occ.set_level(keep)                 # set_level moved it; put it back
        n, cell = dst_occ.n, dst_occ.cell_size
        if abs(yaw) < 1e-9 and abs(dx) < 1e-9 and abs(dz) < 1e-9:
            sj = si = None                          # identity: no resampling at all
        else:
            zi, xi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            xw = (xi - dst_occ.origin + 0.5) * cell     # destination cell -> world
            zw = (zi - dst_occ.origin + 0.5) * cell
            c, sn = math.cos(-yaw), math.sin(-yaw)      # world -> source world -> cell
            sx = (xw - dx) * c - (zw - dz) * sn
            sz = (xw - dx) * sn + (zw - dz) * c
            si = np.clip(np.floor(sx / cell).astype(int) + src_occ.origin, 0, n - 1)
            sj = np.clip(np.floor(sz / cell).astype(int) + src_occ.origin, 0, n - 1)
        folded = 0
        for src, tgt in pairs:
            folded += int((src.grid > UNKNOWN_).sum())
            if hasattr(src, "evidence") and hasattr(tgt, "evidence"):
                add = src.evidence if sj is None else src.evidence[sj, si]
                tgt.evidence[:] = np.clip(tgt.evidence.astype(np.int32) + add.astype(np.int32),
                                          -L_CLAMP_, L_CLAMP_).astype(np.int16)
                tgt._classify(np.arange(n * n))
            else:
                g = src.grid if sj is None else src.grid[sj, si]
                tgt.grid[g == OBSTACLE_] = OBSTACLE_
                tgt.grid[(g == FREE_) & (tgt.grid == UNKNOWN_)] = FREE_
        return {"cells_folded": folded, "layers": len(pairs)}

    def atlas(self) -> dict:
        """What this side knows about the atlas."""
        return {"epochs": [e.as_record() for e in self.epochs],
                "current": self.epoch, "merges": list(self.merges)}

    # ── checkpoints: freezing a boundary, and starting again from it ──

    def dump_spatial(self, path) -> dict:
        """Everything about this place that is OURS, frozen to a file.

        The point of a checkpoint is that a task can be started again from the
        state it actually started from, rather than re-run in a house that has
        since been explored and with a ledger that already knows the ending. So
        what is written is the map, the markers, the world frame and the last
        estimated pose — not the ORB-SLAM3 graph, which lives inside its own
        process and cannot be frozen. A restore therefore reproduces the body
        and this side's spatial memory exactly, and the tracker's internal state
        only approximately; that is worth saying out loud whenever a replayed
        number is set beside its original.
        """
        import pickle
        vm = self.view()                       # live epochs folded: the atlas does not survive
        layers = getattr(vm, "layers", None)
        lays = layers if layers is not None else [vm]
        blob = {
            "version": 2,
            "map_mode": self.map_mode,
            "grids": [l.grid for l in lays],
            "evidence": [getattr(l, "evidence", None) for l in lays],
            "floor_ys": list(getattr(vm, "floor_ys", [0.0])),
            "current_floor": int(getattr(vm, "current_floor", 0)),
            "floor_policy": getattr(vm, "floor_policy", None),
            "level": int(getattr(vm, "level", 0)),
            "level_layer": dict(getattr(vm, "_level_layer", {0: 0})),
            "transitions": list(getattr(vm, "transitions", [])),
            "y_win": list(getattr(self.occ, "_y_win", [])),
            "y_hist": list(getattr(self.occ, "_y_hist", [])),
            "cell_size": float(vm.cell_size),
            "n": int(vm.n),
            "markers": list(self.markers),
            "est_track": list(self.est_track),
            "gt_track": list(self.gt_track),
            "ep_track0": int(self.ep_track0),
            "last_pose": None if self.last_pose is None else self.last_pose.tolist(),
            "T0_inv": None if self._T0_inv is None else self._T0_inv.tolist(),
            "epochs": [e.as_record() for e in self.epochs],
            "merges": list(self.merges),
            "reanchor_log": list(self.reanchor_log),
            "reanchors": int(self.reanchors),
            "lost_ticks": int(self.lost_ticks),
            "tick": int(self.tick),
        }
        f = Path(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump(blob, fh, protocol=4)
        return {"saved": str(f), "grids": len(lays), "markers": len(self.markers),
                "epochs": len(self.epochs), "level": int(getattr(vm, "level", 0)),
                "explored_m2": round(vm.explored_area_m2(), 2)}

    def load_spatial(self, path) -> dict:
        """Read a frozen boundary back. Moves nothing: the body is put back by
        ``restore_body`` afterwards, and doing it the other way round would
        anchor the tracker at the old pose and then swap the map under it."""
        import pickle
        f = Path(path)
        if not f.exists():
            return {"error": f"no spatial dump at {f}"}
        with open(f, "rb") as fh:
            blob = pickle.load(fh)
        if int(blob.get("n", -1)) != int(self.occ.n) or \
                abs(float(blob.get("cell_size", -1)) - float(self.occ.cell_size)) > 1e-9:
            return {"error": "grid geometry differs; refusing to load"}
        grids = blob.get("grids") or []
        evid = blob.get("evidence") or [None] * len(grids)
        occ = self._new_map()
        layers = getattr(occ, "layers", None)
        if layers is not None:
            ll = {int(k): int(v) for k, v in (blob.get("level_layer") or {0: 0}).items()}
            if getattr(occ, "floor_policy", "height") == "stairs" and len(ll) == len(grids):
                for lvl, _ in sorted(ll.items(), key=lambda kv: kv[1]):
                    occ.set_level(lvl)
                occ.set_level(int(blob.get("level", 0)))
                occ.transitions = list(blob.get("transitions") or [])
            while len(layers) < len(grids):
                layers.append(occ.layer_cls(**occ._layer_kw))
            ys = list(blob.get("floor_ys") or [0.0])
            occ.floor_ys = ys + [0.0] * max(0, len(layers) - len(ys))
            occ.current_floor = int(blob.get("current_floor", 0))
            occ._y_win = list(blob.get("y_win") or [])
            occ._y_hist = list(blob.get("y_hist") or [])
            targets = layers
        else:
            targets = [occ]
        for lay, g, e in zip(targets, grids, evid):
            if hasattr(lay, "evidence"):
                if e is not None:
                    lay.evidence[:] = e
                else:                            # a v1/v2 dump read into evidence layers
                    lay.evidence[:] = 0
                    lay.evidence[g == OBSTACLE_] = L_OCC_
                    lay.evidence[g == FREE_] = -L_FREE_
                lay._classify(np.arange(lay.n * lay.n))
            else:
                lay.grid[:] = g
        self.occ = occ
        self.epochs = [MapEpoch(0, self._anchor.copy(), occ, provisional=False, tick=self.tick)]
        self.epoch = 0
        self._claimed_kf.clear()
        self._view = None
        self.map_window = None
        self.markers = list(blob.get("markers") or [])
        self.est_track = list(blob.get("est_track") or [])
        gt = blob.get("gt_track")
        if gt is not None:
            self.gt_track = list(gt)
        self.ep_track0 = len(self.est_track)      # earlier path stays recorded, not redrawn
        self.task_track0 = len(self.est_track)
        self.reanchor_log = list(blob.get("reanchor_log") or [])
        # Kept for restore_body: only a restore that also puts the body back
        # where the dump was taken may continue from the saved estimate.
        self._restore = {"last_pose": blob.get("last_pose"), "T0_inv": blob.get("T0_inv")}
        return {"loaded": str(f), "grids": len(grids), "markers": len(self.markers),
                "level": self.floor_level(), "version": int(blob.get("version", 1)),
                "explored_m2": round(occ.explored_area_m2(), 2)}

    def restore_body(self) -> dict:
        """The body has just been put back where a checkpoint recorded it: open
        a SLAM session that continues from the checkpoint's estimate.

        The world frame, the map and the estimated pose are the checkpoint's,
        and the new session is anchored so its first frame carries on from that
        estimate rather than declaring the restored spot the origin. Called
        after ``load_spatial``; without a loaded checkpoint this is an ordinary
        fresh session at wherever the body now stands."""
        restore = self._restore
        self._restore = None
        if restore is not None and restore.get("T0_inv") is not None:
            self._T0_inv = np.asarray(restore["T0_inv"], float)
        else:
            self._T0_inv = np.linalg.inv(self.env.gt_camera_pose_cv())
        if restore is not None and restore.get("last_pose") is not None:
            self.last_pose = np.asarray(restore["last_pose"], float)
            self._anchor = self.last_pose.copy()
        else:
            self._collapse()
            self._anchor = np.eye(4)
        self.epochs = [MapEpoch(0, self._anchor.copy(), self.occ, provisional=False, tick=self.tick)]
        self.epoch = 0
        self._claimed_kf.clear()
        self._view = None
        self._fresh_atlas = True
        self.ep_track0 = len(self.est_track)
        self.task_track0 = self.ep_track0
        if self.pose_source == "slam":
            if self.slam is not None:
                with contextlib.suppress(Exception):
                    self.slam.close()
            self.slam = OrbSlam3Client()
            self.slam.init(self.env.intrinsics(), fps=10.0)
            # A restored estimate is only right if the new session is re-based
            # on it; RGB-D ORB-SLAM3 may report OK on its very first frame.
            self._need_reanchor = restore is not None and restore.get("last_pose") is not None
        self._feed()  # anchor SLAM on the first frame
        return {"slam": self.status, "continued_from_checkpoint": restore is not None,
                "level": self.floor_level(), "markers": len(self.markers),
                "explored_m2": round(self.view().explored_area_m2(), 2)}

    def mark(self, label: str) -> dict:
        """Drop (or move) a named marker at the current estimated planar pose.
        Markers are annotation only — they live
        on the map render + get_map payload, never touch SLAM or the grid.
        Re-marking an existing label moves it; labels cap at 40 chars."""
        if self.last_pose is None:
            return {"error": "no pose estimate yet"}
        label = str(label).strip()[:40]   # a shorter cap collapses distinct rooms to one prefix
        if not label:
            return {"error": "empty label"}
        x = round(float(self.last_pose[0, 3]), 2)
        z = round(float(self.last_pose[2, 3]), 2)
        yaw = round(math.degrees(yaw_of(self.last_pose)), 1)   # the planar pose (x, z, yaw)
        floor = getattr(self.occ, "current_floor", 0)
        self.markers = [m for m in self.markers if m["label"] != label]
        self.markers.append({"label": label, "x": x, "z": z, "yaw_deg": yaw, "floor": floor,
                             "level": self.floor_level()})
        # ``current_floor``: the layer the marker went on, for the views the
        # harness banks with it (each view's pose estimate is this one, turned)
        return {"marked": {"label": label, "x": x, "z": z, "yaw_deg": yaw},
                "n_markers": len(self.markers), "current_floor": int(floor)}

    # ── read-only queries (no side effects, callable any time) ──

    # ── which floor am I on ───────────────────────────────────────────

    def begin_task(self, index: int = -1) -> dict:
        """A new sub-goal starts here: the drawn path starts over.

        Nothing else moves — the grid, the markers, the SLAM session and the
        pose are the visit's, not the task's.
        """
        self.task_track0 = len(self.est_track)
        return {"task": int(index), "track0": self.task_track0, "tick": self.tick}

    def drawn_track(self) -> list:
        """The line to draw: this sub-goal's, inside this episode's.

        The map holds the geometry and keeps it for the whole visit. The line
        is a working note about what the robot has just been doing, so it is
        scoped to the current task — a whole-visit track soon covers every
        room and says nothing.
        Everything before the slice is still in ``est_track``; it is recorded,
        not redrawn.
        """
        return self.est_track[max(self.ep_track0, self.task_track0):]

    def floor_level(self) -> int:
        """This stack's floor id for the place the robot is standing in.

        Under ``floor_policy="stairs"`` that is the topological level: 0 where
        the chain started, +1 up a staircase, and it can go negative. Under
        the height policy there is no such thing, so the layer index stands in
        — the two agree on a single-storey house and the height policy has no
        better answer to give.
        """
        return int(getattr(self.occ, "level", self.occ.current_floor))

    def floor_mismatch(self) -> bool:
        """Is the body off the floor layer the map plans on?

        ``get_map`` draws, and ``preview_path`` plans on, the current layer.
        Under the height policy that is the registered floor the estimated
        height last settled within ``floor_merge_m`` of; while the estimate is
        further than that from it — on a staircase, or on a new storey that has
        not settled into a layer yet — routes are planned on a floor the body
        is not standing on. The stairs policy names its level by traversal
        rather than by height, and a v1 map has a single layer: neither has a
        mismatch to report.
        """
        occ = self.occ
        ys = getattr(occ, "floor_ys", None)
        if ys is None or self.last_pose is None or getattr(occ, "floor_policy", "height") != "height":
            return False
        return abs(float(self.last_pose[1, 3]) - float(ys[occ.current_floor])) > float(occ.floor_merge_m)

    def body_facts(self) -> dict:
        """The readings a motion reply carries for the recovery controller:
        the explored area of the whole map (every floor layer, every live
        epoch — newly explored area is measured on it), the layer routes are
        planned on, and whether the body is on it. One count over the grids
        and one height comparison: cheap enough for every step."""
        return {"explored_m2": round(float(self.view().explored_area_m2()), 2),
                "current_floor": int(getattr(self.occ, "current_floor", 0)),
                "floor_mismatch": self.floor_mismatch()}

    def set_floor(self, level: int | None = None, marker: str = "", why: str = "") -> dict:
        """Put the map on a level by NAME.

        The wake-up is the one moment the robot's own motion says nothing
        about which floor it is on: it did not walk here, it was carried, and
        no counter over the robot's own movement can see that. What DOES know is the
        place the wake-up recognised: every marker was dropped on a floor, so
        the place graph can hand the level back in. Absolute height is not
        consulted, which is the whole point.
        """
        if not hasattr(self.occ, "set_level"):
            return {"error": f"map_mode {self.map_mode} has no level registry"}
        if level is None and marker:
            hit = next((m for m in self.markers if m.get("label") == marker), None)
            if hit is None:
                return {"error": f"no marker {marker!r}", "markers": len(self.markers)}
            level = hit.get("level", hit.get("floor"))
        if level is None:
            return {"error": "set_floor needs a level or a marker"}
        was = self.floor_level()
        y = None if self.last_pose is None else float(self.last_pose[1, 3])
        layer = self.occ.set_level(int(level), y_hint=y)
        return {"level": int(level), "was": was, "layer": int(layer),
                "layers": len(getattr(self.occ, "layers", [1])),
                "changed": int(level) != was, "why": why or marker}

    def get_pose(self) -> dict:
        if self.last_pose is None:
            return {"slam": self.status, "tick": self.tick, "pose": None}
        p = self.last_pose
        return {"x": round(float(p[0, 3]), 3), "z": round(float(p[2, 3]), 3),
                "yaw_deg": round(math.degrees(yaw_of(p)), 1),
                "slam": self.status, "tick": self.tick}

    def _stable_frontier_ids(self, reps: list) -> list:
        """Match current cluster representatives (world coords) against the
        registry; return the id per rep, updating the registry in place."""
        pairs = []
        for ri, (wx, wz) in enumerate(reps):
            for ei, (_fid, (ox, oz)) in enumerate(self._frontier_registry):
                d = math.hypot(wx - ox, wz - oz)
                if d <= self._frontier_match_m:
                    pairs.append((d, ri, ei))
        pairs.sort()
        assigned: dict = {}
        used: set = set()
        for _d, ri, ei in pairs:
            if ri in assigned or ei in used:
                continue
            assigned[ri] = self._frontier_registry[ei][0]
            used.add(ei)
        ids = []
        registry = []
        for ri, (wx, wz) in enumerate(reps):
            fid = assigned.get(ri)
            if fid is None:
                fid = self._frontier_next_id
                self._frontier_next_id += 1
            ids.append(fid)
            registry.append((fid, (wx, wz)))
        self._frontier_registry = registry  # unmatched old ids retire
        return ids

    def get_map(self, png_path: str | None = None) -> dict:
        vm = self.view()
        fmask = frontier_mask(vm.grid)
        clusters = cluster_frontiers(fmask)
        frontiers = []
        if self.last_pose is not None:
            px, pz = float(self.last_pose[0, 3]), float(self.last_pose[2, 3])
            yaw = yaw_of(self.last_pose)
            cells = [cluster_representative(comp) for comp in clusters]
            reps = [vm.cell_to_world(zi, xi) for zi, xi in cells]
            ids = self._stable_frontier_ids(reps)
            for fid, (zi, xi), (wx, wz), comp in zip(  # noqa: B905 — py3.9 env
                    ids, cells, reps, clusters):
                bearing = math.atan2(wx - px, wz - pz)
                frontiers.append({
                    "id": f"F{fid}",
                    "dir_deg": round(math.degrees(angle_diff(bearing, yaw)), 1),
                    "dist_m": round(math.hypot(wx - px, wz - pz), 2),
                    "size_cells": int(len(comp)),
                    # representative cell for the annotated render — stripped from the
                    # agent-facing payload by the manager
                    "cell": (zi, xi),
                })
            frontiers.sort(key=lambda f: f["dist_m"])
        agent_cell = None
        if self.last_pose is not None:
            agent_cell = vm.world_to_cell(float(self.last_pose[0, 3]),
                                                float(self.last_pose[2, 3]))
        if png_path is not None:
            save_map_png(png_path, vm.grid, fmask, agent_cell, None)
        return {
            "frontiers": frontiers,
            "dir_convention": "dir_deg relative to heading, positive = to the right",
            "free_m2": vm.free_area_m2(),
            "explored_m2": vm.explored_area_m2(),
            "png": png_path,
            "slam": self.status, "tick": self.tick,
        }

    def get_trajectory(self, max_points: int = 200) -> dict:
        """This sub-goal's path, not the whole visit — the same slice the map
        draws, so the picture and the numbers cannot disagree about where the
        robot has just been. Earlier sub-goals stay in ``est_track``."""
        pts = [[round(p[0], 2), round(p[2], 2)] for p in self.drawn_track()]
        stride = max(1, len(pts) // max_points)
        return {"points": pts[::stride], "n_total": len(pts),
                "slam": self.status, "tick": self.tick}

    # ── eval-side helpers (not part of the agent surface) ──

    def map_image(self) -> np.ndarray:
        vm = self.view()
        agent_cell = None
        if self.last_pose is not None:
            agent_cell = vm.world_to_cell(float(self.last_pose[0, 3]),
                                                float(self.last_pose[2, 3]))
        return render_map(vm.grid, frontier_mask(vm.grid), agent_cell, None)

    def metrics(self) -> dict:
        vm = self.view()
        n = min(len(self.est_track), len(self.gt_track))
        ate = float("nan")
        if n:
            e = np.asarray(self.est_track[:n]) - np.asarray(self.gt_track[:n])
            ate = float(np.sqrt((e ** 2).sum(axis=1).mean()))
        return {
            "ticks": self.tick,
            "lost_ticks": self.lost_ticks,
            "tracking_ratio": (self.tick - self.lost_ticks) / max(1, self.tick),
            "reanchors": self.reanchors,
            # Not just how many: when and where, so "where did the map go
            # wrong" has an answer.
            "reanchor_log": list(self.reanchor_log),
            # Sessions that had to be replaced because the container died under
            # us: a run with these in it walked part of its map on a guess.
            "slam_restarts": self.slam_restarts,
            "slam_restart_log": list(self.slam_restart_log),
            # The atlas: how many maps ORB-SLAM3 ended up with, and which of
            # them it later reconnected. A provisional epoch that never merges
            # is a guess this run never got to check — and the record says so.
            "epochs": [e.as_record() for e in self.epochs],
            "merges": list(self.merges),
            "map_mode": self.map_mode,
            "ate_rmse_m": ate,
            "free_m2": vm.free_area_m2(),
            "explored_m2": vm.explored_area_m2(),
            "collisions": self.env.collisions,
        }
