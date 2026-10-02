"""Host-side client for the containerized ORB-SLAM3 server.

Spawns ``docker run -i navharness/orbslam3 python3 /work/_slam_server.py``
with this directory mounted read-only at /work, and speaks the JSON-lines
protocol over the container's stdin/stdout. The image is built from
``docker/orbslam3/``; ORB-SLAM3 is CPU-only, no GPU flags needed.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import select
import subprocess

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

TRACKING_OK = 2
TRACKING_LOST = 4



def _stderr_sink():
    """Where the container's own words go. A file, appended, never discarded."""
    path = os.environ.get("ORBSLAM3_LOG") or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"orbslam3-{os.getpid()}.log")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return open(path, "a", buffering=1)          # noqa: SIM115 — owned by the process
    except OSError:
        return subprocess.DEVNULL                     # a log is not worth an episode

# How long one RPC may take before the container counts as gone. A tracked
# frame is milliseconds; loop closure and map merging can take seconds; nothing
# legitimate takes minutes. Startup loads the vocabulary, so it gets its own.
RPC_TIMEOUT_S = 180.0
INIT_TIMEOUT_S = 900.0


class SlamError(RuntimeError):
    """ORB-SLAM3 refused a call (an exception inside the library, reported
    back by the server, e.g. ``Invalid argument``). The process is alive but
    its state is not to be trusted; the caller treats it like a dead one."""


class SlamDied(RuntimeError):
    """The ORB-SLAM3 container is gone. Seen when its own loop closure throws
    (``terminate called ... cv::Exception``) — a C++ death this process
    cannot prevent, only notice. The caller may open a new session."""


class OrbSlam3Client:
    def __init__(self, image: str = "navharness/orbslam3:latest") -> None:
        self._proc = subprocess.Popen(
            [
                "docker", "run", "-i", "--rm",
                "-v", f"{_THIS_DIR}:/work:ro",
                image, "python3", "/work/_slam_server.py",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # NOT DEVNULL. ORB-SLAM3 is a C++ library in a container: when it
            # dies, what it said on the way out is the only evidence there is.
            # Set ORBSLAM3_LOG to keep the file somewhere else.
            stderr=_stderr_sink(),
            text=True,
        )

    def _rpc(self, msg: dict, timeout_s: float | None = None) -> dict:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
            # A container that stops ANSWERING is as dead as one that exits, and
            # far more expensive: a hung frame holds the scene until the env's
            # own timeout. Wait a bounded time, then take the session as gone.
            wait = RPC_TIMEOUT_S if timeout_s is None else timeout_s
            if wait and not select.select([self._proc.stdout], [], [], wait)[0]:
                with contextlib.suppress(Exception):
                    self._proc.kill()
                raise SlamDied(f"slam_server did not answer {msg.get('cmd')} within {wait:.0f}s")
            line = self._proc.stdout.readline()
        except (BrokenPipeError, OSError) as e:
            raise SlamDied(f"slam_server is gone: {type(e).__name__}: {e}") from e
        if not line:
            raise SlamDied("slam_server closed its stdout (the container died)")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise SlamError(f"slam_server error: {resp.get('error')}")
        return resp

    def init(self, intrinsics: dict, fps: float = 10.0) -> None:
        intr = dict(intrinsics)
        intr["fps"] = fps
        self._rpc({"cmd": "init", "intr": intr}, timeout_s=INIT_TIMEOUT_S)

    def track(self, rgb: np.ndarray, depth_m: np.ndarray, t: float) -> dict:
        """Returns {"pose": 4x4 ndarray | None, "state": int, "ok_frames": int}."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth = np.ascontiguousarray(depth_m, dtype=np.float32)
        h, w = depth.shape
        resp = self._rpc({
            "cmd": "track",
            "rgb": base64.b64encode(rgb.tobytes()).decode(),
            "depth": base64.b64encode(depth.tobytes()).decode(),
            "h": h, "w": w, "t": float(t),
        })
        pose = resp["pose"]
        return {
            "pose": None if pose is None else np.asarray(pose, dtype=np.float64),
            "state": int(resp["state"]),
            "ok_frames": int(resp["ok_frames"]),
        }

    def trajectory(self) -> list:
        return self._rpc({"cmd": "trajectory"})["trajectory"]

    def keyframes(self) -> list:
        """Every atlas map's keyframes, as ORB-SLAM3 currently believes them.

        The one thing a frame-by-frame client cannot see on its own: when a
        loop closes, or when two maps are merged, the poses of keyframes
        already reported change. Reading this is how the map on our side gets
        to hear about it.
        """
        return self._rpc({"cmd": "keyframes"})["keyframes"]

    def close(self) -> None:
        with contextlib.suppress(Exception):  # teardown must not raise
            self._rpc({"cmd": "shutdown"})
        with contextlib.suppress(Exception):
            self._proc.stdin.close()  # type: ignore[union-attr]
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
