"""envserver — serve one env object over HTTP, in its own interpreter.

    <python> -m core.envserver --env '<JSON of the experiment's env block>' --port 9200

The env block is instantiated as written (``_target_`` + its kwargs, the
runner's keys python / port / shapes accepted and ignored by the object);
every public method of the object is a verb:

    POST /<method>      body: JSON kwargs      -> JSON return value
    GET  /health        -> the object's info() (+ "ok"), or 503 while it is not up

Requests are served one at a time on the server thread — the simulators
behind these objects are single-threaded (GL affinity), and the runner's
workers are one per server anyway.

``spawn()`` is the runner side of the same file: start a server for an env
block, wait for /health, return the handle; the child dies with its parent
(PR_SET_PDEATHSIG) so a killed run never leaves a simulator behind.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# ── the runner side ──

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_KEYS = ("python", "port", "shapes")  # the block's keys that are not the object's kwargs


def resolve_python(spec: str | None) -> str:
    """``env.python``: an interpreter path, an env name, or None for this
    interpreter. A name is looked up in the repo's own ``envs/<name>``
    (the standalone layout, ``conda create -p envs/<name>``),
    then in ``<conda root>/envs/<name>`` next to the running interpreter."""
    if not spec:
        return sys.executable
    if "/" in spec:
        return spec
    prefix = Path(sys.prefix)
    root = prefix.parent.parent if prefix.parent.name == "envs" else prefix
    candidates = [REPO_ROOT / "envs" / spec / "bin" / "python", root / "envs" / spec / "bin" / "python"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"env.python={spec!r}: no interpreter at " + " or ".join(map(str, candidates)))


def _set_pdeathsig() -> None:  # runs in the child between fork and exec
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


class ServerHandle:
    def __init__(self, proc: subprocess.Popen, url: str, log_path: Path) -> None:
        self.proc, self.url, self.log_path = proc, url, log_path

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def launch(env_block: dict[str, Any], port: int, log_path: Path) -> ServerHandle:
    """Start ``<env.python> -m core.envserver`` for this block on ``port`` and
    return at once; ``wait_healthy`` blocks until it answers."""
    python = resolve_python(env_block.get("python"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 — the child owns it
    proc = subprocess.Popen(
        [python, "-m", "core.envserver", "--env", json.dumps(env_block), "--port", str(port)],
        cwd=str(REPO_ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        preexec_fn=_set_pdeathsig,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    return ServerHandle(proc, f"http://127.0.0.1:{port}", log_path)


def wait_healthy(handles: list[ServerHandle], timeout_s: float = 600.0) -> list[ServerHandle]:
    """Block until every handle answers /health (a cold habitat scene can take
    a while). One server exiting or the deadline passing stops them all."""
    import requests

    pending = list(handles)
    deadline = time.monotonic() + timeout_s
    try:
        while pending and time.monotonic() < deadline:
            for h in list(pending):
                if h.proc.poll() is not None:
                    raise RuntimeError(f"env server on {h.url} exited with {h.proc.returncode} — see {h.log_path}")
                try:
                    if requests.get(f"{h.url}/health", timeout=5).ok:
                        pending.remove(h)
                except requests.RequestException:
                    pass
            if pending:
                time.sleep(1.0)
        if pending:
            urls = ", ".join(h.url for h in pending)
            raise TimeoutError(f"env server(s) on {urls} not healthy after {timeout_s:.0f}s — see {pending[0].log_path}")
    except BaseException:
        for h in handles:
            h.stop()
        raise
    return handles


def spawn(env_block: dict[str, Any], port: int, log_path: Path, timeout_s: float = 600.0) -> ServerHandle:
    """One server, started and healthy: ``wait_healthy([launch(...)])[0]``."""
    return wait_healthy([launch(env_block, port, log_path)], timeout_s=timeout_s)[0]


# ── wire encodings an env uses in its replies ──


def png_b64(rgb: Any) -> str:
    """A uint8 HxWx3 image as base64 PNG — what the bridges decode."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.asarray(rgb)[:, :, :3].astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ndarray_wire(arr: Any) -> dict[str, Any]:
    """An array as the bridges read it (lossless): base64 bytes + dtype + shape."""
    import base64

    import numpy as np

    arr = np.asarray(arr)
    return {"__ndarray__": base64.b64encode(arr.tobytes()).decode("ascii"),
            "dtype": str(arr.dtype), "shape": list(arr.shape)}


# ── the server side ──


def instantiate(block: dict[str, Any]) -> Any:
    target = block["_target_"]
    mod, _, cls = target.rpartition(".")
    kwargs = {k: v for k, v in block.items() if k != "_target_" and k not in LAUNCH_KEYS}
    return getattr(importlib.import_module(mod), cls)(**kwargs)


def _json_default(obj: Any) -> Any:
    """Stray numpy scalars / arrays in a verb's reply (an env should encode
    its images itself; this only keeps a reply from failing outright)."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def make_handler(env: Any) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # one line per call, no addresses
            sys.stdout.write(f"[envserver] {fmt % args}\n")
            sys.stdout.flush()

        def _send(self, code: int, payload: Any) -> None:
            body = json.dumps(payload, default=_json_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                try:
                    self._send(200, {"ok": True, **env.info()})
                except Exception as exc:
                    self._send(503, {"ok": False, "error": repr(exc)})
                return
            self._send(404, {"error": f"no such route {self.path}"})

        def do_POST(self) -> None:
            verb = self.path.strip("/")
            method = getattr(env, verb, None) if verb and not verb.startswith("_") else None
            if method is None or not callable(method):
                self._send(404, {"error": f"no such verb {verb!r}"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            kwargs = json.loads(self.rfile.read(n) or b"{}") if n else {}
            try:
                self._send(200, method(**kwargs))
            except Exception as exc:
                traceback.print_exc()
                self._send(500, {"error": f"{verb}: {exc!r}"})

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, help="JSON: the experiment's env block")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    env = instantiate(json.loads(args.env))
    server = HTTPServer((args.host, args.port), make_handler(env))
    print(f"[envserver] {type(env).__name__} on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        close = getattr(env, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
