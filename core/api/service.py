"""Loopback service base — an owned subprocess behind a health endpoint.

Everything this package materializes (wire gateways, later local model
servers) has the same lifecycle: bind a port from the pool, spawn the
subprocess with its log under ``outputs/_api_gateway/``, poll health until
alive — dying early is a hard error pointing at the log — terminate on
stop. Subclasses fill in the argv and, optionally, env and cwd.

The master key is a loopback auth token, not a secret: the gateways refuse
unauthenticated calls, Claude Code needs SOMETHING in
``ANTHROPIC_AUTH_TOKEN``, and codex needs an env_key variable to exist —
one constant satisfies all three without inventing key management.
"""

from __future__ import annotations

import socket
import subprocess
import time
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # the runner lives at the repo root
LOG_DIR = REPO_ROOT / "outputs" / "_api_gateway"  # underscore: not a run root
MASTER_KEY = "sk-navharness-gateway"


class LoopbackService(ABC):
    PORT_POOL: tuple[int, ...] = ()
    HEALTH_PATH = "/health/liveliness"
    START_TIMEOUT = 120.0
    POLL_INTERVAL = 1.0

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.port: int | None = None
        self._proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def log_path(self) -> Path:
        return LOG_DIR / f"{self.tag}_{self.port}.log"

    @abstractmethod
    def command(self) -> list[str]:
        """argv to spawn — called with ``self.port`` already bound, so it
        may also drop port-named side files (config yaml) next to the log."""

    def spawn_env(self) -> dict[str, str] | None:
        """None = inherit the parent environment unchanged."""
        return None

    def spawn_cwd(self) -> str | None:
        return None

    def _free_port(self) -> int:
        for port in self.PORT_POOL:
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        raise RuntimeError(
            f"no free port in {self.PORT_POOL[0]}-{self.PORT_POOL[-1]} for {self.tag}")

    def start(self, timeout: float | None = None) -> int:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.port = self._free_port()
        timeout = timeout or self.START_TIMEOUT
        log = self.log_path.open("a")
        self._proc = subprocess.Popen(  # noqa: S603
            self.command(), cwd=self.spawn_cwd(), env=self.spawn_env(),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"{self.tag} endpoint died on startup "
                    f"(rc={self._proc.returncode}) — see {self.log_path}")
            try:
                with urllib.request.urlopen(
                        f"{self.base_url}{self.HEALTH_PATH}", timeout=2):
                    return self.port
            except Exception:  # noqa: BLE001
                time.sleep(self.POLL_INTERVAL)
        self.stop()
        raise RuntimeError(
            f"{self.tag} endpoint failed to come up on :{self.port} in {timeout}s")

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None
