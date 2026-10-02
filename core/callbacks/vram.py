"""VramSampler — peak GPU memory over the run, summary.run_stats.vram_peak_mib.

A 0.5 s ``nvidia-smi`` poll on a daemon thread from on_run_start to
on_run_end. Silent no-op (``null``) where nvidia-smi is absent, so it costs
nothing on API-billed runs. The number is the box's total, not the
run's share — sibling servers on the same GPU count.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Any

from core.callbacks.base import Callback


class VramSampler(Callback):
    writes = ("run_stats.vram_peak_mib",)

    def __init__(self, period_s: float = 0.5) -> None:
        self.peak_mib = 0
        self._period = period_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._period):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.split()
                if out:
                    self.peak_mib = max(self.peak_mib, int(out[0]))
            except Exception:  # noqa: BLE001 — sampling must never break a run
                pass

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.peak_mib or None

    def on_run_start(self, run: Any) -> None:
        self.start()

    def on_run_end(self, run: Any, summary: dict[str, Any]) -> None:
        summary["run_stats"]["vram_peak_mib"] = self.stop()
