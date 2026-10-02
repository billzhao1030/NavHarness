"""StatsReport — stats.json + stats.html after the last episode.

Fires on_run_written (it reads summary.json, episode_{i}.jsonl and raw/
from disk) and delegates to reporting.run_stats.generate — charts, per-tool
cost apportioning, per-episode notes. Best-effort: a failed report never
loses a run. Without it: no stats files; backfill any time with
``python reporting/run_stats.py <run_dir>``.
"""

from __future__ import annotations

from typing import Any

from core.callbacks.base import Callback


class StatsReport(Callback):
    writes = ("stats.json", "stats.html")

    def on_run_written(self, run: Any, summary_path: Any) -> None:
        try:
            from reporting.run_stats import generate
            print(f"[std] stats report -> {generate(run.run_dir)}")
        except Exception as exc:  # noqa: BLE001 — a failed report never loses a run
            print(f"[std] stats report failed (non-fatal): {exc!r}")
