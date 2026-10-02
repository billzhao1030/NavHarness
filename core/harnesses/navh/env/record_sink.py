"""The harness's record, forwarded to the runner's event bus — ours.

``record.py::Recorder`` writes NavHarness's own paper trail
(``episode_{k}.jsonl`` per task, ``live_{k}/`` frames, ``summary.json``)
into the harness's out_dir. The runner's callbacks only see what comes
through the bus (``core.bus.EventBus``), so this subclass forwards every
``emit`` as it happens — the four kinds the bus vocabulary already has
(tool_use / tool_result / thinking / assistant_text) under their own names,
everything else (usage, boundary, recovery, warning, episode_meta, …) under
a ``navh_`` prefix so a task-level event is never mistaken for the runner's
episode-level one — and hands the harness-native record to ``sink.raw``.

Frames: the harness writes a task's ``obs_*.png`` into ``live/`` and moves
them to ``live_{k}/`` when the task's stream closes. Every ``tool_result``
forwarded to the bus carries ``frames`` — the files that appeared in
``live/`` since the previous result, i.e. what THAT call produced (an
observe's frame, close_task's four stop views; nothing for get_map, whose
picture goes to the model but not to disk) — and those files are
hard-linked into the runner's ``live_{i}/`` at that moment (running ids, so
the names never clash), which is where the monitor looks and how it
attributes a picture to a call. The harness's own copy is left exactly where
it expects it (its judge and wake-up read those paths).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from core.panel import json_safe

from ..record import Recorder

BUS_KINDS = frozenset({"tool_use", "tool_result", "thinking", "assistant_text"})


class SinkRecorder(Recorder):
    def __init__(
        self, out_dir: Path, frames_dir: Path, *, sink: Any, live_dir: Path | None
    ) -> None:
        super().__init__(out_dir, frames_dir)
        self.sink = sink
        self.live_dir = live_dir
        self.task_index = -1  # the harness's task counter, stamped on every forwarded event
        self._seen: set[str] = set()  # frames already attributed to a tool_result

    def open_task(self, k: int, meta: dict) -> None:
        self.task_index = k
        super().open_task(k, meta)

    def _new_frames(self) -> list[str]:
        """The obs_*.png that landed in the frames dir since the last call,
        linked into the runner's live folder; their names."""
        try:
            names = sorted(
                f.name for f in self.frames_dir.glob("obs_*.png") if f.name not in self._seen
            )
        except OSError:
            return []
        for name in names:
            self._seen.add(name)
            self._link(self.frames_dir / name)
        return names

    def _link(self, src: Path) -> None:
        if self.live_dir is None:
            return
        dst = self.live_dir / src.name
        try:
            self.live_dir.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    dst.write_bytes(src.read_bytes())
        except OSError as exc:
            print(f"[navh] live frame not linked: {exc!r}")

    def emit(self, kind: str, **payload: Any) -> None:
        super().emit(kind, **payload)
        if self.sink is None:
            return
        rec = json_safe(payload)
        if kind == "tool_result":
            rec["frames"] = self._new_frames()
        try:
            if kind in BUS_KINDS:
                self.sink.emit(kind, {**rec, "task": self.task_index})
            else:
                self.sink.emit(f"navh_{kind}", {**rec, "task": self.task_index})
            self.sink.raw({"type": "navh", "kind": kind, "task": self.task_index, **rec})
        except Exception as exc:
            print(f"[navh] sink forward failed ({kind}): {exc!r}")

    def close_stream(self) -> None:
        # frames a task produced after its last tool_result (none in practice,
        # but the folder is about to be moved) are linked before the move
        self._new_frames()
        super().close_stream()


def exit_content(summary: dict[str, Any], n_goals: int) -> str:
    """The episode-ending step result, as the bus parses it off the ``exit``
    event (``steps_taken_total`` + ``end_reason`` — what the runner's record
    reads for env_steps / called_stop)."""
    rows = summary.get("tasks") or []
    steps = int((summary.get("aggregate") or {}).get("steps") or 0)
    status = str(summary.get("exit_status") or "")
    last = str(rows[-1].get("end_reason") or "") if rows else ""
    if status != "chain_complete":
        reason = "chain_incomplete"
    elif last == "budget":
        reason = "step_budget_exhausted"
    elif last == "turns":
        reason = "turn_budget_exhausted"
    elif last == "inactive":
        reason = "inactive"
    else:
        reason = "subtasks_closed" if n_goals > 1 else "stop_called"
    return json.dumps(
        {
            "steps_taken_total": steps,
            "end_reason": reason,
            "episode_over": True,
            "tasks": len(rows),
            "exit_status": status,
        }
    )
