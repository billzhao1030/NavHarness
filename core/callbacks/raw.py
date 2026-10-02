"""RawLogger — the harness-native stream, raw/episode_{i}.jsonl.

Every record an adapter hands to ``sink.raw`` (a Claude SDK message as
``{"type": <class>, "msg": …}``, a codex event as ``{"event": …}``; the
mini-swe-agent loop has no raw stream) written as-is, one JSON line each,
``ensure_ascii`` off. Opened on the first record,
so an adapter that emits none leaves no file. This is what CostLogger and
reporting/run_stats read per-API-call usage from; without it the cost
split by tool and the token curves are gone, the CLI-settled totals stay.
"""

from __future__ import annotations

import json
from typing import Any

from core.callbacks.base import Callback


class RawLogger(Callback):
    writes = ("raw/episode_{i}.jsonl",)

    def __init__(self) -> None:
        self._files: dict[int, Any] = {}

    def on_raw(self, ep: Any, rec: Any) -> None:
        fh = self._files.get(ep.index)
        if fh is None:
            ep.raw_dir.mkdir(parents=True, exist_ok=True)
            fh = (ep.raw_dir / f"episode_{ep.index}.jsonl").open("w")
            self._files[ep.index] = fh
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()

    def on_episode_end(self, ep: Any, record: dict[str, Any] | None) -> None:
        fh = self._files.pop(ep.index, None)
        if fh is not None:
            fh.close()
