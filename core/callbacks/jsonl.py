"""JsonlLogger — the episode trajectory file, episode_{i}.jsonl.

One line per bus event, ``{"t": <s since episode start>, "kind": …,
**payload}``, flushed as written so a live ``tail -f`` sees every event.
The vocabulary is whatever the adapters emit (core.bus); this class only
persists it. Opened fresh per episode (a resume that re-runs an index
overwrites; a rate-limit retry's previous attempt is archived by the
runner first). Without it there is no episode_{i}.jsonl — run_stats goes
blind, the summary record still lands.
"""

from __future__ import annotations

import json
from typing import Any

from core.callbacks.base import Callback


class JsonlLogger(Callback):
    writes = ("episode_{i}.jsonl",)

    def __init__(self) -> None:
        self._files: dict[int, Any] = {}

    def on_episode_start(self, ep: Any) -> None:
        self._files[ep.index] = (ep.run_dir / f"episode_{ep.index}.jsonl").open("w")

    def on_event(self, ep: Any, kind: str, rec: dict[str, Any]) -> None:
        fh = self._files.get(ep.index)
        if fh is None:
            return
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    def on_episode_end(self, ep: Any, record: dict[str, Any] | None) -> None:
        fh = self._files.pop(ep.index, None)
        if fh is not None:
            fh.close()
