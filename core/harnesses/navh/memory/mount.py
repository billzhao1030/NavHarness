"""House — the memory tiers of ONE house, mounted for a chain and fed at every seam.

Three tiers outlive a task, and a chain mounts whichever its config asks for:

* the place graph (``placegraph.py``) — the shape of the house: every marker
  the agent dropped becomes a place; it outlives the process like the map does,
  and is forgotten with the map when a boundary does not keep it;
* the archive (``archive.py``) — what a closed task leaves behind, kept as a
  FACT only when the judge certified it, otherwise as a claim;
* the house notes — ``house/`` in the workspace (``longterm.py``): the
  journal appended at every seam, the index, overview, room notes and skills
  rewritten by the consolidation session at run end, read by the agent's sessions.

``absorb_*`` is the bite each code tier takes at a task's end. The
orchestrator owns the order these happen in; this module owns what they are.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..config import HarnessConfig
from ..env.bench import Bench
from ..env.types import Goal
from .archive import Archive
from .placegraph import PlaceGraph


class House:
    def __init__(self, cfg: HarnessConfig, bench: Bench, out_dir: Path, rec: Any, *,
                 memory_root: Path, seat_arg: Any) -> None:
        self.cfg = cfg
        self.bench = bench
        self.out_dir = out_dir
        self.rec = rec
        self.memory_root = memory_root
        self.seat_arg = seat_arg
        self.places: PlaceGraph | None = None      # the shape of this house
        self.places_path: Path | None = None
        self.archive: Archive | None = None        # what is worth keeping, and for how long

    @property
    def house_id(self) -> str:
        return self.bench.house_id or f"{self.cfg.task.benchmark}-{self.seat_arg}"

    @property
    def house_dir(self):
        """This house's own directory under the memory root (the workspace's house/)."""
        return self.memory_root / house_slug(self.house_id)

    # ── mounting ─────────────────────────────────────────────────────

    def mount_places(self, frozen: Path | None) -> None:
        """The place graph, when the ladder or the graph itself is on. It
        outlives the process for the same reason the map does: it is a fact
        about the house, not about this run. A replay starts from the graph as
        it stood at its boundary (``frozen``)."""
        cfg = self.cfg
        if not (cfg.ladder.recovery or cfg.memory.place_graph):
            return
        gp = (Path(cfg.memory.map_state).with_suffix(".places.json")
              if cfg.memory.map_state else self.out_dir / "places.json")
        if frozen is not None and frozen.exists():
            self.places = PlaceGraph.load(frozen)
        else:
            self.places = PlaceGraph.load(gp) if gp.exists() else PlaceGraph()
        self.places_path = gp
        self.rec.emit("place_graph", places=len(self.places.places),
                      connectors=len(self.places.connectors),
                      floors=list(self.places.floors()))

    def mount_archive(self) -> None:
        """The archive, when its policy is on."""
        if self.cfg.memory.archive_policy:
            # Through the SAME slug the notes use: a mixed-case scene id
            # must not become two directories differing only in case.
            self.archive = Archive(self.house_dir / "archive.json")
            self.rec.emit("archive", **self.archive.stats())

    # ── at the seam ──────────────────────────────────────────────────

    def absorb_places(self, k: int, marks: list[str], floor: int, marker_photos: dict) -> None:
        """Every marker the agent dropped this task becomes a place."""
        if self.places is None:
            return
        for lab in marks:
            self.places.note_mark(lab, floor, photos=len(marker_photos.get(lab) or []), task=k)
        try:
            self.places.save(self.places_path)
        except OSError as e:
            self.rec.emit("warning", kind_detail="place_graph_save", error=str(e)[:200])

    def forget_places(self) -> None:
        """The place graph goes with the map: when a boundary rebuilds the map
        and its markers (``persist_map=False``), the places named on them are
        forgotten too, so no recovery brief can name a place from an earlier
        task. The emptied graph is what is saved, and what a checkpoint holds."""
        if self.places is None:
            return
        self.places.clear()
        try:
            self.places.save(self.places_path)
        except OSError as e:
            self.rec.emit("warning", kind_detail="place_graph_save", error=str(e)[:200])
        self.rec.emit("place_graph", places=0, connectors=0, floors=[], forgotten=True)

    def absorb_archive(self, k: int, goal: Goal, summary: str, verdict: str, marks: list[str]) -> None:
        """Only the judge's verdict decides whether this becomes a fact. A
        claim the robot merely made about itself is recorded as a claim,
        which is what keeps one confident mistake from becoming doctrine."""
        if self.archive is None:
            return
        self.archive.add(f"{goal.text} — {summary}"[:400], certified=(verdict == "complete"),
                         negative=(verdict == "incomplete"), places=list(marks), task=k)
        self.rec.emit("archive", **self.archive.stats())


_SAFE = re.compile(r"[^a-z0-9._-]+")


def house_slug(s: str) -> str:
    return _SAFE.sub("-", s.strip().lower()).strip("-")[:48] or "note"
