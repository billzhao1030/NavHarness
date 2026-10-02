"""House notes — the long-term tier, one directory per house.

The layout a coding agent keeps under ``.claude/``: an index that every task
is told to read, note files read on demand, and a journal appended as work
happens; consolidation at the end of a run turns the run's records into notes.

    <memory_root>/<house>/          (the workspace's ``house/``)
      MEMORY.md          the index: one line per note ("title — hook")
      house.md           the house overview: rooms, how they connect, landmarks,
                         routes that worked; certified facts carry their day · task,
                         everything else is written as a claim
      rooms/<room>.md    a room note: what is there, what was searched and NOT found
      rooms/<room>/photos/ + photos.json
                         a marked place's views: at most four, one per heading
                         (front, right, back, left), each with the navigator's
                         pose estimate; a re-mark replaces a heading's view
      skills/<slug>.md   a navigation skill: when to use it, the steps, the evidence
      journal/<day>.md   the task records of that day, one line per closed task,
                         appended by the harness at every task boundary

Who writes what: the harness appends the journal and files the photo sets;
the consolidation session — file tools only — is the only writer of the four
house-note files (the index, the overview, the room notes and the skills); a
navigation session only reads. Nothing here is a ground-truth coordinate:
places are addressed by room name, marker label and relation (an estimated
``slam(x, z, yaw)`` is allowed, and a view's pose estimate is one).
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
from pathlib import Path

from .workspace import scrub_coordinates

_SAFE = re.compile(r"[^a-z0-9._-]+")

#: the files the consolidation session may write, relative to the workspace
NOTE_PATTERNS = ("house/MEMORY.md", "house/house.md", "house/rooms/*.md", "house/skills/*.md")

#: the four look-around views a marker banks, in the order the bridge takes them
_HEADINGS = ("front", "right", "back", "left")


def slug(s: str) -> str:
    return _SAFE.sub("-", str(s).strip().lower().rstrip("?")).strip("-")[:48] or "note"


class HouseNotes:
    def __init__(self, house_dir: Path, *, day: str = "") -> None:
        self.dir = Path(house_dir)
        self.day = day or _dt.date.today().isoformat()
        for d in ("rooms", "skills", "journal"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)

    # ── paths ────────────────────────────────────────────────────────

    @property
    def index(self) -> Path:
        return self.dir / "MEMORY.md"

    @property
    def overview(self) -> Path:
        return self.dir / "house.md"

    @property
    def journal_today(self) -> Path:
        return self.dir / "journal" / f"{self.day}.md"

    def is_new(self) -> bool:
        """No consolidated notes yet: the first run in this house."""
        return not self.index.exists()

    def list_notes(self) -> list[str]:
        return sorted(p.relative_to(self.dir).as_posix() for p in self.dir.rglob("*.md")
                      if p.name != "MEMORY.md")

    # ── the journal: the task records, appended at every boundary ────

    def journal(self, k: int, goal: str, label: str, steps: int, markers: list[str], account: str) -> str:
        """One line per closed task, in the order they closed. ``label`` is the
        ledger's status (CERTIFIED / NOT DONE / UNRESOLVED / CLAIMED / stop);
        ``account`` the session's own handover, kept as the robot wrote it."""
        line = f"- task {k + 1} · {label} · {goal}"
        if markers:
            line += f" · markers: {', '.join(markers)}"
        if steps:
            line += f" · {steps} steps"
        if account:
            line += " · account: " + " ".join(account.split())
        line = scrub_coordinates(line)[0]
        new = not self.journal_today.exists()
        with self.journal_today.open("a", encoding="utf-8") as fh:
            if new:
                fh.write(f"# {self.day}\n\n")
            fh.write(line + "\n")
        return line

    # ── photo sets: a place is a marker plus the views taken there ───

    def attach_photos(self, label: str, views: list[dict], task_index: int) -> int:
        """File a marker's banked views under ``rooms/<room>/photos/`` and index
        them in ``rooms/<room>/photos.json``, so a later run can look at the
        place before searching it and a wake-up can compare against it.

        A place keeps at most four views, one per heading: a view replaces the
        one of its heading (same file name, ``<heading>.png``), so a label
        re-marked on a later task or day keeps its latest set, and a note that
        names a view keeps pointing at that heading. Each entry carries the
        label, the day and task, and the view's pose estimate. Returns how many
        views were filed; views already filed from the same mark are not
        filed again."""
        room = slug(label)
        d = self.dir / "rooms" / room / "photos"
        d.mkdir(parents=True, exist_ok=True)
        pj = self.dir / "rooms" / room / "photos.json"
        kept: dict[str, dict] = {}
        for v in self._read_index(pj):
            kept[str(v.get("heading") or "")] = v            # the latest entry of a heading wins
        n = 0
        for i, v in enumerate(views[:len(_HEADINGS)]):
            src = Path(str(v.get("path") or ""))
            if not src.is_file():
                continue
            heading = str(v.get("heading") or _HEADINGS[i])
            entry = {"file": f"{heading}.png", "label": label, "heading": heading, "day": self.day,
                     "task": task_index + 1, "step": v.get("step"), "pose": v.get("pose")}
            old = kept.get(heading)
            if old and all(old.get(k) == entry[k] for k in ("label", "day", "task", "step")):
                continue
            if src.resolve() == (d / entry["file"]).resolve():
                continue                                     # the view filed here already (an earlier run's)
            shutil.copy(src, d / entry["file"])
            kept[heading] = entry
            n += 1
        if n:
            order = {h: i for i, h in enumerate(_HEADINGS)}
            index = sorted(kept.values(), key=lambda e: order.get(str(e.get("heading")), len(order)))
            pj.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
        return n

    @staticmethod
    def _read_index(pj: Path) -> list[dict]:
        try:
            got = json.loads(pj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [v for v in got if isinstance(v, dict)] if isinstance(got, list) else []

    def photo_index(self) -> dict[str, list[dict]]:
        """room -> its ``photos.json`` entries."""
        out: dict[str, list[dict]] = {}
        for pj in sorted((self.dir / "rooms").glob("*/photos.json")):
            entries = self._read_index(pj)
            if entries:
                out[pj.parent.name] = entries
        return out

    def photo_views(self, labels: list[str] | tuple[str, ...] = ()) -> dict[str, list[dict]]:
        """The kept views of every place marked on an earlier run, as the
        wake-up reads banked views: label -> ``[{id, path, heading, pose, …}]``
        with absolute paths. A set is named by the label its entries record;
        ``labels`` (the place graph's) name a set filed without one."""
        by_room = {slug(lab): lab for lab in labels}
        out: dict[str, list[dict]] = {}
        for room, entries in self.photo_index().items():
            for e in entries:
                p = self.dir / "rooms" / room / "photos" / str(e.get("file") or "")
                if not p.is_file():
                    continue
                lab = str(e.get("label") or by_room.get(room) or room)
                out.setdefault(lab, []).append({**e, "id": f"{lab}:{e.get('heading', '')}", "path": str(p)})
        return out


__all__ = ["NOTE_PATTERNS", "HouseNotes", "slug"]
