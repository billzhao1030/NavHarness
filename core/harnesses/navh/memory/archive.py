"""The archive: which experience is worth keeping, and for how long.

The house notes (``longterm.py``) give a house its memory. That is storage.
What a deployment that runs for weeks needs on top of storage is a policy, and
it has to answer four questions storage alone leaves open:

    promotion   which of today's episodes deserve to become standing knowledge
    conflict    when today contradicts last week, which one does tomorrow read
    retrieval   which notes should be in context for THIS task, not all of them
    forgetting  how long a "not here" is allowed to keep steering the robot

Three classes, and the promotion rule is the load-bearing part:

    episodic     one thing that happened once      (cheap, expires)
    semantic     how this house is                 (rooms, relations, routes)
    procedural   how to do something here          (a search or recovery habit)

An episode is promoted only when it has been CERTIFIED more than once, or when
it is certified and explicitly reusable. Everything else stays episodic and
ages out. The reason to be strict is the failure this whole tier invites: a
single confident mistake, promoted, becomes a fact the robot defends for weeks.

Negative evidence is the sharpest case and gets its own rule. "The mug is not
in the kitchen" is true on the day it is observed and quietly rots afterwards,
because someone puts the mug in the kitchen. So it expires — unless it is
re-confirmed, which resets its clock.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .workspace import scrub_coordinates

CLASSES = ("episodic", "semantic", "procedural")
_WORD = re.compile(r"[a-z][a-z0-9_-]{2,}")


def _today() -> str:
    return _dt.date.today().isoformat()


@dataclass
class Item:
    text: str
    kind: str = "episodic"           # one of CLASSES
    certified: bool = False
    negative: bool = False           # a "not here" claim, which is the perishable kind
    places: list[str] = field(default_factory=list)
    seen: int = 1                    # how many deployments have confirmed it
    first_day: str = ""
    last_day: str = ""
    superseded_by: str = ""          # set when a later, certified item contradicts this one
    task: int = -1

    def live(self, today: str, *, negative_days: int = 14) -> bool:
        """Is this still allowed to steer the robot?"""
        if self.superseded_by:
            return False
        if not self.negative:
            return True
        try:
            age = (_dt.date.fromisoformat(today) - _dt.date.fromisoformat(self.last_day)).days
        except (ValueError, TypeError):
            return True
        return age <= negative_days

    def key_words(self) -> set[str]:
        return set(_WORD.findall(self.text.lower())) | {p.lower() for p in self.places}


class Archive:
    """A policy over one house's items. Sits beside the house's prose notes;
    this holds the things a policy has to reason about individually."""

    def __init__(self, path: Path, *, negative_days: int = 14, promote_after: int = 2) -> None:
        self.path = Path(path)
        self.negative_days = int(negative_days)
        self.promote_after = int(promote_after)
        self.items: list[Item] = []
        self._load()

    # ── writing ──────────────────────────────────────────────────────────

    def add(self, text: str, *, certified: bool, negative: bool = False,
            places: list[str] | None = None, kind: str = "episodic",
            task: int = -1, day: str | None = None) -> Item:
        """Record one thing. Merging happens here, not at read time, so that
        ``seen`` actually counts confirmations rather than duplicates."""
        day = day or _today()
        text = scrub_coordinates(str(text))[0].strip()[:400]
        it = Item(text=text, kind=kind if kind in CLASSES else "episodic",
                  certified=bool(certified), negative=bool(negative),
                  places=[str(p)[:48] for p in (places or [])][:6],
                  first_day=day, last_day=day, task=int(task))
        same = self._same_as(it)
        if same is not None:
            same.seen += 1
            same.last_day = day
            same.certified = same.certified or it.certified
            self._maybe_promote(same)
            self._save()
            return same
        self._resolve_conflicts(it)
        self.items.append(it)
        self._maybe_promote(it)
        self._save()
        return it

    def _same_as(self, it: Item) -> Item | None:
        w = it.key_words()
        for other in self.items:
            if other.negative != it.negative or other.superseded_by:
                continue
            ow = other.key_words()
            if w and ow and len(w & ow) / len(w | ow) >= 0.7:
                return other
        return None

    def _resolve_conflicts(self, it: Item) -> None:
        """A certified positive claim about a place supersedes a live negative
        one about the same place, and vice versa. Both stay on disk: the record
        of having been wrong is worth more than a tidy file."""
        if not it.certified or not it.places:
            return
        for other in self.items:
            if other.superseded_by or other.negative == it.negative:
                continue
            if ({p.lower() for p in other.places} & {p.lower() for p in it.places}
                    and len(other.key_words() & it.key_words()) >= 2):
                other.superseded_by = f"{it.first_day}:{it.text[:40]}"

    def _maybe_promote(self, it: Item) -> None:
        """Episodic -> semantic/procedural. Confirmation is the currency."""
        if it.kind != "episodic" or not it.certified:
            return
        if it.seen >= self.promote_after:
            it.kind = "procedural" if it.text.lower().startswith(("to ", "when ", "if ")) else "semantic"

    # ── reading ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, *, k: int = 6, day: str | None = None) -> list[Item]:
        """The notes worth putting in context for THIS task.

        Scored on word overlap with the goal, then by class and confirmation,
        because a semantic fact confirmed three times should outrank an episode
        that merely shares a noun."""
        day = day or _today()
        q = set(_WORD.findall((query or "").lower()))
        live = [i for i in self.items if i.live(day, negative_days=self.negative_days)]
        weight = {"semantic": 2.0, "procedural": 2.0, "episodic": 1.0}

        def score(i: Item) -> float:
            overlap = len(q & i.key_words())
            return overlap * weight[i.kind] + 0.25 * min(i.seen, 4) + (0.5 if i.certified else 0)

        ranked = sorted((i for i in live if score(i) > 0), key=score, reverse=True)
        return ranked[:k]

    def render(self, query: str, *, k: int = 6, day: str | None = None) -> str:
        got = self.retrieve(query, k=k, day=day)
        if not got:
            return ""
        out = ["From what you have learned in this house before:"]
        for i in got:
            tag = "known" if i.certified and i.kind != "episodic" else (
                "certified once" if i.certified else "claimed")
            out.append(f"  - [{tag}] {i.text}")
        return "\n".join(out)

    def stats(self, day: str | None = None) -> dict:
        """The tier's size, and how much of it is still allowed to steer
        anything."""
        day = day or _today()
        live = [i for i in self.items if i.live(day, negative_days=self.negative_days)]
        return {"items": len(self.items), "live": len(live),
                "by_class": {c: sum(1 for i in live if i.kind == c) for c in CLASSES},
                "negative_live": sum(1 for i in live if i.negative),
                "superseded": sum(1 for i in self.items if i.superseded_by),
                "expired": len(self.items) - len(live) - sum(1 for i in self.items if i.superseded_by)}

    # ── disk ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        for raw in d.get("items") or []:
            try:
                self.items.append(Item(**raw))
            except (TypeError, KeyError):
                continue

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"items": [asdict(i) for i in self.items]},
                                        indent=2, ensure_ascii=False))
