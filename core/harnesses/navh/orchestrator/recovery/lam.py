"""λ — what crosses a recovery.

Prose is a bad carrier for what a dying attempt should hand on: a model
writes narrative when asked for constraints ("I went to the kitchen and
then…"), and prose cannot be checked or counted.

So λ is six fields, and only six:

    λ = (searched, ruled_out, last_anchor, untried, uncertain, recommended_scope)

The split that matters most is ``searched`` vs ``ruled_out``. "I looked in the
kitchen" and "the mug is not in the kitchen" are different claims, and a robot
that promotes the first into the second stops looking in the one place the
thing actually is. The long-term memory already draws this line between a
certified fact and a claim; λ draws it at the recovery boundary.

``recommended_scope`` is λ's other job: the note is written by the attempt that
is about to die, which is also the only witness to HOW it died. The controller
treats it as a proposal, never as an instruction.

What the next attempt is handed can be replaced, to ask what the note's
content is worth (``HANDOVERS``, chosen by ``LadderConfig.recovery_handover``):

    structured   the attempt's own λ, as it wrote it (the reference)
    empty        no note: the attempt writes none and the next one reads none
    other        another task's λ in place of the attempt's own; the attempt
                 still writes its own, which is banked for later tasks to lend
    summary      an ordinary prose summary of the attempt, cut to the mean
                 length of the structured notes it is compared with

``other`` and ``summary`` need structured notes from somewhere: notes to lend,
and a length to match. ``load_bank`` reads them from an earlier run.
"""
from __future__ import annotations

import json
import re
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ...memory.workspace import scrub_coordinates
from .ladder import MODEL_CHOICES

_JSON = re.compile(r"\{.*\}", re.S)

#: What a recovery can hand the next attempt; ``structured`` is the reference.
HANDOVERS: tuple[str, ...] = ("structured", "empty", "other", "summary")

#: λ's fields that carry text (``recommended_scope`` is a choice, not content).
_TEXT_FIELDS = ("searched", "ruled_out", "last_anchor", "untried", "uncertain")


@dataclass
class Lambda:
    searched: list[str] = field(default_factory=list)      # places actually visited and looked at
    ruled_out: list[str] = field(default_factory=list)     # places where the goal is NOT, with evidence
    last_anchor: str = ""                                  # the last place we were sure of
    untried: list[str] = field(default_factory=list)       # the promising thing not yet attempted
    uncertain: list[str] = field(default_factory=list)     # guesses, explicitly flagged as guesses
    recommended_scope: str = "attempt"

    def is_empty(self) -> bool:
        return not (self.searched or self.ruled_out or self.untried or self.last_anchor)

    def render(self) -> str:
        """What the next attempt reads. Fixed order, so the shape itself tells
        the model which lines are established and which are guesses."""
        if self.is_empty():
            return "(the previous attempt left no usable note)"
        out = []
        if self.last_anchor:
            out.append(f"Last place you were sure of: {self.last_anchor}.")
        if self.ruled_out:
            out.append("Established — the goal is NOT in: " + "; ".join(self.ruled_out) + ".")
        if self.searched:
            out.append("Already looked at (not proof it is absent): " + "; ".join(self.searched) + ".")
        if self.untried:
            out.append("Not yet tried, most promising first: " + "; ".join(self.untried) + ".")
        if self.uncertain:
            out.append("Only guesses, treat as unproven: " + "; ".join(self.uncertain) + ".")
        return "\n".join(out)

    def words(self) -> int:
        return _field_words(asdict(self))

    def as_record(self) -> dict:
        return asdict(self)


def _strs(v, cap: int = 6) -> list[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [scrub_coordinates(str(x))[0].strip()[:160] for x in v if str(x).strip()][:cap]


def parse(text: str) -> Lambda:
    """Read the model's JSON, and keep only what survives validation.

    A malformed note must degrade to an empty one rather than to a crash: a
    recovery that fails because the note would not parse is the worst possible
    trade, since the attempt is already being thrown away."""
    d = _json_object(text)
    if d is None:
        return Lambda()
    scope = str(d.get("recommended_scope") or "attempt").strip().lower()
    return Lambda(
        searched=_strs(d.get("searched")),
        ruled_out=_strs(d.get("ruled_out")),
        last_anchor=(_strs(d.get("last_anchor"), 1) or [""])[0],
        untried=_strs(d.get("untried")),
        uncertain=_strs(d.get("uncertain")),
        recommended_scope=scope if scope in MODEL_CHOICES else "attempt",
    )


def read_note(raw: str) -> Lambda:
    """The note an attempt wrote, as λ: JSON is parsed; anything else is kept
    whole as a single unproven line, which leaves the note empty."""
    raw = (raw or "").strip()
    if not raw:
        return Lambda()
    return parse(raw) if raw.startswith("{") else Lambda(uncertain=[raw[:1200]])


def _json_object(text: str) -> dict | None:
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    return d if isinstance(d, dict) else None


def _field_words(d: dict) -> int:
    n = 0
    for name in _TEXT_FIELDS:
        v = d.get(name)
        for x in (v if isinstance(v, list) else [v]):
            n += len(str(x).split()) if x else 0
    return n


def note_words(text: str) -> int:
    """How long a note is, in words of content. A structured note counts the
    words of its five text fields, so JSON's keys and punctuation do not count
    towards it; anything else counts every word."""
    text = (text or "").strip()
    if text.startswith("{"):
        d = _json_object(text)
        if d is not None:
            return _field_words(d)
    return len(text.split())


def trim_words(text: str, n: int) -> str:
    """The first ``n`` words of ``text``, its line breaks kept."""
    words = list(re.finditer(r"\S+", text or ""))
    if n <= 0 or len(words) <= n:
        return (text or "").strip()
    return text[:words[n - 1].end()].strip()


# ── what crosses, and where a borrowed note comes from ───────────────────

@dataclass
class Handover:
    """What one recovery hands the next attempt."""
    lam: Lambda                  # the structured note handed over (empty for `empty` and `summary`)
    source: str                  # own | own-empty | empty | other | other-unavailable | summary | summary-empty
    text: str                    # what the next attempt reads at the note's path ("" = no note)
    own: Lambda | None = None    # `other`: the attempt's own note, banked but not handed over
    words: int = 0               # the length of ``text`` (``note_words``)

    def render(self) -> str:
        """The note as the record shows it: λ in its fixed order, a summary as written."""
        if self.source in ("summary", "summary-empty", "empty", "other-unavailable"):
            return self.text or "(no note was handed over)"
        return self.lam.render()


@dataclass
class Banked:
    """A structured note another task wrote, kept to be lent or measured."""
    goal: str
    lam: Lambda
    words: int


def load_bank(path: str) -> list[Banked]:
    """Structured notes from an earlier run, to lend (``other``) or to match in
    length (``summary``).

    ``path`` is a run directory (every ``lessons.jsonl`` below it), one
    ``lessons.jsonl``, or a JSON list. A lessons row counts when the note it
    handed over was the attempt's own (``lam_source`` own), or when it kept the
    attempt's own note beside a borrowed one (``lam_own``); a JSON list holds
    λ's fields directly, with the ``goal`` they were written for when known.
    A note's length is the one recorded when it was handed over (``lam_words``,
    counted on the note as written) and otherwise its fields' words."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"recovery_handover_bank: no such file or directory: {p}")
    rows: list[dict] = []
    files = sorted(p.rglob("lessons.jsonl")) if p.is_dir() else [p]
    for f in files:
        text = f.read_text(encoding="utf-8")
        if f.suffix == ".json":
            data = json.loads(text)
            rows += [d for d in (data if isinstance(data, list) else []) if isinstance(d, dict)]
            continue
        for line in text.splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                rows.append(d)
    out: list[Banked] = []
    for row in rows:
        words = 0
        if isinstance(row.get("lam_own"), dict):
            d = row["lam_own"]
        elif isinstance(row.get("lam"), dict):
            if str(row.get("lam_source") or "own") != "own":
                continue
            d, words = row["lam"], int(row.get("lam_words") or 0)
        elif any(name in row for name in _TEXT_FIELDS):
            d = row
        else:
            continue
        lam = parse(json.dumps(d))
        if not lam.is_empty():
            out.append(Banked(goal=str(row.get("goal") or ""), lam=lam, words=words or lam.words()))
    return out


def mean_words(bank: list[Banked]) -> int:
    """The mean length of the notes in ``bank``, in words (0 for an empty bank)."""
    sizes = [b.words for b in bank if b.words > 0]
    return round(sum(sizes) / len(sizes)) if sizes else 0


def pick_other(pool: list[Banked], goal: str, attempt: int) -> Banked | None:
    """Another task's note for ``goal``: any note in the pool written for a
    different goal. The choice is a hash of the goal and the attempt, so a
    rerun lends the same note."""
    others = [b for b in pool if b.goal.strip() != goal.strip() and not b.lam.is_empty()]
    if not others:
        return None
    return others[zlib.crc32(f"{goal}\x00{attempt}".encode()) % len(others)]
