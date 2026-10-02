"""Fixed-access historical memory — BM25 over the task records and the house notes.

The reference harness lets the navigating session decide when to read its
records and which ones. ``retrieval: bm25`` is the control that takes that
choice away: the harness retrieves historical text on a fixed schedule — at
task start, after a recovery and every ``retrieval_every_turns``
navigation-model turns — with a fixed query (the goal, then the session's
latest description of what it sees), and puts the excerpts into the
conversation itself. The file tools are scoped to the current task's own
files in this configuration, as under ``ledger=none`` (which ``bm25``
holds), and Claude Code's own file tools are not opened on the house notes,
so the schedule cannot be bypassed; the map stays open.

What is indexed is historical text, never the current task's own files:

    run/ledger.md                one record per line (each line is one closed task)
    run/tasks/<j>/   j < k       goal.md, handover.md, lambda*.md (recovery notes), verdict.json
    house/                       MEMORY.md, house.md, rooms/*.md, skills/*.md, and the journal
                                 of earlier days (this run's own task records come from run/)

Records are cut into chunks of ``chunk`` tokens that overlap by ``overlap``.
BM25 (Okapi, k1 = 1.2, b = 0.75, the non-negative idf) scores every chunk
against the query, and ties favour newer records. The best ``top_k`` chunks
that fit a ``budget``-token budget become the excerpts — two chunks of one
record that overlap are merged into one excerpt — each shown as the original
text with its source and the recorded status of the task it belongs to (the
ledger's CERTIFIED / NOT DONE / UNRESOLVED / CLAIMED). Tokens are counted
with the four-characters-per-token approximation, so no tokenizer is needed;
the BM25 terms are lower-cased alphanumeric runs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .workspace import Workspace

K1 = 1.2
B = 0.75

#: the ledger's status words, as the ledger and the journal write them
STATUS = {"complete": "CERTIFIED", "incomplete": "NOT DONE", "unknown": "UNRESOLVED"}
_LEDGER_LINE = re.compile(r"^\s*(\d+)\.\s+(CERTIFIED|NOT DONE|UNRESOLVED|CLAIMED)\b")
_JOURNAL_LINE = re.compile(r"^-\s*task\s+(\d+)\s*·\s*([A-Z][A-Z ]+?)\b(?:\s*\(|\s*·)")
_TERM = re.compile(r"[a-z0-9]+")
_PIECE = re.compile(r"\S+\s*")

#: newer than any house file: a run record's recency is this plus its task index
_RUN_EPOCH = 1e12


def tokens(text: str) -> int:
    """Approximate model tokens: four characters each, at least one per word."""
    return sum(max(1, math.ceil(len(p.strip()) / 4)) for p in _PIECE.findall(text or ""))


def terms(text: str) -> list[str]:
    return _TERM.findall((text or "").lower())


@dataclass(eq=False)
class Record:
    """One historical text: where it is, what it says, the status of its task."""

    source: str                 # workspace-relative path (a ledger line: run/ledger.md)
    text: str
    status: str                 # CERTIFIED / NOT DONE / UNRESOLVED / CLAIMED, or "house note"
    task: int | None            # 1-based task number it belongs to, when it belongs to one
    recency: float              # larger = newer

    def header(self) -> str:
        where = f" · task {self.task}" if self.task else ""
        return f"[{self.source}{where} · {self.status}]"


@dataclass(eq=False)
class Chunk:
    record: Record
    start: int                  # piece range [start, end) of the record
    end: int


@dataclass(eq=False)
class Excerpt:
    record: Record
    start: int
    end: int
    score: float

    def text(self, pieces: list[str]) -> str:
        return "".join(pieces[self.start:self.end]).strip()


# ── the records ───────────────────────────────────────────────────────


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _task_status(task_dir: Path) -> str:
    try:
        verdict = json.loads((task_dir / "verdict.json").read_text(encoding="utf-8")).get("verdict")
    except (OSError, ValueError, AttributeError):
        return "CLAIMED"
    return STATUS.get(str(verdict), "CLAIMED")


def _verdict_text(p: Path) -> str:
    """The completion check's record, as the fields it states (not its bookkeeping)."""
    try:
        v = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(v, dict):
        return ""
    lines = [f"verdict: {v.get('verdict', '')}"]
    if v.get("evidence"):
        lines.append(f"evidence: {v['evidence']}")
    if v.get("searched"):
        lines.append("searched: " + ", ".join(str(s) for s in v["searched"]))
    return "\n".join(lines)


def historical_records(ws: Workspace, k: int, day: str = "") -> list[Record]:
    """Every historical text the fixed policy may retrieve while task ``k``
    (0-based) runs: earlier tasks' records, the ledger, the house notes and
    the journal of days other than ``day``."""
    out: list[Record] = []
    if ws.ledger.exists():
        for line in _read(ws.ledger).splitlines():
            m = _LEDGER_LINE.match(line)
            if not line.strip():
                continue
            task = int(m.group(1)) if m else None
            out.append(Record("run/ledger.md", line.strip(), m.group(2) if m else "CLAIMED", task,
                              _RUN_EPOCH + (task or 0)))
    dirs = sorted((d for d in ws.tasks.iterdir() if d.is_dir() and d.name.isdigit() and int(d.name) < k),
                  key=lambda d: int(d.name)) if ws.tasks.exists() else []
    for d in dirs:
        j = int(d.name)
        status = _task_status(d)
        files = [d / "goal.md", d / "handover.md", *sorted(d.glob("lambda*.md"))]
        for p in files:
            text = _read(p)
            if text:
                out.append(Record(ws.rel(p), text, status, j + 1, _RUN_EPOCH + j + 1))
        text = _verdict_text(d / "verdict.json")
        if text:
            out.append(Record(ws.rel(d / "verdict.json"), text, status, j + 1, _RUN_EPOCH + j + 1))
    house = ws.house
    notes = [house / "MEMORY.md", house / "house.md", *sorted((house / "rooms").glob("*.md")),
             *sorted((house / "skills").glob("*.md"))]
    for p in notes:
        text = _read(p)
        if text:
            out.append(Record(ws.rel(p), text, "house note", None, _mtime(p)))
    for p in sorted((house / "journal").glob("*.md")):
        if p.stem == day:
            continue                     # this run's own task records are indexed from run/
        for line in _read(p).splitlines():
            m = _JOURNAL_LINE.match(line)
            if m:
                out.append(Record(ws.rel(p), line.strip(), m.group(2).strip(), int(m.group(1)), _mtime(p)))
    return out


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


# ── chunks and BM25 ──────────────────────────────────────────────────


def chunk_spans(costs: list[int], size: int, overlap: int) -> list[tuple[int, int]]:
    """Piece ranges of at most ``size`` tokens, each re-reading the last
    ``overlap`` tokens of the one before (a piece longer than ``size`` is a
    chunk of its own)."""
    spans: list[tuple[int, int]] = []
    start, n = 0, len(costs)
    while start < n:
        end, used = start, 0
        while end < n and (end == start or used + costs[end] <= size):
            used += costs[end]
            end += 1
        spans.append((start, end))
        if end >= n:
            break
        back, nxt = 0, end
        while nxt > start + 1 and back < overlap:
            nxt -= 1
            back += costs[nxt]
        start = nxt
    return spans


def bm25(query: list[str], docs: list[list[str]], k1: float = K1, b: float = B) -> list[float]:
    """Okapi BM25 of every document against the query (a term repeated in
    the query counts each time), with the non-negative idf
    ``ln(1 + (N - n + 0.5) / (n + 0.5))``."""
    n_docs = len(docs)
    if not n_docs or not query:
        return [0.0] * n_docs
    avgdl = sum(len(d) for d in docs) / n_docs or 1.0
    df = Counter(t for d in docs for t in set(d))
    idf = {t: math.log(1.0 + (n_docs - df[t] + 0.5) / (df[t] + 0.5)) for t in set(query)}
    scores = []
    for d in docs:
        tf = Counter(d)
        norm = k1 * (1.0 - b + b * len(d) / avgdl)
        scores.append(sum((idf[t] * tf[t] * (k1 + 1.0) / (tf[t] + norm) for t in query if tf[t]), 0.0))
    return scores


# ── the policy ───────────────────────────────────────────────────────


@dataclass
class FixedRetriever:
    """The fixed-access policy's retrieval: what one scheduled call returns."""

    top_k: int = 5
    budget: int = 2000
    chunk: int = 400
    overlap: int = 50

    def retrieve(self, records: list[Record], query: str) -> tuple[list[Excerpt], dict[int, list[str]]]:
        """The excerpts for ``query``, best first, and each record's pieces
        (by ``id``) to render them from."""
        pieces: dict[int, list[str]] = {}
        chunks: list[Chunk] = []
        for rec in records:
            ps = _PIECE.findall(rec.text)
            pieces[id(rec)] = ps
            costs = [max(1, math.ceil(len(p.strip()) / 4)) for p in ps]
            chunks += [Chunk(rec, s, e) for s, e in chunk_spans(costs, self.chunk, self.overlap)]
        scores = bm25(terms(query), [terms("".join(pieces[id(c.record)][c.start:c.end])) for c in chunks])
        ranked = sorted((i for i, s in enumerate(scores) if s > 0.0),
                        key=lambda i: (-scores[i], -chunks[i].record.recency))
        chosen: list[Excerpt] = []
        accepted = 0
        for i in ranked:
            if accepted >= self.top_k:
                break
            c = chunks[i]
            mates = [x for x in chosen if x.record is c.record and x.start < c.end and c.start < x.end]
            joined = Excerpt(c.record, min([c.start] + [x.start for x in mates]),
                             max([c.end] + [x.end for x in mates]), max([scores[i]] + [x.score for x in mates]))
            # the merged excerpt keeps the place of the best chunk it absorbed
            at = chosen.index(mates[0]) if mates else len(chosen)
            trial = [x for x in chosen if x not in mates]
            trial.insert(min(at, len(trial)), joined)
            if self._cost(trial, pieces) > self.budget:
                continue
            chosen = trial
            accepted += 1
        return chosen, pieces

    @staticmethod
    def _cost(excerpts: list[Excerpt], pieces: dict[int, list[str]]) -> int:
        return sum(tokens(x.record.header()) + tokens(x.text(pieces[id(x.record)])) for x in excerpts)

    def render(self, records: list[Record], query: str) -> tuple[str, list[dict]]:
        """The block the harness puts into the conversation, and what it
        held (source, task, status, tokens) for the record."""
        if not records:
            return RETRIEVED_NONE.format(why="there are no earlier records yet"), []
        excerpts, pieces = self.retrieve(records, query)
        if not excerpts:
            return RETRIEVED_NONE.format(why="no earlier record matches the goal or what you described"), []
        parts, held = [RETRIEVED_HEAD], []
        for n, x in enumerate(excerpts, 1):
            text = x.text(pieces[id(x.record)])
            parts.append(f"({n}) {x.record.header()}\n{text}")
            held.append({"source": x.record.source, "task": x.record.task, "status": x.record.status,
                         "tokens": tokens(text), "score": round(x.score, 3)})
        return "\n\n".join(parts), held


RETRIEVED_HEAD = ("Retrieved from the records of earlier tasks and the house notes (the harness "
                  "chooses these: the best matches for the goal and your latest description; each "
                  "is shown with its source and the status recorded for its task):")
RETRIEVED_NONE = "Retrieved from the records of earlier tasks and the house notes: nothing ({why})."


__all__ = ["B", "K1", "Excerpt", "FixedRetriever", "Record", "bm25", "chunk_spans", "historical_records",
           "terms", "tokens"]
