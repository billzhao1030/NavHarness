"""Workspace — the file tree a chain's sessions connect through.

Memory is files, not prompt text. Three writers, each with its own files:

    house/                     this house, across runs (``memory_root/<house>``; longterm.py)
      MEMORY.md                the index                  } written by the consolidation
      house.md                 the house overview         } session at run end, from the
      rooms/<room>.md          room notes                 } ledger, the handovers and the
      skills/<slug>.md         navigation skills          } notes it finds
      rooms/<room>/photos/     written by the harness: a marked place's views (at most
        + photos.json            four, one per heading) and their pose estimates
      journal/<day>.md         written by the harness: one line per closed task
      archive.json             written by code (the archive policy)
    run/
      ledger.md                written by the harness: one line per closed task
      tasks/<k>/
        goal.md                written by the harness before the task opens
        handover.md            written by the agent's session before it may close
        lambda.md              written by the agent's session before a recovery (replaced or
                               cut by the harness under recovery_handover other / summary)
        verdict.json           written by the harness from the judge's verdict

A session reads whatever it wants below the root and writes only the paths
its instruction named for it; ``resolve`` refuses everything else. Paths in
prompts and tool calls are relative to the root (``run/tasks/3/handover.md``).
A read may also be given a SCOPE (``ledger=none``: ``run/tasks/<k>``): then
only what is under it can be read, and a listing above it shows only the way
down to it — the other tasks' records, the ledger and ``house/`` are not
there for that session, though the harness still writes them. A stored
photograph is read as the picture (``session_read``).
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import re
from pathlib import Path

_COORD_PAIR = re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)"      # (1.5, -2.0)
                         r"|-?\d+\.\d+\s*,\s*-?\d+(?:\.\d+)?"                 # 1.5, -2
                         r"|-?\d+\s*,\s*-?\d+\.\d+")                          # 1, -2.0
_XZ_ASSIGN = re.compile(r"\b[xz]\s*=\s*-?\d+(?:\.\d+)?", re.I)
_METRES = re.compile(r"\b-?\d+(?:\.\d+)?\s*(?:m|meters?|metres?)\b", re.I)
# what must survive the scrub: dates, task / day numbers, week tags, and SLAM pose
# annotations written as slam(x, z, yaw). SLAM pose is estimated, not ground truth,
# so it may be attached to a place's photo set; bare coordinates in
# prose are still redacted, and ground-truth pose never enters any text.
_KEEP = re.compile(r"slam\([^)]*\)|\d{4}-\d{2}-\d{2}|\b\d{2}-\d{2}\b|\btasks? \d+\b|\bday \d+\b|\bW\d{2}\b", re.I)


def scrub_coordinates(text: str) -> tuple[str, int]:
    """Redact coordinate-like text. Integer lists such as "tasks 1, 2" or a
    date are not coordinates (a pair needs parentheses or a decimal point)."""
    keep: list[str] = []

    def stash(m: re.Match) -> str:
        keep.append(m.group(0))
        return f"\x00{len(keep) - 1}\x00"

    text = _KEEP.sub(stash, text)
    n = 0
    for pat in (_XZ_ASSIGN, _COORD_PAIR, _METRES):
        text, k = pat.subn("[redacted]", text)
        n += k
    text = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], text)
    return text, n


MAX_WRITE_CHARS = 6000

#: the pictures a session may open with ``read_file`` (a place's stored photographs), by suffix
IMAGE_FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}


def _matches(name: str, pattern: str) -> bool:
    """A writable path given as a pattern (``house/rooms/*.md``): matched part
    by part, so a wildcard never crosses a directory."""
    if not any(c in pattern for c in "*?["):
        return False
    a, b = name.split("/"), pattern.split("/")
    return len(a) == len(b) and all(fnmatch.fnmatchcase(x, y) for x, y in zip(a, b))


class Workspace:
    def __init__(self, root: Path, house_dir: Path) -> None:
        self.root = Path(root)
        self.run = self.root / "run"
        self.tasks = self.run / "tasks"
        self.ledger = self.run / "ledger.md"
        self.house = Path(house_dir)
        self.house.mkdir(parents=True, exist_ok=True)
        self.tasks.mkdir(parents=True, exist_ok=True)
        link = self.root / "house"
        # the house's own directory, reachable from inside the workspace as house/
        if not link.exists() and not link.is_symlink():
            with contextlib.suppress(OSError):
                link.symlink_to(self.house, target_is_directory=True)
        self.index = self.house / "MEMORY.md"      # the house notes' index
        self.overview = self.house / "house.md"    # the house overview

    # ── paths, as the sessions name them ─────────────────────────────

    def rel(self, p: Path) -> str:
        p = Path(p)
        if p.is_relative_to(self.house):
            return "house/" + str(p.relative_to(self.house))
        return str(p.relative_to(self.root))

    def task_dir(self, k: int) -> Path:
        d = self.tasks / str(k)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def goal_path(self, k: int) -> Path:
        return self.task_dir(k) / "goal.md"

    def handover_path(self, k: int) -> Path:
        return self.task_dir(k) / "handover.md"

    def lambda_path(self, k: int, attempt: int) -> Path:
        return self.task_dir(k) / (f"lambda_{attempt}.md" if attempt > 1 else "lambda.md")

    def verdict_path(self, k: int) -> Path:
        return self.task_dir(k) / "verdict.json"

    def resolve(self, name: str) -> Path:
        """A path a session named, made absolute and kept inside the tree."""
        name = str(name or "").strip().lstrip("/")
        if not name or ".." in Path(name).parts:
            raise PermissionError(f"path outside the workspace: {name!r}")
        base = self.house if name == "house" or name.startswith("house/") else self.root
        rel = name[len("house/"):] if name.startswith("house/") else ("" if name == "house" else name)
        p = (base / rel).resolve() if rel else base.resolve()
        if not (p == base.resolve() or p.is_relative_to(base.resolve())):
            raise PermissionError(f"path outside the workspace: {name!r}")
        return p

    # ── the harness's own writes ─────────────────────────────────────

    def write_goal(self, k: int, goal_text: str, index: int, total: int, modality: str) -> Path:
        p = self.goal_path(k)
        p.write_text(f"# Task {index + 1} of {total}\n\nmodality: {modality}\n\n{goal_text}\n",
                     encoding="utf-8")
        return p

    def write_verdict(self, k: int, verdict: dict) -> Path:
        p = self.verdict_path(k)
        p.write_text(json.dumps(verdict, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return p

    def append_ledger(self, line: str) -> None:
        with self.ledger.open("a", encoding="utf-8") as fh:
            fh.write(line.rstrip("\n") + "\n")

    def read_handover(self, k: int) -> str:
        p = self.handover_path(k)
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""

    def read_lambda(self, k: int, attempt: int) -> str:
        p = self.lambda_path(k, attempt)
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""

    def handovers(self) -> list[tuple[int, str]]:
        out = []
        for d in sorted(self.tasks.iterdir(), key=lambda q: int(q.name) if q.name.isdigit() else -1):
            h = d / "handover.md"
            if d.name.isdigit() and h.exists():
                out.append((int(d.name), h.read_text(encoding="utf-8").strip()))
        return out

    # ── the sessions' writes, through the file tool ──────────────────

    def session_write(self, name: str, text: str, allowed: set[str]) -> dict:
        """A session's ``write_file``: only a path it was named; coordinates scrubbed
        the way the ledger scrubs them; over-long text truncated visibly."""
        if name not in allowed and not any(_matches(name, pat) for pat in allowed):
            return {"error": f"you may not write {name!r}", "writable": sorted(allowed)}
        p = self.resolve(name)
        clean, n = scrub_coordinates(str(text or ""))
        clean = clean.strip()
        truncated = False
        if len(clean) > MAX_WRITE_CHARS:
            clean = clean[: MAX_WRITE_CHARS - 24].rstrip() + " …[truncated by length]"
            truncated = True
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(clean + "\n", encoding="utf-8")
        out = {"written": name, "chars": len(clean)}
        if n:
            out["coordinates_removed"] = n
        if truncated:
            out["truncated_to"] = MAX_WRITE_CHARS
        return out

    def _in_scope(self, p: Path, scope: str) -> bool:
        s = self.resolve(scope)
        p = p.resolve()
        return p == s or p.is_relative_to(s)

    def _above_scope(self, p: Path, scope: str) -> bool:
        s = self.resolve(scope)
        p = p.resolve()
        return p != s and s.is_relative_to(p)

    @staticmethod
    def _withheld(name: str, scope: str) -> dict:
        return {"error": f"{name or '.'} is not available: this run gives a task only its own files, "
                         f"under {scope}/"}

    def session_read(self, name: str, *, max_chars: int = 20000, scope: str = "") -> dict:
        """A session's ``read_file``: a file's text, or — for a .png / .jpg —
        its bytes under ``data`` with the image ``format``, for the tool to
        hand back as a picture."""
        try:
            p = self.resolve(name)
        except PermissionError as e:
            return {"error": str(e)}
        if scope and not self._in_scope(p, scope):
            return self._withheld(name, scope)
        if not p.exists():
            return {"error": f"no such file: {name}"}
        if p.is_dir():
            return {"error": f"{name} is a directory; use list_files"}
        fmt = IMAGE_FORMATS.get(p.suffix.lower())
        if fmt is not None:
            # a stored photograph: the bytes travel as an image, not as text
            data = p.read_bytes()
            return {"path": name, "format": fmt, "bytes": len(data), "data": data}
        text = p.read_text(encoding="utf-8", errors="replace")
        out = {"path": name, "text": text[:max_chars]}
        if len(text) > max_chars:
            out["truncated"] = True
        return out

    def session_list(self, name: str = "", *, scope: str = "") -> dict:
        try:
            p = self.resolve(name) if name else self.root
        except PermissionError as e:
            return {"error": str(e)}
        above = bool(scope) and self._above_scope(p, scope)
        if scope and not (above or self._in_scope(p, scope)):
            return self._withheld(name, scope)
        if not p.exists():
            return {"error": f"no such path: {name or '.'}"}
        if p.is_file():
            return {"path": name, "files": [name]}
        entries = []
        for q in sorted(p.iterdir()):
            if q.name.startswith(".") or q.name in ("trace", "live"):
                continue
            if above and not (self._above_scope(q, scope) or self._in_scope(q, scope)):
                continue                       # only the way down to the scope
            entries.append(q.name + ("/" if q.is_dir() else ""))
        return {"path": name or ".", "entries": entries}
