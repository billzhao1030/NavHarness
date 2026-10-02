"""Trace — every hop of a chain, saved whole, filed under its session's kind.

The record (``record.py``) keeps the EVENTS of a task — a tool call, its
result, a note, a verdict — trimmed for the monitor. What it cannot answer
is "what exactly was this session asked on hop 17, and what exactly came
back". This module saves that. One JSON file per hop::

    trace/index.jsonl                   one line per hop or opening: task, seq, purpose, file, tokens
    trace/task_{k}/{seq:04d}_{purpose}_open.json
        options   what the session was opened with: the system prompt, the tool
                  names, the MCP servers (command and args only), max_turns,
                  effort, model, cwd. The CLI transcript never keeps the system
                  prompt, so this file is the only record of it.
    trace/task_{k}/{seq:04d}_{purpose}.json
        request   the prompt as sent — images replaced by references
                  (a frame the task already saved, or trace/images/<sha>.png)
        tools     the tool names the session was opened with
        response  the hop: text, thinking, the tool calls and results in order,
                  turns, subtype, usage, session id, latency
        state     what the caller knew at that moment (optional)
    trace/images/<sha>.png              images that are not a saved frame
                                        (goal photo, map renders, look-around,
                                        marker photo sets), deduplicated

The purpose is the session's ``kind`` (``agent``, ``precheck``, ``certify``,
``wakeup``, ``consolidate``) plus which opening of that session and which
hop — set by ``BaseSession.open`` (the opening, hop 0) and
``BaseSession._account`` (every hop), the only callers.

Design rules: tracing must never change a hop (the session forwards the
prompt untouched and re-raises), never lose one (a write failure is reported
through the recorder as a warning), and never duplicate a frame that is
already on disk (``name_image`` says "this base64 is live/obs_0012.png").
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .blocks import Blocks, count_images
from .client import Hop


class Trace:
    def __init__(self, root: Path, *, rec: Any = None) -> None:
        self.root = Path(root)
        self.rec = rec  # a Recorder, for warnings; optional
        self.images = self.root / "images"
        self.task = -1
        self.seq = 0
        self.calls = 0
        self.bytes = 0
        self.saved_images = 0
        self._names: dict[str, str] = {}  # sha16 -> reference path (relative to out_dir)
        self.state_provider = None  # callable() -> dict, set by the orchestrator
        self.last_file: str | None = None
        self.t0 = time.time()

    # ── where a hop belongs ──────────────────────────────────────────

    def open_task(self, k: int) -> None:
        self.task = int(k)
        self.seq = 0

    # ── images ───────────────────────────────────────────────────────

    @staticmethod
    def _sha(b64: str) -> str:
        return hashlib.sha1(b64.encode("ascii", "ignore")).hexdigest()[:16]

    def name_image(self, b64: str, ref: str) -> None:
        """Tell the trace that this base64 is a file the task already saved."""
        if b64:
            self._names[self._sha(b64)] = ref

    def image_ref(self, b64: str) -> dict:
        sha = self._sha(b64)
        ref = self._names.get(sha)
        if ref is None:
            p = self.images / f"{sha}.png"
            if not p.exists():
                try:
                    self.images.mkdir(parents=True, exist_ok=True)
                    raw = base64.b64decode(b64)
                    p.write_bytes(raw)
                    self.saved_images += 1
                    self.bytes += len(raw)
                except (OSError, ValueError) as e:
                    self._warn("trace_image", e)
            ref = f"trace/images/{sha}.png"
            self._names[sha] = ref
        return {"type": "image_ref", "sha": sha, "ref": ref, "b64_chars": len(b64)}

    def _strip(self, prompt: str | Blocks) -> str | list:
        if isinstance(prompt, str):
            return prompt
        out = []
        for b in prompt:
            if isinstance(b, dict) and b.get("type") == "image":
                out.append(self.image_ref(str((b.get("source") or {}).get("data", ""))))
            else:
                out.append(b)
        return out

    # ── one opening ──────────────────────────────────────────────────

    def record_open(self, *, purpose: dict, model: str, system_prompt: str | None, tools: list[str],
                    mcp: dict | None, max_turns: int, effort: str | None, cwd: str) -> str | None:
        """The options a session was just opened with, saved once per opening."""
        self.seq += 1
        name = f"{self.seq:04d}_{purpose.get('purpose', 'session')}_open.json"
        rel = f"trace/task_{self.task}/{name}" if self.task >= 0 else f"trace/run/{name}"
        servers = {str(k): {"command": v.get("command"), "args": list(v.get("args") or [])}
                   for k, v in (mcp or {}).items() if isinstance(v, dict)}
        doc = {
            "t": round(time.time() - self.t0, 1),
            "task": self.task,
            "seq": self.seq,
            **purpose,
            "open": True,
            "model": model,
            "options": {"system_prompt": system_prompt, "tools": list(tools), "mcp": servers,
                        "max_turns": max_turns, "effort": effort, "cwd": str(cwd)},
        }
        try:
            p = self.root.parent / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(doc, ensure_ascii=False, default=str, indent=1)
            p.write_text(text, encoding="utf-8")
            self.bytes += len(text)
            with (self.root / "index.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "t": doc["t"], "task": self.task, "seq": self.seq, **purpose, "file": rel,
                    "open": True, "turns": None, "prompt_tokens": None, "completion_tokens": None,
                    "latency_s": None, "tools_called": [], "error": None,
                }, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            self._warn("trace_write", e)
            return None
        return rel

    # ── one hop ──────────────────────────────────────────────────────

    def record(self, *, purpose: dict, model: str, prompt: str | Blocks, tools: list[str],
               hop: Hop | None, error: str | None, latency_s: float,
               session_id: str | None = None) -> str | None:
        self.seq += 1
        self.calls += 1
        name = f"{self.seq:04d}_{purpose.get('purpose', 'hop')}.json"
        rel = f"trace/task_{self.task}/{name}" if self.task >= 0 else f"trace/run/{name}"
        state = {}
        if self.state_provider is not None:
            try:
                state = dict(self.state_provider() or {})
            except Exception as e:  # a state snapshot must never cost the hop
                state = {"error": f"{type(e).__name__}: {e}"}
        doc = {
            "t": round(time.time() - self.t0, 1),
            "task": self.task,
            "seq": self.seq,
            **purpose,
            "model": model,
            "session_id": session_id,
            "request": {"prompt": self._strip(prompt), "images": count_images(prompt)},
            "tools": list(tools),
            "response": _hop_doc(hop) if hop is not None else None,
            "error": error,
            "latency_s": round(latency_s, 2),
            "state": state,
        }
        try:
            p = self.root.parent / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(doc, ensure_ascii=False, default=str, indent=1)
            p.write_text(text, encoding="utf-8")
            self.bytes += len(text)
            with (self.root / "index.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "t": doc["t"], "task": self.task, "seq": self.seq, **purpose, "file": rel,
                    "turns": hop.num_turns if hop else None,
                    "prompt_tokens": (doc["response"] or {}).get("prompt_tokens") if hop else None,
                    "completion_tokens": (doc["response"] or {}).get("completion_tokens") if hop else None,
                    "latency_s": doc["latency_s"],
                    "tools_called": hop.tools_called if hop else [],
                    "error": error,
                }, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            self._warn("trace_write", e)
            self.last_file = None
            return None
        self.last_file = rel
        return rel

    def stats(self) -> dict:
        return {"dir": "trace", "calls": self.calls, "images_saved": self.saved_images, "bytes": self.bytes}

    def _warn(self, what: str, e: Exception) -> None:
        if self.rec is not None:
            with contextlib.suppress(Exception):
                self.rec.emit("warning", kind_detail=what, error=str(e)[:200])


def _hop_doc(hop: Hop) -> dict:
    from .blocks import ctx_tokens, out_tokens
    return {
        "text": hop.text,
        "thinking": hop.thinking,
        "story": hop.story,
        "tool_uses": hop.tool_uses,
        "num_turns": hop.num_turns,
        "subtype": hop.subtype,
        "usage": hop.usage,
        "prompt_tokens": ctx_tokens(hop.usage),
        "completion_tokens": out_tokens(hop.usage),
        "session_id": hop.session_id,
        "latency_s": hop.latency_s,
        "error": hop.error,
    }
