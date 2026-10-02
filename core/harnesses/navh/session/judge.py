"""JudgeSession — the clean-room judge, opened twice per task.

Two kinds, one conversation shape: goal + a uniform sample of the task's
frames (the forced look-around always kept) + the mechanical record + the map,
one answer as JSON. ``PrecheckSession`` is the gate BEFORE the STOP (it can
send the agent back to work); ``CertifySession`` runs AFTER the STOP and
decides what the ledger says. The mode switch — none | model — and
the refusal the agent reads live in ``orchestrator/judge.py``; this module is
the model path only, plus the ``Verdict`` both paths return.

A judge is a tool-less session of its own: the question is one hop with the
images in it; when the answer cannot be parsed the session is asked once
more, in the same conversation, for the verdict alone. The verdict is read
from the answers only, never from the model's reasoning: a response that
still does not parse is ``unknown`` (UNRESOLVED in the ledger).
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .. import prompts as P
from ..env.types import Goal
from .base import BaseSession
from .wire import image_block, text_block

_JSON_RE = re.compile(r"\{.*\}", re.S)


@dataclass
class Verdict:
    verdict: str                     # complete | incomplete | unknown | none
    evidence: str = ""
    searched: list[str] = field(default_factory=list)
    confidence: str = ""
    judge: str = "none"
    frames_shown: list[int] = field(default_factory=list)
    raw: str = ""
    error: str = ""
    usage: dict = field(default_factory=dict)   # tokens of the judge's hops, for compute matching
    answered_by: str = "prose"       # prose | retry — which answer the JSON was found in

    def disagrees_with(self, claim_status: str) -> bool:
        """True only when the judge has counter-evidence. ``unknown`` (parse failure,
        oracle unavailable, no frames) never refuses a closure: a refusal must
        carry a reason the agent can act on."""
        return self.verdict in ("complete", "incomplete") and self.verdict != claim_status


class JudgeSession(BaseSession):
    kind: ClassVar[str] = "judge"
    prompt: ClassVar[str] = "CLOSING_CHECK"
    reads: ClassVar[tuple[str, ...]] = ("goal", "frames", "record", "map")
    lifetime: ClassVar[str] = "call"

    def __init__(self, model: str, *, max_frames: int = 12, **kw: Any) -> None:
        super().__init__(model, max_turns=1, **kw)
        self.max_frames = max_frames

    def judge(self, goal: Goal, frames: list[dict], record: dict,
              map_b64: str | None = None) -> Verdict:
        """One verdict. A judge that cannot be reached — the session does not
        open, or the hop fails — is ``unknown``, the verdict that never
        refuses a closure: a broken judge costs the check, not the chain."""
        try:
            self.open()
            return self._judge(goal, frames, record, map_b64)
        except Exception as e:
            return Verdict(verdict="unknown", judge="model", error=f"{type(e).__name__}: {str(e)[:300]}",
                           usage=dict(self.last_usage))
        finally:
            self.close()

    def _judge(self, goal: Goal, frames: list[dict], record: dict, map_b64: str | None) -> Verdict:
        sample = stop_first_sample(frames, self.max_frames)
        blocks = [text_block(P.CLOSING_CHECK.format(goal=goal.text, record=json.dumps(record, ensure_ascii=False)))]
        if goal.image_b64:
            blocks.append(text_block("The goal was given as this photograph:"))
            blocks.append(image_block(goal.image_b64))
        if map_b64:
            blocks.append(text_block(
                "Top-down map the robot built, with the route it drove this task and "
                "its markers on it. Use it to see WHERE it went and where it stopped, "
                "not what it saw:"))
            blocks.append(image_block(map_b64))
        for f in sample:
            try:
                b64 = base64.b64encode(Path(f["path"]).read_bytes()).decode()
            except (OSError, KeyError, TypeError):
                continue
            if self.trace is not None:
                self.trace.name_image(b64, str(f["path"]))
            tag = ("look-around at the stopping point, taken by the instruments, not chosen by the robot"
                   if f.get("kind") == "stop" else "chosen by the robot while it worked")
            blocks.append(text_block(f"frame {f['id']} (step {f.get('step', '?')}) — {tag}"))
            blocks.append(image_block(b64))
        if not sample:
            blocks.append(text_block("No frames were recorded during this task."))
        ids = [f["id"] for f in sample]
        try:
            hop = self.hop(blocks)
        except Exception as e:
            return Verdict(verdict="unknown", judge="model", error=f"{type(e).__name__}: {e}",
                           frames_shown=ids, usage=dict(self.last_usage))
        usage = dict(self.last_usage)
        if hop.error and not (hop.text or hop.thinking):
            # the wire reported a failed hop (the provider refused, the call
            # died): there is no answer to parse, and asking again in the same
            # conversation would only repeat the failure
            return Verdict(verdict="unknown", judge="model", error=hop.error, frames_shown=ids, usage=usage)
        v = parse_verdict(hop.text or "", ids)
        if v.verdict == "unknown" and v.error:
            # the answer does not parse: the same conversation is asked once
            # more for the JSON verdict alone, and that answer is the last word
            try:
                hop2 = self.hop(P.VERDICT_ONLY)
            except Exception:
                hop2 = None
            if hop2 is not None:
                usage["completion_tokens"] += int(self.last_usage.get("completion_tokens", 0) or 0)
                usage["retried"] = True
                again = parse_verdict(hop2.text or "", ids)
                if not again.error:
                    again.answered_by, again.usage = "retry", usage
                    return again
        v.usage = usage
        return v


class PrecheckSession(JudgeSession):
    """The gate: opened on the agent's FIRST close request, before the STOP. A
    disagreement goes back to the agent as counter-evidence; the second
    request is honoured unconditionally."""
    kind: ClassVar[str] = "precheck"
    opens: ClassVar[str] = "close_task (the first request of a task)"
    writes: ClassVar[str] = "refusal_text to the agent, or the closure goes through"


class CertifySession(JudgeSession):
    """The certification: opened after the STOP is scored. Decides what the
    ledger says (CERTIFIED / NOT DONE / UNRESOLVED); reused from the gate when
    the agent has not moved since."""
    kind: ClassVar[str] = "certify"
    opens: ClassVar[str] = "task_end"
    writes: ClassVar[str] = "TaskRow.verdict"


def stop_first_sample(frames: list[dict], k: int) -> list[dict]:
    """Keep every forced look-around frame, fill the rest uniformly. The
    look-around is the only evidence the agent did not choose, so it is
    exactly the evidence a uniform sample must not be allowed to drop."""
    stop = [f for f in frames if f.get("kind") == "stop"]
    rest = [f for f in frames if f.get("kind") != "stop"]
    keep = stop[-k:]
    out = uniform_sample(rest, max(0, k - len(keep))) + keep
    return sorted(out, key=lambda f: f.get("id", 0))


def uniform_sample(frames: list[dict], k: int) -> list[dict]:
    if len(frames) <= k:
        return list(frames)
    if k <= 1:
        return [frames[-1]]
    idx = sorted({round(i * (len(frames) - 1) / (k - 1)) for i in range(k)})
    return [frames[i] for i in idx]


def parse_verdict(raw: str, shown: list[int]) -> Verdict:
    m = _JSON_RE.search(raw or "")
    if not m:
        return Verdict(verdict="unknown", judge="model", raw=raw, frames_shown=shown,
                       error="no JSON in judge output")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return Verdict(verdict="unknown", judge="model", raw=raw, frames_shown=shown,
                       error=f"bad JSON: {e}")
    v = str(obj.get("verdict", "unknown")).lower().strip()
    if v not in ("complete", "incomplete", "unknown"):
        v = "unknown"
    searched = obj.get("searched") or []
    if not isinstance(searched, list):
        searched = [str(searched)]
    return Verdict(verdict=v, evidence=str(obj.get("evidence", ""))[:400],
                   searched=[str(s)[:40] for s in searched][:12],
                   confidence=str(obj.get("confidence", "")), judge="model",
                   frames_shown=shown, raw=raw)
