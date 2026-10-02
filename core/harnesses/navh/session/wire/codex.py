"""The Codex wire — one Codex thread, driven by one task, over the same bridge.

`client.py` holds a live `ClaudeSDKClient` from `open` to `close`. Codex has no
such object: a thread is a file on disk and every hop is a fresh `codex exec`
process that resumes it. So this wire keeps the same three verbs and the same
`Hop`, and everything above it — `BaseSession`, the five session kinds, the
orchestrator, the gates, the ladder — is unchanged.

    open()    remembers nothing but the intent to start a thread. Codex is a
              process per hop, so there is nothing to connect.
    hop()     `codex exec` on the first, `codex exec resume <thread>` after;
              reads the `--json` event stream and returns a `Hop`.
    close()   forgets the thread. A recovery opens a new session, which starts
              a new thread — the same thing a fresh SDK session is.

The model provider is the CLI's own login unless the session's
`endpoint["codex"]` names another (the scripted endpoint of `api=fake`, a
gateway); those `-c` overrides ride every exec of the thread.

WHAT THIS WIRE CANNOT MATCH, recorded rather than hidden:

* Codex keeps its own base instructions. Our system prompt rides as
  `developer_instructions`, and only on the hop that starts the thread; the SDK
  path replaces the system prompt outright. So a Codex session and a Claude
  session are not told exactly the same thing, and the trace records what each
  was told.
* There is no turn cap. `max_turns` is a cap the SDK enforces; here the hop
  follows the thread's rollout, counts model requests, and stops the process
  itself, at the session's cap or the lower one a hop is given. It stops ONLY BETWEEN TURNS — when every tool call the model made has
  its result on the thread — because one exec can carry several MCP calls and
  cutting at a result drops the rest of that exec. A session without tools
  (a completion check, the place question) is not capped at all: there is no
  loop, and its one answer ends the exec.
* `open()` cannot verify the bridge. The SDK wire waits for the MCP server and
  fails at `open` if it never came up; Codex starts the server inside the exec,
  so a broken bridge shows up on the first hop instead. It is still an error,
  just later.
* Reasoning is counted but unreadable: Codex reports reasoning tokens and no
  reasoning text, so `Hop.thinking` is always empty on this wire.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .blocks import Blocks
from .client import Hop, SessionLoop

#: Everything Codex offers that is not ours to give. The tool surface is the
#: bridge and nothing else; a shell in particular is the whole disk, and the
#: dataset with the goal positions is on it.
DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "view_image", "sleep_tool",
    "apps", "plugins", "remote_plugin", "multi_agent", "browser_use", "browser_use_external",
    "computer_use", "image_generation", "in_app_browser", "hooks", "skill_search",
    "tool_suggest", "goals", "memories",
)

_DATA_URL = re.compile(r"^data:image/[a-z]+;base64,(.+)$", re.S)


def toml_str(value: str) -> str:
    # A JSON string is a TOML basic string as long as non-ASCII stays raw:
    # ensure_ascii would emit surrogate pairs, which TOML rejects.
    return json.dumps(str(value), ensure_ascii=False)


def codex_home() -> str:
    """The private CODEX_HOME: the login and nothing else of the operator's.

    Codex reads `$CODEX_HOME/config.toml`, and the operator's own config may
    name MCP servers, models and approvals that would silently join the run.
    """
    # session/wire/codex.py -> wire, session, navh, harnesses, core, REPO.
    # The rollouts inline every MCP image and grow large, so they belong under
    # outputs/, never inside the source tree.
    root = Path(__file__).resolve().parents[5]          # the repo root
    home = Path(os.environ.get("NAVH_CODEX_HOME") or (root / "outputs" / ".codex-home"))
    home.mkdir(parents=True, exist_ok=True)
    auth = Path.home() / ".codex" / "auth.json"
    link = home / "auth.json"
    if auth.is_file() and not link.exists():
        with contextlib.suppress(OSError):
            link.symlink_to(auth)
    return str(home)


def codex_env() -> dict:
    return {**os.environ, "CODEX_HOME": codex_home()}


#: What the seat calls "no preference". The SDK takes it as an absent option;
#: Codex takes an enum and rejects the word, so it is normalised here rather
#: than at the seat — the seat should not have to know which CLI it is feeding.
_EFFORT_UNSET = (None, "", "default")


def codex_args(model: str, effort: str | None) -> list[str]:
    effort = "medium" if effort in _EFFORT_UNSET else effort
    return [*(["-c", f"model = {toml_str(model)}"] if model else []),
            "-c", f"model_reasoning_effort = {toml_str(effort)}",
            "-c", "project_doc_max_bytes = 0",
            *[x for f in DISABLED_FEATURES for x in ("-c", f"features.{f} = false")]]


@dataclass
class CodexSpec:
    """What one hop needs. The Codex counterpart of `ClaudeAgentOptions`."""
    model: str
    cwd: str
    system_prompt: str | None = None
    mcp: dict = field(default_factory=dict)      # {"env": {"command":…, "args":[…], "env": {…}}}
    max_turns: int = 25                          # model requests before this wire stops the process
    effort: str | None = None
    tool_timeout_s: int = 600
    provider: list[str] = field(default_factory=list)   # -c overrides naming another model provider

    @classmethod
    def from_session(cls, s: Any) -> "CodexSpec":
        return cls(model=s.model, cwd=str(s.cwd), system_prompt=s.system_prompt,
                   mcp=dict(s.mcp or {}), max_turns=int(s.max_turns), effort=s.effort,
                   provider=list((getattr(s, "endpoint", None) or {}).get("codex") or []))

    @property
    def state_file(self) -> Path | None:
        env = ((self.mcp or {}).get("env") or {}).get("env") or {}
        p = env.get("NAVH_STATE_FILE")
        return Path(p) if p else None


def _read_state(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _pump(stream, out: queue.Queue) -> None:
    for line in iter(stream.readline, ""):
        out.put(line.rstrip("\n"))
    out.put(None)


class _Rollout:
    """Follows the thread's rollout file, which Codex writes as it goes.

    A turn here is one model request (one `token_usage_record`), which is what
    the SDK's `num_turns` counts — not an MCP call, since one exec may carry
    several. `pending` is the number of tool calls whose output is not yet on
    the thread. Only lines written during this hop count; a resumed thread's
    earlier hops are already accounted for.
    """

    def __init__(self, thread: str | None) -> None:
        self.path: Path | None = None
        self.offset = 0
        self.turns = 0
        self.usage: dict = {}
        self.calls: set[str] = set()
        self.outputs: set[str] = set()
        if thread:
            self._find(thread)
            if self.path is not None:
                self.offset = self.path.stat().st_size

    @property
    def pending(self) -> int:
        return len(self.calls - self.outputs)

    def _find(self, thread: str) -> None:
        hits = sorted(Path(codex_home(), "sessions").glob(f"*/*/*/rollout-*-{thread}.jsonl"))
        self.path = hits[-1] if hits else None

    def follow(self, thread: str | None) -> None:
        if self.path is None and thread:
            self._find(thread)
        if self.path is None:
            return
        try:
            with self.path.open("rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError:
            return
        end = chunk.rfind(b"\n") + 1        # a half-written last line waits for the next read
        self.offset += end
        for line in chunk[:end].splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            payload = row.get("payload") or {}
            kind = row.get("type") or payload.get("type")
            if kind == "token_usage_record" or payload.get("type") == "token_usage_record":
                self.turns += 1
                info = (payload.get("info") or payload.get("usage") or {})
                last = info.get("last_token_usage") or info.get("total_token_usage") or info
                if isinstance(last, dict):
                    self.usage = {
                        "input_tokens": int(last.get("input_tokens", 0) or 0),
                        "cached_input_tokens": int(last.get("cached_input_tokens", 0) or 0),
                        "output_tokens": int(last.get("output_tokens", 0) or 0),
                        "reasoning_output_tokens": int(last.get("reasoning_output_tokens", 0) or 0),
                    }
            elif payload.get("type") == "function_call" or kind == "function_call":
                cid = str(payload.get("call_id") or payload.get("id") or "")
                if cid:
                    self.calls.add(cid)
            elif payload.get("type") == "function_call_output" or kind == "function_call_output":
                cid = str(payload.get("call_id") or payload.get("id") or "")
                if cid:
                    self.outputs.add(cid)


class CodexWire:
    """One Codex thread, with the wire interface `BaseSession` drives."""

    #: How long to wait, after the turn cap is reached, for a tool call whose
    #: result has not landed. Past this the process is stopped anyway: a hop
    #: that never closes its last call is a hung exec, not a busy one.
    settle_s = 60.0

    @staticmethod
    def options_for(session):
        return lambda: CodexSpec.from_session(session)

    def __init__(self, loop: SessionLoop, options: Callable[[], CodexSpec], *,
                 on_message: Callable[[Any], None] | None = None) -> None:
        # `loop` is unused: Codex is a subprocess, driven from the caller's own
        # thread. It is accepted so the two wires are interchangeable.
        self.loop = loop
        self._options = options
        self.on_message = on_message
        self.info: dict = {}
        self.session_id: str | None = None          # the Codex thread id
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    # ── lifecycle ────────────────────────────────────────────────────

    def open(self) -> dict:
        spec = self._options()
        self.session_id = None
        self._open = True
        self.info = {"model": spec.model, "transport": "codex-session", "tools": None}
        return self.info

    def close(self) -> None:
        self._open = False
        self.session_id = None

    # ── one hop ──────────────────────────────────────────────────────

    def hop(self, prompt: str | Blocks, max_turns: int | None = None) -> Hop:
        if not self._open:
            raise RuntimeError("session is not open")
        spec = self._options()
        if max_turns is not None:                  # this hop's cap: never above the session's own
            spec = replace(spec, max_turns=min(spec.max_turns, int(max_turns)))
        t0 = time.time()
        hop = Hop()
        thread = self.session_id
        roll = _Rollout(thread)
        state = spec.state_file
        lines: queue.Queue = queue.Queue()
        errors: list[str] = []
        stopped = False

        text, images = _split_prompt(prompt)
        work = Path(spec.cwd)
        work.mkdir(parents=True, exist_ok=True)
        with _image_files(images, work) as paths:
            argv = self._argv(spec, text, paths, thread)
            with (work / "codex_stderr.log").open("ab") as err:
                proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=err, cwd=str(work), text=True,
                                        start_new_session=True, env=codex_env())
                threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True).start()
                capped_since: float | None = None
                try:
                    while True:
                        try:
                            raw = lines.get(timeout=0.5)
                        except queue.Empty:
                            raw = ""
                        if raw is None:                 # codex ended the turn on its own
                            break
                        if raw:
                            thread = self._event(raw, hop, errors) or thread
                        roll.follow(thread)
                        st = _read_state(state)
                        # A session without tools has no loop to cap: its one
                        # answer ends the turn and the process exits by itself.
                        # Stopping it at the first counted request could cut
                        # the answer off before its event is read.
                        if spec.mcp and (roll.turns >= spec.max_turns or st.get("task_over") or st.get("env_dead")):
                            capped_since = capped_since or time.time()
                            if roll.pending == 0 or (roll.path is None
                                                     and time.time() - capped_since > self.settle_s):
                                stopped = True
                                break
                finally:
                    if proc.poll() is None:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            with contextlib.suppress(ProcessLookupError):
                                os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()

        roll.follow(thread)
        if not stopped:
            while True:                                 # events still buffered after the process ended
                try:
                    raw = lines.get_nowait()
                except queue.Empty:
                    break
                if raw:
                    thread = self._event(raw, hop, errors) or thread
        if not stopped and proc.returncode not in (0, None):
            errors.append(f"codex exited {proc.returncode}")

        self.session_id = thread
        hop.session_id = thread
        hop.num_turns = roll.turns
        hop.usage = roll.usage
        hop.latency_s = round(time.time() - t0, 2)
        hop.error = "; ".join(errors)[:400] or None
        hop.subtype = "success" if (stopped or roll.turns) else ("error" if errors else "no_result")
        return hop

    # ── the command, and the events it writes ────────────────────────

    def _argv(self, spec: CodexSpec, prompt: str, images: list[Path], thread: str | None) -> list[str]:
        server = (spec.mcp or {}).get("env") or {}
        head = ["codex", "exec", "--sandbox", "read-only"] + (["resume", thread] if thread else [])
        argv = [*head, "--json", "--skip-git-repo-check", *codex_args(spec.model, spec.effort), *spec.provider]
        # The system prompt starts the thread; a resume inherits it. Codex keeps
        # its own base instructions either way, which is the recorded difference.
        if spec.system_prompt and not thread:
            argv += ["-c", f"developer_instructions = {toml_str(spec.system_prompt)}"]
        if server:
            table = ", ".join(f"{k} = {toml_str(v)}" for k, v in (server.get("env") or {}).items())
            args = ", ".join(toml_str(a) for a in (server.get("args") or []))
            argv += ["-c", f"mcp_servers.env.command = {toml_str(server.get('command') or sys.executable)}",
                     "-c", f"mcp_servers.env.args = [{args}]",
                     "-c", f"mcp_servers.env.env = {{ {table} }}",
                     "-c", 'mcp_servers.env.default_tools_approval_mode = "approve"',
                     "-c", "mcp_servers.env.startup_timeout_sec = 120",
                     "-c", f"mcp_servers.env.tool_timeout_sec = {spec.tool_timeout_s}"]
        for p in images:
            argv += ["-i", str(p)]
        return [*argv, "--", prompt]

    def _say(self, kind: str, **fields: Any) -> None:
        if self.on_message is not None:
            with contextlib.suppress(Exception):
                self.on_message({"kind": kind, **fields})

    def _event(self, raw: str, hop: Hop, errors: list[str]) -> str | None:
        """One `--json` event, folded into the hop the way the SDK wire folds a
        message. Returns a thread id when the event carries one."""
        try:
            ev = json.loads(raw)
        except ValueError:
            return None
        kind = ev.get("type")
        item = ev.get("item") or {}
        typ = item.get("type")
        if kind == "thread.started":
            return ev.get("thread_id")
        if kind in ("error", "turn.failed"):
            e = ev.get("error")
            msg = str(ev.get("message") or (e.get("message") if isinstance(e, dict) else e) or ev)[:500]
            errors.append(msg)
            self._say("warning", kind_detail="codex_error", error=msg[:300])
        elif kind == "item.started" and typ == "mcp_tool_call":
            name = f"mcp__{item.get('server')}__{item.get('tool')}"
            call = {"id": item.get("id"), "name": name, "input": item.get("arguments")}
            hop.tool_uses.append(call)
            hop.story.append({"role": "assistant", "tool_call": {"name": name, "input": call["input"]}})
            self._say("tool_use", **call)
        elif kind == "item.completed" and typ == "mcp_tool_call":
            texts = _result_texts(item)
            hop.tool_results.append({"tool_use_id": item.get("id"), "texts": texts, "images": 0})
            hop.story.append({"role": "tool", "id": item.get("id"), "texts": texts, "images": 0})
            self._say("tool_result", tool_use_id=item.get("id"), texts=[t[:1500] for t in texts], images=0)
        elif kind == "item.completed" and typ == "command_execution":
            # the shell is switched off; if a Codex release brings one back, it is on the record
            self._say("tool_use", id=item.get("id"), name="shell", input={"command": item.get("command")})
            self._say("tool_result", tool_use_id=item.get("id"),
                      texts=[str(item.get("aggregated_output", ""))[:1500]], images=0)
        elif kind == "item.completed" and typ == "agent_message":
            said = str(item.get("text") or "")
            if said.strip():
                hop.text = (hop.text + "\n" + said).strip() if hop.text else said
                hop.story.append({"role": "assistant", "text": said})
                self._say("assistant_text", text=said[:4000])
        return None


def _result_texts(item: dict) -> list[str]:
    out: list[str] = []
    res = item.get("result")
    blocks = res.get("content") if isinstance(res, dict) else res
    for b in (blocks if isinstance(blocks, list) else [blocks]):
        if isinstance(b, dict):
            if b.get("type") == "text" or "text" in b:
                out.append(str(b.get("text") or ""))
            elif b.get("type") == "image":
                out.append("<image>")
        elif b is not None:
            out.append(str(b))
    return [t for t in out if t]


def _split_prompt(prompt: str | Blocks) -> tuple[str, list[str]]:
    """Codex takes text on the command line and images as `-i` attachments, so
    the interleaving is lost. The text says where each one sat — the judge's
    frame captions depend on that order."""
    if isinstance(prompt, str):
        return prompt, []
    parts: list[str] = []
    images: list[str] = []
    for b in prompt:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            parts.append(str(b.get("text") or ""))
        elif b.get("type") == "image":
            src = b.get("source") or {}
            data = src.get("data") or ""
            m = _DATA_URL.match(str(data))
            images.append(m.group(1) if m else str(data))
            parts.append(f"[image {len(images)}]")
    return "\n".join(p for p in parts if p), images


@contextlib.contextmanager
def _image_files(images: list[str], work: Path):
    """The attachments, on disk for the length of one hop."""
    import tempfile

    paths: list[Path] = []
    tmp = Path(tempfile.mkdtemp(prefix="codex_img_", dir=str(work)))
    try:
        for i, b64 in enumerate(images):
            p = tmp / f"{i + 1}.png"
            with contextlib.suppress(Exception):
                p.write_bytes(base64.b64decode(b64))
                paths.append(p)
        yield paths
    finally:
        with contextlib.suppress(OSError):
            for p in tmp.glob("*"):
                p.unlink()
            tmp.rmdir()


__all__ = ["CodexWire", "CodexSpec", "codex_env", "codex_home", "codex_args", "toml_str"]
