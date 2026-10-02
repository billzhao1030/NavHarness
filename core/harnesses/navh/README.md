# NavHarness

A chain whose every conversation with the model is a session Claude Code holds, and whose sessions
connect through files. The seat is `navh` (with `navhcodex` and `navhmini` for the Codex and
mini-swe-agent transports), through `core/harnesses/navharness.py`; outputs land under
`outputs/navharness/`, and the Claude subscription is the auth.

## The idea

**Every session is Claude Code's.** The agent's own session is a live Agent SDK client that stays
connected for a task (or the chain, under `compact: token`); its tools are an MCP server the CLI
spawns beside it — the harness's bridge — and the CLI runs the tool loop. The judge's two checks,
the wake-up question and the day-end consolidation are sessions of their own, opened for one
question (or one job) and closed. The harness never calls a model directly.

**Memory is files, not prompt text.** The agent's system prompt is fixed (`prompts.SYSTEM`); each
task arrives as one instruction naming the goal, the required reading and the file the session must
write; the session reads and writes those files itself through the bridge's `list_files` /
`read_file` / `write_file` (on Claude Code it also reads `house/` with the CLI's own `Read` / `Glob`
/ `Grep`, fenced to that directory by a PreToolUse hook — `session/notes.py`), and the harness never
pastes a ledger, a note or a previous handover into a prompt:

```
house/                 this house, across days (memory_root/<house>)
  MEMORY.md            the index                    } written by the consolidate
  house.md             the house overview           } session at run end (their only
  rooms/<room>.md      room notes                   } writer), from the ledger, the
  skills/<slug>.md     navigation skills            } handovers and the notes it finds
  rooms/<room>/photos/ written by the harness: a marked place's views, at most four (one per
    + photos.json        heading), each with the navigator's pose estimate
  journal/<day>.md     written by the harness: one line per closed task
  archive.json         written by code (the archive policy)
run/
  ledger.md            written by the harness: one line per closed task, after the judge
  tasks/<k>/
    goal.md            written by the harness before the task opens
    handover.md        written by the agent's session before it may close   ← the link to task k+1
    lambda.md          written by the agent's session before a recovery     ← the link to the next attempt
                       (replaced or cut by the harness under recovery_handover other / summary)
    verdict.json       written by the harness from the judge's verdict
```

**The orchestrator acts between hops.** A hop is one query into the agent's session and up to
`chunk_turns` of the CLI's own tool loop, or what is left of the task's `max_turns_per_task` when
that is fewer (the λ hop included), so a task never takes more turns than its budget; while the
fallback recovery can still fire, no more than are left until `reset_after_turns` on the attempt,
so it fires at that turn. When it
returns, the orchestrator reads the bridge's STATE FILE (steps, frames, markers, the body's readings,
a close request, a recovery proposal, the files written) and answers: a close request goes through
the file gate and the judge's gate and only then becomes the STOP, which the orchestrator issues to
the env itself; a stuck attempt goes up the ladder, and λ is the session's last hop, with the env
tools refusing, before the session is closed and a new one reads the note (`recovery_handover`
replaces what it reads: nothing, another task's note, or a prose summary; `recovery: false` turns
the ladder off, its fallback included). A task also ends without a closure when its turns or steps
are spent or when three hops in a row make no tool call (`end_reason` `turns`, `budget` or
`inactive`); its record then carries no completion claim, and for the certification the harness
takes the stop views `close_task` would have taken. The certification is shown the map as it stood before the STOP,
with the route the task drove. A tool never opens a session; a session
never runs the chain.

## Layout

```
navh/
  orchestrator/  orchestrator.py Orchestrator — seat · task · seam · close, and the task's hops; the only caller here
                 judge.py      Judge — the mode switch (none | model); judge_gate; the refusal the agent reads
                 ledger.py     TaskRow · Ledger — the row a closed task leaves
                 close.py      TaskOutcome · ProviderUnavailable · stop / force_stop — the STOP is the orchestrator's
                 checkpoint.py what is frozen at each seam: resume, replay
                 recovery/     driver.py — a recovery carried out; under it ladder.py (the rungs, the
                               controller), lam.py (λ's data and bank), wakeup.py (the wake-up funnel)
  config.py                    HarnessConfig — the switches, grouped by the layer that reads them
  prompts.py                   the fixed system prompt, the instruction, the judge's brief, the ladder's words
  record.py                    the paper trail
  session/       agent.py      AgentSession — the agent's session over the bridge and the workspace
                 judge.py      PrecheckSession (the gate before the STOP) · CertifySession (after)
                 wakeup.py     WakeupSession — which room is this (a rung above `attempt`)
                 consolidate.py ConsolidateSession — the house notes at run end, with the file tools alone
                 base.py       BaseSession · Run — what every kind is: open / hop / close, the accounting
                 wire/         what base.py stands on: client.py (SessionLoop, the sessions' event loop on
                               its own thread · SdkWire, one connected ClaudeSDKClient driven by one task ·
                               Hop), blocks.py (content blocks, usage, the CLI's environment), trace.py
                               (every hop saved whole), home.py (the CLI's config home)
  bridge/        bridge.py     the MCP server: observe · step · get_goal · get_map · mark · preview_path ·
                               close_task · request_recovery · list_files · read_file · write_file
                 state.py      the state file the bridge keeps and the orchestrator reads and edits; the tool names
  memory/        workspace.py  the file tree, its sandbox and the coordinate scrub; mount.py the code tiers
                 archive.py · placegraph.py
  env/           types.py · bench.py (EnvClient, the benches, map_floor) · bench_es.py · record_sink.py
```

Calls run orchestrator → session → (Claude Code) → bridge → env, and orchestrator → env for the seat,
the STOP, the map and the checkpoints; memory and record are read from every layer and read none of
them back. Depth is calling depth: inside orchestrator/ (orchestrator.py → its five neighbours →
recovery/driver.py → its parts) and inside session/ (the kinds → base.py → wire/). Three roles act: the
orchestrator, the sessions, the tools.

## The orchestrator

| method | what it does |
|---|---|
| `seat()` | replay / resume → `done`; `bench.seat`; a restored map (`map_state`) makes task 0 a task after a relocation; the place graph and the archive mounted; the places' kept views handed to the wake-up; the workspace laid out; the seat frozen |
| `task(k, goal)` | `goal.md`; the state file reset; the agent's session opened (or kept under `compact: token`); the instruction (four look-around views along when the body was moved); then the hops: the session's own tool loop, the state file read, a close request through the gates, a stuck attempt up the ladder, or "Continue."; the env asked what comes next |
| `seam(k, goal, outcome)` | certify (the gate reused when the agent has not moved); the handover FILE as the ledger row's summary; `verdict.json` and a `ledger.md` line; the code tiers' bite; the checkpoint; the summary row |
| `close(status)` | the consolidate session, over a files-only bridge, reads the ledger, the handovers, the house notes and the journal and rewrites the house notes, which nothing else writes; the summary with `sessions`, `workspace`, `trace` |

A close request is gated in `_close`: the **file gate** (the first `close_task` is refused, at no step
cost, until `handover.md` exists), then the **judge's gate** (`precheck: model`; the judge session may
send the first request back with counter-evidence, the second is honoured), then the STOP — an env
action the orchestrator issues. The bridge's `close_task` takes the look-around at the stopping point
and records the claim; it neither judges nor stops. Every refused request is kept in the task's row
(`precheck.refusals`: by the file gate or the judge, with the evaluator's truth at that pose, read
through the env's `stop_truth` without ending the task), and the chain aggregate scores the gate from
it (`gate_refused_tasks`, `gate_correct_refusals`, `gate_rescues`, `gate_harms`); none of it reaches
the agent.

## The sessions

Every conversation with the model is a class of `BaseSession`: the class says what the session is
(`kind`, `prompt`, `opens`, `reads`, `writes`, `lifetime`), an instance holds the wire (a live
`ClaudeSDKClient` on the sessions' own event loop) and a `Run` (the CLI's session id, hops, turns,
tokens, timing, how it ended). Every opening is saved under `trace/` (`NNNN_<kind>_open.json`: the
system prompt, the tool names, the MCP servers, max_turns — the CLI transcript never keeps the
system prompt, so this is its only record), every hop whole beside it, and the summary reports how
often each kind was opened.

| kind | class | opens | tools | writes |
|---|---|---|---|---|
| agent | `AgentSession` | task open (kept across tasks under `compact: token`); a recovery reopens | the bridge, all of it (on Claude Code also `Read` / `Glob` / `Grep`, fenced to `house/`) | `handover.md`; `lambda.md` on its last hop before a recovery |
| precheck | `PrecheckSession` | the first `close_task` (after the file gate) | none | refusal text, or the closure goes through |
| certify | `CertifySession` | task end | none | `verdict.json`, the ledger row |
| wakeup | `WakeupSession` | a rung above `attempt` (the funnel over marker photos) | none | the brief in the next instruction |
| consolidate | `ConsolidateSession` | run end (`notes: true`) | the file tools alone | `house/MEMORY.md`, `house.md`, `rooms/`, `skills/` |

Every kind runs on every core. A session without tools (the two checks, the place question) has no
bridge: on the mini wire it is a plain conversation, one completion per hop through mini's model
layer. The two checks are the seat's model on the seat's core unless `judge_model` and
`judge_transport` name another (the cross-model judging comparison); the wake-up and the
consolidation stay the navigator's, and the summary's `sessions` report says which model and core
held each kind.

## The bridge and the state file

The bridge (`bridge/bridge.py`) is started by the CLI as the session's MCP server (`python
<bridge>`); an arm whose env speaks other verbs wraps it and replaces `_call` (this repo's
`exp_workspace/lhnav_es/mcp/navh_bridge.py`, named through `NAVH_BRIDGE_MODULE`). It forwards env
calls, saves frames, banks a marker's views, and rewrites the state file (`bridge/state.py`) after
every call; it re-reads the file at the start of every call, so what the orchestrator wrote between
hops (a new task, an answered request, what may be written, the λ hop's file-only mode) is in force on
the next one. Hops are sequential, so the file has one writer at a time.

## Verification

`tests/test_navh.py` (the package stands alone; the three roles; the class tree; the flat
switches; the tool names; the seat composes),
`tests/test_navh_bridge.py` (the bridge's tools in-process on a stub env: frames and steps in the
state file, `close_task` without a STOP, the file tools' fence, file-only mode, the ladder's proposal,
an orchestrator edit in force on the next call), `tests/test_navh_orchestrator.py` (the
orchestrator over scripted session wires: the two gates and the STOP, `compact: token`, a task the
env ended, a provider refusal, the ladder with λ as the session's last hop, a refused recovery, the
consolidate session), `tests/test_navh_recovery.py` (the recovery driver on a stub env),
`tests/test_navh_judging.py` (every session kind opened, and the tool-less ones answered, on all three
wires with stubbed backends; a navhmini chain judged end to end through the real bridge; a judge on
another core; the truth kept at each refused close request).
`harness=navh model=opus-5 api=fake run.episodes=0` runs the real Claude CLI against the scripted
endpoint through the real bridge: ten tasks, the sessions report, the trace.
