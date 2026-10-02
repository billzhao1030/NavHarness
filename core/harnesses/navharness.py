"""NavHarness adapter — the package core/harnesses/navh/ as a seat.

One seat word, ``navh``: Claude Code holds every session of the chain
(``transport: sdk-session``) — the agent's own, with this arm's MCP bridge
as its tools; the judge's two checks; the wake-up question; the day-end
consolidation — and the sessions connect through a workspace of files.
The Claude subscription is the auth; ``navhcodex`` / ``navhmini`` put the
Codex CLI or mini-swe-agent in the same place. ``api=fake`` points every
session, on whichever core, at the scripted endpoint (one loopback port that
speaks all three wires).

The two completion checks are the seat's model on the seat's core unless
``judge_model`` / ``judge_transport`` name another — the cross-model judging
comparison, e.g. an Opus 5 judge (``judge_model=claude-opus-5
judge_transport=sdk-session``) for a Qwen3.8-27B navigator on mini-swe-agent,
or a Qwen3.8-27B judge (``judge_model=ollama_chat/qwen3.8-27b-q4-std
judge_transport=mini-session``) for Opus 5 on Claude Code. A judge its core
cannot run is refused when the seat is built.

What this class does, and only this: settles the seat (``prepare``: the
gateway or the subscription, the CLI's config home), runs ONE chain per
episode (``run``: ``Orchestrator`` over ``bench_es.GOATEsBench`` on the env
the runner placed, the bridge the arm's surface names), forwards the
harness's events through the bus (``record_sink.SinkRecorder``) and turns
the chain summary into a ``SessionOutcome``. A GOAT episode is a chain of N
sub-goal tasks; a single-goal line is a chain of one. The harness's own
record lands in ``<run_dir>/navh_{i}/`` (its episode_{k}.jsonl per task,
live_{k}/, the workspace, the trace, summary.json with the per-task rows and
the chain aggregate).

Constructor kwargs are the harness's switches (configs/harness/navh.yaml
beside the experiment; ``harness.<key>=…`` overrides one) — the same names
as ``HarnessConfig``. ``session_mode`` picks the session policy (NavHarness,
or one of the two context-only baselines) and ``ledger`` whether earlier
tasks' records are in view; ``HarnessConfig`` holds what a context-only
policy rules out, and the record says what it held.
"""

from __future__ import annotations

import asyncio
import functools
import os
import sys
from pathlib import Path
from typing import Any

from core.api import ensure_serving, fake_gateway_for, gateway_for, provider_of
from core.episode import EpisodeContext, EventSink, SessionOutcome
from core.harnesses.navh import HARNESS_NAME, HarnessConfig, Orchestrator
from core.harnesses.navh.config import LEDGER_MODES, RETRIEVAL_MODES, SESSION_MODES
from core.harnesses.navh.env.bench_es import EsEnvClient, GOATEsBench, TourEsBench
from core.harnesses.navh.env.record_sink import SinkRecorder, exit_content
from core.harnesses.navh.env.types import EnvUnavailable
from core.harnesses.navh.orchestrator import agent_tool_names
from core.harnesses.navh.orchestrator.close import ProviderUnavailable
from core.harnesses.navh.orchestrator.recovery.lam import HANDOVERS, load_bank, mean_words
from core.panel import json_safe
from core.paths import OUTPUTS
from core.pricing import add_anthropic_usage

NAVH_VERSION = "navharness"
TRANSPORTS = ("sdk-session", "codex-session", "mini-session")
CLAUDE_SESSIONS = OUTPUTS / ".claude-sessions"  # the SDK's transcripts, out of ~/.claude
#: the wire each CLI speaks to its model endpoint; mini-swe-agent speaks litellm itself
CLI_WIRE = {"sdk-session": "anthropic-messages", "codex-session": "openai-chat"}
AUTH = {"sdk-session": "claude subscription (Agent SDK sessions)",
        "codex-session": "the Codex CLI's own login",
        "mini-session": "the provider's key, through litellm"}


def judge_route(model: str, transport: str) -> None:
    """Refuse a judge its core cannot run. A misrouted judge fails on every
    call, and a failed call is an ``unknown`` verdict, which never refuses a
    closure — the run would finish looking like a run without a gate. A
    litellm id (``provider/model``) runs on mini-session only; Claude Code
    runs Claude models, the Codex CLI the models of its own login."""
    vendor = getattr(provider_of(model), "name", None)
    if transport == "sdk-session" and ("/" in model or vendor == "openai"):
        raise ValueError(f"judge_model={model!r} is not a model Claude Code runs; a litellm id judges on "
                         "judge_transport=mini-session")
    if transport == "codex-session" and ("/" in model or vendor == "anthropic"):
        raise ValueError(f"judge_model={model!r} is not a model the Codex CLI runs; a Claude model judges on "
                         "judge_transport=sdk-session, a litellm id on judge_transport=mini-session")


class EsOrchestrator(Orchestrator):
    """``Orchestrator`` whose record is a ``SinkRecorder`` — the one place the
    harness's ``Recorder`` is constructed is its ``__init__``, and every
    writer keeps a reference, so all of them are pointed at ours right after."""

    def __init__(self, cfg: HarnessConfig, model: str, judge: str | None, bench: GOATEsBench, out_dir: Path,
                 *, bridge: Path, sink: Any, live_dir: Path | None, ledger: dict[str, dict[str, int]],
                 effort: str | None = None, endpoint: dict | None = None,
                 judge_endpoint: dict | None = None) -> None:
        super().__init__(cfg, model, judge, bench, out_dir, bridge=bridge,
                         meter=functools.partial(add_anthropic_usage, ledger), effort=effort,
                         endpoint=endpoint, judge_endpoint=judge_endpoint)
        self.rec = SinkRecorder(out_dir, out_dir / "live", sink=sink, live_dir=live_dir)   # forwarded to every writer

    def start_map_capture(self, bench: Any, live_dir: Path | None) -> dict:
        """Ask the env to keep every map it renders, beside the camera frames.
        The bridge keeps the renders the MODEL was shown; this keeps the rest,
        so one run's maps are kept the way another's are. Refused politely by
        an env that does not have the verb."""
        if live_dir is None:
            return {"map_capture": None}
        try:
            out = bench.env.call(bench.fn("set_map_capture"),
                                 {"directory": str(live_dir), "every": 1}) or {}
        except Exception as exc:
            return {"map_capture": None, "error": str(exc)[:200]}
        self.rec.emit("map_capture", **(out if isinstance(out, dict) else {}))
        return out


def _check_handover(handover: str, bank: str) -> None:
    """Refuse a handover variant at the seat rather than in every episode: an
    unknown name, a bank that is not there, or a ``summary`` with no
    structured notes to match in length."""
    if handover not in HANDOVERS:
        raise ValueError(f"recovery_handover must be one of {HANDOVERS}, got {handover!r}")
    if handover not in ("other", "summary"):
        return
    notes = load_bank(bank)                      # raises when the path does not exist
    if handover == "summary" and not mean_words(notes):
        raise ValueError("recovery_handover=summary cuts the summary to the mean length of structured notes: "
                         "set recovery_handover_bank to an earlier structured run (its output directory, a "
                         "lessons.jsonl or a JSON list of notes)")
    print(f"[navh] recovery_handover={handover}: {len(notes)} structured note(s) in the bank"
          + (f", {mean_words(notes)} words on average" if notes else "; only this run's own notes are lent"))


class NavHarnessAdapter:
    name = HARNESS_NAME

    def __init__(
        self,
        *,
        transport: str = "sdk-session",
        # the switches (HarnessConfig, same names)
        persist_map: bool = True,
        precheck: str = "model",
        certify: str = "model",
        compact: str = "boundary",
        chunk_turns: int = 25,
        stop_views: int = 4,
        max_turns_per_task: int | None = 200,  # model turns per task; None / 0: the episode's max_turns
                                               # split over its sub-goals
        step_budget_per_task: int = 500,  # executed actions per task; 0: the episode's step budget, whole,
                                          # for every task
        reset_after_turns: int = 80,  # T_f: turns on one attempt before the harness recovers unasked
        max_resets_per_task: int = 1,
        recovery: bool = True,
        recovery_ceiling: str = "house",
        recovery_min_turns: int = 12,
        place_graph: bool = True,
        archive_policy: bool = False,
        notes: bool = True,  # the house notes (MEMORY.md, house.md, rooms/, skills/), consolidated at run end
        # checkpoints / resume / replay — a chain is many tasks long, so unlike
        # an ES batch episode it can be cut short and restarted, and a single
        # task can be re-run from the boundary it actually began at
        checkpoints: bool = False,  # freeze map + markers + body + cursor at every boundary
        resume_from: str = "",  # a previous run dir for this chain: replay its ledger, skip its tasks
        replay_from: str = "",  # a run dir holding checkpoints/
        replay_task: int = -1,  # which task to seat at and re-run (-1 = off)
        map_state: str = "",  # a file the map + markers are kept in across runs; "{house}" = the scene
        # long-term memory across runs (a deployment): where the houses live, and
        # which day this run is. "" = this chain's own directory, so a benchmark
        # episode starts with an empty memory and never reads another episode's notes.
        memory_root: str = "",
        day: str = "",
        # a deployment day: several episodes of the SAME house, in order, as one
        # chain (the first is the one the runner placed); [] = the placed episode
        chain_ids: list | None = None,
        # the model side
        judge_model: str | None = None,  # None: the seat's model judges too
        judge_transport: str | None = None,  # None: the judge sits on the seat's own core
        # the session policy: harness (NavHarness) | independent (a fresh context-only
        # session per task) | single (one context-only session across tasks, compacted
        # by the backend); and whether earlier tasks' records are in view (text | none)
        session_mode: str = "harness",
        ledger: str = "text",
        # the ladder, continued (LadderConfig): escalation and what a recovery hands over
        recovery_escalation: bool = True,  # a repeat recovery at or below the last scope goes one rung up
        recovery_handover: str = "structured",  # structured | empty | other | summary
        recovery_handover_bank: str = "",  # an earlier structured run: `other` lends from it, `summary` matches it
        # fixed-access historical memory (memory/retrieval.py): "none" is the reference;
        # "bm25" retrieves on a schedule, with the file tools scoped as under ledger=none
        retrieval: str = "none",
        retrieval_top_k: int = 5,
        retrieval_budget_tokens: int = 2000,
        retrieval_chunk_tokens: int = 400,
        retrieval_overlap_tokens: int = 50,
        retrieval_every_turns: int = 10,
    ) -> None:
        if transport not in TRANSPORTS:
            raise ValueError(f"navh transport must be one of {TRANSPORTS}, got {transport!r}")
        if session_mode not in SESSION_MODES:
            raise ValueError(f"navh session_mode must be one of {SESSION_MODES}, got {session_mode!r}")
        if ledger not in LEDGER_MODES:
            raise ValueError(f"navh ledger must be one of {LEDGER_MODES}, got {ledger!r}")
        if retrieval not in RETRIEVAL_MODES:
            raise ValueError(f"navh retrieval must be one of {RETRIEVAL_MODES}, got {retrieval!r}")
        _check_handover(recovery_handover, recovery_handover_bank)
        if judge_transport and judge_transport not in TRANSPORTS:
            raise ValueError(f"navh judge_transport must be one of {TRANSPORTS}, got {judge_transport!r}")
        if judge_model:
            judge_route(str(judge_model), judge_transport or transport)
        self.transport = transport
        self.switches: dict[str, Any] = {
            "persist_map": persist_map,
            "precheck": precheck,
            "certify": certify,
            "compact": compact,
            "chunk_turns": chunk_turns,
            "stop_views": stop_views,
            "reset_after_turns": reset_after_turns,
            "max_resets_per_task": max_resets_per_task,
            "recovery": recovery,
            "recovery_ceiling": recovery_ceiling,
            "recovery_min_turns": recovery_min_turns,
            "place_graph": place_graph,
            "archive_policy": archive_policy,
            "notes": notes,
            "checkpoints": checkpoints,
            "resume_from": resume_from,
            "replay_from": replay_from,
            "replay_task": replay_task,
            "map_state": map_state,
            "day": day,
            "session_mode": session_mode,
            "ledger": ledger,
            "recovery_escalation": recovery_escalation,
            "recovery_handover": recovery_handover,
            # absolute, so the run's recorded config names the notes unambiguously
            "recovery_handover_bank": str(Path(recovery_handover_bank).resolve()) if recovery_handover_bank else "",
            "retrieval": retrieval,
            "retrieval_top_k": retrieval_top_k,
            "retrieval_budget_tokens": retrieval_budget_tokens,
            "retrieval_chunk_tokens": retrieval_chunk_tokens,
            "retrieval_overlap_tokens": retrieval_overlap_tokens,
            "retrieval_every_turns": retrieval_every_turns,
            "judge_transport": judge_transport or "",
        }
        self.memory_root = str(memory_root or "")
        self.chain_ids = [int(c) for c in (chain_ids or [])]
        self.max_turns_per_task = max_turns_per_task
        # A chain's tasks share the episode's budget, and the harness's own
        # field is per TASK — so handing it the episode's number would give
        # every sub-goal the whole budget and overstate to the agent the steps
        # it really has. 0 falls back to the episode's budget.
        self.step_budget_per_task = int(step_budget_per_task or 0)
        self.settings: dict[str, Any] = {"judge_model": judge_model}
        self.inherent: dict[str, Any] = {
            "harness": NAVH_VERSION,
            "transport": transport,
            "auth": AUTH[transport],
            "thinking": "SDK default (effort passed through)",
            "turn_cap": "hard (max_turns per task; a chain of N tasks per episode)",
            # Claude Code holds every session's context; the harness holds files
            "vision_context": "the CLI's own (a session holds its transcript)",
            "context_control": False,
            "context_label": (f"navh compact={compact} files"
                              + ("" if ledger == "text" and retrieval == "none" else " ledger=none")
                              + ("" if retrieval == "none" else f" retrieval={retrieval}")
                              if session_mode == "harness" else f"navh session_mode={session_mode} context-only"),
            "switches": dict(self.switches),
        }
        # A context-only session policy holds the switches it rules out, and the
        # fixed-access control holds ledger=none; the record carries the effective
        # config, and the launch says so up front (on stderr: a dry run's stdout
        # is the resolved yaml).
        held = HarnessConfig.from_flat(benchmark="es", chain_id=0, **self.switches).held
        if held:
            self.inherent["session_policy_held"] = dict(held)
            who = f"session_mode={session_mode}" if session_mode != "harness" else f"retrieval={retrieval}"
            print(f"[navh] {who} holds " + " ".join(f"{k}={v}" for k, v in held.items()), file=sys.stderr)
        self._gateway = None
        self._model_id = ""
        self._endpoint: dict = {}
        self._judge_endpoint: dict = {}

    # ── once per run ──

    def prepare(self, spec) -> None:
        from core.harnesses.navh.session import ensure_relocated

        self._model_id = spec.model_id
        extra = spec.extra_dict
        if extra.get("params"):
            raise RuntimeError("navh takes no sampling params — the Agent SDK exposes none")
        judge_id = self.settings["judge_model"] or spec.model_id
        judge_transport = self.switches["judge_transport"] or self.transport
        if judge_transport != self.transport:
            judge_route(judge_id, judge_transport)     # the seat's own model, moved to another core
        fake = extra.get("api_gateway") == "fake"
        # The seat's gateway: the scripted endpoint of api=fake (it speaks every
        # core's wire), or a cross-API seat's, on its CLI's own wire. mini
        # speaks litellm itself and takes no gateway but the scripted one.
        wire = CLI_WIRE.get(self.transport)
        self._gateway = gateway_for(spec, wire=wire, tag="navh") if wire else fake_gateway_for(spec, tag="navh")
        self.inherent["auth"] = AUTH[self.transport]
        # Claude Code sessions reach a gateway through the environment: the
        # scripted endpoint takes every one, a cross-API gateway the seat's own
        sdk_routed = self._gateway is not None and (fake or self.transport == "sdk-session")
        if self._gateway is not None:
            if not fake and judge_transport == self.transport and judge_id != spec.model_id:
                raise RuntimeError(
                    f"the seat's gateway routes {spec.model_id} only; judge_model={judge_id} on the same core "
                    "would reach a gateway that does not serve it — judge on another judge_transport")
            if sdk_routed:
                os.environ.update(self._gateway.anthropic_env())
            desc = self._gateway.describe()
            self.inherent["api_gateway"] = desc
            self.inherent["auth"] = desc["auth"]
            print(f"[navh] api gateway :{self._gateway.port} — {desc['backend']}")
        # Read before the key below leaves the environment: a mini session that
        # runs a Claude model holds its key itself.
        self._endpoint = self._endpoint_for(spec.model_id, extra, self.transport, fake=fake)
        self._judge_endpoint = self._endpoint_for(judge_id, extra if judge_id == spec.model_id else {},
                                                  judge_transport, fake=fake)
        # A local model served by Ollama is run only at the pinned context and with
        # its sampling fixed server-side: Ollama's default window truncates silently,
        # and unpinned sampling makes no two runs comparable (core/api/ollama.py).
        if not fake:
            for model_id, transport, ex in ((spec.model_id, self.transport, extra),
                                            (judge_id, judge_transport, extra if judge_id == spec.model_id else {})):
                if transport == "mini-session":
                    served = ensure_serving(model_id, ex)
                    if served:
                        self.inherent.setdefault("serving", {})[model_id] = served
        if (not sdk_routed and "sdk-session" in (self.transport, judge_transport)
                and os.environ.pop("ANTHROPIC_API_KEY", None)):
            print("[navh] ANTHROPIC_API_KEY was set — removed so sessions use subscription auth")
        sess = ensure_relocated(CLAUDE_SESSIONS)
        self.inherent["claude_sessions"] = sess
        print(f"[navh] cli sessions -> {sess.get('dir') or '~/.claude'} ({sess.get('reason', 'relocated')})")
        try:
            import claude_agent_sdk

            self.inherent["sdk_version"] = getattr(claude_agent_sdk, "__version__", "?")
        except Exception:
            pass
        self.inherent["model"] = spec.model_id
        self.inherent["judge_model"] = judge_id
        self.inherent["judge_transport"] = judge_transport

    def _endpoint_for(self, model_id: str, extra: dict, transport: str, *, fake: bool) -> dict:
        """What a session on ``transport`` needs, beyond the id, to reach
        ``model_id`` (``BaseSession.endpoint``). Claude Code reads its endpoint
        from the environment ``prepare`` set; the Codex CLI takes the
        gateway's provider overrides, else its own login; mini takes the
        scripted endpoint under ``api=fake``, else the row's ``api_base`` (and
        the provider's key with it) or a local provider's default endpoint,
        and a Claude model's key, which Claude Code's subscription takes out
        of the environment whenever a Claude Code session is in the run."""
        gw = self._gateway
        if transport == "codex-session":
            routed = gw is not None and hasattr(gw, "codex_args") and (fake or transport == self.transport)
            return {"codex": gw.codex_args()} if routed else {}
        if transport != "mini-session":
            return {}
        if fake and gw is not None and hasattr(gw, "litellm_kwargs"):
            return {"mini": gw.litellm_kwargs()}
        provider = provider_of(model_id, extra)
        kwargs: dict[str, Any] = {}
        api_base = extra.get("api_base") or (provider.default_api_base if provider is not None else None)
        if api_base:
            kwargs["api_base"] = api_base
        if provider is not None and (extra.get("api_base") or provider.name == "anthropic"):
            key = provider.prepare(model_id, extra).api_key
            if key:
                kwargs["api_key"] = key
        return {"mini": kwargs} if kwargs else {}

    def finalize(self, run_dir: Path) -> dict[str, Any]:
        if self._gateway is not None:
            self._gateway.stop()
            self._gateway = None
            for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
                os.environ.pop(key, None)
        return {}

    # ── per episode ──

    def _config(self, ctx: EpisodeContext, n_goals: int, house: str = "") -> HarnessConfig:
        per_task = self.max_turns_per_task or max(1, int(ctx.max_turns) // max(1, n_goals))
        out_dir = (ctx.run_dir or ctx.workdir.parent) / f"navh_{ctx.index}"
        switches = dict(self.switches)
        if switches.get("map_state") and "{house}" in switches["map_state"]:
            switches["map_state"] = switches["map_state"].replace("{house}", house or f"episode-{ctx.index}")
        if self.chain_ids and self.chain_ids[0] != int(ctx.index):
            raise ValueError(f"chain_ids={self.chain_ids} must start with the placed episode {ctx.index}")
        return HarnessConfig.from_flat(
            benchmark="es",
            chain_id=ctx.index,
            chain_ids=list(self.chain_ids),
            split=str(ctx.benchmark),
            step_budget=self.step_budget_per_task or int(ctx.step_budget),
            max_turns_per_task=int(per_task),
            instruments=int(ctx.instruments or 0),
            rgb_resolution="512",
            transport=self.transport,
            memory_root=self.memory_root or str(out_dir / "memory"),
            **switches,
        )

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "effort": ctx.effort,
            "judge_model": self.inherent.get("judge_model"),
            "judge_transport": self.inherent.get("judge_transport"),
            "navh_config": json_safe(self._config(ctx, 1).flat()),  # per-task turns settled at seat
            "navh_tools": agent_tool_names(self._config(ctx, 1)),
            "navh_dir": f"navh_{ctx.index}",
        }

    def _bench_cls(self, ctx: EpisodeContext):
        """What one chain is on this experiment — an episode, or a tour.

        The default is the episode the runner placed. An arm whose experiment
        seats a whole IVLN-CE tour says so (``surface.navh_bench: tour``) and
        gets a bench whose boundary is the tour's oracle transit, so the
        ledger, the place graph and the house memory span the tour."""
        arm = getattr(ctx, "arm", None)
        declared = str((getattr(arm, "surface", None) or {}).get("navh_bench") or "episode")
        if declared == "tour":
            return TourEsBench
        if declared != "episode":
            raise RuntimeError(
                f"{getattr(arm, 'name', 'this arm')} declares navh_bench={declared!r}; "
                "it is 'episode' (the default) or 'tour'"
            )
        return GOATEsBench

    def _bridge(self, ctx: EpisodeContext) -> Path:
        """The tool surface the sessions hand their CLI — the ARM's.

        A harness is not a tool set: this repo's arms each own theirs
        (``exp_workspace/<arm>/mcp/``). The harness ships a bridge of its own
        (``navh/bridge/bridge.py``), written against ``auto_host``-style env
        verbs, so an arm that wants this seat names the file that speaks ITS
        env in the surface."""
        arm = getattr(ctx, "arm", None)
        declared = (getattr(arm, "surface", None) or {}).get("navh_bridge") if arm else None
        if not declared:
            raise RuntimeError(
                f"harness navh needs an MCP tool surface for this env, and "
                f"{getattr(arm, 'name', 'this arm')} declares none — add "
                "`surface: {navh_bridge: mcp/navh_bridge.py}` to the experiment's arm"
            )
        path = (Path(arm.dir) / declared).resolve()
        if not path.is_file():
            raise RuntimeError(f"{arm.name}'s surface names navh_bridge={declared}, which is not a file: {path}")
        return path

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        return await asyncio.to_thread(self._run_episode, ctx, sink)

    def _run_episode(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        assert self._model_id, "prepare() first"
        ledger: dict[str, dict[str, int]] = {}   # per call, per model id (seat + judge), core.pricing
        env = EsEnvClient(ctx.server_url)
        spec = env.call("goal_spec", {})
        n_goals = int(spec.get("n_goals") or 1) if isinstance(spec, dict) else 1
        scene = str(spec.get("scene_id") or "") if isinstance(spec, dict) else ""
        cfg = self._config(ctx, n_goals, house=Path(scene).name.split(".")[0] if scene else "")
        bench_cls = self._bench_cls(ctx)
        bench = bench_cls(
            env,
            split=str(ctx.benchmark),
            instruments=int(ctx.instruments or 0),
            step_budget=self.step_budget_per_task or int(ctx.step_budget),
            rgb_resolution="512",
            episode_id=str(spec.get("episode_id") or "") if isinstance(spec, dict) else "",
            scene_id=str(spec.get("scene_id") or "") if isinstance(spec, dict) else "",
        )
        out_dir = (ctx.run_dir or ctx.workdir.parent) / f"navh_{ctx.index}"
        runner = EsOrchestrator(
            cfg, self._model_id, self.settings["judge_model"], bench, out_dir,
            bridge=self._bridge(ctx), sink=sink, live_dir=ctx.live_dir, ledger=ledger, effort=ctx.effort,
            endpoint=self._endpoint, judge_endpoint=self._judge_endpoint,
        )
        runner.start_map_capture(bench, ctx.live_dir)
        sink.emit(
            "system_init",
            {
                "model": self._model_id,
                "transport": self.transport,
                "tools": runner.tool_names(),
                "n_goals": n_goals,
                "max_turns_per_task": cfg.task.max_turns_per_task,
            },
        )
        sink.emit(
            "bridge_status",
            {"status": "connected", "server": ctx.server_url, "navh_dir": str(out_dir),
             "bridge": str(runner.bridge)},
        )
        error: str | None = None
        summary: dict[str, Any] = {}
        try:
            summary = runner.run()
        except EnvUnavailable as exc:
            error = f"EnvUnavailable: {exc}"
        except ProviderUnavailable as exc:
            error = f"ProviderUnavailable: {exc}"
        except Exception as exc:
            error = repr(exc)
        if error:
            sink.emit("driver_error", {"error": error})
        rows = summary.get("tasks") or []
        agg = summary.get("aggregate") or {}
        sink.emit(
            "exit",
            {"content": exit_content(summary, n_goals), "exit_status": summary.get("exit_status")},
        )
        return SessionOutcome(
            usage={
                "peak_prompt_tokens": agg.get("peak_prompt_tokens"),
                "judge_prompt_tokens": agg.get("judge_prompt_tokens"),
                "judge_calls": agg.get("judge_calls"),
                "billing": ledger,
                # the same usage by kind of session: the tasks' (agent, judge,
                # wake-up) and the run end's consolidation
                "tokens": {"tasks": agg.get("tokens") or {}, "run": summary.get("run_tokens") or {}},
            }
            if agg
            else {"billing": ledger},
            cost_usd=None,  # no meter of its own; CostLogger prices the ledger
            turns=sum(int(r.get("turns") or 0) for r in rows) or None,
            error=error,
            extra={
                "exit_status": summary.get("exit_status"),
                "navh_tasks": len(rows),
                "navh_dir": str(out_dir),
                "navh_aggregate": json_safe(
                    {
                        k: agg.get(k)
                        for k in (
                            "tasks",
                            "e_sr",
                            "s_sr",
                            "spl",
                            "false_completion_rate",
                            "claims",
                            "verdicts",
                            "verdict_vs_truth",
                            "precheck_refusals",
                            "gate_refused_tasks",
                            "gate_truth_read",
                            "gate_correct_refusals",
                            "gate_rescues",
                            "gate_harms",
                            "recoveries",
                            "recoveries_by_scope",
                            "resets",
                            "steps",
                        )
                    }
                )
                if agg
                else {},
                "navh_claims": [
                    {
                        "task": r.get("index"),
                        "claim": (r.get("claim") or {}).get("status"),
                        "verdict": (r.get("verdict") or {}).get("verdict"),
                        "end_reason": r.get("end_reason"),
                        "steps": r.get("steps"),
                        "turns": r.get("turns"),
                    }
                    for r in rows
                ],
            },
        )


__all__ = ["NAVH_VERSION", "EsOrchestrator", "NavHarnessAdapter", "judge_route"]
