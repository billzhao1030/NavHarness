"""The fixed-access memory control (``retrieval: bm25``): BM25 over the task
records and the house notes, on a fixed schedule, with the file tools scoped
to the task's own files as under ``ledger=none``.

The retrieval itself (chunks, scores, the budget, merging, recency, sources
and statuses), the records it may draw on, the scope the file tools keep,
the switch's place among the session policies, and the orchestrator keeping
the schedule — at task start, every ten turns, after a recovery — over
scripted sessions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_navh_orchestrator import (  # noqa: F401 — scripted is a fixture
    _Bench,
    _close_request,
    _hop,
    _orch,
    _walk,
    _Wire,
    _write,
    scripted,
)

from core.harnesses.navh import prompts as P
from core.harnesses.navh.config import HarnessConfig
from core.harnesses.navh.memory.retrieval import (
    FixedRetriever,
    Record,
    bm25,
    chunk_spans,
    historical_records,
    tokens,
)
from core.harnesses.navh.memory.workspace import Workspace
from core.harnesses.navh.session import prompt_text
from core.harnesses.navharness import NavHarnessAdapter

# ── the retrieval ──


def test_chunks_overlap_and_cover_the_record() -> None:
    spans = chunk_spans([1] * 1000, 400, 50)
    assert spans == [(0, 400), (350, 750), (700, 1000)]
    assert chunk_spans([1] * 10, 400, 50) == [(0, 10)]
    assert chunk_spans([500, 1, 1], 400, 50) == [(0, 1), (1, 3)], "an over-long piece is a chunk of its own"


def test_bm25_prefers_the_rarer_term_and_never_scores_a_miss() -> None:
    docs = [["fridge", "kitchen"], ["kitchen", "hall"], ["kitchen", "sink"], ["bed"]]
    s = bm25(["fridge", "kitchen"], docs)
    assert s[0] > s[1] == s[2] > 0 and s[3] == 0.0


def _rec(source: str, text: str, task: int | None, status: str = "CERTIFIED") -> Record:
    return Record(source, text, status, task, 1e12 + (task or 0))


def test_top_k_within_the_budget_with_sources_and_statuses() -> None:
    recs = [_rec(f"run/tasks/{j}/handover.md", f"the fridge is in room {j} " * 3, j + 1) for j in range(8)]
    text, held = FixedRetriever(top_k=5, budget=2000).render(recs, "where is the fridge")
    assert len(held) == 5 and text.startswith("Retrieved from")
    assert held[0]["task"] == 8, "equal scores: the newer record first"
    assert "[run/tasks/7/handover.md · task 8 · CERTIFIED]" in text
    tight, held = FixedRetriever(top_k=5, budget=60).render(recs, "fridge")
    assert sum(tokens(f"[{h['source']} · task {h['task']} · {h['status']}]") + h["tokens"] for h in held) <= 60
    assert 1 <= len(held) < 5


def test_overlapping_chunks_of_one_record_are_merged_into_one_excerpt() -> None:
    long = " ".join(f"fridge{i % 3} kitchen word{i}" for i in range(300))     # ~900 tokens
    retr = FixedRetriever(top_k=5, budget=2000, chunk=400, overlap=50)
    excerpts, pieces = retr.retrieve([_rec("house/house.md", long, None, "house note")], "kitchen fridge0")
    assert len(excerpts) == 1, "every chunk of the one record overlaps the next: one excerpt"
    assert excerpts[0].text(pieces[id(excerpts[0].record)]) == long.strip()


def test_nothing_to_retrieve_is_said_plainly() -> None:
    assert "no earlier records" in FixedRetriever().render([], "fridge")[0]
    text, held = FixedRetriever().render([_rec("run/ledger.md", "1. CERTIFIED  bed", 1)], "fridge")
    assert held == [] and "nothing" in text


def _workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "ws", tmp_path / "house")
    ws.append_ledger("    1. CERTIFIED  find a fridge · kitchen")
    ws.append_ledger("    2. NOT DONE   find a bed")
    for k, verdict, handover in ((0, "complete", "the fridge is in the kitchen"),
                                 (1, "incomplete", "searched the hall, no bed"),
                                 (2, None, "")):
        ws.write_goal(k, f"goal {k}", k, 3, "object")
        if handover:
            ws.handover_path(k).write_text(handover)
        if verdict:
            ws.write_verdict(k, {"verdict": verdict, "evidence": "seen", "searched": ["hall"], "usage": {}})
    (ws.house / "house.md").write_text("# house\nthe kitchen is downstairs")
    (ws.house / "journal").mkdir(parents=True, exist_ok=True)
    (ws.house / "journal" / "2026-09-01.md").write_text("# 2026-09-01\n\n- task 1 · CERTIFIED · find a sofa · account: lounge\n")
    (ws.house / "journal" / "2026-09-02.md").write_text("# 2026-09-02\n\n- task 1 · CERTIFIED · find a fridge\n")
    return ws


def test_the_records_are_historical_text_only(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    recs = historical_records(ws, 2, day="2026-09-02")
    sources = {(r.source, r.task, r.status) for r in recs}
    assert ("run/ledger.md", 1, "CERTIFIED") in sources and ("run/ledger.md", 2, "NOT DONE") in sources
    assert ("run/tasks/0/handover.md", 1, "CERTIFIED") in sources
    assert ("run/tasks/1/verdict.json", 2, "NOT DONE") in sources
    assert not any(s.startswith("run/tasks/2/") for s, _, _ in sources), "the current task's own files are not history"
    assert ("house/house.md", None, "house note") in sources
    assert ("house/journal/2026-09-01.md", 1, "CERTIFIED") in sources, "an earlier day's journal"
    assert not any(s == "house/journal/2026-09-02.md" for s, _, _ in sources), "today's journal is this run's records"
    verdict = next(r for r in recs if r.source == "run/tasks/1/verdict.json")
    assert verdict.text.startswith("verdict: incomplete") and "usage" not in verdict.text


def test_the_file_tools_serve_only_the_current_tasks_files(tmp_path: Path) -> None:
    """The scope ``ledger=none`` gives the bridge's file tools, which ``bm25`` holds."""
    ws = _workspace(tmp_path)
    scope = "run/tasks/2"
    assert "text" in ws.session_read("run/tasks/2/goal.md", scope=scope)
    for hidden in ("run/ledger.md", "run/tasks/0/handover.md", "house/house.md", "run/tasks/1/goal.md"):
        assert "not available" in ws.session_read(hidden, scope=scope)["error"], hidden
    assert ws.session_list("", scope=scope)["entries"] == ["run/"]
    assert ws.session_list("run", scope=scope)["entries"] == ["tasks/"]
    assert ws.session_list("run/tasks", scope=scope)["entries"] == ["2/"]
    assert "goal.md" in ws.session_list("run/tasks/2", scope=scope)["entries"]
    assert "error" in ws.session_list("house", scope=scope)
    assert "text" in ws.session_read("run/ledger.md"), "no scope: the reference reads everything"


# ── the switches ──


def test_the_switch_defaults_to_the_reference_and_refuses_what_it_does_not_know() -> None:
    a = NavHarnessAdapter()
    assert a.switches["retrieval"] == "none" and a.switches["retrieval_every_turns"] == 10
    assert HarnessConfig.from_flat(benchmark="es", chain_id=0).memory.retrieval == "none"
    b = NavHarnessAdapter(retrieval="bm25", retrieval_top_k=3)
    assert b.switches["retrieval"] == "bm25" and b.switches["retrieval_top_k"] == 3
    with pytest.raises(ValueError):
        NavHarnessAdapter(retrieval="dense")
    with pytest.raises(ValueError):
        HarnessConfig.from_flat(benchmark="es", chain_id=0, retrieval="dense")


def test_fixed_access_holds_the_records_out_of_view_and_a_context_only_session_holds_it_off() -> None:
    cfg = HarnessConfig.from_flat(benchmark="es", chain_id=0, retrieval="bm25")
    assert cfg.memory.ledger == "none" and cfg.held == {"ledger": "none"}, "one scope for both"
    for mode in ("independent", "single"):
        cfg = HarnessConfig.from_flat(benchmark="es", chain_id=0, session_mode=mode, retrieval="bm25")
        assert cfg.memory.retrieval == "none" and cfg.held["retrieval"] == "none", mode


def test_the_experiment_composes_with_fixed_access() -> None:
    from runner import build, compose_app
    exp = Path(__file__).resolve().parents[1] / "exp_workspace/lhnav_es/configs/std_goat_slam_es_lhnav.yaml"
    app = compose_app(exp, ["harness=navh", "model=opus-5", "harness.retrieval=bm25"])
    adapter, *_ = build(app)
    assert adapter.switches["retrieval"] == "bm25" and adapter.switches["retrieval_budget_tokens"] == 2000


# ── the schedule, through the orchestrator ──


def test_the_schedule_task_start_every_ten_turns_and_after_a_recovery(tmp_path: Path, scripted) -> None:
    seen: list[tuple[int, str]] = []

    def agent(wire, prompt):
        orch = _Wire.orch
        text = prompt_text(prompt)
        seen.append((orch.k, text))
        if orch.k == 0:
            if text.startswith("Task"):
                _walk(orch)
                return _hop("I see a kitchen with a fridge.", 10, "observe", "step")
            if text.startswith("Continue."):
                _write(orch, "run/tasks/0/handover.md", "The fridge is in the kitchen beside the window.")
                _close_request(orch)
                return _hop("DONE", 2, "write_file", "close_task")
            raise AssertionError(text[:80])
        if text.startswith("Task"):
            _walk(orch)
            st = orch.state.read()
            st["recovery_request"] = {"scope": "attempt", "reason": "stuck", "confidence": 0.9}
            orch.state.write(st)
            return _hop("Hall again, no fridge here.", 4, "observe", "step", "request_recovery")
        if "write the note your next self will read" in text:
            _write(orch, "run/tasks/1/lambda.md", json.dumps({"searched": ["hall"], "untried": ["kitchen"]}))
            return _hop("DONE", 1, "write_file")
        if text.startswith("This is a fresh attempt"):
            _write(orch, "run/tasks/1/handover.md", "Back in the kitchen.")
            _close_request(orch)
            return _hop("DONE", 2, "write_file", "close_task")
        raise AssertionError(text[:80])

    _Wire.handlers["agent"] = agent
    o = _orch(tmp_path, _Bench(n=2), notes=False, retrieval="bm25", recovery=True, max_resets_per_task=1,
              recovery_min_turns=0)
    assert o.agent_session.system_prompt == P.SYSTEM_FIXED_ACCESS and o.cfg.memory.ledger == "none"
    assert o.agent_session.mcp["env"]["env"]["NAVH_LEDGER"] == "none", "the bridge scopes the file tools"
    assert getattr(o.agent_session, "notes", None) is None, "no notes fence opened on Claude Code"
    o.run()
    first = seen[0][1]
    assert P.FIXED_ACCESS_REQUIRED in first and "no earlier records yet" in first
    assert seen[1][1].startswith("Continue.") and "Retrieved from" in seen[1][1], "ten turns: a retrieval"
    second = next(t for k, t in seen if k == 1 and t.startswith("Task"))
    assert "[run/tasks/0/handover.md · task 1" in second and "kitchen beside the window" in second
    fresh = next(t for k, t in seen if k == 1 and t.startswith("This is a fresh attempt"))
    assert "Retrieved from" in fresh, "a recovery: a retrieval"
    events = [json.loads(line) for line in (o.out_dir / "episode_1.jsonl").read_text().splitlines()]
    whys = [e["why"] for e in events if e["kind"] == "retrieval"]
    assert whys == ["task_start", "recovery"]
    q = next(e for e in events if e["kind"] == "retrieval")["query"]
    assert q.startswith("find thing 1") and "kitchen with a fridge" in q, "goal + latest description"
    assert [r["context"]["retrievals"] for r in o.rec.rows] == [2, 2]


def test_the_hop_cap_is_the_turns_left_until_the_next_retrieval(tmp_path: Path, scripted) -> None:
    """min(retrieval_every_turns, H, B - turns): a hop that ends early leaves
    the rest of the ten for the next one, which then opens with a retrieval."""
    caps: list = []

    def agent(wire, prompt):
        orch = _Wire.orch
        caps.append((wire.limits[-1], "Retrieved from" in prompt_text(prompt)))
        if len(caps) == 4:
            _write(orch, "run/tasks/0/handover.md", "done")
            _close_request(orch)
            return _hop("DONE", 1, "write_file", "close_task")
        _walk(orch)
        return _hop("A hall.", {1: 7, 2: 3, 3: 10}[len(caps)], "observe", "step")

    _Wire.handlers["agent"] = agent
    o = _orch(tmp_path, _Bench(n=1), notes=False, retrieval="bm25")
    o.run()
    assert caps == [(10, True), (3, False), (10, True), (10, True)]


def test_the_reference_keeps_its_hops_prompt_and_files(tmp_path: Path, scripted) -> None:
    o = _orch(tmp_path, _Bench(n=1), notes=False)
    assert o.retriever is None and o.agent_session.max_turns == 25 and o.agent_session.system_prompt.startswith(
        P.SYSTEM)
    assert o.agent_session.mcp["env"]["env"]["NAVH_LEDGER"] == "text"
    o.run()
