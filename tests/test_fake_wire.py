"""The fake endpoint's brain and both wire encoders, on canned requests — no
server, no harness. The script is the turn index over the tool results seen
so far; the names go back exactly as the request registered them."""

from __future__ import annotations

import json

from core.api.anthropic_wire import synthesize_sse
from core.api.fake_wire import (
    DISCOVER_CALL,
    anthropic_reply,
    chat_reply,
    chat_sse,
    default_script,
    handover_of,
    next_move,
    parse_script,
    responses_reply,
    responses_sse,
    short_name,
)

NAV = {"observe": set(), "step": {"actions"}}


def test_default_script_closes_the_way_the_task_does() -> None:
    assert default_script("r2r", NAV) == [("observe", {}), ("step", {"actions": [2, 1, 3]}),
                                          ("step", {"actions": [0]})]
    navh = {**NAV, "close_task": {"status", "evidence"}, "write_file": {"path", "text"}}
    walk = default_script("goat", navh, "run/tasks/3/handover.md")
    assert [t for t, _ in walk[-2:]] == ["write_file", "close_task"], "the handover is written before closing"
    assert walk[-2][1]["path"] == "run/tasks/3/handover.md"
    body = {"messages": [{"role": "user", "content": "Required reading: run/tasks/2/handover.md\n"
                                                     "run/tasks/3/handover.md   your handover"}]}
    assert handover_of(body) == "run/tasks/3/handover.md"


def test_next_move_walks_the_script_then_ends() -> None:
    assert next_move(0, NAV, "r2r", None) == ("observe", {})
    assert next_move(2, NAV, "r2r", None) == ("step", {"actions": [0]})
    assert next_move(3, NAV, "r2r", None) is None
    assert next_move(0, {}, "r2r", None) is None  # a side call registers no env tool
    custom = parse_script('[["step", {"actions": [1]}], ["step", {"actions": [0]}]]')
    assert next_move(1, NAV, "r2r", custom) == ("step", {"actions": [0]})
    assert next_move(0, {"step": {"actions"}}, "r2r", parse_script([["observe", {}], ["step", {}]])) == ("step", {})


def test_short_name() -> None:
    assert short_name("mcp__env__observe") == short_name("env__observe") == short_name("observe")


def _anthropic_body(n_results: int) -> dict:
    messages: list[dict] = [{"role": "user", "content": "go"}]
    for i in range(n_results):
        messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": "mcp__env__step", "input": {}}]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok"}]})
    return {"model": "claude-fable-5", "stream": True, "messages": messages, "tools": [
        {"name": "mcp__env__observe", "input_schema": {"type": "object", "properties": {}}},
        {"name": "mcp__env__step", "input_schema": {"type": "object",
                                                    "properties": {"actions": {}}}},
    ]}


def test_anthropic_wire_returns_the_registered_name_and_valid_sse() -> None:
    reply = anthropic_reply(_anthropic_body(1), "r2r", None)
    call = next(b for b in reply["content"] if b["type"] == "tool_use")
    assert (call["name"], call["input"]) == ("mcp__env__step", {"actions": [2, 1, 3]})
    assert reply["stop_reason"] == "tool_use"
    assert reply["usage"]["input_tokens"] == 0
    names = [name for name, _ in synthesize_sse(reply)]
    assert names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"]
    done = anthropic_reply(_anthropic_body(3), "r2r", None)
    assert done["stop_reason"] == "end_turn"
    assert all(b["type"] == "text" for b in done["content"])


def _chat_body(n_results: int, stream: bool = True) -> dict:
    messages: list[dict] = [{"role": "user", "content": "go"}]
    for i in range(n_results):
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "step", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    return {"model": "gpt-5.6", "stream": stream, "messages": messages, "tools": [
        {"type": "function", "function": {"name": "observe", "parameters": {"properties": {}}}},
        {"type": "function", "function": {"name": "step",
                                          "parameters": {"properties": {"actions": {}}}}},
    ]}


def test_chat_wire_returns_a_tool_call_then_stops() -> None:
    reply = chat_reply(_chat_body(0), "r2r", None)
    choice = reply["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]["function"]
    assert call["name"] == "observe" and json.loads(call["arguments"]) == {}
    chunks = chat_sse(reply)
    assert chunks[-1] == "data: [DONE]\n\n"
    last = json.loads(chunks[-2].removeprefix("data: "))
    assert last["choices"][0]["finish_reason"] == "tool_calls" and last["usage"]["total_tokens"] == 0
    done = chat_reply(_chat_body(3), "r2r", None)
    assert done["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in done["choices"][0]["message"]


def _responses_body(n_results: int) -> dict:
    items: list[dict] = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}]
    for i in range(n_results):
        items.append({"type": "function_call", "call_id": f"c{i}", "name": "mcp__env__step",
                      "arguments": "{}"})
        items.append({"type": "function_call_output", "call_id": f"c{i}", "output": "ok"})
    return {"model": "gpt-5.6-sol", "stream": True, "input": items, "tools": [
        {"type": "function", "name": "mcp__env__observe", "parameters": {"properties": {}}},
        {"type": "function", "name": "mcp__env__step", "parameters": {"properties": {"actions": {}}}},
        {"type": "local_shell"},  # codex's own — never scripted
    ]}


def test_responses_wire_returns_a_function_call_then_a_message_alone() -> None:
    reply = responses_reply(_responses_body(2), "r2r", None)
    call = next(o for o in reply["output"] if o["type"] == "function_call")
    assert (call["name"], json.loads(call["arguments"])) == ("mcp__env__step", {"actions": [0]})
    assert reply["usage"]["total_tokens"] == 0
    kinds = [json.loads(line.split("data: ", 1)[1]).get("type") for line in responses_sse(reply)]
    assert kinds[0] == "response.created" and kinds[-1] == "response.completed"
    assert kinds.count("response.output_item.done") == 2
    done = responses_reply(_responses_body(3), "r2r", None)
    assert [o["type"] for o in done["output"]] == ["message"]


def _code_mode_body(outputs: list[str], discovered: bool = True) -> dict:
    """codex 0.153: no function tools, an exec custom tool in a namespace item."""
    items: list[dict] = [
        {"type": "additional_tools", "role": "developer", "tools": [
            {"type": "namespace", "name": "functions", "tools": [
                {"type": "custom", "name": "exec", "description": "Run JavaScript code …"},
                {"type": "function", "name": "wait", "parameters": {"properties": {}}},
            ]}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
    ]
    if discovered:
        items += [
            {"type": "custom_tool_call", "call_id": DISCOVER_CALL, "name": "exec", "input": "…"},
            {"type": "custom_tool_call_output", "call_id": DISCOVER_CALL,
             "output": '["mcp__env__observe","mcp__env__step","exec_command"]'},
        ]
    for i, out in enumerate(outputs):
        items += [
            {"type": "custom_tool_call", "call_id": f"call_fake_{i}", "name": "exec", "input": "…"},
            {"type": "custom_tool_call_output", "call_id": f"call_fake_{i}", "output": out},
        ]
    return {"model": "gpt-5.6-sol", "stream": True, "input": items}


def test_code_mode_discovers_then_plays_the_script_through_exec() -> None:
    first = responses_reply(_code_mode_body([], discovered=False), "r2r", None)
    call = next(o for o in first["output"] if o["type"] == "custom_tool_call")
    assert call["name"] == "exec" and call["call_id"] == DISCOVER_CALL and "ALL_TOOLS" in call["input"]
    move = responses_reply(_code_mode_body(["ok"]), "r2r", None)
    call = next(o for o in move["output"] if o["type"] == "custom_tool_call")
    assert call["call_id"] == "call_fake_1"
    assert 'tools["mcp__env__step"]' in call["input"] and '{"actions": [2, 1, 3]}' in call["input"]
    assert "image(c)" in call["input"]
    done = responses_reply(_code_mode_body(["ok", "ok", "ok"]), "r2r", None)
    assert [o["type"] for o in done["output"]] == ["message"]
    kinds = [json.loads(line.split("data: ", 1)[1])["type"] for line in responses_sse(move)]
    assert kinds.count("response.output_item.done") == 2 and kinds[-1] == "response.completed"


def test_the_script_restarts_at_every_prompt_so_a_carried_session_walks_it_per_task() -> None:
    tools = [{"name": "mcp__env__observe", "input_schema": {"properties": {}}},
             {"name": "mcp__env__step", "input_schema": {"properties": {"actions": {}}}}]
    first_task = [{"role": "user", "content": "Task 1"},
                  {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "mcp__env__observe", "input": {}}]},
                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]}]
    mid = anthropic_reply({"tools": tools, "messages": first_task}, "r2r", None)
    assert mid["content"][1]["name"] == "mcp__env__step", "inside a task the walk continues"
    second = anthropic_reply({"tools": tools, "messages": first_task + [{"role": "user", "content": "Task 2"}]}, "r2r", None)
    assert second["content"][1]["name"] == "mcp__env__observe", "a new prompt starts the walk again"
    assert second["content"][1]["id"] != "toolu_fake_0", "tool ids stay unique across the conversation"
