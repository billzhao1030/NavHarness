"""anthropic-messages wire endpoint — /v1/messages for cross-API seats.

Why not the litellm proxy for this wire: its /v1/messages path (and the
underlying ``litellm.anthropic_messages``) resolves the provider EARLY and
hits the chat-completions handler with the literal remainder of the slug —
an ``openai/responses/gpt-5.6`` route reaches OpenAI as the nonexistent
model ``responses/gpt-5.6``. The plain
``litellm.acompletion`` dispatch handles the same slug correctly, so this
endpoint does what the proxy should: translate the anthropic-format request
with litellm's own adapter, call ``acompletion`` directly, translate back.

Streaming: Claude Code always streams. Rather than re-implementing litellm's
per-chunk translation (content-block-index bookkeeping), the endpoint
completes the call NON-streaming and emits one protocol-valid anthropic SSE
sequence from the finished response — message_start, one start/delta/stop
triple per content block, message_delta with stop_reason and usage,
message_stop. For eval turns this only trades time-to-first-token, not
correctness.

Runs as its own subprocess (see ``AnthropicWireGateway`` at the bottom).
Config rides two env vars: AC_SHIM_ROUTES (JSON alias→litellm slug) and
AC_SHIM_KEY (bearer token Claude Code presents via ANTHROPIC_AUTH_TOKEN).
"""

# NB: no `from __future__ import annotations` here — FastAPI resolves the
# handler annotations at runtime, and with postponed annotations the locally
# imported Request type stops resolving (the parameter degrades to a query
# field and every call 422s). Python 3.10 handles the `X | None` forms natively.
import json
import os
import sys

from .service import REPO_ROOT, MASTER_KEY, LoopbackService


def sse_lines(events: list[tuple[str, dict]]):
    for name, data in events:
        yield f"event: {name}\ndata: {json.dumps(data)}\n\n"


def synthesize_sse(anth: dict) -> list[tuple[str, dict]]:
    """One valid SSE sequence from a COMPLETE anthropic response."""
    usage = anth.get("usage") or {}
    events: list[tuple[str, dict]] = [("message_start", {
        "type": "message_start",
        "message": {**anth, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage.get("input_tokens", 0),
                              "output_tokens": 0}}})]
    for i, block in enumerate(anth.get("content") or []):
        btype = block.get("type")
        if btype == "tool_use":
            start = {"type": "tool_use", "id": block["id"],
                     "name": block["name"], "input": {}}
            delta = {"type": "input_json_delta",
                     "partial_json": json.dumps(block.get("input") or {})}
        elif btype == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta",
                     "thinking": block.get("thinking", "")}
        else:
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block.get("text", "")}
        events += [
            ("content_block_start", {"type": "content_block_start",
                                     "index": i, "content_block": start}),
            ("content_block_delta", {"type": "content_block_delta",
                                     "index": i, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": i}),
        ]
    events += [
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": anth.get("stop_reason"),
                                     "stop_sequence": anth.get("stop_sequence")},
                           "usage": {"output_tokens": usage.get("output_tokens", 0)}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return events


# ── server side (runs under `python -m core.api.anthropic_wire`) ──

def _build_app():
    import litellm
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    routes: dict[str, str] = json.loads(os.environ["AC_SHIM_ROUTES"])
    key = os.environ.get("AC_SHIM_KEY", MASTER_KEY)
    adapter = LiteLLMAnthropicMessagesAdapter()
    app = FastAPI()

    def _authed(request: Request) -> bool:
        auth = request.headers.get("authorization", "")
        return key in (request.headers.get("x-api-key"), auth.removeprefix("Bearer ").strip())

    @app.get("/health/liveliness")
    async def liveliness():
        return {"status": "ok", "routes": routes}

    @app.post("/v1/messages")
    async def messages(request: Request):
        if not _authed(request):
            raise HTTPException(401, "bad shim key")
        body = await request.json()
        alias = body.get("model", "")
        if alias not in routes:
            raise HTTPException(404, f"unknown alias {alias!r}; routes: {list(routes)}")
        wants_stream = bool(body.pop("stream", False))
        openai_req, tool_name_mapping = adapter.translate_anthropic_to_openai(body)
        openai_req["model"] = routes[alias]
        openai_req.pop("stream", None)
        resp = await litellm.acompletion(**openai_req, drop_params=True)
        anth = adapter.translate_openai_response_to_anthropic(resp, tool_name_mapping)
        anth = anth if isinstance(anth, dict) else anth.model_dump(exclude_none=False)
        anth["model"] = alias  # the caller asked for the alias, echo it back
        if not wants_stream:
            return JSONResponse(anth)
        return StreamingResponse(sse_lines(synthesize_sse(anth)),
                                 media_type="text/event-stream")

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        if not _authed(request):
            raise HTTPException(401, "bad shim key")
        body = await request.json()
        alias = body.get("model", "")
        try:
            n = litellm.token_counter(model=routes.get(alias, alias),
                                      messages=body.get("messages") or [])
        except Exception:  # noqa: BLE001 — an estimate, never a blocker
            n = sum(len(json.dumps(m)) // 4 for m in body.get("messages") or [])
        return JSONResponse({"input_tokens": int(n)})

    return app


def main() -> None:
    import uvicorn
    port = int(sys.argv[sys.argv.index("--port") + 1])
    uvicorn.run(_build_app(), host="127.0.0.1", port=port, log_level="warning")


# ── client side (constructed by core.api.gateway_for) ──

class AnthropicWireGateway(LoopbackService):
    PORT_POOL = tuple(range(4110, 4120))
    START_TIMEOUT = 60.0
    POLL_INTERVAL = 0.5

    def __init__(self, routes: dict[str, str], tag: str = "shim") -> None:
        super().__init__(tag)
        self.routes = routes

    def command(self) -> list[str]:
        return [sys.executable, "-m", "core.api.anthropic_wire",
                "--port", str(self.port)]

    def spawn_env(self) -> dict[str, str]:
        return {**os.environ, "AC_SHIM_ROUTES": json.dumps(self.routes),
                "AC_SHIM_KEY": MASTER_KEY}

    def spawn_cwd(self) -> str:
        return str(REPO_ROOT)

    def anthropic_env(self) -> dict[str, str]:
        return {"ANTHROPIC_BASE_URL": self.base_url,
                "ANTHROPIC_AUTH_TOKEN": MASTER_KEY}

    def describe(self) -> dict:
        return {"backend": "anthropic-shim (in-process litellm acompletion)",
                "auth": "litellm gateway (vendor API billing)",
                "port": self.port, "routes": dict(self.routes),
                "streaming": "synthesized SSE from a completed response"}


if __name__ == "__main__":
    main()
