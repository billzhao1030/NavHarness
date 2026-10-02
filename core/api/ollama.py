"""ollama serving — the shared local server, pinned context, verified sampling.

The serving contract is endpoint machinery, not harness logic. ollama's
default context is 4096:
every request past it is silently truncated, the run completes, the numbers
look plausible and are garbage. So this module — not a wrapper script the
standard path can be run without — owns the server and pins the context.

Unlike the wire gateways this is NOT an owned-per-run LoopbackService: the
server is shared and long-lived, ``ensure()`` reuses it when it is already
serving at the pinned context AND writing to our log, and kill+restarts it
(user-scoped pkill only) otherwise. OLLAMA_URL is env-overridable for hosts
where 11434 is held by a system ollama that cannot be replaced.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .providers.local import OLLAMA_URL
from .service import REPO_ROOT

SERVE_CTX = 131072
SERVE_LOG = REPO_ROOT / "outputs" / "mini-swe-agent" / "_ollama_serve.log"


def _bin() -> str | None:
    # the user-local install wins over PATH: it is the one we can upgrade
    # without root, so when both exist it is the newer binary
    return next(
        (str(p) for p in [Path.home() / "ollama" / "bin" / "ollama"] if p.exists()), None
    ) or shutil.which("ollama")


def _get(url: str, timeout: float = 2.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:  # noqa: BLE001
        return None


def _post(url: str, payload: dict, timeout: float = 10.0) -> dict | None:
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001
        return None


def _pid() -> str | None:
    # -u scopes to OUR server: a root/system ollama on the box must be neither
    # matched (its /proc is unreadable) nor killed (we couldn't anyway)
    pids = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "ollama serve"],
                          capture_output=True, text=True).stdout.split()
    return pids[0] if pids else None


def _serving_ctx() -> int | None:
    """OLLAMA_CONTEXT_LENGTH the live server was started with. None = no server;
    0 = server up but the variable is unset, i.e. the 4096 truncation default."""
    if _get(f"{OLLAMA_URL}/api/version") is None:
        return None
    pid = _pid()
    if pid is None:
        return 0
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return 0
    for kv in environ:
        if kv.startswith(b"OLLAMA_CONTEXT_LENGTH="):
            return int(kv.split(b"=", 1)[1])
    return 0


def _we_own_the_log() -> bool:
    """Is the live server writing to OUR log? If not we cannot slice its exact
    per-request token counts, and the audit would silently return nothing —
    the same silent-degradation class as an unpinned context. So this forces a
    restart rather than quietly shipping a run with no token accounting."""
    pid = _pid()
    if pid is None:
        return False
    try:
        return Path(f"/proc/{pid}/fd/1").resolve() == SERVE_LOG.resolve()
    except OSError:
        return False


class OllamaServer:
    """ensure → verify → describe; log_offset/audit for token accounting."""

    url = OLLAMA_URL
    ctx = SERVE_CTX
    log = SERVE_LOG

    def ensure(self, model_tag: str) -> dict:
        """Bring the pinned server up (reuse if compatible), verify the model's
        sampling is baked server-side, return the serving describe dict."""
        binary = _bin()
        if binary is None:
            raise RuntimeError("ollama binary not found — a local cell cannot run")

        if _serving_ctx() != SERVE_CTX or not _we_own_the_log():
            subprocess.run(["pkill", "-u", str(os.getuid()), "-f", "ollama serve"],
                           check=False)
            for _ in range(30):
                if _get(f"{OLLAMA_URL}/api/version", 1.0) is None:
                    break
                time.sleep(1)
            SERVE_LOG.parent.mkdir(parents=True, exist_ok=True)
            log = SERVE_LOG.open("a")
            subprocess.Popen(  # noqa: S603
                [binary, "serve"], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "OLLAMA_CONTEXT_LENGTH": str(SERVE_CTX),
                     "OLLAMA_HOST": urllib.parse.urlparse(OLLAMA_URL).netloc},
                start_new_session=True)
            for _ in range(60):
                if _get(f"{OLLAMA_URL}/api/version", 1.0) is not None:
                    break
                time.sleep(1)

        got = _serving_ctx()
        if got != SERVE_CTX:
            raise RuntimeError(
                f"ollama is serving at context {got} (need {SERVE_CTX}) — refusing to "
                "run: past its window ollama truncates silently and the run would look "
                "fine while being worthless"
            )
        print(f"[mini] ollama serving at {SERVE_CTX} ctx (pinned)")

        # The serving layer, read back FROM THE SERVER — the only place it is real.
        show = _post(f"{OLLAMA_URL}/api/show", {"model": model_tag}) or {}
        info = show.get("model_info") or {}
        sampling: dict[str, str] = {}
        for line in (show.get("parameters") or "").splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                sampling[parts[0]] = parts[1].strip()

        # Reproducibility guard, same principle as the context pin: verify, don't
        # trust. litellm's ollama route does not support presence_penalty and
        # drop_params=True makes it vanish silently, so sampling the harness
        # "sets" may never reach the server. The Modelfile is therefore the single
        # source of truth — and this asserts what it actually says.
        deterministic = sampling.get("temperature") == "0" or "seed" in sampling
        if not deterministic:
            raise RuntimeError(
                f"{model_tag} serves with sampling {sampling or '(ollama defaults)'} "
                "— neither temperature=0 nor a seed is pinned, so every episode is a "
                "different random sample and no result here is reproducible. Bake the "
                "sampling into a Modelfile (`ollama create <tag>-greedy -f ...`): the "
                "harness cannot pin presence_penalty through litellm."
            )
        print(f"[mini] sampling pinned server-side: {sampling}")
        # A Modelfile's num_ctx takes precedence over the server's context: the
        # window a request is actually cut at is the model's, when it sets one.
        model_ctx = int(sampling["num_ctx"]) if sampling.get("num_ctx", "").isdigit() else None
        window = model_ctx or SERVE_CTX
        if model_ctx is not None and model_ctx < SERVE_CTX:
            print(f"[mini] {model_tag} sets num_ctx {model_ctx} in its Modelfile: that is the "
                  f"window requests are served with (the server's default is {SERVE_CTX})")

        return {
            "backend": "ollama",
            # OLLAMA_HOST: --version reports the SERVER version at the client's
            # default host — without this it records some other daemon's number
            "version": subprocess.run(
                [binary, "--version"], capture_output=True, text=True,
                env={**os.environ,
                     "OLLAMA_HOST": urllib.parse.urlparse(OLLAMA_URL).netloc},
            ).stdout.strip(),
            "served_context": SERVE_CTX,
            "model_context": model_ctx,       # the Modelfile's num_ctx, which wins when set
            "effective_context": window,
            "native_context": next(
                (v for k, v in info.items() if k.endswith("context_length")), None),
            "quantization": (show.get("details") or {}).get("quantization_level"),
            "sampling": sampling,
            "deterministic": deterministic,
            "note": "sampling comes from the MODELFILE (read back from /api/show), "
                    "not from the harness — litellm's ollama route drops "
                    "presence_penalty, so pinning it client-side would be a no-op",
        }

    def log_offset(self) -> int:
        """Where this run's stretch of the serve log starts."""
        return SERVE_LOG.stat().st_size if SERVE_LOG.exists() else 0

    def audit(self, offset: int) -> dict:
        """Exact per-request prompt-token counts, straight from llama.cpp, for
        one run's stretch of the serve log. Proves the context window was never
        crossed — the claim nothing else in the stack can make. Says so out loud
        when it cannot be produced; a missing audit must not look like a clean one."""
        if not SERVE_LOG.exists():
            return {"prompt_tokens": {"unavailable": f"{SERVE_LOG} missing"}}
        with SERVE_LOG.open(errors="ignore") as fh:
            fh.seek(offset)
            toks = [int(m.group(1)) for line in fh
                    if (m := re.search(r"task\.n_tokens = (\d+)", line))]
        if not toks:
            return {"prompt_tokens": {
                "unavailable": "no `task.n_tokens` lines in this cell's slice of "
                               "the serve log — the server is not the one we started"}}
        return {"prompt_tokens": {
            "llm_requests": len(toks), "peak": max(toks),
            "mean": round(sum(toks) / len(toks)), "sum": sum(toks),
            "served_context": SERVE_CTX,
            "peak_pct_of_window": round(100 * max(toks) / SERVE_CTX, 1),
            "over_32k": sum(1 for t in toks if t > 32768),
            "over_64k": sum(1 for t in toks if t > 65536),
        }}
