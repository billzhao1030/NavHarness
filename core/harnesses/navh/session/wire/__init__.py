"""The wire — what a session stands on.

    client.py   SessionLoop (the sessions' event loop, on a thread of its own) · SdkWire (one
                connected ClaudeSDKClient, driven by one task from connect to disconnect) · Hop
    blocks.py   content blocks (text, image), usage, the CLI's environment
    trace.py    every hop saved whole, filed under its session's kind
    home.py     where the CLI keeps its transcripts (out of ~/.claude), and their pruning

``session/base.py`` is the only caller; the kinds above it never reach in
here, and the orchestrator takes what it needs from the ``session`` package.
"""

from __future__ import annotations

from .blocks import Blocks, image_block, prompt_text, text_block
from .client import Hop, SdkWire, SessionLoop
from .home import ensure_relocated
from .trace import Trace

__all__ = ["Blocks", "Hop", "SdkWire", "SessionLoop", "Trace", "ensure_relocated", "image_block", "prompt_text",
           "text_block"]
