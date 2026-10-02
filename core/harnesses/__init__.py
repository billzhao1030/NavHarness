"""Harness adapters — one per closed/open agent stack.

Each adapter implements core.episode.HarnessAdapter: run ONE clean session
for one episode, emitting through the shared EventBus. Everything else
(placement, prompts, evaluation, artifacts, summary) lives in runner.py,
core.shapes and core.callbacks. To check an adapter without a model,
``api=fake`` points it at the scripted endpoint (core.api.fake_wire).

An adapter is instantiated by runner.py from the ``agent.harness`` block of
the experiment config (``_target_`` + the harness's constants as keyword
arguments; an unknown key is a TypeError from the constructor — loud).
"""

from __future__ import annotations
