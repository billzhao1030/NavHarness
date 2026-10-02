"""The orchestrator — one chain, and everything it reaches for, laid out by who calls whom.

    orchestrator.py  Orchestrator — the only caller in this package
      seat           checkpoint.py restore / replay · ledger.py replay rows · the house mounted
      task           _run_task — the hops: the session runs its own tool loop for one hop,
                     the state file is read, and
                       _close    judge.py judge_gate on the agent's first close request; close.py
                                 stop / force_stop — the STOP is the orchestrator's
                       _recover  recovery/ — the driver decides a rung, the session's last hop
                                 writes λ to its file, the rung is carried out
      seam           judge.py the certification · ledger.py TaskRow · checkpoint.py freeze
      close          the consolidate session, the sessions' report

    checkpoint.py    what is frozen at each seam, and how a run resumes or replays from it
    close.py         TaskOutcome · ProviderUnavailable · force_stop
    judge.py         Judge (none | model) · judge_gate · refusal_text
    ledger.py        TaskRow · Ledger — the row a closed task leaves
    recovery/        driver.py, and under it ladder.py · lam.py · wakeup.py

Depth is calling depth: orchestrator.py calls the five beside it; the driver calls
the three under it; nothing here calls orchestrator.py back.
"""

from __future__ import annotations

from .orchestrator import Orchestrator, agent_tool_names

__all__ = ["Orchestrator", "agent_tool_names"]
