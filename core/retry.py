"""RetryPolicy — the rate-limit retry the run loop applies per episode.

An episode whose session ended in a provider rate limit is re-run in
place, with exponential backoff, up to ``max_attempts`` times; the
previous attempt's files are archived (``*.attempt{k}*``, see
runner.archive_attempt) so nothing is lost. "session limit" / "usage
limit" = the 5 h subscription window (resets on a clock, not transient) —
a retry cannot clear it within the window, but tagging it routes the
episode to error="rate_limited" -> excluded from the scored episodes (not
scored 0 as a navigation failure).
"""

from __future__ import annotations

from dataclasses import dataclass

RATE_LIMIT_MARKERS = ("temporarily limiting", "rate limited", "overloaded",
                      "rate_limit", "429", "session limit", "usage limit",
                      "hit your session")


def is_rate_limited(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in RATE_LIMIT_MARKERS)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    base_backoff_s: float = 30.0     # exponential per attempt
    max_backoff_s: float = 300.0     # per-backoff cap

    def backoff(self, attempt: int) -> float:
        return min(self.base_backoff_s * 2 ** (attempt - 1), self.max_backoff_s)


DEFAULT_RETRY = RetryPolicy()
