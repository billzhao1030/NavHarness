"""The callback contract — how a recorder plugs into the runner.

A callback is a class with any subset of seven hooks; the runner fires them
in registration order at fixed points of the run (below). Callbacks OWN
their outputs: every file a callback writes and every summary / record key
it adds is listed in its ``writes`` tuple, which the runner publishes as
``summary["writers"]`` so any number in a run can be traced to the class
that wrote it. Removing a callback removes exactly those outputs.

    on_run_start(run)                 after cfg + health guards, before episode 0;
                                      run.prior_episodes holds the resumed records
    on_episode_start(ep)              the EpisodeContext is built, nothing emitted yet
    on_event(ep, kind, rec)           every bus.emit — rec = {"t", "kind", **payload}
    on_raw(ep, rec)                   every harness-native record (SDK message /
                                      codex event, already json_safe'd)
    on_episode_end(ep, record)        ALWAYS, in the episode's finally — record is
                                      None when the session raised / timed out
                                      (the runner then writes a stub); may add
                                      keys under record["agent"]
    on_run_end(run, summary)          before the final summary flush; may mutate
                                      summary and summary["run_stats"]
    on_run_written(run, summary_path) after the final flush (for readers of the
                                      file on disk)

Errors: a hook that raises is recorded under ``run_stats.callback_errors``
and the run goes on — unless the callback is ``fatal`` (CodeSnapshot: a run
without its provenance must not start). ``StopRun`` from any hook is not an
error: the runner drains (finishes in-flight episodes, pulls no new index)
and records ``run_stats.stopped_by``.
"""

from __future__ import annotations

from typing import Any


class StopRun(Exception):
    """Raised by a hook to end the run gracefully at the next episode
    boundary (a budget fuse, a quality gate …)."""


class Callback:
    writes: tuple[str, ...] = ()   # dotted summary / record paths + files this class owns
    fatal: bool = False            # an exception here aborts the run

    @property
    def name(self) -> str:
        return type(self).__name__

    def on_run_start(self, run: Any) -> None:
        pass

    def on_episode_start(self, ep: Any) -> None:
        pass

    def on_event(self, ep: Any, kind: str, rec: dict[str, Any]) -> None:
        pass

    def on_raw(self, ep: Any, rec: Any) -> None:
        pass

    def on_episode_end(self, ep: Any, record: dict[str, Any] | None) -> None:
        pass

    def on_run_end(self, run: Any, summary: dict[str, Any]) -> None:
        pass

    def on_run_written(self, run: Any, summary_path: Any) -> None:
        pass


class CallbackSet:
    """The registered callbacks, fired in order with the error policy above."""

    def __init__(self, callbacks: list[Callback], run: Any = None) -> None:
        self.callbacks = list(callbacks)
        self.run = run

    def __iter__(self):
        return iter(self.callbacks)

    def names(self) -> list[str]:
        return [cb.name for cb in self.callbacks]

    def writers(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for cb in self.callbacks:
            for path in cb.writes:
                out[path] = cb.name
        return out

    def fire(self, hook: str, *args: Any) -> None:
        for cb in self.callbacks:
            try:
                getattr(cb, hook)(*args)
            except StopRun:
                raise
            except Exception as exc:  # noqa: BLE001 — see the module docstring
                if cb.fatal:
                    raise
                print(f"[std] callback {cb.name}.{hook} failed (non-fatal): {exc!r}")
                if self.run is not None:
                    self.run.callback_errors[f"{cb.name}.{hook}"] = repr(exc)
