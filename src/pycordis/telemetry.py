"""Lifecycle telemetry and per-fiber stats (spec §3.8).

Every fiber keeps cheap counters, read as an immutable :class:`FiberStats`
snapshot through ``fiber.stats``. When at least one listener is registered for
:data:`TELEMETRY` (``"internal/telemetry"``), each lifecycle step is also
published as a :class:`LifecycleEvent`::

    @ctx.on(TELEMETRY)
    def on_lifecycle(event: LifecycleEvent) -> None:
        print(event.kind, event.fiber, event.duration)

The kinds, and when they are emitted:

- ``"load"``: a load finished and the fiber is ``ACTIVE``; ``duration`` is the
  load time (plugin body and its async setup).
- ``"fail"``: a load failed; ``error`` is why and ``duration`` the time until
  it failed. Its ``state`` is always ``FAILED`` (the outcome), even while the
  unload of what the load set up is still running: that unload follows as its
  own ``"unload"`` event.
- ``"unload"``: an unload finished; ``duration`` is the unload time, including
  the wait for dependents to unload first.
- ``"restart"``: the supervisor scheduled a restart after a failure;
  ``duration`` is the backoff delay before it, ``error`` the failure.
- ``"give-up"``: the supervisor spent its restart budget and leaves the fiber
  ``FAILED``; ``error`` is the last failure.
- ``"timeout"``: a ``load_timeout`` expired (emitted before its ``"fail"``) or
  unload work was abandoned after ``dispose_timeout`` (one event per abandoned
  task or disposer); ``duration`` is the bound and ``error`` the timeout error.
  Its ``state`` is ``UNLOADING``: both kinds are emitted while the fiber unloads.

Without listeners no event is built: the cost on the lifecycle path is one
dictionary lookup per step. Listeners run synchronously with their owner's
context current; an error a listener raises is reported through the logger and
never reaches the lifecycle (an ``async`` listener runs as a task owned by its
fiber). Events are delivered to every listener regardless of isolation scopes.
Durations are measured with :func:`time.perf_counter`, ``ts`` is
:func:`time.time` at emission.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, final

from pycordis._context_internals import lookup
from pycordis.events import EventKey
from pycordis.fiber_core import FiberState, emit_reported

if TYPE_CHECKING:
    from pycordis.events import EventsService
    from pycordis.fiber_core import FiberCore

__all__ = ["TELEMETRY", "FiberStats", "LifecycleEvent", "LifecycleKind"]

type LifecycleKind = Literal["load", "unload", "fail", "restart", "give-up", "timeout"]
"""What a :class:`LifecycleEvent` reports (see :mod:`pycordis.telemetry`)."""


@final
@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """One step in a fiber's lifecycle, published on :data:`TELEMETRY`.

    Example::

        ctx.on(TELEMETRY, lambda event: print(f"{event.fiber} {event.kind} {event.duration}"))
        # worker load 0.0021
    """

    kind: LifecycleKind
    """What happened (see :mod:`pycordis.telemetry` for each kind)."""
    fiber: str
    """The fiber's name (its plugin's name, ``root`` for the root fiber)."""
    uid: int | None
    """The fiber's registry uid (``None`` once disposed, e.g. on its final unload)."""
    state: FiberState
    """The fiber's state when the event was emitted (``FAILED`` for every ``fail``)."""
    duration: float | None
    """Seconds the step took (or the bound / delay it reports); ``None`` if not timed."""
    error: BaseException | None
    """The error behind a ``fail``, ``restart``, ``give-up`` or ``timeout``, else ``None``."""
    ts: float
    """Wall-clock time of emission (:func:`time.time`)."""


TELEMETRY = EventKey[[LifecycleEvent], None]("internal/telemetry")
"""The typed telemetry event: ``ctx.on(TELEMETRY, listener)`` gets each :class:`LifecycleEvent`.

Example::

    durations: list[float] = []
    ctx.on(TELEMETRY, lambda e: e.kind == "load" and durations.append(e.duration or 0.0))
"""


@final
@dataclass(frozen=True, slots=True)
class FiberStats:
    """An immutable snapshot of a fiber's lifecycle counters (``fiber.stats``).

    Example::

        stats = fiber.stats
        print(f"{stats.loads} loads, {stats.failures} failures, last {stats.last_load_duration}")
    """

    loads: int
    """Loads that completed (the fiber became ``ACTIVE``)."""
    failures: int
    """Loads that failed (the fiber became ``FAILED``)."""
    restarts: int
    """Restarts the supervisor scheduled after failures."""
    last_load_duration: float | None
    """Seconds the last completed load took (``None`` before the first)."""
    last_error: BaseException | None
    """The error of the last failed load (kept after the fiber recovers)."""


@final
class StatsRecorder:
    """The mutable counters behind ``fiber.stats``, plus the start times of transitions.

    Example (what a fiber does around a load)::

        recorder.start_load()
        ...
        recorder.loaded()  # returns the load duration

    Internal — read ``fiber.stats`` instead.
    """

    __slots__ = (
        "_load_started",
        "_unload_started",
        "failures",
        "last_error",
        "last_load_duration",
        "loads",
        "restarts",
    )

    def __init__(self) -> None:
        self.loads = 0
        self.failures = 0
        self.restarts = 0
        self.last_load_duration: float | None = None
        self.last_error: BaseException | None = None
        self._load_started = 0.0
        self._unload_started = 0.0

    def start_load(self) -> None:
        """Mark the start of a load."""
        self._load_started = time.perf_counter()

    def load_elapsed(self) -> float:
        """Seconds since the current load started."""
        return time.perf_counter() - self._load_started

    def loaded(self) -> float:
        """Count a completed load; return its duration."""
        duration = self.load_elapsed()
        self.loads += 1
        self.last_load_duration = duration
        return duration

    def failed(self, error: BaseException) -> float:
        """Count a failed load; return the time until it failed."""
        self.failures += 1
        self.last_error = error
        return self.load_elapsed()

    def start_unload(self) -> None:
        """Mark the start of an unload."""
        self._unload_started = time.perf_counter()

    def unload_elapsed(self) -> float:
        """Seconds since the current unload started."""
        return time.perf_counter() - self._unload_started

    def snapshot(self) -> FiberStats:
        """The counters as an immutable :class:`FiberStats`."""
        return FiberStats(
            self.loads, self.failures, self.restarts, self.last_load_duration, self.last_error
        )


def publish(
    fiber: FiberCore,
    kind: LifecycleKind,
    duration: float | None = None,
    error: BaseException | None = None,
    *,
    state: FiberState | None = None,
) -> None:
    """Emit a :class:`LifecycleEvent` for ``fiber`` if anyone listens to :data:`TELEMETRY`.

    ``state`` overrides the fiber's current state in the event. Builds nothing
    without listeners. Listener errors are reported, never raised. Example (a
    fiber that finished loading)::

        publish(fiber, "load", duration)

    Internal — not part of the public API.
    """
    ctx = fiber.ctx
    events: EventsService = lookup(ctx, "events")
    if TELEMETRY.name not in events.hooks:
        return
    if state is None:
        state = fiber.state
    event = LifecycleEvent(kind, fiber.name, fiber.uid, state, duration, error, time.time())
    emit_reported(ctx, TELEMETRY.name, event)
