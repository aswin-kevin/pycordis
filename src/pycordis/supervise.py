"""Supervision: restart policies for failed plugins, load and unload timeouts (spec §3.6).

A plugin opts into supervision with a :class:`RestartPolicy` (plugin metadata
``policy``, ``ctx.plugin(..., policy=...)`` or the loader entry key
``restart``). With ``mode="on-failure"``, a fiber that reaches ``FAILED`` is
loaded again after an exponential :class:`Backoff`; after ``max_restarts``
restarts within ``window`` seconds it stays ``FAILED`` and the supervisor logs
``giving up``. The backoff wait is a task owned by the parent fiber, so
disposing the parent (or the root) cancels it at once. A user ``update()``
of the fiber is a new configuration: it cancels a pending restart and resets
the restart budget.

``load_timeout`` bounds an async plugin body: when it expires the body is
cancelled and the fiber fails with a :class:`LoadTimeoutError` (a
``TimeoutError`` subclass, which counts as a failure for supervision), also
when the body swallows the cancellation and returns late; the unload that
follows is bounded by ``load_timeout`` too unless a ``dispose_timeout`` is
set, so effects the body left in flight cannot delay ``FAILED``.
``dispose_timeout`` bounds an unload:
background tasks still running and disposers still pending when it expires
are abandoned (cancelled) with an error log naming them, and the unload
completes. Only awaiting work can be abandoned: a disposer that blocks the
event loop synchronously cannot be bounded.

Defaults keep v1 behaviour: policy ``never``, no timeouts.
"""

from __future__ import annotations

import asyncio
import math
import random
import warnings
from collections import deque
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, final

from pycordis._guards import is_mapping
from pycordis._utils import report_unhandled
from pycordis.errors import CordisError
from pycordis.telemetry import publish

if TYPE_CHECKING:
    from pycordis.fiber import Fiber

__all__ = ["Backoff", "LoadTimeoutError", "RestartMode", "RestartPolicy"]

type RestartMode = Literal["never", "on-failure"]
"""When a failed plugin is restarted: ``"never"`` (default) or ``"on-failure"``."""

_MODES: tuple[RestartMode, ...] = ("never", "on-failure")
_MIN_WINDOW = 0.01
"""Below this many seconds a restart window cannot hold a restart (warned about)."""


# -- validation helpers ----------------------------------------------------------------


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and not math.isnan(value)


def _check_number(
    value: object, name: str, *, minimum: float, exclusive: bool = False, maximum: float = math.inf
) -> float:
    """Return ``value`` as a float, or raise ``ValueError`` naming ``name`` and the valid range."""
    if _is_number(value):
        number = float(value)
        if (number > minimum if exclusive else number >= minimum) and number <= maximum:
            return number
    if exclusive and minimum == 0:
        expected = "a positive number"
    elif maximum != math.inf:
        expected = f"a number between {minimum:g} and {maximum:g}"
    elif minimum == 0:
        expected = "a non-negative number"
    else:
        expected = f"a number >= {minimum:g}"
    raise ValueError(f'"{name}" must be {expected}')


def _check_count(value: object, name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError(f'"{name}" must be a non-negative integer')


def _check_mode(value: object, name: str) -> RestartMode:
    for mode in _MODES:
        if value == mode:
            return mode
    raise ValueError(f'"{name}" must be "never" or "on-failure"')


def _check_backoff(value: object, name: str) -> None:
    if not isinstance(value, Backoff):
        raise ValueError(f'"{name}" must be a Backoff')


def check_timeout(value: object, name: str) -> float | None:
    """Validate an optional timeout in seconds (``None``, or a positive number).

    Example: ``check_timeout(5, "load_timeout") == 5.0``; ``check_timeout(0, ...)``
    raises ``ValueError('"load_timeout" must be a positive number')``.
    """
    if value is None:
        return None
    return _check_number(value, name, minimum=0, exclusive=True)


def _table(raw: object, name: str, keys: Iterable[str]) -> Mapping[str, object]:
    if not is_mapping(raw):
        raise ValueError(f'"{name}" must be a table')
    allowed = set(keys)
    table: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(f'unknown key "{name}.{key}"')
        table[key] = value
    return table


# -- policy values ---------------------------------------------------------------------


@final
@dataclass(frozen=True)
class Backoff:
    """Exponential backoff between restarts.

    The delay before restart ``n`` (1-based) is ``initial * factor ** (n - 1)``
    capped at ``max``, then scaled by a random factor in ``[1 - jitter, 1 +
    jitter]`` so that many failing plugins do not restart in lockstep.
    Example::

        Backoff(initial=1.0, factor=2.0, max=10.0, jitter=0.0)  # 1, 2, 4, 8, 10, 10 ...
    """

    initial: float = 0.5
    """Delay before the first restart, in seconds (>= 0)."""
    factor: float = 2.0
    """Growth of the delay per restart (>= 1)."""
    max: float = 30.0
    """Upper bound of the delay before jitter, in seconds (>= 0)."""
    jitter: float = 0.1
    """Relative random spread of each delay (0 to 1)."""

    def __post_init__(self) -> None:
        _check_number(self.initial, "initial", minimum=0)
        _check_number(self.factor, "factor", minimum=1)
        _check_number(self.max, "max", minimum=0)
        _check_number(self.jitter, "jitter", minimum=0, maximum=1)

    def delay(self, attempt: int, rng: random.Random) -> float:
        """The delay in seconds before restart ``attempt`` (1-based), jittered with ``rng``.

        Pure apart from ``rng``: a seeded :class:`random.Random` gives a
        reproducible sequence. Example::

            Backoff(jitter=0.0).delay(3, random.Random()) == 2.0
        """
        base = min(self.initial, self.max)
        for _ in range(attempt - 1):
            if base >= self.max:
                break
            base = min(self.max, base * self.factor)
        if not self.jitter:
            return base
        return base * (1 + rng.uniform(-self.jitter, self.jitter))

    @classmethod
    def from_mapping(cls, raw: object, name: str = "backoff") -> Backoff:
        """Build a backoff from a config table, naming bad keys as ``<name>.<key>``.

        Example: ``Backoff.from_mapping({"initial": 1, "max": 5})``. Raises
        ``ValueError`` for a non-table, unknown keys or invalid values.
        """
        table = _table(raw, name, (f.name for f in fields(cls)))
        values = {
            "initial": _check_number(table.get("initial", 0.5), f"{name}.initial", minimum=0),
            "factor": _check_number(table.get("factor", 2.0), f"{name}.factor", minimum=1),
            "max": _check_number(table.get("max", 30.0), f"{name}.max", minimum=0),
            "jitter": _check_number(
                table.get("jitter", 0.1), f"{name}.jitter", minimum=0, maximum=1
            ),
        }
        return cls(**values)


@final
@dataclass(frozen=True)
class RestartPolicy:
    """When and how often a failed plugin is restarted.

    ``mode="never"`` (default) keeps a failed fiber ``FAILED`` (v1 behaviour).
    ``mode="on-failure"`` restarts it after :attr:`backoff`, at most
    :attr:`max_restarts` times within any :attr:`window` seconds. Example::

        ctx.plugin(worker, policy=RestartPolicy(mode="on-failure", max_restarts=3))
    """

    mode: RestartMode = "never"
    """``"never"`` or ``"on-failure"``."""
    max_restarts: int = 5
    """Restarts allowed within :attr:`window` before giving up (>= 0)."""
    window: float = 60.0
    """Length of the sliding window counting restarts, in seconds (> 0).

    It must exceed a restart's latency (backoff delay plus load time), or
    restarts leave the window before they count; below 10 ms a
    ``RuntimeWarning`` is issued."""
    backoff: Backoff = field(default_factory=Backoff)
    """Delay between a failure and its restart."""

    def __post_init__(self) -> None:
        _check_mode(self.mode, "mode")
        _check_count(self.max_restarts, "max_restarts")
        _check_number(self.window, "window", minimum=0, exclusive=True)
        _check_backoff(self.backoff, "backoff")
        if self.window < _MIN_WINDOW:
            warnings.warn(
                f"RestartPolicy window {self.window:g}s is shorter than a restart takes; "
                "restarts may never count against max_restarts",
                RuntimeWarning,
                stacklevel=3,
            )

    @classmethod
    def from_mapping(cls, raw: object, name: str = "restart") -> RestartPolicy:
        """Build a policy from a config table such as the loader's ``restart`` entry key.

        Bad keys are named ``<name>.<key>``. Example::

            RestartPolicy.from_mapping({"mode": "on-failure", "backoff": {"initial": 1}})

        Raises ``ValueError`` for a non-table, unknown keys or invalid values.
        """
        table = _table(raw, name, (f.name for f in fields(cls)))
        backoff = table.get("backoff")
        return cls(
            mode=_check_mode(table.get("mode", "never"), f"{name}.mode"),
            max_restarts=_check_count(table.get("max_restarts", 5), f"{name}.max_restarts"),
            window=_check_number(
                table.get("window", 60.0), f"{name}.window", minimum=0, exclusive=True
            ),
            backoff=Backoff()
            if backoff is None
            else Backoff.from_mapping(backoff, f"{name}.backoff"),
        )


@final
@dataclass(frozen=True)
class Supervision:
    """The supervision settings of one fiber: restart policy and timeouts.

    Example: ``Supervision(load_timeout=5.0)``.

    Internal — pass ``policy`` / ``load_timeout`` / ``dispose_timeout`` to
    ``ctx.plugin()`` or set them as plugin metadata.
    """

    policy: RestartPolicy = field(default_factory=RestartPolicy)
    load_timeout: float | None = None
    dispose_timeout: float | None = None


UNSUPERVISED = Supervision()
"""No restart policy and no timeouts (v1 behaviour). Internal."""


# -- restarts --------------------------------------------------------------------------

_jitter_rng = random.Random()


@final
@dataclass(frozen=True)
class PendingRestart:
    """A restart a :class:`Supervisor` has scheduled: waiting out its backoff.

    Example: ``PendingRestart(attempt=2, max_restarts=5, delay=1.0)`` is the
    second restart of five allowed, due one second after the failure.

    Internal — read through ``fiber.supervisor.pending`` (diagnostics).
    """

    attempt: int
    """1-based number of this restart within the policy window."""
    max_restarts: int
    """Restarts the policy allows within its window."""
    delay: float
    """The backoff waited before this restart, in seconds."""


@final
class Supervisor:
    """Restarts one fiber after it fails, according to its :class:`RestartPolicy`.

    The fiber calls :meth:`on_failed` each time it reaches ``FAILED``, and
    :meth:`cancel` when it is disposed. Example (what ``Fiber`` does)::

        fiber.supervisor = Supervisor(fiber, policy)

    Internal — set a policy through ``ctx.plugin(..., policy=...)``.
    """

    __slots__ = ("_fiber", "_pending", "_policy", "_restarts", "_rng", "_waiting")

    def __init__(
        self, fiber: Fiber, policy: RestartPolicy, rng: random.Random | None = None
    ) -> None:
        self._fiber = fiber
        self._policy = policy
        self._rng = _jitter_rng if rng is None else rng
        self._restarts: deque[float] = deque()
        self._waiting: asyncio.Task[None] | None = None
        self._pending: PendingRestart | None = None

    @property
    def waiting(self) -> bool:
        """Whether a restart is scheduled and its backoff has not elapsed yet."""
        return self._waiting is not None and not self._waiting.done()

    @property
    def pending(self) -> PendingRestart | None:
        """The restart scheduled now, while its backoff has not elapsed (else ``None``)."""
        return self._pending if self.waiting else None

    def on_failed(self) -> None:
        """Schedule a restart of the failed fiber, or give up when the budget is spent."""
        fiber = self._fiber
        policy = self._policy
        now = asyncio.get_running_loop().time()
        while self._restarts and now - self._restarts[0] >= policy.window:
            self._restarts.popleft()
        attempt = len(self._restarts) + 1
        if attempt > policy.max_restarts:
            _log(
                fiber,
                "error",
                f"giving up on {fiber.name} after {len(self._restarts)} restarts "
                f"in {policy.window:g}s",
            )
            publish(fiber, "give-up", error=fiber.error)
            return
        delay = policy.backoff.delay(attempt, self._rng)
        self.cancel()
        try:
            # owned by the parent: its unload (or the root's dispose) cancels the wait
            self._waiting = fiber.parent.fiber.spawn(
                self._restart_after(delay), name=f"pycordis:{fiber.name}:backoff"
            )
        except CordisError:
            return  # the parent is unloading: this fiber is disposed with it
        self._restarts.append(now)
        self._pending = PendingRestart(attempt, policy.max_restarts, delay)
        fiber.recorder.restarts += 1
        _log(
            fiber,
            "warn",
            f"restarting {fiber.name} in {delay:.2f}s (attempt {attempt}/{policy.max_restarts})",
        )
        publish(fiber, "restart", delay, fiber.error)

    def reset(self) -> None:
        """Cancel a scheduled restart and forget past restarts (a new configuration)."""
        self.cancel()
        self._restarts.clear()

    def cancel(self) -> None:
        """Cancel a scheduled restart, if any."""
        waiting = self._waiting
        self._waiting = None
        self._pending = None
        if waiting is not None:
            waiting.cancel()

    async def _restart_after(self, delay: float) -> None:
        await asyncio.sleep(delay)
        self._waiting = None
        self._pending = None
        self._fiber.recover()


def _log(fiber: Fiber, level: Literal["warn", "error"], text: str) -> None:
    # supervision must not fail the unload that reported the failure
    try:
        logger = fiber.ctx.logger("supervisor")
        if level == "warn":
            logger.warn(text)
        else:
            logger.error(text)
    except Exception as unavailable:  # the logger service is unavailable
        report_unhandled(f"pycordis: supervisor: {text}", unavailable)


# -- timeouts --------------------------------------------------------------------------


async def await_load(
    setup: asyncio.Future[None],
    timeout: float | None,
    name: str,
    annotate: Callable[[BaseException], object],
) -> None:
    """Await an async plugin body; fail with :class:`LoadTimeoutError` after ``timeout`` s.

    On expiry the body is cancelled and the :class:`LoadTimeoutError` (naming
    plugin ``name``) is passed to ``annotate`` before it is raised. A body that
    swallows the cancellation and returns anyway still gets the error: the
    deadline passed, so the load failed (the fiber then unloads what the late
    body set up). A ``TimeoutError`` raised by the body itself passes through
    unchanged. Example (a fiber's load)::

        await await_load(setup, fiber.load_timeout, fiber.name, annotate)

    Internal — not part of the public API.
    """
    if timeout is None:
        await setup
        return
    scope = asyncio.timeout(timeout)
    try:
        async with scope:
            await setup
    except TimeoutError:
        if not scope.expired():
            raise
    else:
        if not scope.expired():
            return
        # expired, yet the body swallowed its cancellation and returned late
    error = LoadTimeoutError(f"plugin <{name}> did not load within {timeout:g}s")
    annotate(error)
    raise error from None


class LoadTimeoutError(TimeoutError):
    """Raised (and reported) when a plugin body exceeds its ``load_timeout``.

    A :class:`TimeoutError` subclass, so ``except TimeoutError`` still catches
    it; catch it specifically to tell a load timeout apart from a timeout raised
    by the plugin body itself. Example::

        try:
            await ctx.plugin(worker, load_timeout=5)
        except LoadTimeoutError:
            ...  # the fiber is FAILED; its body was cancelled
    """


def abandoned(what: str, owner: str, timeout: float) -> TimeoutError:
    """The error reported for unload work abandoned after ``timeout`` seconds.

    Example: ``abandoned("task poll", "worker", 10.0)`` reads ``task poll in
    <worker> did not finish within 10s; abandoned``.

    Internal — not part of the public API.
    """
    return TimeoutError(f"{what} in <{owner}> did not finish within {timeout:g}s; abandoned")


async def wait_until[F: asyncio.Future[Any]](  # futures of any result type
    tasks: Iterable[F], deadline: float
) -> set[F]:
    """Wait for ``tasks`` until ``deadline`` (loop time); return those still pending.

    Example: ``stuck = await wait_until(tasks, loop.time() + 5.0)``.

    Internal — not part of the public API.
    """
    waiting = set(tasks)
    if not waiting:
        return set()
    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
    _, pending = await asyncio.wait(waiting, timeout=remaining)
    return pending


_abandoned: set[asyncio.Task[Any]] = set()  # Task is invariant: any result type
"""Cancelled jobs that have not finished yet (kept referenced until they do)."""


async def run_within(
    # Any: asyncio's own coroutine typing (yield and send types)
    jobs: Sequence[tuple[str, Coroutine[Any, Any, None]]],
    deadline: float,
) -> list[str]:
    """Run labelled coroutines concurrently until ``deadline``; return the labels abandoned.

    Jobs still running at the deadline (loop time) are cancelled and abandoned:
    they are not awaited further, but stay referenced until they finish.
    Cancelling the caller cancels every job. Example::

        stuck = await run_within([("close db", close_db())], loop.time() + 5.0)

    Internal — not part of the public API.
    """
    tasks = {asyncio.ensure_future(coro): label for label, coro in jobs}
    try:
        pending = await wait_until(tasks, deadline)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        raise
    for task in pending:
        task.cancel()
        _abandoned.add(task)
        task.add_done_callback(_abandoned.discard)
    return [label for task, label in tasks.items() if task in pending]
