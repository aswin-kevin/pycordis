"""A fiber's identity, lifecycle state and the effects and tasks it owns.

:class:`FiberCore` is the base layer of :class:`~pycordis.fiber.Fiber`: it
registers effects (``ctx.effect``), owns spawned tasks (``ctx.spawn``) and
fiber-local ``internal/update`` hooks, and reports the failures no caller will
observe. The load/unload state machine lives in
:mod:`pycordis.fiber_lifecycle`; construction and config updates in
:mod:`pycordis.fiber`.

Internal — import :class:`~pycordis.fiber.Fiber` and the names below from
:mod:`pycordis.fiber`, which re-exports them.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING, Any

from pycordis._utils import DisposableList, report_unhandled, symbols
from pycordis.current import call_in, task_context
from pycordis.effect import Disposable, EffectDisposer, EffectMeta, EffectResult
from pycordis.errors import CordisError

if TYPE_CHECKING:
    from pycordis.context import Context
    from pycordis.registry import Runtime


class FiberState(IntEnum):
    """Lifecycle state of a fiber.

    ``PENDING`` waits for required services; ``LOADING`` runs the plugin body;
    ``ACTIVE`` is loaded; ``FAILED`` means the body or its config raised;
    ``DISPOSED`` was removed and cannot restart; ``UNLOADING`` runs disposers.

    Example: ``if fiber.state == FiberState.ACTIVE: ...``.
    """

    PENDING = 0
    LOADING = 1
    ACTIVE = 2
    FAILED = 3
    DISPOSED = 4
    UNLOADING = 5


_REPORTED_KEY = "__cordis_reported__"


def mark_reported(error: BaseException) -> None:
    """Mark ``error`` as already reported (kept in the instance ``__dict__``)."""
    error.__dict__[_REPORTED_KEY] = True


def is_reported(error: BaseException | None) -> bool:
    """Whether ``error`` was already reported through :func:`report_error`."""
    return error is not None and error.__dict__.get(_REPORTED_KEY, False) is True


def report_error(source: FiberCore | Context, error: BaseException) -> None:
    """Report an error no caller will observe (cordis ``ctx.logger.error``).

    Routed to ``ctx.logger.error`` of the fiber's context; if the logger cannot be reached
    (the logger service is unavailable) it goes to the running loop's exception
    handler instead. An error object is reported at most once.

    Internal — not part of the public API.
    """
    if is_reported(error):
        return
    mark_reported(error)
    fiber: FiberCore = source if isinstance(source, FiberCore) else source.fiber
    try:
        logger = fiber.ctx.logger()
    except Exception as unavailable:
        # the logger service itself is unavailable: report both, without raising
        report_unhandled(
            f"pycordis: unhandled error in <{fiber.name}>",
            error,
            fiber=fiber,
            logger_error=unavailable,
        )
        return
    logger.error(error)


def emit_reported(context: Context, *args: object) -> None:
    """Emit like ``context.emit(*args)``, reporting listener errors instead of raising.

    Each listener runs with its owner current; an error it raises is reported
    through :func:`report_error` on ``context`` and the remaining listeners
    still run. Internal — not part of the public API.
    """
    rest = list(args)
    try:
        callbacks = context.events.dispatch_owned("emit", rest)
    except Exception as error:
        report_error(context, error)
        return
    for owner, callback in callbacks:
        try:
            returned = call_in(owner, callback, *rest)
        except Exception as error:
            report_error(context, error)
            continue
        if inspect.iscoroutine(returned):
            # owned by the listener's fiber (as in ``emit``), never by the
            # emitting fiber, whose unload would cancel it before it ran
            owner.fiber.spawn_owned(returned, context=owner)


def effect_label(dispose: Disposable) -> str:
    """The label of the effect behind ``dispose``, else its ``repr`` (a raw disposer callable).

    Example: ``effect_label(ctx.effect(start, "poller")) == "poller"``.

    Internal — not part of the public API.
    """
    meta = getattr(dispose, symbols.effect, None)
    return meta.label if isinstance(meta, EffectMeta) else repr(dispose)


_GENERATOR_NONE = (
    "generator effect yielded None; yield a disposer callable "
    "(cleanup code after yield runs immediately)"
)


def warn_generator_none(fiber: FiberCore) -> None:
    """Warn that a generator effect of ``fiber`` yielded ``None`` (fixture-style misuse).

    Example: ``EffectDisposer(..., on_none=partial(warn_generator_none, fiber))``.

    Internal — not part of the public API.
    """
    try:
        fiber.ctx.logger().warn(_GENERATOR_NONE)
    except Exception as unavailable:  # the logger service is unavailable
        report_unhandled(f"pycordis: <{fiber.name}>: {_GENERATOR_NONE}", unavailable)


class FiberCore:
    """Identity, state and owned effects of a fiber (the base of :class:`~pycordis.fiber.Fiber`).

    Example (through a fiber created by ``ctx.plugin()``)::

        task = fiber.spawn(poll())  # cancelled when the fiber unloads

    Internal — construct fibers through ``ctx.plugin()``.
    """

    def __init__(self, parent: Context, runtime: Runtime | None) -> None:
        self.parent = parent
        """The context this fiber was created from."""
        self.runtime: Runtime | None = runtime
        """The shared plugin runtime, or ``None`` for the root fiber."""
        self.state = FiberState.PENDING
        """Current lifecycle state."""
        self._disposables: DisposableList[Disposable] = DisposableList()
        self._update_hooks: DisposableList[Callable[..., object]] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()  # Task is invariant: any result type
        self.ctx: Context
        """The context this fiber's plugin runs in (extends the parent context)."""
        self.uid: int | None
        """Unique id within the registry; 0 for the root fiber, ``None`` once disposed."""

    @property
    def name(self) -> str:
        """The plugin's display name, inherited from the nearest named ancestor, else ``root``."""
        fiber: FiberCore = self
        while True:
            name = getattr(fiber.runtime, "name", None)
            if name:
                return str(name)
            parent_fiber: FiberCore = fiber.parent.fiber
            if parent_fiber is fiber:
                return "root"
            fiber = parent_fiber

    def _assert_effectable(self) -> None:
        """Raise ``CordisError("INACTIVE_EFFECT")`` unless new effects may be registered."""
        self.assert_active()
        if self.state == FiberState.UNLOADING:
            raise CordisError("INACTIVE_EFFECT")

    def assert_active(self) -> None:
        """Raise ``CordisError("INACTIVE_EFFECT")`` if the fiber has been disposed."""
        if self.uid is not None:
            return
        raise CordisError("INACTIVE_EFFECT")

    def effect(
        self,
        execute: Callable[[], EffectResult],
        label: str = "anonymous",
        *,
        context: Context | None = None,
    ) -> EffectDisposer:
        """Register a cleanup-aware effect on this fiber.

        ``execute`` runs immediately; the disposers it produces run in reverse order
        when the returned disposer is called or the fiber unloads, whichever comes
        first. Accepted results: ``None``; a (sync or async) disposer callable; an
        awaitable resolving to one; a generator or async generator yielding them; a
        context manager or async context manager (entered now, exited on dispose).

        Raises ``CordisError("INACTIVE_EFFECT")`` on a disposed or unloading fiber and
        ``TypeError("Invalid effect")`` for any other result.

        Setup and disposers run with ``context`` current
        (:func:`~pycordis.current_context`), by default this fiber's context;
        ``ctx.effect(...)`` passes ``ctx`` itself (the use site, e.g. an
        ``isolate()``-d view), while the effect stays owned by this fiber.
        """
        self._assert_effectable()
        return EffectDisposer(
            execute,
            label,
            context=self.ctx if context is None else context,
            owner_name=lambda: self.name,
            owner=self._disposables,
            report=partial(report_error, self),
            on_none=partial(warn_generator_none, self),
        )

    def spawn[T](
        self,
        coro: Coroutine[Any, Any, T],  # asyncio's coroutine typing
        *,
        name: str | None = None,
        context: Context | None = None,
    ) -> asyncio.Task[T]:
        """Start ``coro`` as an :class:`asyncio.Task` owned by this fiber (``ctx.spawn``).

        The task is cancelled, and awaited, when the fiber unloads, before the
        fiber's disposers run. An exception other than ``CancelledError`` is
        reported through the logger (and still raised to whoever awaits the task).
        The task runs with ``context`` current (:func:`~pycordis.current_context`),
        by default this fiber's context; ``ctx.spawn(...)`` passes ``ctx`` itself
        (the use site), while the task stays owned by this fiber.

        Raises ``CordisError("INACTIVE_EFFECT")`` (closing ``coro``) when the fiber
        is disposed or unloading.
        """
        try:
            self._assert_effectable()
        except CordisError:
            coro.close()
            raise
        return self.spawn_owned(coro, name, context)

    def spawn_owned[T](
        self,
        coro: Coroutine[Any, Any, T],  # asyncio's coroutine typing
        name: str | None = None,
        context: Context | None = None,
    ) -> asyncio.Task[T]:
        """Start ``coro`` as a task owned by this fiber, without the state check of :meth:`spawn`.

        Used for coroutines returned by listeners and timer callbacks, which must be
        owned by their fiber even while it unloads. A failure is reported. The task
        runs with ``context`` current (the use site that registered the listener or
        timer), by default this fiber's context.

        Internal — not part of the public API.
        """
        owner = self.ctx if context is None else context
        task = asyncio.get_running_loop().create_task(coro, name=name, context=task_context(owner))
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def owned_tasks(
        self,
    ) -> list[asyncio.Task[Any]]:  # Task is invariant: any result type
        """The tasks started by :meth:`spawn` / :meth:`spawn_owned` that are still running.

        Internal — not part of the public API.
        """
        return list(self._tasks)

    def disposables(self) -> list[Disposable]:
        """The disposers this fiber currently owns, in registration order.

        Internal — not part of the public API.
        """
        return list(self._disposables)

    def release_disposables(self) -> None:
        """Detach every disposer from this fiber without running it.

        The root context calls this once its built-in services are installed, so
        their listeners are not torn down by a root restart.

        Internal — not part of the public API.
        """
        self._disposables.clear()

    def add_update_hook(
        self, listener: Callable[..., object], *, prepend: bool = False
    ) -> EffectDisposer:
        """Register a fiber-local ``internal/update`` listener, owned as an effect.

        Internal — not part of the public API (``ctx.on("internal/update", ...)``
        routes here).
        """
        hooks = self._update_hooks
        if hooks is None:
            hooks = self._update_hooks = DisposableList()
        method = hooks.unshift if prepend else hooks.push
        return self.effect(lambda: method(listener), 'ctx.on("internal/update")')

    def update_hooks(self) -> list[Callable[..., object]]:
        """The fiber-local ``internal/update`` listeners, in dispatch order.

        Internal — not part of the public API.
        """
        return list(self._update_hooks or ())

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:  # Task is invariant: any result type
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            report_error(self, error)

    def get_effects(self) -> list[EffectMeta]:
        """Return the metadata tree of every live effect registered on this fiber."""
        metas: list[EffectMeta] = []
        for dispose in self._disposables:
            meta = getattr(dispose, symbols.effect, None)
            if isinstance(meta, EffectMeta):
                metas.append(meta)
        return metas
