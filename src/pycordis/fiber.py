"""Fiber: the runtime instance owning a context's effects (port of cordis ``fiber.ts``).

The root fiber of a context is always active; a plugin fiber (created by
``ctx.plugin()``) loads when its required services are available and moves
through :class:`FiberState` driven by its dependency epoch and ``inertia`` task.

:class:`Fiber` is assembled from layers: :class:`~pycordis.fiber_core.FiberCore`
(identity, state, owned effects and tasks, error reporting),
:class:`~pycordis.fiber_lifecycle.FiberLifecycle` (the load/unload state
machine) and this module (construction, restart and config updates). Every
name of those layers used elsewhere is re-exported here.
"""

from __future__ import annotations

import asyncio
import inspect
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from functools import partial
from typing import TYPE_CHECKING, Any

from pycordis._context_internals import events_of, fiber_of, intercept_map, lookup
from pycordis._utils import add_note_once, done_future, report_failure, symbols
from pycordis.diagnose import Diagnosis, diagnose_fiber
from pycordis.effect import EffectResult, EffectRunner
from pycordis.fiber_core import (
    FiberState,
    emit_reported,
    is_reported,
    mark_reported,
    report_error,
    warn_generator_none,
)
from pycordis.fiber_lifecycle import INACTIVE, FiberLifecycle
from pycordis.supervise import UNSUPERVISED, RestartPolicy, Supervision, Supervisor
from pycordis.volatile import diff_only_volatile, volatile_pairs

if TYPE_CHECKING:
    from pycordis.context import Context
    from pycordis.registry import RegistryService, Runtime

__all__ = [
    "INACTIVE",
    "Fiber",
    "FiberState",
    "emit_plugin_disposed",
    "emit_reported",
    "is_reported",
    "mark_reported",
    "report_error",
]

_warned_runtimes: weakref.WeakSet[Runtime] = weakref.WeakSet()
"""Plugins already warned by :func:`~pycordis.fiber_core.warn_generator_none` (once per plugin)."""


def _noop() -> None:
    return None


class Fiber(FiberLifecycle):
    """Runtime instance of one plugin application (the root fiber for a root context).

    A fiber owns the effects registered on its context and tears them down, in
    reverse order, when it unloads.

    Example: ``fiber = await ctx.plugin(server)``; ``await fiber.dispose()``.
    """

    def __init__(
        self,
        parent: Context,
        config: Any = None,
        inject: dict[str, object] | None = None,
        runtime: Runtime | None = None,
        *,
        supervision: Supervision = UNSUPERVISED,
    ) -> None:
        super().__init__(parent, inject, runtime)
        self.policy: RestartPolicy = supervision.policy
        """How this fiber is restarted after a failure (see :mod:`pycordis.supervise`)."""
        self.load_timeout = supervision.load_timeout
        self.dispose_timeout = supervision.dispose_timeout
        if supervision.policy.mode == "on-failure":
            self.supervisor = Supervisor(self, supervision.policy)
        # Any: a fiber runs plugins of every config type (Fiber is not generic over it)
        self.config: Any = config
        """Raw plugin input retained for the next activation."""
        self.notes: list[str] = []
        """Extra notes added to errors raised while running this fiber's plugin (e.g. by
        the loader: ``in loader entry "<id>"``), after ``in plugin <name>``."""
        self._config_revision = 0
        self._effective_config: Any = None
        self._dispose_impl: Callable[[], asyncio.Future[None]]
        if runtime is None:
            self.uid = 0
            self.ctx = parent
            self.state = FiberState.ACTIVE
            self.committed = {}
            self._runner = EffectRunner[str]("0", _noop, self._disposables.push, _noop_annotate)
            # cordis root: `dispose = () => this.restart()`
            self._dispose_impl = lambda: asyncio.ensure_future(self.restart())
            return
        self._init_plugin(parent, runtime)

    def _init_plugin(self, parent: Context, runtime: Runtime) -> None:
        registry: RegistryService = lookup(parent, "registry")
        self.uid = registry.counter
        self.cycles = registry.cycles
        self.ctx = parent.extend(**self._plugin_meta(parent))
        self._runner = EffectRunner[str](
            INACTIVE,
            partial(self._execute_plugin, runtime),
            self._disposables.push,
            self._annotate_plugin_error,
            partial(self._warn_generator_none_once, runtime),
        )
        self._dispose_impl = fiber_of(parent).effect(
            partial(self._register_in_runtime, runtime), "ctx.plugin()"
        )
        self._publish()
        # Keep the initial notification's historical PENDING view. The loader
        # may also extend `inject` in that notification, so resolve dependencies
        # only after publication. A reentrant parent unload makes the child
        # disposer responsible for draining any PENDING effects instead. (Read
        # uid afresh: an observer may have disposed this fiber, clearing it.)
        if self._has_uid() and fiber_of(parent).state != FiberState.UNLOADING:
            for name in self.inject:
                self.check_impl(name)
            self.refresh_epoch()
            # a new fiber starts PENDING without a status change to observe
            if self.state == FiberState.PENDING:
                registry.cycles.entered_pending(self)

    def _plugin_meta(self, parent: Context) -> dict[str, object]:
        """The own attributes of the plugin's context: the fiber and its intercept configs."""
        meta: dict[str, object] = {"fiber": self}
        if self.inject:
            intercept = intercept_map(parent)
            for name, config in self.inject.items():
                if config is None:
                    continue
                intercept = intercept.child(name, config)
            meta[symbols.intercept] = intercept
        return meta

    def _execute_plugin(self, runtime: Runtime) -> EffectResult:
        """The plugin body as an effect: resolve the config, then call or construct the plugin."""
        config = self.resolve_config(self.config)
        self._effective_config = config
        callback = runtime.callback
        if not _is_constructor(callback):
            return callback(self.ctx, config)
        instance = callback(self.ctx, config)
        # cordis class plugins: per-instance init hooks, then an optional init method
        for hook in getattr(instance, symbols.init_hooks, None) or ():
            hook()
        init = getattr(instance, symbols.init, None)
        return init() if init is not None else None

    def _annotate_plugin_error(self, error: BaseException) -> None:
        for note in (f"in plugin <{self.name}>", *self.notes):
            add_note_once(error, note)

    def _warn_generator_none_once(self, runtime: Runtime) -> None:
        if runtime not in _warned_runtimes:
            _warned_runtimes.add(runtime)
            warn_generator_none(self)

    # Any: asyncio's own coroutine typing (yield and send types)
    def _register_in_runtime(self, runtime: Runtime) -> Callable[[], Coroutine[Any, Any, None]]:
        """The ``ctx.plugin()`` effect of the parent: add this fiber to its runtime.

        Its disposer removes the fiber again and unloads it.
        """
        remove = runtime.fibers.push(self)
        return partial(self._dispose_plugin, runtime, remove)

    # Any: asyncio's own coroutine typing (yield and send types)
    def _dispose_plugin(
        self, runtime: Runtime, remove: Callable[[], bool]
    ) -> Coroutine[Any, Any, None]:
        """Unregister and unload this fiber.

        cordis runs an async disposer synchronously up to its first await; this
        keeps that prefix synchronous so a caller observes the fiber removed.
        """
        self.uid = None
        if self.supervisor is not None:
            self.supervisor.cancel()
        emit_plugin_disposed(self.ctx, self)
        registry = self.ctx.registry
        if registry.has(runtime.callback):
            remove()
            if not len(runtime.fibers):
                registry.delete(runtime.callback)
        self._set_epoch(INACTIVE)
        # A PENDING fiber can already own effects registered by an
        # internal/plugin observer. Its epoch is still INACTIVE, so
        # _set_epoch() has no transition to drive; explicitly unload that
        # pre-activation work before reporting disposal complete.
        if self.inertia is None:

            def transition() -> FiberState:
                self.inertia = self._start_unload()
                return FiberState.UNLOADING

            self._update_state(transition)
        return self._settle_inertia()

    def _publish(self) -> None:
        """Announce the new fiber (``internal/plugin``); dispose it if that fails synchronously."""
        try:
            # Publish only after the parent owns a fully assigned disposer. A
            # synchronous observer may dispose either this fiber or its parent.
            events_of(self.ctx).emit("internal/plugin", self)
        except BaseException:
            # Publication failed synchronously. The disposer removes the child
            # from both the parent and runtime before control escapes.
            report_failure(self._dispose_impl(), partial(report_error, self))
            raise

    def _has_uid(self) -> bool:
        # a method call, so type narrowing of ``self.uid`` does not carry over
        return self.uid is not None

    def dispose(self) -> asyncio.Future[None]:
        """Dispose this fiber; for the root fiber, unload every effect and reactivate.

        Example: ``await fiber.dispose()``.
        """
        return self._dispose_impl()

    def diagnose(self) -> Diagnosis:
        """Explain why this fiber, and each fiber below it, is not ``ACTIVE``.

        Returns a :class:`~pycordis.diagnose.Diagnosis` tree (see
        :mod:`pycordis.diagnose`); print it with
        :func:`~pycordis.diagnose.render_diagnosis`. Example::

            print(render_diagnosis(fiber.diagnose()))
        """
        return diagnose_fiber(self)

    def resolve_config(self, config: object) -> Any:  # validated by a schema only the plugin knows
        """Evaluate and validate a candidate config through ``internal/config`` and the schema.

        Does not publish the result or replace the raw config. Raises
        :class:`~pycordis.errors.ValidationError` when the schema rejects it.

        Example: ``fiber.resolve_config({"port": 80})`` (a validated config).
        """
        fiber = fiber_of(self.ctx)
        return events_of(fiber.ctx).waterfall("internal/config", fiber, lambda: config)

    @property
    def raw_config(self) -> Any:
        """Raw input retained for the next activation, neither evaluated nor validated.

        Example: ``fiber.update({**fiber.raw_config, "port": 81})``.
        """
        return self.ctx.fiber.config

    @raw_config.setter
    def raw_config(self, config: object) -> None:
        # retain raw input without changing the running config or triggering an update
        fiber = self.ctx.fiber
        fiber.config = config
        fiber._config_revision += 1

    @property
    def effective_config(self) -> Any:
        """The config passed to activation, or by the latest update that kept the instance.

        Candidate resolution does not publish it; volatile updates retain its
        references. ``None`` before activation and still readable after unload.

        Example: ``fiber.effective_config.port``.
        """
        return self.ctx.fiber._effective_config

    async def restart(self) -> None:
        """Unload this fiber's effects and load it again with its current config.

        Raises ``CordisError("INACTIVE_EFFECT")`` when the fiber is disposed.

        Example: ``await fiber.restart()``.
        """
        await self._begin_restart().wait()

    def _begin_restart(self) -> Fiber:
        """Run the synchronous prefix of cordis ``restart`` and return the target fiber."""
        fiber: Fiber = self.ctx.fiber
        fiber.assert_active()
        fiber._revision += 1
        fiber.refresh_epoch()
        return fiber

    def recover(self) -> None:
        """Clear this fiber's failure and load it again with its current config.

        Called by the :class:`~pycordis.supervise.Supervisor` once a restart's
        backoff has elapsed; does nothing unless the fiber is ``FAILED``. A fiber
        whose dependencies went away meanwhile becomes ``PENDING`` and loads
        when they are back.

        Internal — not part of the public API (``update()`` also recovers).

        Example: ``if fiber.state == FiberState.FAILED: fiber.recover()``.
        """
        if self.uid is None or self._error is None:
            return
        self._error = None
        self._begin_restart()
        if self.state == FiberState.FAILED:
            # no transition started (dependencies missing): leave FAILED for PENDING
            self._update_state(lambda: None)

    def update(self, config: object, no_save: bool = False) -> asyncio.Future[None]:
        """Resolve a candidate, retain it as the raw config and apply it.

        An active fiber resolves ``config`` now, raising
        :class:`~pycordis.errors.ValidationError` synchronously and leaving the
        fiber untouched. If the candidate differs from :attr:`effective_config`
        only inside volatile references, the new snapshots are written into the
        existing references and the plugin keeps running; no ``internal/update``
        listener is consulted. Otherwise the ``internal/update`` waterfall runs:
        listeners ``(fiber, resolved, next)`` may veto the restart by not calling
        ``next`` (``await next()`` is always safe; listeners run with their
        registering context current and a coroutine they return is wrapped to
        keep it so, see :meth:`~pycordis.events.EventsService.waterfall`), in
        which case the resolved candidate becomes :attr:`effective_config`
        (cordis parity) and references handed out earlier stop receiving
        volatile updates. A pending or failed
        fiber defers resolution to its next activation. ``internal/commit(fiber,
        config)`` is emitted unless ``no_save``.

        Pass *raw* config (for example a modified copy of :attr:`raw_config`),
        not the resolved :attr:`effective_config`: a commit listener such as the
        loader persists what it receives, and resolved values may hold expanded
        secrets. The loader puts placeholders back where it can and refuses
        writes that would still persist one.

        Returns a future, already done when nothing needs awaiting. Failures
        after this call returns (a failed restart or an async listener error) are
        reported through the fiber, so dropping the future is silent while
        awaiting it re-raises.

        Example: ``await fiber.update({**fiber.raw_config, "port": 81})``.
        """
        fiber: Fiber = self.ctx.fiber
        fiber.assert_active()
        loop = asyncio.get_running_loop()
        loaded = fiber._runner.epoch != INACTIVE
        resolved = fiber.resolve_config(config) if loaded else None
        if not no_save:
            fiber.ctx.emit("internal/commit", fiber, config)
        fiber.config = config
        fiber._error = None
        if fiber.supervisor is not None:
            # a new configuration: recovered here, with a fresh restart budget
            fiber.supervisor.reset()
        fiber._config_revision += 1
        if loaded and _commit_volatile(fiber._effective_config, resolved):
            return done_future(loop)
        revision = fiber._config_revision
        epoch = fiber._runner.epoch
        committed = fiber.committed
        restarted = False

        def restart() -> asyncio.Future[Fiber]:
            nonlocal restarted
            restarted = True
            return _quiet(asyncio.ensure_future(fiber._begin_restart().wait()))

        def publish() -> None:
            if (
                not restarted
                and fiber.uid is not None
                and fiber._config_revision == revision
                and fiber._runner.epoch == epoch
                and fiber.committed is committed
            ):
                fiber._effective_config = resolved

        result: object = (
            fiber.ctx.waterfall(fiber, "internal/update", resolved, restart)
            if loaded
            else restart()
        )
        if not inspect.isawaitable(result) or (
            isinstance(result, asyncio.Future)
            and result.done()
            and not result.cancelled()
            and result.exception() is None
        ):
            # a listener vetoed the restart (or applied it synchronously)
            publish()
            return done_future(loop)

        async def settle(awaitable: Awaitable[object]) -> None:
            try:
                await awaitable
            except Exception as error:
                report_error(fiber, error)
                raise
            publish()

        return _quiet(loop.create_task(settle(result)))


def emit_plugin_disposed(context: Context, fiber: Fiber) -> None:
    """Notify ``internal/plugin`` without letting one observer break ownership cleanup.

    Internal — not part of the public API.
    """
    emit_reported(context, "internal/plugin", fiber)


def _noop_annotate(_error: BaseException) -> None:
    return None


def _is_constructor(func: Callable[..., object]) -> bool:
    """Whether a plugin callback is instantiated rather than called (cordis ``isConstructor``).

    Only classes are; generator and async functions are plain callables here.
    """
    return inspect.isclass(func)


def _quiet[T](future: asyncio.Future[T]) -> asyncio.Future[T]:
    """Mark ``future``'s exception retrieved: its failure is reported elsewhere."""
    future.add_done_callback(_retrieve)
    return future


def _retrieve[T](future: asyncio.Future[T]) -> None:
    if not future.cancelled():
        future.exception()


def _commit_volatile(effective: object, resolved: object) -> bool:
    """Write ``resolved``'s volatile snapshots into ``effective``'s references.

    Applies only when the two differ solely inside volatile references and at
    least one reference exists (an equal config without any keeps cordis's
    restart semantics). Returns whether the update was applied this way.
    """
    if not diff_only_volatile(effective, resolved):
        return False
    pairs = volatile_pairs(effective, resolved)
    if not pairs:
        return False
    for ref, source in pairs:
        snapshot = source.get()
        if ref.get() != snapshot:
            ref.commit(snapshot)
    return True
