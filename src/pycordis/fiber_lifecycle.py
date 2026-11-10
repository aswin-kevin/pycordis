"""The load/unload state machine of a fiber (port of cordis ``fiber.ts`` transitions).

A plugin fiber loads when its required services are available and moves
through :class:`~pycordis.fiber_core.FiberState` driven by its dependency epoch
and ``inertia`` task. :class:`FiberLifecycle` sits between
:class:`~pycordis.fiber_core.FiberCore` (owned effects) and
:class:`~pycordis.fiber.Fiber` (construction and config updates).

Internal — import :class:`~pycordis.fiber.Fiber` and :data:`INACTIVE` from
:mod:`pycordis.fiber`, which re-exports them.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine, Generator
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal, Self

from pycordis._context_internals import events_of, reflect_of
from pycordis._utils import cancelling, resolve_future
from pycordis.current import task_context
from pycordis.effect import Disposable, EffectRunner, as_disposer_error, run_effect
from pycordis.fiber_core import FiberCore, FiberState, effect_label, report_error
from pycordis.supervise import LoadTimeoutError, abandoned, await_load, run_within, wait_until
from pycordis.telemetry import FiberStats, StatsRecorder, publish
from pycordis.trace import get_traceable

if TYPE_CHECKING:
    from pycordis.context import Context
    from pycordis.diagnose import CycleWatch
    from pycordis.reflect import Impl
    from pycordis.registry import Runtime
    from pycordis.supervise import Supervisor

INACTIVE = "__INACTIVE__"
"""Epoch value of a fiber whose dependencies are not satisfied."""


class FiberLifecycle(FiberCore):
    """Dependency tracking and load/unload transitions of a fiber.

    Example (through a fiber created by ``ctx.plugin()``)::

        fiber = ctx.plugin(MyPlugin)
        await fiber  # waits for the load and re-raises its error

    Internal — construct fibers through ``ctx.plugin()``.
    """

    def __init__(
        self, parent: Context, inject: dict[str, object] | None, runtime: Runtime | None
    ) -> None:
        super().__init__(parent, runtime)
        self.inject: dict[str, object] = inject if inject is not None else {}
        """Resolved dependency map (service name -> intercept config)."""
        self.committed: dict[str, Impl] | None = None
        """Required implementations retained through dependent teardown."""
        self.provisions: set[Impl] = set()
        """Service implementations owned by this fiber."""
        self.inertia: asyncio.Task[None] | None = None
        """The in-flight load/unload transition, if any."""
        self.load_timeout: float | None = None
        """Seconds an async plugin body may take to load (``None``: unbounded)."""
        self.dispose_timeout: float | None = None
        """Seconds an unload may take before stuck work is abandoned (``None``: unbounded)."""
        self.supervisor: Supervisor | None = None
        """Restarts this fiber after failures (``None`` for policy ``never``). Internal."""
        self.cycles: CycleWatch | None = None
        """Told when this fiber enters ``PENDING`` (``None`` for the root). Internal."""
        self.recorder = StatsRecorder()
        """Lifecycle counters behind :attr:`stats`. Internal."""
        self._error: BaseException | None = None
        self._revision = 0
        self._store: dict[str, Impl] = {}
        self._check_errors: dict[str, Exception] = {}
        self._release: Callable[[], None] | None = None
        self._setup: asyncio.Future[None] | None = None
        self._unloading: list[Disposable] = []
        self._load_timed_out = False
        self._runner: EffectRunner[str]

    @property
    def epoch(self) -> str:
        """The dependency epoch of the current activation (``INACTIVE`` when unloaded).

        Internal — not part of the public API.
        """
        return self._runner.epoch

    @property
    def error(self) -> BaseException | None:
        """The error that failed this fiber's last load (``None`` unless ``FAILED``).

        Example::

            fiber = ctx.plugin(broken)
            await settle(ctx)
            fiber.error  # ValueError('bad url'), also shown by ``fiber.diagnose()``
        """
        return self._error

    @property
    def stats(self) -> FiberStats:
        """A snapshot of this fiber's lifecycle counters and timings.

        Example::

            fiber = await ctx.plugin(worker)
            fiber.stats.loads  # 1; also failures, restarts, last_load_duration, last_error
        """
        return self.recorder.snapshot()

    def satisfied(self, name: str) -> bool:
        """Whether injected service ``name`` is available to this fiber (passed its ``check``).

        Internal — not part of the public API (read by :mod:`pycordis.diagnose`).
        """
        return name in self._store

    def check_error(self, name: str) -> Exception | None:
        """The error the ``check()`` of injected service ``name`` raised at its last check.

        ``None`` when the last check returned (either way) or there was none.

        Internal — not part of the public API (read by :mod:`pycordis.diagnose`).
        """
        return self._check_errors.get(name)

    def __await__(self) -> Generator[Any, None, Self]:  # what a Future's __await__ yields
        return self.wait().__await__()

    async def _settle_inertia(self) -> None:
        # Shielded: abandoning the wait must not cancel the transition itself,
        # which would wedge the fiber in LOADING / UNLOADING.
        while (inertia := self.inertia) is not None:
            try:
                await asyncio.shield(inertia)
            except asyncio.CancelledError:
                # the transition itself was cancelled (it reconciles the fiber
                # in a done callback, which already ran); only our own
                # cancellation propagates
                if cancelling() or not inertia.cancelled():
                    raise

    async def wait(self) -> Self:
        """Wait for current lifecycle work and re-raise a startup error (cordis ``await``)."""
        await self._settle_inertia()
        if self._error is not None:
            raise self._error
        return self

    def _get_state(self) -> FiberState:
        if self.uid is None:
            return FiberState.DISPOSED
        if self._error is not None:
            return FiberState.FAILED
        if self._runner.epoch != INACTIVE:
            return FiberState.ACTIVE
        return FiberState.PENDING

    def _update_state(self, callback: Callable[[], FiberState | None]) -> None:
        old_state = self.state
        state = callback()
        self.state = state if state is not None else self._get_state()
        if old_state == self.state:
            return
        events_of(self.ctx).emit("internal/status", self, old_state)
        if self.state == FiberState.PENDING and self.cycles is not None:
            self.cycles.entered_pending(self)

        # only notify changes between ACTIVE and NON-ACTIVE states
        if old_state != FiberState.ACTIVE and self.state != FiberState.ACTIVE:
            return
        for impl in self._registered_provisions():
            self.ctx.isolate(impl.name, impl.isolate).reflect.notify([impl.name])

    def _registered_provisions(self) -> list[Impl]:
        """This fiber's provisions still registered in the reflect store, in store order.

        ``provisions`` also keeps retired ones until their dependents unloaded.
        The store only grows by appending a new record with the next uid, so
        uid order is store order; scanning the (application-wide) store instead
        would make every state change cost one step per provided service.
        """
        store = reflect_of(self.ctx).store
        registered = [impl for impl in self.provisions if store.get(impl.isolate) is impl]
        return sorted(registered, key=_impl_uid)

    def check_impl(self, name: str) -> None:
        """Re-resolve the injected service ``name`` and run its ``check`` hook.

        Stores the implementation this fiber sees for ``name`` (none when it is
        withdrawn or rejected; a raising check is reported). Call
        :meth:`refresh_epoch` afterwards to act on the change. Internal: used by
        service notification. Example: ``fiber.check_impl("db")``.
        """
        if self._check_errors:
            self._check_errors.pop(name, None)
        impl = self.ctx.reflect.get_impl(name, True)
        if impl is None:
            self._store.pop(name, None)
            return
        try:
            if impl.check is not None and not _call_check(impl.check, self.ctx, impl.value):
                self._store.pop(name, None)
                return
        except Exception as error:
            report_error(impl.ctx, error)
            self._check_errors[name] = error
            self._store.pop(name, None)
            return
        self._store[name] = impl

    def refresh_epoch(self) -> None:
        """Recompute this fiber's dependency epoch, loading / unloading it when it changed.

        Internal: used by service notification and restarts. Example:
        ``fiber.refresh_epoch()`` after :meth:`check_impl`.
        """
        epoch = str(self._revision)
        for name in self.inject:
            impl = self._store.get(name)
            if impl is None:
                epoch = INACTIVE
                break
            epoch += f":{impl.uid}"
        self._set_epoch(epoch)

    def _set_epoch(self, epoch: str) -> None:
        old_epoch = self._runner.epoch
        if epoch == old_epoch:
            return
        # a failed fiber only recovers through update(), which clears _error
        if self._error is not None:
            return
        self._runner.epoch = epoch
        # Python extension: an async plugin body still in flight is cancelled
        # once its load is superseded (cordis would keep awaiting it, so e.g. a
        # shutdown during startup I/O could never complete).
        setup = self._setup
        if setup is not None:
            self._setup = None
            setup.cancel()
        if self.inertia is not None:
            return

        def transition() -> FiberState:
            if epoch != INACTIVE and old_epoch == INACTIVE:
                self.inertia = self._start_reload()
                return FiberState.LOADING
            self.inertia = self._start_unload()
            return FiberState.UNLOADING

        self._update_state(transition)

    # Any: asyncio's own coroutine typing (yield and send types)
    def _create_task(
        self, coro: Coroutine[Any, Any, None], kind: Literal["load", "unload"]
    ) -> asyncio.Task[None]:
        # load and unload run the plugin body and its disposers: make the
        # fiber's context current for them (and every task they start)
        return asyncio.get_running_loop().create_task(
            coro, name=f"pycordis:{self.name}:{kind}", context=task_context(self.ctx)
        )

    def _start_reload(self) -> asyncio.Task[None]:
        """Run the synchronous prefix of cordis ``_reload`` now, the rest as a task."""
        self.recorder.start_load()
        self.committed = dict(self._store)
        for impl in self.committed.values():
            if impl.ctx.fiber is self:
                continue
            impl.dependents.add(self)
        task = self._create_task(self._reload(self._runner.epoch), "load")
        task.add_done_callback(self._on_reload_done)
        return task

    def _on_reload_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() or self.inertia is not task:
            return
        # A cancelled load (e.g. loop shutdown) fails the fiber instead of
        # wedging it in LOADING; what it collected stays owned by the fiber
        # and is disposed by its next unload.
        error = RuntimeError("plugin load was cancelled")
        self._error = error
        self._runner.epoch = INACTIVE

        def transition() -> None:
            self.inertia = None

        self._update_state(transition)
        publish(self, "fail", self.recorder.failed(error), error, state=FiberState.FAILED)

    async def _reload(self, old_epoch: str) -> None:
        superseded = False
        failure: Exception | None = None
        try:
            await asyncio.sleep(0)
            if self._runner.epoch == old_epoch:
                superseded = await self._run_setup()
        except Exception as reason:
            report_error(self, reason)
            # the unload that follows is bounded too (see _unload_timeout)
            self._load_timed_out = isinstance(reason, LoadTimeoutError)
            # a superseded load still needs cleanup, but cannot fail the new target
            if self._runner.epoch == old_epoch:
                self._error = failure = reason
                self._runner.epoch = INACTIVE
        loaded = self._runner.epoch == old_epoch and not superseded

        def transition() -> FiberState | None:
            if loaded:
                self.inertia = None
                return None
            self.inertia = self._start_unload()
            return FiberState.UNLOADING

        self._update_state(transition)
        if loaded:
            publish(self, "load", self.recorder.loaded())
        elif failure is not None:
            self._publish_failure(failure)

    def _publish_failure(self, error: Exception) -> None:
        """Count a failed load and publish it (a load timeout as ``timeout`` first)."""
        duration = self.recorder.failed(error)
        if isinstance(error, LoadTimeoutError):
            publish(self, "timeout", duration, error)
        # FAILED is the outcome; the fiber is still UNLOADING what the load set up
        publish(self, "fail", duration, error, state=FiberState.FAILED)

    async def _run_setup(self) -> bool:
        """Run the plugin body; return whether a superseding epoch cancelled it."""
        setup = run_effect(self._runner)
        if setup is None:
            return False
        self._setup = setup
        try:
            await await_load(setup, self.load_timeout, self.name, self._runner.annotate)
        except asyncio.CancelledError as error:
            if cancelling():
                raise
            if self._setup is not setup:
                return True
            # the body cancelled itself: a failure, not a shutdown request
            raise RuntimeError("plugin setup raised CancelledError") from error
        finally:
            if self._setup is setup:
                self._setup = None
        return False

    def _start_unload(self) -> asyncio.Task[None]:
        """Run the synchronous prefix of cordis ``_unload`` now, the rest as a task."""
        self.recorder.start_unload()
        released = self._await_dependents()
        tasks = self._begin_disposal() if released is None else None
        task = self._create_task(self._unload(released, tasks), "unload")
        task.add_done_callback(self._on_unload_done)
        return task

    def _on_unload_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() or self.inertia is not task:
            return
        # A cancelled unload hands its not-yet-run disposers back to the fiber
        # so a later unload (e.g. dispose()) still runs them.
        for dispose in reversed(self._unloading):
            self._disposables.push(dispose)
        self._unloading = []
        self._release = None

        def transition() -> None:
            self.inertia = None

        self._update_state(transition)

    def _await_dependents(self) -> asyncio.Future[None] | None:
        # Dependents are notified when this fiber leaves ACTIVE, but may still
        # access its services during their own teardown, so the effects of this
        # fiber are only disposed after every dependent has unloaded.
        if not any(impl.dependents for impl in self.provisions):
            self._release = None
            return None
        released: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._release = partial(resolve_future, released)
        return released

    def _begin_disposal(
        self,
    ) -> list[asyncio.Task[Any]]:  # Task is invariant: any result type
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        self._unloading = self._disposables.clear()
        return tasks

    async def _unload(
        self, released: asyncio.Future[None] | None, tasks: list[asyncio.Task[Any]] | None
    ) -> None:
        while released is not None:
            await released
            released = self._await_dependents()
        if tasks is None:
            tasks = self._begin_disposal()
        timeout = self._unload_timeout()
        if timeout is None:
            # spawned background work stops before the disposers it may rely on run
            if tasks:
                await asyncio.wait(tasks)
            await asyncio.gather(*(self._run_disposer(d) for d in list(self._unloading)))
        else:
            await self._dispose_within(tasks, timeout)
        # release the providers waiting for this fiber to unload
        for impl in (self.committed or {}).values():
            if self not in impl.dependents:
                continue
            impl.dependents.discard(self)
            if impl.dependents:
                continue
            if impl.release is not None:
                impl.release()
            release = impl.ctx.fiber._release
            if release is not None:
                release()
        self.committed = None
        # strict mode: every listener of a plugin fiber should leave with its unload
        if self.runtime is not None and self.ctx.reflect.strict:
            self.ctx.events.warn_outlived(self)

        def transition() -> FiberState | None:
            if self._runner.epoch == INACTIVE:
                self.inertia = None
                return None
            self.inertia = self._start_reload()
            return FiberState.LOADING

        self._update_state(transition)
        publish(self, "unload", self.recorder.unload_elapsed())
        # every failed load ends here, once what it collected is disposed
        if self.state == FiberState.FAILED and self.supervisor is not None:
            self.supervisor.on_failed()

    def _unload_timeout(self) -> float | None:
        """The bound of the unload starting now: ``dispose_timeout``, else after a
        load timeout the ``load_timeout`` (effects the cancelled body left in
        flight must not delay ``FAILED`` indefinitely)."""
        timed_out = self._load_timed_out
        self._load_timed_out = False
        if self.dispose_timeout is not None or not timed_out:
            return self.dispose_timeout
        return self.load_timeout

    async def _dispose_within(
        self,
        tasks: list[asyncio.Task[Any]],
        timeout: float,  # tasks of any result type
    ) -> None:
        """Stop owned tasks, then run the disposers; abandon what is left after ``timeout``."""
        deadline = asyncio.get_running_loop().time() + timeout
        for task in await wait_until(tasks, deadline):
            self._abandon(f"task {task.get_name()}", timeout)
        jobs = [(effect_label(d), self._run_disposer(d)) for d in list(self._unloading)]
        for label in await run_within(jobs, deadline):
            self._abandon(f"disposer of effect {label}", timeout)

    def _abandon(self, what: str, timeout: float) -> None:
        """Report and publish unload work abandoned after ``timeout`` seconds."""
        error = abandoned(what, self.name, timeout)
        report_error(self, error)
        publish(self, "timeout", timeout, error)

    async def _run_disposer(self, dispose: Disposable) -> None:
        try:
            await asyncio.sleep(0)
            if dispose not in self._unloading:
                return  # handed back by a cancelled unload
            self._unloading.remove(dispose)
            result = dispose()
            if inspect.isawaitable(result):
                await result
        except BaseException as raw:
            # a disposer's own CancelledError must not cancel the unload
            reason = as_disposer_error(raw)
            if reason is None:
                raise
            report_error(self, reason)


def _impl_uid(impl: Impl) -> int:
    return impl.uid


def _call_check(check: Callable[[], bool], ctx: Context, value: object) -> bool:
    """Call a service ``check`` with ``self`` traced to ``ctx`` (cordis ``check.call(traced)``)."""
    if isinstance(check, MethodType) and check.__self__ is value:
        return bool(check.__func__(get_traceable(ctx, value)))
    return bool(check())
