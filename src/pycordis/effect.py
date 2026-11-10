"""Effect engine: effect shapes, single-shot disposers and their metadata.

Port of the effect half of cordis ``fiber.ts`` (``Fiber._execute``, ``Fiber.effect``
and ``runDisposable`` / ``effectInertia``) onto asyncio. ``Fiber.effect`` builds an
:class:`EffectDisposer`; the owning fiber drives its own plugin body through the
same :func:`run_effect` so both accept identical effect shapes.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterable,
)
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, final

from pycordis._guards import is_future
from pycordis._utils import (
    DisposableList,
    add_note_once,
    cancelling,
    done_future,
    report_failure,
    symbols,
)
from pycordis.current import call_in

if TYPE_CHECKING:
    from pycordis.context import Context

type Disposable = Callable[[], object]
"""A cleanup callable; may return an awaitable, which teardown then awaits."""

type EffectResult = object
"""Anything an effect body may return (see :meth:`pycordis.fiber.Fiber.effect`).

:func:`run_effect` dispatches on its shape: ``None``, a context manager (sync
or async), a disposer callable, an awaitable, or a (async) generator of
disposers; anything else is ``TypeError("Invalid effect")``.
"""


@final
@dataclass(slots=True)
class EffectMeta:
    """Diagnostics tree node: an effect's label and the effects nested in it.

    Example: ``[child.label for child in fiber.get_effects()[0].children]``.
    """

    label: str
    children: list[EffectMeta] = field(default_factory=list["EffectMeta"])


@final
class EffectRunner[T]:
    """Execution state shared between :func:`run_effect` and its owner.

    ``epoch`` is compared before every async-iterator step so a superseded setup
    stops; ``collect`` receives each produced disposer; ``annotate`` decorates
    setup errors before they propagate (cordis ``composeError``); ``on_none``,
    if given, is called the first time a generator effect yields ``None``.
    """

    __slots__ = ("annotate", "collect", "epoch", "execute", "on_none")

    def __init__(
        self,
        epoch: T,
        execute: Callable[[], EffectResult],
        collect: Callable[[Disposable], object],
        annotate: Callable[[BaseException], object],
        on_none: Callable[[], object] | None = None,
    ) -> None:
        self.epoch = epoch
        self.execute = execute
        self.collect = collect
        self.annotate = annotate
        self.on_none = on_none


def run_effect[E](runner: EffectRunner[E]) -> asyncio.Future[None] | None:
    """Run ``runner.execute`` once and collect the disposers it produces.

    Returns ``None`` when setup completed synchronously, otherwise a task that
    completes when async setup has finished collecting. Raises
    ``TypeError("Invalid effect")`` for an unsupported shape. Example
    (what :class:`EffectDisposer` does)::

        pending = run_effect(EffectRunner(True, execute, disposers.append, annotate))
    """
    return _Setup(runner).run()


@final
class _Setup[E]:
    """One run of an effect body: dispatch on what it returned, collect its disposers."""

    __slots__ = ("_epoch", "_runner", "_warned")

    def __init__(self, runner: EffectRunner[E]) -> None:
        self._runner = runner
        self._epoch = runner.epoch  # an async generator stops once the epoch moved on
        self._warned = False

    def run(self) -> asyncio.Future[None] | None:
        runner = self._runner
        try:
            # dispatched on its shape with hasattr checks below (cordis duck typing),
            # which a static type cannot express: each branch hands it on typed
            effect: Any = runner.execute()
            if effect is None:
                return None
            # context managers first: @contextmanager objects are also callable
            if hasattr(effect, "__aenter__") and hasattr(effect, "__aexit__"):
                return asyncio.ensure_future(self._from_async_cm(effect))
            if hasattr(effect, "__enter__") and hasattr(effect, "__exit__"):
                effect.__enter__()
                exit_ = effect.__exit__
                runner.collect(lambda: exit_(None, None, None))
                return None
            if callable(effect):
                runner.collect(effect)
                return None
            if inspect.isawaitable(effect):
                return _spawn(self._from_awaitable(effect), effect)
            if hasattr(effect, "__aiter__"):
                return asyncio.ensure_future(self._from_async_iter(effect))
            if not hasattr(effect, "__iter__"):
                raise TypeError("Invalid effect")
            self._from_iter(effect)
            return None
        except Exception as error:
            runner.annotate(error)
            raise

    def _yielded(self, dispose: object) -> None:
        # a fixture-style ``yield`` (cleanup after it) runs its cleanup during setup
        on_none = self._runner.on_none
        if dispose is None and not self._warned and on_none is not None:
            self._warned = True
            on_none()
        self._collect(dispose)

    def _collect(self, dispose: object) -> None:
        if dispose is None:
            return
        if not callable(dispose):
            raise TypeError("Invalid effect")
        self._runner.collect(dispose)

    def _from_iter(self, iterable: Iterable[object]) -> None:
        iterator = iter(iterable)
        try:
            while True:
                try:
                    dispose = next(iterator)
                except StopIteration as stop:
                    self._collect(stop.value)
                    return
                self._yielded(dispose)
        finally:
            # generators have it; a plain iterator may not
            close: Callable[[], object] | None = getattr(iterator, "close", None)
            if close is not None:
                close()

    async def _from_awaitable(self, value: Awaitable[object]) -> None:
        try:
            self._collect(await value)
        except Exception as error:
            self._runner.annotate(error)
            raise

    async def _from_async_cm(self, cm: AbstractAsyncContextManager[object]) -> None:
        try:
            await cm.__aenter__()
        except Exception as error:
            self._runner.annotate(error)
            raise
        exit_ = cm.__aexit__
        self._runner.collect(lambda: exit_(None, None, None))

    async def _from_async_iter(self, iterable: AsyncIterable[object]) -> None:
        iterator = aiter(iterable)
        try:
            while self._runner.epoch == self._epoch:
                try:
                    dispose = await anext(iterator)
                except StopAsyncIteration:
                    return
                self._yielded(dispose)
        except Exception as error:
            self._runner.annotate(error)
            raise
        finally:
            # async generators have it; a plain async iterator may not
            aclose: Callable[[], Awaitable[object]] | None = getattr(iterator, "aclose", None)
            if aclose is not None:
                await aclose()


def as_disposer_error(error: BaseException) -> Exception | None:
    """Classify an error escaping a user disposer.

    Ordinary exceptions are disposer errors. A ``CancelledError`` that does not
    come from cancelling the current task (e.g. a disposer doing
    ``task.cancel(); await task``) is converted into a ``RuntimeError`` so it can
    be reported or raised without cancelling teardown. Returns ``None`` when the
    error must propagate (the current task is being cancelled, or a non-Exception
    such as ``KeyboardInterrupt``).
    """
    if isinstance(error, Exception):
        return error
    if isinstance(error, asyncio.CancelledError) and not cancelling():
        converted = RuntimeError("disposer raised CancelledError")
        converted.__cause__ = error
        return converted
    return None


# Any: asyncio's own coroutine typing (yield and send types)
def _spawn(coro: Coroutine[Any, Any, None], awaitable: object) -> asyncio.Future[None]:
    """Schedule ``coro``; if it is cancelled before awaiting a user coroutine, close that."""
    task = asyncio.ensure_future(coro)
    if inspect.iscoroutine(awaitable):
        pending = awaitable

        def close_if_cancelled(done: asyncio.Future[None]) -> None:
            if done.cancelled():
                pending.close()

        task.add_done_callback(close_if_cancelled)
    return task


def _failed(error: BaseException) -> asyncio.Future[None]:
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    future.set_exception(error)
    return future


@final
class EffectDisposer:
    """Single-shot, awaitable handle returned by :meth:`pycordis.fiber.Fiber.effect`.

    Calling it tears the effect down (disposers run in reverse order) and returns
    a future that is already done when cleanup was synchronous; later calls return
    the in-flight cleanup future, or a done one. Awaiting it waits for async setup
    and yields a disposer callable, re-raising the setup error if setup failed.

    Setup and the collected disposers run with ``context`` (the owning fiber's
    context) current (:func:`~pycordis.current_context`).

    Example::

        dispose = ctx.effect(lambda: lambda: print("cleaned up"))
        await dispose()
    """

    __slots__ = (
        "_barrier",
        "_context",
        "_disposables",
        "_disposal",
        "_disposing",
        "_executing",
        "_in_flight",
        "_owner",
        "_owner_name",
        "_remove_wrapper",
        "_report",
        "_runner",
        "_setup_failed",
        "_task",
        "meta",
    )

    def __init__(
        self,
        execute: Callable[[], EffectResult],
        label: str,
        *,
        owner_name: Callable[[], str],
        owner: DisposableList[Disposable],
        report: Callable[[BaseException], object],
        on_none: Callable[[], object] | None = None,
        context: Context,
    ) -> None:
        self.meta = EffectMeta(label, [])
        self._context = context
        self._owner_name = owner_name
        self._owner = owner
        self._report = report
        self._disposables: list[Disposable] = []
        self._disposing = False
        self._disposal: asyncio.Future[None] | None = None
        self._runner = EffectRunner[bool](True, execute, self._collect, self._annotate, on_none)
        self._task: asyncio.Future[None] | None = None
        self._executing = True
        self._barrier: asyncio.Future[None] | None = None
        self._setup_failed = False
        self._in_flight: asyncio.Future[None] | None = None
        self._remove_wrapper: Callable[[], bool] = lambda: False

        # Make the effect visible to a reentrant owner unload before execute()
        # runs any plugin code. Async teardown stays owner-visible until it
        # settles, allowing an outer effect to join cleanup another caller began.
        self._remove_wrapper = owner.push(self)
        try:
            self._task = self._in_context(partial(run_effect, self._runner))
        except BaseException as reason:
            self._executing = False
            self._setup_failed = True
            self._runner.epoch = False
            try:
                cleanup = self._finalize(self._dispose)
            finally:
                barrier = self._barrier
                if barrier is not None and not barrier.done():
                    barrier.set_exception(reason)
            self._report_outcome(cleanup)
            raise
        self._executing = False
        task = self._task
        barrier = self._barrier
        if barrier is not None:
            if task is None:
                barrier.set_result(None)
            else:
                task.add_done_callback(lambda t: _transfer(t, barrier))
        if task is not None:
            task.add_done_callback(self._on_setup_done)

    @property
    def __cordis_effect__(self) -> EffectMeta:
        """The effect's metadata under the cordis ``Context.effect`` symbol name."""
        return self.meta

    @property
    def active(self) -> bool:
        """Whether this effect has not been disposed yet.

        Example: ``if dispose.active: await dispose()``.
        """
        return self._runner.epoch

    def __call__(self) -> asyncio.Future[None]:
        # A synchronous setup failure can race an owner unload that already
        # captured this wrapper but has not invoked it yet; such callers (and
        # repeated calls) join the in-flight cleanup instead of starting one.
        if not self._runner.epoch:
            return self._in_flight if self._in_flight is not None else done_future()
        self._runner.epoch = False
        return self._finalize(self._dispose_on_call)

    # Any: what awaiting a Future yields to the event loop
    def __await__(self) -> Generator[Any, None, Callable[[], asyncio.Future[None]]]:
        return self._settle().__await__()

    async def _settle(self) -> Callable[[], asyncio.Future[None]]:
        if self._task is not None:
            await asyncio.shield(self._task)
        return self._dispose_async

    def _dispose_async(self) -> asyncio.Future[None]:
        if not self._runner.epoch:
            return self._in_flight if self._in_flight is not None else done_future()
        self._runner.epoch = False
        return self._finalize(self._dispose)

    def _dispose_on_call(self) -> asyncio.Future[None] | None:
        if self._executing:
            return self._dispose_after(self._wait_for_setup())
        if self._task is not None:
            return self._dispose_after(self._task)
        return self._dispose()

    def _wait_for_setup(self) -> asyncio.Future[None]:
        if self._barrier is None:
            self._barrier = asyncio.get_running_loop().create_future()
        return self._barrier

    def _dispose_after(self, setup: asyncio.Future[None]) -> asyncio.Future[None]:
        async def run() -> None:
            try:
                await asyncio.shield(setup)
            except asyncio.CancelledError:
                if cancelling():
                    raise
                # setup was cancelled (not us): dispose what it collected
            except Exception:
                # the setup error stays primary; a cleanup failure is reported
                try:
                    pending = self._dispose()
                    if pending is not None:
                        await pending
                except Exception as error:
                    self._report(error)
                raise
            pending = self._dispose()
            if pending is not None:
                await pending

        return self._in_context(partial(asyncio.ensure_future, run()))

    def _in_context[R](self, callback: Callable[[], R]) -> R:
        """Call ``callback`` with the owning context current; tasks it starts inherit it."""
        return call_in(self._context, callback)

    def _finalize(
        self, callback: Callable[[], asyncio.Future[None] | None]
    ) -> asyncio.Future[None]:
        try:
            result = callback()
        except Exception as error:
            self._remove_wrapper()
            return _failed(error)
        if result is None:
            self._remove_wrapper()
            return done_future()

        def settle(_: asyncio.Future[None]) -> None:
            self._remove_wrapper()
            if self._in_flight is result:
                self._in_flight = None

        result.add_done_callback(settle)
        self._in_flight = result
        return result

    def _collect(self, dispose: Disposable) -> None:
        # a nested disposer handed to us is owned here, not by the fiber
        self._disposables.append(dispose)
        self._owner.delete(dispose.effect if isinstance(dispose, ListenerDisposer) else dispose)
        meta = getattr(dispose, symbols.effect, None)
        if isinstance(meta, EffectMeta):
            self.meta.children.append(meta)

    def _annotate(self, error: BaseException) -> BaseException:
        add_note_once(error, f"in effect {self.meta.label} of <{self._owner_name()}>")
        return error

    def _dispose(self) -> asyncio.Future[None] | None:
        """Run collected disposers in reverse; a raising one never blocks the rest."""
        return self._in_context(self._dispose_collected)

    def _dispose_collected(self) -> asyncio.Future[None] | None:
        if self._disposing:
            return self._disposal
        self._disposing = True
        disposables = self._disposables[::-1]
        self._disposables.clear()
        errors: list[BaseException] = []
        for index, dispose in enumerate(disposables):
            pending = self._run_disposable(dispose, errors)
            if pending is not None:
                rest = disposables[index + 1 :]
                self._disposal = asyncio.ensure_future(self._dispose_rest(pending, rest, errors))
                return self._disposal
        self._raise_first(errors)
        return None

    async def _dispose_rest(
        self, pending: Awaitable[object], rest: list[Disposable], errors: list[BaseException]
    ) -> None:
        await self._await_disposable(pending, errors)
        for dispose in rest:
            following = self._run_disposable(dispose, errors)
            if following is not None:
                await self._await_disposable(following, errors)
        self._raise_first(errors)

    async def _await_disposable(
        self, pending: Awaitable[object], errors: list[BaseException]
    ) -> None:
        try:
            await pending
        except BaseException as raw:
            error = as_disposer_error(raw)
            if error is None:
                raise
            errors.append(self._annotate(error))

    def _run_disposable(
        self, dispose: Disposable, errors: list[BaseException]
    ) -> Awaitable[object] | None:
        try:
            result = dispose()
        except BaseException as raw:
            error = as_disposer_error(raw)
            if error is None:
                raise
            errors.append(self._annotate(error))
            return None
        if is_future(result):
            future = result
            if not future.done():
                return future
            # e.g. a nested EffectDisposer whose cleanup finished synchronously
            if future.cancelled():
                cancelled = RuntimeError("disposer raised CancelledError")
                errors.append(self._annotate(cancelled))
            elif (error := future.exception()) is not None:
                errors.append(self._annotate(error))
            return None
        if inspect.isawaitable(result):
            return result
        return None

    def _raise_first(self, errors: list[BaseException]) -> None:
        if not errors:
            return
        for error in errors[1:]:
            self._report(error)
        raise errors[0]

    def _on_setup_done(self, task: asyncio.Future[None]) -> None:
        if not task.cancelled():
            error = task.exception()
            if error is None:
                return
            self._report(error)
        # a failed or cancelled setup still disposes what it already collected
        try:
            # an already-requested disposal only needs the shared cleanup
            cleanup = self._finalize(self._dispose) if self._runner.epoch else self._dispose()
        except Exception as dispose_error:
            self._report(dispose_error)
            return
        self._report_outcome(cleanup)

    def _report_outcome(self, future: asyncio.Future[None] | None) -> None:
        if future is not None:
            report_failure(future, self._report, eager=True)


@final
class ListenerDisposer:
    """Disposer returned by ``ctx.on()`` / ``ctx.once()`` (cordis ``() => boolean``).

    Calling it removes the listener synchronously and returns ``True`` if the
    listener was still registered, ``False`` otherwise. ``effect`` is the
    underlying registration :class:`EffectDisposer`.

    Example: ``off = ctx.on("ready", handler)``; ``off()`` is ``True`` once.
    """

    __slots__ = ("effect",)

    def __init__(self, effect: EffectDisposer) -> None:
        self.effect = effect

    @property
    def __cordis_effect__(self) -> EffectMeta:
        """The registration effect's metadata."""
        return self.effect.meta

    def __call__(self) -> bool:
        active = self.effect.active
        future = self.effect()
        if future.done() and not future.cancelled():
            future.result()
        return active


def _transfer(source: asyncio.Future[None], target: asyncio.Future[None]) -> None:
    if target.done():
        return
    if source.cancelled():
        target.cancel()
    elif (error := source.exception()) is not None:
        target.set_exception(error)
    else:
        target.set_result(None)
