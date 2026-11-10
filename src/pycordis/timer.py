"""Fiber-owned timers (cordis ships these as the separate ``@cordisjs/plugin-timer``).

``ctx.timer`` is a built-in root service like ``ctx.logger``: no ``inject`` is
needed, and reading it through a context yields that use site, so every timer is
an effect of the *calling* fiber and stops when that fiber unloads. ``sleep``,
``set_timeout`` and ``set_interval`` are also mixed onto contexts.

Timer callbacks may be plain functions or return a coroutine, which then runs as a
task owned by the calling fiber (see :meth:`~pycordis.fiber.Fiber.spawn`). A
callback's exception is reported through the logger.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, final

from pycordis._utils import resolve_future
from pycordis.current import call_in
from pycordis.fiber import FiberState, report_error
from pycordis.trace import Tracker

if TYPE_CHECKING:
    from pycordis.context import Context
    from pycordis.effect import EffectDisposer
    from pycordis.fiber import Fiber

__all__ = ["TimerService"]


def _call(ctx: Context, callback: Callable[..., object], args: tuple[object, ...]) -> None:
    """Run a timer callback for the use site ``ctx``: report its error, own its coroutine.

    The callback and its coroutine run with ``ctx`` current; the coroutine is
    owned by ``ctx``'s fiber.
    """
    fiber: Fiber = ctx.fiber
    try:
        result = call_in(ctx, callback, *args)
    except Exception as error:
        report_error(fiber, error)
        return
    if inspect.iscoroutine(result):
        fiber.spawn_owned(result, context=ctx)


@final
class TimerService:
    """Built-in timer service, installed as ``ctx.timer``.

    Every timer is registered as an effect of the calling context's fiber, so it
    is cancelled when that fiber unloads (or when its disposer is called).
    """

    __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx", no_shadow=True)

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        """The context the service acts for (the use site through a proxy)."""

    async def sleep(self, seconds: float) -> None:
        """Wait ``seconds``; raises ``CancelledError`` if the calling fiber unloads first.

        Raises ``CordisError("INACTIVE_EFFECT")`` on a disposed or unloading fiber.
        """
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()

        def execute() -> Callable[[], None]:
            handle = loop.call_later(seconds, resolve_future, waiter)

            def dispose() -> None:
                handle.cancel()
                waiter.cancel()

            return dispose

        dispose = self.ctx.effect(execute, "ctx.timer.sleep()")
        try:
            await waiter
        finally:
            dispose()

    def set_timeout(self, callback: Callable[[], object], seconds: float) -> EffectDisposer:
        """Call ``callback`` once after ``seconds``; the returned disposer cancels it.

        Like every effect, this raises ``CordisError("INACTIVE_EFFECT")`` on a
        disposed or unloading fiber; for a grace delay inside teardown, use
        :func:`asyncio.sleep` instead.
        """
        return self._schedule(callback, (), seconds, "ctx.timer.set_timeout()")

    def _schedule(
        self, callback: Callable[..., object], args: tuple[object, ...], seconds: float, label: str
    ) -> EffectDisposer:
        ctx = self.ctx
        loop = asyncio.get_running_loop()
        dispose: EffectDisposer | None = None

        def fire() -> None:
            if dispose is not None:
                dispose()
            _call(ctx, callback, args)

        def execute() -> Callable[[], None]:
            return loop.call_later(seconds, fire).cancel

        dispose = self.ctx.effect(execute, label)
        return dispose

    def set_interval(self, callback: Callable[[], object], seconds: float) -> EffectDisposer:
        """Call ``callback`` every ``seconds`` until the returned disposer runs.

        Like every effect, this raises ``CordisError("INACTIVE_EFFECT")`` on a
        disposed or unloading fiber; for a grace delay inside teardown, use
        :func:`asyncio.sleep` instead.
        """
        ctx = self.ctx
        loop = asyncio.get_running_loop()

        def execute() -> Callable[[], None]:
            def fire() -> None:
                nonlocal handle
                handle = loop.call_later(seconds, fire)
                _call(ctx, callback, ())

            handle = loop.call_later(seconds, fire)
            return lambda: handle.cancel()

        return self.ctx.effect(execute, "ctx.timer.set_interval()")

    def debounce(self, callback: Callable[..., object], seconds: float) -> Callable[..., None]:
        """Return a function that calls ``callback`` ``seconds`` after its last call.

        Only the arguments of the last call are used. The pending call is owned by
        the calling fiber and dropped when it unloads; calling the returned function
        once that fiber is disposed or unloading does nothing (for a grace delay
        inside teardown, use :func:`asyncio.sleep`).
        """
        fiber: Fiber = self.ctx.fiber
        pending: EffectDisposer | None = None

        def debounced(*args: object) -> None:
            nonlocal pending
            if not _effectable(fiber):
                return
            if pending is not None:
                pending()
            pending = self._schedule(callback, args, seconds, "ctx.timer.debounce()")

        return debounced

    def throttle(self, callback: Callable[..., object], seconds: float) -> Callable[..., None]:
        """Return a function calling ``callback`` at most once per ``seconds``.

        The first call runs immediately; calls during the window are coalesced into
        one trailing call (with the latest arguments) when the window ends, which
        opens a new window. The window timer is owned by the calling fiber; calling
        the returned function once that fiber is disposed or unloading does nothing
        (for a grace delay inside teardown, use :func:`asyncio.sleep`).
        """
        ctx = self.ctx
        fiber: Fiber = ctx.fiber
        window: EffectDisposer | None = None
        trailing: tuple[object, ...] | None = None

        def close() -> None:
            nonlocal window, trailing
            window = None
            if trailing is not None:
                args, trailing = trailing, None
                run(args)

        def run(args: tuple[object, ...]) -> None:
            nonlocal window
            window = self._schedule(close, (), seconds, "ctx.timer.throttle()")
            _call(ctx, callback, args)

        def throttled(*args: object) -> None:
            nonlocal trailing
            if not _effectable(fiber):
                return
            if window is not None and window.active:
                trailing = args
                return
            run(args)

        return throttled


def _effectable(fiber: Fiber) -> bool:
    return fiber.uid is not None and fiber.state != FiberState.UNLOADING
