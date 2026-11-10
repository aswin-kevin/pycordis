"""Ambient current context: the pycordis context whose code is running (spec §3.3).

The framework sets it while it runs user code on a context's behalf: a plugin
body, an effect's setup and disposers, an event listener, a timer callback and
a task started by ``ctx.spawn``. Plain helper functions can then reach services
without having ``ctx`` threaded through every call::

    from pycordis import current_context

    def search(query: str) -> list[str]:  # e.g. an agent tool
        return current_context().get("index").search(query)

The value is a :class:`contextvars.ContextVar`, so it follows asyncio tasks:
a task started while it is set inherits it.
"""

from __future__ import annotations

import contextvars
import inspect
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from pycordis.context import Context

__all__ = ["current_context", "use_context"]

CURRENT: contextvars.ContextVar[Context] = contextvars.ContextVar("pycordis.current_context")
"""The active pycordis context.

Internal — not part of the public API. Hot paths (event dispatch) call
``CURRENT.set`` / ``CURRENT.reset`` directly; everything else goes through the
helpers below.
"""


def current_context() -> Context:
    """Return the pycordis context whose code is running.

    Raises ``LookupError("no active pycordis context")`` outside plugin bodies,
    effects, listeners, disposers, timer callbacks, spawned tasks and
    :func:`use_context` blocks.

    Example::

        def plugin(ctx: Context, config: None) -> None:
            assert current_context() is ctx
    """
    try:
        return CURRENT.get()
    except LookupError:
        raise LookupError("no active pycordis context") from None


@contextmanager
def use_context(ctx: Context) -> Generator[Context]:
    """Make ``ctx`` the current context inside the ``with`` block, then restore the previous one.

    Example::

        with use_context(root) as ctx:
            assert current_context() is ctx
    """
    token = CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        CURRENT.reset(token)


def call_in[R](ctx: Context, callback: Callable[..., R], *args: object) -> R:
    """Call ``callback(*args)`` with ``ctx`` current.

    Internal — not part of the public API.
    """
    token = CURRENT.set(ctx)
    try:
        return callback(*args)
    finally:
        CURRENT.reset(token)


def call_bound(ctx: Context, callback: Callable[..., object], *args: object) -> object:
    """Like :func:`call_in`, and a coroutine result runs with ``ctx`` current when awaited.

    For listeners whose coroutine is awaited later by the dispatcher's caller
    (``waterfall``, fiber-local ``internal/update`` hooks).

    Internal — not part of the public API.
    """
    result = call_in(ctx, callback, *args)
    if inspect.iscoroutine(result):
        return await_in(ctx, result)
    return result


async def await_in[R](ctx: Context, awaitable: Awaitable[R]) -> R:
    """Await ``awaitable`` with ``ctx`` current (a coroutine body then runs with it set).

    Internal — not part of the public API.
    """
    token = CURRENT.set(ctx)
    try:
        return await awaitable
    finally:
        CURRENT.reset(token)


def task_context(ctx: Context) -> contextvars.Context:
    """A copy of the running :mod:`contextvars` context with ``ctx`` current.

    Pass it as ``create_task(..., context=...)`` so the task runs with ``ctx`` set.

    Internal — not part of the public API.
    """
    copied = contextvars.copy_context()
    copied.run(CURRENT.set, ctx)
    return copied


@final
class ContextSwitch:
    """Switch the current context across a run of listeners, restoring it once.

    Event dispatch calls :meth:`enter` only when the next listener's owner
    differs from the previous one, so a run of listeners sharing an owner (the
    common case) costs one ``set``; a per-listener ``set``/``reset`` would
    dominate the cost of a cheap listener. Nested dispatch and
    :func:`use_context` restore the value they found, so it is still the last
    entered context when control returns here.

    Example::

        with ContextSwitch() as switch:
            switch.enter(ctx)
            listener()

    Internal — not part of the public API.
    """

    __slots__ = ("_token",)

    def __init__(self) -> None:
        self._token: contextvars.Token[Context] | None = None

    def __enter__(self) -> ContextSwitch:
        return self

    def __exit__(self, *_exc: object) -> None:
        token = self._token
        if token is not None:
            self._token = None
            CURRENT.reset(token)

    def enter(self, ctx: Context) -> None:
        """Make ``ctx`` current until the next :meth:`enter` or the end of the block."""
        token = CURRENT.set(ctx)
        if self._token is None:
            self._token = token
