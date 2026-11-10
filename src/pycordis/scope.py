"""Scopes: plugin lifetimes bounded by an ``async with`` block.

``ctx.scope()`` starts an anonymous plugin fiber under ``ctx``, waits until it
is ACTIVE (its injections available) and yields its context. Leaving the block
disposes the fiber, so every effect, listener, spawned task, provided service
and collection item registered through the scope context goes with it::

    async with ctx.scope(name="session-42", inject=[LLM], provide=[MEMORY]) as s:
        s.provide(MEMORY, ConversationMemory())
        await s.plugin(chat_agent)  # sees this session's memory
        await s.use(LLM).complete(prompt)

Isolation: names passed as ``provide`` are isolated to the scope
(``ctx.isolate(name)`` for each, with a fresh label per scope), so two
concurrent scopes providing ``"memory"`` never collide, and neither the parent
nor sibling scopes see the scope's instance. A service provided through the
scope context under a name *not* declared in ``provide`` is an ordinary
provision: visible to the whole application until the scope exits.

The scope fiber is an ordinary plugin fiber once active: if an injected
service goes away during the block, it unloads (tearing down what the block
registered) and waits for the service again, while the block keeps running.

If the context the scope was opened from unloads mid-block, the scope fiber is
disposed with it. The block keeps running (unless its task is owned by that
context's fiber, e.g. started with ``ctx.spawn``, which cancels it) with a
DISPOSED scope context: registering further effects on it (``s.effect``,
``s.on``, ``s.plugin``, ...) raises ``CordisError("INACTIVE_EFFECT")``.

Activation always takes at least one event-loop turn (the plugin load runs as
a task), so ``timeout=0`` always raises ``TimeoutError``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, final

from pycordis._utils import cancelling
from pycordis.errors import CordisError
from pycordis.fiber import Fiber, FiberState, report_error
from pycordis.keys import ServiceKey, ServiceName, service_name
from pycordis.registry import Inject

if TYPE_CHECKING:
    from types import TracebackType

    from pycordis.context import Context
    from pycordis.registry import InjectSpec

__all__ = ["Scope"]


@final
class Scope(AbstractAsyncContextManager["Context"]):
    """The async context manager returned by :meth:`Context.scope <pycordis.context.Context.scope>`.

    Entering starts the scope fiber and waits until it is ACTIVE; ``as`` binds
    the scope's context. Exiting disposes the fiber, also when the block raised
    or was cancelled. A scope can be entered once. Annotate with it when a
    scope is created in one place and entered in another. Example::

        async with ctx.scope(inject=["db"], timeout=5.0) as s:
            await s.db.execute("...")
    """

    def __init__(
        self,
        ctx: Context,
        *,
        name: str | None = None,
        inject: InjectSpec | None = None,
        provide: ServiceName | Sequence[ServiceName] = (),
        config: object = None,
        timeout: float | None = None,
    ) -> None:
        self._ctx = ctx
        self._name = name if name is not None else "scope"
        self._inject = Inject.resolve(inject)
        self._provide = _isolated_names(provide)
        overlap = sorted(set(self._provide) & set(self._inject))
        if overlap:
            raise ValueError(
                f"scope {self._name!r} cannot both inject and provide {', '.join(overlap)}: "
                "a provided name is isolated, so the injection could never be satisfied"
            )
        self._config = config
        self._timeout = timeout
        self._fiber: Fiber | None = None

    async def __aenter__(self) -> Context:
        if self._fiber is not None:
            raise RuntimeError(f"scope {self._name!r} can only be entered once")
        fiber = self._fiber = self._start()
        try:
            await self._until_active(fiber)
        except BaseException:
            await _dispose(fiber)
            raise
        return fiber.ctx

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._fiber is not None, "__aexit__ without __aenter__"
        # Returning None lets the block's exception propagate unchanged.
        await _dispose(self._fiber)

    def _start(self) -> Fiber:
        """Start the scope fiber below a context isolating every provided name."""
        parent = self._ctx
        for name in self._provide:
            parent = parent.isolate(name)

        def apply(_ctx: Context, _config: object) -> None:
            # The body has nothing to do: the scope's context is its fiber's
            # context, handed out once the fiber is ACTIVE.
            return None

        # ``provide`` is declared like plugin metadata, so diagnostics see it
        plugin = SimpleNamespace(
            name=self._name, inject=self._inject, provide=self._provide, apply=apply
        )
        return parent.plugin(plugin, self._config)

    async def _until_active(self, fiber: Fiber) -> None:
        """Wait until ``fiber`` is ACTIVE, raising its startup error or ``TimeoutError``.

        Only the scope's own deadline becomes a scope ``TimeoutError``; a startup
        error that happens to be a ``TimeoutError`` is re-raised unchanged.
        """
        deadline = asyncio.timeout(self._timeout)
        try:
            async with deadline:
                await _await_active(fiber)
        except TimeoutError as error:
            if not deadline.expired():
                raise
            missing = [name for name in self._inject if self._ctx.get(name, False) is None]
            waiting = f" (waiting for {', '.join(missing)})" if missing else ""
            raise TimeoutError(
                f"scope {self._name!r} was not active after {self._timeout:g}s{waiting}"
            ) from error


async def _await_active(fiber: Fiber) -> None:
    """Wait until ``fiber`` is ACTIVE; raise its startup error, or if it is disposed first."""
    changed = asyncio.Event()

    def on_status(target: object, _old: object) -> None:
        if target is fiber:
            changed.set()

    # One listener per waiting scope, registered only for the wait and removed
    # in ``finally``: every fiber status change in the application calls it
    # while it exists, which is cheap (an identity check) and short-lived.
    # Owned by the scope fiber itself: a parent unload tears its own listeners
    # down before its children, which would hide the scope's disposal. The
    # scope fiber emits its own UNLOADING before running its disposers.
    remove = fiber.ctx.on("internal/status", on_status)
    try:
        while (state := fiber.state) != FiberState.ACTIVE:
            if state == FiberState.DISPOSED:
                raise CordisError(
                    "INACTIVE_EFFECT", f"scope <{fiber.name}> was disposed before it became active"
                )
            if state == FiberState.PENDING:
                changed.clear()
                await changed.wait()
            else:
                # LOADING / UNLOADING settle; FAILED re-raises the startup error
                await fiber.wait()
    finally:
        remove()


async def _dispose(fiber: Fiber) -> None:
    """Dispose ``fiber`` and wait for it; disposal errors are reported, never raised.

    Disposal runs shielded: cancelling the waiting task (a cancelled block, or
    a second cancellation while disposing) never cancels the unload itself, so
    the scope's effects are always torn down. A second cancellation only stops
    the waiting.
    """
    disposal = fiber.dispose()
    try:
        await asyncio.shield(disposal)
    except asyncio.CancelledError:
        if cancelling() or not disposal.cancelled():
            # stop waiting: the unload carries on and still reports its failure
            disposal.add_done_callback(partial(_report_disposal, fiber))
            raise
    except Exception as error:
        report_error(fiber.parent, error)


def _report_disposal(fiber: Fiber, disposal: asyncio.Future[None]) -> None:
    if disposal.cancelled():
        return
    error = disposal.exception()
    if error is not None:
        report_error(fiber.parent, error)


def _isolated_names(provide: ServiceName | Sequence[ServiceName]) -> list[str]:
    if isinstance(provide, str | ServiceKey):
        return [service_name(provide)]
    return list(dict.fromkeys(service_name(name) for name in provide))
