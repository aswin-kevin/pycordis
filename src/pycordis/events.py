"""Events service: the event bus installed as ``ctx.events`` (port of cordis ``events.ts``).

Dispatch arguments may start with an optional receiver (``this_arg``): when the
first argument is neither a ``str`` nor an :class:`EventKey` (and more
arguments follow) it is the receiver, whose ``__cordis_filter__(ctx)`` decides
which non-global listeners run. A lone non-name (``emit(123)``) is a bad event
name and raises ``TypeError``.

Python has no ``this``: listeners receive only the event arguments, except for
the events listed in :data:`RECEIVER_EVENTS`, whose listeners get the receiver
as an explicit first positional argument:
``internal/listener(ctx, name, listener, options)``,
``internal/update(fiber, config, next)`` and ``internal/service(ctx, name, value)``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import CoroutineType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, final, overload

from pycordis._context_internals import fiber_of
from pycordis._utils import is_bailed, symbols
from pycordis.current import CURRENT, ContextSwitch, await_in, call_bound, call_in
from pycordis.effect import EffectDisposer, ListenerDisposer
from pycordis.schema import validate_config
from pycordis.trace import Tracker, untraced

if TYPE_CHECKING:
    from contextvars import Token

    from pycordis.context import Context
    from pycordis.fiber import Fiber
    from pycordis.fiber_core import FiberCore

RECEIVER_EVENTS = frozenset({"internal/listener", "internal/update", "internal/service"})
"""Events whose listeners receive the dispatch receiver as first positional argument."""

type Listener = Callable[..., object]
"""An event listener."""


@final
class EventKey[**P, R]:
    """A typed event name: ``EventKey[[int, str], None]("app/ready")``.

    Usable wherever an event name is accepted; listeners and emitted arguments
    are then checked against ``P`` / ``R``. It is equivalent to its plain
    ``name`` at runtime, so typed and string forms interoperate.

    Example::

        READY = EventKey[[str], None]("app/ready")
        ctx.on(READY, lambda name: print(name))
        ctx.emit(READY, "web")
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"EventKey({self.name!r})"


@final
@dataclass(slots=True)
class EventOptions:
    """Options accepted by ``ctx.on()`` / ``ctx.once()``.

    ``prepend`` adds the listener before existing ones; ``global_`` delivers
    events regardless of the receiver's filter. A ``bool`` is shorthand for
    ``prepend`` and a dict ``{"prepend": ..., "global": ...}`` is accepted too.

    Example: ``ctx.on("app/stop", handler, EventOptions(prepend=True))``.
    """

    prepend: bool = False
    global_: bool = False


@final
@dataclass(frozen=True, slots=True)
class Hook:
    """A registered listener record (immutable: dispatch caches snapshots of hooks)."""

    ctx: Context
    callback: Listener
    prepend: bool = False
    global_: bool = False


@final
@dataclass(frozen=True, slots=True)
class _Plan:
    """A snapshot of one event's hooks, cached until a registration for it changes."""

    hooks: tuple[Hook, ...]
    """The hooks in dispatch order."""
    runs: tuple[tuple[Context, tuple[Listener, ...]], ...]
    """The same callbacks, grouped into runs of consecutive hooks sharing an owner."""

    @staticmethod
    def of(hooks: list[Hook]) -> _Plan:
        """Snapshot ``hooks``. Example: ``_Plan.of(events.hooks["x"]).runs``."""
        runs: list[tuple[Context, list[Listener]]] = []
        for hook in hooks:
            if runs and runs[-1][0] is hook.ctx:
                runs[-1][1].append(hook.callback)
            else:
                runs.append((hook.ctx, [hook.callback]))
        return _Plan(tuple(hooks), tuple((owner, tuple(run)) for owner, run in runs))


_NO_PLAN = _Plan((), ())
"""The plan of an event without listeners (never cached)."""


type ListenerOptions = bool | EventOptions | Mapping[str, bool] | None
"""Accepted ``options`` of ``on`` / ``once``: ``prepend`` flag, :class:`EventOptions` or mapping."""


def _normalize_options(options: ListenerOptions) -> EventOptions:
    if isinstance(options, EventOptions):
        return options
    if options is None or isinstance(options, bool):
        return EventOptions(prepend=bool(options))
    return EventOptions(
        prepend=bool(options.get("prepend", False)),
        global_=bool(options.get("global", options.get("global_", False))),
    )


type _DispatchMode = Literal["emit", "parallel", "serial", "bail", "waterfall"]
"""How an event is dispatched (the :class:`EventsService` method name)."""


def _event_name(name: object) -> str:
    """The event name ``name`` stands for (a ``str`` or an :class:`EventKey`)."""
    if isinstance(name, str):
        return name
    if isinstance(name, EventKey):
        return name.name
    raise TypeError(f"event name must be a str or EventKey, got {type(name).__name__}")


@final
class EventsService:
    """Event bus installed as ``ctx.events`` and exposed on every context.

    Supports ``emit`` (sync, fire-and-forget), ``parallel``, ``serial``,
    ``bail`` and ``waterfall`` dispatch. Listeners are owned by the fiber of
    the context that registered them and removed when it unloads.
    """

    __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx", no_shadow=True)

    ctx: Context
    """The context registrations are owned by."""
    hooks: dict[str, list[Hook]]
    """Registered hooks by event name, in dispatch order (shared by every view).

    Dispatch reads a cached snapshot of each list (:class:`Hook` is frozen):
    change the lists only through ``on`` / ``once`` and their disposers (or a
    fiber unload disposing them), which drop the snapshot. Appending to or
    removing from a list directly leaves dispatch on a stale snapshot.

    Internal — not part of the public API.
    """
    _plans: dict[str, _Plan]

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.hooks = {}
        self._plans = {}

        def on_listener(
            ctx: Context, name: str, listener: Listener, options: EventOptions
        ) -> EffectDisposer | None:
            if name == "internal/update" and not options.global_:
                # runs with the registering ctx current; a returned coroutine is
                # wrapped for that too (a different object than the listener's)
                hook = partial(call_bound, ctx, listener)
                return ctx.fiber.add_update_hook(hook, prepend=options.prepend)
            return None

        def on_update(fiber: Fiber, config: object, next: Callable[[], object]) -> object:
            callbacks = fiber.update_hooks()

            def _next() -> object:
                if not callbacks:
                    return next()
                return callbacks.pop(0)(fiber, config, _next)

            return _next()

        def on_config(fiber: Fiber, next: Callable[[], object]) -> object:
            config = next()
            runtime = fiber.runtime
            if not runtime:
                return config
            # cordis ``runtime?.Config``: an emitter may pass a stand-in runtime without one
            schema: object = getattr(runtime, "Config", None)
            return validate_config(schema, config)

        self.on("internal/listener", on_listener)
        self.on("internal/update", on_update, EventOptions(prepend=True, global_=True))
        self.on("internal/config", on_config)

    def _plan(self, name: str) -> _Plan:
        """The cached snapshot of ``name``'s hooks, built on first dispatch after a change."""
        plan = self._plans.get(name)
        if plan is None:
            hooks = self.hooks.get(name)
            if not hooks:
                return _NO_PLAN
            plan = self._plans[name] = _Plan.of(hooks)
        return plan

    def _target(self, mode: _DispatchMode, args: list[object]) -> tuple[object, str]:
        """Consume the receiver and event name from ``args``; announce ``internal/dispatch``.

        Raises ``TypeError`` when no event name is left (``emit()``, or a lone
        non-name such as ``emit(123)``, which is a bad name, not a receiver).
        """
        if not args:
            raise TypeError("an event name is required")
        receiver = not isinstance(args[0], str | EventKey) and len(args) > 1
        this_arg = args.pop(0) if receiver else None
        name = _event_name(args.pop(0))
        if not name.startswith("internal/") and self.hooks.get("internal/dispatch"):
            self.emit("internal/dispatch", mode, name, args, this_arg)
        return this_arg, name

    def _resolve(
        self, mode: _DispatchMode, args: list[object]
    ) -> tuple[object, list[tuple[Hook, Listener]]]:
        events = untraced(self)  # dispatch needs no use site: skip the proxy traps
        this_arg, name = events._target(mode, args)
        return this_arg, events._select(name, this_arg)

    def _select(self, name: str, this_arg: object) -> list[tuple[Hook, Listener]]:
        """The hooks of ``name`` that ``this_arg``'s filter admits, with their callbacks."""
        # a receiver context's filter (``Context.filter``), absent on other receivers
        filter_: Callable[[Context], object] | None = getattr(this_arg, symbols.filter, None)
        result: list[tuple[Hook, Listener]] = []
        for hook in self._plan(name).hooks:
            if hook.global_ or filter_ is None or filter_(hook.ctx):
                callback = hook.callback
                if name in RECEIVER_EVENTS:
                    callback = partial(callback, this_arg)
                result.append((hook, callback))
        return result

    def dispatch(self, mode: _DispatchMode, args: list[object]) -> list[Listener]:
        """Resolve the callbacks an event would reach, consuming receiver and name from ``args``.

        Used for per-listener error containment; each returned callable takes
        the remaining ``args``. The callables do not set the listener's context
        as current; use :meth:`dispatch_owned` to get each listener's owner.
        """
        return [callback for _, callback in self._resolve(mode, args)[1]]

    def warn_outlived(self, fiber: FiberCore) -> None:
        """Strict mode: warn about listeners of ``fiber`` still registered after its unload.

        Internal — not part of the public API.
        """
        for name, hooks in list(self.hooks.items()):
            for hook in list(hooks):
                if hook.ctx.fiber is fiber:
                    self.ctx.root.logger.warn(
                        f'listener for "{name}" outlived fiber <{fiber.name}>'
                    )

    def dispatch_owned(
        self, mode: _DispatchMode, args: list[object]
    ) -> list[tuple[Context, Listener]]:
        """Like :meth:`dispatch`, pairing each callback with the context that registered it.

        The owner's fiber is the one a coroutine result should be scheduled on.
        """
        return [(hook.ctx, callback) for hook, callback in self._resolve(mode, args)[1]]

    @overload
    def emit[**P, R](self, name: EventKey[P, R], /, *args: P.args, **kwargs: P.kwargs) -> None: ...
    @overload
    def emit(self, *args: object) -> None: ...
    def emit(self, *args: object, **kwargs: object) -> None:
        """Run listeners synchronously, ignoring their results.

        A listener returning a coroutine has it scheduled as a task owned by
        the listener's fiber; its failure is reported, not raised. In every
        dispatch mode a listener (and its coroutine) runs with the context that
        registered it current (:func:`~pycordis.current_context`).
        """
        rest = _positional(args, kwargs)
        events = untraced(self)  # dispatch needs no use site: skip the proxy traps
        this_arg, name = events._target("emit", rest)
        if getattr(this_arg, symbols.filter, None) is None and name not in RECEIVER_EVENTS:
            # the common case: every hook runs with the plain arguments
            _emit_runs(events._plan(name).runs, rest)
            return
        owner: Context | None = None
        with ContextSwitch() as switch:
            for hook, callback in events._select(name, this_arg):
                if hook.ctx is not owner:
                    owner = hook.ctx
                    switch.enter(owner)
                result = callback(*rest)
                if inspect.iscoroutine(result):
                    hook.ctx.fiber.spawn_owned(result, context=hook.ctx)

    @overload
    async def parallel[**P, R](
        self, name: EventKey[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> None: ...
    @overload
    async def parallel(self, *args: object) -> None: ...
    async def parallel(self, *args: object, **kwargs: object) -> None:
        """Run all listeners concurrently and wait for them.

        Raises ``ExceptionGroup("parallel listeners failed", errors)`` if any failed.
        """
        rest = _positional(args, kwargs)
        hooks = self._resolve("parallel", rest)[1]
        errors: list[BaseException] = []
        pending: list[Awaitable[object]] = []
        for hook, callback in hooks:
            try:
                result = call_in(hook.ctx, callback, *rest)
            except Exception as error:
                errors.append(error)
                continue
            if inspect.isawaitable(result):
                pending.append(await_in(hook.ctx, result))
        results = await asyncio.gather(*pending, return_exceptions=True)
        errors.extend(r for r in results if isinstance(r, BaseException))
        if errors:
            raise BaseExceptionGroup("parallel listeners failed", errors)

    @overload
    async def serial[**P, R](
        self, name: EventKey[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> Any: ...  # R may be an awaitable, whose result serial returns
    @overload
    async def serial(self, *args: object) -> Any: ...  # a string event: untyped result
    async def serial(self, *args: object, **kwargs: object) -> object:
        """Run listeners in order, awaiting each, until one returns a bail value."""
        rest = _positional(args, kwargs)
        for hook, callback in self._resolve("serial", rest)[1]:
            result = call_in(hook.ctx, callback, *rest)
            if inspect.isawaitable(result):
                result = await await_in(hook.ctx, result)
            if is_bailed(result):
                return result
        return None

    @overload
    def bail[**P, R](
        self, name: EventKey[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> R | None: ...
    @overload
    def bail(self, *args: object) -> Any: ...  # a string event: untyped result
    def bail(self, *args: object, **kwargs: object) -> object:
        """Run listeners synchronously until one returns a bail value, and return it."""
        rest = _positional(args, kwargs)
        owner: Context | None = None
        with ContextSwitch() as switch:
            for hook, callback in self._resolve("bail", rest)[1]:
                if hook.ctx is not owner:
                    owner = hook.ctx
                    switch.enter(owner)
                result = callback(*rest)
                if is_bailed(result):
                    return result
        return None

    @overload
    def waterfall[**P, R](
        self, name: EventKey[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> R: ...
    @overload
    def waterfall(self, *args: object) -> Any: ...  # a string event: untyped result
    def waterfall(self, *args: object, **kwargs: object) -> object:
        """Compose listeners around the final ``next`` argument.

        Listeners run outermost-first, each receiving the event arguments plus a
        ``next`` continuation; one that does not call ``next()`` vetoes the rest
        of the chain. Calling ``next()`` twice raises ``RuntimeError``.

        Each listener runs with the context that registered it current. A
        coroutine a listener returns is wrapped so its body also runs with that
        context when awaited: the caller (or the outer listener's ``next()``)
        receives a different coroutine object than the listener returned, and an
        un-awaited one is reported by Python as ``await_in`` never awaited.
        """
        rest = _positional(args, kwargs)
        hooks = self._resolve("waterfall", rest)[1]
        if not rest:
            raise TypeError("waterfall needs a final next argument")
        inner = rest.pop()

        def dispatch() -> object:
            if not hooks:
                if not callable(inner):
                    raise TypeError(f"{type(inner).__name__!r} object is not callable")
                return inner()
            hook, callback = hooks.pop(0)
            called = False

            def next() -> object:
                nonlocal called
                if called:
                    raise RuntimeError("next() called multiple times")
                called = True
                return dispatch()

            return call_bound(hook.ctx, callback, *rest, next)

        return dispatch()

    def _register(
        self, label: str, name: str, callback: Listener, options: EventOptions
    ) -> EffectDisposer:
        ctx = self.ctx  # the owner: the registering (use-site) context
        events = untraced(self)  # the shared state needs no use site

        def execute() -> Callable[[], bool]:
            hooks = events.hooks.setdefault(name, [])
            hook = Hook(ctx, callback, options.prepend, options.global_)
            if options.prepend:
                hooks.insert(0, hook)
            else:
                hooks.append(hook)
            events._plans.pop(name, None)
            return lambda: events.unregister(name, callback)

        return fiber_of(ctx).effect(execute, label)

    def unregister(self, name: str, callback: Listener) -> bool:
        """Remove the first hook for ``name`` whose callback is ``callback``; report success.

        Internal — not part of the public API (listener disposers call it).
        """
        hooks = self.hooks.get(name)
        if not hooks:
            return False
        for index, hook in enumerate(hooks):
            if hook.callback is callback:
                del hooks[index]
                if not hooks:
                    del self.hooks[name]
                self._plans.pop(name, None)
                return True
        return False

    @overload
    def on[**P, R](
        self, name: EventKey[P, R], listener: Callable[P, R], options: ListenerOptions = None
    ) -> ListenerDisposer: ...
    @overload
    def on[**P, R, F: Callable[..., object]](
        self, name: EventKey[P, R], listener: None = None, options: ListenerOptions = None
    ) -> Callable[[F], F]: ...
    @overload
    def on(
        self, name: str, listener: Listener, options: ListenerOptions = None
    ) -> ListenerDisposer: ...
    @overload
    def on[F: Callable[..., object]](
        self, name: str, listener: None = None, options: ListenerOptions = None
    ) -> Callable[[F], F]: ...
    def on(
        self,
        name: str | EventKey[Any, Any],  # a key of any signature
        listener: Listener | None = None,
        options: ListenerOptions = None,
    ) -> object:
        """Register a listener owned by the calling context's fiber.

        ``options`` may be a ``bool`` (``prepend``), :class:`EventOptions` or a
        dict. Returns a disposer that removes the listener and reports whether
        it was still registered. With ``listener`` omitted, returns a decorator
        that registers the function and returns it unchanged. Raises
        ``CordisError("INACTIVE_EFFECT")`` if the fiber is disposed.
        """
        if listener is None:
            return self._decorator(self.on, name, options)
        event = _event_name(name)
        opts = _normalize_options(options)
        ctx = self.ctx  # the registering (use-site) context
        fiber_of(ctx).assert_active()
        listener = ctx.reflect.bind(listener)
        result = untraced(self).bail(ctx, "internal/listener", event, listener, opts)
        if result:
            return ListenerDisposer(result) if isinstance(result, EffectDisposer) else result
        label = f"ctx.on({json.dumps(event)})"
        return ListenerDisposer(self._register(label, event, listener, opts))

    @overload
    def once[**P, R](
        self, name: EventKey[P, R], listener: Callable[P, R], options: ListenerOptions = None
    ) -> ListenerDisposer: ...
    @overload
    def once[**P, R, F: Callable[..., object]](
        self, name: EventKey[P, R], listener: None = None, options: ListenerOptions = None
    ) -> Callable[[F], F]: ...
    @overload
    def once(
        self, name: str, listener: Listener, options: ListenerOptions = None
    ) -> ListenerDisposer: ...
    @overload
    def once[F: Callable[..., object]](
        self, name: str, listener: None = None, options: ListenerOptions = None
    ) -> Callable[[F], F]: ...
    def once(
        self,
        name: str | EventKey[Any, Any],  # a key of any signature
        listener: Listener | None = None,
        options: ListenerOptions = None,
    ) -> object:
        """Like :meth:`on`, but the listener disposes itself before its first call."""
        if listener is None:
            return self._decorator(self.once, name, options)
        callback = listener
        dispose: Callable[[], object] | None = None

        def wrapper(*args: object) -> object:
            if dispose is not None:
                dispose()
            return callback(*args)

        dispose = self.on(name, wrapper, options)
        return dispose

    @staticmethod
    def _decorator(
        register: Callable[..., object],
        name: str | EventKey[Any, Any],  # a key of any signature
        options: ListenerOptions,
    ) -> Callable[[Listener], Listener]:
        def decorator(listener: Listener) -> Listener:
            register(name, listener, options)
            return listener

        return decorator


def _emit_runs(runs: tuple[tuple[Context, tuple[Listener, ...]], ...], args: list[object]) -> None:
    """Call each run's listeners with its owner current, then restore the caller's context.

    The owner is set with a direct ``CURRENT.set`` (resetting only the first
    token at the end, like :class:`~pycordis.current.ContextSwitch`): when every
    listener has its own owner, a method call per switch cost as much as the
    ``set`` itself. A coroutine result is scheduled as a task of the owner.
    """
    first: Token[Context] | None = None
    try:
        for owner, callbacks in runs:
            token = CURRENT.set(owner)
            if first is None:
                first = token
            for callback in callbacks:
                result = callback(*args)
                # inspect.iscoroutine is this isinstance check behind a call (one
                # per listener here); its type parameters are what isinstance
                # cannot infer
                if isinstance(result, CoroutineType):
                    coro = cast("CoroutineType[Any, Any, Any]", result)
                    owner.fiber.spawn_owned(coro, context=owner)
    finally:
        if first is not None:
            CURRENT.reset(first)


def _positional(args: tuple[object, ...], kwargs: dict[str, object]) -> list[object]:
    if kwargs:
        raise TypeError("event arguments must be positional")
    return list(args)
