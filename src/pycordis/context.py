"""Context: the scoped object every plugin receives (port of cordis ``context.ts``).

A JS context is a ``Proxy`` over a prototype chain; here a context keeps its own
attributes in ``_own`` and falls back to ``_parent`` for everything else, so
:meth:`Context.extend` is the analogue of ``Object.create(parent)``. That state
and the accessors other modules use for it live in :mod:`pycordis._context_internals`.

Typing: services are attributes resolved at runtime, so ``ctx.<name>`` and
``ctx.get("name")`` are ``Any``; :class:`~pycordis.keys.ServiceKey` gives them
a static type (``ctx.use(KEY)``). Events named by a plain string carry untyped
payloads and results for the same reason; an :class:`~pycordis.events.EventKey`
types them.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from types import MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    Protocol,
    Self,
    TypeGuard,
    final,
    overload,
)

from pycordis._context_internals import ContextCore, lookup
from pycordis._guards import MISSING, Missing
from pycordis._utils import Label, ScopeMap, symbols
from pycordis.effect import EffectDisposer, EffectResult
from pycordis.events import EventsService
from pycordis.fiber import Fiber
from pycordis.logger import LoggerService
from pycordis.reflect import ReflectService
from pycordis.registry import RegistryService
from pycordis.scope import Scope
from pycordis.timer import TimerService
from pycordis.trace import trace_cached

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from types import ModuleType, SimpleNamespace, TracebackType

    from pycordis.diagnose import Diagnosis
    from pycordis.effect import ListenerDisposer
    from pycordis.events import (
        EventKey,
        Listener,
        ListenerOptions,
    )
    from pycordis.inspect import FiberNode
    from pycordis.keys import ServiceKey, ServiceName
    from pycordis.registry import InjectSpec, Plugin, Runtime
    from pycordis.supervise import RestartPolicy

_INTERNAL = frozenset({"_own", "_parent", "_traced"})
_STRICT_OFF = frozenset({"", "0", "false", "off", "no"})


class _EffectMethod(Protocol):
    def __call__(self, execute: Callable[[], EffectResult], label: str = ...) -> EffectDisposer: ...


@final
class _Symbol[T]:
    """Class-level cordis symbol constant; instance lookups fall through to ``__getattr__``."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @overload
    def __get__(self, obj: None, owner: type) -> str: ...
    @overload
    def __get__(self, obj: object, owner: type | None = None) -> T: ...
    def __get__(self, obj: object, owner: type | None = None) -> str | T:
        if obj is None:
            return self.symbol
        # AttributeError from a descriptor makes Python try Context.__getattr__
        raise AttributeError(self.symbol)


@final
class _ScopeMethod[**P]:
    """``Context.isolate`` / ``Context.intercept``: a symbol on the class, a method on instances."""

    def __init__(self, symbol: str, func: Callable[Concatenate[Context, P], Context]) -> None:
        self.symbol = symbol
        self.func = func
        self.__doc__ = func.__doc__

    @overload
    def __get__(self, obj: None, owner: type) -> str: ...
    @overload
    def __get__[C: Context](self, obj: C, owner: type | None = None) -> Callable[P, C]: ...
    def __get__(self, obj: object, owner: type | None = None) -> str | Callable[..., object]:
        if obj is None:
            return self.symbol
        return MethodType(self.func, obj)


class Context(ContextCore):
    """Root and child dependency containers for plugins.

    ``Context()`` creates a root context (inside a running event loop); child
    contexts come from :meth:`extend`, :meth:`isolate` and :meth:`intercept` and
    never mutate their parent. ``async with Context() as root:`` disposes the root
    fiber on exit.

    Example::

        async with Context() as root:
            await root.plugin(server, ServerConfig(port=8080))
    """

    __cordis_context__: ClassVar[bool] = True

    def _isolate_method(self, name: str, label: Label | str | None = None) -> Self:
        """Create a child context with an independent service scope for ``name``.

        Below the returned context, the service ``name`` resolves against ``label``
        instead of the parent's scope. ``None`` creates a fresh label; a ``str`` label
        is interned per root, so two ``isolate()`` calls with the same string share a
        scope; a :class:`Label` is used as is.
        """
        if label is None:
            label = Label(name)
        elif isinstance(label, str):
            labels = self._labels
            label = labels.setdefault(label, Label(label))
        return self.extend(**{symbols.isolate: self._isolate.child(name, label)})

    def _intercept_method(self, name: str, config: object) -> Self:
        """Create a child context carrying intercept ``config`` for the service ``name``.

        Plugins loaded below the returned context see ``config`` merged into that
        service's config (ancestor entries first). The parent is not affected.
        """
        return self.extend(**{symbols.intercept: self._intercept.child(name, config)})

    effect = _Symbol[_EffectMethod](symbols.effect)
    """Attribute name under which a disposer exposes its ``EffectMeta`` tree.

    On instances, ``ctx.effect`` is the fiber's :meth:`~pycordis.fiber.Fiber.effect`
    bound to the use site: the effect is owned by ``ctx.fiber`` and its setup and
    disposers run with ``ctx`` itself as :func:`~pycordis.current_context`.
    """
    filter = _Symbol[Callable[..., bool]](symbols.filter)
    """Attribute name of a context's listener filter."""
    isolate = _ScopeMethod(symbols.isolate, _isolate_method)
    """Symbol of the isolate map on the class; :meth:`_isolate_method` on instances."""
    intercept = _ScopeMethod(symbols.intercept, _intercept_method)
    """Symbol of the intercept map on the class; :meth:`_intercept_method` on instances."""

    _isolate: ScopeMap[Label]
    _intercept: ScopeMap[object]
    _labels: dict[str, Label]
    fiber: Fiber
    """The fiber that owns this context."""
    reflect: ReflectService
    """The reflection layer behind service resolution (``ctx.get``, ``ctx.provide``, ...)."""
    registry: RegistryService
    """The plugin registry shared by every context of this root."""
    events: EventsService
    """The event bus shared by every context of this root."""
    logger: LoggerService
    """The logging service (``ctx.logger("name")``, ``ctx.logger.info(...)``)."""
    timer: TimerService
    """Fiber-owned timers (``ctx.timer.sleep(...)``, ``debounce``, ``throttle``, ...)."""
    runtime: Runtime | None
    """The plugin runtime of this context's fiber (``None`` at the root)."""

    if TYPE_CHECKING:
        # mixed-in members, resolved at runtime through ``ctx.reflect`` accessors

        @overload
        def get[T](self, name: ServiceKey[T], strict: bool = True) -> T | None: ...
        @overload
        def get(self, name: str, strict: bool = True) -> Any: ...  # an untyped service
        def get(self, name: ServiceName, strict: bool = True) -> object:
            """See :meth:`ReflectService.get`."""
            ...

        @overload
        def set[T](self, name: ServiceKey[T], value: T) -> bool: ...
        @overload
        def set(self, name: str, value: object) -> bool: ...
        def set(self, name: ServiceName, value: object) -> bool:
            """See :meth:`ReflectService.set`."""
            ...

        @overload
        def provide[T](
            self, name: ServiceKey[T], value: T, check: Callable[[], bool] | None = None
        ) -> EffectDisposer: ...
        @overload
        def provide(
            self, name: str, value: object = None, check: Callable[[], bool] | None = None
        ) -> EffectDisposer: ...
        def provide(
            self, name: ServiceName, value: object = None, check: Callable[[], bool] | None = None
        ) -> EffectDisposer:
            """See :meth:`ReflectService.provide`."""
            ...

        def accessor(
            self,
            name: str,
            *,
            get: Callable[[Context, object], object],
            set: Callable[[Context, object, object], bool] | None = None,
        ) -> EffectDisposer:
            """See :meth:`ReflectService.accessor`."""
            ...

        def mixin(
            self, source: str | object, mixins: Sequence[str] | Mapping[str, str]
        ) -> EffectDisposer:
            """See :meth:`ReflectService.mixin`."""
            ...

        def spawn[T](
            # Any: asyncio's own coroutine typing (yield and send types)
            self,
            coro: Coroutine[Any, Any, T],
            *,
            name: str | None = None,
        ) -> asyncio.Task[T]:
            """See :meth:`Fiber.spawn`; the task is owned by ``ctx.fiber`` and runs with
            ``ctx`` itself (the use site) as :func:`~pycordis.current_context`."""
            ...

        async def sleep(self, seconds: float) -> None:
            """See :meth:`TimerService.sleep`."""
            ...

        def set_timeout(self, callback: Callable[[], object], seconds: float) -> EffectDisposer:
            """See :meth:`TimerService.set_timeout`."""
            ...

        def set_interval(self, callback: Callable[[], object], seconds: float) -> EffectDisposer:
            """See :meth:`TimerService.set_interval`."""
            ...

        def inject(self, deps: InjectSpec, callback: Callable[[Context, None], object]) -> Fiber:
            """See :meth:`RegistryService.inject`."""
            ...

        @overload
        def plugin[C](
            self,
            plugin: Plugin[C],
            config: C | Mapping[str, object] | None = None,
            *,
            policy: RestartPolicy | None = None,
            load_timeout: float | None = None,
            dispose_timeout: float | None = None,
        ) -> Fiber: ...
        @overload
        def plugin(
            self,
            plugin: SimpleNamespace | ModuleType,
            config: Any = None,  # namespaces and modules declare no config type
            *,
            policy: RestartPolicy | None = None,
            load_timeout: float | None = None,
            dispose_timeout: float | None = None,
        ) -> Fiber: ...
        @overload
        def plugin(
            self,
            plugin: object,
            config: None = None,
            *,
            policy: RestartPolicy | None = None,
            load_timeout: float | None = None,
            dispose_timeout: float | None = None,
        ) -> Fiber: ...
        def plugin(
            self,
            plugin: object,
            config: object = None,
            *,
            policy: RestartPolicy | None = None,
            load_timeout: float | None = None,
            dispose_timeout: float | None = None,
        ) -> Fiber:
            """See :meth:`RegistryService.plugin`."""
            ...

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
        def on(self, *args: object, **kwargs: object) -> object:
            """See :meth:`EventsService.on`."""
            ...

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
        def once(self, *args: object, **kwargs: object) -> object:
            """See :meth:`EventsService.once`."""
            ...

        @overload
        def emit[**P, R](
            self, name: EventKey[P, R], /, *args: P.args, **kwargs: P.kwargs
        ) -> None: ...
        @overload
        def emit(self, *args: object) -> None: ...
        def emit(self, *args: object, **kwargs: object) -> None:
            """See :meth:`EventsService.emit`."""
            ...

        @overload
        async def parallel[**P, R](
            self, name: EventKey[P, R], /, *args: P.args, **kwargs: P.kwargs
        ) -> None: ...
        @overload
        async def parallel(self, *args: object) -> None: ...
        async def parallel(self, *args: object, **kwargs: object) -> None:
            """See :meth:`EventsService.parallel`."""
            ...

        @overload
        async def serial[**P, R](
            self, name: EventKey[P, R], /, *args: P.args, **kwargs: P.kwargs
        ) -> Any: ...  # R may be an awaitable, whose result serial returns
        @overload
        async def serial(self, *args: object) -> Any: ...  # a string event: untyped result
        async def serial(self, *args: object, **kwargs: object) -> object:
            """See :meth:`EventsService.serial`."""
            ...

        @overload
        def bail[**P, R](
            self, name: EventKey[P, R], /, *args: P.args, **kwargs: P.kwargs
        ) -> R | None: ...
        @overload
        def bail(self, *args: object) -> Any: ...  # a string event: untyped result
        def bail(self, *args: object, **kwargs: object) -> object:
            """See :meth:`EventsService.bail`."""
            ...

        @overload
        def waterfall[**P, R](
            self, name: EventKey[P, R], /, *args: P.args, **kwargs: P.kwargs
        ) -> R: ...
        @overload
        def waterfall(self, *args: object) -> Any: ...  # a string event: untyped result
        def waterfall(self, *args: object, **kwargs: object) -> object:
            """See :meth:`EventsService.waterfall`."""
            ...

    def __init__(self, *, strict: bool | None = None) -> None:
        """Create a root context.

        ``strict`` enables debug warnings for services read through disposed
        contexts and for listeners that outlive their fiber; ``None`` reads the
        ``PYCORDIS_STRICT`` environment variable (on unless empty, ``0``, ``false``,
        ``off`` or ``no``, case-insensitive).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "pycordis.Context must be created inside a running asyncio event loop"
            ) from None
        self._attach({}, None)
        self._isolate = ScopeMap[Label].empty()
        self._intercept = ScopeMap[object].empty()
        self._labels = {}
        self.fiber = Fiber(self, {}, {}, None)
        if strict is None:
            strict = os.environ.get("PYCORDIS_STRICT", "").strip().lower() not in _STRICT_OFF
        self.reflect = ReflectService(self, strict)
        self.registry = RegistryService(self)
        self.events = EventsService(self)
        self.logger = LoggerService(self)
        self.timer = TimerService(self)
        self.fiber.release_disposables()

    def use[T](self, key: ServiceKey[T]) -> T:
        """Read the service named by ``key``, typed by the key.

        Resolves exactly like attribute access, ``ctx.use(LLM)`` is ``ctx.llm``:
        the same inject requirement applies and the same
        :class:`~pycordis.errors.ServiceAccessError` is raised. Example::

            LLM = ServiceKey[LLMClient]("llm")

            @plugin_meta(inject=[LLM])
            def chat(ctx: Context, config: None) -> None:
                client = ctx.use(LLM)  # LLMClient
        """
        service: T = getattr(self, key.name)
        return service

    def scope(
        self,
        *,
        name: str | None = None,
        inject: InjectSpec | None = None,
        provide: ServiceName | Sequence[ServiceName] = (),
        config: object = None,
        timeout: float | None = None,
    ) -> Scope:
        """Run a block inside a short-lived plugin fiber (a per-request lifetime).

        Entering starts an anonymous plugin named ``name`` (default ``"scope"``)
        under this context with ``inject`` and ``config``, waits until it is
        ACTIVE and binds its context; a startup error is re-raised, and waiting
        longer than ``timeout`` seconds raises ``TimeoutError`` (activation takes
        at least one event-loop turn, so ``timeout=0`` always does). Leaving the
        block disposes the fiber, also on an exception or cancellation, and lets
        the block's exception propagate unchanged; disposal errors are reported
        through the logger. Each name in ``provide`` is isolated to the scope, so
        what the scope provides under it is invisible to this context and to
        concurrent scopes. If this context unloads mid-block, the block keeps
        running with a disposed scope context whose new effects raise
        ``CordisError("INACTIVE_EFFECT")`` (see :mod:`pycordis.scope`). Example::

            async with ctx.scope(name="session-42", inject=[LLM], provide=[MEMORY]) as s:
                s.provide(MEMORY, ConversationMemory())
                await s.plugin(chat_agent)
        """
        return Scope(
            self, name=name, inject=inject, provide=provide, config=config, timeout=timeout
        )

    def inspect(self) -> FiberNode:
        """Snapshot the fiber tree below this context's fiber (:mod:`pycordis.inspect`).

        Example: ``print(ctx.inspect())``.
        """
        from pycordis.inspect import inspect_fiber

        return inspect_fiber(self.fiber)

    def diagnose(self) -> Diagnosis:
        """Explain why each fiber below this context's fiber is not ``ACTIVE``.

        See :mod:`pycordis.diagnose` for the reasons reported. Example::

            print(render_diagnosis(ctx.diagnose()))
        """
        return self.fiber.diagnose()

    @staticmethod
    def is_(value: object) -> TypeGuard[Context]:
        """Return True for contexts, by class marker rather than ``isinstance``.

        Example: ``Context.is_(ctx)`` is ``True``; ``Context.is_(ctx.logger)`` is ``False``.
        """
        return getattr(value, "__cordis_context__", False) is True

    @property
    def root(self) -> Self:
        """The root context of the application.

        Example: ``ctx.root.logger.info("from the root")``.
        """
        node = self
        while node._parent is not None:
            node = node._parent
        return node

    def extend(self, **meta: object) -> Self:
        """Create a child context; ``meta`` entries become its own attributes.

        Extending a shadow context (see :mod:`pycordis.trace`) extends its use site
        and re-applies the shadow on top, so the def site is kept. Example:
        ``child = ctx.extend(request_id="r-1")``.
        """
        shadow = self._own.get(symbols.shadow)
        # cordis: `Object.create(getTraceable(this, this))`
        base = self if shadow is None else self._parent
        if base is None:
            raise RuntimeError("a shadow context must extend its use site")
        child = base._child(meta)
        if shadow is None:
            return child
        return child._child({symbols.shadow: shadow})

    def __getattr__(self, name: str) -> Any:  # a service or attribute of any type
        if name in _INTERNAL:
            raise AttributeError(name)
        value = lookup(self, name, MISSING)
        if value is not MISSING:
            # private names bypass tracing (cordis special properties)
            return value if name.startswith("_") else trace_cached(self, name, value)
        reflect: ReflectService | Missing = lookup(self, "reflect", MISSING)
        if name.startswith("_") or reflect is MISSING:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return reflect.handle_get(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in _INTERNAL:
            object.__setattr__(self, name, value)
            return
        if not name.startswith("_"):
            reflect: ReflectService | Missing = lookup(self, "reflect", MISSING)
            if reflect is not MISSING and reflect.handle_set(self, name, value):
                return
        self._own[name] = value

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        # class symbol constants (``Context.filter``) are not instance members
        static: object = inspect.getattr_static(type(self), name, MISSING)
        if static is not MISSING and not isinstance(static, _Symbol):
            return True
        if lookup(self, name, MISSING) is not MISSING:
            return True
        if name.startswith("_"):
            return False
        reflect: ReflectService | Missing = lookup(self, "reflect", MISSING)
        return reflect is not MISSING and reflect.handle_has(name)

    def __getitem__(self, name: str) -> Any:  # like attribute access: any service
        """``ctx["foo.bar"]``: attribute resolution for names that are not identifiers."""
        return getattr(self, name)

    def __repr__(self) -> str:
        return f"Context <{self.fiber.name}>"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.fiber.dispose()
