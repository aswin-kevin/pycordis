"""Reflect service: service resolution behind every context (port of cordis ``reflect.ts``).

cordis wraps each context in a ``Proxy`` whose traps live in
``ReflectService.handler``. Here :class:`~pycordis.context.Context` keeps its own
attributes and delegates the remaining reads, writes and ``in`` checks to
:meth:`ReflectService.handle_get`, :meth:`ReflectService.handle_set` and
:meth:`ReflectService.handle_has`.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial, wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, final, overload

from pycordis._cascade import Cascade, Steps
from pycordis._context_internals import fiber_of, isolate_map, lookup, registry_of
from pycordis._utils import Label, resolve_future, symbols
from pycordis.effect import EffectDisposer
from pycordis.errors import ServiceAccessError
from pycordis.fiber import FiberState
from pycordis.keys import ServiceKey, ServiceName, service_name
from pycordis.trace import (
    Tracker,
    get_traceable,
    get_with_receiver,
    set_with_receiver,
    trace_cached,
    with_props,
)
from pycordis.trace_attrs import ATOMIC

if TYPE_CHECKING:
    from pycordis.context import Context
    from pycordis.events import EventsService
    from pycordis.fiber_lifecycle import FiberLifecycle


@final
@dataclass(frozen=True)
class ServiceProperty:
    """A context property backed by a provided service implementation."""

    type: Literal["service"] = "service"


@final
@dataclass(frozen=True, kw_only=True)
class AccessorProperty:
    """A computed context property backed by ``get`` / ``set`` hooks.

    ``get(ctx, receiver)`` computes the value for the reading context;
    ``set(ctx, value, receiver)`` stores it and returns ``False`` to reject the
    write. Without ``set`` the property is read-only.
    """

    type: Literal["accessor"] = "accessor"
    get: Callable[[Context, object], object]
    set: Callable[[Context, object, object], bool] | None = None


type Property = ServiceProperty | AccessorProperty
"""A context property definition known by the reflect service."""


@final
@dataclass(eq=False, slots=True)
class Impl:
    """A service implementation record stored in the reflect service.

    ``isolate`` is the actual isolation label it is stored under; ``dependents``
    are fibers retaining it until their teardown finishes, and ``release`` wakes
    the provision disposer once the last of them has unloaded.
    """

    uid: int
    name: str
    ctx: Context
    isolate: Label
    dependents: set[FiberLifecycle]
    value: object = None
    check: Callable[[], bool] | None = None
    release: Callable[[], None] | None = None
    key: ServiceKey[Any] | None = None  # a key of any service type
    """The :class:`~pycordis.keys.ServiceKey` it was provided under, if any."""


def _collision_message(existing: Impl, impl: Impl) -> str:
    """The error for providing ``impl`` where ``existing`` is already registered.

    When a key is involved, both providers are named with what they provided
    under: keys sharing a name share one service slot, whatever their types.
    """
    message = f'service "{impl.name}" has been registered at <{existing.ctx.fiber.name}>'
    if existing.key is None and impl.key is None:
        return message

    def provider(entry: Impl) -> str:
        via = repr(entry.key) if entry.key is not None else f'"{entry.name}"'
        return f"{via} at <{entry.ctx.fiber.name}>"

    return f"{message} (registered by {provider(existing)}, then provided by {provider(impl)})"


@final
class _GetFailed(Exception):
    """A failed service lookup on the hot path (turned into a :class:`ServiceAccessError`)."""

    __slots__ = ("message",)

    def __init__(self, message: str | None) -> None:
        self.message = message


def _get_error(ctx: Context, name: str, message: str | None = None) -> ServiceAccessError:
    return ServiceAccessError(
        message or f'cannot get property "{name}" without inject', name=name, obj=ctx
    )


def _resolve(
    ctx: Context, name: str, def_site: Context, error: ServiceAccessError | None
) -> Any:  # a service of any type
    """Walk the def site's fibers for ``name`` (the default ``internal/get`` behaviour).

    Failures raise ``error`` (with its message updated) when given, else :class:`_GetFailed`.
    """

    def fail(message: str | None = None) -> Exception:
        if error is None:
            return _GetFailed(message)
        if message is not None:
            error.args = (message,)
        return error

    key = isolate_map(def_site).get(name)
    fiber = fiber_of(def_site)
    while True:
        committed = fiber.committed
        impl: Impl | None = committed.get(name) if committed is not None else None
        for provision in fiber.provisions:
            if provision.name == name and isolate_map(provision.ctx).get(name) is key:
                impl = provision
        if impl is not None:
            return trace_cached(ctx, name, impl.value)
        if name in fiber.inject:
            raise fail(f'cannot get required service "{name}" in inactive context')
        if not fiber.runtime:
            raise fail()
        if isolate_map(fiber.parent).get(name) is not key:
            raise fail()
        fiber = fiber_of(fiber.parent)


@final
class ReflectService:
    """Reflection and service-resolution layer installed as ``ctx.reflect``.

    Holds every provided service and declared context property, and mixes core
    service methods onto contexts (``ctx.provide``, ``ctx.on``, ``ctx.effect``,
    ...). Reading it through a context yields a traced proxy whose :attr:`ctx`
    is that context, sharing the same store.
    """

    __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx", no_shadow=True)

    ctx: Context
    """The context this view acts for."""
    store: dict[Label, Impl]
    """Service implementations, keyed by isolation label."""
    props: dict[str, Property]
    """Declared context properties (services and accessors), by name."""
    strict: bool
    """Strict debug mode: warn when a service is read through a disposed context."""
    _uids: Iterator[int]
    _cascade: Cascade

    def __init__(self, ctx: Context, strict: bool = False) -> None:
        self.ctx = ctx
        self.store = {}
        self.props = {}
        self.strict = strict
        self._uids = itertools.count(1)
        self._cascade = Cascade()

        self.mixin("reflect", ["get", "set", "provide", "accessor", "mixin"])
        self.mixin("fiber", ["runtime"])
        # bound to the reading context, so effects and tasks see the use site
        # (e.g. an isolate()-d view) as current, like listeners and timers do
        self.accessor("effect", get=_use_site_effect)
        self.accessor("spawn", get=_use_site_spawn)
        self.mixin("timer", ["sleep", "set_timeout", "set_interval"])
        self.mixin("registry", ["inject", "plugin"])
        self.mixin("events", ["on", "once", "parallel", "emit", "serial", "bail", "waterfall"])

    # -- context traps -----------------------------------------------------

    def handle_get(self, ctx: Context, name: str) -> Any:  # a service of any type
        """Resolve a public name that is not an own or inherited attribute of ``ctx``.

        Internal — not part of the public API (the ``Context.__getattr__`` trap).
        """
        # hot path: no traced proxies, and the error is built only on failure
        prop = self.props.get(name)
        if isinstance(prop, AccessorProperty):
            return prop.get(ctx, lookup(ctx, symbols.receiver))
        if self.strict and prop is not None and ctx.fiber.state == FiberState.DISPOSED:
            ctx.logger.warn(f'service "{name}" read from disposed context <{ctx.fiber.name}>')

        shadow = lookup(ctx, symbols.shadow)
        def_site: Context = ctx if shadow is None else shadow
        # a fiber-less def site cannot declare `inject`, so it keeps unchecked root access
        if not fiber_of(def_site).runtime:
            # pycordis: an undeclared name is an AttributeError rather than None,
            # so hasattr() / getattr(ctx, name, default) keep their meaning
            if prop is None:
                raise _get_error(ctx, name) from None
            impl = self._get_impl_in(ctx, name, False)
            return trace_cached(ctx, name, None if impl is None else impl.value)

        events: EventsService = lookup(ctx, "events")
        if not events.hooks.get("internal/get"):
            try:
                return _resolve(ctx, name, def_site, None)
            except _GetFailed as failed:
                raise _get_error(ctx, name, failed.message) from None

        error = _get_error(ctx, name)
        try:
            return ctx.events.waterfall(
                "internal/get", ctx, name, error, lambda: _resolve(ctx, name, def_site, error)
            )
        except ServiceAccessError as e:
            if e is error:
                raise error.with_traceback(None) from None
            raise

    def handle_set(self, ctx: Context, name: str, value: object) -> bool:
        """Write a public name; return ``False`` when it should become an own attribute.

        Internal — not part of the public API (the ``Context.__setattr__`` trap).
        """
        error = ServiceAccessError(
            f'cannot set property "{name}" without provide', name=name, obj=ctx
        )
        prop = self.props.get(name)
        if prop is None:
            if not ctx.fiber.runtime:
                return False
            raise error
        try:
            if isinstance(prop, AccessorProperty):
                if prop.set is None or not prop.set(ctx, value, lookup(ctx, symbols.receiver)):
                    raise ServiceAccessError(f'cannot set property "{name}"', name=name, obj=ctx)
                return True
            result = ctx.events.waterfall(
                "internal/set", ctx, name, value, error, lambda: ctx.reflect.set(name, value)
            )
            if not result:
                # cordis: a falsy `set` trap result throws in strict mode
                raise ServiceAccessError(f'cannot set property "{name}"', name=name, obj=ctx)
            return True
        except ServiceAccessError as e:
            if e is error:
                raise error.with_traceback(None) from None
            raise

    def handle_has(self, name: str) -> bool:
        """Whether ``name`` is a declared context property.

        Internal — not part of the public API (the ``Context.__contains__`` trap).
        """
        return name in self.props

    # -- public API --------------------------------------------------------

    @overload
    def get[T](self, name: ServiceKey[T], strict: bool = True) -> T | None: ...
    @overload
    def get(self, name: str, strict: bool = True) -> Any: ...  # an untyped service
    def get(self, name: ServiceName, strict: bool = True) -> object:
        """Read a service without the inject requirement.

        With ``strict`` (the default) only implementations whose providing fiber
        is active are returned. Returns ``None`` when the service is not provided.
        A :class:`~pycordis.keys.ServiceKey` types the result: ``ctx.get(LLM)`` is
        ``LLMClient | None``.
        """
        name = service_name(name)
        impl = self.get_impl(name, strict)
        # through the cache even when absent, so a withdrawn service's proxy is dropped
        return trace_cached(self.ctx, name, None if impl is None else impl.value)

    def get_impl(self, name: str, strict: bool = True) -> Impl | None:
        """The implementation record behind :meth:`get` (``None`` when not provided).

        Internal — not part of the public API.
        """
        return self._get_impl_in(self.ctx, name, strict)

    def _get_impl_in(self, ctx: Context, name: str, strict: bool) -> Impl | None:
        key = isolate_map(ctx).get(name)
        impl = None if key is None else self.store.get(key)
        if impl is None:
            return None
        if strict and fiber_of(impl.ctx).state != FiberState.ACTIVE:
            return None
        return impl

    @overload
    def set[T](self, name: ServiceKey[T], value: T) -> bool: ...
    @overload
    def set(self, name: str, value: object) -> bool: ...
    def set(self, name: ServiceName, value: object) -> bool:
        """Overwrite a provided service's value.

        Raises :class:`ServiceAccessError` when ``name`` was never provided in
        this scope, or was provided by another fiber. With a
        :class:`~pycordis.keys.ServiceKey`, ``value`` is type-checked against it.
        """
        name = service_name(name)
        key = isolate_map(self.ctx).get(name)
        impl = None if key is None else self.store.get(key)
        if impl is None:
            raise ServiceAccessError(
                f'cannot set property "{name}" without provide', name=name, obj=self.ctx
            )
        if impl.ctx.fiber is not self.ctx.fiber:
            raise ServiceAccessError(
                f'cannot set property "{name}" in multiple fibers', name=name, obj=self.ctx
            )
        impl.value = value
        return True

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
        """Register a service implementation owned by the current fiber.

        The service becomes visible in this context's isolation scope and is
        unregistered (waking dependents) when the returned disposer runs or the
        fiber unloads. ``check`` is an optional availability predicate consulted
        before dependents load. Raises :class:`ServiceAccessError` if the name is
        already provided in this scope or declared as an accessor. With a
        :class:`~pycordis.keys.ServiceKey`, ``value`` is type-checked against it::

            ctx.provide(LLM, OpenAIClient())
        """
        service_key = name if isinstance(name, ServiceKey) else None
        if service_key is not None:
            value = service_key.coerce(value)
        name = service_name(name)
        ctx = self.ctx

        def execute() -> Callable[[], object]:
            prop = self.props.get(name)
            if prop is not None and prop.type != "service":
                raise ServiceAccessError(
                    f'property "{name}" is already declared as {prop.type}', name=name, obj=self.ctx
                )
            self.props[name] = ServiceProperty()

            root_isolate = isolate_map(ctx.root)
            if not root_isolate.has_own(name):
                root_isolate.set_root(name, Label(name))
            key = isolate_map(ctx)[name]
            impl = Impl(next(self._uids), name, ctx, key, set(), value, check, key=service_key)
            existing = self.store.get(key)
            if existing is not None:
                raise ServiceAccessError(_collision_message(existing, impl), name=name, obj=ctx)
            self.store[key] = impl
            ctx.fiber.provisions.add(impl)
            if ctx.fiber.state == FiberState.ACTIVE:
                self.notify([name])

            # cordis runs this async disposer synchronously up to its first
            # await: the binding is retired and dependents are notified before
            # the caller regains control
            # NOTE: the returned coroutine must always be awaited (every
            # disposer runner does); dropping it skips the provision removal
            # and warns "coroutine was never awaited"
            def dispose() -> Coroutine[Any, Any, None]:  # asyncio's coroutine typing
                if self.store.get(impl.isolate) is impl:
                    del self.store[impl.isolate]
                ctx.isolate(name, impl.isolate).reflect.notify([name])
                # Retired bindings cannot gain new dependents. Wait only for their
                # cleanup, even if a fiber has left the registry or starts a new load.
                # Yield even without direct dependents so child cleanup can still
                # access the inherited service before its provision is removed.
                released: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                impl.release = partial(resolve_future, released)
                if not impl.dependents:
                    resolve_future(released)
                return settle(released)

            async def settle(released: asyncio.Future[None]) -> None:
                if released.done():
                    await asyncio.sleep(0)
                else:
                    await released
                impl.release = None
                # ensure self access before dependencies cleanup
                ctx.fiber.provisions.discard(impl)

            return dispose

        return ctx.fiber.effect(execute, f"ctx.provide({json.dumps(name)})")

    def notify(
        self, names: list[str], filter: Callable[[Context, str], bool] | None = None
    ) -> None:
        """Re-evaluate every fiber requiring one of ``names`` and emit ``internal/service``.

        ``filter(ctx, name)`` restricts notification to matching isolation scopes;
        by default, scopes sharing this context's label for ``name``.

        A dependent that leaves ``ACTIVE`` notifies its own dependents in turn;
        that cascade runs on a :class:`~pycordis._cascade.Cascade` rather than
        the call stack, so a dependency chain of any length can be torn down.
        Called from inside a cascade (e.g. ``ctx.provide`` in an
        ``internal/status`` listener while a teardown cascades), the
        notification is deferred: it runs right after the current step (the
        dependent being refreshed, or the ``internal/service`` emit), before the
        cascade continues, and the call returns before it ran. An exception it
        raises therefore surfaces from the outermost cascade (the call that
        started it, e.g. ``fiber.dispose()``), not from the call that notified.
        """
        self._cascade.run(self._notify_steps(names, filter))

    def _notify_steps(
        self, names: list[str], filter: Callable[[Context, str], bool] | None
    ) -> Steps:
        """The work of :meth:`notify`; it yields after each point that can notify again."""
        if filter is None:
            own = isolate_map(self.ctx)

            def same_scope(ctx: Context, name: str) -> bool:
                return isolate_map(ctx).get(name) is own.get(name)

            filter = same_scope
        scope = filter

        # absent until the registry service is installed; read untraced (no
        # use site needed), as this loop runs once per notification
        registry = registry_of(self.ctx)
        if registry is not None:
            # snapshot: refresh_epoch() emits internal/status synchronously, and an
            # observer may load or dispose plugins while we iterate
            for runtime in list(registry.values()):
                for fiber in runtime.fibers.snapshot():
                    has_update = False
                    for name in names:
                        if name not in fiber.inject:
                            continue
                        if not scope(fiber.ctx, name):
                            continue
                        has_update = True
                        fiber.check_impl(name)
                    if not has_update:
                        continue
                    fiber.refresh_epoch()
                    yield  # the fiber's own cascade runs here

        for name in names:

            def receiver_filter(target: Context, name: str = name) -> bool:
                return scope(target, name)

            receiver = self.ctx.extend(**{symbols.filter: receiver_filter})
            impl = self.get_impl(name, False)
            self.ctx.events.emit(
                receiver, "internal/service", name, None if impl is None else impl.value
            )
            yield  # notifications its listeners started run here

    def accessor(
        self,
        name: str,
        *,
        get: Callable[[Context, object], object],
        set: Callable[[Context, object, object], bool] | None = None,
    ) -> EffectDisposer:
        """Define a computed context property backed by ``get`` / ``set`` hooks.

        The accessor is removed when the returned disposer runs or the current
        fiber unloads. Raises :class:`ServiceAccessError` if ``name`` is already
        declared.
        """

        def execute() -> Callable[[], None]:
            prop = self.props.get(name)
            if prop is not None:
                raise ServiceAccessError(
                    f'property "{name}" is already declared as {prop.type}', name=name, obj=self.ctx
                )
            self.props[name] = AccessorProperty(get=get, set=set)

            def dispose() -> None:
                self.props.pop(name, None)

            return dispose

        return self.ctx.fiber.effect(execute, f"ctx.accessor({json.dumps(name)})")

    def mixin(
        self, source: str | object, mixins: Sequence[str] | Mapping[str, str]
    ) -> EffectDisposer:
        """Expose selected members of a service directly on contexts.

        ``source`` is a context property name (e.g. ``"events"``) or an object;
        ``mixins`` lists member names, or maps member name to context name. Each
        becomes an accessor forwarding to the source as seen from the reading
        context, so ``ctx.on`` is ``ctx.events.on``. Removed when the returned
        disposer runs or the current fiber unloads.
        """
        if isinstance(mixins, Mapping):
            entries = list(mixins.items())
        else:
            entries = [(key, key) for key in mixins]

        def get_target(ctx: Context) -> object:
            if isinstance(source, str):
                return getattr(ctx, source)
            return get_traceable(ctx, source)

        def forward(key: str) -> AccessorProperty:
            def get(ctx: Context, receiver: object) -> object:
                service = get_target(ctx)
                if service is None:
                    return None
                if receiver is None:
                    return getattr(service, key)
                # members of the service win; the rest resolve on the traced receiver
                return get_with_receiver(service, key, with_props(receiver, service))

            def set_(ctx: Context, value: object, receiver: object) -> bool:
                service = get_target(ctx)
                if receiver is None:
                    setattr(service, key, value)
                else:
                    set_with_receiver(service, key, value, with_props(receiver, service))
                return True

            return AccessorProperty(get=get, set=set_)

        def execute() -> Iterator[EffectDisposer]:
            for key, alias in entries:
                prop = forward(key)
                yield self.accessor(alias, get=prop.get, set=prop.set)

        label = json.dumps(source) if isinstance(source, str) else type(source).__name__
        return self.ctx.fiber.effect(execute, f"ctx.mixin({label})")

    def trace[T](self, value: T) -> T:
        """Return ``value`` as seen from this context (:func:`~pycordis.trace.get_traceable`)."""
        return get_traceable(self.ctx, value)

    def bind[F: Callable[..., object]](self, callback: F) -> F:
        """Wrap ``callback`` so its ``self`` and arguments are traced to this context.

        A bound method is rebound to the traced view of its ``self``; positional and
        keyword arguments are traced on every call. The original callback is
        available as ``__wrapped__``.
        """
        ctx = self.ctx
        func: Callable[..., object] = callback
        if isinstance(callback, MethodType):
            func = MethodType(callback.__func__, get_traceable(ctx, callback.__self__))

        # name, docstring and ``__wrapped__``, but not ``__dict__``: a callable
        # object's own attributes are not copied onto the wrapper
        @wraps(callback, updated=())
        def bound(*args: object, **kwargs: object) -> object:
            # every listener call goes through here: a list comprehension is
            # cheaper than unpacking a generator, and listeners rarely get keywords.
            # Plain values (get_traceable's own first check) skip the call.
            traced = [arg if type(arg) in ATOMIC else get_traceable(ctx, arg) for arg in args]
            if not kwargs:
                return func(*traced)
            return func(*traced, **{key: get_traceable(ctx, v) for key, v in kwargs.items()})

        if inspect.iscoroutinefunction(callback):
            inspect.markcoroutinefunction(bound)
        # ``wraps`` types the wrapper by its own signature; it forwards to ``callback``
        return cast(F, bound)


def _use_site_effect(ctx: Context, _receiver: object) -> Callable[..., EffectDisposer]:
    """``ctx.effect``: the fiber's :meth:`~pycordis.fiber.Fiber.effect`, use site current."""
    site = get_traceable(ctx, ctx)  # a service's shadow context resolves to its use site
    return partial(site.fiber.effect, context=site)


def _use_site_spawn(
    ctx: Context, _receiver: object
) -> Callable[..., asyncio.Task[Any]]:  # tasks of any result type
    """``ctx.spawn``: the fiber's :meth:`~pycordis.fiber.Fiber.spawn`, use site current."""
    site = get_traceable(ctx, ctx)  # a service's shadow context resolves to its use site
    return partial(site.fiber.spawn, context=site)
