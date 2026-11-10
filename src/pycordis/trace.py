"""Traceable service proxies (port of cordis ``getTraceable`` and friends in ``utils.ts``).

A service read through a context is wrapped in a :class:`Traceable` so that its
methods see the *caller's* context. A context ``ctx`` handed to
:func:`get_traceable` may be a *shadow*, i.e. a pair of:

- def site: where the accessing code was defined, governs service resolution;
- use site: where the service is consumed, governs intercept, isolate and effects.

A shadow context is a child of the use site carrying the def site as its own
``_shadow`` attribute (cordis ``symbols.shadow``).

Python differences from the JS ``Proxy`` original:

- JS methods receive ``this`` at call time; Python binds ``self`` when the
  attribute is read, so methods read through a proxy come back already bound to
  the right receiver (the proxy, or a shadow receiver whose tracker property is
  a fresh shadow context).
- Only functions defined on the target's class are rebound. Callables stored on
  the instance and ``staticmethod`` / ``classmethod`` / other non-data
  descriptors are returned as their normal attribute value.
- Re-tracing a proxy re-wraps its original target instead of nesting proxies.
- Python looks special methods up on the type, so each proxy is an instance of a
  per-target-type subclass (cached weakly by target type, see
  :mod:`pycordis.trace_special`) that defines exactly
  the special methods the target's type overrides from ``object``: calling,
  context managers (sync and async), awaiting, (async) iteration, containers,
  ``__eq__`` / ``__hash__``, ordering, ``__bool__``, numeric operators and
  conversions. Each forwards to the target with the traced receiver, so
  ``callable(proxy)`` and ``isinstance(proxy, Iterable)`` match the target.
  ``__repr__``, ``__str__`` and ``__dir__`` always show the target. Omitted on
  purpose: descriptor hooks (``__get__`` / ``__set__`` / ``__set_name__``),
  class-level hooks (``__init_subclass__``, ``__class_getitem__``), pickling and
  copying hooks, ``__sizeof__`` and the buffer protocol. In-place operators
  return the target's result, which is not re-wrapped.

Typing: a proxy forwards arbitrary attribute reads, writes and calls to a target
of any type, so the target (:attr:`_State.target`), forwarded values and the
attribute-protocol methods (``__getattribute__``, ``__setattr__``, calls) are
``Any`` here: they are exactly as dynamic as the service behind the proxy.
:func:`get_traceable` and :func:`untraced` keep the caller's static type.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any, TypeGuard, cast, final

from pycordis._context_internals import lookup, own_meta, parent_of, traced_cache
from pycordis._guards import is_dict
from pycordis._utils import symbols
from pycordis.errors import ServiceAccessError
from pycordis.trace_attrs import (
    ATOMIC,
    MISSING,
    class_attr,
    cls_of,
    instance_dict,
    is_context,
    is_data_descriptor,
    is_dunder,
    static_value,
)
from pycordis.trace_special import special_type

if TYPE_CHECKING:
    from pycordis.context import Context
    from pycordis.reflect import ReflectService

__all__ = [
    "Traceable",
    "Tracker",
    "get_traceable",
    "get_with_receiver",
    "invoke",
    "set_with_receiver",
    "trace_cached",
    "with_props",
]


@final
@dataclass(frozen=True)
class Tracker:
    """Opt-in tracing metadata, stored on a value as ``__cordis_tracker__``.

    ``property`` names the attribute holding the value's own context (``"ctx"``);
    reading it through a proxy yields the use site. ``associate`` makes
    ``proxy.x`` read the context property ``f"{associate}.x"`` when declared.
    ``no_shadow`` keeps ``self`` inside methods as the proxy itself instead of a
    shadow receiver.
    """

    property: str | None = None
    associate: str | None = None
    no_shadow: bool = False


def _tracker_of(value: object) -> Tracker | None:
    if type(value) in ATOMIC or isinstance(value, type):
        return None
    try:
        tracker = object.__getattribute__(value, symbols.tracker)
    except AttributeError:
        return None
    return tracker if isinstance(tracker, Tracker) else None


@final
@dataclass(frozen=True, slots=True)
class _State:
    target: Any  # the proxied service, of any type (see the module docstring)
    ctx: Context
    tracker: Tracker
    def_site: Context
    use_site: Context


def _state(proxy: Traceable) -> _State:
    return object.__getattribute__(proxy, "_cordis_state")


class _Forwarding:
    """Special methods shared by :class:`Traceable` and :class:`_WithProps`."""

    __slots__ = ()

    def __repr__(self) -> str:
        return repr(_original(self))

    def __str__(self) -> str:
        return str(_original(self))

    def __dir__(self) -> list[str]:
        return dir(_original(self))


class Traceable(_Forwarding):
    """Transparent proxy over a service, as seen from one context.

    Reading the tracker property (``.ctx``) returns the use site;
    ``_cordis_original`` returns the target and ``_cordis_caller`` the def site.
    Methods of the target come back bound so that ``self`` is the proxy
    (``no_shadow``) or a shadow receiver whose ``ctx`` carries both sites
    (cordis ``createShadowMethod``); their results are traced to the use site.
    ``__class__`` is the target's class, so ``isinstance`` holds. The proxy is
    callable only if the target is (or defines ``__cordis_invoke__``); calling
    dispatches through ``__cordis_invoke__`` when defined.
    """

    __slots__ = ("__weakref__", "_cordis_state")

    def __new__(cls, ctx: Context, target: object, tracker: Tracker) -> Traceable:
        if cls is Traceable:
            callable_ = callable(target) or _has_invoke(target)
            call = _traceable_call if callable_ else None
            cls = special_type(Traceable, cls_of(target), call, _original)
        return object.__new__(cls)

    def __init__(self, ctx: Context, target: object, tracker: Tracker) -> None:
        shadow = lookup(ctx, symbols.shadow)
        def_site = ctx if shadow is None else shadow
        use_site = ctx if shadow is None else parent_of(ctx)
        if use_site is None:
            raise RuntimeError("a shadow context must extend its use site")
        state = _State(target, ctx, tracker, def_site, use_site)
        object.__setattr__(self, "_cordis_state", state)

    def __getattribute__(self, name: str) -> Any:
        return _traceable_get(self, name, self)

    def __setattr__(self, name: str, value: Any) -> None:
        _traceable_set(self, name, value, self)

    def __delattr__(self, name: str) -> None:
        state = _state(self)
        if name in (symbols.original, symbols.caller) or name == state.tracker.property:
            raise ServiceAccessError(f'cannot delete property "{name}"', name=name, obj=self)
        delattr(state.target, name)


def _traceable_call(self: Traceable, *args: Any, **kwargs: Any) -> Any:
    """``__call__`` of proxies over callables (cordis ``apply`` trap)."""
    state = _state(self)
    tracker = state.tracker
    receiver = (
        self
        if tracker.no_shadow
        else _create_shadow(state.use_site, state.target, tracker.property, self)
    )
    return _apply_traceable(receiver, state.target, args, kwargs)


def _has_invoke(target: object) -> bool:
    own = instance_dict(target)
    if own is not None and symbols.invoke in own:
        return True
    return class_attr(type(target), symbols.invoke) is not MISSING


def _is_traceable(value: object) -> TypeGuard[Traceable]:
    return issubclass(type(value), Traceable)


def _is_with_props(value: object) -> TypeGuard[_WithProps]:
    return issubclass(type(value), _WithProps)


def _retrace(ctx: Context, value: object) -> Any:
    """Trace a stored value (cordis ``innerValue[symbols.tracker]``), proxies included."""
    if _is_traceable(value):
        state = _state(value)
        if state.ctx is ctx:
            return value
        return _create_traceable(ctx, state.target, state.tracker)
    inner = _tracker_of(value)
    if inner is not None:
        return _create_traceable(ctx, value, inner)
    return value


def _original(obj: object) -> Any:
    if _is_traceable(obj):
        return _state(obj).target
    return object.__getattribute__(obj, "_target")


def untraced[T](value: T) -> T:
    """The object behind a :class:`Traceable` proxy (``value`` itself otherwise).

    For internal code that does not need the use site: every attribute read
    through a proxy runs its trap, which dominates a cheap method's cost. Example::

        events = untraced(ctx.events)  # the shared EventsService

    Internal — not part of the public API.
    """
    if _is_traceable(value):
        target: T = _state(value).target  # the object the proxy stands in for
        return target
    return value


def _bind_method(func: Callable[..., object], receiver: object) -> MethodType:
    return MethodType(func, receiver)


def _shadow_method(
    use_site: Context, func: Callable[..., object], shadow: object
) -> Callable[..., object]:
    """Port of cordis ``createShadowMethod``: call with ``self=shadow``, trace the result."""

    def method(*args: object, **kwargs: object) -> object:
        return get_traceable(use_site, func(shadow, *args, **kwargs))

    method.__name__ = func.__name__
    method.__qualname__ = func.__qualname__
    method.__doc__ = func.__doc__
    if inspect.iscoroutinefunction(func):
        inspect.markcoroutinefunction(method)
    # after marking: markcoroutinefunction would follow `__func__`
    method.__dict__.update(__func__=func, __self__=shadow)
    return method


def _traceable_get(proxy: Traceable, name: str, receiver: object) -> Any:
    state = _state(proxy)
    target = state.target
    tracker = state.tracker
    if name == symbols.original:
        return target
    if name == symbols.caller:
        return state.def_site
    if name == tracker.property:
        return state.use_site
    if is_dunder(name):
        # cordis: symbol keys pass straight through with the receiver
        attr = class_attr(cls_of(target), name)
        if type(attr) is FunctionType:
            return _bind_method(attr, receiver)
        return getattr(target, name)
    if tracker.associate is not None:
        key = f"{tracker.associate}.{name}"
        if key in _props_of(state.use_site):
            # `ctx` rather than `use_site`, so that `ctx.foo.bar` and `ctx["foo.bar"]`
            # are governed by the same def site
            return getattr(state.ctx.extend(**{symbols.receiver: receiver}), key)
    use_site = state.use_site
    cls = cls_of(target)
    attr = class_attr(cls, name)
    if isinstance(attr, property):
        if attr.fget is None:
            raise AttributeError(f"property {name!r} of {cls.__name__!r} object has no getter")
        shadow = _create_shadow(use_site, target, tracker.property, receiver)
        value = attr.fget(shadow)
    elif attr is not MISSING and is_data_descriptor(attr):
        value = attr.__get__(target, cls)
    else:
        own = instance_dict(target)
        if own is not None and name in own:
            value = own[name]
        elif type(attr) is FunctionType:
            if tracker.no_shadow:
                return _bind_method(attr, receiver)
            shadow = _create_shadow(use_site, target, tracker.property, receiver)
            return _shadow_method(use_site, attr, shadow)
        else:
            value = getattr(target, name)
    return _retrace(use_site, value)


def _traceable_set(proxy: Traceable, name: str, value: object, receiver: object) -> None:
    state = _state(proxy)
    target = state.target
    tracker = state.tracker
    if name in (symbols.original, symbols.caller) or name == tracker.property:
        raise ServiceAccessError(f'cannot set property "{name}"', name=name, obj=proxy)
    if not is_dunder(name) and tracker.associate is not None:
        key = f"{tracker.associate}.{name}"
        if key in _props_of(state.use_site):
            setattr(state.ctx.extend(**{symbols.receiver: receiver}), key, value)
            return
    attr = class_attr(cls_of(target), name)
    if isinstance(attr, property) and attr.fset is not None:
        attr.fset(_create_shadow(state.use_site, target, tracker.property, receiver), value)
        return
    setattr(target, name, value)


def _props_of(ctx: Context) -> Mapping[str, object]:
    reflect: ReflectService | None = lookup(ctx, "reflect")  # absent while bootstrapping
    return {} if reflect is None else reflect.props


def _apply_traceable(
    receiver: object, target: Any, args: tuple[object, ...], kwargs: dict[str, object]
) -> Any:
    hook = class_attr(cls_of(target), symbols.invoke)
    own = instance_dict(target)
    if type(hook) is FunctionType and (own is None or symbols.invoke not in own):
        return hook(receiver, *args, **kwargs)
    try:
        bound = getattr(target, symbols.invoke)
    except AttributeError:
        return target(*args, **kwargs)
    return bound(*args, **kwargs)


def _create_shadow(use_site: Context, target: object, prop: str | None, receiver: object) -> Any:
    """Port of cordis ``createShadow``: overlay ``prop`` with a shadow of the use site."""
    if prop is None:
        return receiver
    value = static_value(target, prop)
    if not value:
        return receiver
    def_site = (lookup(value, symbols.shadow) if is_context(value) else None) or value
    if _is_with_props(receiver):
        # reuse an equivalent shadow receiver instead of stacking another layer
        props: object = object.__getattribute__(receiver, "_props")
        current = props.get(prop) if is_dict(props) else None
        if (
            is_context(current)
            and parent_of(current) is use_site
            and lookup(current, symbols.shadow) is def_site
        ):
            return receiver
    shadow = use_site.extend(**{symbols.shadow: def_site})
    return _WithProps(receiver, {prop: shadow}, readonly=True)


def _create_traceable(ctx: Context, value: object, tracker: Tracker) -> Traceable:
    return Traceable(ctx, value, tracker)


def get_traceable[T](ctx: Context, value: T) -> T:
    """Return ``value`` as seen from ``ctx`` (cordis ``getTraceable``).

    A shadow context resolves to its use site; a value carrying a
    :class:`Tracker` in ``__cordis_tracker__`` is wrapped in a :class:`Traceable`
    (an existing proxy is re-targeted rather than nested); anything else is
    returned unchanged. Example: ``client = get_traceable(ctx, raw_client)``.
    """
    if type(value) in ATOMIC:
        return value
    if is_context(value):
        if symbols.shadow in own_meta(value):
            use_site = parent_of(value)
            if use_site is None:
                raise RuntimeError("a shadow context must extend its use site")
            return use_site  # a context of the same type as the shadow
        return value
    return cast(T, _retrace(ctx, value))  # a proxy stands in for the value it wraps


def trace_cached[T](ctx: Context, name: str, value: T) -> T:
    """:func:`get_traceable` for the context attribute ``name``, cached on ``ctx``.

    The cache keeps one proxy per attribute name and is checked by target and
    tracker identity, so a replaced value is never served stale and the cache
    stays bounded by the number of names read. Example:
    ``trace_cached(ctx, "llm", raw_llm)`` (as ``ctx.llm`` reads it).
    """
    cache: dict[str, Traceable] = traced_cache(ctx)
    if type(value) in ATOMIC:
        # e.g. a withdrawn service: drop the stale proxy so it can be collected
        cache.pop(name, None)
        return value
    cached = cache.get(name)
    if cached is not None:
        state = _state(cached)
        if state.target is value and _tracker_of(value) is state.tracker:
            return cast(T, cached)  # a proxy stands in for the value it wraps
    result = get_traceable(ctx, value)
    if _is_traceable(result):
        cache[name] = result
    return result


def invoke(target: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a callable service through its own traced view (cordis ``createCallable``).

    For use in a service's ``__call__``: traces ``target`` to its own ``ctx`` and
    dispatches through ``__cordis_invoke__`` with the traced receiver. Its
    arguments and result are those of the service's call, hence ``Any``. Example::

        def __call__(self, *args: object) -> object:
            return invoke(self, *args)
    """
    tracker = _tracker_of(target)
    if tracker is None:
        raise TypeError(f"{type(target).__name__!r} object has no tracker")
    proxy: Any = _create_traceable(target.ctx, target, tracker)
    return proxy(*args, **kwargs)


class _WithProps(_Forwarding):
    """Port of cordis ``withProps``: a view whose ``props`` overlay ``target``.

    ``props`` is a dict of overrides or an object whose members win over the
    target's. Special (dunder) names always come from the target.
    """

    __slots__ = ("__weakref__", "_props", "_readonly", "_target")

    def __new__(cls, target: object, props: object, *, readonly: bool = False) -> _WithProps:
        if cls is _WithProps:
            call = _with_props_call if callable(target) else None
            cls = special_type(_WithProps, _real_type(target), call, _original)
        return object.__new__(cls)

    def __init__(self, target: object, props: object, *, readonly: bool = False) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_props", props)
        object.__setattr__(self, "_readonly", readonly)

    def __getattribute__(self, name: str) -> Any:
        return _with_props_get(self, name, self)

    def __setattr__(self, name: str, value: Any) -> None:
        props = object.__getattribute__(self, "_props")
        if not is_dunder(name) and _has(props, name):
            if object.__getattribute__(self, "_readonly"):
                raise ServiceAccessError(f'cannot set property "{name}"', name=name, obj=self)
            if is_dict(props):
                props[name] = value
            else:
                set_with_receiver(props, name, value, self)
            return
        set_with_receiver(object.__getattribute__(self, "_target"), name, value, self)

    def __delattr__(self, name: str) -> None:
        props = object.__getattribute__(self, "_props")
        if not is_dunder(name) and _has(props, name):
            if object.__getattribute__(self, "_readonly"):
                raise ServiceAccessError(f'cannot delete property "{name}"', name=name, obj=self)
            if is_dict(props):
                del props[name]
            else:
                delattr(props, name)
            return
        delattr(object.__getattribute__(self, "_target"), name)


def _with_props_call(self: _WithProps, *args: Any, **kwargs: Any) -> Any:
    return object.__getattribute__(self, "_target")(*args, **kwargs)


def _real_type(obj: object) -> type:
    """The class of the object a (possibly nested) proxy or view finally wraps."""
    while True:
        if _is_traceable(obj):
            obj = _state(obj).target
        elif _is_with_props(obj):
            obj = object.__getattribute__(obj, "_target")
        else:
            return type(obj)


def _has(obj: object, name: str) -> bool:
    if isinstance(obj, dict):
        return name in obj
    if _is_traceable(obj):
        state = _state(obj)
        if name in (symbols.original, symbols.caller, state.tracker.property):
            return True
        obj = state.target
    elif _is_with_props(obj):
        return _has(object.__getattribute__(obj, "_props"), name) or _has(
            object.__getattribute__(obj, "_target"), name
        )
    return inspect.getattr_static(obj, name, MISSING) is not MISSING


def _with_props_get(view: _WithProps, name: str, receiver: object) -> Any:
    props = object.__getattribute__(view, "_props")
    if not is_dunder(name) and _has(props, name):
        if is_dict(props):
            return props[name]
        return get_with_receiver(props, name, receiver)
    return get_with_receiver(object.__getattribute__(view, "_target"), name, receiver)


def get_with_receiver(obj: object, name: str, receiver: object) -> Any:
    """``Reflect.get(obj, name, receiver)``: getters and methods see ``receiver``.

    Returns the attribute's value, as dynamic as ``getattr``. Example:
    ``get_with_receiver(service, "status", proxy)`` runs the getter with ``self=proxy``.
    """
    kind = cls_of(obj)
    if _is_traceable(obj):
        return _traceable_get(obj, name, receiver)
    if _is_with_props(obj):
        return _with_props_get(obj, name, receiver)
    attr = class_attr(kind, name)
    if isinstance(attr, property) and attr.fget is not None:
        return attr.fget(receiver)
    if type(attr) is FunctionType:
        own = instance_dict(obj)
        if own is None or name not in own:
            return _bind_method(attr, receiver)
    return getattr(obj, name)


def set_with_receiver(obj: object, name: str, value: object, receiver: object) -> None:
    """``Reflect.set(obj, name, value, receiver)``: setters see ``receiver``.

    Example: ``set_with_receiver(service, "status", "ok", proxy)``.
    """
    kind = cls_of(obj)
    if _is_traceable(obj):
        _traceable_set(obj, name, value, receiver)
        return
    if _is_with_props(obj):
        _WithProps.__setattr__(obj, name, value)
        return
    attr = class_attr(kind, name)
    if isinstance(attr, property) and attr.fset is not None:
        attr.fset(receiver, value)
        return
    setattr(obj, name, value)


def with_props(target: object, props: object) -> Any:
    """Return a view of ``target`` whose attributes in ``props`` take precedence.

    ``props`` is a dict of overrides (written in place) or an object whose
    members win over ``target``'s; ``None`` returns ``target`` itself. Methods
    and properties read through the view see the view as ``self``. The view
    has ``target``'s attributes plus those of ``props``, so its type is
    dynamic (``Any``). Example::

        view = with_props(service, {"ctx": inner_ctx})  # view.ctx is inner_ctx
    """
    if props is None:
        return target
    return _WithProps(target, props)
