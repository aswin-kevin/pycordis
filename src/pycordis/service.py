"""Base class for services exposing a named API on contexts (port of cordis ``service.ts``).

A :class:`Service` registers itself with ``ctx.reflect.provide`` when constructed
and is removed with the owning fiber. Reading it through a context returns a
traced proxy (see :mod:`pycordis.trace`) whose methods see the caller's context.

Python differences from the JS original:

- cordis returns a function object from the constructor when the class has an
  invoke body (``createCallable``). Here a subclass defining
  ``__cordis_invoke__`` gets a ``__call__`` (unless it defines its own) that
  dispatches through a proxy traced to the service's own context; services
  without one are not callable, so ``callable(ctx.svc)`` is ``False``.
- ``isinstance`` across proxies comes from the proxy's ``__class__``, so no
  ``__instancecheck__`` is needed.
- The intercept-config type parameter ``C`` has no default (PEP 696 defaults
  need Python 3.13 syntax); write ``Service[Any]`` under strict type checking.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from pycordis._context_internals import intercept_configs, isolate_map
from pycordis._utils import merge_entries, symbols
from pycordis.keys import ServiceName, service_name
from pycordis.trace import Tracker, invoke

if TYPE_CHECKING:
    from pycordis.context import Context

__all__ = ["Service"]


def _service_call(self: Service[Any], *args: Any, **kwargs: Any) -> Any:
    """Call the service through its own traced view (cordis ``createCallable``).

    Installed as ``__call__``: its signature is the subclass's ``__cordis_invoke__``.
    """
    return invoke(self, *args, **kwargs)


def _defines(cls: type, name: str) -> bool:
    return any(name in klass.__dict__ for klass in cls.__mro__ if klass is not object)


class Service[C]:
    """Base class for services that expose a named API on ``ctx``.

    ``Service(ctx, name)`` registers the instance as ``name`` in ``ctx`` (via
    ``ctx.reflect.provide(name, self, self.__cordis_check__)``), so it is
    unregistered automatically when the owning fiber unloads. ``name`` defaults
    to the class attribute :attr:`provide`, which ``class Counter(Service,
    provide="counter")`` sets. Used as a class plugin, the registry constructs it
    as ``cls(ctx, config)``; a non-``str`` second argument is that config, not a
    name, and is ignored.

    Optional hooks a subclass may define: ``__cordis_check__()`` (availability
    predicate consulted before dependents load), ``__cordis_invoke__(...)``
    (makes the service callable, e.g. ``ctx.logger("x")``) and
    ``__cordis_init__()`` (run after construction as a class plugin).

    Example::

        class Counter(Service[None], provide="counter"):
            value = 0

        ctx.plugin(Counter)  # ctx.counter.value == 0 once loaded
    """

    provide: ClassVar[str | None] = None
    """Default service name; set by ``class X(Service, provide="name")``.

    A :class:`~pycordis.keys.ServiceKey` works too and is stored as its name:
    ``class Client(Service[Any], provide=LLM)``.
    """

    ctx: Context
    """The context the service was registered in (the use site through a proxy)."""
    name: str
    """The service name this instance is registered under."""
    __cordis_tracker__: Tracker
    """Tracing metadata: ``Tracker(property="ctx", associate=name)``."""

    def __init_subclass__(cls, *, provide: ServiceName | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if provide is not None:
            cls.provide = service_name(provide)
        if _defines(cls, symbols.invoke) and not _defines(cls, "__call__"):
            cls.__call__ = _service_call

    # ``name`` is typed ``object`` because the registry passes the plugin config
    # here: ``ctx.plugin(MyService, MyConfig(...))`` must type-check
    def __init__(self, ctx: Context, name: object = None) -> None:
        if not isinstance(name, str):
            name = type(self).provide
        if name is None:
            raise TypeError(
                f"service name is required: pass it to {type(self).__name__}() "
                'or declare it with `provide="..."`'
            )
        self.ctx = ctx
        self.name = name
        self.__cordis_tracker__ = Tracker(property="ctx", associate=name)
        # never touch `__cordis_init_hooks__` here: `@inject` methods keep a
        # per-instance list that may already have been appended to
        # an optional cordis ``[symbols.check]`` hook a subclass may define
        check: Callable[[], bool] | None = getattr(self, symbols.check, None)
        ctx.reflect.provide(name, self, check)

    def __cordis_filter__(self, ctx: Context) -> bool:
        """Whether ``ctx`` shares this service's isolation scope (listener filter)."""
        return isolate_map(ctx).get(self.name) is isolate_map(self.ctx).get(self.name)

    def __cordis_extend__(self, **props: object) -> Self:
        """Return a shallow copy of this service with ``props`` set on it.

        Read through a proxy, the underlying service is copied. Callable
        services stay callable. The copy is shallow: mutable attributes (lists,
        dicts, caches, exporter tables) are shared with the original, so
        mutating them through either affects both; assign a new object in
        ``props`` to give the copy its own.
        """
        original: Self = getattr(self, symbols.original, self)
        result = copy.copy(original)
        for key, value in props.items():
            setattr(result, key, value)
        return result

    def __cordis_resolve_config__(self, base: C | None = None, head: C | None = None) -> C:
        """Merge intercept configs for this service from ``self.ctx`` and its ancestors.

        Entries added closer to the root apply first; ``base`` is prepended and
        ``head`` appended. Uses ``self.Config.merge(*configs)`` when the service
        declares one, otherwise a shallow dict merge of mappings (other objects
        contribute their attributes).
        """
        configs = intercept_configs(self.ctx, self.name)
        if base is not None:
            configs.insert(0, base)
        if head is not None:
            configs.append(head)
        # an optional ``Config.merge`` hook a subclass may declare
        merge: object = getattr(getattr(self, "Config", None), "merge", None)
        if merge is None:
            merged: object = merge_entries(configs)
        elif callable(merge):
            merged = merge(*configs)
        else:
            raise TypeError(f"{type(self).__name__}.Config.merge is not callable")
        # the service declares C as the shape these entries merge into
        return cast(C, merged)
