"""Prototype-chain state of a context and the package-internal accessors to it.

Internal — not part of the public API.

A :class:`~pycordis.context.Context` keeps its own attributes in ``_own`` and
falls back to ``_parent`` for everything else (cordis ``Object.create(parent)``).
That state lives on :class:`ContextCore`, the base class of ``Context``. Its
names keep a leading underscore on purpose: a context's attribute namespace is
shared with service and accessor names, and names starting with ``_`` never
resolve to a service, so ``ctx.parent`` stays free for a user service.

The rest of pycordis reads this state only through the functions below
(:func:`lookup`, :func:`parent_of`, :func:`own_meta`, :func:`traced_cache`,
:func:`isolate_map`, :func:`intercept_map`, :func:`intercept_configs`). The
ones that touch underscore attributes are static methods of :class:`_Access`,
a never-instantiated subclass of :class:`ContextCore`: Python's convention, as
enforced by pyright's ``reportPrivateUsage``, lets an underscore member be used
inside the declaring class and its subclasses only, so a "friend" subclass is
the one place outside ``ContextCore`` itself that may read it (a module-level
function, even in this module, may not). This module sits below ``context`` in
the import graph so that ``trace``, ``reflect`` and ``fiber`` can import it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self, final

from pycordis._utils import Label, ScopeMap, symbols

if TYPE_CHECKING:
    from pycordis.events import EventsService
    from pycordis.fiber import Fiber
    from pycordis.reflect import ReflectService
    from pycordis.registry import RegistryService

__all__ = [
    "ContextCore",
    "events_of",
    "fiber_of",
    "intercept_configs",
    "intercept_map",
    "isolate_map",
    "lookup",
    "own_meta",
    "parent_of",
    "reflect_of",
    "registry_of",
    "traced_cache",
]


class ContextCore:
    """One node of the context prototype chain: own attributes, parent and proxy cache.

    Base class of :class:`~pycordis.context.Context`.

    Internal — not part of the public API.
    """

    _own: dict[str, Any]
    """Own attributes (cordis: own properties of the context object).

    ``Any``: services and meta of every type live here, read back by name.
    """
    _parent: Self | None
    """The context this one extends; ``None`` for a root."""
    _traced: dict[str, Any]
    """Traced proxies of attributes read through this context, by name (any service type)."""

    def _attach(self, own: dict[str, object], parent: Self | None) -> None:
        """Initialise this node's state (bypassing ``Context.__setattr__``)."""
        object.__setattr__(self, "_own", own)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_traced", {})

    def _child(self, own: dict[str, object]) -> Self:
        """A new node of the same type whose own attributes are ``own`` and parent is ``self``."""
        child = object.__new__(type(self))
        child._attach(own, self)
        return child


@final
class _Access(ContextCore):
    """Friend of :class:`ContextCore` hosting the accessors below; never instantiated."""

    @staticmethod
    def lookup(ctx: ContextCore, name: str, default: object = None) -> Any:  # see ``_own``
        node: ContextCore | None = ctx
        while node is not None:
            own = node._own
            if name in own:
                return own[name]
            node = node._parent
        return default

    @staticmethod
    def parent_of[C: ContextCore](ctx: C) -> C | None:
        return ctx._parent

    @staticmethod
    def own_meta(ctx: ContextCore) -> Mapping[str, object]:
        return ctx._own

    @staticmethod
    def traced_cache(ctx: ContextCore) -> dict[str, Any]:
        return ctx._traced


lookup = _Access.lookup
"""``lookup(ctx, name, default=None)``: own or inherited attribute ``name`` of ``ctx``.

Reads the prototype chain only: no service resolution and no tracing.
"""

parent_of = _Access.parent_of
"""``parent_of(ctx)``: the context ``ctx`` extends (``None`` for a root)."""

own_meta = _Access.own_meta
"""``own_meta(ctx)``: read-only view of the attributes set on ``ctx`` itself."""

traced_cache = _Access.traced_cache
"""``traced_cache(ctx)``: the per-context cache of traced proxies, by attribute name."""


def fiber_of(ctx: ContextCore) -> Fiber:
    """``ctx.fiber`` read without the attribute trap (a fiber is never traced).

    For internal hot paths (plugin load, listener registration). Example:
    ``fiber_of(ctx).assert_active()``.
    """
    fiber: Fiber = lookup(ctx, "fiber")
    return fiber


def events_of(ctx: ContextCore) -> EventsService:
    """The events service behind ``ctx.events``, without its traced proxy.

    Dispatch (``emit``, ``bail``, ``waterfall``, ...) does not depend on the
    reading context, so internal code dispatches on the service itself and
    skips resolving the ``ctx.emit`` accessor. Registration (``on``) needs the
    reading context: use ``ctx.on`` for it. Example:
    ``events_of(ctx).emit("internal/status", fiber, old_state)``.
    """
    events: EventsService = lookup(ctx, "events")
    return events


def reflect_of(ctx: ContextCore) -> ReflectService:
    """The reflect service behind ``ctx.reflect``, without its traced proxy.

    Only for reading its shared state (``store``, ``props``); its methods act
    for the reading context and need ``ctx.reflect``. Example:
    ``reflect_of(ctx).store``.
    """
    reflect: ReflectService = lookup(ctx, "reflect")
    return reflect


def registry_of(ctx: ContextCore) -> RegistryService | None:
    """The registry service behind ``ctx.registry``, untraced; ``None`` until it is installed.

    For internal loops over every plugin (service notification). Example:
    ``registry = registry_of(ctx)``.
    """
    registry: RegistryService | None = lookup(ctx, "registry")
    return registry


def isolate_map(ctx: ContextCore) -> ScopeMap[Label]:
    """The isolation labels visible from ``ctx`` (cordis ``ctx[Context.isolate]``)."""
    isolate: ScopeMap[Label] = lookup(ctx, symbols.isolate)
    return isolate


def intercept_map(ctx: ContextCore) -> ScopeMap[object]:
    """The intercept configs visible from ``ctx`` (cordis ``ctx[Context.intercept]``)."""
    intercept: ScopeMap[object] = lookup(ctx, symbols.intercept)
    return intercept


def intercept_configs(ctx: ContextCore, name: str) -> list[object]:
    """The intercept configs for service ``name`` visible from ``ctx``, root-most first."""
    intercept: ScopeMap[object] | None = intercept_map(ctx)
    configs: list[object] = []
    while intercept is not None and name in intercept:
        if intercept.has_own(name):
            configs.insert(0, intercept[name])
        intercept = intercept.parent
    return configs
