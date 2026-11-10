"""Static attribute introspection for the tracing proxies of :mod:`pycordis.trace`.

The proxies must inspect a target's attributes without running its getters or
their own ``__getattribute__``; these helpers read class and instance
dictionaries directly.

Internal — not part of the public API.
"""

from __future__ import annotations

from types import MemberDescriptorType, NoneType
from typing import TYPE_CHECKING, Any, TypeGuard

from pycordis._guards import MISSING

if TYPE_CHECKING:
    from pycordis.context import Context

__all__ = [
    "ATOMIC",
    "MISSING",
    "class_attr",
    "cls_of",
    "instance_dict",
    "is_context",
    "is_data_descriptor",
    "is_dunder",
    "static_value",
]

ATOMIC: frozenset[type] = frozenset(
    {NoneType, bool, int, float, complex, str, bytes, tuple, list, dict, set, frozenset}
)
"""Value types that can never carry a tracker."""


def is_context(value: object) -> TypeGuard[Context]:
    """Whether ``value`` is a :class:`~pycordis.context.Context` (without importing it).

    Example: ``is_context(ctx)`` is ``True``; ``is_context(ctx.logger)`` is ``False``.
    """
    return getattr(type(value), "__cordis_context__", False) is True


def cls_of(value: object) -> type:
    """The real class of ``value`` (a proxy's own class, not its target's).

    Example: ``cls_of(proxy)`` is the per-target proxy subclass.
    """
    return type(value)


def class_attr(cls: type, name: str) -> Any:  # whatever the class body defined
    """The raw class-dict entry for ``name`` along ``cls``'s MRO, else :data:`MISSING`.

    The entry is whatever the class body defined (a function, descriptor or
    value), hence ``Any``; callers dispatch on its type.

    Example: ``class_attr(Service, "start")`` returns the plain function.
    """
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass.__dict__[name]
    return MISSING


def instance_dict(value: object) -> dict[str, Any] | None:  # instance attributes: any values
    """``value.__dict__`` read without ``__getattribute__``, or ``None`` when it has none.

    Example: ``instance_dict(service).get("ctx")``.
    """
    try:
        return object.__getattribute__(value, "__dict__")
    except AttributeError:
        return None


def static_value(value: object, name: str) -> Any:  # an attribute value of any type
    """A data value of ``value`` (instance dict or plain class attribute), without getters.

    The value is an arbitrary attribute value, hence ``Any``.

    Example: ``static_value(service, "ctx")`` is the service's own context.
    """
    own = instance_dict(value)
    if own is not None and name in own:
        return own[name]
    attr = class_attr(type(value), name)
    if type(attr) is MemberDescriptorType:
        # a `__slots__` member is plain data
        return getattr(value, name, None)
    if attr is MISSING or hasattr(cls_of(attr), "__get__"):
        return None
    return attr


def is_dunder(name: str) -> bool:
    """Whether ``name`` is a special (``__x__``) name.

    Example: ``is_dunder("__call__")`` is ``True``.
    """
    return name.startswith("__") and name.endswith("__")


def is_data_descriptor(attr: object) -> bool:
    """Whether ``attr`` is a data descriptor (defines ``__set__`` or ``__delete__``).

    Example: ``is_data_descriptor(property(f))`` is ``True``.
    """
    kind = type(attr)
    return hasattr(kind, "__set__") or hasattr(kind, "__delete__")
