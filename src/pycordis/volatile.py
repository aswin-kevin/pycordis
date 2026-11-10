"""Volatile config references (port of cosmokit ``volatile.ts``).

A :class:`Volatile` is a stable reference whose deep-frozen snapshot can be
swapped by the owning runtime without changing the reference's identity.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeGuard, cast, final

from pycordis._guards import is_array, is_dict

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_TOKEN: object = object()

type _VolatilePath = tuple[str, ...]
"""Key path of a :class:`Volatile` inside a config (list indices as strings)."""


@final
class Volatile[T]:
    """Stable reference to an immutable config snapshot; create with :func:`create_volatile`.

    Example: ``limits = create_volatile({"rpm": 60})``; ``limits.get()["rpm"]``.
    """

    __slots__ = ("_value",)

    def __init__(self, value: object, _token: object = None) -> None:
        if _token is not _TOKEN:
            raise TypeError("Volatile cannot be constructed directly; use create_volatile()")
        self._value: object = value

    def get(self) -> T:
        """Return the current deep-frozen snapshot.

        Example: ``create_volatile({"rate": 5}).get()["rate"] == 5``.
        """
        # the snapshot is the frozen form of a T (tuples for lists, read-only
        # mappings for dicts), which is what callers read it as
        return cast("T", self._value)

    def commit(self, snapshot: object) -> None:
        """Replace the held snapshot in place; ``snapshot`` must already be deep-frozen.

        The owning runtime uses this to publish a volatile-only config update
        without restarting the plugin.

        Internal — not part of the public API.

        Example: ``ref.commit(new_snapshot)`` (the fiber does this on update).
        """
        self._value = snapshot

    def __repr__(self) -> str:
        return f"Volatile({self._value!r})"


def _snapshot(value: object, ancestors: set[int]) -> object:
    if callable(value):
        raise TypeError("volatile config cannot contain functions")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    key_id = id(value)
    if key_id in ancestors:
        raise TypeError("volatile config cannot contain cycles")
    ancestors.add(key_id)
    try:
        if is_array(value):
            return tuple(_snapshot(item, ancestors) for item in value)
        if is_dict(value):
            return MappingProxyType(
                {key: _snapshot(item, ancestors) for key, item in value.items()}
            )
        raise TypeError("volatile config objects must be plain objects or arrays")
    finally:
        ancestors.discard(key_id)


def create_volatile[T](value: T) -> Volatile[T]:
    """Create a detached reference holding a deep-frozen copy of ``value``.

    Example: ``rate = create_volatile({"per_minute": 60})``.
    """
    return Volatile(_snapshot(value, set()), _TOKEN)


def is_volatile(value: object) -> TypeGuard[Volatile[object]]:
    """Return whether ``value`` is a :class:`Volatile` reference.

    Example: ``is_volatile(create_volatile(1)) is True``.
    """
    return isinstance(value, Volatile)


def volatile_entries(value: object) -> list[tuple[_VolatilePath, Volatile[object]]]:
    """Collect references and their key paths without descending into snapshots.

    Only dicts, lists/tuples and dataclass instances (by field name) are
    traversed; list indices appear as strings. Cyclic ordinary containers are
    visited once per path. Example::

        volatile_entries({"limits": create_volatile(5)})  # [(("limits",), <Volatile>)]
    """
    ancestors: set[int] = set()

    def visit(node: object, path: _VolatilePath) -> list[tuple[_VolatilePath, Volatile[object]]]:
        if is_volatile(node):
            return [(path, node)]
        node_id = id(node)
        items = _named_children(node)
        if items is None or node_id in ancestors:
            return []
        ancestors.add(node_id)
        try:
            out: list[tuple[_VolatilePath, Volatile[object]]] = []
            for key, child in items:
                out.extend(visit(child, (*path, key)))
            return out
        finally:
            ancestors.discard(node_id)

    return visit(value, ())


def diff_only_volatile(old: object, new: object) -> bool:
    """Return whether ``old`` and ``new`` are equal except inside Volatile refs at same paths.

    Dataclass instances are compared field by field and must share their type.
    Example: ``diff_only_volatile({"a": create_volatile(1)}, {"a": create_volatile(2)})``.
    """
    if isinstance(old, Volatile) and isinstance(new, Volatile):
        return True
    if isinstance(old, Volatile) or isinstance(new, Volatile):
        return False
    if _is_dataclass(old) or _is_dataclass(new):
        if type(old) is not type(new) or not (_is_dataclass(old) and _is_dataclass(new)):
            return False
        return all(
            diff_only_volatile(a, b)
            for (_, a), (_, b) in zip(_fields(old), _fields(new), strict=True)
        )
    if is_dict(old) and is_dict(new):
        return old.keys() == new.keys() and all(diff_only_volatile(old[k], new[k]) for k in old)
    if is_array(old) and is_array(new):
        return len(old) == len(new) and all(
            diff_only_volatile(a, b) for a, b in zip(old, new, strict=True)
        )
    return bool(old == new)


def volatile_pairs(old: object, new: object) -> list[tuple[Volatile[object], Volatile[object]]]:
    """Pair the references of ``old`` with those at the same position in ``new``.

    Walks both structures in parallel (keys keep their original types), so it
    expects :func:`diff_only_volatile` to hold; positions that do not line up
    are skipped. Example::

        volatile_pairs({"a": old_ref}, {"a": new_ref}) == [(old_ref, new_ref)]
    """
    if is_volatile(old) and is_volatile(new):
        return [(old, new)]
    children: Iterable[tuple[object, object]]
    if _is_dataclass(old) and _is_dataclass(new) and type(old) is type(new):
        children = zip((v for _, v in _fields(old)), (v for _, v in _fields(new)), strict=True)
    elif is_dict(old) and is_dict(new):
        children = ((old[k], new[k]) for k in old if k in new)
    elif is_array(old) and is_array(new):
        children = zip(old, new, strict=False)
    else:
        return []
    out: list[tuple[Volatile[object], Volatile[object]]] = []
    for a, b in children:
        out.extend(volatile_pairs(a, b))
    return out


def _named_children(node: object) -> list[tuple[str, object]] | None:
    """The children of a traversed container, keyed by name or index (``None``: not one)."""
    if _is_dataclass(node):
        return _fields(node)
    if is_dict(node):
        return [(str(k), v) for k, v in node.items()]
    if is_array(node):
        return [(str(i), v) for i, v in enumerate(node)]
    return None


def _is_dataclass(value: object) -> TypeGuard[DataclassInstance]:
    """Whether ``value`` is a dataclass instance (not a dataclass type)."""
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _fields(value: DataclassInstance) -> list[tuple[str, object]]:
    return [(f.name, getattr(value, f.name)) for f in dataclasses.fields(value)]
