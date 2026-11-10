"""Type guards for untyped data, and the typed :data:`MISSING` sentinel.

Internal — not part of the public API.

``isinstance(value, dict)`` narrows an ``object`` to ``dict[Unknown, Unknown]``,
whose items a strict type checker cannot use. These guards narrow to
``dict[object, object]`` / ``list[object]`` / ``tuple[object, ...]`` instead:
the element types stay honest (``object``, narrowed again before use), so
code walking parsed data needs neither ``Any`` nor ``cast``. They check the
container type only, exactly like the ``isinstance`` they replace. Example::

    if is_dict(node):
        for key, value in node.items():  # key: object, value: object
            ...

:data:`MISSING` marks "no value" where ``None`` is a valid value. Its type,
:class:`Missing`, is a one-member enum, so ``value is MISSING`` narrows
``object | Missing`` to ``object`` (an ``object()`` sentinel cannot do that).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Final, TypeGuard, final

__all__ = [
    "MISSING",
    "Missing",
    "RawTable",
    "is_array",
    "is_dict",
    "is_exception_group",
    "is_future",
    "is_iterable",
    "is_list",
    "is_mapping",
    "is_str_list",
    "is_table",
]

type RawTable = dict[str, Any]
"""A table as parsed from a config file (TOML, JSON or YAML), before validation.

Its values are ``Any`` on purpose: they are whatever the file holds, read by
key and validated where they are read (entry keys by the loader's table
parser, ``config`` by the plugin's ``Config`` schema). This alias is the one
place parsed config data is typed dynamically. Example: ``{"id": "llm", "use": "app:llm"}``.
"""


@final
class Missing(Enum):
    """The type of :data:`MISSING`.

    Example: ``def get(key: str, default: object | Missing = MISSING) -> object: ...``.
    """

    MISSING = "MISSING"

    def __repr__(self) -> str:
        return "MISSING"


MISSING: Final = Missing.MISSING
"""Sentinel for "no value" where ``None`` is a valid value. Example: ``table.get(key, MISSING)``."""


def is_dict(value: object) -> TypeGuard[dict[object, object]]:
    """Whether ``value`` is a ``dict`` (subclasses included). Example: ``is_dict({}) is True``."""
    return isinstance(value, dict)


def is_exception_group(error: BaseException) -> TypeGuard[BaseExceptionGroup[BaseException]]:
    """Whether ``error`` is an exception group. Example: ``for sub in error.exceptions: ...``."""
    return isinstance(error, BaseExceptionGroup)


def is_future(value: object) -> TypeGuard[asyncio.Future[object]]:
    """Whether ``value`` is an :class:`asyncio.Future` (tasks included).

    Unlike :func:`asyncio.isfuture` this is a plain ``isinstance`` check (no
    duck-typed futures). Example: ``if is_future(result): await result``.
    """
    return isinstance(value, asyncio.Future)


def is_iterable(value: object) -> TypeGuard[Iterable[object]]:
    """Whether ``value`` is :class:`~collections.abc.Iterable`. Example: ``is_iterable({1})``."""
    return isinstance(value, Iterable)


def is_list(value: object) -> TypeGuard[list[object]]:
    """Whether ``value`` is a ``list`` (subclasses included). Example: ``is_list([]) is True``."""
    return isinstance(value, list)


def is_array(value: object) -> TypeGuard[list[object] | tuple[object, ...]]:
    """Whether ``value`` is a ``list`` or ``tuple``. Example: ``is_array((1, 2)) is True``."""
    return isinstance(value, list | tuple)


def is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    """Whether ``value`` is a :class:`~collections.abc.Mapping`. Example: ``is_mapping({})``."""
    return isinstance(value, Mapping)


def is_table(value: object) -> TypeGuard[RawTable]:
    """Whether ``value`` is a parsed config table (a ``dict``). Example: ``is_table({"id": 1})``."""
    return isinstance(value, dict)


def is_str_list(value: object) -> TypeGuard[list[str]]:
    """Whether ``value`` is a ``list`` of strings only. Example: ``is_str_list(["a"]) is True``."""
    return is_list(value) and all(isinstance(item, str) for item in value)
