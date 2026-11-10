from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from pycordis.context import Context
from pycordis.fiber import is_reported

if TYPE_CHECKING:
    from pycordis._utils import Label, ScopeMap
    from pycordis.logger import Message


def capture_reports(root: Context) -> list[BaseException]:
    """Capture exceptions reported through ``root.logger.error`` (an exporter on the root)."""
    errors: list[BaseException] = []
    capture_reports_into(root, errors)
    return errors


def capture_reports_into(root: Context, errors: list[BaseException]) -> None:
    """Append exceptions reported through ``root.logger.error`` to ``errors``."""

    class Capture:
        def export(self, message: Message) -> None:
            error = message.args[0]
            # `__cordis_reported__` skips the extra entries logged for a `__cause__`
            if message.type == "error" and isinstance(error, BaseException) and is_reported(error):
                errors.append(error)

    exporters = root.logger.exporters
    exporters[-len(exporters) - 1] = Capture()  # not an effect: keeps fiber effects intact


def apply_plugin(
    name: str, apply: Callable[[Context, Any], object], **attrs: object
) -> SimpleNamespace:
    """An object plugin ``SimpleNamespace(name=name, apply=apply, **attrs)``.

    Typing ``apply`` lets a lambda infer its ``(ctx, config)`` parameters.
    """
    return SimpleNamespace(name=name, apply=apply, **attrs)


def with_meta[F: Callable[..., Any]](plugin: F, **meta: object) -> F:
    """Set cordis metadata attributes (``Config``, ``inject``, ``name``, ...) on a function plugin.

    Equivalent to ``plugin.inject = [...]`` and friends: function attributes are how
    a function plugin declares metadata, but a type checker does not model
    attributes on a ``def`` function, so they are written to its ``__dict__``.
    """
    vars(plugin).update(meta)
    return plugin


def isolate_map(ctx: Context) -> ScopeMap[Label]:
    """``ctx``'s isolate scope map, read through the ``Context.isolate`` symbol.

    The typed equivalent of cordis ``ctx[Context.isolate]``.
    """
    scope: ScopeMap[Label] = getattr(ctx, Context.isolate)
    return scope


def intercept_map(ctx: Context) -> ScopeMap[Any]:
    """``ctx``'s intercept scope map, read through the ``Context.intercept`` symbol.

    The typed equivalent of cordis ``ctx[Context.intercept]``.
    """
    scope: ScopeMap[Any] = getattr(ctx, Context.intercept)
    return scope


LOADER_PLUGINS = Path(__file__).parent / "fixtures" / "loader" / "plugins"


@pytest.fixture
def loadfix() -> Iterator[ModuleType]:
    """Make tests/fixtures/loader/plugins importable and yield the reset ``loadfix`` module."""
    sys.path.insert(0, str(LOADER_PLUGINS))
    try:
        module = importlib.import_module("loadfix")
        module.reset()
        yield module
    finally:
        sys.path.remove(str(LOADER_PLUGINS))
        sys.modules.pop("loadfix", None)
