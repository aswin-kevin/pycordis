"""Plugin resolution for loader entries (spec §6.6 plugin resolution and safety).

A ``use`` string is either ``"package.module:attr.sub"`` (import the module,
then walk the attributes) or a bare name registered under the
``pycordis.plugins`` entry-point group. ``allow`` patterns (``fnmatch``,
case-sensitive; ``pkg.*`` also admits ``pkg`` itself) are checked against the
module name — the part before ``:``, or the entry point's ``module`` — *before*
anything is imported. Attribute segments may not start with ``_``, and every
module reached while walking the attributes must be allowed too, so a reference
cannot escape ``allow`` through re-exported modules.

``allow`` names what a config may import, not a sandbox: importing an allowed
``app.plugins`` also runs its parent package ``app`` (Python imports parents
first), and an allowed module's own code (including a module ``__getattr__``
reached by the attribute walk) runs with the process's full rights.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from collections.abc import Sequence
from fnmatch import fnmatchcase
from types import ModuleType

ENTRY_POINT_GROUP = "pycordis.plugins"
"""Entry-point group under which installed packages expose plugins by name."""

_DOTTED = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")


def import_plugin(use: str, allow: Sequence[str]) -> object:
    """Import the plugin object named by ``use`` if its module is allowed.

    Raises ``ValueError`` for a malformed reference, ``PermissionError`` when the
    module matches no ``allow`` pattern, and ``ImportError`` when the module,
    attribute or entry point cannot be found.

    Example: ``import_plugin("app.plugins:chat", allow=["app.*"])``.
    """
    if ":" in use:
        module, _, attrs = use.partition(":")
        segments = attrs.split(".")
        if (
            not _DOTTED.fullmatch(module)
            or not _DOTTED.fullmatch(attrs)
            or any(seg.startswith("_") for seg in segments)
        ):
            raise ValueError(f'invalid plugin reference "{use}"')
        _check_allowed(module, allow)
        target: object = importlib.import_module(module)
        for attr in segments:
            try:
                target = getattr(target, attr)
            except AttributeError:
                raise ImportError(f'module "{module}" has no attribute "{attrs}"') from None
            if isinstance(target, ModuleType):  # re-exported module: must be allowed too
                _check_allowed(target.__name__, allow)
        return target
    if not use.strip():  # entry-point names are free-form (e.g. "my-plugin")
        raise ValueError(f'invalid plugin reference "{use}"')
    try:
        entry_point = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)[use]
    except KeyError:
        raise ImportError(
            f'no plugin named "{use}" in entry-point group "{ENTRY_POINT_GROUP}"'
        ) from None
    _check_allowed(entry_point.module, allow)
    return entry_point.load()


def _is_allowed(module: str, pattern: str) -> bool:
    # "pkg.*" also admits the package itself
    return fnmatchcase(module, pattern) or (pattern.endswith(".*") and module == pattern[:-2])


def _check_allowed(module: str, allow: Sequence[str]) -> None:
    if not any(_is_allowed(module, pattern) for pattern in allow):
        raise PermissionError(f'plugin module "{module}" is not allowed by loader.allow')
