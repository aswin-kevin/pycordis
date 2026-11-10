"""Fixture plugins for tests/loader/test_service.py.

Counters are keyed by the ``name`` in the plugin config (``"?"`` without one) and
reset by the ``loadfix`` fixture before every test.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Coroutine
from typing import Any, cast

from pycordis.context import Context
from pycordis.current import current_context
from pycordis.registry import plugin_meta
from pycordis.service import Service

runs: Counter[str] = Counter()
"""Body runs per key."""
disposed: Counter[str] = Counter()
"""Disposals per key."""
configs: dict[str, Any] = {}
"""Last effective config per key."""
seen: dict[str, Any] = {}
"""Values observed by plugin bodies, per key."""


def reset() -> None:
    """Clear every counter."""
    runs.clear()
    disposed.clear()
    configs.clear()
    seen.clear()


def _key(config: Any, prefix: str = "") -> str:
    # configs are arbitrary JSON/TOML data; a dict one is keyed by str
    name = cast("dict[str, object]", config).get("name", "?") if isinstance(config, dict) else "?"
    return f"{prefix}{name}"


def _record(key: str, config: Any) -> Callable[[], None]:
    runs[key] += 1
    configs[key] = config

    def dispose() -> None:
        disposed[key] += 1

    return dispose


def echo(_ctx: Context, config: Any) -> Callable[[], None]:
    """Record a run under the config's name."""
    return _record(_key(config), config)


def other(_ctx: Context, config: Any) -> Callable[[], None]:
    """Like :func:`echo`, keyed ``other:<name>``."""
    return _record(_key(config, "other:"), config)


class Provider(Service[Any], provide="llm"):
    """Provides the ``llm`` service."""

    def __init__(self, ctx: Context, config: Any = None) -> None:
        super().__init__(ctx)
        runs[_key(config, "llm:")] += 1

    def resolved(self) -> Any:
        """The intercept config seen from the caller."""
        return self.__cordis_resolve_config__()


def dependent(ctx: Context, config: Any) -> Callable[[], None]:
    """Requires ``llm``; records the intercept config it sees."""
    key = _key(config, "dep:")
    seen[key] = ctx.llm.resolved()
    return _record(key, config)


# a function plugin declares metadata as attributes; written to its __dict__
# because a type checker does not model attributes on a `def` function
vars(dependent).update(inject=["llm"])


def failing(_ctx: Context, _config: Any) -> None:
    """Raise from the plugin body."""
    raise RuntimeError("body failed")


not_a_plugin = 42
"""A module attribute that is not a plugin."""


def ambient(ctx: Context, config: Any) -> None:
    """Record ``(ctx, current_context())`` seen by the body under the config's name."""
    seen[_key(config)] = (ctx, current_context())


def flaky(_ctx: Context, config: Any) -> Callable[[], None]:
    """Fail the first ``config["failures"]`` runs under the config's name, then load."""
    key = _key(config)
    dispose = _record(key, config)
    if runs[key] <= cast("dict[str, int]", config)["failures"]:
        raise RuntimeError(f"flaky run {runs[key]}")
    return dispose


def hung_close(_ctx: Context, config: Any) -> Callable[[], Coroutine[Any, Any, None]]:
    """Load at once; the disposer never finishes on its own."""
    _record(_key(config), config)

    async def close() -> None:
        await asyncio.sleep(10)

    return close


@plugin_meta(dispose_timeout=7.0)
def declared(_ctx: Context, config: Any) -> Callable[[], None]:
    """Declares ``dispose_timeout`` as plugin metadata."""
    return _record(_key(config), config)
