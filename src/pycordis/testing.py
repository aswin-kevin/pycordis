"""Testing helpers with no test-framework dependency.

``async with mock_context(logs) as ctx`` gives a fresh root context whose log
messages are captured by a :class:`MemoryExporter`; ``await settle(ctx)`` waits
until no fiber has lifecycle work in flight.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pycordis.context import Context
from pycordis.logger import Logger, LoggerLevel, Message

if TYPE_CHECKING:
    from pycordis.fiber import Fiber

__all__ = ["MemoryExporter", "mock_context", "settle"]


class MemoryExporter:
    """Logger exporter keeping every message (all levels) in :attr:`messages`."""

    colors: int = 0
    """ANSI color depth used by :meth:`text` (0: plain text)."""
    levels: dict[str, int]
    """Per-logger-name thresholds; ``"default"`` is DEBUG, so every level is kept."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        """Captured messages, oldest first."""
        self.levels = {"default": LoggerLevel.DEBUG}

    def export(self, message: Message) -> None:
        """Record one message."""
        self.messages.append(message)

    def text(self, level: str | int | None = None) -> list[str]:
        """Formatted message texts, optionally only of one level.

        ``level`` is a message type (``"warn"``) or a :class:`LoggerLevel`.
        """
        return [
            Logger.format(self, message)
            for message in self.messages
            if level is None
            or (message.type == level if isinstance(level, str) else message.level == level)
        ]


@asynccontextmanager
async def mock_context(
    exporter: MemoryExporter | None = None, *, strict: bool | None = None
) -> AsyncGenerator[Context]:
    """Yield a new root context with ``exporter`` (default: a new one) capturing logs.

    To read the captured messages, pass your own exporter::

        logs = MemoryExporter()
        async with mock_context(logs) as ctx:
            ...
        assert logs.text("warn") == [...]

    The exporter is registered with ``root.logger.exporter()`` (a root effect),
    and the root is disposed on exit.
    """
    root = Context(strict=strict)
    root.logger.exporter(exporter if exporter is not None else MemoryExporter())
    async with root:
        yield root


def _fibers(root: Context) -> list[Fiber]:
    fibers = [root.fiber]
    for runtime in list(root.registry.values()):
        fibers.extend(runtime.fibers)
    return fibers


async def settle(ctx: Context, *, max_iterations: int = 1000) -> None:
    """Wait until no fiber of ``ctx``'s application has a load or unload in flight.

    Each iteration waits for every in-flight transition; transitions keep
    starting new ones (e.g. a restart loop) for more than ``max_iterations``
    raises ``RuntimeError("pycordis did not settle")``. A plugin body that never
    finishes blocks it indefinitely; bound it with ``async with asyncio.timeout(...)``.
    """
    root = ctx.root
    for _ in range(max_iterations):
        pending: set[asyncio.Task[Any]] = {  # Task is invariant: any result type
            fiber.inertia for fiber in _fibers(root) if fiber.inertia is not None
        }
        if not pending:
            return
        await asyncio.wait(pending)
    raise RuntimeError("pycordis did not settle")
