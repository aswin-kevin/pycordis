"""Fixture plugins for tests/test_cli.py.

:data:`calls` counts plugin-body runs, so a static ``inspect`` can be shown not
to run any plugin.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections import Counter
from dataclasses import dataclass
from typing import Any

from pycordis.context import Context
from pycordis.registry import plugin_meta
from pycordis.service import Service

calls: Counter[str] = Counter()
"""Body runs per plugin."""


class Store(Service[Any], provide="store"):
    """Provides the ``store`` service."""

    @dataclass
    class Config:
        """Store config."""

        label: str = "main"

    def __init__(self, ctx: Context, config: Store.Config) -> None:
        super().__init__(ctx)
        calls["store"] += 1
        self.label = config.label


def consumer(ctx: Context, _config: Any) -> None:
    """Requires ``store``."""
    calls["consumer"] += 1


# a function plugin declares metadata as attributes; written to its __dict__
# because a type checker does not model attributes on a `def` function
vars(consumer).update(inject=["store"])


def ready(ctx: Context, _config: Any) -> None:
    """Print ``ready`` and keep a background task running until disposed."""
    calls["ready"] += 1

    async def forever() -> None:
        await asyncio.Event().wait()

    ctx.spawn(forever())
    print("ready", flush=True)


def send_signals(_ctx: Context, config: Any) -> None:
    """Send the signals named in ``config["signals"]`` to this process, in order."""
    calls["send_signals"] += 1
    loop = asyncio.get_running_loop()
    for name in config["signals"]:
        loop.call_soon(os.kill, os.getpid(), getattr(signal, name))


def hang(ctx: Context, _config: Any) -> None:
    """Print ``ready``; its disposer never returns."""
    calls["hang"] += 1

    async def dispose() -> None:
        await asyncio.Event().wait()

    ctx.effect(lambda: dispose, "hang")
    print("ready", flush=True)


@plugin_meta(name="agent", inject=["llm"], provide="cache")
def agent(ctx: Context, _config: Any) -> None:
    """Waits for ``llm``; declares ``cache`` (half of a dependency cycle)."""
    calls["agent"] += 1


@plugin_meta(name="llm", inject=["cache"], provide="llm")
def llm(ctx: Context, _config: Any) -> None:
    """Waits for ``cache``; declares ``llm`` (the other half of the cycle)."""
    calls["llm"] += 1


async def stall(ctx: Context, _config: Any) -> None:
    """An async body that never finishes loading."""
    calls["stall"] += 1
    await asyncio.Event().wait()


def probe_health(ctx: Context, _config: Any) -> None:
    """Query the CLI's health server once ready, print the answers, then send SIGTERM."""
    calls["probe_health"] += 1
    health = ctx.get("health")

    async def fetch(path: str) -> str:
        reader, writer = await asyncio.open_connection(health.host, health.port)
        writer.write(f"GET {path} HTTP/1.1\r\nHost: probe\r\n\r\n".encode())
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        status = response.split(b" ", 2)[1].decode()
        return f"{status} {response.partition(b'\r\n\r\n')[2].decode()}"

    async def probe() -> None:
        ready = ""
        for _ in range(200):
            ready = await fetch("/readyz")
            if ready.startswith("200"):
                break
            await asyncio.sleep(0.01)
        print(f"health {health.host} {await fetch('/healthz')}", flush=True)
        print(f"ready {ready}", flush=True)
        os.kill(os.getpid(), signal.SIGTERM)

    ctx.spawn(probe())
