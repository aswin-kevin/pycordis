import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import apply_plugin

from pycordis.context import Context
from pycordis.errors import ServiceAccessError
from pycordis.events import EventsService, Listener
from pycordis.fiber import FiberState
from pycordis.inspect import FiberNode, render
from pycordis.testing import MemoryExporter, mock_context, settle


def _provider(ctx: Context, _config: Any) -> None:
    ctx.provide("db", object())


def _dependent(ctx: Context, _config: Any) -> None:
    return None


async def _app(root: Context) -> None:
    root.plugin(SimpleNamespace(name="provider", apply=_provider))
    root.plugin(SimpleNamespace(name="dependent", inject=["db"], apply=_dependent))
    await settle(root)


async def test_inspect_tree() -> None:
    root = Context()
    await _app(root)
    node = root.inspect()
    assert isinstance(node, FiberNode)
    assert (node.name, node.uid, node.state) == ("root", 0, FiberState.ACTIVE)
    assert [child.name for child in node.children] == ["provider", "dependent"]
    provider, dependent = node.children
    assert provider.provides == ["db"]
    assert provider.injects == []
    assert [e.label for e in provider.effects] == ['ctx.provide("db")']
    assert dependent.injects == ["db"]
    assert dependent.state == FiberState.ACTIVE
    assert [e.label for e in node.effects] == ["ctx.plugin()", "ctx.plugin()"]


async def test_render_snapshot() -> None:
    root = Context()
    await _app(root)
    assert render(root.inspect()) == "\n".join(
        [
            "root #0 ACTIVE effects: ctx.plugin(), ctx.plugin()",
            '├─ provider #1 ACTIVE provides: db effects: ctx.provide("db")',
            "└─ dependent #2 ACTIVE injects: db",
        ]
    )


async def test_render_nested() -> None:
    root = Context()

    def outer(ctx: Context, _config: Any) -> None:
        ctx.plugin(apply_plugin("inner", lambda _ctx, _config: None))

    root.plugin(SimpleNamespace(name="outer", apply=outer))
    root.plugin(SimpleNamespace(name="waiting", inject=["missing"], apply=_dependent))
    await settle(root)
    assert render(root.inspect()) == "\n".join(
        [
            "root #0 ACTIVE effects: ctx.plugin(), ctx.plugin()",
            "├─ outer #1 ACTIVE effects: ctx.plugin()",
            "│  └─ inner #3 ACTIVE",
            "└─ waiting #2 PENDING injects: missing",
        ]
    )


# -- testing helpers ----------------------------------------------------------


async def test_mock_context_captures_logs() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as ctx:
        ctx.logger("app").debug("hello %s", "world")
        ctx.logger("app").warn("careful")
    assert [m.type for m in logs.messages] == ["debug", "warn"]
    assert logs.text() == ["hello world", "careful"]
    assert logs.text("warn") == ["careful"]
    # registered as a root effect: removed with the root
    assert all(e is not logs for e in ctx.logger.exporters.values())


async def test_mock_context_default_exporter() -> None:
    async with mock_context() as ctx:
        assert any(isinstance(e, MemoryExporter) for e in ctx.logger.exporters.values())


async def test_settle_waits_for_async_loads() -> None:
    root = Context()
    loaded = asyncio.Event()

    async def slow(_ctx: Context, _config: Any) -> None:
        await asyncio.sleep(0.01)
        loaded.set()

    fiber = root.plugin(slow)
    await settle(root)
    assert loaded.is_set()
    assert fiber.state == FiberState.ACTIVE


async def test_settle_gives_up() -> None:
    root = Context()
    looping = True

    def restless(_ctx: Context, _config: Any) -> None:
        return None

    fiber = root.plugin(restless)

    def on_status(target: Any, _old: FiberState) -> None:
        # every activation immediately schedules another restart
        if looping and target is fiber and target.state == FiberState.ACTIVE:
            target.update({})

    root.on("internal/status", on_status)
    with pytest.raises(RuntimeError, match="pycordis did not settle"):
        await settle(root, max_iterations=3)
    looping = False
    await settle(root)
    assert fiber.state == FiberState.ACTIVE


# -- strict mode ---------------------------------------------------------------


async def _read_after_dispose(root: Context) -> None:
    held: list[Context] = []
    root.provide("db", 1)
    fiber = root.plugin(apply_plugin("dep", lambda ctx, _c: held.append(ctx), inject=["db"]))
    await fiber
    await fiber.dispose()
    with pytest.raises(ServiceAccessError):
        _ = held[0].db


async def test_strict_warns_on_disposed_read() -> None:
    logs = MemoryExporter()
    async with mock_context(logs, strict=True) as root:
        await _read_after_dispose(root)
    assert logs.text("warn") == ['service "db" read from disposed context <dep>']


async def test_strict_off_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYCORDIS_STRICT", raising=False)
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        await _read_after_dispose(root)
    assert logs.text("warn") == []


async def test_strict_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYCORDIS_STRICT", "1")
    assert Context().reflect.strict is True
    for off in ("", "0", "false", "OFF", " No "):
        monkeypatch.setenv("PYCORDIS_STRICT", off)
        assert Context().reflect.strict is False, off
    assert Context(strict=True).reflect.strict is True


async def _leak_listener(root: Context, monkeypatch: pytest.MonkeyPatch) -> None:
    held: list[Context] = []
    fiber = root.plugin(apply_plugin("leaky", lambda c, _: held.append(c)))
    await fiber
    held[0].on("app/ok", lambda: None)  # removed with the fiber
    # simulate a removal bug: the "app/leak" listener's disposer fails to unregister it
    unregister = EventsService.unregister

    def faulty(self: EventsService, name: str, callback: Listener) -> bool:
        return False if name == "app/leak" else unregister(self, name, callback)

    monkeypatch.setattr(EventsService, "unregister", faulty)
    held[0].on("app/leak", lambda: None)
    await fiber.dispose()


async def test_strict_warns_on_outlived_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Context(strict=True)
    logs = MemoryExporter()
    root.logger.exporter(logs)  # an ordinary root exporter sees the warning
    await _leak_listener(root, monkeypatch)
    assert logs.text("warn") == ['listener for "app/leak" outlived fiber <leaky>']
    await root.fiber.dispose()


async def test_outlived_listener_silent_when_not_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Context(strict=False)
    logs = MemoryExporter()
    root.logger.exporter(logs)
    await _leak_listener(root, monkeypatch)
    assert logs.text("warn") == []
    await root.fiber.dispose()


async def test_strict_restart_is_silent() -> None:
    """Restarts (explicit, by config update, by dependency churn) never warn in strict mode."""
    logs = MemoryExporter()
    reads: list[int] = []

    def body(ctx: Context, _config: Any) -> None:
        ctx.on("app/ping", lambda: None)
        ctx.effect(lambda: lambda: None, "noop")
        reads.append(ctx.db)

    def provider(value: int) -> SimpleNamespace:
        return apply_plugin("db", lambda ctx, _c: ctx.provide("db", value))

    async with mock_context(logs, strict=True) as root:
        first = root.plugin(provider(1))
        fiber = root.plugin(SimpleNamespace(name="svc", inject=["db"], apply=body), {"n": 0})
        await settle(root)
        await fiber.restart()
        await fiber.update({"n": 1})
        await first.dispose()  # the dependent unloads, then restarts with a new provider
        root.plugin(provider(2))
        await settle(root)
        assert fiber.state == FiberState.ACTIVE
        assert reads == [1, 1, 1, 2]
        await fiber.dispose()
    assert logs.text("warn") == []
