import asyncio
import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from conftest import capture_reports

from pycordis.context import Context
from pycordis.errors import CordisError, ValidationError
from pycordis.fiber import Fiber, FiberState
from pycordis.registry import plugin_meta
from pycordis.volatile import Volatile


@dataclass
class PortConfig:
    port: int


@dataclass
class LiveConfig:
    port: int
    limits: Volatile[dict[str, int]]


def _tracking(schema: Any) -> tuple[Any, list[Any]]:
    seen: list[Any] = []

    @plugin_meta(Config=schema)
    def plugin(ctx: Context, config: Any) -> None:
        seen.append(config)

    return plugin, seen


def _config_recorder(target: list[Any]) -> Callable[..., None]:
    """An ``internal/commit`` / ``internal/update`` listener recording the config.

    Both events pass ``(fiber, config, ...)``; the update veto's ``next`` is ignored.
    """

    def listener(fiber: Fiber, config: Any, *_: object) -> None:
        target.append(config)

    return listener


async def test_update_restarts_with_new_config() -> None:
    root = Context()
    plugin, seen = _tracking(PortConfig)
    commits: list[tuple[Fiber, Any]] = []

    def on_commit(fiber: Fiber, config: Any) -> None:
        commits.append((fiber, config))

    root.on("internal/commit", on_commit)
    fiber = await root.plugin(plugin, {"port": 1})
    assert fiber.raw_config == {"port": 1}
    assert fiber.effective_config == PortConfig(1)
    result = fiber.update({"port": 2})
    assert isinstance(result, asyncio.Future)
    assert commits == [(fiber, {"port": 2})]
    assert fiber.raw_config == {"port": 2}
    await result
    assert seen == [PortConfig(1), PortConfig(2)]
    assert fiber.effective_config == PortConfig(2)
    assert fiber.state == FiberState.ACTIVE


async def test_update_no_save_suppresses_commit() -> None:
    root = Context()
    plugin, _ = _tracking(PortConfig)
    commits: list[Any] = []
    root.on("internal/commit", _config_recorder(commits))
    fiber = await root.plugin(plugin, {"port": 1})
    await fiber.update({"port": 3}, no_save=True)
    assert commits == []
    assert fiber.effective_config == PortConfig(3)


async def test_update_invalid_config_raises_synchronously() -> None:
    root = Context()
    plugin, seen = _tracking(PortConfig)
    commits: list[Any] = []
    root.on("internal/commit", _config_recorder(commits))
    fiber = await root.plugin(plugin, {"port": 1})
    with pytest.raises(ValidationError):
        fiber.update({"port": "x"})
    assert commits == []
    assert fiber.raw_config == {"port": 1}
    assert fiber.state == FiberState.ACTIVE
    assert fiber.effective_config == PortConfig(1)
    await asyncio.sleep(0)
    assert seen == [PortConfig(1)]


async def test_update_veto_publishes_resolved_config() -> None:
    root = Context()
    seen: list[Any] = []
    vetoes: list[Any] = []

    @plugin_meta(Config=PortConfig)
    def plugin(ctx: Context, config: Any) -> None:
        seen.append(config)

        def veto(fiber: Fiber, cfg: Any, next: Callable[[], object]) -> None:
            vetoes.append((fiber, cfg))

        ctx.on("internal/update", veto)

    fiber = await root.plugin(plugin, {"port": 1})
    result = fiber.update({"port": 5})
    assert result.done()
    await result
    assert vetoes == [(fiber, PortConfig(5))]
    assert seen == [PortConfig(1)]
    assert fiber.effective_config == PortConfig(5)
    assert fiber.raw_config == {"port": 5}


async def test_update_listener_calling_next_restarts() -> None:
    root = Context()
    seen: list[Any] = []
    calls: list[Any] = []

    def plugin(ctx: Context, config: Any) -> None:
        seen.append(config)

        async def listener(fiber: Fiber, cfg: Any, next: Any) -> None:
            calls.append(cfg)
            await next()

        ctx.on("internal/update", listener)

    fiber = await root.plugin(plugin, 1)
    await fiber.update(2)
    assert calls == [2]
    assert seen == [1, 2]
    assert fiber.effective_config == 2


async def test_update_recovers_failed_fiber() -> None:
    root = Context()
    reports = capture_reports(root)
    plugin, seen = _tracking(PortConfig)
    fiber = root.plugin(plugin, {"port": "bad"})
    with pytest.raises(ValidationError):
        await fiber
    assert fiber.state == FiberState.FAILED
    await fiber.update({"port": 7})
    assert fiber.state == FiberState.ACTIVE
    assert seen == [PortConfig(7)]
    assert fiber.effective_config == PortConfig(7)
    assert len(reports) == 1


async def test_update_failure_raises_when_awaited_and_is_quiet_when_dropped() -> None:
    root = Context()
    reports = capture_reports(root)

    def plugin(ctx: Context, config: Any) -> None:
        if config == "boom":
            raise RuntimeError("boom")

    fiber = await root.plugin(plugin, "ok")
    with pytest.raises(RuntimeError, match="boom"):
        await fiber.update("boom")
    assert fiber.state == FiberState.FAILED
    assert len(reports) == 1
    # a dropped failing update is reported once by the fiber, never as an
    # unretrieved future exception
    dropped = fiber.update("boom")
    await asyncio.wait([dropped])
    del dropped
    gc.collect()
    await asyncio.sleep(0)
    assert len(reports) == 2
    assert fiber.state == FiberState.FAILED


async def test_volatile_only_update_keeps_running() -> None:
    root = Context()
    plugin, seen = _tracking(LiveConfig)
    fiber = await root.plugin(plugin, {"port": 1, "limits": {"rate": 10}})
    effective = fiber.effective_config
    ref = effective.limits
    assert ref.get()["rate"] == 10
    result = fiber.update({"port": 1, "limits": {"rate": 20}})
    assert result.done()
    await result
    assert len(seen) == 1
    assert fiber.effective_config is effective
    assert effective.limits is ref
    assert ref.get()["rate"] == 20
    assert fiber.raw_config == {"port": 1, "limits": {"rate": 20}}


async def test_resolve_config_does_not_commit_volatile() -> None:
    root = Context()
    plugin, _ = _tracking(LiveConfig)
    fiber = await root.plugin(plugin, {"port": 1, "limits": {"rate": 10}})
    candidate = fiber.resolve_config({"port": 1, "limits": {"rate": 99}})
    assert candidate.limits.get()["rate"] == 99
    assert fiber.effective_config.limits.get()["rate"] == 10


async def test_mixed_update_restarts() -> None:
    root = Context()
    plugin, seen = _tracking(LiveConfig)
    fiber = await root.plugin(plugin, {"port": 1, "limits": {"rate": 10}})
    effective = fiber.effective_config
    await fiber.update({"port": 2, "limits": {"rate": 20}})
    assert len(seen) == 2
    assert fiber.effective_config is not effective
    assert fiber.effective_config.port == 2
    assert fiber.effective_config.limits.get()["rate"] == 20
    assert effective.limits.get()["rate"] == 10


async def test_equal_config_without_volatile_restarts() -> None:
    root = Context()
    plugin, seen = _tracking(PortConfig)
    fiber = await root.plugin(plugin, {"port": 1})
    await fiber.update({"port": 1})
    assert len(seen) == 2


async def test_volatile_only_change_skips_update_listeners() -> None:
    root = Context()
    seen: list[Any] = []
    vetoes: list[Any] = []
    commits: list[Any] = []
    root.on("internal/commit", _config_recorder(commits))

    @plugin_meta(Config=LiveConfig)
    def plugin(ctx: Context, config: Any) -> None:
        seen.append(config)
        ctx.on("internal/update", _config_recorder(vetoes))

    fiber = await root.plugin(plugin, {"port": 1, "limits": {"rate": 10}})
    effective = fiber.effective_config
    held = effective.limits
    result = fiber.update({"port": 1, "limits": {"rate": 20}})
    assert result.done()
    assert vetoes == []
    assert commits == [{"port": 1, "limits": {"rate": 20}}]
    assert held.get()["rate"] == 20
    await fiber.update({"port": 1, "limits": {"rate": 30}}, no_save=True)
    assert held.get()["rate"] == 30
    assert fiber.effective_config is effective
    assert fiber.raw_config == {"port": 1, "limits": {"rate": 30}}
    assert len(seen) == 1 and vetoes == []


async def test_vetoed_mixed_change_orphans_held_refs() -> None:
    root = Context()

    @plugin_meta(Config=LiveConfig)
    def plugin(ctx: Context, config: Any) -> None:
        ctx.on("internal/update", lambda *_: None)

    fiber = await root.plugin(plugin, {"port": 1, "limits": {"rate": 10}})
    held = fiber.effective_config.limits
    await fiber.update({"port": 2, "limits": {"rate": 20}})  # vetoed, published
    await fiber.update({"port": 2, "limits": {"rate": 30}})  # volatile-only
    # cordis parity: the vetoed candidate's refs replaced the originals
    assert held.get()["rate"] == 10
    assert fiber.effective_config.limits.get()["rate"] == 30


async def test_async_listener_awaiting_next() -> None:
    root = Context()
    seen: list[Any] = []

    @plugin_meta(Config=LiveConfig)
    def plugin(ctx: Context, config: Any) -> None:
        seen.append(config)

        async def listener(fiber: Fiber, cfg: Any, next: Any) -> None:
            await next()

        ctx.on("internal/update", listener)

    fiber = await root.plugin(plugin, {"port": 1, "limits": {"rate": 10}})
    ref = fiber.effective_config.limits
    await fiber.update({"port": 1, "limits": {"rate": 20}})
    assert ref.get()["rate"] == 20 and len(seen) == 1
    await fiber.update({"port": 2, "limits": {"rate": 20}})
    assert len(seen) == 2
    assert fiber.effective_config.port == 2


async def test_async_listener_error_before_next() -> None:
    root = Context()
    reports = capture_reports(root)
    seen: list[Any] = []

    def plugin(ctx: Context, config: Any) -> None:
        seen.append(config)

        async def listener(fiber: Fiber, cfg: Any, next: Any) -> None:
            raise RuntimeError("listener")

        ctx.on("internal/update", listener)

    fiber = await root.plugin(plugin, 1)
    with pytest.raises(RuntimeError, match="listener"):
        await fiber.update(2)
    assert seen == [1]
    assert len(reports) == 1
    # dropped: still reported, never an unretrieved-exception warning
    dropped = fiber.update(3)
    await asyncio.wait([dropped])
    del dropped
    gc.collect()
    await asyncio.sleep(0)
    assert len(reports) == 2


async def test_volatile_update_frozen_dataclass() -> None:
    @dataclass(frozen=True)
    class Frozen:
        port: int
        limits: Volatile[dict[str, int]]

    root = Context()
    plugin, seen = _tracking(Frozen)
    fiber = await root.plugin(plugin, {"port": 1, "limits": {"rate": 10}})
    effective = fiber.effective_config
    await fiber.update({"port": 1, "limits": {"rate": 20}})
    assert fiber.effective_config is effective
    assert effective.limits.get()["rate"] == 20
    assert len(seen) == 1


async def test_stale_update_does_not_publish() -> None:
    root = Context()
    seen: list[Any] = []
    pending: list[Any] = []

    def plugin(ctx: Context, config: Any) -> None:
        seen.append(config)

        async def listener(fiber: Fiber, cfg: Any, next: Any) -> None:
            gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            pending.append(gate)
            await gate
            await next()

        ctx.on("internal/update", listener)

    fiber = await root.plugin(plugin, 1)
    first = fiber.update(2)
    await asyncio.sleep(0)
    fiber.raw_config = 3  # bumps the config revision
    pending[0].set_result(None)
    await first
    assert seen[-1] == 3
    # restart path ran, so its config is the published one
    assert fiber.effective_config == 3


async def test_raw_config_setter_does_not_update() -> None:
    root = Context()
    plugin, seen = _tracking(PortConfig)
    fiber = await root.plugin(plugin, {"port": 1})
    fiber.raw_config = {"port": 9}
    await asyncio.sleep(0)
    assert seen == [PortConfig(1)]
    assert fiber.effective_config == PortConfig(1)
    await fiber.restart()
    assert seen == [PortConfig(1), PortConfig(9)]
    assert fiber.effective_config == PortConfig(9)


async def test_restart_disposed_fiber_raises() -> None:
    root = Context()
    plugin, _ = _tracking(PortConfig)
    fiber = await root.plugin(plugin, {"port": 1})
    await fiber.dispose()
    with pytest.raises(CordisError) as info:
        await fiber.restart()
    assert info.value.code == "INACTIVE_EFFECT"
    with pytest.raises(CordisError):
        fiber.update({"port": 2})


async def test_update_pending_fiber_defers_resolution() -> None:
    root = Context()
    seen: list[Any] = []

    @plugin_meta(inject=["missing"], Config=PortConfig)
    def plugin(ctx: Context, config: Any) -> None:
        seen.append(config)

    fiber = root.plugin(plugin, {"port": 1})
    assert fiber.state == FiberState.PENDING
    # unresolved, so even an invalid candidate is accepted until activation
    result = fiber.update({"port": "later"})
    await result
    assert fiber.raw_config == {"port": "later"}
    assert fiber.state == FiberState.PENDING
    assert seen == []


async def test_listener_returning_resolved_future_is_handled_synchronously() -> None:
    root = Context()
    seen: list[Any] = []

    def plugin(ctx: Context, config: Any) -> None:
        seen.append(config)

        def listener(fiber: Fiber, cfg: Any, next: Any) -> asyncio.Future[None]:
            done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            done.set_result(None)
            return done

        ctx.on("internal/update", listener)

    fiber = await root.plugin(plugin, 1)
    result = fiber.update(2)
    assert result.done()
    assert fiber.effective_config == 2
    assert seen == [1]
