import asyncio
from typing import Any

import pytest
from conftest import capture_reports

from pycordis.context import Context
from pycordis.errors import CordisError
from pycordis.timer import TimerService

TICK = 0.02


async def test_timer_is_builtin_and_traced() -> None:
    root = Context()
    assert isinstance(root.timer, TimerService)
    held: list[Context] = []
    fiber = root.plugin(lambda ctx, _config: held.append(ctx))
    await fiber
    # reading through a plugin context yields that use site, without inject
    assert held[0].timer.ctx is held[0]
    assert "timer" not in fiber.inject


async def test_sleep_completes() -> None:
    root = Context()
    await root.sleep(0)
    await root.timer.sleep(TICK)
    # the internal sleep effect is removed once the sleep finishes
    assert root.fiber.get_effects() == []


async def test_sleep_in_plugin_cancelled_on_dispose() -> None:
    root = Context()
    outcomes: list[str] = []

    async def plugin(ctx: Context, _config: Any) -> None:
        try:
            await ctx.sleep(100)
        except asyncio.CancelledError:
            outcomes.append("cancelled")
            raise

    fiber = root.plugin(plugin)
    for _ in range(5):
        await asyncio.sleep(0)
    await asyncio.wait_for(fiber.dispose(), 1)
    assert outcomes == ["cancelled"]


async def test_sleep_in_foreign_task_cancelled_by_fiber_unload() -> None:
    root = Context()
    held: list[Context] = []
    fiber = root.plugin(lambda ctx, _config: held.append(ctx))
    await fiber
    # a task the fiber does not own: the sleep effect itself cancels the wait
    task = asyncio.ensure_future(held[0].sleep(100))
    await asyncio.sleep(0)
    assert [e.label for e in fiber.get_effects()] == ["ctx.timer.sleep()"]
    await asyncio.wait_for(fiber.dispose(), 1)
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_set_timeout_fires_once_and_cleans_up() -> None:
    root = Context()
    calls: list[int] = []
    root.set_timeout(lambda: calls.append(1), TICK)
    assert [e.label for e in root.fiber.get_effects()] == ["ctx.timer.set_timeout()"]
    await asyncio.sleep(TICK * 5)
    assert calls == [1]
    assert root.fiber.get_effects() == []


async def test_set_timeout_disposer_cancels() -> None:
    root = Context()
    calls: list[int] = []
    dispose = root.set_timeout(lambda: calls.append(1), TICK)
    await dispose()
    await asyncio.sleep(TICK * 6)
    assert calls == []


async def test_set_timeout_coroutine_and_error() -> None:
    root = Context()
    errors = capture_reports(root)
    done = asyncio.Event()

    async def later() -> None:
        done.set()

    def fail() -> None:
        raise ValueError("timeout failed")

    root.set_timeout(later, 0)
    root.set_timeout(fail, 0)
    await asyncio.wait_for(done.wait(), 1)
    await asyncio.sleep(TICK)
    assert [str(e) for e in errors] == ["timeout failed"]


async def test_set_interval_fires_until_disposed() -> None:
    root = Context()
    ticks: list[int] = []
    held: list[Context] = []

    def plugin(ctx: Context, _config: Any) -> None:
        held.append(ctx)
        ctx.set_interval(lambda: ticks.append(len(ticks)), TICK)

    fiber = root.plugin(plugin)
    await fiber
    while len(ticks) < 3:
        await asyncio.sleep(TICK)
    await fiber.dispose()
    count = len(ticks)
    assert count >= 3
    await asyncio.sleep(TICK * 6)
    assert len(ticks) == count


async def test_set_interval_disposer_stops() -> None:
    root = Context()
    ticks: list[int] = []
    dispose = root.set_interval(lambda: ticks.append(1), TICK)
    while len(ticks) < 2:
        await asyncio.sleep(TICK)
    await dispose()
    count = len(ticks)
    await asyncio.sleep(TICK * 6)
    assert len(ticks) == count
    assert root.fiber.get_effects() == []


async def test_debounce_coalesces() -> None:
    root = Context()
    calls: list[tuple[Any, ...]] = []
    debounced = root.timer.debounce(lambda *args: calls.append(args), TICK * 3)
    debounced(1)
    debounced(2)
    debounced(3)
    await asyncio.sleep(TICK * 12)
    assert calls == [(3,)]
    assert root.fiber.get_effects() == []


async def test_throttle_leading_and_trailing() -> None:
    root = Context()
    calls: list[int] = []
    throttled = root.timer.throttle(calls.append, TICK * 5)
    throttled(1)
    throttled(2)
    throttled(3)
    assert calls == [1]
    # trailing call at ~5 ticks opens a second window until ~10 ticks
    await asyncio.sleep(TICK * 8)
    assert calls == [1, 3]
    await asyncio.sleep(TICK * 10)
    assert calls == [1, 3]
    throttled(4)
    assert calls == [1, 3, 4]


async def test_timers_owned_by_calling_fiber() -> None:
    root = Context()
    calls: list[int] = []
    held: list[Context] = []
    fiber = root.plugin(lambda ctx, _config: held.append(ctx))
    await fiber
    held[0].set_timeout(lambda: calls.append(1), TICK)
    debounced = held[0].timer.debounce(lambda: calls.append(2), TICK)
    debounced()
    assert [e.label for e in fiber.get_effects()] == [
        "ctx.timer.set_timeout()",
        "ctx.timer.debounce()",
    ]
    assert [e.label for e in root.fiber.get_effects()] == ["ctx.plugin()"]
    await fiber.dispose()
    await asyncio.sleep(TICK * 6)
    assert calls == []
    with pytest.raises(CordisError):
        held[0].set_timeout(lambda: None, TICK)
    # debounced / throttled functions of a disposed fiber are silent no-ops
    debounced()
    throttled = held[0].timer.throttle(lambda: calls.append(3), TICK)
    throttled()
    await asyncio.sleep(TICK * 3)
    assert calls == []
