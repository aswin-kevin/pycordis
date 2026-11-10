import asyncio
from typing import Any

import pytest
from conftest import capture_reports

from pycordis.context import Context
from pycordis.errors import CordisError


async def test_spawned_task_cancelled_on_dispose() -> None:
    root = Context()
    started = asyncio.Event()
    cancelled: list[bool] = []
    tasks: list[asyncio.Task[None]] = []

    async def forever() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    def plugin(ctx: Context, _config: Any) -> None:
        tasks.append(ctx.spawn(forever(), name="worker"))

    fiber = root.plugin(plugin)
    await fiber
    await started.wait()
    assert tasks[0].get_name() == "worker"
    # owned by the calling (child) fiber, not the root
    assert tasks[0] in fiber.owned_tasks()
    assert tasks[0] not in root.fiber.owned_tasks()
    await fiber.dispose()
    assert cancelled == [True]
    assert tasks[0].cancelled()


async def test_spawned_failure_logged() -> None:
    root = Context()
    errors = capture_reports(root)

    async def boom() -> None:
        raise ValueError("boom")

    task = root.spawn(boom())
    with pytest.raises(ValueError):
        await task
    assert [str(e) for e in errors] == ["boom"]


async def test_spawn_returns_result() -> None:
    root = Context()

    async def value() -> int:
        return 42

    assert await root.spawn(value()) == 42


async def test_spawn_on_disposed_fiber_raises_and_closes_coroutine() -> None:
    root = Context()
    held: list[Context] = []
    fiber = root.plugin(lambda ctx, _config: held.append(ctx))
    await fiber
    await fiber.dispose()

    async def work() -> None:
        return None

    coro = work()
    with pytest.raises(CordisError) as info:
        held[0].spawn(coro)
    assert info.value.code == "INACTIVE_EFFECT"
    # closed, so no "coroutine was never awaited" warning
    assert coro.cr_frame is None
