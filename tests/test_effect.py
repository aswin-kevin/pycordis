import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Iterator
from typing import Any

import pytest
import pytest_asyncio
from conftest import capture_reports_into

from pycordis.context import Context
from pycordis.effect import EffectDisposer, EffectMeta
from pycordis.errors import CordisError
from pycordis.fiber import FiberState


@pytest_asyncio.fixture
async def reported(monkeypatch: pytest.MonkeyPatch) -> list[BaseException]:
    """Capture errors reported through the logger of every root created in the test."""
    errors: list[BaseException] = []
    original = Context.__init__

    def init(self: Context) -> None:
        original(self)
        capture_reports_into(self, errors)

    monkeypatch.setattr(Context, "__init__", init)
    return errors


def notes(error: BaseException) -> list[str]:
    return list(getattr(error, "__notes__", []))


# --- shapes ---------------------------------------------------------------


async def test_disposers_run_in_reverse_and_once() -> None:
    root = Context()
    log: list[str] = []

    def body() -> Iterator[Callable[[], None]]:
        yield lambda: log.append("a")
        yield lambda: log.append("b")

    d = root.fiber.effect(body, "gen")
    await d()
    await d()
    assert log == ["b", "a"]


async def test_none_effect() -> None:
    root = Context()
    d = root.fiber.effect(lambda: None)
    assert root.fiber.get_effects() == [EffectMeta("anonymous", [])]
    await d()
    assert root.fiber.get_effects() == []


async def test_sync_context_manager_effect() -> None:
    root = Context()
    log: list[str] = []

    @contextlib.contextmanager
    def cm() -> Generator[None, None, None]:
        log.append("enter")
        yield
        log.append("exit")

    d = root.fiber.effect(cm)
    assert log == ["enter"]
    fut = d()
    assert fut.done()
    assert log == ["enter", "exit"]


async def test_async_context_manager_effect() -> None:
    root = Context()
    log: list[str] = []

    @contextlib.asynccontextmanager
    async def cm() -> AsyncGenerator[None, None]:
        log.append("enter")
        yield
        log.append("exit")

    d = root.fiber.effect(cm)
    await d
    await d()
    assert log == ["enter", "exit"]


async def test_async_callable_disposer() -> None:
    root = Context()
    log: list[str] = []

    async def cleanup() -> None:
        await asyncio.sleep(0)
        log.append("cleanup")

    d = root.fiber.effect(lambda: cleanup)
    fut = d()
    assert not fut.done()
    await fut
    assert log == ["cleanup"]


async def test_coroutine_effect_and_await_returns_disposer() -> None:
    root = Context()
    log: list[str] = []

    async def body() -> Callable[[], None]:
        await asyncio.sleep(0)
        log.append("setup")
        return lambda: log.append("cleanup")

    d = root.fiber.effect(body, "co")
    dispose = await d
    assert log == ["setup"]
    await dispose()
    assert log == ["setup", "cleanup"]
    await dispose()
    await d()
    assert log == ["setup", "cleanup"]


async def test_await_sync_effect_returns_disposer() -> None:
    root = Context()
    log: list[str] = []
    d = root.fiber.effect(lambda: lambda: log.append("x"))
    dispose = await d
    fut = dispose()
    assert fut.done() and log == ["x"]
    assert d().done() and log == ["x"]


async def test_async_generator_effect() -> None:
    root = Context()
    log: list[str] = []

    async def body() -> AsyncIterator[Callable[[], None]]:
        yield lambda: log.append("a")
        await asyncio.sleep(0)
        yield lambda: log.append("b")

    d = root.fiber.effect(body)
    await d
    await d()
    assert log == ["b", "a"]


async def test_async_generator_stops_when_disposed() -> None:
    root = Context()
    log: list[str] = []
    gate = asyncio.Event()

    async def body() -> AsyncIterator[Callable[[], None]]:
        try:
            yield lambda: log.append("a")
            await gate.wait()
            yield lambda: log.append("b")
            log.append("continued")
            yield lambda: log.append("c")
        finally:
            log.append("closed")

    d = root.fiber.effect(body)
    await asyncio.sleep(0.01)
    fut = d()
    await asyncio.sleep(0)
    assert log == [] and not fut.done()
    gate.set()
    await fut
    assert log == ["closed", "b", "a"]


async def test_effect_returning_effect_disposer() -> None:
    root = Context()
    log: list[str] = []
    d = root.fiber.effect(lambda: root.fiber.effect(lambda: lambda: log.append("inner"), "inner"))
    await d()
    assert log == ["inner"]


# --- invariants -------------------------------------------------------------


async def test_wrapper_registered_before_execute() -> None:
    root = Context()
    seen: list[list[EffectMeta]] = []

    def body() -> None:
        seen.append(root.fiber.get_effects())

    root.fiber.effect(body, "early")
    assert seen == [[EffectMeta("early", [])]]


async def test_sync_setup_failure_rolls_back() -> None:
    root = Context()
    log: list[str] = []

    def gen() -> Iterator[Callable[[], None]]:
        yield lambda: log.append("a")
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom") as info:
        root.fiber.effect(gen, "gen")
    assert log == ["a"]
    assert notes(info.value) == ["in effect gen of <root>"]
    assert root.fiber.get_effects() == []


async def test_sync_setup_failure_with_async_rollback(reported: list[BaseException]) -> None:
    root = Context()
    log: list[str] = []

    async def cleanup() -> None:
        log.append("cleanup")
        raise RuntimeError("cleanup failed")

    def gen() -> Iterator[Any]:
        yield cleanup
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        root.fiber.effect(gen, "gen")
    await asyncio.sleep(0.01)
    assert log == ["cleanup"]
    assert [str(e) for e in reported] == ["cleanup failed"]
    assert notes(reported[0]) == ["in effect gen of <root>"]


async def test_async_setup_failure_disposes_and_reports(reported: list[BaseException]) -> None:
    root = Context()
    log: list[str] = []

    async def body() -> AsyncIterator[Callable[[], None]]:
        yield lambda: log.append("a")
        raise ValueError("late")

    d = root.fiber.effect(body, "agen")
    with pytest.raises(ValueError, match="late") as info:
        await d
    await asyncio.sleep(0)
    assert log == ["a"]
    assert reported == [info.value]
    assert notes(info.value) == ["in effect agen of <root>"]
    assert root.fiber.get_effects() == []
    # cordis: a later dispose waits for the failed setup and re-raises its error
    with pytest.raises(ValueError, match="late"):
        await d()
    assert log == ["a"]
    assert reported == [info.value]


async def test_dispose_during_async_setup_waits() -> None:
    root = Context()
    log: list[str] = []
    ready = asyncio.Event()

    async def body() -> Callable[[], None]:
        await ready.wait()
        log.append("setup-done")
        return lambda: log.append("cleanup")

    d = root.fiber.effect(body)
    fut = d()
    await asyncio.sleep(0)
    assert not fut.done()
    ready.set()
    await fut
    assert log == ["setup-done", "cleanup"]
    assert d() is not fut and d().done()


async def test_dispose_during_failing_async_setup() -> None:
    root = Context()

    async def body() -> None:
        await asyncio.sleep(0)
        raise ValueError("bad")

    d = root.fiber.effect(body, "x")
    fut = d()
    with pytest.raises(ValueError, match="bad") as info:
        await fut
    assert notes(info.value) == ["in effect x of <root>"]


def _sole_disposer(ctx: Context) -> EffectDisposer:
    """The one disposer registered on ``ctx.fiber``.

    Re-entrancy tests need the effect's disposer while its body is still running,
    before ``effect()`` has returned it, so they read the fiber's disposer list.
    """
    (wrapper,) = ctx.fiber.disposables()
    assert isinstance(wrapper, EffectDisposer)
    return wrapper


async def test_reentrant_dispose_during_sync_execute() -> None:
    root = Context()
    log: list[str] = []
    futures: list[asyncio.Future[None]] = []

    def body() -> Callable[[], None]:
        wrapper = _sole_disposer(root)
        futures.append(wrapper())
        return lambda: log.append("cleanup")

    d = root.fiber.effect(body)
    assert not futures[0].done()
    await futures[0]
    assert log == ["cleanup"]
    assert d().done()


async def test_reentrant_dispose_then_sync_failure() -> None:
    root = Context()
    futures: list[asyncio.Future[None]] = []

    def body() -> None:
        wrapper = _sole_disposer(root)
        futures.append(wrapper())
        raise ValueError("sync")

    with pytest.raises(ValueError, match="sync"):
        root.fiber.effect(body)
    with pytest.raises(ValueError, match="sync"):
        await futures[0]


async def test_nested_effect_meta() -> None:
    root = Context()

    def outer() -> Iterator[EffectDisposer]:
        yield root.fiber.effect(lambda: None, "inner")

    d = root.fiber.effect(outer, "outer")
    assert root.fiber.get_effects() == [EffectMeta("outer", [EffectMeta("inner", [])])]
    assert d.meta == EffectMeta("outer", [EffectMeta("inner", [])])
    await d()
    assert root.fiber.get_effects() == []


async def test_invalid_effect(reported: list[BaseException]) -> None:
    root = Context()
    with pytest.raises(TypeError, match="Invalid effect"):
        root.fiber.effect(lambda: 42)
    with pytest.raises(TypeError, match="Invalid effect"):
        root.fiber.effect(lambda: iter([42]))

    async def body() -> int:
        return 42

    d = root.fiber.effect(body, "co")
    with pytest.raises(TypeError, match="Invalid effect"):
        await d
    assert root.fiber.get_effects() == []


async def test_effect_on_inactive_fiber() -> None:
    root = Context()
    errors: list[CordisError] = []

    def cleanup() -> None:
        assert root.fiber.state == FiberState.UNLOADING
        try:
            root.fiber.effect(lambda: None)
        except CordisError as e:
            errors.append(e)

    root.fiber.effect(lambda: cleanup)
    await root.fiber.dispose()
    assert [e.code for e in errors] == ["INACTIVE_EFFECT"]
    assert str(errors[0]) == "cannot create effect on inactive context"

    root.fiber.uid = None
    with pytest.raises(CordisError) as info:
        root.fiber.effect(lambda: None)
    assert info.value.code == "INACTIVE_EFFECT"
    with pytest.raises(CordisError):
        root.fiber.assert_active()


async def test_raising_disposer_does_not_block_others(reported: list[BaseException]) -> None:
    root = Context()
    log: list[str] = []

    def boom() -> None:
        raise RuntimeError("boom")

    async def aboom() -> None:
        raise RuntimeError("aboom")

    def body() -> Iterator[Callable[[], Any]]:
        yield lambda: log.append("a")
        yield aboom
        yield boom
        yield lambda: log.append("d")

    d = root.fiber.effect(body, "multi")
    with pytest.raises(RuntimeError, match="boom") as info:
        await d()
    assert log == ["d", "a"]
    assert notes(info.value) == ["in effect multi of <root>"]
    assert [str(e) for e in reported] == ["aboom"]
    assert notes(reported[0]) == ["in effect multi of <root>"]


async def test_root_dispose_survives_raising_disposer(reported: list[BaseException]) -> None:
    root = Context()
    log: list[str] = []

    def boom() -> None:
        raise RuntimeError("boom")

    root.fiber.effect(lambda: lambda: log.append("first"), "first")
    root.fiber.effect(lambda: boom, "bad")
    root.fiber.effect(lambda: lambda: log.append("last"), "last")
    await root.fiber.dispose()
    assert log == ["last", "first"]
    assert [str(e) for e in reported] == ["boom"]
    assert notes(reported[0]) == ["in effect bad of <root>"]
    assert root.fiber.state == FiberState.ACTIVE
    assert root.fiber.get_effects() == []
    root.fiber.effect(lambda: None, "again")
    assert root.fiber.get_effects() == [EffectMeta("again", [])]


async def test_second_call_returns_in_flight_future() -> None:
    root = Context()
    gate = asyncio.Event()

    async def cleanup() -> None:
        await gate.wait()

    d = root.fiber.effect(lambda: cleanup)
    f1 = d()
    f2 = d()
    assert f1 is f2 and not f1.done()
    gate.set()
    await f1
    f3 = d()
    assert f3.done() and f3 is not f1


async def test_owner_unload_joins_in_flight_cleanup() -> None:
    root = Context()
    gate = asyncio.Event()
    log: list[str] = []

    async def cleanup() -> None:
        await gate.wait()
        log.append("cleanup")

    d = root.fiber.effect(lambda: cleanup, "slow")
    fut = d()
    # still visible to the owner while its async teardown is pending
    assert root.fiber.get_effects() == [EffectMeta("slow", [])]
    task = root.fiber.dispose()
    await asyncio.sleep(0.01)
    assert not task.done()
    gate.set()
    await task
    assert fut.done() and log == ["cleanup"]


async def test_root_dispose_during_async_setup() -> None:
    root = Context()
    log: list[str] = []
    ready = asyncio.Event()

    async def body() -> Callable[[], None]:
        await ready.wait()
        return lambda: log.append("cleanup")

    root.fiber.effect(body)
    task = root.fiber.dispose()
    await asyncio.sleep(0.01)
    assert not task.done()
    ready.set()
    await task
    assert log == ["cleanup"]


async def test_note_added_once(reported: list[BaseException]) -> None:
    root = Context()

    async def body() -> None:
        raise ValueError("once")

    d = root.fiber.effect(body, "e")
    with pytest.raises(ValueError) as info:
        await d
    with pytest.raises(ValueError):
        await d()
    assert notes(info.value) == ["in effect e of <root>"]
    assert reported == [info.value]


async def test_wait_returns_fiber() -> None:
    root = Context()
    assert await root.fiber.wait() is root.fiber


async def test_reentrant_dispose_during_async_setup() -> None:
    root = Context()
    log: list[str] = []
    futures: list[asyncio.Future[None]] = []

    async def setup() -> Callable[[], None]:
        await asyncio.sleep(0)
        return lambda: log.append("cleanup")

    def body() -> Any:
        wrapper = _sole_disposer(root)
        futures.append(wrapper())
        return setup()

    root.fiber.effect(body)
    await futures[0]
    assert log == ["cleanup"]


async def test_mixed_async_disposers_all_run(reported: list[BaseException]) -> None:
    root = Context()
    log: list[str] = []

    async def ok(tag: str) -> None:
        await asyncio.sleep(0)
        log.append(tag)

    async def bad() -> None:
        raise RuntimeError("bad")

    def body() -> Iterator[Callable[[], Any]]:
        yield lambda: ok("first")
        yield bad
        yield lambda: ok("second")
        yield lambda: ok("third")

    d = root.fiber.effect(body, "mixed")
    with pytest.raises(RuntimeError, match="bad"):
        await d()
    assert log == ["third", "second", "first"]
    assert reported == []


async def test_nested_failure_surfaces_through_outer() -> None:
    root = Context()

    def boom() -> None:
        raise RuntimeError("inner boom")

    def outer() -> Iterator[EffectDisposer]:
        yield root.fiber.effect(lambda: boom, "inner")

    d = root.fiber.effect(outer, "outer")
    with pytest.raises(RuntimeError, match="inner boom") as info:
        await d()
    assert notes(info.value) == ["in effect inner of <root>", "in effect outer of <root>"]


async def test_failed_async_setup_with_pending_dispose(reported: list[BaseException]) -> None:
    root = Context()
    log: list[str] = []

    async def cleanup() -> None:
        log.append("cleanup")

    def broken() -> None:
        raise RuntimeError("dispose failed")

    async def body() -> AsyncIterator[Callable[[], Any]]:
        yield cleanup
        yield broken
        await asyncio.sleep(0)
        raise ValueError("setup failed")

    d = root.fiber.effect(body, "x")
    await asyncio.sleep(0)
    fut = d()
    with pytest.raises(ValueError, match="setup failed"):
        await fut
    assert log == ["cleanup"]
    assert sorted(str(e) for e in reported) == ["dispose failed", "setup failed"]


async def test_cancelled_async_setup_still_disposes(reported: list[BaseException]) -> None:
    root = Context()
    log: list[str] = []
    setup_tasks: list[asyncio.Task[Any] | None] = []

    async def body() -> AsyncIterator[Callable[[], None]]:
        # async setup runs as a task of its own: the one to cancel
        setup_tasks.append(asyncio.current_task())
        yield lambda: log.append("cleanup")
        await asyncio.Event().wait()

    d = root.fiber.effect(body, "c")
    await asyncio.sleep(0.01)
    (task,) = setup_tasks
    assert task is not None
    task.cancel()
    await asyncio.sleep(0.01)
    assert task.cancelled() and reported == []
    assert log == ["cleanup"] and root.fiber.get_effects() == []
    await d()
    assert log == ["cleanup"]


async def test_setup_raising_cancelled_error_still_disposes(
    reported: list[BaseException],
) -> None:
    root = Context()
    log: list[str] = []

    async def body() -> AsyncIterator[Callable[[], None]]:
        yield lambda: log.append("cleanup")
        raise asyncio.CancelledError()

    d = root.fiber.effect(body, "ag")
    await asyncio.sleep(0.01)
    assert log == ["cleanup"] and root.fiber.get_effects() == []
    await asyncio.wait_for(d(), 1)
    assert log == ["cleanup"]
    assert reported == []


async def test_cancelled_setup_then_root_dispose() -> None:
    root = Context()
    log: list[str] = []

    async def body() -> AsyncIterator[Callable[[], None]]:
        yield lambda: log.append("cleanup")
        raise asyncio.CancelledError()

    root.fiber.effect(body, "ag")
    await root.fiber.dispose()
    assert log == ["cleanup"] and root.fiber.state == FiberState.ACTIVE


def _cancel_and_await_effect() -> Callable[[], Any]:
    async def worker() -> None:
        await asyncio.sleep(100)

    task = asyncio.ensure_future(worker())

    async def cleanup() -> None:
        task.cancel()
        await task

    return cleanup


async def test_root_dispose_survives_cancel_and_await_disposer(
    reported: list[BaseException],
) -> None:
    log: list[str] = []
    async with Context() as root:
        root.fiber.effect(lambda: lambda: log.append("first"), "first")
        root.fiber.effect(_cancel_and_await_effect, "canceller")
    assert log == ["first"]
    assert root.fiber.state == FiberState.ACTIVE and root.fiber.inertia is None
    assert await root.fiber.wait() is root.fiber
    root.fiber.effect(lambda: None, "after")
    assert [type(e) for e in reported] == [RuntimeError]
    assert isinstance(reported[0].__cause__, asyncio.CancelledError)
    assert notes(reported[0]) == ["in effect canceller of <root>"]


async def test_cancel_and_await_disposer_does_not_stop_others() -> None:
    root = Context()
    log: list[str] = []

    def body() -> Iterator[Callable[[], Any]]:
        yield lambda: log.append("a")
        yield _cancel_and_await_effect()

    d = root.fiber.effect(body, "gen")
    with pytest.raises(RuntimeError, match="disposer raised CancelledError"):
        await d()
    assert log == ["a"]


async def test_sync_disposer_raising_cancelled_error() -> None:
    root = Context()
    log: list[str] = []

    def cancel() -> None:
        raise asyncio.CancelledError()

    def body() -> Iterator[Callable[[], Any]]:
        yield lambda: log.append("a")
        yield cancel

    d = root.fiber.effect(body, "gen")
    with pytest.raises(RuntimeError, match="disposer raised CancelledError"):
        await d()
    assert log == ["a"]


async def test_cancelling_the_disposal_propagates() -> None:
    root = Context()
    gate = asyncio.Event()

    async def slow() -> None:
        await gate.wait()

    def body() -> Iterator[Callable[[], Any]]:
        yield lambda: None
        yield slow

    d = root.fiber.effect(body)
    fut = d()
    await asyncio.sleep(0)
    fut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fut


async def test_sync_setup_cancelled_error_rolls_back() -> None:
    root = Context()
    log: list[str] = []

    def gen() -> Iterator[Callable[[], None]]:
        yield lambda: log.append("a")
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        root.fiber.effect(gen, "gen")
    assert log == ["a"] and root.fiber.get_effects() == []


# -- generator effects yielding None (fixture-style misuse) ----------------------

GEN_NONE = (
    "generator effect yielded None; yield a disposer callable "
    "(cleanup code after yield runs immediately)"
)


async def test_generator_yielding_none_warns_once_per_plugin() -> None:
    from pycordis.testing import MemoryExporter, mock_context

    log: list[str] = []

    def gen_plugin(_ctx: Context, _config: Any) -> Iterator[None]:
        log.append("setup")
        yield
        log.append("cleanup")

    async def agen_plugin(_ctx: Context, _config: Any) -> AsyncIterator[None]:
        yield
        yield

    def good(_ctx: Context, _config: Any) -> Iterator[Callable[[], None]]:
        yield lambda: log.append("disposed")

    logs = MemoryExporter()
    async with mock_context(logs) as root:
        first = await root.plugin(gen_plugin)
        assert log == ["setup", "cleanup"]  # drained during setup
        await root.plugin(gen_plugin)
        await first.restart()
        await root.plugin(agen_plugin)
        await root.plugin(good)
        assert logs.text("warn") == [GEN_NONE, GEN_NONE]


async def test_effect_generator_yielding_none_warns() -> None:
    from pycordis.testing import MemoryExporter, mock_context

    def gen() -> Iterator[None]:
        yield
        yield

    logs = MemoryExporter()
    async with mock_context(logs) as root:
        root.effect(gen)
        root.effect(gen)
        assert logs.text("warn") == [GEN_NONE, GEN_NONE]
