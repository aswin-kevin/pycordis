"""Lifecycle telemetry events and fiber stats (spec §3.8)."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Callable
from typing import Any

import pytest
from conftest import capture_reports, with_meta

import pycordis.telemetry
from pycordis import TELEMETRY, Backoff, FiberStats, LifecycleEvent, LoadTimeoutError
from pycordis.context import Context
from pycordis.fiber import FiberState
from pycordis.supervise import RestartPolicy

FAST = Backoff(initial=0.01, factor=1.0, max=0.01, jitter=0.0)


def record(root: Context) -> list[LifecycleEvent]:
    """Collect every telemetry event emitted in ``root``'s application."""
    events: list[LifecycleEvent] = []
    root.on(TELEMETRY, events.append)
    return events


def kinds(events: list[LifecycleEvent], name: str | None = None) -> list[str]:
    return [event.kind for event in events if name is None or event.fiber == name]


def quick(_ctx: Context, _config: Any) -> None:
    return None


def broken(_ctx: Context, _config: Any) -> None:
    raise ValueError("bad url")


def failing_until(outcomes: list[bool]) -> Callable[[Context, Any], None]:
    """A plugin ``flaky`` popping one outcome per load and raising on ``False``."""

    def flaky(_ctx: Context, _config: Any) -> None:
        if not outcomes.pop(0):
            raise RuntimeError("boom")

    return flaky


async def until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.002)


# -- events ----------------------------------------------------------------------------


async def test_load_and_unload_events_carry_fiber_state_and_duration() -> None:
    async with Context() as root:
        events = record(root)
        before = time.time()
        fiber = await root.plugin(quick)
        uid = fiber.uid
        await fiber.dispose()
        assert kinds(events) == ["load", "unload"]
        load, unload = events
        assert (load.fiber, load.uid, load.state, load.error) == (
            "quick",
            uid,
            FiberState.ACTIVE,
            None,
        )
        assert load.duration is not None and 0 <= load.duration < 1
        assert before <= load.ts <= unload.ts <= time.time()
        # disposal clears the uid before the unload runs
        assert (unload.fiber, unload.uid, unload.state) == ("quick", None, FiberState.DISPOSED)
        assert unload.duration is not None and unload.duration >= 0


async def test_load_duration_measures_an_async_body() -> None:
    async def slow(_ctx: Context, _config: Any) -> None:
        await asyncio.sleep(0.03)

    async with Context() as root:
        events = record(root)
        fiber = await root.plugin(slow)
        assert events[0].duration is not None and events[0].duration >= 0.025
        assert fiber.stats.last_load_duration == events[0].duration


async def test_restart_emits_unload_then_load() -> None:
    async with Context() as root:
        events = record(root)
        fiber = await root.plugin(quick)
        await fiber.restart()
        assert kinds(events) == ["load", "unload", "load"]
        assert events[1].state == FiberState.LOADING  # the reload already started


async def test_failure_emits_fail_then_unload() -> None:
    async with Context() as root:
        errors = capture_reports(root)
        events = record(root)
        fiber = root.plugin(broken)
        with pytest.raises(ValueError, match="bad url") as info:
            await fiber
        assert kinds(events) == ["fail", "unload"]
        fail, unload = events
        assert fail.error is info.value
        assert fail.state == FiberState.FAILED  # the outcome, while the unload runs
        assert fail.duration is not None and fail.duration >= 0
        assert (unload.state, unload.error) == (FiberState.FAILED, None)
        assert errors == [info.value]


async def test_supervised_restart_emits_restart_with_its_backoff() -> None:
    policy = RestartPolicy(mode="on-failure", backoff=FAST)
    async with Context() as root:
        capture_reports(root)
        events = record(root)
        fiber = root.plugin(failing_until([False, True]), policy=policy)
        await until(lambda: fiber.state == FiberState.ACTIVE)
        assert kinds(events) == ["fail", "unload", "restart", "load"]
        restart = events[2]
        assert restart.duration == 0.01
        assert isinstance(restart.error, RuntimeError)
        assert restart.state == FiberState.FAILED
        stats = fiber.stats
        assert (stats.loads, stats.failures, stats.restarts) == (1, 1, 1)


async def test_giving_up_emits_give_up() -> None:
    policy = RestartPolicy(mode="on-failure", max_restarts=0, backoff=FAST)
    async with Context() as root:
        capture_reports(root)
        events = record(root)
        fiber = root.plugin(failing_until([False]), policy=policy)
        await until(lambda: "give-up" in kinds(events))
        assert kinds(events) == ["fail", "unload", "give-up"]
        assert events[-1].error is fiber.error
        assert fiber.stats.restarts == 0


async def test_load_timeout_emits_timeout_then_fail() -> None:
    async def hang(_ctx: Context, _config: Any) -> None:
        await asyncio.sleep(10)

    async with Context() as root:
        capture_reports(root)
        events = record(root)
        fiber = root.plugin(hang, load_timeout=0.02)
        with pytest.raises(LoadTimeoutError) as info:
            await fiber
        assert kinds(events) == ["timeout", "fail", "unload"]
        timeout, fail = events[:2]
        assert timeout.error is fail.error is info.value
        assert (timeout.state, fail.state) == (FiberState.UNLOADING, FiberState.FAILED)
        assert timeout.duration is not None and timeout.duration >= 0.015


async def test_dispose_timeout_emits_timeout_per_abandoned_disposer() -> None:
    def hung(ctx: Context, _config: Any) -> None:
        async def close() -> None:
            await asyncio.sleep(10)

        ctx.effect(lambda: close, "hung-close")

    async with Context() as root:
        capture_reports(root)
        events = record(root)
        fiber = await root.plugin(hung, dispose_timeout=0.02)
        await fiber.dispose()
        assert kinds(events) == ["load", "timeout", "unload"]
        timeout = events[1]
        assert timeout.duration == 0.02
        assert isinstance(timeout.error, TimeoutError)
        assert "hung-close" in str(timeout.error)
        assert timeout.state == FiberState.UNLOADING


async def test_cancelled_load_emits_fail() -> None:
    async def wait_forever(_ctx: Context, _config: Any) -> None:
        await asyncio.Event().wait()

    async with Context() as root:
        events = record(root)
        fiber = root.plugin(wait_forever)
        await asyncio.sleep(0.01)
        assert fiber.inertia is not None
        fiber.inertia.cancel()
        with pytest.raises(RuntimeError, match="plugin load was cancelled"):
            await fiber
        assert kinds(events) == ["fail"]
        assert events[0].state == FiberState.FAILED
        assert fiber.stats.failures == 1


async def test_listener_errors_are_reported_not_raised_into_the_lifecycle() -> None:
    def explode(_event: LifecycleEvent) -> None:
        raise RuntimeError("listener broke")

    async with Context() as root:
        errors = capture_reports(root)
        root.on(TELEMETRY, explode)
        events = record(root)
        fiber = await root.plugin(quick)
        assert fiber.state == FiberState.ACTIVE
        assert [str(error) for error in errors] == ["listener broke"]
        assert kinds(events) == ["load"]  # later listeners still run


async def test_async_listener_runs_as_a_task() -> None:
    seen: list[str] = []

    async def listener(event: LifecycleEvent) -> None:
        await asyncio.sleep(0)
        seen.append(event.kind)

    async with Context() as root:
        root.on(TELEMETRY, listener)
        await root.plugin(quick)
        await until(lambda: seen == ["load"])


async def test_no_event_is_built_without_listeners(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []
    real = pycordis.telemetry.LifecycleEvent

    def counting(*args: Any) -> LifecycleEvent:
        event = real(*args)
        built.append(event)
        return event

    monkeypatch.setattr(pycordis.telemetry, "LifecycleEvent", counting)
    async with Context() as root:
        capture_reports(root)
        fiber = await root.plugin(quick)
        await fiber.restart()
        with pytest.raises(ValueError):
            await root.plugin(broken)
        assert built == []
        dispose = root.on(TELEMETRY, lambda _event: None)
        await fiber.restart()
        assert len(built) == 2  # unload + load
        dispose()
        await fiber.restart()
        assert len(built) == 2


async def test_listener_owned_by_a_plugin_stops_with_it() -> None:
    events: list[LifecycleEvent] = []

    def observer(ctx: Context, _config: Any) -> None:
        ctx.on(TELEMETRY, events.append)

    async with Context() as root:
        watcher = await root.plugin(observer)
        await root.plugin(quick)
        await watcher.dispose()
        count = len(events)
        await root.plugin(with_meta(quick, name="other"))
        assert len(events) == count
        assert "quick" in [event.fiber for event in events]


# -- stats -----------------------------------------------------------------------------


async def test_stats_count_loads_failures_and_keep_the_last_error() -> None:
    async with Context() as root:
        capture_reports(root)
        assert root.fiber.stats == FiberStats(0, 0, 0, None, None)
        fiber = root.plugin(failing_until([False, True, True]))
        with pytest.raises(RuntimeError) as info:
            await fiber
        stats = fiber.stats
        assert (stats.loads, stats.failures, stats.restarts) == (0, 1, 0)
        assert stats.last_error is info.value
        assert stats.last_load_duration is None
        await fiber.update(None)
        await fiber.restart()
        stats = fiber.stats
        assert (stats.loads, stats.failures) == (2, 1)
        assert stats.last_error is info.value  # kept after recovering
        assert stats.last_load_duration is not None and stats.last_load_duration >= 0


async def test_stats_is_an_immutable_snapshot() -> None:
    async with Context() as root:
        fiber = await root.plugin(quick)
        snapshot = fiber.stats
        with pytest.raises(dataclasses.FrozenInstanceError):
            snapshot.loads = 5  # pyright: ignore[reportAttributeAccessIssue]  # frozen: negative test
        await fiber.restart()
        assert snapshot.loads == 1
        assert fiber.stats.loads == 2


def test_event_and_stats_are_frozen_values() -> None:
    event = LifecycleEvent("load", "p", 1, FiberState.ACTIVE, 0.5, None, 1.0)
    assert dataclasses.replace(event, kind="unload").kind == "unload"
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.kind = "fail"  # pyright: ignore[reportAttributeAccessIssue]  # frozen: negative test
    assert TELEMETRY.name == "internal/telemetry"
