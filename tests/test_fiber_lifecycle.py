import asyncio
import gc
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import capture_reports

from pycordis.context import Context
from pycordis.errors import CordisError, ServiceAccessError
from pycordis.fiber import INACTIVE, Fiber, FiberState


def _record_states(root: Context, target: list[tuple[Fiber, FiberState]]) -> None:
    def on_status(fiber: Fiber, _old: FiberState) -> None:
        target.append((fiber, fiber.state))

    root.on("internal/status", on_status)


def _states(log: list[tuple[Fiber, FiberState]], fiber: Fiber) -> list[FiberState]:
    return [state for owner, state in log if owner is fiber]


def _injecting(
    name: str, inject: list[str], apply: Callable[[Context, Any], Any]
) -> SimpleNamespace:
    return SimpleNamespace(name=name, inject=inject, apply=apply)


async def _until(predicate: Callable[[], bool], limit: int = 100) -> None:
    for _ in range(limit):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not reached")


# -- state transitions --------------------------------------------------------


async def test_plugin_without_inject_states() -> None:
    root = Context()
    log: list[tuple[Fiber, FiberState]] = []
    _record_states(root, log)

    def plugin(_ctx: Context, _config: Any) -> None:
        return None

    fiber = root.plugin(plugin)
    # the LOADING transition is synchronous (cordis `_reload` sync prefix)
    assert _states(log, fiber) == [FiberState.LOADING]
    assert fiber.inertia is not None
    assert fiber.inertia.get_name() == "pycordis:plugin:load"
    await fiber
    assert _states(log, fiber) == [FiberState.LOADING, FiberState.ACTIVE]
    assert fiber.inertia is None
    log.clear()
    await fiber.dispose()
    assert _states(log, fiber) == [FiberState.UNLOADING, FiberState.DISPOSED]
    assert fiber.uid is None
    assert fiber.state == FiberState.DISPOSED


async def test_epoch_format() -> None:
    root = Context()
    seen: list[str] = []

    def dependent(_ctx: Context, _config: Any) -> None:
        return None

    root.provide("db", 1)
    root.provide("cache", 2)
    fiber = root.plugin(_injecting("dep", ["db", "cache"], dependent))
    await fiber
    db = root.reflect.get_impl("db")
    cache = root.reflect.get_impl("cache")
    assert db is not None and cache is not None
    seen.append(fiber.epoch)
    assert seen == [f"0:{db.uid}:{cache.uid}"]
    assert fiber.committed == {"db": db, "cache": cache}
    assert db.dependents == {fiber} and cache.dependents == {fiber}
    await fiber.dispose()
    assert fiber.epoch == INACTIVE
    assert fiber.committed is None
    assert not db.dependents


async def test_plugin_raises() -> None:
    root = Context()
    reports = capture_reports(root)
    log: list[tuple[Fiber, FiberState]] = []
    _record_states(root, log)
    error = RuntimeError("boom")

    def plugin(_ctx: Context, _config: Any) -> None:
        raise error

    fiber = root.plugin(plugin)
    with pytest.raises(RuntimeError) as info:
        await fiber
    assert info.value is error
    assert fiber.state == FiberState.FAILED
    # cordis: a failed load still unloads what it collected
    assert _states(log, fiber) == [FiberState.LOADING, FiberState.UNLOADING, FiberState.FAILED]
    assert reports == [error]
    # waiting again re-raises without reporting twice
    with pytest.raises(RuntimeError):
        await fiber.wait()
    assert reports == [error]
    await fiber.dispose()
    assert fiber.state == FiberState.DISPOSED


async def test_async_plugin_raises() -> None:
    root = Context()
    reports = capture_reports(root)
    disposed: list[str] = []

    async def plugin(ctx: Context, _config: Any) -> None:
        ctx.effect(lambda: lambda: disposed.append("partial"))
        await asyncio.sleep(0)
        raise ValueError("late")

    fiber = root.plugin(plugin)
    with pytest.raises(ValueError, match="late"):
        await fiber
    assert fiber.state == FiberState.FAILED
    assert len(reports) == 1
    await fiber.dispose()
    assert disposed == ["partial"]


# -- dependencies --------------------------------------------------------------


async def test_inject_pending_until_provided() -> None:
    root = Context()
    log: list[Any] = []

    def dependent(ctx: Context, _config: Any) -> Callable[[], None]:
        log.append(("on", ctx.db))
        return lambda: log.append("off")

    def provider(ctx: Context, _config: Any) -> None:
        ctx.provide("db", "x")

    dep = root.plugin(_injecting("D", ["db"], dependent))
    await dep
    assert dep.state == FiberState.PENDING
    assert dep.inertia is None
    assert log == []

    prov = root.plugin(provider)
    await prov
    await dep
    assert dep.state == FiberState.ACTIVE
    assert log == [("on", "x")]

    await prov.dispose()
    await dep
    assert dep.state == FiberState.PENDING
    assert log == [("on", "x"), "off"]
    # it reactivates once the service returns
    await root.plugin(provider)
    await dep
    assert dep.state == FiberState.ACTIVE
    assert log == [("on", "x"), "off", ("on", "x")]


async def test_teardown_order_dependent_first() -> None:
    root = Context()
    log: list[str] = []

    def provider(ctx: Context, _config: Any) -> Callable[[], None]:
        ctx.provide("db", "x")
        return lambda: log.append("P-off")

    def dependent(_ctx: Context, _config: Any) -> Callable[[], None]:
        return lambda: log.append("D-off")

    prov = root.plugin(provider)
    dep = root.plugin(_injecting("D", ["db"], dependent))
    await prov
    await dep
    assert dep.state == FiberState.ACTIVE
    await prov.dispose()
    assert log == ["D-off", "P-off"]
    assert dep.state == FiberState.PENDING
    assert prov.state == FiberState.DISPOSED


async def test_provider_waits_for_every_dependent() -> None:
    root = Context()
    log: list[str] = []
    gate = asyncio.Event()

    def provider(ctx: Context, _config: Any) -> Callable[[], None]:
        ctx.provide("db", "x")
        return lambda: log.append("P-off")

    def fast(_ctx: Context, _config: Any) -> Callable[[], None]:
        return lambda: log.append("fast-off")

    def slow(_ctx: Context, _config: Any) -> Callable[[], Any]:
        async def off() -> None:
            await gate.wait()
            log.append("slow-off")

        return off

    prov = root.plugin(provider)
    deps = [
        root.plugin(_injecting("fast", ["db"], fast)),
        root.plugin(_injecting("slow", ["db"], slow)),
    ]
    await prov
    for dep in deps:
        await dep
    disposal = prov.dispose()
    await _until(lambda: log == ["fast-off"])
    for _ in range(5):
        await asyncio.sleep(0)
    assert log == ["fast-off"]
    gate.set()
    await disposal
    assert log == ["fast-off", "slow-off", "P-off"]


async def test_failed_fiber_ignores_service_changes() -> None:
    root = Context()
    capture_reports(root)
    runs: list[int] = []

    def dependent(_ctx: Context, _config: Any) -> None:
        runs.append(1)
        raise RuntimeError("broken")

    undo = root.provide("db", 1)
    dep = root.plugin(_injecting("D", ["db"], dependent))
    with pytest.raises(RuntimeError):
        await dep
    await undo()
    root.provide("db", 2)
    with pytest.raises(RuntimeError):
        await dep
    assert runs == [1]
    assert dep.state == FiberState.FAILED


async def test_dependent_reads_service_during_teardown() -> None:
    root = Context()
    seen: list[Any] = []

    def provider(ctx: Context, _config: Any) -> None:
        ctx.provide("db", "x")

    def dependent(ctx: Context, _config: Any) -> Callable[[], None]:
        return lambda: seen.append(ctx.db)

    prov = root.plugin(provider)
    dep = root.plugin(_injecting("D", ["db"], dependent))
    await prov
    await dep
    await prov.dispose()
    assert seen == ["x"]
    with pytest.raises(ServiceAccessError, match="inactive context"):
        _ = dep.ctx.db


async def test_service_swap_reloads_dependent() -> None:
    root = Context()
    log: list[Any] = []

    def dependent(ctx: Context, _config: Any) -> Callable[[], None]:
        log.append(("on", ctx.db))
        return lambda: log.append("off")

    undo = root.provide("db", 1)
    dep = root.plugin(_injecting("D", ["db"], dependent))
    await dep
    # swap without a gap: the old binding goes away synchronously and the new one
    # arrives while the dependent is still unloading
    pending = undo()
    root.provide("db", 2)
    await pending
    await dep
    assert log == [("on", 1), "off", ("on", 2)]
    assert dep.state == FiberState.ACTIVE


async def test_service_swap_with_gap() -> None:
    root = Context()
    log: list[Any] = []
    states: list[tuple[Fiber, FiberState]] = []
    _record_states(root, states)

    def dependent(ctx: Context, _config: Any) -> Callable[[], None]:
        log.append(("on", ctx.db))
        return lambda: log.append("off")

    undo = root.provide("db", 1)
    dep = root.plugin(_injecting("D", ["db"], dependent))
    await dep
    await undo()
    root.provide("db", 2)
    await dep
    assert log == [("on", 1), "off", ("on", 2)]
    assert _states(states, dep) == [
        FiberState.LOADING,
        FiberState.ACTIVE,
        FiberState.UNLOADING,
        FiberState.PENDING,
        FiberState.LOADING,
        FiberState.ACTIVE,
    ]


async def test_epoch_change_while_loading_reloads() -> None:
    root = Context()
    log: list[Any] = []
    gate = asyncio.Event()

    async def dependent(ctx: Context, _config: Any) -> Callable[[], None]:
        value = ctx.db
        log.append(("on", value))
        if value == 1:
            await gate.wait()
        return lambda: log.append(("off", value))

    undo = root.provide("db", 1)
    dep = root.plugin(_injecting("D", ["db"], dependent))
    await _until(lambda: bool(log))
    undo()
    root.provide("db", 2)
    await dep
    # the superseded load was cancelled, then the new target loaded
    assert log == [("on", 1), ("on", 2)]
    assert dep.state == FiberState.ACTIVE
    await dep.dispose()
    assert log[-1] == ("off", 2)


async def test_service_check_and_failing_check() -> None:
    root = Context()
    reports = capture_reports(root)
    allowed = False
    calls: list[int] = []

    def check() -> bool:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("check failed")
        return allowed

    def dependent(_ctx: Context, _config: Any) -> None:
        return None

    dep = root.plugin(_injecting("D", ["db"], dependent))
    root.provide("db", "x", check)
    await dep
    assert dep.state == FiberState.PENDING
    assert [str(e) for e in reports] == ["check failed"]
    root.reflect.notify(["db"])
    await dep
    assert dep.state == FiberState.PENDING
    allowed = True
    root.reflect.notify(["db"])
    await dep
    assert dep.state == FiberState.ACTIVE


async def test_real_fiber_walk_resolution() -> None:
    root = Context()
    root.provide("foo", 1)
    contexts: dict[str, Context] = {}

    def capture(key: str) -> Callable[[Context, Any], None]:
        def apply(ctx: Context, _config: Any) -> None:
            contexts[key] = ctx

        return apply

    # found in an ancestor fiber's provisions (the root fiber)
    await root.plugin(_injecting("plain", [], capture("plain")))
    assert contexts["plain"].foo == 1
    with pytest.raises(ServiceAccessError, match=r'^cannot get property "bar" without inject$'):
        _ = contexts["plain"].bar
    # required and committed
    dep = root.plugin(_injecting("dep", ["foo"], capture("dep")))
    await dep
    assert contexts["dep"].foo == 1
    # required but not available → inactive
    missing = root.plugin(_injecting("missing", ["bar"], capture("missing")))
    await missing
    with pytest.raises(
        ServiceAccessError, match=r'^cannot get required service "bar" in inactive context$'
    ):
        _ = missing.ctx.bar

    # own provision in an isolated scope
    def provider(ctx: Context, _config: Any) -> None:
        ctx.provide("baz", "own")
        contexts["provider"] = ctx

    await root.isolate("baz").plugin(provider)
    assert contexts["provider"].baz == "own"
    # invisible outside the isolated scope
    assert root.baz is None


async def test_body_cancelling_itself_fails() -> None:
    root = Context()
    reports = capture_reports(root)

    async def plugin(_ctx: Context, _config: Any) -> None:
        raise asyncio.CancelledError

    fiber = root.plugin(plugin)
    with pytest.raises(RuntimeError, match="plugin setup raised CancelledError"):
        await fiber
    assert fiber.state == FiberState.FAILED
    assert len(reports) == 1


async def test_restart_twice_while_body_awaits() -> None:
    root = Context()
    log: list[str] = []
    gate = asyncio.Event()

    async def plugin(ctx: Context, _config: Any) -> Callable[[], None]:
        run = len([e for e in log if e.startswith("run")]) + 1
        log.append(f"run{run}")
        ctx.effect(lambda: lambda: log.append(f"partial{run}"))
        if run == 1:
            await gate.wait()
        return lambda: log.append(f"off{run}")

    fiber = root.plugin(plugin)
    await _until(lambda: log == ["run1"])
    await asyncio.wait_for(asyncio.gather(fiber.restart(), fiber.restart()), 1)
    # the superseded body was cancelled and its partial effects disposed
    # before the latest target loaded, exactly once
    assert log == ["run1", "partial1", "run2"]
    assert fiber.state == FiberState.ACTIVE
    await fiber.dispose()
    assert log[3:] == ["off2", "partial2"]


async def test_superseded_load_back_to_same_epoch_reloads() -> None:
    root = Context()
    runs: list[str] = []
    allowed = True

    def check() -> bool:
        return allowed

    async def plugin(_ctx: Context, _config: Any) -> None:
        runs.append("run")
        if len(runs) == 1:
            await asyncio.Event().wait()

    root.provide("db", 1, check)
    fiber = root.plugin(_injecting("D", ["db"], plugin))
    await _until(lambda: bool(runs))
    # A -> INACTIVE -> A again (same provider uid) while the body is in flight
    allowed = False
    root.reflect.notify(["db"])
    allowed = True
    root.reflect.notify(["db"])
    await asyncio.wait_for(fiber.wait(), 1)
    # the cancelled load is not reported ACTIVE: it unloads and loads again
    assert runs == ["run", "run"]
    assert fiber.state == FiberState.ACTIVE


async def test_cancelled_load_recovers_on_dispose() -> None:
    root = Context()
    log: list[str] = []
    states: list[tuple[Fiber, FiberState]] = []
    _record_states(root, states)

    async def plugin(ctx: Context, _config: Any) -> None:
        ctx.effect(lambda: lambda: log.append("partial"))
        await asyncio.Event().wait()

    fiber = root.plugin(plugin)
    await _until(lambda: bool(fiber.get_effects()))
    inertia = fiber.inertia
    assert inertia is not None
    inertia.cancel()
    with pytest.raises(RuntimeError, match=r"^plugin load was cancelled$"):
        await fiber
    assert fiber.inertia is None
    assert fiber.state == FiberState.FAILED
    await asyncio.wait_for(fiber.dispose(), 1)
    assert log == ["partial"]
    assert fiber.state == FiberState.DISPOSED
    assert _states(states, fiber) == [
        FiberState.LOADING,
        FiberState.FAILED,
        FiberState.UNLOADING,
        FiberState.DISPOSED,
    ]


async def test_load_cancelled_before_it_starts() -> None:
    root = Context()

    def plugin(_ctx: Context, _config: Any) -> None:
        return None

    fiber = root.plugin(plugin)
    assert fiber.inertia is not None
    fiber.inertia.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        await fiber
    await fiber.dispose()
    assert fiber.state == FiberState.DISPOSED


async def test_cancelled_unload_hands_back_disposers() -> None:
    root = Context()
    log: list[str] = []
    gate = asyncio.Event()

    async def slow() -> None:
        await gate.wait()
        log.append("slow")

    root.effect(lambda: lambda: log.append("first"))
    root.effect(lambda: slow)
    disposal = root.fiber.dispose()
    await _until(lambda: root.fiber.state == FiberState.UNLOADING)
    inertia = root.fiber.inertia
    assert inertia is not None
    # disposers start after one yield; cancel before they run
    inertia.cancel()
    # the waiter is not wedged: it settles once the fiber is reconciled
    await asyncio.wait_for(disposal, 1)
    assert root.fiber.inertia is None
    assert root.fiber.state == FiberState.ACTIVE
    assert len(root.fiber.get_effects()) == 2
    gate.set()
    await asyncio.wait_for(root.fiber.dispose(), 1)
    assert sorted(log) == ["first", "slow"]
    assert root.fiber.get_effects() == []


# -- hierarchy and disposal ------------------------------------------------------


async def test_parent_dispose_disposes_children() -> None:
    root = Context()
    log: list[str] = []
    children: list[Fiber] = []

    def child(_ctx: Context, _config: Any) -> Callable[[], None]:
        return lambda: log.append("child-off")

    def parent(ctx: Context, _config: Any) -> Callable[[], None]:
        children.append(ctx.plugin(child))
        return lambda: log.append("parent-off")

    fiber = root.plugin(parent)
    await fiber
    [kid] = children
    await kid
    assert kid.state == FiberState.ACTIVE
    await fiber.dispose()
    assert kid.state == FiberState.DISPOSED
    assert fiber.state == FiberState.DISPOSED
    assert sorted(log) == ["child-off", "parent-off"]
    assert not root.registry.has(child)


async def test_effect_on_disposed_context() -> None:
    root = Context()

    def plugin(_ctx: Context, _config: Any) -> None:
        return None

    fiber = root.plugin(plugin)
    await fiber
    await fiber.dispose()
    with pytest.raises(CordisError) as info:
        fiber.ctx.effect(lambda: None)
    assert info.value.code == "INACTIVE_EFFECT"
    with pytest.raises(CordisError):
        fiber.ctx.plugin(plugin)


async def test_dispose_twice_and_pending_dispose() -> None:
    root = Context()

    def dependent(_ctx: Context, _config: Any) -> None:
        return None

    fiber = root.plugin(_injecting("D", ["db"], dependent))
    await fiber
    assert fiber.state == FiberState.PENDING
    await fiber.dispose()
    assert fiber.state == FiberState.DISPOSED
    await fiber.dispose()
    assert fiber.state == FiberState.DISPOSED


async def test_root_dispose_while_plugin_awaiting(caplog: pytest.LogCaptureFixture) -> None:
    """Review Focus 2: shutdown during startup completes cleanly."""
    root = Context()
    log: list[str] = []
    event = asyncio.Event()
    started = asyncio.Event()

    async def plugin(ctx: Context, _config: Any) -> Callable[[], None]:
        ctx.effect(lambda: lambda: log.append("partial"))
        started.set()
        await event.wait()
        return lambda: log.append("full")

    fiber = root.plugin(plugin)
    await started.wait()
    assert fiber.state == FiberState.LOADING
    await asyncio.wait_for(root.fiber.dispose(), 1)
    assert log == ["partial"]
    assert fiber.state == FiberState.DISPOSED
    assert root.fiber.state == FiberState.ACTIVE
    assert fiber.inertia is None and not fiber.ctx.fiber.owned_tasks()
    del plugin, fiber
    gc.collect()
    assert not [r for r in caplog.records if r.name == "asyncio"]


async def test_root_dispose_before_plugin_body_starts() -> None:
    root = Context()

    async def plugin(_ctx: Context, _config: Any) -> None:
        await asyncio.sleep(0)

    fiber = root.plugin(plugin)
    await root.fiber.dispose()
    assert fiber.state == FiberState.DISPOSED


async def test_unload_cancels_spawned_tasks_before_disposers() -> None:
    root = Context()
    log: list[str] = []

    async def forever() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            log.append("task-cancelled")

    def plugin(ctx: Context, _config: Any) -> Callable[[], None]:
        ctx.fiber.spawn_owned(forever())
        return lambda: log.append("disposer")

    fiber = root.plugin(plugin)
    await fiber
    await asyncio.sleep(0)
    await fiber.dispose()
    assert log == ["task-cancelled", "disposer"]


async def test_cancel_dispose_awaiter_does_not_wedge_plugin() -> None:
    root = Context()
    release = asyncio.Event()
    log: list[str] = []

    async def slow_disposer() -> None:
        await release.wait()
        log.append("disposed")

    def plugin(_ctx: Context, _config: Any) -> Callable[[], Any]:
        return slow_disposer

    fiber = root.plugin(plugin)
    await fiber

    async def dispose() -> None:
        await fiber.dispose()

    awaiter = asyncio.create_task(dispose())
    await _until(lambda: fiber.state == FiberState.UNLOADING)
    for _ in range(5):
        await asyncio.sleep(0)
    awaiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await awaiter
    assert fiber.inertia is not None and not fiber.inertia.cancelled()
    release.set()
    assert await fiber.wait() is fiber
    assert fiber.state == FiberState.DISPOSED
    assert log == ["disposed"]


async def test_cancel_wait_does_not_cancel_load() -> None:
    root = Context()
    gate = asyncio.Event()

    async def plugin(_ctx: Context, _config: Any) -> None:
        await gate.wait()

    fiber = root.plugin(plugin)
    waiter = asyncio.create_task(fiber.wait())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    gate.set()
    await fiber
    assert fiber.state == FiberState.ACTIVE


async def test_cancel_root_dispose_awaiter_does_not_wedge_root() -> None:
    root = Context()
    release = asyncio.Event()

    async def slow_disposer() -> None:
        await release.wait()

    root.effect(lambda: slow_disposer)

    async def dispose() -> None:
        await root.fiber.dispose()

    awaiter = asyncio.create_task(dispose())
    await _until(lambda: root.fiber.state == FiberState.UNLOADING)
    awaiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await awaiter
    release.set()
    await root.fiber.wait()
    assert root.fiber.state == FiberState.ACTIVE
    # the root is usable again
    root.effect(lambda: None)


# -- carried fixes ---------------------------------------------------------------


async def test_status_observer_may_load_plugins_during_notify() -> None:
    root = Context()
    loaded: list[Fiber] = []
    triggered: list[bool] = []

    def extra(_ctx: Context, _config: Any) -> None:
        return None

    def on_status(fiber: Fiber, _old: FiberState) -> None:
        if fiber.state == FiberState.LOADING and not triggered:
            triggered.append(True)
            loaded.append(root.plugin(SimpleNamespace(name="extra", apply=extra)))

    def dependent(_ctx: Context, _config: Any) -> None:
        return None

    dep = root.plugin(_injecting("D", ["db"], dependent))
    await dep
    root.on("internal/status", on_status)
    root.provide("db", 1)
    await dep
    [extra_fiber] = loaded
    await extra_fiber
    assert dep.state == FiberState.ACTIVE
    assert extra_fiber.state == FiberState.ACTIVE


async def test_plugin_publish_rollback_on_base_exception() -> None:
    root = Context()

    class Interrupt(BaseException):
        pass

    raised: list[Fiber] = []

    def on_plugin(fiber: Fiber) -> None:
        # e.g. Ctrl-C landing inside an observer (only the first notification)
        if not raised:
            raised.append(fiber)
            raise Interrupt

    def plugin(_ctx: Context, _config: Any) -> None:
        return None

    root.on("internal/plugin", on_plugin)
    with pytest.raises(Interrupt):
        root.plugin(plugin)
    assert not root.registry.has(plugin)
    assert raised[0].uid is None
    # the rollback disposal settles in the background
    await _until(lambda: len(root.fiber.get_effects()) == 1)
    assert [meta.label for meta in root.fiber.get_effects()] == ['ctx.on("internal/plugin")']
    assert raised[0].state == FiberState.DISPOSED


async def test_plugin_on_unloading_parent_leaves_no_runtime() -> None:
    root = Context()
    errors: list[BaseException] = []

    def late(_ctx: Context, _config: Any) -> None:
        return None

    def parent(ctx: Context, _config: Any) -> Callable[[], None]:
        def dispose() -> None:
            try:
                ctx.plugin(late)
            except CordisError as error:
                errors.append(error)

        return dispose

    undo = root.provide("db", 1)
    fiber = root.plugin(_injecting("parent", ["db"], parent))
    await fiber
    # the fiber is UNLOADING (uid still set) when its disposer runs
    await undo()
    await fiber
    assert fiber.state == FiberState.PENDING
    assert [getattr(e, "code", None) for e in errors] == ["INACTIVE_EFFECT"]
    assert not root.registry.has(late)


async def test_async_plugin_listener_observes_disposal() -> None:
    root = Context()
    seen: list[tuple[str, int | None]] = []

    async def on_plugin(fiber: Fiber) -> None:
        seen.append((fiber.name, fiber.uid))

    def plugin(_ctx: Context, _config: Any) -> None:
        return None

    root.on("internal/plugin", on_plugin)
    fiber = root.plugin(plugin)
    uid = fiber.uid
    await fiber
    await fiber.dispose()
    await _until(lambda: len(seen) == 2)
    assert seen == [("plugin", uid), ("plugin", None)]


async def test_set_from_another_fiber() -> None:
    root = Context()
    contexts: list[Context] = []

    def provider(ctx: Context, _config: Any) -> None:
        ctx.provide("x", 1)

    def other(ctx: Context, _config: Any) -> None:
        contexts.append(ctx)

    await root.plugin(provider)
    await root.plugin(other)
    [ctx] = contexts
    with pytest.raises(ServiceAccessError, match=r'^cannot set property "x" in multiple fibers'):
        ctx.set("x", 2)
    assert root.x == 1


async def test_context_manager_root_dispose() -> None:
    log: list[str] = []
    async with Context() as root:

        def plugin(_ctx: Context, _config: Any) -> Callable[[], None]:
            return lambda: log.append("off")

        await root.plugin(plugin)
    assert log == ["off"]


# -- deep dependency chains ---------------------------------------------------


def _chain_link(index: int, disposed: list[int]) -> SimpleNamespace:
    """Plugin ``p{index}``: injects ``s{index - 1}`` (none for the head), provides ``s{index}``."""

    def apply(ctx: Context, _config: Any) -> None:
        ctx.provide(f"s{index}", index)
        ctx.effect(lambda: lambda: disposed.append(index))

    inject = [f"s{index - 1}"] if index > 0 else []
    return _injecting(f"p{index}", inject, apply)


async def test_a_1000_deep_active_chain_tears_down_dependents_first() -> None:
    # every unload notifies the next dependent synchronously: a recursive
    # cascade overflowed the stack well before 1000 links
    n = 1000
    disposed: list[int] = []
    root = Context()
    errors = capture_reports(root)
    fibers = [root.plugin(_chain_link(i, disposed)) for i in range(n)]
    await _until(lambda: all(f.state == FiberState.ACTIVE for f in fibers), limit=4 * n)
    started = time.perf_counter()
    await fibers[0].dispose()
    assert disposed == list(reversed(range(n)))
    assert all(f.state == FiberState.PENDING for f in fibers[1:])
    await root.fiber.dispose()
    assert time.perf_counter() - started < 5.0
    assert errors == []
    assert all(f.state == FiberState.DISPOSED for f in fibers)


async def test_a_teardown_cascade_announces_in_depth_first_order() -> None:
    # p1 and p2 inject s0, p3 injects both: each dependent's cascade (status,
    # then its own services) completes before the next dependent is told
    root = Context()
    log: list[str] = []

    def link(index: int, inject: list[str]) -> SimpleNamespace:
        return _injecting(f"p{index}", inject, lambda ctx, _: ctx.provide(f"s{index}", index))

    links = [link(0, []), link(1, ["s0"]), link(2, ["s0"]), link(3, ["s1", "s2"])]
    fibers = [root.plugin(p) for p in links]
    await _until(lambda: all(f.state == FiberState.ACTIVE for f in fibers))

    def on_status(fiber: Fiber, old: FiberState) -> None:
        log.append(f"{fiber.name} {old.name}->{fiber.state.name}")

    def on_service(_ctx: Context, name: str, _value: object) -> None:
        log.append(name)

    root.on("internal/status", on_status)
    root.on("internal/service", on_service)
    await fibers[0].dispose()
    assert log == [
        "p0 ACTIVE->UNLOADING",
        "p1 ACTIVE->UNLOADING",
        "p3 ACTIVE->UNLOADING",
        "s3",
        "s1",
        "p2 ACTIVE->UNLOADING",
        "s2",
        "s0",
        "s3",
        "p3 UNLOADING->PENDING",
        "s1",
        "s2",
        "p1 UNLOADING->PENDING",
        "p2 UNLOADING->PENDING",
        "s0",
        "p0 UNLOADING->DISPOSED",
    ]


def _chain_with_waiter() -> list[SimpleNamespace]:
    """p0 -> p1 -> p2 dependency chain, plus ``w`` waiting for a service ``extra``."""

    def providing(name: str) -> Callable[[Context, Any], Any]:
        return lambda ctx, _: ctx.provide(name, name)

    return [
        _injecting("p0", [], providing("s0")),
        _injecting("p1", ["s0"], providing("s1")),
        _injecting("p2", ["s1"], lambda _ctx, _config: None),
        _injecting("w", ["extra"], lambda _ctx, _config: None),
    ]


async def test_a_notify_started_during_a_cascade_runs_after_the_current_step() -> None:
    root = Context()
    fibers = [root.plugin(p) for p in _chain_with_waiter()]
    await _until(lambda: [f.state for f in fibers][:3] == [FiberState.ACTIVE] * 3)
    log: list[str] = []

    def on_status(fiber: Fiber, old: FiberState) -> None:
        log.append(f"{fiber.name} {old.name}->{fiber.state.name}")
        if fiber.name == "p1" and fiber.state == FiberState.UNLOADING:
            root.provide("extra", 1)
            # deferred: the provide's notification runs once this step ends
            log.append(f"w is {fibers[3].state.name} after provide")

    root.on("internal/status", on_status)
    await fibers[0].dispose()
    assert log[:5] == [
        "p0 ACTIVE->UNLOADING",
        "p1 ACTIVE->UNLOADING",
        "w is PENDING after provide",
        "w PENDING->LOADING",  # before the cascade moves on to p2
        "p2 ACTIVE->UNLOADING",
    ]


async def test_an_error_in_a_deferred_notify_surfaces_from_the_outermost_cascade() -> None:
    root = Context()
    fibers = [root.plugin(p) for p in _chain_with_waiter()]
    await _until(lambda: [f.state for f in fibers][:3] == [FiberState.ACTIVE] * 3)
    provide_returned: list[bool] = []

    def on_status(fiber: Fiber, _old: FiberState) -> None:
        if fiber.name == "p1" and fiber.state == FiberState.UNLOADING:
            root.provide("extra", 1)
            provide_returned.append(True)
        if fiber.name == "w" and fiber.state == FiberState.LOADING:
            raise RuntimeError("listener boom")

    root.on("internal/status", on_status)
    with pytest.raises(RuntimeError, match="listener boom"):
        await fibers[0].dispose()
    assert provide_returned == [True]  # the provide itself did not raise
