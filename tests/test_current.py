"""Ambient current context: ``current_context()`` / ``use_context()`` (spec §3.3)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest
from conftest import capture_reports

import pycordis
from pycordis import Context, Service, current_context, use_context


def _try_current() -> Context | None:
    try:
        return current_context()
    except LookupError:
        return None


def test_outside_any_context_raises_lookup_error() -> None:
    with pytest.raises(LookupError, match="no active pycordis context"):
        current_context()


def test_exported_from_package() -> None:
    assert pycordis.current_context is current_context
    assert pycordis.use_context is use_context


async def test_use_context_sets_and_nesting_restores() -> None:
    outer = Context()
    inner = Context()
    with use_context(outer) as entered:
        assert entered is outer
        assert current_context() is outer
        with use_context(inner):
            assert current_context() is inner
        assert current_context() is outer
    assert _try_current() is None


async def test_use_context_restores_after_exception() -> None:
    ctx = Context()
    with pytest.raises(ValueError), use_context(ctx):
        raise ValueError("boom")
    assert _try_current() is None


async def test_plugin_body_sees_its_own_context() -> None:
    root = Context()
    seen: list[tuple[Context, Context]] = []

    def plugin(ctx: Context, _config: Any) -> None:
        seen.append((ctx, current_context()))

    await root.plugin(plugin)
    [(ctx, current)] = seen
    assert current is ctx
    assert current is not root


async def test_async_plugin_body_sees_its_own_context_across_awaits() -> None:
    root = Context()
    seen: list[tuple[Context, Context]] = []

    async def plugin(ctx: Context, _config: Any) -> None:
        await asyncio.sleep(0)
        seen.append((ctx, current_context()))

    await root.plugin(plugin)
    [(ctx, current)] = seen
    assert current is ctx


async def test_class_plugin_constructor_sees_its_own_context() -> None:
    root = Context()
    seen: list[tuple[Context, Context]] = []

    class Plugin:
        def __init__(self, ctx: Context, _config: Any) -> None:
            seen.append((ctx, current_context()))

    await root.plugin(Plugin)
    [(ctx, current)] = seen
    assert current is ctx


async def test_nested_plugins_see_their_own_context() -> None:
    root = Context()
    seen: dict[str, tuple[Context, Context]] = {}

    def child(ctx: Context, _config: Any) -> None:
        seen["child"] = (ctx, current_context())

    async def parent(ctx: Context, _config: Any) -> None:
        await ctx.plugin(child)
        seen["parent"] = (ctx, current_context())

    await root.plugin(parent)
    for name in ("parent", "child"):
        ctx, current = seen[name]
        assert current is ctx, name
    assert seen["parent"][0] is not seen["child"][0]


async def test_generator_plugin_disposer_sees_its_own_context() -> None:
    root = Context()
    owners: list[Context] = []
    seen: list[Context] = []

    def plugin(ctx: Context, _config: Any) -> Iterator[Callable[[], None]]:
        owners.append(ctx)
        yield lambda: seen.append(current_context())

    fiber = root.plugin(plugin)
    await fiber
    await fiber.dispose()
    assert seen == owners


async def test_async_disposer_sees_its_own_context() -> None:
    root = Context()
    owners: list[Context] = []
    seen: list[Context] = []

    async def cleanup() -> None:
        await asyncio.sleep(0)
        seen.append(current_context())

    def plugin(ctx: Context, _config: Any) -> Callable[[], Any]:
        owners.append(ctx)
        return cleanup

    fiber = root.plugin(plugin)
    await fiber
    await fiber.dispose()
    assert seen == owners


async def test_effect_setup_and_disposer_see_owning_context() -> None:
    root = Context()
    owners: list[Context] = []
    seen: list[tuple[str, Context]] = []

    async def plugin(ctx: Context, _config: Any) -> None:
        owners.append(ctx)

        def execute() -> Callable[[], None]:
            seen.append(("setup", current_context()))
            return lambda: seen.append(("dispose", current_context()))

        dispose = ctx.effect(execute)
        # disposed from outside the plugin's own task
        root.spawn(_call_later(dispose))

    await root.plugin(plugin)
    await asyncio.sleep(0.01)
    [ctx] = owners
    assert seen == [("setup", ctx), ("dispose", ctx)]


async def test_effect_disposers_see_the_derived_view_they_were_registered_on() -> None:
    root = Context()
    seen: list[tuple[str, Context]] = []
    views: list[Context] = []

    async def plugin(ctx: Context, _config: Any) -> None:
        for view in (ctx.extend(tag="request"), ctx.isolate("db")):
            views.append(view)

            def execute() -> Callable[[], None]:
                seen.append(("setup", current_context()))
                return lambda: seen.append(("dispose", current_context()))

            # the effect is owned by ctx.fiber but runs with the view current
            await view.effect(execute)()

    await root.plugin(plugin)
    extended, isolated = views
    assert seen == [
        ("setup", extended),
        ("dispose", extended),
        ("setup", isolated),
        ("dispose", isolated),
    ]


async def _call_later(dispose: Callable[[], Any]) -> None:
    await asyncio.sleep(0)
    await dispose()


async def test_effect_called_outside_any_context_sets_owner() -> None:
    root = Context()
    seen: list[Context] = []

    async def setup() -> AsyncIterator[Callable[[], Any]]:
        seen.append(current_context())

        async def cleanup() -> None:
            await asyncio.sleep(0)
            seen.append(current_context())

        yield cleanup

    dispose = root.effect(setup)
    await dispose
    await dispose()
    assert seen == [root, root]
    assert _try_current() is None


async def test_listener_sees_registering_context_in_every_dispatch_mode() -> None:
    root = Context()
    seen: list[tuple[str, Context]] = []
    owners: list[Context] = []

    def plugin(ctx: Context, _config: Any) -> None:
        owners.append(ctx)
        ctx.on("app/emit", lambda: seen.append(("emit", current_context())))
        ctx.on("app/bail", lambda: seen.append(("bail", current_context())))

        async def parallel() -> None:
            await asyncio.sleep(0)
            seen.append(("parallel", current_context()))

        async def serial() -> None:
            await asyncio.sleep(0)
            seen.append(("serial", current_context()))

        def waterfall(next: Callable[[], Any]) -> Any:
            seen.append(("waterfall", current_context()))
            return next()

        ctx.on("app/parallel", parallel)
        ctx.on("app/serial", serial)
        ctx.on("app/waterfall", waterfall)

    await root.plugin(plugin)
    root.emit("app/emit")
    root.bail("app/bail")
    await root.parallel("app/parallel")
    await root.serial("app/serial")
    assert root.waterfall("app/waterfall", lambda: 1) == 1
    [ctx] = owners
    assert seen == [
        ("emit", ctx),
        ("bail", ctx),
        ("parallel", ctx),
        ("serial", ctx),
        ("waterfall", ctx),
    ]
    assert _try_current() is None


async def test_dispatch_restores_caller_context_with_mixed_owners() -> None:
    root = Context()
    seen: list[Context] = []
    owners: list[Context] = []

    def plugin(ctx: Context, _config: Any) -> None:
        owners.append(ctx)
        ctx.on("app/event", lambda: seen.append(current_context()))

    root.on("app/event", lambda: seen.append(current_context()))
    await root.plugin(plugin)
    await root.plugin(plugin)
    root.on("app/event", lambda: seen.append(current_context()))
    caller = Context()
    with use_context(caller):
        root.emit("app/event")
        assert current_context() is caller
    first, second = owners
    assert seen == [root, first, second, root]


async def test_emit_restores_context_when_listener_raises() -> None:
    root = Context()
    root.on("app/event", _raise)
    with pytest.raises(ValueError):
        root.emit("app/event")
    assert _try_current() is None


def _raise() -> None:
    raise ValueError("boom")


async def test_coroutine_listener_spawned_by_emit_sees_registering_context() -> None:
    root = Context()
    seen: list[Context] = []
    owners: list[Context] = []
    done = asyncio.Event()

    def plugin(ctx: Context, _config: Any) -> None:
        owners.append(ctx)

        async def listener() -> None:
            await asyncio.sleep(0)
            seen.append(current_context())
            done.set()

        ctx.on("app/event", listener)

    await root.plugin(plugin)
    root.emit("app/event")
    await done.wait()
    assert seen == owners


async def test_async_waterfall_listeners_see_their_own_context() -> None:
    root = Context()
    seen: list[tuple[str, Context]] = []
    owners: dict[str, Context] = {}

    def make(label: str) -> Callable[[Context, Any], None]:
        def plugin(ctx: Context, _config: Any) -> None:
            owners[label] = ctx

            async def listener(next: Callable[[], Any]) -> Any:
                result = await next()
                seen.append((label, current_context()))
                return result

            ctx.on("app/waterfall", listener)

        return plugin

    await root.plugin(make("outer"))
    await root.plugin(make("inner"))

    async def final() -> int:
        return 7

    assert await root.waterfall("app/waterfall", final) == 7
    assert seen == [("inner", owners["inner"]), ("outer", owners["outer"])]


async def test_spawned_task_sees_owning_context() -> None:
    root = Context()
    seen: list[tuple[Context, Context]] = []
    tasks: list[asyncio.Task[None]] = []

    def plugin(ctx: Context, _config: Any) -> None:
        async def work() -> None:
            await asyncio.sleep(0)
            seen.append((ctx, current_context()))

        tasks.append(ctx.spawn(work()))

    await root.plugin(plugin)
    await tasks[0]
    [(ctx, current)] = seen
    assert current is ctx


async def test_spawn_from_outside_any_context_sets_owner() -> None:
    root = Context()

    async def work() -> Context:
        return current_context()

    assert await root.spawn(work()) is root
    assert _try_current() is None


async def test_timer_callbacks_see_calling_context() -> None:
    root = Context()
    seen: list[tuple[str, Context]] = []
    owners: list[Context] = []
    done = asyncio.Event()

    def plugin(ctx: Context, _config: Any) -> None:
        owners.append(ctx)
        ctx.set_timeout(lambda: seen.append(("timeout", current_context())), 0)

        def interval() -> None:
            seen.append(("interval", current_context()))
            dispose()

        dispose = ctx.set_interval(interval, 0)
        ctx.timer.debounce(lambda: seen.append(("debounce", current_context())), 0)()
        ctx.timer.throttle(lambda: seen.append(("throttle", current_context())), 0)()

        async def later() -> None:
            seen.append(("async", current_context()))
            done.set()

        ctx.set_timeout(later, 0.01)

    await root.plugin(plugin)
    await done.wait()
    [ctx] = owners
    assert sorted(label for label, _ in seen) == [
        "async",
        "debounce",
        "interval",
        "throttle",
        "timeout",
    ]
    assert all(current is ctx for _, current in seen)


async def test_timer_callback_error_is_reported_and_context_restored() -> None:
    root = Context()
    errors = capture_reports(root)
    fired = asyncio.Event()

    def boom() -> None:
        fired.set()
        raise ValueError("boom")

    root.set_timeout(boom, 0)
    await fired.wait()
    await asyncio.sleep(0)
    assert [str(error) for error in errors] == ["boom"]
    assert _try_current() is None


async def test_plugin_disposed_observer_sees_registering_context() -> None:
    root = Context()
    seen: list[Context] = []
    owners: list[Context] = []

    def observer(ctx: Context, _config: Any) -> None:
        owners.append(ctx)

        def on_plugin(_fiber: object) -> None:
            seen.append(current_context())

        ctx.on("internal/plugin", on_plugin)

    await root.plugin(observer)
    fiber = root.plugin(lambda _ctx, _config: None)
    seen.clear()
    await fiber.dispose()
    assert seen == owners


async def test_loader_style_helper_reaches_services_without_ctx() -> None:
    """A plain helper reaches the calling plugin's services through the ambient context."""
    root = Context()
    seen: list[object] = []

    def helper() -> None:
        seen.append(current_context().get("answer"))

    def provider(ctx: Context, _config: Any) -> None:
        ctx.provide("answer", 42)

    def consumer(ctx: Context, _config: Any) -> None:
        ctx.on("app/ask", helper)

    await root.plugin(provider)
    await root.plugin(consumer)
    root.emit("app/ask")
    assert seen == [42]


async def test_fiber_local_update_hook_sees_registering_context() -> None:
    root = Context()
    seen: list[tuple[str, Context]] = []
    owners: list[Context] = []

    def plugin(ctx: Context, _config: Any) -> None:
        owners.append(ctx)

        async def on_update(_fiber: Any, _config: Any, next: Callable[[], Any]) -> None:
            await asyncio.sleep(0)
            seen.append(("update", current_context()))
            await next()

        ctx.on("internal/update", on_update)

    fiber = root.plugin(plugin, {"value": 1})
    await fiber
    await fiber.update({"value": 2})
    assert seen == [("update", owners[0])]


async def test_serial_and_parallel_restore_caller_context() -> None:
    root = Context()
    seen: list[Context] = []

    async def listener() -> None:
        await asyncio.sleep(0)
        seen.append(current_context())

    root.on("app/event", listener)
    caller = Context()
    with use_context(caller):
        await root.serial("app/event")
        await root.parallel("app/event")
        assert current_context() is caller
    assert seen == [root, root]


async def _record_use_site(view: Context, seen: dict[str, object]) -> None:
    """Record ``current_context()`` from an effect, a task, a timer and a listener of ``view``."""
    fired = asyncio.Event()

    def setup() -> None:
        seen["effect"] = current_context()

    async def task() -> None:
        seen["spawn"] = current_context()

    def timer() -> None:
        seen["timer"] = current_context()
        fired.set()

    view.effect(setup)
    await view.spawn(task())
    view.set_timeout(timer, 0)
    view.on("app/probe", lambda: seen.__setitem__("listener", current_context()))
    view.emit("app/probe")
    await fired.wait()


@pytest.mark.parametrize("derive", ["extend", "isolate"])
async def test_derived_context_is_current_for_effect_spawn_timer_and_listener(
    derive: str,
) -> None:
    root = Context()
    root.provide("db", "shared")
    seen: dict[str, object] = {}
    views: list[Context] = []

    async def plugin(ctx: Context, _config: Any) -> None:
        if derive == "isolate":
            view = ctx.isolate("db")
            view.provide("db", "isolated")
        else:
            view = ctx.extend()
        views.append(view)
        await _record_use_site(view, seen)

    await root.plugin(plugin)
    [view] = views
    assert seen == {kind: view for kind in ("effect", "spawn", "timer", "listener")}
    expected = "isolated" if derive == "isolate" else "shared"
    assert view.get("db") == expected
    assert all(current_context_db(seen[kind]) == expected for kind in seen)


def current_context_db(ctx: object) -> object:
    assert isinstance(ctx, Context)
    return ctx.get("db")


async def test_service_method_effects_see_the_calling_context() -> None:
    """Inside a service method ``self.ctx`` is a shadow; the caller is the use site."""
    root = Context()
    seen: dict[str, object] = {}
    callers: list[Context] = []

    class Probe(Service[None], provide="probe"):
        async def record(self) -> None:
            await _record_use_site(self.ctx, seen)

    await root.plugin(Probe)

    async def consumer(ctx: Context, _config: Any) -> None:
        callers.append(ctx)
        probe: Probe = ctx.probe
        await probe.record()

    await root.inject(["probe"], consumer)
    [caller] = callers
    assert seen == {kind: caller for kind in ("effect", "spawn", "timer", "listener")}
