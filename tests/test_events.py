import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import capture_reports

from pycordis._utils import DisposableList, symbols
from pycordis.context import Context
from pycordis.current import current_context
from pycordis.effect import EffectDisposer, ListenerDisposer
from pycordis.errors import ValidationError
from pycordis.events import EventKey, EventOptions, EventsService, Hook
from pycordis.fiber import Fiber, emit_plugin_disposed


async def test_events_installed_on_root() -> None:
    root = Context()
    assert isinstance(root.events, EventsService)
    # each context reads a traced view sharing one hook table
    assert root.extend().events.hooks is root.events.hooks
    # built-in listeners are not owned by the root fiber's disposables
    assert root.fiber.disposables() == []
    assert set(root.events.hooks) == {
        "internal/listener",
        "internal/update",
        "internal/config",
    }


def _recorder(seen: list[str], prefix: str) -> Callable[[object], None]:
    """A listener appending ``f"{prefix}{value}"`` to ``seen``."""

    def listener(value: object) -> None:
        seen.append(f"{prefix}{value}")

    return listener


async def test_emit_order_honours_prepend() -> None:
    root = Context()
    seen: list[str] = []
    root.on("x", _recorder(seen, "a"))
    root.on("x", _recorder(seen, "b"), True)
    root.on("x", _recorder(seen, "c"), {"prepend": True})
    root.on("x", _recorder(seen, "d"), EventOptions(prepend=False))
    assert root.emit("x", 1) is None
    assert seen == ["c1", "b1", "a1", "d1"]


async def test_emit_sync_error_propagates() -> None:
    root = Context()

    def boom() -> None:
        raise ValueError("boom")

    root.on("x", boom)
    with pytest.raises(ValueError, match="boom"):
        root.emit("x")


async def test_serial_returns_first_bailed_value_and_stops() -> None:
    root = Context()
    seen: list[str] = []

    async def a() -> None:
        seen.append("a")

    async def b() -> bool:
        seen.append("b")
        return False

    async def c() -> str:
        seen.append("c")
        return "C"

    def d() -> str:
        seen.append("d")
        return "D"

    for listener in (a, b, c, d):
        root.on("x", listener)
    assert await root.serial("x") == "C"
    assert seen == ["a", "b", "c"]
    assert await root.serial("missing") is None


async def test_serial_sync_listener_values() -> None:
    root = Context()
    root.on("x", lambda: 0)
    assert await root.serial("x") == 0  # 0 bails: only None/False don't


async def test_bail_sync() -> None:
    root = Context()
    seen: list[str] = []

    def a(v: int) -> None:
        seen.append("a")

    def b(v: int) -> int:
        seen.append("b")
        return v * 2

    def c(v: int) -> int:
        seen.append("c")
        return 0

    for listener in (a, b, c):
        root.on("x", listener)
    assert root.bail("x", 21) == 42
    assert seen == ["a", "b"]
    assert root.bail("missing") is None


async def test_parallel_collects_failures() -> None:
    root = Context()
    seen: list[str] = []

    async def ok() -> None:
        await asyncio.sleep(0)
        seen.append("ok")

    async def fail1() -> None:
        await asyncio.sleep(0)
        raise ValueError("one")

    def fail2() -> None:
        raise KeyError("two")

    for listener in (fail1, ok, fail2):
        root.on("x", listener)
    with pytest.raises(ExceptionGroup) as info:
        await root.parallel("x")
    assert info.value.message == "parallel listeners failed"
    assert len(info.value.exceptions) == 2
    assert {type(e) for e in info.value.exceptions} == {ValueError, KeyError}
    assert seen == ["ok"]


async def test_parallel_success() -> None:
    root = Context()
    seen: list[str] = []

    async def a() -> None:
        seen.append("a-start")
        await asyncio.sleep(0)
        seen.append("a-end")

    root.on("x", a)
    root.on("x", lambda: seen.append("b"))
    assert await root.parallel("x") is None
    # sync listeners run at call time; coroutines start when gathered
    assert seen == ["b", "a-start", "a-end"]


async def test_parallel_reraises_base_exception() -> None:
    root = Context()

    def interrupt() -> None:
        raise KeyboardInterrupt

    root.on("x", interrupt)
    root.on("x", lambda: 1 / 0)
    with pytest.raises(KeyboardInterrupt):
        await root.parallel("x")


async def test_waterfall_ordering() -> None:
    root = Context()
    seen: list[str] = []

    def outer(value: int, next: Any) -> Any:
        seen.append("outer-before")
        result = next()
        seen.append("outer-after")
        return result + 1

    def inner(value: int, next: Any) -> Any:
        seen.append("inner-before")
        result = next()
        seen.append("inner-after")
        return result * 10

    def core() -> int:
        seen.append("core")
        return 4

    root.on("w", outer)
    root.on("w", inner)
    assert root.waterfall("w", 7, core) == 41
    assert seen == ["outer-before", "inner-before", "core", "inner-after", "outer-after"]


async def test_waterfall_veto_and_double_next() -> None:
    root = Context()
    called: list[str] = []

    def veto(next: Callable[[], object]) -> str:
        return "vetoed"

    root.on("w", veto)
    assert root.waterfall("w", lambda: called.append("core")) == "vetoed"
    assert called == []

    other = Context()

    def twice(next: Any) -> None:
        next()
        next()

    other.on("w", twice)
    with pytest.raises(RuntimeError, match=r"^next\(\) called multiple times$"):
        other.waterfall("w", lambda: None)
    assert other.waterfall("empty", lambda: "inner") == "inner"


async def test_once_fires_once() -> None:
    root = Context()
    seen: list[int] = []
    dispose = root.once("x", seen.append)
    root.emit("x", 1)
    root.emit("x", 2)
    assert seen == [1]
    assert dispose() is False


async def test_once_decorator() -> None:
    root = Context()
    seen: list[int] = []

    @root.once("x")
    def listener(v: int) -> None:
        seen.append(v)

    root.emit("x", 1)
    root.emit("x", 2)
    assert seen == [1]
    assert callable(listener)


async def test_disposer_true_then_false() -> None:
    root = Context()
    seen: list[int] = []
    dispose = root.on("x", seen.append)
    assert isinstance(dispose, ListenerDisposer)
    assert isinstance(dispose.effect, EffectDisposer)
    assert dispose() is True
    assert dispose() is False
    root.emit("x", 1)
    assert seen == []
    assert "x" not in root.events.hooks


async def test_effect_label() -> None:
    root = Context()
    dispose = root.on("app/ready", lambda: None)
    assert dispose.effect.meta.label == 'ctx.on("app/ready")'
    assert [m.label for m in root.fiber.get_effects()] == ['ctx.on("app/ready")']
    assert dispose.__cordis_effect__ is dispose.effect.meta


async def test_listener_removed_with_owning_effect() -> None:
    root = Context()
    seen: list[int] = []
    disposer = root.fiber.effect(lambda: root.on("x", seen.append), "outer")
    root.emit("x", 1)
    # the nested registration is owned by the outer effect, not the fiber
    assert [m.label for m in root.fiber.get_effects()] == ["outer"]
    assert [m.label for m in disposer.meta.children] == ['ctx.on("x")']
    await disposer()
    root.emit("x", 2)
    assert seen == [1]


async def test_listener_removed_on_fiber_dispose() -> None:
    root = Context()
    seen: list[int] = []
    dispose = root.on("x", seen.append)
    await root.fiber.dispose()
    root.emit("x", 1)
    assert seen == []
    assert dispose() is False
    # built-ins survive root dispose
    assert "internal/update" in root.events.hooks


async def test_registration_owned_by_calling_context() -> None:
    root = Context()
    child = root.extend(tag="child")
    child.on("x", lambda: None)
    hook = root.events.hooks["x"][0]
    assert isinstance(hook, Hook)
    assert hook.ctx is child and hook.prepend is False and hook.global_ is False


async def test_filter_skips_non_global_hooks() -> None:
    root = Context()
    seen: list[str] = []
    root.on("x", _recorder(seen, "local"))
    root.on("x", _recorder(seen, "global"), {"global": True})
    root.on("x", _recorder(seen, "global_"), EventOptions(global_=True))
    contexts: list[Any] = []

    def deny(ctx: Any) -> bool:
        contexts.append(ctx)
        return False

    receiver = SimpleNamespace(**{symbols.filter: deny})
    root.emit(receiver, "x", 1)
    assert seen == ["global1", "global_1"]
    assert contexts == [root]
    root.emit(SimpleNamespace(), "x", 2)  # receiver without filter: no filtering
    root.emit(None, "x", 3)  # None is a receiver too
    assert seen == [
        "global1",
        "global_1",
        "local2",
        "global2",
        "global_2",
        "local3",
        "global3",
        "global_3",
    ]


async def test_decorator_form() -> None:
    root = Context()
    seen: list[int] = []

    @root.on("x")
    def listener(v: int) -> None:
        seen.append(v)

    root.emit("x", 5)
    assert seen == [5]
    assert listener.__name__ == "listener"


async def test_coroutine_listener_under_emit() -> None:
    root = Context()
    errors = capture_reports(root)
    seen: list[str] = []

    async def ok() -> None:
        await asyncio.sleep(0)
        seen.append("ok")

    async def fail() -> None:
        await asyncio.sleep(0)
        raise ValueError("async boom")

    root.on("x", ok)
    root.on("x", fail)
    root.emit("x")  # does not raise
    tasks = root.fiber.owned_tasks()
    assert len(tasks) == 2
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)
    assert seen == ["ok"]
    assert [str(e) for e in errors] == ["async boom"]
    assert not root.fiber.owned_tasks()


async def test_coroutine_listener_under_filtered_emit_runs_as_owner_task() -> None:
    root = Context()
    seen: list[Context] = []
    owners: list[Context] = []

    async def listener() -> None:
        await asyncio.sleep(0)
        seen.append(current_context())

    def plugin(ctx: Context, _config: Any) -> None:
        owners.append(ctx)
        ctx.on("x", listener)

    def admit(_ctx: Context) -> bool:
        return True

    fiber = await root.plugin(plugin)
    receiver = SimpleNamespace(**{symbols.filter: admit})
    root.emit(receiver, "x")
    tasks = fiber.owned_tasks()
    assert len(tasks) == 1
    await asyncio.gather(*tasks)
    assert seen == owners


async def test_spawn_cancellation_not_reported() -> None:
    root = Context()
    errors = capture_reports(root)
    task = root.fiber.spawn_owned(asyncio.sleep(10), name="sleeper")
    assert task.get_name() == "sleeper"
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert errors == []
    assert not root.fiber.owned_tasks()


async def test_internal_dispatch() -> None:
    root = Context()
    seen: list[tuple[Any, ...]] = []
    root.on("x", lambda *_: None)
    root.emit("x", 1)  # no internal/dispatch listener yet: not emitted
    root.on("internal/dispatch", lambda *args: seen.append(args))
    root.emit("x", 1)
    assert seen == [("emit", "x", [1], None)]
    root.emit("internal/status", 1)  # internal events are not reported
    receiver = SimpleNamespace()
    root.bail(receiver, "y", 2)
    assert seen[-1] == ("bail", "y", [2], receiver)
    assert len(seen) == 2


async def test_dispatch_returns_callables() -> None:
    root = Context()
    seen: list[Any] = []
    root.on("x", seen.append)
    callbacks = root.events.dispatch("emit", ["x", 1])
    assert len(callbacks) == 1
    callbacks[0](9)
    assert seen == [9]


async def test_event_key() -> None:
    root = Context()
    ready: EventKey[[int, str], None] = EventKey("app/ready")
    assert ready.name == "app/ready" and repr(ready) == "EventKey('app/ready')"
    seen: list[tuple[int, str]] = []

    def listener(a: int, b: str) -> None:
        seen.append((a, b))

    dispose = root.events.on(ready, listener)
    assert dispose.effect.meta.label == 'ctx.on("app/ready")'
    root.events.emit(ready, 1, "a")
    root.emit("app/ready", 2, "b")  # interoperates with the plain name
    root.emit(SimpleNamespace(), ready, 3, "c")
    assert seen == [(1, "a"), (2, "b"), (3, "c")]


async def test_kwargs_rejected() -> None:
    root = Context()
    with pytest.raises(TypeError, match="positional"):
        root.emit("x", a=1)  # pyright: ignore[reportCallIssue, reportArgumentType]


NAME_ERROR = "event name must be a str or EventKey, got int"


async def test_a_lone_non_name_is_rejected_as_the_event_name() -> None:
    root = Context()
    with pytest.raises(TypeError, match=NAME_ERROR):
        root.emit(123)
    with pytest.raises(TypeError, match=NAME_ERROR):
        await root.parallel(123)
    with pytest.raises(TypeError, match=NAME_ERROR):
        await root.serial(123)
    with pytest.raises(TypeError, match=NAME_ERROR):
        root.bail(123)
    with pytest.raises(TypeError, match=NAME_ERROR):
        root.waterfall(123)


async def test_a_receiver_without_an_event_name_is_rejected() -> None:
    root = Context()
    with pytest.raises(TypeError, match=NAME_ERROR):
        root.emit(SimpleNamespace(), 123)
    with pytest.raises(TypeError, match="an event name is required"):
        root.emit()
    with pytest.raises(TypeError, match="an event name is required"):
        root.waterfall()


async def test_waterfall_without_a_next_continuation_is_rejected() -> None:
    root = Context()

    def listener(next: Callable[[], object]) -> object:
        return next()

    root.on("chain", listener)
    with pytest.raises(TypeError, match="waterfall needs a final next argument"):
        root.waterfall("chain")


async def test_a_receiver_stays_optional() -> None:
    root = Context()
    seen: list[object] = []
    root.on("x", seen.append)
    root.emit("x", 1)
    root.emit(SimpleNamespace(), "x", 2)
    assert seen == [1, 2]


async def test_inactive_fiber_rejects_on() -> None:
    root = Context()
    root.fiber.uid = None
    from pycordis.errors import CordisError

    with pytest.raises(CordisError):
        root.on("x", lambda: None)


async def test_custom_internal_listener_result_replaces_registration() -> None:
    root = Context()
    sentinel = object()

    def replace_y(ctx: Context, name: str, listener: object, options: EventOptions) -> object:
        return sentinel if name == "y" else None

    root.on("internal/listener", replace_y)
    assert root.on("y", lambda: None) is sentinel
    assert "y" not in root.events.hooks


async def test_internal_listener_receives_context_first() -> None:
    root = Context()
    child = root.extend()
    seen: list[Any] = []
    root.on("internal/listener", lambda *args: seen.append(args))
    child.on("z", print)
    # the listener arrives wrapped by `ctx.reflect.bind`
    assert seen[0][0] is child and seen[0][1] == "z" and seen[0][2].__wrapped__ is print
    assert seen[0][3] == EventOptions()


async def test_internal_update_routed_to_fiber_hooks() -> None:
    root = Context()
    seen: list[str] = []

    def first(fiber: Any, config: Any, next: Any) -> Any:
        seen.append(f"first:{config}")
        return next()

    def second(fiber: Any, config: Any, next: Any) -> Any:
        assert fiber is root.fiber
        seen.append(f"second:{config}")
        return next()

    def prepended(fiber: Any, config: Any, next: Any) -> Any:
        seen.append(f"pre:{config}")
        return next()

    d1 = root.on("internal/update", first)
    root.on("internal/update", second)
    root.on("internal/update", prepended, True)
    assert len(root.fiber.update_hooks()) == 3
    assert 'ctx.on("internal/update")' in [m.label for m in root.fiber.get_effects()]

    result = root.waterfall(root.fiber, "internal/update", "cfg", lambda: "restarted")
    assert result == "restarted"
    assert seen == ["pre:cfg", "first:cfg", "second:cfg"]

    assert d1() is True
    assert d1() is False
    assert len(root.fiber.update_hooks()) == 2
    await root.fiber.dispose()
    assert root.fiber.update_hooks() == []


async def test_internal_update_veto_and_global() -> None:
    root = Context()

    def veto(fiber: Fiber, config: object, next: Callable[[], object]) -> str:
        return "veto"

    root.on("internal/update", veto)
    assert root.waterfall(root.fiber, "internal/update", 1, lambda: "restarted") == "veto"
    # a global listener is a normal event hook, not a per-fiber hook
    other = Context()
    seen: list[Any] = []

    def observer(fiber: Any, config: Any, next: Any) -> Any:
        seen.append(fiber)
        return next()

    other.on("internal/update", observer, {"global": True})
    assert other.fiber.update_hooks() == []
    assert other.waterfall(other.fiber, "internal/update", 1, lambda: "r") == "r"
    assert seen == [other.fiber]


async def test_internal_config_validates_with_runtime_schema() -> None:
    from dataclasses import dataclass

    @dataclass
    class Config:
        port: int

    root = Context()
    stand_in = SimpleNamespace(runtime=SimpleNamespace(Config=Config))
    result = root.waterfall("internal/config", stand_in, lambda: {"port": 1})
    assert result == Config(port=1)
    with pytest.raises(ValidationError):
        root.waterfall("internal/config", stand_in, lambda: {"port": "x"})
    no_schema = SimpleNamespace(runtime=SimpleNamespace())
    assert root.waterfall("internal/config", no_schema, lambda: {"a": 1}) == {"a": 1}
    # root fiber: no runtime, config passes through
    assert root.waterfall("internal/config", root.fiber, lambda: 5) == 5


async def test_unknown_attribute_still_raises() -> None:
    root = Context()
    with pytest.raises(AttributeError):
        _ = root.nothing


async def test_emit_plugin_disposed_contains_errors() -> None:
    root = Context()
    errors = capture_reports(root)
    seen: list[Fiber] = []

    def bad(fiber: Fiber) -> None:
        raise ValueError("sync observer")

    async def bad_async(fiber: Fiber) -> None:
        raise ValueError("async observer")

    root.on("internal/plugin", bad)
    root.on("internal/plugin", bad_async)
    root.on("internal/plugin", seen.append)
    emit_plugin_disposed(root, root.fiber)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert seen == [root.fiber]
    assert sorted(str(e) for e in errors) == ["async observer", "sync observer"]


async def test_emit_plugin_disposed_dispatch_failure_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Context()
    errors = capture_reports(root)

    def dispatch_owned(self: EventsService, mode: str, args: list[Any]) -> list[Any]:
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(EventsService, "dispatch_owned", dispatch_owned)
    emit_plugin_disposed(root, root.fiber)
    assert [str(e) for e in errors] == ["dispatch failed"]


async def test_unregister_edge_cases() -> None:
    root = Context()
    seen: list[int] = []
    d1 = root.on("x", seen.append)
    d2 = root.on("x", seen.append)
    assert d1() is True
    root.emit("x", 1)
    assert seen == [1]
    assert d2() is True
    events = root.events
    assert events.unregister("x", seen.append) is False
    root.on("y", print)
    assert events.unregister("y", seen.append) is False


async def test_disposable_list_unshift() -> None:
    items: DisposableList[str] = DisposableList()
    items.push("b")
    remove_a = items.unshift("a")
    items.push("c")
    assert list(items) == ["a", "b", "c"]
    assert remove_a() is True and remove_a() is False
    assert list(items) == ["b", "c"]
    assert items.delete("c") and list(items) == ["b"]


# -- dispatch snapshots stay in step with registrations ------------------------


def _named(log: list[str], label: str) -> Callable[[], None]:
    def listener() -> None:
        log.append(label)

    return listener


async def test_dispatch_sees_every_registration_change_between_emits() -> None:
    root = Context()
    log: list[str] = []
    dispose_a = root.on("x", _named(log, "a"))
    root.emit("x")
    root.on("x", _named(log, "b"))
    root.emit("x")
    root.on("x", _named(log, "first"), True)
    root.emit("x")
    assert dispose_a()
    root.emit("x")
    root.once("x", _named(log, "once"), True)
    root.emit("x")
    root.emit("x")
    assert log == [
        *["a"],
        *["a", "b"],
        *["first", "a", "b"],
        *["first", "b"],
        *["once", "first", "b"],
        *["first", "b"],
    ]


async def test_dispatch_drops_listeners_of_a_disposed_fiber_after_an_emit() -> None:
    root = Context()
    log: list[str] = []
    fiber = await root.plugin(lambda ctx, _: ctx.on("x", _named(log, "plugin")))
    root.on("x", _named(log, "root"))
    root.emit("x")
    await fiber.dispose()
    root.emit("x")
    assert log == ["plugin", "root", "root"]


async def test_registrations_during_an_emit_apply_from_the_next_one() -> None:
    root = Context()
    log: list[str] = []
    removable: list[ListenerDisposer] = []

    def first() -> None:
        log.append("first")
        if removable:  # on the first emit: add "late", remove "b"
            root.on("x", _named(log, "late"))
            dispose_b = removable.pop()
            dispose_b()

    root.on("x", first)
    removable.append(root.on("x", _named(log, "b")))
    root.emit("x")  # a snapshot: "b" still runs, "late" does not yet
    root.emit("x")
    assert log == ["first", "b", "first", "late"]


async def test_alternating_owners_each_see_their_own_context() -> None:
    root = Context()
    seen: list[tuple[str, Context]] = []
    owners: dict[str, Context] = {}

    def owner(label: str) -> Callable[[Context, Any], None]:
        def apply(ctx: Context, _config: Any) -> None:
            owners[label] = ctx
            ctx.on("x", lambda: seen.append((label, current_context())))

        return apply

    await root.plugin(SimpleNamespace(name="a", apply=owner("a")))
    await root.plugin(SimpleNamespace(name="b", apply=owner("b")))
    root.on("x", lambda: seen.append(("root", current_context())))
    await root.plugin(SimpleNamespace(name="c", apply=owner("a2")))
    for _ in range(2):
        root.emit("x")
    expected = [("a", owners["a"]), ("b", owners["b"]), ("root", root), ("a2", owners["a2"])]
    assert seen == expected * 2
    with pytest.raises(LookupError):
        current_context()
