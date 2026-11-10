import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from conftest import apply_plugin, capture_reports, intercept_map, with_meta

from pycordis._utils import DisposableList, symbols
from pycordis.context import Context
from pycordis.errors import CordisError, ValidationError
from pycordis.fiber import Fiber, FiberState
from pycordis.registry import (
    Inject,
    InjectSpec,
    RegistryService,
    Runtime,
    inject,
    plugin_meta,
)
from pycordis.trace import Tracker

# -- installation ------------------------------------------------------------


async def test_registry_installed_and_traced() -> None:
    root = Context()
    assert isinstance(root.registry, RegistryService)
    assert root.registry.ctx is root
    child = root.extend()
    assert child.registry.ctx is child
    assert len(root.registry) == 0
    assert root.fiber.get_effects() == []


async def test_counter_increments() -> None:
    root = Context()
    first = root.registry.counter
    assert root.registry.counter == first + 1
    assert root.extend().registry.counter == first + 2


# -- plugin shapes -----------------------------------------------------------


async def test_function_plugin() -> None:
    root = Context()
    seen: list[tuple[Context, Any]] = []
    disposed: list[str] = []

    def plugin(ctx: Context, config: Any) -> Callable[[], None]:
        seen.append((ctx, config))
        return lambda: disposed.append("x")

    fiber = root.plugin(plugin, {"a": 1})
    assert isinstance(fiber, Fiber)
    assert fiber.state == FiberState.LOADING
    assert seen == []
    assert await fiber is fiber
    assert fiber.state == FiberState.ACTIVE
    [(ctx, config)] = seen
    assert ctx is fiber.ctx and ctx.fiber is fiber and config == {"a": 1}
    assert fiber.parent is root
    assert fiber.runtime is root.registry.get(plugin)
    assert fiber.uid is not None and fiber.uid > 0
    assert fiber.name == "plugin"
    assert ctx.runtime is fiber.runtime
    await fiber.dispose()
    assert disposed == ["x"]
    assert fiber.state == FiberState.DISPOSED
    assert fiber.uid is None
    assert not root.registry.has(plugin)


async def test_class_plugin() -> None:
    root = Context()
    events: list[Any] = []

    class Plugin:
        def __init__(self, ctx: Context, config: Any) -> None:
            self.ctx = ctx
            events.append(("init", config))
            self.__cordis_init_hooks__ = [lambda: events.append("hook")]

        def __cordis_init__(self) -> Callable[[], None]:
            events.append("cordis_init")
            return lambda: events.append("dispose")

    fiber = await root.plugin(Plugin, 5)
    assert events == [("init", 5), "hook", "cordis_init"]
    assert fiber.name == "Plugin"
    await fiber.dispose()
    assert events[-1] == "dispose"


async def test_plain_class_plugin() -> None:
    root = Context()
    instances: list[Any] = []

    class Plugin:
        def __init__(self, ctx: Context, config: Any) -> None:
            instances.append(self)

    fiber = await root.plugin(Plugin)
    assert len(instances) == 1 and fiber.state == FiberState.ACTIVE


async def test_object_plugin() -> None:
    root = Context()
    seen: list[Any] = []

    class Obj:
        name = "obj-plugin"

        def apply(self, ctx: Context, config: Any) -> None:
            seen.append((ctx, config))

    obj = Obj()
    fiber = await root.plugin(obj, 3)
    assert seen == [(fiber.ctx, 3)]
    assert fiber.name == "obj-plugin"
    assert root.registry.has(obj) and root.registry.get(obj) is fiber.runtime

    def apply(ctx: Context, config: Any) -> None:
        seen.append(config)

    ns = SimpleNamespace(apply=apply)  # no name: the runtime stays anonymous
    other = await root.plugin(ns, 4)
    assert seen[-1] == 4
    assert other.runtime is not None and other.runtime.name is None
    assert other.name == "root"


async def test_generator_plugin() -> None:
    root = Context()
    events: list[str] = []

    def plugin(ctx: Context, config: Any) -> Iterator[Callable[[], None]]:
        events.append("start")
        yield lambda: events.append("first")
        yield lambda: events.append("second")

    fiber = await root.plugin(plugin)
    assert events == ["start"]
    await fiber.dispose()
    assert events == ["start", "second", "first"]


async def test_async_generator_plugin() -> None:
    root = Context()
    events: list[str] = []

    async def plugin(ctx: Context, config: Any) -> AsyncIterator[Callable[[], None]]:
        await asyncio.sleep(0)
        events.append("start")
        yield lambda: events.append("dispose")

    fiber = await root.plugin(plugin)
    assert events == ["start"]
    assert fiber.state == FiberState.ACTIVE
    await fiber.dispose()
    assert events == ["start", "dispose"]


async def test_async_function_plugin() -> None:
    root = Context()
    events: list[str] = []

    async def plugin(ctx: Context, config: Any) -> Callable[[], None]:
        await asyncio.sleep(0)
        events.append("start")
        return lambda: events.append("dispose")

    fiber = await root.plugin(plugin)
    assert events == ["start"]
    await fiber.dispose()
    assert events == ["start", "dispose"]


async def test_effects_inside_plugin_are_owned_by_fiber() -> None:
    root = Context()
    events: list[str] = []

    def plugin(ctx: Context) -> None:
        ctx.on("custom", lambda: events.append("custom"))
        ctx.effect(lambda: lambda: events.append("effect"))

    fiber = await root.plugin(lambda ctx, _: plugin(ctx))
    root.emit("custom")
    assert events == ["custom"]
    assert [m.label for m in fiber.get_effects()] == ['ctx.on("custom")', "anonymous"]
    assert [m.label for m in root.fiber.get_effects()] == ["ctx.plugin()"]
    await fiber.dispose()
    root.emit("custom")
    assert events == ["custom", "effect"]
    assert root.fiber.get_effects() == []


async def test_nested_plugin_disposed_with_parent() -> None:
    root = Context()
    inner: list[Fiber] = []

    def child(ctx: Context, config: Any) -> None:
        pass

    def parent(ctx: Context, config: Any) -> None:
        inner.append(ctx.plugin(child))

    fiber = await root.plugin(parent)
    [child_fiber] = inner
    await child_fiber
    assert child_fiber.parent is fiber.ctx
    assert child_fiber.state == FiberState.ACTIVE
    await fiber.dispose()
    assert child_fiber.state == FiberState.DISPOSED
    assert len(root.registry) == 0


# -- errors --------------------------------------------------------------------


async def test_invalid_plugin() -> None:
    root = Context()
    with pytest.raises(TypeError) as info:
        root.plugin(1)
    assert str(info.value) == (
        'invalid plugin, expect function or object with an "apply" method, received int'
    )
    assert root.registry.resolve(object()) is None

    class Bad:
        @property
        def apply(self) -> Any:
            raise RuntimeError("boom")

    assert root.registry.resolve(Bad()) is None
    with pytest.raises(TypeError, match="invalid plugin") as raised:
        root.plugin(Bad())
    # the getter's own error is kept as the cause, not swallowed
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "boom"
    assert not root.registry.has(1)
    assert root.registry.get(1) is None
    assert root.registry.delete(1) is None


async def test_plugin_on_disposed_context() -> None:
    root = Context()
    fiber = await root.plugin(lambda ctx, config: None)
    await fiber.dispose()
    with pytest.raises(CordisError) as info:
        fiber.ctx.plugin(lambda ctx, config: None)
    assert info.value.code == "INACTIVE_EFFECT"


async def test_startup_error_surfaces_through_await() -> None:
    root = Context()
    reports = capture_reports(root)
    error = RuntimeError("startup")

    def plugin(ctx: Context, config: Any) -> None:
        ctx.effect(lambda: lambda: None)
        raise error

    fiber = root.plugin(plugin)
    with pytest.raises(RuntimeError) as info:
        await fiber
    assert info.value is error
    assert fiber.state == FiberState.FAILED
    with pytest.raises(RuntimeError):
        await fiber.wait()
    assert reports == [error]
    assert fiber.get_effects() == []
    await fiber.dispose()
    assert fiber.state == FiberState.DISPOSED


async def test_config_validation_error() -> None:
    root = Context()
    reports = capture_reports(root)

    @dataclass
    class Config:
        port: int

    seen: list[Any] = []

    @plugin_meta(Config=Config)
    def plugin(ctx: Context, config: Config) -> None:
        seen.append(config)

    fiber = await root.plugin(plugin, {"port": 80})
    assert seen == [Config(80)]
    runtime = root.registry.get(plugin)
    assert runtime is not None
    assert runtime.Config is Config
    bad = root.plugin(plugin, {"port": "x"})
    with pytest.raises(ValidationError):
        await bad
    assert bad.state == FiberState.FAILED
    assert len(reports) == 1
    assert fiber.state == FiberState.ACTIVE


async def test_internal_plugin_event_and_rollback() -> None:
    root = Context()
    seen: list[Any] = []

    def on_plugin(fiber: Fiber) -> None:
        seen.append((fiber, fiber.state))

    root.on("internal/plugin", on_plugin)

    def plugin(ctx: Context, config: Any) -> None:
        pass

    fiber = root.plugin(plugin)
    assert seen == [(fiber, FiberState.PENDING)]
    await fiber
    await fiber.dispose()
    # published on disposal, before the fiber leaves its current state
    assert seen[-1] == (fiber, FiberState.ACTIVE)
    assert fiber.uid is None

    error = RuntimeError("observer")
    failed: list[Fiber] = []

    def bad(fiber: Fiber) -> None:
        if fiber.uid is not None:
            failed.append(fiber)
            raise error

    root.on("internal/plugin", bad)
    with pytest.raises(RuntimeError) as info:
        root.plugin(plugin)
    assert info.value is error
    # rolled back before control escapes
    [rolled_back] = failed
    assert rolled_back.uid is None
    assert not root.registry.has(plugin)
    for _ in range(10):
        await asyncio.sleep(0)
    assert rolled_back.state == FiberState.DISPOSED
    assert [m.label for m in root.fiber.get_effects()] == [
        'ctx.on("internal/plugin")',
        'ctx.on("internal/plugin")',
    ]


# -- registry map ------------------------------------------------------------


async def test_same_callback_shares_runtime() -> None:
    root = Context()

    def plugin(ctx: Context, config: Any) -> None:
        pass

    a = await root.plugin(plugin, 1)
    b = await root.plugin(plugin, 2)
    runtime = root.registry.get(plugin)
    assert isinstance(runtime, Runtime)
    assert a.runtime is runtime and b.runtime is runtime
    assert list(runtime.fibers) == [a, b]
    assert runtime.callback is plugin and runtime.name == "plugin"
    assert len(root.registry) == 1
    assert list(root.registry) == [plugin]
    assert list(root.registry.keys()) == [plugin]
    assert list(root.registry.values()) == [runtime]
    assert list(root.registry.items()) == [(plugin, runtime)]
    visited: list[tuple[Runtime, Any]] = []
    root.registry.for_each(lambda value, key: visited.append((value, key)))
    assert visited == [(runtime, plugin)]
    assert a.uid != b.uid

    await a.dispose()
    assert list(runtime.fibers) == [b]
    assert root.registry.has(plugin)
    await b.dispose()
    assert not root.registry.has(plugin)
    assert len(root.registry) == 0


async def test_registry_delete_disposes_all_fibers() -> None:
    root = Context()
    disposed: list[int] = []

    def plugin(ctx: Context, config: int) -> Callable[[], None]:
        return lambda: disposed.append(config)

    a = await root.plugin(plugin, 1)
    b = await root.plugin(plugin, 2)
    runtime = root.registry.delete(plugin)
    assert runtime is a.runtime
    assert not root.registry.has(plugin)
    for _ in range(20):
        await asyncio.sleep(0)
    assert sorted(disposed) == [1, 2]
    assert a.state == FiberState.DISPOSED and b.state == FiberState.DISPOSED
    assert root.fiber.get_effects() == []


async def test_root_dispose_unloads_plugins() -> None:
    disposed: list[str] = []
    async with Context() as root:
        await root.plugin(lambda ctx, config: lambda: disposed.append("x"))
    assert disposed == ["x"]
    assert len(root.registry) == 0


# -- names -------------------------------------------------------------------


async def test_plugin_name_rules() -> None:
    root = Context()

    def apply(ctx: Context, config: Any) -> None:
        pass

    fiber = await root.plugin(apply)
    assert fiber.runtime is not None and fiber.runtime.name is None
    assert fiber.name == "root"
    anon = await root.plugin(lambda ctx, config: None)
    assert anon.runtime is not None and anon.runtime.name is None
    assert anon.name == "root"

    @plugin_meta(name="custom")
    def named(ctx: Context, config: Any) -> None:
        pass

    assert (await root.plugin(named)).name == "custom"

    inner: list[Fiber] = []

    def outer(ctx: Context, config: Any) -> None:
        inner.append(ctx.plugin(lambda ctx, config: None))

    await root.plugin(outer)
    assert inner[0].name == "outer"


# -- inject ------------------------------------------------------------------


async def test_inject_resolve_forms() -> None:
    assert Inject.resolve(None) == {}
    assert Inject.resolve(["a", "b"]) == {"a": None, "b": None}
    assert Inject.resolve({"a": {"x": 1}, "b": None}) == {"a": {"x": 1}, "b": None}
    result: dict[str, Any] = {"z": 1}
    assert Inject.resolve(["a"], result) is result
    assert result == {"z": 1, "a": None}
    with pytest.raises(TypeError):
        Inject.resolve("abc")  # pyright: ignore[reportArgumentType]


async def test_inject_class_decorator_inheritance() -> None:
    @inject("a")
    class Base:
        # declared for type checkers only; ``@inject`` sets the class attribute
        inject: ClassVar[InjectSpec]

    @inject("c", {"level": 2})
    @inject("b")
    class Child(Base):
        pass

    class Plain(Base):
        inject: ClassVar[InjectSpec] = ["p"]

    @inject("q")
    class FromPlain(Plain):
        pass

    @inject("own")
    class OwnList:
        inject: ClassVar[list[str]] = ["x"]

    assert Inject.resolve(Base.inject) == {"a": None}
    assert Inject.resolve(Child.inject) == {"a": None, "b": None, "c": {"level": 2}}
    assert Inject.resolve(FromPlain.inject) == {"p": None, "q": None}
    assert Inject.resolve(OwnList.inject) == {"x": None, "own": None}


async def test_inject_loads_with_available_service() -> None:
    root = Context()
    root.provide("foo", 1)
    seen: list[Any] = []
    fiber = await root.inject(["foo"], lambda ctx, config: seen.append(ctx.foo))
    assert seen == [1]
    assert fiber.state == FiberState.ACTIVE
    assert fiber.inject == {"foo": None}


async def test_inject_pending_without_service() -> None:
    root = Context()
    seen: list[Any] = []
    fiber = await root.inject({"foo": {"x": 1}}, lambda ctx, config: seen.append(ctx))
    assert fiber.state == FiberState.PENDING
    assert seen == []
    assert intercept_map(fiber.ctx).get("foo") == {"x": 1}
    assert intercept_map(root).get("foo") is None
    await fiber.dispose()
    assert fiber.state == FiberState.DISPOSED


async def test_inject_method_decorator() -> None:
    root = Context()
    root.provide("foo", 1)
    seen: list[Any] = []

    class Plugin:
        __cordis_tracker__ = Tracker(property="ctx")

        def __init__(self, ctx: Context, config: Any) -> None:
            self.ctx = ctx

        @inject("foo", {"k": 1})
        @inject("bar")
        def on_ready(self) -> None:
            seen.append(self.ctx)

        def plain(self) -> str:
            return "plain"

    metadata = getattr(Plugin.on_ready, symbols.metadata)
    assert metadata == {"inject": {"bar": None, "foo": {"k": 1}}}
    fiber = await root.plugin(Plugin)
    assert seen == []
    root.provide("bar", 2)
    for _ in range(5):
        await asyncio.sleep(0)
    [inner] = seen
    assert inner.fiber is not fiber and inner.fiber.parent is fiber.ctx
    assert inner.fiber.inject == {"foo": {"k": 1}, "bar": None}
    instance_hooks = len(root.registry)
    assert instance_hooks == 2  # Plugin + the method's inner plugin


async def test_inject_method_without_tracker() -> None:
    root = Context()
    root.provide("foo", 1)
    selves: list[Any] = []

    class Plugin:
        def __init__(self, ctx: Context, config: Any) -> None:
            self.ctx = ctx

        @inject("foo")
        def ready(self) -> None:
            selves.append(self)

    await root.plugin(Plugin)
    for _ in range(5):
        await asyncio.sleep(0)
    [self_] = selves
    assert type(self_) is Plugin
    # the method still works as a plain method
    self_.ready()
    assert len(selves) == 2


async def test_inject_rejects_other_targets() -> None:
    with pytest.raises(TypeError) as info:
        inject("foo")(staticmethod(lambda: None))
    assert str(info.value) == "@Inject() can only be used on class or class methods"


# -- dependency bookkeeping --------------------------------------------------


async def test_provider_plugin_and_dependent() -> None:
    root = Context()
    events: list[str] = []

    def provider(ctx: Context, config: Any) -> None:
        ctx.provide("svc", "value")
        ctx.effect(lambda: lambda: events.append("provider"))

    @plugin_meta(inject=["svc"])
    def dependent(ctx: Context, config: Any) -> Callable[[], None]:
        events.append(f"dependent:{ctx.svc}")

        def dispose() -> None:
            # still readable during own teardown
            events.append(f"dispose:{ctx.svc}")

        return dispose

    dep = root.plugin(dependent)
    await dep
    assert dep.state == FiberState.PENDING
    prov = await root.plugin(provider)
    for _ in range(5):
        await asyncio.sleep(0)
    await dep
    assert dep.state == FiberState.ACTIVE
    assert events == ["dependent:value"]
    await prov.dispose()
    await dep
    assert events == ["dependent:value", "dispose:value", "provider"]
    assert dep.state == FiberState.PENDING


async def test_runtime_dataclass() -> None:
    def cb() -> None:
        pass

    fibers: DisposableList[Fiber] = DisposableList()
    runtime = Runtime(cb, fibers)
    assert runtime.name is None and runtime.Config is None
    assert runtime != Runtime(cb, fibers)


async def test_inject_respects_service_check() -> None:
    root = Context()
    reports = capture_reports(root)
    root.provide("off", 1, check=lambda: False)
    error = RuntimeError("check")

    def broken() -> bool:
        raise error

    root.provide("broken", 2, check=broken)

    class Svc:
        __cordis_tracker__ = Tracker(property="ctx")

        def __init__(self, ctx: Context) -> None:
            self.ctx = ctx
            self.seen: list[Any] = []

        def check(self) -> bool:
            self.seen.append(self.ctx)
            return True

    svc = Svc(root)
    root.provide("svc", svc, check=svc.check)

    off = await root.inject(["off"], lambda ctx, config: None)
    assert off.state == FiberState.PENDING
    bad = await root.inject(["broken"], lambda ctx, config: None)
    assert bad.state == FiberState.PENDING
    assert reports == [error]
    ok = await root.inject(["svc"], lambda ctx, config: None)
    assert ok.state == FiberState.ACTIVE
    # check runs with `self` traced to the dependent's context
    assert svc.seen == [ok.ctx]


async def test_init_hooks_descriptor_on_class() -> None:
    class Plugin:
        @inject("foo")
        def ready(self) -> None:
            pass

    hooks = Plugin.__dict__["__cordis_init_hooks__"]
    assert getattr(Plugin, symbols.init_hooks) is hooks
    instance = Plugin()
    first: list[Any] = getattr(instance, symbols.init_hooks)
    assert len(first) == 1
    assert getattr(instance, symbols.init_hooks) is first


# -- metadata of module plugins ----------------------------------------------------


async def test_module_plugin_ignores_mistyped_metadata() -> None:
    import types

    import pycordis

    module = types.ModuleType("modplug")
    seen: list[Any] = []

    def apply(_ctx: Context, config: Any) -> None:
        seen.append(config)

    # names imported into the module, not plugin metadata
    module.__dict__.update(apply=apply, inject=inject, name=pycordis, Config=types, provide=print)
    root = Context()
    fiber = await root.plugin(module, {"a": 1})
    assert fiber.state == FiberState.ACTIVE
    assert fiber.inject == {}
    assert seen == [{"a": 1}]
    runtime = root.registry.get(module)
    assert runtime is not None and runtime.name is None and runtime.Config is None

    # well-typed module metadata is still honoured
    typed = types.ModuleType("typed")

    def typed_apply(_ctx: Context, _config: Any) -> None: ...

    typed.__dict__.update(apply=typed_apply, inject=["svc"], name="typed")
    typed_fiber = root.plugin(typed)
    assert typed_fiber.inject == {"svc": None}
    typed_runtime = root.registry.get(typed)
    assert typed_runtime is not None and typed_runtime.name == "typed"
    await root.fiber.dispose()


async def test_object_plugin_with_invalid_inject_raises() -> None:
    root = Context()
    plugin = apply_plugin("bad", lambda _ctx, _config: None, inject=42)
    with pytest.raises(
        TypeError, match="plugin 'bad' has invalid \"inject\": expected list or mapping, got int"
    ):
        root.plugin(plugin)

    def fn(_ctx: Context, _config: Any) -> None: ...

    with_meta(fn, inject=inject)
    with pytest.raises(TypeError, match=r"plugin 'fn' has invalid \"inject\".*got function"):
        root.plugin(fn)
    assert len(root.registry) == 0


async def test_invalid_inject_items_are_type_errors_naming_the_problem() -> None:
    root = Context()

    def apply(_ctx: Context, _config: None) -> None:
        return None

    numbered = SimpleNamespace(inject=[1], apply=apply)
    with pytest.raises(TypeError, match="inject must name services by str or ServiceKey, got int"):
        root.plugin(numbered)

    class Scalar:
        inject = 5

    with pytest.raises(TypeError, match="inject must be a list or mapping, got int"):
        inject("db")(Scalar)
