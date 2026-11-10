import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import isolate_map

from pycordis._utils import Label
from pycordis.context import Context
from pycordis.effect import EffectDisposer
from pycordis.errors import ServiceAccessError
from pycordis.events import EventsService
from pycordis.reflect import AccessorProperty, Impl, ReflectService, ServiceProperty


async def test_reflect_installed_with_default_mixins() -> None:
    root = Context()
    assert isinstance(root.reflect, ReflectService)
    assert root.reflect.ctx is root
    child = root.extend()
    # traced: the service sees the use-site context but shares state
    assert child.reflect.ctx is child
    assert child.reflect.store is root.reflect.store
    for name in (
        *("get", "set", "provide", "accessor", "mixin"),
        *("runtime", "effect", "inject", "plugin"),
        *("on", "once", "parallel", "emit", "serial", "bail", "waterfall"),
    ):
        assert isinstance(root.reflect.props[name], AccessorProperty)
    # mixin effects are not owned by the root fiber
    assert root.fiber.get_effects() == []


async def test_provide_and_get_at_root() -> None:
    root = Context()
    dispose = root.provide("foo", 1)
    assert isinstance(dispose, EffectDisposer)
    assert dispose.meta.label == 'ctx.provide("foo")'
    assert root.foo == 1
    assert root.extend().foo == 1
    assert root.get("foo") == 1
    assert root["foo"] == 1
    assert "foo" in root
    assert isinstance(root.reflect.props["foo"], ServiceProperty)
    impl = root.reflect.get_impl("foo")
    assert isinstance(impl, Impl)
    assert (impl.name, impl.value, impl.ctx, impl.uid) == ("foo", 1, root, 1)
    assert impl in root.fiber.provisions
    assert impl.isolate is isolate_map(root)["foo"]


async def test_set_provided_service() -> None:
    root = Context()
    root.provide("foo", 1)
    root.foo = 2
    assert root.foo == 2 and root.get("foo") == 2
    assert root.set("foo", 3) is True
    assert root.foo == 3


async def test_set_without_provide() -> None:
    root = Context()
    with pytest.raises(ServiceAccessError, match=r'^cannot set property "bar" without provide$'):
        root.set("bar", 1)
    # an undeclared public name on a root-fiber context is a plain attribute
    root.bar = 1
    assert root.bar == 1 and "bar" not in root.reflect.props


async def test_get_undeclared_raises_attribute_error() -> None:
    root = Context()
    with pytest.raises(ServiceAccessError, match=r'^cannot get property "nope" without inject$'):
        _ = root.nope
    assert not hasattr(root, "nope")
    assert getattr(root, "nope", 5) == 5
    assert "nope" not in root
    assert root.get("nope") is None


async def test_attribute_error_has_no_chained_context() -> None:
    root = Context()
    try:
        _ = root.nope
    except ServiceAccessError as error:
        assert error.__cause__ is None
        assert error.__suppress_context__ or error.__context__ is None
    else:  # pragma: no cover
        pytest.fail("expected ServiceAccessError")


async def test_duplicate_provide() -> None:
    root = Context()
    root.provide("foo", 1)
    with pytest.raises(ServiceAccessError, match=r'^service "foo" has been registered at <root>'):
        root.provide("foo", 2)
    assert root.foo == 1


async def test_dispose_provide_removes_and_notifies() -> None:
    root = Context()
    seen: list[tuple[Any, ...]] = []

    def on_service(ctx: Context, name: str, value: object) -> None:
        seen.append((name, value))

    root.on("internal/service", on_service)
    dispose = root.provide("foo", 1)
    assert seen == [("foo", 1)]
    impl = root.reflect.get_impl("foo")
    assert impl is not None
    await dispose()
    assert seen == [("foo", 1), ("foo", None)]
    assert root.get("foo") is None
    assert root.foo is None  # still declared as a service
    assert impl not in root.fiber.provisions
    assert impl.release is None
    # can be provided again after disposal
    root.provide("foo", 2)
    assert root.foo == 2


async def test_provision_waits_for_dependents() -> None:
    root = Context()
    dispose = root.provide("foo", 1)
    impl = root.reflect.get_impl("foo")
    assert impl is not None
    impl.dependents.add(root.fiber)
    pending = dispose()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not pending.done()
    assert root.get("foo") is None  # unregistered immediately
    assert impl in root.fiber.provisions  # kept for dependent teardown
    assert impl.release is not None
    impl.release()
    await pending
    assert impl not in root.fiber.provisions


async def test_isolated_provide_invisible_at_root() -> None:
    root = Context()
    isolated = root.isolate("foo")
    seen: list[Any] = []

    def on_service(ctx: Context, name: str, value: object) -> None:
        seen.append(value)

    root.on("internal/service", on_service)
    isolated.provide("foo", 1)
    assert isolated.foo == 1 and isolated.get("foo") == 1
    assert root.get("foo") is None
    assert root.foo is None
    # the root listener's scope does not match the isolated provision
    assert seen == []
    root.provide("foo", 2)
    assert (root.foo, isolated.foo) == (2, 1)
    # a shared str label joins scopes
    a, b = root.isolate("bar", "s"), root.isolate("bar", "s")
    a.provide("bar", 3)
    assert b.bar == 3 and root.get("bar") is None


async def test_strict_get_requires_active_fiber() -> None:
    root = Context()
    root.provide("foo", 1)
    root.fiber.state = root.fiber.state.__class__.LOADING
    try:
        assert root.get("foo") is None
        assert root.get("foo", False) == 1
        assert root.foo == 1  # attribute access is non-strict at root
    finally:
        root.fiber.state = root.fiber.state.__class__.ACTIVE


async def test_accessor_get_set() -> None:
    root = Context()
    store: dict[str, Any] = {"v": 1}
    calls: list[Any] = []

    def get(ctx: Context, receiver: Any) -> Any:
        calls.append(ctx)
        return store["v"]

    def set_(ctx: Context, value: Any, receiver: Any) -> bool:
        store["v"] = value
        return True

    dispose = root.accessor("prop", get=get, set=set_)
    assert dispose.meta.label == 'ctx.accessor("prop")'
    child = root.extend()
    assert child.prop == 1 and calls == [child]
    child.prop = 5
    assert store["v"] == 5 and root.prop == 5
    assert "prop" in root
    with pytest.raises(
        ServiceAccessError, match=r'^property "prop" is already declared as accessor'
    ):
        root.accessor("prop", get=get)
    with pytest.raises(
        ServiceAccessError, match=r'^property "prop" is already declared as accessor'
    ):
        root.provide("prop", 1)
    await dispose()
    assert "prop" not in root.reflect.props
    assert "prop" not in root


async def test_accessor_read_only_and_rejecting_setter() -> None:
    root = Context()
    root.accessor("ro", get=lambda ctx, receiver: 1)
    with pytest.raises(ServiceAccessError, match=r'^cannot set property "ro"$'):
        root.ro = 2
    root.accessor("veto", get=lambda ctx, receiver: 1, set=lambda ctx, value, receiver: False)
    with pytest.raises(ServiceAccessError, match=r'^cannot set property "veto"$'):
        root.veto = 2


async def test_service_name_declared_as_service_rejects_accessor() -> None:
    root = Context()
    root.provide("svc", 1)
    with pytest.raises(ServiceAccessError, match=r'^property "svc" is already declared as service'):
        root.accessor("svc", get=lambda ctx, receiver: 1)


async def test_accessor_removed_on_root_dispose() -> None:
    root = Context()
    root.accessor("prop", get=lambda ctx, receiver: 1)
    root.provide("foo", 1)
    await root.fiber.dispose()
    assert "prop" not in root.reflect.props
    assert root.get("foo") is None
    # built-in mixins survive the root restart
    assert isinstance(root.reflect.props["on"], AccessorProperty)


async def test_mixin_forwards_and_binds() -> None:
    root = Context()
    emitted: list[Any] = []
    root.on("app/x", emitted.append)
    root.emit("app/x", 1)
    assert emitted == [1]

    child = root.extend()
    seen: list[int] = []
    # listener registered through the mixin is owned by the calling context
    disposer = root.fiber.effect(lambda: child.on("app/y", seen.append), "outer")
    assert [m.label for m in disposer.meta.children] == ['ctx.on("app/y")']
    assert child.on.__self__.ctx is child
    await disposer()
    root.emit("app/y", 2)
    assert seen == []


async def test_mixin_object_source_and_mapping() -> None:
    root = Context()

    class Source:
        value = 3

        def __init__(self) -> None:
            self.calls: list[Any] = []

        def hello(self, x: int) -> int:
            self.calls.append(x)
            return x + 1

    source = Source()
    dispose = root.mixin(source, {"hello": "greet", "value": "val"})
    assert dispose.meta.label.startswith("ctx.mixin(")
    assert root.greet(1) == 2 and source.calls == [1]
    assert root.val == 3
    root.val = 4
    assert source.value == 4
    await dispose()
    assert "greet" not in root and "val" not in root


async def test_mixin_named_source_and_nullable() -> None:
    root = Context()
    holder = SimpleNamespace(x=1)
    root.holder = holder
    root.mixin("holder", ["x"])
    assert root.x == 1
    root.holder = None
    assert root.x is None


async def test_fiber_mixins() -> None:
    root = Context()
    assert root.runtime is None
    seen: list[int] = []
    dispose = root.effect(lambda: lambda: seen.append(1), "eff")
    assert isinstance(dispose, EffectDisposer)
    await dispose()
    assert seen == [1]


async def test_events_traced_per_context() -> None:
    root = Context()
    child = root.extend()
    assert isinstance(child.events, EventsService)
    assert child.events.ctx is child
    assert child.events.hooks is root.events.hooks


async def test_contains_and_getitem() -> None:
    root = Context()
    assert "extend" in root and "fiber" in root and "on" in root
    assert "_isolate" in root and "_nope" not in root
    root.provide("foo.bar", 1)
    assert root["foo.bar"] == 1
    assert "foo.bar" in root
    with pytest.raises(ServiceAccessError):
        root["missing.name"]


async def test_internal_get_skipped_for_root_def_site() -> None:
    root = Context()
    calls: list[Any] = []
    root.on("internal/get", lambda *args: calls.append(args))
    root.provide("foo", 1)
    assert root.foo == 1
    assert calls == []


def _plugin_fiber(parent: Context, **kwargs: Any) -> Any:
    """A minimal stand-in for a plugin fiber (real ones arrive with the registry)."""
    return SimpleNamespace(
        runtime=SimpleNamespace(name="stub"),
        parent=parent,
        committed=kwargs.get("committed", {}),
        provisions=kwargs.get("provisions", set()),
        inject=kwargs.get("inject", {}),
    )


async def test_internal_get_waterfall_overrides_read() -> None:
    root = Context()
    plugin = root.extend(fiber=_plugin_fiber(root))
    seen: list[Any] = []

    def on_get(ctx: Context, name: str, error: Exception, next: Any) -> Any:
        seen.append((ctx, name, str(error)))
        return "override" if name == "foo" else next()

    root.on("internal/get", on_get)
    assert plugin.foo == "override"
    assert seen == [(plugin, "foo", 'cannot get property "foo" without inject')]
    with pytest.raises(ServiceAccessError, match=r'^cannot get property "bar" without inject$'):
        _ = plugin.bar


async def test_internal_get_listener_added_later_intercepts() -> None:
    root = Context()
    root.provide("foo", 1)
    plugin = root.extend(fiber=_plugin_fiber(root))
    assert plugin.foo == 1  # fast path: no internal/get listener yet
    with pytest.raises(ServiceAccessError, match=r'^cannot get property "bar" without inject$'):
        _ = plugin.bar
    seen: list[str] = []

    def on_get(_ctx: Context, name: str, _error: Exception, next: Any) -> Any:
        seen.append(name)
        return "override" if name == "bar" else next()

    dispose = root.on("internal/get", on_get)
    assert plugin.foo == 1
    assert plugin.bar == "override"
    assert seen == ["foo", "bar"]
    dispose()
    with pytest.raises(ServiceAccessError, match=r'^cannot get property "bar" without inject$'):
        _ = plugin.bar
    assert seen == ["foo", "bar"]


async def test_service_access_error_traceback_is_short() -> None:
    root = Context()
    plugin = root.extend(fiber=_plugin_fiber(root, inject={"bar": None}))
    for ctx, name in ((plugin, "bar"), (plugin, "baz"), (root, "baz")):
        with pytest.raises(ServiceAccessError) as info:
            getattr(ctx, name)
        assert info.value.__cause__ is None and info.value.__suppress_context__
        assert info.value.name == name and info.value.obj is ctx


async def test_fiber_walk_resolution() -> None:
    root = Context()
    root.provide("foo", 1)
    # found in an ancestor fiber's provisions (the root fiber)
    plugin = root.extend(fiber=_plugin_fiber(root))
    assert plugin.foo == 1
    # required but not committed → inactive
    required = root.extend(fiber=_plugin_fiber(root, inject={"bar": None}))
    with pytest.raises(
        ServiceAccessError, match=r'^cannot get required service "bar" in inactive context$'
    ):
        _ = required.bar
    # committed implementation wins
    impl = Impl(99, "bar", root, Label("bar"), set(), value="committed")
    committed = root.extend(fiber=_plugin_fiber(root, committed={"bar": impl}))
    assert committed.bar == "committed"
    # a provision matching the def-site isolation key
    isolated = root.isolate("baz")
    own = Impl(98, "baz", isolated, isolate_map(isolated)["baz"], set(), value="own")
    provider = isolated.extend(fiber=_plugin_fiber(isolated, provisions={own}))
    assert provider.baz == "own"
    # the walk stops at a parent with a different isolation key
    inner = isolated.extend(fiber=_plugin_fiber(root))
    with pytest.raises(ServiceAccessError, match=r'^cannot get property "foo" without inject$'):
        _ = inner.extend(fiber=_plugin_fiber(root.isolate("foo"))).foo


async def test_internal_set_waterfall_and_plugin_set() -> None:
    root = Context()
    root.provide("foo", 1)
    seen: list[Any] = []

    def on_set(ctx: Context, name: str, value: Any, error: Exception, next: Any) -> Any:
        seen.append((name, value, str(error)))
        return True if value == "veto" else next()

    root.on("internal/set", on_set)
    root.foo = 2
    root.foo = "veto"
    assert root.foo == 2
    assert seen == [
        ("foo", 2, 'cannot set property "foo" without provide'),
        ("foo", "veto", 'cannot set property "foo" without provide'),
    ]
    plugin = root.extend(fiber=_plugin_fiber(root))
    with pytest.raises(ServiceAccessError, match=r'^cannot set property "nope" without provide$'):
        plugin.nope = 1


async def test_set_from_another_fiber() -> None:
    root = Context()
    root.provide("foo", 1)
    plugin = root.extend(fiber=_plugin_fiber(root))
    with pytest.raises(ServiceAccessError, match=r'^cannot set property "foo" in multiple fibers'):
        plugin.set("foo", 2)
    # through attribute assignment the inner error propagates unchanged
    with pytest.raises(ServiceAccessError, match=r'^cannot set property "foo" in multiple fibers'):
        plugin.foo = 2
    assert root.foo == 1


async def test_set_in_unprovided_isolated_scope() -> None:
    root = Context()
    root.provide("foo", 1)
    with pytest.raises(ServiceAccessError, match=r'^cannot set property "foo" without provide'):
        root.isolate("foo").set("foo", 2)


async def test_accessor_error_propagates() -> None:
    root = Context()

    def get(ctx: Context, receiver: Any) -> Any:
        raise ServiceAccessError("inner")

    root.accessor("bad", get=get)
    with pytest.raises(ServiceAccessError, match=r"^inner$"):
        _ = root.bad


async def test_bind_wraps_and_contains_non_str() -> None:
    root = Context()

    def callback() -> None:
        return None

    bound = root.reflect.bind(callback)
    assert bound is not callback and bound() is None
    assert 1 not in root


async def test_internal_set_listener_may_raise_carrier_error() -> None:
    root = Context()
    root.provide("foo", 1)

    def on_set(ctx: Context, name: str, value: Any, error: Exception, next: Any) -> Any:
        raise error

    root.on("internal/set", on_set)
    with pytest.raises(ServiceAccessError, match=r'^cannot set property "foo" without provide'):
        root.foo = 2
    assert root.foo == 1


async def test_internal_set_falsy_result_rejects_write() -> None:
    root = Context()
    root.provide("foo", 1)
    root.on("internal/set", lambda *_: False)
    with pytest.raises(ServiceAccessError, match=r'^cannot set property "foo"$'):
        root.foo = 2
    assert root.foo == 1


async def test_contains_excludes_class_symbols() -> None:
    root = Context()
    assert "filter" not in root and not hasattr(root, "filter")
    assert "effect" in root and hasattr(root, "effect")  # mixed-in accessor
    assert "isolate" in root and "root" in root


async def test_service_access_error_name_and_obj() -> None:
    root = Context()
    child = root.extend()
    with pytest.raises(ServiceAccessError) as info:
        _ = child.nope
    assert info.value.name == "nope" and info.value.obj is child
    with pytest.raises(ServiceAccessError) as info:
        root.set("bar", 1)
    assert info.value.name == "bar" and info.value.obj is root
