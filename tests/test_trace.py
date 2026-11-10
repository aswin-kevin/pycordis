from collections.abc import AsyncIterator, Iterable, Iterator, Sized
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from pycordis._context_internals import lookup, own_meta, parent_of, traced_cache
from pycordis._utils import symbols
from pycordis.context import Context
from pycordis.errors import ServiceAccessError
from pycordis.events import EventsService
from pycordis.fiber import Fiber
from pycordis.reflect import ReflectService
from pycordis.trace import Traceable, Tracker, get_traceable, invoke, with_props


class Svc:
    """A service traced with the cordis default tracker (``property="ctx"``)."""

    __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx")
    inner: "Svc"
    """Set by tests that nest one traced service inside another."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.value = 1
        self._where = 0

    def where(self) -> Any:
        return self.ctx

    def sites(self) -> tuple[Any, Any]:
        ctx = self.ctx
        # a shadow context's parent is the use site; it has no public accessor
        return parent_of(ctx), lookup(ctx, symbols.shadow)

    def me(self) -> Any:
        return self

    def listen(self, seen: list[Any]) -> Any:
        return self.ctx.on("app/x", seen.append)

    def caller_of_events(self) -> Any:
        events: Any = self.ctx.events
        return events._cordis_caller, events.ctx

    def nested(self) -> Any:
        return self.where()

    def make(self) -> "Svc":
        return Svc(self.ctx)

    @property
    def prop(self) -> Any:
        return self.ctx

    @property
    def settable(self) -> int:
        return self._where

    @settable.setter
    def settable(self, value: int) -> None:
        self._where = value
        self.last_ctx = self.ctx

    @staticmethod
    def static() -> int:
        return 7

    @classmethod
    def klass(cls) -> type:
        return cls


class Raw:
    """A service traced with ``no_shadow``: ``self`` stays the proxy."""

    __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx", no_shadow=True)

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.items = [1, 2]

    def me(self) -> Any:
        return self

    def make(self) -> Svc:
        return Svc(self.ctx)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[int]:
        return iter(self.items)

    def __contains__(self, item: object) -> bool:
        return item in self.items

    def __getitem__(self, index: int) -> int:
        return self.items[index]

    def __repr__(self) -> str:
        return f"Raw({self.items})"


def _scoped(root: Context) -> Context:
    """A child context owning its own (root-style) fiber."""
    child = root.extend()
    child.fiber = Fiber(child)
    return child


async def test_property_returns_use_site() -> None:
    root = Context()
    svc = Svc(root)
    root.provide("svc", svc)
    child = root.extend(tag="child")
    proxy = child.svc
    assert isinstance(proxy, Traceable)
    assert proxy.ctx is child
    assert root.svc.ctx is root
    assert isinstance(proxy, Svc)
    assert proxy._cordis_original is svc
    assert proxy._cordis_caller is child
    assert proxy.value == 1
    assert svc.ctx is root  # target untouched


async def test_method_sees_shadow_context() -> None:
    root = Context()
    svc = Svc(root)
    root.provide("svc", svc)
    child = root.extend()
    # inside the method, the use site governs the context and the def site (the
    # service's own ctx) rides along as its shadow
    assert child.svc.sites() == (child, root)
    # a shadow context returned from a method is traced back to its use site
    assert child.svc.where() is child
    # services read inside the method resolve with the def site as caller
    caller, use = child.svc.caller_of_events()
    assert caller is root and use is child
    # nested self-calls keep the same use site
    assert child.svc.nested() is child
    # self is a shadow receiver, not the proxy, but behaves like the service
    me = child.svc.me()
    assert isinstance(me, Svc)
    # a shadow context's parent is the use site; it has no public accessor
    assert parent_of(me.ctx) is child
    assert me.value == 1


async def test_method_registers_on_callers_fiber() -> None:
    root = Context()
    svc = Svc(root)
    root.provide("svc", svc)
    caller = _scoped(root)
    seen: list[Any] = []
    caller.svc.listen(seen)
    provider_seen: list[Any] = []
    root.svc.listen(provider_seen)
    hooks = root.events.hooks["app/x"]
    assert [hook.ctx for hook in hooks] == [caller, root]
    root.emit("app/x", 1)
    assert seen == [1] and provider_seen == [1]
    # disposing the caller removes its listener; the provider's survives
    await caller.fiber.dispose()
    root.emit("app/x", 2)
    assert seen == [1] and provider_seen == [1, 2]


async def test_method_results_traced_to_use_site() -> None:
    root = Context()
    root.provide("svc", Svc(root))
    child = root.extend()
    made = child.svc.make()
    assert isinstance(made, Traceable)
    assert made.ctx is child


async def test_property_getter_and_setter_receive_shadow() -> None:
    root = Context()
    svc = Svc(root)
    root.provide("svc", svc)
    child = root.extend()
    assert parent_of(child.svc.prop) is child
    child.svc.settable = 5
    assert svc.settable == 5
    # a shadow context's parent is the use site; it has no public accessor
    assert parent_of(svc.last_ctx) is child
    child.svc.value = 3
    assert svc.value == 3
    assert child.svc.static() == 7
    assert child.svc.klass() is Svc
    with pytest.raises(AttributeError):
        _ = child.svc.missing
    del child.svc.value
    assert not hasattr(svc, "value")


async def test_reserved_names_are_read_only() -> None:
    root = Context()
    root.provide("svc", Svc(root))
    proxy = root.extend().svc
    for name in ("ctx", "_cordis_original", "_cordis_caller"):
        with pytest.raises(ServiceAccessError, match=f'cannot set property "{name}"'):
            setattr(proxy, name, 1)


async def test_no_shadow_keeps_self() -> None:
    root = Context()
    raw = Raw(root)
    root.provide("raw", raw)
    child = root.extend()
    proxy = child.raw
    assert proxy.me() is proxy
    assert proxy.me().ctx is child
    # no_shadow method results are not traced
    assert type(proxy.make()) is Svc
    # dunder forwarding
    assert len(proxy) == 2 and list(proxy) == [1, 2] and 2 in proxy and proxy[0] == 1
    assert repr(proxy) == "Raw([1, 2])" and str(proxy) == "Raw([1, 2])"
    assert bool(proxy) is True
    assert "items" in dir(proxy)
    assert proxy.__class__ is Raw


async def test_dunder_missing_raises_type_error() -> None:
    root = Context()
    root.provide("svc", Svc(root))
    proxy = root.svc
    with pytest.raises(TypeError):
        len(proxy)
    with pytest.raises(TypeError):
        iter(proxy)
    with pytest.raises(TypeError):
        _ = proxy[0]
    with pytest.raises(TypeError):
        _ = 1 in proxy
    assert bool(proxy) is True


async def test_associate() -> None:
    root = Context()

    class Foo:
        def __init__(self, ctx: Context) -> None:
            self.ctx = ctx
            self.__cordis_tracker__ = Tracker(property="ctx", associate="foo")
            self.own = "own"

    root.provide("foo", Foo(root))
    root.provide("foo.bar", 1)
    assert root.foo.bar == 1
    assert root["foo.bar"] == 1
    assert root.foo.own == "own"
    root.foo.bar = 2
    assert root.foo.bar == 2


async def test_mixin_with_receiver() -> None:
    root = Context()

    class Foo:
        __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx", associate="foo")

        def __init__(self, ctx: Context) -> None:
            self.ctx = ctx
            self.only_foo = "foo"

    class Ext:
        def __init__(self) -> None:
            self.own = "ext"
            self.flag = 0

        def describe(self) -> tuple[Any, Any]:
            # members of the mixin source win; the rest comes from the receiver
            receiver: Any = self
            return receiver.own, receiver.only_foo

    root.provide("foo", Foo(root))
    ext = Ext()
    root.mixin(ext, {"describe": "foo.describe", "flag": "foo.flag"})
    assert root.foo.describe() == ("ext", "foo")
    assert root.foo.flag == 0
    root.foo.flag = 4
    assert ext.flag == 4


async def test_callable_service_invoke() -> None:
    root = Context()

    class Callable_:
        __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx")

        def __init__(self, ctx: Context) -> None:
            self.ctx = ctx

        def __cordis_invoke__(self, name: str) -> tuple[str, Any]:
            return name, self.ctx

        def __call__(self, name: str) -> Any:
            return invoke(self, name)

    class RawCallable(Callable_):
        __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx", no_shadow=True)

        def __cordis_invoke__(self, name: str) -> tuple[str, Any]:
            return name, self

    root.provide("call", Callable_(root))
    root.provide("raw_call", RawCallable(root))
    child = root.extend()
    name, ctx = child.call("a")
    assert name == "a" and parent_of(ctx) is child and lookup(ctx, symbols.shadow) is root
    proxy = child.raw_call
    assert proxy("b") == ("b", proxy)
    # calling the target directly traces it to its own context (cordis createCallable)
    target = root.call._cordis_original
    name, ctx = target("c")
    assert parent_of(ctx) is root
    # a tracked callable without an invoke hook is just called
    plain_tracked: Any = SimpleNamespace(__cordis_tracker__=Tracker(), ctx=root)
    with pytest.raises(TypeError):
        get_traceable(root, plain_tracked)()


async def test_callable_without_invoke() -> None:
    root = Context()

    class Fn:
        __cordis_tracker__: ClassVar[Tracker] = Tracker()

        def __call__(self, x: int) -> int:
            return x * 2

    proxy = get_traceable(root, Fn())
    assert proxy(2) == 4


async def test_cache_returns_same_proxy() -> None:
    root = Context()
    root.provide("svc", Svc(root))
    child = root.extend()
    assert child.svc is child.svc
    assert child.events is child.events
    assert child.reflect is child.reflect
    assert child.svc is not root.svc
    # a replaced service is not served from the cache
    other = Svc(root)
    root.svc = other
    assert child.svc._cordis_original is other


async def test_get_traceable_passthrough() -> None:
    root = Context()
    child = root.extend()
    assert get_traceable(root, 1) == 1
    assert get_traceable(root, None) is None
    assert get_traceable(root, child) is child
    obj = SimpleNamespace(a=1)
    assert get_traceable(root, obj) is obj
    assert get_traceable(root, Svc) is Svc  # classes are never traced
    bogus = SimpleNamespace(__cordis_tracker__="not a tracker")
    assert get_traceable(root, bogus) is bogus
    # a traced value is re-traced to the new context
    proxy: Any = get_traceable(root, Svc(root))
    again = get_traceable(child, proxy)
    assert again._cordis_original is proxy._cordis_original
    assert again.ctx is child


async def test_extend_from_shadow_keeps_shadow_on_top() -> None:
    root = Context()
    root.provide("svc", Svc(root))
    child = root.extend()
    shadow = child.svc.prop
    assert parent_of(shadow) is child
    extended = shadow.extend(tag=1)
    assert lookup(extended, symbols.shadow) is root
    assert extended.tag == 1
    base = get_traceable(root, extended)
    assert parent_of(base) is child and base.tag == 1
    isolated = shadow.isolate("svc")
    assert lookup(isolated, symbols.shadow) is root
    # extend from a plain context adds no shadow layer
    assert symbols.shadow not in own_meta(child.extend())


async def test_tracker_without_property() -> None:
    root = Context()

    class Bare:
        __cordis_tracker__: ClassVar[Tracker] = Tracker()

        def me(self) -> Any:
            return self

    proxy = get_traceable(root, Bare())
    assert proxy.me() is proxy


async def test_target_ctx_missing() -> None:
    root = Context()

    class NoCtx:
        __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx")
        ctx = None

        def me(self) -> Any:
            return self

    proxy = get_traceable(root, NoCtx())
    assert proxy.me() is proxy


async def test_with_props() -> None:
    target = SimpleNamespace(a=1, b=2)
    view = with_props(target, {"a": 10})
    assert view.a == 10 and view.b == 2
    view.a = 11
    view.b = 3
    assert view.a == 11 and target.a == 1 and target.b == 3
    assert view.__class__ is SimpleNamespace
    assert repr(view) == repr(target)
    assert with_props(target, None) is target
    func_view = with_props(len, {})
    assert func_view([1]) == 1


async def test_bind_traces_self_and_args() -> None:
    root = Context()
    svc = Svc(root)
    child = root.extend()
    seen: list[Any] = []

    def callback(*args: Any, **kwargs: Any) -> str:
        seen.append((args, kwargs))
        return "ok"

    bound = child.reflect.bind(callback)
    assert bound(svc, 1, key=svc) == "ok"
    (args, kwargs) = seen[0]
    assert args[0].ctx is child and args[1] == 1 and kwargs["key"].ctx is child
    assert bound.__name__ == "callback"

    class Holder:
        __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx")

        def __init__(self) -> None:
            self.ctx = root

        def method(self) -> Any:
            return self.ctx

    method = child.reflect.bind(Holder().method)
    assert method() is child
    assert child.reflect.trace(svc).ctx is child


async def test_bind_marks_coroutine_functions() -> None:
    import inspect

    root = Context()

    async def callback() -> int:
        return 1

    bound = root.reflect.bind(callback)
    assert inspect.iscoroutinefunction(bound)
    assert await bound() == 1


async def test_listeners_receive_traced_args() -> None:
    root = Context()
    svc = Svc(root)
    child = root.extend()
    seen: list[Any] = []
    child.on("app/x", seen.append)
    root.emit("app/x", svc)
    assert seen[0].ctx is child and seen[0]._cordis_original is svc


async def test_builtin_services_trackers() -> None:
    root = Context()
    for value in (root.events, root.reflect):
        assert isinstance(value, Traceable)
    assert isinstance(root.events, EventsService)
    assert isinstance(root.reflect, ReflectService)
    tracker = getattr(EventsService, symbols.tracker)
    assert tracker == Tracker(property="ctx", no_shadow=True)
    assert getattr(ReflectService, symbols.tracker) == tracker
    events: Any = root.events
    assert not hasattr(events._cordis_original, "_bind")


class Slotted:
    """A slotted service: no instance ``__dict__``, members are data descriptors."""

    __slots__ = ("ctx", "value")
    __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx")

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.value = 5

    async def sites(self) -> tuple[Any, Any]:
        ctx = self.ctx
        assert ctx is not None  # only `assign_ctx` (expected to fail) writes None
        return parent_of(ctx), lookup(ctx, symbols.shadow)

    def assign_ctx(self) -> None:
        self.ctx = None

    write_only = property(fset=lambda self, value: None)


async def test_slotted_service_and_async_method() -> None:
    import inspect

    root = Context()
    root.provide("slotted", Slotted(root))
    child = root.extend()
    proxy = child.slotted
    assert proxy.value == 5
    assert inspect.iscoroutinefunction(proxy.sites)
    assert await proxy.sites() == (child, root)
    with pytest.raises(AttributeError, match="has no getter"):
        _ = proxy.write_only
    # the shadow receiver's ctx is read-only
    with pytest.raises(ServiceAccessError, match='cannot set property "ctx"'):
        proxy.assign_ctx()


async def test_tracked_attribute_values_are_traced() -> None:
    root = Context()
    svc = Svc(root)
    svc.inner = Svc(root)
    root.provide("svc", svc)
    child = root.extend()
    inner = child.svc.inner
    assert isinstance(inner, Traceable) and inner.ctx is child


async def test_tracker_property_absent_on_target() -> None:
    root = Context()

    class NoAttr:
        __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx")

        def me(self) -> Any:
            return self

    proxy = get_traceable(root, NoAttr())
    assert proxy.me() is proxy


async def test_invoke_variants() -> None:
    root = Context()

    def increment(x: int) -> int:
        return x + 1

    hooked: Any = SimpleNamespace(
        __cordis_tracker__=Tracker(), __cordis_invoke__=increment, ctx=root
    )
    assert get_traceable(root, hooked)(1) == 2
    with pytest.raises(TypeError, match="has no tracker"):
        invoke(object())


async def test_mixin_of_traced_service_with_receiver() -> None:
    root = Context()

    class Foo:
        __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx", associate="foo")

        def __init__(self, ctx: Context) -> None:
            self.ctx = ctx
            self.only_foo = "foo"

    class Ext:
        def __init__(self) -> None:
            self.own = "ext"

        def bump(self) -> None:
            receiver: Any = self
            receiver.own = "bumped"
            receiver.only_foo = "changed"

    foo = Foo(root)
    ext = Ext()
    root.provide("foo", foo)
    root.provide("svc", Svc(root))
    root.mixin(ext, {"bump": "foo.bump"})
    root.mixin("svc", {"sites": "foo.sites", "value": "foo.value", "nested": "foo.nested"})
    child = root.extend()
    child.foo.bump()
    assert ext.own == "bumped" and foo.only_foo == "changed"
    # the traced source's methods still see the use site and def site
    use_site, def_site = child.foo.sites()
    assert parent_of(use_site) is child and def_site is root
    assert child.foo.value == 1
    # `self.where` inside the mixed-in method resolves on the traced source
    assert parent_of(child.foo.nested()) is child


async def test_with_props_nesting_and_properties() -> None:
    class Target:
        def __init__(self) -> None:
            self.stored = 0
            self.base = "target"

        @property
        def me(self) -> Any:
            return self

        @property
        def setter(self) -> int:
            return self.stored

        @setter.setter
        def setter(self, value: int) -> None:
            self.stored = value

    target = Target()
    inner = with_props(target, {"a": 1})
    outer = with_props(inner, {"b": 2})
    assert outer.a == 1 and outer.b == 2 and outer.base == "target"
    assert outer.me is outer
    outer.setter = 3
    assert target.stored == 3
    outer.base = "changed"
    assert target.base == "changed"
    # an object overlay: its members win, nested views are searched too
    overlay = with_props(SimpleNamespace(), outer)
    assert overlay.a == 1 and overlay.base == "changed"


async def test_stored_proxy_retraced_to_use_site() -> None:
    root = Context()

    class Holder:
        __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx")

        def __init__(self, ctx: Context) -> None:
            self.ctx = ctx
            self.events = ctx.events  # a proxy traced to the provider

        def listen(self, seen: list[Any]) -> Any:
            return self.events.on("app/x", seen.append)

    root.provide("holder", Holder(root))
    caller = _scoped(root)
    assert caller.holder.events.ctx is caller
    seen: list[Any] = []
    caller.holder.listen(seen)
    hooks = root.events.hooks["app/x"]
    assert [hook.ctx for hook in hooks] == [caller]
    await caller.fiber.dispose()
    root.emit("app/x", 1)
    assert seen == []
    assert "app/x" not in root.events.hooks


async def test_withdrawn_service_dropped_from_cache() -> None:
    root = Context()
    root.provide("svc", Svc(root))
    _ = root.svc
    assert "svc" in traced_cache(root)
    root.svc = None
    assert root.svc is None
    assert "svc" not in traced_cache(root)

    plugin = _scoped(root)
    plugin.provide("svc2", Svc(plugin))
    _ = root.svc2
    assert "svc2" in traced_cache(root)
    await plugin.fiber.dispose()
    assert root.svc2 is None
    assert "svc2" not in traced_cache(root)


async def test_callable_only_if_target_is() -> None:
    root = Context()
    root.provide("svc", Svc(root))
    assert not callable(root.svc)
    with pytest.raises(TypeError):
        root.svc()
    assert not callable(with_props(SimpleNamespace(), {}))
    assert not isinstance(root.svc, Iterable) and not isinstance(root.svc, Sized)
    root.provide("raw", Raw(root))
    # Sized first: narrowing to the generic Iterable would leave root.raw partially unknown
    assert isinstance(root.raw, Sized) and isinstance(root.raw, Iterable)


async def test_special_methods_forwarded() -> None:
    root = Context()
    log: list[Any] = []

    class Special:
        __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx", no_shadow=True)

        def __init__(self, ctx: Context) -> None:
            self.ctx = ctx
            self.n = 3

        def __enter__(self) -> Any:
            log.append(("enter", self.ctx))
            return self

        def __exit__(self, *exc: object) -> None:
            log.append("exit")

        async def __aenter__(self) -> int:
            return 1

        async def __aexit__(self, *exc: object) -> None:
            log.append("aexit")

        def __aiter__(self) -> AsyncIterator[int]:
            async def gen() -> AsyncIterator[int]:
                yield self.n

            return gen()

        def __eq__(self, other: object) -> bool:
            return other == self.n

        def __hash__(self) -> int:
            return 42

        def __lt__(self, other: int) -> bool:
            return self.n < other

        def __add__(self, other: int) -> int:
            return self.n + other

        def __radd__(self, other: int) -> int:
            return other + self.n

        def __neg__(self) -> int:
            return -self.n

        def __int__(self) -> int:
            return self.n

        def __setitem__(self, key: str, value: int) -> None:
            log.append((key, value))

        def __bool__(self) -> bool:
            return False

    root.provide("special", Special(root))
    child = root.extend()
    proxy = child.special
    with proxy as entered:
        assert entered is proxy
    assert log[:2] == [("enter", child), "exit"]
    async with proxy as value:
        assert value == 1
    assert log[-1] == "aexit"
    assert [x async for x in proxy] == [3]
    assert proxy == 3 and proxy != 4 and hash(proxy) == 42
    assert proxy < 4 and proxy + 1 == 4 and 1 + proxy == 4 and -proxy == -3
    assert int(proxy) == 3 and not proxy
    proxy["k"] = 1
    assert log[-1] == ("k", 1)
    assert type(proxy) is type(root.special)  # one proxy class per target type
    assert f"{proxy}".startswith("<")  # object.__format__ falls back to str()


async def test_unhashable_target_unhashable_proxy() -> None:
    root = Context()

    class EqOnly:
        __cordis_tracker__: ClassVar[Tracker] = Tracker()

        def __eq__(self, other: object) -> bool:
            return True

    proxy = get_traceable(root, EqOnly())
    with pytest.raises(TypeError):
        hash(proxy)
    assert proxy == 1


async def test_reserved_names_cannot_be_deleted() -> None:
    root = Context()
    svc = Svc(root)
    root.provide("svc", svc)
    proxy = root.extend().svc
    for name in ("ctx", "_cordis_original", "_cordis_caller"):
        with pytest.raises(ServiceAccessError, match=f'cannot delete property "{name}"'):
            delattr(proxy, name)
    assert svc.ctx is root
    shadow = proxy.me()
    with pytest.raises(ServiceAccessError, match='cannot delete property "ctx"'):
        del shadow.ctx
    view = with_props(SimpleNamespace(a=1), {"b": 2})
    del view.b
    del view.a
    assert not hasattr(view, "b") and not hasattr(view, "a")
    overlay_source = SimpleNamespace(c=3)
    overlay = with_props(SimpleNamespace(), overlay_source)
    del overlay.c
    assert not hasattr(overlay_source, "c")


async def test_no_shadow_service_through_receiver() -> None:
    root = Context()

    class Foo:
        __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx", associate="foo")

        def __init__(self, ctx: Context) -> None:
            self.ctx = ctx
            self.only_foo = "foo"

    root.provide("foo", Foo(root))
    root.provide("raw", Raw(root))
    root.mixin("raw", {"me": "foo.me"})
    child = root.extend()
    me: Any = child.foo.me()
    # `self` is the mixin view over the receiver, not a shadow
    assert not isinstance(me, Traceable)
    assert me.items == [1, 2] and me.only_foo == "foo"
    assert parent_of(me.ctx) is child and symbols.shadow not in own_meta(me.ctx)
    # special methods follow the receiver (cordis: a proxy over the receiver)
    with pytest.raises(TypeError):
        len(me)
