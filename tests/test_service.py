import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from conftest import apply_plugin

from pycordis._utils import symbols
from pycordis.context import Context
from pycordis.errors import ServiceAccessError
from pycordis.fiber import FiberState
from pycordis.logger import Logger, Message
from pycordis.registry import inject
from pycordis.service import Service
from pycordis.trace import Tracker

# -- README quick start ---------------------------------------------------------


async def test_readme_quick_start() -> None:
    class Counter(Service[Any]):
        value = 0

        def __init__(self, ctx: Context, _config: Any = None) -> None:
            super().__init__(ctx, "counter")

        def next(self) -> int:
            self.value += 1
            return self.value

    def greeter(ctx: Context, _config: Any) -> None:
        def on_ready(message: str) -> None:
            ctx.logger.info("%s #%d", message, ctx.counter.next())

        ctx.on("app/ready", on_ready)

    root = Context()
    lines: list[str] = []

    class Lines:
        def export(self, message: Message) -> None:
            lines.append(Logger.format(self, message))

    root.logger.exporter(Lines())
    await root.plugin(Counter)
    await root.inject(["counter"], greeter)

    root.emit("app/ready", "started")
    root.emit("app/ready", "started")
    assert lines == ["started #1", "started #2"]
    assert [m.name for m in root.logger.buffer] == ["greeter", "greeter"]
    assert root.counter.value == 2
    await root.fiber.dispose()
    root.emit("app/ready", "started")
    assert lines == ["started #1", "started #2"]


# -- construction ---------------------------------------------------------------


async def test_provide_class_keyword() -> None:
    class Counter(Service[Any], provide="counter"):
        def next(self) -> int:
            return 1

    assert Counter.provide == "counter"
    root = Context()
    fiber = await root.plugin(Counter)
    assert fiber.state == FiberState.ACTIVE
    assert root.counter.next() == 1
    service = root.counter
    assert isinstance(service, Counter)
    assert isinstance(service, Service)
    assert service.name == "counter"
    assert service.__cordis_tracker__ == Tracker(property="ctx", associate="counter")
    # reading the tracker property through the proxy yields the use site
    assert service.ctx is root
    await fiber.dispose()
    assert root.counter is None


async def test_explicit_name_and_non_string_config() -> None:
    class Svc(Service[Any], provide="default"):
        pass

    root = Context()
    Svc(root, "explicit")
    assert isinstance(root.explicit, Svc)
    # the registry constructs class plugins as cls(ctx, config): a config is not a name
    await root.plugin(Svc, {"some": "config"})
    assert isinstance(root.default, Svc)


async def test_missing_name_raises() -> None:
    class Anonymous(Service[Any]):
        pass

    with pytest.raises(TypeError, match="service name"):
        Anonymous(Context())


async def test_provide_inherited_by_subclass() -> None:
    class Base(Service[Any], provide="base"):
        pass

    class Child(Base):
        pass

    assert Child.provide == "base"
    root = Context()
    Child(root)
    assert isinstance(root.base, Child)


async def test_inject_method_hooks_survive_service_init() -> None:
    log: list[Any] = []

    class Svc(Service[Any], provide="svc"):
        def __init__(self, ctx: Context, _config: Any = None) -> None:
            # touching the hook list before Service.__init__ must not lose hooks
            hooks: list[Callable[[], None]] = getattr(self, symbols.init_hooks)
            hooks.append(lambda: log.append("extra"))
            super().__init__(ctx)

        @inject("db")
        def with_db(self) -> None:
            log.append(("db", self.ctx.db))

    root = Context()
    root.provide("db", 42)
    await root.plugin(Svc)
    for _ in range(10):
        await asyncio.sleep(0)
    assert log == ["extra", ("db", 42)]


# -- callable services ----------------------------------------------------------


async def test_non_invokable_service_not_callable() -> None:
    class Plain(Service[Any], provide="plain"):
        pass

    root = Context()
    plain = Plain(root)
    assert not callable(plain)
    assert not callable(root.plain)
    with pytest.raises(TypeError):
        root.plain()


async def test_invokable_service_callable() -> None:
    seen: list[Any] = []

    class Greeter(Service[Any], provide="greet"):
        def __cordis_invoke__(self, who: str) -> str:
            seen.append(self.ctx)
            return f"hello {who}"

    root = Context()
    greet = Greeter(root)
    child = root.extend()
    assert callable(greet)
    assert callable(child.greet)
    assert child.greet("a") == "hello a"
    # direct call traces to the service's own context
    assert greet("b") == "hello b"
    # inside invoke, self.ctx is (a shadow of) the use site
    assert seen[0] is not None
    assert seen[0]._parent is child
    assert seen[1]._parent is root


async def test_explicit_call_not_overridden() -> None:
    class Custom(Service[Any], provide="custom"):
        def __cordis_invoke__(self) -> str:
            return "invoke"

        def __call__(self) -> str:
            return "call"

    root = Context()
    assert Custom(root)() == "call"
    # the proxy dispatches through __cordis_invoke__ (cordis apply trap)
    assert root.custom() == "invoke"


# -- check / filter / extend -----------------------------------------------------


async def test_check_false_keeps_dependents_pending() -> None:
    allowed = False

    class Guarded(Service[Any], provide="guarded"):
        def __cordis_check__(self) -> bool:
            return allowed

    def dependent(_ctx: Context, _config: Any) -> None:
        return None

    root = Context()
    dep = root.plugin(SimpleNamespace(name="D", inject=["guarded"], apply=dependent))
    await root.plugin(Guarded)
    await dep
    assert dep.state == FiberState.PENDING
    allowed = True
    root.reflect.notify(["guarded"])
    await dep
    assert dep.state == FiberState.ACTIVE


async def test_check_sees_traced_self() -> None:
    seen: list[Any] = []

    class Guarded(Service[Any], provide="guarded"):
        def __cordis_check__(self) -> bool:
            seen.append(self.ctx)
            return True

    root = Context()
    dep = root.plugin(apply_plugin("D", lambda c, _: None, inject=["guarded"]))
    await root.plugin(Guarded)
    await dep
    assert dep.state == FiberState.ACTIVE
    assert seen and seen[0] is dep.ctx


async def test_filter_compares_isolation() -> None:
    class Svc(Service[Any], provide="svc"):
        pass

    root = Context()
    svc = Svc(root)
    assert svc.__cordis_filter__(root)
    assert svc.__cordis_filter__(root.extend())
    assert not svc.__cordis_filter__(root.isolate("svc"))
    assert svc.__cordis_filter__(root.isolate("other"))


async def test_filter_scopes_emit_with_this_arg() -> None:
    class Svc(Service[Any], provide="svc"):
        pass

    root = Context()
    svc = Svc(root)
    calls: list[str] = []
    root.on("evt", lambda: calls.append("root"))
    root.isolate("svc").on("evt", lambda: calls.append("isolated"))
    root.emit(svc, "evt")
    assert calls == ["root"]


async def test_extend_copies_and_sets_props() -> None:
    class Svc(Service[Any], provide="svc"):
        prefix = "a"

        def show(self) -> str:
            return self.prefix

    root = Context()
    svc = Svc(root)
    ext = svc.__cordis_extend__(prefix="b")
    assert ext is not svc
    assert isinstance(ext, Svc)
    assert ext.show() == "b"
    assert svc.show() == "a"
    assert ext.name == "svc"
    # through a proxy, the original is copied (not the proxy)
    ext2 = root.svc.__cordis_extend__(prefix="c")
    assert type(ext2) is Svc
    assert ext2.prefix == "c"


async def test_extend_keeps_callable() -> None:
    class Fn(Service[Any], provide="fn"):
        tag = "x"

        def __cordis_invoke__(self) -> str:
            return self.tag

    root = Context()
    ext = Fn(root).__cordis_extend__(tag="y")
    assert callable(ext)
    assert ext() == "y"


# -- resolve_config --------------------------------------------------------------


async def test_resolve_config_merges_root_first() -> None:
    class Svc(Service[Any], provide="svc"):
        pass

    root = Context()
    Svc(root)
    assert root.svc.__cordis_resolve_config__() == {}
    child = root.intercept("svc", {"a": 1, "b": 1}).intercept("other", {"z": 9})
    grandchild = child.intercept("svc", {"b": 2})
    resolved = grandchild.svc.__cordis_resolve_config__({"base": 0, "a": 0}, {"head": 3})
    assert resolved == {"base": 0, "a": 1, "b": 2, "head": 3}
    assert list(resolved) == ["base", "a", "b", "head"]
    assert child.svc.__cordis_resolve_config__(head={"b": 5}) == {"a": 1, "b": 5}


async def test_resolve_config_uses_config_merge() -> None:
    calls: list[tuple[Any, ...]] = []

    class MergingConfig:
        @staticmethod
        def merge(*configs: Any) -> Any:
            calls.append(configs)
            return "merged"

    class Svc(Service[Any], provide="svc"):
        Config: ClassVar[Any] = MergingConfig

    root = Context()
    Svc(root)
    ctx = root.intercept("svc", {"a": 1})
    assert ctx.svc.__cordis_resolve_config__(None, {"h": 1}) == "merged"
    assert calls == [({"a": 1}, {"h": 1})]


async def test_resolve_config_accepts_objects() -> None:
    class Svc(Service[Any], provide="svc"):
        pass

    root = Context()
    Svc(root)
    ctx = root.intercept("svc", SimpleNamespace(a=1))
    assert ctx.svc.__cordis_resolve_config__() == {"a": 1}


# -- Review Focus 5: dependency teardown edge cases --------------------------------


async def test_dependent_reads_service_in_own_disposer_during_provider_removal() -> None:
    seen: list[Any] = []

    class X(Service[Any], provide="x"):
        def get(self) -> str:
            return "value"

    def dependent(ctx: Context, _config: Any) -> Callable[[], None]:
        return lambda: seen.append(ctx.x.get())

    root = Context()
    provider = await root.plugin(X)
    dep = await root.plugin(SimpleNamespace(name="D", inject=["x"], apply=dependent))
    assert dep.state == FiberState.ACTIVE
    await provider.dispose()
    assert seen == ["value"]
    assert dep.state == FiberState.PENDING
    with pytest.raises(ServiceAccessError) as info:
        _ = dep.ctx.x
    assert str(info.value) == 'cannot get required service "x" in inactive context'


async def test_dependent_async_disposer_reads_service() -> None:
    seen: list[Any] = []

    class X(Service[Any], provide="x"):
        pass

    def dependent(ctx: Context, _config: Any) -> Callable[[], Any]:
        async def dispose() -> None:
            seen.append(ctx.x.name)

        return dispose

    root = Context()
    provider = await root.plugin(X)
    await root.plugin(SimpleNamespace(name="D", inject=["x"], apply=dependent))
    await provider.dispose()
    assert seen == ["x"]


async def test_a_non_callable_config_merge_is_a_type_error() -> None:
    class Broken(Service[dict[str, int]], provide="broken"):
        class Config:
            merge = 5

    root = Context()
    service = Broken(root)
    with pytest.raises(TypeError, match=r"Broken\.Config\.merge is not callable"):
        service.__cordis_resolve_config__()
