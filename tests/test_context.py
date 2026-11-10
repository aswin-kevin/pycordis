import pytest
from conftest import intercept_map, isolate_map

from pycordis._context_internals import own_meta
from pycordis._utils import Label, ScopeMap, symbols
from pycordis.context import Context
from pycordis.fiber import Fiber, FiberState


async def test_extend_does_not_mutate_parent() -> None:
    root = Context()
    child = root.extend(foo=1)
    assert child.foo == 1 and child.root is root
    with pytest.raises(AttributeError):
        _ = root.foo


async def test_extend_chain_and_shadowing() -> None:
    root = Context()
    a = root.extend(x=1, y=2)
    b = a.extend(y=3)
    assert (b.x, b.y, a.y) == (1, 3, 2)
    assert b.fiber is root.fiber and b.root is root
    b.z = 4
    b._private = 5
    assert b.z == 4 and b._private == 5
    with pytest.raises(AttributeError):
        _ = a.z


async def test_extend_preserves_subclass() -> None:
    class AppContext(Context):
        pass

    root = AppContext()
    child = root.extend()
    assert type(child) is AppContext and child.root is root


async def test_isolate_and_intercept_scopes() -> None:
    root = Context()
    a = root.isolate("db")
    b = root.isolate("db")
    assert isolate_map(a)["db"] is not isolate_map(b)["db"]
    s1, s2 = root.isolate("db", "shared"), root.isolate("db", "shared")
    assert isolate_map(s1)["db"] is isolate_map(s2)["db"]
    i = root.intercept("logger", {"level": 3}).intercept("logger", {"name": "x"})
    parent = intercept_map(i).parent
    assert parent is not None
    assert intercept_map(i)["logger"] == {"name": "x"} and parent["logger"] == {"level": 3}
    assert "db" not in isolate_map(root)


async def test_isolate_label_forms() -> None:
    root = Context()
    label = Label("custom")
    assert isolate_map(root.isolate("db", label))["db"] is label
    fresh = isolate_map(root.isolate("db"))["db"]
    assert isinstance(fresh, Label) and fresh.name == "db"
    # str labels are interned per root
    other = Context()
    assert (
        isolate_map(root.isolate("db", "s"))["db"]
        is not isolate_map(other.isolate("db", "s"))["db"]
    )
    nested = root.extend().isolate("db", "s")
    assert isolate_map(nested)["db"] is isolate_map(root.isolate("cache", "s"))["cache"]


async def test_class_constants() -> None:
    assert Context.effect == symbols.effect == "__cordis_effect__"
    assert Context.filter == symbols.filter
    assert Context.isolate == symbols.isolate
    assert Context.intercept == symbols.intercept
    root = Context()
    assert isinstance(getattr(root, Context.isolate), ScopeMap)
    # the symbol must alias the stored scope map itself, so read the storage directly
    assert getattr(root, Context.intercept) is own_meta(root)[symbols.intercept]
    with pytest.raises(AttributeError):
        _ = root.filter

    def f(*_: object) -> bool:
        return True

    assert getattr(root.extend(**{Context.filter: f}), Context.filter) is f


def test_context_requires_running_loop() -> None:
    with pytest.raises(RuntimeError, match="running asyncio event loop"):
        Context()


async def test_context_is() -> None:
    assert Context.is_(Context()) and not Context.is_(object())
    assert Context.is_(Context().extend()) and not Context.is_(None)


async def test_root_fiber_and_repr() -> None:
    root = Context()
    assert isinstance(root.fiber, Fiber)
    assert root.fiber.ctx is root and root.fiber.uid == 0
    assert root.fiber.state == FiberState.ACTIVE
    assert root.fiber.name == "root"
    assert root.fiber.get_effects() == []
    assert repr(root) == "Context <root>"


async def test_async_with_disposes_root() -> None:
    seen: list[int] = []
    async with Context() as root:
        root.fiber.effect(lambda: lambda: seen.append(1))
    assert seen == [1]
    assert root.fiber.state == FiberState.ACTIVE


async def test_uninitialised_context_attribute_error() -> None:
    ctx = object.__new__(Context)
    with pytest.raises(AttributeError):
        _ = ctx.anything
