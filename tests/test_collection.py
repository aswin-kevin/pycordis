import asyncio
from dataclasses import dataclass

import pytest
from conftest import capture_reports

from pycordis.collection import Collection, CollectionChange
from pycordis.context import Context
from pycordis.errors import CordisError
from pycordis.events import EventKey
from pycordis.fiber import Fiber, FiberState
from pycordis.keys import ServiceKey
from pycordis.registry import plugin_meta


@dataclass(frozen=True)
class Tool:
    label: str


TOOLS = ServiceKey[Collection[Tool]]("tools")


def contributor(name: str, *labels: str) -> object:
    """A plugin named ``name`` adding one tool per label, named after the label."""

    @plugin_meta(name=name, inject=[TOOLS])
    def apply(ctx: Context, config: None) -> None:
        for label in labels:
            ctx.use(TOOLS).add(Tool(label), name=label)

    return apply


async def tools_root() -> tuple[Context, Fiber]:
    root = Context()
    fiber = await root.plugin(Collection.of(TOOLS))
    return root, fiber


async def test_of_provides_the_collection() -> None:
    root, fiber = await tools_root()
    assert fiber.state == FiberState.ACTIVE
    assert fiber.name == "collection:tools"
    tools = root.use(TOOLS)
    assert isinstance(tools, Collection)
    assert len(tools) == 0
    assert list(tools) == []
    await fiber.dispose()
    assert root.get(TOOLS) is None


async def test_items_from_two_plugins_in_insertion_order() -> None:
    root, _ = await tools_root()
    await root.plugin(contributor("search", "web", "docs"))
    await root.plugin(contributor("math", "calc"))
    tools = root.use(TOOLS)
    assert [tool.label for tool in tools] == ["web", "docs", "calc"]
    assert tools.items() == [("web", Tool("web")), ("docs", Tool("docs")), ("calc", Tool("calc"))]
    assert len(tools) == 3
    assert tools.get("docs") == Tool("docs")
    assert tools.get("missing") is None
    assert "calc" in tools
    assert "missing" not in tools


async def test_unload_removes_only_that_plugins_items() -> None:
    root, _ = await tools_root()
    search = await root.plugin(contributor("search", "web", "docs"))
    await root.plugin(contributor("math", "calc"))
    await search.dispose()
    assert root.use(TOOLS).items() == [("calc", Tool("calc"))]


async def test_duplicate_name_names_both_owners() -> None:
    root, _ = await tools_root()
    await root.plugin(contributor("search", "web"))
    rival = root.plugin(contributor("rival", "web"))
    with pytest.raises(ValueError, match="cannot add it from rival"):
        await rival
    assert rival.state == FiberState.FAILED
    with pytest.raises(ValueError) as caught:
        root.use(TOOLS).add(Tool("again"), name="web")
    assert str(caught.value) == (
        'collection "tools" already has "web" (added by search); cannot add it from root'
    )
    # the original entry is untouched
    assert root.use(TOOLS).get("web") == Tool("web")
    assert len(root.use(TOOLS)) == 1


async def test_default_names_use_the_calling_fiber() -> None:
    root, _ = await tools_root()

    @plugin_meta(name="anon", inject=[TOOLS])
    def anon(ctx: Context, config: None) -> None:
        tools = ctx.use(TOOLS)
        tools.add(Tool("a"), name="anon#2")  # an explicit name a default would collide with
        tools.add(Tool("b"))
        tools.add(Tool("c"))

    await root.plugin(anon)
    root.use(TOOLS).add(Tool("d"))
    assert [name for name, _ in root.use(TOOLS).items()] == [
        "anon#2",
        "anon#1",
        "anon#3",
        "root#4",
    ]


async def test_disposer_removes_and_is_idempotent() -> None:
    root, _ = await tools_root()
    changes: list[CollectionChange[Tool]] = []
    root.on(Collection.changes(TOOLS), changes.append)
    dispose = root.use(TOOLS).add(Tool("web"), name="web")
    assert "web" in root.use(TOOLS)
    await dispose()
    assert "web" not in root.use(TOOLS)
    await dispose()
    assert [change.kind for change in changes] == ["add", "remove"]
    # the name is free again
    root.use(TOOLS).add(Tool("web 2"), name="web")
    assert root.use(TOOLS).get("web") == Tool("web 2")


async def test_change_events() -> None:
    root, _ = await tools_root()
    changes: list[CollectionChange[Tool]] = []
    root.on("collection/tools", changes.append)
    search = await root.plugin(contributor("search", "web"))
    await search.dispose()
    assert changes == [
        CollectionChange("add", "web", Tool("web")),
        CollectionChange("remove", "web", Tool("web")),
    ]


async def test_changes_key() -> None:
    key = Collection.changes(TOOLS)
    assert isinstance(key, EventKey)
    assert key.name == "collection/tools"


async def test_iteration_is_over_a_snapshot() -> None:
    root, _ = await tools_root()
    tools = root.use(TOOLS)
    disposers = [tools.add(Tool(label), name=label) for label in ("a", "b", "c")]
    seen: list[str] = []
    for tool in tools:
        seen.append(tool.label)
        for dispose in disposers:
            dispose()
    assert seen == ["a", "b", "c"]
    assert len(tools) == 0


async def test_listener_sees_the_item_registered() -> None:
    root, _ = await tools_root()
    present: list[bool] = []

    def on_change(change: CollectionChange[Tool]) -> None:
        present.append(change.name in root.use(TOOLS))

    root.on(Collection.changes(TOOLS), on_change)
    dispose = root.use(TOOLS).add(Tool("web"), name="web")
    await dispose()
    assert present == [True, False]


async def test_collection_unload_with_contributors() -> None:
    root, fiber = await tools_root()
    search = await root.plugin(contributor("search", "web"))
    await fiber.dispose()
    assert search.state == FiberState.PENDING
    await root.plugin(Collection.of(TOOLS))
    for _ in range(10):
        await asyncio.sleep(0)
    assert search.state == FiberState.ACTIVE
    assert root.use(TOOLS).items() == [("web", Tool("web"))]


async def test_constructed_directly() -> None:
    root = Context()
    tools = Collection(root, TOOLS)
    assert root.use(TOOLS) is not None
    tools.add(Tool("web"), name="web")
    assert root.use(TOOLS).get("web") == Tool("web")
    assert repr(tools) == "Collection('tools', ['web'])"


async def test_change_events_stay_inside_isolated_scopes() -> None:
    root, _ = await tools_root()
    scope = root.isolate("tools")
    await scope.plugin(Collection.of(TOOLS))
    assert root.use(TOOLS) is not scope.use(TOOLS)
    seen_root: list[str] = []
    seen_scope: list[str] = []
    root.on(Collection.changes(TOOLS), lambda change: seen_root.append(change.name))
    scope.on(Collection.changes(TOOLS), lambda change: seen_scope.append(change.name))
    root.use(TOOLS).add(Tool("root"), name="root-tool")
    scope.use(TOOLS).add(Tool("scope"), name="scope-tool")
    assert root.use(TOOLS).items() == [("root-tool", Tool("root"))]
    assert scope.use(TOOLS).items() == [("scope-tool", Tool("scope"))]
    assert seen_root == ["root-tool"]
    assert seen_scope == ["scope-tool"]


async def test_raising_listener_is_reported_not_raised() -> None:
    root, _ = await tools_root()
    errors = capture_reports(root)

    def explode(change: CollectionChange[Tool]) -> None:
        raise RuntimeError(f"boom on {change.kind}")

    after: list[str] = []
    root.on(Collection.changes(TOOLS), explode)
    root.on(Collection.changes(TOOLS), lambda change: after.append(change.kind))
    search = await root.plugin(contributor("search", "web"))
    assert search.state == FiberState.ACTIVE
    assert root.use(TOOLS).get("web") == Tool("web")
    await search.dispose()
    assert "web" not in root.use(TOOLS)
    assert [str(error) for error in errors] == ["boom on add", "boom on remove"]
    # the listeners after the raising one still ran
    assert after == ["add", "remove"]


async def test_add_after_provider_unload_raises() -> None:
    root, fiber = await tools_root()
    stale = root.use(TOOLS)
    await fiber.dispose()
    with pytest.raises(CordisError, match='collection "tools" is no longer active') as caught:
        stale.add(Tool("late"), name="late")
    assert caught.value.code == "INACTIVE_EFFECT"
    assert len(stale) == 0


async def test_provider_unload_drops_items_of_contributors_without_the_injection() -> None:
    root, fiber = await tools_root()
    changes: list[CollectionChange[Tool]] = []
    root.on(Collection.changes(TOOLS), changes.append)
    stale = root.use(TOOLS)
    dispose = stale.add(Tool("web"), name="web")  # the root does not inject TOOLS
    stale.add(Tool("docs"), name="docs")
    changes.clear()
    await fiber.dispose()
    # the closed collection is empty, and says so
    assert (stale.items(), list(stale), len(stale)) == ([], [], 0)
    assert stale.get("web") is None and "web" not in stale
    assert changes == [
        CollectionChange("remove", "web", Tool("web")),
        CollectionChange("remove", "docs", Tool("docs")),
    ]
    # the item's own disposer, run later, is a harmless no-op
    await dispose()
    await dispose()
    assert len(changes) == 2


async def test_add_from_a_disposer_of_an_unloading_fiber_raises() -> None:
    root, _ = await tools_root()
    errors = capture_reports(root)

    @plugin_meta(name="late", inject=[TOOLS])
    def late(ctx: Context, config: None) -> None:
        def add_on_unload() -> None:
            ctx.use(TOOLS).add(Tool("ghost"), name="ghost")

        ctx.effect(lambda: add_on_unload)

    fiber = await root.plugin(late)
    await fiber.dispose()
    assert len(root.use(TOOLS)) == 0
    assert [type(error) for error in errors] == [CordisError]
