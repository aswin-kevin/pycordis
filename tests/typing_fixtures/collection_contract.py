"""Typing contract: collections are typed by their service key.

Checked by tests/test_typing_contracts.py; lines ending in ``# expect-error``
(optionally ``: ruleName``) must be the only lines pyright reports.
"""

from typing import assert_type

from pycordis import Collection, CollectionChange, Context, ServiceKey


class Tool:
    pass


TOOLS = ServiceKey[Collection[Tool]]("tools")


async def provides(ctx: Context) -> None:
    await ctx.plugin(Collection.of(TOOLS))
    Collection(ctx, TOOLS)


def reads(ctx: Context) -> None:
    tools = ctx.use(TOOLS)
    assert_type(tools.get("web"), Tool | None)
    assert_type(tools.items(), list[tuple[str, Tool]])
    for tool in tools:
        assert_type(tool, Tool)
    tools.add(Tool(), name="web")
    tools.add("not a tool")  # expect-error: reportArgumentType


def listens(ctx: Context) -> None:
    def on_tool(change: CollectionChange[Tool]) -> None:
        assert_type(change.item, Tool)

    def on_text(change: CollectionChange[str]) -> None:
        pass

    ctx.on(Collection.changes(TOOLS), on_tool)
    ctx.on(Collection.changes(TOOLS), on_text)  # expect-error
