"""An agent-shaped workload, for profiling pycordis as an agent runtime would use it.

An application provides an LLM client and a tool collection; 20 tool plugins
contribute tools. Each of 500 requests runs in its own scope with an isolated
conversation memory: it reads services, looks up tools by name, calls them
(the tools reach their services through ``current_context()``), emits progress
events that two observers listen to, and leaves the scope.

Run with ``uv run python bench/bench_agent.py``; profile with
``uv run python -m cProfile -s tottime bench/bench_agent.py``.
"""

from __future__ import annotations

import asyncio
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pycordis import (
    Collection,
    Context,
    EventKey,
    ServiceKey,
    current_context,
    plugin_meta,
    use_context,
)

REQUESTS = 500
TOOL_COUNT = 20
TURNS = 5


@dataclass
class LLM:
    """A fake model: picks tool names round-robin."""

    tools: list[str]
    calls: int = 0

    def choose(self) -> str:
        self.calls += 1
        return self.tools[self.calls % len(self.tools)]


@dataclass
class Memory:
    """A conversation transcript."""

    lines: list[str] = field(default_factory=list[str])


type Tool = Callable[[str], str]

LLM_KEY = ServiceKey[LLM]("llm")
TOOLS = ServiceKey[Collection[Tool]]("tools")
MEMORY = ServiceKey[Memory]("memory")
PROGRESS = EventKey[[str, int], None]("agent/progress")


def tool_plugin(index: int) -> Callable[[Context, None], None]:
    """Plugin ``tool_{index}``: adds a tool that records into the caller's memory."""

    def tool(query: str) -> str:
        memory = current_context().use(MEMORY)
        memory.lines.append(f"tool_{index}: {query}")
        return query.upper()

    @plugin_meta(name=f"tool_{index}", inject=[TOOLS])
    def plugin(ctx: Context, _config: None) -> None:
        ctx.use(TOOLS).add(tool, name=f"tool_{index}")

    return plugin


async def request(root: Context, number: int, progress: list[int]) -> None:
    """One agent request: a scope with its own memory, TURNS tool calls."""
    async with root.scope(name=f"request-{number}", inject=[LLM_KEY, TOOLS], provide=[MEMORY]) as s:
        s.provide(MEMORY, Memory())
        llm = s.use(LLM_KEY)
        tools = s.use(TOOLS)
        with use_context(s):  # the agent loop is not a plugin body: name the request
            for turn in range(TURNS):
                tool = tools.get(llm.choose())
                assert tool is not None
                tool(f"question {number}.{turn}")
                s.emit(PROGRESS, f"request-{number}", turn)
        assert len(s.use(MEMORY).lines) == TURNS
    progress.append(number)


async def main() -> None:
    """Build the application, run the requests, report the elapsed time."""
    print(f"python {platform.python_version()} on {platform.platform()} ({platform.machine()})")
    root = Context()
    await root.plugin(Collection.of(TOOLS))
    for index in range(TOOL_COUNT):
        await root.plugin(tool_plugin(index))
    root.provide(LLM_KEY, LLM([f"tool_{i}" for i in range(TOOL_COUNT)]))
    events: list[int] = []
    for _ in range(2):
        root.on(PROGRESS, lambda _name, turn: events.append(turn))
    progress: list[int] = []
    start = time.perf_counter()
    for number in range(REQUESTS):
        await request(root, number, progress)
    seconds = time.perf_counter() - start
    assert len(progress) == REQUESTS
    assert len(events) == 2 * REQUESTS * TURNS
    print(f"{'agent requests':<52} {REQUESTS:>12,} ops  {seconds:7.3f} s")
    await root.fiber.dispose()


if __name__ == "__main__":
    asyncio.run(main())
