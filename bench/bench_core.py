"""Microbenchmarks for pycordis core paths (spec section 8, v2 §3.10).

Run with ``uv run python bench/bench_core.py``. Each benchmark prints its
operation count, elapsed seconds and operations per second. ``--repeat N``
runs each timed section N times and reports the fastest (less noise when
comparing two versions); ``--only TEXT`` runs the benchmarks whose name
contains TEXT.
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import time
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from pycordis import Collection, Context, Service, ServiceKey, current_context, plugin_meta

SERVICE_READS = 1_000_000
EMIT_LISTENERS = 100
EMIT_CALLS = 10_000
PLUGIN_COUNT = 1_000
SCOPE_COUNT = 1_000
COLLECTION_ITEMS = 10_000
CONTEXT_LOOKUPS = 1_000_000
CHAIN_DEPTH = 1_000

ITEMS = ServiceKey[Collection[int]]("bench_items")

type Timed = Callable[[], Awaitable[float]]
"""One run of a benchmark's timed section, returning its elapsed seconds."""


def report(name: str, operations: int, seconds: float) -> None:
    """Print one result line."""
    print(
        f"{name:<52} {operations:>12,} ops  {seconds:7.3f} s  {operations / seconds:>14,.0f} ops/s"
    )


async def best_of(repeat: int, timed: Timed) -> float:
    """The fastest of ``repeat`` runs of ``timed``."""
    return min([await timed() for _ in range(repeat)])


async def bench_service_reads(repeat: int) -> None:
    """1,000,000 traced service attribute reads from a plugin context and from the root."""

    class Counter(Service[Any], provide="counter"):
        value = 1

    root = Context()
    await root.plugin(Counter)
    contexts: dict[str, Any] = {}

    def plugin(ctx: Context, _config: Any) -> None:
        contexts["plugin"] = ctx

    await root.inject(["counter"], plugin)
    for label, ctx in (("plugin context", contexts["plugin"]), ("root context", root)):

        async def timed(ctx: Any = ctx) -> float:
            start = time.perf_counter()
            for _ in range(SERVICE_READS):
                _ = ctx.counter.value
            return time.perf_counter() - start

        report(f"service attribute read, {label}", SERVICE_READS, await best_of(repeat, timed))
    await root.fiber.dispose()


def _emit_timer(root: Context, *args: Any) -> Timed:
    """Time ``EMIT_CALLS`` calls of ``root.emit(*args, i)``."""

    async def timed() -> float:
        start = time.perf_counter()
        for i in range(EMIT_CALLS):
            root.emit(*args, i)
        return time.perf_counter() - start

    return timed


def _report_emit(label: str, seconds: float) -> None:
    report(f"emit to {EMIT_LISTENERS} listeners{label} (calls)", EMIT_CALLS, seconds)
    report(
        f"emit to {EMIT_LISTENERS} listeners{label} (listener calls)",
        EMIT_CALLS * EMIT_LISTENERS,
        seconds,
    )


async def bench_emit(repeat: int) -> None:
    """``emit`` to 100 listeners of one owner, 10,000 times."""
    root = Context()
    hits = 0

    def listener(_value: int) -> None:
        nonlocal hits
        hits += 1

    for _ in range(EMIT_LISTENERS):
        root.on("bench/event", listener)
    seconds = await best_of(repeat, _emit_timer(root, "bench/event"))
    assert hits == EMIT_LISTENERS * EMIT_CALLS * repeat
    _report_emit("", seconds)
    await root.fiber.dispose()


async def bench_emit_many_owners(repeat: int) -> None:
    """``emit`` to 100 listeners, each registered by its own plugin (owner switches)."""
    root = Context()
    hits = 0

    def listener(_value: int) -> None:
        nonlocal hits
        hits += 1

    def owner(ctx: Context, _config: Any) -> None:
        ctx.on("bench/event", listener)

    for index in range(EMIT_LISTENERS):
        await root.plugin(SimpleNamespace(name=f"owner_{index}", apply=owner))
    seconds = await best_of(repeat, _emit_timer(root, "bench/event"))
    assert hits == EMIT_LISTENERS * EMIT_CALLS * repeat
    _report_emit(f", {EMIT_LISTENERS} owners", seconds)
    await root.fiber.dispose()


class _Receiver:
    """An emit receiver whose filter admits every listener's context."""

    def __cordis_filter__(self, _ctx: Context) -> bool:
        return True


async def bench_emit_filtered(repeat: int) -> None:
    """``emit`` with a filtering receiver to 100 listeners, 10,000 times."""
    root = Context()
    hits = 0

    def listener(_value: int) -> None:
        nonlocal hits
        hits += 1

    for _ in range(EMIT_LISTENERS):
        root.on("bench/event", listener)
    seconds = await best_of(repeat, _emit_timer(root, _Receiver(), "bench/event"))
    assert hits == EMIT_LISTENERS * EMIT_CALLS * repeat
    _report_emit(", filtered", seconds)
    await root.fiber.dispose()


async def bench_load_plugins(repeat: int) -> None:
    """Load 1,000 distinct function plugins and wait for all to be active."""

    def make(index: int) -> Callable[[Context, Any], None]:
        def plugin(ctx: Context, _config: Any) -> None:
            ctx.on("bench/never", lambda: index)

        plugin.__name__ = f"plugin_{index}"
        return plugin

    async def timed() -> float:
        root = Context()
        plugins = [make(i) for i in range(PLUGIN_COUNT)]
        start = time.perf_counter()
        fibers = [root.plugin(plugin) for plugin in plugins]
        await asyncio.gather(*(fiber.wait() for fiber in fibers))
        seconds = time.perf_counter() - start
        await root.fiber.dispose()
        return seconds

    report("load plugins", PLUGIN_COUNT, await best_of(repeat, timed))


async def bench_scopes(repeat: int) -> None:
    """Enter and exit 1,000 scopes, one after another."""
    root = Context()

    async def timed() -> float:
        start = time.perf_counter()
        for _ in range(SCOPE_COUNT):
            async with root.scope(name="request"):
                pass
        return time.perf_counter() - start

    report("scope enter + exit", SCOPE_COUNT, await best_of(repeat, timed))
    await root.fiber.dispose()


async def bench_collection(repeat: int) -> None:
    """Add 10,000 named items to a collection, then remove them, with one change listener."""
    root = Context()
    await root.plugin(Collection.of(ITEMS))
    items = root.use(ITEMS)
    changes = 0

    def on_change(_change: object) -> None:
        nonlocal changes
        changes += 1

    root.on(Collection.changes(ITEMS), on_change)

    async def timed() -> float:
        start = time.perf_counter()
        disposers = [items.add(i, name=f"item-{i}") for i in range(COLLECTION_ITEMS)]
        for dispose in disposers:
            await dispose()
        return time.perf_counter() - start

    seconds = await best_of(repeat, timed)
    assert changes == 2 * COLLECTION_ITEMS * repeat
    report("collection add + remove", COLLECTION_ITEMS, seconds)
    await root.fiber.dispose()


async def bench_current_context(repeat: int) -> None:
    """1,000,000 ``current_context()`` calls inside a plugin body."""
    seconds: list[float] = []

    def plugin(ctx: Context, _config: Any) -> None:
        start = time.perf_counter()
        for _ in range(CONTEXT_LOOKUPS):
            current_context()
        seconds.append(time.perf_counter() - start)

    async def timed() -> float:
        root = Context()
        await root.plugin(plugin)
        await root.fiber.dispose()
        return seconds[-1]

    report("current_context() lookup", CONTEXT_LOOKUPS, await best_of(repeat, timed))


def _chain_link(index: int) -> Callable[[Context, Any], None]:
    """Plugin ``link_{index}``: injects the previous link's service, provides its own."""

    @plugin_meta(
        name=f"link_{index}",
        inject=[f"link_{index - 1}"] if index else [],
        provide=f"link_{index}",
    )
    def link(ctx: Context, _config: Any) -> None:
        ctx.provide(f"link_{index}", index)

    return link


async def bench_chain_teardown(repeat: int) -> None:
    """Dispose the head of a 1,000-deep ACTIVE dependency chain (every link unloads)."""

    async def timed() -> float:
        root = Context()
        fibers = [root.plugin(_chain_link(i)) for i in range(CHAIN_DEPTH)]
        while not all(fiber.state.name == "ACTIVE" for fiber in fibers):
            await asyncio.sleep(0)
        start = time.perf_counter()
        await fibers[0].dispose()
        seconds = time.perf_counter() - start
        await root.fiber.dispose()
        return seconds

    report(f"tear down a {CHAIN_DEPTH}-deep chain", CHAIN_DEPTH, await best_of(repeat, timed))


BENCHMARKS: list[tuple[str, Callable[[int], Awaitable[None]]]] = [
    ("service", bench_service_reads),
    ("emit", bench_emit),
    ("emit owners", bench_emit_many_owners),
    ("emit filtered", bench_emit_filtered),
    ("load", bench_load_plugins),
    ("scope", bench_scopes),
    ("collection", bench_collection),
    ("current_context", bench_current_context),
    ("chain", bench_chain_teardown),
]


async def main() -> None:
    """Run the selected benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1, help="report the best of N runs")
    parser.add_argument("--only", default="", help="run benchmarks whose name contains this")
    args = parser.parse_args()
    repeat: int = args.repeat
    only: str = args.only
    print(f"python {platform.python_version()} on {platform.platform()} ({platform.machine()})")
    for name, benchmark in BENCHMARKS:
        if only in name:
            await benchmark(repeat)


if __name__ == "__main__":
    asyncio.run(main())
