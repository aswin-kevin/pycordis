"""Diagnostics: why a fiber is not ACTIVE, dependency cycles (spec §3.7)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import pytest
from conftest import apply_plugin, capture_reports, with_meta

import pycordis.diagnose as diagnose_module
from pycordis import Diagnosis, Reason, render_diagnosis
from pycordis.context import Context
from pycordis.fiber import Fiber, FiberState
from pycordis.keys import ServiceKey
from pycordis.registry import plugin_meta
from pycordis.service import Service
from pycordis.supervise import Backoff, RestartPolicy
from pycordis.testing import MemoryExporter, mock_context
from pycordis.testing import settle as settle_transitions

LLM = ServiceKey[str]("llm")


async def settle(root: Context) -> None:
    """Settle every transition, then one more loop turn: cycle checks run on the next turn."""
    await settle_transitions(root)
    await asyncio.sleep(0)


def plugin(
    name: str,
    *,
    inject: list[str] | None = None,
    provide: str | list[str] | ServiceKey[Any] | None = None,
    apply: Callable[[Context, Any], object] | None = None,
) -> Any:
    """An object plugin named ``name`` declaring ``inject`` / ``provide`` metadata.

    Without ``apply`` the body does nothing; each such plugin gets its own body,
    so that it gets its own registry runtime (and name).
    """

    def noop(_ctx: Context, _config: Any) -> None:
        return None

    body = noop if apply is None else apply
    return plugin_meta(name=name, inject=inject, provide=provide)(apply_plugin(name, body))


def providing(name: str, value: object = True) -> Callable[[Context, Any], None]:
    """A plugin body providing service ``name``."""

    def apply(ctx: Context, _config: Any) -> None:
        ctx.provide(name, value)

    return apply


def raising(error: Exception) -> Callable[[Context, Any], None]:
    def apply(_ctx: Context, _config: Any) -> None:
        raise error

    return apply


def reasons_of(diagnosis: Diagnosis, name: str) -> tuple[Reason, ...]:
    """The reasons of the (only) fiber called ``name`` in the tree."""
    found = [node for node in walk(diagnosis) if node.fiber == name]
    assert len(found) == 1, found
    return found[0].reasons


def walk(diagnosis: Diagnosis) -> list[Diagnosis]:
    nodes = [diagnosis]
    for child in diagnosis.children:
        nodes.extend(walk(child))
    return nodes


def cycle_warnings(logs: MemoryExporter) -> list[str]:
    return [
        str(message.args[0])
        for message in logs.messages
        if message.name == "diagnose" and message.type == "warn"
    ]


# -- the tree --------------------------------------------------------------------------


async def test_healthy_tree_has_no_reasons() -> None:
    async with Context() as root:
        root.plugin(plugin("db", provide="db", apply=providing("db")))
        root.plugin(plugin("app", inject=["db"]))
        await settle(root)
        diagnosis = root.diagnose()
        assert diagnosis == Diagnosis(
            fiber="root",
            uid=0,
            state=FiberState.ACTIVE,
            reasons=(),
            children=(
                Diagnosis("db", 1, FiberState.ACTIVE, (), ()),
                Diagnosis("app", 2, FiberState.ACTIVE, (), ()),
            ),
        )
        assert diagnosis.healthy
        assert diagnosis.unhealthy() is None


async def test_fiber_diagnose_covers_its_subtree() -> None:
    async with Context() as root:

        def outer(ctx: Context, _config: Any) -> None:
            ctx.plugin(plugin("inner", inject=["missing"]))

        fiber = root.plugin(plugin("outer", apply=outer))
        root.plugin(plugin("sibling"))
        await settle(root)
        diagnosis = fiber.diagnose()
        assert diagnosis.fiber == "outer"
        assert [child.fiber for child in diagnosis.children] == ["inner"]
        assert not diagnosis.healthy


async def test_unhealthy_keeps_only_paths_to_stuck_fibers() -> None:
    async with Context() as root:

        def outer(ctx: Context, _config: Any) -> None:
            ctx.plugin(plugin("fine"))
            ctx.plugin(plugin("stuck", inject=["missing"]))

        root.plugin(plugin("outer", apply=outer))
        root.plugin(plugin("other"))
        await settle(root)
        pruned = root.diagnose().unhealthy()
        assert pruned is not None
        assert [node.fiber for node in walk(pruned)] == ["root", "outer", "stuck"]


# -- PENDING: why a service is missing -------------------------------------------------


async def test_missing_service_nobody_provides() -> None:
    async with Context() as root:
        root.plugin(plugin("app", inject=["db"]))
        await settle(root)
        assert reasons_of(root.diagnose(), "app") == (
            Reason("missing-service", "db", 'no plugin declares or currently provides "db"'),
        )


async def test_missing_service_names_a_declared_provider_still_waiting() -> None:
    async with Context() as root:
        root.plugin(plugin("db", inject=["config"], provide="db", apply=providing("db")))
        root.plugin(plugin("app", inject=["db"]))
        await settle(root)
        assert reasons_of(root.diagnose(), "app") == (
            Reason(
                "missing-service", "db", '"db" is declared by db #1 (PENDING), not provided yet'
            ),
        )
        assert reasons_of(root.diagnose(), "db") == (
            Reason(
                "missing-service", "config", 'no plugin declares or currently provides "config"'
            ),
        )


async def test_missing_service_provider_not_active_yet() -> None:
    async with Context() as root:
        release = asyncio.Event()

        async def slow(ctx: Context, _config: Any) -> None:
            ctx.provide("db", 1)
            await release.wait()

        root.plugin(plugin("db", apply=slow))
        root.plugin(plugin("app", inject=["db"]))
        await asyncio.sleep(0.01)
        assert reasons_of(root.diagnose(), "app") == (
            Reason("missing-service", "db", '"db" is provided by db #1, which is LOADING'),
        )
        assert reasons_of(root.diagnose(), "db") == (
            Reason("loading", None, "running the plugin body"),
        )
        release.set()
        await settle(root)
        assert root.diagnose().healthy


async def test_provider_failed_names_its_error() -> None:
    async with Context() as root:
        root.plugin(plugin("db", provide="db", apply=raising(ValueError("bad url"))))
        root.plugin(plugin("app", inject=["db"]))
        await settle(root)
        diagnosis = root.diagnose()
        assert reasons_of(diagnosis, "app") == (
            Reason("provider-failed", "db", '"db" provider db #1 failed: ValueError: bad url'),
        )
        assert reasons_of(diagnosis, "db") == (Reason("failed", None, "ValueError: bad url"),)


async def test_provider_failed_with_a_service_key_declaration() -> None:
    class Client(Service[Any], provide=LLM):
        def __init__(self, ctx: Context, config: Any) -> None:
            super().__init__(ctx)
            raise RuntimeError("no api key")

    async with Context() as root:
        root.plugin(Client)
        root.plugin(plugin("chat", inject=["llm"]))
        await settle(root)
        assert reasons_of(root.diagnose(), "chat") == (
            Reason(
                "provider-failed",
                "llm",
                '"llm" provider Client #1 failed: RuntimeError: no api key',
            ),
        )


async def test_provider_failed_with_a_service_key_attribute() -> None:
    async with Context() as root:
        # metadata assigned directly may hold the key itself
        root.plugin(with_meta(raising(KeyError("model")), name="client", provide=LLM))
        root.plugin(plugin("chat", inject=["llm"]))
        await settle(root)
        assert reasons_of(root.diagnose(), "chat") == (
            Reason(
                "provider-failed", "llm", "\"llm\" provider client #1 failed: KeyError: 'model'"
            ),
        )


async def test_check_failed() -> None:
    def unhealthy(ctx: Context, _config: Any) -> None:
        ctx.provide("db", 1, lambda: False)

    async with Context() as root:
        root.plugin(plugin("db", apply=unhealthy))
        root.plugin(plugin("app", inject=["db"]))
        await settle(root)
        assert reasons_of(root.diagnose(), "app") == (
            Reason("check-failed", "db", '"db" is provided by db #1 but its check() rejected it'),
        )


async def test_dependent_not_yet_notified_of_an_active_provider() -> None:
    # internal/status is emitted before dependents are notified: an observer
    # diagnosing at that moment sees the service provided but not picked up
    async with Context() as root:
        seen: list[tuple[Reason, ...]] = []
        provider = root.plugin(plugin("db", apply=providing("db")))
        root.plugin(plugin("app", inject=["db"]))

        def on_status(fiber: object, _old: FiberState) -> None:
            if fiber is provider and provider.state == FiberState.ACTIVE:
                seen.append(reasons_of(root.diagnose(), "app"))

        root.on("internal/status", on_status)
        await provider.restart()
        await settle(root)
        assert seen == [
            (Reason("missing-service", "db", '"db" is provided by db #1 but not picked up yet'),)
        ]


async def test_check_that_raised_names_its_error() -> None:
    def broken_check() -> bool:
        raise ValueError("probe down")

    def unhealthy(ctx: Context, _config: Any) -> None:
        ctx.provide("db", 1, broken_check)

    async with mock_context() as root:
        root.plugin(plugin("db", apply=unhealthy))
        root.plugin(plugin("app", inject=["db"]))
        await settle(root)
        assert reasons_of(root.diagnose(), "app") == (
            Reason(
                "check-failed",
                "db",
                '"db" is provided by db #1 but its check() raised ValueError: probe down',
            ),
        )


async def test_isolated_provision() -> None:
    async with Context() as root:
        root.isolate("db").plugin(plugin("db", apply=providing("db")))
        root.plugin(plugin("app", inject=["db"]))
        await settle(root)
        assert reasons_of(root.diagnose(), "app") == (
            Reason("isolated", "db", '"db" is provided by db #1 in another isolated scope'),
        )


async def test_isolated_declaration() -> None:
    async with Context() as root:
        root.isolate("db").plugin(plugin("db", inject=["config"], provide="db"))
        root.plugin(plugin("app", inject=["db"]))
        await settle(root)
        assert reasons_of(root.diagnose(), "app") == (
            Reason("isolated", "db", '"db" is only declared by db #1 in another isolated scope'),
        )


async def test_scope_waiting_for_a_service() -> None:
    async with Context() as root:
        with pytest.raises(TimeoutError):
            async with root.scope(name="session-1", inject=["db"], timeout=0.01):
                pass
        task = asyncio.ensure_future(root.scope(name="session-2", inject=["db"]).__aenter__())
        await asyncio.sleep(0.01)
        assert reasons_of(root.diagnose(), "session-2") == (
            Reason("missing-service", "db", 'no plugin declares or currently provides "db"'),
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_scope_provided_names_count_as_declared() -> None:
    async with Context() as root:
        task = asyncio.ensure_future(
            root.scope(name="session", inject=["db"], provide=["memory"]).__aenter__()
        )
        root.plugin(plugin("memory-user", inject=["memory"]))
        await asyncio.sleep(0.01)
        # the scope isolates "memory": a root-level injection never sees it
        assert reasons_of(root.diagnose(), "memory-user") == (
            Reason(
                "isolated",
                "memory",
                '"memory" is only declared by session #2 in another isolated scope',
            ),
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# -- other states ----------------------------------------------------------------------


async def test_failed_with_restart_backoff() -> None:
    policy = RestartPolicy(
        mode="on-failure", max_restarts=3, backoff=Backoff(initial=30.0, jitter=0.0)
    )
    async with mock_context() as root:
        fiber = root.plugin(plugin("worker", apply=raising(RuntimeError("boom"))), policy=policy)
        await settle(root)
        assert fiber.state == FiberState.FAILED
        assert reasons_of(root.diagnose(), "worker") == (
            Reason("failed", None, "RuntimeError: boom"),
            Reason("backoff", None, "restart 1/3 after a 30.00s backoff"),
        )


async def test_failed_after_giving_up_has_no_backoff() -> None:
    policy = RestartPolicy(mode="on-failure", max_restarts=0)
    async with mock_context() as root:
        root.plugin(plugin("worker", apply=raising(RuntimeError("boom"))), policy=policy)
        await settle(root)
        assert reasons_of(root.diagnose(), "worker") == (
            Reason("failed", None, "RuntimeError: boom"),
        )


async def test_loading_names_the_load_timeout() -> None:
    async with Context() as root:

        async def slow(_ctx: Context, _config: Any) -> None:
            await asyncio.sleep(10)

        root.plugin(plugin("slow", apply=slow), load_timeout=5)
        await asyncio.sleep(0)
        assert reasons_of(root.diagnose(), "slow") == (
            Reason("loading", None, "running the plugin body (load_timeout 5s)"),
        )


async def test_unloading_running_disposers() -> None:
    async with Context() as root:
        hold = asyncio.Event()

        def apply(ctx: Context, _config: Any) -> None:
            ctx.effect(lambda: hold.wait, "hold")

        fiber = root.plugin(plugin("closing", apply=apply))
        await fiber
        disposal = fiber.dispose()
        await asyncio.sleep(0)
        # disposed fibers leave the registry: diagnose the fiber itself
        assert fiber.diagnose().reasons == (Reason("unloading", None, "running disposers"),)
        hold.set()
        await disposal


async def test_unloading_waits_for_dependents() -> None:
    async with Context() as root:
        hold = asyncio.Event()

        def dependent(ctx: Context, _config: Any) -> None:
            ctx.effect(lambda: hold.wait, "hold")

        provider = root.plugin(plugin("db", apply=providing("db")))
        root.plugin(plugin("app", inject=["db"], apply=dependent))
        await settle(root)
        restart = asyncio.ensure_future(provider.restart())
        await asyncio.sleep(0)
        assert reasons_of(root.diagnose(), "db") == (
            Reason("unloading", None, "waiting for dependents to unload: app #2"),
        )
        hold.set()
        await restart
        await settle(root)


# -- cycles ----------------------------------------------------------------------------


async def test_two_plugins_injecting_each_other_report_the_cycle_path() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        root.plugin(plugin("a", inject=["llm"], provide="cache"))
        root.plugin(plugin("b", inject=["cache"], provide=LLM))
        await settle(root)
        diagnosis = root.diagnose()
        path = ("a", "llm", "b", "cache", "a")
        assert reasons_of(diagnosis, "a") == (
            Reason("cycle", "llm", 'cycle #1: a -> "llm" -> b -> "cache" -> a', path),
        )
        assert reasons_of(diagnosis, "b") == (
            Reason("cycle", "cache", 'waits for "cache" in cycle #1 (see a #1)', path),
        )
        assert cycle_warnings(logs) == [
            'dependency cycle: a -> "llm" -> b -> "cache" -> a (these plugins wait for each other)'
        ]


async def test_three_plugin_cycle_and_a_bystander() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        root.plugin(plugin("a", inject=["x"], provide="z"))
        root.plugin(plugin("b", inject=["y"], provide="x"))
        root.plugin(plugin("c", inject=["z"], provide="y"))
        root.plugin(plugin("user", inject=["x"]))
        await settle(root)
        diagnosis = root.diagnose()
        path = ("a", "x", "b", "y", "c", "z", "a")
        for name in ("a", "b", "c"):
            (reason,) = reasons_of(diagnosis, name)
            assert (reason.kind, reason.path) == ("cycle", path)
        # waiting on a cycle member is not itself part of the cycle
        assert reasons_of(diagnosis, "user") == (
            Reason("missing-service", "x", '"x" is declared by b #2 (PENDING), not provided yet'),
        )
        assert cycle_warnings(logs) == [
            'dependency cycle: a -> "x" -> b -> "y" -> c -> "z" -> a '
            "(these plugins wait for each other)"
        ]


async def test_self_cycle() -> None:
    async with Context() as root:
        root.plugin(plugin("loop", inject=["me"], provide="me"))
        await settle(root)
        (reason,) = reasons_of(root.diagnose(), "loop")
        assert reason.path == ("loop", "me", "loop")


async def test_two_cycles_sharing_a_fiber_form_one_group() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        root.plugin(plugin("hub", inject=["p", "q"], provide=["h1", "h2"]))
        root.plugin(plugin("p", inject=["h1"], provide="p"))
        root.plugin(plugin("q", inject=["h2"], provide="q"))
        await settle(root)
        diagnosis = root.diagnose()
        # one group of fibers waiting for each other: one cycle, its path shown once
        path = ("hub", "p", "p", "h1", "hub")
        assert reasons_of(diagnosis, "hub") == (
            Reason("cycle", "p", 'cycle #1: hub -> "p" -> p -> "h1" -> hub', path),
            Reason("cycle", "q", 'waits for "q" in cycle #1 (see hub #1)', path),
        )
        assert reasons_of(diagnosis, "q") == (
            Reason("cycle", "h2", 'waits for "h2" in cycle #1 (see hub #1)', path),
        )
        # created in one go: one group formed, one warning
        assert cycle_warnings(logs) == [
            'dependency cycle: hub -> "p" -> p -> "h1" -> hub (these plugins wait for each other)',
        ]


async def test_a_fiber_joining_a_warned_cycle_group_later_is_warned_about() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        root.plugin(plugin("hub", inject=["p", "q"], provide=["h1", "h2"]))
        root.plugin(plugin("p", inject=["h1"], provide="p"))
        await settle(root)
        root.plugin(plugin("q", inject=["h2"], provide="q"))
        await settle(root)
        assert cycle_warnings(logs) == [
            'dependency cycle: hub -> "p" -> p -> "h1" -> hub (these plugins wait for each other)',
            'dependency cycle: hub -> "q" -> q -> "h2" -> hub (these plugins wait for each other)',
        ]


async def test_separate_cycles_are_numbered_by_lowest_uid() -> None:
    async with Context() as root:
        root.plugin(plugin("a", inject=["x"], provide="y"))
        root.plugin(plugin("c", inject=["z"], provide="z"))
        root.plugin(plugin("b", inject=["y"], provide="x"))
        await settle(root)
        diagnosis = root.diagnose()
        assert [r.detail for r in reasons_of(diagnosis, "a")] == [
            'cycle #1: a -> "x" -> b -> "y" -> a'
        ]
        assert [r.detail for r in reasons_of(diagnosis, "c")] == ['cycle #2: c -> "z" -> c']


async def test_cycle_warning_is_logged_once_per_formation() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        a = root.plugin(plugin("a", inject=["llm"], provide="cache"))
        root.plugin(plugin("b", inject=["cache"], provide="llm"))
        await settle(root)
        # unrelated activity does not repeat the warning
        root.plugin(plugin("noise", inject=["nothing"], provide="other"))
        await settle(root)
        assert len(cycle_warnings(logs)) == 1
        # breaking the cycle (a's dependency appears) and forming it again warns again
        llm = root.plugin(plugin("llm", apply=providing("llm")))
        await settle(root)
        assert a.state == FiberState.ACTIVE
        await llm.dispose()
        await settle(root)
        assert a.state == FiberState.PENDING
        assert len(cycle_warnings(logs)) == 2


async def test_cycle_through_a_current_provision() -> None:
    # "b" provides "llm" from its body without declaring it; the provision stays
    # registered while b waits for "cache" (an effect registered while PENDING)
    async with Context() as root:
        root.plugin(plugin("a", inject=["llm"], provide="cache"))
        b = root.plugin(plugin("b", inject=["cache"]))
        b.ctx.provide("llm", 1)
        await settle(root)
        (reason,) = reasons_of(root.diagnose(), "b")
        assert reason.path == ("a", "llm", "b", "cache", "a")


async def test_disposing_a_cycle_member_breaks_it() -> None:
    async with Context() as root:
        root.plugin(plugin("a", inject=["llm"], provide="cache"))
        b = root.plugin(plugin("b", inject=["cache"], provide="llm"))
        await settle(root)
        await b.dispose()
        assert reasons_of(root.diagnose(), "a") == (
            Reason("missing-service", "llm", 'no plugin declares or currently provides "llm"'),
        )


# -- rendering -------------------------------------------------------------------------


async def test_render_diagnosis_snapshot() -> None:
    async with mock_context() as root:

        def group(ctx: Context, _config: Any) -> None:
            ctx.plugin(plugin("a", inject=["llm"], provide="cache"))
            ctx.plugin(plugin("b", inject=["cache"], provide="llm"))

        root.plugin(plugin("db", provide="db", apply=raising(ValueError("bad url"))))
        root.plugin(plugin("agents", apply=group))
        root.plugin(plugin("app", inject=["db", "queue"]))
        await settle(root)
        assert render_diagnosis(root.diagnose()) == "\n".join(
            [
                "root #0 ACTIVE",
                "├─ db #1 FAILED",
                "│     failed: ValueError: bad url",
                "├─ agents #2 ACTIVE",
                "│  ├─ a #4 PENDING",
                '│  │     cycle: cycle #1: a -> "llm" -> b -> "cache" -> a',
                "│  └─ b #5 PENDING",
                '│        cycle: waits for "cache" in cycle #1 (see a #4)',
                "└─ app #3 PENDING",
                '      provider-failed: "db" provider db #1 failed: ValueError: bad url',
                '      missing-service: no plugin declares or currently provides "queue"',
            ]
        )


async def test_render_reasons_of_a_fiber_with_children() -> None:
    async with Context() as root:
        hold = asyncio.Event()

        async def parent(ctx: Context, _config: Any) -> None:
            ctx.plugin(plugin("child"))
            await hold.wait()

        fiber = root.plugin(plugin("parent", apply=parent))
        await asyncio.sleep(0.01)
        assert render_diagnosis(fiber.diagnose()) == "\n".join(
            [
                "parent #1 LOADING",
                "│  loading: running the plugin body",
                "└─ child #2 ACTIVE",
            ]
        )
        assert render_diagnosis(root.diagnose()).splitlines()[:3] == [
            "root #0 ACTIVE",
            "└─ parent #1 LOADING",
            "   │  loading: running the plugin body",
        ]
        hold.set()
        await settle(root)


def test_reason_and_diagnosis_are_frozen() -> None:
    reason = Reason("failed", None, "boom")
    with pytest.raises(AttributeError):
        reason.detail = "other"  # pyright: ignore[reportAttributeAccessIssue]  # negative test: frozen
    assert str(reason) == "failed: boom"


async def test_fiber_type_exports() -> None:
    async with Context() as root:
        fiber = root.plugin(plugin("x"))
        assert isinstance(fiber, Fiber)
        assert isinstance(fiber.diagnose(), Diagnosis)


# -- cycles: only stuck hops count ------------------------------------------------------


async def test_a_loading_alternative_provider_is_no_cycle() -> None:
    # "b" is a fallback provider of llm waiting for a's cache, while "c" (the
    # real llm provider) is still loading: a healthy app in the making
    logs = MemoryExporter()
    release = asyncio.Event()

    async def slow_llm(ctx: Context, _config: Any) -> None:
        await release.wait()
        ctx.provide("llm", 1)

    async with mock_context(logs) as root:
        root.plugin(plugin("c", provide="llm", apply=slow_llm))
        root.plugin(plugin("a", inject=["llm"], provide="cache", apply=providing("cache")))
        root.plugin(plugin("b", inject=["cache"], provide="llm"))
        await asyncio.sleep(0.01)
        assert reasons_of(root.diagnose(), "a") == (
            Reason(
                "missing-service", "llm", '"llm" is declared by c #1 (LOADING), not provided yet'
            ),
        )
        release.set()
        await settle(root)
        assert root.diagnose().healthy
        assert cycle_warnings(logs) == []


async def test_a_wrapper_providing_into_an_isolated_child_is_no_cycle() -> None:
    # "wrapper" injects db and provides a wrapped db to an isolated child scope;
    # its declaration matches its own injection, but the real db is loading
    logs = MemoryExporter()
    release = asyncio.Event()

    async def real_db(ctx: Context, _config: Any) -> None:
        await release.wait()
        ctx.provide("db", "real")

    def wrapper(ctx: Context, _config: Any) -> None:
        ctx.isolate("db").provide("db", "wrapped")

    async with mock_context(logs) as root:
        root.plugin(plugin("db", provide="db", apply=real_db))
        root.plugin(plugin("wrapper", inject=["db"], provide="db", apply=wrapper))
        await asyncio.sleep(0.01)
        assert [r.kind for r in reasons_of(root.diagnose(), "wrapper")] == ["missing-service"]
        release.set()
        await settle(root)
        assert root.diagnose().healthy
        assert cycle_warnings(logs) == []


async def test_a_failed_provider_with_a_restart_pending_is_no_cycle() -> None:
    policy = RestartPolicy(mode="on-failure", backoff=Backoff(initial=30.0, jitter=0.0))
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        root.plugin(plugin("c", provide="llm", apply=raising(RuntimeError("boom"))), policy=policy)
        await settle(root)
        root.plugin(plugin("a", inject=["llm"], provide="cache"))
        root.plugin(plugin("b", inject=["cache"], provide="llm"))
        await settle(root)
        assert cycle_warnings(logs) == []
        assert [r.kind for r in reasons_of(root.diagnose(), "a")] == ["provider-failed"]


# -- cycles: bounded work ----------------------------------------------------------------


async def test_a_long_pending_chain_is_checked_fast_without_recursion() -> None:
    # p_i injects s_{i+1} and provides s_i, created providers first behind a
    # missing "base": every new fiber enters PENDING behind a long PENDING chain
    n = 1000
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        started = time.perf_counter()
        for i in reversed(range(n)):
            inject = [f"s{i + 1}"] if i + 1 < n else ["base"]
            root.plugin(plugin(f"p{i}", inject=inject, provide=f"s{i}", apply=providing(f"s{i}")))
        await settle(root)
        diagnosis = root.diagnose()
        elapsed = time.perf_counter() - started
        assert reasons_of(diagnosis, "p0") == (
            Reason(
                "missing-service", "s1", '"s1" is declared by p1 #999 (PENDING), not provided yet'
            ),
        )
        assert cycle_warnings(logs) == []
        # generous for slow CI runners under coverage; catches a quadratic cycle scan
        assert elapsed < 10.0, elapsed


async def test_a_long_cycle_is_found_without_recursion() -> None:
    n = 1000
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        for i in range(n):
            root.plugin(plugin(f"p{i}", inject=[f"s{(i + 1) % n}"], provide=f"s{i}"))
        await settle(root)
        (reason,) = reasons_of(root.diagnose(), "p0")
        assert reason.kind == "cycle"
        assert len(reason.path) == 2 * n + 1
        assert len(cycle_warnings(logs)) == 1


async def test_mutual_injection_warns_a_bounded_number_of_times() -> None:
    k = 8
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        for i in range(k):
            others = [f"s{j}" for j in range(k) if j != i]
            root.plugin(plugin(f"n{i}", inject=others, provide=f"s{i}"))
        await settle(root)
        # one group formed in one go: one warning, not one per path
        assert cycle_warnings(logs) == [
            'dependency cycle: n0 -> "s1" -> n1 -> "s0" -> n0 (these plugins wait for each other)'
        ]
        diagnosis = root.diagnose()
        details = [r.detail for node in walk(diagnosis) for r in node.reasons]
        assert len(details) == k * (k - 1)
        assert sum(detail.startswith("cycle #1: ") for detail in details) == 1


async def test_an_error_in_the_cycle_hook_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("graph bug")

    monkeypatch.setattr(diagnose_module, "_warn_new_cycles", broken)
    async with Context() as root:
        reports = capture_reports(root)
        root.plugin(plugin("a", inject=["llm"], provide="cache", apply=providing("cache")))
        b = root.plugin(plugin("b", inject=["cache"], provide="llm"))
        await settle(root)
        assert [str(error) for error in reports] == ["graph bug"]
        root.provide("llm", 1)
        await settle(root)
        assert b.state == FiberState.ACTIVE


async def test_cycle_path_is_a_shortest_one_inside_its_group() -> None:
    # a -> (b | c) -> d -> a is a diamond (d is reached twice); a also waits for
    # e, which waits for a missing service and is outside the cycle group
    async with Context() as root:
        root.plugin(plugin("a", inject=["x", "y", "w"], provide="back"))
        root.plugin(plugin("b", inject=["z"], provide="x"))
        root.plugin(plugin("c", inject=["z"], provide="y"))
        root.plugin(plugin("d", inject=["back"], provide="z"))
        root.plugin(plugin("e", inject=["missing"], provide="w"))
        await settle(root)
        path = ("a", "x", "b", "z", "d", "back", "a")
        assert reasons_of(root.diagnose(), "a") == (
            Reason("cycle", "x", 'cycle #1: a -> "x" -> b -> "z" -> d -> "back" -> a', path),
            Reason("cycle", "y", 'waits for "y" in cycle #1 (see a #1)', path),
            Reason("missing-service", "w", '"w" is declared by e #5 (PENDING), not provided yet'),
        )


async def test_no_warning_when_the_fiber_left_pending_before_the_check() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        root.plugin(plugin("a", inject=["llm"], provide="cache"))
        root.provide("llm", 1)  # in the same loop turn: "a" is LOADING by the check
        await settle(root)
        assert cycle_warnings(logs) == []
