import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from conftest import capture_reports

from pycordis.collection import Collection
from pycordis.context import Context
from pycordis.errors import CordisError
from pycordis.fiber import Fiber, FiberState
from pycordis.keys import ServiceKey
from pycordis.registry import plugin_meta


@dataclass
class Memory:
    owner: str


MEMORY = ServiceKey[Memory]("memory")
DB = ServiceKey[str]("db")


def baseline(root: Context) -> tuple[int, int, int, dict[str, int]]:
    """Everything a scope may touch on the root: registry, root effects, services, hooks."""
    hooks = {name: len(hooks) for name, hooks in root.events.hooks.items() if hooks}
    return (len(root.registry), len(root.fiber.disposables()), len(root.reflect.store), hooks)


async def test_block_sees_injected_services() -> None:
    async with Context() as root:
        root.provide(DB, "sqlite")
        root.provide("cache", {"hits": 0})
        async with root.scope(inject=[DB, "cache"]) as s:
            assert s.fiber.state == FiberState.ACTIVE
            assert s.use(DB) == "sqlite"
            assert s["cache"] == {"hits": 0}


async def test_scope_fiber_is_named_and_inspectable() -> None:
    async with Context() as root:
        async with root.scope(name="session-42") as s:
            assert s.fiber.name == "session-42"
            assert [child.name for child in root.inspect().children] == ["session-42"]
        async with root.scope() as s:
            assert s.fiber.name == "scope"
        assert root.inspect().children == []


async def test_config_reaches_the_scope_fiber() -> None:
    async with Context() as root, root.scope(config={"user": "ada"}) as s:
        assert s.fiber.effective_config == {"user": "ada"}


async def test_provided_names_are_invisible_outside() -> None:
    async with Context() as root:
        async with root.scope(provide=[MEMORY]) as s:
            s.provide(MEMORY, Memory("s"))
            assert s.use(MEMORY) == Memory("s")
            assert root.get(MEMORY) is None
        assert root.get(MEMORY) is None


async def test_concurrent_scopes_providing_the_same_name_do_not_collide() -> None:
    seen: list[str] = []

    @plugin_meta(inject=[MEMORY])
    def agent(ctx: Context, config: None) -> None:
        seen.append(ctx.use(MEMORY).owner)

    async with (
        Context() as root,
        root.scope(provide=MEMORY) as a,
        root.scope(provide="memory") as b,
    ):
        a.provide(MEMORY, Memory("a"))
        b.provide(MEMORY, Memory("b"))
        assert a.use(MEMORY) == Memory("a")
        assert b.use(MEMORY) == Memory("b")
        assert root.get(MEMORY) is None
        await a.plugin(agent)
        await b.plugin(agent)
        assert seen == ["a", "b"]


async def test_parent_provision_of_a_provided_name_is_shadowed() -> None:
    async with Context() as root:
        root.provide(MEMORY, Memory("root"))
        async with root.scope(provide=[MEMORY]) as s:
            assert s.get(MEMORY) is None
            s.provide(MEMORY, Memory("s"))
            assert s.use(MEMORY) == Memory("s")
            assert root.use(MEMORY) == Memory("root")


async def test_inject_and_provide_of_one_name_is_rejected() -> None:
    async with Context() as root:
        with pytest.raises(ValueError, match="cannot both inject and provide memory"):
            root.scope(inject=[MEMORY], provide=[MEMORY])


async def test_exit_disposes_effects_listeners_and_tasks() -> None:
    async with Context() as root:
        before = baseline(root)
        events: list[str] = []
        async with root.scope() as s:
            s.effect(lambda: lambda: events.append("effect disposed"))
            s.on("ping", lambda: events.append("ping"))
            task = s.spawn(asyncio.sleep(3600))
            fiber = s.fiber
            root.emit("ping")
        root.emit("ping")
        assert events == ["ping", "effect disposed"]
        assert task.cancelled()
        assert fiber.state == FiberState.DISPOSED
        assert baseline(root) == before


async def test_collection_items_added_in_a_scope_are_removed_on_exit() -> None:
    tools = ServiceKey[Collection[str]]("tools")
    async with Context() as root:
        await root.plugin(Collection.of(tools))
        async with root.scope(inject=[tools]) as s:
            s.use(tools).add("search", name="search")
            assert list(root.use(tools)) == ["search"]
        assert list(root.use(tools)) == []


async def test_block_exception_propagates_unchanged_after_disposal() -> None:
    async with Context() as root:
        disposed: list[bool] = []
        error = ValueError("boom")
        with pytest.raises(ValueError) as caught:
            async with root.scope() as s:
                s.effect(lambda: lambda: disposed.append(True))
                raise error
        assert caught.value is error
        assert disposed == [True]


async def test_dispose_error_is_reported_without_masking_the_block_error() -> None:
    async with Context() as root:
        reports = capture_reports(root)
        failures = [RuntimeError("first disposer"), RuntimeError("second disposer")]

        def fail() -> None:
            raise failures[len(reports)]

        with pytest.raises(ValueError, match="block"):
            async with root.scope() as s:
                s.effect(lambda: fail)
                raise ValueError("block")
        assert reports == failures[:1]

        async with root.scope() as s:
            s.effect(lambda: fail)
        assert reports == failures


def control_disposal(
    monkeypatch: pytest.MonkeyPatch,
    settle: Callable[[asyncio.Future[None]], object] | None = None,
) -> list[asyncio.Future[None]]:
    """Make ``Fiber.dispose`` start the real disposal but return a future the test settles.

    ``settle``, if given, is called with the future soon after it is returned.

    Stands in for a disposal future that fails or is cancelled outright:
    disposer errors are reported by the unload itself and never reach it.
    """
    futures: list[asyncio.Future[None]] = []
    dispose = Fiber.dispose

    def controlled(fiber: Fiber) -> asyncio.Future[None]:
        dispose(fiber)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        futures.append(future)
        if settle is not None:
            loop.call_soon(settle, future)
        return future

    monkeypatch.setattr(Fiber, "dispose", controlled)
    return futures


async def test_a_failing_disposal_future_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    async with Context() as root:
        reports = capture_reports(root)
        failure = RuntimeError("dispose failed")
        control_disposal(monkeypatch, lambda future: future.set_exception(failure))
        async with root.scope() as s:
            fiber = s.fiber
        monkeypatch.undo()
        assert reports == [failure]
        await fiber.wait()
        assert fiber.state == FiberState.DISPOSED


async def test_a_cancelled_disposal_future_ends_the_scope_quietly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Context() as root:
        reports = capture_reports(root)
        control_disposal(monkeypatch, asyncio.Future[None].cancel)
        async with root.scope():
            pass
        monkeypatch.undo()
        assert reports == []


async def abandon_disposal(root: Context, monkeypatch: pytest.MonkeyPatch) -> asyncio.Future[None]:
    """Cancel a scope's block, then cancel again while it waits for disposal."""
    futures = control_disposal(monkeypatch)
    entered = asyncio.Event()

    async def session() -> None:
        async with root.scope():
            entered.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(session())
    await entered.wait()
    task.cancel()
    while not futures:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.undo()
    return futures[0]


async def test_an_abandoned_disposal_still_reports_its_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Context() as root:
        reports = capture_reports(root)
        disposal = await abandon_disposal(root, monkeypatch)
        failure = RuntimeError("late failure")
        disposal.set_exception(failure)
        await asyncio.sleep(0)
        assert reports == [failure]


async def test_an_abandoned_disposal_that_is_cancelled_reports_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Context() as root:
        reports = capture_reports(root)
        disposal = await abandon_disposal(root, monkeypatch)
        disposal.cancel()
        await asyncio.sleep(0)
        assert reports == []
        disposal = await abandon_disposal(root, monkeypatch)
        disposal.set_result(None)
        await asyncio.sleep(0)
        assert reports == []


async def test_cancelled_block_disposes_then_propagates() -> None:
    async with Context() as root:
        before = baseline(root)
        entered = asyncio.Event()
        cleanup: list[str] = []

        async def slow_cleanup() -> None:
            await asyncio.sleep(0.01)
            cleanup.append("done")

        async def session() -> None:
            async with root.scope() as s:
                s.effect(lambda: slow_cleanup)
                entered.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(session())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # the cancelled task waited for the whole unload before re-raising
        assert cleanup == ["done"]
        assert baseline(root) == before


async def test_second_cancellation_abandons_waiting_but_disposal_completes() -> None:
    async with Context() as root:
        before = baseline(root)
        entered = asyncio.Event()
        disposing = asyncio.Event()
        release = asyncio.Event()

        async def blocked_cleanup() -> None:
            disposing.set()
            await release.wait()

        async def session() -> None:
            async with root.scope() as s:
                s.effect(lambda: blocked_cleanup)
                entered.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(session())
        await entered.wait()
        task.cancel()
        await disposing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert baseline(root) != before  # the unload is still running
        release.set()
        for _ in range(10):
            await asyncio.sleep(0)
        assert baseline(root) == before


async def test_cancelled_while_injections_pending() -> None:
    async with Context() as root:
        before = baseline(root)
        started = asyncio.Event()

        async def session() -> None:
            scope = root.scope(name="pending", inject=["missing"])
            started.set()
            async with scope:
                pytest.fail("the scope must not become active")

        task = asyncio.create_task(session())
        await started.wait()
        for _ in range(3):
            await asyncio.sleep(0)
        assert [child.state for child in root.inspect().children] == [FiberState.PENDING]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert baseline(root) == before
        assert root.inspect().children == []


async def test_timeout_waiting_for_missing_injection() -> None:
    async with Context() as root:
        before = baseline(root)
        with pytest.raises(
            TimeoutError, match=r"'chat' was not active after 0.01s \(waiting for missing, db\)$"
        ):
            async with root.scope(name="chat", inject=["missing", DB], timeout=0.01):
                pytest.fail("the scope must not become active")
        assert baseline(root) == before
        root.provide(DB, "db")
        with pytest.raises(TimeoutError, match=r"after 0.01s \(waiting for missing\)$"):
            async with root.scope(inject=["missing", DB], timeout=0.01):
                pytest.fail("the scope must not become active")


async def test_timeout_without_missing_injection() -> None:
    async with Context() as root:
        root.provide("flaky", 1, check=lambda: False)
        with pytest.raises(TimeoutError, match=r"'scope' was not active after 0.01s$"):
            async with root.scope(inject=["flaky"], timeout=0.01):
                pytest.fail("the scope must not become active")


async def test_scope_activates_once_injection_arrives() -> None:
    async with Context() as root:

        async def provide_later() -> None:
            for _ in range(3):
                await asyncio.sleep(0)
            root.provide(DB, "late")

        provider = asyncio.create_task(provide_later())
        async with root.scope(inject=[DB], timeout=1.0) as s:
            assert s.use(DB) == "late"
        await provider


def fail_config(marker: str) -> Callable[[Any, Callable[[], Any]], Any]:
    """An ``internal/config`` listener failing the load of a fiber whose config is ``marker``."""

    def on_config(fiber: Any, next: Callable[[], Any]) -> Any:
        config = next()
        if config == marker:
            raise ValueError(f"bad config {marker}")
        return config

    return on_config


async def test_startup_failure_is_raised_and_disposed() -> None:
    async with Context() as root:
        capture_reports(root)
        root.on("internal/config", fail_config("boom"))
        before = baseline(root)
        with pytest.raises(ValueError, match="bad config boom"):
            async with root.scope(config="boom"):
                pytest.fail("the scope must not become active")
        assert baseline(root) == before


async def test_startup_failure_after_pending_injection() -> None:
    async with Context() as root:
        capture_reports(root)
        root.on("internal/config", fail_config("boom"))

        async def provide_later() -> None:
            for _ in range(3):
                await asyncio.sleep(0)
            root.provide(DB, "late")

        provider = asyncio.create_task(provide_later())
        with pytest.raises(ValueError, match="bad config boom"):
            async with root.scope(inject=[DB], config="boom", timeout=1.0):
                pytest.fail("the scope must not become active")
        await provider


async def test_parent_disposed_while_waiting() -> None:
    async with Context() as root:
        outcome: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()

        async def run_scope(ctx: Context) -> None:
            try:
                async with ctx.scope(name="inner", inject=["missing"]):
                    pytest.fail("the scope must not become active")
            except CordisError as error:
                outcome.set_result(error)

        def host(ctx: Context, config: None) -> None:
            # a root-owned task outlives the host fiber's unload
            root.spawn(run_scope(ctx))

        fiber: Fiber = await root.plugin(host)
        for _ in range(3):
            await asyncio.sleep(0)
        await fiber.dispose()
        error = await outcome
        assert str(error) == "scope <inner> was disposed before it became active"


async def test_scope_can_be_entered_once() -> None:
    async with Context() as root:
        scope = root.scope(name="once")
        async with scope:
            pass
        with pytest.raises(RuntimeError, match="'once' can only be entered once"):
            async with scope:
                pytest.fail("unreachable")


async def test_sequential_scopes_do_not_leak() -> None:
    tools = ServiceKey[Collection[str]]("tools")
    async with Context() as root:
        await root.plugin(Collection.of(tools))
        root.provide(DB, "db")
        before = baseline(root)
        for index in range(1000):
            async with root.scope(
                name=f"request-{index}", inject=[DB, tools], provide=[MEMORY]
            ) as s:
                s.provide(MEMORY, Memory(str(index)))
                s.use(tools).add(f"tool-{index}")
                s.on("ping", lambda: None)
        assert baseline(root) == before
        assert len(root.use(tools)) == 0


async def test_a_startup_timeout_error_is_not_rewritten_as_a_scope_timeout() -> None:
    async with Context() as root:
        capture_reports(root)
        failure = TimeoutError("upstream timed out")

        def on_config(fiber: Any, next: Callable[[], Any]) -> Any:
            config = next()
            if config == "slow":
                raise failure
            return config

        root.on("internal/config", on_config)
        for timeout in (None, 1.0):
            with pytest.raises(TimeoutError) as caught:
                async with root.scope(config="slow", timeout=timeout):
                    pytest.fail("the scope must not become active")
            assert caught.value is failure


async def test_timeout_message_formats_the_deadline() -> None:
    async with Context() as root:
        with pytest.raises(
            TimeoutError, match=r"^scope 'scope' was not active after 0.5s \(waiting for missing\)$"
        ):
            async with root.scope(inject=["missing"], timeout=0.5):
                pytest.fail("the scope must not become active")


async def test_timeout_zero_always_fails() -> None:
    async with Context() as root:
        with pytest.raises(TimeoutError, match=r"after 0s$"):
            async with root.scope(timeout=0):
                pytest.fail("the scope must not become active")


async def test_parent_unload_mid_block_leaves_a_disposed_scope_context() -> None:
    async with Context() as root:
        entered = asyncio.Event()
        resume = asyncio.Event()
        outcome: asyncio.Future[tuple[FiberState, CordisError]] = (
            asyncio.get_running_loop().create_future()
        )

        async def session(ctx: Context) -> None:
            async with ctx.scope(name="inner") as s:
                entered.set()
                await resume.wait()
                # the block keeps running: its task is owned by the root
                with pytest.raises(CordisError) as caught:
                    s.effect(lambda: None)
                outcome.set_result((s.fiber.state, caught.value))

        def host(ctx: Context, config: None) -> None:
            root.spawn(session(ctx))

        fiber: Fiber = await root.plugin(host)
        await entered.wait()
        await fiber.dispose()
        resume.set()
        state, error = await outcome
        assert state == FiberState.DISPOSED
        assert error.code == "INACTIVE_EFFECT"


async def test_parent_unload_cancels_a_block_owned_by_the_parent() -> None:
    async with Context() as root:
        entered = asyncio.Event()
        cleanup: list[str] = []

        async def session(ctx: Context) -> None:
            async with ctx.scope() as s:
                s.effect(lambda: lambda: cleanup.append("scope"))
                entered.set()
                await asyncio.sleep(3600)

        tasks: list[asyncio.Task[None]] = []

        def host(ctx: Context, config: None) -> None:
            tasks.append(ctx.spawn(session(ctx)))

        fiber: Fiber = await root.plugin(host)
        await entered.wait()
        await fiber.dispose()
        assert tasks[0].cancelled()
        assert cleanup == ["scope"]


def test_scope_is_the_return_type_of_context_scope() -> None:
    from pycordis import Scope

    assert inspect.signature(Context.scope).return_annotation in (Scope, "Scope")
