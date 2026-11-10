"""Supervision: restart policies, load and dispose timeouts (spec §3.6)."""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from conftest import capture_reports, with_meta

from pycordis import LoadTimeoutError
from pycordis.context import Context
from pycordis.fiber import Fiber, FiberState
from pycordis.registry import plugin_meta
from pycordis.supervise import Backoff, RestartPolicy
from pycordis.testing import MemoryExporter, mock_context

FAST = Backoff(initial=0.01, factor=1.0, max=0.01, jitter=0.0)
"""A fixed 10 ms backoff without jitter: restart messages are predictable."""


def on_failure(**options: Any) -> RestartPolicy:
    return RestartPolicy(mode="on-failure", backoff=options.pop("backoff", FAST), **options)


async def until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    """Poll ``predicate`` until it holds; fail the test after ``timeout`` seconds."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.002)


def supervisor_lines(logs: MemoryExporter, level: str) -> list[str]:
    return [
        str(message.args[0])
        for message in logs.messages
        if message.name == "supervisor" and message.type == level
    ]


def other_tasks() -> list[asyncio.Task[Any]]:
    return [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]


def backoff_tasks() -> list[asyncio.Task[Any]]:
    return [task for task in asyncio.all_tasks() if task.get_name().endswith(":backoff")]


def failing_until(outcomes: list[bool], name: str = "flaky") -> Callable[[Context, Any], None]:
    """A plugin that pops one outcome per load and raises ``RuntimeError("boom")`` on False."""

    def plugin(_ctx: Context, _config: Any) -> None:
        if not outcomes.pop(0):
            raise RuntimeError("boom")

    return with_meta(plugin, name=name)


# -- Backoff / RestartPolicy values ----------------------------------------------------


def test_defaults_match_v1_behaviour() -> None:
    policy = RestartPolicy()
    assert policy.mode == "never"
    assert (policy.max_restarts, policy.window) == (5, 60.0)
    assert policy.backoff == Backoff(initial=0.5, factor=2.0, max=30.0, jitter=0.1)


def test_backoff_grows_by_factor_and_caps_at_max() -> None:
    backoff = Backoff(initial=1.0, factor=2.0, max=5.0, jitter=0.0)
    rng = random.Random(0)
    assert [backoff.delay(n, rng) for n in range(1, 6)] == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_backoff_never_overflows_for_large_attempts() -> None:
    backoff = Backoff(initial=1.0, factor=1e300, max=7.0, jitter=0.0)
    assert backoff.delay(10_000, random.Random(0)) == 7.0


def test_jitter_is_bounded_and_deterministic_for_a_seed() -> None:
    backoff = Backoff(initial=2.0, factor=1.0, max=2.0, jitter=0.25)
    first = [backoff.delay(1, random.Random(seed)) for seed in range(200)]
    assert all(1.5 <= delay <= 2.5 for delay in first)
    assert len(set(first)) > 100  # actually jittered
    assert first == [backoff.delay(1, random.Random(seed)) for seed in range(200)]


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Backoff(initial=-1), '"initial" must be a non-negative number'),
        (lambda: Backoff(factor=0.5), '"factor" must be a number >= 1'),
        (lambda: Backoff(max=-0.1), '"max" must be a non-negative number'),
        (lambda: Backoff(jitter=1.5), '"jitter" must be a number between 0 and 1'),
        (lambda: Backoff(initial=float("nan")), '"initial" must be a non-negative number'),
        (lambda: RestartPolicy(max_restarts=-1), '"max_restarts" must be a non-negative integer'),
        (lambda: RestartPolicy(window=0), '"window" must be a positive number'),
    ],
)
def test_invalid_values_are_rejected(build: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


def test_invalid_types_are_rejected() -> None:
    # deliberate negative tests: the type checker would reject these calls
    with pytest.raises(ValueError, match='"mode" must be "never" or "on-failure"'):
        RestartPolicy(mode="always")  # pyright: ignore[reportArgumentType] -- invalid on purpose
    with pytest.raises(ValueError, match='"max_restarts" must be a non-negative integer'):
        RestartPolicy(max_restarts=1.5)  # pyright: ignore[reportArgumentType] -- invalid on purpose
    with pytest.raises(ValueError, match='"backoff" must be a Backoff'):
        RestartPolicy(backoff={})  # pyright: ignore[reportArgumentType] -- invalid on purpose


def test_policy_from_mapping() -> None:
    policy = RestartPolicy.from_mapping(
        {"mode": "on-failure", "max_restarts": 3, "backoff": {"initial": 1, "jitter": 0}}
    )
    assert policy == RestartPolicy(
        mode="on-failure", max_restarts=3, backoff=Backoff(initial=1.0, jitter=0.0)
    )
    assert RestartPolicy.from_mapping({}) == RestartPolicy()


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ({"bogus": 1}, 'unknown key "restart.bogus"'),
        ({"mode": "always"}, '"restart.mode" must be "never" or "on-failure"'),
        ({"max_restarts": True}, '"restart.max_restarts" must be a non-negative integer'),
        ({"max_restarts": 2.0}, '"restart.max_restarts" must be a non-negative integer'),
        ({"window": "60"}, '"restart.window" must be a positive number'),
        ({"backoff": 1}, '"restart.backoff" must be a table'),
        ({"backoff": {"nope": 1}}, 'unknown key "restart.backoff.nope"'),
        ({"backoff": {"factor": 0}}, '"restart.backoff.factor" must be a number >= 1'),
        ({"backoff": {"jitter": -0.1}}, '"restart.backoff.jitter" must be a number between'),
    ],
)
def test_policy_from_mapping_names_the_bad_key(table: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RestartPolicy.from_mapping(table)


# -- restart on failure ----------------------------------------------------------------


async def test_never_is_the_default_and_keeps_a_failed_fiber_failed() -> None:
    logs = MemoryExporter()
    outcomes = [False, True]
    async with mock_context(logs) as root:
        fiber = root.plugin(failing_until(outcomes))
        with pytest.raises(RuntimeError, match="boom"):
            await fiber
        await asyncio.sleep(0.03)
        assert fiber.state == FiberState.FAILED
        assert outcomes == [True]
        assert supervisor_lines(logs, "warn") == []


async def test_on_failure_retries_until_the_plugin_loads() -> None:
    logs = MemoryExporter()
    outcomes = [False, False, True]
    async with mock_context(logs) as root:
        errors = capture_reports(root)
        fiber = root.plugin(failing_until(outcomes), policy=on_failure())
        await until(lambda: fiber.state == FiberState.ACTIVE)
        assert outcomes == []
        assert supervisor_lines(logs, "warn") == [
            "restarting flaky in 0.01s (attempt 1/5)",
            "restarting flaky in 0.01s (attempt 2/5)",
        ]
        assert [str(error) for error in errors] == ["boom", "boom"]
        assert await fiber is fiber
        assert backoff_tasks() == []


async def test_gives_up_after_max_restarts_within_the_window() -> None:
    logs = MemoryExporter()
    loads: list[int] = []

    def broken(_ctx: Context, _config: Any) -> None:
        loads.append(1)
        raise RuntimeError("boom")

    async with mock_context(logs) as root:
        fiber = root.plugin(broken, policy=on_failure(max_restarts=2))
        await until(lambda: bool(supervisor_lines(logs, "error")))
        await asyncio.sleep(0.05)  # no busy loop: nothing else happens
        assert len(loads) == 3
        assert fiber.state == FiberState.FAILED
        assert supervisor_lines(logs, "error") == ["giving up on broken after 2 restarts in 60s"]
        assert backoff_tasks() == []


async def test_restarts_older_than_the_window_are_forgotten() -> None:
    logs = MemoryExporter()
    outcomes = [False, True, False, True, False]
    async with mock_context(logs) as root:
        fiber = root.plugin(failing_until(outcomes), policy=on_failure(max_restarts=1, window=0.1))
        await until(lambda: fiber.state == FiberState.ACTIVE)
        await asyncio.sleep(0.15)  # the first restart leaves the window
        with pytest.raises(RuntimeError, match="boom"):
            await fiber.restart()
        await until(lambda: fiber.state == FiberState.ACTIVE)
        with pytest.raises(RuntimeError, match="boom"):  # within the window again
            await fiber.restart()
        await asyncio.sleep(0.03)
        assert fiber.state == FiberState.FAILED
        assert outcomes == []
        assert supervisor_lines(logs, "warn") == ["restarting flaky in 0.01s (attempt 1/1)"] * 2
        assert supervisor_lines(logs, "error") == ["giving up on flaky after 1 restarts in 0.1s"]


async def test_policy_from_plugin_metadata_and_call_override() -> None:
    logs = MemoryExporter()
    outcomes = [False, True]
    plugin = plugin_meta(name="meta", policy=on_failure())(failing_until(outcomes, "meta"))
    async with mock_context(logs) as root:
        fiber = root.plugin(plugin)
        assert fiber.policy == on_failure()
        await until(lambda: fiber.state == FiberState.ACTIVE)
        outcomes.append(False)
        # an explicit policy wins over the plugin's metadata
        other = root.plugin(plugin, policy=RestartPolicy())
        with pytest.raises(RuntimeError, match="boom"):
            await other
        await asyncio.sleep(0.03)
        assert other.state == FiberState.FAILED
        assert supervisor_lines(logs, "warn") == ["restarting meta in 0.01s (attempt 1/5)"]


async def test_update_during_backoff_recovers_and_cancels_the_wait() -> None:
    logs = MemoryExporter()
    outcomes = [False, True]
    async with mock_context(logs) as root:
        capture_reports(root)
        slow = on_failure(backoff=Backoff(initial=30.0, jitter=0.0))
        fiber = root.plugin(failing_until(outcomes), policy=slow)
        await until(lambda: bool(supervisor_lines(logs, "warn")))
        await fiber.update({"fixed": True})
        assert fiber.state == FiberState.ACTIVE
        assert fiber.supervisor is not None and not fiber.supervisor.waiting
        await asyncio.sleep(0)
        assert backoff_tasks() == []
        fiber.recover()  # not FAILED: nothing to do
        assert fiber.state == FiberState.ACTIVE
        assert outcomes == []


async def test_update_resets_the_restart_budget() -> None:
    logs = MemoryExporter()
    outcomes = [False, True, True, False, True]
    async with mock_context(logs) as root:
        capture_reports(root)
        fiber = root.plugin(failing_until(outcomes), policy=on_failure(max_restarts=1))
        await until(lambda: fiber.state == FiberState.ACTIVE)
        await fiber.update({"v": 2})  # a new configuration: fresh budget
        with pytest.raises(RuntimeError, match="boom"):
            await fiber.restart()
        await until(lambda: fiber.state == FiberState.ACTIVE)
        assert outcomes == []
        assert supervisor_lines(logs, "warn") == ["restarting flaky in 0.01s (attempt 1/1)"] * 2
        assert supervisor_lines(logs, "error") == []


def test_a_window_shorter_than_a_restart_warns() -> None:
    with pytest.warns(RuntimeWarning, match="window 0.001s is shorter than a restart takes"):
        RestartPolicy(mode="on-failure", window=0.001)


async def test_restart_waits_for_missing_dependencies() -> None:
    outcomes = [False, True]
    plugin = with_meta(failing_until(outcomes), inject=["db"])
    async with Context() as root:
        capture_reports(root)
        dispose = root.provide("db", object())
        fiber = root.plugin(plugin, policy=on_failure(backoff=Backoff(0.03, 1.0, 0.03, 0.0)))
        await until(lambda: fiber.state == FiberState.FAILED)
        await dispose()
        await until(lambda: fiber.state == FiberState.PENDING)
        root.provide("db", object())
        await until(lambda: fiber.state == FiberState.ACTIVE)
        assert outcomes == []


async def test_load_cancelled_from_outside_is_not_restarted() -> None:
    logs = MemoryExporter()
    started = asyncio.Event()

    async def slow(_ctx: Context, _config: Any) -> None:
        started.set()
        await asyncio.sleep(10)

    async with mock_context(logs) as root:
        fiber = root.plugin(slow, policy=on_failure())
        await started.wait()
        assert fiber.inertia is not None
        fiber.inertia.cancel()
        await until(lambda: fiber.state == FiberState.FAILED)
        await asyncio.sleep(0.03)
        assert fiber.state == FiberState.FAILED
        assert supervisor_lines(logs, "warn") == []


# -- backoff waits are owned by the parent ---------------------------------------------


async def failed_in_backoff(root: Context, logs: MemoryExporter) -> Fiber:
    """Start a plugin that fails forever, with a 30 s backoff; return it once it is waiting."""

    def forever(_ctx: Context, _config: Any) -> None:
        raise RuntimeError("boom")

    slow = on_failure(backoff=Backoff(initial=30.0, jitter=0.0))
    fiber = root.plugin(forever, policy=slow)
    await until(lambda: bool(supervisor_lines(logs, "warn")))
    assert fiber.supervisor is not None and fiber.supervisor.waiting
    assert len(backoff_tasks()) == 1
    return fiber


async def measure(awaitable: Awaitable[object]) -> float:
    started = time.monotonic()
    await awaitable
    return time.monotonic() - started


async def test_root_dispose_during_backoff_is_immediate() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        capture_reports(root)
        fiber = await failed_in_backoff(root, logs)
        assert await measure(root.fiber.dispose()) < 0.5
        assert fiber.state == FiberState.DISPOSED
        await asyncio.sleep(0)
        assert backoff_tasks() == []
        assert supervisor_lines(logs, "warn") == ["restarting forever in 30.00s (attempt 1/5)"]


async def test_disposing_the_fiber_cancels_its_backoff() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        capture_reports(root)
        fiber = await failed_in_backoff(root, logs)
        assert await measure(fiber.dispose()) < 0.5
        await asyncio.sleep(0)
        assert backoff_tasks() == []
        assert fiber.supervisor is not None and not fiber.supervisor.waiting


async def test_parent_unload_cancels_a_child_backoff() -> None:
    logs = MemoryExporter()
    children: list[Fiber] = []

    def forever(_ctx: Context, _config: Any) -> None:
        raise RuntimeError("boom")

    def parent(ctx: Context, _config: Any) -> None:
        slow = on_failure(backoff=Backoff(initial=30.0, jitter=0.0))
        children.append(ctx.plugin(forever, policy=slow))

    async with mock_context(logs) as root:
        capture_reports(root)
        outer = await root.plugin(parent)
        await until(lambda: bool(supervisor_lines(logs, "warn")))
        assert await measure(outer.dispose()) < 0.5
        assert children[0].state == FiberState.DISPOSED
        await asyncio.sleep(0)
        assert backoff_tasks() == []


async def test_child_failing_while_its_parent_unloads_is_not_restarted() -> None:
    logs = MemoryExporter()
    gate = asyncio.Event()
    children: list[Fiber] = []

    async def child(_ctx: Context, _config: Any) -> None:
        await gate.wait()
        raise RuntimeError("late")

    def parent(ctx: Context, _config: Any) -> None:
        ctx.provide("svc", object())
        children.append(ctx.plugin(child, policy=on_failure()))

    def dependent(ctx: Context, _config: Any) -> Callable[[], Awaitable[None]]:
        async def close() -> None:
            gate.set()  # the child fails while the parent waits for this unload
            await asyncio.sleep(0.03)

        return close

    async with mock_context(logs) as root:
        capture_reports(root)
        outer = await root.plugin(parent)
        await root.plugin(with_meta(dependent, inject=["svc"]))
        await outer.dispose()
        assert children[0].state == FiberState.DISPOSED
        assert supervisor_lines(logs, "warn") == []
        assert backoff_tasks() == []


# -- load_timeout ----------------------------------------------------------------------


def hanging() -> Callable[[Context, Any], Awaitable[None]]:
    """A fresh plugin named ``hang`` whose body never finishes (metadata never leaks)."""

    async def hang(_ctx: Context, _config: Any) -> None:
        await asyncio.sleep(10)

    return hang


async def test_load_timeout_fails_the_fiber_with_a_timeout_error() -> None:
    async with Context() as root:
        errors = capture_reports(root)
        fiber = root.plugin(hanging(), load_timeout=0.02)
        assert fiber.load_timeout == 0.02
        with pytest.raises(LoadTimeoutError, match=r"did not load within 0\.02s") as info:
            await fiber
        assert "in plugin <hang>" in (info.value.__notes__ or [])
        assert fiber.state == FiberState.FAILED
        assert errors == [info.value]


async def test_load_timeout_counts_as_a_failure_for_supervision() -> None:
    logs = MemoryExporter()
    attempts: list[int] = []

    async def slow_once(_ctx: Context, _config: Any) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(10)

    async with mock_context(logs) as root:
        capture_reports(root)
        fiber = root.plugin(slow_once, policy=on_failure(), load_timeout=0.02)
        await until(lambda: fiber.state == FiberState.ACTIVE)
        assert len(attempts) == 2
        assert supervisor_lines(logs, "warn") == ["restarting slow_once in 0.01s (attempt 1/5)"]


async def test_load_timeout_bounds_the_unload_of_a_hung_effect() -> None:
    logs = MemoryExporter()
    gate = asyncio.Event()
    attempts: list[int] = []

    class HungSetup:
        async def __aenter__(self) -> None:
            await gate.wait()

        async def __aexit__(self, *_exc: object) -> None:
            return None

    async def body(ctx: Context, _config: Any) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            await ctx.effect(HungSetup, "hung-setup")  # the setup outlives the body

    async with mock_context(logs) as root:
        errors = capture_reports(root)
        started = time.monotonic()
        fiber = root.plugin(body, policy=on_failure(), load_timeout=0.05)
        await until(lambda: bool(supervisor_lines(logs, "warn")))
        assert time.monotonic() - started < 0.2  # ~2 x load_timeout, not until the effect ends
        await until(lambda: fiber.state == FiberState.ACTIVE)
        assert len(attempts) == 2
        assert [str(error) for error in errors] == [
            "plugin <body> did not load within 0.05s",
            "disposer of effect hung-setup in <body> did not finish within 0.05s; abandoned",
        ]
        assert isinstance(errors[0], TimeoutError)
        gate.set()  # the abandoned setup finishes and its effect is torn down
        await until(lambda: other_tasks() == [])


async def test_load_timeout_holds_when_the_body_swallows_its_cancellation() -> None:
    disposed: list[str] = []

    async def stubborn(ctx: Context, _config: Any) -> None:
        ctx.effect(lambda: lambda: disposed.append("effect"), "late")
        with contextlib.suppress(asyncio.CancelledError):  # misbehaving: carries on
            await asyncio.sleep(10)
        await asyncio.sleep(0.03)  # and returns well after the deadline

    async with Context() as root:
        errors = capture_reports(root)
        fiber = root.plugin(stubborn, load_timeout=0.02)
        with pytest.raises(
            LoadTimeoutError, match=r"plugin <stubborn> did not load within"
        ) as info:
            await fiber
        assert fiber.state == FiberState.FAILED
        assert disposed == ["effect"]  # what the late body set up is torn down
        assert errors == [info.value]


async def test_dispose_timeout_wins_over_load_timeout_for_that_unload() -> None:
    gate = asyncio.Event()

    class HungSetup:
        async def __aenter__(self) -> None:
            await gate.wait()

        async def __aexit__(self, *_exc: object) -> None:
            return None

    async def body(ctx: Context, _config: Any) -> None:
        await ctx.effect(HungSetup, "hung-setup")

    async with Context() as root:
        errors = capture_reports(root)
        fiber = root.plugin(body, load_timeout=0.02, dispose_timeout=0.1)
        await until(lambda: fiber.state == FiberState.FAILED)
        assert "within 0.1s; abandoned" in str(errors[-1])
        gate.set()
        await until(lambda: other_tasks() == [])


async def test_body_timeout_error_is_not_mistaken_for_the_load_timeout() -> None:
    async def own(_ctx: Context, _config: Any) -> None:
        await asyncio.sleep(0)
        raise TimeoutError("upstream")

    async with Context() as root:
        capture_reports(root)
        with pytest.raises(TimeoutError) as info:
            await root.plugin(own, load_timeout=5)
        assert str(info.value) == "upstream"
        assert not isinstance(info.value, LoadTimeoutError)


async def test_load_timeout_from_metadata_and_fast_bodies_are_unaffected() -> None:
    async def quick(_ctx: Context, _config: Any) -> None:
        await asyncio.sleep(0)

    async with Context() as root:
        capture_reports(root)
        fiber = await root.plugin(plugin_meta(load_timeout=0.5)(quick))
        assert fiber.load_timeout == 0.5
        assert fiber.state == FiberState.ACTIVE
        with pytest.raises(TimeoutError):
            await root.plugin(plugin_meta(load_timeout=0.01)(hanging()))


async def test_superseded_load_with_timeout_is_cancelled_quietly() -> None:
    async with Context() as root:
        errors = capture_reports(root)
        fiber = root.plugin(hanging(), load_timeout=5)
        await asyncio.sleep(0.01)
        await fiber.dispose()
        assert fiber.state == FiberState.DISPOSED
        assert errors == []


@pytest.mark.parametrize("value", [0, -1, float("nan"), True])
async def test_invalid_timeouts_are_rejected(value: float) -> None:
    async with Context() as root:
        with pytest.raises(ValueError, match='"load_timeout" must be a positive number'):
            root.plugin(hanging(), load_timeout=value)
        with pytest.raises(ValueError, match='"dispose_timeout" must be a positive number'):
            plugin_meta(dispose_timeout=value)
        assert len(root.registry) == 0


async def test_invalid_metadata_is_rejected() -> None:
    plugin = with_meta(hanging(), policy="on-failure")
    async with Context() as root:
        with pytest.raises(
            TypeError, match=r'plugin .hang. has invalid "policy": expected RestartPolicy, got str'
        ):
            root.plugin(plugin)
        with pytest.raises(ValueError, match='"load_timeout" must be a positive number'):
            root.plugin(with_meta(hanging(), load_timeout="5"))


# -- dispose_timeout -------------------------------------------------------------------


def hung_disposer(ctx: Context, _config: Any) -> None:
    async def close() -> None:
        await asyncio.sleep(10)

    ctx.effect(lambda: close, "hung-close")
    ctx.effect(lambda: lambda: None, "quick-close")


async def test_dispose_timeout_abandons_a_hung_disposer() -> None:
    async with Context() as root:
        errors = capture_reports(root)
        fiber = await root.plugin(hung_disposer, dispose_timeout=0.02)
        assert fiber.dispose_timeout == 0.02
        assert await measure(fiber.dispose()) < 0.5
        assert fiber.state == FiberState.DISPOSED
        assert [str(error) for error in errors] == [
            "disposer of effect hung-close in <hung_disposer> did not finish within 0.02s; "
            "abandoned"
        ]
        assert isinstance(errors[0], TimeoutError)
        # the abandoned disposer was cancelled: no task is left behind
        await until(lambda: other_tasks() == [])


async def test_dispose_timeout_bounds_a_restart_too() -> None:
    async with Context() as root:
        errors = capture_reports(root)
        fiber = await root.plugin(hung_disposer, dispose_timeout=0.02)
        assert await measure(fiber.restart()) < 0.5
        assert fiber.state == FiberState.ACTIVE
        assert len(errors) == 1


async def test_dispose_timeout_reports_a_task_ignoring_cancellation() -> None:
    release = asyncio.Event()
    finished: list[bool] = []

    async def stubborn() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await release.wait()  # ignores the cancellation until released
            finished.append(True)

    def plugin(ctx: Context, _config: Any) -> None:
        ctx.spawn(stubborn(), name="stubborn")

    async with Context() as root:
        errors = capture_reports(root)
        fiber = await root.plugin(plugin, dispose_timeout=0.02)
        assert await measure(fiber.dispose()) < 0.5
        assert [str(error) for error in errors] == [
            "task stubborn in <plugin> did not finish within 0.02s; abandoned"
        ]
        release.set()
        await until(lambda: finished == [True])


async def test_without_dispose_timeout_unload_waits_for_disposers() -> None:
    def slowish(ctx: Context, _config: Any) -> None:
        async def close() -> None:
            await asyncio.sleep(0.05)

        ctx.effect(lambda: close, "slowish")

    async with Context() as root:
        errors = capture_reports(root)
        fiber = await root.plugin(slowish)
        assert fiber.dispose_timeout is None
        assert await measure(fiber.dispose()) >= 0.04
        assert errors == []


async def test_raw_disposer_label_falls_back_to_its_repr() -> None:
    def plugin(_ctx: Context, _config: Any) -> Callable[[], Awaitable[None]]:
        async def stuck() -> None:
            await asyncio.sleep(10)

        return stuck

    async with Context() as root:
        errors = capture_reports(root)
        fiber = await root.plugin(plugin, dispose_timeout=0.02)
        await fiber.dispose()
        assert len(errors) == 1
        assert "plugin.<locals>.stuck at 0x" in str(errors[0])


async def test_cancelled_unload_with_dispose_timeout_cancels_its_disposers() -> None:
    async with Context() as root:
        capture_reports(root)
        fiber = await root.plugin(hung_disposer, dispose_timeout=5)
        dispose = fiber.dispose()
        await asyncio.sleep(0.01)
        assert fiber.inertia is not None
        fiber.inertia.cancel()
        await dispose
        # the running disposers were cancelled with the unload, not left sleeping
        await until(lambda: other_tasks() == [])
