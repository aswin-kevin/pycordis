import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from pycordis._context_internals import intercept_configs
from pycordis._utils import (
    DisposableList,
    Label,
    ScopeMap,
    add_note_once,
    cancelling,
    done_future,
    hyphenate,
    is_bailed,
    merge_entries,
    report_failure,
    resolve_future,
    symbols,
)
from pycordis.context import Context
from pycordis.errors import CordisError, Issue, ValidationError


def test_disposable_list_order_and_delete():
    dl = DisposableList[object]()
    a, b, c = object(), object(), object()
    dl.push(a)
    rm = dl.push(b)
    dl.push(c)
    assert list(dl) == [a, b, c] and len(dl) == 3
    assert rm() is True and rm() is False
    assert dl.delete(c) is True and dl.delete(c) is False
    dl.push(c)
    assert dl.clear() == [c, a] and len(dl) == 0


def test_scope_map_in_vs_own():
    root = ScopeMap[int].empty()
    a = root.child("x", 1)
    b = a.child("y", 2)
    assert "x" in b and not b.has_own("x") and b.has_own("y") and b["x"] == 1
    assert "x" not in root
    assert b.parent is a and root.parent is None
    assert b.get("zz") is None and b.get("zz", 5) == 5
    assert list(b.own_items()) == [("y", 2)]


def test_scope_map_set_root():
    root = ScopeMap[int].empty()
    child = root.child("a", 1)
    root.set_root("r", 9)
    assert child["r"] == 9


def test_label():
    a, b = Label("x"), Label("x")
    assert a != b and a == a and repr(a) == "Label('x')"


def test_symbols():
    assert symbols.invoke == "__cordis_invoke__"
    assert symbols.init_hooks == "__cordis_init_hooks__"
    with pytest.raises(AttributeError):
        # the frozen dataclass must reject the write the checker also rejects
        symbols.invoke = "x"  # pyright: ignore[reportAttributeAccessIssue]


def test_validation_error_message():
    err = ValidationError([Issue("bad", ("a", 0)), Issue("worse")])
    assert str(err) == "invalid config:\n  - bad (at a.0)\n  - worse"
    assert isinstance(err, TypeError)


def test_cordis_error_default_message():
    e = CordisError("INACTIVE_EFFECT")
    assert e.code == "INACTIVE_EFFECT" and str(e) == "cannot create effect on inactive context"


@pytest.mark.parametrize(
    "v,bailed", [(None, False), (False, False), (0, True), ("", True), ([], True)]
)
def test_is_bailed(v: object, bailed: bool):
    assert is_bailed(v) is bailed


def test_hyphenate():
    names = ["FooBar", "fooBar", "foo_bar", "foo"]
    assert [hyphenate(s) for s in names] == ["foo-bar"] * 3 + ["foo"]


def test_disposable_list_prunes_ids() -> None:
    items: DisposableList[object] = DisposableList()
    a, b = object(), object()
    dispose_a = items.push(a)
    items.unshift(b)
    assert dispose_a()
    assert items.delete(b)
    assert items.index_size == 0
    # a value pushed twice: disposing the stale entry keeps the newer id mapping
    first = items.push(a)
    items.push(a)
    assert first()
    assert items.delete(a)
    assert items.index_size == 0 and len(items) == 0


async def test_future_helpers() -> None:
    done = done_future()
    assert done.done() and done.result() is None
    loop = asyncio.get_running_loop()
    assert done_future(loop).result() is None
    pending: asyncio.Future[None] = loop.create_future()
    resolve_future(pending)
    resolve_future(pending)  # already done: no InvalidStateError
    assert pending.result() is None


async def test_cancelling() -> None:
    assert not cancelling()
    seen: list[bool] = []

    async def body() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            seen.append(cancelling())
            raise

    task = asyncio.ensure_future(body())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == [True]


async def test_report_failure() -> None:
    loop = asyncio.get_running_loop()
    reported: list[BaseException] = []
    error = ValueError("boom")
    failed: asyncio.Future[None] = loop.create_future()
    failed.set_exception(error)
    report_failure(failed, reported.append, eager=True)
    assert reported == [error]  # checked right away
    later: asyncio.Future[None] = loop.create_future()
    report_failure(later, reported.append)
    later.set_exception(error)
    cancelled: asyncio.Future[None] = loop.create_future()
    report_failure(cancelled, reported.append)
    cancelled.cancel()
    ok = done_future()
    report_failure(ok, reported.append)
    await asyncio.sleep(0)
    assert reported == [error, error]


def test_add_note_once() -> None:
    error = ValueError()
    add_note_once(error, "a")
    add_note_once(error, "a")
    add_note_once(error, "b")
    assert error.__notes__ == ["a", "b"]


async def test_intercept_configs_and_merge() -> None:
    root = Context()
    ctx = root.intercept("svc", {"a": 1}).intercept("other", 2).intercept("svc", {"b": 2})
    assert intercept_configs(ctx, "svc") == [{"a": 1}, {"b": 2}]
    assert intercept_configs(root, "svc") == []
    merged = merge_entries([{"a": 1, "b": 1}, SimpleNamespace(b=2, c=3)])
    assert merged == {"a": 1, "b": 2, "c": 3}


def test_missing_sentinel_repr() -> None:
    from pycordis._guards import MISSING

    assert repr(MISSING) == "MISSING"


def test_merge_entries_keeps_non_str_keys() -> None:
    assert merge_entries([{1: "a"}, {"b": 2}]) == {1: "a", "b": 2}


def test_report_unhandled_without_a_loop_keeps_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from pycordis._utils import report_unhandled

    seen: list[Any] = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    error = RuntimeError("x")
    report_unhandled("pycordis: failed", error, logger_error=KeyError("logger"))
    [args] = seen
    assert args.exc_value is error
    assert args.err_msg == "pycordis: failed (logger_error=KeyError('logger'))"
