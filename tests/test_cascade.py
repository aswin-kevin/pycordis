"""The trampoline behind service-change cascades (``pycordis._cascade``)."""

from __future__ import annotations

import pytest

from pycordis._cascade import Cascade, Steps


def test_nested_work_runs_in_recursive_order_without_stack_growth() -> None:
    cascade = Cascade()
    visited: list[str] = []

    def visit(path: str, depth: int) -> Steps:
        visited.append(path)
        if depth == 0:
            return
        for child in "ab":
            cascade.run(visit(path + child, depth - 1))
            yield

    cascade.run(visit("", 2))
    assert visited == ["", "a", "aa", "ab", "b", "ba", "bb"]

    def chain(depth: int) -> Steps:
        if depth:
            cascade.run(chain(depth - 1))
        yield

    cascade.run(chain(100_000))  # recursion this deep would overflow the stack


def test_work_started_in_one_step_runs_in_call_order() -> None:
    cascade = Cascade()
    log: list[str] = []

    def leaf(name: str) -> Steps:
        log.append(name)
        yield

    def parent() -> Steps:
        cascade.run(leaf("first"))
        cascade.run(leaf("second"))
        log.append("rest of the step")
        yield
        log.append("next step")

    cascade.run(parent())
    assert log == ["rest of the step", "first", "second", "next step"]


def test_an_error_abandons_the_cascade_and_leaves_it_usable() -> None:
    cascade = Cascade()
    log: list[str] = []

    def failing() -> Steps:
        yield from ()  # a generator failing on its first step
        raise RuntimeError("boom")

    def parent() -> Steps:
        cascade.run(failing())
        yield
        log.append("not reached")

    with pytest.raises(RuntimeError, match="boom"):
        cascade.run(parent())
    assert log == []

    def ok() -> Steps:
        log.append("ran")
        yield

    cascade.run(ok())
    assert log == ["ran"]
