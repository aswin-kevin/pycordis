from types import MappingProxyType
from typing import Any

import pytest

from pycordis.volatile import (
    Volatile,
    create_volatile,
    diff_only_volatile,
    is_volatile,
    volatile_entries,
    volatile_pairs,
)


def test_snapshot_is_deep_frozen() -> None:
    v = create_volatile({"a": [1, {"b": 2}]})
    snap: Any = v.get()
    with pytest.raises(TypeError):
        snap["a"] = 1
    assert snap["a"] == (1, {"b": 2}) and isinstance(snap["a"][1], MappingProxyType)


def test_scalars_and_tuples() -> None:
    assert create_volatile(None).get() is None
    assert create_volatile("s").get() == "s"
    assert create_volatile((1, 2.5, True)).get() == (1, 2.5, True)


@pytest.mark.parametrize("bad,msg", [({"f": len}, "functions"), ({"o": object()}, "plain objects")])
def test_rejects(bad: Any, msg: str) -> None:
    with pytest.raises(TypeError, match=msg):
        create_volatile(bad)


def test_rejects_cycles() -> None:
    d: dict[str, Any] = {}
    d["self"] = d
    with pytest.raises(TypeError, match="cycles"):
        create_volatile(d)


def test_shared_non_cyclic_reference_allowed() -> None:
    shared = [1]
    assert create_volatile({"a": shared, "b": shared}).get()["a"] == (1,)


def test_not_user_constructible() -> None:
    with pytest.raises(TypeError):
        Volatile(1)


def test_is_volatile() -> None:
    assert is_volatile(create_volatile(1))
    assert not is_volatile({"a": 1}) and not is_volatile(None)


def test_entries_and_commit() -> None:
    ref = create_volatile({"x": 1})
    cfg = {"a": {"b": ref}, "c": 2, "d": [ref], "e": object()}
    assert volatile_entries(cfg) == [(("a", "b"), ref), (("d", "0"), ref)]
    assert volatile_entries(ref) == [((), ref)]
    ref.commit(MappingProxyType({"x": 2}))
    assert ref.get()["x"] == 2


def test_entries_cyclic_ordinary_fields() -> None:
    ref = create_volatile(1)
    d: dict[str, Any] = {"r": ref}
    d["self"] = d
    assert volatile_entries(d) == [(("r",), ref)]


def test_diff_only_volatile() -> None:
    r1, r2 = create_volatile(1), create_volatile(2)
    assert diff_only_volatile({"a": r1, "b": 1}, {"a": r2, "b": 1})
    assert not diff_only_volatile({"a": r1, "b": 1}, {"a": r2, "b": 2})
    assert not diff_only_volatile({"a": r1}, {"a": 1})
    assert not diff_only_volatile({"a": 1}, {"a": 1, "b": 2})
    assert diff_only_volatile([r1, 1], [r2, 1])
    assert not diff_only_volatile([r1], [r2, 1])
    assert not diff_only_volatile([1], {"a": 1})


def test_repr() -> None:
    assert repr(create_volatile(1)) == "Volatile(1)"


def test_dataclass_entries_and_diff() -> None:
    from dataclasses import dataclass

    @dataclass
    class Inner:
        ref: Any

    @dataclass
    class Outer:
        inner: Inner
        port: int

    @dataclass
    class Other:
        inner: Inner
        port: int

    r1, r2 = create_volatile(1), create_volatile(2)
    old, new = Outer(Inner(r1), 80), Outer(Inner(r2), 80)
    assert volatile_entries(old) == [(("inner", "ref"), r1)]
    assert diff_only_volatile(old, new)
    assert not diff_only_volatile(old, Outer(Inner(r2), 81))
    assert not diff_only_volatile(old, Other(Inner(r2), 80))
    assert not diff_only_volatile(old, {"inner": Inner(r2), "port": 80})
    # dataclass types themselves are not traversed
    assert volatile_entries(Outer) == []


def test_volatile_pairs_keep_key_types() -> None:
    a1, a2, b1, b2 = (create_volatile(i) for i in range(4))
    old = {1: a1, "1": b1, "l": [a1], "x": 0}
    new = {1: a2, "1": b2, "l": [a2], "x": 0}
    assert volatile_pairs(old, new) == [(a1, a2), (b1, b2), (a1, a2)]
    assert volatile_pairs(a1, 1) == []
    assert volatile_pairs({"k": a1}, {}) == []
