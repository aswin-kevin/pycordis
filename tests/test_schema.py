"""Tests for config schema adapters."""

from __future__ import annotations

from collections.abc import Coroutine, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import pytest

from pycordis.errors import Issue, ValidationError
from pycordis.schema import validate_config
from pycordis.volatile import Volatile


class Mode(Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass
class Inner:
    host: str
    tags: list[str] = field(default_factory=list[str])


@dataclass
class Cfg:
    port: int
    inner: Inner
    ratio: float = 1.0
    nick: str | None = None
    level: Literal["a", "b"] = "a"
    mode: Mode = Mode.FAST
    extra: Volatile[dict[str, int]] | None = None
    pair: tuple[int, ...] = ()
    table: dict[str, int] = field(default_factory=dict[str, int])
    anything: Any = None


def issues_of(schema: object, raw: Any) -> tuple[Issue, ...]:
    with pytest.raises(ValidationError) as ei:
        validate_config(schema, raw)
    return ei.value.issues


def test_none_schema_passthrough() -> None:
    raw = {"a": 1}
    assert validate_config(None, raw) is raw


def test_dataclass_instance_passes_through() -> None:
    built = Cfg(port=80, inner=Inner("h"))
    assert validate_config(Cfg, built) is built


def test_instance_of_another_dataclass_is_rejected() -> None:
    assert issues_of(Cfg, Inner("h")) == (Issue("expected object"),)


def test_mapping_is_still_validated_against_a_dataclass() -> None:
    assert validate_config(Cfg, {"port": 80, "inner": {"host": "h"}}) == Cfg(80, Inner("h"))
    assert issues_of(Cfg, {"port": "x", "inner": {"host": "h"}}) == (
        Issue("expected int, got str", ("port",)),
    )


def test_valid_nested_dataclass() -> None:
    cfg = validate_config(
        Cfg,
        {
            "port": 80,
            "inner": {"host": "h", "tags": ["x"]},
            "ratio": 2,
            "mode": "slow",
            "pair": (1, 2),
            "table": {"k": 1},
            "anything": object,
        },
    )
    assert isinstance(cfg, Cfg)
    assert cfg.inner == Inner("h", ["x"])
    assert cfg.mode is Mode.SLOW
    assert cfg.ratio == 2.0
    assert isinstance(cfg.ratio, float)
    assert cfg.pair == (1, 2)
    assert cfg.nick is None


def test_none_input_means_empty() -> None:
    @dataclass
    class D:
        a: int = 3

    assert validate_config(D, None) == D(3)


def test_type_mismatch_text() -> None:
    with pytest.raises(ValidationError) as ei:
        validate_config(Cfg, {"port": "x", "inner": {"host": "h"}})
    assert str(ei.value) == "invalid config:\n  - expected int, got str (at port)"


def test_not_mapping() -> None:
    assert issues_of(Cfg, 5) == (Issue("expected object"),)


def test_missing_and_unknown() -> None:
    issues = issues_of(Cfg, {"inner": {"host": "h"}, "bogus": 1})
    assert Issue("missing required field", ("port",)) in issues
    assert Issue("unknown key", ("bogus",)) in issues


def test_bool_not_int() -> None:
    assert issues_of(Cfg, {"port": True, "inner": {"host": "h"}}) == (
        Issue("expected int, got bool", ("port",)),
    )


def test_literal_union_enum_errors() -> None:
    issues = issues_of(
        Cfg, {"port": 1, "inner": {"host": "h"}, "level": "c", "nick": 3, "mode": "x"}
    )
    by_path = {i.path: i.message for i in issues}
    assert by_path == {
        ("level",): "expected 'a' | 'b', got str",
        ("nick",): "expected str, got int",
        ("mode",): "expected one of 'fast', 'slow', got str",  # the value is never echoed
    }


def test_nested_paths_with_indices() -> None:
    issues = issues_of(Cfg, {"port": 1, "inner": {"host": "h", "tags": ["a", 2]}})
    assert issues == (Issue("expected str, got int", ("inner", "tags", 1)),)


def test_container_errors() -> None:
    issues = issues_of(
        Cfg,
        {
            "port": 1,
            "inner": {"host": "h", "tags": "ab"},
            "table": {"k": "v", 1: 2},
            "pair": 5,
        },
    )
    assert sorted((i.path, i.message) for i in issues) == sorted(
        [
            (("inner", "tags"), "expected list[str], got str"),
            (("table", "k"), "expected int, got str"),
            (("table",), "expected str key, got int"),
            (("pair",), "expected tuple[int, ...], got int"),
        ]
    )


def test_volatile_field() -> None:
    cfg: Cfg = validate_config(Cfg, {"port": 1, "inner": {"host": "h"}, "extra": {"a": 1}})
    assert isinstance(cfg.extra, Volatile)
    assert cfg.extra.get()["a"] == 1
    bad = issues_of(Cfg, {"port": 1, "inner": {"host": "h"}, "extra": {"a": "s"}})
    assert bad == (Issue("expected int, got str", ("extra", "a")),)


def test_callable_schema() -> None:
    def double(raw: int) -> int:
        return raw * 2

    assert validate_config(double, 2) == 4

    def bad(_: Any) -> Any:
        raise ValueError("nope")

    with pytest.raises(ValidationError) as ei:
        validate_config(bad, 1)
    assert str(ei.value) == "invalid config:\n  - nope"


class StdSchema:
    def __init__(self, result: Any) -> None:
        self.result = result

    def validate(self, raw: Any) -> Any:
        return self.result


class Res:
    def __init__(self, value: Any = None, issues: Any = None) -> None:
        self.value = value
        self.issues = issues


def test_standard_schema_shapes() -> None:
    assert validate_config(StdSchema(Res(value=7)), 1) == 7
    assert validate_config(StdSchema({"value": 8}), 1) == 8
    got = issues_of(StdSchema(Res(issues=[{"message": "m", "path": ["a", {"key": 1}]}])), 1)
    assert got == (Issue("m", ("a", 1)),)
    obj = type("I", (), {"message": "o", "path": None})()
    assert issues_of(StdSchema({"issues": [obj]}), 1) == (Issue("o"),)


def _coro_schema(_raw: object) -> Coroutine[Any, Any, int]:
    return _coro()


async def _coro() -> int:
    return 1


def test_async_rejected() -> None:
    with pytest.raises(TypeError, match="Async config validation is not supported"):
        validate_config(StdSchema(_coro()), 1)
    with pytest.raises(TypeError, match="Async config validation is not supported"):
        validate_config(_coro_schema, 1)


class PydanticErr(Exception):
    def errors(self) -> list[dict[str, Any]]:
        return [{"loc": ("a", 0), "msg": "bad"}, {"loc": (), "msg": "root"}]


class FakeModel:
    @classmethod
    def model_validate(cls, raw: Any) -> Any:
        if raw == "boom":
            raise RuntimeError("other")
        if raw:
            return cls()
        raise PydanticErr

    def validate(self, raw: Any) -> Any:  # must lose to model_validate
        raise AssertionError


def test_pydantic_duck_typed() -> None:
    assert isinstance(validate_config(FakeModel, 1), FakeModel)
    assert issues_of(FakeModel, 0) == (Issue("bad", ("a", 0)), Issue("root"))
    with pytest.raises(RuntimeError):
        validate_config(FakeModel, "boom")


def test_unsupported_annotation() -> None:
    @dataclass
    class U:
        a: set[int]

    with pytest.raises(TypeError, match="unsupported"):
        validate_config(U, {"a": []})


def test_nested_optional_none_stays_none() -> None:
    @dataclass
    class Opt:
        inner: Inner | None = None

    assert validate_config(Opt, {"inner": None}) == Opt(None)
    assert issues_of(Opt, {"inner": 5}) == (Issue("expected object", ("inner",)),)


def test_validation_error_passthrough() -> None:
    def nested(_: Any) -> Any:
        validate_config(Cfg, {"port": "x", "inner": {"host": "h"}})

    issues = issues_of(nested, 1)
    assert issues == (Issue("expected int, got str", ("port",)),)


def test_callable_type_error() -> None:
    def bad(_: Any) -> Any:
        raise TypeError("tt")

    assert issues_of(bad, 1) == (Issue("tt"),)


def test_unsupported_schema() -> None:
    with pytest.raises(TypeError, match="unsupported config schema"):
        validate_config(5, 1)


def test_malformed_error_dicts() -> None:
    class E(Exception):
        def errors(self) -> list[dict[str, Any]]:
            return [{}]

    class M:
        @staticmethod
        def model_validate(raw: Any) -> Any:
            raise E

    assert issues_of(M, 1) == (Issue("invalid value"),)
    assert issues_of(StdSchema(Res(issues=[{"path": None}])), 1) == (Issue("invalid value"),)


def test_mapping_bare_and_unions() -> None:
    @dataclass
    class B:
        m: Mapping[str, int]
        # bare generics on purpose: the schema must accept unparameterized containers
        ls: list  # pyright: ignore[reportMissingTypeArgument]
        d: dict  # pyright: ignore[reportMissingTypeArgument]
        u: int | str
        f: float = 0.0

    got = validate_config(B, {"m": {"a": 1}, "ls": [1, "x"], "d": {"k": 1}, "u": "s"})
    assert got.m == {"a": 1} and got.ls == [1, "x"] and got.u == "s"
    assert issues_of(B, {"m": {}, "ls": [], "d": {}, "u": 1.5, "f": True}) == (
        Issue("expected int | str, got float", ("u",)),
        Issue("expected float, got bool", ("f",)),
    )


def test_list_tuple_rules() -> None:
    assert issues_of(Cfg, {"port": 1, "inner": {"host": "h", "tags": ("a",)}})
    cfg = validate_config(Cfg, {"port": 1, "inner": {"host": "h"}, "pair": [1, 2]})
    assert cfg.pair == (1, 2)


def test_unsupported_eager() -> None:
    @dataclass
    class T:
        a: tuple[int, str] = (1, "a")

    with pytest.raises(TypeError, match="unsupported"):
        validate_config(T, {"a": 5})
    with pytest.raises(TypeError, match="unsupported"):
        validate_config(T, {})


def test_a_non_callable_model_validate_is_a_type_error() -> None:
    class Schema:
        model_validate = 5

    with pytest.raises(TypeError, match="'int' object is not callable"):
        validate_config(Schema, {})
