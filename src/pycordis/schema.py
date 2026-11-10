"""Config schema adapters (port of cordis ``resolveConfig``).

:func:`validate_config` validates a raw plugin config against a schema that may
be a dataclass, a pydantic-style model, a Standard-Schema-shaped object or a
plain callable. Validation is synchronous only.

Typing: a schema is any of those shapes, so it is probed by attribute
(``model_validate``, ``validate``), and annotations are runtime type forms
(``list[int]``, ``Literal["a"]``, ...) typed ``object``. The validated value's
type is decided by the schema at runtime, so :func:`validate_config` returns
``Any``.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
from collections.abc import Mapping
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    TypeGuard,
    Union,
    final,
    get_args,
    get_origin,
    get_type_hints,
)

from ._guards import MISSING, is_array, is_iterable, is_mapping
from .errors import Issue, ValidationError
from .volatile import Volatile, create_volatile

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_ASYNC_MSG = "Async config validation is not supported"
type _Path = tuple[str | int, ...]


@final
class _Failed(Enum):
    """The type of :data:`_FAIL`: a value failed validation (its issues were recorded)."""

    FAIL = "FAIL"


_FAIL: Final = _Failed.FAIL


def validate_config(schema: object | None, raw: object) -> Any:
    """Validate ``raw`` against ``schema`` and return the validated value.

    ``None`` schema returns ``raw`` unchanged, and so does a dataclass schema
    given an instance of itself. Raises :class:`ValidationError`
    on invalid input and :class:`TypeError` if a validator is asynchronous.
    Example: ``validate_config(ServerConfig, {"port": 80}) == ServerConfig(port=80)``.
    """
    if schema is None:
        return raw
    if _is_dataclass_type(schema):
        if isinstance(raw, schema):
            # an already-built config, e.g. ``ctx.plugin(chat, ChatConfig(2))``
            return raw
        issues: list[Issue] = []
        value = _check(schema, {} if raw is None else raw, (), issues)
        if issues:
            raise ValidationError(issues)
        return value
    # duck-typed schema libraries: probe the protocol method each one defines
    model_validate: object = getattr(schema, "model_validate", MISSING)
    if model_validate is not MISSING:
        return _run_pydantic(model_validate, raw)
    validate: object = getattr(schema, "validate", MISSING)
    if validate is not MISSING:
        return _run_standard(validate, raw)
    if callable(schema):
        return _run_callable(schema, raw)
    raise TypeError(f"unsupported config schema: {schema!r}")


def _is_dataclass_type(obj: object) -> TypeGuard[type[DataclassInstance]]:
    return isinstance(obj, type) and dataclasses.is_dataclass(obj)


def _is_class(tp: object) -> TypeGuard[type[object]]:
    """Whether the annotation ``tp`` is a plain class (not a generic alias or special form)."""
    return isinstance(tp, type)


def _reject_async(result: object) -> None:
    if inspect.isawaitable(result):
        close: object = getattr(result, "close", None)  # a coroutine's; other awaitables vary
        if callable(close):
            close()
        raise TypeError(_ASYNC_MSG)


def _call(fn: object, raw: object) -> object:
    """``fn(raw)``; a non-callable raises Python's own ``TypeError``."""
    if not callable(fn):
        raise TypeError(f"{type(fn).__name__!r} object is not callable")
    return fn(raw)


def _run_callable(fn: object, raw: object) -> object:
    try:
        result = _call(fn, raw)
    except ValidationError:
        raise
    except (ValueError, TypeError) as e:
        raise ValidationError([Issue(str(e))]) from e
    _reject_async(result)
    return result


def _run_pydantic(model_validate: object, raw: object) -> object:
    try:
        result = _call(model_validate, raw)
    except Exception as e:
        # a pydantic ValidationError lists its problems as dicts from errors()
        errors: object = getattr(e, "errors", None)
        if not callable(errors):
            raise
        raise ValidationError([_pydantic_issue(err) for err in _iterate(errors())]) from e
    _reject_async(result)
    return result


def _pydantic_issue(err: object) -> Issue:
    """An :class:`Issue` from one pydantic error dict (``{"msg": ..., "loc": (...)}``)."""
    message = _field(err, "msg")
    location = _field(err, "loc")
    path = tuple(_path_part(part) for part in location) if is_array(location) else ()
    return Issue(str("invalid value" if message is None else message), path)


def _path_part(part: object) -> str | int:
    return part if isinstance(part, str | int) else str(part)


def _iterate(items: object) -> list[object]:
    """The items of an iterable result field (none when it is absent or not iterable)."""
    return list(items) if is_iterable(items) else []


def _field(obj: object, name: str) -> object:
    if is_mapping(obj):
        return obj.get(name)
    return getattr(obj, name, None)  # a duck-typed result object's field, if present


def _std_issue(item: object) -> Issue:
    path: list[str | int] = []
    for seg in _iterate(_field(item, "path")):
        key = _field(seg, "key") if not isinstance(seg, (str, int)) else seg
        path.append(_path_part(key))
    message = _field(item, "message")
    return Issue("invalid value" if message is None else str(message), tuple(path))


def _run_standard(validate: object, raw: object) -> object:
    result = _call(validate, raw)
    _reject_async(result)
    issues = _field(result, "issues")
    if issues:
        raise ValidationError([_std_issue(i) for i in _iterate(issues)])
    return _field(result, "value")


def _type_name(tp: object) -> str:
    if tp is Ellipsis:
        return "..."
    if tp is type(None) or tp is None:
        return "None"
    if tp is Any:
        return "any"
    origin = get_origin(tp)
    if origin is Literal:
        return " | ".join(repr(a) for a in get_args(tp))
    if origin in (Union, types.UnionType):
        return " | ".join(_type_name(a) for a in get_args(tp))
    if origin is not None:
        args = get_args(tp)
        base = _name_of(origin)
        return f"{base}[{', '.join(_type_name(a) for a in args)}]" if args else base
    return _name_of(tp)


def _name_of(tp: object) -> str:
    """A class's ``__name__``; ``str(tp)`` for other type forms."""
    return tp.__name__ if isinstance(tp, type) else str(tp)


def _mismatch(tp: object, value: object, path: _Path, issues: list[Issue]) -> _Failed:
    issues.append(Issue(f"expected {_type_name(tp)}, got {type(value).__name__}", path))
    return _FAIL


def _check(tp: object, value: object, path: _Path, issues: list[Issue]) -> object:
    """Validate ``value`` against ``tp``; append issues and return ``_FAIL`` on failure."""
    if tp is Any:
        return value
    if tp is None or tp is type(None):
        return value if value is None else _mismatch(tp, value, path, issues)
    if tp is bool:
        return value if isinstance(value, bool) else _mismatch(tp, value, path, issues)
    if tp is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
        return value if ok else _mismatch(tp, value, path, issues)
    if tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return _mismatch(tp, value, path, issues)
    if tp is str:
        return value if isinstance(value, str) else _mismatch(tp, value, path, issues)

    origin = get_origin(tp)
    if origin is Literal:
        for a in get_args(tp):
            if type(a) is type(value) and a == value:
                return value
        return _mismatch(tp, value, path, issues)
    if origin in (Union, types.UnionType):
        members = get_args(tp)
        rest = [m for m in members if m is not type(None)]
        if len(rest) == 1 and value is not None:
            return _check(rest[0], value, path, issues)
        for member in members:
            trial: list[Issue] = []
            result = _check(member, value, path, trial)
            if result is not _FAIL:
                return result
        return _mismatch(tp, value, path, issues)
    if origin is Volatile:
        (inner,) = get_args(tp)
        result = _check(inner, value, path, issues)
        return result if result is _FAIL else create_volatile(result)
    if origin is list or tp is list:
        return _seq(tp, value, path, issues, "list")
    if origin is tuple:
        return _seq(tp, value, path, issues, "tuple")
    if origin in (dict, Mapping) or tp is dict:
        return _map(tp, value, path, issues)
    if _is_class(tp):
        if issubclass(tp, Enum):
            try:
                return tp(value)
            except ValueError:
                members = ", ".join(repr(m.value) for m in tp)
                issues.append(Issue(f"expected one of {members}, got {type(value).__name__}", path))
                return _FAIL
        if _is_dataclass_type(tp):
            return _dataclass(tp, value, path, issues)
    raise TypeError(f"unsupported annotation in config schema: {tp!r}")


def _assert_supported(tp: object) -> None:
    """Raise ``TypeError`` for annotations the validator cannot handle, regardless of input."""
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is Literal:
        return
    if origin in (Union, types.UnionType, Volatile, list, dict, Mapping):
        for a in args:
            _assert_supported(a)
        return
    if origin is tuple:
        if not (len(args) == 2 and args[1] is Ellipsis):
            raise TypeError(f"unsupported annotation in config schema: {tp!r}")
        _assert_supported(args[0])
        return
    if origin is not None or not (
        tp is Any
        or tp is None
        or (
            _is_class(tp)
            and (
                tp in (list, dict)
                or tp.__module__ == "builtins"
                or issubclass(tp, Enum)
                or _is_dataclass_type(tp)
            )
        )
    ):
        raise TypeError(f"unsupported annotation in config schema: {tp!r}")


def _seq(
    tp: object,
    value: object,
    path: _Path,
    issues: list[Issue],
    kind: Literal["list", "tuple"],
) -> object:
    args = get_args(tp)
    if kind == "tuple" and not (len(args) == 2 and args[1] is Ellipsis):
        raise TypeError(f"unsupported annotation in config schema: {tp!r}")
    if not is_array(value) or (kind == "list" and not isinstance(value, list)):
        return _mismatch(tp, value, path, issues)
    item_tp: object = args[0] if args else Any
    out: list[object] = []
    bad = False
    for i, item in enumerate(value):
        res = _check(item_tp, item, (*path, i), issues)
        bad = bad or res is _FAIL
        out.append(res)
    if bad:
        return _FAIL
    return tuple(out) if kind == "tuple" else out


def _map(tp: object, value: object, path: _Path, issues: list[Issue]) -> object:
    if not is_mapping(value):
        return _mismatch(tp, value, path, issues)
    args = get_args(tp)
    val_tp: object = args[1] if len(args) == 2 else Any
    out: dict[str, object] = {}
    bad = False
    for k, v in value.items():
        if not isinstance(k, str):
            issues.append(Issue(f"expected str key, got {type(k).__name__}", path))
            bad = True
            continue
        res = _check(val_tp, v, (*path, k), issues)
        bad = bad or res is _FAIL
        out[k] = res
    return _FAIL if bad else out


def _dataclass(
    cls: type[DataclassInstance], value: object, path: _Path, issues: list[Issue]
) -> object:
    if not is_mapping(value):
        issues.append(Issue("expected object", path))
        return _FAIL
    data = value
    hints = get_type_hints(cls, include_extras=True)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    for name in fields:
        _assert_supported(hints.get(name, Any))
    bad = False
    for key in data:
        if key not in fields:
            issues.append(Issue("unknown key", (*path, _path_part(key))))
            bad = True
    kwargs: dict[str, object] = {}
    for name, f in fields.items():
        if name not in data:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                issues.append(Issue("missing required field", (*path, name)))
                bad = True
            continue
        res = _check(hints.get(name, Any), data[name], (*path, name), issues)
        if res is _FAIL:
            bad = True
        else:
            kwargs[name] = res
    return _FAIL if bad else cls(**kwargs)
