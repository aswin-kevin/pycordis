"""Per-target-type proxy classes defining exactly the target's special methods.

Python looks special methods up on the type, so each proxy of
:mod:`pycordis.trace` is an instance of a subclass, created here and cached
weakly by target type, that forwards the special methods the target's type
overrides from ``object`` (see the :mod:`pycordis.trace` docstring for the
list and the omissions).

Internal — not part of the public API.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any, cast

FORWARDED = (
    *("__enter__", "__exit__", "__aenter__", "__aexit__", "__await__"),
    *("__iter__", "__next__", "__aiter__", "__anext__", "__reversed__"),
    *("__len__", "__length_hint__", "__contains__"),
    *("__getitem__", "__setitem__", "__delitem__"),
    *("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__bool__", "__format__"),
    *("__neg__", "__pos__", "__abs__", "__invert__", "__round__", "__trunc__"),
    *("__floor__", "__ceil__", "__int__", "__float__", "__complex__", "__index__"),
    *("__bytes__", "__fspath__"),
    *(
        f"__{prefix}{op}__"
        for op in (
            *("add", "sub", "mul", "matmul", "truediv", "floordiv", "mod", "divmod"),
            *("pow", "lshift", "rshift", "and", "xor", "or"),
        )
        for prefix in ("", "r", "i")
        if not (prefix == "i" and op == "divmod")
    ),
)
"""Special methods forwarded when the target's type overrides them."""

type Unwrap = Callable[[object], object]
"""Returns the object a proxy finally wraps (used in error messages)."""


def _special(obj: object, name: str, unwrap: Unwrap) -> Any:  # the target's method, any signature
    try:
        return getattr(obj, name)
    except AttributeError:
        raise TypeError(f"{type(unwrap(obj)).__name__!r} object has no {name}") from None


def _forwarder(name: str, unwrap: Unwrap) -> Callable[..., object]:
    def forward(self: object, *args: object) -> object:
        return _special(self, name, unwrap)(*args)

    forward.__name__ = forward.__qualname__ = name
    return forward


type _Key = tuple[type, Callable[..., object] | None, Unwrap]

_special_types: weakref.WeakKeyDictionary[type, dict[_Key, type]] = weakref.WeakKeyDictionary()


def special_type[B](
    base: type[B], cls: type, call: Callable[..., object] | None, unwrap: Unwrap
) -> type[B]:
    """The subclass of ``base`` defining the special methods ``cls`` overrides.

    Each forwarder reads the method through the proxy (so it sees the traced
    receiver); ``call`` becomes ``__call__`` when given. The result is cached
    per target type under ``(base, call, unwrap)``, so a different ``call`` or
    ``unwrap`` never reuses a subclass built for another.

    Example: ``special_type(Traceable, type(target), None, original)``.
    """
    per_type = _special_types.get(cls)
    if per_type is None:
        per_type = _special_types[cls] = {}
    key = (base, call, unwrap)
    cached = per_type.get(key)
    if cached is not None:
        return cast(type[B], cached)  # cached under ``base`` above, so a subclass of it
    namespace: dict[str, object] = {"__slots__": ()}
    for name in FORWARDED:
        attr = getattr(cls, name, None)
        if attr is not None and attr is not getattr(object, name, None):
            namespace[name] = _forwarder(name, unwrap)
    target_hash = getattr(cls, "__hash__", None)
    if target_hash is None:
        namespace["__hash__"] = None
    elif "__eq__" in namespace or target_hash is not object.__hash__:
        namespace["__hash__"] = _forwarder("__hash__", unwrap)
    if call is not None:
        namespace["__call__"] = call
    # three-argument type() is typed as returning plain ``type``; it is a subclass of base
    created = cast(type[B], type(base.__name__, (base,), namespace))
    per_type[key] = created
    return created
