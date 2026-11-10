"""Internal utilities shared across pycordis."""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import final, overload

from pycordis._guards import is_mapping

_MISSING: object = object()


@final
class DisposableList[T]:
    """Ordered collection with O(1) deletion by value identity (port of cordis DisposableList).

    Example: ``remove = fibers.push(fiber)``; ``remove()`` takes it out again.
    """

    __slots__ = ("_ids", "_map", "_sn")

    def __init__(self) -> None:
        self._sn = 0
        self._map: dict[int, T] = {}
        self._ids: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._map)

    @property
    def index_size(self) -> int:
        """Entries in the identity index behind :meth:`delete` (pruned with the values).

        Internal — not part of the public API.
        """
        return len(self._ids)

    def __iter__(self) -> Iterator[T]:
        return iter(self.snapshot())

    def snapshot(self) -> list[T]:
        """The values in order, as a new list (later changes do not affect it).

        Example: ``for fiber in runtime.fibers.snapshot(): ...``.
        """
        return list(self._map.values())

    def push(self, value: T) -> Callable[[], bool]:
        """Append ``value`` and return a callable that removes it, reporting success."""
        self._sn += 1
        sn = self._sn
        self._map[sn] = value
        self._ids[id(value)] = sn

        return partial(self._remove, sn, id(value))

    def unshift(self, value: T) -> Callable[[], bool]:
        """Prepend ``value`` and return a callable that removes it, reporting success."""
        self._sn += 1
        sn = self._sn
        self._map = {sn: value, **self._map}
        self._ids[id(value)] = sn

        return partial(self._remove, sn, id(value))

    def delete(self, value: T) -> bool:
        """Remove ``value`` by identity; return whether it was present."""
        sn = self._ids.get(id(value))
        return sn is not None and self._remove(sn, id(value))

    def _remove(self, sn: int, key: int) -> bool:
        if self._ids.get(key) == sn:
            del self._ids[key]
        return self._map.pop(sn, _MISSING) is not _MISSING

    def clear(self) -> list[T]:
        """Empty the list, returning the removed values in reverse insertion order."""
        values = list(self._map.values())
        self._map.clear()
        self._ids.clear()
        values.reverse()
        return values


@final
class Label:
    """Unique isolation label compared by identity."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Label({self.name!r})"


@final
class ScopeMap[V]:
    """Immutable chained mapping; lookups fall through to ancestors."""

    __slots__ = ("_own", "_parent")

    def __init__(self, own: dict[str, V], parent: ScopeMap[V] | None) -> None:
        self._own = own
        self._parent = parent

    @classmethod
    def empty(cls) -> ScopeMap[V]:
        """Create a new root map."""
        return cls({}, None)

    @property
    def parent(self) -> ScopeMap[V] | None:
        """The enclosing map, or ``None`` for a root."""
        return self._parent

    def child(self, key: str, value: V) -> ScopeMap[V]:
        """Return a new map layering ``key -> value`` over this one."""
        return ScopeMap({key: value}, self)

    def set_root(self, key: str, value: V) -> None:
        """Mutate this map's own entries (used only for the root isolate map)."""
        self._own[key] = value

    def __getitem__(self, key: str) -> V:
        node: ScopeMap[V] | None = self
        while node is not None:
            if key in node._own:
                return node._own[key]
            node = node._parent
        raise KeyError(key)

    @overload
    def get(self, key: str) -> V | None: ...
    @overload
    def get[D](self, key: str, default: D) -> V | D: ...
    def get(self, key: str, default: object = None) -> object:
        """Look up ``key`` through ancestors, returning ``default`` if absent."""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        node: ScopeMap[V] | None = self
        while node is not None:
            if key in node._own:
                return True
            node = node._parent
        return False

    def has_own(self, key: str) -> bool:
        """Whether ``key`` is defined on this map itself, ignoring ancestors."""
        return key in self._own

    def own_items(self) -> Iterator[tuple[str, V]]:
        """Iterate this map's own entries."""
        return iter(list(self._own.items()))


@final
@dataclass(frozen=True)
class _Symbols:
    """Namespace of attribute names used to avoid public name collisions."""

    invoke: str = "__cordis_invoke__"
    check: str = "__cordis_check__"
    init: str = "__cordis_init__"
    filter: str = "__cordis_filter__"
    extend: str = "__cordis_extend__"
    tracker: str = "__cordis_tracker__"
    resolve_config: str = "__cordis_resolve_config__"
    config: str = "__cordis_config__"
    init_hooks: str = "__cordis_init_hooks__"
    effect: str = "__cordis_effect__"
    metadata: str = "__cordis_metadata__"
    # attribute names of a context's isolate / intercept scope maps, so that
    # ``getattr(ctx, Context.isolate)`` mirrors cordis ``ctx[Context.isolate]``
    isolate: str = "_isolate"
    intercept: str = "_intercept"
    # def-site context carried by a shadow context, and the traced receiver of
    # an accessor read; private names, so context resolution never traces them
    shadow: str = "_shadow"
    receiver: str = "_receiver"
    # a traceable proxy's target and def-site context
    original: str = "_cordis_original"
    caller: str = "_cordis_caller"


symbols = _Symbols()

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def hyphenate(name: str) -> str:
    """Convert ``FooBar``, ``fooBar`` or ``foo_bar`` to ``foo-bar``."""
    return _CAMEL.sub("-", name).replace("_", "-").lower()


def is_bailed(value: object) -> bool:
    """Return True unless ``value`` is ``None`` or ``False`` (i.e. the hook bailed)."""
    return value is not None and value is not False


# -- futures and tasks -------------------------------------------------------------


def resolve_future(future: asyncio.Future[None]) -> None:
    """Set ``future``'s result to ``None`` unless it is already done."""
    if not future.done():
        future.set_result(None)


def done_future(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Future[None]:
    """A future already resolved to ``None`` (on the running loop by default)."""
    future: asyncio.Future[None] = (loop or asyncio.get_running_loop()).create_future()
    future.set_result(None)
    return future


def cancelling() -> bool:
    """Whether the current task itself is being cancelled."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def report_failure[T](
    future: asyncio.Future[T],
    report: Callable[[BaseException], object],
    *,
    eager: bool = False,
) -> None:
    """Pass ``future``'s exception (not a cancellation) to ``report`` once it is done.

    With ``eager``, a future that is already done is checked right away instead
    of from a done callback.
    """

    def check(done: asyncio.Future[T]) -> None:
        if not done.cancelled() and (error := done.exception()) is not None:
            report(error)

    if eager and future.done():
        check(future)
    else:
        future.add_done_callback(check)


def add_note_once(error: BaseException, note: str) -> None:
    """Add ``note`` to ``error`` unless it already carries it."""
    if note not in getattr(error, "__notes__", ()):
        error.add_note(note)


def merge_entries(configs: list[object]) -> dict[object, object]:
    """Shallow-merge ``configs`` in order: mappings by key, other objects by attribute.

    Keys are kept as they are (a mapping's non-str keys included). Example:
    ``merge_entries([{"a": 1}, {"b": 2}]) == {"a": 1, "b": 2}``.
    """
    result: dict[object, object] = {}
    for config in configs:
        if is_mapping(config):
            result.update(config)
        else:
            result.update(vars(config))
    return result


def report_unhandled(message: str, error: BaseException, **context: object) -> None:
    """Report ``error``, which no caller observes, outside the logger service.

    For the paths that cannot log through ``ctx.logger`` (the logger itself
    failed, or is unavailable): the running loop's exception handler gets
    ``message``, ``error`` and ``context``; outside a running loop it goes to
    :func:`sys.unraisablehook`. Never raises. Example::

        report_unhandled("pycordis: logger exporter Console failed", failure)
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # the hook takes a message only: keep the context in it
        details = ", ".join(f"{key}={value!r}" for key, value in context.items())
        text = f"{message} ({details})" if details else message
        sys.unraisablehook(_Unraisable(type(error), error, error.__traceback__, text))
        return
    loop.call_exception_handler({"message": message, "exception": error, **context})


@final
@dataclass
class _Unraisable:
    """The argument of :func:`sys.unraisablehook` (the ``sys.UnraisableHookArgs`` protocol)."""

    exc_type: type[BaseException]
    exc_value: BaseException | None
    exc_traceback: TracebackType | None
    err_msg: str | None
    object: object = None
