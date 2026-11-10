"""Lifecycle-bound registries: named items contributed by plugins.

A :class:`Collection` is a service holding an ordered, name-unique set of
items. Each item is owned by the fiber that added it and is removed when that
fiber unloads, so a tool registry, route table or command list never holds
entries from plugins that are gone::

    TOOLS = ServiceKey[Collection[Tool]]("tools")

    await ctx.plugin(Collection.of(TOOLS))

    @plugin_meta(inject=[TOOLS])
    def search(ctx: Context, config: None) -> None:
        ctx.use(TOOLS).add(SearchTool(), name="search")

Every add and removal is announced as a :class:`CollectionChange` on the event
``collection/<name>`` (typed as :meth:`Collection.changes`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, final

from pycordis.errors import CordisError
from pycordis.events import EventKey
from pycordis.fiber import emit_reported
from pycordis.keys import ServiceKey
from pycordis.registry import plugin_meta
from pycordis.service import Service

if TYPE_CHECKING:
    from pycordis.context import Context
    from pycordis.effect import EffectDisposer

__all__ = ["Collection", "CollectionChange"]


@final
@dataclass(frozen=True, slots=True)
class CollectionChange[T]:
    """An item added to or removed from a :class:`Collection`.

    Emitted as ``collection/<name>``; listen with the typed key::

        def on_change(change: CollectionChange[Tool]) -> None:
            print(change.kind, change.name)

        ctx.on(Collection.changes(TOOLS), on_change)
    """

    kind: Literal["add", "remove"]
    """Whether the item was added or removed."""
    name: str
    """The item's name in the collection."""
    item: T
    """The item itself."""


@final
@dataclass(frozen=True, slots=True)
class _Entry[T]:
    item: T
    owner: str


@final
class _Entries[T]:
    """The mutable state of a collection, shared by every traced view of it."""

    def __init__(self, service: Collection[T]) -> None:
        self.service = service
        self.event = f"collection/{service.name}"
        self.by_name: dict[str, _Entry[T]] = {}
        self.serial = 0
        self.active = True

    def close(self) -> None:
        """Deactivate and drop every remaining item, announcing each removal.

        Items left at this point belong to contributors that did not inject
        the key (they outlive the provider); a stale reference must not keep
        serving them. Their own disposers then find nothing to remove.
        """
        self.active = False
        dropped = list(self.by_name.items())
        self.by_name.clear()
        for name, entry in dropped:
            self.announce("remove", name, entry.item)

    def discard(self, name: str, entry: _Entry[T]) -> None:
        """Remove ``entry`` (stored as ``name``) and announce it, unless already gone."""
        if self.by_name.get(name) is entry:
            del self.by_name[name]
            self.announce("remove", name, entry.item)

    def announce(self, kind: Literal["add", "remove"], name: str, item: T) -> None:
        # Emitted from the provider's context with the service as receiver, so
        # the service's isolation filter keeps an isolated collection's changes
        # inside its scope. Listener errors are reported, never raised into the
        # add or removal that triggered them.
        change = CollectionChange(kind, name, item)
        emit_reported(self.service.ctx, self.service, self.event, change)

    def next_name(self, owner: str) -> str:
        while True:
            self.serial += 1
            name = f"{owner}#{self.serial}"
            if name not in self.by_name:
                return name


class Collection[T](Service[None]):
    """An ordered, name-unique registry whose items unload with their contributors.

    Provide one with ``ctx.plugin(Collection.of(KEY))`` (or construct it as
    ``Collection(ctx, KEY)`` to tie it to ``ctx``'s fiber), then add items from
    any plugin that injects ``KEY``::

        TOOLS = ServiceKey[Collection[Tool]]("tools")
        await ctx.plugin(Collection.of(TOOLS))

        tools = ctx.use(TOOLS)
        dispose = tools.add(Tool("web"), name="web")
        [tool for tool in tools]  # insertion order
        await dispose()  # or let the adding plugin unload

    Iteration and :meth:`items` work on a snapshot, so removing items while
    iterating is safe.

    Contributors must inject the key (``plugin_meta(inject=[KEY])``) for their
    items to come back after the provider restarts: the injection reloads them
    against the new collection, while items added without it are lost with
    the old one. When the provider unloads, the collection drops every item
    still in it (announcing each ``remove``), so a stale reference is empty.
    """

    def __init__(self, ctx: Context, key: ServiceKey[Collection[T]]) -> None:
        super().__init__(ctx, key.name)
        entries = _Entries[T](self)
        self._entries = entries
        ctx.effect(lambda: entries.close, f"Collection({key.name!r})")

    @staticmethod
    def of[U](key: ServiceKey[Collection[U]]) -> Callable[[Context, None], None]:
        """A plugin (named ``collection:<name>``) that provides an empty collection as ``key``.

        Example::

            fiber = await ctx.plugin(Collection.of(TOOLS))
        """

        @plugin_meta(name=f"collection:{key.name}", provide=key)
        def collection(ctx: Context, config: None) -> None:
            Collection(ctx, key)

        return collection

    @staticmethod
    def changes[U](key: ServiceKey[Collection[U]]) -> EventKey[[CollectionChange[U]], None]:
        """The typed event announcing changes to the collection named by ``key``.

        Example::

            ctx.on(Collection.changes(TOOLS), lambda change: print(change.name))
        """
        return EventKey[[CollectionChange[U]], None](f"collection/{key.name}")

    def add(self, item: T, *, name: str | None = None) -> EffectDisposer:
        """Add ``item`` as ``name``, owned by the calling fiber; return its disposer.

        The item is removed when the returned disposer is called (later calls do
        nothing), when the calling fiber unloads or when the collection's
        provider unloads. ``name`` defaults to
        ``"<fiber name>#<n>"``. Raises ``ValueError`` when ``name`` is taken,
        and ``CordisError("INACTIVE_EFFECT")`` when the collection's provider
        has unloaded or the calling fiber is unloading. Change listeners that
        raise are reported through the logger, never raised here. Example::

            dispose = ctx.use(TOOLS).add(Tool("web"), name="web")
            await dispose()
        """
        entries = self._entries
        if not entries.active:
            raise CordisError("INACTIVE_EFFECT", f'collection "{self.name}" is no longer active')
        owner = self.ctx.fiber.name
        if name is None:
            name = entries.next_name(owner)
        existing = entries.by_name.get(name)
        if existing is not None:
            raise ValueError(
                f'collection "{self.name}" already has "{name}" '
                f"(added by {existing.owner}); cannot add it from {owner}"
            )
        key = name
        entry = _Entry(item, owner)

        def register() -> Callable[[], None]:
            entries.by_name[key] = entry
            return lambda: entries.discard(key, entry)

        dispose = self.ctx.effect(register, f"ctx.{self.name}.add()")
        entries.announce("add", key, item)
        return dispose

    def get(self, name: str) -> T | None:
        """The item named ``name``, or ``None``. Example: ``ctx.use(TOOLS).get("web")``."""
        entry = self._entries.by_name.get(name)
        return None if entry is None else entry.item

    def items(self) -> list[tuple[str, T]]:
        """A snapshot of ``(name, item)`` pairs in insertion order.

        Example: ``dict(ctx.use(TOOLS).items())``.
        """
        return [(name, entry.item) for name, entry in self._entries.by_name.items()]

    def __iter__(self) -> Iterator[T]:
        """Iterate over a snapshot of the items in insertion order."""
        return iter([entry.item for entry in self._entries.by_name.values()])

    def __len__(self) -> int:
        """The number of items."""
        return len(self._entries.by_name)

    def __contains__(self, name: object) -> bool:
        """Whether an item named ``name`` exists: ``"web" in ctx.use(TOOLS)``."""
        return name in self._entries.by_name

    def __repr__(self) -> str:
        return f"Collection({self.name!r}, {list(self._entries.by_name)!r})"
