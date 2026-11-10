"""The loader's entry tree: its state, ownership hooks and reconciliation engine.

:class:`LoaderTree` holds one :class:`Node` per entry id and reconciles the
running fibers with a new entry list (see the :mod:`pycordis.loader.service`
docstring for the diff rules). It also owns the loader's event listeners
(ownership of entry fibers, config interpolation, write-back of commits), the
file watcher and the file writer. The entry-level API mutations
(``create`` / ``update`` / ``remove``) live in :mod:`pycordis.loader.edits`.

Internal — use :class:`~pycordis.loader.Loader`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, final

from pycordis._utils import add_note_once
from pycordis.events import EventOptions
from pycordis.loader.entry import Entry, LoaderOptions
from pycordis.loader.files import LoaderConfigError, load_tree
from pycordis.loader.interpolate import Expansions, expand
from pycordis.loader.overlay import overlay_path
from pycordis.loader.persist import (
    REFUSED,
    Edit,
    Writer,
    committed_config,
    leak_target,
    restore_placeholders,
)
from pycordis.loader.resolve import import_plugin
from pycordis.loader.watch import FileWatcher
from pycordis.loader.writeback import write_keys
from pycordis.registry import plugin_supervision

if TYPE_CHECKING:
    from pycordis.context import Context
    from pycordis.fiber import Fiber
    from pycordis.loader.service import LoaderConfig
    from pycordis.logger import Logger

_STRUCTURAL = (
    *("use", "group", "include", "isolate", "intercept", "inject"),
    *("restart", "load_timeout", "dispose_timeout"),
)
_MAX_SETTLE = 1000


@final
class Node:
    """Loader bookkeeping for one entry id.

    Example: ``tree.nodes["db"].fiber`` is the live fiber of entry ``db``.

    Internal — not part of the public API.
    """

    __slots__ = ("entry", "expansions", "fiber", "parent", "scope")

    def __init__(self, entry: Entry, parent: Node | None) -> None:
        self.entry = entry
        self.parent = parent
        self.fiber: Fiber | None = None
        # groups: the context the children run in, while the group's body is loaded
        self.scope: Context | None = None
        # every interpolation result seen for this entry id, kept for its lifetime
        self.expansions = Expansions()


@final
class LoaderTree:
    """The loader's state and reconciliation engine (kept off the traced service).

    Example: ``tree.apply(tree.load())`` reconciles the running fibers with the file.

    Internal — not part of the public API.
    """

    def __init__(self, ctx: Context, config: LoaderConfig) -> None:
        self.ctx = ctx
        self.config = config
        self.path = Path(config.path)
        self.options = LoaderOptions()
        self.logger: Logger = ctx.logger("loader")
        self.top: list[Entry] = []
        # set once ``Loader.start()`` applied the loaded file tree
        self.started = False
        self.nodes: dict[str, Node] = {}
        self.owned: dict[Fiber, Node] = {}
        self.creating: tuple[Context, Node, object] | None = None
        self.waits: list[asyncio.Future[None]] = []
        # files the last load attempt read or looked for (watched with the tree's includes)
        self.watched: list[Path] = []
        self.watcher: FileWatcher | None = None
        self.watch_task: asyncio.Task[None] | None = None
        self.writer = Writer(self.logger)
        ctx.on("internal/plugin", self._on_plugin)
        ctx.on("internal/commit", self._on_commit)
        # first: put placeholders back into the raw config an update just retained
        # (global: a plain internal/update listener only sees its own fiber's updates)
        ctx.on("internal/update", self._on_update, EventOptions(prepend=True, global_=True))
        # outermost: annotate every config error of an owned fiber with its entry id
        ctx.on("internal/config", self._annotate_config, True)
        # innermost: expand the raw config before the schema validates it
        ctx.on("internal/config", self._expand_config)
        # the loader's fiber unloading disposes every entry fiber with it
        ctx.effect(lambda: self._detach, "loader")

    def _detach(self) -> None:
        """Forget every fiber when the loader's fiber unloads (they are disposed with it)."""
        self.owned.clear()
        self.watch_task = self.watcher = None  # the fiber cancelled the task
        for node in self.nodes.values():
            node.fiber = None
            node.scope = None

    # -- listeners ------------------------------------------------------------

    def _on_plugin(self, fiber: Fiber) -> None:
        if fiber.uid is None:  # disposed
            node = self.owned.pop(fiber, None)
            if node is not None and node.fiber is fiber:
                node.fiber = None
            return
        creating = self.creating
        if (
            creating is None
            or fiber.parent is not creating[0]
            or fiber.runtime is None
            or fiber.runtime.callback != fiber.ctx.registry.resolve(creating[2])
        ):
            return
        self.creating = None
        node = creating[1]
        for name in node.entry.inject:
            fiber.inject.setdefault(name, None)
        fiber.notes.append(_entry_note(node.entry.id))
        self.owned[fiber] = node

    def _on_commit(self, fiber: Fiber, config: object) -> None:
        node = self.owned.get(fiber)
        if node is None or node.entry.use is None:
            return
        entry = node.entry
        # a plugin passing its expanded config gets the file's placeholders back
        committed = committed_config(entry, config, node.expansions)
        if committed is REFUSED:
            target = leak_target(entry, config, node.expansions)
            self.logger.error(secret_message(target, entry.id))
            return
        config = committed
        entry.config = config  # a reload of the saved file then sees no change
        write_keys(self, entry, {"config": config})

    def _annotate_config(self, fiber: Fiber, next: Callable[[], object]) -> object:
        node = self.owned.get(fiber)
        if node is None:
            return next()
        try:
            return next()
        except Exception as error:
            _note(error, node.entry.id)
            raise

    def _on_update(self, fiber: Fiber, resolved: object, next: Callable[[], object]) -> object:
        node = self.owned.get(fiber)
        if node is not None:
            self._retemplate_raw(fiber, node)
        return next()

    def _expand_config(self, fiber: Fiber, next: Callable[[], object]) -> object:
        config = next()
        node = self.owned.get(fiber)
        if node is None:
            return config
        raw = config
        # expanded values handed back by the plugin are not interpolated again
        config = restore_placeholders(raw, node.entry, node.expansions)
        if config is not raw and raw is fiber.config:
            # resolving the fiber's own raw config (not an update candidate,
            # which must leave it untouched should it fail)
            fiber.config = config
        return expand(config, node.entry.source.parent, track=node.expansions)

    def _retemplate_raw(self, fiber: Fiber, node: Node) -> None:
        """Put the entry's placeholders back into ``fiber``'s retained raw config."""
        raw = fiber.config
        config = restore_placeholders(raw, node.entry, node.expansions)
        if config is not raw:
            fiber.config = config  # the raw input only: no revision, no restart

    # -- file tree ------------------------------------------------------------

    def load(self) -> list[Entry]:
        """Read the file tree and refresh :attr:`options`."""
        self.watched = []
        entries, options = load_tree(self.path, files=self.watched, profile=self.config.profile)
        if self.config.watch is not None:
            options.watch = self.config.watch
        if self.config.allow is not None:
            options.allow = list(self.config.allow)
        if self.config.readonly is not None:
            options.readonly = self.config.readonly
        self.options = options
        return entries

    async def reload(self) -> None:
        self.ctx.fiber.assert_active()
        try:
            entries = self.load()
        except (LoaderConfigError, RuntimeError) as error:
            self.logger.error("config reload failed: %s", error)
            return
        self.apply(entries)
        self.watch()
        await self.settle()

    # -- file watching --------------------------------------------------------

    def watch(self) -> None:
        """Start, retune or stop the file watcher to follow :attr:`options`."""
        task = self.watch_task
        running = task is not None and not task.done()
        if not self.options.watch:
            if task is not None and running:
                task.cancel()
            self.watch_task = self.watcher = None
        elif running and self.watcher is not None:
            self.watcher.interval = self.options.poll_interval
        else:
            watcher = FileWatcher(self.watch_paths, self.options.poll_interval, self.hot_reload)
            self.watcher = watcher
            self.watch_task = self.ctx.spawn(watcher.run(), name="loader watcher")

    def watch_paths(self) -> list[Path]:
        """The root file, the files of the last load attempt, the tree's includes and overlays."""
        files = [self.path.resolve()]
        for node in self.nodes.values():
            entry = node.entry
            if entry.include is not None:
                files.append((entry.source.parent / entry.include).resolve())
        profile = self.config.profile
        overlays = [] if profile is None else [overlay_path(file, profile) for file in files]
        return [*files, *overlays, *self.watched]

    async def hot_reload(self) -> None:
        try:
            await self.reload()
        except Exception as error:
            self.logger.error(error)

    # -- persistence ----------------------------------------------------------

    def persist(self, source: Path, edit: Edit, *, strict: bool = False) -> bool:
        """Apply ``edit`` to ``source`` and write it (see :meth:`Writer.write`)."""
        return self.writer.write(
            source, edit, readonly=self.options.readonly, watcher=self.watcher, strict=strict
        )

    def walk(self) -> list[Node]:
        """Nodes of the tree, depth first in file order."""
        result: list[Node] = []

        def visit(entries: list[Entry]) -> None:
            for entry in entries:
                node = self.nodes.get(entry.id)
                if node is not None and node.entry is entry:
                    result.append(node)
                    visit(entry.children)

        visit(self.top)
        return result

    def lookup(self, id: str) -> Node:
        node = self.nodes.get(id)
        if node is None:
            raise KeyError(f'unknown loader entry "{id}"')
        return node

    # -- reconciliation -------------------------------------------------------

    def apply(self, entries: list[Entry]) -> None:
        """Reconcile the running tree with ``entries``."""
        parents: dict[str, str | None] = {}

        def index(items: list[Entry], parent: str | None) -> None:
            for entry in items:
                parents[entry.id] = parent
                index(entry.children, entry.id)

        def prune(items: list[Entry], parent: str | None) -> None:
            for entry in items:
                node = self.nodes[entry.id]
                if entry.id not in parents or parents[entry.id] != parent:
                    self.drop(node)  # removed, or moved to another group
                else:
                    prune(entry.children, entry.id)

        index(entries, None)
        prune(self.top, None)
        self.top = entries
        self.sync(self.ctx, None, entries)

    def sync(self, scope: Context | None, parent: Node | None, entries: list[Entry]) -> None:
        """Bring the children of ``parent`` in line with ``entries`` (removals already done).

        ``scope`` is the context children run in, ``None`` while the group is inactive.
        """
        for entry in entries:
            node = self.nodes.get(entry.id)
            if node is not None and node.parent is parent:
                self.change(scope, node, entry)
            else:
                node = self.register(entry, parent)
                if scope is not None and not entry.disabled:
                    self.start(scope, node)

    def change(self, scope: Context | None, node: Node, new: Entry) -> None:
        old = node.entry
        node.entry = new
        if old == new:
            self.rebind(node, new)
            return
        if scope is None or new.disabled:
            self.stop(node)
            self.replace_children(node, old, new)
        elif (
            node.fiber is None
            or old.disabled
            or any(getattr(old, key) != getattr(new, key) for key in _STRUCTURAL)
        ):
            self.stop(node)
            self.replace_children(node, old, new)
            self.start(scope, node)
        elif new.use is None:
            self.sync(node.scope, node, new.children)
        elif old.config != new.config:
            self.reconfigure(node, new.config)

    def rebind(self, node: Node, entry: Entry) -> None:
        """Point ``node``'s subtree at the equal entries of ``entry`` (identity used by walk)."""
        node.entry = entry
        for child in entry.children:
            sub = self.nodes.get(child.id)
            if sub is not None and sub.parent is node:
                self.rebind(sub, child)

    def register(self, entry: Entry, parent: Node | None) -> Node:
        node = Node(entry, parent)
        self.nodes[entry.id] = node
        for child in entry.children:
            self.register(child, node)
        return node

    def children(self, node: Node, entries: Iterable[Entry]) -> list[Node]:
        result: list[Node] = []
        for entry in entries:
            child = self.nodes.get(entry.id)
            if child is not None and child.parent is node:
                result.append(child)
        return result

    def replace_children(self, node: Node, old: Entry, new: Entry) -> None:
        for child in self.children(node, old.children):
            self.drop(child)
        for entry in new.children:
            self.register(entry, node)

    def drop(self, node: Node) -> None:
        """Dispose ``node``'s fiber and forget it with its subtree."""
        self.stop(node)
        for child in self.children(node, node.entry.children):
            self.drop(child)
        if self.nodes.get(node.entry.id) is node:
            del self.nodes[node.entry.id]

    def stop(self, node: Node) -> None:
        fiber = node.fiber
        node.fiber = None
        node.scope = None
        if fiber is not None:
            self.waits.append(fiber.dispose())

    def start(self, scope: Context, node: Node) -> None:
        """Create ``node``'s fiber, logging (not raising) a failure."""
        try:
            self.launch(scope, node, self.plugin_of(node))
        except Exception as error:
            self.report(node, error)

    def plugin_of(self, node: Node) -> object:
        entry = node.entry
        if entry.use is not None:
            return import_plugin(entry.use, self.options.allow)
        name = f"group:{entry.group}" if entry.group is not None else f"include:{entry.include}"

        def apply(ctx: Context, _config: object) -> Callable[[], None]:
            node.scope = ctx
            for child in self.children(node, node.entry.children):
                if not child.entry.disabled:
                    self.start(ctx, child)

            def leave() -> None:
                if node.scope is ctx:
                    node.scope = None

            return leave

        return SimpleNamespace(name=name, apply=apply)

    def launch(self, scope: Context, node: Node, plugin: object) -> Fiber:
        entry = node.entry
        ctx = scope
        for name, label in entry.isolate.items():
            ctx = ctx.isolate(name, label)
        for name, config in entry.intercept.items():
            ctx = ctx.intercept(name, config)
        dispose_timeout = entry.dispose_timeout
        if dispose_timeout is None and plugin_supervision(plugin).dispose_timeout is None:
            dispose_timeout = self.config.dispose_timeout
        self.creating = (ctx, node, plugin)
        try:
            fiber = ctx.plugin(
                plugin,
                entry.config,
                policy=entry.policy,
                load_timeout=entry.load_timeout,
                dispose_timeout=dispose_timeout,
            )
        finally:
            self.creating = None
        node.fiber = fiber
        return fiber

    def reconfigure(self, node: Node, config: object) -> None:
        fiber = node.fiber
        assert fiber is not None
        try:
            self.waits.append(fiber.update(config, no_save=True))
        except Exception as error:
            self.report(node, error)

    def report(self, node: Node, error: Exception) -> None:
        _note(error, node.entry.id)
        self.logger.error(error)

    async def settle(self, scope: Iterable[Node] | None = None) -> None:
        """Wait for pending disposals and updates, then for owned fibers' transitions.

        ``scope`` limits the fibers waited for to those subtrees (default: all).
        """
        roots = None if scope is None else list(scope)
        for _ in range(_MAX_SETTLE):
            pending: set[asyncio.Future[None]] = set(self.waits)
            self.waits.clear()
            nodes = self.nodes.values() if roots is None else self.subtrees(roots)
            for node in nodes:
                if node.fiber is not None and node.fiber.inertia is not None:
                    pending.add(node.fiber.inertia)
            if not pending:
                return
            await asyncio.wait(pending)
            for future in pending:
                if not future.cancelled():
                    future.exception()  # reported where it happened
        self.logger.warn("loader tree did not settle")

    def subtrees(self, roots: list[Node]) -> list[Node]:
        result: list[Node] = []
        stack = list(roots)
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(self.children(node, node.entry.children))
        return result


def secret_message(source: Path, id: str) -> str:
    """The refusal logged for a write that would persist an interpolated secret.

    Example: ``secret_message(Path("app.toml"), "db")``.

    Internal — not part of the public API.
    """
    return f'cannot save {source}: would persist an interpolated secret (entry "{id}")'


def _entry_note(id: str) -> str:
    return f'in loader entry "{id}"'


def _note(error: BaseException, id: str) -> None:
    add_note_once(error, _entry_note(id))
