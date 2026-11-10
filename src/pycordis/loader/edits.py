"""Entry-level edits of the loader tree: ``create``, ``update`` and ``remove``.

Each applies its change to the in-memory tree synchronously, persists it to
the file the entry is declared in, then waits for the affected fibers to
settle (see :class:`~pycordis.loader.Loader` for the public contract).

Internal — use :meth:`Loader.create <pycordis.loader.Loader.create>`,
:meth:`~pycordis.loader.Loader.update` and :meth:`~pycordis.loader.Loader.remove`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from pycordis._guards import RawTable
from pycordis.loader.entry import Entry
from pycordis.loader.files import parse_entry
from pycordis.loader.persist import (
    REFUSED,
    AppendEdit,
    committed_config,
    declaration,
    leak_target,
    new_config_leaks,
    relocate,
    remove_edit,
)
from pycordis.loader.tree import LoaderTree, Node, secret_message
from pycordis.loader.writeback import shift, write_keys

if TYPE_CHECKING:
    from pycordis.fiber import Fiber

_PATCHABLE = frozenset(
    {
        *("use", "config", "disabled", "isolate", "intercept", "inject"),
        *("restart", "load_timeout", "dispose_timeout"),
    }
)


def _target(tree: LoaderTree, parent: Node | None) -> tuple[Path, tuple[str | int, ...]]:
    """The file and location prefix new children of ``parent`` are declared at."""
    if parent is None:
        return tree.path.resolve(), ("plugins",)
    entry = parent.entry
    if entry.include is not None:
        return (entry.source.parent / entry.include).resolve(), ("plugins",)
    return entry.source, (*entry.location, "plugins")


async def create_entry(
    tree: LoaderTree, raw: Mapping[str, object], parent_id: str | None
) -> Fiber | None:
    """Add the entry ``raw`` under ``parent_id`` (or the top level) and start it.

    Example: ``fiber = await create_entry(tree, {"use": "app:Echo"}, None)``.
    The contract is :meth:`Loader.create <pycordis.loader.Loader.create>`.
    """
    tree.ctx.fiber.assert_active()
    parent = None if parent_id is None else tree.lookup(parent_id)
    if parent is not None and parent.entry.use is not None:
        raise ValueError(f'entry "{parent_id}" is not a group')
    siblings = tree.top if parent is None else parent.entry.children
    source, base = _target(tree, parent)
    # its index once appended (siblings an overlay adds live in another file)
    index = sum(1 for e in siblings if e.source == source and e.location[:-1] == base)
    entry = parse_entry(
        raw,
        source=source,
        location=(*base, index),
        parent_id=parent_id or "",
        siblings=siblings,
        existing=[node.entry for node in tree.nodes.values()],
    )
    if entry.use is not None and new_config_leaks(entry.config, source.parent):
        _refuse_secret(tree, source, entry.id)
    scope = None if entry.disabled else tree.ctx if parent is None else parent.scope
    siblings.append(entry)
    node = tree.register(entry, parent)
    if scope is None:
        _persist_create(tree, entry, raw, parent, source, base)
        return None  # starts once enabled, or when its group activates
    try:
        fiber = tree.launch(scope, node, tree.plugin_of(node))
    except BaseException:
        # import or plugin errors: the entry is not added
        siblings[:] = [e for e in siblings if e is not entry]
        tree.drop(node)
        raise
    _persist_create(tree, entry, raw, parent, source, base)
    await tree.settle([node])
    return fiber


def _refuse_secret(tree: LoaderTree, source: Path, id: str) -> NoReturn:
    """Log and raise the refusal of a write that would persist a secret."""
    message = secret_message(source, id)
    tree.logger.error(message)
    raise ValueError(message)


def _persist_create(
    tree: LoaderTree,
    entry: Entry,
    raw: Mapping[str, object],
    parent: Node | None,
    source: Path,
    base: tuple[str | int, ...],
) -> None:
    check = parent.entry if parent is not None and parent.entry.include is None else None
    edit = AppendEdit(base, dict(raw), check)
    if tree.persist(source, edit):
        relocate(entry, (*base, edit.index))


async def update_entry(tree: LoaderTree, id: str, patch: Mapping[str, object]) -> None:
    """Apply ``patch`` to the entry ``id`` and reconcile it.

    Example: ``await update_entry(tree, "db", {"disabled": True})``.
    The contract is :meth:`Loader.update <pycordis.loader.Loader.update>`.
    """
    tree.ctx.fiber.assert_active()
    node = tree.lookup(id)
    for key in patch:
        if key not in _PATCHABLE:
            raise ValueError(f'cannot patch "{key}"')
    old = node.entry
    config = patch.get("config")
    if config is not None and old.use is not None:
        # an expanded value gets its placeholder back, like a commit
        committed = committed_config(old, config, node.expansions)
        if committed is REFUSED:
            _refuse_secret(tree, leak_target(old, config, node.expansions), old.id)
        patch = {**patch, "config": committed}
    raw = _raw(old)
    for key, value in patch.items():
        if value is None:
            raw.pop(key, None)
        else:
            raw[key] = value
    parent = node.parent
    excluded = {n.entry.id for n in tree.subtrees([node])}
    new = parse_entry(
        raw,
        source=old.source,
        location=old.location,
        parent_id="" if parent is None else parent.entry.id,
        existing=[n.entry for n in tree.nodes.values() if n.entry.id not in excluded],
    )
    new.explicit_id = old.explicit_id
    new.overlay = old.overlay  # the same declarations; write_keys keeps them current
    if old.use is None:
        new = replace(new, group=old.group, include=old.include, children=old.children)
    siblings = tree.top if parent is None else parent.entry.children
    siblings[next(i for i, e in enumerate(siblings) if e is old)] = new
    tree.change(tree.ctx if parent is None else parent.scope, node, new)

    write_keys(tree, old, patch)
    await tree.settle([node])


async def remove_entry(tree: LoaderTree, id: str) -> None:
    """Remove the entry ``id`` with its subtree and dispose its fiber.

    Example: ``await remove_entry(tree, "db")``.
    The contract is :meth:`Loader.remove <pycordis.loader.Loader.remove>`.
    """
    tree.ctx.fiber.assert_active()
    node = tree.lookup(id)
    parent = node.parent
    siblings = tree.top if parent is None else parent.entry.children
    entry = node.entry
    doomed = _declarations(tree, node)
    siblings[:] = [e for e in siblings if e is not entry]
    tree.drop(node)

    for source, declarations in doomed.items():
        if tree.persist(source, remove_edit(declarations)):
            for removed, overlay in declarations:
                shift(tree, source, declaration(removed, overlay))
    await tree.settle(())


def _declarations(tree: LoaderTree, node: Node) -> dict[Path, list[tuple[Entry, bool]]]:
    """The tables to delete per file to remove ``node``: ``(entry, overlay)`` pairs.

    That is the entry's own declaration and the profile overlay tables of the
    entries of its subtree declared in the same file (an include's children
    stay in the included file), each list in descending location order and
    without tables nested in another one deleted anyway.
    """
    root = node.entry
    found: dict[Path, list[tuple[Entry, bool]]] = {root.source: [(root, False)]}
    for sub in tree.subtrees([node]):
        entry = sub.entry
        if entry.overlay is not None and (sub is node or entry.source == root.source):
            found.setdefault(entry.overlay.source, []).append((entry, True))
    result: dict[Path, list[tuple[Entry, bool]]] = {}
    for source, items in found.items():
        locations = {declaration(e, overlay) for e, overlay in items}
        outermost = [
            (e, overlay)
            for e, overlay in items
            if not any(_inside(declaration(e, overlay), other) for other in locations)
        ]
        outermost.sort(key=lambda item: declaration(*item), reverse=True)
        result[source] = outermost
    return result


def _inside(location: tuple[str | int, ...], other: tuple[str | int, ...]) -> bool:
    """Whether ``location`` lies strictly inside the table at ``other``."""
    return len(location) > len(other) and location[: len(other)] == other


def _raw(entry: Entry) -> RawTable:
    """The entry's own keys as a config table (groups and includes as a placeholder group)."""
    raw: RawTable = {"id": entry.id}
    if entry.use is not None:
        raw["use"] = entry.use
        if entry.config is not None:
            raw["config"] = entry.config
    else:
        raw["group"] = entry.group or "include"
    if entry.disabled:
        raw["disabled"] = True
    if entry.isolate:
        labels = list(entry.isolate.values())
        raw["isolate"] = list(entry.isolate) if None in labels else dict(entry.isolate)
    if entry.intercept:
        raw["intercept"] = dict(entry.intercept)
    if entry.inject:
        raw["inject"] = list(entry.inject)
    if entry.restart is not None:
        raw["restart"] = dict(entry.restart)
    if entry.load_timeout is not None:
        raw["load_timeout"] = entry.load_timeout
    if entry.dispose_timeout is not None:
        raw["dispose_timeout"] = entry.dispose_timeout
    return raw
