"""Writing loader entries back to the files that define their values.

Without a profile an entry's values all live in the file declaring it. With a
profile overlay (:class:`~pycordis.loader.entry.Overlay`), each value goes to
the file that defines it: :func:`~pycordis.loader.overlay.split_keys` splits
the merged values, both shares are written separately and the overlay's
recorded tables follow. Used by the tree's commit listener, the entry edits
and :meth:`Loader.save <pycordis.loader.Loader.save>`.

Internal — use :class:`~pycordis.loader.Loader`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from pycordis._guards import RawTable
from pycordis.loader.entry import Entry
from pycordis.loader.overlay import record, shifted, split_keys
from pycordis.loader.persist import meaningful, patch_edit, persisted, save_edit

if TYPE_CHECKING:
    from pycordis.loader.tree import LoaderTree

__all__ = ["save_tree", "shift", "write_keys"]

type _Write = tuple[Entry, bool, RawTable]
"""``(entry, overlay, keys)``: keys to write to an entry's base or overlay declaration."""


def write_keys(tree: LoaderTree, entry: Entry, values: Mapping[str, object]) -> None:
    """Write ``entry``'s merged ``values`` (``None``: remove) to the files defining them.

    Without a profile overlay that is the entry's own file; otherwise
    :func:`~pycordis.loader.overlay.split_keys` decides, and each file is only
    touched when its share changed. Errors are logged (see :meth:`LoaderTree.persist`);
    the overlay's recorded tables follow only the shares that were written, so a
    failed (or ``readonly``) write leaves the provenance as the files have it.
    Example: ``write_keys(tree, entry, {"config": config})``.
    """
    overlay = entry.overlay
    if overlay is None:
        tree.persist(entry.source, patch_edit(entry, values))
        return
    base_patch, over_patch = split_keys(overlay, values)
    over_written = bool(over_patch) and tree.persist(
        overlay.source, patch_edit(entry, over_patch, overlay=True)
    )
    base_written = bool(base_patch) and tree.persist(entry.source, patch_edit(entry, base_patch))
    # after the writes: their checks read the recorded tables
    record(overlay, base_patch if base_written else {}, over_patch if over_written else {})


def save_tree(tree: LoaderTree) -> None:
    """Write every entry's keys to the files defining them (:meth:`Loader.save`).

    Each file is attempted; failures are raised together as an ``ExceptionGroup``.
    Example: ``save_tree(tree)``.
    """
    tree.ctx.fiber.assert_active()
    by_source: dict[Path, list[_Write]] = {}
    shares: list[tuple[Entry, RawTable, RawTable]] = []
    for node in tree.walk():
        entry = node.entry
        keys = persisted(entry)
        if entry.overlay is None:
            by_source.setdefault(entry.source, []).append((entry, False, keys))
            continue
        # the overlay's keys go to the overlay; the base keeps its shadowed values
        base_patch, over_patch = split_keys(entry.overlay, keys, changes_only=False)
        by_source.setdefault(entry.source, []).append((entry, False, base_patch))
        if over_patch:
            by_source.setdefault(entry.overlay.source, []).append((entry, True, over_patch))
        # what the save changes (it leaves values that mean the same alone)
        base_change = meaningful(base_patch, entry.overlay.base)
        shares.append((entry, base_change, meaningful(over_patch, entry.overlay.table)))
    failures: list[Exception] = []
    written: set[Path] = set()
    for source, writes in by_source.items():
        try:
            if tree.persist(source, save_edit(writes), strict=True):
                written.add(source)
        except Exception as error:
            failures.append(error)
    for entry, base_change, over_change in shares:
        overlay = entry.overlay
        assert overlay is not None
        base_change = base_change if entry.source in written else {}
        record(overlay, base_change, over_change if overlay.source in written else {})
    if failures:
        raise ExceptionGroup("cannot save config files", failures)


def shift(tree: LoaderTree, source: Path, removed: tuple[str | int, ...]) -> None:
    """Update the locations recorded in ``source`` after its element at ``removed`` was deleted.

    Example: ``shift(tree, path, ("plugins", 0))`` after deleting the first entry.
    """
    for node in tree.nodes.values():
        entry = node.entry
        if entry.source == source:
            entry.location = shifted(entry.location, removed)
        if entry.overlay is not None and entry.overlay.source == source:
            entry.overlay.location = shifted(entry.overlay.location, removed)
