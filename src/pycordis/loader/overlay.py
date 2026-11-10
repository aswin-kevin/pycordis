"""Config profiles: overlay file naming, merging and splitting values back (spec §3.9).

With a profile ``dev``, every config file ``<stem><suffix>`` read by the loader
(the root file and each included file) may have an overlay ``<stem>.dev<suffix>``
next to it. :mod:`pycordis.loader.files` layers it over its file at parse time;
this module holds the pure data rules:

* :func:`deep_merge` — how an overlay value combines with the base value
  (tables merged key by key, everything else, lists included, replaced);
* :func:`split_keys` — the inverse for write-back: which file a merged value
  goes to (the overlay when it defines that key or config path, else the base);
* :func:`shifted` — keeping recorded locations valid after an array element of
  a file is deleted;
* :class:`Layer` / :func:`overlaid` — sorting an overlay's entries against its
  base file by id and layering one overlay table over its base table.

Internal — use :class:`~pycordis.loader.LoaderConfig` ``profile`` or
:func:`~pycordis.loader.load_tree` ``profile``.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, final

from pycordis._guards import MISSING, RawTable, is_dict, is_list, is_table
from pycordis.loader.entry import Overlay
from pycordis.loader.table import (
    ENTRY_KINDS,
    EntryKind,
    LoaderConfigError,
    Location,
    check_keys,
    fields,
    format_location,
)

__all__ = [
    "PROFILE_ENV",
    "Layer",
    "check_profile",
    "deep_merge",
    "explicit_tables",
    "overlaid",
    "overlay_path",
    "record",
    "shifted",
    "split_keys",
]


PROFILE_ENV = "PYCORDIS_PROFILE"
"""Environment variable naming the profile when none is given (``os.environ[PROFILE_ENV]``)."""

_PROFILE = re.compile(r"[A-Za-z0-9_-]+")
_DEFAULTS: dict[str, object] = {"disabled": False, "inject": [], "intercept": {}, "isolate": []}
"""Entry keys whose absence can also be written as a value (what ``None`` means for them)."""


def check_profile(name: str) -> str:
    """Return ``name`` when it is a valid profile name, else raise ``ValueError``.

    A profile name is letters, digits, ``_`` and ``-`` only, so an overlay path
    built from it can never leave the config file's directory.
    Example: ``check_profile("dev") == "dev"``; ``check_profile("../x")`` raises.
    """
    if not _PROFILE.fullmatch(name):
        raise ValueError(f"invalid profile name {name!r}: use only letters, digits, '_' and '-'")
    return name


def overlay_path(path: Path, profile: str) -> Path:
    """The overlay of the config file ``path`` for ``profile``: ``<stem>.<profile><suffix>``.

    Example: ``overlay_path(Path("conf/app.toml"), "dev") == Path("conf/app.dev.toml")``.
    """
    return path.with_name(f"{path.stem}.{profile}{path.suffix}")


def deep_merge(base: object, over: object) -> Any:  # the merged config, as dynamic as its inputs
    """``over`` layered on ``base``: tables merged recursively, any other value replaced.

    New dicts are built for merged tables; neither input is modified.
    Example: ``deep_merge({"a": 1, "t": {"x": 1}}, {"t": {"y": 2}})``
    is ``{"a": 1, "t": {"x": 1, "y": 2}}``; lists are replaced, not concatenated.
    """
    if not (is_dict(base) and is_dict(over)):
        return over
    merged = dict(base)
    for key, value in over.items():
        merged[key] = deep_merge(base[key], value) if key in base else value
    return merged


def split_keys(
    overlay: Overlay, values: Mapping[str, object], *, changes_only: bool = True
) -> tuple[RawTable, RawTable]:
    """Split merged entry ``values`` into ``(base patch, overlay patch)``.

    ``values`` maps entry keys to their new merged value (``None``: the key is
    removed, i.e. back to its default). A key the overlay defines goes to the
    overlay only, any other to the base. A removed key the overlay defines is
    written to the overlay as its default value (``disabled = false``,
    ``inject = []``, ``intercept = {}``, ``isolate = []``) when the base defines
    the key, so the base keeps its own value; a key without a default form
    (``config``, ``restart``, timeouts, ``use``) is removed from both files then,
    as the base value would otherwise come back. ``config`` is split path by
    path (see :func:`_split_config`), so a value lands in the overlay exactly
    when the overlay defines it. With ``changes_only``, only values that differ
    from what the files hold (``overlay.table`` / ``overlay.base``) are kept, so
    an empty patch means that file needs no write. Example::

        split_keys(Overlay(..., table={"disabled": True}, base={"disabled": True}),
                   {"disabled": None, "inject": ["llm"]})
        # -> ({"inject": ["llm"]}, {"disabled": False})
    """
    base_patch: RawTable = {}
    over_patch: RawTable = {}
    for key, value in values.items():
        over: object = overlay.table.get(key, MISSING)
        base: object = overlay.base.get(key, MISSING)
        if value is None:
            if over is MISSING:
                base_patch[key] = None
            elif base is not MISSING and key in _DEFAULTS:
                over_patch[key] = copy.deepcopy(_DEFAULTS[key])
            else:
                over_patch[key] = None
                base_patch[key] = None
        elif key == "config":
            over_part, base_part = _split_config(value, over, base)
            if over_part is not MISSING:
                over_patch[key] = over_part
            base_patch[key] = None if base_part is MISSING else base_part
        elif over is not MISSING:
            over_patch[key] = value
        else:
            base_patch[key] = value
    if not changes_only:
        return base_patch, over_patch
    return _changes(base_patch, overlay.base), _changes(over_patch, overlay.table)


def _split_config(new: object, over: object, base: object) -> tuple[object, object]:
    """``(overlay part, base part)`` of the merged config value ``new``.

    ``over`` / ``base`` are the files' current values at this path (``MISSING``
    when undefined). Where both files and ``new`` hold tables, keys are split
    recursively and keys missing from ``new`` leave both files; elsewhere the
    overlay, when it defines the path, owns the whole value (and the base keeps
    its own, shadowed one). ``deep_merge(base part, overlay part) == new``.
    """
    if over is MISSING:
        return MISSING, new
    if not (is_dict(new) and is_dict(over) and is_dict(base)):
        return new, base
    return _split_tables(new, over, base)


def _split_tables(
    new: dict[object, object], over: dict[object, object], base: dict[object, object]
) -> tuple[dict[object, object], dict[object, object]]:
    """:func:`_split_config` of three tables, key by key."""
    over_part: dict[object, object] = {}
    base_part: dict[object, object] = {}
    for key, value in new.items():
        o, b = _split_config(value, over.get(key, MISSING), base.get(key, MISSING))
        if o is not MISSING:
            over_part[key] = o
        if b is not MISSING:
            base_part[key] = b
    return over_part, base_part


def _changes(patch: RawTable, table: Mapping[str, object]) -> RawTable:
    """The entries of ``patch`` that change ``table`` (``None``: remove the key)."""
    return {key: value for key, value in patch.items() if value != table.get(key)}


def record(
    overlay: Overlay, base_patch: Mapping[str, object], over_patch: Mapping[str, object]
) -> None:
    """Apply the patches of :func:`split_keys` to the tables ``overlay`` keeps.

    Example: ``record(overlay, {}, {"disabled": True})`` after writing the overlay.
    """
    for table, patch in ((overlay.base, base_patch), (overlay.table, over_patch)):
        for key, value in patch.items():
            if value is None:
                table.pop(key, None)
            else:
                table[key] = copy.deepcopy(value)


def shifted(location: Location, removed: Location) -> Location:
    """``location`` after the array element at ``removed`` was deleted from the same file.

    Locations after the removed element in the same array move up by one.
    Example: ``shifted(("plugins", 3), ("plugins", 1)) == ("plugins", 2)``.
    """
    depth = len(removed) - 1
    if len(location) <= depth or location[:depth] != removed[:depth]:
        return location
    index, gone = location[depth], removed[depth]
    if isinstance(index, int) and isinstance(gone, int) and index > gone:
        return (*location[:depth], index - 1, *location[depth + 1 :])
    return location


# -- parsing -----------------------------------------------------------------------


@final
@dataclass
class Layer:
    """An overlay file's entries, sorted for layering over its base file.

    ``matched`` maps base entry ids to the overlay table (and its location)
    overriding them; ``appended`` maps the id of a base group (``""`` for the
    top level) to the new entries the overlay lists in it. Example::

        layer = Layer(overlay_file, data)
        layer.collect(data.get("plugins", []), ("plugins",), "", explicit_tables(base_plugins))
    """

    source: Path
    data: RawTable
    matched: dict[str, tuple[RawTable, Location]] = field(default_factory=lambda: {})
    appended: dict[str, list[tuple[object, Location]]] = field(default_factory=lambda: {})

    def collect(
        self, raw: object, location: Location, group: str, base: dict[str, RawTable]
    ) -> None:
        """Sort the overlay entries ``raw`` listed in ``group`` against the base tables by id."""
        if not is_list(raw):
            raise LoaderConfigError(
                f'{self.source}: "plugins" must be an array (at {format_location(location)})'
            )
        for i, table in enumerate(raw):
            where = (*location, i)
            if not is_table(table):
                raise LoaderConfigError(
                    f"{self.source}: entry {format_location(where)} must be a table"
                )
            id = table.get("id")
            target = base.get(id) if isinstance(id, str) else None
            if not isinstance(id, str) or target is None:
                self.appended.setdefault(group, []).append((table, where))
                continue
            if id in self.matched:
                first = format_location(self.matched[id][1])
                raise LoaderConfigError(
                    f'{self.source}: entry "{id}" is overlaid twice'
                    f" (at {first} and {format_location(where)})"
                )
            self.matched[id] = (table, where)
            if "plugins" in table:
                if "group" not in target:
                    raise LoaderConfigError(
                        f'{self.source}: "plugins" is only allowed on group entries'
                        f" in entry {format_location(where)}"
                    )
                self.collect(table["plugins"], (*where, "plugins"), id, base)


def explicit_tables(raw: object) -> dict[str, RawTable]:
    """The entry tables of a file's ``plugins`` (groups included) by explicit id.

    Tolerant of malformed data: the base parse reports it.
    Example: ``explicit_tables([{"id": "a", "use": "m:p"}]) == {"a": {...}}``.
    """
    found: dict[str, RawTable] = {}
    pending = [raw]
    while pending:
        items = pending.pop()
        if not is_list(items):
            continue
        for table in items:
            if is_table(table):
                id = table.get("id")
                if isinstance(id, str):
                    found.setdefault(id, table)
                if "group" in table:
                    pending.append(table.get("plugins"))
    return found


def overlaid(
    base: RawTable, kind: EntryKind, match: tuple[RawTable, Location], source: Path
) -> tuple[RawTable, Overlay]:
    """The base entry table (of ``kind``) with the overlay table ``match`` from ``source`` on top.

    Returns the merged table and the :class:`Overlay` record of the entry.
    Raises :class:`LoaderConfigError` naming the overlay for an invalid overlay table.
    Example: ``overlaid({"id": "a", "use": "m:p"}, "use", ({"id": "a", "disabled": True},
    ("plugins", 0)), Path("app.dev.toml"))[0]["disabled"] is True``.
    """
    over, location = match
    where = f"entry {format_location(location)}"

    def fail(message: str) -> LoaderConfigError:
        return LoaderConfigError(f"{source}: {message} in {where}")

    check_keys(over, fail)
    for key in ENTRY_KINDS:
        replaces_use = key == "use" and kind == "use"  # an overlay may swap the plugin
        if key in over and not replaces_use and over[key] != base.get(key):
            raise fail(f'cannot change "{key}" of entry "{over["id"]}"')
    fields(over, kind, fail)
    merged = {**base, **{k: v for k, v in over.items() if k != "plugins"}}
    if "config" in base and "config" in over:
        merged["config"] = deep_merge(base["config"], over["config"])
    overlay = Overlay(
        source=source,
        location=location,
        table=copy.deepcopy({k: v for k, v in over.items() if k not in ("id", "plugins")}),
        base=copy.deepcopy({k: v for k, v in base.items() if k != "plugins"}),
    )
    return merged, overlay
