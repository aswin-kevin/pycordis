"""Reading loader config files into an :class:`Entry` tree (spec §6.6).

Parsing is purely structural: config values are kept raw (no interpolation)
so they can be written back unchanged, and plugins are not imported. Editing
files lives in :mod:`pycordis.loader.persist`. A top-level ``$schema`` string
(an editor's pointer to the file's JSON Schema, such as ``pycordis schema``
output) is allowed in every file, overlays included, and ignored; write-back
leaves it in place.

Profiles (spec §3.9): with ``profile="dev"``, each file read (the root and
every included file) is layered with its overlay ``<stem>.dev<suffix>`` next
to it, when that exists. The overlay has the same shape as a config file:

* ``[loader]`` keys (root overlay only) override the root file's. Overlays
  are trusted like the base file: an overlay's ``allow`` can widen the plugin
  allow-list;
* an overlay entry whose explicit ``id`` names an entry of the base file (at
  any depth) overrides that entry's keys, its ``config`` deep-merged (tables
  merged key by key, other values, lists included, replaced). It may set
  ``disabled``, replace ``use`` and, on a group, list more ``plugins``; it
  cannot change a ``group`` or ``include`` value or an entry's kind;
* any other overlay entry (no ``id``, or an id the base file does not
  declare) is a new entry appended to the group it is listed in (the top level
  for the overlay's top-level ``plugins``), after the base file's entries.

Overlay errors name the overlay file. See :class:`~pycordis.loader.entry.Overlay`
for how a merged entry records where each value is defined.
"""

from __future__ import annotations

import importlib
import json
import os
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import final

from pycordis._guards import RawTable, is_list, is_table
from pycordis.loader.entry import Entry, LoaderOptions
from pycordis.loader.overlay import Layer, check_profile, explicit_tables, overlaid, overlay_path
from pycordis.loader.table import (
    ENTRY_KINDS,
    EntryKind,
    LoaderConfigError,
    Location,
    check_keys,
    fields,
    format_location,
    strings,
)

__all__ = [
    "ENTRY_KINDS",
    "LoaderConfigError",
    "format_location",
    "load_tree",
    "parse_entry",
    "parse_text",
    "read_config",
]

# an editor's pointer to the file's JSON Schema (e.g. ``pycordis schema`` output)
_SCHEMA_KEY = "$schema"
_ROOT_KEYS = frozenset({"plugins", "loader", _SCHEMA_KEY})


def read_config(path: Path) -> RawTable:
    """Parse a config file by suffix (``.toml``, ``.json``, ``.yaml`` / ``.yml``).

    Raises :class:`LoaderConfigError` (naming the file) when the file cannot be
    read or decoded, on syntax errors or when the document is not a table,
    and ``RuntimeError`` for YAML when PyYAML is not installed.

    Example: ``read_config(Path("app.toml"))["plugins"]``.
    """
    suffix = path.suffix.lower()
    if suffix not in (".toml", ".json", ".yaml", ".yml"):
        raise LoaderConfigError(f'{path}: unsupported config file type "{suffix}"')
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise LoaderConfigError(f"{path}: cannot read file: {error.strerror or error}") from error
    except UnicodeDecodeError as error:
        raise LoaderConfigError(f"{path}: file is not valid UTF-8: {error.reason}") from error
    return parse_text(path, text)


def parse_text(path: Path, text: str) -> RawTable:
    """Parse ``text`` as the config format of ``path``'s suffix (already checked)."""
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = _load_yaml(path, text)
    else:
        try:
            data = tomllib.loads(text) if suffix == ".toml" else json.loads(text)
        except ValueError as error:
            raise LoaderConfigError(f"{path}: {error}") from error
    if not is_table(data):
        raise LoaderConfigError(f"{path}: the document must be a table")
    return data


def _load_yaml(path: Path, text: str) -> object:
    try:
        yaml = importlib.import_module("yaml")
    except ImportError:
        raise RuntimeError("YAML config requires PyYAML to be installed") from None
    try:
        data: object = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise LoaderConfigError(f"{path}: {error}") from error
    return {} if data is None else data


def load_tree(
    path: Path, *, files: list[Path] | None = None, profile: str | None = None
) -> tuple[list[Entry], LoaderOptions]:
    """Load the root config ``path`` and its includes into an entry tree.

    Returns the top-level entries and the root file's ``[loader]`` options.
    Raises :class:`LoaderConfigError` (a ``ValueError``) for unreadable or
    malformed files, missing includes, unknown keys, duplicate ids and include
    cycles; ``RuntimeError`` for YAML without PyYAML. When ``files`` is given,
    the resolved path of every file the load reads or looks for is appended to
    it, also when the load fails (the hot-reload watch set). ``profile``
    layers each file's overlay over it (see the module docstring); the root
    path is resolved first, so a symlinked root file's overlay is looked up
    next to the symlink's target. An invalid profile name raises
    ``ValueError``. The environment is not read here (see
    :class:`~pycordis.loader.LoaderConfig`). Example::

        entries, options = load_tree(Path("app.toml"), profile="dev")
    """
    if profile is not None:
        check_profile(profile)
    root = path.resolve()
    parser = _Parser(root.parent, files, profile)
    data = parser.read(root)
    layer = parser.overlay(root, data)
    loader = data.get("loader", {})
    options = _options(loader, root)
    if layer is not None and "loader" in layer.data:
        _options(layer.data["loader"], layer.source)  # names the overlay on errors
        options = _options({**loader, **layer.data["loader"]}, root)
    return parser.entries_of(root, data, "", layer), options


def parse_entry(
    raw: Mapping[str, object],
    *,
    source: Path,
    location: tuple[str | int, ...],
    parent_id: str,
    siblings: Sequence[Entry] = (),
    existing: Iterable[Entry] = (),
) -> Entry:
    """Parse one entry table as if it were declared in ``source`` at ``location``.

    Used by the loader API to add or rewrite entries without a file. ``siblings``
    are the entries already in the target group (a generated id never collides
    with theirs) and ``existing`` every entry whose id is taken. Includes are
    read relative to ``source``. Raises :class:`LoaderConfigError` like
    :func:`load_tree`.
    """
    parser = _Parser(source.parent)
    for entry in existing:
        parser.ids[entry.id] = f"{entry.source} {format_location(entry.location)}"
    counts: dict[str, int] = {}
    for sibling in siblings:
        if not sibling.explicit_id:
            key = _entry_key(sibling)
            counts[key] = counts.get(key, 0) + 1
    kinds: list[tuple[EntryKind, str]] = [
        (kind, value) for kind in ENTRY_KINDS if isinstance(value := raw.get(kind), str)
    ]
    if "id" not in raw and len(kinds) == 1:
        key = _key(*kinds[0])
        while f"{parent_id}/{key}#{counts.get(key, 0)}" in parser.ids:
            counts[key] = counts.get(key, 0) + 1
    return parser.entry(dict(raw), source, location, parent_id, counts)


def _entry_key(entry: Entry) -> str:
    """The generated-id key of ``entry``: its ``use``, ``group:<name>`` or ``include:<path>``."""
    if entry.use is not None:
        return entry.use
    if entry.group is not None:
        return _key("group", entry.group)
    return _key("include", str(entry.include))


def _key(kind: EntryKind, value: str) -> str:
    return value if kind == "use" else f"{kind}:{value}"


def _options(raw: object, source: Path) -> LoaderOptions:
    if not is_table(raw):
        raise LoaderConfigError(f'{source}: "loader" must be a table')
    table = raw
    options = LoaderOptions()
    for key, value in table.items():
        if key in ("readonly", "watch"):
            if not isinstance(value, bool):
                raise LoaderConfigError(f'{source}: [loader] "{key}" must be a boolean')
            setattr(options, key, value)
        elif key == "allow":
            options.allow = strings(value, f'{source}: [loader] "allow"')
        elif key == "poll_interval":
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise LoaderConfigError(
                    f'{source}: [loader] "poll_interval" must be a positive number'
                )
            options.poll_interval = float(value)
        else:
            raise LoaderConfigError(f'{source}: unknown [loader] key "{key}"')
    return options


@final
class _Parser:
    def __init__(
        self, root_dir: Path, files: list[Path] | None = None, profile: str | None = None
    ) -> None:
        self.root_dir = root_dir
        self.files = files
        self.profile = profile
        self.stack: list[tuple[tuple[int, int], Path]] = []
        self.ids: dict[str, str] = {}

    def read(self, path: Path) -> RawTable:
        if self.files is not None:
            self.files.append(path)
        data = read_config(path)
        for key in data:
            if key not in _ROOT_KEYS:
                raise LoaderConfigError(f'{path}: unknown top-level key "{key}"')
        if not isinstance(data.get(_SCHEMA_KEY, ""), str):
            raise LoaderConfigError(f'{path}: "{_SCHEMA_KEY}" must be a string')
        return data

    def overlay(self, path: Path, data: RawTable, *, root: bool = True) -> Layer | None:
        """The profile overlay of the file ``path`` (whose data is ``data``), if it exists."""
        if self.profile is None:
            return None
        source = overlay_path(path, self.profile)
        if self.files is not None:
            self.files.append(source)
        if not source.is_file():
            return None
        layer = Layer(source, self.read(source))
        if not root and "loader" in layer.data:
            raise LoaderConfigError(f"{source}: [loader] is only allowed in the root config file")
        base = explicit_tables(data.get("plugins"))
        layer.collect(layer.data.get("plugins", []), ("plugins",), "", base)
        return layer

    def entries_of(
        self, path: Path, data: RawTable, parent_id: str, layer: Layer | None = None
    ) -> list[Entry]:
        # keyed on file identity so hardlinks / case-insensitive names still match
        try:
            stat = path.stat()
        except OSError as error:
            raise LoaderConfigError(
                f"{path}: cannot read file: {error.strerror or error}"
            ) from error
        identity = (stat.st_dev, stat.st_ino)
        keys = [key for key, _ in self.stack]
        if identity in keys:
            start = keys.index(identity)
            chain = [p for _, p in self.stack[start:]] + [self.stack[start][1]]
            names = " -> ".join(os.path.relpath(p, self.root_dir) for p in chain)
            raise LoaderConfigError(f"include cycle: {names}")
        self.stack.append((identity, path))
        appended = [] if layer is None else layer.appended.pop("", [])
        try:
            return self.entries(
                data.get("plugins", []), path, ("plugins",), parent_id, layer, appended
            )
        finally:
            self.stack.pop()

    def entries(
        self,
        raw: object,
        source: Path,
        location: Location,
        parent_id: str,
        layer: Layer | None = None,
        appended: Sequence[tuple[object, Location]] = (),
    ) -> list[Entry]:
        """Parse the entry array ``raw``, then the overlay entries ``appended`` to it."""
        if not is_list(raw):
            raise LoaderConfigError(
                f'{source}: "plugins" must be an array (at {format_location(location)})'
            )
        counts: dict[str, int] = {}
        result = [
            self.entry(item, source, (*location, i), parent_id, counts, layer)
            for i, item in enumerate(raw)
        ]
        if layer is not None:
            result += [
                self.entry(item, layer.source, where, parent_id, counts) for item, where in appended
            ]
        return result

    def entry(
        self,
        raw: object,
        source: Path,
        location: Location,
        parent_id: str,
        counts: dict[str, int],
        layer: Layer | None = None,
    ) -> Entry:
        where = f"entry {format_location(location)}"
        if not is_table(raw):
            raise LoaderConfigError(f"{source}: {where} must be a table")
        item = raw

        def fail(message: str) -> LoaderConfigError:
            return LoaderConfigError(f"{source}: {message} in {where}")

        check_keys(item, fail)
        kinds: list[EntryKind] = [k for k in ENTRY_KINDS if k in item]
        if len(kinds) != 1:
            raise fail('an entry must have exactly one of "use", "group" or "include"')
        kind = kinds[0]
        parsed = fields(item, kind, fail)
        overlay = None
        key = item.get("id")
        if layer is not None and isinstance(key, str) and key in layer.matched:
            item, overlay = overlaid(item, kind, layer.matched.pop(key), layer.source)
            parsed = fields(item, kind, fail)
        value: str = item[kind]
        entry_id = self._entry_id(item, kind, value, parent_id, counts, fail)
        self.ids[entry_id] = f"{source} {format_location(location)}"

        entry = Entry(
            id=entry_id,
            explicit_id="id" in item,
            use=value if kind == "use" else None,
            group=value if kind == "group" else None,
            include=value if kind == "include" else None,
            config=item.get("config"),
            disabled=parsed.disabled,
            isolate=parsed.isolate,
            intercept=parsed.intercept,
            inject=parsed.inject,
            restart=parsed.restart,
            load_timeout=parsed.load_timeout,
            dispose_timeout=parsed.dispose_timeout,
            source=source,
            location=location,
            overlay=overlay,
        )
        if kind == "group":
            appended = [] if layer is None or "id" not in item else layer.appended.pop(entry_id, [])
            entry.children = self.entries(
                item.get("plugins", []), source, (*location, "plugins"), entry_id, layer, appended
            )
        elif kind == "include":
            entry.children = self._included(source, value, entry_id)
        return entry

    def _entry_id(
        self,
        item: RawTable,
        kind: EntryKind,
        value: str,
        parent_id: str,
        counts: dict[str, int],
        fail: Callable[[str], LoaderConfigError],
    ) -> str:
        """The entry's explicit ``id``, else ``<parent>/<kind key>#<n>``; unique or ``fail``."""
        if "id" in item:
            entry_id: str = item["id"]
        else:
            key = _key(kind, value)
            n = counts.get(key, 0)
            counts[key] = n + 1
            entry_id = f"{parent_id}/{key}#{n}"
        if entry_id in self.ids:
            raise fail(f'duplicate entry id "{entry_id}" (first declared at {self.ids[entry_id]})')
        return entry_id

    def _included(self, source: Path, value: str, entry_id: str) -> list[Entry]:
        """The entries of the file an ``include`` entry of ``source`` names."""
        target = (source.parent / value).resolve()
        if self.files is not None:
            self.files.append(target)
        if not target.is_file():
            raise LoaderConfigError(f'{source}: include "{value}" not found ({target})')
        data = self.read(target)
        if "loader" in data:
            raise LoaderConfigError(f"{target}: [loader] is only allowed in the root config file")
        include_layer = self.overlay(target, data, root=False)
        return self.entries_of(target, data, entry_id, include_layer)
