"""Writing loader changes back to config files (spec §6.6 write-back).

:class:`ConfigFile` edits one file in place (TOML format-preserving, JSON with
its detected indent, YAML re-serialized) and :func:`write_atomic` replaces it
atomically. :class:`Writer` applies the loader's edits (honouring ``readonly``
and the file watcher), and the helpers below build those edits and guard
committed configs against persisting interpolated secrets (see
:mod:`pycordis.loader.interpolate`).
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import re
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, final

from pycordis._guards import MISSING, Missing, RawTable, is_array, is_dict, is_list, is_table
from pycordis.loader.entry import Entry
from pycordis.loader.files import ENTRY_KINDS, LoaderConfigError, format_location, parse_text
from pycordis.loader.interpolate import Expansions, leaks, retemplate, secrets
from pycordis.loader.overlay import split_keys
from pycordis.loader.table import EntryKind
from pycordis.tomledit import TomlDocument

if TYPE_CHECKING:
    from pycordis.loader.watch import FileWatcher
    from pycordis.logger import Logger

__all__ = ["ConfigFile", "Writer", "write_atomic"]

type Location = tuple[str | int, ...]
type Edit = Callable[[ConfigFile], None]


@final
class ConfigFile:
    """One config file opened for editing; :meth:`write` replaces it atomically.

    Paths are data locations like :attr:`Entry.location`. TOML is edited with
    :class:`~pycordis.tomledit.TomlDocument`, so comments and formatting are
    kept (a value array that grows is re-rendered whole). JSON and YAML are
    re-serialized: JSON with the file's detected indent (default 2), its
    trailing-newline presence and line endings; YAML through PyYAML (its
    comments are lost).
    """

    def __init__(self, path: Path, text: str) -> None:
        self.path = path
        """The file."""
        self.original = text
        """The text the file had when opened."""
        suffix = path.suffix.lower()
        self.kind: Literal["toml", "json", "yaml"] = (
            "toml" if suffix == ".toml" else "json" if suffix == ".json" else "yaml"
        )
        """``"toml"``, ``"json"`` or ``"yaml"``."""
        self._toml: TomlDocument | None = None
        self._data: RawTable = {}  # the parsed JSON / YAML document (unused for TOML)
        if self.kind == "toml":
            try:
                self._toml = TomlDocument(text)
            except tomllib.TOMLDecodeError as error:
                raise LoaderConfigError(f"{path}: {error}") from error
        else:
            self._data = parse_text(path, text)

    @classmethod
    def open(cls, path: Path) -> ConfigFile:
        """Read ``path`` (``.toml``, ``.json``, ``.yaml`` / ``.yml``) for editing.

        Raises :class:`LoaderConfigError` like :func:`read_config`.
        """
        suffix = path.suffix.lower()
        if suffix not in (".toml", ".json", ".yaml", ".yml"):
            raise LoaderConfigError(f'{path}: unsupported config file type "{suffix}"')
        try:
            text = path.read_bytes().decode("utf-8")  # keeps CRLF line endings
        except OSError as error:
            raise LoaderConfigError(
                f"{path}: cannot read file: {error.strerror or error}"
            ) from error
        except UnicodeDecodeError as error:
            raise LoaderConfigError(f"{path}: file is not valid UTF-8: {error.reason}") from error
        return cls(path, text)

    def get(self, location: Location, default: object | Missing = MISSING) -> Any:
        """The value at ``location``; ``default`` (else ``KeyError``) when absent.

        Returns file data, whose shape only the caller knows (hence ``Any``).
        Example: ``file.get(("plugins", 0, "use"))``.
        """
        node = self._lookup(location)
        if node is not MISSING:
            return node
        if default is MISSING:
            raise KeyError(location)
        return default

    def _lookup(self, location: Location) -> object | Missing:
        """The value at ``location``, else ``MISSING``."""
        node: object = self._toml.data if self._toml is not None else self._data
        for part in location:
            node = _child(node, part)
            if node is MISSING:
                return MISSING
        return node

    def set(self, location: Location, value: object) -> None:
        """Set ``location`` (its parent must exist) to ``value``.

        Raises ``TypeError`` for a value TOML cannot represent.
        Example: ``file.set(("plugins", 0, "disabled"), True)``.
        """
        if self._toml is not None:
            self._toml.set(location, value)
            return
        *parents, last = location
        container = self._container(parents)
        if is_dict(container):
            container.update({last: value})
        elif is_list(container) and isinstance(last, int):
            container[last] = value
        else:
            raise TypeError(f"cannot set {last!r} in {type(container).__name__} at {location!r}")

    def delete(self, location: Location) -> None:
        """Remove ``location`` (a key or an array element); absent is a no-op.

        Example: ``file.delete(("plugins", 2))``.
        """
        if self._lookup(location) is MISSING:
            return
        if self._toml is not None:
            self._toml.delete(location)
            return
        *parents, last = location
        container = self._container(parents)
        if is_dict(container):
            container.pop(last)
        elif is_list(container) and isinstance(last, int):
            del container[last]
        else:
            raise TypeError(f"cannot delete {last!r} in {type(container).__name__} at {location!r}")

    def _container(self, parents: Sequence[str | int]) -> object:
        """The JSON / YAML node at ``parents`` (``KeyError`` / ``IndexError`` when absent)."""
        node: object = self._data
        for part in parents:
            node = _child(node, part)
            if node is MISSING:
                raise KeyError(parents)
        return node

    def append(self, location: Location, value: object) -> int:
        """Append ``value`` to the array at ``location`` (created if absent); return its index.

        Example: ``file.append(("plugins",), {"use": "app:chat"}) == 3``.
        """
        current: object = self.get(location, [])
        items = list(current) if is_array(current) else []
        self.set(location, [*items, value])
        return len(items)

    def check(self, entry: Entry, *, overlay: bool = False) -> None:
        """Raise :class:`LoaderConfigError` unless ``entry`` is declared at its location here.

        Guards writes against a file edited since it was loaded. ``overlay``
        checks the entry's profile overlay table (matched by its id) instead of
        its base declaration. Example: ``file.check(entry)`` before editing.
        """
        location = declaration(entry, overlay)
        table: object = self.get(location, None)
        ok = is_table(table)
        if is_table(table) and not overlay:
            ok = all(table.get(kind) == _declared_kind(entry, kind) for kind in ENTRY_KINDS)
        if ok and is_table(table) and (overlay or entry.explicit_id):
            ok = table.get("id") == entry.id
        if not ok:
            raise LoaderConfigError(
                f'{self.path}: entry "{entry.id}" is no longer at {format_location(location)}'
                " (the file changed; reload it first)"
            )

    def dumps(self) -> str:
        """The edited text."""
        if self._toml is not None:
            return self._toml.dumps()
        if self.kind == "json":
            match = re.search(r"\n([ \t]+)\S", self.original)
            indent = match.group(1) if match else 2
            text = json.dumps(self._data, indent=indent, ensure_ascii=False)
            if self.original.endswith("\n"):
                text += "\n"
        else:
            yaml = importlib.import_module("yaml")
            # safe_dump returns the text when given no stream
            text = str(yaml.safe_dump(self._data, sort_keys=False, allow_unicode=True))
        if "\r\n" in self.original:
            text = text.replace("\n", "\r\n")
        return text

    def write(self) -> bool:
        """Atomically replace the file when its text changed; return whether it did."""
        text = self.dumps()
        if text == self.original:
            return False
        write_atomic(self.path, text)
        self.original = text
        return True


def _declared_kind(entry: Entry, kind: EntryKind) -> object:
    """The ``use`` / ``group`` / ``include`` value of ``entry``'s base declaration.

    It differs from the merged entry's when a profile overlay replaces ``use``.
    """
    if entry.overlay is not None:
        return entry.overlay.base.get(kind)
    match kind:
        case "use":
            return entry.use
        case "group":
            return entry.group
        case "include":
            return entry.include


def write_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` (UTF-8, line endings as given) atomically.

    The text goes to a temporary file in the same directory, which is flushed,
    fsync-ed, given the original file's permission bits (a new file keeps the
    temporary file's owner-only mode) and moved over ``path`` with
    :func:`os.replace`. On failure the temporary file is removed and ``path`` is
    untouched.
    """
    try:
        mode: int | None = path.stat().st_mode & 0o7777
    except FileNotFoundError:
        mode = None
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temp:
        tmp = Path(temp.name)
        try:
            temp.write(text.encode("utf-8"))
            temp.flush()
            os.fsync(temp.fileno())
        except BaseException:
            temp.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@final
class Writer:
    """Applies the loader's file edits; logs a YAML comment-loss warning once."""

    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        self.yaml_warned = False

    def write(
        self,
        source: Path,
        edit: Edit,
        *,
        readonly: bool,
        watcher: FileWatcher | None,
        strict: bool = False,
    ) -> bool:
        """Apply ``edit`` to ``source`` and write it; return whether the file is in sync.

        A no-op returning ``False`` when ``readonly``. Errors are logged (raised
        when ``strict``) and return ``False``. A write the ``watcher`` has not
        seen a change before is recorded, so it does not trigger a reload.
        """
        if readonly:
            return False
        try:
            file = ConfigFile.open(source)
            edit(file)
            # a file changed behind the watcher's back is left for it to notice
            fresh = watcher is not None and watcher.unchanged(source)
            if file.write():
                if file.kind == "yaml" and not self.yaml_warned:
                    self.yaml_warned = True
                    self.logger.warn("saving YAML config drops comments")
                if fresh and watcher is not None:
                    watcher.record(source)
        except Exception as error:
            if strict:
                raise
            self.logger.error("cannot save %s: %s", source, error)
            return False
        return True


def _child(node: object, part: str | int) -> object | Missing:
    """Element ``part`` of an array or key ``part`` of a table (``MISSING`` when absent)."""
    if isinstance(part, int) and is_list(node):
        return node[part] if 0 <= part < len(node) else MISSING
    if isinstance(part, str) and is_dict(node):
        return node.get(part, MISSING)
    return MISSING


# -- secrets --------------------------------------------------------------------------


@final
class Refused(Enum):
    """The type of :data:`REFUSED`."""

    REFUSED = "REFUSED"


REFUSED: Final = Refused.REFUSED
"""Returned by :func:`committed_config` for a commit that would leak a secret."""


def restore_placeholders(value: object, entry: Entry, known: Expansions) -> object:
    """``value`` with the expanded leaves of ``entry``'s raw config put back as placeholders."""
    return retemplate(value, entry.config, entry.source.parent, known.leaves)


def committed_config(entry: Entry, config: object, known: Expansions) -> object | Refused:
    """The config to store for a commit of ``config`` to ``entry``, or :data:`REFUSED`.

    Re-templates ``config`` and adds the secrets of ``entry``'s raw config, as
    expanded now, to ``known``. Refused when the result would still persist
    one of ``known``'s secrets; otherwise a deep copy is returned. Example::

        if committed_config(entry, config, known) is REFUSED: ...
    """
    config = restore_placeholders(config, entry, known)
    known.secrets |= secrets(entry.config, entry.source.parent)
    if leaks(config, entry.config, known.secrets):
        return REFUSED
    return _copy(config)


def leak_target(entry: Entry, config: object, known: Expansions) -> Path:
    """The file a refused commit of ``config`` (see :func:`committed_config`) would leak into.

    The entry's own file, or its profile overlay when the overlay's share of the
    config holds the secret. Example: ``secret_message(leak_target(entry, config, known), id)``.
    """
    overlay = entry.overlay
    if overlay is None:
        return entry.source
    config = restore_placeholders(config, entry, known)
    _, over_patch = split_keys(overlay, {"config": config}, changes_only=False)
    over_config = over_patch.get("config")
    if leaks(over_config, overlay.table.get("config"), known.secrets):
        return overlay.source
    return entry.source


def new_config_leaks(config: object, source_dir: Path) -> bool:
    """Whether ``config`` holds the expansion of one of its own placeholders literally.

    The check for a config that has no previous version (:meth:`Loader.create`).
    Example: ``new_config_leaks({"k": "${env:TOKEN}"}, Path("."))`` is ``False``.
    """
    return leaks(config, None, secrets(config, source_dir))


# -- edits ----------------------------------------------------------------------------


def declaration(entry: Entry, overlay: bool = False) -> Location:
    """Where ``entry`` is declared in its file: its base location, or its overlay table's.

    Example: ``declaration(entry) == entry.location``.
    """
    if not overlay:
        return entry.location
    assert entry.overlay is not None
    return entry.overlay.location


@final
class AppendEdit:
    """An edit appending ``value`` to the entry array at ``location``.

    ``check`` is an entry (the parent group) that must still be at its
    location. After the edit ran, :attr:`index` is the new element's index.
    """

    def __init__(self, location: Location, value: object, check: Entry | None = None) -> None:
        self.location = location
        self.value = value
        self.check = check
        self.index = 0

    def __call__(self, file: ConfigFile) -> None:
        if self.check is not None:
            file.check(self.check)
        self.index = file.append(self.location, self.value)


def patch_edit(entry: Entry, patch: Mapping[str, object], *, overlay: bool = False) -> Edit:
    """An edit setting (``None``: removing) the keys of ``patch`` on ``entry``.

    ``overlay`` edits the entry's overlay table instead of its base declaration.
    """

    def edit(file: ConfigFile) -> None:
        file.check(entry, overlay=overlay)
        location = declaration(entry, overlay)
        for key, value in patch.items():
            put(file, (*location, key), value)

    return edit


def remove_edit(declarations: list[tuple[Entry, bool]]) -> Edit:
    """An edit deleting the declarations ``(entry, overlay)`` (see :func:`declaration`).

    Every declaration is checked first; they are then deleted in the given
    order, which must not shift the ones after it (descending locations).
    """

    def edit(file: ConfigFile) -> None:
        for entry, overlay in declarations:
            file.check(entry, overlay=overlay)
        for entry, overlay in declarations:
            file.delete(declaration(entry, overlay))

    return edit


def save_edit(writes: list[tuple[Entry, bool, RawTable]]) -> Edit:
    """An edit rewriting the keys ``(entry, overlay, keys)`` whose file value means something else.

    ``keys`` are ``persisted(entry)`` (or the file's share of them, see
    :func:`~pycordis.loader.overlay.split_keys`), written to the entry's base
    declaration, or its overlay table when ``overlay``.
    """

    def edit(file: ConfigFile) -> None:
        for entry, overlay, keys in writes:
            file.check(entry, overlay=overlay)
            for key, value in keys.items():
                location = (*declaration(entry, overlay), key)
                if _meaning(key, file.get(location, None)) != _meaning(key, value):
                    put(file, location, value)

    return edit


def persisted(entry: Entry) -> RawTable:
    """The file keys :meth:`Loader.save` writes for ``entry`` (``None``: absent).

    Example: ``persisted(entry)["disabled"]`` is ``True`` or ``None``.
    """
    keys: RawTable = {}
    if entry.use is not None:
        keys["config"] = entry.config
    keys["disabled"] = True if entry.disabled else None
    labels = list(entry.isolate.values())
    if not entry.isolate:
        keys["isolate"] = None
    elif None in labels:
        keys["isolate"] = list(entry.isolate)
    else:
        keys["isolate"] = dict(entry.isolate)
    keys["intercept"] = dict(entry.intercept) or None
    keys["inject"] = list(entry.inject) or None
    keys["restart"] = entry.restart
    keys["load_timeout"] = entry.load_timeout
    keys["dispose_timeout"] = entry.dispose_timeout
    return keys


def meaningful(patch: Mapping[str, object], table: Mapping[str, object]) -> RawTable:
    """The keys of ``patch`` whose value means something else than in ``table``.

    Defaults and list forms are normalised as :meth:`Loader.save` compares them.
    Example: ``meaningful({"disabled": None}, {"disabled": False}) == {}``.
    """
    return {k: v for k, v in patch.items() if _meaning(k, table.get(k)) != _meaning(k, v)}


def _meaning(key: str, value: object) -> object:
    """``value`` of the entry key ``key`` with defaults and list forms normalised."""
    if key == "disabled":
        return bool(value)
    if key == "isolate":
        if is_list(value):
            return dict.fromkeys(value)
        return value or {}
    if key in ("intercept", "inject"):
        return value or None
    return value


def _copy(value: object) -> object:
    """A deep copy of ``value``; ``value`` itself when it cannot be copied.

    A config holding an uncopyable object (a lock, a client) is still stored:
    it is only written back to the file, which fails loudly if it cannot be.
    """
    try:
        return copy.deepcopy(value)
    except Exception:  # deepcopy raises TypeError, copy.Error or anything __deepcopy__ raises
        return value


def put(file: ConfigFile, location: Location, value: object) -> None:
    """Set ``location`` to ``value``, or remove it when ``value`` is ``None``."""
    if value is None:
        file.delete(location)
    else:
        file.set(location, value)


def relocate(entry: Entry, location: Location) -> None:
    """Move ``entry`` (and a group's children, declared in the same file) to ``location``."""
    entry.location = location
    if entry.group is not None:
        for i, child in enumerate(entry.children):
            relocate(child, (*location, "plugins", i))
