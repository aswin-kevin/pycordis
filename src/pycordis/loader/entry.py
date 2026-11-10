"""Config-file entry model for the built-in loader (spec §6.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, final

from pycordis._guards import RawTable
from pycordis.supervise import RestartPolicy


@final
@dataclass(kw_only=True)
class Overlay:
    """Where a profile overlay redefines an :class:`Entry` (spec §3.9).

    ``source`` is the overlay file and ``location`` the overlay table's path in
    its data. ``table`` holds the keys the overlay defines (raw, without ``id``
    and ``plugins``) and ``base`` the keys of the base declaration (raw, without
    ``plugins``); the loader keeps both in step with its writes, so it knows
    which file defines each value. Example::

        Overlay(source=Path("app.dev.toml"), location=("plugins", 0),
                table={"config": {"model": "large"}}, base={"id": "a", "use": "m:p"})
    """

    source: Path
    location: tuple[str | int, ...]
    table: RawTable
    base: RawTable


@final
@dataclass(kw_only=True)
class Entry:
    """One entry of a loader config file: a plugin, a group or an include.

    Exactly one of :attr:`use`, :attr:`group` and :attr:`include` is set.

    ``id`` is the explicit ``id`` when given; otherwise it is generated as
    ``"<parent-id>/<key>#<n>"`` where ``<parent-id>`` is the enclosing group's or
    include's id (explicit or generated; empty at the top level, so top-level
    generated ids start with ``/``), ``<key>`` is the ``use`` string,
    ``group:<name>`` or ``include:<path>``, and ``n`` counts earlier *id-less*
    siblings with the same key. Generated ids are stable across reparses;
    reordering two id-less entries with the same key swaps their ids.

    ``config`` is the raw, un-interpolated value from the file (``None`` when
    absent). ``isolate`` maps service names to a shared label, or ``None`` for a
    fresh private scope (the list form). ``source`` is the file the entry is
    declared in (the included file for entries of an include) and ``location``
    is the entry's path inside that file's data, e.g. ``("plugins", 3,
    "plugins", 0)``; write-back targets ``location + ("config",)``.

    With a profile, an entry matched by an overlay file keeps its base
    declaration in ``source`` / ``location`` and records the overlay in
    :attr:`overlay`; its values (``config`` included) are the merged ones. An
    entry only the overlay declares has the overlay file as ``source`` and no
    ``overlay``.

    Example: ``[item.entry.id for item in ctx.loader.entries() if item.entry.disabled]``.
    """

    id: str
    explicit_id: bool
    use: str | None = None
    config: Any = None  # raw file data for a plugin imported at runtime (ctx.plugin ruling)
    disabled: bool = False
    isolate: dict[str, str | None] = field(default_factory=lambda: {})
    intercept: RawTable = field(default_factory=lambda: {})
    inject: list[str] = field(default_factory=lambda: [])
    restart: RawTable | None = None
    load_timeout: float | None = None
    dispose_timeout: float | None = None
    group: str | None = None
    include: str | None = None
    children: list[Entry] = field(default_factory=lambda: [])
    source: Path
    location: tuple[str | int, ...]
    overlay: Overlay | None = None

    @property
    def policy(self) -> RestartPolicy | None:
        """The :class:`~pycordis.supervise.RestartPolicy` of :attr:`restart` (``None`` if absent).

        Example: ``Entry(..., restart={"mode": "on-failure"}).policy.mode == "on-failure"``.
        """
        return None if self.restart is None else RestartPolicy.from_mapping(self.restart)


@final
@dataclass
class LoaderOptions:
    """Options from the root config file's ``[loader]`` table.

    Example: ``LoaderOptions(allow=["app.*"]).allow``.
    """

    readonly: bool = False
    watch: bool = False
    allow: list[str] = field(default_factory=lambda: ["*"])
    poll_interval: float = 1.0
