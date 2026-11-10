"""The ``loader`` service: reconciles a config entry tree into live fibers (spec §6.6).

Each entry maps to one fiber, created under its parent's context after applying
the entry's ``isolate`` labels and ``intercept`` configs. A group (or include)
is itself a fiber whose body creates its children in its own context, so
disposing it tears down the whole subtree. The entry's extra ``inject`` names
are merged into the new fiber's dependency map from an ``internal/plugin``
listener, before dependencies are resolved (the cordis loader pattern).

Config interpolation (``${env:...}``, ``${file:...}``) runs in an
``internal/config`` waterfall listener for fibers owned by the loader, so a
fiber's ``raw_config`` stays un-expanded. Errors of one entry (import denied or
failed, invalid config, a failing body) are logged with the entry id and never
stop its siblings.

Reconciliation diffs the current tree against a new one by entry id, per group,
in file order: removed entries are disposed, added ones created; a change of
``use`` / ``group`` / ``include`` / ``isolate`` / ``intercept`` / ``inject``
or of the supervision keys ``restart`` / ``load_timeout`` / ``dispose_timeout``
recreates the entry; ``disabled = true`` disposes it but keeps it in the tree;
a change of ``config`` only goes through ``fiber.update(config, no_save=True)``
(so plugins may veto the restart); unchanged entries are untouched.

Write-back: ``internal/commit(fiber, config)`` of a loader-owned plugin fiber
(``fiber.update()`` without ``no_save``) stores ``config`` in the in-memory
entry and writes it to the file the entry is declared in, at
``entry.location + ("config",)``. :meth:`Loader.create`, :meth:`Loader.update`
and :meth:`Loader.remove` persist their change the same way, and
:meth:`Loader.save` flushes every entry's keys. Each write edits only the
affected keys (see :mod:`pycordis.loader.persist`), refuses to
touch an entry that is no longer at its recorded location, and replaces the
file atomically. ``[loader] readonly = true`` disables every write. Write
errors are logged (``save()`` raises them, as an ``ExceptionGroup``).

Secrets: a committed config is first re-templated against the entry's raw
config: a string equal to the expansion of the raw ``${...}`` string at the
same path (now, or as seen by an earlier resolution of the entry) is replaced
by that raw string, so a plugin passing its resolved config back keeps the
file's placeholders. If the result would still persist a secret of the entry
(see :mod:`pycordis.loader.interpolate`: ``${file:...}`` contents and
``${env:...}`` values of at least 8 characters, remembered for the entry's
lifetime), the commit is refused and logged (``cannot save <path>: would
persist an interpolated secret``) and the in-memory entry keeps its old config.
Resolution re-templates the same way before expanding (and puts the
placeholders back into the fiber's retained raw config), so expanded values
are never interpolated twice; an update candidate that fails to resolve leaves
the raw config untouched. :meth:`Loader.update` applies the same guard to a
``config`` patch and :meth:`Loader.create` refuses a new config holding one of
its own secrets literally; both log the refusal and raise ``ValueError``.
Plugins should still pass raw config (a modified copy of ``fiber.raw_config``).

Profiles: with :attr:`LoaderConfig.profile`, every file is layered with its
overlay (see :mod:`pycordis.loader.files` for the merge rules) and an entry
an overlay redefines keeps both declarations
(:class:`~pycordis.loader.entry.Overlay`). Every write above then targets the
file that defines the value: a key, or a ``config`` path, the overlay defines
goes to the overlay, anything else to the base file; a removed key leaves
both. ``remove`` deletes the entry's overlay tables too (those of its subtree
in the same file included); ``create`` writes to the file declaring the
parent group. Each file is checked to still hold its declaration before it
is written, and the two files are written separately (one write can fail,
logged, while the other succeeds). Reconciliation is unchanged: an overlay edit restarts
(or ``update``-s) only the entries whose merged values changed.

Hot reload (``[loader] watch = true``): a :class:`~pycordis.loader.watch.FileWatcher`
spawned on the loader's fiber polls the root file, every included file and
their profile overlays (also ones that do not exist yet) every
``poll_interval`` seconds and runs :meth:`Loader.reload` on a change; an
invalid file is logged once (``config reload failed: ...``) and the running
tree is kept. The loader's own writes do not trigger a reload.

The reconciliation engine is :mod:`pycordis.loader.tree`; the entry-level
``create`` / ``update`` / ``remove`` edits are :mod:`pycordis.loader.edits`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

from pycordis.loader.edits import create_entry, remove_entry, update_entry
from pycordis.loader.entry import Entry, LoaderOptions
from pycordis.loader.overlay import PROFILE_ENV, check_profile
from pycordis.loader.tree import LoaderTree
from pycordis.loader.writeback import save_tree
from pycordis.service import Service
from pycordis.supervise import check_timeout

if TYPE_CHECKING:
    from pycordis.context import Context
    from pycordis.fiber import Fiber

__all__ = ["LoadedEntry", "Loader", "LoaderConfig"]


@dataclass
class LoaderConfig:
    """Options of the :class:`Loader` service.

    ``path`` is the root config file (relative to the working directory).
    ``watch``, ``allow`` and ``readonly`` override the file's ``[loader]``
    options when not ``None`` (CLI flags). ``dispose_timeout`` (seconds) bounds
    the unload of every entry that sets none itself, neither by its
    ``dispose_timeout`` key nor by plugin metadata (the CLI passes 10);
    ``None`` leaves such unloads unbounded. ``profile`` selects config
    overlays: each config file ``<stem><suffix>`` is layered with
    ``<stem>.<profile><suffix>`` next to it (next to the symlink *target* when
    the file is a symlink, as paths are resolved; see
    :mod:`pycordis.loader.files`). When ``profile`` is ``None`` the
    ``PYCORDIS_PROFILE`` environment variable is used if set and non-empty (an
    explicit ``profile``, like the CLI's ``--profile``, wins). Names are
    letters, digits, ``_`` and ``-``; anything else raises ``ValueError``
    (naming ``PYCORDIS_PROFILE`` when it came from there). Overlays are trusted
    like the base file: an overlay's ``[loader] allow`` can widen the plugin
    allow-list. Example::

        LoaderConfig(path="app.toml", dispose_timeout=10.0, profile="dev")
    """

    path: str = "pycordis.toml"
    watch: bool | None = None
    allow: list[str] | None = None
    readonly: bool | None = None
    dispose_timeout: float | None = None
    profile: str | None = None

    def __post_init__(self) -> None:
        check_timeout(self.dispose_timeout, "dispose_timeout")
        if self.profile is not None:
            check_profile(self.profile)
            return
        name = os.environ.get(PROFILE_ENV) or None
        if name is not None:
            try:
                self.profile = check_profile(name)
            except ValueError as error:
                raise ValueError(f"{PROFILE_ENV}: {error}") from None


@final
@dataclass
class LoadedEntry:
    """An entry of the loaded tree and its live fiber.

    ``fiber`` is ``None`` while the entry is disabled, inside an inactive group,
    or when its plugin could not be imported or started.

    ``entry.config`` is the config last applied to the tree. It can differ from
    the fiber's running config when ``fiber.update()`` rejected it (the error
    is logged and the fiber keeps its previous config).

    Example: ``ctx.loader.get("llm").fiber``.
    """

    entry: Entry
    fiber: Fiber | None


class Loader(Service[Any], provide="loader"):
    """The ``loader`` service: loads a config file into fibers and keeps them in sync.

    Used as ``ctx.plugin(Loader, LoaderConfig(path=...))`` (or constructed
    directly), then ``await ctx.loader.start()``. Entries run in the loader's
    own context, so disposing the loader disposes every entry.

    ``start()``, ``reload()``, ``create()``, ``update()`` and ``remove()`` wait
    for the affected fibers to settle; do not await them from the body of a
    plugin the loader itself is loading. Each applies its change to the tree
    synchronously, before its first await, so concurrent calls (e.g.
    ``asyncio.gather(loader.reload(), loader.reload())``) are safe and the
    last applied state wins. Once the loader's fiber is disposed they raise
    ``CordisError("INACTIVE_EFFECT")``.

    Example::

        ctx.plugin(Loader, LoaderConfig(path=Path("app.toml"), allow=["app.*"]))
        await ctx.loader.start()
    """

    def __init__(self, ctx: Context, config: LoaderConfig | None = None) -> None:
        super().__init__(ctx)
        self._tree = LoaderTree(ctx, config if isinstance(config, LoaderConfig) else LoaderConfig())

    @property
    def path(self) -> Path:
        """The root config file.

        Example: ``ctx.loader.path.name == "app.toml"``.
        """
        return self._tree.path

    @property
    def options(self) -> LoaderOptions:
        """Effective options: the file's ``[loader]`` table with overrides applied.

        Example: ``ctx.loader.options.allow``.
        """
        return self._tree.options

    @property
    def started(self) -> bool:
        """Whether :meth:`start` has loaded the config file and created its entries.

        ``False`` until then (and when ``start()`` raised for an invalid file);
        it stays ``True`` afterwards. The fibers may still be loading: see
        :meth:`entries` for their states. The health server's ``/readyz``
        is not ready while it is ``False``.

        Example: ``assert ctx.loader.started``.
        """
        return self._tree.started

    def owned_fiber_count(self) -> int:
        """Number of live plugin fibers the loader currently owns.

        Internal — not part of the public API.

        Example: ``ctx.loader.owned_fiber_count() == len(ctx.loader.entries())``.
        """
        return len(self._tree.owned)

    async def start(self) -> None:
        """Load the config file and create its entries.

        Raises :class:`~pycordis.loader.LoaderConfigError` (or ``RuntimeError``
        for YAML without PyYAML) when the file tree is invalid; per-entry errors
        are logged instead.

        Example: ``await ctx.loader.start()``.
        """
        tree = self._tree
        tree.ctx.fiber.assert_active()
        tree.apply(tree.load())
        tree.started = True
        tree.watch()
        await tree.settle()

    async def reload(self) -> None:
        """Re-read the config file tree and reconcile the running entries with it.

        When the file tree is invalid, the error is logged once (``config reload
        failed: ...``) and the running tree is left untouched. Starts or stops
        the file watcher when ``watch`` changed.

        Example: ``await ctx.loader.reload()``.
        """
        await self._tree.reload()

    async def create(self, entry: Mapping[str, object], parent: str | None = None) -> Fiber | None:
        """Add ``entry`` (a config table) to the top level or to the group ``parent``.

        The entry is appended to the in-memory tree and its fiber returned once
        its load settled (a failing plugin body leaves it ``FAILED``). A disabled
        entry, or one whose parent group is not active, is added without a fiber
        and ``None`` is returned; it starts when enabled / when the group
        activates. Raises ``KeyError`` for an unknown ``parent``, ``ValueError``
        when ``parent`` is not a group,
        :class:`~pycordis.loader.LoaderConfigError` for an invalid entry or
        duplicate id, ``ValueError`` when its config holds the expansion of one
        of its own ``${...}`` secrets literally (logged too), and the import
        error when its plugin cannot be imported (the entry is then not added).
        A disabled or inactive entry is not imported yet: its import, ``allow``
        check and config validation happen when it is enabled, and failures are
        then logged with its entry id.

        Example: ``await ctx.loader.create({"id": "chat", "use": "app:chat"})``.
        """
        return await create_entry(self._tree, entry, parent)

    async def update(self, id: str, patch: Mapping[str, object]) -> None:
        """Change keys of the entry ``id`` and reconcile it.

        ``patch`` may set ``use``, ``config``, ``disabled``, ``isolate``,
        ``intercept``, ``inject``, ``restart``, ``load_timeout`` and
        ``dispose_timeout``; a ``None`` value removes the key. Raises
        ``KeyError`` for an unknown id, ``ValueError`` for another key and
        :class:`~pycordis.loader.LoaderConfigError` for an invalid result. A
        ``config`` is re-templated and guarded like a commit (see the module
        docstring); a refused one is logged and raises ``ValueError``, and the
        entry is left unchanged.

        Example: ``await ctx.loader.update("chat", {"disabled": True})``.
        """
        await update_entry(self._tree, id, patch)

    async def remove(self, id: str) -> None:
        """Remove the entry ``id`` (with its subtree) and dispose its fiber.

        Raises ``KeyError`` for an unknown id.

        Example: ``await ctx.loader.remove("chat")``.
        """
        await remove_entry(self._tree, id)

    def get(self, id: str) -> LoadedEntry | None:
        """The entry ``id`` and its fiber, or ``None`` when there is no such entry.

        Example: ``ctx.loader.get("chat")``.
        """
        node = self._tree.nodes.get(id)
        return None if node is None else LoadedEntry(node.entry, node.fiber)

    def entries(self) -> list[LoadedEntry]:
        """Every entry of the tree, depth first in file order.

        Example: ``[item.entry.id for item in ctx.loader.entries()]``.
        """
        return [LoadedEntry(node.entry, node.fiber) for node in self._tree.walk()]

    async def save(self) -> None:
        """Write every entry's keys back to the file it is declared in.

        Each entry is rewritten in place at its recorded location; keys whose
        file value already means the same are left alone, so an in-sync tree
        writes nothing. Does nothing when ``readonly``. Every file is attempted;
        the failures are then raised together as an ``ExceptionGroup`` (in file
        order) of :class:`~pycordis.loader.LoaderConfigError` when a file cannot
        be read or an entry is no longer at its location (the file changed since
        it was loaded), ``TypeError`` for a value the format cannot represent,
        and ``OSError`` when writing fails.

        Example: ``await ctx.loader.save()``.
        """
        save_tree(self._tree)
