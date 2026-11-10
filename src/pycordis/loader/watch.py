"""Polling file watcher for loader hot reload (spec §6.6).

Stdlib only: every ``interval`` seconds each watched file is ``stat``-ed; a
change of ``(st_mtime_ns, st_size)`` is confirmed by a SHA-256 of the content,
so touching a file or rewriting identical bytes is not a change. A file that
disappears (an editor saving via rename) or reappears is a change. Changes are
debounced: the callback runs once the files have been quiet for ``debounce``
seconds.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import final

__all__ = ["FileWatcher"]

type _Signature = tuple[int, int, bytes] | None
"""``(mtime_ns, size, sha256)`` of a file, ``None`` when it cannot be read."""


@final
class FileWatcher:
    """Polls the files returned by ``paths()`` and awaits ``on_change()`` when they change.

    ``paths`` is called on every poll, so the watched set may change (e.g. after
    a reload adds an include); a newly listed path is recorded silently and a
    path no longer listed is forgotten. Run :meth:`run` as a task (the loader
    uses ``ctx.spawn``); cancel it to stop watching. An exception raised by
    ``on_change`` ends :meth:`run`.
    """

    def __init__(
        self,
        paths: Callable[[], list[Path]],
        interval: float,
        on_change: Callable[[], Awaitable[None]],
        *,
        debounce: float = 0.2,
    ) -> None:
        self.paths = paths
        """Returns the files to watch."""
        self.interval = interval
        """Seconds between polls."""
        self.on_change = on_change
        """Awaited after a (debounced) change."""
        self.debounce = debounce
        """Seconds the files must stay unchanged before :attr:`on_change` runs."""
        self._seen: dict[Path, _Signature] = {}

    async def run(self) -> None:
        """Watch until cancelled."""
        self._scan()  # baseline
        while True:
            await asyncio.sleep(self.interval)
            if not self._scan():
                continue
            while True:
                await asyncio.sleep(self.debounce)
                if not self._scan():
                    break
            await self.on_change()
            self._scan(track_only=True)  # baseline paths the change added

    def unchanged(self, path: Path) -> bool:
        """Whether ``path`` still matches its recorded state (``True`` when not watched).

        The loader checks this before writing a file: only a file that had no
        unnoticed external change is :meth:`record`-ed after the write, so such a
        change still triggers a reload.
        """
        if path not in self._seen:
            return True
        seen = self._seen[path]
        return _digest(_signature(path, seen)) == _digest(seen)

    def record(self, path: Path) -> None:
        """Record ``path``'s current content as seen (after the owner's own write)."""
        self._seen[path] = _signature(path, None)

    def _scan(self, *, track_only: bool = False) -> bool:
        """Refresh the recorded states; return whether a known file changed."""
        changed = False
        current: dict[Path, _Signature] = {}
        for path in self.paths():
            if path in current:
                continue
            if path not in self._seen:
                current[path] = _signature(path, None)
                continue
            seen = self._seen[path]
            if track_only:
                current[path] = seen
                continue
            signature = _signature(path, seen)
            changed = changed or _digest(signature) != _digest(seen)
            current[path] = signature
        self._seen = current
        return changed


def _signature(path: Path, seen: _Signature) -> _Signature:
    """The state of ``path``, hashing its content only when its stat differs from ``seen``."""
    try:
        stat = path.stat()
        if seen is not None and (stat.st_mtime_ns, stat.st_size) == seen[:2]:
            return seen
        digest = hashlib.sha256(path.read_bytes()).digest()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size, digest)


def _digest(signature: _Signature) -> bytes | None:
    """The content hash of a signature (``None`` for a missing file)."""
    return None if signature is None else signature[2]
