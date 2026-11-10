from __future__ import annotations

import asyncio
import os
import textwrap
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import pycordis.loader.tree as tree_module
from pycordis.context import Context
from pycordis.loader.service import Loader, LoaderConfig
from pycordis.loader.watch import FileWatcher
from pycordis.testing import MemoryExporter, mock_context, settle

FAST = "[loader]\nwatch = true\npoll_interval = 0.05\n"


def write(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def bump(path: Path) -> None:
    """Move the mtime forward so a same-size rewrite within one tick is still noticed."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


async def eventually(check: Callable[[], bool], timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.02)


@asynccontextmanager
async def started(
    path: Path, logs: MemoryExporter | None = None, **options: Any
) -> AsyncGenerator[tuple[Context, Loader], None]:
    async with mock_context(logs) as root:
        await root.plugin(Loader, LoaderConfig(path=str(path), **options))
        loader: Loader = root.loader
        await loader.start()
        await settle(root)
        yield root, loader


def count_loads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []
    real = tree_module.load_tree

    def spy(path: Path, **kwargs: Any) -> Any:
        calls.append(path)
        return real(path, **kwargs)

    monkeypatch.setattr(tree_module, "load_tree", spy)
    return calls


# -- FileWatcher ------------------------------------------------------------------


async def test_file_watcher_detects_changes(tmp_path: Path) -> None:
    a, b = tmp_path / "a.toml", tmp_path / "b.toml"
    a.write_text("x = 1\n")
    paths = [a]
    changes = 0

    async def on_change() -> None:
        nonlocal changes
        changes += 1

    watcher = FileWatcher(lambda: paths, 0.01, on_change, debounce=0.02)
    task = asyncio.create_task(watcher.run())
    try:
        await asyncio.sleep(0.05)
        assert changes == 0
        a.write_text("x = 2\n")
        bump(a)
        await eventually(lambda: changes == 1)
        # same content, new mtime: no change
        bump(a)
        await asyncio.sleep(0.1)
        assert changes == 1
        # a new path is baselined silently; a removed file is a change
        b.write_text("y = 1\n")
        paths.append(b)
        await asyncio.sleep(0.1)
        assert changes == 1
        b.unlink()
        await eventually(lambda: changes == 2)
        # it comes back: another change
        b.write_text("y = 1\n")
        await eventually(lambda: changes == 3)
        # the owner's own write, recorded, is not a change
        assert watcher.unchanged(a)
        a.write_text("x = 33\n")
        assert not watcher.unchanged(a)
        watcher.record(a)
        assert watcher.unchanged(a)
        await asyncio.sleep(0.1)
        assert changes == 3
        # an untracked path counts as unchanged
        assert watcher.unchanged(tmp_path / "other.toml")
        # paths dropped from the set are forgotten
        paths.remove(b)
        await asyncio.sleep(0.05)
        assert watcher.unchanged(b)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_file_watcher_debounces(tmp_path: Path) -> None:
    a = tmp_path / "a.toml"
    a.write_text("x = 0\n")
    changes = 0

    async def on_change() -> None:
        nonlocal changes
        changes += 1

    watcher = FileWatcher(lambda: [a], 0.01, on_change, debounce=0.1)
    task = asyncio.create_task(watcher.run())
    try:
        await asyncio.sleep(0.03)
        for i in range(1, 6):  # a burst of saves within the debounce window
            a.write_text(f"x = {i}\n")
            bump(a)
            await asyncio.sleep(0.03)
        await eventually(lambda: changes == 1)
        await asyncio.sleep(0.2)
        assert changes == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# -- hot reload -----------------------------------------------------------------


def config(name: str, extra: str = "") -> str:
    entry = f'[[plugins]]\nid = "a"\nuse = "loadfix:echo"\nconfig = {{ name = "{name}"{extra} }}\n'
    return f"{FAST}\n{entry}"


async def test_hot_reload_applies_changes(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, config("a"))
    logs = MemoryExporter()
    async with started(path, logs) as (_root, loader):
        assert loadfix.runs["a"] == 1
        write(path, config("a", ", v = 2"))
        bump(path)
        await eventually(lambda: loadfix.runs["a"] == 2)
        assert loadfix.configs["a"] == {"name": "a", "v": 2}

        # a syntax error keeps the running tree and logs once
        write(path, config("a", ", v = "))
        bump(path)
        await eventually(lambda: len(logs.text("error")) > 0)
        await asyncio.sleep(0.4)
        assert len(logs.text("error")) == 1
        assert "config reload failed: " in logs.text("error")[0]
        assert loadfix.runs["a"] == 2
        entry = loader.get("a")
        assert entry is not None and entry.fiber is not None

        # the next valid save applies
        write(path, config("a", ", v = 3"))
        bump(path)
        await eventually(lambda: loadfix.runs["a"] == 3)
        assert loadfix.configs["a"] == {"name": "a", "v": 3}
        assert len(logs.text("error")) == 1


async def test_own_write_does_not_reload(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, config("a"))
    loads = count_loads(monkeypatch)
    async with started(path) as (_root, loader):
        assert len(loads) == 1
        entry = loader.get("a")
        assert entry is not None and entry.fiber is not None
        await entry.fiber.update({"name": "a", "v": 9})
        await loader.create({"id": "n", "use": "loadfix:echo"})
        await asyncio.sleep(0.5)
        assert len(loads) == 1
        assert "v = 9" in path.read_text()
        # an external change afterwards is still noticed
        write(path, path.read_text().replace("v = 9", "v = 10"))
        bump(path)
        await eventually(lambda: len(loads) == 2)


async def test_external_edit_before_own_write_still_reloads(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, config("a").replace("poll_interval = 0.05", "poll_interval = 0.3"))
    loads = count_loads(monkeypatch)
    async with started(path) as (_root, loader):
        await asyncio.sleep(0.05)  # the watcher took its baseline
        # an external edit lands, then the loader writes before the next poll
        write(path, path.read_text() + '\n[[plugins]]\nid = "x"\nuse = "loadfix:echo"\n')
        bump(path)
        entry = loader.get("a")
        assert entry is not None and entry.fiber is not None
        await entry.fiber.update({"name": "a", "v": 1})
        await eventually(lambda: loader.get("x") is not None)
        assert len(loads) == 2


async def test_hot_reload_watches_includes(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, f'{FAST}\n[[plugins]]\nid = "inc"\ninclude = "sub.toml"\n')
    sub = tmp_path / "sub.toml"
    write(sub, '[[plugins]]\nid = "s"\nuse = "loadfix:echo"\nconfig = { name = "s" }\n')
    logs = MemoryExporter()
    async with started(path, logs) as (_root, loader):
        write(sub, '[[plugins]]\nid = "s"\nuse = "loadfix:echo"\nconfig = { name = "t" }\n')
        bump(sub)
        await eventually(lambda: loadfix.runs["t"] == 1)

        # a new include that is broken, then fixed: the fix is noticed
        extra = tmp_path / "extra.toml"
        write(extra, "plugins = [\n")
        write(path, path.read_text() + '\n[[plugins]]\nid = "x"\ninclude = "extra.toml"\n')
        bump(path)
        await eventually(lambda: len(logs.text("error")) == 1)
        write(extra, '[[plugins]]\nid = "e"\nuse = "loadfix:echo"\nconfig = { name = "e" }\n')
        bump(extra)
        await eventually(lambda: loadfix.runs["e"] == 1)

        # a missing include that is created later
        write(path, path.read_text() + '\n[[plugins]]\nid = "y"\ninclude = "later.toml"\n')
        bump(path)
        await eventually(lambda: len(logs.text("error")) == 2)
        write(
            tmp_path / "later.toml",
            '[[plugins]]\nid = "l"\nuse = "loadfix:echo"\nconfig = { name = "l" }\n',
        )
        await eventually(lambda: loadfix.runs["l"] == 1)
        assert loader.get("l") is not None


async def test_file_removed_temporarily(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, config("a"))
    logs = MemoryExporter()
    async with started(path, logs) as (_root, loader):
        text = path.read_text()
        path.unlink()
        await eventually(lambda: len(logs.text("error")) == 1)
        entry = loader.get("a")
        assert entry is not None and entry.fiber is not None
        path.write_text(text.replace('"a" }', '"a", v = 5 }'))
        await eventually(lambda: loadfix.configs["a"] == {"name": "a", "v": 5})


async def test_watch_toggled_by_reload(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, config("a"))
    loads = count_loads(monkeypatch)
    async with started(path) as (_root, loader):
        # turning watch off through the file stops the watcher
        write(path, config("a").replace("watch = true", "watch = false"))
        bump(path)
        await eventually(lambda: len(loads) == 2)
        assert not loader.options.watch
        write(path, config("b").replace("watch = true", "watch = false"))
        bump(path)
        await asyncio.sleep(0.4)
        assert len(loads) == 2
        # a manual reload turns it back on
        write(path, config("c"))
        await loader.reload()
        assert loader.options.watch
        write(path, config("d"))
        bump(path)
        await eventually(lambda: loadfix.runs["d"] == 1)


async def test_watch_disabled_by_default(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, config("a").replace("watch = true", "watch = false"))
    loads = count_loads(monkeypatch)
    async with started(path) as (_root, _loader):
        write(path, config("b").replace("watch = true", "watch = false"))
        bump(path)
        await asyncio.sleep(0.4)
        assert len(loads) == 1


async def test_hot_reload_error_is_logged(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, config("a"))
    logs = MemoryExporter()
    async with started(path, logs) as (_root, _loader):

        def broken(*args: Any, **kwargs: Any) -> Any:
            # not a config error: escapes the reload and reaches the watcher's handler
            raise ValueError("unexpected")

        monkeypatch.setattr(tree_module, "load_tree", broken)
        write(path, config("b"))
        bump(path)
        await eventually(lambda: len(logs.text("error")) >= 1)
        assert "unexpected" in logs.text("error")[0]
