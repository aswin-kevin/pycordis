from __future__ import annotations

import asyncio
import textwrap
import tomllib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pycordis.context import Context
from pycordis.errors import CordisError
from pycordis.fiber import Fiber, FiberState
from pycordis.loader import LoaderConfigError
from pycordis.loader.service import LoadedEntry, Loader, LoaderConfig
from pycordis.testing import MemoryExporter, mock_context, settle

BASE = """
[[plugins]]
id = "llm"
use = "loadfix:Provider"

[[plugins]]
id = "dep"
use = "loadfix:dependent"

[[plugins]]
id = "grp"
group = "g"
plugins = [
  { id = "a", use = "loadfix:echo", config = { name = "a", v = 1 } },
  { id = "b", use = "loadfix:echo", config = { name = "b" } },
]
"""


def write(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")


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


def loaded_entry(loader: Loader, id: str) -> LoadedEntry:
    loaded = loader.get(id)
    assert loaded is not None, id
    return loaded


def fiber(loader: Loader, id: str) -> Fiber:
    loaded = loader.get(id)
    assert loaded is not None and loaded.fiber is not None, id
    return loaded.fiber


def state(loader: Loader, id: str) -> FiberState | None:
    loaded = loader.get(id)
    assert loaded is not None, id
    return None if loaded.fiber is None else loaded.fiber.state


async def reload(root: Context, loader: Loader, path: Path, text: str) -> None:
    write(path, text)
    await loader.reload()
    await settle(root)


def errors(logs: MemoryExporter) -> str:
    return "\n".join(logs.text("error"))


# -- start --------------------------------------------------------------------


async def test_start_loads_tree(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (_root, loader):
        assert [e.entry.id for e in loader.entries()] == ["llm", "dep", "grp", "a", "b"]
        for id in ("llm", "dep", "grp", "a", "b"):
            assert state(loader, id) == FiberState.ACTIVE, id
        assert loadfix.runs == {"llm:?": 1, "dep:?": 1, "a": 1, "b": 1}
        # children run inside the group's fiber
        assert fiber(loader, "a").parent.fiber is fiber(loader, "grp")
        assert loader.get("missing") is None
        loaded = loader.get("a")
        assert isinstance(loaded, LoadedEntry)
        assert loaded.entry.config == {"name": "a", "v": 1}


async def test_loaded_plugin_body_sees_its_own_context(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(
        path,
        """
        [[plugins]]
        id = "amb"
        use = "loadfix:ambient"
        config = { name = "amb" }
        """,
    )
    async with started(path) as (root, loader):
        ctx, current = loadfix.seen["amb"]
        assert current is ctx
        assert ctx.fiber is fiber(loader, "amb")
        assert current is not root


async def test_loader_is_a_service(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, "")
    async with mock_context() as root:
        loader = Loader(root, LoaderConfig(path=str(path)))
        assert root.loader is not None
        await loader.start()
        assert loader.entries() == []


async def test_start_raises_config_errors(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, "plugins = 1")
    async with mock_context() as root:
        loader = Loader(root, LoaderConfig(path=str(path)))
        with pytest.raises(LoaderConfigError):
            await loader.start()


# -- reconciliation -----------------------------------------------------------


async def test_config_only_change_restarts_once(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (root, loader):
        before = {id: fiber(loader, id) for id in ("llm", "dep", "grp", "a", "b")}
        await reload(root, loader, path, BASE.replace("v = 1", "v = 2"))
        assert loadfix.runs == {"llm:?": 1, "dep:?": 1, "a": 2, "b": 1}
        assert loadfix.disposed == {"a": 1}
        assert loadfix.configs["a"] == {"name": "a", "v": 2}
        # every fiber kept: the config change went through fiber.update()
        assert {id: fiber(loader, id) for id in before} == before
        assert fiber(loader, "a").raw_config == {"name": "a", "v": 2}
        assert loader.get("a") == LoadedEntry(loaded_entry(loader, "a").entry, before["a"])


async def test_unchanged_reload_touches_nothing(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (root, loader):
        await reload(root, loader, path, BASE + "\n# just a comment\n")
        assert loadfix.runs == {"llm:?": 1, "dep:?": 1, "a": 1, "b": 1}
        assert loadfix.disposed == {}


async def test_use_change_recreates(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (root, loader):
        old = fiber(loader, "a")
        text = BASE.replace('id = "a", use = "loadfix:echo"', 'id = "a", use = "loadfix:other"')
        await reload(root, loader, path, text)
        assert old.state == FiberState.DISPOSED
        assert fiber(loader, "a") is not old
        assert state(loader, "a") == FiberState.ACTIVE
        assert loadfix.runs == {"llm:?": 1, "dep:?": 1, "a": 1, "b": 1, "other:a": 1}
        assert loadfix.disposed == {"a": 1}


async def test_structural_keys_recreate(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (root, loader):
        old = fiber(loader, "dep")
        text = BASE.replace(
            'id = "dep"\nuse = "loadfix:dependent"',
            'id = "dep"\nuse = "loadfix:dependent"\nintercept = { llm = { level = 3 } }',
        )
        await reload(root, loader, path, text)
        assert fiber(loader, "dep") is not old
        assert loadfix.seen["dep:?"] == {"level": 3}
        assert loadfix.runs["dep:?"] == 2


async def test_remove_disposes(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (root, loader):
        old = fiber(loader, "b")
        await reload(
            root,
            loader,
            path,
            BASE.replace('  { id = "b", use = "loadfix:echo", config = { name = "b" } },\n', ""),
        )
        assert old.state == FiberState.DISPOSED
        assert loader.get("b") is None
        assert loadfix.disposed == {"b": 1}
        assert loadfix.runs == {"llm:?": 1, "dep:?": 1, "a": 1, "b": 1}


async def test_disable_and_reenable(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    disabled = BASE.replace('config = { name = "b" }', 'config = { name = "b" }, disabled = true')
    async with started(path) as (root, loader):
        old = fiber(loader, "b")
        await reload(root, loader, path, disabled)
        assert old.state == FiberState.DISPOSED
        loaded = loader.get("b")
        assert loaded is not None and loaded.fiber is None and loaded.entry.disabled
        assert loadfix.disposed == {"b": 1}
        await reload(root, loader, path, BASE)
        assert state(loader, "b") == FiberState.ACTIVE
        assert loadfix.runs["b"] == 2


async def test_disabled_at_start(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE.replace('group = "g"', 'group = "g"\ndisabled = true'))
    async with started(path) as (_root, loader):
        assert [state(loader, id) for id in ("grp", "a", "b")] == [None, None, None]
        assert loadfix.runs == {"llm:?": 1, "dep:?": 1}


async def test_group_disposal_tears_down_subtree(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        children = [fiber(loader, "a"), fiber(loader, "b")]
        await reload(
            root, loader, path, BASE.replace('group = "g"', 'group = "g"\ndisabled = true')
        )
        assert [f.state for f in children] == [FiberState.DISPOSED] * 2
        assert loadfix.disposed == {"a": 1, "b": 1}
        assert [state(loader, id) for id in ("grp", "a", "b")] == [None, None, None]
        # re-enabling the group recreates its children
        await reload(root, loader, path, BASE)
        assert [state(loader, id) for id in ("grp", "a", "b")] == [FiberState.ACTIVE] * 3
        # removing the group removes the subtree
        await reload(root, loader, path, BASE.split('[[plugins]]\nid = "grp"')[0])
        assert [loader.get(id) for id in ("grp", "a", "b")] == [None, None, None]
        # the group teardown and the loader both dispose the children: idempotent
        assert loadfix.disposed == {"a": 2, "b": 2}
        assert errors(logs) == ""


async def test_group_children_reconcile(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (root, loader):
        group = fiber(loader, "grp")
        text = BASE.replace(
            '{ id = "b", use = "loadfix:echo", config = { name = "b" } },',
            '{ id = "c", use = "loadfix:echo", config = { name = "c" } },',
        )
        await reload(root, loader, path, text)
        assert fiber(loader, "grp") is group
        assert loader.get("b") is None
        assert state(loader, "c") == FiberState.ACTIVE
        assert fiber(loader, "c").parent.fiber is group
        assert loadfix.runs == {"llm:?": 1, "dep:?": 1, "a": 1, "b": 1, "c": 1}
        assert [e.entry.id for e in loader.entries()] == ["llm", "dep", "grp", "a", "c"]


async def test_entry_moves_between_groups(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (root, loader):
        text = (
            BASE.replace('  { id = "b", use = "loadfix:echo", config = { name = "b" } },\n', "")
            + '\n[[plugins]]\nid = "b"\nuse = "loadfix:echo"\nconfig = { name = "b" }\n'
        )
        await reload(root, loader, path, text)
        assert state(loader, "b") == FiberState.ACTIVE
        assert fiber(loader, "b").parent.fiber is not fiber(loader, "grp")
        assert loadfix.runs["b"] == 2
        assert [e.entry.id for e in loader.entries()] == ["llm", "dep", "grp", "a", "b"]


async def test_create_disabled_entry(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (_root, loader):
        created = await loader.create(
            {"id": "d", "use": "loadfix:echo", "config": {"name": "d"}, "disabled": True}
        )
        assert created is None
        assert loader.entries()[-1] == LoadedEntry(loaded_entry(loader, "d").entry, None)
        assert loadfix.runs["d"] == 0
        await loader.update("d", {"disabled": False})
        assert state(loader, "d") == FiberState.ACTIVE
        assert loadfix.runs["d"] == 1


async def test_inject_extends_dependencies(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    text = """
    [[plugins]]
    id = "x"
    use = "loadfix:echo"
    inject = ["llm"]
    config = { name = "x" }
    """
    write(path, text)
    async with started(path) as (root, loader):
        assert state(loader, "x") == FiberState.PENDING
        assert "llm" in fiber(loader, "x").inject
        await reload(root, loader, path, text + '\n[[plugins]]\nuse = "loadfix:Provider"\n')
        assert state(loader, "x") == FiberState.ACTIVE
        assert loadfix.runs["x"] == 1


async def test_group_inject_gates_subtree(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    text = """
    [[plugins]]
    id = "grp"
    group = "g"
    inject = ["llm"]
    plugins = [{ id = "a", use = "loadfix:echo", config = { name = "a" } }]
    """
    write(path, text)
    async with started(path) as (root, loader):
        assert state(loader, "grp") == FiberState.PENDING
        assert state(loader, "a") is None
        # reconciling into an inactive group only updates the tree
        changed = text.replace('{ name = "a" }', '{ name = "a", v = 2 }')
        await reload(root, loader, path, changed)
        assert state(loader, "a") is None
        await reload(root, loader, path, changed + '\n[[plugins]]\nuse = "loadfix:Provider"\n')
        assert state(loader, "a") == FiberState.ACTIVE
        assert loadfix.configs["a"] == {"name": "a", "v": 2}


async def test_isolate_labels(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(
        path,
        """
        [[plugins]]
        use = "loadfix:Provider"
        isolate = { llm = "shared" }

        [[plugins]]
        id = "same"
        use = "loadfix:dependent"
        isolate = { llm = "shared" }
        config = { name = "same" }

        [[plugins]]
        id = "private"
        use = "loadfix:dependent"
        isolate = ["llm"]
        config = { name = "private" }

        [[plugins]]
        id = "outside"
        use = "loadfix:dependent"
        config = { name = "outside" }
        """,
    )
    async with started(path) as (_root, loader):
        assert state(loader, "same") == FiberState.ACTIVE
        assert state(loader, "private") == FiberState.PENDING
        assert state(loader, "outside") == FiberState.PENDING


# -- per-entry errors ---------------------------------------------------------


async def test_missing_env_var_fails_one_entry(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review Focus 4: a missing env var fails only its entry, logged with id and name."""
    monkeypatch.delenv("PYCORDIS_TEST_SECRET", raising=False)
    path = tmp_path / "pycordis.toml"
    text = """
    [[plugins]]
    id = "needs-secret"
    use = "loadfix:echo"
    config = { name = "s", key = "${env:PYCORDIS_TEST_SECRET}" }

    [[plugins]]
    id = "sibling"
    use = "loadfix:echo"
    config = { name = "sib" }
    """
    write(path, text)
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        assert state(loader, "needs-secret") == FiberState.FAILED
        assert state(loader, "sibling") == FiberState.ACTIVE
        text_ = errors(logs)
        assert "PYCORDIS_TEST_SECRET" in text_
        assert 'loader entry "needs-secret"' in text_
        assert loadfix.runs == {"sib": 1}
        # the raw config stays un-expanded
        raw = fiber(loader, "needs-secret").raw_config
        assert raw["key"] == "${env:PYCORDIS_TEST_SECRET}"
        # fixing the environment and changing the config recovers the entry
        monkeypatch.setenv("PYCORDIS_TEST_SECRET", "hunter2")
        await reload(root, loader, path, text.replace('name = "s"', 'name = "s2"'))
        assert state(loader, "needs-secret") == FiberState.ACTIVE
        assert loadfix.configs["s2"]["key"] == "hunter2"
        assert fiber(loader, "needs-secret").raw_config["key"] == "${env:PYCORDIS_TEST_SECRET}"


async def test_file_interpolation_relative_to_entry_source(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "secret.txt").write_text("s3cret\n", encoding="utf-8")
    write(
        sub / "extra.toml",
        """
        [[plugins]]
        id = "inner"
        use = "loadfix:echo"
        config = { name = "inner", key = "${file:secret.txt}" }
        """,
    )
    path = tmp_path / "pycordis.toml"
    write(path, '[[plugins]]\nid = "inc"\ninclude = "sub/extra.toml"\n')
    async with started(path) as (_root, loader):
        assert state(loader, "inc") == FiberState.ACTIVE
        assert loadfix.configs["inner"]["key"] == "s3cret"
        assert fiber(loader, "inner").parent.fiber is fiber(loader, "inc")


async def test_import_denied_by_allow(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(
        path,
        """
        [loader]
        allow = ["nothing.*"]

        [[plugins]]
        id = "denied"
        use = "json:dumps"

        [[plugins]]
        id = "ok"
        use = "loadfix:echo"
        config = { name = "ok" }
        """,
    )
    logs = MemoryExporter()
    # the LoaderConfig option overrides the file's [loader] allow
    async with started(path, logs, allow=["loadfix"]) as (_root, loader):
        assert loader.options.allow == ["loadfix"]
        assert state(loader, "denied") is None
        assert state(loader, "ok") == FiberState.ACTIVE
        text = errors(logs)
        assert 'plugin module "json" is not allowed by loader.allow' in text
        assert 'loader entry "denied"' in text
        assert len(logs.text("error")) == 1


async def test_bad_plugins_are_per_entry(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(
        path,
        """
        [[plugins]]
        id = "missing"
        use = "pycordis_no_such_module:x"

        [[plugins]]
        id = "not-plugin"
        use = "loadfix:not_a_plugin"

        [[plugins]]
        id = "fails"
        use = "loadfix:failing"

        [[plugins]]
        id = "ok"
        use = "loadfix:echo"
        config = { name = "ok" }
        """,
    )
    logs = MemoryExporter()
    async with started(path, logs) as (_root, loader):
        assert state(loader, "missing") is None
        assert state(loader, "not-plugin") is None
        assert state(loader, "fails") == FiberState.FAILED
        assert state(loader, "ok") == FiberState.ACTIVE
        text = errors(logs)
        for id in ("missing", "not-plugin", "fails"):
            assert f'loader entry "{id}"' in text
        assert "body failed" in text
        failure = next(m for m in logs.messages if "body failed" in str(m.args[0]))
        reported = failure.args[0]
        assert isinstance(reported, BaseException)
        assert reported.__notes__ == ["in plugin <failing>", 'in loader entry "fails"']


async def test_reload_config_error_keeps_tree(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        before = [(e.entry, e.fiber) for e in loader.entries()]
        await reload(root, loader, path, BASE + "\n[[plugins]]\nbogus = 1\n")
        assert [(e.entry, e.fiber) for e in loader.entries()] == before
        assert len(logs.text("error")) == 1
        assert 'unknown key "bogus"' in errors(logs)
        assert loadfix.disposed == {}
        # a later valid file applies
        await reload(root, loader, path, BASE.replace("v = 1", "v = 3"))
        assert loadfix.configs["a"]["v"] == 3


async def test_reload_invalid_config_update_is_logged(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        old = fiber(loader, "a")
        await reload(root, loader, path, BASE.replace("v = 1", 'v = "${env:PYCORDIS_NOPE_X}"'))
        # fiber.update() rejects the candidate synchronously: the entry keeps running
        assert fiber(loader, "a") is old and old.state == FiberState.ACTIVE
        assert loadfix.configs["a"] == {"name": "a", "v": 1}
        text = errors(logs)
        assert "PYCORDIS_NOPE_X" in text and 'loader entry "a"' in text


# -- API ------------------------------------------------------------------------


async def test_create_update_remove(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (_root, loader):
        created = await loader.create({"use": "loadfix:echo", "config": {"name": "n"}})
        assert created is not None and created.state == FiberState.ACTIVE
        new = loader.entries()[-1]
        assert new.fiber is created
        assert new.entry.id == "/loadfix:echo#0"
        assert new.entry.location == ("plugins", 3)
        again = await loader.create({"use": "loadfix:echo", "config": {"name": "m"}})
        assert loader.entries()[-1].entry.id == "/loadfix:echo#1"
        assert again is not None and again.state == FiberState.ACTIVE

        child = await loader.create(
            {"id": "c", "use": "loadfix:echo", "config": {"name": "c"}}, "grp"
        )
        assert child is not None and child.parent.fiber is fiber(loader, "grp")
        assert [e.entry.id for e in loader.entries()][3:6] == ["a", "b", "c"]
        assert loaded_entry(loader, "c").entry.location == ("plugins", 2, "plugins", 2)

        await loader.update("c", {"config": {"name": "c", "v": 2}})
        assert loadfix.runs["c"] == 2 and loadfix.configs["c"] == {"name": "c", "v": 2}
        assert fiber(loader, "c") is child

        await loader.update("c", {"use": "loadfix:other"})
        assert fiber(loader, "c") is not child
        assert loadfix.runs["other:c"] == 1
        assert loaded_entry(loader, "c").entry.explicit_id

        await loader.update("/loadfix:echo#0", {"disabled": True})
        assert state(loader, "/loadfix:echo#0") is None
        assert not loaded_entry(loader, "/loadfix:echo#1").entry.explicit_id
        await loader.update("/loadfix:echo#0", {"disabled": None})
        assert state(loader, "/loadfix:echo#0") == FiberState.ACTIVE

        await loader.update("grp", {"intercept": {"llm": {"level": 1}}})
        assert state(loader, "a") == FiberState.ACTIVE
        assert [e.entry.id for e in loader.entries()][3:6] == ["a", "b", "c"]

        await loader.remove("grp")
        assert [loader.get(id) for id in ("grp", "a", "b", "c")] == [None] * 4
        await loader.remove("/loadfix:echo#0")
        assert loader.get("/loadfix:echo#0") is None
        assert [e.entry.id for e in loader.entries()] == ["llm", "dep", "/loadfix:echo#1"]


async def test_api_errors(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path, allow=["loadfix"]) as (_root, loader):
        with pytest.raises(KeyError, match="nope"):
            await loader.create({"use": "loadfix:echo"}, "nope")
        with pytest.raises(ValueError, match='entry "a" is not a group'):
            await loader.create({"use": "loadfix:echo"}, "a")
        with pytest.raises(LoaderConfigError, match="duplicate entry id"):
            await loader.create({"id": "a", "use": "loadfix:echo"})
        with pytest.raises(PermissionError):
            await loader.create({"use": "json:dumps"})
        with pytest.raises(TypeError, match="invalid plugin"):
            await loader.create({"id": "np", "use": "loadfix:not_a_plugin"})
        with pytest.raises(KeyError, match="nope"):
            await loader.update("nope", {})
        with pytest.raises(ValueError, match='cannot patch "id"'):
            await loader.update("a", {"id": "z"})
        with pytest.raises(LoaderConfigError, match="config"):
            await loader.update("grp", {"config": {}})
        with pytest.raises(KeyError, match="nope"):
            await loader.remove("nope")
        assert [e.entry.id for e in loader.entries()] == ["llm", "dep", "grp", "a", "b"]


async def test_api_groups_includes_and_scopes(tmp_path: Path, loadfix: ModuleType) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    write(sub / "extra.toml", "")
    path = tmp_path / "pycordis.toml"
    write(
        path,
        """
        [[plugins]]
        id = "inc"
        include = "sub/extra.toml"

        [[plugins]]
        id = "off"
        group = "off"
        disabled = true
        """,
    )
    async with started(path, watch=True, readonly=True) as (root, loader):
        assert loader.path == path
        assert (loader.options.watch, loader.options.readonly) == (True, True)
        child = await loader.create({"use": "loadfix:echo", "config": {"name": "i"}}, "inc")
        assert child is not None
        assert child.parent.fiber is fiber(loader, "inc")
        created = loader.get("inc/loadfix:echo#0")
        assert created is not None
        assert created.entry.source == (sub / "extra.toml").resolve()
        assert created.entry.location == ("plugins", 0)
        # under an inactive group: added without a fiber, started when the group activates
        assert (
            await loader.create(
                {"id": "o1", "use": "loadfix:echo", "config": {"name": "o1"}}, "off"
            )
            is None
        )
        assert [(e.entry.id, e.fiber) for e in loader.entries()][-1] == ("o1", None)
        await loader.update("off", {"disabled": None})
        assert state(loader, "o1") == FiberState.ACTIVE
        assert fiber(loader, "o1").parent.fiber is fiber(loader, "off")

        group = await loader.create(
            {"id": "h", "group": "h", "plugins": [{"id": "h1", "use": "loadfix:echo"}]}
        )
        assert group is not None and group.state == FiberState.ACTIVE
        assert state(loader, "h1") == FiberState.ACTIVE
        assert fiber(loader, "h1").parent.fiber is group

        # patching keeps the other keys (isolate, intercept, inject)
        await loader.create(
            {
                "id": "x",
                "use": "loadfix:echo",
                "config": {"name": "x"},
                "isolate": ["cache"],
                "intercept": {"llm": {"level": 2}},
                "inject": ["llm"],
            }
        )
        assert state(loader, "x") == FiberState.PENDING
        await loader.update("x", {"config": {"name": "x", "v": 1}})
        entry = loaded_entry(loader, "x").entry
        assert entry.isolate == {"cache": None}
        assert entry.intercept == {"llm": {"level": 2}}
        assert entry.inject == ["llm"]
        await loader.update("x", {"isolate": {"cache": "shared"}, "inject": None})
        assert loaded_entry(loader, "x").entry.isolate == {"cache": "shared"}
        assert state(loader, "x") == FiberState.ACTIVE
        await settle(root)


async def test_unrelated_plugins_are_ignored(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (root, loader):
        configs: list[Any] = []

        def plain(_ctx: Context, config: Any) -> None:
            configs.append(config)

        await root.plugin(plain, {"key": "${env:NOT_EXPANDED}"})
        assert configs == [{"key": "${env:NOT_EXPANDED}"}]
        assert len(loader.entries()) == 5


async def test_disposing_the_loader_disposes_entries(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with mock_context() as root:
        loader_fiber = await root.plugin(Loader, LoaderConfig(path=str(path)))
        loader: Loader = root.loader
        await loader.start()
        entries = [e.fiber for e in loader.entries()]
        await loader_fiber.dispose()
        assert all(f is not None and f.state == FiberState.DISPOSED for f in entries)
        assert loadfix.disposed == {"dep:?": 1, "a": 1, "b": 1}
        # nothing of the loader outlives it
        assert loader.owned_fiber_count() == 0
        assert all(e.fiber is None for e in loader.entries())
        configs: list[Any] = []

        def plain(_ctx: Context, config: Any) -> None:
            configs.append(config)

        await root.plugin(plain, {"key": "${env:NOT_EXPANDED}"})
        assert configs == [{"key": "${env:NOT_EXPANDED}"}]
        for call in (
            loader.start(),
            loader.reload(),
            loader.create({"use": "loadfix:echo"}),
            loader.update("a", {}),
            loader.remove("a"),
        ):
            with pytest.raises(CordisError, match="inactive context"):
                await call


async def test_settle_gives_up(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pycordis.loader.tree as tree

    monkeypatch.setattr(tree, "_MAX_SETTLE", 0)
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        loader = Loader(root, LoaderConfig(path=str(path)))
        await loader.start()
        assert logs.text("warn") == ["loader tree did not settle"]
        await settle(root)


async def test_group_restart_recreates_children(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    group = """
    [[plugins]]
    id = "grp"
    group = "g"
    inject = ["llm"]
    plugins = [{ id = "a", use = "loadfix:echo", config = { name = "a" } }]
    """
    provider = '\n[[plugins]]\nid = "llm"\nuse = "loadfix:Provider"\n'
    write(path, group + provider)
    async with started(path) as (root, loader):
        first = fiber(loader, "a")
        await reload(root, loader, path, group)
        assert state(loader, "grp") == FiberState.PENDING
        assert first.state == FiberState.DISPOSED and state(loader, "a") is None
        await reload(root, loader, path, group + provider)
        assert state(loader, "a") == FiberState.ACTIVE
        assert fiber(loader, "a") is not first
        assert loadfix.runs["a"] == 2 and loadfix.disposed["a"] == 1


async def test_concurrent_reloads(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (root, loader):
        write(path, BASE.replace("v = 1", "v = 2"))
        await asyncio.gather(loader.reload(), loader.reload())
        await settle(root)
        assert loadfix.runs["a"] == 2 and loadfix.configs["a"]["v"] == 2
        assert loadfix.runs["b"] == 1


async def test_reload_while_group_body_in_flight(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with mock_context() as root:
        loader = Loader(root, LoaderConfig(path=str(path)))

        async def change() -> None:
            # start() has created the group fiber, whose body has not run yet
            assert state(loader, "grp") == FiberState.LOADING
            assert state(loader, "a") is None
            write(path, BASE.replace("v = 1", "v = 2"))
            await loader.reload()

        await asyncio.gather(loader.start(), change())
        await settle(root)
        assert state(loader, "a") == FiberState.ACTIVE
        assert loadfix.runs["a"] == 1 and loadfix.configs["a"]["v"] == 2


async def test_unchanged_reload_keeps_group_children(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, BASE)
    async with started(path) as (root, loader):
        await reload(root, loader, path, BASE)
        assert [e.entry.id for e in loader.entries()] == ["llm", "dep", "grp", "a", "b"]


# -- supervision keys --------------------------------------------------------------------


async def test_entry_restart_policy_and_timeouts(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(
        path,
        """
        [[plugins]]
        id = "flaky"
        use = "loadfix:flaky"
        config = { name = "f", failures = 1 }
        restart = { mode = "on-failure", backoff = { initial = 0.01, jitter = 0 } }
        load_timeout = 5
        dispose_timeout = 0.5
        """,
    )
    logs = MemoryExporter()
    async with started(path, logs) as (_root, loader):
        flaky = fiber(loader, "flaky")
        async with asyncio.timeout(2):
            while flaky.state != FiberState.ACTIVE:
                await asyncio.sleep(0.005)
        assert loadfix.runs["f"] == 2
        assert flaky.policy.mode == "on-failure"
        assert (flaky.load_timeout, flaky.dispose_timeout) == (5.0, 0.5)
        assert "flaky run 1" in errors(logs)
        assert any("restarting flaky in 0.01s (attempt 1/5)" in line for line in logs.text("warn"))


async def test_supervision_key_change_recreates_the_entry(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, '[[plugins]]\nid = "a"\nuse = "loadfix:echo"\nconfig = { name = "a" }\n')
    async with started(path) as (root, loader):
        before = fiber(loader, "a")
        assert before.load_timeout is None
        await reload(
            root,
            loader,
            path,
            '[[plugins]]\nid = "a"\nuse = "loadfix:echo"\nconfig = { name = "a" }\n'
            "load_timeout = 3\n",
        )
        after = fiber(loader, "a")
        assert after is not before
        assert after.load_timeout == 3.0
        assert loadfix.runs["a"] == 2


async def test_loader_dispose_timeout_is_the_entry_default(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    path = tmp_path / "pycordis.toml"
    write(
        path,
        """
        [[plugins]]
        id = "hung"
        use = "loadfix:hung_close"
        config = { name = "hung" }

        [[plugins]]
        id = "own"
        use = "loadfix:echo"
        dispose_timeout = 3

        [[plugins]]
        id = "meta"
        use = "loadfix:declared"
        """,
    )
    logs = MemoryExporter()
    async with started(path, logs, dispose_timeout=0.02) as (_root, loader):
        assert fiber(loader, "hung").dispose_timeout == 0.02
        assert fiber(loader, "own").dispose_timeout == 3.0  # the entry key wins
        assert fiber(loader, "meta").dispose_timeout == 7.0  # then plugin metadata
        async with asyncio.timeout(1):
            await loader.remove("hung")
        assert "disposer of effect <function hung_close.<locals>.close" in errors(logs)
        assert "in <hung_close> did not finish within 0.02s; abandoned" in errors(logs)


def test_loader_config_rejects_a_bad_dispose_timeout() -> None:
    with pytest.raises(ValueError, match='"dispose_timeout" must be a positive number'):
        LoaderConfig(dispose_timeout=0)


async def test_update_patches_supervision_keys(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, '[[plugins]]\nid = "a"\nuse = "loadfix:echo"\n')
    async with started(path) as (_root, loader):
        await loader.update(
            "a", {"restart": {"mode": "on-failure"}, "load_timeout": 4, "dispose_timeout": 2}
        )
        a = fiber(loader, "a")
        assert a.policy.mode == "on-failure"
        assert (a.load_timeout, a.dispose_timeout) == (4.0, 2.0)
        (written,) = tomllib.loads(path.read_text())["plugins"]
        assert written["restart"] == {"mode": "on-failure"}
        assert (written["load_timeout"], written["dispose_timeout"]) == (4, 2)
        with pytest.raises(LoaderConfigError, match='"load_timeout" must be a positive number'):
            await loader.update("a", {"load_timeout": -1})
        await loader.update("a", {"restart": None, "load_timeout": None, "dispose_timeout": None})
        assert fiber(loader, "a").policy.mode == "never"
        assert "restart" not in path.read_text()
