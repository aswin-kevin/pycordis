"""Config profiles: ``<stem>.<profile><suffix>`` overlays layered over a config file (spec §3.9)."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import textwrap
import tomllib
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pycordis import cli
from pycordis.context import Context
from pycordis.fiber import Fiber
from pycordis.loader import Entry, LoaderConfigError, Overlay, load_tree
from pycordis.loader.overlay import deep_merge, overlay_path, shifted, split_keys
from pycordis.loader.service import Loader, LoaderConfig
from pycordis.testing import MemoryExporter, mock_context, settle

BASE = """\
[loader]
allow = ["loadfix"]

[[plugins]]
id = "a"
use = "loadfix:echo"
config = { name = "a", model = "small", opts = { temp = 1, top = 5 }, tags = ["x", "y"] }

[[plugins]]
id = "b"
use = "loadfix:echo"
config = { name = "b" }

[[plugins]]
id = "grp"
group = "g"
plugins = [
  { id = "c", use = "loadfix:echo", config = { name = "c" } },
]
"""

OVERLAY = """\
[[plugins]]
id = "a"
config = { model = "large", opts = { temp = 2 }, tags = ["z"] }
"""


def write(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def setup(tmp_path: Path, base: str = BASE, overlay: str | None = OVERLAY) -> tuple[Path, Path]:
    path = tmp_path / "app.toml"
    write(path, base)
    dev = tmp_path / "app.dev.toml"
    if overlay is not None:
        write(dev, overlay)
    return path, dev


def by_id(entries: list[Entry]) -> dict[str, Entry]:
    found: dict[str, Entry] = {}

    def visit(items: list[Entry]) -> None:
        for entry in items:
            found[entry.id] = entry
            visit(entry.children)

    visit(entries)
    return found


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


def fiber(loader: Loader, id: str) -> Fiber:
    loaded = loader.get(id)
    assert loaded is not None and loaded.fiber is not None, id
    return loaded.fiber


def entry_of(loader: Loader, id: str) -> Entry:
    loaded = loader.get(id)
    assert loaded is not None, id
    return loaded.entry


async def eventually(check: Callable[[], bool], timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.02)


# -- parsing and merge rules -------------------------------------------------------


def test_overlay_path_is_next_to_the_file() -> None:
    assert overlay_path(Path("/etc/app.toml"), "dev") == Path("/etc/app.dev.toml")
    assert overlay_path(Path("conf/base.json"), "prod_1") == Path("conf/base.prod_1.json")


@pytest.mark.parametrize("name", ["", "../x", "a/b", "a\\b", "..", ".", "dev.toml", "a b"])
def test_profile_names_are_validated(tmp_path: Path, name: str) -> None:
    path, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="invalid profile name"):
        load_tree(path, profile=name)
    with pytest.raises(ValueError, match="invalid profile name"):
        LoaderConfig(path=str(path), profile=name)


def test_missing_overlay_is_a_no_op(tmp_path: Path) -> None:
    path, _ = setup(tmp_path, overlay=None)
    files: list[Path] = []
    assert load_tree(path, profile="dev", files=files) == load_tree(path)
    # the absent overlay is still watched, so creating it triggers a reload
    assert tmp_path / "app.dev.toml" in files


def test_without_profile_the_overlay_is_ignored(tmp_path: Path) -> None:
    path, _ = setup(tmp_path)
    entries, _ = load_tree(path)
    assert by_id(entries)["a"].config["model"] == "small"
    assert by_id(entries)["a"].overlay is None


def test_config_is_deep_merged(tmp_path: Path) -> None:
    path, dev = setup(tmp_path)
    entries, _ = load_tree(path, profile="dev")
    a = by_id(entries)["a"]
    # dicts merged, lists replaced, other keys untouched
    assert a.config == {
        "name": "a",
        "model": "large",
        "opts": {"temp": 2, "top": 5},
        "tags": ["z"],
    }
    assert a.source == path and a.location == ("plugins", 0)
    assert a.overlay == Overlay(
        source=dev,
        location=("plugins", 0),
        table={"config": {"model": "large", "opts": {"temp": 2}, "tags": ["z"]}},
        base={
            "id": "a",
            "use": "loadfix:echo",
            "config": {
                "name": "a",
                "model": "small",
                "opts": {"temp": 1, "top": 5},
                "tags": ["x", "y"],
            },
        },
    )
    assert by_id(entries)["b"].overlay is None


def test_overlay_schema_key_is_ignored(tmp_path: Path) -> None:
    path, _ = setup(tmp_path, overlay='"$schema" = "./overlay.schema.json"\n' + OVERLAY)
    entries, _ = load_tree(path, profile="dev")
    assert by_id(entries)["a"].config["model"] == "large"


def test_entry_keys_override_and_disable(tmp_path: Path) -> None:
    path, _ = setup(
        tmp_path,
        overlay="""\
        [[plugins]]
        id = "b"
        use = "loadfix:other"
        disabled = true
        inject = ["x"]
        restart = { mode = "on-failure" }

        [[plugins]]
        id = "grp"
        disabled = true
        """,
    )
    found = by_id(load_tree(path, profile="dev")[0])
    b = found["b"]
    assert (b.use, b.disabled, b.inject, b.config) == ("loadfix:other", True, ["x"], {"name": "b"})
    assert b.policy is not None and b.policy.mode == "on-failure"
    assert found["grp"].disabled and found["grp"].group == "g"


def test_config_replaces_a_non_table(tmp_path: Path) -> None:
    path, _ = setup(
        tmp_path,
        base='[[plugins]]\nid = "a"\nuse = "m:p"\nconfig = 3\n'
        '[[plugins]]\nid = "b"\nuse = "m:p"\nconfig = { x = 1 }\n'
        '[[plugins]]\nid = "c"\nuse = "m:p"\n',
        overlay='[[plugins]]\nid = "a"\nconfig = { x = 1 }\n'
        '[[plugins]]\nid = "b"\nconfig = "s"\n'
        '[[plugins]]\nid = "c"\nconfig = { y = 2 }\n',
    )
    found = by_id(load_tree(path, profile="dev")[0])
    assert [found[k].config for k in "abc"] == [{"x": 1}, "s", {"y": 2}]


def test_unmatched_entries_are_appended_to_their_group(tmp_path: Path) -> None:
    path, dev = setup(
        tmp_path,
        overlay="""\
        [[plugins]]
        use = "loadfix:echo"
        config = { name = "extra" }

        [[plugins]]
        id = "new"
        use = "loadfix:echo"

        [[plugins]]
        id = "grp"
        plugins = [
          { id = "c", config = { name = "c2" } },
          { use = "loadfix:echo" },
        ]
        """,
    )
    entries, _ = load_tree(path, profile="dev")
    assert [e.id for e in entries] == ["a", "b", "grp", "/loadfix:echo#0", "new"]
    extra = entries[3]
    assert (extra.source, extra.location, extra.overlay) == (dev, ("plugins", 0), None)
    group = entries[2]
    assert [e.id for e in group.children] == ["c", "grp/loadfix:echo#0"]
    assert group.children[0].config == {"name": "c2"}
    assert group.children[0].overlay is not None
    assert group.children[0].overlay.location == ("plugins", 2, "plugins", 0)
    assert group.children[1].location == ("plugins", 2, "plugins", 1)


def test_generated_ids_continue_after_the_base_siblings(tmp_path: Path) -> None:
    path, _ = setup(
        tmp_path,
        base='[[plugins]]\nuse = "m:p"\n',
        overlay='[[plugins]]\nuse = "m:p"\n',
    )
    entries, _ = load_tree(path, profile="dev")
    assert [e.id for e in entries] == ["/m:p#0", "/m:p#1"]


def test_ids_match_anywhere_in_the_file(tmp_path: Path) -> None:
    path, _ = setup(tmp_path, overlay='[[plugins]]\nid = "c"\nconfig = { name = "deep" }\n')
    group = load_tree(path, profile="dev")[0][2]
    assert [e.id for e in group.children] == ["c"]
    assert group.children[0].config == {"name": "deep"}


def test_loader_options_are_overridden(tmp_path: Path) -> None:
    path, _ = setup(tmp_path, overlay="[loader]\nreadonly = true\npoll_interval = 2\n")
    _, options = load_tree(path, profile="dev")
    assert (options.allow, options.readonly, options.poll_interval) == (["loadfix"], True, 2.0)


def test_included_files_have_their_own_overlays(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    write(sub / "part.json", json.dumps({"plugins": [{"id": "p", "use": "m:p", "config": {}}]}))
    write(sub / "part.dev.json", json.dumps({"plugins": [{"id": "p", "config": {"v": 1}}]}))
    path, _ = setup(
        tmp_path, base='[[plugins]]\nid = "inc"\ninclude = "sub/part.json"\n', overlay=None
    )
    files: list[Path] = []
    entries, _ = load_tree(path, profile="dev", files=files)
    part = entries[0].children[0]
    assert part.config == {"v": 1}
    assert part.overlay is not None and part.overlay.source == sub / "part.dev.json"
    assert {tmp_path / "app.dev.toml", sub / "part.dev.json"} <= set(files)


def test_overlay_can_override_a_json_or_yaml_base(tmp_path: Path) -> None:
    path = tmp_path / "app.json"
    write(path, json.dumps({"plugins": [{"id": "a", "use": "m:p", "config": {"x": 1}}]}))
    write(tmp_path / "app.dev.json", json.dumps({"plugins": [{"id": "a", "config": {"y": 2}}]}))
    assert load_tree(path, profile="dev")[0][0].config == {"x": 1, "y": 2}


@pytest.mark.parametrize(
    ("overlay", "message"),
    [
        ('[[plugins]]\nid = "a"\nnope = 1\n', 'unknown key "nope" in entry plugins[0]'),
        ('[[plugins]]\nid = "a"\ndisabled = "yes"\n', '"disabled" must be a boolean'),
        ('[[plugins]]\nid = "a"\nload_timeout = -1\n', "load_timeout"),
        ('[[plugins]]\nid = "a"\nuse = ""\n', '"use" must be a non-empty string'),
        ('[[plugins]]\nid = "a"\ngroup = "x"\n', 'cannot change "group" of entry "a"'),
        ('[[plugins]]\nid = "grp"\ngroup = "x"\n', 'cannot change "group" of entry "grp"'),
        ('[[plugins]]\nid = "grp"\nuse = "m:p"\n', 'cannot change "use" of entry "grp"'),
        ('[[plugins]]\nid = "grp"\nconfig = {}\n', '"config" is only allowed on "use" entries'),
        ('[[plugins]]\nid = "a"\nplugins = []\n', '"plugins" is only allowed on group entries'),
        ('[[plugins]]\nid = "grp"\nplugins = 3\n', '"plugins" must be an array'),
        ('[[plugins]]\nid = "a"\n[[plugins]]\nid = "a"\n', 'entry "a" is overlaid twice'),
        ("plugins = [1]\n", "entry plugins[0] must be a table"),
        ("plugins = 1\n", '"plugins" must be an array'),
        ("other = 1\n", 'unknown top-level key "other"'),
        ('"$schema" = false\n', '"$schema" must be a string'),
        ('[loader]\nwatch = "x"\n', '[loader] "watch" must be a boolean'),
        ('[[plugins]]\nid = "zz"\nconfig = {}\n', 'an entry must have exactly one of "use"'),
        ('[[plugins]]\nid = "b"\nuse = "m:p"\n[[plugins]]\nid = "b"\n', "overlaid twice"),
        ("plugins = [\n", "Invalid value"),
    ],
)
def test_overlay_errors_name_the_overlay(tmp_path: Path, overlay: str, message: str) -> None:
    path, dev = setup(tmp_path, overlay=overlay)
    with pytest.raises(LoaderConfigError, match=re.escape(message)) as info:
        load_tree(path, profile="dev")
    assert str(info.value).startswith(f"{dev}:")


def test_appended_duplicate_id_names_the_overlay(tmp_path: Path) -> None:
    write(tmp_path / "part.toml", '[[plugins]]\nid = "p"\nuse = "m:p"\n')
    path, dev = setup(
        tmp_path,
        base='[[plugins]]\nid = "inc"\ninclude = "part.toml"\n',
        # "p" lives in the included file: the root overlay cannot match it
        overlay='[[plugins]]\nid = "p"\nuse = "m:q"\n',
    )
    with pytest.raises(LoaderConfigError, match='duplicate entry id "p"') as info:
        load_tree(path, profile="dev")
    assert str(info.value).startswith(f"{dev}:")


def test_include_overlay_may_not_set_loader_options(tmp_path: Path) -> None:
    write(tmp_path / "part.toml", '[[plugins]]\nid = "p"\nuse = "m:p"\n')
    write(tmp_path / "part.dev.toml", "[loader]\nwatch = true\n")
    path, _ = setup(tmp_path, base='[[plugins]]\nid = "inc"\ninclude = "part.toml"\n', overlay=None)
    with pytest.raises(LoaderConfigError, match=r"\[loader\] is only allowed in the root") as info:
        load_tree(path, profile="dev")
    assert str(info.value).startswith(f"{tmp_path / 'part.dev.toml'}:")


@pytest.mark.parametrize(
    "base", ["plugins = [1]\n", '[[plugins]]\nid = "g"\ngroup = "g"\nplugins = 3\n']
)
def test_a_malformed_base_is_reported_with_an_overlay(tmp_path: Path, base: str) -> None:
    path, _ = setup(tmp_path, base=base, overlay='[[plugins]]\nid = "g"\ndisabled = true\n')
    with pytest.raises(LoaderConfigError) as info:
        load_tree(path, profile="dev")
    assert str(info.value).startswith(f"{path}:")


def test_shifted_moves_only_later_elements_of_the_same_array() -> None:
    removed = ("plugins", 1, "plugins", 2)
    assert shifted(("plugins", 1, "plugins", 3, "plugins", 0), removed) == (
        "plugins",
        1,
        "plugins",
        2,
        "plugins",
        0,
    )
    assert shifted(("plugins", 1, "plugins", 1), removed) == ("plugins", 1, "plugins", 1)
    assert shifted(("plugins", 2, "plugins", 3), removed) == ("plugins", 2, "plugins", 3)
    assert shifted(("plugins", 1), removed) == ("plugins", 1)


@pytest.mark.parametrize(
    ("base", "over", "new"),
    [
        ({"a": 1, "t": {"x": 1, "y": 2}}, {"t": {"x": 5}}, {"a": 2, "t": {"x": 6, "z": 3}}),
        ({"a": 1}, {"a": [1]}, {"a": [2]}),
        ({"t": 1}, {"t": {"x": 1}}, {"t": {"x": 2, "y": 3}}),
        ({"t": {"x": 1}}, {"t": 5}, {"t": {"x": 2}}),
        ({}, {"t": {"x": 1}}, {}),
        (3, {"x": 1}, "s"),
    ],
)
def test_split_keys_inverts_deep_merge(base: Any, over: Any, new: Any) -> None:
    overlay = Overlay(source=Path("o"), location=(), table={"config": over}, base={"config": base})
    base_patch, over_patch = split_keys(overlay, {"config": new}, changes_only=False)
    assert deep_merge(base_patch["config"], over_patch["config"]) == new


# -- reconciliation (Review Focus 5) ----------------------------------------------


async def test_overlay_change_restarts_only_that_entry(tmp_path: Path, loadfix: ModuleType) -> None:
    path, dev = setup(tmp_path)
    async with started(path, profile="dev") as (root, loader):
        assert loadfix.configs["a"]["model"] == "large"
        before = dict(loadfix.runs)
        others = {id: fiber(loader, id) for id in ("b", "c", "grp")}
        write(dev, OVERLAY.replace('"large"', '"huge"'))
        await loader.reload()
        await settle(root)
        assert loadfix.runs["a"] == before["a"] + 1
        assert loadfix.configs["a"]["model"] == "huge"
        assert {k: v for k, v in loadfix.runs.items() if k != "a"} == {
            k: v for k, v in before.items() if k != "a"
        }
        assert all(fiber(loader, id) is f for id, f in others.items())

        # the overlay disabling an entry stops only that entry
        write(
            dev, OVERLAY.replace('"large"', '"huge"') + '[[plugins]]\nid = "b"\ndisabled = true\n'
        )
        await loader.reload()
        await settle(root)
        assert loader.get("b") is not None and fiber_or_none(loader, "b") is None
        assert loadfix.disposed["b"] == 1 and loadfix.disposed["a"] == 1
        assert loadfix.runs["a"] == before["a"] + 1


def fiber_or_none(loader: Loader, id: str) -> Fiber | None:
    loaded = loader.get(id)
    assert loaded is not None
    return loaded.fiber


async def test_creating_the_overlay_triggers_a_hot_reload(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    fast = 'allow = ["loadfix"]\nwatch = true\npoll_interval = 0.05'
    path, dev = setup(tmp_path, base=BASE.replace('allow = ["loadfix"]', fast), overlay=None)
    async with started(path, profile="dev") as (_root, _loader):
        assert loadfix.configs["a"]["model"] == "small"
        write(dev, OVERLAY)
        await eventually(lambda: loadfix.configs["a"]["model"] == "large")
        assert loadfix.runs["b"] == 1


# -- write-back --------------------------------------------------------------------


async def test_commit_writes_each_value_to_its_file(tmp_path: Path, loadfix: ModuleType) -> None:
    path, dev = setup(tmp_path)
    logs = MemoryExporter()
    async with started(path, logs, profile="dev") as (root, loader):
        a = fiber(loader, "a")
        config = dict(a.config)
        config["model"] = "xl"  # overlay-defined
        config["opts"] = {"temp": 3, "top": 6}  # temp from the overlay, top from the base
        config["name"] = "a"
        config["new"] = True  # defined nowhere: the base
        await a.update(config)
        await settle(root)
        assert dev.read_text() == OVERLAY.replace(
            'model = "large", opts = { temp = 2 }', 'model = "xl", opts = { temp = 3 }'
        )
        assert path.read_text() == BASE.replace(
            'temp = 1, top = 5 }, tags = ["x", "y"] }',
            'temp = 1, top = 6 }, tags = ["x", "y"], new = true }',
        )
        # a reload sees no change: nothing restarts
        runs = dict(loadfix.runs)
        await loader.reload()
        await settle(root)
        assert dict(loadfix.runs) == runs

        # a removed key leaves both files
        await a.update({"name": "a", "opts": {"top": 6}, "tags": ["z"]})
        await settle(root)
        assert load_tree(path, profile="dev")[0][0].config == {
            "name": "a",
            "opts": {"top": 6},
            "tags": ["z"],
        }
        assert "temp" not in dev.read_text() and "model" not in path.read_text()
    assert logs.text("error") == []


async def test_commit_keeps_overlay_placeholders(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROFILE_TOKEN", "s3cret-token-value")
    overlay = '[[plugins]]\nid = "a"\nconfig = { token = "${env:PROFILE_TOKEN}" }\n'
    path, dev = setup(tmp_path, overlay=overlay)
    logs = MemoryExporter()
    async with started(path, logs, profile="dev") as (root, loader):
        a = fiber(loader, "a")
        assert loadfix.configs["a"]["token"] == "s3cret-token-value"
        resolved = dict(loadfix.configs["a"])
        resolved["model"] = "medium"
        await a.update(resolved)  # the expanded config handed back
        await settle(root)
        assert dev.read_text() == overlay
        assert 'model = "medium"' in path.read_text()
        assert "s3cret" not in path.read_text()

        # a literal secret in a base-owned key is refused
        leaked = dict(fiber(loader, "a").config)
        leaked["model"] = "s3cret-token-value"
        await fiber(loader, "a").update(leaked)
        await settle(root)
        assert "s3cret" not in path.read_text() + dev.read_text()
    assert any("would persist an interpolated secret" in line for line in logs.text("error"))


async def test_update_targets_the_defining_file(tmp_path: Path, loadfix: ModuleType) -> None:
    path, dev = setup(
        tmp_path,
        overlay='[[plugins]]\nid = "b"\nuse = "loadfix:other"\ninject = []\n',
    )
    async with started(path, profile="dev") as (_root, loader):
        await loader.update("b", {"inject": ["llm"], "disabled": True})
        # inject is overlay-defined, disabled is not
        assert load_overlay(dev) == [{"id": "b", "use": "loadfix:other", "inject": ["llm"]}]
        assert load_overlay(path)[1] == {
            "id": "b",
            "use": "loadfix:echo",
            "config": {"name": "b"},
            "disabled": True,
        }
        # use is overridden by the overlay, so the base's check still passes
        await loader.update("b", {"use": "loadfix:echo", "disabled": None, "inject": None})
        assert load_overlay(dev) == [{"id": "b", "use": "loadfix:echo"}]
        assert "disabled" not in load_overlay(path)[1]
        assert entry_of(loader, "b").overlay is not None


def load_overlay(path: Path) -> list[dict[str, Any]]:
    return tomllib.loads(path.read_text())["plugins"]


async def test_remove_deletes_the_entry_from_both_files(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    path, dev = setup(
        tmp_path,
        overlay="""\
        [[plugins]]
        id = "a"
        config = { model = "large" }

        [[plugins]]
        id = "c"
        config = { name = "c2" }

        [[plugins]]
        id = "b"
        config = { model = "b2" }

        [[plugins]]
        id = "grp"
        plugins = [{ id = "d", use = "loadfix:echo", config = { name = "d" } }]
        """,
    )
    async with started(path, profile="dev") as (root, loader):
        await loader.remove("a")
        assert [e["id"] for e in load_overlay(dev)] == ["c", "b", "grp"]
        assert [e["id"] for e in load_overlay(path)] == ["b", "grp"]
        # locations follow, so later writes still find their entries
        await fiber(loader, "b").update({"name": "b", "model": "b3"})
        await settle(root)
        assert load_overlay(dev)[1] == {"id": "b", "config": {"model": "b3"}}

        # a group takes its children's overlays (also matched elsewhere) with it
        await loader.remove("grp")
        assert [e["id"] for e in load_overlay(dev)] == ["b"]
        assert [e["id"] for e in load_overlay(path)] == ["b"]
        assert loadfix.disposed["c2"] == 1 and loadfix.disposed["d"] == 1


async def test_create_in_an_overlay_group_writes_the_overlay(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    path, dev = setup(
        tmp_path,
        overlay='[[plugins]]\nid = "og"\ngroup = "dev-only"\nplugins = []\n',
    )
    async with started(path, profile="dev") as (_root, loader):
        await loader.create({"id": "n", "use": "loadfix:echo", "config": {"name": "n"}}, "og")
        assert loadfix.runs["n"] == 1
        assert load_overlay(dev)[0]["plugins"] == [
            {"id": "n", "use": "loadfix:echo", "config": {"name": "n"}}
        ]
        await loader.create({"id": "m", "use": "loadfix:echo", "config": {"name": "m"}}, "grp")
        assert load_overlay(path)[2]["plugins"][1]["id"] == "m"
        n = entry_of(loader, "n")
        assert (n.source, n.location) == (dev, ("plugins", 0, "plugins", 0))
        m = entry_of(loader, "m")
        assert (m.source, m.location) == (path, ("plugins", 2, "plugins", 1))


async def test_save_writes_overlay_values_to_the_overlay(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    path, dev = setup(tmp_path, overlay=OVERLAY + '\n[[plugins]]\nid = "b"\ndisabled = false\n')
    async with started(path, profile="dev") as (_root, loader):
        base_text, dev_text = path.read_text(), dev.read_text()
        await loader.save()  # in sync: nothing written
        assert (path.read_text(), dev.read_text()) == (base_text, dev_text)

        entry_of(loader, "a").config["opts"]["temp"] = 9
        entry_of(loader, "a").config["opts"]["top"] = 7
        entry_of(loader, "b").disabled = True
        await loader.save()
        overlay = load_overlay(dev)
        assert overlay[0]["config"]["opts"] == {"temp": 9}
        assert overlay[1] == {"id": "b", "disabled": True}
        base = load_overlay(path)
        assert base[0]["config"]["opts"] == {"temp": 1, "top": 7}
        assert "disabled" not in base[1]


async def test_write_to_a_moved_overlay_entry_is_refused(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    path, dev = setup(tmp_path)
    logs = MemoryExporter()
    async with started(path, logs, profile="dev") as (root, loader):
        write(dev, '[[plugins]]\nid = "zz"\nuse = "loadfix:echo"\n')
        a = fiber(loader, "a")
        await a.update({**a.config, "model": "xl"})
        await settle(root)
    assert any('entry "a" is no longer at plugins[0]' in line for line in logs.text("error"))


# -- CLI and environment -----------------------------------------------------------


@pytest.fixture
def profile_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("PYCORDIS_PROFILE", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))  # main() may append the cwd
    path, _ = setup(tmp_path)
    write(tmp_path / "app.ci.toml", '[[plugins]]\nid = "b"\ndisabled = true\n')
    return path


def test_cli_profile_flag(
    profile_files: Path, loadfix: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["inspect", "-c", str(profile_files), "--profile", "ci"]) == 0
    assert "b  loadfix:echo  (disabled)" in capsys.readouterr().out


def test_cli_profile_from_environment(
    profile_files: Path,
    loadfix: ModuleType,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYCORDIS_PROFILE", "ci")
    assert cli.main(["inspect", "-c", str(profile_files)]) == 0
    assert "b  loadfix:echo  (disabled)" in capsys.readouterr().out
    # the flag wins over the environment
    assert cli.main(["inspect", "-c", str(profile_files), "--profile", "dev"]) == 0
    out = capsys.readouterr().out
    assert "(disabled)" not in out


def test_cli_rejects_bad_profile_names(
    profile_files: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["inspect", "-c", str(profile_files), "--profile", "../etc"])
    assert info.value.code == 2
    assert "invalid profile name" in capsys.readouterr().err
    monkeypatch.setenv("PYCORDIS_PROFILE", "a/b")
    assert cli.main(["inspect", "--run", "-c", str(profile_files)]) == 1
    assert "PYCORDIS_PROFILE: invalid profile name" in capsys.readouterr().err


def test_cli_run_uses_the_profile(profile_files: Path, loadfix: ModuleType) -> None:
    assert cli.main(["inspect", "--run", "-c", str(profile_files), "--profile", "ci"]) == 0
    assert (loadfix.runs["a"], loadfix.runs["b"]) == (1, 0)


# -- fix round 1 -------------------------------------------------------------------

DEFAULTED_BASE = """\
[loader]
allow = ["loadfix"]

[[plugins]]
id = "a"
use = "loadfix:echo"
disabled = true
inject = ["llm"]
config = { name = "a" }

[[plugins]]
id = "b"
use = "loadfix:echo"
config = { name = "b", k = 1 }
"""


async def test_save_keeps_base_values_an_overlay_resets_to_defaults(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    overlay = '[[plugins]]\nid = "a"\ndisabled = false\ninject = []\nintercept = {}\n'
    path, dev = setup(tmp_path, base=DEFAULTED_BASE, overlay=overlay)
    async with started(path, profile="dev") as (_root, loader):
        assert loadfix.runs["a"] == 1
        before = (path.read_bytes(), dev.read_bytes())
        await loader.save()
        assert (path.read_bytes(), dev.read_bytes()) == before


async def test_clearing_an_overlay_key_writes_its_default_to_the_overlay(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    overlay = '[[plugins]]\nid = "a"\ninject = ["db"]\nload_timeout = 5\n'
    base_text = DEFAULTED_BASE.replace('inject = ["llm"]\n', 'inject = ["llm"]\nload_timeout = 3\n')
    path, dev = setup(tmp_path, base=base_text, overlay=overlay)
    base = path.read_text()
    async with started(path, profile="dev") as (_root, loader):
        await loader.update("a", {"inject": None})
        # the base's inject stays; the overlay states the default explicitly
        assert path.read_text() == base
        assert load_overlay(dev)[0]["inject"] == []
        assert entry_of(loader, "a").inject == []
        # a key without a default form leaves both files
        await loader.update("a", {"load_timeout": None})
        assert "load_timeout" not in dev.read_text() + path.read_text()
        # the merged result after a reload is what was asked for
        a = by_id(load_tree(path, profile="dev")[0])["a"]
        assert (a.inject, a.load_timeout) == ([], None)


def test_loader_config_reads_the_profile_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYCORDIS_PROFILE", "staging")
    assert LoaderConfig().profile == "staging"
    assert LoaderConfig(profile="dev").profile == "dev"  # explicit wins
    monkeypatch.setenv("PYCORDIS_PROFILE", "")
    assert LoaderConfig().profile is None
    monkeypatch.setenv("PYCORDIS_PROFILE", "../etc")
    with pytest.raises(ValueError, match="PYCORDIS_PROFILE: invalid profile name"):
        LoaderConfig()


async def test_failed_overlay_write_keeps_its_provenance(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    path, dev = setup(tmp_path)
    logs = MemoryExporter()
    async with started(path, logs, profile="dev") as (root, loader):
        write(dev, '[[plugins]]\nid = "zz"\nuse = "loadfix:echo"\n')  # moved behind our back
        a = fiber(loader, "a")
        await a.update({**a.config, "model": "xl", "name": "a2"})
        await settle(root)
        overlay = entry_of(loader, "a").overlay
        assert overlay is not None
        assert overlay.table["config"]["model"] == "large"  # the failed share is not recorded
        assert overlay.base["config"]["name"] == "a2"  # the written one is
        assert 'name = "a2"' in path.read_text()


async def test_secret_refusal_names_the_target_file(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROFILE_TOKEN", "s3cret-token-value")
    overlay = '[[plugins]]\nid = "a"\nconfig = { token = "${env:PROFILE_TOKEN}", model = "l" }\n'
    path, dev = setup(tmp_path, overlay=overlay)
    logs = MemoryExporter()
    async with started(path, logs, profile="dev") as (root, loader):
        a = fiber(loader, "a")
        await a.update({**loadfix.configs["a"], "model": "s3cret-token-value"})
        await settle(root)
        # the plugin runs with the refused config; put the overlay value back
        await a.update({**loadfix.configs["a"], "model": "l", "name": "s3cret-token-value"})
        await settle(root)
    refusals = [line for line in logs.text("error") if "interpolated secret" in line]
    assert len(refusals) == 2
    assert f"cannot save {dev}:" in refusals[0]
    assert f"cannot save {path}:" in refusals[1]
