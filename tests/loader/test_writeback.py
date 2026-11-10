from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pycordis.context import Context
from pycordis.errors import ValidationError
from pycordis.fiber import Fiber
from pycordis.loader import LoaderConfigError
from pycordis.loader.persist import ConfigFile, write_atomic
from pycordis.loader.service import Loader, LoaderConfig
from pycordis.testing import MemoryExporter, mock_context, settle

COMMENTED = """\
# root config
[loader]
allow = ["loadfix"]  # only fixtures

[[plugins]]
id = "a"  # the main one
use = "loadfix:echo"
config = { name = "a", model = "x" }  # inline config

[[plugins]]
id = "b"
use = "loadfix:echo"

  [plugins.config]
  name = "b"
  model = 'x'  # literal

[[plugins]]
id = "grp"
group = "g"
plugins = [
  { id = "c", use = "loadfix:echo", config = { name = "c" } },  # child
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


def fiber(loader: Loader, id: str) -> Fiber:
    loaded = loader.get(id)
    assert loaded is not None and loaded.fiber is not None, id
    return loaded.fiber


def errors(logs: MemoryExporter) -> list[str]:
    return logs.text("error")


# -- internal/commit write-back -------------------------------------------------


async def test_commit_rewrites_only_the_value(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    path.write_bytes(COMMENTED.encode())
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        await fiber(loader, "a").update({"name": "a", "model": "y"})
        await fiber(loader, "b").update({"name": "b", "model": "y"})
        await fiber(loader, "c").update({"name": "c", "n": 1})
        await settle(root)
        expected = (
            COMMENTED.replace('model = "x" }', 'model = "y" }')
            .replace("model = 'x'", "model = 'y'")
            .replace('config = { name = "c" }', 'config = { name = "c", n = 1 }')
        )
        assert path.read_bytes() == expected.encode()
        # the in-memory entry follows, so a reload sees no change (no restart)
        entry = loader.get("a")
        assert entry is not None and entry.entry.config == {"name": "a", "model": "y"}
        runs = dict(loadfix.runs)
        await loader.reload()
        await settle(root)
        assert dict(loadfix.runs) == runs
    assert errors(logs) == []


async def test_commit_keeps_crlf_and_mode(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    text = '[[plugins]]\r\nid = "a"\r\nuse = "loadfix:echo"\r\n'
    path.write_bytes(text.encode())
    path.chmod(0o640)
    async with started(path) as (_root, loader):
        await fiber(loader, "a").update({"name": "a"})
        assert path.read_bytes() == (text + '\r\n[plugins.config]\r\nname = "a"\r\n').encode()
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
        # setting the config to None removes the key
        await fiber(loader, "a").update(None)
        assert path.read_bytes() == (text + "\r\n").encode()
    assert [p.name for p in tmp_path.iterdir()] == ["pycordis.toml"]  # no temp files left


async def test_commit_writes_the_included_file(tmp_path: Path, loadfix: ModuleType) -> None:
    root_text = '[[plugins]]\nid = "inc"\ninclude = "sub.toml"\n'
    sub_text = '# sub\n[[plugins]]\nid = "s"\nuse = "loadfix:echo"\nconfig = { name = "s" }\n'
    write(tmp_path / "pycordis.toml", root_text)
    write(tmp_path / "sub.toml", sub_text)
    async with started(tmp_path / "pycordis.toml") as (_root, loader):
        await fiber(loader, "s").update({"name": "t"})
    assert (tmp_path / "pycordis.toml").read_text() == root_text
    assert (tmp_path / "sub.toml").read_text() == sub_text.replace('"s" }', '"t" }')


async def test_readonly_skips_every_write(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    path.write_bytes(COMMENTED.encode())
    async with started(path, readonly=True) as (_root, loader):
        await fiber(loader, "a").update({"name": "a", "model": "y"})
        await loader.create({"id": "n", "use": "loadfix:echo"})
        await loader.update("b", {"disabled": True})
        await loader.remove("c")
        await loader.save()
        entry = loader.get("a")
        assert entry is not None and entry.entry.config == {"name": "a", "model": "y"}
    assert path.read_bytes() == COMMENTED.encode()


async def test_no_save_and_foreign_fibers_do_not_write(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    path.write_bytes(COMMENTED.encode())
    async with started(path) as (root, loader):
        await fiber(loader, "a").update({"name": "a", "model": "z"}, no_save=True)
        other = root.plugin(loadfix.echo, {"name": "o"})
        await other.update({"name": "p"})
        # a group fiber has no config to save
        await fiber(loader, "grp").update({"x": 1})
    assert path.read_bytes() == COMMENTED.encode()


async def test_json_indent_and_trailing_newline(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.json"
    data: dict[str, list[dict[str, object]]] = {
        "plugins": [{"id": "a", "use": "loadfix:echo", "config": {"name": "é"}}]
    }
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    async with started(path) as (_root, loader):
        await fiber(loader, "a").update({"name": "ü", "n": 2})
        data["plugins"][0]["config"] = {"name": "ü", "n": 2}
        assert path.read_text(encoding="utf-8") == json.dumps(data, indent=4, ensure_ascii=False)
        path.write_text(json.dumps(data, indent="\t") + "\n")
        await loader.reload()
        await fiber(loader, "a").update({"name": "v"})
        data["plugins"][0]["config"] = {"name": "v"}
        assert path.read_text() == json.dumps(data, indent="\t") + "\n"


def test_json_compact_and_crlf(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text('{"plugins": []}')
    file = ConfigFile.open(path)
    assert file.append(("plugins",), {"use": "x:y"}) == 0
    assert file.write()
    assert path.read_text() == '{\n  "plugins": [\n    {\n      "use": "x:y"\n    }\n  ]\n}'
    path.write_bytes(b'{\r\n  "plugins": [],\r\n  "x": 1\r\n}\r\n')
    file = ConfigFile.open(path)
    file.delete(("x",))
    file.delete(("missing", 0))
    assert not ConfigFile.open(path).write()  # unchanged: nothing written
    assert file.write()
    assert path.read_bytes() == b'{\r\n  "plugins": []\r\n}\r\n'


def _fake_yaml() -> ModuleType:
    module = ModuleType("yaml")

    class YAMLError(Exception):
        pass

    def safe_load(text: str) -> Any:
        return json.loads(text) if text.strip() else None

    def safe_dump(data: Any, **options: Any) -> str:
        assert options == {"sort_keys": False, "allow_unicode": True}
        return json.dumps(data, indent=1) + "\n"

    module.__dict__.update(YAMLError=YAMLError, safe_load=safe_load, safe_dump=safe_dump)
    return module


async def test_yaml_warns_once(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "yaml", _fake_yaml())
    path = tmp_path / "pycordis.yaml"
    path.write_text(json.dumps({"plugins": [{"id": "a", "use": "loadfix:echo"}]}))
    logs = MemoryExporter()
    async with started(path, logs) as (_root, loader):
        await fiber(loader, "a").update({"name": "a"})
        await fiber(loader, "a").update({"name": "b"})
    expected = {"plugins": [{"id": "a", "use": "loadfix:echo", "config": {"name": "b"}}]}
    assert path.read_text() == json.dumps(expected, indent=1) + "\n"
    assert logs.text("warn") == ["saving YAML config drops comments"]


async def test_write_failure_is_logged(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, '[[plugins]]\nid = "a"\nuse = "loadfix:echo"\n')
    logs = MemoryExporter()
    async with started(path, logs) as (_root, loader):
        # not representable in TOML: logged, the fiber still updates
        await fiber(loader, "a").update({"name": "a", "x": object()})
        assert len(errors(logs)) == 1
        assert "cannot save" in errors(logs)[0]
    assert path.read_text() == '[[plugins]]\nid = "a"\nuse = "loadfix:echo"\n'


async def test_externally_moved_entry_is_not_overwritten(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    path = tmp_path / "pycordis.toml"
    write(path, '[[plugins]]\nid = "a"\nuse = "loadfix:echo"\n')
    logs = MemoryExporter()
    async with started(path, logs) as (_root, loader):
        changed = '[[plugins]]\nid = "z"\nuse = "loadfix:echo"\n'
        write(path, changed)
        await fiber(loader, "a").update({"name": "a"})
        assert path.read_text() == changed
        assert len(errors(logs)) == 1
        assert 'entry "a" is no longer at plugins[0]' in errors(logs)[0]
        with pytest.raises(ExceptionGroup) as raised:
            await loader.save()
        assert raised.group_contains(LoaderConfigError, match="no longer at")


async def test_save_attempts_every_file_and_raises_all_failures(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    path = tmp_path / "pycordis.toml"
    write(
        path,
        '[[plugins]]\nid = "a"\nuse = "loadfix:echo"\n'
        '[[plugins]]\nid = "ix"\ninclude = "x.toml"\n'
        '[[plugins]]\nid = "iy"\ninclude = "y.toml"\n',
    )
    write(tmp_path / "x.toml", '[[plugins]]\nid = "b"\nuse = "loadfix:echo"\n')
    y_entry = '[[plugins]]\nid = "c"\nuse = "loadfix:echo"\nconfig = { n = 1 }\n'
    write(tmp_path / "y.toml", y_entry)
    async with started(path) as (_root, loader):
        # the first two files changed underneath the loader; the last one only drifted
        write(path, path.read_text().replace('id = "a"', 'id = "z"'))
        write(tmp_path / "x.toml", '[[plugins]]\nid = "q"\nuse = "loadfix:echo"\n')
        write(tmp_path / "y.toml", y_entry.replace("n = 1", "n = 2"))
        with pytest.raises(ExceptionGroup) as raised:
            await loader.save()
        failures = raised.value.exceptions
        assert [type(error) for error in failures] == [LoaderConfigError, LoaderConfigError]
        assert 'entry "a"' in str(failures[0]) and 'entry "b"' in str(failures[1])
        assert "config = { n = 1 }" in (tmp_path / "y.toml").read_text()


# -- loader API persistence -----------------------------------------------------


async def test_api_create_update_remove_persist(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    path.write_bytes(COMMENTED.encode())
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        await loader.create({"id": "n", "use": "loadfix:echo", "config": {"name": "n"}})
        await loader.create({"id": "d", "use": "loadfix:echo"}, "grp")
        await loader.update("b", {"disabled": True, "inject": ["llm"]})
        await loader.update("b", {"inject": None})
        await loader.remove("a")
        # locations of the following entries moved up
        await fiber(loader, "n").update({"name": "m"})
        await fiber(loader, "c").update({"name": "c2"})
        await settle(root)
        text = path.read_text()
        assert text.startswith('# root config\n[loader]\nallow = ["loadfix"]  # only fixtures\n')
        assert "the main one" not in text
        assert "model = 'x'  # literal" in text
        assert "disabled = true" in text
        assert "inject" not in text
        runs = dict(loadfix.runs)
        await loader.reload()
        await settle(root)
        assert dict(loadfix.runs) == runs  # the file matches the running tree
        assert [e.entry.id for e in loader.entries()] == ["b", "grp", "c", "d", "n"]
        n = loader.get("n")
        assert n is not None and n.entry.config == {"name": "m"}
        c = loader.get("c")
        assert c is not None and c.entry.config == {"name": "c2"}
    assert errors(logs) == []


async def test_api_create_in_include_and_new_file_array(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    write(tmp_path / "pycordis.toml", '[[plugins]]\nid = "inc"\ninclude = "sub.toml"\n')
    write(tmp_path / "sub.toml", "# empty include\n")
    async with started(tmp_path / "pycordis.toml") as (_root, loader):
        await loader.create({"id": "s", "use": "loadfix:echo"}, "inc")
        await loader.create({"id": "e", "group": "e"})
        await loader.create({"id": "e1", "use": "loadfix:echo"}, "e")
    assert (
        tmp_path / "sub.toml"
    ).read_text() == '# empty include\nplugins = [{ id = "s", use = "loadfix:echo" }]\n'
    assert "e1" in (tmp_path / "pycordis.toml").read_text()


async def test_save_flushes_the_tree(tmp_path: Path, loadfix: ModuleType) -> None:
    path = tmp_path / "pycordis.toml"
    path.write_bytes(COMMENTED.encode())
    async with started(path, readonly=True) as (_root, loader):
        await loader.update("a", {"config": {"name": "a", "model": "s"}, "isolate": ["x"]})
        await loader.update("b", {"disabled": True, "intercept": {"llm": {"v": 1}}})
        await loader.update("c", {"isolate": {"llm": "shared"}})
    # readonly: nothing written; a writable loader on the same file then flushes on save()
    assert path.read_bytes() == COMMENTED.encode()
    async with started(path) as (_root, loader):
        await loader.save()
        assert path.read_bytes() == COMMENTED.encode()  # nothing to change
        entry = loader.get("a")
        assert entry is not None
        entry.entry.config = {"name": "a", "model": "s"}
        entry.entry.isolate = {"x": None}
        entry.entry.inject = ["llm"]
        b = loader.get("b")
        assert b is not None
        b.entry.disabled = True
        b.entry.intercept = {"llm": {"v": 1}}
        b.entry.config = None
        c = loader.get("c")
        assert c is not None
        c.entry.isolate = {"llm": "shared"}
        c.entry.restart = {"mode": "on-failure"}
        c.entry.dispose_timeout = 5.0
        await loader.save()
        text = path.read_text()
        assert 'config = { name = "a", model = "s" }  # inline config' in text
        assert 'isolate = ["x"]' in text
        assert 'inject = ["llm"]' in text
        assert "disabled = true" in text
        assert "literal" not in text
        assert 'isolate = { llm = "shared" }' in text
        assert 'restart = { mode = "on-failure" }' in text
        assert "dispose_timeout = 5.0" in text
        await loader.save()
        assert path.read_text() == text


# -- secrets: interpolated values are never persisted ----------------------------

SECRET = "sk-SECRET-123"


def secret_file(tmp_path: Path, config: str) -> Path:
    path = tmp_path / "pycordis.toml"
    write(path, f'[[plugins]]\nid = "a"\nuse = "loadfix:echo"\nconfig = {config}\n')
    return path


async def test_commit_keeps_placeholder_for_expanded_value(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYC_SECRET", SECRET)
    path = secret_file(tmp_path, '{ token = "${env:PYC_SECRET}", count = 0 }')
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        a = fiber(loader, "a")
        assert a.effective_config["token"] == SECRET
        await a.update({**a.effective_config, "count": 1})
        await settle(root)
        text = path.read_text()
        assert SECRET not in text
        assert 'config = { token = "${env:PYC_SECRET}", count = 1 }' in text
        entry = loader.get("a")
        assert entry is not None
        assert entry.entry.config == {"token": "${env:PYC_SECRET}", "count": 1}
        await loader.save()  # the in-memory entry cannot leak either
        assert SECRET not in path.read_text()
    assert errors(logs) == []


async def test_commit_keeps_nested_and_mixed_placeholders(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYC_SECRET", SECRET)
    (tmp_path / "key.txt").write_text("file-secret\n")
    config = (
        '{ auth = { header = "Bearer ${env:PYC_SECRET}", keys = ["${file:key.txt}", "plain"] }, '
        "n = 0 }"
    )
    path = secret_file(tmp_path, config)
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        a = fiber(loader, "a")
        effective = a.effective_config
        assert effective["auth"] == {"header": f"Bearer {SECRET}", "keys": ["file-secret", "plain"]}
        await a.update({**effective, "n": 1})
        await settle(root)
        text = path.read_text()
        assert SECRET not in text and "file-secret" not in text
        assert text.endswith(config.replace("n = 0", "n = 1") + "\n")
    assert errors(logs) == []


async def test_commit_refuses_to_persist_a_secret(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYC_SECRET", SECRET)
    config = '{ header = "Bearer ${env:PYC_SECRET}", home = "${env:PYC_UNSET:-/srv}" }'
    path = secret_file(tmp_path, config)
    before = path.read_text()
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        a = fiber(loader, "a")
        # the secret moved to another key, and the placeholder string was changed
        await a.update({"header": f"Token {SECRET}", "home": "/srv"})
        await settle(root)
        assert path.read_text() == before
        assert errors(logs) == [
            f'cannot save {path}: would persist an interpolated secret (entry "a")'
        ]
        entry = loader.get("a")
        assert entry is not None
        assert entry.entry.config == {
            "header": "Bearer ${env:PYC_SECRET}",
            "home": "${env:PYC_UNSET:-/srv}",
        }
        await loader.save()
        assert path.read_text() == before
        # a default value is not a secret: writing it as a literal is allowed
        await a.update({"header": f"Bearer {SECRET}", "home": "/srv", "x": "/srv"})
        await settle(root)
        assert path.read_text().endswith(
            'config = { header = "Bearer ${env:PYC_SECRET}", home = "${env:PYC_UNSET:-/srv}", '
            'x = "/srv" }\n'
        )
    assert len(errors(logs)) == 1


async def test_commit_refusal_also_applies_when_readonly(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYC_SECRET", SECRET)
    path = secret_file(tmp_path, '{ token = "${env:PYC_SECRET}" }')
    async with started(path, readonly=True) as (_root, loader):
        await fiber(loader, "a").update({"token": "x", "copy": SECRET})
        entry = loader.get("a")
        assert entry is not None and entry.entry.config == {"token": "${env:PYC_SECRET}"}


async def test_commit_copies_the_config(tmp_path: Path, loadfix: ModuleType) -> None:
    path = secret_file(tmp_path, '{ name = "a" }')
    async with started(path) as (_root, loader):
        config: dict[str, Any] = {"name": "a", "nested": {"k": [1]}}
        await fiber(loader, "a").update(config)
        config["name"] = "changed"
        config["nested"]["k"].append(2)
        entry = loader.get("a")
        assert entry is not None
        assert entry.entry.config == {"name": "a", "nested": {"k": [1]}}


async def test_short_env_value_is_retemplated_but_never_refused(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYC_W", "4")
    path = secret_file(tmp_path, '{ host = "v${env:PYC_W}-host" }')
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        a = fiber(loader, "a")
        assert a.effective_config == {"host": "v4-host"}
        await a.update({"host": "v4-host", "other": "v4-host", "n": 4})
        await settle(root)
        assert path.read_text().endswith(
            'config = { host = "v${env:PYC_W}-host", other = "v4-host", n = 4 }\n'
        )
    assert errors(logs) == []


async def test_eight_char_env_secret_is_refused_as_substring(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYC_LONG", "12345678")
    path = secret_file(tmp_path, '{ token = "${env:PYC_LONG}" }')
    before = path.read_text()
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        await fiber(loader, "a").update({"token": "x", "copy": "id-12345678-z"})
        await settle(root)
        assert path.read_text() == before
    assert len(errors(logs)) == 1


async def test_rotated_file_secret_keeps_placeholder_and_is_refused(
    tmp_path: Path, loadfix: ModuleType
) -> None:
    key = tmp_path / "key.txt"
    key.write_text("old-file\n")
    path = secret_file(tmp_path, '{ k = "${file:key.txt}", n = 0 }')
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        a = fiber(loader, "a")
        effective = a.effective_config
        assert effective["k"] == "old-file"
        key.write_text("new-file\n")  # rotated after load
        await a.update({**effective, "n": 1})
        await settle(root)
        text = path.read_text()
        assert 'config = { k = "${file:key.txt}", n = 1 }' in text
        assert a.effective_config["k"] == "new-file"
        await a.update({"k": "${file:key.txt}", "n": 2, "copy": "old-file"})
        await settle(root)
        assert path.read_text() == text
        assert "old-file" not in text
    assert len(errors(logs)) == 1


async def test_secret_remembered_after_placeholder_is_overwritten(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYC_SECRET", SECRET)
    path = secret_file(tmp_path, '{ token = "${env:PYC_SECRET}" }')
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        a = fiber(loader, "a")
        await a.update({"token": "lit"})
        await settle(root)
        text = path.read_text()
        assert 'config = { token = "lit" }' in text
        await a.update({"token": "lit", "x": SECRET})
        await settle(root)
        assert path.read_text() == text
        await loader.save()
        assert SECRET not in path.read_text()
    assert len(errors(logs)) == 1


async def test_non_string_leaf_equal_to_a_secret_is_refused(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYC_PIN", "98765432")
    monkeypatch.setenv("PYC_SHORT", "1234")
    path = secret_file(tmp_path, '{ pin = "${env:PYC_PIN}", short = "${env:PYC_SHORT}" }')
    before = path.read_text()
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        a = fiber(loader, "a")
        await a.update({"pin": 98765432, "short": "${env:PYC_SHORT}"})
        await settle(root)
        assert path.read_text() == before
        assert len(errors(logs)) == 1
        # a short value is not a secret: an int equal to it may be written
        await a.update({"pin": "${env:PYC_PIN}", "short": 1234})
        await settle(root)
        assert "short = 1234" in path.read_text()
    assert len(errors(logs)) == 1


async def test_expanded_config_is_not_interpolated_again(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYC_OTHER", "zzz")
    (tmp_path / "key.txt").write_text("abc${env:PYC_OTHER}xyz\n")
    path = secret_file(tmp_path, '{ k = "${file:key.txt}", n = 0 }')
    logs = MemoryExporter()
    async with started(path, logs) as (root, loader):
        a = fiber(loader, "a")
        effective = a.effective_config
        assert effective["k"] == "abc${env:PYC_OTHER}xyz"
        await a.update({**effective, "n": 1})
        await settle(root)
        assert a.effective_config == {"k": "abc${env:PYC_OTHER}xyz", "n": 1}
        assert a.raw_config == {"k": "${file:key.txt}", "n": 1}
        assert 'config = { k = "${file:key.txt}", n = 1 }' in path.read_text()
        # an unsaved update gets its raw config re-templated synchronously too
        pending = a.update({**a.effective_config, "n": 2}, no_save=True)
        assert a.raw_config == {"k": "${file:key.txt}", "n": 2}
        await pending
        await settle(root)
        assert a.effective_config == {"k": "abc${env:PYC_OTHER}xyz", "n": 2}
    assert errors(logs) == []


async def test_api_create_and_update_refuse_to_persist_a_secret(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYC_SECRET", SECRET)
    path = secret_file(tmp_path, '{ token = "${env:PYC_SECRET}" }')
    before = path.read_text()
    logs = MemoryExporter()
    message = f'cannot save {path}: would persist an interpolated secret (entry "a")'
    async with started(path, logs) as (_root, loader):
        a = fiber(loader, "a")
        with pytest.raises(ValueError, match="would persist an interpolated secret"):
            await loader.update("a", {"config": {"token": "x", "copy": SECRET}})
        assert path.read_text() == before
        assert a.raw_config == {"token": "${env:PYC_SECRET}"}
        entry = loader.get("a")
        assert entry is not None and entry.entry.config == {"token": "${env:PYC_SECRET}"}
        # an expanded value passed back gets its placeholder, like a commit
        await loader.update("a", {"config": {"token": SECRET, "n": 1}})
        assert 'config = { token = "${env:PYC_SECRET}", n = 1 }' in path.read_text()
        assert SECRET not in path.read_text()
        # create: a literal copy of the entry's own interpolated secret is refused
        with pytest.raises(ValueError, match="would persist an interpolated secret"):
            await loader.create(
                {
                    "id": "b",
                    "use": "loadfix:echo",
                    "config": {"k": "${env:PYC_SECRET}", "c": SECRET},
                }
            )
        assert loader.get("b") is None
        assert SECRET not in path.read_text()
        await loader.create(
            {"id": "b", "use": "loadfix:echo", "config": {"k": "${env:PYC_SECRET}"}}
        )
        assert fiber(loader, "b").effective_config == {"k": SECRET}
    assert errors(logs) == [message, message.replace('"a"', '"b"')]


async def test_create_disabled_entry_with_unresolvable_interpolations(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYC_UNSET", raising=False)
    path = secret_file(tmp_path, '{ name = "a" }')
    logs = MemoryExporter()
    config = {"k": "${env:PYC_UNSET}", "f": "${file:missing.txt}"}
    async with started(path, logs) as (_root, loader):
        created = await loader.create(
            {"id": "d", "use": "loadfix:echo", "disabled": True, "config": config}
        )
        assert created is None
        entry = loader.get("d")
        assert entry is not None and entry.fiber is None
        assert entry.entry.config == config
        assert '"${env:PYC_UNSET}"' in path.read_text()
    assert errors(logs) == []


async def test_failed_candidate_leaves_raw_config_untouched(
    tmp_path: Path, loadfix: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYC_SECRET", SECRET)
    monkeypatch.delenv("PYC_UNSET", raising=False)
    path = secret_file(tmp_path, '{ token = "${env:PYC_SECRET}" }')
    async with started(path) as (_root, loader):
        a = fiber(loader, "a")
        retained = {"token": SECRET}
        a.raw_config = retained  # e.g. left by a volatile-only update
        with pytest.raises(ValidationError):
            a.update({"token": SECRET, "bad": "${env:PYC_UNSET}"})
        assert a.raw_config is retained
        assert a.resolve_config({"token": SECRET}) == {"token": SECRET}
        assert a.raw_config is retained  # a candidate never touches the raw config
        await a.restart()  # resolving the raw config itself restores its placeholders
        assert a.raw_config == {"token": "${env:PYC_SECRET}"}
        assert a.effective_config == {"token": SECRET}


# -- atomic write ----------------------------------------------------------------


def test_write_atomic_cleans_up_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "a.toml"
    path.write_text("x = 1\n")

    def fail(_src: object, _dst: object) -> None:
        raise OSError("boom")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="boom"):
        write_atomic(path, "x = 2\n")
    assert path.read_text() == "x = 1\n"
    assert [p.name for p in tmp_path.iterdir()] == ["a.toml"]


def test_write_atomic_new_file(tmp_path: Path) -> None:
    path = tmp_path / "new.toml"
    write_atomic(path, "x = 1\n")
    assert path.read_text() == "x = 1\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_config_file_errors(tmp_path: Path) -> None:
    path = tmp_path / "a.toml"
    path.write_text("x = [1]\n")
    file = ConfigFile.open(path)
    assert file.get(("x", 0)) == 1
    assert file.get(("y",), None) is None
    assert file.get(("x", 5), None) is None
    with pytest.raises(KeyError):
        file.get(("y",))
    file.delete(("y",))  # absent: no-op
    assert not file.write()
    (tmp_path / "b.toml").write_text("x = \n")
    with pytest.raises(LoaderConfigError):
        ConfigFile.open(tmp_path / "b.toml")


def test_config_file_open_errors(tmp_path: Path) -> None:
    with pytest.raises(LoaderConfigError, match="unsupported config file type"):
        ConfigFile.open(tmp_path / "a.ini")
    with pytest.raises(LoaderConfigError, match="cannot read file"):
        ConfigFile.open(tmp_path / "missing.toml")
    (tmp_path / "bad.json").write_bytes(b"\xff")
    with pytest.raises(LoaderConfigError, match="not valid UTF-8"):
        ConfigFile.open(tmp_path / "bad.json")


def test_json_nested_edits(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text('{"plugins": [{"use": "x:y", "config": {"a": 1, "b": 2}}]}\n')
    file = ConfigFile.open(path)
    file.delete(("plugins", 0, "config", "a"))
    file.set(("plugins", 0, "config", "c"), 3)
    assert file.write()
    assert json.loads(path.read_text()) == {"plugins": [{"use": "x:y", "config": {"b": 2, "c": 3}}]}


def test_write_atomic_fsync_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "a.toml"
    path.write_text("x = 1\n")

    def fail(_fd: int) -> None:
        raise OSError("disk")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="disk"):
        write_atomic(path, "x = 2\n")
    assert path.read_text() == "x = 1\n"
    assert [p.name for p in tmp_path.iterdir()] == ["a.toml"]


def test_set_through_a_mismatched_path_is_a_type_error(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text('{"plugins": [{"use": "x:y"}], "x": 1}')
    file = ConfigFile.open(path)
    with pytest.raises(TypeError, match="cannot set 'name' in list"):
        file.set(("plugins", "name"), 1)
    with pytest.raises(TypeError, match="cannot set 'y' in int"):
        file.set(("x", "y"), 1)
    file.delete(("x", "y"))  # absent: a no-op, nothing is addressed
    assert file.get(()) == {"plugins": [{"use": "x:y"}], "x": 1}
