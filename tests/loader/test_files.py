from __future__ import annotations

import dataclasses
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from pycordis.loader import Entry, LoaderConfigError, LoaderOptions, load_tree
from pycordis.loader.files import read_config
from pycordis.supervise import RestartPolicy

FIXTURES = Path(__file__).parent.parent / "fixtures" / "loader"


def walk(entries: list[Entry]) -> list[Entry]:
    out: list[Entry] = []
    for entry in entries:
        out.append(entry)
        out.extend(walk(entry.children))
    return out


def strip_source(entries: list[Entry], root: Path) -> list[Entry]:
    """Entries declared in ``root`` get a placeholder source, others keep theirs."""
    return [
        dataclasses.replace(
            e,
            source=Path("<root>") if e.source == root else e.source,
            children=strip_source(e.children, root),
        )
        for e in entries
    ]


def test_toml_tree() -> None:
    root = (FIXTURES / "basic.toml").resolve()
    extra = (FIXTURES / "sub" / "extra.toml").resolve()
    entries, options = load_tree(FIXTURES / "basic.toml")
    assert options == LoaderOptions(
        readonly=True, watch=True, allow=["my_agent.*", "pycordis.*"], poll_interval=0.5
    )
    llm, search0, search1, channels, include = entries
    assert llm == Entry(
        id="llm",
        explicit_id=True,
        use="my_agent.services:OpenAIClient",
        config={"model": "gpt-x", "api_key": "${env:OPENAI_API_KEY}"},
        source=root,
        location=("plugins", 0),
    )
    assert search0 == Entry(
        id="/my_agent.tools:search#0",
        explicit_id=False,
        use="my_agent.tools:search",
        disabled=True,
        isolate={"cache": None},
        intercept={"logger": {"level": 3}},
        inject=["llm_client"],
        source=root,
        location=("plugins", 1),
    )
    assert search1.id == "/my_agent.tools:search#1"
    assert search1.isolate == {"cache": "shared", "db": "main"}
    assert search1.config is None
    assert channels.id == "/group:channels#0"
    assert channels.group == "channels"
    assert channels.use is None
    assert channels.inject == ["llm"]
    assert [c.id for c in channels.children] == [
        "/group:channels#0/my_agent.channels:slack#0",
        "/group:channels#0/my_agent.channels:slack#1",
        "irc",
    ]
    assert [c.location for c in channels.children] == [
        ("plugins", 3, "plugins", 0),
        ("plugins", 3, "plugins", 1),
        ("plugins", 3, "plugins", 2),
    ]
    assert include.id == "/include:./sub/extra.toml#0"
    assert include.include == "./sub/extra.toml"
    assert include.disabled is True
    one, nested = include.children
    assert one == Entry(
        id="/include:./sub/extra.toml#0/my_agent.extra:one#0",
        explicit_id=False,
        use="my_agent.extra:one",
        config={"key": "${file:secret.txt}"},
        source=extra,
        location=("plugins", 0),
    )
    assert nested.source == extra
    assert nested.location == ("plugins", 1)
    (two,) = nested.children
    assert two.id == "/include:./sub/extra.toml#0/group:nested#0/my_agent.extra:two#0"
    assert two.source == extra
    assert two.location == ("plugins", 1, "plugins", 0)


def test_json_tree_equals_toml_tree() -> None:
    toml_entries, toml_options = load_tree(FIXTURES / "basic.toml")
    json_entries, json_options = load_tree(FIXTURES / "basic.json")
    assert json_options == toml_options
    assert strip_source(json_entries, (FIXTURES / "basic.json").resolve()) == strip_source(
        toml_entries, (FIXTURES / "basic.toml").resolve()
    )


@pytest.mark.parametrize("name", ["basic.toml", "basic.json"])
def test_location_walk_matches_raw_entry(name: str) -> None:
    entries, _ = load_tree(FIXTURES / name)
    for entry in walk(entries):
        if entry.source.suffix == ".json":
            data: Any = json.loads(entry.source.read_text())
        else:
            data = tomllib.loads(entry.source.read_text())
        node = data
        for part in entry.location:
            node = node[part]
        assert node.get("id", entry.id) == entry.id
        assert node.get("use") == entry.use
        assert node.get("group") == entry.group
        assert node.get("include") == entry.include
        assert node.get("config") == entry.config


def test_ids_stable_across_reparse() -> None:
    first, _ = load_tree(FIXTURES / "basic.toml")
    second, _ = load_tree(FIXTURES / "basic.toml")
    assert [e.id for e in walk(first)] == [e.id for e in walk(second)]
    ids = [e.id for e in walk(first)]
    assert len(ids) == len(set(ids))


def test_explicit_id_does_not_shift_generated_ids(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text('[[plugins]]\nuse = "m:a"\n\n[[plugins]]\nuse = "m:a"\n')
    before = [e.id for e in load_tree(cfg)[0]]
    cfg.write_text('[[plugins]]\nid = "x"\nuse = "m:a"\n\n' + cfg.read_text())
    after = [e.id for e in load_tree(cfg)[0]]
    assert before == ["/m:a#0", "/m:a#1"]
    assert after == ["x", "/m:a#0", "/m:a#1"]


def test_explicit_id_group_prefixes_children(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text('[[plugins]]\nid = "g"\ngroup = "x"\nplugins = [{ use = "m:a" }]\n')
    (group,) = load_tree(cfg)[0]
    assert group.explicit_id
    assert group.children[0].id == "g/m:a#0"


def test_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text("")
    assert load_tree(cfg) == ([], LoaderOptions())
    assert LoaderOptions().allow == ["*"]


def test_include_cycle() -> None:
    with pytest.raises(ValueError) as info:
        load_tree(FIXTURES / "cycle_a.toml")
    assert str(info.value) == "include cycle: cycle_a.toml -> cycle_b.toml -> cycle_a.toml"


def test_self_include_cycle(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text('[[plugins]]\ninclude = "a.toml"\n')
    with pytest.raises(ValueError, match=r"^include cycle: a.toml -> a.toml$"):
        load_tree(tmp_path / "a.toml")


def test_same_file_included_twice_is_not_a_cycle(tmp_path: Path) -> None:
    (tmp_path / "b.toml").write_text('[[plugins]]\nuse = "m:b"\n')
    (tmp_path / "a.toml").write_text(
        '[[plugins]]\ninclude = "b.toml"\n\n[[plugins]]\ninclude = "b.toml"\n'
    )
    first, second = load_tree(tmp_path / "a.toml")[0]
    assert first.children[0].id == "/include:b.toml#0/m:b#0"
    assert second.children[0].id == "/include:b.toml#1/m:b#0"


def test_include_relative_to_including_file(tmp_path: Path) -> None:
    (tmp_path / "d1" / "d2").mkdir(parents=True)
    (tmp_path / "root.toml").write_text('[[plugins]]\ninclude = "d1/a.toml"\n')
    (tmp_path / "d1" / "a.toml").write_text('[[plugins]]\ninclude = "d2/b.json"\n')
    (tmp_path / "d1" / "d2" / "b.json").write_text('{"plugins": [{"use": "m:x"}]}')
    (inc,) = load_tree(tmp_path / "root.toml")[0]
    (inner,) = inc.children
    assert inner.source == (tmp_path / "d1" / "a.toml").resolve()
    (leaf,) = inner.children
    assert leaf.source == (tmp_path / "d1" / "d2" / "b.json").resolve()
    assert leaf.location == ("plugins", 0)


def test_missing_include(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text('[[plugins]]\ninclude = "nope.toml"\n')
    with pytest.raises(LoaderConfigError, match=r'include "nope.toml"'):
        load_tree(tmp_path / "a.toml")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[[plugins]]\nuse = "m:a"\nbogus = 1\n', 'unknown key "bogus"'),
        ("[[plugins]]\nconfig = {}\n", 'exactly one of "use", "group" or "include"'),
        ('[[plugins]]\nuse = "m:a"\ngroup = "g"\n', 'exactly one of "use", "group" or "include"'),
        ("[[plugins]]\nuse = 1\n", '"use" must be a non-empty string'),
        ('[[plugins]]\nid = ""\nuse = "m:a"\n', '"id" must be a non-empty string'),
        ('[[plugins]]\nuse = "m:a"\ndisabled = "no"\n', '"disabled" must be a boolean'),
        ('[[plugins]]\nuse = "m:a"\nisolate = [1]\n', '"isolate" must be'),
        ('[[plugins]]\nuse = "m:a"\nisolate = { a = 1 }\n', '"isolate" must be'),
        ('[[plugins]]\nuse = "m:a"\nisolate = "a"\n', '"isolate" must be'),
        ('[[plugins]]\nuse = "m:a"\nisolate = ["a", "a"]\n', "more than once"),
        ('[[plugins]]\nuse = "m:a"\nintercept = [1]\n', '"intercept" must be a table'),
        ('[[plugins]]\nuse = "m:a"\ninject = "x"\n', '"inject" must be a list of strings'),
        ('[[plugins]]\nuse = "m:a"\nplugins = []\n', '"plugins" is only allowed on group'),
        ('[[plugins]]\ngroup = "g"\nconfig = {}\n', '"config" is only allowed on "use"'),
        ('[[plugins]]\ngroup = "g"\nplugins = 1\n', '"plugins" must be an array'),
        ("plugins = [1]\n", "must be a table"),
        ("plugins = 1\n", '"plugins" must be an array'),
        ("other = 1\n", 'unknown top-level key "other"'),
        ('"$schema" = 1\n', '"$schema" must be a string'),
        ('[[plugins]]\nuse = "m:a"\nrestart = 1\n', '"restart" must be a table'),
        (
            '[[plugins]]\nuse = "m:a"\nrestart = { mode = "always" }\n',
            '"restart.mode" must be "never" or "on-failure"',
        ),
        ('[[plugins]]\nuse = "m:a"\nrestart = { bogus = 1 }\n', 'unknown key "restart.bogus"'),
        (
            '[[plugins]]\nuse = "m:a"\nrestart = { max_restarts = -1 }\n',
            '"restart.max_restarts" must be a non-negative integer',
        ),
        (
            '[[plugins]]\nuse = "m:a"\nrestart = { backoff = { initial = -1 } }\n',
            '"restart.backoff.initial" must be a non-negative number',
        ),
        (
            '[[plugins]]\nuse = "m:a"\nload_timeout = 0\n',
            '"load_timeout" must be a positive number',
        ),
        ('[[plugins]]\nuse = "m:a"\nload_timeout = true\n', '"load_timeout" must be a positive'),
        (
            '[[plugins]]\nuse = "m:a"\ndispose_timeout = "10"\n',
            '"dispose_timeout" must be a positive number',
        ),
        ("[loader]\nbogus = 1\n", 'unknown [loader] key "bogus"'),
        ("loader = 1\n", '"loader" must be a table'),
        ("[loader]\nreadonly = 1\n", '"readonly" must be a boolean'),
        ('[loader]\nallow = "x"\n', '"allow" must be a list of strings'),
        ("[loader]\npoll_interval = 0\n", '"poll_interval" must be a positive number'),
        ("[loader]\npoll_interval = true\n", '"poll_interval" must be a positive number'),
        (
            '[[plugins]]\nid = "x"\nuse = "m:a"\n\n[[plugins]]\nid = "x"\nuse = "m:b"\n',
            'duplicate entry id "x"',
        ),
    ],
)
def test_invalid_entries(tmp_path: Path, text: str, message: str) -> None:
    cfg = tmp_path / "bad.toml"
    cfg.write_text(text)
    with pytest.raises(LoaderConfigError) as info:
        load_tree(cfg)
    assert message in str(info.value)
    assert "bad.toml" in str(info.value)


def test_schema_key_is_accepted_and_ignored(tmp_path: Path) -> None:
    root = tmp_path / "app.json"
    root.write_text(
        json.dumps(
            {
                "$schema": "./pycordis.schema.json",
                "plugins": [{"id": "a", "use": "m:a"}, {"id": "p", "include": "part.toml"}],
            }
        )
    )
    (tmp_path / "part.toml").write_text(
        '"$schema" = "./pycordis.schema.json"\n[[plugins]]\nid = "b"\nuse = "m:b"\n'
    )
    entries, options = load_tree(root)
    assert [entry.id for entry in walk(entries)] == ["a", "p", "b"]
    assert options == LoaderOptions()


def test_supervision_keys(tmp_path: Path) -> None:
    cfg = tmp_path / "a.toml"
    cfg.write_text(
        '[[plugins]]\nuse = "m:a"\nrestart = { mode = "on-failure", max_restarts = 3 }\n'
        "load_timeout = 5\ndispose_timeout = 2.5\n\n"
        '[[plugins]]\nuse = "m:b"\n'
    )
    (supervised, plain), _ = load_tree(cfg)
    assert supervised.restart == {"mode": "on-failure", "max_restarts": 3}
    assert supervised.policy == RestartPolicy(mode="on-failure", max_restarts=3)
    assert (supervised.load_timeout, supervised.dispose_timeout) == (5.0, 2.5)
    assert isinstance(supervised.load_timeout, float)
    assert (plain.restart, plain.policy, plain.load_timeout, plain.dispose_timeout) == (
        None,
        None,
        None,
        None,
    )


def test_bad_restart_names_entry_location(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.toml"
    cfg.write_text('[[plugins]]\nuse = "m:a"\nrestart = { window = 0 }\n')
    with pytest.raises(
        LoaderConfigError,
        match=r'"restart\.window" must be a positive number in entry plugins\[0\]',
    ):
        load_tree(cfg)


def test_unknown_key_names_location(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.toml"
    cfg.write_text('[[plugins]]\ngroup = "g"\nplugins = [{ use = "m:a", nope = 1 }]\n')
    with pytest.raises(ValueError, match=r'unknown key "nope" in entry plugins\[0\]\.plugins\[0\]'):
        load_tree(cfg)


def test_loader_table_only_in_root(tmp_path: Path) -> None:
    (tmp_path / "b.toml").write_text("[loader]\nwatch = true\n")
    (tmp_path / "a.toml").write_text('[[plugins]]\ninclude = "b.toml"\n')
    with pytest.raises(ValueError, match=r"b\.toml.*\[loader\] is only allowed in the root"):
        load_tree(tmp_path / "a.toml")


def test_parse_error_names_file(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text("[[plugins]\n")
    with pytest.raises(ValueError, match=r"a\.toml"):
        load_tree(tmp_path / "a.toml")
    (tmp_path / "a.json").write_text("{")
    with pytest.raises(ValueError, match=r"a\.json"):
        load_tree(tmp_path / "a.json")
    (tmp_path / "a.json").write_text("[]")
    with pytest.raises(ValueError, match=r"a\.json.*must be a table"):
        load_tree(tmp_path / "a.json")


def test_unsupported_suffix(tmp_path: Path) -> None:
    (tmp_path / "a.ini").write_text("")
    with pytest.raises(ValueError, match=r'unsupported config file type ".ini"'):
        load_tree(tmp_path / "a.ini")


def test_yaml_requires_pyyaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.yaml").write_text("plugins: []\n")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(RuntimeError) as info:
        load_tree(tmp_path / "a.yaml")
    assert str(info.value) == "YAML config requires PyYAML to be installed"


def test_yaml_tree(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    (tmp_path / "a.yml").write_text("plugins:\n  - use: m:a\n    config: {x: 1}\n")
    (entry,) = load_tree(tmp_path / "a.yml")[0]
    assert entry.config == {"x": 1}
    assert read_config(tmp_path / "a.yml") == {"plugins": [{"use": "m:a", "config": {"x": 1}}]}


def test_yaml_empty_document(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    (tmp_path / "a.yaml").write_text("")
    assert load_tree(tmp_path / "a.yaml") == ([], LoaderOptions())


def test_yaml_syntax_error_names_file(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    (tmp_path / "a.yaml").write_text("plugins: [\n")
    with pytest.raises(ValueError, match=r"a\.yaml"):
        load_tree(tmp_path / "a.yaml")


def test_unreadable_files_raise_loader_config_error(tmp_path: Path) -> None:
    with pytest.raises(LoaderConfigError, match=r"missing\.toml: cannot read file"):
        load_tree(tmp_path / "missing.toml")
    (tmp_path / "bin.toml").write_bytes(b"\xff\xfe = 1\n")
    with pytest.raises(LoaderConfigError, match=r"bin\.toml: file is not valid UTF-8"):
        load_tree(tmp_path / "bin.toml")
    (tmp_path / "a.toml").write_text('[[plugins]]\ninclude = "bin.toml"\n')
    with pytest.raises(LoaderConfigError, match=r"bin\.toml"):
        load_tree(tmp_path / "a.toml")


def test_hardlinked_include_cycle(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text('[[plugins]]\ninclude = "b.toml"\n')
    os.link(tmp_path / "a.toml", tmp_path / "b.toml")
    with pytest.raises(LoaderConfigError) as info:
        load_tree(tmp_path / "a.toml")
    assert str(info.value) == "include cycle: a.toml -> a.toml"


def test_symlinked_include(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "b.toml").write_text('[[plugins]]\nuse = "m:b"\n')
    (tmp_path / "link.toml").symlink_to(tmp_path / "real" / "b.toml")
    (tmp_path / "a.toml").write_text('[[plugins]]\ninclude = "link.toml"\n')
    (inc,) = load_tree(tmp_path / "a.toml")[0]
    (leaf,) = inc.children
    assert leaf.use == "m:b"
    # write-back goes to the real file the entry is declared in
    assert leaf.source == (tmp_path / "real" / "b.toml").resolve()
