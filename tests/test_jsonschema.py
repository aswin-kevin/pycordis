"""Tests for the JSON Schema export (spec §3.9)."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, make_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from pycordis import cli, jsonschema
from pycordis.errors import ValidationError
from pycordis.fiber import FiberState
from pycordis.loader import Entry, load_tree, read_config
from pycordis.loader.service import Loader, LoaderConfig
from pycordis.schema import validate_config
from pycordis.testing import mock_context
from pycordis.volatile import Volatile

FIXTURES = Path(__file__).parent / "fixtures"
LOADER_FIXTURES = FIXTURES / "loader"


# -- a tiny validator for the JSON Schema subset the exporter emits ------------------


def _is_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return (
            isinstance(value, int | float) and not isinstance(value, bool) and value == int(value)
        )
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    return isinstance(value, dict)  # "object"


def valid(schema: Mapping[str, Any], value: Any, root: Mapping[str, Any] | None = None) -> bool:
    """Whether ``value`` satisfies ``schema`` (``root`` resolves ``$ref``)."""
    root = schema if root is None else root

    def sub(s: Mapping[str, Any], v: Any) -> bool:
        return valid(s, v, root)

    if "$ref" in schema:
        target: Any = root
        for part in [p for p in schema["$ref"].removeprefix("#").split("/") if p]:
            target = target[part]
        return sub(target, value)
    types = schema.get("type")
    if types is not None and not any(
        _is_type(value, t) for t in ([types] if isinstance(types, str) else types)
    ):
        return False
    if "enum" in schema and not any(type(a) is type(value) and a == value for a in schema["enum"]):
        return False
    if "const" in schema and schema["const"] != value:
        return False
    if "anyOf" in schema and not any(sub(s, value) for s in schema["anyOf"]):
        return False
    if "oneOf" in schema and sum(sub(s, value) for s in schema["oneOf"]) != 1:
        return False
    if not all(sub(s, value) for s in schema.get("allOf", [])):
        return False
    if "if" in schema and sub(schema["if"], value) and not sub(schema.get("then", {}), value):
        return False
    return _valid_numbers(schema, value) and _valid_containers(schema, value, sub)


def _valid_numbers(schema: Mapping[str, Any], value: Any) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return True
    return (
        value >= schema.get("minimum", value)
        and value <= schema.get("maximum", value)
        and value > schema.get("exclusiveMinimum", value - 1)
    )


def _valid_containers(schema: Mapping[str, Any], value: Any, sub: Any) -> bool:
    if isinstance(value, str):
        return len(value) >= schema.get("minLength", 0)
    if isinstance(value, list):
        items = cast("list[Any]", value)
        if schema.get("uniqueItems") and len(set(map(json.dumps, items))) != len(items):
            return False
        return all(sub(schema.get("items", {}), item) for item in items)
    if not isinstance(value, dict):
        return True
    table = cast("dict[str, Any]", value)
    props: Mapping[str, Any] = schema.get("properties", {})
    extra = schema.get("additionalProperties", True)
    for key, item in table.items():
        if key in props:
            if not sub(props[key], item):
                return False
        elif extra is False or (isinstance(extra, dict) and not sub(extra, item)):
            return False
    if not all(key in table for key in schema.get("required", [])):
        return False
    return all(
        all(need in table for need in needs)
        for key, needs in schema.get("dependentRequired", {}).items()
        if key in table
    )


type Json = dict[str, Any]


def config_json_schema(schema: object) -> Json:
    return cast("Json", jsonschema.config_json_schema(schema))


def loader_json_schema(
    entries: Sequence[Entry],
    allow: Sequence[str] = ("*",),
    *,
    overlay: bool = False,
    skipped: Callable[[str, Exception], None] | None = None,
) -> Json:
    return cast(
        "Json", jsonschema.loader_json_schema(entries, allow, overlay=overlay, skipped=skipped)
    )


# -- dataclass schemas ---------------------------------------------------------------


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class Inner:
    """Inner part."""

    name: str
    weight: float = 1.0


@dataclass
class Everything:
    """Every supported annotation."""

    text: str
    count: int
    secret: Volatile[str]
    ratio: float = 0.5
    flag: bool = False
    mode: Literal["a", "b"] = "a"
    maybe: int | None = None
    either: int | str = 1
    names: list[str] = field(default_factory=lambda: [])
    pair: tuple[int, ...] = (1, 2)
    table: dict[str, int] = field(default_factory=lambda: {})
    mapping: Mapping[str, Inner] = field(default_factory=lambda: {})
    color: Color = Color.RED
    inner: Inner = field(default_factory=lambda: Inner("x"))
    anything: Any = None
    plain_list: list[Any] = field(default_factory=lambda: [])
    plain_dict: dict[str, Any] = field(default_factory=lambda: {})
    derived: int = field(default=0, init=False)


GOOD: dict[str, Any] = {"text": "t", "count": 2, "secret": "s"}
CASES: list[tuple[str, dict[str, Any]]] = [
    ("good minimal", GOOD),
    ("all fields", {**GOOD, "ratio": 2, "flag": True, "mode": "b", "maybe": None, "either": "s"}),
    ("missing required", {"text": "t", "count": 2}),
    ("unknown key", {**GOOD, "nope": 1}),
    ("non-init field", {**GOOD, "derived": 1}),
    ("bool is not int", {**GOOD, "count": True}),
    ("int for float", {**GOOD, "ratio": 3}),
    ("bool is not float", {**GOOD, "ratio": False}),
    ("float is not int", {**GOOD, "count": 2.5}),
    ("bad literal", {**GOOD, "mode": "c"}),
    ("literal wrong type", {**GOOD, "mode": 1}),
    ("optional none", {**GOOD, "maybe": None}),
    ("optional wrong", {**GOOD, "maybe": "x"}),
    ("union wrong", {**GOOD, "either": 1.5}),
    ("list ok", {**GOOD, "names": ["a", "b"]}),
    ("list bad item", {**GOOD, "names": ["a", 1]}),
    ("list not list", {**GOOD, "names": "a"}),
    ("tuple as array", {**GOOD, "pair": [3, 4]}),
    ("tuple bad item", {**GOOD, "pair": [3, "x"]}),
    ("dict ok", {**GOOD, "table": {"a": 1}}),
    ("dict bad value", {**GOOD, "table": {"a": "x"}}),
    ("mapping nested", {**GOOD, "mapping": {"k": {"name": "n"}}}),
    ("mapping nested bad", {**GOOD, "mapping": {"k": {}}}),
    ("enum value", {**GOOD, "color": "blue"}),
    ("enum bad", {**GOOD, "color": "green"}),
    ("enum by name is wrong", {**GOOD, "color": "BLUE"}),
    ("nested ok", {**GOOD, "inner": {"name": "n", "weight": 2}}),
    ("nested unknown", {**GOOD, "inner": {"name": "n", "zzz": 1}}),
    ("nested not object", {**GOOD, "inner": 1}),
    ("volatile is inner type", {**GOOD, "secret": "x"}),
    ("volatile bad", {**GOOD, "secret": 1}),
    ("any", {**GOOD, "anything": [1, {"a": None}]}),
    ("bare list", {**GOOD, "plain_list": [1, "a"]}),
    ("bare dict", {**GOOD, "plain_dict": {"a": [1]}}),
]


@pytest.mark.parametrize(("label", "config"), CASES, ids=[c[0] for c in CASES])
def test_schema_agrees_with_runtime_validator(label: str, config: dict[str, Any]) -> None:
    del label
    try:
        validate_config(Everything, config)
        runtime_ok = True
    except ValidationError:
        runtime_ok = False
    assert valid(config_json_schema(Everything), config) is runtime_ok


def test_dataclass_schema_shape() -> None:
    schema = config_json_schema(Everything)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "Everything"
    assert schema["description"] == "Every supported annotation."
    assert schema["required"] == ["text", "count", "secret"]
    assert schema["additionalProperties"] is False
    props = cast("dict[str, Any]", schema["properties"])
    assert "derived" not in props
    assert props["ratio"] == {"type": "number", "default": 0.5}
    assert props["color"] == {"enum": ["red", "blue"], "default": "red"}
    assert props["pair"]["default"] == [1, 2]
    assert props["secret"] == {"type": "string"}
    assert props["inner"]["$ref"] == "#/$defs/Inner"
    assert "cannot be written in JSON" in props["inner"]["description"]
    assert schema["$defs"]["Inner"]["description"] == "Inner part."


@dataclass
class PlainConfig:
    a: int


def test_generated_dataclass_docstring_is_not_a_description() -> None:
    assert "description" not in config_json_schema(PlainConfig)
    json.dumps(config_json_schema(Everything))  # serialisable


@dataclass
class Node:
    """A self-referential config."""

    value: int
    children: list[Node] = field(default_factory=lambda: [])
    parent: Node | None = None


def test_self_referential_dataclass_uses_refs() -> None:
    schema = config_json_schema(Node)
    assert schema["properties"]["children"]["items"] == {"$ref": "#"}
    tree = {"value": 1, "children": [{"value": 2, "parent": {"value": 3}}]}
    assert valid(schema, tree)
    assert not valid(schema, {"value": 1, "children": [{"value": "x"}]})


@dataclass
class Wrapper:
    node: Node
    other: Node | None = None


def test_nested_self_referential_dataclass_terminates() -> None:
    schema = config_json_schema(Wrapper)
    assert list(schema["$defs"]) == ["Node"]
    assert schema["$defs"]["Node"]["properties"]["parent"]["anyOf"][0] == {"$ref": "#/$defs/Node"}
    assert valid(schema, {"node": {"value": 1, "children": [{"value": 2}]}})
    assert not valid(schema, {"node": {"value": "x"}})


@dataclass
class Odd:
    pair: tuple[int, str] = (1, "a")
    items: set[int] = field(default_factory=lambda: set[int]())
    lit: Literal[b"x"] = b"x"
    mapping: Mapping[str, int] = field(default_factory=lambda: {})


def test_unsupported_annotations_accept_anything_with_a_description() -> None:
    props = config_json_schema(Odd)["properties"]
    for name in ("pair", "items", "lit"):
        assert "Unsupported annotation" in props[name]["description"], name
        assert "type" not in props[name]
    assert props["mapping"]["additionalProperties"] == {"type": "integer"}


def test_non_dataclass_schema_accepts_anything() -> None:
    config_models: list[object] = [None, dict, str]
    for schema in config_models:
        result = config_json_schema(schema)
        assert result["$schema"].endswith("2020-12/schema")
        assert "description" in result
        assert "type" not in result


# -- the loader file format ----------------------------------------------------------


@pytest.fixture
def plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """An importable module ``schemafix`` with a typed plugin, an untyped one and a model."""
    (tmp_path / "schemafix.py").write_text(
        "from dataclasses import dataclass\n"
        "from typing import Literal\n\n"
        "@dataclass\nclass ChatConfig:\n    model: str\n    retries: int = 3\n"
        "    mode: Literal['fast', 'slow'] = 'fast'\n\n"
        "def chat(ctx, config): pass\nchat.Config = ChatConfig\n\n"
        "def bare(ctx, config): pass\n\n"
        "def modelled(ctx, config): pass\nmodelled.Config = lambda raw: raw\n"
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    monkeypatch.delitem(sys.modules, "schemafix", raising=False)
    return "schemafix"


def _entries(tmp_path: Path, text: str) -> list[Entry]:
    path = tmp_path / "app.toml"
    path.write_text(text)
    return load_tree(path)[0]


def test_loader_schema_types_each_plugin_config(plugins: str, tmp_path: Path) -> None:
    entries = _entries(
        tmp_path,
        f"""
[[plugins]]
use = "{plugins}:chat"
[[plugins]]
use = "{plugins}:chat"
[[plugins]]
use = "{plugins}:bare"
[[plugins]]
use = "{plugins}:modelled"
""",
    )
    schema = loader_json_schema(entries)
    clauses = schema["$defs"]["entry"]["allOf"]
    assert len(clauses) == 1  # deduplicated; untyped plugins add nothing
    assert clauses[0]["if"]["properties"]["use"] == {"const": f"{plugins}:chat"}
    assert schema["$defs"]["ChatConfig"]["required"] == ["model"]

    def check(config: dict[str, Any], use: str = f"{plugins}:chat") -> bool:
        return valid(schema, {"plugins": [{"use": use, "config": config}]})

    assert check({"model": "m"})
    assert not check({"model": "m", "retries": "x"})
    assert not check({"model": "m", "mode": "other"})
    assert not check({})
    assert check({"anything": 1}, f"{plugins}:bare")


def test_loader_schema_types_nested_group_entries(plugins: str, tmp_path: Path) -> None:
    entries = _entries(
        tmp_path,
        f'[[plugins]]\ngroup = "g"\n[[plugins.plugins]]\nuse = "{plugins}:chat"\n',
    )
    schema = loader_json_schema(entries)
    nested: Json = {
        "plugins": [{"group": "g", "plugins": [{"use": f"{plugins}:chat", "config": {}}]}]
    }
    assert not valid(schema, nested)
    good = {
        "plugins": [
            {"group": "g", "plugins": [{"use": f"{plugins}:chat", "config": {"model": "m"}}]}
        ]
    }
    assert valid(schema, good)


def test_loader_schema_skips_unimportable_and_disallowed(plugins: str, tmp_path: Path) -> None:
    entries = _entries(
        tmp_path,
        f"""
[[plugins]]
use = "{plugins}:chat"
[[plugins]]
use = "nosuchmodule_x:p"
[[plugins]]
use = "{plugins}:missing_attr"
[[plugins]]
use = "{plugins}:chat"
id = "off"
disabled = true
[[plugins]]
use = "json:loads"
""",
    )
    skipped: list[tuple[str, str]] = []
    schema = loader_json_schema(
        entries, [plugins], skipped=lambda use, error: skipped.append((use, type(error).__name__))
    )
    assert skipped == [
        ("nosuchmodule_x:p", "PermissionError"),
        (f"{plugins}:missing_attr", "ImportError"),
        ("json:loads", "PermissionError"),
    ]
    assert len(schema["$defs"]["entry"]["allOf"]) == 1


def test_loader_schema_default_note_goes_to_stderr(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    entries = _entries(tmp_path, '[[plugins]]\nuse = "nosuchmodule_x:p"\n')
    loader_json_schema(entries)
    err = capsys.readouterr().err
    assert "nosuchmodule_x:p" in err
    assert "left untyped" in err


def test_disabled_entries_are_not_imported(tmp_path: Path) -> None:
    entries = _entries(tmp_path, '[[plugins]]\nuse = "nosuchmodule_x:p"\ndisabled = true\n')
    skipped: list[str] = []
    loader_json_schema(entries, skipped=lambda use, _e: skipped.append(use))
    assert skipped == []


@pytest.fixture(scope="module")
def structure() -> dict[str, Any]:
    return loader_json_schema([])


VALID_FILES: list[Json] = [
    {},
    {"$schema": "./pycordis.schema.json", "plugins": []},
    {"plugins": []},
    {"loader": {"readonly": True, "watch": False, "allow": ["a.*"], "poll_interval": 0.5}},
    {"plugins": [{"use": "m:p", "config": {"x": 1}, "id": "a", "disabled": True}]},
    {"plugins": [{"use": "m:p", "isolate": ["db"], "inject": ["x"], "intercept": {"a": 1}}]},
    {"plugins": [{"use": "m:p", "isolate": {"db": "shared"}}]},
    {"plugins": [{"group": "g", "plugins": [{"include": "x.toml"}]}]},
    {"plugins": [{"use": "m:p", "restart": {"mode": "on-failure", "backoff": {"max": 5}}}]},
    {"plugins": [{"use": "m:p", "load_timeout": 5, "dispose_timeout": 2.5}]},
]


@pytest.mark.parametrize("doc", VALID_FILES)
def test_loader_schema_accepts_valid_files(structure: dict[str, Any], doc: dict[str, Any]) -> None:
    assert valid(structure, doc)


INVALID_FILES: list[Json] = [
    {"extra": 1},
    {"$schema": 1},
    {"loader": {"readonly": "yes"}},
    {"loader": {"poll_interval": 0}},
    {"loader": {"poll_interval": True}},
    {"loader": {"allow": "a"}},
    {"loader": {"unknown": 1}},
    {"plugins": {}},
    {"plugins": [{}]},
    {"plugins": [{"use": "m:p", "group": "g"}]},
    {"plugins": [{"use": ""}]},
    {"plugins": [{"group": "g", "config": {}}]},
    {"plugins": [{"use": "m:p", "plugins": []}]},
    {"plugins": [{"use": "m:p", "bogus": 1}]},
    {"plugins": [{"use": "m:p", "disabled": "no"}]},
    {"plugins": [{"use": "m:p", "isolate": ["a", "a"]}]},
    {"plugins": [{"use": "m:p", "isolate": {"a": 1}}]},
    {"plugins": [{"use": "m:p", "inject": [1]}]},
    {"plugins": [{"use": "m:p", "restart": {"mode": "sometimes"}}]},
    {"plugins": [{"use": "m:p", "restart": {"max_restarts": -1}}]},
    {"plugins": [{"use": "m:p", "restart": {"backoff": {"jitter": 2}}}]},
    {"plugins": [{"use": "m:p", "load_timeout": 0}]},
    {"plugins": [{"group": "g", "plugins": [{"use": "m:p", "x": 1}]}]},
]


@pytest.mark.parametrize("doc", INVALID_FILES)
def test_loader_schema_rejects_invalid_files(
    structure: dict[str, Any], doc: dict[str, Any]
) -> None:
    assert not valid(structure, doc)


def test_loader_schema_accepts_every_fixture_config() -> None:
    schema = loader_json_schema([])
    files = [*LOADER_FIXTURES.rglob("*.toml"), *LOADER_FIXTURES.rglob("*.json")]
    assert len(files) >= 5
    for path in files:
        assert valid(schema, read_config(path)), path.name


def test_loader_schema_with_resolved_fixture_tree() -> None:
    entries, options = load_tree(LOADER_FIXTURES / "basic.toml")
    notes: list[str] = []
    schema = loader_json_schema(entries, options.allow, skipped=lambda use, _e: notes.append(use))
    assert "my_agent.services:OpenAIClient" in notes  # not importable here: left untyped
    assert valid(schema, read_config(LOADER_FIXTURES / "basic.toml"))


# -- CLI -----------------------------------------------------------------------------


def _write_app(tmp_path: Path, plugins: str) -> Path:
    path = tmp_path / "app.toml"
    path.write_text(f'[[plugins]]\nuse = "{plugins}:chat"\nconfig = {{ model = "m" }}\n')
    return path


def test_cli_schema_prints_to_stdout(
    plugins: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _write_app(tmp_path, plugins)
    assert cli.main(["schema", "-c", str(app)]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["$schema"].endswith("2020-12/schema")
    assert "ChatConfig" in schema["$defs"]
    assert valid(schema, read_config(app))


def test_cli_schema_writes_output_file(
    plugins: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _write_app(tmp_path, plugins)
    out = tmp_path / "out.json"
    assert cli.main(["schema", "-c", str(app), "-o", str(out)]) == 0
    assert capsys.readouterr().out == ""
    assert "ChatConfig" in json.loads(out.read_text())["$defs"]


def test_cli_schema_honours_allow_override(
    plugins: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _write_app(tmp_path, plugins)
    assert cli.main(["schema", "-c", str(app), "--allow", "other.*"]) == 0
    captured = capsys.readouterr()
    assert "ChatConfig" not in json.loads(captured.out)["$defs"]
    assert re.search(r"not allowed", captured.err)


def test_cli_schema_config_error_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[[plugins]]\nbogus = 1\n")
    assert cli.main(["schema", "-c", str(bad)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "pycordis: error:" in captured.err
    assert cli.main(["schema", "-c", str(tmp_path / "missing.toml")]) == 1


def test_cli_schema_unwritable_output_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = tmp_path / "app.toml"
    app.write_text("")
    assert cli.main(["schema", "-c", str(app), "-o", str(tmp_path / "no" / "out.json")]) == 1
    assert "cannot write" in capsys.readouterr().err


# -- naming and bare containers ------------------------------------------------------


class First:
    @dataclass
    class Item:
        a: int


class Second:
    @dataclass
    class Item:
        b: str


@dataclass
class Both:
    one: First.Item
    two: Second.Item


def test_same_named_dataclasses_get_distinct_defs() -> None:
    schema = config_json_schema(Both)
    assert sorted(schema["$defs"]) == ["Item", "Item_2"]
    assert schema["properties"]["one"] != schema["properties"]["two"]
    assert valid(schema, {"one": {"a": 1}, "two": {"b": "x"}})
    assert not valid(schema, {"one": {"b": "x"}, "two": {"b": "x"}})


def test_bare_list_and_dict_annotations() -> None:
    bare = make_dataclass("Bare", [("items", list), ("table", dict)])
    schema = config_json_schema(bare)
    assert schema["properties"]["items"] == {"type": "array"}
    assert schema["properties"]["table"] == {"type": "object"}
    assert valid(schema, {"items": [1, "a"], "table": {"k": 1}})
    assert not valid(schema, {"items": {}, "table": []})


def test_plain_class_annotation_is_unsupported() -> None:
    odd = make_dataclass("Odd", [("when", Path)])
    assert "Unsupported annotation" in config_json_schema(odd)["properties"]["when"]["description"]


# -- non-finite defaults and stable output -------------------------------------------


@dataclass
class Limits:
    top: float = float("inf")
    low: float = -float("inf")
    none: float = float("nan")
    ok: float = 1.5
    many: list[float] = field(default_factory=lambda: [1.0, float("nan")])


def test_non_finite_defaults_are_omitted_so_output_is_strict_json() -> None:
    schema = config_json_schema(Limits)
    props = schema["properties"]
    for name in ("top", "low", "none", "many"):
        assert "default" not in props[name], name
        assert "cannot be written in JSON" in props[name]["description"], name
    assert props["ok"]["default"] == 1.5
    json.dumps(schema, allow_nan=False)


@dataclass
class Late:
    x: int


@dataclass
class Early:
    y: int


@dataclass
class Order1:
    late: Late
    early: Early


@dataclass
class Order2:
    early: Early
    late: Late


def test_defs_are_sorted_so_reordering_does_not_change_output() -> None:
    assert list(config_json_schema(Order1)["$defs"]) == ["Early", "Late"]
    assert config_json_schema(Order1)["$defs"] == config_json_schema(Order2)["$defs"]
    assert list(loader_json_schema([])["$defs"]) == ["backoff", "entry", "restart"]


# -- profile overlays ----------------------------------------------------------------

BASE = """
[[plugins]]
id = "m"
use = "{p}:chat"
config = {{ model = "a" }}
[[plugins]]
id = "off"
use = "{p}:chat"
disabled = true
[[plugins]]
id = "g"
group = "grp"
"""


def _overlay(plugins: str, tmp_path: Path) -> Json:
    return loader_json_schema(_entries(tmp_path, BASE.format(p=plugins)), overlay=True)


def test_overlay_schema_accepts_partial_entries(plugins: str, tmp_path: Path) -> None:
    schema = _overlay(plugins, tmp_path)
    assert "overlay_entry" in schema["$defs"]
    assert "entry" not in schema["$defs"]
    assert schema["properties"]["plugins"]["items"] == {"$ref": "#/$defs/overlay_entry"}
    good = {
        "loader": {"watch": True},
        "plugins": [
            {"id": "m", "config": {"retries": 9}},  # partial: no required model
            {"id": "off", "disabled": False},
            {"id": "g", "plugins": [{"use": f"{plugins}:chat", "config": {"model": "x"}}]},
            {"use": f"{plugins}:chat", "config": {"anything": 1}},  # a new entry
            {"group": "new", "plugins": [{"id": "g2", "disabled": True}]},
        ],
    }
    assert valid(schema, good)


def test_overlay_schema_allows_a_schema_string(plugins: str, tmp_path: Path) -> None:
    schema = _overlay(plugins, tmp_path)
    assert valid(schema, {"$schema": "./overlay.schema.json"})
    assert not valid(schema, {"$schema": None})


async def test_schema_key_round_trips_through_load_and_save(plugins: str, tmp_path: Path) -> None:
    app = tmp_path / "app.json"
    pointer = "./app.schema.json"
    entry = {"id": "m", "use": f"{plugins}:chat", "config": {"model": "a"}}
    app.write_text(json.dumps({"$schema": pointer, "plugins": [entry]}, indent=2))
    entries, options = load_tree(app)
    schema = loader_json_schema(entries, options.allow)
    assert valid(schema, read_config(app))
    async with mock_context() as root:
        await root.plugin(Loader, LoaderConfig(path=str(app)))
        loader: Loader = root.loader
        await loader.start()
        loaded = loader.get("m")
        assert loaded is not None and loaded.fiber is not None
        assert loaded.fiber.state == FiberState.ACTIVE
        await loader.update("m", {"config": {"model": "b"}})
        await loader.save()
    saved = read_config(app)
    assert saved["$schema"] == pointer
    assert saved["plugins"] == [{"id": "m", "use": f"{plugins}:chat", "config": {"model": "b"}}]
    assert valid(schema, saved)


MALFORMED_OVERLAY_ENTRIES: list[Json] = [
    {"config": {"retries": 9}},  # no kind and no id
    {"id": "m", "config": {"retries": "many"}},  # typed from the base plugin
    {"id": "m", "config": {"nope": 1}},
    {"id": "off", "config": {"mode": "other"}},  # a disabled base entry is typed too
    {"id": "m", "use": "a:b", "group": "g"},
    {"id": "m", "bogus": 1},
    {"id": "m", "restart": {"mode": "x"}},
    {"id": "g", "plugins": [{"config": {}}]},
]


@pytest.mark.parametrize("entry", MALFORMED_OVERLAY_ENTRIES)
def test_overlay_schema_rejects_malformed_entries(
    plugins: str, tmp_path: Path, entry: Json
) -> None:
    assert not valid(_overlay(plugins, tmp_path), {"plugins": [entry]})


def test_base_schema_rejects_what_overlay_allows(plugins: str, tmp_path: Path) -> None:
    entries = _entries(tmp_path, BASE.format(p=plugins))
    partial = {"plugins": [{"id": "m", "config": {"retries": 9}}]}
    assert not valid(loader_json_schema(entries), partial)
    assert valid(loader_json_schema(entries, overlay=True), partial)


def test_overlay_for_real_profile_file_validates(plugins: str, tmp_path: Path) -> None:
    base = tmp_path / "app.toml"
    base.write_text(BASE.format(p=plugins))
    overlay_file = tmp_path / "app.dev.toml"
    overlay_file.write_text('[[plugins]]\nid = "m"\nconfig = { retries = 9 }\n')
    entries, _ = load_tree(base, profile="dev")  # the loader accepts it
    schema = loader_json_schema(entries, overlay=True)
    assert valid(schema, read_config(overlay_file))
    overlay_file.write_text('[[plugins]]\nid = "m"\nconfig = { retries = "x" }\n')
    assert not valid(schema, read_config(overlay_file))


def test_cli_schema_overlay_flag(
    plugins: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = tmp_path / "app.toml"
    app.write_text(BASE.format(p=plugins))
    assert cli.main(["schema", "-c", str(app), "--overlay"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert "overlay_entry" in schema["$defs"]
    assert valid(schema, {"plugins": [{"id": "m", "config": {"retries": 2}}]})


def test_unbuildable_config_schema_is_reported_and_left_untyped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "brokenfix.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\nclass Cfg:\n    x: 'Undefined'\n\n"
        "def p(ctx, config): pass\np.Config = Cfg\n"
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    monkeypatch.delitem(sys.modules, "brokenfix", raising=False)
    entries = _entries(tmp_path, '[[plugins]]\nuse = "brokenfix:p"\n')
    seen: list[str] = []
    schema = loader_json_schema(entries, skipped=lambda use, _e: seen.append(use))
    assert seen == ["brokenfix:p"]
    assert valid(schema, {"plugins": [{"use": "brokenfix:p", "config": {"x": 1}}]})
