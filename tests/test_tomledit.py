import copy
import datetime as dt
import math
import random
import tomllib
from pathlib import Path
from typing import Any

import pytest

from pycordis.tomledit import TomlDocument, dumps_value

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "toml").glob("*.toml"))

SRC = """# top comment
[[plugins]]
use = "a:b"   # trailing
config = { model = "x", n = 1 }

[[plugins]]
use = "c:d"
[plugins.config]
key = 'single'  # keep me
"""


def read(path: Path) -> str:
    return path.read_bytes().decode()


type TomlValue = (
    str
    | int
    | float
    | bool
    | dt.datetime
    | dt.date
    | dt.time
    | list[TomlValue]
    | dict[str, TomlValue]
)
"""A value ``tomllib`` can produce."""


def same(a: TomlValue, b: TomlValue) -> bool:
    """Strict structural equality (1 != 1.0 != True; nan == nan)."""
    if type(a) is not type(b):
        return False
    # a and b now share a type; narrowing both lets the checker see that
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    return bool(a == b)


# --- brief tests -----------------------------------------------------------


def test_set_scalar_preserves_everything_else() -> None:
    doc = TomlDocument.parse(SRC)
    doc.set(("plugins", 0, "config", "model"), "y")
    assert doc.dumps() == SRC.replace('model = "x"', 'model = "y"')


def test_set_in_table_section_and_insert_key() -> None:
    doc = TomlDocument.parse(SRC)
    doc.set(("plugins", 1, "config", "new"), [1, 2])
    out = doc.dumps()
    assert "key = 'single'  # keep me\nnew = [1, 2]\n" in out and "# top comment" in out


def test_delete_key() -> None:
    doc = TomlDocument.parse(SRC)
    doc.delete(("plugins", 0, "config", "n"))
    assert doc.dumps() == SRC.replace('{ model = "x", n = 1 }', '{ model = "x" }')
    doc.delete(("plugins", 0, "config", "model"))
    assert "config = {}\n" in doc.dumps()


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_roundtrip_fixtures(path: Path) -> None:
    text = read(path)
    doc = TomlDocument.parse(text)
    assert doc.dumps() == text
    assert same(doc.data, tomllib.loads(text))


def test_crlf_fixture_really_has_crlf() -> None:
    assert "\r\n" in read(next(p for p in FIXTURES if p.name == "crlf.toml"))


def test_replace_config_dict_diff() -> None:
    doc = TomlDocument.parse(SRC)
    doc.set(("plugins", 1, "config"), {"key": "single", "z": 1})
    assert doc.dumps() == SRC + "z = 1\n"
    doc.set(("plugins", 1, "config"), {"z": 2})
    assert doc.dumps().endswith("[plugins.config]\nz = 2\n")


# --- get / data ------------------------------------------------------------


def test_get_and_data_are_copies() -> None:
    doc = TomlDocument.parse(SRC)
    assert doc.get(("plugins", 1, "config", "key")) == "single"
    assert doc.get(()) == doc.data
    doc.data["plugins"].clear()
    doc.get(("plugins",)).clear()
    assert len(doc.data["plugins"]) == 2
    with pytest.raises(KeyError):
        doc.get(("nope",))
    with pytest.raises(IndexError):
        doc.get(("plugins", 5))


def test_invalid_toml_rejected() -> None:
    with pytest.raises(tomllib.TOMLDecodeError):
        TomlDocument.parse("a = ")


# --- set: replacing values ---------------------------------------------------


def test_set_preserves_literal_quote_style() -> None:
    doc = TomlDocument.parse(SRC)
    doc.set(("plugins", 1, "config", "key"), "other")
    assert "key = 'other'  # keep me\n" in doc.dumps()
    doc.set(("plugins", 1, "config", "key"), "it's")
    assert 'key = "it\'s"  # keep me\n' in doc.dumps()


def test_set_same_value_is_noop_and_type_strict() -> None:
    doc = TomlDocument.parse("a = 1\n")
    doc.set(("a",), 1)
    assert doc.dumps() == "a = 1\n"
    doc.set(("a",), 1.0)
    assert doc.dumps() == "a = 1.0\n"
    doc.set(("a",), True)
    assert doc.dumps() == "a = true\n"


def test_set_array_elements_keeps_comments() -> None:
    text = read(next(p for p in FIXTURES if p.name == "arrays.toml"))
    doc = TomlDocument.parse(text)
    doc.set(("multi",), [1, 20, 3])
    assert "  20,\n" in doc.dumps() and "# one" in doc.dumps()
    doc.set(("multi",), [9])
    assert "multi = [9]\n" in doc.dumps()
    doc.set(("integers", 0), 7)
    assert "integers = [ 7, 2, 3 ]" in doc.dumps()


def test_set_dotted_and_crlf() -> None:
    doc = TomlDocument.parse("a.b.c = 1\r\na.b.d = 2\r\nz = 0\r\n")
    doc.set(("a", "b", "c"), 5)
    doc.set(("a", "b", "e"), 3)
    doc.set(("a", "x"), "y")
    assert doc.dumps() == 'a.b.c = 5\r\na.b.d = 2\r\na.b.e = 3\r\na.x = "y"\r\nz = 0\r\n'


def test_set_table_to_scalar_and_back() -> None:
    doc = TomlDocument.parse("top = 1\n\n[t]\nx = 1\n[t.sub]\ny = 2\n\n[u]\nk = 1\n")
    doc.set(("t",), 5)
    assert doc.dumps() == "top = 1\nt = 5\n\n\n[u]\nk = 1\n"
    doc.set(("t",), {"a": 1})
    assert doc.dumps() == "top = 1\nt = { a = 1 }\n\n\n[u]\nk = 1\n"
    doc.delete(("t",))
    doc.set(("t",), {"a": 1, "b": {"c": [1]}})
    assert doc.data["t"] == {"a": 1, "b": {"c": [1]}}
    assert doc.dumps() == "top = 1\n\n\n[u]\nk = 1\n\n[t]\na = 1\n\n[t.b]\nc = [1]\n"


def test_set_scalar_to_inline_table() -> None:
    doc = TomlDocument.parse("a = 1  # c\n")
    doc.set(("a",), {"k": "v", "n": {"m": 1}})
    assert doc.dumps() == 'a = { k = "v", n = { m = 1 } }  # c\n'


# --- set: inserting -----------------------------------------------------------


def test_insert_into_inline_table() -> None:
    doc = TomlDocument.parse(SRC)
    doc.set(("plugins", 0, "config", "extra"), {"q": True})
    assert 'config = { model = "x", n = 1, extra = { q = true } }' in doc.dumps()
    doc2 = TomlDocument.parse("e = {}\n")
    doc2.set(("e", "k"), 1)
    assert doc2.dumps() == "e = { k = 1 }\n"


def test_insert_missing_intermediate_tables() -> None:
    doc = TomlDocument.parse("[a]\nx = 1\n\n# next\n[b]\ny = 2\n")
    doc.set(("a", "new", "deep", "k"), 1)
    assert doc.dumps() == "[a]\nx = 1\n\n[a.new.deep]\nk = 1\n\n# next\n[b]\ny = 2\n"
    doc.set(("c", "d"), "e")
    assert doc.dumps().endswith('[b]\ny = 2\n\n[c]\nd = "e"\n')


def test_insert_into_root_before_first_header() -> None:
    doc = TomlDocument.parse(SRC)
    doc.set(("title",), "t")
    assert doc.dumps() == SRC.replace("[[plugins]]", 'title = "t"\n[[plugins]]', 1)
    doc2 = TomlDocument.parse("# only a comment")
    doc2.set(("k",), 1)
    assert doc2.dumps() == "# only a comment\nk = 1\n"
    doc3 = TomlDocument.parse("")
    doc3.set(("t", "k"), 1)
    assert doc3.dumps() == "[t]\nk = 1\n"


def test_insert_matches_indentation_and_header_without_keys() -> None:
    doc = TomlDocument.parse("[t]\n  a = 1\n[u]  # c\n[v]")
    doc.set(("t", "b"), 2)
    doc.set(("u", "x"), 1)
    doc.set(("v", "y"), 1)
    assert doc.dumps() == "[t]\n  a = 1\n  b = 2\n[u]  # c\nx = 1\n[v]\ny = 1\n"


def test_insert_under_implicit_super_table() -> None:
    doc = TomlDocument.parse("[a.b.c]\nd = 1\n")
    doc.set(("a", "b", "k"), 1)
    doc.set(("a", "t"), {"z": 1})
    assert doc.dumps() == "a.b.k = 1\na.t = { z = 1 }\n[a.b.c]\nd = 1\n"


def test_emptying_dotted_table_keeps_it() -> None:
    doc = TomlDocument.parse("a.x = 1\nb = 2\n")
    doc.set(("a",), {})
    assert doc.data == {"a": {}, "b": 2}
    assert doc.dumps() == "b = 2\n\n[a]\n"


def test_insert_under_inline_dotted_key_rerenders_value() -> None:
    doc = TomlDocument.parse("d = { x.y = 1 }  # c\n")
    doc.set(("d", "x", "z"), 2)
    assert doc.dumps() == "d = { x = { y = 1, z = 2 } }  # c\n"


def test_insert_into_array_of_tables_element() -> None:
    text = read(next(p for p in FIXTURES if p.name == "aot.toml"))
    doc = TomlDocument.parse(text)
    doc.set(("fruits", 0, "varieties", 0, "taste"), "sweet")
    doc.set(("fruits", 0, "extra"), {"k": 1})
    out = doc.dumps()
    assert 'name = "red delicious"\ntaste = "sweet"\n' in out
    assert 'name = "granny smith"\n\n[fruits.extra]\nk = 1\n\n\n[[fruits]]' in out


def test_set_array_of_tables_length_change() -> None:
    doc = TomlDocument.parse(SRC)
    doc.set(("plugins",), [{"use": "a:b", "config": {"model": "x", "n": 1}}])
    assert doc.dumps() == SRC.split('\n[[plugins]]\nuse = "c:d"')[0] + "\n"
    doc.set(("plugins",), [{"use": "a:b", "config": {"model": "x", "n": 1}}, {"use": "e"}])
    assert doc.dumps().endswith('n = 1 }\n\n[[plugins]]\nuse = "e"\n\n')
    doc.set(("plugins",), [])
    assert doc.dumps() == "# top comment\n\nplugins = []\n"


def test_set_errors() -> None:
    doc = TomlDocument.parse("a = 1\nl = [1]\n")
    with pytest.raises(TypeError):
        doc.set(("a", "b"), 1)
    with pytest.raises(IndexError):
        doc.set(("l", 3), 1)
    with pytest.raises(TypeError):
        doc.set(("l", "x"), 1)
    with pytest.raises(TypeError):
        doc.set(("x",), object())
    with pytest.raises(TypeError):
        doc.set((), 1)
    assert doc.dumps() == "a = 1\nl = [1]\n"


def test_set_accepts_tuples() -> None:
    doc = TomlDocument.parse("a = [1]\n")
    doc.set(("a",), (1, 2))
    doc.set(("b",), {"x": (1,), "y": [(2, 3)]})
    assert doc.data == {"a": [1, 2], "b": {"x": [1], "y": [[2, 3]]}}
    assert doc.dumps() == "a = [1, 2]\n\n[b]\nx = [1]\ny = [[2, 3]]\n"


def test_set_rejects_unrepresentable_values_up_front() -> None:
    doc = TomlDocument.parse("a = 1\n")
    odd = dt.timezone(dt.timedelta(seconds=30))
    for bad in (dt.time(1, tzinfo=dt.UTC), dt.datetime(2020, 1, 1, tzinfo=odd), None, {1: 2}):
        with pytest.raises(TypeError):
            doc.set(("a",), bad)
    assert doc.dumps() == "a = 1\n"


def test_array_of_tables_element_must_be_table() -> None:
    doc = TomlDocument.parse(SRC)
    with pytest.raises(TypeError, match="array-of-tables element must be a table"):
        doc.set(("plugins", 0), 5)
    assert doc.dumps() == SRC


def test_negative_indices_rejected() -> None:
    doc = TomlDocument.parse(SRC)
    for call in (
        lambda: doc.get(("plugins", -1)),
        lambda: doc.set(("plugins", -1, "use"), "x"),
        lambda: doc.delete(("plugins", -1)),
    ):
        with pytest.raises(IndexError):
            call()
    assert doc.dumps() == SRC


# --- delete ----------------------------------------------------------------


def test_delete_line_with_comment_and_multiline_value() -> None:
    text = read(next(p for p in FIXTURES if p.name == "arrays.toml"))
    doc = TomlDocument.parse(text)
    doc.delete(("multi",))
    assert "multi" not in doc.dumps() and "# one" not in doc.dumps()
    doc = TomlDocument.parse(SRC)
    doc.delete(("plugins", 0, "use"))
    assert "trailing" not in doc.dumps()


def test_delete_array_elements() -> None:
    doc = TomlDocument.parse(
        "a = [1, 2, 3]\nm = [\n  1,  # one\n  2,\n]\nn = [\n  'a',\n  'b'\n]\n"
    )
    doc.delete(("a", 0))
    doc.delete(("a", 1))
    doc.delete(("m", 0))
    doc.delete(("n", 1))
    assert doc.dumps() == "a = [2]\nm = [\n  2,\n]\nn = [\n  'a'\n]\n"
    doc.delete(("a", 0))
    assert doc.dumps().startswith("a = []\n")


def test_delete_tables() -> None:
    text = read(next(p for p in FIXTURES if p.name == "aot.toml"))
    doc = TomlDocument.parse(text)
    doc.delete(("fruits", 0))
    out = doc.dumps()
    assert "apple" not in out and "granny" not in out and "# first fruit" in out
    assert doc.get(("fruits", 0, "name")) == "banana"
    doc.delete(("x",))
    assert "[x" not in doc.dumps()
    doc.delete(("products",))
    assert doc.data == {
        "title": "aot",
        "fruits": [{"name": "banana", "varieties": [{"name": "plantain"}]}],
    }


def test_delete_sole_array_of_tables_element() -> None:
    doc = TomlDocument.parse("[[a]]\nx = 1\n")
    doc.delete(("a", 0))
    assert doc.dumps() == "a = []\n"


def test_delete_dotted_table_and_inline_dotted() -> None:
    doc = TomlDocument.parse("a.b = 1\na.c = 2\nd = { x.y = 1, x.z = 2, w = 3 }\n")
    doc.delete(("a",))
    doc.delete(("d", "x", "y"))
    assert doc.data == {"d": {"x": {"z": 2}, "w": 3}}
    doc.delete(("d", "x"))
    assert doc.dumps() == "d = { w = 3 }\n"


def test_delete_errors() -> None:
    doc = TomlDocument.parse("a = 1\n")
    with pytest.raises(KeyError):
        doc.delete(("b",))
    with pytest.raises(ValueError):
        doc.delete(())


def test_inconsistent_output_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = TomlDocument.parse("a = 1\n")

    def dumps_two(_value: object) -> str:
        return "2"

    def dumps_broken(_value: object) -> str:
        return "["

    monkeypatch.setattr("pycordis.tomledit.dumps_value", dumps_two)
    with pytest.raises(RuntimeError, match="tomledit produced inconsistent output"):
        doc.set(("a",), 3)
    assert doc.dumps() == "a = 1\n" and doc.data == {"a": 1}
    monkeypatch.setattr("pycordis.tomledit.dumps_value", dumps_broken)
    with pytest.raises(RuntimeError, match="tomledit produced inconsistent output"):
        doc.set(("a",), 3)
    assert doc.dumps() == "a = 1\n"


# --- dumps_value ---------------------------------------------------------------


def test_dumps_value() -> None:
    assert dumps_value('a"b\\\n\x01\x7f') == '"a\\"b\\\\\\n\\u0001\\u007f"'
    assert dumps_value(True) == "true" and dumps_value(False) == "false"
    assert dumps_value(-3) == "-3"
    assert dumps_value(1.5) == "1.5" and dumps_value(1e300) == "1e+300"
    assert dumps_value(math.inf) == "inf" and dumps_value(-math.inf) == "-inf"
    assert dumps_value(math.nan) == "nan"
    assert dumps_value(dt.date(2020, 1, 2)) == "2020-01-02"
    assert dumps_value(dt.time(1, 2, 3)) == "01:02:03"
    stamp = dt.datetime(2020, 1, 2, 3, 4, 5, tzinfo=dt.UTC)
    assert dumps_value(stamp) == "2020-01-02T03:04:05+00:00"
    assert dumps_value([1, "a", list[int]()]) == '[1, "a", []]'
    assert dumps_value((1, 2)) == "[1, 2]"
    assert dumps_value({}) == "{}"
    assert dumps_value({"a b": 1, "c": {"d": 2}}) == '{ "a b" = 1, c = { d = 2 } }'
    with pytest.raises(TypeError):
        dumps_value(None)
    with pytest.raises(TypeError):
        dumps_value({1: 2})
    with pytest.raises(TypeError, match="UTC offset"):
        dumps_value(dt.time(1, tzinfo=dt.UTC))
    with pytest.raises(TypeError, match="whole minutes"):
        dumps_value(dt.datetime(2020, 1, 1, tzinfo=dt.timezone(dt.timedelta(seconds=1))))
    plus_hour = dt.timezone(dt.timedelta(hours=1))
    assert dumps_value(dt.datetime(2020, 1, 1, tzinfo=plus_hour)) == "2020-01-01T00:00:00+01:00"
    value = {"s": "x", "l": [1.5, dt.date(2020, 1, 1)], "t": {"u": [{"v": False}]}}
    assert tomllib.loads("k = " + dumps_value(value))["k"] == value


# --- fuzz-ish safety net ----------------------------------------------------------


def leaf_paths(value: TomlValue, path: tuple[str | int, ...] = ()) -> list[tuple[str | int, ...]]:
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in leaf_paths(v, (*path, k))]
    if isinstance(value, list) and value:
        return [p for i, v in enumerate(value) for p in leaf_paths(v, (*path, i))]
    return [path]


def table_paths(value: TomlValue, path: tuple[str | int, ...] = ()) -> list[tuple[str | int, ...]]:
    out: list[tuple[str | int, ...]] = []
    if isinstance(value, dict):
        out.append(path)
        for k, v in value.items():
            out += table_paths(v, (*path, k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out += table_paths(v, (*path, i))
    return out


def apply(
    data: Any, path: tuple[str | int, ...], value: Any = None, *, delete: bool = False
) -> None:
    node = data
    for seg in path[:-1]:
        node = node[seg]
    if delete:
        del node[path[-1]]
    else:
        node[path[-1]] = copy.deepcopy(value)


def is_scalar(value: Any) -> bool:
    return not isinstance(value, list | dict)


def comment_lines(text: str, exclude: set[str]) -> list[str]:
    lines = (ln.strip() for ln in text.splitlines())
    return [ln for ln in lines if ln.startswith("#") and ln not in exclude]


VALUES: list[Any] = [
    0,
    -5,
    2.5,
    True,
    "s",
    "it's",
    "multi\nline",
    [1, 2],
    {"k": "v"},
    [],
    {},
    [{"a": 1}],
]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
@pytest.mark.parametrize("seed", range(6))
def test_random_operations_stay_consistent(path: Path, seed: int) -> None:
    rng = random.Random(f"{path.name}-{seed}")
    doc = TomlDocument.parse(read(path))
    expected = tomllib.loads(read(path))
    for _ in range(12):
        leaves = [p for p in leaf_paths(expected) if p]
        op = rng.random()
        if leaves and op < 0.45:
            target = rng.choice(leaves)
            value = rng.choice(VALUES)
            old = doc.get(target)
            before = doc.dumps()
            doc.set(target, value)
            apply(expected, target, value)
            if is_scalar(old) and is_scalar(value):
                # only the value span changes: every comment line elsewhere survives
                inside = {ln.strip() for v in (old, value) for ln in str(v).splitlines()}
                assert comment_lines(doc.dumps(), inside) == comment_lines(before, inside)
        elif leaves and op < 0.7:
            target = rng.choice(leaves)
            doc.delete(target)
            apply(expected, target, delete=True)
        elif op < 0.85:  # whole-table replacement: drop one key, add one
            target = rng.choice(table_paths(expected))
            table = copy.deepcopy(doc.get(target))
            if table:
                del table[rng.choice(sorted(table))]
            table[f"t{rng.randrange(1000)}"] = rng.choice(VALUES)
            doc.set(target, table)
            if target:
                apply(expected, target, table)
            else:
                expected = copy.deepcopy(table)
        else:
            target = (*rng.choice(table_paths(expected)), f"new{rng.randrange(1000)}")
            value = rng.choice(VALUES)
            doc.set(target, value)
            apply(expected, target, value)
        assert same(tomllib.loads(doc.dumps()), expected)
        assert same(doc.data, expected)


def test_get_through_a_scalar_is_a_type_error() -> None:
    doc = TomlDocument('a = "text"\n')
    with pytest.raises(TypeError, match="cannot address 0 in str"):
        doc.get(("a", 0))
