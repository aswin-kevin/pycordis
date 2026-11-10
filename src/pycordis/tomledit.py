"""Format-preserving TOML editor (spec §6.6 write-back).

The document text is scanned into spans — table headers, key/value lines and,
inside values, array elements and inline-table entries — each tagged with its
absolute data path. Every edit is a text splice on those spans, so comments,
ordering, whitespace and quoting elsewhere are untouched. After each public
mutation the text is re-read with :mod:`tomllib` and compared (strictly) with
the intended data; any difference raises ``RuntimeError`` and the edit is
rolled back.

Placement rules for new content:

* a new key in a ``[table]`` / ``[[array]]`` section goes on the line after the
  section's last key line (dotted siblings: after the last sibling line), with
  that line's indentation; a section without keys gets it right after the
  header; the root table without keys gets it before the first header (or at
  end of file);
* a new key in an inline table is appended before ``}`` (kept on one line);
* a new *table* below a section becomes a new ``[path]`` section placed at the
  end of the parent's region (after its last sub-section), or at end of file
  for the root; new array-of-tables elements go after the last element;
* tables that exist only implicitly (super-tables of headers, dotted keys)
  receive dotted keys (``a.b.key = value``) in the nearest owning section;
* paths inside inline tables that use dotted keys fall back to re-rendering
  the enclosing top-level value.

Known limits: replaced values are rendered fresh, so a multiline string that is
replaced becomes a single-line basic string (a single-line literal string stays
literal when possible); a scalar replaced by a dict becomes an inline table; a
value array whose length changes is re-rendered whole (comments inside it are
lost); deleting a section keeps the comment lines above its header. Tuples are
written as arrays. Int path segments must be non-negative (IndexError).
"""

from __future__ import annotations

import copy
import datetime as dt
import math
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, final

from pycordis._guards import MISSING, Missing, is_array, is_dict, is_list

__all__ = ["TomlDocument", "dumps_value"]

type Path = tuple[str | int, ...]

type _TomlTable = dict[str, Any]
"""A parsed TOML table as :func:`tomllib.loads` returns it.

Its values are ``Any`` on purpose: callers index parsed documents of a shape
only they know (``doc.data["plugins"][0]["config"]``), as with ``tomllib``.
"""

_BARE = re.compile(r"[A-Za-z0-9_-]+")
_SCALAR = re.compile(r"[0-9A-Za-z_:.+\-]+")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}
_ERROR = "tomledit produced inconsistent output"


def dumps_value(value: object) -> str:
    """Render ``value`` as a TOML literal (dicts become inline tables)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return '"' + "".join(_escape(c) for c in value) + '"'
    if isinstance(value, dt.time) and value.tzinfo is not None:
        raise TypeError("TOML local times cannot carry a UTC offset")
    offset = value.utcoffset() if isinstance(value, dt.datetime) else None
    if offset is not None and (offset.seconds % 60 or offset.microseconds):
        raise TypeError("TOML datetime offsets must be whole minutes")
    if isinstance(value, dt.date | dt.time):
        return value.isoformat()
    if is_array(value):
        return "[" + ", ".join(dumps_value(v) for v in value) + "]"
    if is_dict(value):
        items = value.items()
        if not items:
            return "{}"
        return "{ " + ", ".join(f"{_key(k)} = {dumps_value(v)}" for k, v in items) + " }"
    raise TypeError(f"cannot represent {type(value).__name__} in TOML")


def _escape(c: str) -> str:
    if c in _ESCAPES:
        return _ESCAPES[c]
    return f"\\u{ord(c):04x}" if ord(c) < 0x20 or c == "\x7f" else c


def _key(key: object) -> str:
    if not isinstance(key, str):
        raise TypeError(f"TOML keys must be str, not {type(key).__name__}")
    return key if _BARE.fullmatch(key) else dumps_value(key)


def _same(a: object, b: object) -> bool:
    """Strict structural equality: types must match (``1 != 1.0``), nan equals nan."""
    if type(a) is not type(b):
        return False
    if is_dict(a) and is_dict(b):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if is_list(a) and is_list(b):
        return len(a) == len(b) and all(_same(i, j) for i, j in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    return bool(a == b)


def _lookup(data: object, path: Path) -> object | Missing:
    node = data
    for seg in path:
        if is_dict(node) and isinstance(seg, str):
            node = node.get(seg, MISSING)
        elif is_list(node) and isinstance(seg, int) and 0 <= seg < len(node):
            node = node[seg]
        else:
            return MISSING
    return node


def _apply(data: object, path: Path, value: object | Missing = MISSING) -> None:
    """Assign (or delete when ``value`` is missing) ``path`` in ``data``, creating tables."""
    node = data
    for i, seg in enumerate(path):
        last = i == len(path) - 1
        if is_dict(node) and isinstance(seg, str):
            table = node
            if last:
                if value is MISSING:
                    del table[seg]
                else:
                    table[seg] = copy.deepcopy(value)
                return
            node = table.setdefault(seg, {})
        elif is_list(node) and isinstance(seg, int):
            array = node
            if not 0 <= seg < len(array):
                raise IndexError(f"array index {seg} out of range at {path[:i]!r}")
            if last:
                if value is MISSING:
                    del array[seg]
                else:
                    array[seg] = copy.deepcopy(value)
                return
            node = array[seg]
        else:
            kind = type(node).__name__
            raise TypeError(f"cannot address {seg!r} in {kind} at {path[:i]!r}")


# --- span model ---------------------------------------------------------------------


@final
@dataclass(eq=False)
class _Value:
    start: int
    end: int
    kind: Literal["scalar", "array", "inline"]
    items: list[_Item] = field(default_factory=lambda: list[_Item]())


@final
@dataclass(eq=False)
class _Item:
    start: int  # value start (array) or key start (inline table)
    keys: tuple[str, ...]
    value: _Value
    comma: int | None = None


@final
@dataclass(eq=False)
class _Section:
    abs: Path
    line_start: int
    own_end: int  # end of the header line or of the section's last key line
    indent: str


@final
@dataclass(eq=False)
class _Entry:
    abs: Path
    section: _Section
    line_start: int
    line_end: int
    indent: str
    value: _Value


@final
class _Scanner:
    """Permissive TOML scanner; the input has already been validated by tomllib."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.root = _Section((), 0, 0, "")
        self.sections: list[_Section] = [self.root]
        self.entries: list[_Entry] = []

    def scan(self) -> None:
        text, aot_count = self.text, dict[Path, int]()
        section = self.root
        while True:
            self.ws()
            if self.pos >= len(text):
                return
            c = text[self.pos]
            if c in "#\r\n":
                self.line_tail()
                continue
            line_start = text.rfind("\n", 0, self.pos) + 1
            indent = text[line_start : self.pos]
            if c == "[":
                aot = text.startswith("[[", self.pos)
                self.pos += 2 if aot else 1
                self.ws()
                keys = self.key()
                self.pos += 2 if aot else 1
                path: Path = ()
                for i, k in enumerate(keys):
                    path = (*path, k)
                    if aot and i == len(keys) - 1:
                        aot_count[path] = aot_count.get(path, 0) + 1
                    if path in aot_count:
                        path = (*path, aot_count[path] - 1)
                self.line_tail()
                section = _Section(path, line_start, self.pos, indent)
                self.sections.append(section)
                continue
            keys = self.key()
            self.pos += 1  # "="
            self.ws()
            value = self.value()
            self.line_tail()
            abs_path = (*section.abs, *keys)
            self.entries.append(_Entry(abs_path, section, line_start, self.pos, indent, value))
            section.own_end = self.pos

    def ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1

    def line_tail(self) -> None:
        """Skip whitespace, an optional comment and one newline."""
        self.ws()
        if self.pos < len(self.text) and self.text[self.pos] == "#":
            nl = self.text.find("\n", self.pos)
            self.pos = len(self.text) if nl < 0 else nl
            if self.text[self.pos - 1] == "\r":
                self.pos -= 1
        if self.text.startswith("\r\n", self.pos):
            self.pos += 2
        elif self.text.startswith("\n", self.pos):
            self.pos += 1

    def trivia(self) -> None:
        """Skip whitespace, newlines and comments (inside arrays and inline tables)."""
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n#":
            self.line_tail()

    def key(self) -> tuple[str, ...]:
        keys: list[str] = []
        while True:
            self.ws()
            start, c = self.pos, self.text[self.pos]
            if c in "\"'":
                self.string()
                raw = self.text[start : self.pos]
                keys.append(raw[1:-1] if c == "'" else tomllib.loads(f"k = {raw}")["k"])
            else:
                match = _BARE.match(self.text, self.pos)
                assert match is not None
                keys.append(match.group())
                self.pos = match.end()
            self.ws()
            if self.text[self.pos] != ".":
                return tuple(keys)
            self.pos += 1

    def string(self) -> None:
        text, pos = self.text, self.pos
        quote = text[pos]
        if text.startswith(quote * 3, pos):
            i = pos + 3
            while not text.startswith(quote * 3, i):
                i += 2 if quote == '"' and text[i] == "\\" else 1
            j = i + 3
            while j < len(text) and text[j] == quote and j - i < 5:
                j += 1
            self.pos = j
            return
        i = pos + 1
        while text[i] != quote:
            i += 2 if quote == '"' and text[i] == "\\" else 1
        self.pos = i + 1

    def value(self) -> _Value:
        text, start = self.text, self.pos
        c = text[start]
        if c in "\"'":
            self.string()
            return _Value(start, self.pos, "scalar")
        if c in "[{":
            node = _Value(start, start, "array" if c == "[" else "inline")
            self.pos += 1
            while True:
                self.trivia()
                if text[self.pos] in "]}":
                    break
                item_start = self.pos
                keys: tuple[str, ...] = ()
                if c == "{":
                    keys = self.key()
                    self.pos += 1
                    self.trivia()
                item = _Item(item_start, keys, self.value())
                node.items.append(item)
                self.trivia()
                if text[self.pos] == ",":
                    item.comma = self.pos
                    self.pos += 1
            self.pos += 1
            node.end = self.pos
            return node
        match = _SCALAR.match(text, start)
        assert match is not None
        self.pos = match.end()
        if _DATE.fullmatch(match.group()) and re.match(r" \d", text[self.pos : self.pos + 2]):
            time_part = _SCALAR.match(text, self.pos + 1)
            assert time_part is not None  # tomllib accepted the document: a time follows
            self.pos = time_part.end()
        return _Value(start, self.pos, "scalar")


# --- document ---------------------------------------------------------------------------


@final
class TomlDocument:
    """A TOML document that can be edited while preserving its formatting."""

    def __init__(self, text: str) -> None:
        """Parse ``text``; raises :class:`tomllib.TOMLDecodeError` if it is not valid TOML."""
        self._text = text
        self._data: _TomlTable = tomllib.loads(text)
        self._index()

    @classmethod
    def parse(cls, text: str) -> TomlDocument:
        """Parse ``text`` into a document (same as the constructor)."""
        return cls(text)

    @property
    def data(self) -> _TomlTable:
        """A deep copy of the document's data (equal to ``tomllib.loads(self.dumps())``)."""
        return copy.deepcopy(self._data)

    def dumps(self) -> str:
        """Return the document text."""
        return self._text

    def get(self, path: Path) -> Any:  # a TOML value of a shape only the caller knows
        """Return a deep copy of the value at ``path`` (KeyError/IndexError if absent).

        Example: ``TomlDocument("a = [1, 2]").get(("a", 1)) == 2``.
        """
        _check_indices(path)
        node = _lookup(self._data, path)
        if node is MISSING:
            _raise_absent(self._data, path)
        return copy.deepcopy(node)

    def set(self, path: Path, value: object) -> None:
        """Set ``path`` to ``value``, editing only the text that must change.

        Missing intermediate tables are created; int segments must index existing
        array elements. Dict values are applied as a recursive per-key diff.
        Tuples are treated as lists.
        """
        _check_indices(path)
        value = _normalise(value)
        dumps_value(value)  # reject unrepresentable values before editing
        if not path:
            if not is_dict(value):
                raise TypeError("the document root must be set to a dict")
            expected: object = copy.deepcopy(value)
        else:
            expected = copy.deepcopy(self._data)
            _apply(expected, path, value)
        self._transaction(lambda: self._set(path, value), expected)

    def delete(self, path: Path) -> None:
        """Remove ``path`` (its key line(s), table sections or array element)."""
        if not path:
            raise ValueError("cannot delete the document root")
        _check_indices(path)
        if _lookup(self._data, path) is MISSING:
            raise KeyError(path)
        expected = copy.deepcopy(self._data)
        _apply(expected, path)
        self._transaction(lambda: self._delete(path), expected)

    # --- internals ---------------------------------------------------------------------

    def _index(self) -> None:
        scanner = _Scanner(self._text)
        scanner.scan()
        self._root, self._sections, self._entries = scanner.root, scanner.sections, scanner.entries
        self._nl = "\r\n" if "\r\n" in self._text else "\n"

    def _splice(self, start: int, end: int, new: str) -> None:
        self._text = self._text[:start] + new + self._text[end:]
        try:
            self._data = tomllib.loads(self._text)
        except tomllib.TOMLDecodeError as err:
            raise RuntimeError(_ERROR) from err
        self._index()

    def _transaction(self, edit: Callable[[], None], expected: object) -> None:
        text, data = self._text, self._data
        try:
            edit()
            if not _same(self._data, expected):
                raise RuntimeError(_ERROR)
        except BaseException:
            self._text, self._data = text, data
            self._index()
            raise

    def _find(self, path: Path) -> tuple[_Value, _Value | None, int] | None:
        """Locate the value span at ``path`` with its container (None at entry level)."""
        for entry in self._entries:
            if path[: len(entry.abs)] != entry.abs:
                continue
            node, parent, index = entry.value, None, -1
            rest = path[len(entry.abs) :]
            while rest:
                for i, item in enumerate(node.items):
                    keys = item.keys or (i,)
                    if rest[: len(keys)] == keys:
                        node, parent, index, rest = item.value, node, i, rest[len(keys) :]
                        break
                else:
                    return None
            return node, parent, index
        return None

    def _enclosing(self, path: Path) -> _Entry | None:
        return next((e for e in self._entries if path[: len(e.abs)] == e.abs), None)

    def _rerender(self, entry: _Entry, path: Path, value: object | Missing = MISSING) -> None:
        whole = copy.deepcopy(self._data)
        _apply(whole, path, value)
        new = dumps_value(_lookup(whole, entry.abs))
        self._splice(entry.value.start, entry.value.end, new)

    def _set(self, path: Path, value: object) -> None:
        old = _lookup(self._data, path)
        if old is MISSING:
            return self._insert(path, value)
        if _same(old, value):
            return None
        found = self._find(path)
        if is_dict(old) and is_dict(value):
            for k in [k for k in old if k not in value]:
                self._delete((*path, _str_key(k)))
            for k, v in value.items():
                self._set((*path, _str_key(k)), v)  # _delete re-creates a table that vanished
            return None
        if is_list(old) and is_list(value):
            olds, news = old, value
            if found is not None and len(olds) == len(news):
                for i, v in enumerate(news):
                    self._set((*path, i), v)
                return None
            if found is None and news and all(is_dict(v) for v in news):
                for i in range(min(len(olds), len(news))):
                    self._set((*path, i), news[i])
                for i in reversed(range(len(news), len(olds))):
                    self._delete((*path, i))
                if len(news) > len(olds):
                    self._add_sections(path, news[len(olds) :], aot=True)
                return None
        if found is not None:
            node = found[0]
            new_text = dumps_value(value)
            was_literal = self._text[node.start] == "'" and not self._text.startswith(
                "'''", node.start
            )
            if was_literal and isinstance(value, str) and "'" not in value and _escape_free(value):
                new_text = f"'{value}'"  # keep the single-line literal quoting style
            return self._splice(node.start, node.end, new_text)
        if isinstance(path[-1], int):  # an array-of-tables element given a non-table
            raise TypeError("array-of-tables element must be a table")
        self._delete(path)
        self._insert(path, value)
        return None

    def _delete(self, path: Path) -> None:
        container = _lookup(self._data, path[:-1])
        found = self._find(path)
        if found is not None:
            node, parent, index = found
            if parent is None:
                entry = next(e for e in self._entries if e.value is node)
                self._splice(entry.line_start, entry.line_end, "")
            else:
                self._remove_item(parent, index)
        elif (entry := self._enclosing(path)) is not None:
            self._rerender(entry, path)
        else:
            under = sorted(
                (start, end) for start, end, p in self._pieces() if p[: len(path)] == path
            )
            merged: list[list[int]] = []
            for start, end in under:
                if merged and not self._text[merged[-1][1] : start].strip():
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            text = self._text
            for start, end in reversed(merged):
                text = text[:start] + text[end:]
            self._splice(0, len(self._text), text)
        if _lookup(self._data, path[:-1]) is MISSING:  # an implicit table/array vanished
            self._insert(path[:-1], [] if isinstance(container, list) else {})

    def _pieces(self) -> list[tuple[int, int, Path]]:
        """Spans of every section (header plus own key lines) and key line, with paths."""
        pieces = [(s.line_start, s.own_end, s.abs) for s in self._sections if s is not self._root]
        return pieces + [(e.line_start, e.line_end, e.abs) for e in self._entries]

    def _remove_item(self, parent: _Value, index: int) -> None:
        items, text = parent.items, self._text
        if len(items) == 1:
            return self._splice(parent.start, parent.end, "[]" if parent.kind == "array" else "{}")
        item = items[index]
        if item.comma is None:
            return self._splice(items[index - 1].value.end, item.value.end, "")
        after = item.comma + 1
        while after < len(text) and text[after] in " \t":
            after += 1
        line_start = text.rfind("\n", 0, item.start) + 1
        if not text[line_start : item.start].strip() and (
            after >= len(text) or text[after] in "#\r\n"
        ):
            scanner = _Scanner(text)
            scanner.pos = after
            scanner.line_tail()
            return self._splice(line_start, scanner.pos, "")
        return self._splice(item.start, after, "")

    def _insert(self, path: Path, value: object) -> None:
        depth = len(path) - 1
        while depth > 0 and _lookup(self._data, path[:depth]) is MISSING:
            depth -= 1
        parent, key = path[:depth], path[depth]
        for seg in reversed(path[depth + 1 :]):
            value = {seg: value}
        found = self._find(parent)
        if found is not None:
            node = found[0]
            pair = f"{_key(key)} = {dumps_value(value)}"
            if not node.items:
                return self._splice(node.start, node.end, "{ " + pair + " }")
            end = node.items[-1].value.end
            return self._splice(end, end, ", " + pair)
        if (entry := self._enclosing(parent)) is not None:
            return self._rerender(entry, (*parent, key), value)
        section, prefix = self._owner(parent)
        if is_dict(value) and not prefix:
            return self._add_sections((*parent, key), [value], aot=False)
        return self._insert_line(section, (*prefix, _str_key(key)), value)

    def _owner(self, table: Path) -> tuple[_Section, tuple[str, ...]]:
        """The section whose key lines define ``table``, and the dotted prefix within it."""
        for section in self._sections:
            if section.abs == table:
                return section, ()
        for e in self._entries:
            if len(e.section.abs) < len(table) and e.abs[: len(table)] == table:
                return e.section, tuple(_str_key(k) for k in table[len(e.section.abs) :])
        section, prefix = self._owner(table[:-1])
        return section, (*prefix, _str_key(table[-1]))

    def _insert_line(self, section: _Section, keys: tuple[str, ...], value: object) -> None:
        mine = [e for e in self._entries if e.section is section]
        under = (*section.abs, *keys[:-1])
        siblings = [e for e in mine if len(keys) > 1 and e.abs[: len(under)] == under]
        if ref := (siblings or mine):
            pos, indent = ref[-1].line_end, ref[-1].indent
        elif section is self._root:
            pos = self._sections[1].line_start if len(self._sections) > 1 else len(self._text)
            indent = ""
        else:
            pos, indent = section.own_end, section.indent
        lead = self._nl if pos and self._text[pos - 1] != "\n" else ""
        line = f"{indent}{'.'.join(_key(k) for k in keys)} = {dumps_value(value)}{self._nl}"
        self._splice(pos, pos, lead + line)

    def _add_sections(self, path: Path, tables: list[object], *, aot: bool) -> None:
        """Insert ``[path]`` (or ``[[path]]``) sections at the end of the parent's region."""
        region = path if aot else path[:-1]
        ends = [end for _, end, p in self._pieces() if p[: len(region)] == region]
        pos = max(ends, default=len(self._text))
        names = [k for k in path if isinstance(k, str)]
        body = self._nl.join(self._render(names, t, aot=aot) for t in tables)
        lead = (self._nl if self._text[pos - 1] != "\n" else "") + self._nl if pos else ""
        self._splice(pos, pos, lead + body)

    def _render(self, names: list[str], table: object, *, aot: bool = False) -> str:
        nl = self._nl
        if not is_dict(table):
            raise TypeError(f"a table section needs a dict, not {type(table).__name__}")
        plain = [_str_key(k) for k, v in table.items() if not is_dict(v)]
        subs = [_str_key(k) for k, v in table.items() if is_dict(v)]
        blocks: list[str] = []
        if plain or not subs or aot:
            header = ".".join(_key(n) for n in names)
            lines = [f"[[{header}]]" if aot else f"[{header}]"]
            lines += [f"{_key(k)} = {dumps_value(table[k])}" for k in plain]
            blocks.append(nl.join(lines) + nl)
        blocks += [self._render([*names, k], table[k]) for k in subs]
        return nl.join(blocks)


def _escape_free(value: str) -> bool:
    return all(_escape(c) == c for c in value)


def _normalise(value: object) -> object:
    """Copy ``value`` with tuples turned into lists (what tomllib reads back)."""
    if is_array(value):
        return [_normalise(v) for v in value]
    if is_dict(value):
        return {k: _normalise(v) for k, v in value.items()}
    return value


def _str_key(key: object) -> str:
    """``key`` as a table key; ``TypeError`` (as :func:`dumps_value` raises) for any other."""
    if not isinstance(key, str):
        raise TypeError(f"TOML keys must be str, not {type(key).__name__}")
    return key


def _raise_absent(data: object, path: Path) -> NoReturn:
    """Raise the ``KeyError`` / ``IndexError`` that indexing ``data`` along ``path`` raises."""
    node = data
    for seg in path:
        if is_dict(node):
            if seg not in node:
                raise KeyError(seg)
            node = node[seg]
        elif is_list(node) and isinstance(seg, int):
            if seg >= len(node):
                raise IndexError(f"array index {seg} out of range")
            node = node[seg]
        else:
            raise TypeError(f"cannot address {seg!r} in {type(node).__name__}")
    raise KeyError(path)  # only reached when called for a path that exists


def _check_indices(path: Path) -> None:
    for seg in path:
        if isinstance(seg, int) and seg < 0:
            raise IndexError(f"negative array index {seg} in {path!r}")
