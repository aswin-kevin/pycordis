"""JSON Schema export for pycordis configs (spec §3.9).

:func:`config_json_schema` turns a dataclass ``Config`` into a JSON Schema
(draft 2020-12) and :func:`loader_json_schema` describes a whole loader config
file, typing each entry's ``config`` from its plugin's dataclass ``Config``.
Point an editor at the output for autocompletion and validation, or run
``pycordis schema``.

The schema mirrors the runtime validator of :mod:`pycordis.schema`: dataclass
fields (required unless they have a default), ``bool`` / ``int`` / ``float`` /
``str`` (``bool`` is not an integer, an integer is a number), ``None``,
``Literal``, unions and ``Optional``, ``list[T]``, ``tuple[T, ...]`` (an array),
``dict[str, T]`` / ``Mapping[str, T]``, ``Enum`` (by value), nested dataclasses
and ``Volatile[T]`` (as ``T``). Unknown keys are rejected
(``additionalProperties: false``), like the runtime's ``unknown key`` issue.

Intentional differences: JSON Schema cannot tell ``1.0`` from ``1``, so a float
written as ``1.0`` satisfies ``integer`` here but not at runtime; an annotation
the runtime refuses (a non-variadic ``tuple``, ``set``, a custom class, ...)
becomes ``{}`` (anything) with a ``description`` instead of an error; and
interpolation placeholders (``${env:X}``) are plain strings to the schema, as
they are to the validator, which sees the already-expanded value.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeGuard,
    Union,
    final,
    get_args,
    get_origin,
    get_type_hints,
)

from pycordis._guards import is_array, is_mapping
from pycordis.loader.entry import Entry
from pycordis.loader.resolve import import_plugin
from pycordis.registry import plugin_config_type
from pycordis.volatile import Volatile

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = ["config_json_schema", "loader_json_schema"]

type Schema = dict[str, object]

DRAFT = "https://json-schema.org/draft/2020-12/schema"
_PRIMITIVES: dict[object, str] = {bool: "boolean", int: "integer", float: "number", str: "string"}
_LITERAL_TYPES = (bool, int, str, type(None))


def config_json_schema(schema: object) -> Schema:
    """The JSON Schema (draft 2020-12) of a dataclass ``Config`` type.

    Nested dataclasses go to ``$defs`` and are referenced with ``$ref``, so
    self-referential dataclasses terminate. A schema that is not a dataclass
    (a pydantic model, a callable, ``None``) cannot be described: the result
    accepts anything and says so in its ``description``. Example::

        @dataclass
        class Chat:
            model: str
            retries: int = 3

        config_json_schema(Chat)["required"] == ["model"]
    """
    if not _is_dataclass(schema):
        reason = "no configuration schema" if schema is None else "not a dataclass config"
        return {"$schema": DRAFT, "description": f"Any value ({reason})."}
    builder = _Builder(root=schema)
    root = builder.dataclass_schema(schema)
    result: Schema = {"$schema": DRAFT, **root}
    if builder.defs:
        result["$defs"] = dict(sorted(builder.defs.items()))
    return result


def loader_json_schema(
    entries: Sequence[Entry],
    allow: Sequence[str] = ("*",),
    *,
    overlay: bool = False,
    skipped: Callable[[str, Exception], None] | None = None,
) -> Schema:
    """The JSON Schema of a loader config file (spec §6.6).

    Describes the top-level ``plugins`` array, the ``[loader]`` table, an
    optional ``$schema`` string (so a file can point its editor at this schema;
    the loader ignores it) and every entry key. For each enabled ``use`` found
    in ``entries`` (a parsed tree, see :func:`~pycordis.loader.load_tree`)
    whose plugin has a dataclass ``Config``, the entry schema gains
    ``if use == X then config: <schema>``.
    Plugins are imported to find their ``Config``, honouring ``allow``; one
    that cannot be imported (or whose schema cannot be built) is reported
    through ``skipped(use, error)`` (default: a note on stderr) and left
    untyped. Disabled entries are not imported.

    Use the default schema for the root file and for included files, and
    ``overlay=True`` for profile overlay files (``app.dev.toml``). An overlay
    entry may be partial: an entry whose ``id`` names a base entry needs no
    ``use`` / ``group`` / ``include``, and its ``config`` is deep-merged into
    the base one, so the overlay schema types it from that entry's plugin with
    no required fields. Entries of ``entries`` are the *base* tree, and (as
    overlays may enable a disabled entry) disabled ones are imported too. New
    overlay entries (unknown ``id``) are typed only structurally.

    ``[loader]`` is only valid in the root file (and the root overlay); the
    loader rejects it in included files and their overlays, which the schema
    cannot tell apart from the root file. Example::

        entries, options = load_tree(Path("pycordis.toml"))
        schema = loader_json_schema(entries, options.allow)
        overlay_schema = loader_json_schema(entries, options.allow, overlay=True)
    """
    report = skipped if skipped is not None else _note_skipped
    builder = _Builder(reserved=_STATIC_DEFS, partial=overlay)
    entry = _entry_schema(overlay)
    if overlay:
        configs = _configs_by_id(entries, allow, builder, report)
        entry["allOf"] = [_when_id(name, config) for name, config in configs.items()]
    else:
        typed = _typed_configs(entries, allow, builder, report)
        entry["allOf"] = [_when_use(use, config) for use, config in typed.items()]
    entry_def = "overlay_entry" if overlay else "entry"
    return {
        "$schema": DRAFT,
        "title": "pycordis config overlay" if overlay else "pycordis config",
        "type": "object",
        "properties": {
            "$schema": {
                "type": "string",
                "description": "The JSON Schema of this file; ignored by the loader.",
            },
            "loader": _loader_table_schema(),
            "plugins": {"type": "array", "items": {"$ref": f"#/$defs/{entry_def}"}},
        },
        "additionalProperties": False,
        "$defs": dict(sorted({**builder.defs, **_static_defs(), entry_def: entry}.items())),
    }


def _note_skipped(use: str, error: Exception) -> None:
    print(f"pycordis: schema: {use}: config left untyped ({error})", file=sys.stderr)


def _config_type_of(
    use: str, allow: Sequence[str], report: Callable[[str, Exception], None]
) -> type[DataclassInstance] | None:
    """The dataclass ``Config`` of plugin ``use``; failures are reported and give ``None``."""
    try:
        config_type = plugin_config_type(import_plugin(use, allow))
    except Exception as error:
        report(use, error)
        return None
    return config_type if _is_dataclass(config_type) else None


def _typed_configs(
    entries: Sequence[Entry],
    allow: Sequence[str],
    builder: _Builder,
    report: Callable[[str, Exception], None],
) -> dict[str, Schema]:
    """``use`` -> config schema for each distinct plugin with a dataclass ``Config``."""
    typed: dict[str, Schema] = {}
    seen: set[str] = set()
    for entry in _walk(entries, include_disabled=False):
        use = entry.use
        if use is None or use in seen:
            continue
        seen.add(use)
        if (config_type := _config_type_of(use, allow, report)) is not None:
            typed[use] = _build(builder, config_type, use, report)
    return typed


def _configs_by_id(
    entries: Sequence[Entry],
    allow: Sequence[str],
    builder: _Builder,
    report: Callable[[str, Exception], None],
) -> dict[str, Schema]:
    """Entry id -> config schema, for base entries with an explicit id (what overlays match)."""
    by_use: dict[str, Schema | None] = {}
    configs: dict[str, Schema] = {}
    for entry in _walk(entries, include_disabled=True):
        use = entry.use
        if use is None or not entry.explicit_id:
            continue
        if use not in by_use:
            config_type = _config_type_of(use, allow, report)
            by_use[use] = None if config_type is None else _build(builder, config_type, use, report)
        schema = by_use[use]
        if schema is not None:
            configs[entry.id] = schema
    return configs


def _build(
    builder: _Builder,
    config_type: type[DataclassInstance],
    use: str,
    report: Callable[[str, Exception], None],
) -> Schema:
    try:
        return builder.dataclass_ref(config_type)
    except Exception as error:
        report(use, error)
        return {}


def _walk(entries: Sequence[Entry], *, include_disabled: bool) -> Iterator[Entry]:
    for entry in entries:
        if entry.disabled and not include_disabled:
            continue
        yield entry
        yield from _walk(entry.children, include_disabled=include_disabled)


def _when_use(use: str, config: Schema) -> Schema:
    return {
        "if": {"properties": {"use": {"const": use}}, "required": ["use"]},
        "then": {"properties": {"config": config}},
    }


def _when_id(name: str, config: Schema) -> Schema:
    return {
        "if": {"properties": {"id": {"const": name}}, "required": ["id"]},
        "then": {"properties": {"config": config}},
    }


# -- the loader file format ----------------------------------------------------------

_NUMBER_OR_NULL: Schema = {"type": ["number", "null"], "exclusiveMinimum": 0}
_STRING_LIST: Schema = {"type": "array", "items": {"type": "string"}}
_STATIC_DEFS = frozenset({"entry", "overlay_entry", "restart", "backoff"})


def _loader_table_schema() -> Schema:
    return {
        "type": "object",
        "description": "Options of the loader itself (root config file only).",
        "properties": {
            "readonly": {"type": "boolean"},
            "watch": {"type": "boolean"},
            "allow": {**_STRING_LIST, "description": "Allowed plugin module patterns."},
            "poll_interval": {"type": "number", "exclusiveMinimum": 0},
        },
        "additionalProperties": False,
    }


def _entry_schema(overlay: bool) -> Schema:
    """The entry table; an overlay entry may leave out its kind when it has an ``id``."""
    name: Schema = {"type": "string", "minLength": 1}
    ref = "#/$defs/overlay_entry" if overlay else "#/$defs/entry"
    one_kind: list[object] = [
        {"required": ["use"]},
        {"required": ["group"]},
        {"required": ["include"]},
    ]
    kinds: Schema = {"oneOf": one_kind}
    kindless: Schema = {"required": ["id"], "not": {"anyOf": one_kind}}
    rules: Schema = (
        {"oneOf": [*one_kind, kindless]}
        if overlay
        else {**kinds, "dependentRequired": {"config": ["use"], "plugins": ["group"]}}
    )
    return {
        "type": "object",
        "description": "A plugin ('use'), a group or an include.",
        "properties": {
            "id": name,
            "use": {**name, "description": "'module:attr' or an entry-point name."},
            "group": name,
            "include": name,
            "config": {"description": "The plugin's config ('use' entries only)."},
            "disabled": {"type": "boolean"},
            "isolate": {
                "oneOf": [
                    {**_STRING_LIST, "uniqueItems": True},
                    {"type": "object", "additionalProperties": {"type": "string"}},
                ]
            },
            "intercept": {"type": "object"},
            "inject": _STRING_LIST,
            "plugins": {"type": "array", "items": {"$ref": ref}},
            "restart": {"$ref": "#/$defs/restart"},
            "load_timeout": _NUMBER_OR_NULL,
            "dispose_timeout": _NUMBER_OR_NULL,
        },
        "additionalProperties": False,
        **rules,
    }


def _static_defs() -> dict[str, object]:
    return {
        "restart": {
            "type": "object",
            "description": "Restart policy of a failed plugin.",
            "properties": {
                "mode": {"enum": ["never", "on-failure"]},
                "max_restarts": {"type": "integer", "minimum": 0},
                "window": {"type": "number", "exclusiveMinimum": 0},
                "backoff": {"$ref": "#/$defs/backoff"},
            },
            "additionalProperties": False,
        },
        "backoff": {
            "type": "object",
            "properties": {
                "initial": {"type": "number", "minimum": 0},
                "factor": {"type": "number", "minimum": 1},
                "max": {"type": "number", "minimum": 0},
                "jitter": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "additionalProperties": False,
        },
    }


# -- dataclass types -----------------------------------------------------------------


def _is_dataclass(obj: object) -> TypeGuard[type[DataclassInstance]]:
    return isinstance(obj, type) and dataclasses.is_dataclass(obj)


@final
class _Builder:
    """Builds schemas for annotations, collecting dataclasses into ``defs``."""

    def __init__(
        self,
        root: type | None = None,
        reserved: frozenset[str] = frozenset(),
        partial: bool = False,
    ) -> None:
        self.defs: dict[str, object] = {}
        self._root = root
        self._partial = partial  # no required fields: a config an overlay merges
        self._names: dict[type, str] = {}
        self._taken = set(reserved)

    def dataclass_ref(self, cls: type[DataclassInstance]) -> Schema:
        """A ``$ref`` to ``cls`` (``#`` for the root type), defining it on first use."""
        if cls is self._root:
            return {"$ref": "#"}
        if cls not in self._names:
            name = self._unique_name(cls.__name__)
            self._names[cls] = name
            self.defs[name] = self.dataclass_schema(cls)
        return {"$ref": f"#/$defs/{self._names[cls]}"}

    def _unique_name(self, base: str) -> str:
        name, n = base, 1
        while name in self._taken:
            n += 1
            name = f"{base}_{n}"
        self._taken.add(name)
        return name

    def dataclass_schema(self, cls: type[DataclassInstance]) -> Schema:
        """The object schema of dataclass ``cls``."""
        hints = get_type_hints(cls, include_extras=True)
        properties: dict[str, object] = {}
        required: list[str] = []
        for f in dataclasses.fields(cls):
            if not f.init:
                continue
            prop = self.type_schema(hints.get(f.name, Any))
            _add_default(prop, f)
            properties[f.name] = prop
            if not self._partial and _is_required(f):
                required.append(f.name)
        schema: Schema = {"title": cls.__name__, "type": "object"}
        doc = cls.__doc__
        if doc and not doc.startswith(f"{cls.__name__}("):  # not the generated signature
            schema["description"] = doc.strip()
        schema["properties"] = properties
        if required:
            schema["required"] = required
        schema["additionalProperties"] = False
        return schema

    def type_schema(self, tp: object) -> Schema:
        """The schema of annotation ``tp``; ``{}`` plus a description when unsupported."""
        if tp is Any:
            return {}
        if tp is None or tp is type(None):
            return {"type": "null"}
        if tp in _PRIMITIVES:
            return {"type": _PRIMITIVES[tp]}
        if tp is list:
            return {"type": "array"}
        if tp is dict:
            return {"type": "object"}
        if get_origin(tp) is not None:
            return self._generic(tp)
        if _is_dataclass(tp):
            return self.dataclass_ref(tp)
        if _is_enum_type(tp):
            return {"enum": [member.value for member in tp]}
        return _unsupported(tp)

    def _generic(self, tp: object) -> Schema:
        origin = get_origin(tp)
        args = get_args(tp)
        if origin is Literal:
            if all(type(a) in _LITERAL_TYPES for a in args):
                return {"enum": list(args)}
            return _unsupported(tp)
        if origin in (Union, types.UnionType):
            return {"anyOf": [self.type_schema(a) for a in args]}
        if origin is Volatile:
            return self.type_schema(args[0])
        if origin is list:
            return {"type": "array", "items": self.type_schema(args[0])}
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": self.type_schema(args[0])}
        if origin in (dict, Mapping) and len(args) == 2:
            return {"type": "object", "additionalProperties": self.type_schema(args[1])}
        return _unsupported(tp)


def _is_enum_type(tp: object) -> TypeGuard[type[Enum]]:
    return isinstance(tp, type) and issubclass(tp, Enum)


def _is_required(f: dataclasses.Field[Any]) -> bool:
    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


def _unsupported(tp: object) -> Schema:
    return {"description": f"Unsupported annotation {tp!r}: any value is accepted here."}


def _add_default(prop: Schema, f: dataclasses.Field[Any]) -> None:
    """Record the field's default in ``prop`` when it is expressible in JSON."""
    if f.default is not dataclasses.MISSING:
        value: object = f.default
    elif f.default_factory is not dataclasses.MISSING:
        value = f.default_factory()
    else:
        return
    ok, plain = _plain(value)
    if ok:
        prop["default"] = plain
        return
    note = f"Default {value!r} cannot be written in JSON."
    prop["description"] = f"{prop['description']} {note}" if "description" in prop else note


def _plain(value: object) -> tuple[bool, object]:
    """``value`` as plain JSON data, and whether that is possible."""
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        return False, None  # JSON has no NaN / Infinity
    if value is None or isinstance(value, (bool, int, float, str)):
        return True, value
    if is_array(value):
        items = [_plain(v) for v in value]
        return all(ok for ok, _ in items), [v for _, v in items]
    if is_mapping(value):
        table = value
        pairs = {k: _plain(v) for k, v in table.items() if isinstance(k, str)}
        ok = len(pairs) == len(table) and all(good for good, _ in pairs.values())
        return ok, {k: v for k, (_, v) in pairs.items()}
    return False, None
