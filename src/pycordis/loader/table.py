"""Validation of loader config entry tables (spec §6.6), shared by base files and overlays.

Internal — the parsing entry points are :func:`~pycordis.loader.load_tree` and
:func:`~pycordis.loader.files.parse_entry`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, final

from pycordis._guards import RawTable, is_dict, is_mapping, is_str_list, is_table
from pycordis.supervise import RestartPolicy, check_timeout

type Location = tuple[str | int, ...]
"""A path inside a config file's data, e.g. ``("plugins", 0, "config")``."""

ENTRY_KEYS: frozenset[str] = frozenset(
    {
        *("id", "use", "group", "include", "config"),
        *("disabled", "isolate", "intercept", "inject", "plugins"),
        *("restart", "load_timeout", "dispose_timeout"),
    }
)
type EntryKind = Literal["use", "group", "include"]
"""What an entry is: a plugin (``use``), a group of entries, or an included file."""

ENTRY_KINDS: tuple[EntryKind, ...] = ("use", "group", "include")
"""The keys naming what an entry is; exactly one is set."""


class LoaderConfigError(ValueError):
    """A loader config file (or one of its includes) is missing, unreadable or invalid.

    Every failure of :func:`~pycordis.loader.load_tree` /
    :func:`~pycordis.loader.read_config` except the missing PyYAML
    ``RuntimeError`` is raised as this type, with the file named in the
    message, so callers (hot reload) can catch a single exception type.
    Example: ``except LoaderConfigError as error: logger.error(error)``.
    """


def format_location(location: Location) -> str:
    """``location`` as a readable path.

    Example: ``format_location(("plugins", 0, "plugins", 1)) == "plugins[0].plugins[1]"``.
    """
    out = ""
    for part in location:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else part)
    return out


def strings(value: object, what: str) -> list[str]:
    """``value`` as a list of strings; else :class:`LoaderConfigError` (``what`` must be ...).

    Example: ``strings(["a"], '"inject"') == ["a"]``.
    """
    if not is_str_list(value):
        raise LoaderConfigError(f"{what} must be a list of strings")
    return list(value)


def check_keys(item: RawTable, fail: Callable[[str], LoaderConfigError]) -> None:
    """Raise ``fail('unknown key "..."')`` for a key of ``item`` no entry table may have.

    Example: ``check_keys({"use": "m:p"}, fail)`` passes.
    """
    for key in item:
        if key not in ENTRY_KEYS:
            raise fail(f'unknown key "{key}"')


@final
@dataclass(frozen=True)
class Fields:
    """The parsed value keys of an entry table (defaults for absent keys).

    Example: ``fields({"use": "m:p"}, "use", fail).disabled is False``.
    """

    disabled: bool
    isolate: dict[str, str | None]
    intercept: RawTable
    inject: list[str]
    restart: RawTable | None
    load_timeout: float | None
    dispose_timeout: float | None


def fields(item: RawTable, kind: EntryKind, fail: Callable[[str], LoaderConfigError]) -> Fields:
    """Validate the value keys of the entry table ``item`` of ``kind``; return them parsed.

    ``fail`` builds the error (naming the file and entry) for a message.
    Example: ``fields({"use": "m:p", "inject": ["db"]}, "use", fail).inject == ["db"]``.
    """
    for key in ("id", *ENTRY_KINDS):
        if key in item and (not isinstance(item[key], str) or not item[key]):
            raise fail(f'"{key}" must be a non-empty string')
    if "config" in item and kind != "use":
        raise fail('"config" is only allowed on "use" entries')
    if "plugins" in item and kind != "group":
        raise fail('"plugins" is only allowed on group entries')
    disabled = item.get("disabled", False)
    if not isinstance(disabled, bool):
        raise fail('"disabled" must be a boolean')
    intercept: object = item.get("intercept", {})
    if not is_table(intercept):
        raise fail('"intercept" must be a table')
    try:
        inject = strings(item.get("inject", []), '"inject"')
        restart = restart_table(item.get("restart"))
        load_timeout = check_timeout(item.get("load_timeout"), "load_timeout")
        dispose_timeout = check_timeout(item.get("dispose_timeout"), "dispose_timeout")
    except ValueError as error:
        raise fail(str(error)) from None
    return Fields(
        disabled=disabled,
        isolate=isolate_table(item.get("isolate", {}), fail),
        intercept=dict(intercept),
        inject=inject,
        restart=restart,
        load_timeout=load_timeout,
        dispose_timeout=dispose_timeout,
    )


def restart_table(raw: object) -> RawTable | None:
    """The ``restart`` table, validated as a :class:`RestartPolicy` (``None`` when absent).

    Example: ``restart_table({"mode": "on-failure"}) == {"mode": "on-failure"}``.
    """
    if raw is None:
        return None
    RestartPolicy.from_mapping(raw)  # raises ValueError naming the bad key
    if not is_mapping(raw):  # from_mapping already rejected it; narrows the type
        raise ValueError('"restart" must be a table')
    return {str(key): value for key, value in raw.items()}  # keys checked by from_mapping


def isolate_table(raw: object, fail: Callable[[str], LoaderConfigError]) -> dict[str, str | None]:
    """The ``isolate`` value as names to labels (``None``: a private scope, the list form).

    Example: ``isolate_table(["db"], fail) == {"db": None}``.
    """
    if is_str_list(raw):
        if len(set(raw)) != len(raw):
            raise fail('"isolate" lists a service name more than once')
        return dict.fromkeys(raw)
    if is_dict(raw):
        labels = {key: value for key, value in raw.items() if isinstance(value, str)}
        if len(labels) == len(raw):
            return {str(key): label for key, label in labels.items()}
    raise fail('"isolate" must be a list of service names or a table of labels')
