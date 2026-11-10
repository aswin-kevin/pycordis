"""Config value interpolation: ``${env:NAME}``, ``${env:NAME:-default}``, ``${file:path}``.

Interpolation is applied when a fiber's config is resolved, never at parse
time, so the raw (un-expanded) config is what gets written back to the file.

* ``${env:NAME}`` — the environment variable ``NAME``; unset is an error.
* ``${env:NAME:-default}`` — ``default`` when ``NAME`` is unset *or empty*
  (shell semantics). The default runs up to the first ``}``; no nesting.
* ``${file:path}`` — the UTF-8 contents of ``path`` (relative paths resolve
  against the config's source directory, or the working directory when none
  is given) with one trailing newline removed; files over 1 MiB are rejected.
  Only regular files are read (symlinks are followed; the type is checked on
  the opened descriptor, so it cannot change between check and read); a
  UTF-8 BOM is dropped.
* ``$${`` — a literal ``${``.

All problems in a value are collected into one :class:`ValidationError` whose
issues carry the path of the offending string within the config.

Secret tracking (used by the loader's write-back): an :class:`Expansions`
passed to :func:`expand` records what each interpolated string expanded to and
which token values count as secrets: every non-empty ``${file:...}`` value and
every ``${env:...}`` value of at least :data:`MIN_SECRET_LENGTH` characters.
Shorter environment values are too likely to occur by chance (``"4"``,
``"true"``) to refuse writes containing them; they are only put back as
placeholders on an exact match (see :func:`retemplate`).
"""

from __future__ import annotations

import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from pathlib import Path
from typing import final

from pycordis._guards import is_array, is_dict
from pycordis.errors import Issue, ValidationError

MAX_FILE_SIZE = 1024 * 1024
"""Largest file ``${file:...}`` reads (1 MiB)."""

MIN_SECRET_LENGTH = 8
"""Shortest ``${env:...}`` value treated as a secret by :func:`leaks` (file values always are)."""

_Loc = tuple[str | int, ...]


@final
@dataclass
class Expansions:
    """What the interpolations of a config expanded to; filled by :func:`expand`.

    ``secrets`` holds the token values counted as secrets (see the module
    docstring); ``leaves`` maps each interpolated string to the expansions it
    was seen with. Both only grow, so earlier (rotated) values stay known.
    """

    secrets: set[str] = field(default_factory=set[str])
    leaves: dict[str, set[str]] = field(default_factory=dict[str, set[str]])

    def update(self, other: Expansions) -> None:
        """Merge ``other`` into this record."""
        self.secrets |= other.secrets
        for text, values in other.leaves.items():
            self.leaves.setdefault(text, set()).update(values)


_TOKEN = re.compile(r"\$\$\{|\$\{([^}]*)\}|\$\{")


def expand(
    value: object,
    source_dir: Path | None = None,
    *,
    path: _Loc = (),
    track: Expansions | None = None,
) -> object:
    """Return a copy of ``value`` with interpolations replaced in every string.

    Dicts, lists and tuples are walked recursively; other values are returned
    as-is. ``path`` is prefixed to issue paths. ``track`` records the
    expansions (also those made before an error). Raises
    :class:`ValidationError` listing every failed interpolation.

    Example: ``expand({"key": "${env:API_KEY}"})`` gives the key's value.
    """
    issues: list[Issue] = []
    result = _walk(value, path, source_dir, issues, track)
    if issues:
        raise ValidationError(issues)
    return result


def _walk(
    value: object, path: _Loc, base: Path | None, issues: list[Issue], track: Expansions | None
) -> object:
    if isinstance(value, str):
        result = _expand_str(value, path, base, issues, track)
        if track is not None and "${" in value:
            track.leaves.setdefault(value, set()).add(result)
        return result
    if is_dict(value):
        items = value.items()
        return {k: _walk(v, (*path, _key(k)), base, issues, track) for k, v in items}
    if is_array(value):
        seq = [_walk(v, (*path, i), base, issues, track) for i, v in enumerate(value)]
        return tuple(seq) if isinstance(value, tuple) else seq
    return value


def _key(key: object) -> str | int:
    return key if isinstance(key, int) else str(key)


def _expand_str(
    text: str, path: _Loc, base: Path | None, issues: list[Issue], track: Expansions | None
) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token == "$${":
            return "${"
        body = match.group(1)
        if body is None:
            issues.append(Issue(f'unterminated interpolation in "{text}"', path))
            return token
        kind, _, arg = body.partition(":")
        name, sep, default = arg.partition(":-")
        if kind == "env" and name:
            current = os.environ.get(name)
            if sep and not current:
                return default
            if current is not None:
                if track is not None and len(current) >= MIN_SECRET_LENGTH:
                    track.secrets.add(current)
                return current
            issues.append(Issue(f"environment variable {name} is not set", path))
        elif kind == "file" and arg:
            failed = len(issues)
            content = _read_file(arg, base, path, issues)
            if track is not None and content and len(issues) == failed:
                track.secrets.add(content)
            return content
        else:
            issues.append(Issue(f'unknown interpolation "{token}"', path))
        return token

    return _TOKEN.sub(replace, text)


def _read_file(name: str, base: Path | None, path: _Loc, issues: list[Issue]) -> str:
    target = Path(name)
    if not target.is_absolute() and base is not None:
        target = base / target
    try:
        data = _read_regular(target)
    except OSError as error:
        issues.append(Issue(f"cannot read file {name}: {error.strerror or error}", path))
        return ""
    if data is None:  # never block on FIFOs, devices or /dev/stdin
        issues.append(Issue(f"cannot read file {name}: not a regular file", path))
        return ""
    if len(data) > MAX_FILE_SIZE:
        issues.append(Issue(f"file {name} exceeds the 1 MiB limit", path))
        return ""
    try:
        content = data.decode("utf-8-sig")  # drops a BOM
    except UnicodeDecodeError as error:
        issues.append(Issue(f"file {name} is not valid UTF-8: {error.reason}", path))
        return ""
    if content.endswith("\r\n"):
        return content[:-2]
    return content.removesuffix("\n")


def _read_regular(target: Path) -> bytes | None:
    """Up to ``MAX_FILE_SIZE + 1`` bytes of ``target``; ``None`` when it is not a regular file.

    A path that is not a regular file is never opened: opening a device can
    have side effects (a tty becoming the controlling terminal, a serial port
    resetting its board, a watchdog arming). The type is checked again on the
    opened descriptor, so a path swapped for a FIFO or device after the first
    check is not read either; the open is non-blocking and ``O_NOCTTY`` for
    that case. Raises ``OSError`` when it cannot be read (missing, no
    permission).
    """
    if not stat.S_ISREG(os.stat(target).st_mode):
        return None
    flags = os.O_RDONLY
    if sys.platform == "win32":
        flags |= os.O_BINARY  # no CRLF translation, no 0x1A end-of-file
    else:
        flags |= os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC
    fd = os.open(target, flags)
    try:
        regular = stat.S_ISREG(os.fstat(fd).st_mode)
        if regular and sys.platform != "win32":
            os.set_blocking(fd, True)  # a regular file: plain blocking reads
    except BaseException:
        os.close(fd)
        raise
    if not regular:
        os.close(fd)
        return None
    with os.fdopen(fd, "rb") as file:  # takes ownership of fd and closes it
        return file.read(MAX_FILE_SIZE + 1)


def retemplate(
    value: object,
    raw: object,
    source_dir: Path | None = None,
    known: Mapping[str, Set[str]] | None = None,
) -> object:
    """Return ``value`` with expanded leaves put back as the placeholders in ``raw``.

    ``value`` and ``raw`` are walked together (dict keys, list and tuple
    indexes); wherever the ``raw`` leaf is a string with an interpolation and
    ``value`` holds exactly its expansion, the ``raw`` string is used instead.
    ``known`` maps interpolated strings to earlier expansions (see
    :attr:`Expansions.leaves`) that also count, so a value expanded before a
    secret was rotated still gets its placeholder back. Other leaves are kept.
    A container is rebuilt (as a plain dict / list / tuple) only when one of
    its leaves changed, else returned as is; leaves are not copied. Example::

        # with API_KEY=sk-123 in the environment
        retemplate({"key": "sk-123"}, {"key": "${env:API_KEY}"}) == {"key": "${env:API_KEY}"}
    """
    if isinstance(raw, str):
        if isinstance(value, str) and value != raw and "${" in raw:
            if known is not None and value in known.get(raw, ()):
                return raw
            try:
                expanded = expand(raw, source_dir)
            except ValidationError:
                return value
            if expanded == value:
                return raw
        return value
    if is_dict(value) and is_dict(raw):
        items = value.items()
        mapping = {k: retemplate(v, raw[k], source_dir, known) if k in raw else v for k, v in items}
        same = all(mapping[k] is v for k, v in items)
        return value if same and type(value) is dict else mapping
    if is_array(value) and is_array(raw):
        seq = [
            retemplate(v, raw[i], source_dir, known) if i < len(raw) else v
            for i, v in enumerate(value)
        ]
        unchanged = all(a is b for a, b in zip(seq, value, strict=True))
        if unchanged and type(value) in (list, tuple):
            return value
        return tuple(seq) if isinstance(value, tuple) else seq
    return value


def secrets(raw: object, source_dir: Path | None = None) -> set[str]:
    """The secret values the ``${env:...}`` / ``${file:...}`` tokens of ``raw`` expand to now.

    Secrets are the non-empty ``${file:...}`` contents and the ``${env:...}``
    values of at least :data:`MIN_SECRET_LENGTH` characters. A default
    (``${env:NAME:-default}`` with ``NAME`` unset or empty) is written in the
    file already and is not a secret; tokens that fail to expand are skipped.
    Example: ``secrets({"key": "${file:token.txt}"}, Path("conf"))`` is ``{<token>}``.
    """
    track = Expansions()
    _walk(raw, (), source_dir, [], track)
    return track.secrets


def leaks(value: object, raw: object, found: Set[str]) -> bool:
    """Whether ``value`` would persist one of ``found`` (see :func:`secrets`).

    A string leaks when it contains a secret; any other leaf (a number, say)
    when its ``str()`` equals one (an integral float also by its integer form).
    A leaf identical to the ``raw`` leaf at the same path was in the file
    already and is not counted. Example: ``leaks({"k": "sk-123"}, {}, {"sk-123"}) is True``.
    """
    if isinstance(value, str):
        return value != raw and any(secret in value for secret in found)
    if is_dict(value):
        old: dict[object, object] = raw if is_dict(raw) else {}
        return any(leaks(k, None, found) or leaks(v, old.get(k), found) for k, v in value.items())
    if is_array(value):
        old_seq: Sequence[object] = raw if is_array(raw) else []
        return any(
            leaks(v, old_seq[i] if i < len(old_seq) else None, found) for i, v in enumerate(value)
        )
    if value is None:
        return False
    forms = {str(value)}
    if isinstance(value, float) and value.is_integer():
        forms.add(str(int(value)))  # 12345678.0 also matches "12345678"
    if forms.isdisjoint(found):
        return False
    return not (type(raw) is type(value) and raw == value)
