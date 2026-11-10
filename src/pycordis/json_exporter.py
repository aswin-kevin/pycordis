"""JSON lines log exporter for log pipelines (spec §3.8).

:class:`JsonExporter` writes one JSON object per log message, one per line::

    ctx.logger.exporter(JsonExporter())  # to sys.stderr
    # {"ts": 1700000000.5, "level": "info", "name": "worker", "message": "started",
    #  "plugin": "worker", "error_type": null, "traceback": null}
"""

from __future__ import annotations

import dataclasses
import json
import sys
import traceback
from collections.abc import Mapping
from typing import Literal, TextIO

from pycordis.logger import DEFAULT_MAX_LENGTH, Logger, Message, clip_lines

__all__ = ["JsonExporter"]


class JsonExporter:
    """Write each log message as one JSON object per line (``sys.stderr`` by default).

    Every line has the same keys, in this order:

    - ``ts``: Unix time of the message in seconds (float).
    - ``level``: ``"error"``, ``"warn"``, ``"info"`` or ``"debug"``.
    - ``name``: the logger name.
    - ``message``: the rendered text, printf-style like :class:`ConsoleExporter`
      (arguments JSON can't encode are rendered with ``repr``); an exception
      argument is rendered as ``"<error_type>: <text>"``.
    - ``plugin``: the name of the fiber that logged it, or ``null``.
    - ``error_type``: the type of the first exception argument (module-qualified
      unless built in, e.g. ``"ValueError"``, ``"app.db.DbError"``), or ``null``.
    - ``traceback``: that exception's formatted traceback (without its chain:
      the logger logs a cause as its own message first), or ``null``.

    ``levels`` maps logger names (and ``"default"``) to the most verbose
    :class:`~pycordis.LoggerLevel` written, like :class:`ConsoleExporter`;
    ``max_length`` cuts each line of ``message`` and ``traceback`` (default
    10240 characters, longer lines end in ``...``), so one huge exception
    cannot produce an unbounded record.

    Rendering never raises: a message whose arguments cannot be rendered is
    written as a line whose ``message`` describes the failure (``error_type``
    and ``traceback`` are kept). Text is written as UTF-8 JSON; a line the
    stream cannot encode (lone surrogates, an ASCII-only stream) is written
    again with every non-ASCII character escaped, so it is never dropped. Each
    line is flushed. Other errors of the stream itself (e.g. it was closed) are
    reported by the logger like any exporter failure. Not thread-safe. Example::

        ctx.logger.exporter(JsonExporter(open("app.jsonl", "a"), levels={"default": 3}))
    """

    colors: Literal[False] = False

    def __init__(
        self,
        stream: TextIO | None = None,
        levels: Mapping[str, int] | None = None,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        """Where the lines are written."""
        self.levels: Mapping[str, int] = levels or {}
        """Per-logger-name maximum levels (``"default"`` for the rest)."""
        self.max_length = max_length
        """Per-line length limit of ``message`` and ``traceback`` (longer lines end in ``...``)."""

    def export(self, message: Message) -> None:
        """Write ``message`` as one JSON line and flush.

        Example: ``exporter.export(message)`` (the logger calls it).
        """
        try:
            record = self.record(message)
        except Exception as failure:
            record = self._fallback(message, failure)
        try:
            self.stream.write(_dumps(record, ascii_only=False))
        except UnicodeEncodeError:
            # lone surrogates (e.g. a surrogateescape file name) or a non-UTF
            # stream: escaped JSON is plain ASCII, which every stream can encode
            self.stream.write(_dumps(record, ascii_only=True))
        self.stream.flush()

    def record(self, message: Message) -> dict[str, object]:
        """The JSON object written for ``message`` (keys as documented above).

        Raises when an argument cannot be rendered (:meth:`export` falls back).
        Example: ``JsonExporter().record(message)["level"] == "info"``.
        """
        args = tuple(
            _describe(arg) if isinstance(arg, BaseException) else arg for arg in message.args
        )
        text = Logger.format(self, dataclasses.replace(message, args=args))
        return {**self._base(message), "message": text}

    def _base(self, message: Message) -> dict[str, object]:
        """Every key of a line, with all but ``message`` filled in (these never fail)."""
        fiber = message.fiber() if message.fiber is not None else None
        error = next((arg for arg in message.args if isinstance(arg, BaseException)), None)
        return {
            "ts": message.ts,
            "level": message.type,
            "name": message.name,
            "message": None,
            "plugin": fiber.name if fiber is not None else None,
            "error_type": _type_name(error) if error is not None else None,
            "traceback": self._traceback(error) if error is not None else None,
        }

    def _traceback(self, error: BaseException) -> str:
        # the traceback module shows an exception whose __str__ raises as
        # "<exception str() failed>", so this does not fail on such errors
        text = "".join(traceback.format_exception(error, chain=False))
        return clip_lines(text, self.max_length)

    def _fallback(self, message: Message, failure: Exception) -> dict[str, object]:
        """A line for a message whose text could not be rendered (error fields kept)."""
        try:
            detail = f"{_type_name(failure)}: {failure}; args: {message.args!r}"
        except Exception:
            detail = _type_name(failure)  # the arguments (or the error) cannot be shown either
        text = clip_lines(f"<unrenderable log message: {detail}>", self.max_length)
        return {**self._base(message), "message": text}


def _dumps(record: dict[str, object], *, ascii_only: bool) -> str:
    return json.dumps(record, ensure_ascii=ascii_only, default=repr) + "\n"


def _type_name(error: BaseException) -> str:
    """``ValueError`` for built-ins, else the module-qualified name of ``error``'s type."""
    kind = type(error)
    if kind.__module__ == "builtins":
        return kind.__qualname__
    return f"{kind.__module__}.{kind.__qualname__}"


def _describe(error: BaseException) -> str:
    """``"<type>: <text>"`` (just the type when the text is empty)."""
    text = str(error)
    return f"{_type_name(error)}: {text}" if text else _type_name(error)
