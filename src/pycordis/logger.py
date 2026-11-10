"""Logger service and exporters (port of cordis ``logger.ts``).

``ctx.logger`` is a callable service: ``ctx.logger("name")`` returns a named
:class:`Logger`, and ``ctx.logger.info(...)`` logs under a name derived from the
calling plugin. Messages are structured :class:`Message` records handed to every
registered :class:`Exporter`; the service itself keeps the last ``buffer_size``
messages in :attr:`LoggerService.buffer`.

Python differences from the JS original:

- ``%o`` / ``%O`` render with ``json.dumps(value, default=repr)``.
- An exception as the first argument is rendered as a formatted traceback without
  its chain (``chain=False``): a ``__cause__`` (or an unsuppressed implicit
  ``__context__``) is logged first as its own entry, and
  an ``ExceptionGroup`` without a cause logs each sub-exception instead of itself.
- Extra arguments are JSON-rendered only for ``dict`` / ``list`` / ``tuple`` / ``set``
  / ``frozenset``; others use ``str``. Output is split with ``splitlines`` for the
  per-line length limit, and ``%d`` / ``%i`` / ``%f`` of a non-number render ``nan``.
- A failing exporter does not stop the others; its exception goes to the running
  loop's exception handler (``sys.unraisablehook`` outside a running loop).
- An exporter's optional attributes (``colors``, ``max_length``, ``levels``,
  ``formatters``) are read with ``getattr`` defaults, so a plain object with an
  ``export`` method is a valid exporter.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import os
import re
import sys
import time
import traceback
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    SupportsFloat,
    SupportsIndex,
    TextIO,
    final,
)

from pycordis._context_internals import intercept_configs
from pycordis._guards import is_exception_group
from pycordis._utils import hyphenate, merge_entries, report_unhandled, symbols
from pycordis.fiber import is_reported
from pycordis.trace import Tracker, invoke

if TYPE_CHECKING:
    from pycordis.context import Context
    from pycordis.effect import EffectDisposer
    from pycordis.fiber import Fiber

__all__ = [
    "DEFAULT_MAX_LENGTH",
    "ConsoleExporter",
    "Exporter",
    "Formatter",
    "Logger",
    "LoggerLevel",
    "LoggerService",
    "Message",
    "StdlibLoggingExporter",
    "c16",
    "c256",
    "capture_stdlib_logging",
    "clip_lines",
    "default_formatters",
]

c16: list[int] = [6, 2, 3, 4, 5, 1]
"""ANSI 16-color palette indexes used for logger name coloring."""
c256: list[int] = [
    20, 21, 26, 27, 32, 33, 38, 39, 40, 41, 42, 43, 44, 45, 56, 57, 62,
    63, 68, 69, 74, 75, 76, 77, 78, 79, 80, 81, 92, 93, 98, 99, 112, 113,
    129, 134, 135, 148, 149, 160, 161, 162, 163, 164, 165, 166, 167, 168,
    169, 170, 171, 172, 173, 178, 179, 184, 185, 196, 197, 198, 199, 200,
    201, 202, 203, 204, 205, 206, 207, 208, 209, 214, 215, 220, 221,
]  # fmt: skip
"""ANSI 256-color palette indexes used for logger name coloring."""

_STDLIB_MARK = "pycordis_exported"
DEFAULT_MAX_LENGTH = 10240
"""Default per-line length limit of rendered messages (an exporter's ``max_length``)."""


class LoggerLevel(IntEnum):
    """Numeric severity used when exporters decide whether to emit a message.

    Example: ``ctx.intercept("logger", {"level": LoggerLevel.DEBUG})``.
    """

    ERROR = 0
    WARN = 1
    INFO = 2
    DEBUG = 3


@final
@dataclass(slots=True)
class Message:
    """Structured log record delivered to exporters.

    Example: ``[m.args for m in ctx.logger.buffer if m.type == "error"]``.
    """

    sn: int
    ts: float
    name: str
    type: str
    level: int
    args: tuple[object, ...]
    fiber: weakref.ref[Fiber] | None = None


type _LogType = Literal["error", "warn", "info", "debug"]
"""The level name a :class:`Logger` method records in :attr:`Message.type`."""

type Formatter = Callable[[object, object, Message], object]
"""Formatter used to resolve a printf-style placeholder: ``(value, exporter, message)``."""


class Exporter(Protocol):
    """Sink that receives structured log messages.

    Optional attributes read with defaults: ``colors: int | Literal[False]``,
    ``max_length: int`` (default 10240), ``levels: Mapping[str, int]`` (per logger
    name, plus ``"default"``) and ``formatters: Mapping[str, Formatter]``.

    Example::

        class ListExporter:
            def export(self, message: Message) -> None:
                seen.append(message)

        ctx.logger.exporter(ListExporter())
    """

    def export(self, message: Message) -> None:
        """Receive one message.

        Example: ``def export(self, message: Message) -> None: print(message.args)``.
        """
        ...


def _to_number(value: object) -> float:
    """``float(value)``, or NaN when ``value`` has none (``%d`` / ``%f`` of a non-number)."""
    if isinstance(value, int | float | str):
        convertible: SupportsFloat | SupportsIndex | str | bytes | bytearray = value
    elif isinstance(value, SupportsFloat | SupportsIndex | bytes | bytearray):
        convertible = value  # rarer types: only these pay for the protocol checks
    else:
        return math.nan
    try:
        return float(convertible)
    except (TypeError, ValueError, OverflowError):
        return math.nan


def _trunc(value: object) -> float | int:
    number = _to_number(value)
    return int(number) if math.isfinite(number) else number


def _json(value: object, exporter: object, message: Message) -> str:
    return json.dumps(value, default=repr)


def _color_format(value: object, exporter: object, message: Message) -> str:
    colors = getattr(exporter, "colors", None)
    return Logger.color(exporter, Logger.code(message.name, colors), value)


def _str(value: object, exporter: object, message: Message) -> str:
    return str(value)


def _int(value: object, exporter: object, message: Message) -> float | int:
    return _trunc(value)


def _float(value: object, exporter: object, message: Message) -> float:
    return _to_number(value)


def _empty(value: object, exporter: object, message: Message) -> str:
    return ""


default_formatters: dict[str, Formatter] = {
    "s": _str,
    "d": _int,
    "i": _int,
    "f": _float,
    "o": _json,
    "O": _json,
    "c": _empty,
    "C": _color_format,
}
"""Built-in placeholder formatters used by :meth:`Logger.format`."""


def _int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


class Logger:
    """Logger facade for one named subsystem.

    Example: ``ctx.logger("db").info("connected to %s", url)``.
    """

    def __init__(
        self,
        name: str,
        service: LoggerService,
        level: int | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> None:
        self.name = name
        """The logger name shown with each message."""
        self.level = level
        """Default maximum level exported when an exporter has no own threshold."""
        self.meta: Mapping[str, object] = meta or {}
        """Message fields merged into every record from this logger."""
        self._service = service

    @staticmethod
    def color(exporter: object, code: int, value: object, decoration: str = "") -> str:
        """Wrap ``value`` in ANSI color ``code`` if the exporter has colors enabled.

        Example: ``Logger.color(exporter, 3, "ready")``.
        """
        colors = getattr(exporter, "colors", None)
        if not colors:
            return str(value)
        prefix = str(code) if code < 8 else f"8;5;{code}"
        return f"\x1b[3{prefix}{decoration if colors >= 2 else ''}m{value}\x1b[0m"

    @staticmethod
    def code(name: str, colors: int | Literal[False] | None) -> int:
        """Pick a stable palette color for ``name`` (``0`` when colors are off).

        Example: ``Logger.code("db", 2)`` (the same name always gets the same color).
        """
        value = 0
        for unit in memoryview(name.encode("utf-16-le")).cast("H"):
            value = _int32((value << 3) - value + unit + 13)
        palette = [] if not colors else c256 if colors >= 2 else c16
        return palette[abs(value) % len(palette)] if palette else 0

    @staticmethod
    def format(exporter: object, message: Message) -> str:
        """Render ``message`` printf-style with ``exporter``'s formatters and limits.

        Example: ``Logger.format(exporter, message)`` gives ``"connected to db"``.
        """
        args = list(message.args)
        first = args[0] if args else None
        if isinstance(first, BaseException):
            args[0] = "".join(traceback.format_exception(first, chain=False)).rstrip("\n")
            template = "%s"
        elif isinstance(first, str):
            template = first
            del args[0]
        else:
            template = "%o"

        formatters: Mapping[str, Formatter] = getattr(exporter, "formatters", None) or {}

        def replace(match: re.Match[str]) -> str:
            char = match.group(1)
            if char == "%":
                return "%"
            formatter = formatters.get(char) or default_formatters.get(char)
            if formatter is None:
                return match.group(0)
            return str(formatter(args.pop(0) if args else None, exporter, message))

        text = _PLACEHOLDER.sub(replace, template)
        o_formatter = formatters.get("o") or default_formatters["o"]
        for arg in args:
            if _json_rendered(arg):
                text += f" {o_formatter(arg, exporter, message)}"
            else:
                text += f" {arg}"

        return clip_lines(text, getattr(exporter, "max_length", DEFAULT_MAX_LENGTH))

    def error(self, *args: object) -> None:
        """Log at ERROR level.

        Example: ``ctx.logger("db").error(error)``.
        """
        self._log("error", LoggerLevel.ERROR, args)

    def warn(self, *args: object) -> None:
        """Log at WARN level.

        Example: ``ctx.logger("db").warn("slow query: %dms", 250)``.
        """
        self._log("warn", LoggerLevel.WARN, args)

    def info(self, *args: object) -> None:
        """Log at INFO level.

        Example: ``ctx.logger("db").info("connected to %s", url)``.
        """
        self._log("info", LoggerLevel.INFO, args)

    def debug(self, *args: object) -> None:
        """Log at DEBUG level.

        Example: ``ctx.logger("db").debug("row %o", row)``.
        """
        self._log("debug", LoggerLevel.DEBUG, args)

    def _log(self, type_: _LogType, level: int, args: tuple[object, ...]) -> None:
        if len(args) == 1 and isinstance(args[0], BaseException):
            error = args[0]
            cause = error.__cause__
            context = error.__context__
            # implicit chaining ("during handling of ..."), unless reported on its own
            if cause is None and not error.__suppress_context__ and not is_reported(context):
                cause = context
            if cause is not None:
                self._log(type_, level, (cause,))
            elif is_exception_group(error):
                for sub in error.exceptions:
                    self._log(type_, level, (sub,))
                return

        service = self._service
        sn = service.next_sn()
        ts = time.time()
        for exporter in list(service.exporters.values()):
            levels: Mapping[str, int] = getattr(exporter, "levels", None) or {}
            target = levels.get(self.name)
            if target is None:
                target = levels.get("default")
            if target is None:
                target = self.level if self.level is not None else LoggerLevel.INFO
            if target < level:
                continue
            # ``meta`` may set any Message field (cordis spreads it into the record)
            fields: dict[str, Any] = {
                "sn": sn,
                "ts": ts,
                "type": type_,
                "level": level,
                "name": self.name,
                **self.meta,
                "args": args,
            }
            try:
                exporter.export(Message(**fields))
            except Exception as failure:
                # one broken exporter must not starve the others nor lose the error
                report_unhandled(
                    f"pycordis: logger exporter {type(exporter).__name__} failed", failure
                )


def _json_rendered(value: object) -> bool:
    """Whether an extra log argument renders as JSON (a container) rather than with ``str``."""
    return isinstance(value, (dict, list, tuple, set, frozenset))


def _option[T](config: Mapping[object, object], key: str, kind: type[T]) -> T | None:
    """The logger intercept option ``key`` (``None`` when unset); ``TypeError`` if mistyped."""
    value = config.get(key)
    if value is None or isinstance(value, kind):
        return value
    raise TypeError(f'logger config "{key}" must be {kind.__name__}, got {type(value).__name__}')


def clip_lines(text: str, max_length: int) -> str:
    """Cut every line of ``text`` to ``max_length`` characters, marking cut lines with ``...``.

    Lines are split with :meth:`str.splitlines` and joined with ``\\n`` (a
    trailing newline is dropped). Example::

        clip_lines("abcdef\\nxy", 3) == "abc...\\nxy"
    """
    return "\n".join(
        line[:max_length] + ("..." if len(line) > max_length else "")
        for line in text.splitlines() or [""]
    )


_PLACEHOLDER = re.compile(r"%([a-zA-Z%])")


@final
class _BufferExporter:
    colors: int = 3

    def __init__(self, service: LoggerService) -> None:
        self._service = service

    def export(self, message: Message) -> None:
        self._service.buffer.append(message)


@final
class LoggerService:
    """Built-in logging service, installed as ``ctx.logger``.

    Call ``ctx.logger(name)`` for a named :class:`Logger`, or ``ctx.logger.info(...)``
    (also ``error`` / ``warn`` / ``debug``) to log under a name derived from the
    calling plugin (hyphenated). ``ctx.intercept("logger", {"name": ..., "level": ...})``
    overrides the name and default level for plugins loaded below that context.
    """

    __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx", no_shadow=True)

    def __init__(self, ctx: Context, buffer_size: int = 1000) -> None:
        self.ctx = ctx
        """The context the service was installed on (the use site through a proxy)."""
        self.buffer_size = buffer_size
        """Maximum number of messages kept in :attr:`buffer`."""
        self.buffer: collections.deque[Message] = collections.deque(maxlen=buffer_size)
        """The most recent messages, oldest first."""
        self.exporters: dict[int, Exporter] = {}
        """Registered exporters by id."""
        self._sn_message = 0
        self._sn_exporter = 0
        self.exporters[0] = _BufferExporter(self)

    def __call__(self, name: str | None = None) -> Logger:
        """Call through the service's own traced view (``ctx.logger("name")``).

        See :meth:`__cordis_invoke__`. Example: ``ctx.logger("db").info("connected")``.
        """
        logger: Logger = invoke(self, name)
        return logger

    def next_sn(self) -> int:
        """Allocate the serial number of the next message (shared by every logger).

        Internal — not part of the public API.
        """
        self._sn_message += 1
        return self._sn_message

    def _resolve_config(self) -> dict[object, object]:
        return merge_entries(intercept_configs(self.ctx, "logger"))

    def exporter(self, exporter: Exporter) -> EffectDisposer:
        """Register an exporter, disposed with the calling fiber; returns its disposer."""
        service: LoggerService = getattr(self, symbols.original, self)

        def register() -> Callable[[], object]:
            service._sn_exporter += 1
            key = service._sn_exporter
            service.exporters[key] = exporter
            return partial(service.exporters.pop, key, None)

        return self.ctx.effect(register, "ctx.logger.exporter()")

    def __cordis_invoke__(self, name: str | None = None) -> Logger:
        """Create a :class:`Logger`; the default name is the calling fiber's, hyphenated."""
        config = self._resolve_config()
        caller: Context = getattr(self, symbols.caller, None) or self.ctx
        fiber = caller.fiber
        if name is None:
            name = _option(config, "name", str)
        if name is None:
            name = hyphenate(fiber.name)
        service: LoggerService = getattr(self, symbols.original, self)
        level = _option(config, "level", int)
        return Logger(name, service, level, {"fiber": weakref.ref(fiber)})

    def error(self, *args: object) -> None:
        """Log at ERROR level under the calling plugin's logger."""
        self().error(*args)

    def warn(self, *args: object) -> None:
        """Log at WARN level under the calling plugin's logger."""
        self().warn(*args)

    def info(self, *args: object) -> None:
        """Log at INFO level under the calling plugin's logger."""
        self().info(*args)

    def debug(self, *args: object) -> None:
        """Log at DEBUG level under the calling plugin's logger."""
        self().debug(*args)


def _auto_colors(stream: TextIO) -> int:
    isatty = getattr(stream, "isatty", None)
    if os.environ.get("NO_COLOR") or not (callable(isatty) and isatty()):
        return 0
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return 3
    return 1


class ConsoleExporter:
    """Write ``HH:MM:SS [T] name text`` lines to a text stream (default ``sys.stderr``).

    ``colors`` defaults to 0 when the stream is not a TTY or ``NO_COLOR`` is set,
    3 when ``COLORTERM`` is ``truecolor`` / ``24bit``, else 1.

    Example: ``ctx.logger.exporter(ConsoleExporter(colors=False))``.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        colors: int | Literal[False] | None = None,
        levels: Mapping[str, int] | None = None,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.colors: int | Literal[False] = _auto_colors(self.stream) if colors is None else colors
        self.levels: Mapping[str, int] = levels or {}

    def export(self, message: Message) -> None:
        """Write one formatted line.

        Example: ``exporter.export(message)`` (the logger calls it).
        """
        clock = time.strftime("%H:%M:%S", time.localtime(message.ts))
        name = Logger.color(self, Logger.code(message.name, self.colors), message.name, ";1")
        text = Logger.format(self, message)
        self.stream.write(f"{clock} [{message.type[0].upper()}] {name} {text}\n")


_STDLIB_LEVELS = {
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


class StdlibLoggingExporter:
    """Forward messages to ``logging.getLogger(f"{prefix}.{name}")`` at the mapped level.

    Example: ``ctx.logger.exporter(StdlibLoggingExporter())``.
    """

    colors: Literal[False] = False

    def __init__(self, prefix: str = "pycordis", levels: Mapping[str, int] | None = None) -> None:
        self.prefix = prefix
        self.levels: Mapping[str, int] = levels or {}

    def export(self, message: Message) -> None:
        """Log one formatted message through the standard library.

        Example: ``exporter.export(message)`` (the logger calls it).
        """
        logging.getLogger(f"{self.prefix}.{message.name}").log(
            _STDLIB_LEVELS.get(message.type, logging.INFO),
            "%s",
            Logger.format(self, message),
            extra={_STDLIB_MARK: True},
        )


def capture_stdlib_logging(ctx: Context, level: int = logging.INFO) -> EffectDisposer:
    """Route stdlib ``logging`` records at or above ``level`` into ``ctx.logger``.

    Installs a handler on the root logger (removed when the disposer runs or the
    fiber unloads). Records produced by :class:`StdlibLoggingExporter` are skipped
    so the two can be combined without looping. The root logger's own level still
    decides which records reach the handler.

    Example: ``remove = capture_stdlib_logging(ctx)``; ``remove()`` stops it.
    """

    @final
    class Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if getattr(record, _STDLIB_MARK, False):
                return
            text = record.getMessage()
            if record.exc_info:
                text += "\n" + logging.Formatter().formatException(record.exc_info)
            logger = ctx.logger(record.name)
            method = (
                logger.error
                if record.levelno >= logging.ERROR
                else logger.warn
                if record.levelno >= logging.WARNING
                else logger.info
                if record.levelno >= logging.INFO
                else logger.debug
            )
            method("%s", text)

    def install() -> Callable[[], None]:
        handler = Handler(level)
        root = logging.getLogger()
        root.addHandler(handler)
        return lambda: root.removeHandler(handler)

    return ctx.effect(install, "capture_stdlib_logging()")
