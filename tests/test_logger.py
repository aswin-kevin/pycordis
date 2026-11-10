import asyncio
import io
import logging
import re
import time
from collections.abc import Iterator
from typing import Any

import pytest
from conftest import capture_reports

from pycordis._utils import symbols
from pycordis.context import Context
from pycordis.errors import ServiceAccessError
from pycordis.fiber import report_error
from pycordis.logger import (
    ConsoleExporter,
    Formatter,
    Logger,
    LoggerLevel,
    LoggerService,
    Message,
    StdlibLoggingExporter,
    c16,
    c256,
    capture_stdlib_logging,
)
from pycordis.supervise import RestartPolicy


class Sink:
    def __init__(self, **attrs: Any) -> None:
        self.messages: list[Message] = []
        for key, value in attrs.items():
            setattr(self, key, value)

    def export(self, message: Message) -> None:
        self.messages.append(message)


def _message(*args: Any, name: str = "test", type_: str = "info") -> Message:
    return Message(1, 0.0, name, type_, LoggerLevel.INFO, args)


# -- service --------------------------------------------------------------------


async def test_info_reaches_buffer_with_args() -> None:
    root = Context()
    root.logger.info("%s #%d", "x", 3)
    [message] = root.logger.buffer
    assert (message.name, message.type, message.level) == ("root", "info", LoggerLevel.INFO)
    assert message.args == ("%s #%d", "x", 3)
    assert message.sn == 1 and message.ts <= time.time()
    assert message.fiber is not None and message.fiber() is root.fiber


async def test_named_logger_and_buffer_limit() -> None:
    root = Context()
    assert isinstance(root.logger("db"), Logger)
    root.logger("db").warn("w")
    root.logger("db").debug("d")  # above the default INFO threshold: buffer skips it
    root.logger("db").error("e")
    assert [(m.name, m.type) for m in root.logger.buffer] == [("db", "warn"), ("db", "error")]
    assert root.logger.buffer.maxlen == 1000
    for index in range(1100):
        root.logger.info("n%d", index)
    assert len(root.logger.buffer) == 1000
    assert root.logger.buffer[-1].args == ("n%d", 1099)
    assert root.logger.buffer[-1].sn == 1103


async def test_plugin_logger_name_is_hyphenated_plugin_name() -> None:
    root = Context()

    def my_plugin(ctx: Context, _config: Any) -> None:
        ctx.logger.info("hello")
        ctx.logger("explicit").info("hi")

    await root.plugin(my_plugin)
    assert [m.name for m in root.logger.buffer] == ["my-plugin", "explicit"]


async def test_intercept_name_and_level() -> None:
    root = Context()
    sink = Sink()
    root.logger.exporter(sink)

    def plugin(ctx: Context, _config: Any) -> None:
        ctx.logger.info("a")
        ctx.logger.debug("hidden")

    await root.intercept("logger", {"name": "custom"}).plugin(plugin)
    assert [m.name for m in sink.messages] == ["custom"]
    await root.intercept("logger", {"level": LoggerLevel.DEBUG}).plugin(plugin)
    assert [(m.name, m.type) for m in sink.messages[1:]] == [
        ("plugin", "info"),
        ("plugin", "debug"),
    ]


async def test_intercept_option_of_wrong_type_is_a_type_error() -> None:
    root = Context()
    with pytest.raises(TypeError, match='logger config "name" must be str, got int'):
        root.intercept("logger", {"name": 5}).logger()
    with pytest.raises(TypeError, match='logger config "level" must be int, got str'):
        root.intercept("logger", {"level": "debug"}).logger("x")


async def test_exporter_levels_per_name_and_default() -> None:
    root = Context()
    sink = Sink(levels={"noisy": LoggerLevel.DEBUG, "default": LoggerLevel.WARN})
    root.logger.exporter(sink)
    root.logger("noisy").debug("1")
    root.logger("other").info("2")
    root.logger("other").warn("3")
    assert [m.args[0] for m in sink.messages] == ["1", "3"]


async def test_exporter_disposed_with_fiber() -> None:
    root = Context()
    sink = Sink()

    def plugin(ctx: Context, _config: Any) -> None:
        ctx.logger.exporter(sink)

    fiber = await root.plugin(plugin)
    root.logger.info("a")
    await fiber.dispose()
    root.logger.info("b")
    assert [m.args[0] for m in sink.messages] == ["a"]


async def test_exporter_disposer() -> None:
    root = Context()
    sink = Sink()
    dispose = root.logger.exporter(sink)
    root.logger.info("a")
    dispose()
    root.logger.info("b")
    assert [m.args[0] for m in sink.messages] == ["a"]


# -- exceptions -----------------------------------------------------------------


async def test_exception_with_cause_logs_two_messages() -> None:
    root = Context()
    try:
        try:
            raise ValueError("inner")
        except ValueError as cause:
            raise RuntimeError("outer") from cause
    except RuntimeError as error:
        root.logger.error(error)
    kinds: list[type[object]] = [type(m.args[0]) for m in root.logger.buffer]
    assert kinds == [ValueError, RuntimeError]
    text = Logger.format(Sink(), root.logger.buffer[1])
    assert "RuntimeError: outer" in text and "ValueError" not in text


async def test_exception_with_implicit_context_logs_two_messages() -> None:
    root = Context()
    try:
        try:
            raise ValueError("inner")
        except ValueError:
            raise RuntimeError("outer")  # noqa: B904  (deliberate: implicit __context__)
    except RuntimeError as error:
        root.logger.error(error)
    assert [type(m.args[0]) for m in root.logger.buffer] == [ValueError, RuntimeError]
    root.logger.buffer.clear()
    # `from None` suppresses the context
    try:
        try:
            raise ValueError("inner")
        except ValueError:
            raise RuntimeError("outer") from None
    except RuntimeError as error:
        root.logger.error(error)
    assert [type(m.args[0]) for m in root.logger.buffer] == [RuntimeError]


async def test_failing_exporter_without_loop_goes_to_unraisablehook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import threading

    seen: list[Any] = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    root = Context()

    class Broken:
        def export(self, message: Message) -> None:
            raise OSError("disk full")

    root.logger.exporter(Broken())
    # e.g. a stdlib logging call from a worker thread: no running loop there
    thread = threading.Thread(target=root.logger.info, args=("x",))
    thread.start()
    thread.join()
    assert len(seen) == 1
    assert isinstance(seen[0].exc_value, OSError)
    assert seen[0].err_msg == "pycordis: logger exporter Broken failed"


async def test_exception_group_logs_each() -> None:
    root = Context()
    group = ExceptionGroup("many", [ValueError("a"), KeyError("b")])
    root.logger.error(group)
    assert [type(m.args[0]) for m in root.logger.buffer] == [ValueError, KeyError]


# -- format ---------------------------------------------------------------------


def test_format_placeholders() -> None:
    sink = Sink()
    assert Logger.format(sink, _message("%s %d %o %%", "a", "3.9", {"k": [1]})) == (
        'a 3 {"k": [1]} %'
    )
    assert Logger.format(sink, _message("%i %f %O %c|", 7.8, "2.5", (1, 2), "css")) == (
        "7 2.5 [1, 2] |"
    )
    assert Logger.format(sink, _message("%d", "x")) == "nan"
    assert Logger.format(sink, _message("%z %s", "only")) == "%z only"


def test_format_extra_args_and_non_string_first() -> None:
    sink = Sink()
    assert Logger.format(sink, _message("a", 1, {"b": 2}, None)) == 'a 1 {"b": 2} None'
    assert Logger.format(sink, _message({"x": 1})) == '{"x": 1}'
    assert Logger.format(sink, _message(object)) == "\"<class 'object'>\""
    assert Logger.format(sink, _message()) == "null"


def test_format_max_length_per_line() -> None:
    sink = Sink(max_length=3)
    assert Logger.format(sink, _message("abcdef\nxy")) == "abc...\nxy"
    assert len(Logger.format(Sink(), _message("x" * 20000))) == 10240 + 3


def test_format_custom_formatter_and_color() -> None:
    formatters: dict[str, Formatter] = {"s": lambda value, _e, _m: f"<{value}>"}
    sink = Sink(formatters=formatters, colors=0)
    assert Logger.format(sink, _message("%s %C", "a", "b")) == "<a> b"
    colored = Sink(colors=1)
    code = Logger.code("test", 1)
    assert Logger.format(colored, _message("%C", "b")) == f"\x1b[3{code}mb\x1b[0m"


def test_color_and_code() -> None:
    assert Logger.color(Sink(), 3, "v") == "v"
    assert Logger.color(Sink(colors=False), 3, "v") == "v"
    assert Logger.color(Sink(colors=1), 3, "v", ";1") == "\x1b[33mv\x1b[0m"
    assert Logger.color(Sink(colors=3), 3, "v", ";1") == "\x1b[33;1mv\x1b[0m"
    assert Logger.color(Sink(colors=3), 200, "v") == "\x1b[38;5;200mv\x1b[0m"
    assert Logger.code("x", False) == 0
    assert Logger.code("abc", 1) in c16 and Logger.code("abc", 3) in c256
    # JS: hash = ((hash << 3) - hash) + charCode + 13 over "abc" (no overflow)
    h = 0
    for ch in "abc":
        h = h * 7 + ord(ch) + 13
    assert Logger.code("abc", 1) == c16[h % len(c16)]
    assert Logger.code("a" * 40, 3) in c256  # int32 wrap-around


# -- exporters ------------------------------------------------------------------


def test_console_exporter_plain_line() -> None:
    stream = io.StringIO()
    exporter = ConsoleExporter(stream, colors=0)
    ts = time.mktime((2026, 1, 2, 3, 4, 5, 0, 0, -1))
    exporter.export(Message(1, ts, "app", "warn", 1, ("%s!", "hey")))
    assert stream.getvalue() == "03:04:05 [W] app hey!\n"


def test_console_exporter_colors() -> None:
    stream = io.StringIO()
    exporter = ConsoleExporter(stream, colors=3)
    exporter.export(Message(1, 0.0, "app", "error", 0, ("x",)))
    code = Logger.code("app", 3)
    assert re.search(rf"\[E\] \x1b\[38;5;{code};1mapp\x1b\[0m x\n$", stream.getvalue())


def test_console_exporter_auto_colors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("COLORTERM", raising=False)
    assert ConsoleExporter(io.StringIO()).colors == 0
    assert ConsoleExporter(Tty()).colors == 1
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert ConsoleExporter(Tty()).colors == 3
    monkeypatch.setenv("NO_COLOR", "1")
    assert ConsoleExporter(Tty()).colors == 0
    assert ConsoleExporter(levels={"a": 1}).stream is not None


@pytest.fixture
def stdlib_root() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    root.setLevel(logging.DEBUG)
    yield root
    root.setLevel(level)
    root.handlers[:] = handlers


async def test_stdlib_exporter_forwards(
    caplog: pytest.LogCaptureFixture, stdlib_root: logging.Logger
) -> None:
    root = Context()
    root.logger.exporter(StdlibLoggingExporter())
    with caplog.at_level(logging.DEBUG):
        root.logger("svc").warn("%s!", "w")
        root.logger("svc").error("e")
    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        ("pycordis.svc", logging.WARNING, "w!"),
        ("pycordis.svc", logging.ERROR, "e"),
    ]


async def test_capture_stdlib_round_trip(stdlib_root: logging.Logger) -> None:
    root = Context()
    dispose = capture_stdlib_logging(root.extend(), logging.INFO)
    logging.getLogger("lib.x").debug("skipped")
    logging.getLogger("lib.x").info("50%% of %s", "it")
    logging.getLogger("lib.x").warning("w")
    logging.getLogger("lib.x").error("e")
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("lib.x").exception("failed")
    got = [(m.name, m.type, Logger.format(Sink(), m)) for m in root.logger.buffer]
    assert got[:3] == [
        ("lib.x", "info", "50% of it"),
        ("lib.x", "warn", "w"),
        ("lib.x", "error", "e"),
    ]
    assert got[3][2].startswith("failed\nTraceback") and "ValueError: boom" in got[3][2]
    dispose()
    logging.getLogger("lib.x").error("after")
    assert len(root.logger.buffer) == 4


async def test_capture_stdlib_no_loop(stdlib_root: logging.Logger) -> None:
    root = Context()
    root.logger.exporter(StdlibLoggingExporter())
    capture_stdlib_logging(root, logging.DEBUG)
    root.logger("a").info("once")
    logging.getLogger("b").info("also once")
    assert [(m.name, m.args) for m in root.logger.buffer] == [
        ("a", ("once",)),
        ("b", ("%s", "also once")),
    ]


# -- error reporting ------------------------------------------------------------


async def test_plugin_failure_logged_exactly_once() -> None:
    root = Context()

    def failing(_ctx: Context, _config: Any) -> None:
        raise RuntimeError("sync fail")

    async def failing_async(_ctx: Context, _config: Any) -> None:
        raise RuntimeError("async fail")

    with pytest.raises(RuntimeError, match="sync fail"):
        await root.plugin(failing)
    with pytest.raises(RuntimeError, match="async fail"):
        await root.plugin(failing_async)
    await asyncio.sleep(0.01)
    errors = [m for m in root.logger.buffer if m.type == "error"]
    assert [str(m.args[0]) for m in errors] == ["sync fail", "async fail"]
    assert [m.name for m in errors] == ["failing", "failing-async"]


async def test_exception_group_with_cause_logs_cause_then_group() -> None:
    root = Context()
    group = ExceptionGroup("many", [ValueError("a")])
    group.__cause__ = KeyError("c")
    root.logger.error(group)
    assert [type(m.args[0]) for m in root.logger.buffer] == [KeyError, ExceptionGroup]


async def test_failing_exporter_isolated_and_reported_to_loop_handler() -> None:
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    seen: list[BaseException] = []
    loop.set_exception_handler(lambda _loop, context: seen.append(context["exception"]))
    try:
        root = Context()

        class Broken:
            def export(self, message: Message) -> None:
                raise OSError("disk full")

        sink = Sink()
        root.logger.exporter(Broken())
        root.logger.exporter(sink)
        error = RuntimeError("x")
        report_error(root.fiber, error)
        report_error(root.fiber, error)  # deduplicated
        assert [m.args[0] for m in sink.messages] == [error]  # later exporters still got it
        assert [type(e) for e in seen] == [OSError]  # exporter's failure only, not `error`
    finally:
        loop.set_exception_handler(previous)


async def test_reported_error_does_not_reach_loop_handler() -> None:
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    seen: list[BaseException] = []
    loop.set_exception_handler(lambda _loop, context: seen.append(context["exception"]))
    try:
        root = Context()

        def failing(_ctx: Context, _config: Any) -> None:
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            await root.plugin(failing)
        assert [str(m.args[0]) for m in root.logger.buffer if m.type == "error"] == ["nope"]
        assert seen == []
    finally:
        loop.set_exception_handler(previous)


async def test_report_falls_back_when_logger_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    seen: list[BaseException] = []
    loop.set_exception_handler(lambda _loop, context: seen.append(context["exception"]))
    try:
        root = Context()

        def unavailable(self: LoggerService, name: str | None = None) -> Logger:
            raise ServiceAccessError("logger unavailable", name="logger", obj=root)

        # `ctx.logger()` dispatches through the service's invoke hook
        monkeypatch.setattr(LoggerService, "__cordis_invoke__", unavailable)
        error = RuntimeError("x")
        report_error(root.fiber, error)
        assert seen == [error]
    finally:
        loop.set_exception_handler(previous)


async def test_generator_none_warning_falls_back_when_logger_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    seen: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: seen.append(context))
    try:
        root = Context()

        def unavailable(self: LoggerService, name: str | None = None) -> Logger:
            raise ServiceAccessError("logger unavailable", name="logger", obj=root)

        def fixture_style(_ctx: Context, _config: None) -> Iterator[None]:
            yield None  # the misuse that is warned about

        monkeypatch.setattr(LoggerService, "__cordis_invoke__", unavailable)
        await root.plugin(fixture_style)
        assert [type(c["exception"]) for c in seen] == [ServiceAccessError]
        assert "generator effect yielded None" in seen[0]["message"]
    finally:
        loop.set_exception_handler(previous)


async def test_supervisor_log_falls_back_when_logger_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    seen: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: seen.append(context))
    try:
        root = Context()

        def unavailable(self: LoggerService, name: str | None = None) -> Logger:
            raise ServiceAccessError("logger unavailable", name="logger", obj=root)

        def failing(_ctx: Context, _config: None) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(LoggerService, "__cordis_invoke__", unavailable)
        policy = RestartPolicy(mode="on-failure", max_restarts=0)
        with pytest.raises(RuntimeError):
            await root.plugin(failing, policy=policy)
        messages = [c["message"] for c in seen]
        assert any(m.startswith("pycordis: supervisor: ") for m in messages), messages
    finally:
        loop.set_exception_handler(previous)


# -- installation ---------------------------------------------------------------


async def test_logger_is_plain_property_not_provided_service() -> None:
    root = Context()
    original = getattr(root.logger, symbols.original)
    assert getattr(root.isolate("logger").logger, symbols.original) is original
    assert root.fiber.get_effects() == []
    fired: list[str] = []
    root.on("internal/get", lambda *args: fired.append("get"))

    def plugin(ctx: Context, _config: Any) -> None:
        ctx.logger.info("hi")

    await root.isolate("logger").plugin(plugin)
    assert [m.args for m in root.logger.buffer] == [("hi",)]
    assert fired == []


async def test_plugin_under_isolated_logger_reports_via_logger() -> None:
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    seen: list[BaseException] = []
    loop.set_exception_handler(lambda _loop, context: seen.append(context["exception"]))
    try:
        root = Context()

        def failing(_ctx: Context, _config: Any) -> None:
            raise RuntimeError("iso")

        with pytest.raises(RuntimeError):
            await root.isolate("logger").plugin(failing)
        assert [str(m.args[0]) for m in root.logger.buffer if m.type == "error"] == ["iso"]
        assert seen == []
    finally:
        loop.set_exception_handler(previous)


async def test_capture_reports_ignores_plain_text_errors() -> None:
    async with Context() as root:
        errors = capture_reports(root)
        root.logger("app").error("plain text, not an exception")
        assert errors == []
