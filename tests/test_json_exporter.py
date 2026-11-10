"""JsonExporter: one JSON object per log message (spec §3.8)."""

from __future__ import annotations

import io
import json
import weakref
from typing import Any

import pytest

from pycordis import JsonExporter, LoggerLevel, Message
from pycordis.context import Context

KEYS = ["ts", "level", "name", "message", "plugin", "error_type", "traceback"]


class FlushCounter(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def message(*args: Any, type_: str = "info", fiber: Any = None) -> Message:
    return Message(1, 1700000000.5, "app", type_, LoggerLevel.INFO, args, fiber)


class Unprintable:
    def __str__(self) -> str:
        raise RuntimeError("no str")

    def __repr__(self) -> str:
        return "<Unprintable>"


def test_one_json_object_per_line_with_documented_keys() -> None:
    stream = io.StringIO()
    exporter = JsonExporter(stream)
    exporter.export(message("hello %s", "world"))
    exporter.export(message("second", type_="warn"))
    first, second = lines(stream)
    assert list(first) == KEYS
    assert first == {
        "ts": 1700000000.5,
        "level": "info",
        "name": "app",
        "message": "hello world",
        "plugin": None,
        "error_type": None,
        "traceback": None,
    }
    assert (second["level"], second["message"]) == ("warn", "second")


def test_exception_fields_come_from_an_exception_argument() -> None:
    stream = io.StringIO()
    try:
        raise ValueError("bad url")
    except ValueError as error:
        JsonExporter(stream).export(message(error, type_="error"))
    (line,) = lines(stream)
    assert line["message"] == "ValueError: bad url"
    assert line["error_type"] == "ValueError"
    assert line["traceback"].startswith("Traceback (most recent call last):")
    assert line["traceback"].endswith("\nValueError: bad url")
    assert "test_exception_fields_come_from_an_exception_argument" in line["traceback"]


def test_exception_after_a_text_argument_and_qualified_error_type() -> None:
    class CustomError(Exception):
        pass

    stream = io.StringIO()
    JsonExporter(stream).export(message("request failed:", CustomError("nope")))
    (line,) = lines(stream)
    error_type = f"{__name__}.{CustomError.__qualname__}"
    assert line["error_type"] == error_type
    assert line["message"] == f"request failed: {error_type}: nope"
    # never raised: no frames, only the exception line
    assert line["traceback"] == f"{error_type}: nope"


def test_non_serialisable_arguments_are_rendered_with_repr() -> None:
    stream = io.StringIO()
    JsonExporter(stream).export(message("%o and", {"key": object()}, {1, 2}))
    (line,) = lines(stream)
    assert line["message"].startswith('{"key": "<object object at')
    assert line["message"].endswith('>"} and "{1, 2}"')


def test_unrenderable_message_falls_back_to_a_minimal_line() -> None:
    stream = io.StringIO()
    JsonExporter(stream).export(message("value:", Unprintable(), type_="warn"))
    (line,) = lines(stream)
    assert list(line) == KEYS
    assert (line["level"], line["name"]) == ("warn", "app")
    assert line["message"] == (
        "<unrenderable log message: RuntimeError: no str; args: ('value:', <Unprintable>)>"
    )


def test_fallback_survives_an_unrepresentable_argument() -> None:
    class Hopeless:
        def __str__(self) -> str:
            raise RuntimeError("no str")

        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    stream = io.StringIO()
    JsonExporter(stream).export(message(Hopeless()))
    (line,) = lines(stream)
    assert line["message"] == "<unrenderable log message: RuntimeError>"


def test_unicode_and_newlines_stay_on_one_line() -> None:
    stream = io.StringIO()
    JsonExporter(stream).export(message("first\nsecond ✓"))
    assert stream.getvalue().count("\n") == 1
    assert lines(stream)[0]["message"] == "first\nsecond ✓"


def test_each_line_is_flushed() -> None:
    stream = FlushCounter()
    exporter = JsonExporter(stream)
    exporter.export(message("a"))
    exporter.export(message("b"))
    assert stream.flushes == 2


def test_message_without_a_live_fiber_has_no_plugin() -> None:
    class Gone:
        pass

    gone = Gone()
    ref = weakref.ref(gone)
    del gone
    stream = io.StringIO()
    JsonExporter(stream).export(message("x", fiber=ref))
    assert lines(stream)[0]["plugin"] is None


async def test_plugin_is_the_fiber_name_and_levels_filter() -> None:
    stream = io.StringIO()

    def worker(ctx: Context, _config: Any) -> None:
        ctx.logger.info("started")
        ctx.logger.debug("hidden")

    async with Context() as root:
        root.logger.exporter(JsonExporter(stream))
        await root.plugin(worker)
        quiet = io.StringIO()
        root.logger.exporter(JsonExporter(quiet, levels={"default": LoggerLevel.WARN}))
        root.logger("app").info("dropped")
        root.logger("app").warn("kept")
    assert [(line["name"], line["plugin"], line["message"]) for line in lines(stream)][:1] == [
        ("worker", "worker", "started")
    ]
    assert [line["message"] for line in lines(quiet)] == ["kept"]


def test_defaults_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    JsonExporter().export(message("to stderr"))
    assert json.loads(capsys.readouterr().err)["message"] == "to stderr"


def test_lone_surrogates_are_written_escaped_to_a_utf8_stream() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8")
    exporter = JsonExporter(stream)
    exporter.export(message("file \udcff.txt ✓"))  # a surrogateescape-decoded file name
    exporter.export(message("after ✓"))
    first, second = raw.getvalue().decode("utf-8").splitlines()
    assert "\\udcff" in first and "\\u2713" in first  # the whole line was escaped
    assert json.loads(first)["message"] == "file \udcff.txt ✓"
    assert second == json.dumps(json.loads(second), ensure_ascii=False)  # others stay UTF-8
    assert json.loads(second)["message"] == "after ✓"


def test_an_ascii_only_stream_gets_escaped_lines() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii")
    JsonExporter(stream).export(message("café"))
    assert json.loads(raw.getvalue().decode("ascii"))["message"] == "café"


def test_huge_exceptions_are_cut_per_line() -> None:
    stream = io.StringIO()
    exporter = JsonExporter(stream, max_length=100)
    exporter.export(message(ValueError("y" * 3_000_000), type_="error"))
    assert len(stream.getvalue()) < 1000
    (line,) = lines(stream)
    assert line["message"] == "ValueError: " + "y" * 88 + "..."
    assert line["traceback"] == "ValueError: " + "y" * 88 + "..."


def test_default_max_length_bounds_a_line() -> None:
    stream = io.StringIO()
    JsonExporter(stream).export(message(ValueError("y" * 3_000_000)))
    assert len(stream.getvalue()) < 2 * 10240 + 1000


def test_fallback_keeps_the_error_fields_when_str_raises() -> None:
    class Garbled(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no str")

    stream = io.StringIO()
    try:
        raise Garbled
    except Garbled as error:
        JsonExporter(stream).export(message(error, type_="error"))
    (line,) = lines(stream)
    assert line["message"].startswith("<unrenderable log message: RuntimeError: no str; args:")
    assert line["error_type"] == f"{__name__}.{Garbled.__qualname__}"
    assert line["traceback"].startswith("Traceback (most recent call last):")
    assert "<exception str() failed>" in line["traceback"]
