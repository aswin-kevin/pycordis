"""Tests for the health and readiness server (spec §3.8, plan Review Focus 4).

Every request goes through a raw ``asyncio.open_connection`` client: no HTTP
library is involved, so malformed, oversized and slow requests can be sent
exactly as written.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import textwrap
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from conftest import apply_plugin

from pycordis.context import Context
from pycordis.diagnose import Diagnosis
from pycordis.errors import ValidationError
from pycordis.fiber import FiberState
from pycordis.health import HealthServer
from pycordis.inspect import FiberNode
from pycordis.loader.service import Loader, LoaderConfig
from pycordis.registry import plugin_meta
from pycordis.testing import MemoryExporter, mock_context, settle

# -- a raw HTTP client ----------------------------------------------------------------


@dataclass
class Response:
    """A parsed HTTP response."""

    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


async def exchange(port: int, raw: bytes) -> bytes:
    """Send ``raw`` and return everything the server sends until it closes."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(raw)
        with contextlib.suppress(ConnectionError):  # the server may close mid-send
            await writer.drain()
        return await reader.read()
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


def parse(data: bytes) -> Response:
    head, separator, body = data.partition(b"\r\n\r\n")
    assert separator, data
    status_line, *lines = head.decode("ascii").split("\r\n")
    version, status, _reason = status_line.split(" ", 2)
    assert version == "HTTP/1.1"
    headers = {name.lower(): value.strip() for name, value in (h.split(":", 1) for h in lines)}
    return Response(int(status), headers, body)


async def send(port: int, raw: bytes) -> Response:
    return parse(await exchange(port, raw))


async def get(port: int, target: str, method: str = "GET") -> Response:
    return await send(port, f"{method} {target} HTTP/1.1\r\nHost: test\r\n\r\n".encode())


async def start_health(root: Context, **config: Any) -> HealthServer:
    """Load a :class:`HealthServer` under ``root`` and return the service."""
    await root.plugin(HealthServer, HealthServer.Config(**config))
    health: HealthServer = root.health
    return health


@contextlib.asynccontextmanager
async def serving(**config: Any) -> AsyncGenerator[tuple[Context, HealthServer]]:
    async with mock_context() as root:
        yield root, await start_health(root, **config)


def plugin(
    name: str,
    *,
    inject: list[str] | None = None,
    provide: str | None = None,
    apply: Callable[[Context, Any], object] | None = None,
) -> Any:
    """An object plugin named ``name`` with ``inject`` / ``provide`` metadata."""

    def noop(_ctx: Context, _config: Any) -> None:
        return None

    body = noop if apply is None else apply
    return plugin_meta(name=name, inject=inject, provide=provide)(apply_plugin(name, body))


def raising(message: str) -> Callable[[Context, Any], None]:
    def apply(_ctx: Context, _config: Any) -> None:
        raise RuntimeError(message)

    return apply


# -- routes ---------------------------------------------------------------------------


async def test_healthz_is_ok_with_safe_headers() -> None:
    async with serving() as (_root, health):
        response = await get(health.port, "/healthz")
        assert response.status == 200
        assert response.json() == {"status": "ok"}
        assert response.headers == {
            "content-type": "application/json",
            "content-length": str(len(response.body)),
            "cache-control": "no-store",
            "x-content-type-options": "nosniff",
            "connection": "close",
        }


async def test_query_string_is_ignored_but_paths_match_exactly() -> None:
    async with serving() as (_root, health):
        assert (await get(health.port, "/healthz?probe=1")).status == 200
        for target in ("/healthz/", "/HEALTHZ", "//healthz", "/./healthz", "/healthz%20", "/"):
            response = await get(health.port, target)
            assert response.status == 404, target
            assert response.json() == {"error": "not found"}


async def test_other_methods_are_405() -> None:
    async with serving() as (_root, health):
        for method in ("POST", "HEAD", "DELETE", "OPTIONS"):
            response = await get(health.port, "/healthz", method)
            assert response.status == 405, method
            assert response.headers["allow"] == "GET"
            assert response.json() == {"error": "method not allowed"}


async def test_http_1_0_requests_are_served() -> None:
    async with serving() as (_root, health):
        response = await send(health.port, b"GET /healthz HTTP/1.0\r\n\r\n")
        assert response.status == 200


async def test_inspect_returns_the_fiber_tree_without_configs() -> None:
    async with serving() as (root, health):
        await root.plugin(plugin("db", provide="db"), {"password": "s3cret-value"})
        await root.plugin(plugin("app", inject=["db"]))
        response = await get(health.port, "/inspect")
        assert response.status == 200
        assert b"s3cret" not in response.body
        tree = response.json()
        assert (tree["name"], tree["uid"], tree["state"]) == ("root", 0, "ACTIVE")
        children = {child["name"]: child for child in tree["children"]}
        assert set(children) == {"HealthServer", "db", "app"}
        assert children["HealthServer"]["provides"] == ["health"]
        db = children["db"]
        assert db == {
            "name": "db",
            "uid": db["uid"],
            "state": "ACTIVE",
            "provides": [],
            "injects": [],
            "children": [],
        }
        assert children["app"]["injects"] == ["db"]


# -- readiness ------------------------------------------------------------------------

LOADER_TREE = """
[[plugins]]
id = "llm"
use = "loadfix:Provider"

[[plugins]]
id = "dep"
use = "loadfix:dependent"

[[plugins]]
id = "off"
use = "loadfix:nope"
disabled = true

[[plugins]]
group = "idle"
disabled = true
plugins = [{ id = "inner", use = "loadfix:nope" }]
"""


async def start_loader(root: Context, path: Path, text: str) -> Loader:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    await root.plugin(Loader, LoaderConfig(path=str(path)))
    loader: Loader = root.loader
    await loader.start()
    await settle(root)
    return loader


async def test_readyz_without_a_loader_is_not_ready() -> None:
    async with serving() as (_root, health):
        response = await get(health.port, "/readyz")
        assert response.status == 503
        assert response.json() == {
            "status": "not-ready",
            "reasons": [
                {
                    "entry": None,
                    "fiber": None,
                    "uid": None,
                    "state": None,
                    "kind": "no-loader",
                    "service": None,
                    "detail": "no loader service is running",
                }
            ],
        }


async def test_readyz_is_not_ready_until_the_loader_has_started(
    loadfix: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "app.toml"
    path.write_text(textwrap.dedent(LOADER_TREE), encoding="utf-8")
    async with serving() as (root, health):
        await root.plugin(Loader, LoaderConfig(path=str(path)))
        loader: Loader = root.loader
        assert not loader.started
        response = await get(health.port, "/readyz")
        assert response.status == 503
        assert response.json() == {
            "status": "not-ready",
            "reasons": [
                {
                    "entry": None,
                    "fiber": None,
                    "uid": None,
                    "state": None,
                    "kind": "not-started",
                    "service": None,
                    "detail": "the loader has not loaded its config yet",
                }
            ],
        }
        await loader.start()
        await settle(root)
        assert loader.started
        response = await get(health.port, "/readyz")
        assert (response.status, response.json()) == (200, {"status": "ready"})


async def test_readyz_is_ready_when_every_enabled_entry_is_active(
    loadfix: ModuleType, tmp_path: Path
) -> None:
    async with serving() as (root, health):
        await start_loader(root, tmp_path / "app.toml", LOADER_TREE)
        response = await get(health.port, "/readyz")
        assert (response.status, response.json()) == (200, {"status": "ready"})


async def test_readyz_reports_stuck_and_missing_entries(
    loadfix: ModuleType, tmp_path: Path
) -> None:
    tree = """
    [[plugins]]
    id = "dep"
    use = "loadfix:dependent"

    [[plugins]]
    id = "gone"
    use = "loadfix:nope"
    """
    async with serving() as (root, health):
        loader = await start_loader(root, tmp_path / "app.toml", tree)
        dep = loader.get("dep")
        assert dep is not None and dep.fiber is not None
        response = await get(health.port, "/readyz")
        assert response.status == 503
        assert response.json() == {
            "status": "not-ready",
            "reasons": [
                {
                    "entry": "dep",
                    "fiber": "dependent",
                    "uid": dep.fiber.uid,
                    "state": "PENDING",
                    "kind": "missing-service",
                    "service": "llm",
                    "detail": 'no plugin declares or currently provides "llm"',
                },
                {
                    "entry": "gone",
                    "fiber": None,
                    "uid": None,
                    "state": None,
                    "kind": "not-started",
                    "service": None,
                    "detail": "the entry has no running fiber (see the logs)",
                },
            ],
        }


async def test_readyz_withholds_error_messages(loadfix: ModuleType, tmp_path: Path) -> None:
    tree = """
    [[plugins]]
    id = "boom"
    use = "loadfix:failing"
    """
    async with serving() as (root, health):
        await start_loader(root, tmp_path / "app.toml", tree)
        response = await get(health.port, "/readyz")
        assert response.status == 503
        assert b"body failed" not in response.body
        (reason,) = response.json()["reasons"]
        assert (reason["entry"], reason["state"], reason["kind"]) == ("boom", "FAILED", "failed")
        assert reason["detail"] is None


async def test_readyz_all_requires_every_fiber_active() -> None:
    async with serving(ready_requires="all") as (root, health):
        assert (await get(health.port, "/readyz")).status == 200
        app = root.plugin(plugin("app", inject=["db"]))
        root.plugin(plugin("broken", apply=raising("password=s3cret")))
        await settle(root)
        response = await get(health.port, "/readyz")
        assert response.status == 503
        assert b"s3cret" not in response.body
        reasons = response.json()["reasons"]
        assert [(r["fiber"], r["state"], r["kind"], r["service"]) for r in reasons] == [
            ("app", "PENDING", "missing-service", "db"),
            ("broken", "FAILED", "failed", None),
        ]
        assert reasons[0]["uid"] == app.uid
        assert all("entry" not in reason for reason in reasons)


async def test_readyz_caps_the_number_of_reasons() -> None:
    async with serving(ready_requires="all") as (root, health):
        for index in range(60):
            root.plugin(plugin(f"app{index}", inject=["a", "b"]))
        await settle(root)
        body = (await get(health.port, "/readyz")).json()
        assert len(body["reasons"]) == 100
        assert body["omitted"] == 20


# -- hostile clients (Review Focus 4) -------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        b"GET /healthz HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody",
        b"GET /healthz HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
        b"GET /healthz HTTP/1.1\r\nContent-Length: nope\r\n\r\n",
        b"GET /healthz HTTP/1.1\r\nContent-Length: " + b"9" * 5000 + b"\r\n\r\n",
        b"GET /healthz HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n",
        b"GET /healthz\r\n\r\n",
        b"GET  /healthz HTTP/1.1\r\n\r\n",
        b"GET /healthz HTTP/2.0\r\n\r\n",
        b"GET /healthz HTTP/1.1 extra\r\n\r\n",
        b"GET http://evil.example/healthz HTTP/1.1\r\n\r\n",
        b"GET /healthz HTTP/1.1\r\nno colon here\r\n\r\n",
        b"GET /healthz HTTP/1.1\r\nHost : test\r\n\r\n",
        b"GET /healthz HTTP/1.1\r\nX: a\r\n folded\r\n\r\n",
        b"G\xffT /healthz HTTP/1.1\r\n\r\n",
        b"GET /\xff<script> HTTP/1.1\r\n\r\n",
        b"\r\nGET /healthz HTTP/1.1\r\n\r\n",
        b"GET /healthz HTTP/1.1\nHost: test\n\n",
        b"GET /healthz HTTP/1.1\r\nHost: test\n\r\n",
    ],
)
async def test_malformed_requests_are_400_without_reflection(raw: bytes) -> None:
    async with serving() as (_root, health):
        response = await send(health.port, raw)
        assert response.status == 400
        assert response.json() == {"error": "bad request"}
        assert b"evil" not in response.body and b"script" not in response.body


async def test_empty_content_length_zero_is_accepted() -> None:
    async with serving() as (_root, health):
        for length in (b"0", b"000"):
            raw = b"GET /healthz HTTP/1.1\r\nContent-Length: " + length + b"\r\n\r\n"
            assert (await send(health.port, raw)).status == 200


async def test_oversized_request_is_rejected_and_closed() -> None:
    async with serving() as (root, health):
        await root.plugin(plugin("app"))
        raw = b"GET /healthz HTTP/1.1\r\nX-Big: " + b"a" * 9000 + b"\r\n\r\n"
        response = await send(health.port, raw)
        assert response.status == 431
        assert response.json() == {"error": "request too large"}
        # endless headers without the blank line: also cut off at the limit
        endless = b"GET /healthz HTTP/1.1\r\n" + b"X: y\r\n" * 2000
        assert (await send(health.port, endless)).status == 431
        assert (await get(health.port, "/healthz")).status == 200
        assert root.diagnose().healthy


async def test_slow_client_is_timed_out() -> None:
    async with serving(timeout=0.4, head_timeout=0.2) as (root, health):
        reader, writer = await asyncio.open_connection("127.0.0.1", health.port)
        writer.write(b"GET /healthz HTTP/1.1\r\n")  # never finishes the headers
        started = time.monotonic()
        assert await asyncio.wait_for(reader.read(), 5) == b""
        assert time.monotonic() - started < 2
        writer.close()
        await writer.wait_closed()
        assert (await get(health.port, "/healthz")).status == 200
        assert root.fiber.state == FiberState.ACTIVE


async def test_client_disconnecting_mid_request_is_harmless() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        health = await start_health(root)
        _reader, writer = await asyncio.open_connection("127.0.0.1", health.port)
        writer.write(b"GET /hea")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        assert (await get(health.port, "/healthz")).status == 200
    assert logs.text("error") == []


async def test_connections_beyond_the_cap_are_closed_at_once() -> None:
    async with serving(max_connections=2, timeout=5.0) as (_root, health):
        idle = [await asyncio.open_connection("127.0.0.1", health.port) for _ in range(2)]
        await asyncio.sleep(0.05)  # let the server accept both
        reader, writer = await asyncio.open_connection("127.0.0.1", health.port)
        with contextlib.suppress(ConnectionError):
            assert await asyncio.wait_for(reader.read(), 2) == b""
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()
        for _reader, idle_writer in idle:
            idle_writer.close()
            await idle_writer.wait_closed()
        await asyncio.sleep(0.05)  # the idle connections are released
        assert (await get(health.port, "/healthz")).status == 200


# -- binding and lifecycle ------------------------------------------------------------


async def test_binds_loopback_by_default() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        health = await start_health(root)
        assert health.host == "127.0.0.1"
        assert health.port > 0
        assert f"health server listening on http://127.0.0.1:{health.port}" in logs.text("info")
        assert logs.text("warn") == []


async def test_binding_all_interfaces_warns() -> None:
    logs = MemoryExporter()
    async with mock_context(logs) as root:
        health = await start_health(root, host="0.0.0.0")
        assert health.host == "0.0.0.0"
        (warning,) = logs.text("warn")
        assert "0.0.0.0" in warning and "not loopback" in warning


async def test_port_before_listening_raises() -> None:
    async with mock_context() as root:
        health = HealthServer(root, HealthServer.Config())
        with pytest.raises(RuntimeError, match="not listening"):
            _ = health.port
        with pytest.raises(RuntimeError, match="not listening"):
            _ = health.host


async def test_dispose_closes_the_server_and_open_connections() -> None:
    async with mock_context() as root:
        fiber = await root.plugin(HealthServer)
        health: HealthServer = root.health
        port = health.port
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /healthz HTTP/1.1\r\n")  # in flight during dispose
        await writer.drain()
        await asyncio.sleep(0.05)
        started = time.monotonic()
        await fiber.dispose()
        assert time.monotonic() - started < 1
        with contextlib.suppress(ConnectionError):
            assert await asyncio.wait_for(reader.read(), 2) == b""
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()
        with pytest.raises(OSError):
            await asyncio.open_connection("127.0.0.1", port)


async def test_a_port_in_use_fails_the_plugin() -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = cast("tuple[str, int]", taken.getsockname())[1]
        async with mock_context() as root:
            fiber = root.plugin(HealthServer, HealthServer.Config(port=port))
            with pytest.raises(OSError):
                await fiber
            assert fiber.state == FiberState.FAILED


async def test_config_comes_from_a_mapping() -> None:
    async with mock_context() as root:
        await root.plugin(HealthServer, {"ready_requires": "all", "timeout": 2})
        health: HealthServer = root.health
        assert (await get(health.port, "/healthz")).status == 200


@pytest.mark.parametrize(
    "config",
    [
        {"port": -1},
        {"port": 65536},
        {"host": ""},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
        {"head_timeout": 0},
        {"timeout": 3.0, "head_timeout": 4.0},
        {"max_connections": 0},
        {"ready_requires": "some"},
    ],
)
async def test_invalid_config_is_rejected(config: dict[str, Any]) -> None:
    async with mock_context() as root:
        fiber = root.plugin(HealthServer, config)
        with pytest.raises((ValidationError, ValueError)):
            await fiber


async def test_an_endpoint_error_is_a_logged_500(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_self: Context) -> Diagnosis:
        raise RuntimeError("diagnosis broke: s3cret")

    logs = MemoryExporter()
    async with mock_context(logs) as root:
        health = await start_health(root, ready_requires="all")
        monkeypatch.setattr(Context, "diagnose", broken)
        response = await get(health.port, "/readyz")
        assert (response.status, response.json()) == (500, {"error": "internal error"})
        assert b"s3cret" not in response.body
        assert any("diagnosis broke" in text for text in logs.text("error"))
        assert (await get(health.port, "/healthz")).status == 200


async def test_a_fiber_not_active_without_reasons_is_still_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # e.g. a PENDING fiber diagnosed just before it picks up its last service
    waiting = Diagnosis("app", 7, FiberState.PENDING, (), ())

    def diagnose(_self: Context) -> Diagnosis:
        return Diagnosis("root", 0, FiberState.ACTIVE, (), (waiting,))

    async with serving(ready_requires="all") as (_root, health):
        monkeypatch.setattr(Context, "diagnose", diagnose)
        response = await get(health.port, "/readyz")
        assert response.status == 503
        assert response.json()["reasons"] == [
            {
                "fiber": "app",
                "uid": 7,
                "state": "PENDING",
                "kind": "not-active",
                "service": None,
                "detail": None,
            }
        ]


async def test_long_details_are_clipped() -> None:
    async with serving(ready_requires="all") as (root, health):
        root.plugin(plugin("app", inject=["x" * 300]))
        await settle(root)
        (reason,) = (await get(health.port, "/readyz")).json()["reasons"]
        assert len(reason["detail"]) == 200
        assert reason["detail"].endswith("...")


def test_config_checks_ready_requires_when_built_directly() -> None:
    options: dict[str, Any] = {"ready_requires": "some"}
    with pytest.raises(ValueError, match="ready_requires"):
        HealthServer.Config(**options)


async def test_a_large_rejected_body_is_not_read_forever() -> None:
    async with serving() as (_root, health):
        raw = b"GET /healthz HTTP/1.1\r\nContent-Length: 300000\r\n\r\n" + b"x" * 300_000
        with contextlib.suppress(ConnectionError):
            response = await send(health.port, raw)
            assert response.status == 400
        assert (await get(health.port, "/healthz")).status == 200


async def test_a_client_not_reading_its_response_is_cut_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huge = FiberNode("root", 0, FiberState.ACTIVE, ["x" * 1000] * 20_000, [], [])

    def inspect(_self: Context) -> FiberNode:
        return huge

    async with serving(timeout=0.2, head_timeout=0.2) as (_root, health):
        monkeypatch.setattr(Context, "inspect", inspect)
        _reader, writer = await asyncio.open_connection("127.0.0.1", health.port)
        writer.write(b"GET /inspect HTTP/1.1\r\n\r\n")  # and never read the 20 MB
        await writer.drain()
        await asyncio.sleep(0.5)
        monkeypatch.undo()
        assert (await get(health.port, "/healthz")).status == 200
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


async def test_an_idle_connection_frees_its_slot_after_the_head_timeout() -> None:
    async with serving(max_connections=1, timeout=5.0, head_timeout=0.2) as (_root, health):
        reader, writer = await asyncio.open_connection("127.0.0.1", health.port)
        started = time.monotonic()
        assert await asyncio.wait_for(reader.read(), 2) == b""  # sent nothing at all
        assert time.monotonic() - started < 1.5
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)
        assert (await get(health.port, "/healthz")).status == 200


async def test_lf_only_requests_are_rejected_without_waiting() -> None:
    async with serving(timeout=5.0, head_timeout=3.0) as (_root, health):
        started = time.monotonic()
        response = await send(health.port, b"GET /healthz HTTP/1.1\n\n")
        assert response.status == 400
        assert time.monotonic() - started < 1


async def test_a_bind_failure_is_logged_in_one_line() -> None:
    logs = MemoryExporter()
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = cast("tuple[str, int]", taken.getsockname())[1]
        async with mock_context(logs) as root:
            with pytest.raises(OSError):
                await root.plugin(HealthServer, HealthServer.Config(port=port))
    (error,) = logs.text("error")
    assert f"cannot listen on 127.0.0.1:{port}: " in error
    assert "Traceback" not in error


def test_head_timeout_defaults_to_two_seconds_capped_by_timeout() -> None:
    assert HealthServer.Config().head_deadline == 2.0
    assert HealthServer.Config(timeout=1.0).head_deadline == 1.0
    assert HealthServer.Config(timeout=3.0, head_timeout=0.5).head_deadline == 0.5


async def test_a_short_timeout_alone_is_accepted() -> None:
    async with mock_context() as root:
        await root.plugin(HealthServer, {"timeout": 1.0})
        health: HealthServer = root.health
        assert (await get(health.port, "/healthz")).status == 200


def test_an_infinite_timeout_is_named_in_the_error() -> None:
    with pytest.raises(ValueError, match="timeout must be a positive finite number, got inf"):
        HealthServer.Config(timeout=float("inf"))
