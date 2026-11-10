"""A dependency-free health and readiness HTTP server (spec §3.8).

:class:`HealthServer` is a plugin providing the ``health`` service: it listens
on ``127.0.0.1`` (an ephemeral port by default) while its fiber is loaded and
answers three read-only endpoints with small JSON documents::

    await ctx.plugin(HealthServer, HealthServer.Config(port=8080))
    # GET http://127.0.0.1:8080/readyz  ->  200 {"status":"ready"}

The CLI starts one with ``pycordis start --health-port PORT [--health-host HOST]``.

What each endpoint exposes (and nothing else):

- ``GET /healthz``: ``200 {"status":"ok"}`` while the process runs its event loop.
- ``GET /readyz``: ``200 {"status":"ready"}``, or ``503 {"status":"not-ready",
  "reasons":[...]}``. With ``ready_requires="loader"`` (the default) it is ready
  when the ``loader`` service exists, has started (``Loader.start()`` loaded
  its config) and every enabled loader entry has an ``ACTIVE`` fiber; with
  ``"all"`` when every fiber of the application is ``ACTIVE``. Each reason is an
  object with the fiber's ``name`` (key ``fiber``), ``uid`` and ``state``, the
  :class:`~pycordis.diagnose.Reason` ``kind``, the ``service`` concerned and a
  short ``detail`` (plus ``entry``, the loader entry id, in ``"loader"`` mode).
  Reasons that would carry an error message (``failed``, ``provider-failed``,
  ``check-failed``) have a ``null`` detail: error texts can hold secrets; read
  them from the logs or ``pycordis doctor``. Two more kinds exist here: ``no-loader`` and
  ``not-started`` (with a ``null`` entry: the loader has not started yet;
  with an entry id: an enabled entry without a fiber, its import or start
  failed). At most 100 reasons are listed; ``omitted`` counts the rest.
- ``GET /inspect``: ``200`` with the fiber tree: per fiber ``name``, ``uid``,
  ``state``, ``provides``, ``injects`` and ``children``. Configs are never
  included (they may hold secrets). The tree is built synchronously on the
  event loop for each request: cheap for typical trees, proportional to the
  number of fibers.

The server is deliberately minimal and closes every connection after one
response (``Connection: close``). It reads only a request line and headers
(lines ending in CRLF), at most 8 KiB of them (``431`` beyond); it answers
``400`` to anything malformed (a bare-LF line at once, without waiting for
more), to a request with a body (``Content-Length`` above 0, or any
``Transfer-Encoding``) and to a target that is not a plain path; any method but
``GET`` gets ``405``. Paths match exactly after dropping a query string, so
``/healthz/`` is a ``404``. After a ``400`` or ``431`` up to 64 KiB of what the
client still sends is read and dropped so the response arrives intact; a client
sending more (e.g. a head far beyond 8 KiB) gets its connection reset instead,
possibly before it reads the response. Responses never repeat request data.
Every response is JSON with ``Content-Length``, ``Cache-Control: no-store`` and
``X-Content-Type-Options: nosniff``.

Limits: the request head must arrive within ``head_timeout`` seconds (default
2, or ``timeout`` if that is shorter) and the whole exchange within
``timeout`` (default 5), and at most ``max_connections`` (default 64) are
served at once: more are closed unanswered. Idle connections so free their
slot quickly, but a peer that can reach the port can still keep every slot busy
(opening connections faster than they time out) and make probes fail: never
expose the port to untrusted networks.

There is no authentication: binding an address that is not loopback (e.g.
``host="0.0.0.0"``) must be asked for and logs a warning. A failure to listen
(e.g. the port is in use) is logged as one line and fails the plugin. Unloading
the plugin closes the listening socket and every open connection.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import math
import re
from collections.abc import AsyncGenerator, Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pycordis.fiber import FiberState
from pycordis.fiber_core import mark_reported
from pycordis.loader.service import LoadedEntry, Loader
from pycordis.service import Service

if TYPE_CHECKING:
    from pycordis.context import Context
    from pycordis.diagnose import Diagnosis, Reason, ReasonKind
    from pycordis.inspect import FiberNode

__all__ = ["HealthServer"]

MAX_HEAD = 8192
"""Most bytes of request line and headers read (431 beyond)."""
MAX_REASONS = 100
"""Most reasons listed by ``/readyz``."""
DEFAULT_HEAD_TIMEOUT = 2.0
"""Seconds allowed for a request head when ``head_timeout`` is not set (capped by ``timeout``)."""
_MAX_DETAIL = 200
_MAX_DISCARD = 65536
"""Most unread request bytes swallowed before closing after a rejection."""
_WITHHELD: frozenset[ReasonKind] = frozenset({"failed", "provider-failed", "check-failed"})
"""Reason kinds whose detail carries an error message."""
_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_TARGET = re.compile(r"/[\x21-\x7e]*")
_FIELD_VALUE = re.compile(r"[\t\x20-\x7e]*")
_STATUS = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    431: "Request Header Fields Too Large",
    500: "Internal Server Error",
    503: "Service Unavailable",
}

type _Json = dict[str, object]


class HealthServer(Service[Any], provide="health"):
    """Serve ``/healthz``, ``/readyz`` and ``/inspect`` over HTTP (see :mod:`pycordis.health`).

    Used as a class plugin; once its fiber is ``ACTIVE`` the server listens and
    :attr:`port` is the bound port (useful with the default ``port=0``)::

        await ctx.plugin(HealthServer, HealthServer.Config(port=0))
        health: HealthServer = ctx.health
        print(f"curl http://127.0.0.1:{health.port}/readyz")

    A port already in use fails the plugin with the ``OSError``.
    """

    @dataclass
    class Config:
        """Options of :class:`HealthServer`.

        ``host`` is the address to bind (only loopback by default; ``"0.0.0.0"``
        exposes the server on every interface, with a warning); ``port`` the
        TCP port (``0``: any free one). ``ready_requires`` chooses what
        ``/readyz`` waits for: ``"loader"`` (every enabled loader entry
        ``ACTIVE``) or ``"all"`` (every fiber ``ACTIVE``). ``timeout`` bounds a
        connection in seconds, ``head_timeout`` (at most ``timeout``) the
        wait for its request line and headers (``None``: 2 seconds, or
        ``timeout`` if shorter; see :attr:`head_deadline`),
        ``max_connections`` the connections served at once. Example::

            HealthServer.Config(port=8080, ready_requires="all")
            HealthServer.Config(timeout=1.0).head_deadline == 1.0
        """

        host: str = "127.0.0.1"
        port: int = 0
        ready_requires: Literal["loader", "all"] = "loader"
        timeout: float = 5.0
        head_timeout: float | None = None
        max_connections: int = 64

        def __post_init__(self) -> None:
            if not self.host:
                raise ValueError("host must not be empty (that would bind every interface)")
            if not 0 <= self.port <= 65535:
                raise ValueError(f"port must be between 0 and 65535, got {self.port}")
            if self.ready_requires not in ("loader", "all"):
                raise ValueError(
                    f'ready_requires must be "loader" or "all", got {self.ready_requires!r}'
                )
            # finite: an infinite timeout would let one idle client hold its slot forever
            if not 0 < self.timeout < math.inf:
                raise ValueError(f"timeout must be a positive finite number, got {self.timeout}")
            if self.head_timeout is not None and not 0 < self.head_timeout <= self.timeout:
                raise ValueError(
                    f"head_timeout must be positive and at most timeout ({self.timeout}), "
                    f"got {self.head_timeout}"
                )
            if self.max_connections < 1:
                raise ValueError(f"max_connections must be at least 1, got {self.max_connections}")

        @property
        def head_deadline(self) -> float:
            """Seconds allowed for the request head: ``head_timeout``, else ``min(2, timeout)``.

            Example: ``HealthServer.Config(timeout=1.0).head_deadline == 1.0``.
            """
            if self.head_timeout is None:
                return min(DEFAULT_HEAD_TIMEOUT, self.timeout)
            return self.head_timeout

    def __init__(self, ctx: Context, config: HealthServer.Config | None = None) -> None:
        super().__init__(ctx)
        self._config = config if config is not None else HealthServer.Config()
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.Task[Any]] = set()  # Task is invariant: any result type
        self._closing = False

    def __cordis_init__(self) -> contextlib.AbstractAsyncContextManager[None]:
        """The plugin's effect: listen while loaded (the load waits until bound)."""
        return self._listening()

    @property
    def port(self) -> int:
        """The TCP port the server listens on; ``RuntimeError`` while it is not listening.

        Example: ``f"http://127.0.0.1:{ctx.health.port}/readyz"``.
        """
        port: int = self._address()[1]
        return port

    @property
    def host(self) -> str:
        """The address the server listens on; ``RuntimeError`` while it is not listening.

        Example: ``ctx.health.host == "127.0.0.1"``.
        """
        return self._address()[0]

    def _address(self) -> tuple[str, int]:
        """``(host, port)`` of the listening socket (IPv6 extras dropped)."""
        if self._server is None or not self._server.sockets:
            raise RuntimeError("the health server is not listening")
        address: tuple[str, int] = self._server.sockets[0].getsockname()[:2]
        return address

    # -- lifecycle ---------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _listening(self) -> AsyncGenerator[None]:
        """Listen while the effect lives; on exit close the server and every connection."""
        config = self._config
        # `limit` bounds what readuntil() buffers: the request head limit
        try:
            server = await asyncio.start_server(
                self._serve, config.host, config.port, limit=MAX_HEAD
            )
        except OSError as error:
            # an environment problem, not a bug: one line instead of a traceback
            reason = error.strerror or str(error)
            self.ctx.logger("health").error(
                "cannot listen on %s:%d: %s", config.host, config.port, reason
            )
            mark_reported(error)
            raise
        self._server = server
        self._closing = False
        self._announce(server)
        try:
            yield
        finally:
            await self._close(server)

    def _announce(self, server: asyncio.Server) -> None:
        logger = self.ctx.logger("health")
        for sock in server.sockets:
            # (host, port) for IPv4, (host, port, flowinfo, scope_id) for IPv6
            address: tuple[str, int] = sock.getsockname()[:2]
            host, port = address
            url = f"http://{f'[{host}]' if ':' in host else host}:{port}"
            logger.info("health server listening on %s", url)
            if not _is_loopback(host):
                logger.warn(
                    "health server address %s is not loopback: the unauthenticated "
                    "endpoints are reachable from other machines, and any peer can keep "
                    "all %d connection slots busy so probes fail; do not expose it to "
                    "untrusted networks",
                    host,
                    self._config.max_connections,
                )

    async def _close(self, server: asyncio.Server) -> None:
        self._closing = True
        server.close()
        connections = list(self._connections)
        for task in connections:
            task.cancel()
        await asyncio.gather(*connections, return_exceptions=True)
        # also waits for accepted connections whose handler had not started yet
        await server.wait_closed()
        self._server = None

    # -- connections -------------------------------------------------------------------

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Answer one request on a new connection, then close it."""
        if self._closing or len(self._connections) >= self._config.max_connections:
            writer.close()
            return
        task = asyncio.current_task()
        assert task is not None, "stream handlers run as tasks"
        self._connections.add(task)
        try:
            async with asyncio.timeout(self._config.timeout):
                response, rejected = await self._respond(reader)
                writer.write(response)
                if rejected:
                    await _discard_rest(reader, writer)
                writer.close()
                await writer.wait_closed()
        except (TimeoutError, OSError, asyncio.IncompleteReadError):
            pass  # a slow or vanished client: just close
        finally:
            self._connections.discard(task)
            _hang_up(writer)

    async def _respond(self, reader: asyncio.StreamReader) -> tuple[bytes, bool]:
        """The response to the request on ``reader``, and whether the request was rejected."""
        try:
            async with asyncio.timeout(self._config.head_deadline):
                head = await _read_head(reader)
        except asyncio.LimitOverrunError:
            return _error(431, "request too large"), True
        if head is None:
            return _error(400, "bad request"), True
        request = _parse_head(head)
        if request is None:
            return _error(400, "bad request"), True
        method, path = request
        if method != "GET":
            return _error(405, "method not allowed", allow="GET"), False
        route = self._routes().get(path)
        if route is None:
            return _error(404, "not found"), False
        try:
            status, body = route()
        except Exception as error:
            self.ctx.logger("health").error(error)
            return _error(500, "internal error"), False
        return _response(status, body), False

    def _routes(self) -> dict[str, Callable[[], tuple[int, _Json]]]:
        return {"/healthz": _healthz, "/readyz": self._readyz, "/inspect": self._inspect}

    # -- endpoints ---------------------------------------------------------------------

    def _readyz(self) -> tuple[int, _Json]:
        if self._config.ready_requires == "all":
            reasons = list(_all_reasons(self.ctx.root.diagnose()))
        else:
            reasons = list(self._loader_reasons())
        if not reasons:
            return 200, {"status": "ready"}
        body: _Json = {"status": "not-ready", "reasons": reasons[:MAX_REASONS]}
        if len(reasons) > MAX_REASONS:
            body["omitted"] = len(reasons) - MAX_REASONS
        return 503, body

    def _loader_reasons(self) -> Iterator[_Json]:
        loader: object = self.ctx.get("loader")
        if not isinstance(loader, Loader):
            yield {"entry": None} | _bare_reason("no-loader", "no loader service is running")
            return
        if not loader.started:
            detail = "the loader has not loaded its config yet"
            yield {"entry": None} | _bare_reason("not-started", detail)
            return
        for loaded in _enabled(loader.entries()):
            fiber = loaded.fiber
            if fiber is None:
                detail = "the entry has no running fiber (see the logs)"
                yield {"entry": loaded.entry.id} | _bare_reason("not-started", detail)
            elif fiber.state != FiberState.ACTIVE:
                for reason in _fiber_reasons(fiber.diagnose()):
                    yield {"entry": loaded.entry.id} | reason

    def _inspect(self) -> tuple[int, _Json]:
        return 200, _tree(self.ctx.root.inspect())


# -- request parsing -------------------------------------------------------------------


async def _read_head(reader: asyncio.StreamReader) -> bytes | None:
    """Read a request head line by line up to its blank line; ``None`` at a bare-LF line.

    Raises ``LimitOverrunError`` past :data:`MAX_HEAD` bytes (``reader``'s own
    limit bounds each line to that too).
    """
    head = bytearray()
    while not head.endswith(b"\r\n\r\n"):
        line = await reader.readuntil(b"\n")
        head += line
        if len(head) > MAX_HEAD:
            raise asyncio.LimitOverrunError("request head too large", len(head))
        if not line.endswith(b"\r\n"):
            return None
    return bytes(head)


def _parse_head(head: bytes) -> tuple[str, str] | None:
    """``(method, path)`` of a request head (ending in a blank line), ``None`` if malformed.

    Accepts HTTP/1.0 and 1.1 with an origin-form target; rejects a body
    (``Content-Length`` above 0, repeated, or any ``Transfer-Encoding``),
    obsolete line folding and non-ASCII bytes. The query string is dropped.
    """
    try:
        text = head[: -len(b"\r\n\r\n")].decode("ascii")
    except UnicodeDecodeError:
        return None
    request_line, *fields = text.split("\r\n")
    parts = request_line.split(" ")
    if len(parts) != 3:
        return None
    method, target, version = parts
    if (
        not _TOKEN.fullmatch(method)
        or not _TARGET.fullmatch(target)
        or version not in ("HTTP/1.0", "HTTP/1.1")
        or not _fields_allowed(fields)
    ):
        return None
    return method, target.partition("?")[0]


def _fields_allowed(fields: list[str]) -> bool:
    """Whether the header lines are well formed and announce no request body."""
    lengths = 0
    for line in fields:
        name, colon, value = line.partition(":")
        if not colon or not _TOKEN.fullmatch(name) or not _FIELD_VALUE.fullmatch(value):
            return False  # also a folded line: it starts with whitespace
        name = name.lower()
        value = value.strip(" \t")
        if name == "transfer-encoding":
            return False
        if name == "content-length":
            lengths += 1
            # zeros only: never int() an attacker-sized number
            if lengths > 1 or not value.isdigit() or value.strip("0"):
                return False
    return True


async def _discard_rest(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """After a rejection: end our side and swallow what the client still sends.

    Closing a socket with unread input resets the connection, which can destroy
    the response before the client reads it. Bounded by :data:`_MAX_DISCARD`
    bytes (and the connection timeout).
    """
    writer.write_eof()
    discarded = 0
    while discarded < _MAX_DISCARD:
        chunk = await reader.read(4096)
        if not chunk:
            return
        discarded += len(chunk)


def _hang_up(writer: asyncio.StreamWriter) -> None:
    """Close the connection: gracefully if all is sent, else at once (no waiting on a client)."""
    transport = writer.transport
    if transport.get_write_buffer_size():
        transport.abort()
    else:
        transport.close()


# -- responses -------------------------------------------------------------------------


def _response(status: int, body: _Json, *, allow: str | None = None) -> bytes:
    payload = json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    lines = [
        f"HTTP/1.1 {status} {_STATUS[status]}",
        "Content-Type: application/json",
        f"Content-Length: {len(payload)}",
        "Cache-Control: no-store",
        "X-Content-Type-Options: nosniff",
        "Connection: close",
    ]
    if allow is not None:
        lines.append(f"Allow: {allow}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + payload


def _error(status: int, message: str, *, allow: str | None = None) -> bytes:
    return _response(status, {"error": message}, allow=allow)


def _healthz() -> tuple[int, _Json]:
    return 200, {"status": "ok"}


def _tree(node: FiberNode) -> _Json:
    return {
        "name": node.name,
        "uid": node.uid,
        "state": node.state.name,
        "provides": node.provides,
        "injects": node.injects,
        "children": [_tree(child) for child in node.children],
    }


def _enabled(entries: list[LoadedEntry]) -> list[LoadedEntry]:
    """The entries that should be running: not disabled, nor inside a disabled group."""
    inactive: set[str] = set()
    for loaded in entries:
        if loaded.entry.disabled:
            stack = list(loaded.entry.children)
            while stack:
                child = stack.pop()
                inactive.add(child.id)
                stack.extend(child.children)
    return [
        loaded
        for loaded in entries
        if not loaded.entry.disabled and loaded.entry.id not in inactive
    ]


def _all_reasons(diagnosis: Diagnosis) -> Iterator[_Json]:
    """The reasons of every fiber in the tree that is not ACTIVE, by uid order."""
    stack = [diagnosis]
    while stack:
        node = stack.pop()
        if node.state != FiberState.ACTIVE:
            yield from _fiber_reasons(node)
        stack.extend(reversed(node.children))


def _fiber_reasons(node: Diagnosis) -> Iterator[_Json]:
    """One reason object per reason of ``node`` (one ``not-active`` if it has none)."""
    fiber: _Json = {"fiber": node.fiber, "uid": node.uid, "state": node.state.name}
    if not node.reasons:
        yield fiber | {"kind": "not-active", "service": None, "detail": None}
    for reason in node.reasons:
        yield fiber | _reason(reason)


def _reason(reason: Reason) -> _Json:
    detail = None if reason.kind in _WITHHELD else _clip(reason.detail)
    return {"kind": reason.kind, "service": reason.service, "detail": detail}


def _bare_reason(kind: str, detail: str) -> _Json:
    """A reason about no particular fiber."""
    return {
        "fiber": None,
        "uid": None,
        "state": None,
        "kind": kind,
        "service": None,
        "detail": detail,
    }


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_DETAIL else text[: _MAX_DETAIL - 3] + "..."


def _is_loopback(host: str) -> bool:
    """Whether a bound socket address (always an IP literal) is a loopback address."""
    return ipaddress.ip_address(host).is_loopback
