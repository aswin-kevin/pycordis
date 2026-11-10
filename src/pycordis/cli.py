"""The ``pycordis`` command line (spec §6.6 CLI); also ``python -m pycordis``.

``pycordis [start] [-c pycordis.toml] [--watch] [--allow PATTERN]
[--log-level LEVEL] [--dispose-timeout SECONDS]`` creates a root context,
installs a :class:`~pycordis.logger.ConsoleExporter` and the
:class:`~pycordis.loader.Loader`, loads the config file and runs until
SIGINT or SIGTERM. The first signal logs ``shutting down`` and disposes the
root; a second one (of either kind) exits immediately with ``os._exit(130)``,
so a forced exit is always 130. Signals are handled on the event loop, so a
disposer that blocks the loop synchronously delays signal handling until it
returns. The console exporter stays registered through the disposal, so
errors logged by disposers are shown. ``--dispose-timeout`` (default 10
seconds; ``0`` disables it) bounds the unload of every entry that sets no
``dispose_timeout`` of its own (see :class:`~pycordis.loader.LoaderConfig`), so
a hung disposer cannot block shutdown: it is abandoned with an error log. A
disposer that blocks the event loop synchronously cannot be abandoned.

``--profile NAME`` (every subcommand) selects config overlays: each config
file ``<stem><suffix>`` is layered with ``<stem>.NAME<suffix>`` next to it
(see :class:`~pycordis.loader.LoaderConfig`). Without the flag the
``PYCORDIS_PROFILE`` environment variable is used when set and non-empty; the
flag wins. A name other than letters, digits, ``_`` and ``-`` is a usage error
for the flag and a startup error for the variable.

``--health-port PORT`` (``start`` only; ``0`` picks a free port) also runs a
:class:`~pycordis.health.HealthServer` at the root, started before the loader,
on ``--health-host`` (default ``127.0.0.1``; another address must be given
explicitly and logs a warning): ``/healthz``, ``/readyz`` (ready once every
enabled loader entry is ACTIVE) and ``/inspect``. A port that cannot be bound is
a startup error.

``pycordis inspect -c FILE`` is a static dry run: it parses the entry tree,
imports each enabled entry's plugin (honouring ``allow``) and checks its config
against the plugin's ``Config`` (after interpolation) — without calling any
plugin. Import, ``allow`` and config errors print ``✗ {id}: ...`` lines and make
the exit code 1. Injections that no enabled entry declares with ``provide``
print ``⚠ {id}: cannot inject ...`` warnings that do not affect the exit code:
services a plugin provides only from its body (``ctx.provide(...)``) cannot be
seen statically. Isolation is honoured: a service provided only inside an
entry whose ``isolate`` covers its name does not satisfy an injection outside that
scope (and vice versa; equal labels share a scope), which is reported as
``cannot inject "<name>": only provided in another isolated scope``.
``pycordis inspect --run`` starts the tree, waits for it to settle, prints
``render(root.inspect())`` and disposes it; SIGINT/SIGTERM are handled as for
``start``.

``pycordis schema [-c FILE] [--overlay] [-o OUT.json]`` prints (or writes to ``OUT.json``) the
JSON Schema of the config file format (:func:`~pycordis.jsonschema.loader_json_schema`),
each entry's ``config`` typed from its plugin's dataclass ``Config``. Plugins are
imported (honouring ``allow``) to find it; one that cannot be imported is noted on
stderr and left untyped. ``--overlay`` gives the schema of profile overlay files
(partial entries matched by ``id``; see
:func:`~pycordis.jsonschema.loader_json_schema`). A config that cannot be loaded
exits 1.

``pycordis doctor -c FILE [--timeout SECONDS]`` starts the tree, waits for it
to settle (at most ``--timeout`` seconds, default 10, covering startup too, so
a plugin body that never finishes cannot hang it), prints the diagnosis of
every fiber that is not ACTIVE (:func:`~pycordis.diagnose.render_diagnosis`,
loader entry fibers annotated ``[entry <id>]``, with ``still starting after
...`` first on a timeout) or ``all fibers are
ACTIVE``, and disposes the tree. It exits 0 when every fiber is ACTIVE, else 1;
signals are handled as for ``start``.

Exit codes: 0 success (also after a SIGTERM-triggered shutdown), 1 startup or
config error (``pycordis: error: ...`` on stderr, or ``✗`` lines for
``inspect``) or a fiber that is not ACTIVE for ``doctor``, 130 after a
SIGINT-triggered shutdown or a forced second-signal exit, 2 usage error
(argparse).

The working directory is appended to ``sys.path`` (unless already present), so
plugins of the current project resolve when run as the ``pycordis`` script;
being appended, files there cannot shadow standard-library or installed modules.
That guarantee is the console script's only: ``python -m pycordis`` starts with
the working directory first on ``sys.path`` (Python's own ``-m`` rule).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import final

from pycordis.context import Context
from pycordis.diagnose import render_diagnosis
from pycordis.errors import ValidationError
from pycordis.health import HealthServer
from pycordis.inspect import render
from pycordis.jsonschema import loader_json_schema
from pycordis.loader import Entry, Loader, LoaderConfig, LoaderConfigError, expand, import_plugin
from pycordis.loader.files import load_tree
from pycordis.loader.overlay import PROFILE_ENV, check_profile
from pycordis.logger import ConsoleExporter, LoggerLevel
from pycordis.registry import plugin_config_type, plugin_inject, plugin_provide
from pycordis.schema import validate_config
from pycordis.testing import settle

__all__ = ["main"]

_LEVELS = {
    "error": LoggerLevel.ERROR,
    "warn": LoggerLevel.WARN,
    "info": LoggerLevel.INFO,
    "debug": LoggerLevel.DEBUG,
}
_BUILTIN_SERVICES = frozenset({"loader"})
"""Services provided by the CLI itself (the only injectable built-in)."""
_EXIT_SIGINT = 130
_YAML_MISSING = "YAML config requires PyYAML"
_DISPOSE_TIMEOUT = 10.0
"""Default unload bound of loader entries under the CLI (spec §3.6)."""
_DOCTOR_TIMEOUT = 10.0
"""Default bound of ``doctor``'s startup and settling, in seconds."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line with ``argv`` (default ``sys.argv[1:]``); return the exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    health = args.health_port is not None or args.health_host is not None
    # no subcommand means start (``command`` is then None)
    if args.command in ("inspect", "doctor") and health:
        parser.error("--health-port and --health-host apply to start only")
    if args.health_host is not None and args.health_port is None:
        parser.error("--health-host requires --health-port")
    try:
        args.profile = _profile(args.profile)
    except _StartupError as error:
        return _startup_error(error)
    cwd = os.getcwd()
    # last: never shadows stdlib / installed modules. This holds for the `pycordis`
    # console script only; `python -m pycordis` already has the cwd first on sys.path
    if cwd not in sys.path and "" not in sys.path:
        sys.path.append(cwd)
    if args.command == "schema":
        return _schema(args)
    if args.command == "inspect" and not args.run:
        return _inspect_static(args)
    return asyncio.run(_run(args))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pycordis", description="Run a pycordis config file.")
    _add_options(parser, True)
    _add_health_options(parser, True)
    parser.set_defaults(command="start", run=False, timeout=_DOCTOR_TIMEOUT)
    commands = parser.add_subparsers(dest="command", metavar="{start,inspect,doctor,schema}")
    start = commands.add_parser("start", help="load the config and run until interrupted")
    _add_options(start, False)
    _add_health_options(start, False)
    inspect = commands.add_parser("inspect", help="check the config without running plugins")
    _add_options(inspect, False)
    inspect.add_argument(
        "--run", action="store_true", help="start the tree and print the live fiber tree"
    )
    doctor = commands.add_parser("doctor", help="start the tree and explain what is not ACTIVE")
    _add_options(doctor, False, watch=False)
    doctor.add_argument(
        "--timeout",
        type=_seconds,
        default=_DOCTOR_TIMEOUT,
        metavar="SECONDS",
        help="give up waiting for the tree to settle after this long (default: 10)",
    )
    schema = commands.add_parser("schema", help="print the JSON Schema of the config file")
    _add_options(schema, False, watch=False, runtime=False)
    schema.add_argument(
        "--overlay",
        action="store_true",
        help="schema of a profile overlay file (partial entries) instead of a base file",
    )
    schema.add_argument(
        "-o", "--output", metavar="PATH", help="write the schema to this file (default: stdout)"
    )
    return parser


def _add_options(
    parser: argparse.ArgumentParser, defaults: bool, watch: bool = True, runtime: bool = True
) -> None:
    """Add the options shared by every command (``watch`` / ``runtime``: their groups too).

    ``defaults`` is set on the top-level parser only: subcommands default to
    ``SUPPRESS`` so they never reset a top-level value. Example:
    ``_add_options(doctor, False, watch=False)``.
    """

    def default(value: object) -> object:
        return value if defaults else argparse.SUPPRESS

    parser.add_argument(
        "-c",
        "--config",
        default=default("pycordis.toml"),
        metavar="PATH",
        help="config file (default: pycordis.toml)",
    )
    if watch:
        parser.add_argument(
            "--watch",
            action="store_true",
            default=default(False),
            help="reload when the config file changes",
        )
    parser.add_argument(
        "--allow",
        action="append",
        default=default(None),
        metavar="PATTERN",
        help="allowed plugin module pattern (repeatable; overrides [loader] allow)",
    )
    if runtime:
        parser.add_argument(
            "--log-level",
            choices=list(_LEVELS),
            default=default("info"),
            help="console log level (default: info)",
        )
    parser.add_argument(
        "--profile",
        type=_profile_name,
        default=default(None),
        metavar="NAME",
        help="layer each config file with its NAME overlay (default: $PYCORDIS_PROFILE)",
    )
    if runtime:
        parser.add_argument(
            "--dispose-timeout",
            type=_seconds,
            default=default(_DISPOSE_TIMEOUT),
            metavar="SECONDS",
            help="abandon an entry's unload after this long (default: 10; 0 disables)",
        )


def _add_health_options(parser: argparse.ArgumentParser, defaults: bool) -> None:
    """Add ``--health-port`` / ``--health-host`` (see :func:`_add_options` for ``defaults``)."""
    default = None if defaults else argparse.SUPPRESS
    parser.add_argument(
        "--health-port",
        type=_port,
        default=default,
        metavar="PORT",
        help="serve /healthz, /readyz and /inspect on this port (0: any free port)",
    )
    parser.add_argument(
        "--health-host",
        type=_host,
        default=default,
        metavar="HOST",
        help="address of the health server (default: 127.0.0.1)",
    )


def _seconds(text: str) -> float:
    """Parse a ``--dispose-timeout`` / ``--timeout`` value: a non-negative number of seconds."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not value >= 0:  # also rejects nan
        raise argparse.ArgumentTypeError(f"must be a non-negative number, got {text!r}")
    return value


def _profile_name(text: str) -> str:
    """Parse a ``--profile`` value: a profile name (see :func:`check_profile`)."""
    try:
        return check_profile(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _profile(flag: str | None) -> str | None:
    """The profile in effect: ``flag``, else a non-empty ``PYCORDIS_PROFILE``, else ``None``."""
    if flag is not None:
        return flag
    name = os.environ.get(PROFILE_ENV) or None
    if name is None:
        return None
    try:
        return check_profile(name)
    except ValueError as error:
        raise _StartupError(f"{PROFILE_ENV}: {error}") from None


def _port(text: str) -> int:
    """Parse a ``--health-port`` value: a TCP port number (0 picks a free one)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {text!r}") from None
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"must be between 0 and 65535, got {text!r}")
    return value


def _host(text: str) -> str:
    """Parse a ``--health-host`` value: a non-empty address (empty would bind every interface)."""
    if not text:
        raise argparse.ArgumentTypeError("must not be empty")
    return text


@final
class _StartupError(Exception):
    """A startup problem reported as ``pycordis: error: ...`` without a traceback."""


# -- start / inspect --run / doctor -------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    root = Context()
    # registered directly, not as a root effect: disposal-time errors stay visible
    root.logger.exporters[-1] = ConsoleExporter(levels={"default": _LEVELS[args.log_level]})
    logger = root.logger("pycordis")
    loader = Loader(
        root,
        LoaderConfig(
            path=args.config,
            watch=args.watch or None,
            allow=args.allow,
            dispose_timeout=args.dispose_timeout or None,
            profile=args.profile,
        ),
    )

    async def work() -> int:
        if args.command == "doctor":
            return await _doctor(root, loader, args.timeout)
        if args.health_port is not None:
            host = "127.0.0.1" if args.health_host is None else args.health_host
            await _start_health(root, host, args.health_port)
        await loader.start()
        if args.run:
            await settle(root)
            print(render(root.inspect()), flush=True)
        return 0

    received: list[int] = []
    stop = asyncio.Event()

    def on_signal(signum: int) -> None:
        if received:
            _flush()
            os._exit(_EXIT_SIGINT)
        received.append(signum)
        stop.set()

    restore = _install_signals(on_signal)
    try:
        working = asyncio.ensure_future(work())
        stopping = asyncio.ensure_future(stop.wait())
        await asyncio.wait({working, stopping}, return_when=asyncio.FIRST_COMPLETED)
        finite = args.run or args.command == "doctor"
        if not received and working.done() and (finite or working.exception() is not None):
            stopping.cancel()
            await asyncio.gather(stopping, return_exceptions=True)
            await root.fiber.dispose()
            error = working.exception()
            if error is None:
                return working.result()
            if _is_startup_error(error):
                return _startup_error(error)
            raise error
        await stopping
        logger.info("shutting down")
        await root.fiber.dispose()
        working.cancel()  # no-op unless interrupted while starting
        await asyncio.gather(working, return_exceptions=True)
        if not working.cancelled() and (error := working.exception()) is not None:
            logger.debug("startup ended by shutdown: %s", error)
    finally:
        restore()
    return _EXIT_SIGINT if received[0] == signal.SIGINT else 0


async def _doctor(root: Context, loader: Loader, timeout: float) -> int:
    """Start the tree, settle within ``timeout`` seconds and print what is not ACTIVE.

    Returns the exit code: 0 when every fiber is ACTIVE, else 1.
    """
    deadline = asyncio.timeout(timeout)
    try:
        async with deadline:
            await loader.start()
            await settle(root)
    except TimeoutError:
        if not deadline.expired():
            raise
        print(f"pycordis doctor: still starting after {timeout:g}s")
    await asyncio.sleep(0)  # let cycle warnings (checked on the next loop turn) log first
    stuck = root.diagnose().unhealthy()
    if stuck is None:
        print("all fibers are ACTIVE", flush=True)
        return 0
    print(render_diagnosis(stuck, labels=_entry_labels(loader)), flush=True)
    return 1


async def _start_health(root: Context, host: str, port: int) -> None:
    """Run a :class:`HealthServer` at the root; a failure to listen is a startup error.

    The server logs that failure as one line itself (no traceback).
    """
    try:
        await root.plugin(HealthServer, HealthServer.Config(host=host, port=port))
    except OSError as error:
        reason = error.strerror or str(error)
        raise _StartupError(f"cannot start the health server on {host}:{port}: {reason}") from None


def _entry_labels(loader: Loader) -> dict[int, str]:
    """``entry <id>`` for the uid of each loader entry's live fiber."""
    return {
        fiber.uid: f"entry {loaded.entry.id}"
        for loaded in loader.entries()
        if (fiber := loaded.fiber) is not None and fiber.uid is not None
    }


def _install_signals(handler: Callable[[int], None]) -> Callable[[], None]:
    """Route SIGINT/SIGTERM to ``handler`` on the running loop; return the uninstaller."""
    loop = asyncio.get_running_loop()
    signals = (signal.SIGINT, signal.SIGTERM)
    try:
        for signum in signals:
            loop.add_signal_handler(signum, handler, signum)
    except NotImplementedError:  # pragma: no cover - Windows event loops
        previous = {signum: signal.getsignal(signum) for signum in signals}
        for signum in signals:
            signal.signal(signum, lambda s, _f: loop.call_soon_threadsafe(handler, s))

        def restore_signal() -> None:
            for signum, old in previous.items():
                signal.signal(signum, old)

        return restore_signal

    def restore() -> None:
        for signum in signals:
            loop.remove_signal_handler(signum)

    return restore


def _flush() -> None:
    for stream in (sys.stdout, sys.stderr):
        # at exit a closed stream (ValueError) or a gone reader (BrokenPipeError,
        # OSError) has nowhere left to report to
        with contextlib.suppress(OSError, ValueError):
            stream.flush()


def _is_startup_error(error: BaseException) -> bool:
    """A config problem reported without a traceback (not a bug)."""
    return isinstance(error, (LoaderConfigError, _StartupError)) or (
        isinstance(error, RuntimeError) and str(error).startswith(_YAML_MISSING)
    )


def _startup_error(error: BaseException) -> int:
    print(f"pycordis: error: {error}", file=sys.stderr)
    return 1


# -- schema -----------------------------------------------------------------------------


def _schema(args: argparse.Namespace) -> int:
    try:
        entries, options = load_tree(Path(args.config), profile=args.profile)
    except (LoaderConfigError, RuntimeError) as error:
        if not _is_startup_error(error):
            raise
        return _startup_error(error)
    allow: list[str] = args.allow if args.allow is not None else options.allow
    text = (
        json.dumps(
            loader_json_schema(entries, allow, overlay=args.overlay), indent=2, allow_nan=False
        )
        + "\n"
    )
    if args.output is None:
        sys.stdout.write(text)
        return 0
    try:
        Path(args.output).write_text(text, encoding="utf-8")
    except OSError as error:
        return _startup_error(OSError(f"cannot write {args.output}: {error.strerror or error}"))
    return 0


# -- inspect (static) ------------------------------------------------------------------


def _inspect_static(args: argparse.Namespace) -> int:
    try:
        entries, options = load_tree(Path(args.config), profile=args.profile)
    except (LoaderConfigError, RuntimeError) as error:
        if not _is_startup_error(error):
            raise
        return _startup_error(error)
    allow: list[str] = args.allow if args.allow is not None else options.allow
    lines: list[str] = []
    errors: list[str] = []
    # (name, scope) pairs; built-ins are provided at the root scope
    provided: set[tuple[str, _Scope]] = {(name, ()) for name in _BUILTIN_SERVICES}
    needs: list[tuple[str, list[tuple[str, _Scope]]]] = []

    def visit(items: list[Entry], depth: int, active: bool, scopes: _Scopes) -> None:
        for entry in items:
            enabled = active and not entry.disabled
            use = _label(entry)
            indent = "  " * depth
            own = _isolate(scopes, entry)
            if not enabled:
                lines.append(f"{indent}{entry.id}  {use}  (disabled)")
                visit(entry.children, depth + 1, False, own)
                continue
            provides: list[str] = []
            injects: list[str] = []
            if entry.use is not None:
                provides, injects = _check_plugin(entry, allow, errors)
            injects += [name for name in entry.inject if name not in injects]
            provided.update((name, own.get(name, ())) for name in provides)
            needs.append((entry.id, [(name, own.get(name, ())) for name in injects]))
            lines.append(
                f"{indent}{entry.id}  {use}  "
                f"provides=[{', '.join(provides)}] injects=[{', '.join(injects)}]"
            )
            visit(entry.children, depth + 1, True, own)

    visit(entries, 0, True, {})
    names = {name for name, _ in provided}
    warnings = [
        f'{id}: cannot inject "{name}": '
        + ("only provided in another isolated scope" if name in names else "no entry provides it")
        for id, injects in needs
        for name, scope in injects
        if (name, scope) not in provided
    ]
    for line in lines:
        print(line)
    for error in errors:
        print(f"✗ {error}")
    for warning in warnings:
        print(f"⚠ {warning}")
    return 1 if errors else 0


type _Scope = tuple[str, ...]
"""Static model of an isolation label: ``()`` root, ``("label", l)``, ``("entry", id)``."""
type _Scopes = dict[str, _Scope]


def _isolate(scopes: _Scopes, entry: Entry) -> _Scopes:
    """The service scopes ``entry`` (and its children) run in, given its parent's.

    Mirrors :meth:`Context.isolate`: a string label is shared by every entry
    using it, no label creates a scope private to the entry's subtree.
    """
    if not entry.isolate:
        return scopes
    own = dict(scopes)
    for name, label in entry.isolate.items():
        own[name] = ("entry", entry.id) if label is None else ("label", label)
    return own


def _label(entry: Entry) -> str:
    if entry.use is not None:
        return entry.use
    if entry.group is not None:
        return f"group:{entry.group}"
    return f"include:{entry.include}"


def _check_plugin(
    entry: Entry, allow: Sequence[str], errors: list[str]
) -> tuple[list[str], list[str]]:
    """Import ``entry``'s plugin and validate its config; return (provides, injects).

    Problems are appended to ``errors``. Reads only the plugin's metadata; the
    plugin itself is never called.
    """
    assert entry.use is not None
    try:
        plugin = import_plugin(entry.use, allow)
        injects = list(plugin_inject(plugin))
    except Exception as error:
        errors.append(f"{entry.id}: {error}")
        return [], []
    provides = list(plugin_provide(plugin))
    try:
        validate_config(plugin_config_type(plugin), expand(entry.config, entry.source.parent))
    except ValidationError as error:
        errors.extend(
            f"{entry.id}: invalid config: {issue.message}"
            + (f" (at {'.'.join(map(str, issue.path))})" if issue.path else "")
            for issue in error.issues
        )
    except Exception as error:
        errors.append(f"{entry.id}: invalid config: {error}")
    return provides, injects
