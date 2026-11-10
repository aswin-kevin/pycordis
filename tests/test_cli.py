"""Tests for the ``pycordis`` command line (spec §6.6 CLI)."""

from __future__ import annotations

import asyncio
import importlib
import os
import runpy
import select
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from pycordis import cli
from pycordis.context import Context

FIXTURES = Path(__file__).parent / "fixtures" / "cli"
PLUGINS = FIXTURES / "plugins"
APP = str(FIXTURES / "app.toml")
BROKEN = str(FIXTURES / "broken.toml")


@pytest.fixture(autouse=True)
def _restore_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main()`` puts the working directory on ``sys.path``; undo it after each test."""
    monkeypatch.setattr(sys, "path", list(sys.path))


@pytest.fixture
def clifix(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Make the CLI fixture plugins importable; restore ``sys.path`` afterwards."""
    monkeypatch.setattr(sys, "path", [str(PLUGINS), *sys.path])
    monkeypatch.delenv("CLIFIX_LABEL", raising=False)
    sys.modules.pop("clifix", None)
    module = importlib.import_module("clifix")
    try:
        yield module
    finally:
        sys.modules.pop("clifix", None)


# -- inspect (static) ------------------------------------------------------------


def test_inspect_static_prints_tree_without_running_plugins(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["inspect", "-c", APP]) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "store  clifix:Store  provides=[store] injects=[]",
        "consumer  clifix:consumer  provides=[] injects=[store]",
        "/group:extras#0  group:extras  provides=[] injects=[]",
        "  ready  clifix:ready  provides=[] injects=[]",
        "off  clifix:missing  (disabled)",
    ]
    assert not clifix.calls


def test_inspect_static_flags_errors(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["inspect", "-c", BROKEN]) == 1
    assert capsys.readouterr().out.splitlines() == [
        "store  clifix:Store  provides=[store] injects=[]",
        "consumer  clifix:consumer  provides=[] injects=[store, nothing]",
        "denied  os:getcwd  provides=[] injects=[]",
        "missing  clifix:missing  provides=[] injects=[]",
        "✗ store: invalid config: expected str, got int (at label)",
        '✗ denied: plugin module "os" is not allowed by loader.allow',
        '✗ missing: module "clifix" has no attribute "missing"',
        '⚠ consumer: cannot inject "nothing": no entry provides it',
    ]
    assert not clifix.calls


def test_inspect_static_unsatisfied_inject_is_a_warning(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = tmp_path / "app.toml"
    config.write_text(
        '[loader]\nallow = ["clifix"]\n'
        '[[plugins]]\nid = "consumer"\nuse = "clifix:consumer"\ninject = ["loader"]\n'
    )
    assert cli.main(["inspect", "-c", str(config)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "consumer  clifix:consumer  provides=[] injects=[store, loader]",
        '⚠ consumer: cannot inject "store": no entry provides it',
    ]


def test_inspect_static_injection_respects_isolation(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = tmp_path / "app.toml"
    config.write_text(
        '[loader]\nallow = ["clifix"]\n'
        # provided only inside an isolated group: invisible outside it
        '[[plugins]]\nid = "inside"\ngroup = "in"\nisolate = ["store"]\nplugins = [\n'
        '  { id = "s1", use = "clifix:Store", config = { label = "a" } },\n'
        '  { id = "c1", use = "clifix:consumer" },\n]\n'
        '[[plugins]]\nid = "c2"\nuse = "clifix:consumer"\n'
        # an isolated consumer does not see a provider outside its scope
        '[[plugins]]\nid = "c3"\nuse = "clifix:consumer"\nisolate = { store = "shared" }\n'
        # equal labels share one scope
        '[[plugins]]\nid = "s2"\nuse = "clifix:Store"\nconfig = { label = "b" }\n'
        'isolate = { store = "shared" }\n'
        '[[plugins]]\nid = "c4"\nuse = "clifix:consumer"\nisolate = { store = "other" }\n'
    )
    assert cli.main(["inspect", "-c", str(config)]) == 0
    assert capsys.readouterr().out.splitlines()[-2:] == [
        '⚠ c2: cannot inject "store": only provided in another isolated scope',
        '⚠ c4: cannot inject "store": only provided in another isolated scope',
    ]


def test_inspect_static_interpolates_config(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLIFIX_LABEL", "")  # empty -> default
    assert cli.main(["inspect", "-c", APP]) == 0
    capsys.readouterr()


def test_inspect_static_bad_interpolation(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = tmp_path / "app.toml"
    config.write_text(
        '[[plugins]]\nid = "store"\nuse = "clifix:Store"\n'
        'config = { label = "${env:CLIFIX_NOPE}" }\n'
    )
    assert cli.main(["inspect", "-c", str(config), "--allow", "clifix"]) == 1
    errors = [line for line in capsys.readouterr().out.splitlines() if line.startswith("✗")]
    assert len(errors) == 1
    assert errors[0].startswith("✗ store: invalid config: ")
    assert "CLIFIX_NOPE" in errors[0]


def test_inspect_static_allow_override(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["inspect", "-c", APP, "--allow", "nothing.*"]) == 1
    errors = [line for line in capsys.readouterr().out.splitlines() if line.startswith("✗")]
    assert '✗ store: plugin module "clifix" is not allowed by loader.allow' in errors


def test_inspect_static_invalid_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    missing = tmp_path / "nope.toml"
    assert cli.main(["inspect", "-c", str(missing)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"pycordis: error: {missing.resolve()}: cannot read file")


def test_yaml_without_pyyaml_is_a_startup_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "yaml", None)  # import yaml -> ImportError
    config = tmp_path / "app.yaml"
    config.write_text("plugins: []\n")
    for argv in (["inspect"], ["inspect", "--run"], ["start"], ["schema"]):
        assert cli.main([*argv, "-c", str(config)]) == 1
        err = capsys.readouterr().err
        assert err == "pycordis: error: YAML config requires PyYAML to be installed\n"


def test_other_runtime_errors_propagate(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_tree(_path: Path, **_options: object) -> None:
        raise RuntimeError("bug")

    async def broken_start(self: object) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(cli, "load_tree", broken_tree)
    with pytest.raises(RuntimeError, match="bug"):
        cli.main(["inspect", "-c", APP])
    with pytest.raises(RuntimeError, match="bug"):
        cli.main(["schema", "-c", APP])
    monkeypatch.setattr(cli.Loader, "start", broken_start)
    with pytest.raises(RuntimeError, match="bug"):
        cli.main(["inspect", "--run", "-c", APP])
    capsys.readouterr()


def test_cwd_is_appended_after_stdlib(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", cwd)])
    cli.main(["inspect", "-c", str(tmp_path / "nope.toml")])
    capsys.readouterr()
    stdlib = os.path.dirname(os.__file__)
    assert sys.path[-1] == cwd
    assert sys.path.index(cwd) > sys.path.index(stdlib)
    cli.main(["inspect", "-c", str(tmp_path / "nope.toml")])  # not added twice
    capsys.readouterr()
    assert sys.path.count(cwd) == 1


def test_inspect_static_bad_metadata(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    mod = tmp_path / "clibad.py"
    mod.write_text(
        "def plugin(ctx, config):\n    pass\n\nplugin.inject = 'llm'\n\n"
        "def multi(ctx, config):\n    pass\n\nmulti.provide = ['a', 'b']\n"
    )
    config = tmp_path / "app.toml"
    config.write_text(
        '[[plugins]]\nid = "bad"\nuse = "clibad:plugin"\n'
        '[[plugins]]\nid = "multi"\nuse = "clibad:multi"\n'
    )
    sys.path.insert(0, str(tmp_path))
    try:
        assert cli.main(["inspect", "-c", str(config)]) == 1
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("clibad", None)
    out = capsys.readouterr().out.splitlines()
    assert "multi  clibad:multi  provides=[a, b] injects=[]" in out
    assert "✗ bad: inject must be a list of service names or a mapping" in out


# -- inspect --run -----------------------------------------------------------------


def test_inspect_run_prints_live_tree(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["inspect", "--run", "-c", APP, "--log-level", "error"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("ready\nroot #0 ACTIVE")
    assert "provides: loader" in out
    assert "Store #" in out and "provides: store" in out
    assert "consumer #" in out and "injects: store" in out
    assert clifix.calls == {"store": 1, "consumer": 1, "ready": 1}


def test_inspect_run_invalid_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli.main(["inspect", "--run", "-c", str(tmp_path / "nope.toml")]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("pycordis: error: ")
    assert "cannot read file" in captured.err
    assert "Traceback" not in captured.err


# -- start ---------------------------------------------------------------------------


def test_start_missing_config(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli.main(["-c", str(tmp_path / "nope.toml")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("pycordis: error: ")
    assert "Traceback" not in err


def test_start_sigint_exits_130(clifix: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["start", "-c", str(FIXTURES / "sigint.toml")]) == 130
    captured = capsys.readouterr()
    assert captured.out == "ready\n"
    assert "shutting down" in captured.err


def test_start_sigterm_exits_0(clifix: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["-c", str(FIXTURES / "sigterm.toml"), "--log-level", "warn"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "ready\n"
    assert "shutting down" not in captured.err  # info is below the warn threshold


def test_start_second_signal_forces_exit(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    exits: list[int] = []
    monkeypatch.setattr(cli.os, "_exit", exits.append)
    assert cli.main(["-c", str(FIXTURES / "twice.toml")]) == 130
    assert exits == [130]
    capsys.readouterr()


def test_start_signal_during_startup(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # the signal arrives while loader.start() is still waiting for a slow body
    mod = tmp_path / "clislow.py"
    mod.write_text(
        "import asyncio, os, signal\n"
        "async def slow(ctx, config):\n"
        "    os.kill(os.getpid(), signal.SIGINT)\n"
        "    await asyncio.sleep(60)\n"
    )
    config = tmp_path / "app.toml"
    config.write_text('[[plugins]]\nuse = "clislow:slow"\n')
    sys.path.insert(0, str(tmp_path))
    try:
        started = time.monotonic()
        assert cli.main(["-c", str(config)]) == 130
        assert time.monotonic() - started < 5
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("clislow", None)
    capsys.readouterr()


def test_watch_flag_reaches_loader(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []
    original = cli.Loader

    class Spy(original):
        def __init__(self, ctx: Context, config: cli.LoaderConfig) -> None:
            seen.append(config)
            super().__init__(ctx, config)

    monkeypatch.setattr(cli, "Loader", Spy)
    args = ["inspect", "--run", "-c", APP, "--watch", "--allow", "clifix", "--allow", "x.*"]
    assert cli.main(args) == 0
    assert seen == [
        cli.LoaderConfig(path=APP, watch=True, allow=["clifix", "x.*"], dispose_timeout=10.0)
    ]
    capsys.readouterr()


@pytest.mark.parametrize(
    ("flag", "expected"),
    [([], 10.0), (["--dispose-timeout", "2.5"], 2.5), (["--dispose-timeout", "0"], None)],
)
def test_dispose_timeout_defaults_to_10s_and_is_overridable(
    clifix: ModuleType,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    flag: list[str],
    expected: float | None,
) -> None:
    seen: list[cli.LoaderConfig] = []
    original = cli.Loader

    class Spy(original):
        def __init__(self, ctx: Context, config: cli.LoaderConfig) -> None:
            seen.append(config)
            super().__init__(ctx, config)

    monkeypatch.setattr(cli, "Loader", Spy)
    assert cli.main(["inspect", "--run", "-c", APP, "--allow", "clifix", *flag]) == 0
    assert [config.dispose_timeout for config in seen] == [expected]
    capsys.readouterr()


@pytest.mark.parametrize("value", ["-1", "nan", "soon"])
def test_dispose_timeout_rejects_bad_values(capsys: pytest.CaptureFixture[str], value: str) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["--dispose-timeout", value])
    assert info.value.code == 2
    assert "--dispose-timeout" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_start_subprocess_sigint() -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(PLUGINS), *sys.path])}
    proc = subprocess.Popen(
        [sys.executable, "-m", "pycordis", "-c", APP],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], 10)
        assert ready, "plugin did not report ready"
        assert proc.stdout.readline() == "ready\n"
        proc.send_signal(signal.SIGINT)
        started = time.monotonic()
        _, err = proc.communicate(timeout=5)
        assert time.monotonic() - started < 5
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 130, err
    assert "shutting down" in err
    assert "Task was destroyed" not in err
    assert "never awaited" not in err
    assert "Traceback" not in err


def test_inspect_static_include_and_async_schema(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    mod = tmp_path / "cliasync.py"
    mod.write_text(
        "async def check(raw):\n    return raw\n\n"
        "def plugin(ctx, config):\n    pass\n\nplugin.Config = check\n"
    )
    (tmp_path / "sub.toml").write_text('[[plugins]]\nid = "inner"\nuse = "cliasync:plugin"\n')
    config = tmp_path / "app.toml"
    config.write_text('[[plugins]]\nid = "inc"\ninclude = "sub.toml"\n')
    sys.path.insert(0, str(tmp_path))
    try:
        assert cli.main(["inspect", "-c", str(config)]) == 1
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("cliasync", None)
    assert capsys.readouterr().out.splitlines() == [
        "inc  include:sub.toml  provides=[] injects=[]",
        "  inner  cliasync:plugin  provides=[] injects=[]",
        "✗ inner: invalid config: Async config validation is not supported",
    ]


def test_inspect_run_handles_sigint(clifix: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["inspect", "--run", "-c", str(FIXTURES / "sigint.toml")]) == 130
    captured = capsys.readouterr()
    assert "shutting down" in captured.err
    assert "Traceback" not in captured.err


def test_start_error_after_shutdown_is_logged(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def late(self: object) -> None:
        os.kill(os.getpid(), signal.SIGINT)
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise ValueError("late failure") from None

    monkeypatch.setattr(cli.Loader, "start", late)
    assert cli.main(["-c", APP, "--log-level", "debug"]) == 130
    assert "startup ended by shutdown: late failure" in capsys.readouterr().err


def test_start_unexpected_error_propagates(
    clifix: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def boom(self: object) -> None:
        raise KeyError("boom")

    monkeypatch.setattr(cli.Loader, "start", boom)
    with pytest.raises(KeyError, match="boom"):
        cli.main(["-c", APP])
    capsys.readouterr()


def test_python_m_pycordis(
    clifix: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["pycordis", "inspect", "-c", APP])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("pycordis", run_name="__main__", alter_sys=True)
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("store  clifix:Store")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_start_subprocess_second_sigint_forces_exit() -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(PLUGINS), *sys.path])}
    proc = subprocess.Popen(
        [sys.executable, "-m", "pycordis", "-c", str(FIXTURES / "hang.toml")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], 10)
        assert ready, "plugin did not report ready"
        assert proc.stdout.readline() == "ready\n"
        proc.send_signal(signal.SIGINT)
        time.sleep(1)
        assert proc.poll() is None, "first SIGINT should wait for the hanging disposer"
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 130, err
    assert "shutting down" in err


# -- doctor --------------------------------------------------------------------------


def test_doctor_all_active_exits_0(clifix: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["doctor", "-c", APP, "--log-level", "error"]) == 0
    assert capsys.readouterr().out == "ready\nall fibers are ACTIVE\n"


def test_doctor_reports_a_dependency_cycle(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    started = time.monotonic()
    assert cli.main(["doctor", "-c", str(FIXTURES / "cycle.toml")]) == 1
    assert time.monotonic() - started < 5
    captured = capsys.readouterr()
    assert captured.out == "\n".join(
        [
            "root #0 ACTIVE",
            "├─ agent #2 [entry agent] PENDING",
            '│     cycle: cycle #1: agent -> "llm" -> llm -> "cache" -> agent',
            "└─ llm #3 [entry llm] PENDING",
            '      cycle: waits for "cache" in cycle #1 (see agent #2)',
            "",
        ]
    )
    warning = 'dependency cycle: agent -> "llm" -> llm -> "cache" -> agent'
    assert captured.err.count(warning) == 1
    assert clifix.calls["agent"] == clifix.calls["llm"] == 0


def test_doctor_times_out_instead_of_hanging(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    started = time.monotonic()
    assert cli.main(["doctor", "-c", str(FIXTURES / "stall.toml"), "--timeout", "0.2"]) == 1
    assert time.monotonic() - started < 5
    captured = capsys.readouterr()
    assert captured.out == "\n".join(
        [
            "pycordis doctor: still starting after 0.2s",
            "root #0 ACTIVE",
            "└─ stall #1 [entry stall] LOADING",
            "      loading: running the plugin body",
            "",
        ]
    )
    assert clifix.calls["stall"] == 1


def test_doctor_invalid_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli.main(["doctor", "-c", str(tmp_path / "nope.toml")]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("pycordis: error: ")
    assert captured.out == ""


def test_doctor_handles_sigint(clifix: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["doctor", "-c", str(FIXTURES / "sigint.toml")]) == 130
    captured = capsys.readouterr()
    assert "shutting down" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("value", ["-1", "nan", "soon"])
def test_doctor_rejects_bad_timeouts(capsys: pytest.CaptureFixture[str], value: str) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["doctor", "--timeout", value])
    assert info.value.code == 2
    assert "--timeout" in capsys.readouterr().err


def test_doctor_propagates_a_timeout_error_of_its_own_making(
    clifix: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # only doctor's own deadline means "still starting"; any other TimeoutError is a bug
    async def boom(self: object) -> None:
        raise TimeoutError("from the loader")

    monkeypatch.setattr(cli.Loader, "start", boom)
    with pytest.raises(TimeoutError, match="from the loader"):
        cli.main(["doctor", "-c", APP])
    capsys.readouterr()


# -- health server -------------------------------------------------------------------


def test_start_health_port_serves_on_loopback(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["start", "-c", str(FIXTURES / "health.toml"), "--health-port", "0"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        'health 127.0.0.1 200 {"status":"ok"}',
        'ready 200 {"status":"ready"}',
    ]
    assert "health server listening on http://127.0.0.1:" in captured.err


def test_no_health_server_without_the_flag(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["inspect", "--run", "-c", APP]) == 0
    assert "HealthServer" not in capsys.readouterr().out


def test_start_health_port_in_use_is_a_startup_error(
    clifix: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = str(cast("tuple[str, int]", taken.getsockname())[1])
        assert cli.main(["-c", APP, "--health-port", port]) == 1
    err = capsys.readouterr().err
    assert f"pycordis: error: cannot start the health server on 127.0.0.1:{port}: " in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "argv",
    [
        ["--health-port", "-1"],
        ["--health-port", "65536"],
        ["--health-port", "http"],
        ["--health-host", "0.0.0.0"],
        ["--health-port", "0", "--health-host", ""],
        ["doctor", "--health-port", "0"],
        ["--health-port", "0", "doctor"],
        ["--health-host", "::1", "inspect"],
    ],
)
def test_health_flags_are_validated(capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(argv)
    assert info.value.code == 2
    assert "--health-" in capsys.readouterr().err
