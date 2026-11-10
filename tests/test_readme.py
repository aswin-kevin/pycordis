"""README examples must run; the quick start must match ``test_readme_quick_start`` exactly."""

import re
import sys
import types
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _quick_start() -> str:
    text = (ROOT / "README.md").read_text("utf-8")
    match = re.search(r"## Quick start\n.*?```python\n(.*?)```", text, re.S)
    assert match
    return match.group(1)


def _test_body() -> str:
    text = (ROOT / "tests" / "test_service.py").read_text("utf-8")
    match = re.search(
        r"async def test_readme_quick_start\(\) -> None:\n(.*?)\n\n\n# --", text, re.S
    )
    assert match
    return match.group(1)


def test_readme_quick_start_matches_test_body() -> None:
    assert _test_body() in _quick_start()


async def test_readme_quick_start_runs() -> None:
    namespace: dict[str, object] = {"__name__": "readme_example"}
    code = _quick_start().replace("asyncio.run(main())", "")
    exec(compile(code, "README.md", "exec"), namespace)
    await cast(Callable[[], Awaitable[None]], namespace["main"])()


async def test_readme_generator_plugin_example() -> None:
    from pycordis import Context

    text = (ROOT / "README.md").read_text("utf-8")
    match = re.search(r"```python\n(def watcher\(ctx, config\):\n.*?)```", text, re.S)
    assert match
    log: list[str] = []

    class Handle:
        def close(self) -> None:
            log.append("closed")

    class Timer:
        def cancel(self) -> None:
            log.append("cancelled")

    def open_connection(_config: object) -> Handle:
        return Handle()

    namespace: dict[str, object] = {
        "open_connection": open_connection,
        "start_timer": Timer,
    }
    exec(compile(match.group(1), "README.md", "exec"), namespace)
    async with Context() as root:
        fiber = await root.plugin(namespace["watcher"])
        assert log == []
        await fiber.dispose()
        assert log == ["cancelled", "closed"]


def _programs() -> list[str]:
    """README python blocks that are complete programs, other than the quick start."""
    text = (ROOT / "README.md").read_text("utf-8")
    blocks: list[str] = re.findall(r"^```python\n(.*?)^```$", text, re.S | re.M)
    return [block for block in blocks if "asyncio.run(main())" in block][1:]


def test_readme_programs_run(capsys: pytest.CaptureFixture[str]) -> None:
    programs = _programs()
    assert len(programs) >= 2  # typed service keys, collections
    for number, program in enumerate(programs):
        module = types.ModuleType(f"readme_program_{number}")
        sys.modules[module.__name__] = module
        try:
            exec(compile(program, "README.md", "exec"), module.__dict__)
        finally:
            del sys.modules[module.__name__]
    assert capsys.readouterr().out == "echo: hi\nadd search\nremove search\n"


def _diagnostics_example() -> tuple[str, str, str]:
    """The plugin module, config file and ``pycordis doctor`` output of README Diagnostics."""
    text = (ROOT / "README.md").read_text("utf-8")
    section = re.search(r"^### Diagnostics\n(.*?)^The reason kinds are", text, re.S | re.M)
    assert section
    body = section.group(1)
    plugins = re.search(r"^```python\n(# app_plugins\.py\n.*?)^```$", body, re.S | re.M)
    config = re.search(r"^```toml\n(# app\.toml\n.*?)^```$", body, re.S | re.M)
    output = re.search(r"^```text\n\$ pycordis doctor -c app\.toml\n(.*?)^```$", body, re.S | re.M)
    assert plugins and config and output
    return plugins.group(1), config.group(1), output.group(1)


def test_readme_doctor_example_prints_its_documented_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pycordis import cli

    plugins, config, output = _diagnostics_example()
    module = types.ModuleType("app_plugins")
    exec(compile(plugins, "README.md", "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, "app_plugins", module)
    path = tmp_path / "app.toml"
    path.write_text(config, "utf-8")
    assert cli.main(["doctor", "-c", str(path)]) == 1
    assert capsys.readouterr().out == output
