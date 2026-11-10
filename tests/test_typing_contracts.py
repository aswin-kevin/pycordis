"""Static typing contracts, checked by running pyright on ``tests/typing_fixtures``.

Each fixture ends the lines pyright must reject with ``# expect-error``, or with
``# expect-error: reportRuleName`` to also require that rule; every other line
must type-check cleanly under strict mode. The fixtures are excluded from the
main pyright run (see ``pyproject.toml``). The test copies them to a temporary
directory with its own pyright config, points pyright at the running
interpreter, and checks every fixture with a single pyright invocation. It is
skipped only when pyright is not installed.
"""

import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "typing_fixtures"
MARKER = re.compile(r"# expect-error(?::\s*(?P<rule>\w+))?\s*$")


@dataclass(frozen=True)
class Expectation:
    """A line pyright must reject, optionally with a specific rule."""

    line: int
    rule: str | None


def _pyright_command() -> list[str]:
    """The pyright CLI next to the running interpreter, else the one on ``PATH``."""
    local = Path(sys.executable).parent / "pyright"
    if local.exists():
        return [str(local)]
    found = shutil.which("pyright")
    if found is None:
        pytest.skip("pyright is not installed in this environment")
    return [found]


def _expectations(path: Path) -> dict[int, Expectation]:
    expected: dict[int, Expectation] = {}
    for number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        match = MARKER.search(line)
        if match:
            expected[number] = Expectation(number, match.group("rule"))
    return expected


def _fixture_names() -> list[str]:
    return sorted(path.name for path in FIXTURES.glob("*.py"))


def _run_pyright(config_dir: Path) -> dict[str, dict[int, set[str]]]:
    """Run pyright once over every fixture; return ``{file: {line: rules}}``.

    A crash, a timeout or unreadable output fails the test: pyright is
    installed, so the contract must actually be checked.
    """
    # pyright only accepts relative "include" paths, so the fixtures are copied
    # next to a config written for this run
    for name in _fixture_names():
        shutil.copy(FIXTURES / name, config_dir / name)
    config = config_dir / "pyrightconfig.json"
    settings = {"include": ["."], "typeCheckingMode": "strict", "pythonVersion": "3.12"}
    config.write_text(json.dumps(settings), "utf-8")
    command = [*_pyright_command(), "--outputjson", "-p", str(config)]
    completed = subprocess.run(
        [*command, "--pythonpath", sys.executable],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        pytest.fail(
            f"pyright produced no JSON report (exit {completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    analyzed = report["summary"]["filesAnalyzed"]
    assert analyzed == len(_fixture_names()), f"pyright analyzed {analyzed} fixture files"
    reported: defaultdict[str, defaultdict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for diagnostic in report["generalDiagnostics"]:
        if diagnostic["severity"] in ("error", "warning"):
            line = diagnostic["range"]["start"]["line"] + 1
            rule = diagnostic.get("rule", "")
            reported[Path(diagnostic["file"]).name][line].add(rule)
    return {file: dict(lines) for file, lines in reported.items()}


@pytest.fixture(scope="module")
def reported(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[int, set[str]]]:
    return _run_pyright(tmp_path_factory.mktemp("pyright"))


@pytest.mark.parametrize("fixture", _fixture_names())
def test_fixture_reports_exactly_the_marked_lines(
    fixture: str, reported: dict[str, dict[int, set[str]]]
) -> None:
    expected = _expectations(FIXTURES / fixture)
    assert expected, f"{fixture} marks no '# expect-error' lines"
    actual = reported.get(fixture, {})
    assert sorted(set(actual) - set(expected)) == [], "unexpected pyright diagnostics"
    assert sorted(set(expected) - set(actual)) == [], "marked lines pyright accepted"
    wrong_rule = {
        line: sorted(actual[line])
        for line, expectation in expected.items()
        if expectation.rule is not None and expectation.rule not in actual[line]
    }
    assert wrong_rule == {}, "marked lines reported under a different rule"
