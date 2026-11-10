"""``docs/guide-agents.md`` must run: its python blocks are one program, its text block the output.

Every fenced ``python`` block is valid Python on its own; together, in document
order, they form one program whose standard output must equal the guide's single
fenced ``text`` block. The program runs as a real module (registered in
``sys.modules``) so dataclasses and type hints resolve as they would in a file.
"""

import io
import re
import sys
import types
from collections.abc import Generator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "guide-agents.md"
MODULE = "pycordis_guide_agents"


def _blocks(language: str) -> list[str]:
    text = GUIDE.read_text("utf-8")
    return re.findall(rf"^```{language}\n(.*?)^```$", text, re.S | re.M)


@contextmanager
def _module(name: str) -> Generator[types.ModuleType]:
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        yield module
    finally:
        del sys.modules[name]


def _fence_tags() -> list[str]:
    """The info string of every fenced block of the guide, in order (``""`` if untagged)."""
    tags: list[str] = []
    inside = False
    for line in GUIDE.read_text("utf-8").splitlines():
        if not line.startswith("```"):
            continue
        if not inside:
            tags.append(line.removeprefix("```").strip())
        inside = not inside
    assert not inside, "unclosed fenced block"
    return tags


def test_the_guide_has_exactly_its_known_blocks() -> None:
    # a retagged (```py), untagged or added block would otherwise drop out of the run
    assert _fence_tags() == ["python"] * 7 + ["text"]


def test_every_python_block_is_valid_python() -> None:
    blocks = _blocks("python")
    assert len(blocks) == 7
    for number, block in enumerate(blocks, 1):
        compile(block, f"{GUIDE.name} block {number}", "exec")


def test_the_guide_runs_and_prints_its_documented_output() -> None:
    program = "\n\n".join(_blocks("python"))
    [expected] = _blocks("text")
    stdout = io.StringIO()
    with _module(MODULE) as module, redirect_stdout(stdout):
        exec(compile(program, str(GUIDE), "exec"), module.__dict__)
    assert stdout.getvalue() == expected
