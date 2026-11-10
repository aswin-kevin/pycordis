from __future__ import annotations

import os
from pathlib import Path

import pytest

from pycordis.errors import Issue, ValidationError
from pycordis.loader import expand
from pycordis.loader.interpolate import (
    MAX_FILE_SIZE,
    MIN_SECRET_LENGTH,
    Expansions,
    leaks,
    retemplate,
    secrets,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "loader"

needs_fifo = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo (POSIX)")


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYCORDIS_A", "alpha")
    monkeypatch.setenv("PYCORDIS_LONG", "longsecret")
    monkeypatch.setenv("PYCORDIS_EMPTY", "")
    monkeypatch.delenv("PYCORDIS_UNSET", raising=False)


def test_env() -> None:
    assert expand("${env:PYCORDIS_A}") == "alpha"
    assert expand("x-${env:PYCORDIS_A}-${env:PYCORDIS_A}!") == "x-alpha-alpha!"
    assert expand("${env:PYCORDIS_EMPTY}") == ""


def test_env_default() -> None:
    assert expand("${env:PYCORDIS_UNSET:-fallback}") == "fallback"
    assert expand("${env:PYCORDIS_UNSET:-}") == ""
    assert expand("${env:PYCORDIS_UNSET:-a:-b c}") == "a:-b c"
    # shell semantics: `:-` also applies when the variable is set but empty
    assert expand("${env:PYCORDIS_EMPTY:-fallback}") == "fallback"
    assert expand("${env:PYCORDIS_A:-fallback}") == "alpha"


def test_escape() -> None:
    assert expand("$${env:PYCORDIS_A}") == "${env:PYCORDIS_A}"
    assert expand("$$${env:PYCORDIS_A}") == "$${env:PYCORDIS_A}"
    assert expand("cost: $5 {x} $") == "cost: $5 {x} $"


def test_recursive_and_non_strings_untouched() -> None:
    value = {"a": ["${env:PYCORDIS_A}", 1, None, {"b": "${env:PYCORDIS_A}"}], "c": True}
    assert expand(value) == {"a": ["alpha", 1, None, {"b": "alpha"}], "c": True}
    assert expand(("${env:PYCORDIS_A}",)) == ("alpha",)
    assert expand(1.5) == 1.5


def test_raw_value_not_mutated() -> None:
    raw = {"k": ["${env:PYCORDIS_A}"]}
    expand(raw)
    assert raw == {"k": ["${env:PYCORDIS_A}"]}


def test_missing_env_carries_path() -> None:
    value = {"llm": {"keys": ["ok", "${env:PYCORDIS_UNSET}"]}}
    with pytest.raises(ValidationError) as info:
        expand(value)
    assert info.value.issues == (
        Issue("environment variable PYCORDIS_UNSET is not set", ("llm", "keys", 1)),
    )


def test_missing_env_base_path_and_all_issues() -> None:
    with pytest.raises(ValidationError) as info:
        expand({"a": "${env:PYCORDIS_UNSET}", "b": "${env:PYCORDIS_UNSET}"}, path=("cfg",))
    assert [i.path for i in info.value.issues] == [("cfg", "a"), ("cfg", "b")]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("${nope}", 'unknown interpolation "${nope}"'),
        ("${env:}", 'unknown interpolation "${env:}"'),
        ("${file:}", 'unknown interpolation "${file:}"'),
        ("${env::-x}", 'unknown interpolation "${env::-x}"'),
        ("abc ${env:X", 'unterminated interpolation in "abc ${env:X"'),
    ],
)
def test_malformed(text: str, message: str) -> None:
    with pytest.raises(ValidationError) as info:
        expand({"k": text})
    assert info.value.issues == (Issue(message, ("k",)),)


def test_file_secret_relative_to_source_dir() -> None:
    assert expand("${file:secret.txt}", FIXTURES / "sub") == "s3cr3t"
    assert expand("pre-${file:sub/secret.txt}", FIXTURES) == "pre-s3cr3t"


def test_file_absolute_and_only_one_newline_stripped(tmp_path: Path) -> None:
    (tmp_path / "s").write_bytes(b"line\r\n\r\n")
    assert expand(f"${{file:{tmp_path / 's'}}}") == "line\r\n"
    (tmp_path / "t").write_bytes(b"no-newline")
    assert expand("${file:t}", tmp_path) == "no-newline"


def test_file_relative_to_cwd_without_source_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "s").write_text("v\n")
    monkeypatch.chdir(tmp_path)
    assert expand("${file:s}") == "v"


def test_file_errors(tmp_path: Path) -> None:
    (tmp_path / "big").write_bytes(b"x" * (MAX_FILE_SIZE + 1))
    (tmp_path / "ok").write_bytes(b"x" * MAX_FILE_SIZE)
    (tmp_path / "bin").write_bytes(b"\xff\xfe")
    assert expand("${file:ok}", tmp_path) == "x" * MAX_FILE_SIZE
    with pytest.raises(ValidationError) as info:
        expand(["${file:big}", "${file:missing}", "${file:bin}"], tmp_path)
    big, missing, binary = info.value.issues
    assert big == Issue("file big exceeds the 1 MiB limit", (0,))
    assert missing.message.startswith("cannot read file missing:")
    assert missing.path == (1,)
    assert binary.message.startswith("file bin is not valid UTF-8")


@needs_fifo
def test_file_must_be_regular(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "fifo")
    (tmp_path / "dir").mkdir()
    with pytest.raises(ValidationError) as info:
        expand(["${file:fifo}", "${file:dir}"], tmp_path)
    assert info.value.issues == (
        Issue("cannot read file fifo: not a regular file", (0,)),
        Issue("cannot read file dir: not a regular file", (1,)),
    )


def test_file_bom_dropped(tmp_path: Path) -> None:
    (tmp_path / "s").write_bytes(b"\xef\xbb\xbftoken\n")
    assert expand("${file:s}", tmp_path) == "token"


def test_file_symlink_followed(tmp_path: Path) -> None:
    (tmp_path / "real").write_text("linked\n")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert expand("${file:link}", tmp_path) == "linked"


# -- write-back helpers -----------------------------------------------------------


def test_retemplate_restores_placeholders(tmp_path: Path) -> None:
    (tmp_path / "k").write_text("filekey\n")
    raw_b = ["x ${env:PYCORDIS_A}", "${file:k}"]
    raw = {
        "a": "${env:PYCORDIS_A}",
        "b": raw_b,
        "c": ("${env:PYCORDIS_A}",),
        "lit": "$${x}",
        "bad": "${env:PYCORDIS_UNSET}",
        "n": 1,
    }
    value = {
        "a": "alpha",
        "b": ["x alpha", "filekey", "extra"],
        "c": ("alpha",),
        "lit": "${x}",
        "bad": "",
        "n": 2,
        "new": "alpha",
    }
    expected = {**raw, "b": [*raw_b, "extra"], "n": 2, "new": "alpha", "bad": ""}
    assert retemplate(value, raw, tmp_path) == expected
    # a changed value, a type change and a non-container stay as given
    assert retemplate({"a": "other", "b": 1}, raw) == {"a": "other", "b": 1}
    assert retemplate("alpha", {"a": 1}) == "alpha"
    assert retemplate(None, "${env:PYCORDIS_A}") is None


def test_secrets_and_leaks(tmp_path: Path) -> None:
    (tmp_path / "k").write_text("key\n")
    raw = {
        "a": "Bearer ${env:PYCORDIS_LONG}",
        "b": ["${file:k}", "${file:missing}", "${env:PYCORDIS_UNSET:-dflt}"],
        "c": ("${env:PYCORDIS_EMPTY}", "$${env:X}", "${oops", 3, "${env:PYCORDIS_A}"),
    }
    found = secrets(raw, tmp_path)
    # file contents always count; env values only from MIN_SECRET_LENGTH characters
    assert MIN_SECRET_LENGTH == 8
    assert found == {"longsecret", "key"}
    assert not leaks(raw, raw, found)
    assert leaks({"z": "has longsecret"}, raw, found)
    assert leaks({"longsecret": 1}, raw, found)  # keys too
    assert leaks({"b": ["x", "key"]}, raw, found)
    assert leaks(("longsecret",), None, found)
    assert not leaks({"b": ["dflt"], "n": 1, "s": "alpha"}, raw, found)
    # a value already in the file at the same path is not new
    assert not leaks({"a": "longsecret"}, {"a": "longsecret"}, found)


def test_leaks_non_string_leaves() -> None:
    found = {"98765432"}
    assert leaks({"pin": 98765432}, {"pin": "${env:PIN}"}, found)
    assert leaks([98765432.0], None, {"98765432.0"})
    assert not leaks({"pin": 98765432}, {"pin": 98765432}, found)  # in the file already
    assert not leaks({"pin": 1234, "x": None, "y": True}, None, found)
    # an integral float matches its integer form too
    assert leaks({"pin": 98765432.0}, None, found)
    assert not leaks({"pin": 98765432.5}, None, found)


def test_expand_tracks_expansions(tmp_path: Path) -> None:
    (tmp_path / "k").write_text("key\n")
    track = Expansions()
    raw = {"a": "x-${env:PYCORDIS_A}", "b": "${env:PYCORDIS_LONG}", "c": "${file:k}", "d": "lit"}
    assert expand(raw, tmp_path, track=track) == {
        "a": "x-alpha",
        "b": "longsecret",
        "c": "key",
        "d": "lit",
    }
    assert track.secrets == {"longsecret", "key"}
    assert track.leaves == {
        "x-${env:PYCORDIS_A}": {"x-alpha"},
        "${env:PYCORDIS_LONG}": {"longsecret"},
        "${file:k}": {"key"},
    }
    # partial results are recorded before an error is raised
    failed = Expansions()
    with pytest.raises(ValidationError):
        expand(["${file:k}", "${file:missing}"], tmp_path, track=failed)
    assert failed.secrets == {"key"}
    merged = Expansions({"old"}, {"${file:k}": {"old"}})
    merged.update(track)
    assert merged.secrets == {"old", "longsecret", "key"}
    assert merged.leaves["${file:k}"] == {"old", "key"}


def test_retemplate_uses_known_expansions(tmp_path: Path) -> None:
    (tmp_path / "k").write_text("new\n")
    raw = {"k": "${file:k}", "n": [1, {"x": "${file:k}"}]}
    known = {"${file:k}": {"old"}}
    assert retemplate({"k": "old", "n": [1, {"x": "new"}]}, raw, tmp_path, known) == raw
    assert retemplate({"k": "other"}, raw, tmp_path, known) == {"k": "other"}
    # nothing to restore: the very same object comes back
    same = {"k": "${file:k}", "n": [1, {"x": "z"}]}
    assert retemplate(same, raw, tmp_path, known) is same


@needs_fifo
def test_non_regular_paths_are_rejected_without_being_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.mkfifo(tmp_path / "fifo")  # no writer: opening it for reading could block
    (tmp_path / "dir").mkdir()
    opened: list[str] = []
    real_open = os.open

    def recording_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        opened.append(os.fspath(path))
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", recording_open)
    names = ["fifo", "dir"]
    if Path("/dev/null").exists():
        names.append("/dev/null")  # a device node
    with pytest.raises(ValidationError) as info:
        expand([f"${{file:{name}}}" for name in names], tmp_path)
    assert [issue.message.endswith("not a regular file") for issue in info.value.issues] == [
        True
    ] * len(names)
    assert opened == []


@needs_fifo
def test_a_path_swapped_after_the_type_check_is_rejected_on_its_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "secret").write_text("v")
    os.mkfifo(tmp_path / "fifo")
    real_open = os.open

    def swapped_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        # the regular file was replaced by a FIFO between the stat and the open
        return real_open(tmp_path / "fifo", flags, mode)

    monkeypatch.setattr(os, "open", swapped_open)
    with pytest.raises(ValidationError) as info:
        expand("${file:secret}", tmp_path)
    assert info.value.issues == (Issue("cannot read file secret: not a regular file", ()),)
