from __future__ import annotations

import importlib.metadata
import json
import os.path
from collections.abc import Callable
from typing import Any

import pytest

from pycordis.loader import import_plugin


def test_module_attribute() -> None:
    assert import_plugin("json:dumps", ["*"]) is json.dumps
    assert import_plugin("json:JSONDecoder.decode", ["*"]) is json.JSONDecoder.decode
    assert import_plugin("os.path:join", ["os.*"]) is os.path.join


def test_allowlist_matches_module_part() -> None:
    assert import_plugin("json:dumps", ["js*"]) is json.dumps
    with pytest.raises(PermissionError) as info:
        import_plugin("json:dumps", ["my_agent.*", "pycordis.*"])
    assert str(info.value) == 'plugin module "json" is not allowed by loader.allow'
    with pytest.raises(PermissionError):
        import_plugin("os.path:join", ["os"])
    with pytest.raises(PermissionError):
        import_plugin("json:dumps", [])


def test_denied_module_is_not_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(importlib, "import_module", calls.append)
    with pytest.raises(PermissionError):
        import_plugin("evil.mod:x", ["safe.*"])
    assert calls == []


def test_bad_references() -> None:
    with pytest.raises(ImportError, match=r'module "json" has no attribute "nope.x"'):
        import_plugin("json:nope.x", ["*"])
    with pytest.raises(ModuleNotFoundError):
        import_plugin("pycordis_no_such_module:x", ["*"])
    for bad in ("json:", ":x", "json:a..b", ""):
        with pytest.raises(ValueError, match="invalid plugin reference"):
            import_plugin(bad, ["*"])


def fake_entry_points(eps: list[tuple[str, str]]) -> Callable[..., Any]:
    def entry_points(**params: Any) -> importlib.metadata.EntryPoints:
        assert params == {"group": "pycordis.plugins"}
        return importlib.metadata.EntryPoints(
            importlib.metadata.EntryPoint(name, value, "pycordis.plugins") for name, value in eps
        )

    return entry_points


def test_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib.metadata, "entry_points", fake_entry_points([("my-search", "json:dumps")])
    )
    assert import_plugin("my-search", ["json"]) is json.dumps


def test_entry_point_allowlist_uses_ep_module(monkeypatch: pytest.MonkeyPatch) -> None:
    # loading would raise ModuleNotFoundError, so PermissionError proves nothing was imported
    eps = fake_entry_points([("search", "evil_pycordis_missing.tools:x")])
    monkeypatch.setattr(importlib.metadata, "entry_points", eps)
    with pytest.raises(PermissionError) as info:
        import_plugin("search", ["my_agent.*", "search"])
    assert str(info.value) == (
        'plugin module "evil_pycordis_missing.tools" is not allowed by loader.allow'
    )


def test_entry_point_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points([]))
    with pytest.raises(ImportError, match=r'no plugin named "search" in entry-point group'):
        import_plugin("search", ["*"])


def test_package_star_admits_package_itself() -> None:
    assert import_plugin("json:dumps", ["json.*"]) is json.dumps
    assert import_plugin("json.decoder:JSONDecoder", ["json.*"]) is json.JSONDecoder
    with pytest.raises(PermissionError):
        import_plugin("jsonx:y", ["json.*"])


@pytest.mark.parametrize(
    "use", ["json:__loader__.load_module", "json:__builtins__", "json:_private", "json:a._b"]
)
def test_private_attributes_rejected(use: str) -> None:
    with pytest.raises(ValueError, match="invalid plugin reference"):
        import_plugin(use, ["*"])


@pytest.mark.parametrize(
    ("use", "allow", "module"),
    [
        ("os.path:os.system", ["os.path"], "os"),
        ("json:codecs.open", ["json"], "codecs"),
        ("json:decoder.JSONDecoder", ["json"], "json.decoder"),
    ],
)
def test_reexported_module_must_be_allowed(use: str, allow: list[str], module: str) -> None:
    with pytest.raises(PermissionError) as info:
        import_plugin(use, allow)
    assert str(info.value) == f'plugin module "{module}" is not allowed by loader.allow'


def test_reexported_allowed_module_resolves() -> None:
    assert import_plugin("json:decoder.JSONDecoder", ["json.*"]) is json.JSONDecoder
