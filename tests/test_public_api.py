import importlib
import inspect

import pycordis

EXPECTED = {
    "Context",
    "Fiber",
    "FiberState",
    "Service",
    "Inject",
    "inject",
    "plugin_meta",
    "Plugin",
    "PluginObject",
    "ServiceKey",
    "EventKey",
    "EventOptions",
    "CordisError",
    "ValidationError",
    "Issue",
    "Volatile",
    "create_volatile",
    "Logger",
    "LoggerLevel",
    "Message",
    "Exporter",
    "ConsoleExporter",
    "StdlibLoggingExporter",
    "capture_stdlib_logging",
    "EffectMeta",
    "EffectDisposer",
    "ListenerDisposer",
    "ServiceAccessError",
    "run",
    "current_context",
    "use_context",
    "Collection",
    "CollectionChange",
    "Scope",
    "Backoff",
    "RestartPolicy",
    "LoadTimeoutError",
    "Diagnosis",
    "Reason",
    "render_diagnosis",
    "TELEMETRY",
    "LifecycleEvent",
    "LifecycleKind",
    "FiberStats",
    "HealthServer",
    "JsonExporter",
    "config_json_schema",
    "loader_json_schema",
}


def test_all_matches_expected() -> None:
    assert sorted(set(pycordis.__all__)) == sorted(EXPECTED)
    assert len(pycordis.__all__) == len(EXPECTED)


def test_every_name_importable_and_documented() -> None:
    for name in pycordis.__all__:
        value = getattr(pycordis, name)
        assert inspect.getdoc(value), name


def test_inspect_submodule_does_not_shadow_stdlib() -> None:
    module = importlib.import_module("pycordis.inspect")
    assert hasattr(module, "render")
    # pycordis modules keep using the standard library module
    import pycordis.context as context_module

    assert context_module.inspect is inspect
    assert inspect.getattr_static is not None


def test_run() -> None:
    async def main() -> int:
        async with pycordis.Context() as root:
            return await root.spawn(_answer())

    assert pycordis.run(main) == 42


async def _answer() -> int:
    return 42


def test_extension_points_are_not_final() -> None:
    """Classes users subclass (extension points, exporter bases, test doubles) stay open."""
    from pycordis.health import HealthServer
    from pycordis.json_exporter import JsonExporter
    from pycordis.loader import Loader, LoaderConfig
    from pycordis.testing import MemoryExporter

    open_classes: list[type] = [
        pycordis.Context,
        pycordis.Service,
        pycordis.Fiber,
        pycordis.Logger,
        pycordis.ConsoleExporter,
        pycordis.StdlibLoggingExporter,
        pycordis.Collection,
        JsonExporter,
        MemoryExporter,
        Loader,
        LoaderConfig,
        HealthServer,
        HealthServer.Config,
    ]
    assert [cls.__name__ for cls in open_classes if getattr(cls, "__final__", False)] == []

    class TaggedLogger(pycordis.Logger):  # pyright would reject this for a final class
        tag = "x"

    assert TaggedLogger.tag == "x"
