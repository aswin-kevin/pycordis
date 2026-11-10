"""pycordis: asyncio plugin framework."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from pycordis.collection import Collection, CollectionChange
from pycordis.context import Context
from pycordis.current import current_context, use_context
from pycordis.diagnose import Diagnosis, Reason, render_diagnosis
from pycordis.effect import EffectDisposer, EffectMeta, ListenerDisposer
from pycordis.errors import CordisError, Issue, ServiceAccessError, ValidationError
from pycordis.events import EventKey, EventOptions
from pycordis.fiber import Fiber, FiberState
from pycordis.health import HealthServer
from pycordis.json_exporter import JsonExporter
from pycordis.jsonschema import config_json_schema, loader_json_schema
from pycordis.keys import ServiceKey
from pycordis.logger import (
    ConsoleExporter,
    Exporter,
    Logger,
    LoggerLevel,
    Message,
    StdlibLoggingExporter,
    capture_stdlib_logging,
)
from pycordis.registry import Inject, Plugin, PluginObject, inject, plugin_meta
from pycordis.scope import Scope
from pycordis.service import Service
from pycordis.supervise import Backoff, LoadTimeoutError, RestartPolicy
from pycordis.telemetry import TELEMETRY, FiberStats, LifecycleEvent, LifecycleKind
from pycordis.volatile import Volatile, create_volatile

__all__ = [
    "TELEMETRY",
    "Backoff",
    "Collection",
    "CollectionChange",
    "ConsoleExporter",
    "Context",
    "CordisError",
    "Diagnosis",
    "EffectDisposer",
    "EffectMeta",
    "EventKey",
    "EventOptions",
    "Exporter",
    "Fiber",
    "FiberState",
    "FiberStats",
    "HealthServer",
    "Inject",
    "Issue",
    "JsonExporter",
    "LifecycleEvent",
    "LifecycleKind",
    "ListenerDisposer",
    "LoadTimeoutError",
    "Logger",
    "LoggerLevel",
    "Message",
    "Plugin",
    "PluginObject",
    "Reason",
    "RestartPolicy",
    "Scope",
    "Service",
    "ServiceAccessError",
    "ServiceKey",
    "StdlibLoggingExporter",
    "ValidationError",
    "Volatile",
    "capture_stdlib_logging",
    "config_json_schema",
    "create_volatile",
    "current_context",
    "inject",
    "loader_json_schema",
    "plugin_meta",
    "render_diagnosis",
    "run",
    "use_context",
]


def run[T](main: Callable[[], Coroutine[Any, Any, T]]) -> T:  # asyncio's coroutine typing
    """Run ``main()`` in a new event loop (:func:`asyncio.run`) and return its result.

    Example: ``pycordis.run(main)`` with ``async def main() -> None: ...``.
    """
    return asyncio.run(main())
