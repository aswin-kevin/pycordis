"""Typing contract: ``ctx.plugin(plugin, config)`` checks the config type.

Checked by tests/test_typing_contracts.py; lines ending in ``# expect-error``
(optionally ``: ruleName``) must be the only lines pyright reports.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, assert_type

from pycordis import Context, Plugin, Service


@dataclass
class PortConfig:
    port: int


@dataclass
class OtherConfig:
    name: str


def server(ctx: Context, config: PortConfig) -> None: ...


class Server:
    def __init__(self, ctx: Context, config: PortConfig) -> None: ...


class ServerObject:
    def apply(self, ctx: Context, config: PortConfig) -> None: ...


def untyped(ctx: Context, config: Any) -> None: ...


def function_plugins(ctx: Context) -> None:
    ctx.plugin(server, PortConfig(80))
    ctx.plugin(server, {"port": 80})  # raw mapping: validated by a Config schema
    ctx.plugin(server)
    ctx.plugin(server, None)
    ctx.plugin(server, OtherConfig("x"))  # expect-error: reportCallIssue
    ctx.plugin(server, 80)  # expect-error


def class_plugins(ctx: Context) -> None:
    ctx.plugin(Server, PortConfig(80))
    ctx.plugin(Server)
    ctx.plugin(Server, OtherConfig("x"))  # expect-error


class PlainService(Service[Any], provide="plain"):
    pass


def service_plugins(ctx: Context) -> None:
    ctx.plugin(PlainService)
    ctx.plugin(PlainService, PortConfig(80))
    ctx.plugin(PlainService, {"port": 80})


def object_plugins(ctx: Context) -> None:
    ctx.plugin(ServerObject(), PortConfig(80))
    ctx.plugin(ServerObject(), OtherConfig("x"))  # expect-error


def lambdas_infer_the_config(ctx: Context) -> None:
    ctx.plugin(lambda ctx, config: assert_type(config, PortConfig), PortConfig(80))
    ctx.plugin(lambda ctx, config: None)


def untyped_plugins(ctx: Context, opaque: object, anything: Any) -> None:
    ctx.plugin(untyped, 80)
    ctx.plugin(SimpleNamespace(apply=server), OtherConfig("x"))
    ctx.plugin(opaque)
    ctx.plugin(opaque, anything)
    ctx.plugin(opaque, OtherConfig("x"))  # expect-error


def through_alias(ctx: Context, plugin: Plugin[PortConfig]) -> None:
    ctx.plugin(plugin, PortConfig(80))
    ctx.plugin(plugin, OtherConfig("x"))  # expect-error
