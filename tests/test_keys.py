"""Typed service keys (``ServiceKey``) and typed plugin metadata (``plugin_meta``)."""

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from pycordis import ServiceKey, plugin_meta
from pycordis._utils import symbols
from pycordis.context import Context
from pycordis.errors import ServiceAccessError
from pycordis.fiber import FiberState
from pycordis.keys import service_name
from pycordis.registry import inject
from pycordis.service import Service


class LLMClient:
    def __init__(self, model: str) -> None:
        self.model = model


LLM = ServiceKey[LLMClient]("llm")


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


# -- ServiceKey ------------------------------------------------------------------


def test_key_exposes_name_and_repr() -> None:
    assert LLM.name == "llm"
    assert repr(LLM) == "ServiceKey('llm')"


def test_keys_compare_and_hash_by_identity() -> None:
    twin = ServiceKey[LLMClient]("llm")
    assert twin != LLM
    assert len({LLM, twin, LLM}) == 2


def test_coerce_returns_the_value_unchanged() -> None:
    client = LLMClient("small")
    assert LLM.coerce(client) is client


def test_key_requires_a_name() -> None:
    with pytest.raises(ValueError, match="non-empty service name"):
        ServiceKey[object]("")


def test_key_rejects_private_names() -> None:
    with pytest.raises(ValueError, match="must not start with '_'"):
        ServiceKey[object]("_secret")


async def test_collision_with_a_key_names_both_providers() -> None:
    root = Context()
    root.provide(LLM, LLMClient("small"))

    def other(_ctx: Context, _config: Any) -> None: ...

    fiber = await root.plugin(other)
    with pytest.raises(ServiceAccessError) as info:
        fiber.ctx.provide("llm", object())
    assert str(info.value) == (
        'service "llm" has been registered at <root> '
        "(registered by ServiceKey('llm') at <root>, then provided by \"llm\" at <other>)"
    )


def test_service_name_accepts_strings_and_keys() -> None:
    assert service_name("llm") == "llm"
    assert service_name(LLM) == "llm"


# -- provide / use / get / set ---------------------------------------------------


async def test_provide_with_key_is_readable_by_name_and_key() -> None:
    root = Context()
    client = LLMClient("small")
    root.provide(LLM, client)
    assert root.use(LLM) is client
    assert root.get(LLM) is client
    assert root.llm is client


async def test_get_with_key_returns_none_when_not_provided() -> None:
    root = Context()
    assert root.get(LLM) is None
    assert root.get(LLM, strict=False) is None


async def test_set_with_key_replaces_the_value() -> None:
    root = Context()
    root.provide(LLM, LLMClient("small"))
    large = LLMClient("large")
    assert root.set(LLM, large) is True
    assert root.use(LLM) is large


async def test_use_without_inject_raises_like_attribute_access() -> None:
    root = Context()
    await root.plugin(lambda ctx, _config: ctx.provide(LLM, LLMClient("small")))
    errors: list[ServiceAccessError] = []

    # a sibling plugin's service needs `inject`, whichever way it is read
    def plugin(ctx: Context, _config: Any) -> None:
        for read in (lambda: ctx.use(LLM), lambda: ctx.llm):
            with pytest.raises(ServiceAccessError) as info:
                read()
            errors.append(info.value)

    await root.plugin(plugin)
    by_key, by_attribute = errors
    assert str(by_key) == str(by_attribute) == 'cannot get property "llm" without inject'
    assert by_key.name == by_attribute.name == "llm"


async def test_use_of_unknown_key_at_root_raises() -> None:
    root = Context()
    with pytest.raises(ServiceAccessError, match='cannot get property "llm"'):
        root.use(LLM)


# -- inject with keys ------------------------------------------------------------


async def test_inject_list_with_key_waits_for_the_service() -> None:
    root = Context()
    seen: list[LLMClient] = []

    @plugin_meta(inject=[LLM, "db"])
    def chat(ctx: Context, _config: Any) -> None:
        seen.append(ctx.use(LLM))

    fiber = root.plugin(chat)
    assert fiber.inject == {"llm": None, "db": None}
    await fiber
    assert fiber.state == FiberState.PENDING
    root.provide("db", object())
    client = LLMClient("small")
    root.provide(LLM, client)
    await _settle()
    await fiber
    assert fiber.state == FiberState.ACTIVE
    assert seen == [client]


async def test_inject_mapping_with_key_keeps_its_config() -> None:
    root = Context()
    fiber = root.inject({LLM: {"model": "tiny"}}, lambda _ctx, _config: None)
    assert fiber.inject == {"llm": {"model": "tiny"}}


async def test_inject_decorator_accepts_a_key() -> None:
    @inject(LLM)
    class Chat:
        def __init__(self, ctx: Context, _config: Any) -> None:
            self.client = ctx.use(LLM)

    assert vars(Chat)["inject"] == {"llm": None}
    root = Context()
    fiber = root.plugin(Chat)
    await fiber
    assert fiber.state == FiberState.PENDING
    root.provide(LLM, LLMClient("small"))
    await _settle()
    assert fiber.state == FiberState.ACTIVE


# -- Service with a key ------------------------------------------------------------


async def test_service_subclass_provides_under_a_key() -> None:
    class Client(Service[Any], provide=LLM):
        pass

    assert Client.provide == "llm"
    root = Context()
    client = Client(root)
    # reads return a proxy traced to the reading context; compare the original
    assert getattr(root.get(LLM), symbols.original) is client


# -- plugin_meta -------------------------------------------------------------------


@dataclass
class PortConfig:
    port: int


def test_plugin_meta_sets_function_metadata() -> None:
    @plugin_meta(name="server", inject=[LLM], provide=["http", LLM], Config=PortConfig)
    def server(_ctx: Context, _config: PortConfig) -> None: ...

    assert vars(server) == {
        "name": "server",
        "inject": {"llm": None},
        "provide": ["http", "llm"],
        "Config": PortConfig,
    }


def test_plugin_meta_sets_only_given_metadata_and_single_provide() -> None:
    @plugin_meta(provide=LLM)
    class Provider:
        def __init__(self, ctx: Context, _config: Any) -> None:
            ctx.provide(LLM, LLMClient("small"))

    assert vars(Provider)["provide"] == "llm"
    assert "name" not in vars(Provider)
    assert "inject" not in vars(Provider)


def test_plugin_meta_rejects_a_string_inject() -> None:
    with pytest.raises(TypeError, match="inject must be a list"):
        plugin_meta(inject="llm")  # pyright: ignore[reportArgumentType] - runtime check


async def test_plugin_meta_config_and_name_are_used_by_the_registry() -> None:
    seen: list[PortConfig] = []

    @plugin_meta(name="server", Config=PortConfig)
    def server(_ctx: Context, config: PortConfig) -> None:
        seen.append(config)

    root = Context()
    fiber = await root.plugin(server, {"port": 80})
    assert fiber.name == "server"
    built = PortConfig(81)
    await root.plugin(server, built)
    assert seen == [PortConfig(80), built]
    assert seen[1] is built
