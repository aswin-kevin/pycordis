"""Typing contract: service keys type ``use`` / ``get`` / ``provide`` / ``set``.

Checked by tests/test_typing_contracts.py; lines ending in ``# expect-error``
(optionally ``: ruleName``) must be the only lines pyright reports.
"""

from typing import Any, assert_type

from pycordis import Context, Service, ServiceKey, plugin_meta


class LLMClient:
    def complete(self, prompt: str) -> str:
        return prompt


LLM = ServiceKey[LLMClient]("llm")
ANIMAL = ServiceKey[object]("animal")


def reads(ctx: Context) -> None:
    assert_type(ctx.use(LLM), LLMClient)
    assert_type(ctx.get(LLM), LLMClient | None)
    assert_type(ctx.get("llm"), Any)
    ctx.use("llm")  # expect-error


def writes(ctx: Context) -> None:
    ctx.provide(LLM, LLMClient())
    ctx.provide("llm", "anything")
    ctx.set(LLM, LLMClient())
    ctx.provide(LLM, "not a client")  # expect-error: reportArgumentType
    ctx.set(LLM, 42)  # expect-error
    ctx.provide(LLM)  # expect-error


class LocalClient(LLMClient):
    pass


LOCAL = ServiceKey[LocalClient]("local")


def widen() -> ServiceKey[LLMClient]:
    return LOCAL  # expect-error: reportReturnType


def narrow() -> ServiceKey[LocalClient]:
    return LLM  # expect-error: reportReturnType


def subclass_values(ctx: Context) -> None:
    ctx.provide(LLM, LocalClient())
    ctx.provide(LOCAL, LLMClient())  # expect-error: reportArgumentType
    ctx.provide(ANIMAL, LLMClient())


class Client(Service[Any], provide=LLM):
    pass


@plugin_meta(name="chat", inject=[LLM, "db"], provide=LLM)
def chat(ctx: Context, config: None) -> None:
    ctx.use(LLM).complete("hi")


@plugin_meta(inject={LLM: {"model": "small"}, "db": None})
def mapped(ctx: Context, config: None) -> None: ...


@plugin_meta(inject="llm")  # expect-error
def bad_inject(ctx: Context, config: None) -> None: ...


def inject_lists(ctx: Context) -> None:
    names: list[str] = ["db"]
    keys = [LLM]
    ctx.inject(names, lambda ctx, config: None)
    ctx.inject(keys, lambda ctx, config: None)
    ctx.inject((LLM, "db"), lambda ctx, config: None)
