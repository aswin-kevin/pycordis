"""Typed service keys: a service name carrying its value type for static checking.

A :class:`ServiceKey` is sugar over a plain service name. Every API that takes a
service name also takes a key and resolves it to :attr:`ServiceKey.name` once,
at the call boundary, so runtime resolution stays string-based::

    LLM = ServiceKey[LLMClient]("llm")

    ctx.provide(LLM, OpenAIClient())  # a type checker verifies the value
    client = ctx.use(LLM)  # typed as LLMClient
"""

from __future__ import annotations

from typing import Any, final

__all__ = ["ServiceKey", "ServiceName", "service_name"]


@final
class ServiceKey[T]:
    """A service name typed with the type of the service it names.

    Keys compare and hash by identity. Two keys with the same name share one
    service slot at runtime: types are not checked across keys, so give each
    service a single key. Names starting with ``_`` are private context
    attributes and are rejected. Example::

        COUNTER = ServiceKey[int]("counter")
        ctx.provide(COUNTER, 0)
        ctx.use(COUNTER) + 1
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("a service key needs a non-empty service name")
        if name.startswith("_"):
            raise ValueError(
                f"service key name {name!r} must not start with '_' "
                "(underscore names are private context attributes)"
            )
        self._name = name

    @property
    def name(self) -> str:
        """The service name this key resolves to (e.g. ``"llm"``).

        Example: ``ServiceKey[int]("counter").name == "counter"``.
        """
        return self._name

    def coerce(self, value: T) -> T:
        """Return ``value`` unchanged, typed as this key's service.

        Internal: used to check provided values against the key's type. Taking
        and returning ``T`` also makes the key invariant: a key both accepts
        values (``ctx.provide``) and produces them (``ctx.use``), so a
        ``ServiceKey[Dog]`` must not pass as a ``ServiceKey[Animal]``.

        Example: ``COUNTER.coerce(0)`` type-checks only for an ``int``.
        """
        return value

    def __repr__(self) -> str:
        return f"ServiceKey({self._name!r})"


type ServiceName = str | ServiceKey[Any]
"""A service named by a plain string or a typed :class:`ServiceKey`.

``ServiceKey[Any]``: a key of any service type (keys are invariant, so
``ServiceKey[object]`` would accept only keys typed exactly ``object``).
"""


def service_name(name: ServiceName) -> str:
    """The plain service name behind ``name``.

    Example::

        service_name("llm") == service_name(ServiceKey[object]("llm")) == "llm"
    """
    return name if isinstance(name, str) else name.name
