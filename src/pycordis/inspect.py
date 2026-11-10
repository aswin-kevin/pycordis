"""Fiber-tree introspection: ``ctx.inspect()`` snapshots and :func:`render`.

Note: this is ``pycordis.inspect``; pycordis modules import the standard library
``inspect`` with an absolute import, so the two never collide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, final

from pycordis.effect import EffectMeta
from pycordis.fiber import FiberState

if TYPE_CHECKING:
    from pycordis.fiber import Fiber

__all__ = ["FiberNode", "inspect_fiber", "render"]


@final
@dataclass
class FiberNode:
    """Snapshot of one fiber and the fibers started from its context."""

    name: str
    uid: int | None
    state: FiberState
    provides: list[str]
    injects: list[str]
    effects: list[EffectMeta]
    children: list[FiberNode] = field(default_factory=list["FiberNode"])


def inspect_fiber(fiber: Fiber) -> FiberNode:
    """Build the :class:`FiberNode` tree rooted at ``fiber`` from its registry."""
    children: dict[Fiber, list[Fiber]] = {}
    for runtime in list(fiber.ctx.registry.values()):
        for child in list(runtime.fibers):
            children.setdefault(child.parent.fiber, []).append(child)

    def build(node: Fiber) -> FiberNode:
        kids = sorted(children.get(node, ()), key=lambda f: -1 if f.uid is None else f.uid)
        return FiberNode(
            name=node.name,
            uid=node.uid,
            state=node.state,
            provides=[impl.name for impl in sorted(node.provisions, key=lambda i: i.uid)],
            injects=list(node.inject),
            effects=node.get_effects(),
            children=[build(kid) for kid in kids],
        )

    return build(fiber)


def _line(node: FiberNode) -> str:
    parts = [node.name, "#-" if node.uid is None else f"#{node.uid}", node.state.name]
    if node.provides:
        parts.append("provides: " + ", ".join(node.provides))
    if node.injects:
        parts.append("injects: " + ", ".join(node.injects))
    if node.effects:
        parts.append("effects: " + ", ".join(effect.label for effect in node.effects))
    return " ".join(parts)


def render(node: FiberNode) -> str:
    """Render a fiber tree as text, one fiber per line, with ``├─`` / ``└─`` branches.

    Each line reads ``name #uid STATE`` followed by the provided services,
    injected services and top-level effect labels, when present.
    """
    lines = [_line(node)]

    def walk(parent: FiberNode, prefix: str) -> None:
        for index, child in enumerate(parent.children):
            last = index == len(parent.children) - 1
            lines.append(prefix + ("└─ " if last else "├─ ") + _line(child))
            walk(child, prefix + ("   " if last else "│  "))

    walk(node, "")
    return "\n".join(lines)
