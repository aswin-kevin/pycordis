"""Run nested, re-entrant work without growing the Python stack.

Internal — not part of the public API.

A service change cascades: a provider leaving ``ACTIVE`` notifies its
dependents, each of which leaves ``ACTIVE`` and notifies its own dependents,
and so on. Written as plain recursion, a 1000-plugin dependency chain overflows
the interpreter stack. :class:`Cascade` runs each unit of work as a generator
of *steps* and keeps the nesting on an explicit stack instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import final

type Steps = Iterator[None]
"""A unit of work: each ``next()`` runs one step; it may start nested work."""


@final
class Cascade:
    """A trampoline for nested work, preserving depth-first (recursive) order.

    :meth:`run` called outside a cascade runs the work to completion before
    returning. Called from inside a step, it only records the work: once that
    step ends, the recorded work runs (in call order, each to completion)
    before the step's own generator resumes. That is the order plain recursion
    would produce, except that the remainder of the step itself runs before the
    nested work does, so a step should end right after anything that can start
    nested work (yield there).

    Example::

        cascade = Cascade()

        def visit(node: Node) -> Steps:
            for child in node.children:
                cascade.run(visit(child))
                yield  # the child's subtree runs here, before the next child

        cascade.run(visit(root))

    An exception raised by a step propagates out of the outermost :meth:`run`
    and abandons every unfinished unit, like it would unwind a recursion.
    """

    __slots__ = ("_running", "_started")

    def __init__(self) -> None:
        self._started: list[Steps] = []
        self._running = False

    def run(self, steps: Steps) -> None:
        """Run ``steps`` now, or right after the current step when called from one."""
        if self._running:
            self._started.append(steps)
            return
        self._running = True
        stack = [steps]
        try:
            while stack:
                try:
                    next(stack[-1])
                except StopIteration:
                    stack.pop()
                if self._started:
                    # the first work started during the step must run first,
                    # so it goes on top of the stack
                    stack.extend(reversed(self._started))
                    self._started.clear()
        finally:
            self._running = False
            self._started.clear()
