"""Diagnostics: why a fiber is not ACTIVE, and dependency cycles (spec §3.7).

``ctx.diagnose()`` (or ``fiber.diagnose()``) snapshots the fiber tree as a
:class:`Diagnosis`; every fiber that is not ``ACTIVE`` carries the
:class:`Reason` s it is stuck, computed from the live state:

- ``missing-service``: an injected service is not provided in the fiber's
  isolation scope (the detail names a plugin declaring it, if any);
- ``provider-failed``: a plugin declaring the service in that scope is ``FAILED``
  (the detail carries its error);
- ``check-failed``: the service is provided but its ``check()`` rejected it
  (returned False, or raised: the detail names the error);
- ``isolated``: the service is provided (or declared) only in another isolated
  scope;
- ``cycle``: the fiber is part of a dependency cycle: fibers that wait for each
  other, each hop blocked because every provider of the service is stuck too.
  ``path`` spells it, e.g. ``a -> "llm" -> b -> "cache" -> a``: ``a`` waits for
  ``llm``, which ``b`` would provide, ``b`` waits for ``cache``, which ``a``
  would provide. Each cycle (a group of fibers waiting for each other) is
  numbered and its path shown once, on the first hop; its other members refer
  to it;
- ``failed`` (with the error) and ``backoff`` (a restart is scheduled);
- ``loading`` / ``unloading``: a transition is in flight.

Which plugin *would* provide a service comes from its declared ``provide``
metadata (:func:`~pycordis.registry.plugin_meta`, ``Service`` subclasses) and
from what fibers currently provide; services a plugin provides only from its
body, before it ever loaded, cannot be known. Example::

    print(render_diagnosis(ctx.diagnose()))
    # root #0 ACTIVE
    # ├─ a #1 PENDING
    # │     cycle: cycle #1: a -> "llm" -> b -> "cache" -> a
    # └─ b #2 PENDING
    #       cycle: waits for "cache" in cycle #1 (see a #1)

A cycle is also logged as a warning (logger ``diagnose``) once each time it
forms, checked on the event-loop turn after (see :class:`CycleWatch`): cordis
would leave its plugins ``PENDING`` without a word.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Literal, final

from pycordis._context_internals import isolate_map, lookup
from pycordis.fiber_core import FiberState, report_error

if TYPE_CHECKING:
    from pycordis.fiber_lifecycle import FiberLifecycle
    from pycordis.reflect import Impl
    from pycordis.registry import RegistryService

__all__ = [
    "CycleWatch",
    "Diagnosis",
    "Reason",
    "ReasonKind",
    "diagnose_fiber",
    "format_path",
    "render_diagnosis",
]

type ReasonKind = Literal[
    "missing-service",
    "provider-failed",
    "check-failed",
    "isolated",
    "cycle",
    "backoff",
    "failed",
    "loading",
    "unloading",
]
"""Why a fiber is not ACTIVE (see :mod:`pycordis.diagnose`)."""


@final
@dataclass(frozen=True, slots=True)
class Reason:
    """One reason a fiber is not ACTIVE.

    ``service`` is the injected service concerned, if any; ``path`` the
    dependency cycle for kind ``"cycle"``. Example::

        Reason("missing-service", "db", 'no plugin provides "db"')
        str(reason) == 'missing-service: no plugin provides "db"'
    """

    kind: ReasonKind
    """What kind of problem this is."""
    service: str | None
    """The injected service the fiber waits for, or ``None``."""
    detail: str
    """A human-readable explanation."""
    path: tuple[str, ...] = ()
    """For a cycle: fiber and service names alternating, ending where it starts."""

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


@final
@dataclass(frozen=True)
class Diagnosis:
    """A fiber's state and the reasons it is not ACTIVE, with the fibers below it.

    Built by ``ctx.diagnose()`` / ``fiber.diagnose()``. Example::

        diagnosis = ctx.diagnose()
        if not diagnosis.healthy:
            print(render_diagnosis(diagnosis.unhealthy()))
    """

    fiber: str
    """The fiber's name."""
    uid: int | None
    """The fiber's uid (``None`` once disposed)."""
    state: FiberState
    """The fiber's lifecycle state."""
    reasons: tuple[Reason, ...]
    """Why the fiber is not ACTIVE (empty when it is)."""
    children: tuple[Diagnosis, ...]
    """The fibers started from this fiber's context, by uid."""

    @property
    def healthy(self) -> bool:
        """Whether this fiber and every fiber below it is ACTIVE.

        Example: ``assert ctx.diagnose().healthy``.
        """
        return self.state == FiberState.ACTIVE and all(child.healthy for child in self.children)

    def unhealthy(self) -> Diagnosis | None:
        """This tree reduced to the fibers that are not ACTIVE and their ancestors.

        ``None`` when the whole tree is :attr:`healthy`. Example::

            stuck = ctx.diagnose().unhealthy()
            print("all good" if stuck is None else render_diagnosis(stuck))
        """
        children = tuple(
            pruned for child in self.children if (pruned := child.unhealthy()) is not None
        )
        if self.state == FiberState.ACTIVE and not children:
            return None
        return replace(self, children=children)


def render_diagnosis(diagnosis: Diagnosis, *, labels: Mapping[int, str] | None = None) -> str:
    """Render a diagnosis tree as text: one ``name #uid STATE`` line per fiber.

    Each reason follows its fiber's line, indented below it, as ``kind:
    detail``; branches are drawn with ``├─`` / ``└─``. ``labels`` maps fiber
    uids to a note shown in brackets after the uid (``pycordis doctor`` passes
    loader entry ids). Example::

        root #0 ACTIVE
        └─ app #1 [entry web] PENDING
              missing-service: no plugin declares or currently provides "db"
    """
    lines: list[str] = []
    notes = labels or {}

    def walk(node: Diagnosis, head: str, below: str) -> None:
        uid = "-" if node.uid is None else node.uid
        note = None if node.uid is None else notes.get(node.uid)
        title = f"{node.fiber} #{uid}" + ("" if note is None else f" [{note}]")
        lines.append(f"{head}{title} {node.state.name}")
        indent = below + ("│  " if node.children else "   ")
        lines.extend(indent + str(reason) for reason in node.reasons)
        for index, child in enumerate(node.children):
            last = index == len(node.children) - 1
            walk(child, below + ("└─ " if last else "├─ "), below + ("   " if last else "│  "))

    walk(diagnosis, "", "")
    return "\n".join(lines)


def diagnose_fiber(fiber: FiberLifecycle) -> Diagnosis:
    """Diagnose ``fiber`` and the fibers below it (``fiber.diagnose()``).

    Example: ``diagnose_fiber(ctx.fiber).healthy``.

    Internal — call ``ctx.diagnose()`` or ``fiber.diagnose()``.
    """
    graph = _WaitGraph(fiber)
    children: dict[FiberLifecycle, list[FiberLifecycle]] = {}
    for node in graph.fibers:
        children.setdefault(node.parent.fiber, []).append(node)

    def build(node: FiberLifecycle) -> Diagnosis:
        return Diagnosis(
            fiber=node.name,
            uid=node.uid,
            state=node.state,
            reasons=tuple(graph.reasons(node)),
            children=tuple(build(child) for child in children.get(node, ())),
        )

    return build(fiber)


# -- the waits-for graph ---------------------------------------------------------------

type _Step = tuple[FiberLifecycle, str]
"""One hop of a cycle: a fiber and the service it waits for."""


@final
@dataclass(frozen=True)
class _Cycle:
    """One reported cycle: its number, the fiber it is spelled from and its path."""

    number: int
    start: FiberLifecycle
    path: tuple[str, ...]


def _uid(fiber: FiberLifecycle) -> int:
    return -1 if fiber.uid is None else fiber.uid


def _ref(fiber: FiberLifecycle) -> str:
    return f"{fiber.name} #{'-' if fiber.uid is None else fiber.uid}"


def _application_fibers(fiber: FiberLifecycle) -> list[FiberLifecycle]:
    """Every plugin fiber of ``fiber``'s application, by uid."""
    registry: RegistryService = lookup(fiber.ctx, "registry")
    fibers: list[FiberLifecycle] = [
        node for runtime in list(registry.values()) for node in runtime.fibers
    ]
    return sorted(fibers, key=_uid)


def _declares(fiber: FiberLifecycle, name: str) -> bool:
    return fiber.runtime is not None and name in fiber.runtime.provide


def _unsatisfied(fiber: FiberLifecycle) -> list[str]:
    return [name for name in fiber.inject if not fiber.satisfied(name)]


def _progressing(fiber: FiberLifecycle) -> bool:
    """Whether a provider is on its way to (re)providing: loading, active, or about to retry."""
    state = fiber.state
    if state == FiberState.FAILED:
        return fiber.supervisor is not None and fiber.supervisor.pending is not None
    return state != FiberState.PENDING


def format_path(path: Sequence[str]) -> str:
    """A cycle path as text, services quoted: ``a -> "llm" -> b -> "cache" -> a``.

    ``path`` alternates fiber and service names (:attr:`Reason.path`). Example:
    ``format_path(("a", "llm", "a")) == 'a -> "llm" -> a'``.

    Internal — not part of the public API.
    """
    return " -> ".join(f'"{name}"' if index % 2 else name for index, name in enumerate(path))


def _path(steps: Sequence[_Step]) -> tuple[str, ...]:
    """The names of a cycle's hops, ending where it starts."""
    names = [name for fiber, service in steps for name in (fiber.name, service)]
    return (*names, steps[0][0].name)


def _canonical(steps: Sequence[_Step]) -> list[_Step]:
    """The rotation of a cycle starting at its lowest-uid fiber (one spelling per cycle)."""
    start = min(range(len(steps)), key=lambda index: _uid(steps[index][0]))
    return [*steps[start:], *steps[:start]]


def _strongly_connected[T](nodes: Sequence[T], successors: Callable[[T], list[T]]) -> list[list[T]]:
    """The strongly connected components of a graph (Tarjan's algorithm, iterative).

    Components come out in reverse topological order; each lists its nodes
    in discovery order. Iterative, so a long chain cannot exhaust the stack.
    """
    index: dict[T, int] = {}
    low: dict[T, int] = {}
    stack: list[T] = []
    on_stack: set[T] = set()
    components: list[list[T]] = []

    def enter(node: T) -> None:
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)

    for root in nodes:
        if root in index:
            continue
        enter(root)
        work: list[tuple[T, Iterator[T]]] = [(root, iter(successors(root)))]
        while work:
            node, targets = work[-1]
            child = next((t for t in targets if _visit(t, node, index, low, on_stack)), None)
            if child is not None:
                enter(child)
                work.append((child, iter(successors(child))))
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[T] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member is node:
                        break
                components.append(component)
    return components


def _visit[T](target: T, node: T, index: dict[T, int], low: dict[T, int], on_stack: set[T]) -> bool:
    """Tarjan step for edge ``node -> target``: whether ``target`` is new (to descend into)."""
    if target not in index:
        return True
    if target in on_stack:
        low[node] = min(low[node], index[target])
    return False


@final
class _WaitGraph:
    """Who waits for whom among one application's fibers, computed on demand.

    An edge ``waiter -(service)-> provider`` exists when the PENDING ``waiter``
    lacks ``service`` and every fiber that provides or declares it in the
    waiter's scope is stuck (PENDING, or FAILED without a scheduled restart);
    the edges then lead to the PENDING ones. A hop that some loading, active
    or retrying provider will satisfy is no part of a cycle.
    """

    def __init__(self, member: FiberLifecycle) -> None:
        self._member = member
        self._candidates: dict[str, list[FiberLifecycle]] | None = None
        self._edges: dict[FiberLifecycle, list[tuple[FiberLifecycle, str]]] = {}
        self._cycles: dict[_Step, _Cycle] | None = None

    @cached_property
    def fibers(self) -> list[FiberLifecycle]:
        """Every plugin fiber of the application, by uid; listed on first use.

        Lazy: a cycle check whose entered fibers wait only for progressing
        providers never needs the application-wide list.
        """
        return _application_fibers(self._member)

    def _by_name(self) -> dict[str, list[FiberLifecycle]]:
        """Fibers declaring or holding each service name (any scope), by uid; built once."""
        if self._candidates is None:
            self._candidates = {}
            for fiber in self.fibers:
                names = dict.fromkeys(fiber.runtime.provide if fiber.runtime else ())
                names.update(dict.fromkeys(impl.name for impl in fiber.provisions))
                for name in names:
                    self._candidates.setdefault(name, []).append(fiber)
        return self._candidates

    def providers(self, waiter: FiberLifecycle, name: str) -> list[FiberLifecycle]:
        """Fibers that provide, or declare they provide, ``name`` in ``waiter``'s scope."""
        label = isolate_map(waiter.ctx).get(name)
        return [
            fiber
            for fiber in self._by_name().get(name, ())
            if (_declares(fiber, name) and isolate_map(fiber.ctx).get(name) is label)
            or any(impl.name == name and impl.isolate is label for impl in fiber.provisions)
        ]

    def edges(self, waiter: FiberLifecycle) -> list[tuple[FiberLifecycle, str]]:
        """``(provider, service)`` for each stuck hop of ``waiter`` (a PENDING fiber)."""
        edges = self._edges.get(waiter)
        if edges is None:
            edges = self._edges[waiter] = []
            for name in _unsatisfied(waiter):
                edges.extend((provider, name) for provider in self._blocked_on(waiter, name))
        return edges

    def _blocked_on(self, waiter: FiberLifecycle, name: str) -> list[FiberLifecycle]:
        """The PENDING providers ``waiter`` waits for on ``name``; empty unless all are stuck."""
        impl = waiter.ctx.reflect.get_impl(name, False)
        # a disposed provider's service lingers until its dependents unloaded
        holder = None if impl is None else impl.ctx.fiber
        if holder is not None and holder.uid is not None and _progressing(holder):
            return []
        providers = self.providers(waiter, name)
        if any(_progressing(provider) for provider in providers):
            return []
        return [provider for provider in providers if provider.state == FiberState.PENDING]

    def cycle_through(self, start: FiberLifecycle, group: set[FiberLifecycle]) -> list[_Step]:
        """A shortest cycle through ``start``, visiting only fibers of its cyclic ``group``.

        ``group`` is the strongly connected group of ``start`` and holds a
        cycle (:meth:`is_cyclic`), so the search always comes back to
        ``start``. Breadth-first with a visited set: linear in the graph,
        never recursive.
        """
        came_from: dict[FiberLifecycle, _Step] = {}
        queue = deque([start])
        while True:
            node = queue.popleft()
            for target, service in self.edges(node):
                if target is start:
                    steps = [(node, service)]
                    while node is not start:
                        step = came_from[node]
                        steps.append(step)
                        node = step[0]
                    steps.reverse()
                    return steps
                if target in came_from or target not in group:
                    continue
                came_from[target] = (node, service)
                queue.append(target)

    def cycles(self) -> dict[_Step, _Cycle]:
        """The cycle each stuck ``(fiber, service)`` hop belongs to.

        Fibers waiting for each other form a strongly connected group; each
        group is one numbered cycle (by its lowest uid), spelled as a shortest
        cycle through its lowest-uid fiber. Linear in the graph.
        """
        if self._cycles is None:
            self._cycles = {}
            pending = [fiber for fiber in self.fibers if fiber.state == FiberState.PENDING]
            groups = [set(group) for group in _strongly_connected(pending, self.targets)]
            cyclic = [group for group in groups if self.is_cyclic(group)]
            for number, group in enumerate(sorted(cyclic, key=_lowest_uid), start=1):
                start = min(group, key=_uid)
                cycle = _Cycle(number, start, _path(self.cycle_through(start, group)))
                for member in sorted(group, key=_uid):
                    for target, service in self.edges(member):
                        if target in group:
                            self._cycles.setdefault((member, service), cycle)
        return self._cycles

    def targets(self, fiber: FiberLifecycle) -> list[FiberLifecycle]:
        """The fibers ``fiber`` waits for (see :meth:`edges`)."""
        return [target for target, _service in self.edges(fiber)]

    def is_cyclic(self, group: set[FiberLifecycle]) -> bool:
        """Whether a strongly connected ``group`` holds a cycle (more than one fiber, or a loop)."""
        if len(group) > 1:
            return True
        (only,) = group
        return any(target is only for target in self.targets(only))

    # -- reasons -----------------------------------------------------------------------

    def reasons(self, fiber: FiberLifecycle) -> list[Reason]:
        """Why ``fiber`` is not ACTIVE (empty when it is, or once disposed)."""
        state = fiber.state
        if state == FiberState.PENDING:
            return [self._waiting(fiber, name) for name in _unsatisfied(fiber)]
        if state == FiberState.LOADING:
            return [_loading(fiber)]
        if state == FiberState.UNLOADING:
            return [_unloading(fiber)]
        if state == FiberState.FAILED:
            return _failed(fiber)
        return []

    def _waiting(self, fiber: FiberLifecycle, name: str) -> Reason:
        """Why the PENDING ``fiber`` does not have service ``name``."""
        cycle = self.cycles().get((fiber, name))
        if cycle is not None:
            return _in_cycle(fiber, name, cycle)
        impl = fiber.ctx.reflect.get_impl(name, False)
        if impl is not None:
            return _provided(fiber, name, impl)
        providers = self.providers(fiber, name)
        failed = [p for p in providers if p.state == FiberState.FAILED]
        if failed:
            return _provider_failed(name, failed[0])
        elsewhere = [impl for impl in fiber.ctx.reflect.store.values() if impl.name == name]
        if elsewhere:
            provider = _ref(elsewhere[0].ctx.fiber)
            return Reason(
                "isolated", name, f'"{name}" is provided by {provider} in another isolated scope'
            )
        if providers:
            provider = providers[0]
            detail = f'"{name}" is declared by {_ref(provider)} ({provider.state.name})'
            return Reason("missing-service", name, f"{detail}, not provided yet")
        declared = [other for other in self.fibers if _declares(other, name)]
        if declared:
            detail = f'"{name}" is only declared by {_ref(declared[0])} in another isolated scope'
            return Reason("isolated", name, detail)
        return Reason("missing-service", name, f'no plugin declares or currently provides "{name}"')


def _lowest_uid(group: set[FiberLifecycle]) -> int:
    return min(_uid(fiber) for fiber in group)


def _in_cycle(fiber: FiberLifecycle, name: str, cycle: _Cycle) -> Reason:
    """The ``cycle`` reason: the path in full once (on its first hop), else a reference."""
    if fiber is cycle.start and cycle.path[1] == name:
        detail = f"cycle #{cycle.number}: {format_path(cycle.path)}"
    else:
        detail = f'waits for "{name}" in cycle #{cycle.number} (see {_ref(cycle.start)})'
    return Reason("cycle", name, detail, cycle.path)


def _provided(waiter: FiberLifecycle, name: str, impl: Impl) -> Reason:
    """Why a service that is registered in the waiter's scope is still unavailable to it."""
    provider = impl.ctx.fiber
    if provider.state != FiberState.ACTIVE:
        detail = f'"{name}" is provided by {_ref(provider)}, which is {provider.state.name}'
        return Reason("missing-service", name, detail)
    if impl.check is not None:
        raised = waiter.check_error(name)
        verdict = "rejected it" if raised is None else f"raised {_describe(raised)}"
        detail = f'"{name}" is provided by {_ref(provider)} but its check() {verdict}'
        return Reason("check-failed", name, detail)
    # the waiter is notified right after the provider's status change (seen
    # by an ``internal/status`` observer diagnosing in between)
    detail = f'"{name}" is provided by {_ref(provider)} but not picked up yet'
    return Reason("missing-service", name, detail)


def _provider_failed(name: str, provider: FiberLifecycle) -> Reason:
    detail = f'"{name}" provider {_ref(provider)} failed: {_describe(provider.error)}'
    return Reason("provider-failed", name, detail)


def _describe(error: BaseException | None) -> str:
    assert error is not None, "a FAILED fiber carries the error that failed it"
    return f"{type(error).__name__}: {error}"


def _loading(fiber: FiberLifecycle) -> Reason:
    timeout = fiber.load_timeout
    bound = "" if timeout is None else f" (load_timeout {timeout:g}s)"
    return Reason("loading", None, f"running the plugin body{bound}")


def _unloading(fiber: FiberLifecycle) -> Reason:
    dependents = {dependent for impl in fiber.provisions for dependent in impl.dependents}
    if not dependents:
        return Reason("unloading", None, "running disposers")
    names = ", ".join(_ref(dependent) for dependent in sorted(dependents, key=_uid))
    return Reason("unloading", None, f"waiting for dependents to unload: {names}")


def _failed(fiber: FiberLifecycle) -> list[Reason]:
    reasons = [Reason("failed", None, _describe(fiber.error))]
    pending = None if fiber.supervisor is None else fiber.supervisor.pending
    if pending is not None:
        attempt = f"{pending.attempt}/{pending.max_restarts}"
        detail = f"restart {attempt} after a {pending.delay:.2f}s backoff"
        reasons.append(Reason("backoff", None, detail))
    return reasons


# -- cycle warnings --------------------------------------------------------------------


@final
class CycleWatch:
    """Warns when fibers entering ``PENDING`` close a dependency cycle.

    Every fiber of a cycle is PENDING while it exists, so a cycle forms when
    its last fiber enters ``PENDING``. Fibers entering PENDING are collected
    and checked together once per event-loop turn (a loader start creating
    many entries is one check), in time linear in the fibers and their
    dependencies: each group of fibers waiting for each other that one of
    them joined is warned about once, spelled as a shortest cycle through it.
    An existing cycle is not warned about again until one of its fibers left
    ``PENDING``. Cycles closed by a new provision of an already PENDING fiber
    are not warned about (``diagnose()`` still finds them). An error in the
    check is reported, never raised. Example (what a fiber entering PENDING
    does)::

        registry.cycles.entered_pending(fiber)

    Internal — one per application, as ``registry.cycles``.
    """

    __slots__ = ("_entered",)

    def __init__(self) -> None:
        self._entered: list[FiberLifecycle] = []

    def entered_pending(self, fiber: FiberLifecycle) -> None:
        """Note that ``fiber`` became PENDING; check for new cycles on the next loop turn."""
        # cheap exit: a cycle member waits for something and could provide something
        if not fiber.inject or not (fiber.provisions or _declared_any(fiber)):
            return
        if not self._entered:
            asyncio.get_running_loop().call_soon(self._check)
        self._entered.append(fiber)

    def _check(self) -> None:
        entered = [fiber for fiber in self._entered if fiber.state == FiberState.PENDING]
        self._entered = []
        if not entered:
            return
        try:
            _warn_new_cycles(entered)
        except Exception as error:
            report_error(entered[0], error)


def _warn_new_cycles(entered: list[FiberLifecycle]) -> None:
    """Warn once about each group of fibers waiting for each other that ``entered`` joined."""
    graph = _WaitGraph(entered[0])
    new = set(entered)
    # a group that an entered fiber joined is reachable from it: searching from
    # the entered fibers only keeps a check proportional to what they wait for,
    # not to every PENDING fiber (a long chain tearing down enters one per turn)
    roots = sorted(new, key=_uid)
    groups = [set(group) for group in _strongly_connected(roots, graph.targets)]
    for group in sorted(groups, key=_lowest_uid):
        joined = sorted(group & new, key=_uid)
        if not joined or not graph.is_cyclic(group):
            continue
        steps = graph.cycle_through(joined[0], group)
        _warn_cycle(joined[0], _path(_canonical(steps)))


def _declared_any(fiber: FiberLifecycle) -> bool:
    return fiber.runtime is not None and bool(fiber.runtime.provide)


def _warn_cycle(fiber: FiberLifecycle, path: tuple[str, ...]) -> None:
    text = f"dependency cycle: {format_path(path)} (these plugins wait for each other)"
    fiber.ctx.logger("diagnose").warn(text)
