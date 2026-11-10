"""Plugin registry, ``Inject`` and plugin shapes (port of cordis ``registry.ts``).

Supported plugin shapes:

- a function ``plugin(ctx, config)``; generator and async-generator functions
  are functions whose returned iterator becomes the plugin's effect;
- a class, constructed as ``cls(ctx, config)``; then its instance's
  ``__cordis_init_hooks__`` run and the result of ``__cordis_init__()`` (if
  defined) becomes the effect;
- an object with an ``apply(ctx, config)`` method (e.g. a module).

Plugin metadata is read as attributes: ``name``, ``Config``, ``inject``,
``provide``, ``intercept`` and the supervision settings ``policy``,
``load_timeout`` and ``dispose_timeout`` (:mod:`pycordis.supervise`);
:func:`plugin_meta` sets them with a typed decorator.
"""

from __future__ import annotations

import inspect
from collections.abc import (
    Callable,
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from dataclasses import dataclass
from types import FunctionType, MethodType, ModuleType, SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    cast,
    final,
    overload,
    runtime_checkable,
)

from pycordis._context_internals import fiber_of
from pycordis._guards import is_array, is_dict, is_iterable, is_mapping
from pycordis._utils import DisposableList, report_failure, symbols
from pycordis.diagnose import CycleWatch
from pycordis.fiber import Fiber, report_error
from pycordis.keys import ServiceKey, ServiceName, service_name
from pycordis.supervise import UNSUPERVISED, RestartPolicy, Supervision, check_timeout
from pycordis.trace import Tracker, get_traceable, untraced

if TYPE_CHECKING:
    from pycordis.context import Context

type InjectSpec = (
    list[str]
    | list[ServiceKey[Any]]
    | list[ServiceName]
    | tuple[ServiceName, ...]
    | Mapping[str, object]
    | Mapping[ServiceKey[Any], object]
    | Mapping[ServiceName, object]
)
"""Service dependency declaration: names or keys, or a name/key -> intercept config map.

Example: ``inject=["db", LLM]`` or ``inject={"db": None, LLM: {"model": "small"}}``.
(``ServiceKey[Any]`` is a key of any service type, as in :data:`~pycordis.keys.ServiceName`.)
"""

type _InjectMap = dict[str, object]
"""A normalized ``inject`` declaration: service name -> intercept config (``None``: none).

Example: ``{"db": None, "llm": {"model": "small"}}``.
"""

type _PluginCallback = Callable[..., object]
"""The executable entrypoint of a plugin: the plugin itself, or its ``apply`` method.

It takes ``(ctx, config)`` with a config type only the plugin knows, hence
``...``. Example: ``ctx.registry.resolve(plugin)``.
"""


class PluginObject[C](Protocol):
    """An object plugin: anything with an ``apply(ctx, config)`` method.

    Example::

        class Greeter:
            def apply(self, ctx: Context, config: GreeterConfig) -> None: ...
    """

    def apply(self, ctx: Context, config: C, /) -> object:
        """Start the plugin in ``ctx`` with its validated ``config``.

        Example: ``def apply(self, ctx: Context, config: GreeterConfig) -> None: ...``.
        """
        ...


type Plugin[C] = Callable[[Context, C], object] | PluginObject[C]
"""A plugin whose config type ``C`` a type checker can see.

Covers a function ``plugin(ctx, config: C)``, a class whose ``__init__(self, ctx,
config: C)`` (classes are callables), and an object with ``apply(ctx, config: C)``.
Example: ``def start(plugin: Plugin[ServerConfig]) -> None: ctx.plugin(plugin, cfg)``.
"""


@final
@dataclass(eq=False, slots=True, weakref_slot=True)
class Runtime:
    """Registry record shared by every fiber of one plugin callback."""

    callback: _PluginCallback
    """The executable entrypoint all fibers share (registry identity key)."""
    fibers: DisposableList[Fiber]
    """Every live fiber of this plugin (one per ``ctx.plugin()`` call)."""
    name: str | None = None
    """Display name taken from the first registered plugin shape."""
    Config: object | None = None
    """Config schema applied to each fiber's config."""
    provide: tuple[str, ...] = ()
    """Service names the plugin declares it provides (:func:`plugin_provide`)."""


@final
class Inject:
    """Utilities for normalizing plugin dependency declarations.

    Example: ``Inject.resolve(["db", LLM]) == {"db": None, "llm": None}``.
    """

    @staticmethod
    def resolve(inject: InjectSpec | None, result: _InjectMap | None = None) -> _InjectMap:
        """Convert a list / mapping ``inject`` declaration into a plain dict.

        Keys are replaced by their service names; names listed without config map
        to ``None``. Entries are written into ``result`` (a new dict by default),
        which is returned. Class inject maps built by :func:`inject` already
        contain their bases' entries. Example::

            Inject.resolve(["db", LLM]) == {"db": None, "llm": None}
        """
        return _resolve_declared(inject, {} if result is None else result)


def _resolve_declared(inject: object, result: _InjectMap) -> _InjectMap:
    """:meth:`Inject.resolve` of a declaration read from plugin metadata (not yet checked)."""
    if not inject:
        return result
    if isinstance(inject, str):
        raise TypeError("inject must be a list of service names or a mapping")
    if is_mapping(inject):
        for name, config in inject.items():
            result[_declared_service(name)] = config
    elif is_iterable(inject):
        for name in inject:
            result[_declared_service(name)] = None
    else:
        raise TypeError(f"inject must be a list or mapping, got {type(inject).__name__}")
    return result


def _declared_service(name: object) -> str:
    """The service name an ``inject`` item stands for (a name or a :class:`ServiceKey`)."""
    if isinstance(name, str):
        return name
    if isinstance(name, ServiceKey):
        return name.name
    raise TypeError(f"inject must name services by str or ServiceKey, got {type(name).__name__}")


def _base_inject(cls: type) -> _InjectMap:
    """The inject map ``cls`` starts from: its own declaration, else the nearest inherited one.

    Maps built by :func:`inject` already contain their bases' entries, so the
    lookup through the MRO yields every inherited dependency, base first.
    """
    declared: object = getattr(cls, "inject", None)  # metadata: a class may not declare it
    return _resolve_declared(declared, {})


@runtime_checkable
class _HasContext(Protocol):
    """A class plugin's instance: it keeps the context it was constructed with as ``ctx``."""

    ctx: Context


@final
class _InitHooks:
    """Class attribute seeding each instance's ``__cordis_init_hooks__`` list.

    A non-data descriptor: the first read on an instance builds its list (one
    hook per :func:`inject`-decorated method) and stores it on the instance, so
    later reads and appends see that list (cordis ``addInitializer``).
    """

    @overload
    def __get__(self, obj: None, owner: type) -> _InitHooks: ...
    @overload
    def __get__(self, obj: object, owner: type | None = None) -> list[Callable[[], None]]: ...
    def __get__(
        self, obj: object, owner: type | None = None
    ) -> _InitHooks | list[Callable[[], None]]:
        if obj is None:
            return self
        cls = type(obj)
        names: dict[str, None] = {}
        for klass in reversed(cls.__mro__):
            for name, value in klass.__dict__.items():
                if isinstance(value, _InjectMethod):
                    names[name] = None
        hooks: list[Callable[[], None]] = []
        for name in names:
            method = inspect.getattr_static(cls, name)
            if isinstance(method, _InjectMethod):
                hooks.append(method.hook(obj))
        obj.__dict__[symbols.init_hooks] = hooks
        return hooks


@final
class _InjectMethod:
    """A method decorated with :func:`inject`; it also runs as an inner plugin.

    Reading it from an instance gives the ordinary bound method.
    """

    def __init__(self, func: FunctionType) -> None:
        self.func = func
        self.inject: _InjectMap = {}
        metadata: dict[str, object] = func.__dict__.setdefault(symbols.metadata, {})
        metadata["inject"] = self.inject
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__
        self.__wrapped__ = func

    @property
    def __cordis_metadata__(self) -> dict[str, object]:
        """Decorator metadata (``{"inject": {...}}``), shared with the function."""
        return self.func.__dict__[symbols.metadata]

    def __set_name__(self, owner: type, name: str) -> None:
        if not isinstance(inspect.getattr_static(owner, symbols.init_hooks, None), _InitHooks):
            setattr(owner, symbols.init_hooks, _InitHooks())

    @overload
    def __get__(self, obj: None, owner: type) -> _InjectMethod: ...
    @overload
    def __get__(self, obj: object, owner: type | None = None) -> MethodType: ...
    def __get__(self, obj: object, owner: type | None = None) -> _InjectMethod | MethodType:
        if obj is None:
            return self
        return MethodType(self.func, obj)

    def hook(self, obj: object) -> Callable[[], None]:
        """The init hook running this method under ``obj.ctx.inject(...)``."""
        func = self.func
        inject = self.inject

        def run() -> None:
            tracker = getattr(obj, symbols.tracker, None)
            prop = tracker.property if isinstance(tracker, Tracker) else None

            # named `apply` so the inner fiber stays anonymous and inherits its
            # parent's name (cordis passes an anonymous arrow function)
            def apply(ctx: Context, _config: None) -> object:
                # cordis `withProps(this, { [property]: ctx })`: a traced self
                # whose tracker property is the inner context
                return func(get_traceable(ctx, obj) if prop else obj)

            if not isinstance(obj, _HasContext):
                raise TypeError(
                    f"@inject method {func.__name__!r} needs self.ctx (the plugin's context)"
                )
            obj.ctx.inject(dict(inject), apply)

        return run


def inject[T](name: ServiceName, config: object = None) -> Callable[[T], T]:
    """Declare a service dependency on a class plugin or a method (cordis ``@Inject``).

    On a class, ``name`` is added to the class's ``inject`` map, which starts as a
    copy of the inherited one so the base is never mutated. On a method, the
    method is run as an inner plugin, ``self.ctx.inject({name: config}, ...)``,
    once the instance has been constructed as a plugin and the service is
    available; inside it ``self.ctx`` is the inner context when the class has a
    :class:`~pycordis.trace.Tracker` property. Stacked decorators on one method
    share a single dependency map. ``name`` may be a :class:`~pycordis.keys.ServiceKey`.
    Example::

        @inject("db")
        class Repo:
            def __init__(self, ctx: Context, config: None) -> None: ...

            @inject(LLM)
            def summarize(self) -> None: ...  # runs once "llm" is available
    """
    name = service_name(name)

    def decorator(value: T) -> T:
        if isinstance(value, type):
            _class_inject(value)[name] = config
            return value
        if isinstance(value, _InjectMethod):
            value.inject[name] = config
            return value
        if isinstance(value, FunctionType):
            method = _InjectMethod(value)
            method.inject[name] = config
            # a descriptor standing in for the function: read from an instance
            # it gives the bound method, so the name keeps its static type
            return cast(T, method)
        raise TypeError("@Inject() can only be used on class or class methods")

    return decorator


def _class_inject(cls: type) -> dict[object, object]:
    """The ``inject`` map ``cls`` itself declares, created from the inherited one if absent.

    A created map starts as a copy, so a base class's map is never mutated.
    """
    own: object = vars(cls).get("inject")
    if is_dict(own):
        return own
    created: dict[object, object] = {}
    created.update(_base_inject(cls))
    cls.inject = created  # plugin metadata, set on an arbitrary user class
    return created


def plugin_meta[P](
    *,
    name: str | None = None,
    inject: InjectSpec | None = None,
    provide: ServiceName | Sequence[ServiceName] | None = None,
    Config: object | None = None,
    policy: RestartPolicy | None = None,
    load_timeout: float | None = None,
    dispose_timeout: float | None = None,
) -> Callable[[P], P]:
    """Declare a plugin's metadata with a typed decorator.

    Sets the attributes the registry reads (``name``, ``inject``, ``provide``,
    ``Config``, ``policy``, ``load_timeout``, ``dispose_timeout``) on a
    function, class or object plugin; arguments left as ``None`` are not set.
    Keys are stored as their service names, and ``inject`` as its normalized
    map (:meth:`Inject.resolve`). Assigning the attributes directly
    (``plugin.inject = [...]``) keeps working at runtime. Raises ``ValueError``
    for a timeout that is not a positive number. Example::

        @plugin_meta(name="chat", inject=[LLM, "db"], Config=ChatConfig)
        def chat(ctx: Context, config: ChatConfig) -> None: ...

        @plugin_meta(policy=RestartPolicy(mode="on-failure"), load_timeout=30)
        async def worker(ctx: Context, config: None) -> None: ...
    """
    check_timeout(load_timeout, "load_timeout")
    check_timeout(dispose_timeout, "dispose_timeout")
    meta: dict[str, object] = {}
    if name is not None:
        meta["name"] = name
    if inject is not None:
        meta["inject"] = Inject.resolve(inject)
    if provide is not None:
        meta["provide"] = _provide_names(provide)
    if Config is not None:
        meta["Config"] = Config
    if policy is not None:
        meta["policy"] = policy
    if load_timeout is not None:
        meta["load_timeout"] = load_timeout
    if dispose_timeout is not None:
        meta["dispose_timeout"] = dispose_timeout

    def decorator(plugin: P) -> P:
        for attribute, value in meta.items():
            setattr(plugin, attribute, value)
        return plugin

    return decorator


def _provide_names(provide: ServiceName | Sequence[ServiceName]) -> str | list[str]:
    if isinstance(provide, str | ServiceKey):
        return service_name(provide)
    return [service_name(name) for name in provide]


def _apply_method(plugin: object) -> _PluginCallback | None:
    """The ``apply`` method of a non-callable object plugin (``None``: it has none).

    Reading ``apply`` runs user code (a property, ``__getattr__``) that may raise.
    """
    apply: object = getattr(plugin, "apply", None)  # metadata probe: any object may lack it
    return apply if callable(apply) else None


def plugin_inject(plugin: object) -> _InjectMap:
    """The normalized ``inject`` declaration of ``plugin`` (see :meth:`Inject.resolve`).

    A module's attributes are mostly names it imported, so a module ``inject``
    that is not a list or mapping (e.g. ``from pycordis import inject``) is
    ignored. Any other plugin with such an ``inject`` raises ``TypeError``.
    Example: ``plugin_inject(plugin_meta(inject=["db"])(start)) == {"db": None}``.
    """
    spec: object = getattr(plugin, "inject", None)
    kind = type(spec).__name__
    iterable = not callable(spec) and is_iterable(spec)
    if spec is None or isinstance(spec, str) or is_mapping(spec) or iterable:
        return _resolve_declared(spec, {})
    if isinstance(plugin, ModuleType):
        return {}
    raise TypeError(
        f'plugin {_plugin_name(plugin)!r} has invalid "inject": '
        f"expected list or mapping, got {kind}"
    )


def plugin_provide(plugin: object) -> tuple[str, ...]:
    """The service names ``plugin`` declares in its ``provide`` metadata.

    ``provide`` may be a name, a :class:`~pycordis.keys.ServiceKey`, or a list
    or tuple of them (as :func:`plugin_meta` and ``Service`` subclasses set
    it); anything else, and any other item, is ignored: a module's ``provide``
    is often an unrelated imported name. Example::

        @plugin_meta(provide=[LLM, "cache"])
        def agent(ctx: Context, config: None) -> None: ...

        plugin_provide(agent) == ("llm", "cache")
    """
    declared: object = getattr(plugin, "provide", None)
    items: Sequence[object] = declared if is_array(declared) else [declared]
    names = (_declared_name(item) for item in items)
    return tuple(name for name in names if name is not None)


def _declared_name(item: object) -> str | None:
    """The service name ``item`` declares in ``provide`` (``None`` when it is no name)."""
    if isinstance(item, str):
        return item
    if isinstance(item, ServiceKey):
        return item.name
    return None


def plugin_config_type(plugin: object) -> object | None:
    """The ``Config`` schema of ``plugin`` (a module imported as ``Config`` is ignored).

    Example: ``plugin_config_type(plugin_meta(Config=ChatConfig)(chat)) is ChatConfig``.
    """
    config_type: object = getattr(plugin, "Config", None)
    if isinstance(plugin, ModuleType) and isinstance(config_type, ModuleType):
        return None
    return config_type


def plugin_supervision(
    plugin: object,
    policy: RestartPolicy | None = None,
    load_timeout: float | None = None,
    dispose_timeout: float | None = None,
) -> Supervision:
    """The supervision settings of ``plugin``: explicit arguments over its metadata.

    Each argument left as ``None`` falls back to the plugin's ``policy`` /
    ``load_timeout`` / ``dispose_timeout`` attribute. Raises ``TypeError`` for a
    ``policy`` attribute that is not a :class:`~pycordis.supervise.RestartPolicy`
    and ``ValueError`` for a timeout that is not a positive number. A module's
    attributes of the wrong type are ignored (they are usually unrelated
    names). Example::

        plugin_supervision(worker, load_timeout=5.0).load_timeout == 5.0
    """
    module = isinstance(plugin, ModuleType)
    if policy is None:
        declared: object = getattr(plugin, "policy", None)
        if isinstance(declared, RestartPolicy):
            policy = declared
        elif declared is not None and not module:
            raise TypeError(
                f'plugin {_plugin_name(plugin)!r} has invalid "policy": '
                f"expected RestartPolicy, got {type(declared).__name__}"
            )
    if load_timeout is None:
        load_timeout = _declared_timeout(plugin, "load_timeout", module)
    if dispose_timeout is None:
        dispose_timeout = _declared_timeout(plugin, "dispose_timeout", module)
    if policy is None and load_timeout is None and dispose_timeout is None:
        return UNSUPERVISED
    return Supervision(
        policy if policy is not None else RestartPolicy(),
        check_timeout(load_timeout, "load_timeout"),
        check_timeout(dispose_timeout, "dispose_timeout"),
    )


def _declared_timeout(plugin: object, attribute: str, module: bool) -> float | None:
    """A timeout attribute of ``plugin`` (``None`` when absent, or ill-typed on a module)."""
    value: object = getattr(plugin, attribute, None)
    if module and (isinstance(value, bool) or not isinstance(value, int | float)):
        return None
    return check_timeout(value, attribute)


def _plugin_name(plugin: object) -> str | None:
    """The display name of ``plugin``: its ``name`` metadata, else a function's or class's name."""
    name: object = getattr(plugin, "name", None)  # metadata probe: any plugin may lack it
    if not isinstance(name, str) and (inspect.isfunction(plugin) or inspect.isclass(plugin)):
        name = plugin.__name__
    if not isinstance(name, str) or name in ("apply", "<lambda>", ""):
        return None
    return name


def _callback_of(plugin: object) -> _PluginCallback | None:
    """The callback identifying ``plugin`` (see :meth:`RegistryService.resolve`); may raise."""
    if callable(plugin):
        return plugin
    return _apply_method(plugin)


def _checked_callback(plugin: object) -> _PluginCallback:
    """The callback identifying ``plugin``; ``TypeError`` (chaining a getter's error) if none."""
    message = (
        'invalid plugin, expect function or object with an "apply" method, received '
        + type(plugin).__name__
    )
    try:
        callback = _callback_of(plugin)
    except Exception as error:
        raise TypeError(message) from error
    if callback is None:
        raise TypeError(message)
    return callback


@final
class RegistryService:
    """Plugin registry installed as ``ctx.registry``; mixes ``plugin`` / ``inject`` into contexts.

    It normalizes plugin shapes, tracks one :class:`Runtime` per plugin callback,
    starts fibers, and is a read-only mapping view of callback -> runtime.
    """

    __cordis_tracker__: ClassVar[Tracker] = Tracker(property="ctx", no_shadow=True)

    ctx: Context
    """The context plugins are loaded from."""
    cycles: CycleWatch
    """Warns when the application's fibers form a dependency cycle. Internal."""
    _counter: int
    _internal: dict[_PluginCallback, Runtime]

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.cycles = CycleWatch()
        self._counter = 0
        self._internal = {}

    @property
    def counter(self) -> int:
        """Allocate the next fiber uid (increments on every read)."""
        self._counter += 1
        return self._counter

    def __len__(self) -> int:
        return len(self._internal)

    def __iter__(self) -> Iterator[_PluginCallback]:
        return iter(list(self._internal))

    def resolve(self, plugin: object) -> _PluginCallback | None:
        """Return the callback identifying ``plugin``, or ``None`` if it is not a plugin.

        An object whose ``apply`` cannot be read (its getter raises) is not a
        plugin here; :meth:`plugin` reports such an error as the cause of its
        ``TypeError``. Example: ``ctx.registry.resolve(module) is module.apply``.
        """
        try:
            return _callback_of(plugin)
        except Exception:  # documented: an unreadable ``apply`` means "not a plugin"
            return None

    def get(self, plugin: object) -> Runtime | None:
        """Return the runtime of ``plugin``, or ``None`` when it is not registered."""
        key = self.resolve(plugin)
        return None if key is None else self._internal.get(key)

    def has(self, plugin: object) -> bool:
        """Whether ``plugin`` has a registered runtime."""
        key = self.resolve(plugin)
        return key is not None and key in self._internal

    def delete(self, plugin: object) -> Runtime | None:
        """Dispose every fiber of ``plugin`` and remove its runtime; return it (or ``None``)."""
        key = self.resolve(plugin)
        runtime = None if key is None else self._internal.pop(key, None)
        if runtime is None:
            return None
        for fiber in runtime.fibers:
            report_failure(
                fiber.dispose(), lambda error, fiber=fiber: report_error(fiber.parent, error)
            )
        return runtime

    def keys(self) -> KeysView[_PluginCallback]:
        """The registered plugin callbacks."""
        return self._internal.keys()

    def values(self) -> ValuesView[Runtime]:
        """The registered plugin runtimes."""
        return self._internal.values()

    def items(self) -> ItemsView[_PluginCallback, Runtime]:
        """``(callback, runtime)`` pairs (cordis ``entries``)."""
        return self._internal.items()

    def for_each(self, callback: Callable[[Runtime, _PluginCallback], object]) -> None:
        """Call ``callback(runtime, key)`` for every registered runtime."""
        for key, runtime in list(self._internal.items()):
            callback(runtime, key)

    def inject(self, deps: InjectSpec, callback: Callable[[Context, None], object]) -> Fiber:
        """Run ``callback(ctx, config)`` as a plugin whenever the ``deps`` services are available.

        Shorthand for ``ctx.plugin(<object with inject and apply>)``: the callback
        is unloaded and re-run whenever a required service changes; its config is
        always ``None``. Returns the fiber; awaiting it settles once loading
        finished. Example::

            ctx.inject(["db"], lambda ctx, _: ctx.db.connect())
        """
        name = getattr(callback, "__name__", None)
        return self.plugin(SimpleNamespace(inject=deps, apply=callback, name=name))

    # Typed plugins (functions, classes, ``apply`` objects) check ``config`` against
    # their config parameter; a raw mapping is always accepted because a ``Config``
    # schema converts it at runtime, and so is ``None`` (or no config): a schema
    # whose fields all have defaults builds the config from nothing, and the checker
    # cannot see whether a plugin has one (the ``object`` overload would accept an
    # omitted config anyway). Namespaces and modules take any config. A plugin
    # of unknown shape (``object``) has nothing to check against, so it only takes a
    # config that is itself unchecked (``Any``) or ``None``: an ``object`` overload
    # accepting any config would also swallow every typed-plugin mismatch.
    #
    # Pyright reports a mismatch against the ``object`` overload ("... is not
    # assignable to None"). Moving ``Plugin[C]`` after it would name the plugin's
    # config type instead, but the ``object`` overload would then match first for
    # ``ctx.plugin(lambda ctx, config: ...)`` and leave the lambda's parameters
    # unknown, so the order stays.
    @overload
    def plugin[C](
        self,
        plugin: Plugin[C],
        config: C | Mapping[str, object] | None = None,
        *,
        policy: RestartPolicy | None = None,
        load_timeout: float | None = None,
        dispose_timeout: float | None = None,
    ) -> Fiber: ...
    @overload
    def plugin(
        self,
        plugin: SimpleNamespace | ModuleType,
        config: Any = None,  # namespaces and modules declare no config type to check
        *,
        policy: RestartPolicy | None = None,
        load_timeout: float | None = None,
        dispose_timeout: float | None = None,
    ) -> Fiber: ...
    @overload
    def plugin(
        self,
        plugin: object,
        config: None = None,
        *,
        policy: RestartPolicy | None = None,
        load_timeout: float | None = None,
        dispose_timeout: float | None = None,
    ) -> Fiber: ...
    def plugin(
        self,
        plugin: object,
        config: object = None,
        *,
        policy: RestartPolicy | None = None,
        load_timeout: float | None = None,
        dispose_timeout: float | None = None,
    ) -> Fiber:
        """Start ``plugin`` in the current context and return its fiber.

        Creates (or reuses) the plugin's runtime, then starts a new fiber under
        the current context. The fiber is awaitable: ``await ctx.plugin(p)``
        settles once loading finished, re-raising a config or startup error.
        Raises ``TypeError`` for an unsupported shape and
        ``CordisError("INACTIVE_EFFECT")`` if the current fiber is disposed.

        Type checkers check ``config`` against the plugin's config parameter
        (:data:`Plugin`); a raw mapping is always accepted, since a ``Config``
        schema validates it at runtime, and so is an omitted (``None``) config,
        since a schema whose fields all have defaults builds one from nothing
        (without a schema the body receives ``None``). A plugin typed as plain
        ``object`` only takes an ``Any`` or ``None`` config. Example::

            def server(ctx: Context, config: ServerConfig) -> None: ...

            ctx.plugin(server, ServerConfig(port=80))  # ok
            ctx.plugin(server, OtherConfig())  # type error

        Pyright words the error after the last overload: ``No overloads for
        "plugin" match`` with ``"OtherConfig" is not assignable to "None"``;
        it means the config does not match the plugin's config type.

        ``policy``, ``load_timeout`` and ``dispose_timeout`` override the plugin's
        metadata of the same names for this fiber (see :mod:`pycordis.supervise`;
        default: no restarts, no timeouts). Raises ``ValueError`` for a timeout
        that is not a positive number. Example::

            ctx.plugin(worker, policy=RestartPolicy(mode="on-failure"), load_timeout=30)
        """
        registry = untraced(self)  # only the registering context needs the proxy
        ctx = self.ctx
        callback = _checked_callback(plugin)
        fiber_of(ctx).assert_active()

        deps = plugin_inject(plugin)
        supervision = plugin_supervision(plugin, policy, load_timeout, dispose_timeout)
        runtime = registry._internal.get(callback)
        created = runtime is None
        if runtime is None:
            runtime = Runtime(
                callback,
                DisposableList(),
                _plugin_name(plugin),
                plugin_config_type(plugin),
                plugin_provide(plugin),
            )
            registry._internal[callback] = runtime

        try:
            return Fiber(ctx, config, deps, runtime, supervision=supervision)
        except BaseException:
            # e.g. INACTIVE_EFFECT from an unloading parent: never leave behind
            # a runtime that no fiber was ever added to
            if created and not len(runtime.fibers) and registry._internal.get(callback) is runtime:
                del registry._internal[callback]
            raise
