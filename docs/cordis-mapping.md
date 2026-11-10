# cordis → pycordis name mapping

pycordis ports cordis 4.1.0. Single-word names keep their spelling; camelCase
becomes snake_case. **Event names (`internal/plugin`, ...) and error message
strings are unchanged.** Python-specific behaviour is called out in the last
column where it differs from cordis. Rows marked **pycordis-only** have no
cordis counterpart; most are collected at the end in
[pycordis-only additions](#pycordis-only-additions).

## Naming rules

| cordis | pycordis |
| --- | --- |
| `getEffects` | `get_effects` |
| `rawConfig` | `raw_config` |
| `effectiveConfig` | `effective_config` |
| `resolveConfig` | `resolve_config` |
| `assertActive` | `assert_active` |
| `isBailed` | `is_bailed` |
| `entries` (registry) | `items` |
| `size` (registry) | `len(ctx.registry)` |
| `forEach` | `for_each` |
| `await` (fiber) | `wait` (keyword clash): `await fiber.wait()` |
| `Context.is` | `Context.is_` |

## Context (`pycordis.Context`)

| cordis | pycordis | Notes |
| --- | --- | --- |
| `new Context()` | `Context()` | Needs a running event loop. `Context(strict=True)` enables strict diagnostics (or `PYCORDIS_STRICT=1`). |
| `ctx.extend(meta)` | `ctx.extend(**meta)` | Child falls back to the parent chain; the parent is never mutated. |
| `ctx.isolate(name, label?)` | `ctx.isolate(name, label=None)` | `Context.isolate` on the class is the attribute-name constant `"_isolate"`; on an instance it is the method. |
| `ctx.intercept(name, config)` | `ctx.intercept(name, config)` | Same descriptor duality, `Context.intercept` is `"_intercept"` on the class. |
| `ctx.root` | `ctx.root` | |
| `ctx.fiber` | `ctx.fiber` | |
| `ctx.get(name, strict?)` | `ctx.get(name, strict=True)` | `name` may also be a `ServiceKey[T]`, typing the result `T \| None` (as for `set` and `provide` below). |
| `ctx.set(name, value)` | `ctx.set(name, value)` | Returns `bool`; a getter-only accessor raises `ServiceAccessError` (cordis returns `false`). |
| `ctx.provide(name, value?, check?)` | `ctx.provide(name, value=None, check=None)` | With a `ServiceKey[T]`, pyright checks `value` is a `T`. |
| `ctx.accessor(name, {get, set})` | `ctx.accessor(name, get=..., set=...)` | |
| `ctx.mixin(source, mixins)` | `ctx.mixin(source, mixins)` | |
| `ctx.inject(deps, cb)` | `ctx.inject(deps, callback)` | `deps` may mix strings and `ServiceKey`s. |
| `ctx.plugin(plugin, config?)` | `ctx.plugin(plugin, config=None, *, policy=None, load_timeout=None, dispose_timeout=None)` | Returns the `Fiber`, which is awaitable: `await ctx.plugin(...)` waits for the load and raises its error. `config` is type-checked against the plugin's config parameter. The keyword-only supervision options are pycordis-only. |
| `ctx.on` / `ctx.once` | `ctx.on` / `ctx.once` | Return a `pycordis.ListenerDisposer` (call it to remove; returns `bool`; `.effect` is the `EffectDisposer`). Also decorators: `@ctx.on("name")`. |
| `ctx.emit` / `parallel` / `serial` / `bail` / `waterfall` | same | `parallel` and `serial` are `async`. |
| `ctx.inspect()` | `ctx.inspect()` | Returns a `FiberNode` tree; `pycordis.inspect.render()` prints it. |
| `Context.is(value)` | `Context.is_(value)` | Type guard. |
| - | `ctx.use(KEY)` | **pycordis-only.** Typed read through a `ServiceKey[T]`; resolves exactly like `ctx.<name>` (same `inject` check). |
| - | `ctx.scope(name=None, inject=None, provide=(), config=None, timeout=None)` | **pycordis-only.** `async with` block with a plugin lifetime (`pycordis.Scope`). |
| - | `ctx.diagnose()` | **pycordis-only.** `Diagnosis` tree explaining every fiber that is not `ACTIVE`. |
| `ctx[name]` | `ctx["name"]` | Same as attribute access (`getattr(ctx, name)`), including the `inject` check: an undeclared service raises `ServiceAccessError`. Use it for names that are not identifiers (`"foo.bar"`). `ctx.get(name)` is the unchecked read that returns `None`. |
| reading an undeclared service → `undefined` | raises `ServiceAccessError` (`AttributeError`) | `cannot get property "x" without inject`; `hasattr` / `getattr(ctx, "x", default)` keep working. |
| `Context.effect`, `Context.filter`, `Context.isolate`, `Context.intercept` (symbols) | class attributes holding the names `__cordis_effect__`, `__cordis_filter__`, `_isolate`, `_intercept` | Only these four exist as `Context` class constants. `Context.extend` is the instance method `ctx.extend(**meta)`, not a symbol. |
| `declare module 'cordis' { interface Context {...} }` | `class AppContext(Context): counter: Counter` | Typing only. |
| `using` / `await using` | `async with Context() as root:` | Disposes the root on exit. |

## Fiber (`pycordis.Fiber`, `pycordis.FiberState`)

| cordis | pycordis | Notes |
| --- | --- | --- |
| `fiber.state` / `FiberState.*` | `fiber.state` / `FiberState.PENDING, LOADING, ACTIVE, FAILED, DISPOSED, UNLOADING` | `IntEnum`, same numbering. |
| `fiber.name` | `fiber.name` | |
| `fiber.uid`, `fiber.parent`, `fiber.runtime` | same | |
| `fiber.committed`, `fiber.provisions`, `fiber.inject`, `fiber.inertia` | same | `inertia` is the in-flight transition's `asyncio.Task`, or `None`. |
| `fiber.await()` | `await fiber.wait()` (or `await fiber`) | Returns the fiber or raises the load error. |
| `fiber.dispose()` | `fiber.dispose()` | Returns an `asyncio.Future`. |
| `fiber.restart()` | `await fiber.restart()` | |
| `fiber.update(config, noSave?)` | `fiber.update(config, no_save=False)` | Volatile-only changes bypass `internal/update`. Pass raw config (see the secrets note in [reference.md](reference.md)). |
| `fiber.effect(execute, label)` | `fiber.effect(execute, label="anonymous")` | Accepts disposers, awaitables, (async) generators and (async) context managers. |
| `fiber.getEffects()` | `fiber.get_effects()` | |
| `fiber.assertActive()` | `fiber.assert_active()` | |
| `fiber.rawConfig` | `fiber.raw_config` | |
| `fiber.effectiveConfig` | `fiber.effective_config` | |
| `fiber.resolveConfig(config)` | `fiber.resolve_config(config)` | |
| - | `fiber.spawn(coro, name=None)` / `ctx.spawn(...)` | **pycordis-only.** Fiber-owned `asyncio.Task`, cancelled on unload, run with the use-site context as `current_context()`. An in-flight async plugin body is cancelled when its epoch changes. |
| - | `fiber.diagnose()`, `fiber.stats`, `fiber.error` | **pycordis-only.** Diagnosis below this fiber, a `FiberStats` snapshot, and the last load error (or `None`). |

## Effects (`pycordis.EffectDisposer`, `pycordis.EffectMeta`)

| cordis | pycordis | Notes |
| --- | --- | --- |
| effect disposer `() => Promise<void>` | `EffectDisposer` (callable, returns `asyncio.Future`) | `.active` property; a raising disposer does not stop the rest, the first error is raised and the others reported. |
| `EffectMeta` | `EffectMeta(label, children)` | |
| `() => boolean` listener disposer | `ListenerDisposer` | Returns `bool`. |

## Registry and Inject (`ctx.registry`, `pycordis.Inject`, `pycordis.inject`)

| cordis | pycordis | Notes |
| --- | --- | --- |
| `registry.plugin(plugin, config)` | `registry.plugin(plugin, config=None, *, policy=None, load_timeout=None, dispose_timeout=None)` | Plugin shapes: function, class, object with `apply`, plus generator and async-generator functions. |
| `registry.inject(deps, cb)` | `registry.inject(deps, callback)` | |
| `registry.resolve` / `get` / `has` / `delete` | same | |
| `registry.keys()` / `values()` | `keys()` / `values()` | |
| `registry.entries()` | `registry.items()` | |
| `registry.size` | `len(registry)` | |
| `registry.forEach(cb)` | `registry.for_each(cb)` | |
| `[Symbol.iterator]` | `iter(registry)` | |
| `Inject.resolve(inject)` | `Inject.resolve(inject)` | List or dict form; items are strings or `ServiceKey`s. Anything else raises `TypeError`. |
| `@Inject(name, config?)` | `@inject(name, config=None)` | Decorates classes (adds to `inject`) and methods (runs the method as an inner plugin once the service exists). Stacked `@inject` on one method merge into one inner plugin. |
| plugin metadata `name`, `Config`, `inject`, `provide`, `intercept` | same attribute names | pycordis also reads `policy`, `load_timeout` and `dispose_timeout`. |
| - | `@plugin_meta(name=..., inject=..., provide=..., Config=..., policy=..., load_timeout=..., dispose_timeout=...)` | **pycordis-only.** Typed decorator setting those attributes (strict pyright rejects `fn.inject = ...`). |

## Service (`pycordis.Service`)

| cordis | pycordis | Notes |
| --- | --- | --- |
| `class X extends Service` + `super(ctx, 'name')` | `class X(Service)` + `super().__init__(ctx, "name")` | |
| `static provide` | `class X(Service, provide="name")` | `provide=` also takes a `ServiceKey`. |
| `Service.invoke`, `.check`, `.filter`, `.init`, `.extend`, `.tracker`, `.resolveConfig` (symbol-keyed hooks) | dunder hooks you define or override: `__cordis_invoke__`, `__cordis_check__`, `__cordis_filter__`, `__cordis_init__`, `__cordis_extend__`, `__cordis_tracker__`, `__cordis_resolve_config__(base, head)` | There are no `Service.invoke` etc. attributes. Only subclasses defining `__cordis_invoke__` become callable. `__cordis_resolve_config__` uses `Config.merge(*configs)` when declared, else a shallow dict merge. |
| `Service.Config.merge` | `Config.merge` | |
| `service.ctx` | `service.ctx` | Through a traced proxy: the use-site context. |
| generic `Service<C>` | `Service[C]` | No default for `C` (PEP 696 needs 3.13+); strict users write `Service[Any]`. |

## Reflect (`ctx.reflect`)

`get(name, strict=True)`, `set`, `provide`, `accessor`, `mixin`, `notify`,
`trace`, `bind`, `props`, `store` keep their names. Error messages are
unchanged; the one difference is that missing services raise
`ServiceAccessError` instead of returning `undefined`.

## Events (`ctx.events`, `pycordis.EventKey`, `pycordis.EventOptions`)

| cordis | pycordis | Notes |
| --- | --- | --- |
| `emit`, `bail`, `waterfall` | same | Coroutines returned by `emit` listeners are scheduled as tasks owned by the listener's fiber; failures are logged. |
| `parallel` | `await ctx.parallel(...)` | Raises `ExceptionGroup` where cordis raises `AggregateError`. Coroutine listeners start after all sync listeners have run. |
| `serial` | `await ctx.serial(...)` | |
| `on(name, listener, options)` | `on(name, listener, options=None)` | Options: `prepend`, `global_` (cordis `global`, a keyword in Python). |
| `isBailed` | `is_bailed` | `None` and `False` do not bail. |
| leading `thisArg` | leading `this_arg` | A Context or object with a filter, detected as in cordis. |
| `internal/plugin`, `internal/status`, `internal/service`, `internal/update`, `internal/commit`, `internal/config`, `internal/get`, `internal/set`, `internal/listener`, `internal/dispatch` | identical | Same argument order. |
| - | `EventKey[P, R]` | **pycordis-only.** Typed event keys. Event names that are neither `str` nor `EventKey` raise `TypeError`. |
| - | `internal/telemetry` (`pycordis.TELEMETRY`) | **pycordis-only.** `LifecycleEvent` per load, unload, failure, restart, give-up and timeout. |
| - | `collection/<name>` (`Collection.changes(KEY)`) | **pycordis-only.** `CollectionChange` on every add and removal. |

## Schema / config validation

| cordis | pycordis | Notes |
| --- | --- | --- |
| Standard Schema `~standard.validate` | object with `validate(value)` returning a value or issues; dataclass; pydantic model (`model_validate`); plain callable | All raise `ValidationError` with the cordis format `invalid config:\n  - msg (at a.b)`. |
| `Schema` issues | `Issue(message, path)` | |
| cosmokit `Volatile` / `createVolatile` | `Volatile[T]` / `create_volatile(value)` | `.get()` returns a deep-frozen snapshot (`MappingProxyType` / tuples). Also `is_volatile` in `pycordis.volatile`. |
| async validation | rejected, as in cordis | |

## Errors

| cordis | pycordis |
| --- | --- |
| `CordisError` (code `INACTIVE_EFFECT`) | `CordisError(code, message=None)` with `.code` |
| `AggregateError` | `ExceptionGroup` |
| validation `TypeError` | `ValidationError(TypeError)` with `.issues` |
| - | `pycordis.ServiceAccessError(AttributeError)` (property read/write/declare failures) |
| - | `pycordis.LoadTimeoutError(TimeoutError)` (a plugin body exceeded its `load_timeout`) |

## Logger (`ctx.logger`)

| cordis | pycordis | Notes |
| --- | --- | --- |
| `ctx.logger(name)` | `ctx.logger("name")` | Returns a `Logger`. |
| `ctx.logger.info/warn/error/debug` | same | Name defaults to the hyphenated fiber name. |
| `Logger.Level` | `LoggerLevel` | `ERROR`, `WARN`, `INFO`, `DEBUG`. |
| `Logger.Record` | `Message` | |
| `Logger.Target` / exporters | `Exporter` protocol; `ctx.logger.exporter(exporter)` returns an `EffectDisposer` | |
| `Logger.format`, `color`, `code` | `Logger.format`, `Logger.color`, `Logger.code` | `%s %d %i %f %o %O %c %C`. |
| buffer | `ctx.logger.buffer` (ring buffer, size 1000) | |
| - | `ConsoleExporter`, `StdlibLoggingExporter`, `JsonExporter`, `capture_stdlib_logging(ctx)` | **pycordis-only.** `JsonExporter` writes JSON lines. |

## Timers (cordis-plugin-timer)

| cordis | pycordis |
| --- | --- |
| `ctx.sleep(ms)` | `await ctx.sleep(seconds)` (seconds, not milliseconds) |
| `ctx.setTimeout(cb, ms)` | `ctx.set_timeout(cb, seconds)` |
| `ctx.setInterval(cb, ms)` | `ctx.set_interval(cb, seconds)` |
| `ctx.timer.debounce` / `throttle` | `ctx.timer.debounce(cb, seconds)` / `throttle(...)` |

Timers are fiber-owned and raise `INACTIVE_EFFECT` on unloading fibers.

## Loader (replaces cordis-plugin-loader / include)

Python-native design with the same model; see [reference.md](reference.md) for the file format.

| Item | pycordis |
| --- | --- |
| Service | `ctx.loader` (`pycordis.loader.Loader`, options `LoaderConfig(path, watch, allow, readonly, dispose_timeout, profile)`) |
| API | `async`: `start()`, `reload()`, `create(entry, parent=None)`, `update(id, patch)`, `remove(id)`, `save()`; sync: `get(id)`, `entries()` |
| Data layer | `Entry`, `LoaderOptions`, `LoadedEntry`, `Overlay`, `load_tree`, `read_config`, `expand`, `import_plugin`, `LoaderConfigError` |
| Entry keys (pycordis-only) | `restart`, `load_timeout`, `dispose_timeout` |
| Profiles (pycordis-only) | `<stem>.<profile><suffix>` overlays, `--profile` / `PYCORDIS_PROFILE` / `LoaderConfig(profile=...)` |
| Entry points | group `pycordis.plugins` |
| CLI | `pycordis [start\|inspect\|doctor\|schema]`, `python -m pycordis` |

## pycordis-only additions

Everything here is new in pycordis; cordis 4.1.0 has no counterpart. [reference.md](reference.md)
documents each one, and [guide-agents.md](guide-agents.md) uses most of them
together.

| Feature | pycordis API | Notes |
| --- | --- | --- |
| Typed service keys | `ServiceKey[T](name)`; `ctx.use(KEY)`, `ctx.get(KEY)`, `ctx.provide(KEY, value)`, `ctx.set(KEY, value)`, `inject=[KEY, "name"]`, `class X(Service, provide=KEY)` | Keys resolve to their plain name at the call boundary; compared by identity; names starting with `_` are rejected. |
| Typed plugin metadata and configs | `@plugin_meta(...)`; the `Plugin[C]` type (a `(ctx, config: C)` callable or a `PluginObject[C]`); `ctx.plugin(plugin, config)` checked against `C` | A mapping config is always accepted (validated at runtime); a dataclass `Config` instance passes unchanged. |
| Ambient context | `current_context()`, `use_context(ctx)` | `contextvars`-based; set for plugin bodies, effects, disposers, listeners, timers and spawned tasks. `LookupError("no active pycordis context")` outside. |
| Scopes | `async with ctx.scope(...) as s:` (`pycordis.Scope`) | Waits for injections (`timeout`), disposes on exit; `provide=[...]` names are isolated per scope. |
| Collections | `Collection[T]`, `Collection.of(KEY)`, `Collection.changes(KEY)`, `CollectionChange` | Ordered, name-unique; items removed when the adding fiber unloads; `add()` returns an `EffectDisposer`. |
| Supervision | `RestartPolicy(mode, max_restarts, window, backoff)`, `Backoff(initial, factor, max, jitter)`; `policy=` / `load_timeout=` / `dispose_timeout=` | `mode` is `"never"` (default) or `"on-failure"`. Load timeouts raise `LoadTimeoutError`. |
| Diagnostics | `ctx.diagnose()`, `fiber.diagnose()`, `Diagnosis`, `Reason`, `render_diagnosis()`; `pycordis doctor` | Missing or failed providers, rejected `check()`, isolation mismatches, dependency cycles (also logged as a warning), backoff, failures. |
| Telemetry | `TELEMETRY`, `LifecycleEvent`, `LifecycleKind`, `fiber.stats` (`FiberStats`) | Events are built only while a listener exists. |
| JSON logs | `JsonExporter(stream=None, levels=None, *, max_length=10240)` | One JSON object per line. |
| Health server | `HealthServer`, `HealthServer.Config(...)`; `pycordis start --health-port PORT [--health-host HOST]` | `/healthz`, `/readyz`, `/inspect`; read-only, `127.0.0.1` by default. |
| Config profiles | `--profile NAME`, `PYCORDIS_PROFILE`, `LoaderConfig(profile=...)` | Overlay entries matched by `id`, `config` deep-merged; write-back targets the file defining the value. |
| JSON Schema | `config_json_schema(Config)`, `loader_json_schema(entries, allow=("*",), *, overlay=False, skipped=None)`; `pycordis schema [--overlay] [-o FILE]` | Draft 2020-12; each entry's `config` typed from its plugin's dataclass `Config`. |

## Extras with no cordis counterpart

`ctx.spawn`, `async with Context()`, `root.inspect()`, strict mode,
`pycordis.testing` (`mock_context`, `settle`, `MemoryExporter`), `EventKey`,
`pycordis.run(main)`, `pycordis.tomledit.TomlDocument`, and the
[pycordis-only additions](#pycordis-only-additions) above.
