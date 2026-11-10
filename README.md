# pycordis

A zero-dependency, asyncio-native plugin framework for Python 3.12+. It is a
faithful port of [cordis](https://github.com/cordiverse/cordis) 4.1.0: a tree of
**contexts**, **plugins** that run in **fibers**, **services** resolved through
dependency injection, and **events** — with the same lifecycle, the same event
names and the same error messages. On top of that it adds structured
concurrency, a built-in config-file loader with a command line, introspection
and testing helpers.

- No runtime dependencies (`dependencies = []`), pyright-strict typed, `py.typed`.
- Plugins reload automatically when a service they inject is replaced, and tear
  down dependents first.
- Names follow Python conventions (`snake_case`); see
  [docs/cordis-mapping.md](https://github.com/aswin-kevin/pycordis/blob/main/docs/cordis-mapping.md) for the cordis → pycordis table.
- Beyond cordis: typed service keys and configs, per-request scopes,
  lifecycle-bound collections, supervision with restarts and timeouts,
  diagnostics, telemetry, a health server, config profiles and JSON Schema
  export. [docs/guide-agents.md](https://github.com/aswin-kevin/pycordis/blob/main/docs/guide-agents.md) puts them together in an
  LLM agent example.

## Install

```sh
pip install pycordis        # or: uv add pycordis
```

Optional: PyYAML enables `.yaml` / `.yml` loader config files. TOML (`tomllib`)
and JSON are built in.

## Quick start

A service, a plugin that injects it, and an event (the cordis README example):

```python
import asyncio
from typing import Any

from pycordis import Context, Logger, Message, Service


async def main() -> None:
    class Counter(Service[Any]):
        value = 0

        def __init__(self, ctx: Context, _config: Any = None) -> None:
            super().__init__(ctx, "counter")

        def next(self) -> int:
            self.value += 1
            return self.value

    def greeter(ctx: Context, _config: Any) -> None:
        def on_ready(message: str) -> None:
            ctx.logger.info("%s #%d", message, ctx.counter.next())

        ctx.on("app/ready", on_ready)

    root = Context()
    lines: list[str] = []

    class Lines:
        def export(self, message: Message) -> None:
            lines.append(Logger.format(self, message))

    root.logger.exporter(Lines())
    await root.plugin(Counter)
    await root.inject(["counter"], greeter)

    root.emit("app/ready", "started")
    root.emit("app/ready", "started")
    assert lines == ["started #1", "started #2"]
    assert [m.name for m in root.logger.buffer] == ["greeter", "greeter"]
    assert root.counter.value == 2
    await root.fiber.dispose()
    root.emit("app/ready", "started")
    assert lines == ["started #1", "started #2"]


asyncio.run(main())
```

## Concepts

**Context.** `Context()` creates a root. `ctx.extend(...)`, `ctx.isolate(name)`
and `ctx.intercept(name, config)` derive children that never mutate the parent.
Services are read as attributes (`ctx.counter`); reading one in a context that
did not declare it with `inject` raises `pycordis.ServiceAccessError` (a subclass of
`AttributeError`, so `hasattr` / `getattr(ctx, "x", default)` work as usual).
Names that are not identifiers use `ctx["foo.bar"]`. `Context.is_(value)` is the
type guard. `async with Context() as root:` disposes the root on exit.

**Plugins.** `await ctx.plugin(plugin, config)` returns the plugin's `Fiber`.
A plugin is a function `(ctx, config)`, a class (constructed as
`cls(ctx, config)`), an object with `apply(ctx, config)`, or a generator /
async generator function that *yields disposer callables* (cordis semantics).
The generator is drained during setup, so it is not a pytest-style fixture: code
after a bare `yield` runs immediately, not on unload (pycordis warns when a
generator yields `None`). Put cleanup in the yielded callable:

```python
def watcher(ctx, config):
    handle = open_connection(config)  # setup
    yield handle.close  # runs when the fiber unloads
    timer = start_timer()
    yield lambda: timer.cancel()  # disposers run in reverse order
```

Metadata
attributes: `name`, `Config`, `inject`, `provide`, `intercept`. A fiber moves
through `PENDING → LOADING → ACTIVE` (or `FAILED`), and `UNLOADING → DISPOSED`
on removal; `await fiber.wait()` returns the fiber or raises the load error and
`await fiber.dispose()` removes it. Errors from plugin bodies are re-raised to
the awaiter *and* reported through `ctx.logger`.

**Services.** Subclass `Service` and name it with
`class Counter(Service, provide="counter")` or `super().__init__(ctx, "counter")`.
Declare what a plugin needs with `plugin.inject = ["counter"]`, a dict
(service name → intercept config for that service, `None` for none), or
`@inject("counter")` on a class or method; `ctx.inject(["counter"], fn)` runs a
function plugin with those dependencies (as in the quick start). A subclass that overrides an
`@inject`-decorated method without decorating the override drops that method's
hook (decorate the override again). For a module used as a plugin, metadata
attributes of the wrong type are ignored, so `from pycordis import inject` at
module level is not mistaken for its `inject` list. When a provider
appears, changes or disappears, dependents are reloaded or unloaded
automatically, dependents first. `ctx.provide(name, value)`,
`ctx.accessor(name, get=..., set=...)` and `ctx.mixin(...)` are the lower-level
reflection API (`ctx.reflect`). Assigning to a getter-only accessor raises
`ServiceAccessError`. Services expose tracing: a service read through a context
sees that context (`service.ctx`).

**Events.** `ctx.on(name, listener)` / `ctx.once(...)` (also usable as a
decorator) return a `pycordis.ListenerDisposer`: calling it removes the listener
synchronously and returns `True` if it was still registered. Dispatch modes:
`emit`, `bail`, `waterfall`, plus async `parallel` and `serial`. `parallel`
raises an `ExceptionGroup`. Coroutines returned from `emit` listeners run as
tasks owned by the listener's fiber; failures are logged. `EventKey[P, R]` gives
type-checked signatures; plain strings work too. The built-in `internal/*`
events (`internal/plugin`, `internal/status`, `internal/service`,
`internal/update`, `internal/commit`, `internal/config`, ...) keep cordis' names
and argument order.

**Config.** A plugin's `Config` can be a dataclass (nested dataclasses,
`list[...]`, `dict[...]`, `Literal`, unions, enums are validated), an object
with `validate(value)`, a pydantic model, or a plain callable that raises to
reject. Failures raise `ValidationError`:

```text
invalid config:
  - expected int, got str (at retries)
```

`Volatile[T]` fields hold a stable reference whose `.get()` returns a frozen
snapshot. `fiber.update(config)` that changes only volatile values updates the
snapshots in place and the plugin keeps running; listeners of `internal/update`
are not consulted for such volatile-only changes. Any other change goes through
the `internal/update` waterfall: a listener may veto the restart by not calling
`next` (`await next()` is always safe). Use `fiber.raw_config`,
`fiber.effective_config` and `fiber.resolve_config(...)` to inspect configs.

**Structured concurrency.** Everything a plugin creates is owned by its fiber:
`ctx.spawn(coro, name=...)` starts a task that is cancelled and awaited on
unload; `ctx.sleep(s)`, `ctx.set_timeout(cb, s)`, `ctx.set_interval(cb, s)`,
`ctx.timer.debounce` and `ctx.timer.throttle` are fiber-owned as well. An
in-flight async plugin body is cancelled when its fiber's epoch changes
(shutdown, a service swap, a restart), so async plugin bodies must be
cancellation-safe. A task that ignores cancellation blocks unload (as with
`asyncio.TaskGroup`) unless the fiber has a `dispose_timeout`.

**Logging.** `ctx.logger.info("%s #%d", name, n)` logs under the plugin's
hyphenated name; `ctx.logger("name")` returns a named `Logger`. Levels are
`ERROR`, `WARN`, `INFO`, `DEBUG`; exporters have per-name thresholds
(`ctx.logger.exporter(exporter)`); the last 1000 messages stay in
`ctx.logger.buffer`. `ConsoleExporter` is coloured and TTY-aware,
`StdlibLoggingExporter` forwards to `logging`, and
`capture_stdlib_logging(ctx)` routes third-party `logging` records into
pycordis exporters.

**Introspection.** `root.inspect()` returns a tree of fibers (state, provided
and injected services, effect labels); `pycordis.inspect.render` prints it.
`root.diagnose()` explains what is not `ACTIVE` (see
[Diagnostics](#diagnostics)). Because `inspect` is a `Context` method, a
service named `"inspect"` is not reachable as `ctx.inspect`; read it with
`ctx.get("inspect")`. `Context(strict=True)` (or `PYCORDIS_STRICT=1`) warns
about services read in disposed contexts. It also checks, at the end of each non-root
fiber's unload (not at root dispose), that every listener that fiber registered
was removed, and warns about listeners that outlived it.

**Testing helpers** (`pycordis.testing`, no pytest dependency):

```python
from pycordis.testing import MemoryExporter, mock_context, settle

logs = MemoryExporter()
async with mock_context(logs) as ctx:
    await ctx.plugin(my_plugin)
    await settle(ctx)  # wait for pending load/unload work
    assert logs.text("warn") == []
```

## Typed services, scopes and collections

These are pycordis additions with no cordis counterpart. They are opt-in: plain
strings, attribute metadata and plain cordis-style code keep working.

### Typed service keys

A `ServiceKey[T]` is a service name that carries the service's type. Every API
that takes a service name also takes a key, and pyright checks the values:

```python
import asyncio
from typing import Protocol

from pycordis import Context, Service, ServiceKey, plugin_meta


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


LLM = ServiceKey[LLMClient]("llm")


class EchoLLM(Service[None], provide=LLM):  # a key works wherever a name does
    def complete(self, prompt: str) -> str:
        return f"echo: {prompt}"


@plugin_meta(inject=[LLM])
def chat(ctx: Context, config: None) -> None:
    print(ctx.use(LLM).complete("hi"))  # ctx.use(LLM) is typed LLMClient


async def main() -> None:
    async with Context() as root:
        await root.plugin(EchoLLM)
        await root.plugin(chat)  # prints "echo: hi"
        assert root.get(LLM) is not None  # LLMClient | None


asyncio.run(main())
```

`ctx.use(KEY)` resolves exactly like `ctx.llm`, with the same `inject` check and
the same `ServiceAccessError`. `ctx.get(KEY)` is the unchecked read
(`T | None`), and `ctx.provide(KEY, value)` / `ctx.set(KEY, value)` check
`value` against `T`. `inject=[LLM, "db"]` mixes keys and strings, and
`ctx.scope(provide=...)` and `Collection` take keys too. Keys compare by
identity, so give each service one key. Names starting with `_` are rejected.

### `plugin_meta` and typed configs

`@plugin_meta(...)` declares plugin metadata with a typed decorator. Strict
pyright rejects ad-hoc `fn.inject = [...]` assignments on functions; the
decorator does the same job and is checked. It takes `name`, `inject`,
`provide`, `Config`, `policy`, `load_timeout` and `dispose_timeout`, and works
on functions, classes and objects. The attribute form still works at runtime.

`ctx.plugin(plugin, config)` checks `config` against the type of the plugin's
config parameter: `(ctx, config: Cfg)` for a function, `__init__(self, ctx,
config: Cfg)` for a class. A mapping is always accepted, because the runtime
schema validates it:

```python
from dataclasses import dataclass


@dataclass
class ChatConfig:
    model: str
    retries: int = 3


@plugin_meta(name="chat", inject=[LLM], Config=ChatConfig)
def chat(ctx: Context, config: ChatConfig) -> None:
    ctx.logger.info("using %s", config.model)


await root.plugin(chat, ChatConfig(model="small"))  # an instance passes unchanged
await root.plugin(chat, {"model": "small"})  # validated into ChatConfig(model="small")
root.plugin(chat, "small")  # pyright error: "Literal['small']" is not assignable
```

A plugin whose static type is plain `object` accepts only an `Any` or `None`
config.

Omitting the config (or passing `None`) always type-checks, like a mapping: a
`Config` schema whose fields all have defaults builds the config from nothing,
and the checker cannot see whether a plugin has one. Without a schema, the body
then receives `None`, so give such a plugin an explicit config, or type its
parameter `Cfg | None`.

A mismatch is reported as `No overloads for "plugin" match the provided
arguments`, followed by `Argument of type "Literal['small']" cannot be assigned
to parameter "config" of type "None"`. The `None` comes from the last
`ctx.plugin` overload, the one for plugins of unknown shape. Read it as "this
config does not match the plugin's config type". Hover the plugin to see the
type it expects.

### Current context

`current_context()` returns the context whose code is running. It is set while
a plugin body, an effect or its disposer, an event listener, a timer callback
or a task from `ctx.spawn` runs. Plain helpers, such as an agent's tool
functions, can reach services without a `ctx` parameter. Outside all of these it
raises `LookupError("no active pycordis context")`. `use_context(ctx)` sets it
for a `with` block, for example in a callback that another library calls:

```python
from pycordis import current_context, use_context


def search(query: str) -> list[str]:  # an agent tool: no ctx parameter
    return current_context().use(INDEX).search(query)


with use_context(root):
    search("pycordis")
```

The value is a `contextvars.ContextVar`, so asyncio tasks inherit it. Inside a
derived context (`ctx.extend`, an isolated view), it is that context: the use
site, not the fiber's own context.

### Scopes

`ctx.scope(...)` gives a block of code a plugin lifetime, for example per
request or per conversation. Entering starts an anonymous plugin, waits until
its injections are available (at most `timeout` seconds, then `TimeoutError`)
and binds its context. Leaving disposes everything the block registered
through that context: effects, listeners, tasks, provided services and
collection items. This happens also on an exception or a cancellation:

```python
async with ctx.scope(name="session-42", inject=[LLM], provide=[MEMORY], timeout=5) as s:
    s.provide(MEMORY, ConversationMemory())  # visible only inside this scope
    await s.plugin(chat_agent)  # sees this session's memory
    reply = await s.spawn(answer(prompt))  # runs with s as current_context()
```

Names in `provide` are isolated to the scope, so concurrent scopes that provide
`"memory"` never collide. Scope isolation is opt-in: a service the block
provides under a name *not* listed in `provide` is visible to the whole
application until the scope exits. A scope cannot both inject and provide a
name (`ValueError`). If an injected service goes away mid-block, the scope
fiber unloads (tearing down what the block registered) and waits for the
service again, while the block keeps running. See `pycordis.scope` for what
happens when the parent unloads.

### Collections

A `Collection[T]` is a service holding an ordered, name-unique registry of
items: tools, routes, prompt templates, commands. Each item is owned by the
fiber that added it and is removed when that fiber unloads:

```python
import asyncio
from dataclasses import dataclass

from pycordis import Collection, CollectionChange, Context, ServiceKey, plugin_meta


@dataclass(frozen=True)
class Tool:
    description: str


TOOLS = ServiceKey[Collection[Tool]]("tools")


@plugin_meta(inject=[TOOLS])  # inject the key so items come back after a provider restart
def web_tools(ctx: Context, config: None) -> None:
    ctx.use(TOOLS).add(Tool("search the web"), name="search")


async def main() -> None:
    async with Context() as root:
        await root.plugin(Collection.of(TOOLS))  # provides an empty collection as TOOLS

        def on_change(change: CollectionChange[Tool]) -> None:
            print(change.kind, change.name)

        root.on(Collection.changes(TOOLS), on_change)
        fiber = await root.plugin(web_tools)  # prints "add search"
        tools = root.use(TOOLS)
        assert [name for name, _ in tools.items()] == ["search"]
        assert tools.get("search") == Tool("search the web")
        await fiber.dispose()  # prints "remove search"
        assert "search" not in tools and len(tools) == 0


asyncio.run(main())
```

`add(item, name=None)` returns an `EffectDisposer`: call it to remove the item
early. Without a name, it is `"<fiber name>#<n>"`. A duplicate name raises
`ValueError` naming both owners. Iteration and `items()` work on snapshots.
Changes are announced on `collection/<name>` (`Collection.changes(KEY)` is the
typed event key) and respect isolation. A change listener that raises is
logged; the error never reaches `add` or the removal. When the provider
unloads, the collection drops every item still in it (one `remove` change
each), so a reference kept past that is empty and its `add` raises
`CordisError("INACTIVE_EFFECT")`.

## Supervision, diagnostics and observability

### Supervision and timeouts

A plugin can be restarted when it fails, and its load and unload can be
bounded. Set these per call, as plugin metadata
(`plugin_meta(policy=..., load_timeout=..., dispose_timeout=...)`) or as loader
entry keys (`restart`, `load_timeout`, `dispose_timeout`). The defaults match
cordis: never restart, no timeouts.

```python
from pycordis import Backoff, RestartPolicy

policy = RestartPolicy(
    mode="on-failure",  # or "never" (default)
    max_restarts=5,  # within any `window` seconds, then give up
    window=60.0,
    backoff=Backoff(initial=0.5, factor=2.0, max=30.0, jitter=0.1),
)
ctx.plugin(worker, policy=policy, load_timeout=30, dispose_timeout=10)
```

**Restarts.** With `on-failure`, a `FAILED` fiber is loaded again after the
backoff, and `restarting worker in 0.50s (attempt 1/5)` is logged as a warning.
The delay before restart `n` is `initial * factor ** (n - 1)`, capped at `max`
and scaled by a random factor within `±jitter`. After `max_restarts` restarts
within `window` seconds, the fiber stays `FAILED` and
`giving up on worker after 5 restarts in 60s` is logged once. `window` must
exceed a restart's latency (backoff delay plus load time), or restarts leave it
before they count; a window under 10 ms warns. The backoff wait is owned by the
parent, so disposing the parent or the root cancels it at once. A
`fiber.update(...)` is a new configuration: it cancels a pending restart and
resets the restart budget.

**Timeouts.** `load_timeout` bounds an async plugin body. When it expires, the
body is cancelled and the fiber fails with `LoadTimeoutError`, a `TimeoutError`
subclass that also counts as a failure for supervision. A body that catches
the cancellation and returns anyway still fails with `LoadTimeoutError` once it
returns, and what it set up is unloaded. A sync body cannot be interrupted.
Unless a `dispose_timeout` is set, the unload that follows is bounded by
`load_timeout` as well, so an effect the body left in flight cannot delay
`FAILED`. `dispose_timeout` bounds an unload: tasks still running and
disposers still pending when it expires are cancelled and abandoned, with an
error naming the effect label and the plugin, and the unload completes. Only
awaiting work can be abandoned: a disposer that blocks the event loop
synchronously cannot be bounded. The CLI sets `dispose_timeout` to 10 seconds
by default (`--dispose-timeout`).

### Diagnostics

`ctx.diagnose()` (or `fiber.diagnose()`) explains why each fiber below it is
not `ACTIVE`. It returns a `Diagnosis` tree: `healthy`, `unhealthy()` (only
what is stuck, with its ancestors), and per fiber its `Reason`s (`kind`,
`service`, `detail`, and `path` for a cycle):

```python
stuck = root.diagnose().unhealthy()
if stuck is not None:
    print(render_diagnosis(stuck))
```

`pycordis doctor` prints the same rendering for a config file. Take a plugin
that injects a service nobody provides:

```python
# app_plugins.py
from pycordis import Context, plugin_meta


@plugin_meta(name="web-search", inject=["search_api"])
def web_search(ctx: Context, config: None) -> None:
    ctx.logger.info("searching with %s", ctx.search_api)
```

```toml
# app.toml
[[plugins]]
id = "search"
use = "app_plugins:web_search"
```

It stays `PENDING`, and doctor explains why and exits 1:

```text
$ pycordis doctor -c app.toml
root #0 ACTIVE
└─ web-search #1 [entry search] PENDING
      missing-service: no plugin declares or currently provides "search_api"
```

The reason kinds are:

- `missing-service`: nobody provides the injected service. The detail names a
  plugin that declares it, if one exists.
- `provider-failed`: the provider is `FAILED`. The detail includes its error.
- `check-failed`: the service's `check()` rejected it.
- `isolated`: the service is provided only in another isolated scope.
- `cycle`: the fiber is in a dependency cycle (`a -> "llm" -> b -> "cache" -> a`).
- `failed`: the load failed (with the error).
- `backoff`: a restart is scheduled.
- `loading` / `unloading`: a transition is in flight.

A dependency cycle is also logged as a warning when it forms. cordis leaves
such plugins `PENDING` without a word.

`pycordis doctor -c app.toml [--timeout SECONDS]` does this for a config file.
It starts the tree, waits for it to settle (default 10 seconds, startup
included), prints the diagnosis of everything not `ACTIVE`, with loader entries
labelled `[entry <id>]`, or `all fibers are ACTIVE`. Then it disposes the tree.
The exit code is 0 when everything is `ACTIVE`, else 1.

### Telemetry and JSON logs

Every fiber keeps counters, read as a `FiberStats` snapshot:
`fiber.stats.loads`, `failures`, `restarts`, `last_load_duration` and
`last_error`. Listeners of `TELEMETRY` (`"internal/telemetry"`) also get a
`LifecycleEvent` for each step: `kind` (`"load"`, `"unload"`, `"fail"`,
`"restart"`, `"give-up"` or `"timeout"`), `fiber`, `uid`, `state`, `duration`,
`error` and `ts`. Events are built only while a listener exists; without one,
each lifecycle step costs a single dictionary lookup:

```python
from pycordis import TELEMETRY, JsonExporter, LifecycleEvent


def on_lifecycle(event: LifecycleEvent) -> None:
    if event.kind in ("fail", "restart", "give-up"):
        print(event.kind, event.fiber, event.error)


root.on(TELEMETRY, on_lifecycle)
root.logger.exporter(JsonExporter())  # one JSON object per line, on stderr
```

`JsonExporter(stream=None, levels=None, *, max_length=10240)` writes `ts`,
`level`, `name`, `message`, `plugin`, `error_type` and `traceback` on every
line, for log pipelines. It never drops a record: a line the stream cannot
encode is written again ASCII-escaped.

### Health server

`HealthServer` is a plugin (the `health` service). It answers three read-only
JSON endpoints, with no dependencies:

```python
await root.plugin(HealthServer, HealthServer.Config(port=0))  # port 0: any free port
server: HealthServer = root.health
print(f"http://127.0.0.1:{server.port}/readyz")
```

- `GET /healthz` returns `200 {"status":"ok"}` while the event loop runs.
- `GET /readyz` returns `200 {"status":"ready"}`, or `503` with `reasons` from
  diagnostics. With `ready_requires="loader"` (the default), it is ready once
  the loader has started (`Loader.started`: `start()` loaded its config) and
  every enabled loader entry is `ACTIVE`; before that the reason is
  `not-started` with a `null` entry. With `"all"`, every fiber must be
  `ACTIVE`. Error messages are withheld from the reasons, because they can hold
  secrets.
- `GET /inspect` returns the fiber tree, without configs.

`HealthServer.Config` takes `host` (default `"127.0.0.1"`), `port`,
`ready_requires`, `timeout` (5 s per connection), `head_timeout` (2 s for the
request head) and `max_connections` (64). The CLI starts one with
`pycordis start --health-port PORT [--health-host HOST]`. Anything but `GET`
is answered `405`, an unknown path `404`, a request with a body `400`, and a
request head over 8 KiB `431`. See
`pycordis.health` for the exact limits.

## Config-file loader

The loader turns a config file into a tree of live fibers and keeps the two in
sync. It is the `loader` service (`pycordis.loader.Loader`) and what the CLI
runs.

```toml
[loader]
readonly = false         # true: never write back to the file
watch = true             # hot reload on file change (poll-based)
poll_interval = 1.0      # seconds between polls
allow = ["my_agent.*", "pycordis.*"]   # importable plugin modules

[[plugins]]
id = "llm"                               # optional stable id
use = "my_agent.services:OpenAIClient"   # module:attr, or an entry-point name
config = { model = "gpt-x", api_key = "${env:OPENAI_API_KEY}" }

[[plugins]]
use = "my_agent.tools:search"
disabled = false
isolate = ["cache"]              # private scope for service "cache"
# isolate = { cache = "shared" } # or join a labelled scope shared by entries
intercept = { logger = { level = 3 } }
inject = ["llm_client"]          # extra required SERVICE names

[[plugins]]
group = "channels"               # a group is a plugin that loads its children
plugins = [ { use = "my_agent.channels:slack" } ]

[[plugins]]
include = "./plugins/extra.toml" # entries from another file
```

Entry keys:

| Key | Meaning |
| --- | --- |
| `id` | Stable identity. Otherwise generated as `<parent-id>/<key>#<n>` (key is the `use` string, `group:<name>` or `include:<path>`; `n` counts earlier id-less siblings with the same key). Generated ids are never written to your file; reordering two id-less entries with the same key swaps their ids and restarts them, so give them ids. |
| `use` | `package.module:attr.sub` or a name registered under the `pycordis.plugins` entry-point group. Exactly one of `use`, `group`, `include` per entry. |
| `config` | Plugin config (validated against the plugin's `Config`). |
| `disabled` | `true` keeps the entry in the tree without running it. |
| `isolate` | List (fresh private scope per name) or table `{ name = "label" }` (equal labels share a scope). |
| `intercept` | Table `{ service = { ...config } }` merged into that service's config below the entry. |
| `inject` | Extra required **service names** (never entry ids). |
| `restart` | Restart policy table: `mode` (`"never"` / `"on-failure"`), `max_restarts`, `window`, `backoff = { initial, factor, max, jitter }`; omitted keys take the `RestartPolicy` defaults. |
| `load_timeout` | Seconds (> 0) an async plugin body may take to load before it fails with `LoadTimeoutError` (a `TimeoutError`). |
| `dispose_timeout` | Seconds (> 0) the entry's unload may take before stuck work is abandoned. Precedence: this key, then plugin metadata, then the loader's default (`LoaderConfig.dispose_timeout`; the CLI sets 10). |
| `group` / `plugins` | A named group: children share a context; `disabled`, `isolate`, `intercept`, `inject` apply to the whole subtree. |
| `include` | Path to another config file (relative to the including file; cycles are an error), usable with the group keys. |

The `[loader]` table is only read from the root file. Unknown keys, `config` on
a group or include, duplicate ids and malformed interpolations are errors.

**Interpolation** is evaluated when a plugin's config is resolved:
`${env:NAME}` (error if unset), `${env:NAME:-default}` (default if unset *or
empty*), `${file:path}` (UTF-8 contents, one trailing newline removed, max
1 MiB, relative to the declaring file), and `$${` for a literal `${`.

**Reconciliation.** On reload, unchanged entries are untouched, a changed
`config` goes through `fiber.update()` (so plugins may veto the restart),
changed `use` / `isolate` / `intercept` / `inject` / `restart` /
`load_timeout` / `dispose_timeout` recreate the entry, removed
entries are disposed and added ones created, in file order. A rejected new file
keeps the old tree; an identical bad file is not retried until it changes.
Errors from one entry (import denied, invalid config, a failing body) are
logged with the entry id and never stop its siblings.

**Programmatic API.** `async`: `await ctx.loader.start()`, `reload()`,
`create(entry, parent=None)`, `update(id, patch)`, `remove(id)`, `save()`;
sync: `get(id)`, `entries()`.
`create()` of a disabled entry returns `None`.

**Write-back.** `fiber.update(config)` persists the config to the file the
entry is declared in (an included entry writes to the included file), replacing
the file atomically after verifying the result parses to the intended data.
TOML is edited by a built-in format-preserving editor: comments, ordering and
quoting elsewhere are kept. Documented limits (see `pycordis.tomledit`): a
replaced multiline string becomes single-line; a scalar replaced by a table
becomes an inline table; an array whose length changes is re-rendered whole
(comments inside it are lost); deleting a section keeps the comment lines above
its header. JSON keeps its detected indent; YAML is re-serialized (comments
lost). Write failures from `fiber.update` are logged (`cannot save <path>: ...`);
`loader.save()` tries every file, then raises the failures as an
`ExceptionGroup` (handle with `except* LoaderConfigError`). The loader refuses to write when an entry is no longer
at its recorded location (the file changed underneath it).

> **Secrets.** Write-back persists the *raw* config. Pass raw values to
> `fiber.update`, for example a modified copy of `fiber.raw_config` that still
> contains `${env:OPENAI_API_KEY}` — never the resolved config a plugin
> receives or `fiber.effective_config`, which hold expanded secrets that would
> then be written to disk in plaintext. Use `fiber.update(config, no_save=True)` to
> apply a change without persisting it, or set `readonly = true`. As a safety
> net the loader puts a placeholder back wherever the committed value equals
> its expansion (also an expansion seen earlier, before a secret was rotated),
> and refuses (logging `cannot save <path>: would persist an interpolated
> secret`) a write that would still contain a secret of that entry: any
> non-empty `${file:...}` content, or a `${env:...}` value of at least 8
> characters, as a substring of a string or as the `str()` of another value
> (an int PIN; an integral float such as `12345678.0` also by its integer form).
> `loader.update(id, {"config": ...})` and `loader.create(...)` apply the same
> guard and raise `ValueError` (besides logging) when they refuse. Shorter environment values (`"4"`, `"true"`) only get their
> placeholder back on an exact match and never block a write. The loader
> remembers every secret an entry expanded to for as long as the entry exists,
> so dropping a placeholder does not make its value writable, and it puts the
> placeholders back into the config it resolves and into `fiber.raw_config`, so
> expanded values passed to `fiber.update` are never interpolated a second time
> (after a volatile-only update, which bypasses `internal/update`,
> `fiber.raw_config` keeps the passed values until the fiber's next resolution).

**Hot reload.** With `watch = true` the loader polls the root file and every
include (mtime + size, confirmed by a content hash), debounced. A file that
fails to parse or validate is logged once and the running tree is untouched.
The loader's own writes do not trigger a reload. Profile overlays are watched
too, also ones that do not exist yet.

**Profiles.** With a profile, each config file `<stem><suffix>` (the root file
and every include) is layered with an overlay `<stem>.<profile><suffix>` next
to it, when that exists. Select the profile with `--profile NAME`, the
`PYCORDIS_PROFILE` environment variable (the flag wins) or
`LoaderConfig(profile=...)`. A name is letters, digits, `_` and `-` only, so an
overlay path never leaves the file's directory. An overlay has the same shape
as a config file:

```toml
# pycordis.dev.toml, layered over pycordis.toml with --profile dev
[loader]
watch = true                       # [loader] keys override the root file's

[[plugins]]
id = "llm"                         # matches the base entry with this id
config = { model = "local-model" } # deep-merged: the base `timeout` is kept

[[plugins]]
use = "my_agent.tools:debug_tools" # a new entry, appended after the base ones
```

An overlay entry whose `id` names an entry of its base file overrides that
entry's keys. Its `config` is deep-merged: tables merge key by key, and other
values, lists included, are replaced. It may set `disabled`, replace `use` and
add children to a group. It cannot change an entry's kind or its `group` /
`include` value. Any other overlay entry is new and is appended. Overlay ids
match only within their own base file: overlay an included entry in the
include's overlay. Overlays are trusted like the base file, so an overlay's
`allow` can widen the allow-list. Write-back goes to the file that defines
the value: a key or `config` path set in the overlay is written there, and
anything else goes to the base file. One limit: clearing a key from a profile
run also removes it from the base file, because an overlay cannot express
"unset".

**JSON Schema.** `pycordis schema -c pycordis.toml -o pycordis.schema.json`
writes a JSON Schema (draft 2020-12) of the config file. Each entry's `config`
is typed from its plugin's dataclass `Config`. The plugins are imported
(honouring `allow`) to find it, and one that cannot be imported is noted on
stderr and left untyped. `--overlay` writes the schema of profile overlay
files instead, where entries matched by `id` may be partial. Associate the
schema in your editor, for example with a `#:schema ./pycordis.schema.json`
comment at the top of a TOML file (Taplo / Even Better TOML) or `json.schemas`
in VS Code for JSON. A JSON (or YAML) file can also name it with a top-level
`"$schema": "./pycordis.schema.json"` key: the schema allows it, and the loader
ignores it and keeps it on write-back. From Python, `config_json_schema(ChatConfig)`
gives the schema of one dataclass:

```python
config_json_schema(ChatConfig) == {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ChatConfig",
    "type": "object",
    "properties": {"model": {"type": "string"}, "retries": {"type": "integer", "default": 3}},
    "required": ["model"],
    "additionalProperties": False,
}
```

`loader_json_schema(entries, allow=("*",), *, overlay=False, skipped=None)`
describes a whole parsed tree. `skipped(use, error)` is called for each plugin
that cannot be imported or whose schema cannot be built (default: a note on
stderr). The schema is slightly more lenient than the runtime: it cannot tell
`1.0` from `1`, and annotations the runtime refuses become `{}` with a
`description`.

## Command line

```sh
pycordis [start] [-c pycordis.toml] [--watch] [--allow PATTERN] [--log-level LEVEL]
                 [--profile NAME] [--dispose-timeout SECONDS]
                 [--health-port PORT [--health-host HOST]]
pycordis inspect [-c pycordis.toml] [--run] [--watch] [--allow PATTERN] [--log-level LEVEL]
                 [--profile NAME] [--dispose-timeout SECONDS]
pycordis doctor [-c pycordis.toml] [--timeout SECONDS] [--allow PATTERN] [--log-level LEVEL]
                [--profile NAME] [--dispose-timeout SECONDS]
pycordis schema [-c pycordis.toml] [--overlay] [-o schema.json] [--allow PATTERN]
                [--profile NAME]
python -m pycordis ...          # equivalent
```

`-c` is also `--config`, and `-o` is also `--output`. `LEVEL` is one of
`error`, `warn`, `info` or `debug`.

- Every subcommand takes `--profile NAME` (default: `$PYCORDIS_PROFILE`) to
  layer each config file with its overlay (see **Profiles** above).
- `start` (the default) creates the root context, installs the console exporter
  and the loader, loads the file and runs until SIGINT or SIGTERM, then
  disposes the root gracefully. A second signal exits immediately (130).
  `--allow` is repeatable and overrides `[loader] allow`; `--log-level` is one
  of `error`, `warn`, `info`, `debug`. `--dispose-timeout` (default `10`
  seconds, `0` disables) bounds the unload of every entry without its own
  `dispose_timeout`, so a hung (awaiting) disposer cannot block shutdown; one
  that blocks the event loop synchronously still can. `--health-port PORT`
  (`0` picks a free port) also serves `GET /healthz` (alive), `/readyz` (200
  once the loader has started and every enabled loader entry is ACTIVE, else
  503 with the reasons from diagnostics, error messages withheld) and `/inspect` (the fiber tree, no
  configs) on `127.0.0.1`, or on `--health-host` (another address logs a
  warning). See `pycordis.health.HealthServer` for the exact responses; it can
  also be loaded as a plugin.
- `inspect` is a static dry run: it parses the tree, imports the enabled
  plugins (honouring `allow`) and validates configs (after interpolation)
  without calling any plugin. Import, `allow` and config errors print `✗ <id>:
  ...` and exit 1. An injected service that no enabled entry declares with
  `provide` prints `⚠ <id>: cannot inject ...` and does **not** change the exit
  code, because services a plugin provides only from its body are invisible
  statically. `isolate` is honoured: a service provided only inside an
  isolated scope does not satisfy an injection outside it (or the reverse;
  equal labels share a scope) and is reported as `only provided in another
  isolated scope`.
- `inspect --run` starts the tree, waits for it to settle, prints the live
  fiber tree and disposes it. Like `doctor`, it then honours `--log-level`
  and `--dispose-timeout` as `start` does (`--watch` is accepted but has no
  lasting effect, as the tree is disposed at once).
- `doctor` starts the tree, waits for it to settle (`--timeout`, default 10
  seconds), prints the diagnosis of every fiber that is not `ACTIVE` (see
  **Diagnostics**) and disposes it. It exits 1 if anything is stuck or failed.
- `schema` prints the JSON Schema of the config file, or with `--overlay` of a
  profile overlay file; `-o` writes it to a file (see **JSON Schema**).
- Exit codes: `0` success (including a clean SIGTERM shutdown), `1` startup or
  config error (or, for `doctor`, a fiber that is not `ACTIVE`), `130` after SIGINT shutdown or a forced exit, `2` usage error.
- The `pycordis` console script **appends** the working directory to
  `sys.path`, so project plugins resolve but files there cannot shadow the
  standard library or installed packages. `python -m pycordis` does not get
  this protection: Python itself puts the working directory first on
  `sys.path` for `-m`.

## Security notes

- **Zero runtime dependencies.** `dependencies = []`: the wheel's metadata
  declares no requirements, so installing pycordis pulls in nothing else.
  PyYAML is used only if you install it yourself and load a YAML file.
- **Config files are code.** `use` imports and runs Python. Load only config
  (and profile overlays) you trust.
- **`allow` is not a sandbox.** It restricts which modules a config file may
  import (`[loader] allow = [...]` or `--allow`); the default `["*"]` allows
  everything. Patterns are case-sensitive `fnmatch` patterns matched against
  the *module name* before anything is imported; `pkg.*` also matches `pkg`
  itself. A reference whose attribute path walks into another module needs
  that module to be allowed too, and attribute segments starting with `_` are
  rejected, so a config cannot reach dunders or private names. But an allowed
  module's code runs with the full rights of the process, and an object it
  re-exports (a function imported into `my_agent`'s namespace) still resolves.
  For entry points the module is checked before loading; the attribute path
  comes from the installed package's metadata and is trusted at install time.
- **Health server** (`--health-port`, `HealthServer`): read-only and
  unauthenticated, bound to `127.0.0.1` unless you pass another host, which
  logs a warning. It serves three `GET` endpoints, accepts no request bodies,
  never echoes request data, and never serves configs or error messages. It
  serves at most 64 connections at once and drops a connection whose request
  head takes over 2 s (5 s for the whole exchange). A peer that can reach the
  port can still keep every slot busy and make probes fail, so never expose it
  to untrusted networks.
- **Secrets are re-templated on write-back.** Keep secrets in `${env:...}` /
  `${file:...}` placeholders. When the loader writes a config back, it puts the
  placeholder back wherever a value equals a secret it expanded, and it refuses
  a write that would still contain one (see the write-back note above).
  `${file:...}` reads only regular files, at most 1 MiB: a FIFO, device or
  directory is rejected without being opened.

## Performance

`uv run python bench/bench_core.py --repeat 5` (best of 5 runs), Python 3.14.6,
Apple M5 (macOS 27, arm64):

| Benchmark | Time |
| --- | --- |
| 1,000,000 `ctx.counter.value` reads, plugin context | 1.40 s (~716,000 reads/s) |
| 1,000,000 `ctx.counter.value` reads, root context | 1.09 s (~916,000 reads/s) |
| `emit` to 100 listeners × 10,000 | 0.134 s (~7.5 M listener calls/s) |
| `emit` to 100 listeners of 100 different plugins × 10,000 | 0.213 s (~4.7 M listener calls/s) |
| `emit` with a filtering receiver to 100 listeners × 10,000 | 0.199 s (~5.0 M listener calls/s) |
| Load 1,000 plugins | 0.040 s (~24,700 plugins/s) |
| Enter and exit 1,000 scopes | 0.181 s (~5,500 scopes/s) |
| Add and remove 10,000 collection items | 0.200 s (~50,000 items/s) |
| 1,000,000 `current_context()` calls | 0.017 s (~60 M calls/s) |
| Tear down a 1,000-deep dependency chain | 0.41 s |

`bench/bench_agent.py` (500 agent requests, each in its own scope, calling
tools that use `current_context()`) runs in 0.234 s.

A plugin-context read resolves `ctx.counter` (the `inject` check and fiber
walk; the `internal/get` waterfall runs only while it has listeners) and then
`.value` on the tracing proxy that lets a service see its caller's context.
Cache `service = ctx.counter` in hot loops.

Each listener runs with its owner as `current_context()`. Emitting to
listeners of many different plugins costs one context switch each time the
owner changes, about 80 ns per switch on this machine. Listeners of the same
owner that were registered one after another share a switch.

## Development

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest --cov=pycordis --cov-fail-under=95
```

## License

MIT. pycordis is a port of [cordis](https://github.com/cordiverse/cordis),
Copyright (c) 2021-present Shigma, also MIT; see [LICENSE](https://github.com/aswin-kevin/pycordis/blob/main/LICENSE).

See [CHANGELOG.md](https://github.com/aswin-kevin/pycordis/blob/main/CHANGELOG.md) for release notes and
[SECURITY.md](https://github.com/aswin-kevin/pycordis/blob/main/SECURITY.md) to report a vulnerability.
