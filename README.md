<p align="center">
  <img src="https://raw.githubusercontent.com/aswin-kevin/pycordis/main/docs/assets/architecture.jpg" alt="pycordis: a root Context connecting LLM, Vector DB, Memory, Tools and connector plugins, with per-user scopes, auto-restart, hot-reloaded agent.toml and health checks" width="720">
</p>

<p align="center">
  <b>The plugin runtime for Python AI agents.</b><br>
  Your model clients, tools, memory and connectors become plugins that<br>
  start when they're ready, restart when they fail, and clean up after themselves.
</p>

<p align="center">
  <a href="https://pypi.org/project/pycordis-framework/"><img src="https://img.shields.io/pypi/v/pycordis-framework?color=6f42c1" alt="PyPI"></a>
  <a href="https://pypi.org/project/pycordis-framework/"><img src="https://img.shields.io/pypi/pyversions/pycordis-framework" alt="Python"></a>
  <img src="https://img.shields.io/badge/dependencies-0-brightgreen" alt="Zero dependencies">
  <img src="https://img.shields.io/badge/typing-strict-blue" alt="Strictly typed">
  <a href="https://github.com/aswin-kevin/pycordis/actions/workflows/ci.yml"><img src="https://github.com/aswin-kevin/pycordis/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/aswin-kevin/pycordis/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT"></a>
</p>

<p align="center">
  <a href="https://github.com/aswin-kevin/pycordis#quickstart">Quickstart</a> ·
  <a href="https://github.com/aswin-kevin/pycordis#a-real-agent">A real agent</a> ·
  <a href="https://github.com/aswin-kevin/pycordis/blob/main/docs/guide-agents.md">Agent guide</a> ·
  <a href="https://github.com/aswin-kevin/pycordis/blob/main/docs/reference.md">Reference</a>
</p>

pycordis is **not** another LLM framework. It sits *underneath* the one you
already use (the OpenAI or Anthropic SDK, PydanticAI, LangGraph…) and runs
everything around it. Think FastAPI's `Depends`, plus a process supervisor, for
long-running async apps.

```sh
pip install pycordis-framework      # Python 3.12+, zero dependencies
```

## The problem

Every real agent's `main()` ends up like this:

```python
llm = AsyncAnthropic()  # crashes if the key is missing
db = await VectorDB.connect(url)  # must come before search_docs
tools = [search_docs(db)]
if SLACK_TOKEN:  # optional tool, ad-hoc check
    tools.append(slack_tool(SLACK_TOKEN))
memories: dict[str, Memory] = {}  # global, shared by every user
try:
    await serve(llm, tools, memories)
finally:
    await db.close()  # and the Slack client? the scheduler?
```

Startup order is maintained by hand. One flaky dependency kills the whole
process. Per-user state is a global dict. And when a tool is missing, you get to
ask *"why didn't it load?"*

With pycordis, each piece is a plugin that declares what it **provides** and what
it **needs**. The runtime handles the order, the restarts, the isolation and the
cleanup.

## Concepts in 30 seconds

| Term | What it is |
| --- | --- |
| **Plugin** | A function (or class) that sets something up. Unloading it undoes everything it did. |
| **Service** | A value a plugin `provide`s under a typed `ServiceKey`, like your LLM client. |
| **inject** | The services a plugin needs. It starts when they all exist, and stops if one goes away. |
| **Scope** | A child context, such as one per conversation, with its own services. Torn down on exit. |
| **Collection** | A registry, like your tools, that many plugins add to. Items leave with their plugin. |

## Quickstart

```python
import asyncio
from collections.abc import Callable

from pycordis import Context, ServiceKey, plugin_meta

DB = ServiceKey[dict[str, str]]("db")


@plugin_meta(name="search", inject=[DB])  # needs DB
def search(ctx: Context, config: None) -> Callable[[], None]:
    print(f"search: online, {len(ctx.use(DB))} docs")
    return lambda: print("search: offline")  # cleanup, run on unload


@plugin_meta(name="db", provide=DB)  # provides DB
async def db(ctx: Context, config: None) -> None:
    await asyncio.sleep(0.1)  # connecting...
    ctx.provide(DB, {"intro": "hello"})


async def main() -> None:
    async with Context() as app:
        app.plugin(search)  # registered first: it simply waits for DB
        conn = await app.plugin(db)
        await conn.dispose()  # pull the DB out from under it


asyncio.run(main())
```

```text
search: online, 1 docs
search: offline
```

`search` started the moment `DB` existed, and shut itself down the moment it
went away. No ordering in `main()`, no `try/finally`, no globals.

## A real agent

A model client with a restart policy, a tool registry you can change while it runs, and private memory per user:

```python
import asyncio
from collections.abc import Callable

from pycordis import Collection, Context, RestartPolicy, ServiceKey
from pycordis import current_context, plugin_meta


class Model:  # stand-in: use AsyncOpenAI, AsyncAnthropic, Ollama...
    async def reply(self, text: str, tools: list[str]) -> str:
        return f"echo {text!r} · tools: {tools}"


LLM = ServiceKey[Model]("llm")
TOOLS = ServiceKey[Collection[Callable[[str], str]]]("tools")
MEMORY = ServiceKey[list[str]]("memory")


@plugin_meta(name="llm", provide=LLM, policy=RestartPolicy(mode="on-failure"))
async def llm(ctx: Context, config: None) -> None:
    ctx.provide(LLM, Model())  # if this raises, pycordis retries with backoff


@plugin_meta(name="web-search", inject=[TOOLS])
def web_search(ctx: Context, config: None) -> None:
    ctx.use(TOOLS).add(lambda query: f"results for {query}", name="web_search")


async def turn(text: str) -> str:
    chat = current_context()  # the conversation this turn belongs to
    chat.use(MEMORY).append(text)
    tools = [name for name, _ in chat.use(TOOLS).items()]
    return await chat.use(LLM).reply(text, tools)


async def main() -> None:
    async with Context() as app:
        await app.plugin(Collection.of(TOOLS))
        app.plugin(llm)  # no await: it starts in the background
        search = await app.plugin(web_search)  # await: wait until it runs

        chat = app.scope(name="alice", inject=[LLM, TOOLS], provide=[MEMORY])
        async with chat as alice:
            alice.provide(MEMORY, [])  # alice's own memory
            print(await alice.spawn(turn("hi")))  # spawn = run inside alice's scope
            await search.dispose()  # unplug a tool, live
            print(await alice.spawn(turn("still there?")))
            print("memory:", alice.use(MEMORY))


asyncio.run(main())
```

```text
echo 'hi' · tools: ['web_search']
echo 'still there?' · tools: []
memory: ['hi', 'still there?']
```

The tool vanished when its plugin unloaded, and nobody cleaned anything up. The
[agent guide](https://github.com/aswin-kevin/pycordis/blob/main/docs/guide-agents.md)
builds the full version: a flaky LLM that recovers, two users, health checks and
diagnostics.

## Why pycordis

- 🔌 **Start in any order.** A plugin runs as soon as what it needs exists.
- ♻️ **Self-healing.** Failing plugins restart with backoff, dependents wait instead of crashing, and `load_timeout` catches hangs.
- 💬 **Per-user isolation.** `app.scope()` gives each conversation its own services, torn down even after an error.
- 🧰 **Live tool registries.** With `Collection[T]`, tools appear and disappear with their plugin.
- 🩺 **"Why isn't it running?"** `pycordis doctor` names the missing service, the failed plugin or the dependency cycle.
- ⚙️ **Config files with hot reload.** TOML or JSON (YAML with PyYAML), `${env:KEY}` secrets, dev/prod profiles.
- 📈 **Ops built in.** `/healthz`, `/readyz` and `/inspect` endpoints, lifecycle telemetry, JSON logs.
- 🎯 **Typed, zero dependencies.** `ctx.use(LLM)` is a `Model` to your type checker. Pure standard library, nothing extra in your supply chain.

## Run it from a config file

Plugins take typed config:

```python
# my_agent.py
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from pycordis import Context, ServiceKey, plugin_meta

LLM = ServiceKey[AsyncAnthropic]("llm")


@dataclass
class LLMConfig:
    api_key: str


@plugin_meta(name="llm", provide=LLM, Config=LLMConfig)
def llm(ctx: Context, config: LLMConfig) -> None:
    ctx.provide(LLM, AsyncAnthropic(api_key=config.api_key))


# needs a "slack_client" service that nothing provides yet: see `pycordis doctor` below
@plugin_meta(name="slack", inject=["slack_client"])
def slack(ctx: Context, config: None) -> None: ...
```

The config file decides what runs:

```toml
# agent.toml
[loader]
watch = true                    # edit while it runs: only what changed reloads

[[plugins]]
id = "llm"
use = "my_agent:llm"
config = { api_key = "${env:ANTHROPIC_API_KEY}" }     # checked against LLMConfig
restart = { mode = "on-failure", max_restarts = 5 }   # self-healing

[[plugins]]
id = "slack"
use = "my_agent:slack"
disabled = false                # true: kept in the file, not running
```

Start it, with health endpoints for your orchestrator:

```sh
pycordis start -c agent.toml --health-port 8080    # GET /healthz /readyz /inspect
# in a container, add --health-host 0.0.0.0 so the orchestrator's probes reach it
```

If something doesn't start, ask why:

```text
$ pycordis doctor -c agent.toml
root #0 ACTIVE
└─ slack #2 [entry slack] PENDING
      missing-service: no plugin declares or currently provides "slack_client"
```

## How it compares

| | pycordis | hand-written `main()` | FastAPI lifespan + `Depends` |
| --- | :---: | :---: | :---: |
| Start order from dependencies | ✅ | by hand | by hand |
| Restart a failed part while the rest keeps running | ✅ | ❌ | ❌ |
| Add or remove parts while running | ✅ | ❌ | ❌ |
| Per-conversation state with cleanup | ✅ | by hand | per request |
| Explains why something didn't start | ✅ | ❌ | ❌ |

pycordis works *with* agent frameworks, not against them: it owns the
lifecycles, and your framework owns the reasoning.

## FAQ

**Does it replace LangChain, LangGraph or PydanticAI?** No. Use them inside
plugins. pycordis manages their clients, tools and lifetimes.

**Do I need the config file or the CLI?** No. Everything works from plain
Python. The loader and `pycordis` command are optional.

**Is it production-ready?** It's 0.1 (beta): strictly typed, 1,250+ tests and 98%
coverage, but the API may still change before 1.0.

**What's cordis?** The TypeScript plugin framework behind
[Koishi](https://koishi.chat). pycordis is a faithful asyncio port of
[cordis](https://github.com/cordiverse/cordis) 4.1.0, with the same lifecycle,
events and error messages. You don't need to know it. If you do, the
[mapping table](https://github.com/aswin-kevin/pycordis/blob/main/docs/cordis-mapping.md)
lists every API side by side.

## Learn more

- 🤖 **[Build an agent](https://github.com/aswin-kevin/pycordis/blob/main/docs/guide-agents.md)**: an LLM agent end to end.
- 📖 **[Reference](https://github.com/aswin-kevin/pycordis/blob/main/docs/reference.md)**: every feature in depth.
- 🔁 **[cordis → pycordis](https://github.com/aswin-kevin/pycordis/blob/main/docs/cordis-mapping.md)**: the API mapping.
- 📝 **[Changelog](https://github.com/aswin-kevin/pycordis/blob/main/CHANGELOG.md)** · 🔐 **[Security](https://github.com/aswin-kevin/pycordis/blob/main/SECURITY.md)**

## License

MIT. pycordis is a port of [cordis](https://github.com/cordiverse/cordis)
(MIT, © 2021-present Shigma).
