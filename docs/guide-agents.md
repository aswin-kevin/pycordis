# Guide: an LLM agent on pycordis

This guide builds a small tool-using agent from pycordis parts:

- an LLM client **service** behind a typed `ServiceKey`;
- a tools **`Collection`** that plugins contribute to, whose items go away when
  the contributing plugin unloads;
- a **scope** per conversation, holding that conversation's memory;
- a **`RestartPolicy`** on the LLM client, which fails to connect once and
  recovers;
- tools that reach their conversation through **`current_context()`**;
- a **`HealthServer`** and **`diagnose()`** to explain a plugin that cannot start.

The LLM is a fake: deterministic, no network, no API key. Swap in a real client
by providing another `LLMClient` under the same key.

Every `python` block below runs, in order, as one program in
`tests/test_guide.py`, and its output must match the `text` block at the end. To
try it, paste the blocks into one file and run it with `python`.

## 1. Types and keys

The agent talks to its parts through three service keys. A key names a service
and carries its type, so `ctx.use(LLM)` is typed `LLMClient` and
`ctx.provide(LLM, value)` is checked against it.

```python
import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pycordis import (
    Backoff,
    Collection,
    CollectionChange,
    Context,
    Fiber,
    HealthServer,
    Logger,
    Message,
    RestartPolicy,
    Service,
    ServiceKey,
    current_context,
    plugin_meta,
    render_diagnosis,
)


@dataclass(frozen=True)
class ToolCall:
    """The model asks for ``tool`` to be run on ``argument``."""

    tool: str
    argument: str


class LLMClient(Protocol):
    def complete(self, prompt: str, tools: list[str]) -> ToolCall | str: ...


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    run: Callable[[str], str]


class Memory:
    """What one conversation remembers."""

    def __init__(self) -> None:
        self.notes: list[str] = []


LLM = ServiceKey[LLMClient]("llm")
TOOLS = ServiceKey[Collection[Tool]]("tools")
MEMORY = ServiceKey[Memory]("memory")
```

## 2. A flaky LLM client with a restart policy

`FakeLLM` is a `Service` provided as `LLM`. Its first connection attempt fails,
as a real endpoint might while it starts. The `RestartPolicy` in its plugin
metadata loads it again after a short backoff instead of leaving it `FAILED`.
Plugins and scopes that inject `LLM` simply wait until it is there.

```python
LLM_POLICY = RestartPolicy(
    mode="on-failure",
    max_restarts=3,  # within `window` seconds, then give up
    window=60.0,
    backoff=Backoff(initial=0.01, factor=2.0, max=1.0, jitter=0.0),
)


@plugin_meta(name="fake-llm", policy=LLM_POLICY)
class FakeLLM(Service[None], provide=LLM):
    """A scripted model: ``<tool> <argument>`` calls a tool, anything else is echoed."""

    connect_attempts = 0

    def __init__(self, ctx: Context, config: None = None) -> None:
        FakeLLM.connect_attempts += 1
        if FakeLLM.connect_attempts == 1:
            raise ConnectionError("model endpoint not reachable yet")
        super().__init__(ctx)

    def complete(self, prompt: str, tools: list[str]) -> ToolCall | str:
        if prompt.startswith("tool result: "):
            return f"Done. {prompt.removeprefix('tool result: ')}"
        command, _, argument = prompt.partition(" ")
        if command in tools:
            return ToolCall(command, argument)
        return f"You said: {prompt}"
```

## 3. Tools: a collection plugins contribute to

`Collection.of(TOOLS)` is a plugin providing an empty collection. Tool plugins
inject `TOOLS` and add to it. Each item is owned by the plugin that added it:
when that plugin unloads, its tools are removed (and the change is announced on
`Collection.changes(TOOLS)`).

The tool functions are plain functions with no `ctx` parameter. They find the
conversation they run in with `current_context()`, which is the scope context
while a conversation's task runs (step 4).

```python
def remember(text: str) -> str:
    memory = current_context().use(MEMORY)
    memory.notes.append(text)
    return f"noted ({len(memory.notes)} so far)"


def recall(_text: str) -> str:
    notes = current_context().use(MEMORY).notes
    return "; ".join(notes) if notes else "nothing yet"


def shout(text: str) -> str:
    return text.upper() + "!"


@plugin_meta(name="memory-tools", inject=[TOOLS])
def memory_tools(ctx: Context, config: None) -> None:
    tools = ctx.use(TOOLS)
    tools.add(Tool("remember", "store a note", remember), name="remember")
    tools.add(Tool("recall", "list the notes", recall), name="recall")


@plugin_meta(name="shout-tool", inject=[TOOLS])
def shout_tool(ctx: Context, config: None) -> None:
    ctx.use(TOOLS).add(Tool("shout", "shout the text", shout), name="shout")
```

## 4. A scope per conversation

`root.scope(...)` runs a block inside a short-lived plugin. It waits until the
scope's injections (`LLM`, `TOOLS`) are available, at most `timeout` seconds,
and disposes everything the block registered when it exits. `provide=[MEMORY]`
isolates the name `memory` to the scope, so each conversation has its own
`Memory` and concurrent conversations never see each other's.

`s.spawn(...)` runs each turn as a task owned by the scope, with the scope as
`current_context()`: that is how the tools above find this conversation's
memory.

```python
async def respond(prompt: str) -> str:
    """One agent turn: ask the model, run the tools it calls, return its answer."""
    ctx = current_context()
    llm = ctx.use(LLM)
    tools = ctx.use(TOOLS)
    answer = llm.complete(prompt, [tool.name for tool in tools])
    while isinstance(answer, ToolCall):
        tool = tools.get(answer.tool)
        result = tool.run(answer.argument) if tool else f"no tool {answer.tool!r}"
        answer = llm.complete(f"tool result: {result}", [])
    return answer


async def converse(root: Context, name: str, prompts: list[str]) -> None:
    async with root.scope(name=name, inject=[LLM, TOOLS], provide=[MEMORY], timeout=5) as s:
        s.provide(MEMORY, Memory())
        for prompt in prompts:
            reply = await s.spawn(respond(prompt))
            print(f"{name}> {prompt}")
            print(f"  {reply}")
```

## 5. Starting the application

The root starts the collection, the LLM client and the tool plugins. It does
not wait for the LLM: its first load fails and the supervisor restarts it,
while the conversations wait for it in their scopes. A small exporter prints
warnings and errors (the failed connection, the restart).

```python
class PrintProblems:
    """Print warnings and errors, one line each (no tracebacks)."""

    def export(self, message: Message) -> None:
        if message.type not in ("warn", "error"):
            return
        error = message.args[0] if message.args else None
        if isinstance(error, BaseException):
            text = f"{type(error).__name__}: {error}"
        else:
            text = Logger.format(self, message)
        print(f"[{message.type}] {message.name}: {text}")


async def start(root: Context) -> tuple[Fiber, Fiber]:
    root.logger.exporter(PrintProblems())
    await root.plugin(Collection.of(TOOLS))

    def on_tools_change(change: CollectionChange[Tool]) -> None:
        print(f"tools: {change.kind} {change.name}")

    root.on(Collection.changes(TOOLS), on_tools_change)
    llm = root.plugin(FakeLLM)  # not awaited: its first load fails
    await root.plugin(memory_tools)
    shouting = await root.plugin(shout_tool)
    return llm, shouting
```

## 6. Health checks and diagnostics

`HealthServer` serves `/healthz`, `/readyz` and `/inspect` on `127.0.0.1`
(`port=0` picks a free port; the CLI equivalent is `pycordis start
--health-port PORT`). With `ready_requires="all"`, `/readyz` is ready while
every fiber is `ACTIVE`.

Then a plugin injects `search_api`, which nobody provides. It stays `PENDING`,
`/readyz` turns `503` and lists why, and `root.diagnose()` explains it in full
(`pycordis doctor -c app.toml` prints the same diagnosis for a config file).

```python
async def http_get(port: int, path: str) -> tuple[int, object]:
    """A minimal HTTP/1.1 GET returning the status and the decoded JSON body."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
    await writer.drain()
    response = await reader.read()  # the server closes after one response
    writer.close()
    await writer.wait_closed()
    head, _, body = response.partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(body)


@plugin_meta(name="web-search", inject=["search_api"])
def web_search(ctx: Context, config: None) -> None:
    ctx.logger.info("searching with %s", ctx.search_api)


async def check_health(root: Context) -> None:
    await root.plugin(HealthServer, HealthServer.Config(port=0, ready_requires="all"))
    server: HealthServer = root.health
    print("readyz:", await http_get(server.port, "/readyz"))

    root.plugin(web_search)  # waits for "search_api" forever
    status, body = await http_get(server.port, "/readyz")
    print("readyz:", status, json.dumps(body, indent=2))

    stuck = root.diagnose().unhealthy()  # only what is not ACTIVE, with its ancestors
    assert stuck is not None
    print(render_diagnosis(stuck))
```

## 7. Putting it together

```python
async def main() -> None:
    async with Context() as root:
        llm, shouting = await start(root)

        await converse(root, "alice", ["remember buy milk", "remember call bob", "recall"])
        await converse(root, "bob", ["recall", "shout hello", "how are you?"])
        stats = llm.stats
        print(f"llm: {llm.state.name}, {stats.failures} failure, {stats.restarts} restart")

        await shouting.dispose()  # its "shout" tool goes with it
        print("tools:", [tool.name for tool in root.use(TOOLS)])

        await check_health(root)


asyncio.run(main())
```

Output:

```text
[error] fake-llm: ConnectionError: model endpoint not reachable yet
tools: add remember
tools: add recall
[warn] supervisor: restarting fake-llm in 0.01s (attempt 1/3)
tools: add shout
alice> remember buy milk
  Done. noted (1 so far)
alice> remember call bob
  Done. noted (2 so far)
alice> recall
  Done. buy milk; call bob
bob> recall
  Done. nothing yet
bob> shout hello
  Done. HELLO!
bob> how are you?
  You said: how are you?
llm: ACTIVE, 1 failure, 1 restart
tools: remove shout
tools: ['remember', 'recall']
readyz: (200, {'status': 'ready'})
readyz: 503 {
  "status": "not-ready",
  "reasons": [
    {
      "fiber": "web-search",
      "uid": 8,
      "state": "PENDING",
      "kind": "missing-service",
      "service": "search_api",
      "detail": "no plugin declares or currently provides \"search_api\""
    }
  ]
}
root #0 ACTIVE
└─ web-search #8 PENDING
      missing-service: no plugin declares or currently provides "search_api"
```

What to notice:

- The LLM failed once and was restarted by its policy; the first conversation
  just waited for it. `llm.stats` counts the failure and the restart.
- `alice` and `bob` each got their own `Memory`: `bob`'s `recall` finds nothing.
- Disposing `shout-tool` removed its tool from the collection, and the change
  was announced; the `memory-tools` items stayed.
- `/readyz` reports the stuck plugin without error texts; `diagnose()` and
  `pycordis doctor` give the full explanation.
