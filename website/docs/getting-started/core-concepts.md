---
title: Core Concepts
---

# Core Concepts

Six concepts cover everything CubePi does. Read this page once, then
the rest of the docs become a lookup table.

## Agent

`Agent` is the stateful façade: you construct it with a provider, a
model, optional tools, and optional middleware/checkpointer. You drive
it through three methods:

- `await agent.prompt(message)` — start a new turn from a user message.
- `await agent.resume()` — continue from the last persisted message
  (used with a checkpointer).
- `agent.steer(message)` / `agent.follow_up(message)` — queue a
  message mid-flight or after the current run.

The agent owns an `AgentState` (system prompt, tools, model, message
history, pending tool calls, streaming flag) and a list of subscribers:

```python
unsubscribe = agent.subscribe(my_listener)
# ...
unsubscribe()
```

Subscribers receive every `AgentEvent` the loop emits. They can be
sync or async.

## Tool

An `AgentTool` is a name + description + Pydantic parameter model +
async `execute` function:

```python
from pydantic import BaseModel
from cubepi import AgentTool, AgentToolResult, TextContent

class SearchParams(BaseModel):
    query: str
    limit: int = 10

async def execute(tool_call_id, params: SearchParams, *, signal=None, on_update=None):
    # do work; respect `signal` if you can be aborted
    return AgentToolResult(content=[TextContent(text=f"…")])

search = AgentTool(
    name="search",
    description="Search the corpus",
    parameters=SearchParams,
    execute=execute,
)
```

The Pydantic schema is auto-converted to JSON Schema and passed to the
model. Arg parsing, error wrapping, and parallel execution are
handled by the framework. See [Tool Use](../guides/agents/tool-use)
for `execution_mode`, `on_update` (incremental progress), and
`terminate` (end the turn from a tool).

## Provider

A `Provider` is anything matching this Protocol:

```python
class Provider(Protocol):
    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream: ...
```

It returns a `MessageStream` — a single async iterator that yields
`StreamEvent`s and exposes the final `AssistantMessage` via `await
stream.result()`. Built-in providers:

- `AnthropicProvider` — Claude (Messages API, with thinking, caching,
  tool use).
- `OpenAIProvider` — GPT family (Chat Completions API).
- `OpenAIResponsesProvider` — GPT family (Responses API, server-side
  state).
- `FauxProvider` — deterministic test double (no network).

Write your own by implementing one method. See [Providers / Custom](../guides/providers/custom).

## Stream and events

Streams and events are two layers:

- **Provider streams** — `MessageStream` yields *provider* events:
  `start`, `text_start`, `text_delta`, `text_end`, `thinking_*`,
  `toolcall_*`, `done`, `error`. This is the raw token stream.
- **Agent events** — what `agent.subscribe(...)` receives. Eleven
  types covering the entire loop: `agent_start`, `agent_end`,
  `turn_start`, `turn_end`, `message_start`, `message_update`,
  `message_end`, `tool_execution_start`, `tool_execution_update`,
  `tool_execution_end`. `message_update` wraps the provider event
  inside `event.stream_event`.

Subscribe to agent events for UI; for low-level token routing dig into
`event.stream_event`. See [Streaming Events](../guides/agents/streaming).

## Middleware

`Middleware` is a class with up to seven typed hooks:

| Hook | When it runs | Composition rule |
|---|---|---|
| `transform_context` | Before each model call, on the message list | Chained — each receives previous result |
| `convert_to_llm` | Right before serialisation to the provider | Last implementation wins |
| `transform_system_prompt` | Before each model call, on the system prompt | Chained |
| `before_tool_call` | Per tool call, after arg validation | First `block=True` short-circuits |
| `after_tool_call` | Per tool call, after `execute` | Later override earlier |
| `after_model_response` | After the assistant message lands | Returns a `TurnAction` controlling flow |
| `should_stop_after_turn` | At each turn boundary | Any `True` stops |

Pass middleware as a list to `Agent(middleware=[...])`. See
[Middleware → Composition](../guides/middleware/composition).

## Checkpointer

A `Checkpointer` is anything matching:

```python
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Message]) -> None: ...
    async def save_extra(self, thread_id: str, extra: dict) -> None: ...
```

Bind one to an agent with `Agent(checkpointer=cp, thread_id="…")` and
the loop will append each new message as it lands, restoring history
on the first `prompt()`. Built-in backends: `MemoryCheckpointer`,
`SQLiteCheckpointer`, `PostgresCheckpointer`. See
[Checkpointing → SQLite](../guides/checkpointing/sqlite).

## Tracer (optional)

`Tracer` produces OpenTelemetry spans aligned with the
[GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/),
so any OTLP backend (Jaeger, Tempo, Honeycomb, Datadog, AWS X-Ray, …)
can ingest agent runs without custom instrumentation. Install the
extra:

```bash
pip install "cubepi[tracing]"           # OTel SDK
pip install "cubepi[tracing-otlp]"      # + OTLP/HTTP exporter
```

then wrap your agent in an `async with`:

```python
from cubepi.tracing import Tracer
from cubepi.tracing.exporters import JsonlSpanExporter

async with (
    Tracer(
        service_name="my-bot",
        agent_name="assistant",
        exporters=[JsonlSpanExporter(directory="./cubepi-traces")],
    ) as tracer,
    tracer.attached(agent),
):
    await agent.prompt("…")
```

Each run emits an `invoke_agent` root span containing one
`cubepi.turn` per LLM round-trip, plus `chat` (CLIENT) and
`execute_tool` children. By default **no prompt content or model
output is recorded** — opt in with `Tracer(record_content=True)` and
a `redact` callback for PII. Pair with `Meter(...)` for token /
duration / TTFC histograms. Full guide:
[Tracing → Overview](../guides/tracing/overview).

## Putting it together

```
User code
   │
   ▼
┌──────────────────────────────────────────┐
│ Agent                                     │
│  ├─ AgentState (messages, tools, …)       │
│  ├─ Middleware ── compose_middleware()    │
│  ├─ Checkpointer ── append on message_end │
│  └─ run_agent_loop  ◀──── the actual loop │
│       │                                   │
│       ▼                                   │
│  Provider.stream() → MessageStream        │
│       │                                   │
│       └─ events → emit → subscribers      │
└──────────────────────────────────────────┘
```

That diagram is the whole framework. The rest of this site is just
the details.
