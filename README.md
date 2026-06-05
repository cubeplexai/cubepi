<p align="center">
  <img src="https://raw.githubusercontent.com/cubeplexai/cubepi/main/website/static/img/brand/cubepi-social-preview.png" alt="CubePi" width="800">
</p>

[![CI](https://github.com/cubeplexai/cubepi/actions/workflows/ci.yml/badge.svg)](https://github.com/cubeplexai/cubepi/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cubeplexai/cubepi/graph/badge.svg)](https://codecov.io/gh/cubeplexai/cubepi)
[![PyPI](https://img.shields.io/pypi/v/cubepi)](https://pypi.org/project/cubepi/)
[![Python](https://img.shields.io/pypi/pyversions/cubepi)](https://pypi.org/project/cubepi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-cubepi.ai-blue)](https://cubepi.ai)
[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-1f6feb)](https://deepwiki.com/cubeplexai/cubepi)
[![skills.sh](https://skills.sh/b/cubeplexai/cubepi)](https://skills.sh/cubeplexai/cubepi)

CubePi is a Pythonic, async-native agent framework designed for high performance, readability, and production-grade persistence. It provides a leaner alternative to graph-based agent runtimes by modeling agent logic as a linear while loop that developers can easily trace and debug.

## Why CubePi

| | langgraph | CubePi |
|---|---|---|
| **Abstraction** | Graph nodes + edges + channels — you model your agent as a state machine | Plain async functions — `run_agent_loop` is a while loop you can read in 5 minutes |
| **Streaming** | Callback-based, multiple handler types | `async for event in stream` — one pattern everywhere |
| **Checkpointing** | Full snapshot per step — serializes entire message list on every channel change | Append-only — writes only new messages, O(1) DB I/O regardless of conversation length |
| **Dependencies** | Pulls in langchain-core, langgraph-sdk, and transitive deps | 3 core deps: `pydantic`, `anthropic`, `openai` |
| **Tool execution** | Tools are graph nodes with manual wiring | Declare tools as functions, framework handles routing and parallel execution |
| **Multi-provider** | Via langchain chat model adapters | Native `Provider` protocol — Anthropic, OpenAI built in, add your own with one class |
| **Middleware** | Graph-level middleware on node entry/exit | Agent-level middleware with 8 typed hooks and declarative composition rules |
| **Observability** | LangSmith / Langfuse integration, full trace visualization | Native OpenTelemetry — `Tracer`, `Meter`, GenAI semconv, OTLP / JSONL exporters built in |

## Install

```bash
pip install cubepi

# Optional extras
pip install cubepi[sqlite]     # SQLite checkpointer
pip install cubepi[postgres]   # Postgres checkpointer
pip install cubepi[mysql]      # MySQL checkpointer
pip install cubepi[mcp]        # MCP tool loaders
pip install cubepi[tracing]    # OpenTelemetry tracing + metrics
pip install cubepi[tracing-otlp]  # Adds the OTLP/HTTP span exporter
pip install cubepi[trace-cli]  # `cubepi trace` terminal viewer
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add cubepi
uv add cubepi[sqlite,postgres,mysql,mcp,tracing]
```

## Quick Start

```python
import asyncio
from pydantic import BaseModel
from cubepi import Agent, AgentTool, Model
from cubepi.agent.types import AgentToolResult
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.base import TextContent

provider = AnthropicProvider(provider_id="anthropic", api_key="sk-...")

class GetWeatherParams(BaseModel):
    city: str

async def get_weather(tool_call_id, params: GetWeatherParams, *, signal=None, on_update=None):
    return AgentToolResult(
        content=[TextContent(text=f"72°F and sunny in {params.city}")]
    )

agent = Agent(
    model=provider.model("claude-sonnet-4-5-20250929"),
    tools=[
        AgentTool(
            name="get_weather",
            description="Get current weather for a city",
            parameters=GetWeatherParams,
            execute=get_weather,
        ),
    ],
    system_prompt="You are a helpful weather assistant.",
)

def on_event(event, signal=None):
    if event.type == "text_delta":
        print(event.delta, end="", flush=True)

agent.subscribe(on_event)
asyncio.run(agent.prompt("What's the weather in Tokyo?"))
```

For a guided tour of the architecture, browse the
[DeepWiki for this repo](https://deepwiki.com/cubeplexai/cubepi) or the
[Core Concepts guide](https://cubepi.ai/docs/getting-started/core-concepts).

## Core Concepts

### Providers

Abstract LLM interaction behind a `Provider` protocol. All providers return `MessageStream` — an async iterator of `StreamEvent`s.

```python
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.openai import OpenAIProvider
from cubepi.providers import FauxProvider

# Real providers
anthropic = AnthropicProvider(provider_id="anthropic", api_key="...")
openai = OpenAIProvider(provider_id="openai", api_key="...")

# Test provider — no API calls, fully deterministic
faux = FauxProvider(provider_id="faux")
faux.set_responses(["Hello!", "How can I help?"])
```

### Tools

Declare tools with a name, a Pydantic model for parameters, and an async `execute` returning `AgentToolResult`. The framework handles JSON Schema derivation, argument parsing, parallel execution, and error wrapping.

```python
from pydantic import BaseModel
from cubepi import AgentTool
from cubepi.agent.types import AgentToolResult
from cubepi.providers.base import TextContent

class SearchParams(BaseModel):
    query: str

async def execute(tool_call_id, params: SearchParams, *, signal=None, on_update=None):
    return AgentToolResult(content=[TextContent(text=f"Results for: {params.query}")])

tool = AgentTool(
    name="search",
    description="Search the web",
    parameters=SearchParams,
    execute=execute,
    execution_mode="parallel",  # or "sequential"
)
```

### Middleware

Composable hooks that modify behavior without touching the core loop:

```python
from cubepi import Middleware, compose_middleware
from cubepi.agent.types import BeforeToolCallResult

class LoggingMiddleware(Middleware):
    async def transform_context(self, messages, *, ctx, signal=None):
        print(f"Context has {len(messages)} messages")
        return messages

class SafetyMiddleware(Middleware):
    async def before_tool_call(self, ctx, *, signal=None):
        if ctx.tool_call.name == "dangerous_tool":
            return BeforeToolCallResult(block=True, reason="Blocked by policy")
        return None

hooks = compose_middleware([LoggingMiddleware(), SafetyMiddleware()])
```

**Composition rules:**

| Hook | Rule |
|------|------|
| `transform_context` | Chained — each receives previous result |
| `convert_to_llm` | Last implementation wins |
| `before_tool_call` | Any block stops execution |
| `after_tool_call` | Later overrides earlier |
| `transform_system_prompt` | Chained — each receives previous result |
| `after_model_response` | Returns `TurnAction`; last decision wins, messages concatenate |
| `should_stop_after_turn` | Any true stops |
| `on_run_end` | Messages concatenate; non-empty result triggers one extra model turn |

### Checkpointer

Persist conversation state with append-only semantics:

```python
from cubepi.checkpointer import (
    MemoryCheckpointer,
    SQLiteCheckpointer,
    PostgresCheckpointer,
    MySQLCheckpointer,
)

# In-memory for dev/test
cp = MemoryCheckpointer()

# SQLite for lightweight persistence
async with SQLiteCheckpointer("agent.db") as cp:
    agent = Agent(model=model, checkpointer=cp, thread_id="conv-1")

# Postgres for production
async with PostgresCheckpointer("postgresql://...") as cp:
    agent = Agent(model=model, checkpointer=cp, thread_id="conv-1")

# MySQL for production
async with MySQLCheckpointer("mysql://...") as cp:
    agent = Agent(model=model, checkpointer=cp, thread_id="conv-1")
```

Postgres and MySQL never issue DDL at runtime — your app owns the schema via
Alembic. See the host-integration runbooks
([Postgres](cubepi/checkpointer/postgres/README.md) ·
[MySQL](cubepi/checkpointer/mysql/README.md)) and the runnable
[`examples/`](examples/).

### FauxProvider for Testing

Ship your agent tests without API keys:

```python
from cubepi.providers import FauxProvider, faux_text, faux_tool_call, faux_assistant_message

provider = FauxProvider(provider_id="faux")
provider.set_responses([
    faux_assistant_message([
        faux_tool_call("search", {"query": "python"}),
    ]),
    faux_assistant_message("Here are the results..."),
])

agent = Agent(model=provider.model("test"), tools=[search_tool])
agent.subscribe(lambda event, signal=None: None)  # subscribe before prompt to receive events
await agent.prompt("Search for python")
# Streams realistic deltas — content_block_start, text_delta, etc.
```

### Tracing

Attach a `Tracer` and every agent run produces OpenTelemetry spans
aligned with the [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —
ingestible by Jaeger, Tempo, Honeycomb, Datadog, AWS X-Ray, or any
OTLP-compatible backend without custom instrumentation:

```python
from cubepi.tracing import Tracer, tracing_context
from cubepi.tracing.exporters import JsonlSpanExporter

async with (
    Tracer(
        service_name="my-bot",
        agent_name="assistant",
        exporters=[JsonlSpanExporter(directory="./cubepi-traces")],
    ) as tracer,
    tracer.attached(agent),
):
    with tracing_context(tags=["beta-arm"], metadata={"user_id": "u-42"}):
        await agent.prompt("Hello.")
# On exit: detach (closes any cancelled-run spans + flush) + tracer shutdown.
```

Span tree per run:

```
trace
└── invoke_agent  14425.8ms  [0x1cd97cdb]         ← one per agent.prompt()
    ├── cubepi.turn  1283.1ms  [0x5cfda93e]        ← one per LLM round-trip
    │   ├── chat deepseek-v4-flash  1208.7ms  tok 6845/68  [0x0d130229]
    │   └── execute_tool subagent  9610.2ms  subagent  [0x38bdd10a]
    │       └── invoke_agent  9601.0ms  [0x8094f99b]   ← subagent run, nested
    │           └── cubepi.turn  9598.4ms  [0x57c5cfc7]
    │               ├── chat deepseek-v4-flash  1190.3ms  [0x8205ca6b]
    │               └── execute_tool web_search  6500.2ms  web_search  [0xca4e59fc]
    └── cubepi.turn  491.9ms  ERROR  [0xce25f242]
        └── chat deepseek-v4-flash  427.2ms  ERROR  [0x0bff68ec]
            └── error: Error code: 400 - ... `tool_use` ids were found without
                `tool_result` blocks immediately after: call_01_...
```

No prompts / model outputs are recorded by default. Opt in with
`Tracer(record_content=True)` plus a `redact` callback for PII. Pair
with `Meter(...)` for `gen_ai.client.operation.duration` / TTFC /
token-usage histograms. Full guide: https://cubepi.ai/docs/guides/tracing/overview

#### Inspecting traces from the terminal

With `JsonlSpanExporter` writing to `./cubepi-traces`, inspect runs with the
`cubepi trace` CLI (install the extra: `pip install cubepi[trace-cli]`). All
subcommands take `--dir` (default `./cubepi-traces`):

```bash
cubepi trace ls                 # recent runs, newest first; the `input`
                                #   column shows the user message + `status`
cubepi trace view <run_id>      # render a run as a tree; errors print inline
                                #   under the failing span (no flag needed).
                                #   A unique run-id PREFIX is enough.
cubepi trace view <run> --content   # also expand prompts / tool args / results
cubepi trace view <run> -v          # expand ALL span attributes (verbose)
cubepi trace follow <run_id>    # stream spans live as they complete
cubepi trace stats --by model   # token / latency / error aggregates
cubepi trace stats --by tool --since 2026-01-01
```

Typical debugging flow: `ls` (find the run by its `input`), then
`view <prefix>` and read the inline `error:` line under any `ERROR` span. Need
content only recorded with `Tracer(record_content=True)`.

**Token / cache fields.** The recorder reconciles to the GenAI semconv, so
`gen_ai.usage.input_tokens` is the **inclusive** total prompt
(`input + cache_read + cache_creation`) and `gen_ai.usage.cache_read.input_tokens`
is a subset of it. From trace fields, cache hit rate is
`cache_read / input_tokens` (≤ 100%) — do **not** add `cache_read` to the
denominator.

Coding agents debugging cubepi/consumer apps can install the
[`cubepi-trace` skill](skills/cubepi-trace/SKILL.md):

```bash
npx skills add cubeplexai/cubepi@cubepi-trace -a claude-code
```

## AI Agents

Two skills are available for coding agents (Claude Code, Cursor, Codex, …) working
with this repo:

| Skill | Install | Purpose |
|-------|---------|---------|
| `cubepi` | `npx skills add cubeplexai/cubepi@cubepi -a claude-code` | Build agents — API reference, tools, middleware, checkpointing, MCP, HITL |
| `cubepi-trace` | `npx skills add cubeplexai/cubepi@cubepi-trace -a claude-code` | Debug runs — inspect OTel spans, token counts, tool results, streaming failures |

## Requirements

- Python >= 3.11
- Core: `pydantic`, `anthropic`, `openai`
- Optional: `aiosqlite` (`[sqlite]`), `asyncpg` + `sqlalchemy` + `msgpack` (`[postgres]`), `aiomysql` + `sqlalchemy` + `msgpack` + `cryptography` (`[mysql]`), `mcp` (`[mcp]`), `opentelemetry-sdk` (`[tracing]`), `opentelemetry-exporter-otlp-proto-http` (`[tracing-otlp]`), `rich` (`[trace-cli]`)

## Credits

Architecture inspired by pi-agent-core (TypeScript); CubePi is an independent Python reimplementation with Pydantic v2, asyncio-native primitives, and built-in checkpointing.

## License

MIT
