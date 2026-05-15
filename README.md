<p align="center">
  <img src="https://raw.githubusercontent.com/cubeplexai/cubepi/main/assets/brand/cubepi-logo.svg" alt="cubepi logo" width="160">
</p>

<h1 align="center">cubepi</h1>

[![CI](https://github.com/cubeplexai/cubepi/actions/workflows/ci.yml/badge.svg)](https://github.com/cubeplexai/cubepi/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cubeplexai/cubepi/graph/badge.svg)](https://codecov.io/gh/cubeplexai/cubepi)
[![PyPI](https://img.shields.io/pypi/v/cubepi)](https://pypi.org/project/cubepi/)
[![Python](https://img.shields.io/pypi/pyversions/cubepi)](https://pypi.org/project/cubepi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Pythonic async-native agent framework. Built to replace [langgraph](https://github.com/langchain-ai/langgraph) with something simpler, faster, and easier to reason about.

Inspired by [pi-agent-core](https://github.com/anthropics/pi-agent-core) (TypeScript), redesigned for Python.

## Why cubepi

### vs langgraph

| | langgraph | cubepi |
|---|---|---|
| **Abstraction** | Graph nodes + edges + channels — you model your agent as a state machine | Plain async functions — `run_agent_loop` is a while loop you can read in 5 minutes |
| **Streaming** | Callback-based, multiple handler types | `async for event in stream` — one pattern everywhere |
| **Checkpointing** | Full snapshot per step — serializes entire message list on every channel change | Append-only — writes only new messages, O(1) DB I/O regardless of conversation length |
| **Dependencies** | Pulls in langchain-core, langgraph-sdk, and transitive deps | 3 core deps: `pydantic`, `anthropic`, `openai` |
| **Tool execution** | Tools are graph nodes with manual wiring | Declare tools as functions, framework handles routing and parallel execution |
| **Multi-provider** | Via langchain chat model adapters | Native Provider protocol — Anthropic, OpenAI built in, add your own with one class |
| **Middleware** | Graph-level middleware on node entry/exit | Agent-level middleware with 5 typed hooks and declarative composition rules |
| **Observability** | LangSmith / Langfuse integration, full trace visualization | Events + middleware hooks — bring your own tracing |

### vs pi-agent-core

cubepi is a Python port of pi's architecture with Pythonic improvements:

| | pi-agent-core | cubepi |
|---|---|---|
| **Language** | TypeScript | Python (async-native) |
| **Type system** | Zod schemas | Pydantic v2 — validation, serialization, JSON Schema generation in one |
| **Cancel signal** | `AbortSignal` (Web API) | `asyncio.Event` — same semantics, native to Python |
| **Middleware** | Hooks only (callbacks on Agent) | Hooks + composable Middleware protocol with `compose_middleware()` |
| **Checkpointing** | Not built in | Built-in `MemoryCheckpointer` + `SQLiteCheckpointer` |
| **Test utility** | Internal test helpers | `FauxProvider` as public API — ship it, use it in your tests |

## Install

```bash
pip install cubepi

# With SQLite checkpointer
pip install cubepi[sqlite]
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add cubepi
uv add cubepi[sqlite]
```

## Quick Start

```python
import asyncio
from cubepi import Agent, AgentTool, Model
from cubepi.providers import AnthropicProvider

provider = AnthropicProvider(api_key="sk-...")

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72°F and sunny in {city}"

agent = Agent(
    model=Model(provider=provider, model="claude-sonnet-4-20250514"),
    tools=[
        AgentTool(
            name="get_weather",
            description="Get current weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            execute=get_weather,
        ),
    ],
    system_prompt="You are a helpful weather assistant.",
)

async def main():
    stream = await agent.prompt("What's the weather in Tokyo?")
    async for event in stream:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
    print()

asyncio.run(main())
```

## Architecture

```
cubepi/
├── providers/        # LLM provider abstraction
│   ├── base.py       # Provider protocol, message types, MessageStream
│   ├── anthropic.py  # Anthropic provider
│   ├── openai.py     # OpenAI provider
│   └── faux.py       # Test utility — pre-configured responses with realistic streaming
├── agent/            # Agent runtime
│   ├── agent.py      # Stateful Agent class
│   ├── loop.py       # Stateless core loop (the actual algorithm)
│   ├── tools.py      # Tool execution engine (sequential + parallel)
│   └── types.py      # Events, AgentTool, AgentContext, hook types
├── middleware/        # Composable middleware protocol
│   └── base.py       # 5 hooks with distinct composition rules
└── checkpointer/     # Persistence
    ├── base.py       # Checkpointer protocol
    ├── memory.py     # In-memory (dev/test)
    └── sqlite.py     # SQLite (lightweight persistence)
```

## Core Concepts

### Providers

Abstract LLM interaction behind a `Provider` protocol. All providers return `MessageStream` — an async iterator of `StreamEvent`s.

```python
from cubepi.providers import AnthropicProvider, OpenAIProvider, FauxProvider

# Real providers
anthropic = AnthropicProvider(api_key="...")
openai = OpenAIProvider(api_key="...")

# Test provider — no API calls, fully deterministic
faux = FauxProvider()
faux.set_responses(["Hello!", "How can I help?"])
```

### Tools

Declare tools with a name, JSON Schema parameters, and a sync or async execute function. The framework handles argument parsing, parallel execution, and error wrapping.

```python
from cubepi import AgentTool

tool = AgentTool(
    name="search",
    description="Search the web",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    execute=lambda query: f"Results for: {query}",
    sequential=False,  # allow parallel execution (default)
)
```

### Middleware

Composable hooks that modify behavior without touching the core loop:

```python
from cubepi import Middleware, compose_middleware

class LoggingMiddleware(Middleware):
    async def transform_context(self, messages, *, signal=None):
        print(f"Context has {len(messages)} messages")
        return messages

class SafetyMiddleware(Middleware):
    async def before_tool_call(self, ctx, *, signal=None):
        if ctx.tool_call.name == "dangerous_tool":
            return BeforeToolCallResult(block=True, content="Blocked by policy")
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
| `should_stop_after_turn` | Any true stops |

### Checkpointer

Persist conversation state with append-only semantics:

```python
from cubepi.checkpointer import MemoryCheckpointer, SQLiteCheckpointer

# In-memory for dev/test
cp = MemoryCheckpointer()

# SQLite for lightweight persistence
async with SQLiteCheckpointer("agent.db") as cp:
    agent = Agent(model=model, checkpointer=cp, thread_id="conv-1")
```

### FauxProvider for Testing

Ship your agent tests without API keys:

```python
from cubepi.providers import FauxProvider, faux_text, faux_tool_call, faux_assistant_message

provider = FauxProvider()
provider.set_responses([
    faux_assistant_message([
        faux_tool_call("search", {"query": "python"}),
    ]),
    faux_assistant_message("Here are the results..."),
])

agent = Agent(model=Model(provider=provider, model="test"), tools=[search_tool])
stream = await agent.prompt("Search for python")
# Streams realistic deltas — content_block_start, text_delta, etc.
```

## Requirements

- Python >= 3.11
- Core: `pydantic`, `anthropic`, `openai`
- Optional: `aiosqlite` (for `SQLiteCheckpointer`)

## License

MIT
