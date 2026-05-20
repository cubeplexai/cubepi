# cubepi Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement cubepi, a Pythonic async-native agent framework with multi-provider LLM streaming, hook/middleware extensibility, and optional checkpointing.

**Architecture:** Bottom-up build order — provider types first, then streaming, then FauxProvider (enables all later tests), then agent loop, Agent class, middleware, and checkpointers. Each task produces working, tested code.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest + pytest-asyncio, anthropic SDK, openai SDK, aiosqlite

**Spec:** `docs/specs/2026-05-09-cubepi-framework-design.md`

**Pi source reference:** `$HOME/pi/packages/agent/src/` (agent-loop.ts, agent.ts, types.ts) and `$HOME/pi/packages/ai/src/providers/faux.ts`

---

## File Structure

```
cubepi/
├── __init__.py                  # Public re-exports
├── py.typed                     # PEP 561 marker
├── providers/
│   ├── __init__.py              # Re-exports: Provider, MessageStream, types
│   ├── base.py                  # Provider Protocol, MessageStream, StreamEvent, all message/content types
│   ├── faux.py                  # FauxProvider (public test utility)
│   ├── anthropic.py             # AnthropicProvider
│   └── openai.py                # OpenAIProvider
├── agent/
│   ├── __init__.py              # Re-exports: Agent, run_agent_loop, AgentTool, etc.
│   ├── types.py                 # AgentEvent, AgentContext, AgentTool, hook types, AgentState
│   ├── loop.py                  # run_agent_loop, run_agent_loop_continue, internal runLoop
│   ├── tools.py                 # Tool execution engine (sequential/parallel)
│   └── agent.py                 # Agent class (stateful wrapper)
├── middleware/
│   ├── __init__.py              # Re-exports: Middleware, compose_middleware
│   └── base.py                  # Middleware Protocol + compose_middleware()
├── checkpointer/
│   ├── __init__.py              # Re-exports: Checkpointer, CheckpointData
│   ├── base.py                  # Checkpointer Protocol, CheckpointData
│   ├── memory.py                # MemoryCheckpointer
│   └── sqlite.py                # SQLiteCheckpointer
tests/
├── conftest.py                  # Shared fixtures (faux provider helpers)
├── providers/
│   ├── test_base.py             # Message type tests
│   ├── test_faux.py             # FauxProvider tests
│   ├── test_anthropic.py        # AnthropicProvider unit tests
│   └── test_openai.py           # OpenAIProvider unit tests
├── agent/
│   ├── test_types.py            # AgentTool, event type tests
│   ├── test_loop.py             # Agent loop tests (ported from pi)
│   ├── test_tools.py            # Tool execution engine tests
│   ├── test_agent.py            # Agent class tests (ported from pi)
│   └── test_e2e.py              # E2E tests (ported from pi)
├── middleware/
│   └── test_base.py             # Middleware composition tests
└── checkpointer/
    ├── test_memory.py           # MemoryCheckpointer tests
    └── test_sqlite.py           # SQLiteCheckpointer tests
pyproject.toml                   # Project config, dependencies, pytest config
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `cubepi/__init__.py`
- Create: `cubepi/py.typed`
- Create: `cubepi/providers/__init__.py`
- Create: `cubepi/agent/__init__.py`
- Create: `cubepi/middleware/__init__.py`
- Create: `cubepi/checkpointer/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/providers/__init__.py`
- Create: `tests/agent/__init__.py`
- Create: `tests/middleware/__init__.py`
- Create: `tests/checkpointer/__init__.py`

- [ ] **Step 1: Initialize project with uv and add dependencies**

```bash
cd /home/chris/cubepi
uv init --name cubepi --python ">=3.11"
uv add pydantic anthropic openai
uv add --optional sqlite aiosqlite
uv add --dev pytest pytest-asyncio pytest-cov
```

Then edit pyproject.toml to add pytest config and build target:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.hatch.build.targets.wheel]
packages = ["cubepi"]
```

- [ ] **Step 2: Create empty package init files and py.typed marker**

Create `cubepi/__init__.py` (empty for now, will add re-exports as modules are built):
```python
"""cubepi — Pythonic async-native agent framework."""
```

Create `cubepi/py.typed` (empty file — PEP 561 marker).

Create empty `__init__.py` files in:
- `cubepi/providers/__init__.py`
- `cubepi/agent/__init__.py`
- `cubepi/middleware/__init__.py`
- `cubepi/checkpointer/__init__.py`
- `tests/__init__.py`
- `tests/conftest.py` (empty for now)
- `tests/providers/__init__.py`
- `tests/agent/__init__.py`
- `tests/middleware/__init__.py`
- `tests/checkpointer/__init__.py`

- [ ] **Step 3: Verify pytest runs**

```bash
uv run pytest --co  # Should collect 0 tests, no errors
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml cubepi/ tests/
git commit -m "chore: scaffold cubepi project structure"
```

---

### Task 2: Provider Base Types

**Files:**
- Create: `cubepi/providers/base.py`
- Create: `tests/providers/test_base.py`

This is the foundation — all other modules depend on these types. Matches pi-ai's type system with Pydantic models.

- [ ] **Step 1: Write tests for message types**

Create `tests/providers/test_base.py`:

```python
import asyncio
from typing import Any

from cubepi.providers.base import (
    AssistantMessage,
    Content,
    ImageContent,
    MessageStream,
    Model,
    ModelCost,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class TestMessageTypes:
    def test_text_content_defaults(self):
        tc = TextContent()
        assert tc.type == "text"
        assert tc.text == ""

    def test_text_content_with_value(self):
        tc = TextContent(text="hello")
        assert tc.text == "hello"

    def test_image_content(self):
        ic = ImageContent(source="base64data", media_type="image/png")
        assert ic.type == "image"
        assert ic.source == "base64data"
        assert ic.media_type == "image/png"

    def test_thinking_content(self):
        tc = ThinkingContent(thinking="step by step")
        assert tc.type == "thinking"
        assert tc.thinking == "step by step"

    def test_tool_call(self):
        tc = ToolCall(id="tc-1", name="search", arguments={"query": "hello"})
        assert tc.type == "tool_call"
        assert tc.id == "tc-1"
        assert tc.name == "search"
        assert tc.arguments == {"query": "hello"}

    def test_user_message(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        assert msg.role == "user"
        assert msg.timestamp is None

    def test_assistant_message_defaults(self):
        msg = AssistantMessage(content=[TextContent(text="hello")])
        assert msg.role == "assistant"
        assert msg.stop_reason == "stop"
        assert msg.error_message is None
        assert msg.usage is None

    def test_assistant_message_with_tool_calls(self):
        msg = AssistantMessage(
            content=[
                TextContent(text="Let me search."),
                ToolCall(id="tc-1", name="search", arguments={"q": "test"}),
            ],
            stop_reason="tool_use",
        )
        assert len(msg.content) == 2
        assert msg.content[1].type == "tool_call"

    def test_tool_result_message(self):
        msg = ToolResultMessage(
            tool_call_id="tc-1",
            tool_name="search",
            content=[TextContent(text="result")],
        )
        assert msg.role == "tool_result"
        assert msg.is_error is False

    def test_model_defaults(self):
        m = Model(id="gpt-4o", provider="openai")
        assert m.context_window == 200_000
        assert m.max_tokens == 8192
        assert m.reasoning is False
        assert m.cost is None

    def test_usage(self):
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0

    def test_model_cost(self):
        c = ModelCost(input=3.0, output=15.0)
        assert c.cache_read == 0

    def test_tool_definition(self):
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert td.name == "search"


class TestMessageStream:
    async def test_stream_iteration_and_result(self):
        stream = MessageStream()
        msg = AssistantMessage(content=[TextContent(text="hello")])

        async def produce():
            await asyncio.sleep(0)
            stream.push(StreamEvent(type="text_delta", delta="hello"))
            stream.push(StreamEvent(type="done"))
            stream.set_result(msg)

        asyncio.create_task(produce())

        events = []
        async for event in stream:
            events.append(event)

        assert len(events) == 2
        assert events[0].type == "text_delta"
        assert events[1].type == "done"

        result = await stream.result()
        assert result.content[0].text == "hello"

    async def test_stream_error_event(self):
        stream = MessageStream()
        error_msg = AssistantMessage(
            content=[],
            stop_reason="error",
            error_message="API error",
        )

        async def produce():
            await asyncio.sleep(0)
            stream.push(StreamEvent(type="error", error_message="API error"))
            stream.set_result(error_msg)

        asyncio.create_task(produce())

        events = []
        async for event in stream:
            events.append(event)

        result = await stream.result()
        assert result.stop_reason == "error"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/providers/test_base.py -v
```

Expected: ImportError — `cubepi.providers.base` module does not exist yet.

- [ ] **Step 3: Implement provider base types**

Create `cubepi/providers/base.py`:

```python
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


class ModelCost(BaseModel):
    input: float = 0
    output: float = 0
    cache_read: float = 0
    cache_write: float = 0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class Model(BaseModel):
    id: str
    provider: str
    api: str = ""
    reasoning: bool = False
    context_window: int = 200_000
    max_tokens: int = 8192
    cost: ModelCost | None = None


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = ""


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: str = ""
    media_type: str = ""


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str = ""


Content = TextContent | ImageContent


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[Content]
    timestamp: float | None = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[Content | ThinkingContent | ToolCall]
    stop_reason: str = "stop"
    error_message: str | None = None
    usage: Usage | None = None
    timestamp: float | None = None


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[Content]
    is_error: bool = False
    timestamp: float | None = None


Message = UserMessage | AssistantMessage | ToolResultMessage


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class StreamEvent(BaseModel):
    type: Literal[
        "start",
        "text_start", "text_delta", "text_end",
        "thinking_start", "thinking_delta", "thinking_end",
        "toolcall_start", "toolcall_delta", "toolcall_end",
        "done", "error",
    ]
    delta: str | None = None
    partial: AssistantMessage | None = None
    error_message: str | None = None


class MessageStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._result_future: asyncio.Future[AssistantMessage] = asyncio.get_event_loop().create_future()

    def push(self, event: StreamEvent) -> None:
        self._queue.put_nowait(event)
        if event.type in ("done", "error"):
            self._queue.put_nowait(None)

    def set_result(self, message: AssistantMessage) -> None:
        if not self._result_future.done():
            self._result_future.set_result(message)

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __anext__(self) -> StreamEvent:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def result(self) -> AssistantMessage:
        return await self._result_future


@runtime_checkable
class Provider(Protocol):
    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        thinking: ThinkingLevel = "off",
        signal: asyncio.Event | None = None,
    ) -> MessageStream: ...
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/providers/test_base.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Update providers `__init__.py` with re-exports**

Update `cubepi/providers/__init__.py`:

```python
from cubepi.providers.base import (
    AssistantMessage,
    Content,
    ImageContent,
    Message,
    MessageStream,
    Model,
    ModelCost,
    Provider,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
)

__all__ = [
    "AssistantMessage",
    "Content",
    "ImageContent",
    "Message",
    "MessageStream",
    "Model",
    "ModelCost",
    "Provider",
    "StreamEvent",
    "TextContent",
    "ThinkingContent",
    "ThinkingLevel",
    "ToolCall",
    "ToolDefinition",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
]
```

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/ tests/providers/
git commit -m "feat: add provider base types and message stream"
```

---

### Task 3: FauxProvider

**Files:**
- Create: `cubepi/providers/faux.py`
- Create: `tests/providers/test_faux.py`

Port pi's `faux.ts`. This is the test infrastructure everything else depends on. The cubepi version is simpler — it implements the `Provider` protocol directly instead of using pi's global API registry.

- [ ] **Step 1: Write FauxProvider tests**

Create `tests/providers/test_faux.py`:

```python
import asyncio

from cubepi.providers.base import (
    AssistantMessage,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_text, faux_thinking, faux_tool_call


class TestFauxHelpers:
    def test_faux_text(self):
        block = faux_text("hello")
        assert block.type == "text"
        assert block.text == "hello"

    def test_faux_thinking(self):
        block = faux_thinking("step by step")
        assert block.type == "thinking"
        assert block.thinking == "step by step"

    def test_faux_tool_call(self):
        block = faux_tool_call("search", {"q": "test"}, id="tc-1")
        assert block.type == "tool_call"
        assert block.id == "tc-1"
        assert block.name == "search"

    def test_faux_tool_call_auto_id(self):
        block = faux_tool_call("search", {"q": "test"})
        assert block.id.startswith("tool:")

    def test_faux_assistant_message_string(self):
        msg = faux_assistant_message("hello")
        assert msg.role == "assistant"
        assert len(msg.content) == 1
        assert msg.content[0].type == "text"
        assert msg.content[0].text == "hello"
        assert msg.stop_reason == "stop"

    def test_faux_assistant_message_blocks(self):
        msg = faux_assistant_message([faux_text("hi"), faux_tool_call("search", {"q": "x"}, id="t1")])
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "tool_call"

    def test_faux_assistant_message_tool_use_stop_reason(self):
        msg = faux_assistant_message(
            [faux_tool_call("search", {"q": "x"}, id="t1")],
            stop_reason="tool_use",
        )
        assert msg.stop_reason == "tool_use"


class TestFauxProvider:
    def _make_model(self):
        from cubepi.providers.base import Model
        return Model(id="faux-1", provider="faux")

    async def test_basic_text_response(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("hello world")])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "hello world"
        assert result.stop_reason == "stop"
        assert any(e.type == "done" for e in events)

    async def test_responses_consumed_in_order(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message("first"),
            faux_assistant_message("second"),
        ])
        model = self._make_model()

        stream1 = await provider.stream(model, [])
        _ = [e async for e in stream1]
        r1 = await stream1.result()

        stream2 = await provider.stream(model, [])
        _ = [e async for e in stream2]
        r2 = await stream2.result()

        assert r1.content[0].text == "first"
        assert r2.content[0].text == "second"

    async def test_error_when_queue_exhausted(self):
        provider = FauxProvider()
        model = self._make_model()

        stream = await provider.stream(model, [])
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.stop_reason == "error"
        assert "No more faux responses" in (result.error_message or "")

    async def test_set_responses_replaces_queue(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("old")])
        provider.set_responses([faux_assistant_message("new")])
        model = self._make_model()

        stream = await provider.stream(model, [])
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "new"

    async def test_append_responses(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("first")])
        provider.append_responses([faux_assistant_message("second")])
        model = self._make_model()

        assert provider.pending_response_count == 2

    async def test_async_response_factory(self):
        async def factory(context, model):
            return faux_assistant_message("dynamic response")

        provider = FauxProvider()
        provider.set_responses([factory])
        model = self._make_model()

        stream = await provider.stream(model, [])
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "dynamic response"

    async def test_streams_text_deltas(self):
        provider = FauxProvider(token_size_min=1, token_size_max=1)
        provider.set_responses([faux_assistant_message("AB")])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        event_types = [e.type for e in events]

        assert "start" in event_types
        assert "text_start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types
        assert "done" in event_types

    async def test_streams_thinking_deltas(self):
        provider = FauxProvider(token_size_min=1, token_size_max=1)
        provider.set_responses([faux_assistant_message([faux_thinking("think"), faux_text("ok")])])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        event_types = [e.type for e in events]

        assert "thinking_start" in event_types
        assert "thinking_delta" in event_types
        assert "thinking_end" in event_types

    async def test_streams_tool_call_deltas(self):
        provider = FauxProvider(token_size_min=1, token_size_max=1)
        provider.set_responses([
            faux_assistant_message(
                [faux_tool_call("search", {"q": "test"}, id="t1")],
                stop_reason="tool_use",
            ),
        ])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        event_types = [e.type for e in events]

        assert "toolcall_start" in event_types
        assert "toolcall_delta" in event_types
        assert "toolcall_end" in event_types

    async def test_abort_before_start(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("hello")])
        model = self._make_model()
        signal = asyncio.Event()
        signal.set()

        stream = await provider.stream(model, [], signal=signal)
        events = [e async for e in stream]
        result = await stream.result()

        assert result.stop_reason == "aborted"

    async def test_abort_mid_stream(self):
        provider = FauxProvider(tokens_per_second=20, token_size_min=2, token_size_max=2)
        provider.set_responses([
            faux_assistant_message("one two three four five six seven eight nine ten"),
        ])
        model = self._make_model()
        signal = asyncio.Event()

        stream = await provider.stream(model, [], signal=signal)

        events = []
        count = 0
        async for event in stream:
            events.append(event)
            count += 1
            if count >= 3:
                signal.set()

        result = await stream.result()
        assert result.stop_reason == "aborted"

    async def test_error_message_passthrough(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message("", stop_reason="error", error_message="API rate limit"),
        ])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        result = await stream.result()

        assert result.stop_reason == "error"
        assert any(e.type == "error" for e in events)

    async def test_multiple_tool_calls_in_one_message(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message(
                [
                    faux_tool_call("search", {"q": "a"}, id="t1"),
                    faux_tool_call("search", {"q": "b"}, id="t2"),
                ],
                stop_reason="tool_use",
            ),
        ])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        result = await stream.result()

        toolcall_starts = [e for e in events if e.type == "toolcall_start"]
        assert len(toolcall_starts) == 2

    async def test_call_count_tracking(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("a"), faux_assistant_message("b")])
        model = self._make_model()

        assert provider.call_count == 0

        s1 = await provider.stream(model, [])
        _ = [e async for e in s1]
        assert provider.call_count == 1

        s2 = await provider.stream(model, [])
        _ = [e async for e in s2]
        assert provider.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/providers/test_faux.py -v
```

Expected: ImportError — `cubepi.providers.faux` does not exist yet.

- [ ] **Step 3: Implement FauxProvider**

Create `cubepi/providers/faux.py`:

```python
from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Awaitable, Callable

from cubepi.providers.base import (
    AssistantMessage,
    Message,
    MessageStream,
    Model,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDefinition,
    ThinkingLevel,
    Usage,
)

FauxContentBlock = TextContent | ThinkingContent | ToolCall

FauxResponseFactory = Callable[
    [list[Message], Model],
    AssistantMessage | Awaitable[AssistantMessage],
]

FauxResponseStep = AssistantMessage | FauxResponseFactory


def _random_id(prefix: str) -> str:
    import random
    return f"{prefix}:{int(time.time() * 1000)}:{random.randbytes(6).hex()}"


def faux_text(text: str) -> TextContent:
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    return ThinkingContent(thinking=thinking)


def faux_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    id: str | None = None,
) -> ToolCall:
    return ToolCall(id=id or _random_id("tool"), name=name, arguments=arguments)


def faux_assistant_message(
    content: str | FauxContentBlock | list[FauxContentBlock],
    *,
    stop_reason: str = "stop",
    error_message: str | None = None,
) -> AssistantMessage:
    if isinstance(content, str):
        blocks: list[FauxContentBlock] = [faux_text(content)]
    elif isinstance(content, list):
        blocks = content
    else:
        blocks = [content]
    return AssistantMessage(
        content=blocks,
        stop_reason=stop_reason,
        error_message=error_message,
        usage=Usage(),
        timestamp=time.time(),
    )


def _split_by_token_size(text: str, min_size: int, max_size: int) -> list[str]:
    import random
    chunks: list[str] = []
    i = 0
    while i < len(text):
        token_size = random.randint(min_size, max_size)
        char_size = max(1, token_size * 4)
        chunks.append(text[i : i + char_size])
        i += char_size
    return chunks or [""]


class FauxProvider:
    def __init__(
        self,
        *,
        tokens_per_second: float | None = None,
        token_size_min: int = 3,
        token_size_max: int = 5,
    ) -> None:
        self._responses: list[FauxResponseStep] = []
        self._tokens_per_second = tokens_per_second
        self._min = max(1, min(token_size_min, token_size_max))
        self._max = max(self._min, token_size_max)
        self.call_count = 0

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        self._responses = list(responses)

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        self._responses.extend(responses)

    @property
    def pending_response_count(self) -> int:
        return len(self._responses)

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        thinking: ThinkingLevel = "off",
        signal: asyncio.Event | None = None,
    ) -> MessageStream:
        ms = MessageStream()
        self.call_count += 1

        step = self._responses.pop(0) if self._responses else None

        async def _produce() -> None:
            try:
                if step is None:
                    error_msg = AssistantMessage(
                        content=[],
                        stop_reason="error",
                        error_message="No more faux responses queued",
                        usage=Usage(),
                        timestamp=time.time(),
                    )
                    ms.push(StreamEvent(type="error", error_message=error_msg.error_message))
                    ms.set_result(error_msg)
                    return

                if callable(step):
                    import inspect
                    if inspect.iscoroutinefunction(step):
                        resolved = await step(messages, model)
                    else:
                        resolved = step(messages, model)
                else:
                    resolved = step

                await self._stream_with_deltas(ms, resolved, signal)
            except Exception as exc:
                error_msg = AssistantMessage(
                    content=[],
                    stop_reason="error",
                    error_message=str(exc),
                    usage=Usage(),
                    timestamp=time.time(),
                )
                ms.push(StreamEvent(type="error", error_message=str(exc)))
                ms.set_result(error_msg)

        asyncio.create_task(_produce())
        return ms

    async def _stream_with_deltas(
        self,
        stream: MessageStream,
        message: AssistantMessage,
        signal: asyncio.Event | None,
    ) -> None:
        partial = AssistantMessage(
            content=[],
            stop_reason=message.stop_reason,
            usage=message.usage,
            timestamp=message.timestamp,
        )

        if signal and signal.is_set():
            aborted = self._make_aborted(partial)
            stream.push(StreamEvent(type="error", error_message="Request was aborted"))
            stream.set_result(aborted)
            return

        stream.push(StreamEvent(type="start", partial=partial.model_copy(deep=True)))

        for block in message.content:
            if signal and signal.is_set():
                aborted = self._make_aborted(partial)
                stream.push(StreamEvent(type="error", error_message="Request was aborted"))
                stream.set_result(aborted)
                return

            if isinstance(block, ThinkingContent):
                partial.content.append(ThinkingContent(thinking=""))
                stream.push(StreamEvent(type="thinking_start", partial=partial.model_copy(deep=True)))
                for chunk in _split_by_token_size(block.thinking, self._min, self._max):
                    await self._schedule_chunk(chunk)
                    if signal and signal.is_set():
                        aborted = self._make_aborted(partial)
                        stream.push(StreamEvent(type="error", error_message="Request was aborted"))
                        stream.set_result(aborted)
                        return
                    last = partial.content[-1]
                    if isinstance(last, ThinkingContent):
                        partial.content[-1] = ThinkingContent(thinking=last.thinking + chunk)
                    stream.push(StreamEvent(type="thinking_delta", delta=chunk, partial=partial.model_copy(deep=True)))
                stream.push(StreamEvent(type="thinking_end", partial=partial.model_copy(deep=True)))

            elif isinstance(block, TextContent):
                partial.content.append(TextContent(text=""))
                stream.push(StreamEvent(type="text_start", partial=partial.model_copy(deep=True)))
                for chunk in _split_by_token_size(block.text, self._min, self._max):
                    await self._schedule_chunk(chunk)
                    if signal and signal.is_set():
                        aborted = self._make_aborted(partial)
                        stream.push(StreamEvent(type="error", error_message="Request was aborted"))
                        stream.set_result(aborted)
                        return
                    last = partial.content[-1]
                    if isinstance(last, TextContent):
                        partial.content[-1] = TextContent(text=last.text + chunk)
                    stream.push(StreamEvent(type="text_delta", delta=chunk, partial=partial.model_copy(deep=True)))
                stream.push(StreamEvent(type="text_end", partial=partial.model_copy(deep=True)))

            elif isinstance(block, ToolCall):
                partial.content.append(ToolCall(id=block.id, name=block.name, arguments={}))
                stream.push(StreamEvent(type="toolcall_start", partial=partial.model_copy(deep=True)))
                json_str = json.dumps(block.arguments)
                for chunk in _split_by_token_size(json_str, self._min, self._max):
                    await self._schedule_chunk(chunk)
                    if signal and signal.is_set():
                        aborted = self._make_aborted(partial)
                        stream.push(StreamEvent(type="error", error_message="Request was aborted"))
                        stream.set_result(aborted)
                        return
                    stream.push(StreamEvent(type="toolcall_delta", delta=chunk, partial=partial.model_copy(deep=True)))
                last = partial.content[-1]
                if isinstance(last, ToolCall):
                    partial.content[-1] = ToolCall(id=block.id, name=block.name, arguments=block.arguments)
                stream.push(StreamEvent(type="toolcall_end", partial=partial.model_copy(deep=True)))

        if message.stop_reason in ("error", "aborted"):
            stream.push(StreamEvent(type="error", error_message=message.error_message))
            stream.set_result(message)
            return

        stream.push(StreamEvent(type="done"))
        stream.set_result(message)

    async def _schedule_chunk(self, chunk: str) -> None:
        if not self._tokens_per_second or self._tokens_per_second <= 0:
            await asyncio.sleep(0)
            return
        tokens = max(1, math.ceil(len(chunk) / 4))
        delay = tokens / self._tokens_per_second
        await asyncio.sleep(delay)

    @staticmethod
    def _make_aborted(partial: AssistantMessage) -> AssistantMessage:
        return partial.model_copy(update={
            "stop_reason": "aborted",
            "error_message": "Request was aborted",
            "timestamp": time.time(),
        })
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/providers/test_faux.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cubepi/providers/faux.py tests/providers/test_faux.py
git commit -m "feat: add FauxProvider test utility"
```

---

### Task 4: Agent Types

**Files:**
- Create: `cubepi/agent/types.py`
- Create: `tests/agent/test_types.py`

Defines AgentEvent (11 types), AgentContext, AgentTool, hook type aliases, and AgentState. These correspond to pi's `types.ts`.

- [ ] **Step 1: Write tests for agent types**

Create `tests/agent/test_types.py`:

```python
from typing import Any

from pydantic import BaseModel

from cubepi.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    AfterToolCallContext,
    AfterToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ShouldStopAfterTurnContext,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from cubepi.providers.base import (
    AssistantMessage,
    StreamEvent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


class TestAgentEvents:
    def test_agent_start_event(self):
        e = AgentStartEvent()
        assert e.type == "agent_start"

    def test_agent_end_event(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        e = AgentEndEvent(messages=[msg])
        assert e.type == "agent_end"
        assert len(e.messages) == 1

    def test_turn_start_event(self):
        e = TurnStartEvent()
        assert e.type == "turn_start"

    def test_turn_end_event(self):
        msg = AssistantMessage(content=[TextContent(text="hi")])
        e = TurnEndEvent(message=msg, tool_results=[])
        assert e.type == "turn_end"

    def test_message_start_event(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        e = MessageStartEvent(message=msg)
        assert e.type == "message_start"

    def test_message_update_event(self):
        msg = AssistantMessage(content=[TextContent(text="h")])
        se = StreamEvent(type="text_delta", delta="h")
        e = MessageUpdateEvent(message=msg, stream_event=se)
        assert e.type == "message_update"

    def test_message_end_event(self):
        msg = AssistantMessage(content=[TextContent(text="hello")])
        e = MessageEndEvent(message=msg)
        assert e.type == "message_end"

    def test_tool_execution_events(self):
        start = ToolExecutionStartEvent(tool_call_id="t1", tool_name="search", args={"q": "test"})
        assert start.type == "tool_execution_start"

        update = ToolExecutionUpdateEvent(
            tool_call_id="t1", tool_name="search", args={"q": "test"},
            partial_result=AgentToolResult(content=[TextContent(text="partial")]),
        )
        assert update.type == "tool_execution_update"

        end = ToolExecutionEndEvent(
            tool_call_id="t1", tool_name="search",
            result=AgentToolResult(content=[TextContent(text="done")]),
            is_error=False,
        )
        assert end.type == "tool_execution_end"


class TestAgentTool:
    async def test_tool_definition_generation(self):
        class SearchParams(BaseModel):
            query: str
            limit: int = 10

        async def execute(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text=f"found: {params.query}")])

        tool = AgentTool(
            name="search",
            description="Search the web",
            parameters=SearchParams,
            execute=execute,
        )

        defn = tool.to_definition()
        assert defn.name == "search"
        assert defn.description == "Search the web"
        assert "query" in defn.parameters.get("properties", {})
        assert "limit" in defn.parameters.get("properties", {})

    async def test_tool_execution(self):
        class EchoParams(BaseModel):
            text: str

        async def execute(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text=params.text)])

        tool = AgentTool(
            name="echo",
            description="Echo text",
            parameters=EchoParams,
            execute=execute,
        )

        result = await tool.execute("t1", EchoParams(text="hello"))
        assert result.content[0].text == "hello"


class TestAgentContext:
    def test_context_creation(self):
        ctx = AgentContext(system_prompt="You are helpful.", messages=[], tools=[])
        assert ctx.system_prompt == "You are helpful."
        assert ctx.messages == []


class TestHookTypes:
    def test_before_tool_call_result_defaults(self):
        r = BeforeToolCallResult()
        assert r.block is False
        assert r.reason is None

    def test_before_tool_call_result_block(self):
        r = BeforeToolCallResult(block=True, reason="Not allowed")
        assert r.block is True
        assert r.reason == "Not allowed"

    def test_after_tool_call_result_partial_override(self):
        r = AfterToolCallResult(terminate=True)
        assert r.terminate is True
        assert r.content is None
        assert r.is_error is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agent/test_types.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement agent types**

Create `cubepi/agent/types.py`:

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, Literal, TypeVar

from pydantic import BaseModel

from cubepi.providers.base import (
    AssistantMessage,
    Content,
    Message,
    Model,
    StreamEvent,
    TextContent,
    ThinkingLevel,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
)

TParams = TypeVar("TParams", bound=BaseModel)
TMessage = TypeVar("TMessage")


class AgentToolResult(BaseModel):
    content: list[Content]
    details: Any = None
    terminate: bool | None = None


@dataclass
class AgentTool(Generic[TParams]):
    name: str
    description: str
    parameters: type[TParams]
    execute: Callable[..., Awaitable[AgentToolResult]]
    label: str = ""
    execution_mode: Literal["sequential", "parallel"] | None = None

    def to_definition(self) -> ToolDefinition:
        schema = self.parameters.model_json_schema()
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=schema,
        )


@dataclass
class AgentContext:
    system_prompt: str
    messages: list[Any]
    tools: list[AgentTool] | None = None


# --- Hook context types ---

class BeforeToolCallResult(BaseModel):
    block: bool = False
    reason: str | None = None


class AfterToolCallResult(BaseModel):
    content: list[Content] | None = None
    details: Any = None
    is_error: bool | None = None
    terminate: bool | None = None


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: Any
    context: AgentContext


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: Any
    result: AgentToolResult
    is_error: bool
    context: AgentContext


@dataclass
class ShouldStopAfterTurnContext:
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[Any]


# --- Event types (11 total, matching pi) ---

class AgentStartEvent(BaseModel):
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[Any]


class TurnStartEvent(BaseModel):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(BaseModel):
    type: Literal["turn_end"] = "turn_end"
    message: Any
    tool_results: list[ToolResultMessage]


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: Any


class MessageUpdateEvent(BaseModel):
    type: Literal["message_update"] = "message_update"
    message: Any
    stream_event: StreamEvent


class MessageEndEvent(BaseModel):
    type: Literal["message_end"] = "message_end"
    message: Any


class ToolExecutionStartEvent(BaseModel):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: Any


class ToolExecutionUpdateEvent(BaseModel):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: Any = None
    partial_result: Any = None


class ToolExecutionEndEvent(BaseModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False


AgentEvent = (
    AgentStartEvent | AgentEndEvent
    | TurnStartEvent | TurnEndEvent
    | MessageStartEvent | MessageUpdateEvent | MessageEndEvent
    | ToolExecutionStartEvent | ToolExecutionUpdateEvent | ToolExecutionEndEvent
)

AgentEventSink = Callable[[AgentEvent], Awaitable[None]]

AgentListener = Callable[[AgentEvent, asyncio.Event | None], Awaitable[None] | None]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agent/test_types.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cubepi/agent/types.py tests/agent/test_types.py
git commit -m "feat: add agent event types, AgentTool, and hook context types"
```

---

### Task 5: Tool Execution Engine

**Files:**
- Create: `cubepi/agent/tools.py`
- Create: `tests/agent/test_tools.py`

Port pi's tool execution pipeline: `prepareToolCall` → `executePreparedToolCall` → `finalizeExecutedToolCall`, with sequential and parallel modes.

- [ ] **Step 1: Write tests for tool execution**

Create `tests/agent/test_tools.py`:

```python
import asyncio

from pydantic import BaseModel

from cubepi.agent.tools import execute_tool_calls
from cubepi.agent.types import (
    AfterToolCallResult,
    AgentContext,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
)
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall


class EchoParams(BaseModel):
    value: str


def make_echo_tool(
    *,
    name: str = "echo",
    execution_mode=None,
    execute_fn=None,
) -> AgentTool:
    async def default_execute(tool_call_id, params, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echoed: {params.value}")])

    return AgentTool(
        name=name,
        description="Echo tool",
        parameters=EchoParams,
        execute=execute_fn or default_execute,
        execution_mode=execution_mode,
    )


def make_assistant_msg(tool_calls: list[ToolCall]) -> AssistantMessage:
    return AssistantMessage(content=tool_calls, stop_reason="tool_use")


def make_context(tools: list[AgentTool]) -> AgentContext:
    return AgentContext(system_prompt="", messages=[], tools=tools)


class TestSequentialExecution:
    async def test_single_tool_call(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg([ToolCall(id="t1", name="echo", arguments={"value": "hi"})])

        events = []
        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential",
            emit=lambda e: events.append(e),
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].tool_call_id == "t1"
        assert not batch.messages[0].is_error
        assert batch.terminate is False

    async def test_multiple_tool_calls_run_sequentially(self):
        order = []

        async def tracked_execute(tool_call_id, params, *, signal=None, on_update=None):
            order.append(f"start:{params.value}")
            await asyncio.sleep(0.01)
            order.append(f"end:{params.value}")
            return AgentToolResult(content=[TextContent(text=f"echoed: {params.value}")])

        tool = make_echo_tool(execute_fn=tracked_execute)
        ctx = make_context([tool])
        msg = make_assistant_msg([
            ToolCall(id="t1", name="echo", arguments={"value": "first"}),
            ToolCall(id="t2", name="echo", arguments={"value": "second"}),
        ])

        batch = await execute_tool_calls(ctx, msg, tool_execution="sequential", emit=lambda e: None)

        assert order == ["start:first", "end:first", "start:second", "end:second"]

    async def test_unknown_tool_returns_error(self):
        ctx = make_context([])
        msg = make_assistant_msg([ToolCall(id="t1", name="unknown", arguments={})])

        batch = await execute_tool_calls(ctx, msg, tool_execution="sequential", emit=lambda e: None)

        assert len(batch.messages) == 1
        assert batch.messages[0].is_error
        assert "not found" in batch.messages[0].content[0].text.lower()


class TestParallelExecution:
    async def test_tools_run_concurrently(self):
        first_resolved = False
        parallel_observed = False
        release = asyncio.Event()

        async def slow_execute(tool_call_id, params, *, signal=None, on_update=None):
            nonlocal first_resolved, parallel_observed
            if params.value == "first":
                await release.wait()
                first_resolved = True
            if params.value == "second" and not first_resolved:
                parallel_observed = True
            return AgentToolResult(content=[TextContent(text=f"echoed: {params.value}")])

        tool = make_echo_tool(execute_fn=slow_execute)
        ctx = make_context([tool])
        msg = make_assistant_msg([
            ToolCall(id="t1", name="echo", arguments={"value": "first"}),
            ToolCall(id="t2", name="echo", arguments={"value": "second"}),
        ])

        async def run():
            await asyncio.sleep(0.02)
            release.set()

        asyncio.create_task(run())
        batch = await execute_tool_calls(ctx, msg, tool_execution="parallel", emit=lambda e: None)

        assert parallel_observed
        assert len(batch.messages) == 2
        assert batch.messages[0].tool_call_id == "t1"
        assert batch.messages[1].tool_call_id == "t2"

    async def test_sequential_tool_forces_sequential_mode(self):
        order = []

        async def tracked(tool_call_id, params, *, signal=None, on_update=None):
            order.append(f"start:{params.value}")
            await asyncio.sleep(0.01)
            order.append(f"end:{params.value}")
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = make_echo_tool(execute_fn=tracked, execution_mode="sequential")
        ctx = make_context([tool])
        msg = make_assistant_msg([
            ToolCall(id="t1", name="echo", arguments={"value": "a"}),
            ToolCall(id="t2", name="echo", arguments={"value": "b"}),
        ])

        batch = await execute_tool_calls(ctx, msg, tool_execution="parallel", emit=lambda e: None)

        assert order[0] == "start:a"
        assert order[1] == "end:a"


class TestBeforeToolCall:
    async def test_block_prevents_execution(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg([ToolCall(id="t1", name="echo", arguments={"value": "hi"})])

        async def before(ctx_arg, *, signal=None):
            return BeforeToolCallResult(block=True, reason="Blocked by test")

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential",
            before_tool_call=before, emit=lambda e: None,
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].is_error
        assert "Blocked by test" in batch.messages[0].content[0].text


class TestAfterToolCall:
    async def test_override_result(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg([ToolCall(id="t1", name="echo", arguments={"value": "hi"})])

        async def after(ctx_arg, *, signal=None):
            return AfterToolCallResult(
                content=[TextContent(text="overridden")],
                terminate=True,
            )

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential",
            after_tool_call=after, emit=lambda e: None,
        )

        assert batch.messages[0].content[0].text == "overridden"
        assert batch.terminate is True


class TestTermination:
    async def test_all_terminate_stops_loop(self):
        async def term_execute(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="done")], terminate=True)

        tool = make_echo_tool(execute_fn=term_execute)
        ctx = make_context([tool])
        msg = make_assistant_msg([ToolCall(id="t1", name="echo", arguments={"value": "a"})])

        batch = await execute_tool_calls(ctx, msg, tool_execution="sequential", emit=lambda e: None)
        assert batch.terminate is True

    async def test_partial_terminate_continues(self):
        async def maybe_term(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(
                content=[TextContent(text="done")],
                terminate=(params.value == "first"),
            )

        tool = make_echo_tool(execute_fn=maybe_term)
        ctx = make_context([tool])
        msg = make_assistant_msg([
            ToolCall(id="t1", name="echo", arguments={"value": "first"}),
            ToolCall(id="t2", name="echo", arguments={"value": "second"}),
        ])

        batch = await execute_tool_calls(ctx, msg, tool_execution="parallel", emit=lambda e: None)
        assert batch.terminate is False


class TestToolEvents:
    async def test_emits_execution_lifecycle_events(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg([ToolCall(id="t1", name="echo", arguments={"value": "hi"})])

        events = []
        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential",
            emit=lambda e: events.append(e),
        )

        types = [e.type for e in events]
        assert "tool_execution_start" in types
        assert "tool_execution_end" in types
        start_idx = types.index("tool_execution_start")
        end_idx = types.index("tool_execution_end")
        assert start_idx < end_idx
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agent/test_tools.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement tool execution engine**

Create `cubepi/agent/tools.py`:

```python
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from cubepi.agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from cubepi.providers.base import (
    Content,
    TextContent,
    ToolCall,
    ToolResultMessage,
    AssistantMessage,
)


@dataclass
class ToolCallBatch:
    messages: list[ToolResultMessage]
    terminate: bool


@dataclass
class _PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool
    args: Any


@dataclass
class _ImmediateOutcome:
    result: AgentToolResult
    is_error: bool


@dataclass
class _FinalizedOutcome:
    tool_call: ToolCall
    result: AgentToolResult
    is_error: bool


def _error_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)])


def _make_tool_result_message(finalized: _FinalizedOutcome) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        content=finalized.result.content,
        is_error=finalized.is_error,
        timestamp=time.time(),
    )


async def _prepare_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    before_tool_call: Callable | None,
    signal: asyncio.Event | None,
) -> _PreparedToolCall | _ImmediateOutcome:
    tool = None
    if context.tools:
        for t in context.tools:
            if t.name == tool_call.name:
                tool = t
                break

    if tool is None:
        return _ImmediateOutcome(
            result=_error_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        validated_args = tool.parameters.model_validate(tool_call.arguments)
    except (ValidationError, Exception) as exc:
        return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)

    if before_tool_call:
        try:
            before_ctx = BeforeToolCallContext(
                assistant_message=assistant_message,
                tool_call=tool_call,
                args=validated_args,
                context=context,
            )
            before_result = await before_tool_call(before_ctx, signal=signal)
            if before_result and before_result.block:
                return _ImmediateOutcome(
                    result=_error_result(before_result.reason or "Tool execution was blocked"),
                    is_error=True,
                )
        except Exception as exc:
            return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)

    return _PreparedToolCall(tool_call=tool_call, tool=tool, args=validated_args)


async def _execute_prepared(
    prepared: _PreparedToolCall,
    signal: asyncio.Event | None,
    emit: Callable,
) -> tuple[AgentToolResult, bool]:
    try:
        result = await prepared.tool.execute(
            prepared.tool_call.id,
            prepared.args,
            signal=signal,
            on_update=lambda partial: emit(ToolExecutionUpdateEvent(
                tool_call_id=prepared.tool_call.id,
                tool_name=prepared.tool_call.name,
                args=prepared.tool_call.arguments,
                partial_result=partial,
            )),
        )
        return result, False
    except Exception as exc:
        return _error_result(str(exc)), True


async def _finalize(
    context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    result: AgentToolResult,
    is_error: bool,
    after_tool_call: Callable | None,
    signal: asyncio.Event | None,
) -> _FinalizedOutcome:
    if after_tool_call:
        try:
            after_ctx = AfterToolCallContext(
                assistant_message=assistant_message,
                tool_call=prepared.tool_call,
                args=prepared.args,
                result=result,
                is_error=is_error,
                context=context,
            )
            after_result = await after_tool_call(after_ctx, signal=signal)
            if after_result:
                result = AgentToolResult(
                    content=after_result.content if after_result.content is not None else result.content,
                    details=after_result.details if after_result.details is not None else result.details,
                    terminate=after_result.terminate if after_result.terminate is not None else result.terminate,
                )
                is_error = after_result.is_error if after_result.is_error is not None else is_error
        except Exception as exc:
            result = _error_result(str(exc))
            is_error = True

    return _FinalizedOutcome(tool_call=prepared.tool_call, result=result, is_error=is_error)


def _should_terminate(finalized: list[_FinalizedOutcome]) -> bool:
    return len(finalized) > 0 and all(f.result.terminate is True for f in finalized)


async def execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    *,
    tool_execution: str = "parallel",
    before_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    signal: asyncio.Event | None = None,
    emit: Callable,
) -> ToolCallBatch:
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]

    has_sequential = any(
        t.execution_mode == "sequential"
        for tc in tool_calls
        if context.tools
        for t in context.tools
        if t.name == tc.name
    )

    if tool_execution == "sequential" or has_sequential:
        return await _execute_sequential(
            context, assistant_message, tool_calls,
            before_tool_call, after_tool_call, signal, emit,
        )
    return await _execute_parallel(
        context, assistant_message, tool_calls,
        before_tool_call, after_tool_call, signal, emit,
    )


async def _execute_sequential(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    before_tool_call: Callable | None,
    after_tool_call: Callable | None,
    signal: asyncio.Event | None,
    emit: Callable,
) -> ToolCallBatch:
    finalized_list: list[_FinalizedOutcome] = []
    messages: list[ToolResultMessage] = []

    for tc in tool_calls:
        await emit(ToolExecutionStartEvent(
            tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments,
        ))

        preparation = await _prepare_tool_call(
            context, assistant_message, tc, before_tool_call, signal,
        )

        if isinstance(preparation, _ImmediateOutcome):
            finalized = _FinalizedOutcome(
                tool_call=tc, result=preparation.result, is_error=preparation.is_error,
            )
        else:
            result, is_error = await _execute_prepared(preparation, signal, emit)
            finalized = await _finalize(
                context, assistant_message, preparation,
                result, is_error, after_tool_call, signal,
            )

        await emit(ToolExecutionEndEvent(
            tool_call_id=tc.id, tool_name=tc.name,
            result=finalized.result, is_error=finalized.is_error,
        ))
        tool_msg = _make_tool_result_message(finalized)
        await emit(MessageStartEvent(message=tool_msg))
        await emit(MessageEndEvent(message=tool_msg))
        finalized_list.append(finalized)
        messages.append(tool_msg)

    return ToolCallBatch(messages=messages, terminate=_should_terminate(finalized_list))


async def _execute_parallel(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    before_tool_call: Callable | None,
    after_tool_call: Callable | None,
    signal: asyncio.Event | None,
    emit: Callable,
) -> ToolCallBatch:
    entries: list[_FinalizedOutcome | asyncio.Task] = []

    for tc in tool_calls:
        await emit(ToolExecutionStartEvent(
            tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments,
        ))

        preparation = await _prepare_tool_call(
            context, assistant_message, tc, before_tool_call, signal,
        )

        if isinstance(preparation, _ImmediateOutcome):
            finalized = _FinalizedOutcome(
                tool_call=tc, result=preparation.result, is_error=preparation.is_error,
            )
            await emit(ToolExecutionEndEvent(
                tool_call_id=tc.id, tool_name=tc.name,
                result=finalized.result, is_error=finalized.is_error,
            ))
            entries.append(finalized)
        else:
            async def _run(prep=preparation):
                result, is_error = await _execute_prepared(prep, signal, emit)
                fin = await _finalize(
                    context, assistant_message, prep,
                    result, is_error, after_tool_call, signal,
                )
                await emit(ToolExecutionEndEvent(
                    tool_call_id=prep.tool_call.id, tool_name=prep.tool_call.name,
                    result=fin.result, is_error=fin.is_error,
                ))
                return fin

            entries.append(asyncio.create_task(_run()))

    finalized_list: list[_FinalizedOutcome] = []
    for entry in entries:
        if isinstance(entry, asyncio.Task):
            finalized_list.append(await entry)
        else:
            finalized_list.append(entry)

    messages: list[ToolResultMessage] = []
    for finalized in finalized_list:
        tool_msg = _make_tool_result_message(finalized)
        await emit(MessageStartEvent(message=tool_msg))
        await emit(MessageEndEvent(message=tool_msg))
        messages.append(tool_msg)

    return ToolCallBatch(messages=messages, terminate=_should_terminate(finalized_list))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agent/test_tools.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cubepi/agent/tools.py tests/agent/test_tools.py
git commit -m "feat: add tool execution engine with sequential/parallel modes"
```

---

### Task 6: Agent Loop

**Files:**
- Create: `cubepi/agent/loop.py`
- Create: `tests/agent/test_loop.py`

Port pi's `agent-loop.ts` — the stateless core loop with nested structure: outer (follow-up) → inner (tool calls + steering).

- [ ] **Step 1: Write agent loop tests**

Create `tests/agent/test_loop.py`. This file ports all tests from pi's `agent-loop.test.ts`. Due to the length, the test file should contain the following test classes and methods:

```python
import asyncio
from typing import Any

from pydantic import BaseModel

from cubepi.agent.loop import run_agent_loop, run_agent_loop_continue
from cubepi.agent.types import (
    AgentContext,
    AgentEvent,
    AgentTool,
    AgentToolResult,
)
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    MessageStream,
    Model,
    StreamEvent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    Usage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_text, faux_tool_call


def make_model() -> Model:
    return Model(id="faux-1", provider="faux")


def make_user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)])


def identity_converter(messages: list[Any]) -> list[Message]:
    return [m for m in messages if hasattr(m, "role") and m.role in ("user", "assistant", "tool_result")]


class EchoParams(BaseModel):
    value: str


def make_echo_tool(*, execution_mode=None, execute_fn=None) -> AgentTool:
    async def default_execute(tool_call_id, params, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echoed: {params.value}")])

    return AgentTool(
        name="echo",
        description="Echo tool",
        parameters=EchoParams,
        execute=execute_fn or default_execute,
        execution_mode=execution_mode,
    )


class TestAgentLoop:
    async def test_emit_events_with_agent_message_types(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Hi there!")])
        context = AgentContext(system_prompt="You are helpful.", messages=[], tools=[])
        user_prompt = make_user_message("Hello")

        events: list[AgentEvent] = []
        messages = await run_agent_loop(
            prompts=[user_prompt],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            emit=lambda e: events.append(e),
        )

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "turn_start" in event_types
        assert "message_start" in event_types
        assert "message_end" in event_types
        assert "turn_end" in event_types
        assert "agent_end" in event_types

    async def test_custom_message_types_via_convert_to_llm(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Response")])

        notification = {"role": "notification", "text": "info"}
        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[notification],
            tools=[],
        )
        user_prompt = make_user_message("Hello")

        converted: list[Message] = []

        def converter(messages):
            result = [m for m in messages if hasattr(m, "role") and m.role in ("user", "assistant", "tool_result")]
            converted.extend(result)
            return result

        await run_agent_loop(
            prompts=[user_prompt],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=converter,
            emit=lambda e: None,
        )

        assert len(converted) == 1
        assert converted[0].role == "user"

    async def test_transform_context_before_convert_to_llm(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Response")])

        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[
                make_user_message("old 1"),
                faux_assistant_message("old resp 1"),
                make_user_message("old 2"),
                faux_assistant_message("old resp 2"),
            ],
            tools=[],
        )

        transformed_len = []
        converted_len = []

        async def transform(messages, *, signal=None):
            result = messages[-2:]
            transformed_len.append(len(result))
            return result

        def converter(messages):
            converted_len.append(len(messages))
            return identity_converter(messages)

        await run_agent_loop(
            prompts=[make_user_message("new")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=converter,
            transform_context=transform,
            emit=lambda e: None,
        )

        assert transformed_len[0] == 2
        assert converted_len[0] == 2

    async def test_tool_calls_and_results(self):
        executed = []

        async def echo_execute(tool_call_id, params, *, signal=None, on_update=None):
            executed.append(params.value)
            return AgentToolResult(content=[TextContent(text=f"echoed: {params.value}")])

        tool = make_echo_tool(execute_fn=echo_execute)
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message(
                [faux_tool_call("echo", {"value": "hello"}, id="tool-1")],
                stop_reason="tool_use",
            ),
            faux_assistant_message("done"),
        ])

        context = AgentContext(system_prompt="", messages=[], tools=[tool])
        events: list[AgentEvent] = []

        await run_agent_loop(
            prompts=[make_user_message("echo something")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            emit=lambda e: events.append(e),
        )

        assert executed == ["hello"]
        event_types = [e.type for e in events]
        assert "tool_execution_start" in event_types
        assert "tool_execution_end" in event_types

    async def test_should_stop_after_turn(self):
        tool = make_echo_tool()
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message(
                [faux_tool_call("echo", {"value": "hello"}, id="tool-1")],
                stop_reason="tool_use",
            ),
            faux_assistant_message("should not run"),
        ])

        context = AgentContext(system_prompt="", messages=[], tools=[tool])
        stop_called = []

        async def should_stop(ctx):
            stop_called.append(True)
            return True

        events: list[AgentEvent] = []
        messages = await run_agent_loop(
            prompts=[make_user_message("echo something")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            should_stop_after_turn=should_stop,
            emit=lambda e: events.append(e),
        )

        assert len(stop_called) == 1
        assert provider.call_count == 1
        roles = [m.role for m in messages]
        assert roles == ["user", "assistant", "tool_result"]

    async def test_steering_messages_injected_after_tool_calls(self):
        executed = []

        async def echo_execute(tool_call_id, params, *, signal=None, on_update=None):
            executed.append(params.value)
            return AgentToolResult(content=[TextContent(text=f"ok:{params.value}")])

        tool = make_echo_tool(execute_fn=echo_execute)
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message(
                [
                    faux_tool_call("echo", {"value": "first"}, id="tool-1"),
                    faux_tool_call("echo", {"value": "second"}, id="tool-2"),
                ],
                stop_reason="tool_use",
            ),
            faux_assistant_message("done"),
        ])

        context = AgentContext(system_prompt="", messages=[], tools=[tool])
        steering_delivered = False

        async def get_steering():
            nonlocal steering_delivered
            if len(executed) >= 1 and not steering_delivered:
                steering_delivered = True
                return [make_user_message("interrupt")]
            return []

        events: list[AgentEvent] = []
        await run_agent_loop(
            prompts=[make_user_message("start")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            get_steering_messages=get_steering,
            tool_execution="sequential",
            emit=lambda e: events.append(e),
        )

        assert executed == ["first", "second"]

    async def test_terminate_when_all_tool_results_terminate(self):
        async def term_execute(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="done")], terminate=True)

        tool = make_echo_tool(execute_fn=term_execute)
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message(
                [faux_tool_call("echo", {"value": "hello"}, id="tool-1")],
                stop_reason="tool_use",
            ),
        ])

        context = AgentContext(system_prompt="", messages=[], tools=[tool])
        messages = await run_agent_loop(
            prompts=[make_user_message("echo something")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            emit=lambda e: None,
        )

        assert provider.call_count == 1
        roles = [m.role for m in messages]
        assert roles == ["user", "assistant", "tool_result"]

    async def test_error_stop_reason_ends_loop(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message("", stop_reason="error", error_message="API error"),
        ])

        context = AgentContext(system_prompt="", messages=[], tools=[])
        events: list[AgentEvent] = []

        messages = await run_agent_loop(
            prompts=[make_user_message("hello")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            emit=lambda e: events.append(e),
        )

        event_types = [e.type for e in events]
        assert "agent_end" in event_types
        assert messages[-1].role == "assistant"
        assert messages[-1].stop_reason == "error"


class TestAgentLoopContinue:
    async def test_raises_when_no_messages(self):
        provider = FauxProvider()
        context = AgentContext(system_prompt="", messages=[], tools=[])

        try:
            await run_agent_loop_continue(
                context=context,
                provider=provider,
                model=make_model(),
                convert_to_llm=identity_converter,
                emit=lambda e: None,
            )
            assert False, "Should have raised"
        except ValueError as e:
            assert "no messages" in str(e).lower()

    async def test_continue_without_user_message_events(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Response")])

        context = AgentContext(
            system_prompt="",
            messages=[make_user_message("Hello")],
            tools=[],
        )

        events: list[AgentEvent] = []
        messages = await run_agent_loop_continue(
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            emit=lambda e: events.append(e),
        )

        assert len(messages) == 1
        assert messages[0].role == "assistant"

        message_ends = [e for e in events if e.type == "message_end"]
        assert len(message_ends) == 1
        assert message_ends[0].message.role == "assistant"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agent/test_loop.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement agent loop**

Create `cubepi/agent/loop.py`:

```python
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from cubepi.agent.tools import ToolCallBatch, execute_tool_calls
from cubepi.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentEventSink,
    AgentStartEvent,
    AgentTool,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ShouldStopAfterTurnContext,
    TurnEndEvent,
    TurnStartEvent,
)
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    MessageStream,
    Model,
    Provider,
    StreamEvent,
    ThinkingLevel,
    ToolCall,
    ToolResultMessage,
)


async def run_agent_loop(
    *,
    prompts: list[Any],
    context: AgentContext,
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    emit: Callable,
    transform_context: Callable | None = None,
    before_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    should_stop_after_turn: Callable | None = None,
    get_steering_messages: Callable | None = None,
    get_follow_up_messages: Callable | None = None,
    thinking: ThinkingLevel = "off",
    tool_execution: str = "parallel",
    signal: asyncio.Event | None = None,
    system_prompt: str | None = None,
) -> list[Any]:
    new_messages: list[Any] = list(prompts)
    current_context = AgentContext(
        system_prompt=system_prompt if system_prompt is not None else context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=context.tools,
    )

    await emit(AgentStartEvent())
    await emit(TurnStartEvent())
    for prompt in prompts:
        await emit(MessageStartEvent(message=prompt))
        await emit(MessageEndEvent(message=prompt))

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        provider=provider,
        model=model,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        should_stop_after_turn=should_stop_after_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
        thinking=thinking,
        tool_execution=tool_execution,
        signal=signal,
        emit=emit,
    )
    return new_messages


async def run_agent_loop_continue(
    *,
    context: AgentContext,
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    emit: Callable,
    transform_context: Callable | None = None,
    before_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    should_stop_after_turn: Callable | None = None,
    get_steering_messages: Callable | None = None,
    get_follow_up_messages: Callable | None = None,
    thinking: ThinkingLevel = "off",
    tool_execution: str = "parallel",
    signal: asyncio.Event | None = None,
    system_prompt: str | None = None,
) -> list[Any]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    last = context.messages[-1]
    if hasattr(last, "role") and last.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[Any] = []
    current_context = AgentContext(
        system_prompt=system_prompt if system_prompt is not None else context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    await emit(AgentStartEvent())
    await emit(TurnStartEvent())

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        provider=provider,
        model=model,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        should_stop_after_turn=should_stop_after_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
        thinking=thinking,
        tool_execution=tool_execution,
        signal=signal,
        emit=emit,
    )
    return new_messages


async def _run_loop(
    *,
    current_context: AgentContext,
    new_messages: list[Any],
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    transform_context: Callable | None,
    before_tool_call: Callable | None,
    after_tool_call: Callable | None,
    should_stop_after_turn: Callable | None,
    get_steering_messages: Callable | None,
    get_follow_up_messages: Callable | None,
    thinking: ThinkingLevel,
    tool_execution: str,
    signal: asyncio.Event | None,
    emit: Callable,
) -> None:
    first_turn = True
    pending_messages: list[Any] = []
    if get_steering_messages:
        pending_messages = await get_steering_messages() or []

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            if not first_turn:
                await emit(TurnStartEvent())
            else:
                first_turn = False

            if pending_messages:
                for msg in pending_messages:
                    await emit(MessageStartEvent(message=msg))
                    await emit(MessageEndEvent(message=msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                pending_messages = []

            message = await _stream_assistant_response(
                current_context, provider, model, convert_to_llm,
                transform_context, thinking, signal, emit,
            )
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await emit(TurnEndEvent(message=message, tool_results=[]))
                await emit(AgentEndEvent(messages=new_messages))
                return

            tool_calls = [c for c in message.content if isinstance(c, ToolCall)]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False

            if tool_calls:
                batch = await execute_tool_calls(
                    current_context, message,
                    tool_execution=tool_execution,
                    before_tool_call=before_tool_call,
                    after_tool_call=after_tool_call,
                    signal=signal,
                    emit=emit,
                )
                tool_results = batch.messages
                has_more_tool_calls = not batch.terminate

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await emit(TurnEndEvent(message=message, tool_results=tool_results))

            if should_stop_after_turn:
                stop_ctx = ShouldStopAfterTurnContext(
                    message=message,
                    tool_results=tool_results,
                    context=current_context,
                    new_messages=new_messages,
                )
                if await should_stop_after_turn(stop_ctx):
                    await emit(AgentEndEvent(messages=new_messages))
                    return

            if get_steering_messages:
                pending_messages = await get_steering_messages() or []

        if get_follow_up_messages:
            follow_ups = await get_follow_up_messages() or []
            if follow_ups:
                pending_messages = follow_ups
                continue

        break

    await emit(AgentEndEvent(messages=new_messages))


async def _stream_assistant_response(
    context: AgentContext,
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    transform_context: Callable | None,
    thinking: ThinkingLevel,
    signal: asyncio.Event | None,
    emit: Callable,
) -> AssistantMessage:
    messages = context.messages
    if transform_context:
        messages = await transform_context(messages, signal=signal)

    llm_messages = convert_to_llm(messages)
    if asyncio.iscoroutine(llm_messages):
        llm_messages = await llm_messages

    tools_defs = None
    if context.tools:
        tools_defs = [t.to_definition() for t in context.tools]

    stream = await provider.stream(
        model, llm_messages,
        system_prompt=context.system_prompt,
        tools=tools_defs,
        thinking=thinking,
        signal=signal,
    )

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in stream:
        if event.type == "start":
            partial_message = event.partial
            if partial_message:
                context.messages.append(partial_message)
                added_partial = True
                await emit(MessageStartEvent(message=partial_message.model_copy(deep=True)))

        elif event.type in (
            "text_start", "text_delta", "text_end",
            "thinking_start", "thinking_delta", "thinking_end",
            "toolcall_start", "toolcall_delta", "toolcall_end",
        ):
            if partial_message and event.partial:
                partial_message = event.partial
                context.messages[-1] = partial_message
                await emit(MessageUpdateEvent(
                    message=partial_message.model_copy(deep=True),
                    stream_event=event,
                ))

        elif event.type in ("done", "error"):
            final_message = await stream.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                await emit(MessageStartEvent(message=final_message))
            await emit(MessageEndEvent(message=final_message))
            return final_message

    final_message = await stream.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await emit(MessageStartEvent(message=final_message))
    await emit(MessageEndEvent(message=final_message))
    return final_message
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agent/test_loop.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cubepi/agent/loop.py tests/agent/test_loop.py
git commit -m "feat: add agent loop with nested loop structure"
```

---

### Task 7: Agent Class

**Files:**
- Create: `cubepi/agent/agent.py`
- Create: `tests/agent/test_agent.py`
- Create: `tests/agent/test_e2e.py`

Port pi's `Agent` class — stateful wrapper with message queues, subscribe/prompt/resume/abort/reset. Also port all e2e tests.

- [ ] **Step 1: Write Agent class tests**

Create `tests/agent/test_agent.py` porting all tests from pi's `agent.test.ts`. The test file should contain these test classes and methods:

```python
import asyncio

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentEvent, AgentTool, AgentToolResult
from cubepi.providers.base import (
    AssistantMessage,
    Model,
    TextContent,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message


def make_model() -> Model:
    return Model(id="faux-1", provider="faux")


class TestAgentInit:
    def test_default_state(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        assert agent.state.system_prompt == ""
        assert agent.state.thinking == "off"
        assert agent.state.tools == []
        assert agent.state.messages == []
        assert agent.state.is_streaming is False
        assert agent.state.streaming_message is None
        assert agent.state.pending_tool_calls == set()
        assert agent.state.error_message is None

    def test_custom_initial_state(self):
        provider = FauxProvider()
        agent = Agent(
            provider=provider,
            model=make_model(),
            system_prompt="You are a helpful assistant.",
            thinking="low",
        )

        assert agent.state.system_prompt == "You are a helpful assistant."
        assert agent.state.thinking == "low"


class TestAgentSubscribe:
    def test_subscribe_and_unsubscribe(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        count = 0

        def listener(event, signal=None):
            nonlocal count
            count += 1

        unsub = agent.subscribe(listener)
        assert count == 0

        unsub()

    async def test_events_emitted_on_prompt(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("ok")])
        agent = Agent(provider=provider, model=make_model())

        events = []
        agent.subscribe(lambda e, s=None: events.append(e.type))

        await agent.prompt("hello")

        assert "agent_start" in events
        assert "message_start" in events
        assert "message_end" in events
        assert "agent_end" in events

    async def test_full_lifecycle_events_for_thrown_run_failures(self):
        async def bad_stream(*args, **kwargs):
            raise RuntimeError("provider exploded")

        provider = FauxProvider()
        provider.stream = bad_stream
        agent = Agent(provider=provider, model=make_model())

        events = []
        agent.subscribe(lambda e, s=None: events.append(e.type))

        await agent.prompt("hello")

        assert events == [
            "agent_start", "turn_start",
            "message_start", "message_end",
            "message_start", "message_end",
            "turn_end", "agent_end",
        ]
        last_msg = agent.state.messages[-1]
        assert last_msg.role == "assistant"
        assert last_msg.stop_reason == "error"
        assert last_msg.error_message == "provider exploded"
        assert agent.state.error_message == "provider exploded"

    async def test_await_async_subscribers_before_prompt_resolves(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("ok")])
        agent = Agent(provider=provider, model=make_model())

        barrier = asyncio.Event()
        listener_finished = False

        async def listener(event, signal=None):
            nonlocal listener_finished
            if event.type == "agent_end":
                await barrier.wait()
                listener_finished = True

        agent.subscribe(listener)

        prompt_resolved = False

        async def run_prompt():
            nonlocal prompt_resolved
            await agent.prompt("hello")
            prompt_resolved = True

        task = asyncio.create_task(run_prompt())
        await asyncio.sleep(0.05)

        assert not prompt_resolved
        assert not listener_finished
        assert agent.state.is_streaming is True

        barrier.set()
        await task

        assert listener_finished
        assert prompt_resolved
        assert agent.state.is_streaming is False


class TestAgentState:
    def test_tools_are_copied(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        tools = [AgentTool(name="t", description="t", parameters=type("P", (object,), {"model_json_schema": classmethod(lambda cls: {})}), execute=lambda *a, **k: None)]
        agent.state.tools = tools
        assert agent.state.tools is not tools

    def test_messages_are_copied(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        messages = [UserMessage(content=[TextContent(text="hi")])]
        agent.state.messages = messages
        assert agent.state.messages is not messages


class TestAgentQueues:
    def test_steer_queues_message(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        msg = UserMessage(content=[TextContent(text="steering")])
        agent.steer(msg)
        assert msg not in agent.state.messages

    def test_follow_up_queues_message(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        msg = UserMessage(content=[TextContent(text="follow-up")])
        agent.follow_up(msg)
        assert msg not in agent.state.messages


class TestAgentAbort:
    def test_abort_without_active_run_does_not_throw(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())
        agent.abort()


class TestAgentPromptGuards:
    async def test_raises_when_prompt_called_while_streaming(self):
        barrier = asyncio.Event()
        provider = FauxProvider()

        async def slow_stream(*args, **kwargs):
            from cubepi.providers.base import MessageStream, StreamEvent
            ms = MessageStream()

            async def produce():
                await barrier.wait()
                msg = faux_assistant_message("ok")
                ms.push(StreamEvent(type="done"))
                ms.set_result(msg)

            asyncio.create_task(produce())
            return ms

        provider.stream = slow_stream
        agent = Agent(provider=provider, model=make_model())

        task = asyncio.create_task(agent.prompt("first"))
        await asyncio.sleep(0.02)
        assert agent.state.is_streaming is True

        try:
            await agent.prompt("second")
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "already processing" in str(e).lower()

        barrier.set()
        await task

    async def test_raises_when_resume_called_while_streaming(self):
        barrier = asyncio.Event()
        provider = FauxProvider()

        async def slow_stream(*args, **kwargs):
            from cubepi.providers.base import MessageStream, StreamEvent
            ms = MessageStream()

            async def produce():
                await barrier.wait()
                msg = faux_assistant_message("ok")
                ms.push(StreamEvent(type="done"))
                ms.set_result(msg)

            asyncio.create_task(produce())
            return ms

        provider.stream = slow_stream
        agent = Agent(provider=provider, model=make_model())

        task = asyncio.create_task(agent.prompt("first"))
        await asyncio.sleep(0.02)

        try:
            await agent.resume()
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "already processing" in str(e).lower()

        barrier.set()
        await task


class TestAgentResume:
    async def test_resume_processes_follow_up_messages(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message("Initial response"),
            faux_assistant_message("Processed"),
        ])
        agent = Agent(provider=provider, model=make_model())

        await agent.prompt("Initial")
        agent.follow_up(UserMessage(content=[TextContent(text="follow-up")]))
        await agent.resume()

        has_follow_up = any(
            hasattr(m, "content") and hasattr(m, "role") and m.role == "user"
            and any(
                hasattr(c, "text") and c.text == "follow-up"
                for c in (m.content if isinstance(m.content, list) else [])
            )
            for m in agent.state.messages
        )
        assert has_follow_up
        assert agent.state.messages[-1].role == "assistant"
```

- [ ] **Step 2: Write E2E tests**

Create `tests/agent/test_e2e.py` porting pi's `e2e.test.ts`:

```python
import asyncio

from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentEvent, AgentTool, AgentToolResult
from cubepi.providers.base import (
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from cubepi.providers.faux import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)


def make_model() -> Model:
    return Model(id="faux-1", provider="faux")


class CalculateParams(BaseModel):
    expression: str


def make_calculate_tool() -> AgentTool:
    async def execute(tool_call_id, params, *, signal=None, on_update=None):
        try:
            result = eval(params.expression)
            return AgentToolResult(
                content=[TextContent(text=f"{params.expression} = {result}")]
            )
        except Exception as e:
            raise RuntimeError(str(e))

    return AgentTool(
        name="calculate",
        description="Calculate a math expression",
        parameters=CalculateParams,
        execute=execute,
    )


class TestE2EBasic:
    async def test_basic_text_prompt(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("4")])
        agent = Agent(
            provider=provider,
            model=make_model(),
            system_prompt="You are a helpful assistant.",
        )

        await agent.prompt("What is 2+2?")

        assert agent.state.is_streaming is False
        assert len(agent.state.messages) == 2
        assert agent.state.messages[0].role == "user"
        assert agent.state.messages[1].role == "assistant"
        assert "4" in agent.state.messages[1].content[0].text

    async def test_tool_execution_with_pending_tracking(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message(
                [
                    faux_text("Let me calculate that."),
                    faux_tool_call("calculate", {"expression": "123 * 456"}, id="calc-1"),
                ],
                stop_reason="tool_use",
            ),
            faux_assistant_message("The result is 56088."),
        ])
        agent = Agent(
            provider=provider,
            model=make_model(),
            system_prompt="Always use the calculator tool for math.",
            tools=[make_calculate_tool()],
        )

        pending_events = []
        agent.subscribe(lambda e, s=None: (
            pending_events.append({"type": e.type, "ids": list(agent.state.pending_tool_calls)})
            if e.type in ("tool_execution_start", "tool_execution_end") else None
        ))

        await agent.prompt("Calculate 123 * 456")

        assert agent.state.is_streaming is False
        assert len(agent.state.messages) >= 4
        tool_result = next(m for m in agent.state.messages if m.role == "tool_result")
        assert "56088" in tool_result.content[0].text
        assert agent.state.pending_tool_calls == set()

    async def test_abort_during_streaming(self):
        provider = FauxProvider(tokens_per_second=20, token_size_min=2, token_size_max=2)
        provider.set_responses([
            faux_assistant_message(
                "one two three four five six seven eight nine ten eleven twelve thirteen"
            ),
        ])
        agent = Agent(provider=provider, model=make_model())

        prompt_task = asyncio.create_task(agent.prompt("Count"))
        await asyncio.sleep(0.03)
        agent.abort()
        await prompt_task

        assert agent.state.is_streaming is False
        last_msg = agent.state.messages[-1]
        assert last_msg.role == "assistant"
        assert last_msg.stop_reason == "aborted"

    async def test_lifecycle_events_during_streaming(self):
        provider = FauxProvider(token_size_min=1, token_size_max=1)
        provider.set_responses([faux_assistant_message("1 2 3 4 5")])
        agent = Agent(provider=provider, model=make_model())

        events = []
        agent.subscribe(lambda e, s=None: events.append(e.type))

        await agent.prompt("Count from 1 to 5")

        assert "agent_start" in events
        assert "message_start" in events
        assert "message_update" in events
        assert "message_end" in events
        assert "agent_end" in events

    async def test_context_across_multiple_turns(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message("Nice to meet you, Alice."),
            lambda msgs, model: faux_assistant_message(
                "Your name is Alice."
                if any(
                    hasattr(m, "content")
                    and isinstance(m.content, list)
                    and any(hasattr(c, "text") and "Alice" in c.text for c in m.content)
                    for m in msgs
                    if hasattr(m, "role") and m.role == "user"
                )
                else "I don't know your name."
            ),
        ])
        agent = Agent(provider=provider, model=make_model())

        await agent.prompt("My name is Alice.")
        assert len(agent.state.messages) == 2

        await agent.prompt("What is my name?")
        assert len(agent.state.messages) == 4
        assert "alice" in agent.state.messages[3].content[0].text.lower()

    async def test_thinking_content_preserved(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message([faux_thinking("step by step"), faux_text("4")]),
        ])
        agent = Agent(
            provider=provider,
            model=Model(id="faux-reasoning", provider="faux", reasoning=True),
            thinking="low",
        )

        await agent.prompt("What is 2+2?")

        assistant_msg = agent.state.messages[1]
        assert assistant_msg.content[0].type == "thinking"
        assert assistant_msg.content[0].thinking == "step by step"
        assert assistant_msg.content[1].type == "text"
        assert assistant_msg.content[1].text == "4"


class TestE2EResume:
    async def test_raises_when_no_messages(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        try:
            await agent.resume()
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "no messages" in str(e).lower()

    async def test_continue_from_user_message(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("HELLO WORLD")])
        agent = Agent(provider=provider, model=make_model())

        agent.state.messages = [
            UserMessage(content=[TextContent(text="Say HELLO WORLD")]),
        ]

        await agent.resume()

        assert agent.state.is_streaming is False
        assert len(agent.state.messages) == 2
        assert agent.state.messages[1].role == "assistant"

    async def test_continue_from_tool_result(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("The answer is 8.")])
        agent = Agent(
            provider=provider,
            model=make_model(),
            tools=[make_calculate_tool()],
        )

        agent.state.messages = [
            UserMessage(content=[TextContent(text="What is 5 + 3?")]),
            AssistantMessage(
                content=[
                    TextContent(text="Let me calculate."),
                    ToolCall(id="calc-1", name="calculate", arguments={"expression": "5 + 3"}),
                ],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="calc-1",
                tool_name="calculate",
                content=[TextContent(text="5 + 3 = 8")],
            ),
        ]

        await agent.resume()

        assert len(agent.state.messages) >= 4
        assert agent.state.messages[-1].role == "assistant"
        assert "8" in agent.state.messages[-1].content[0].text
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/agent/test_agent.py tests/agent/test_e2e.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement Agent class**

Create `cubepi/agent/agent.py`:

```python
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, TypeVar

from cubepi.agent.loop import run_agent_loop, run_agent_loop_continue
from cubepi.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AgentTool,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    Model,
    Provider,
    TextContent,
    ThinkingLevel,
    Usage,
)

TMessage = TypeVar("TMessage")


def _default_convert_to_llm(messages: list[Any]) -> list[Message]:
    return [m for m in messages if hasattr(m, "role") and m.role in ("user", "assistant", "tool_result")]


class _MessageQueue:
    def __init__(self, mode: str = "one-at-a-time") -> None:
        self.mode = mode
        self._messages: list[Any] = []

    def enqueue(self, message: Any) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return len(self._messages) > 0

    def drain(self) -> list[Any]:
        if self.mode == "all":
            drained = self._messages[:]
            self._messages = []
            return drained
        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def clear(self) -> None:
        self._messages = []


@dataclass
class AgentState:
    system_prompt: str = ""
    model: Model = field(default_factory=lambda: Model(id="unknown", provider="unknown"))
    thinking: ThinkingLevel = "off"
    is_streaming: bool = False
    streaming_message: Any = None
    error_message: str | None = None
    _tools: list[AgentTool] = field(default_factory=list)
    _messages: list[Any] = field(default_factory=list)
    _pending_tool_calls: set[str] = field(default_factory=set)

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    @property
    def messages(self) -> list[Any]:
        return self._messages

    @messages.setter
    def messages(self, value: list[Any]) -> None:
        self._messages = list(value)

    @property
    def pending_tool_calls(self) -> set[str]:
        return self._pending_tool_calls

    @pending_tool_calls.setter
    def pending_tool_calls(self, value: set[str]) -> None:
        self._pending_tool_calls = set(value)


class Agent(Generic[TMessage]):
    def __init__(
        self,
        *,
        provider: Provider,
        model: Model,
        system_prompt: str = "",
        tools: list[AgentTool] | None = None,
        thinking: ThinkingLevel = "off",
        convert_to_llm: Callable | None = None,
        transform_context: Callable | None = None,
        before_tool_call: Callable | None = None,
        after_tool_call: Callable | None = None,
        should_stop_after_turn: Callable | None = None,
        steering_mode: str = "one-at-a-time",
        follow_up_mode: str = "one-at-a-time",
        tool_execution: str = "parallel",
        checkpointer: Any = None,
        thread_id: str | None = None,
    ) -> None:
        self._provider = provider
        self._state = AgentState(
            system_prompt=system_prompt,
            model=model,
            thinking=thinking,
        )
        if tools:
            self._state.tools = tools
        self.convert_to_llm = convert_to_llm or _default_convert_to_llm
        self.transform_context = transform_context
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.should_stop_after_turn = should_stop_after_turn
        self.tool_execution = tool_execution
        self.checkpointer = checkpointer
        self.thread_id = thread_id

        self._steering_queue = _MessageQueue(steering_mode)
        self._follow_up_queue = _MessageQueue(follow_up_mode)
        self._listeners: list[Callable] = []
        self._active_signal: asyncio.Event | None = None
        self._active_done: asyncio.Event | None = None

    @property
    def state(self) -> AgentState:
        return self._state

    def subscribe(self, listener: Callable) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def steer(self, message: Any) -> None:
        self._steering_queue.enqueue(message)

    def follow_up(self, message: Any) -> None:
        self._follow_up_queue.enqueue(message)

    def abort(self) -> None:
        if self._active_signal:
            self._active_signal.set()

    async def wait_for_idle(self) -> None:
        if self._active_done:
            await self._active_done.wait()

    def reset(self) -> None:
        self._state.messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    async def prompt(self, message: str | Any | list[Any]) -> None:
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages."
            )

        if isinstance(message, str):
            from cubepi.providers.base import UserMessage, TextContent
            messages = [UserMessage(content=[TextContent(text=message)], timestamp=time.time())]
        elif isinstance(message, list):
            messages = message
        else:
            messages = [message]

        await self._run_prompt(messages)

    async def resume(self) -> None:
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        if not self._state.messages:
            raise RuntimeError("No messages to continue from")

        last = self._state.messages[-1]
        if hasattr(last, "role") and last.role == "assistant":
            steering = self._steering_queue.drain()
            if steering:
                await self._run_prompt(steering)
                return

            follow_ups = self._follow_up_queue.drain()
            if follow_ups:
                await self._run_prompt(follow_ups)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continuation()

    async def _run_prompt(self, messages: list[Any]) -> None:
        await self._run_with_lifecycle(lambda signal: run_agent_loop(
            prompts=messages,
            context=self._create_context_snapshot(),
            provider=self._provider,
            model=self._state.model,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            should_stop_after_turn=self.should_stop_after_turn,
            get_steering_messages=lambda: self._steering_queue.drain(),
            get_follow_up_messages=lambda: self._follow_up_queue.drain(),
            thinking=self._state.thinking,
            tool_execution=self.tool_execution,
            signal=signal,
            emit=lambda e: self._process_event(e),
        ))

    async def _run_continuation(self) -> None:
        await self._run_with_lifecycle(lambda signal: run_agent_loop_continue(
            context=self._create_context_snapshot(),
            provider=self._provider,
            model=self._state.model,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            should_stop_after_turn=self.should_stop_after_turn,
            get_steering_messages=lambda: self._steering_queue.drain(),
            get_follow_up_messages=lambda: self._follow_up_queue.drain(),
            thinking=self._state.thinking,
            tool_execution=self.tool_execution,
            signal=signal,
            emit=lambda e: self._process_event(e),
        ))

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

    async def _run_with_lifecycle(self, executor: Callable) -> None:
        signal = asyncio.Event()
        done = asyncio.Event()
        self._active_signal = signal
        self._active_done = done
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            await executor(signal)
        except Exception as error:
            await self._handle_run_failure(error, signal.is_set())
        finally:
            self._state.is_streaming = False
            self._state.streaming_message = None
            self._state.pending_tool_calls = set()
            self._active_signal = None
            done.set()
            self._active_done = None

    async def _handle_run_failure(self, error: Exception, aborted: bool) -> None:
        failure_message = AssistantMessage(
            content=[TextContent(text="")],
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            usage=Usage(),
            timestamp=time.time(),
        )
        await self._process_event(MessageStartEvent(message=failure_message))
        await self._process_event(MessageEndEvent(message=failure_message))
        await self._process_event(TurnEndEvent(message=failure_message, tool_results=[]))
        await self._process_event(AgentEndEvent(messages=[failure_message]))

    async def _process_event(self, event: AgentEvent) -> None:
        if event.type == "message_start":
            self._state.streaming_message = event.message
        elif event.type == "message_update":
            self._state.streaming_message = event.message
        elif event.type == "message_end":
            self._state.streaming_message = None
            self._state.messages.append(event.message)
        elif event.type == "tool_execution_start":
            self._state.pending_tool_calls = self._state.pending_tool_calls | {event.tool_call_id}
        elif event.type == "tool_execution_end":
            self._state.pending_tool_calls = self._state.pending_tool_calls - {event.tool_call_id}
        elif event.type == "turn_end":
            msg = event.message
            if hasattr(msg, "role") and msg.role == "assistant" and hasattr(msg, "error_message") and msg.error_message:
                self._state.error_message = msg.error_message
        elif event.type == "agent_end":
            self._state.streaming_message = None

        for listener in self._listeners:
            result = listener(event, self._active_signal)
            if asyncio.iscoroutine(result):
                await result
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/agent/test_agent.py tests/agent/test_e2e.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Update agent `__init__.py` with re-exports**

Update `cubepi/agent/__init__.py`:

```python
from cubepi.agent.agent import Agent, AgentState
from cubepi.agent.loop import run_agent_loop, run_agent_loop_continue
from cubepi.agent.tools import execute_tool_calls
from cubepi.agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentListener,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ShouldStopAfterTurnContext,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

__all__ = [
    "Agent",
    "AgentState",
    "run_agent_loop",
    "run_agent_loop_continue",
    "execute_tool_calls",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "AgentContext",
    "AgentEndEvent",
    "AgentEvent",
    "AgentListener",
    "AgentStartEvent",
    "AgentTool",
    "AgentToolResult",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "ShouldStopAfterTurnContext",
    "ToolExecutionEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]
```

- [ ] **Step 7: Commit**

```bash
git add cubepi/agent/ tests/agent/
git commit -m "feat: add Agent class with event system, queues, and lifecycle management"
```

---

### Task 8: Middleware Protocol and Composition

**Files:**
- Create: `cubepi/middleware/base.py`
- Create: `tests/middleware/test_base.py`

Implement the Middleware protocol and `compose_middleware()` function. This is new in cubepi (not in pi).

- [ ] **Step 1: Write middleware tests**

Create `tests/middleware/test_base.py`:

```python
from typing import Any

from cubepi.middleware.base import Middleware, compose_middleware
from cubepi.agent.types import (
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    AfterToolCallContext,
    ShouldStopAfterTurnContext,
    AgentContext,
    AgentToolResult,
)
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


class TestComposeMiddlewareEmpty:
    def test_empty_list_returns_empty_dict(self):
        hooks = compose_middleware([])
        assert hooks == {}


class TestTransformContext:
    async def test_chained_transform_context(self):
        class AddPrefix(Middleware):
            async def transform_context(self, messages, *, signal=None):
                return [UserMessage(content=[TextContent(text="PREFIX")])] + list(messages)

        class AddSuffix(Middleware):
            async def transform_context(self, messages, *, signal=None):
                return list(messages) + [UserMessage(content=[TextContent(text="SUFFIX")])]

        hooks = compose_middleware([AddPrefix(), AddSuffix()])
        result = await hooks["transform_context"]([UserMessage(content=[TextContent(text="middle")])], signal=None)

        assert len(result) == 3
        assert result[0].content[0].text == "PREFIX"
        assert result[1].content[0].text == "middle"
        assert result[2].content[0].text == "SUFFIX"


class TestConvertToLlm:
    async def test_last_implementation_wins(self):
        class First(Middleware):
            async def convert_to_llm(self, messages):
                return [UserMessage(content=[TextContent(text="first")])]

        class Second(Middleware):
            async def convert_to_llm(self, messages):
                return [UserMessage(content=[TextContent(text="second")])]

        hooks = compose_middleware([First(), Second()])
        result = await hooks["convert_to_llm"]([])

        assert len(result) == 1
        assert result[0].content[0].text == "second"


class TestBeforeToolCall:
    async def test_any_block_stops_execution(self):
        class Allower(Middleware):
            async def before_tool_call(self, ctx, *, signal=None):
                return None

        class Blocker(Middleware):
            async def before_tool_call(self, ctx, *, signal=None):
                return BeforeToolCallResult(block=True, reason="Blocked")

        hooks = compose_middleware([Allower(), Blocker()])

        ctx = BeforeToolCallContext(
            assistant_message=AssistantMessage(content=[]),
            tool_call=ToolCall(id="t1", name="test", arguments={}),
            args={},
            context=AgentContext(system_prompt="", messages=[]),
        )
        result = await hooks["before_tool_call"](ctx, signal=None)

        assert result is not None
        assert result.block is True
        assert result.reason == "Blocked"

    async def test_no_block_returns_none(self):
        class Allower(Middleware):
            async def before_tool_call(self, ctx, *, signal=None):
                return None

        hooks = compose_middleware([Allower()])
        ctx = BeforeToolCallContext(
            assistant_message=AssistantMessage(content=[]),
            tool_call=ToolCall(id="t1", name="test", arguments={}),
            args={},
            context=AgentContext(system_prompt="", messages=[]),
        )
        result = await hooks["before_tool_call"](ctx, signal=None)
        assert result is None


class TestAfterToolCall:
    async def test_later_overrides_earlier(self):
        class First(Middleware):
            async def after_tool_call(self, ctx, *, signal=None):
                return AfterToolCallResult(content=[TextContent(text="first")])

        class Second(Middleware):
            async def after_tool_call(self, ctx, *, signal=None):
                return AfterToolCallResult(content=[TextContent(text="second")])

        hooks = compose_middleware([First(), Second()])

        ctx = AfterToolCallContext(
            assistant_message=AssistantMessage(content=[]),
            tool_call=ToolCall(id="t1", name="test", arguments={}),
            args={},
            result=AgentToolResult(content=[TextContent(text="original")]),
            is_error=False,
            context=AgentContext(system_prompt="", messages=[]),
        )
        result = await hooks["after_tool_call"](ctx, signal=None)

        assert result.content[0].text == "second"


class TestShouldStopAfterTurn:
    async def test_any_true_stops(self):
        class NoStop(Middleware):
            async def should_stop_after_turn(self, ctx):
                return False

        class YesStop(Middleware):
            async def should_stop_after_turn(self, ctx):
                return True

        hooks = compose_middleware([NoStop(), YesStop()])

        ctx = ShouldStopAfterTurnContext(
            message=AssistantMessage(content=[]),
            tool_results=[],
            context=AgentContext(system_prompt="", messages=[]),
            new_messages=[],
        )
        result = await hooks["should_stop_after_turn"](ctx)
        assert result is True

    async def test_all_false_continues(self):
        class NoStop(Middleware):
            async def should_stop_after_turn(self, ctx):
                return False

        hooks = compose_middleware([NoStop()])
        ctx = ShouldStopAfterTurnContext(
            message=AssistantMessage(content=[]),
            tool_results=[],
            context=AgentContext(system_prompt="", messages=[]),
            new_messages=[],
        )
        result = await hooks["should_stop_after_turn"](ctx)
        assert result is False


class TestPartialMiddleware:
    async def test_middleware_with_only_some_hooks(self):
        class OnlyTransform(Middleware):
            async def transform_context(self, messages, *, signal=None):
                return messages

        hooks = compose_middleware([OnlyTransform()])
        assert "transform_context" in hooks
        assert "convert_to_llm" not in hooks
        assert "before_tool_call" not in hooks
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/middleware/test_base.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement middleware**

Create `cubepi/middleware/base.py`:

```python
from __future__ import annotations

from typing import Any, Callable


class Middleware:
    async def transform_context(self, messages: list, *, signal=None) -> list:
        raise NotImplementedError

    async def convert_to_llm(self, messages: list) -> list:
        raise NotImplementedError

    async def before_tool_call(self, ctx: Any, *, signal=None) -> Any:
        raise NotImplementedError

    async def after_tool_call(self, ctx: Any, *, signal=None) -> Any:
        raise NotImplementedError

    async def should_stop_after_turn(self, ctx: Any) -> bool:
        raise NotImplementedError


def _has_method(middleware: Middleware, name: str) -> bool:
    method = getattr(type(middleware), name, None)
    base_method = getattr(Middleware, name, None)
    return method is not None and method is not base_method


def compose_middleware(middlewares: list[Middleware]) -> dict[str, Callable]:
    hooks: dict[str, Callable] = {}

    transform_chain = [m for m in middlewares if _has_method(m, "transform_context")]
    if transform_chain:
        async def composed_transform(messages, *, signal=None):
            result = messages
            for mw in transform_chain:
                result = await mw.transform_context(result, signal=signal)
            return result

        hooks["transform_context"] = composed_transform

    convert_impls = [m for m in middlewares if _has_method(m, "convert_to_llm")]
    if convert_impls:
        last = convert_impls[-1]

        async def composed_convert(messages):
            return await last.convert_to_llm(messages)

        hooks["convert_to_llm"] = composed_convert

    before_chain = [m for m in middlewares if _has_method(m, "before_tool_call")]
    if before_chain:
        async def composed_before(ctx, *, signal=None):
            for mw in before_chain:
                result = await mw.before_tool_call(ctx, signal=signal)
                if result and getattr(result, "block", False):
                    return result
            return None

        hooks["before_tool_call"] = composed_before

    after_chain = [m for m in middlewares if _has_method(m, "after_tool_call")]
    if after_chain:
        async def composed_after(ctx, *, signal=None):
            last_result = None
            for mw in after_chain:
                result = await mw.after_tool_call(ctx, signal=signal)
                if result is not None:
                    last_result = result
            return last_result

        hooks["after_tool_call"] = composed_after

    stop_chain = [m for m in middlewares if _has_method(m, "should_stop_after_turn")]
    if stop_chain:
        async def composed_stop(ctx):
            for mw in stop_chain:
                if await mw.should_stop_after_turn(ctx):
                    return True
            return False

        hooks["should_stop_after_turn"] = composed_stop

    return hooks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/middleware/test_base.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Update middleware `__init__.py`**

Update `cubepi/middleware/__init__.py`:

```python
from cubepi.middleware.base import Middleware, compose_middleware

__all__ = ["Middleware", "compose_middleware"]
```

- [ ] **Step 6: Commit**

```bash
git add cubepi/middleware/ tests/middleware/
git commit -m "feat: add Middleware protocol and compose_middleware"
```

---

### Task 9: Checkpointer — Protocol, Memory, and SQLite

**Files:**
- Create: `cubepi/checkpointer/base.py`
- Create: `cubepi/checkpointer/memory.py`
- Create: `cubepi/checkpointer/sqlite.py`
- Create: `tests/checkpointer/test_memory.py`
- Create: `tests/checkpointer/test_sqlite.py`

- [ ] **Step 1: Write MemoryCheckpointer tests**

Create `tests/checkpointer/test_memory.py`:

```python
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import TextContent, UserMessage


class TestMemoryCheckpointer:
    async def test_load_empty_thread(self):
        cp = MemoryCheckpointer()
        data = await cp.load("thread-1")
        assert data is None

    async def test_append_and_load(self):
        cp = MemoryCheckpointer()
        msg1 = UserMessage(content=[TextContent(text="hello")])
        msg2 = UserMessage(content=[TextContent(text="world")])

        await cp.append("thread-1", [msg1])
        await cp.append("thread-1", [msg2])

        data = await cp.load("thread-1")
        assert data is not None
        assert len(data.messages) == 2
        assert data.messages[0].content[0].text == "hello"
        assert data.messages[1].content[0].text == "world"

    async def test_save_extra(self):
        cp = MemoryCheckpointer()
        await cp.append("thread-1", [UserMessage(content=[TextContent(text="hi")])])
        await cp.save_extra("thread-1", {"compaction_index": 5})

        data = await cp.load("thread-1")
        assert data is not None
        assert data.extra["compaction_index"] == 5

    async def test_save_extra_merges(self):
        cp = MemoryCheckpointer()
        await cp.append("thread-1", [UserMessage(content=[TextContent(text="hi")])])
        await cp.save_extra("thread-1", {"a": 1})
        await cp.save_extra("thread-1", {"b": 2})

        data = await cp.load("thread-1")
        assert data.extra == {"a": 1, "b": 2}

    async def test_multiple_threads(self):
        cp = MemoryCheckpointer()
        await cp.append("t1", [UserMessage(content=[TextContent(text="t1")])])
        await cp.append("t2", [UserMessage(content=[TextContent(text="t2")])])

        d1 = await cp.load("t1")
        d2 = await cp.load("t2")

        assert d1.messages[0].content[0].text == "t1"
        assert d2.messages[0].content[0].text == "t2"
```

- [ ] **Step 2: Write SQLiteCheckpointer tests**

Create `tests/checkpointer/test_sqlite.py`:

```python
import os
import tempfile

import pytest

from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.providers.base import TextContent, UserMessage


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


class TestSQLiteCheckpointer:
    async def test_load_empty_thread(self, db_path):
        async with SQLiteCheckpointer(db_path) as cp:
            data = await cp.load("thread-1")
            assert data is None

    async def test_append_and_load(self, db_path):
        async with SQLiteCheckpointer(db_path) as cp:
            msg1 = UserMessage(content=[TextContent(text="hello")])
            msg2 = UserMessage(content=[TextContent(text="world")])
            await cp.append("thread-1", [msg1])
            await cp.append("thread-1", [msg2])

            data = await cp.load("thread-1")
            assert data is not None
            assert len(data.messages) == 2
            assert data.messages[0].content[0].text == "hello"

    async def test_save_extra(self, db_path):
        async with SQLiteCheckpointer(db_path) as cp:
            await cp.append("thread-1", [UserMessage(content=[TextContent(text="hi")])])
            await cp.save_extra("thread-1", {"index": 42})

            data = await cp.load("thread-1")
            assert data.extra["index"] == 42

    async def test_persistence_across_instances(self, db_path):
        async with SQLiteCheckpointer(db_path) as cp:
            await cp.append("thread-1", [UserMessage(content=[TextContent(text="persist")])])

        async with SQLiteCheckpointer(db_path) as cp:
            data = await cp.load("thread-1")
            assert data is not None
            assert data.messages[0].content[0].text == "persist"

    async def test_multiple_threads(self, db_path):
        async with SQLiteCheckpointer(db_path) as cp:
            await cp.append("t1", [UserMessage(content=[TextContent(text="t1")])])
            await cp.append("t2", [UserMessage(content=[TextContent(text="t2")])])

            d1 = await cp.load("t1")
            d2 = await cp.load("t2")
            assert d1.messages[0].content[0].text == "t1"
            assert d2.messages[0].content[0].text == "t2"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/checkpointer/ -v
```

Expected: ImportError.

- [ ] **Step 4: Implement checkpointer base and MemoryCheckpointer**

Create `cubepi/checkpointer/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class CheckpointData:
    messages: list[Any] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Any]) -> None: ...
    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None: ...
```

Create `cubepi/checkpointer/memory.py`:

```python
from __future__ import annotations

from typing import Any

from cubepi.checkpointer.base import CheckpointData


class MemoryCheckpointer:
    def __init__(self) -> None:
        self._store: dict[str, CheckpointData] = {}

    async def load(self, thread_id: str) -> CheckpointData | None:
        return self._store.get(thread_id)

    async def append(self, thread_id: str, messages: list[Any]) -> None:
        if thread_id not in self._store:
            self._store[thread_id] = CheckpointData()
        self._store[thread_id].messages.extend(messages)

    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None:
        if thread_id not in self._store:
            self._store[thread_id] = CheckpointData()
        self._store[thread_id].extra.update(extra)
```

- [ ] **Step 5: Run MemoryCheckpointer tests**

```bash
pytest tests/checkpointer/test_memory.py -v
```

Expected: All PASS.

- [ ] **Step 6: Implement SQLiteCheckpointer**

Create `cubepi/checkpointer/sqlite.py`:

```python
from __future__ import annotations

import json
from typing import Any

import aiosqlite

from cubepi.checkpointer.base import CheckpointData


class SQLiteCheckpointer:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def __aenter__(self) -> SQLiteCheckpointer:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  thread_id TEXT NOT NULL,"
            "  message_json TEXT NOT NULL,"
            "  created_at REAL NOT NULL DEFAULT (julianday('now'))"
            ")"
        )
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS thread_extra ("
            "  thread_id TEXT PRIMARY KEY,"
            "  extra_json TEXT NOT NULL DEFAULT '{}'"
            ")"
        )
        await self._db.commit()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def load(self, thread_id: str) -> CheckpointData | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT message_json FROM messages WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        )
        rows = await cursor.fetchall()
        if not rows:
            extra_cursor = await self._db.execute(
                "SELECT extra_json FROM thread_extra WHERE thread_id = ?",
                (thread_id,),
            )
            extra_row = await extra_cursor.fetchone()
            if not extra_row:
                return None

        messages = []
        for row in rows:
            msg_data = json.loads(row[0])
            messages.append(_deserialize_message(msg_data))

        extra_cursor = await self._db.execute(
            "SELECT extra_json FROM thread_extra WHERE thread_id = ?",
            (thread_id,),
        )
        extra_row = await extra_cursor.fetchone()
        extra = json.loads(extra_row[0]) if extra_row else {}

        return CheckpointData(messages=messages, extra=extra)

    async def append(self, thread_id: str, messages: list[Any]) -> None:
        assert self._db is not None
        for msg in messages:
            msg_json = _serialize_message(msg)
            await self._db.execute(
                "INSERT INTO messages (thread_id, message_json) VALUES (?, ?)",
                (thread_id, msg_json),
            )
        await self._db.commit()

    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None:
        assert self._db is not None
        existing_cursor = await self._db.execute(
            "SELECT extra_json FROM thread_extra WHERE thread_id = ?",
            (thread_id,),
        )
        existing_row = await existing_cursor.fetchone()
        if existing_row:
            existing_extra = json.loads(existing_row[0])
            existing_extra.update(extra)
            await self._db.execute(
                "UPDATE thread_extra SET extra_json = ? WHERE thread_id = ?",
                (json.dumps(existing_extra), thread_id),
            )
        else:
            await self._db.execute(
                "INSERT INTO thread_extra (thread_id, extra_json) VALUES (?, ?)",
                (thread_id, json.dumps(extra)),
            )
        await self._db.commit()


def _serialize_message(msg: Any) -> str:
    if hasattr(msg, "model_dump"):
        return json.dumps(msg.model_dump())
    return json.dumps(msg)


def _deserialize_message(data: dict) -> Any:
    from cubepi.providers.base import AssistantMessage, ToolResultMessage, UserMessage

    role = data.get("role")
    if role == "user":
        return UserMessage.model_validate(data)
    elif role == "assistant":
        return AssistantMessage.model_validate(data)
    elif role == "tool_result":
        return ToolResultMessage.model_validate(data)
    return data
```

- [ ] **Step 7: Run all checkpointer tests**

```bash
pytest tests/checkpointer/ -v
```

Expected: All tests PASS.

- [ ] **Step 8: Update checkpointer `__init__.py`**

Update `cubepi/checkpointer/__init__.py`:

```python
from cubepi.checkpointer.base import Checkpointer, CheckpointData
from cubepi.checkpointer.memory import MemoryCheckpointer

__all__ = ["Checkpointer", "CheckpointData", "MemoryCheckpointer"]
```

- [ ] **Step 9: Commit**

```bash
git add cubepi/checkpointer/ tests/checkpointer/
git commit -m "feat: add Checkpointer protocol with Memory and SQLite implementations"
```

---

### Task 10: AnthropicProvider

**Files:**
- Create: `cubepi/providers/anthropic.py`
- Create: `tests/providers/test_anthropic.py`

- [ ] **Step 1: Write AnthropicProvider unit tests**

Create `tests/providers/test_anthropic.py`:

```python
import pytest

from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.base import (
    Model,
    TextContent,
    ToolCall,
    ToolDefinition,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
)


class TestAnthropicMessageConversion:
    def test_convert_user_message(self):
        msg = UserMessage(content=[TextContent(text="hello")])
        result = AnthropicProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "hello"

    def test_convert_assistant_message(self):
        msg = AssistantMessage(content=[TextContent(text="hi")])
        result = AnthropicProvider._convert_message(msg)
        assert result["role"] == "assistant"

    def test_convert_assistant_with_tool_call(self):
        msg = AssistantMessage(
            content=[ToolCall(id="tc-1", name="search", arguments={"q": "test"})],
            stop_reason="tool_use",
        )
        result = AnthropicProvider._convert_message(msg)
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["id"] == "tc-1"
        assert result["content"][0]["name"] == "search"
        assert result["content"][0]["input"] == {"q": "test"}

    def test_convert_tool_result(self):
        msg = ToolResultMessage(
            tool_call_id="tc-1",
            tool_name="search",
            content=[TextContent(text="result")],
        )
        result = AnthropicProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "tc-1"


class TestAnthropicToolConversion:
    def test_convert_tool_definition(self):
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        result = AnthropicProvider._convert_tool(td)
        assert result["name"] == "search"
        assert result["description"] == "Search the web"
        assert result["input_schema"]["type"] == "object"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/providers/test_anthropic.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement AnthropicProvider**

Create `cubepi/providers/anthropic.py`:

```python
from __future__ import annotations

import asyncio
import time
from typing import Any

from cubepi.providers.base import (
    AssistantMessage,
    Content,
    ImageContent,
    Message,
    MessageStream,
    Model,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class AnthropicProvider:
    def __init__(self, *, api_key: str | None = None) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        thinking: ThinkingLevel = "off",
        signal: asyncio.Event | None = None,
    ) -> MessageStream:
        ms = MessageStream()

        api_messages = [self._convert_message(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": model.id,
            "messages": api_messages,
            "max_tokens": model.max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]
        if thinking != "off":
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget(thinking, model)}

        async def _produce() -> None:
            try:
                async with self._client.messages.stream(**kwargs) as stream:
                    partial = AssistantMessage(content=[], usage=Usage(), timestamp=time.time())
                    ms.push(StreamEvent(type="start", partial=partial.model_copy(deep=True)))

                    async for event in stream:
                        if signal and signal.is_set():
                            aborted = partial.model_copy(update={
                                "stop_reason": "aborted",
                                "error_message": "Request was aborted",
                            })
                            ms.push(StreamEvent(type="error", error_message="Request was aborted"))
                            ms.set_result(aborted)
                            return

                        self._handle_event(event, partial, ms)

                    final_msg = stream.get_final_message()
                    result = self._convert_response(final_msg)
                    ms.push(StreamEvent(type="done"))
                    ms.set_result(result)

            except Exception as exc:
                error_msg = AssistantMessage(
                    content=[],
                    stop_reason="error",
                    error_message=str(exc),
                    usage=Usage(),
                    timestamp=time.time(),
                )
                ms.push(StreamEvent(type="error", error_message=str(exc)))
                ms.set_result(error_msg)

        asyncio.create_task(_produce())
        return ms

    @staticmethod
    def _convert_message(msg: Message) -> dict[str, Any]:
        if isinstance(msg, UserMessage):
            content = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    content.append({"type": "text", "text": c.text})
                elif isinstance(c, ImageContent):
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": c.media_type, "data": c.source},
                    })
            return {"role": "user", "content": content}

        elif isinstance(msg, AssistantMessage):
            content = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    content.append({"type": "text", "text": c.text})
                elif isinstance(c, ThinkingContent):
                    content.append({"type": "thinking", "thinking": c.thinking})
                elif isinstance(c, ToolCall):
                    content.append({
                        "type": "tool_use",
                        "id": c.id,
                        "name": c.name,
                        "input": c.arguments,
                    })
            return {"role": "assistant", "content": content}

        elif isinstance(msg, ToolResultMessage):
            tool_content = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    tool_content.append({"type": "text", "text": c.text})
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": tool_content,
                    "is_error": msg.is_error,
                }],
            }

        return {"role": "user", "content": []}

    @staticmethod
    def _convert_tool(td: ToolDefinition) -> dict[str, Any]:
        return {
            "name": td.name,
            "description": td.description,
            "input_schema": td.parameters,
        }

    @staticmethod
    def _thinking_budget(level: ThinkingLevel, model: Model) -> int:
        budgets = {
            "minimal": 1024,
            "low": 2048,
            "medium": 4096,
            "high": 8192,
            "xhigh": 16384,
        }
        return budgets.get(level, 4096)

    def _handle_event(self, event: Any, partial: AssistantMessage, ms: MessageStream) -> None:
        etype = getattr(event, "type", "")
        if etype == "content_block_start":
            block = event.content_block
            if block.type == "text":
                partial.content.append(TextContent(text=""))
                ms.push(StreamEvent(type="text_start", partial=partial.model_copy(deep=True)))
            elif block.type == "thinking":
                partial.content.append(ThinkingContent(thinking=""))
                ms.push(StreamEvent(type="thinking_start", partial=partial.model_copy(deep=True)))
            elif block.type == "tool_use":
                partial.content.append(ToolCall(id=block.id, name=block.name, arguments={}))
                ms.push(StreamEvent(type="toolcall_start", partial=partial.model_copy(deep=True)))
        elif etype == "content_block_delta":
            delta = event.delta
            if hasattr(delta, "text"):
                if partial.content and isinstance(partial.content[-1], TextContent):
                    partial.content[-1] = TextContent(text=partial.content[-1].text + delta.text)
                ms.push(StreamEvent(type="text_delta", delta=delta.text, partial=partial.model_copy(deep=True)))
            elif hasattr(delta, "thinking"):
                if partial.content and isinstance(partial.content[-1], ThinkingContent):
                    partial.content[-1] = ThinkingContent(thinking=partial.content[-1].thinking + delta.thinking)
                ms.push(StreamEvent(type="thinking_delta", delta=delta.thinking, partial=partial.model_copy(deep=True)))
            elif hasattr(delta, "partial_json"):
                ms.push(StreamEvent(type="toolcall_delta", delta=delta.partial_json, partial=partial.model_copy(deep=True)))
        elif etype == "content_block_stop":
            if partial.content:
                last = partial.content[-1]
                if isinstance(last, TextContent):
                    ms.push(StreamEvent(type="text_end", partial=partial.model_copy(deep=True)))
                elif isinstance(last, ThinkingContent):
                    ms.push(StreamEvent(type="thinking_end", partial=partial.model_copy(deep=True)))
                elif isinstance(last, ToolCall):
                    ms.push(StreamEvent(type="toolcall_end", partial=partial.model_copy(deep=True)))

    @staticmethod
    def _convert_response(response: Any) -> AssistantMessage:
        content: list[Any] = []
        for block in response.content:
            if block.type == "text":
                content.append(TextContent(text=block.text))
            elif block.type == "thinking":
                content.append(ThinkingContent(thinking=block.thinking))
            elif block.type == "tool_use":
                content.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        stop_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_use",
            "max_tokens": "length",
        }

        return AssistantMessage(
            content=content,
            stop_reason=stop_reason_map.get(response.stop_reason, response.stop_reason),
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            ),
            timestamp=time.time(),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/providers/test_anthropic.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cubepi/providers/anthropic.py tests/providers/test_anthropic.py
git commit -m "feat: add AnthropicProvider with streaming support"
```

---

### Task 11: OpenAIProvider

**Files:**
- Create: `cubepi/providers/openai.py`
- Create: `tests/providers/test_openai.py`

- [ ] **Step 1: Write OpenAIProvider unit tests**

Create `tests/providers/test_openai.py`:

```python
from cubepi.providers.openai import OpenAIProvider
from cubepi.providers.base import (
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)


class TestOpenAIMessageConversion:
    def test_convert_user_message(self):
        msg = UserMessage(content=[TextContent(text="hello")])
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "hello"

    def test_convert_assistant_message(self):
        msg = AssistantMessage(content=[TextContent(text="hi")])
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "hi"

    def test_convert_assistant_with_tool_calls(self):
        msg = AssistantMessage(
            content=[ToolCall(id="tc-1", name="search", arguments={"q": "test"})],
            stop_reason="tool_use",
        )
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "tc-1"
        assert result["tool_calls"][0]["function"]["name"] == "search"

    def test_convert_tool_result(self):
        msg = ToolResultMessage(
            tool_call_id="tc-1",
            tool_name="search",
            content=[TextContent(text="result")],
        )
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "tc-1"
        assert result["content"] == "result"


class TestOpenAIToolConversion:
    def test_convert_tool_definition(self):
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        result = OpenAIProvider._convert_tool(td)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["parameters"]["type"] == "object"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/providers/test_openai.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement OpenAIProvider**

Create `cubepi/providers/openai.py`:

```python
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from cubepi.providers.base import (
    AssistantMessage,
    ImageContent,
    Message,
    MessageStream,
    Model,
    StreamEvent,
    TextContent,
    ThinkingLevel,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class OpenAIProvider:
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        import openai

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        thinking: ThinkingLevel = "off",
        signal: asyncio.Event | None = None,
    ) -> MessageStream:
        ms = MessageStream()

        api_messages: list[dict[str, Any]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(self._convert_message(m) for m in messages)

        kwargs: dict[str, Any] = {
            "model": model.id,
            "messages": api_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        async def _produce() -> None:
            try:
                response = await self._client.chat.completions.create(**kwargs)
                partial = AssistantMessage(content=[], usage=Usage(), timestamp=time.time())
                ms.push(StreamEvent(type="start", partial=partial.model_copy(deep=True)))

                current_text = ""
                tool_calls_in_progress: dict[int, dict[str, Any]] = {}
                text_started = False

                async for chunk in response:
                    if signal and signal.is_set():
                        aborted = partial.model_copy(update={
                            "stop_reason": "aborted",
                            "error_message": "Request was aborted",
                        })
                        ms.push(StreamEvent(type="error", error_message="Request was aborted"))
                        ms.set_result(aborted)
                        return

                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    if delta.content:
                        if not text_started:
                            partial.content.append(TextContent(text=""))
                            ms.push(StreamEvent(type="text_start", partial=partial.model_copy(deep=True)))
                            text_started = True
                        current_text += delta.content
                        if partial.content and isinstance(partial.content[-1], TextContent):
                            partial.content[-1] = TextContent(text=current_text)
                        ms.push(StreamEvent(type="text_delta", delta=delta.content, partial=partial.model_copy(deep=True)))

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_in_progress:
                                if text_started:
                                    ms.push(StreamEvent(type="text_end", partial=partial.model_copy(deep=True)))
                                    text_started = False
                                tool_calls_in_progress[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": tc_delta.function.name if tc_delta.function else "",
                                    "arguments": "",
                                }
                                partial.content.append(ToolCall(
                                    id=tool_calls_in_progress[idx]["id"],
                                    name=tool_calls_in_progress[idx]["name"],
                                    arguments={},
                                ))
                                ms.push(StreamEvent(type="toolcall_start", partial=partial.model_copy(deep=True)))
                            if tc_delta.function and tc_delta.function.arguments:
                                tool_calls_in_progress[idx]["arguments"] += tc_delta.function.arguments
                                ms.push(StreamEvent(
                                    type="toolcall_delta",
                                    delta=tc_delta.function.arguments,
                                    partial=partial.model_copy(deep=True),
                                ))

                    finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
                    if finish_reason:
                        if text_started:
                            ms.push(StreamEvent(type="text_end", partial=partial.model_copy(deep=True)))

                        for idx, tc_data in tool_calls_in_progress.items():
                            try:
                                args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            for i, c in enumerate(partial.content):
                                if isinstance(c, ToolCall) and c.id == tc_data["id"]:
                                    partial.content[i] = ToolCall(
                                        id=tc_data["id"], name=tc_data["name"], arguments=args,
                                    )
                            ms.push(StreamEvent(type="toolcall_end", partial=partial.model_copy(deep=True)))

                        stop_map = {"stop": "stop", "tool_calls": "tool_use", "length": "length"}
                        final = partial.model_copy(update={
                            "stop_reason": stop_map.get(finish_reason, finish_reason),
                        })
                        ms.push(StreamEvent(type="done"))
                        ms.set_result(final)
                        return

                ms.push(StreamEvent(type="done"))
                ms.set_result(partial)

            except Exception as exc:
                error_msg = AssistantMessage(
                    content=[],
                    stop_reason="error",
                    error_message=str(exc),
                    usage=Usage(),
                    timestamp=time.time(),
                )
                ms.push(StreamEvent(type="error", error_message=str(exc)))
                ms.set_result(error_msg)

        asyncio.create_task(_produce())
        return ms

    @staticmethod
    def _convert_message(msg: Message) -> dict[str, Any]:
        if isinstance(msg, UserMessage):
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            return {"role": "user", "content": "\n".join(text_parts)}

        elif isinstance(msg, AssistantMessage):
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            tool_calls = [c for c in msg.content if isinstance(c, ToolCall)]

            result: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                result["content"] = "\n".join(text_parts)
            if tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
            return result

        elif isinstance(msg, ToolResultMessage):
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": "\n".join(text_parts),
            }

        return {"role": "user", "content": ""}

    @staticmethod
    def _convert_tool(td: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            },
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/providers/test_openai.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cubepi/providers/openai.py tests/providers/test_openai.py
git commit -m "feat: add OpenAIProvider with streaming support"
```

---

### Task 12: Package Re-exports and Final Integration

**Files:**
- Modify: `cubepi/__init__.py`
- Modify: `cubepi/providers/__init__.py`

- [ ] **Step 1: Update top-level `__init__.py`**

```python
"""cubepi — Pythonic async-native agent framework."""

from cubepi.agent import Agent, AgentState, AgentTool, AgentToolResult, run_agent_loop, run_agent_loop_continue
from cubepi.middleware import Middleware, compose_middleware
from cubepi.providers import (
    AssistantMessage,
    Message,
    MessageStream,
    Model,
    Provider,
    StreamEvent,
    TextContent,
    ThinkingLevel,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)

__all__ = [
    "Agent",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "AssistantMessage",
    "Message",
    "MessageStream",
    "Middleware",
    "Model",
    "Provider",
    "StreamEvent",
    "TextContent",
    "ThinkingLevel",
    "ToolCall",
    "ToolDefinition",
    "ToolResultMessage",
    "UserMessage",
    "compose_middleware",
    "run_agent_loop",
    "run_agent_loop_continue",
]
```

- [ ] **Step 2: Update providers `__init__.py` to include all providers**

Add to the existing `cubepi/providers/__init__.py`:

```python
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_text, faux_thinking, faux_tool_call

# Lazy imports for optional providers
def get_anthropic_provider():
    from cubepi.providers.anthropic import AnthropicProvider
    return AnthropicProvider

def get_openai_provider():
    from cubepi.providers.openai import OpenAIProvider
    return OpenAIProvider
```

Add `FauxProvider`, `faux_assistant_message`, `faux_text`, `faux_thinking`, `faux_tool_call` to `__all__`.

- [ ] **Step 3: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add cubepi/__init__.py cubepi/providers/__init__.py
git commit -m "feat: add package re-exports and finalize public API"
```

---

### Task 13: Full Test Suite Run and Coverage

- [ ] **Step 1: Run full test suite with coverage**

```bash
pytest tests/ -v --cov=cubepi --cov-report=term-missing
```

Expected: All tests PASS, coverage report shows each module.

- [ ] **Step 2: Fix any failures**

If any tests fail, fix the underlying issue and re-run.

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: resolve test suite issues"
```
