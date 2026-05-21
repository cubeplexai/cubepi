# cubepi Improvement Plan (P0 + P1 + P2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring cubepi's type safety, feature completeness, and code quality up to parity with pi-agent-core across 11 items (P0–P2).

**Architecture:** Changes are layered bottom-up: first fix the foundation types in `providers/base.py` and `agent/types.py` (P0), then wire missing features through the agent layer (P1), then clean up code smells (P2). Each task is independently committable and testable.

**Tech Stack:** Python 3.11+, Pydantic v2, asyncio, pytest-asyncio

**Baseline:** 265 tests pass. Run `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q` to verify.

---

## File Map

### Files to modify

| File | What changes |
|---|---|
| `cubepi/providers/base.py` | Add `content_index` to `StreamEvent`, add `provider_id`/`model_id`/`response_id` to `AssistantMessage`, add `details` to `ToolResultMessage`, rename `_invoke_on_payload`→`invoke_on_payload` / `_invoke_on_response`→`invoke_on_response`, store task ref in `MessageStream` |
| `cubepi/agent/types.py` | Replace `Any` with `Message` union in event types, `AgentContext`, `ShouldStopAfterTurnContext`; add `emit_event` helper |
| `cubepi/agent/agent.py` | Type callbacks/state, integrate checkpointer, remove `hasattr` checks |
| `cubepi/agent/loop.py` | Type parameters, remove local `_emit`, add initial steering poll |
| `cubepi/agent/tools.py` | Remove local `_emit`, pass `details` in `_make_tool_result_message` |
| `cubepi/providers/anthropic.py` | Add `content_index` to events, fill `provider_id`/`model_id`, rename imports |
| `cubepi/providers/openai.py` | Add `content_index` to events, fill metadata, rename imports, add image support |
| `cubepi/providers/openai_responses.py` | Add `content_index` to events, fill metadata, remove dead code |
| `cubepi/providers/faux.py` | Add `content_index` to events |

### Files to create

| File | Purpose |
|---|---|
| `tests/providers/test_content_index.py` | Tests for `content_index` on `StreamEvent` |
| `tests/agent/test_checkpointer_integration.py` | Tests for checkpointer wired into Agent |

### Existing test files to modify

| File | What changes |
|---|---|
| `tests/providers/test_base.py` | Add tests for new `AssistantMessage` fields, `ToolResultMessage.details` |
| `tests/agent/test_loop.py` | Add test for initial steering poll |
| `tests/agent/test_tools.py` | Add test for `details` propagation |
| `tests/providers/test_faux.py` | Update assertions for `content_index` |
| `tests/providers/test_openai.py` | Add image conversion test |

---

## Task 1: Add `content_index` to `StreamEvent`

**Files:**
- Modify: `cubepi/providers/base.py:148-165`
- Test: `tests/providers/test_base.py`

- [ ] **Step 1: Write the failing test**

In `tests/providers/test_base.py`, add:

```python
class TestStreamEvent:
    def test_content_index_default_none(self):
        event = StreamEvent(type="text_delta", delta="hi")
        assert event.content_index is None

    def test_content_index_set(self):
        event = StreamEvent(type="text_start", content_index=0)
        assert event.content_index == 0

    def test_content_index_on_start_event(self):
        event = StreamEvent(type="start")
        assert event.content_index is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_base.py::TestStreamEvent -v`
Expected: FAIL with `unexpected keyword argument 'content_index'`

- [ ] **Step 3: Add `content_index` field to `StreamEvent`**

In `cubepi/providers/base.py`, change:

```python
class StreamEvent(BaseModel):
    type: Literal[
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
        "error",
    ]
    content_index: int | None = None
    delta: str | None = None
    partial: AssistantMessage | None = None
    error_message: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/providers/test_base.py::TestStreamEvent -v`
Expected: PASS

- [ ] **Step 5: Run full suite to verify no regressions**

Run: `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q`
Expected: All pass (existing tests don't assert `content_index is absent`)

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/base.py tests/providers/test_base.py
git commit -m "feat: add content_index field to StreamEvent"
```

---

## Task 2: Fill `content_index` in AnthropicProvider

**Files:**
- Modify: `cubepi/providers/anthropic.py:259-347`
- Test: `tests/providers/test_content_index.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/providers/test_content_index.py`:

```python
from cubepi.providers.base import StreamEvent
from cubepi.providers.faux import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from cubepi.providers.base import Model


def make_model() -> Model:
    return Model(id="faux-1", provider="faux")


class TestFauxContentIndex:
    async def test_text_events_have_content_index(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Hello")])
        stream = await provider.stream(make_model(), [])

        events: list[StreamEvent] = []
        async for event in stream:
            events.append(event)

        text_events = [e for e in events if e.type in ("text_start", "text_delta", "text_end")]
        assert len(text_events) >= 2
        for e in text_events:
            assert e.content_index == 0

    async def test_thinking_then_text_indices(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message([faux_thinking("hmm"), faux_text("answer")])
        ])
        stream = await provider.stream(make_model(), [])

        events: list[StreamEvent] = []
        async for event in stream:
            events.append(event)

        thinking_events = [e for e in events if e.type.startswith("thinking_")]
        text_events = [e for e in events if e.type.startswith("text_")]
        for e in thinking_events:
            assert e.content_index == 0
        for e in text_events:
            assert e.content_index == 1

    async def test_tool_call_content_index(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message([
                faux_text("Let me help"),
                faux_tool_call("search", {"q": "test"}),
            ])
        ])
        stream = await provider.stream(make_model(), [])

        events: list[StreamEvent] = []
        async for event in stream:
            events.append(event)

        text_events = [e for e in events if e.type.startswith("text_")]
        tool_events = [e for e in events if e.type.startswith("toolcall_")]
        for e in text_events:
            assert e.content_index == 0
        for e in tool_events:
            assert e.content_index == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_content_index.py -v`
Expected: FAIL — `content_index` is `None` everywhere

- [ ] **Step 3: Update FauxProvider to emit `content_index`**

In `cubepi/providers/faux.py`, in `_stream_with_deltas`, track a block counter. Before each block loop (ThinkingContent, TextContent, ToolCall), compute `block_idx = len(partial.content) - 1` after appending the empty placeholder, then pass `content_index=block_idx` to every `StreamEvent` within that block.

For **ThinkingContent** blocks (around line 320-353), after `partial.content.append(ThinkingContent(...))`:
```python
block_idx = len(partial.content) - 1
```
Then every `StreamEvent` in that block gets `content_index=block_idx`.

For **TextContent** blocks (around line 355-385), same pattern.

For **ToolCall** blocks (around line 387-424), same pattern.

The `"start"` event keeps `content_index=None` (it's a message-level event, not a content-block event).

Full replacement for the block iteration in `_stream_with_deltas` (the `for block in message.content:` loop body):

```python
if isinstance(block, ThinkingContent):
    partial.content.append(ThinkingContent(thinking=""))
    block_idx = len(partial.content) - 1
    stream.push(
        StreamEvent(
            type="thinking_start",
            content_index=block_idx,
            partial=partial.model_copy(deep=True),
        )
    )
    for chunk in _split_by_token_size(block.thinking, self._min, self._max):
        await self._schedule_chunk(chunk)
        if signal and signal.is_set():
            aborted = self._make_aborted(partial)
            stream.push(
                StreamEvent(
                    type="error", error_message="Request was aborted"
                )
            )
            stream.set_result(aborted)
            return
        last = partial.content[-1]
        if isinstance(last, ThinkingContent):
            partial.content[-1] = ThinkingContent(
                thinking=last.thinking + chunk
            )
        stream.push(
            StreamEvent(
                type="thinking_delta",
                content_index=block_idx,
                delta=chunk,
                partial=partial.model_copy(deep=True),
            )
        )
    stream.push(
        StreamEvent(
            type="thinking_end",
            content_index=block_idx,
            partial=partial.model_copy(deep=True),
        )
    )

elif isinstance(block, TextContent):
    partial.content.append(TextContent(text=""))
    block_idx = len(partial.content) - 1
    stream.push(
        StreamEvent(
            type="text_start",
            content_index=block_idx,
            partial=partial.model_copy(deep=True),
        )
    )
    for chunk in _split_by_token_size(block.text, self._min, self._max):
        await self._schedule_chunk(chunk)
        if signal and signal.is_set():
            aborted = self._make_aborted(partial)
            stream.push(
                StreamEvent(
                    type="error", error_message="Request was aborted"
                )
            )
            stream.set_result(aborted)
            return
        last = partial.content[-1]
        if isinstance(last, TextContent):
            partial.content[-1] = TextContent(text=last.text + chunk)
        stream.push(
            StreamEvent(
                type="text_delta",
                content_index=block_idx,
                delta=chunk,
                partial=partial.model_copy(deep=True),
            )
        )
    stream.push(
        StreamEvent(
            type="text_end",
            content_index=block_idx,
            partial=partial.model_copy(deep=True),
        )
    )

elif isinstance(block, ToolCall):
    partial.content.append(
        ToolCall(id=block.id, name=block.name, arguments={})
    )
    block_idx = len(partial.content) - 1
    stream.push(
        StreamEvent(
            type="toolcall_start",
            content_index=block_idx,
            partial=partial.model_copy(deep=True),
        )
    )
    json_str = json.dumps(block.arguments)
    for chunk in _split_by_token_size(json_str, self._min, self._max):
        await self._schedule_chunk(chunk)
        if signal and signal.is_set():
            aborted = self._make_aborted(partial)
            stream.push(
                StreamEvent(
                    type="error", error_message="Request was aborted"
                )
            )
            stream.set_result(aborted)
            return
        stream.push(
            StreamEvent(
                type="toolcall_delta",
                content_index=block_idx,
                delta=chunk,
                partial=partial.model_copy(deep=True),
            )
        )
    last = partial.content[-1]
    if isinstance(last, ToolCall):
        partial.content[-1] = ToolCall(
            id=block.id, name=block.name, arguments=block.arguments
        )
    stream.push(
        StreamEvent(
            type="toolcall_end",
            content_index=block_idx,
            partial=partial.model_copy(deep=True),
        )
    )
```

- [ ] **Step 4: Update AnthropicProvider to emit `content_index`**

In `cubepi/providers/anthropic.py`, `_handle_event` method: Anthropic emits a block index via `event.index` on `content_block_start`, `content_block_delta`, `content_block_stop`. Use it:

In `_handle_event`, the Anthropic SDK provides `event.index` on `content_block_start`/`content_block_delta`/`content_block_stop`. Pass `content_index=event.index` to every `StreamEvent`. For example, the `content_block_start` handler:

```python
if etype == "content_block_start":
    block = event.content_block
    idx = getattr(event, "index", len(partial.content))
    if block.type == "text":
        partial.content.append(TextContent(text=""))
        ms.push(
            StreamEvent(
                type="text_start",
                content_index=idx,
                partial=partial.model_copy(deep=True),
            )
        )
    elif block.type == "thinking":
        partial.content.append(ThinkingContent(thinking=""))
        ms.push(
            StreamEvent(
                type="thinking_start",
                content_index=idx,
                partial=partial.model_copy(deep=True),
            )
        )
    elif block.type == "tool_use":
        partial.content.append(
            ToolCall(id=block.id, name=block.name, arguments={})
        )
        ms.push(
            StreamEvent(
                type="toolcall_start",
                content_index=idx,
                partial=partial.model_copy(deep=True),
            )
        )
```

Apply the same `idx = getattr(event, "index", ...)` pattern to `content_block_delta` and `content_block_stop` events.

- [ ] **Step 5: Update OpenAIProvider to emit `content_index`**

In `cubepi/providers/openai.py`, track the content index. Text always starts at index 0 (or after any prior blocks). Tool calls use the index computed from `len(partial.content) - 1` after appending.

Key changes in `_produce()`:
- After `partial.content.append(TextContent(text=""))`, compute `text_idx = len(partial.content) - 1` and pass `content_index=text_idx` to all text events.
- After `partial.content.append(ToolCall(...))`, compute `tc_idx = len(partial.content) - 1` and pass `content_index=tc_idx` to toolcall_start/delta/end events.

- [ ] **Step 6: Update OpenAIResponsesProvider to emit `content_index`**

In `cubepi/providers/openai_responses.py`, after each `partial.content.append(...)`, compute `block_idx = len(partial.content) - 1`. Pass `content_index=block_idx` to the corresponding start/delta/end events.

- [ ] **Step 7: Run tests**

Run: `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add cubepi/providers/base.py cubepi/providers/faux.py cubepi/providers/anthropic.py cubepi/providers/openai.py cubepi/providers/openai_responses.py tests/providers/test_content_index.py
git commit -m "feat: populate content_index on all provider StreamEvents"
```

---

## Task 3: Add provider metadata to `AssistantMessage`

**Files:**
- Modify: `cubepi/providers/base.py:121-128`
- Modify: `cubepi/providers/anthropic.py:349-382`
- Modify: `cubepi/providers/openai.py` (in `_produce`)
- Modify: `cubepi/providers/openai_responses.py` (in `_produce`)
- Test: `tests/providers/test_base.py`

- [ ] **Step 1: Write the failing test**

In `tests/providers/test_base.py`, add:

```python
class TestAssistantMessageMetadata:
    def test_default_metadata_fields(self):
        msg = AssistantMessage(content=[])
        assert msg.provider_id == ""
        assert msg.model_id == ""
        assert msg.response_id is None

    def test_metadata_fields_set(self):
        msg = AssistantMessage(
            content=[],
            provider_id="anthropic",
            model_id="claude-sonnet-4-20250514",
            response_id="msg_abc123",
        )
        assert msg.provider_id == "anthropic"
        assert msg.model_id == "claude-sonnet-4-20250514"
        assert msg.response_id == "msg_abc123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_base.py::TestAssistantMessageMetadata -v`
Expected: FAIL

- [ ] **Step 3: Add fields to `AssistantMessage`**

In `cubepi/providers/base.py`:

```python
class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[Content | ThinkingContent | ToolCall]
    stop_reason: str = "stop"
    error_message: str | None = None
    usage: Usage | None = None
    provider_id: str = ""
    model_id: str = ""
    response_id: str | None = None
    timestamp: float | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/providers/test_base.py::TestAssistantMessageMetadata -v`
Expected: PASS

- [ ] **Step 5: Fill metadata in AnthropicProvider**

In `cubepi/providers/anthropic.py`, update `_convert_response` to accept `model: Model` and fill the fields:

Change `_convert_response` signature from `@staticmethod` to an instance method (or pass model):

```python
@staticmethod
def _convert_response(response: Any, model: Model) -> AssistantMessage:
    # ... existing content parsing ...
    return AssistantMessage(
        content=content,
        stop_reason=stop_reason_map.get(response.stop_reason, response.stop_reason),
        usage=Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        ),
        provider_id=model.provider,
        model_id=model.id,
        response_id=getattr(response, "id", None),
        timestamp=time.time(),
    )
```

Update the call site in `_produce()` from `self._convert_response(final_msg)` to `self._convert_response(final_msg, model)`.

Also fill metadata on the partial message created at stream start:

```python
partial = AssistantMessage(
    content=[], usage=Usage(),
    provider_id=model.provider, model_id=model.id,
    timestamp=time.time(),
)
```

- [ ] **Step 6: Fill metadata in OpenAIProvider and OpenAIResponsesProvider**

Apply the same pattern: set `provider_id=model.provider, model_id=model.id` on the initial `partial` and final messages in both providers.

- [ ] **Step 7: Run full suite**

Run: `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add cubepi/providers/base.py cubepi/providers/anthropic.py cubepi/providers/openai.py cubepi/providers/openai_responses.py tests/providers/test_base.py
git commit -m "feat: add provider_id, model_id, response_id to AssistantMessage"
```

---

## Task 4: Replace `Any` with typed `Message` union in agent layer

**Files:**
- Modify: `cubepi/agent/types.py`
- Modify: `cubepi/agent/agent.py`
- Modify: `cubepi/agent/loop.py`
- Modify: `cubepi/checkpointer/base.py`
- Test: `tests/agent/test_types.py`

- [ ] **Step 1: Write the failing test**

In `tests/agent/test_types.py`, add:

```python
from cubepi.agent.types import AgentContext, AgentEndEvent, TurnEndEvent, MessageStartEvent
from cubepi.providers.base import AssistantMessage, TextContent, ToolResultMessage, UserMessage


class TestTypedMessages:
    def test_agent_context_accepts_message_union(self):
        ctx = AgentContext(
            system_prompt="test",
            messages=[
                UserMessage(content=[TextContent(text="hi")]),
                AssistantMessage(content=[TextContent(text="hello")]),
            ],
        )
        assert len(ctx.messages) == 2

    def test_agent_end_event_typed_messages(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        event = AgentEndEvent(messages=[msg])
        assert event.messages[0].role == "user"

    def test_turn_end_event_typed_message(self):
        msg = AssistantMessage(content=[TextContent(text="done")])
        event = TurnEndEvent(message=msg, tool_results=[])
        assert event.message.role == "assistant"

    def test_message_start_event_typed(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        event = MessageStartEvent(message=msg)
        assert event.message.role == "user"
```

- [ ] **Step 2: Run test — these should pass already (Any accepts anything)**

Run: `pytest tests/agent/test_types.py::TestTypedMessages -v`
Expected: PASS (since `Any` accepts `Message` types)

This test validates the new types work after we change them. The real value is that *invalid* types would now fail mypy.

- [ ] **Step 3: Update `agent/types.py` — replace `Any` with `Message`**

```python
from cubepi.providers.base import (
    AssistantMessage,
    Content,
    Message,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
)

# ... AgentToolResult unchanged ...

@dataclass
class AgentContext:
    system_prompt: str
    messages: list[Message]
    tools: list[AgentTool] | None = None

# ... BeforeToolCallContext, AfterToolCallContext unchanged (already typed) ...

@dataclass
class ShouldStopAfterTurnContext:
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[Message]

# Event types — replace Any with Message:

class AgentEndEvent(BaseModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[Message]

class TurnEndEvent(BaseModel):
    type: Literal["turn_end"] = "turn_end"
    message: AssistantMessage
    tool_results: list[ToolResultMessage]

class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: Message

class MessageUpdateEvent(BaseModel):
    type: Literal["message_update"] = "message_update"
    message: AssistantMessage
    stream_event: StreamEvent

class MessageEndEvent(BaseModel):
    type: Literal["message_end"] = "message_end"
    message: Message
```

- [ ] **Step 4: Update `agent/agent.py` — type state and callbacks**

Replace `list[Any]` with `list[Message]` in `AgentState`:

```python
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    Model,
    OnPayloadCallback,
    OnResponseCallback,
    Provider,
    StreamOptions,
    TextContent,
    ThinkingLevel,
    ToolResultMessage,
    Usage,
    UserMessage,
)

# ...

@dataclass
class AgentState:
    system_prompt: str = ""
    model: Model = field(
        default_factory=lambda: Model(id="unknown", provider="unknown")
    )
    thinking: ThinkingLevel = "off"
    is_streaming: bool = False
    streaming_message: Message | None = None
    error_message: str | None = None
    _tools: list[AgentTool] = field(default_factory=list)
    _messages: list[Message] = field(default_factory=list)
    _pending_tool_calls: set[str] = field(default_factory=set)

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self._messages = list(value)

    # ... tools and pending_tool_calls unchanged ...
```

Update `_default_convert_to_llm`:

```python
def _default_convert_to_llm(messages: list[Message]) -> list[Message]:
    return [
        m for m in messages
        if m.role in ("user", "assistant", "tool_result")
    ]
```

Update `_MessageQueue` to use `Message`:

```python
class _MessageQueue:
    def __init__(self, mode: str = "one-at-a-time") -> None:
        self.mode = mode
        self._messages: list[Message] = []

    def enqueue(self, message: Message) -> None:
        self._messages.append(message)

    # ... rest unchanged, just change return types to list[Message] ...
```

Update `Agent` class signatures:

```python
class Agent:
    def __init__(
        self,
        *,
        provider: Provider,
        model: Model,
        system_prompt: str = "",
        tools: list[AgentTool] | None = None,
        thinking: ThinkingLevel = "off",
        convert_to_llm: Callable[[list[Message]], list[Message]] | None = None,
        transform_context: Callable[..., Awaitable[list[Message]]] | None = None,
        before_tool_call: Callable[..., Awaitable[BeforeToolCallResult | None]] | None = None,
        after_tool_call: Callable[..., Awaitable[AfterToolCallResult | None]] | None = None,
        should_stop_after_turn: Callable[..., Awaitable[bool]] | None = None,
        # ... rest unchanged ...
    ) -> None:
```

Update `prompt()`:

```python
async def prompt(self, message: str | Message | list[Message]) -> None:
```

Update `steer()` and `follow_up()`:

```python
def steer(self, message: Message) -> None:
def follow_up(self, message: Message) -> None:
```

Remove `hasattr` checks in `_process_event`:

```python
elif event.type == "turn_end":
    if isinstance(event.message, AssistantMessage) and event.message.error_message:
        self._state.error_message = event.message.error_message
```

Remove `hasattr` in `resume()`:

```python
last = self._state._messages[-1]
if isinstance(last, AssistantMessage):
    # ...
```

- [ ] **Step 5: Update `agent/loop.py` — type parameters**

Change `list[Any]` to `list[Message]` in `run_agent_loop`, `run_agent_loop_continue`, and `_run_loop` parameter types and return types. Add the `Message` import.

- [ ] **Step 6: Run full suite**

Run: `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add cubepi/agent/types.py cubepi/agent/agent.py cubepi/agent/loop.py tests/agent/test_types.py
git commit -m "refactor: replace Any with typed Message union across agent layer"
```

---

## Task 5: Add `details` to `ToolResultMessage`

**Files:**
- Modify: `cubepi/providers/base.py:130-137`
- Modify: `cubepi/agent/tools.py:60-67`
- Test: `tests/agent/test_tools.py`

- [ ] **Step 1: Write the failing test**

In `tests/agent/test_tools.py`, add:

```python
class TestToolResultDetails:
    async def test_details_propagated_to_tool_result_message(self):
        async def execute_with_details(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(
                content=[TextContent(text="result")],
                details={"execution_time": 42},
            )

        tool = AgentTool(
            name="detailed",
            description="Tool with details",
            parameters=EchoParams,
            execute=execute_with_details,
        )
        context = AgentContext(
            system_prompt="test", messages=[], tools=[tool]
        )
        assistant_msg = AssistantMessage(
            content=[ToolCall(id="tc1", name="detailed", arguments={"value": "x"})],
            stop_reason="tool_use",
        )

        events: list = []
        batch = await execute_tool_calls(
            context,
            assistant_msg,
            emit=lambda e: events.append(e),
        )
        assert len(batch.messages) == 1
        assert batch.messages[0].details == {"execution_time": 42}
```

(Import `AgentToolResult`, `AgentTool`, `AgentContext` from `cubepi.agent.types`, `ToolCall`, `AssistantMessage`, `TextContent` from `cubepi.providers.base`, and `execute_tool_calls` from `cubepi.agent.tools`. Use `EchoParams` from the existing test file or define a local one.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_tools.py::TestToolResultDetails -v`
Expected: FAIL — `ToolResultMessage` has no `details` field, or it's `None`

- [ ] **Step 3: Add `details` field to `ToolResultMessage`**

In `cubepi/providers/base.py`:

```python
class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[Content]
    details: Any = None
    is_error: bool = False
    timestamp: float | None = None
```

Add `Any` to the imports from `typing` if not already there.

- [ ] **Step 4: Pass `details` in `_make_tool_result_message`**

In `cubepi/agent/tools.py`:

```python
def _make_tool_result_message(finalized: _FinalizedOutcome) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        content=finalized.result.content,
        details=finalized.result.details,
        is_error=finalized.is_error,
        timestamp=time.time(),
    )
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/base.py cubepi/agent/tools.py tests/agent/test_tools.py
git commit -m "feat: propagate details from AgentToolResult to ToolResultMessage"
```

---

## Task 6: Initial steering message poll in `_run_loop`

**Files:**
- Modify: `cubepi/agent/loop.py:137-238`
- Test: `tests/agent/test_loop.py`

- [ ] **Step 1: Write the failing test**

In `tests/agent/test_loop.py`, add:

```python
class TestInitialSteeringPoll:
    async def test_steering_messages_polled_before_first_turn(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message("response to steering"),
        ])

        steering_called = False

        async def get_steering():
            nonlocal steering_called
            if not steering_called:
                steering_called = True
                return [make_user_message("steering message")]
            return []

        context = AgentContext(
            system_prompt="test", messages=[], tools=[]
        )
        events: list[AgentEvent] = []

        await run_agent_loop(
            prompts=[make_user_message("initial")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            get_steering_messages=get_steering,
            emit=lambda e: events.append(e),
        )

        # Steering message should appear as a message_start/end event
        # before the assistant response
        message_events = [e for e in events if e.type in ("message_start", "message_end")]
        # Events: initial prompt start/end, steering start/end, assistant start/end, agent_end
        messages_in_order = [
            e.message for e in events if e.type == "message_end"
        ]
        assert len(messages_in_order) >= 3
        # First is the initial prompt, second is steering, third is assistant
        assert messages_in_order[0].role == "user"
        assert messages_in_order[1].role == "user"
        assert messages_in_order[2].role == "assistant"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_loop.py::TestInitialSteeringPoll -v`
Expected: FAIL — steering message not polled at start, so only 2 message_end events (prompt + assistant)

- [ ] **Step 3: Add initial steering poll to `_run_loop`**

In `cubepi/agent/loop.py`, at the beginning of `_run_loop`, after `first_turn = True`:

```python
    opts = stream_options or StreamOptions()
    first_turn = True

    # Poll for steering messages at start (user may have typed while waiting)
    pending_messages: list = []
    if get_steering_messages:
        pending_messages = await get_steering_messages() or []

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls:
            if not first_turn:
                await _emit(emit, TurnStartEvent())
            else:
                first_turn = False

            # Inject pending messages before next assistant response
            if pending_messages:
                for msg in pending_messages:
                    await _emit(emit, MessageStartEvent(message=msg))
                    await _emit(emit, MessageEndEvent(message=msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                pending_messages = []

            message = await _stream_assistant_response(
                # ... unchanged ...
            )
```

And change the steering check after tool execution to populate `pending_messages` instead of injecting directly:

```python
            # Check for steering messages after tool execution
            if get_steering_messages and has_more_tool_calls:
                pending_messages = await get_steering_messages() or []
```

Remove the old inline injection code (the `for msg in steering:` loop that was there before).

- [ ] **Step 4: Run tests**

Run: `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add cubepi/agent/loop.py tests/agent/test_loop.py
git commit -m "fix: poll steering messages at loop start before first assistant turn"
```

---

## Task 7: Integrate Checkpointer into Agent

**Files:**
- Modify: `cubepi/agent/agent.py`
- Create: `tests/agent/test_checkpointer_integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_checkpointer_integration.py`:

```python
from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import Model, TextContent, UserMessage
from cubepi.providers.faux import FauxProvider, faux_assistant_message


def make_model() -> Model:
    return Model(id="faux-1", provider="faux")


class TestCheckpointerIntegration:
    async def test_messages_persisted_on_message_end(self):
        checkpointer = MemoryCheckpointer()
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Hello!")])

        agent = Agent(
            provider=provider,
            model=make_model(),
            checkpointer=checkpointer,
            thread_id="thread-1",
        )
        await agent.prompt("Hi")

        data = await checkpointer.load("thread-1")
        assert data is not None
        assert len(data.messages) == 2  # user + assistant

    async def test_history_restored_on_prompt(self):
        checkpointer = MemoryCheckpointer()
        provider = FauxProvider()

        # First agent session: send a message
        provider.set_responses([faux_assistant_message("First reply")])
        agent1 = Agent(
            provider=provider,
            model=make_model(),
            checkpointer=checkpointer,
            thread_id="thread-1",
        )
        await agent1.prompt("First message")

        # Second agent session: restore history, send another
        provider.set_responses([faux_assistant_message("Second reply")])
        agent2 = Agent(
            provider=provider,
            model=make_model(),
            checkpointer=checkpointer,
            thread_id="thread-1",
        )
        await agent2.prompt("Second message")

        assert len(agent2.state.messages) == 4  # 2 from first + 2 from second

    async def test_no_checkpointer_works_as_before(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Hi")])

        agent = Agent(provider=provider, model=make_model())
        await agent.prompt("Hello")
        assert len(agent.state.messages) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_checkpointer_integration.py -v`
Expected: FAIL — messages not persisted, history not restored

- [ ] **Step 3: Integrate checkpointer into Agent**

In `cubepi/agent/agent.py`:

1. In `prompt()`, before `await self._run_prompt(messages)`, restore history:

```python
async def prompt(self, message: str | Message | list[Message]) -> None:
    if self._state.is_streaming:
        raise RuntimeError(
            "Agent is already processing a prompt. "
            "Use steer() or follow_up() to queue messages."
        )

    # Restore history from checkpointer if this is first prompt
    if self.checkpointer and self.thread_id and not self._state._messages:
        data = await self.checkpointer.load(self.thread_id)
        if data and data.messages:
            self._state._messages = list(data.messages)

    if isinstance(message, str):
        messages = [
            UserMessage(content=[TextContent(text=message)], timestamp=time.time())
        ]
    elif isinstance(message, list):
        messages = message
    else:
        messages = [message]

    await self._run_prompt(messages)
```

2. In `_process_event()`, persist on `message_end`:

```python
elif event.type == "message_end":
    self._state.streaming_message = None
    self._state._messages.append(event.message)
    # Persist to checkpointer
    if self.checkpointer and self.thread_id:
        await self.checkpointer.append(self.thread_id, [event.message])
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add cubepi/agent/agent.py tests/agent/test_checkpointer_integration.py
git commit -m "feat: integrate checkpointer into Agent for message persistence"
```

---

## Task 8: OpenAI Provider image support

**Files:**
- Modify: `cubepi/providers/openai.py:250-254`
- Modify: `cubepi/providers/openai_responses.py:464-475`
- Test: `tests/providers/test_openai.py`

- [ ] **Step 1: Write the failing test**

In `tests/providers/test_openai.py`, add:

```python
from cubepi.providers.openai import OpenAIProvider
from cubepi.providers.openai_responses import OpenAIResponsesProvider
from cubepi.providers.base import ImageContent, TextContent, UserMessage


class TestOpenAIImageConversion:
    def test_user_message_with_image(self):
        msg = UserMessage(content=[
            TextContent(text="What's in this image?"),
            ImageContent(source="base64data", media_type="image/png"),
        ])
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "What's in this image?"}
        assert result["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,base64data"},
        }

    def test_user_message_text_only_stays_simple(self):
        msg = UserMessage(content=[TextContent(text="hello")])
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "hello"


class TestOpenAIResponsesImageConversion:
    def test_user_message_with_image(self):
        msg = UserMessage(content=[
            TextContent(text="Describe this"),
            ImageContent(source="imgdata", media_type="image/jpeg"),
        ])
        result = OpenAIResponsesProvider._build_input([msg])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "input_text", "text": "Describe this"}
        assert content[1] == {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,imgdata",
        }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_openai.py::TestOpenAIImageConversion -v`
Expected: FAIL — images are dropped, content is a string not a list

- [ ] **Step 3: Update `OpenAIProvider._convert_message` for images**

In `cubepi/providers/openai.py`:

```python
@staticmethod
def _convert_message(msg: Message) -> dict[str, Any]:
    if isinstance(msg, UserMessage):
        has_images = any(isinstance(c, ImageContent) for c in msg.content)
        if not has_images:
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            return {"role": "user", "content": "\n".join(text_parts)}
        content: list[dict[str, Any]] = []
        for c in msg.content:
            if isinstance(c, TextContent):
                content.append({"type": "text", "text": c.text})
            elif isinstance(c, ImageContent):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{c.media_type};base64,{c.source}"},
                })
        return {"role": "user", "content": content}

    # ... rest unchanged ...
```

- [ ] **Step 4: Update `OpenAIResponsesProvider._build_input` for images**

In `cubepi/providers/openai_responses.py`, in the `UserMessage` branch of `_build_input`:

```python
if isinstance(msg, UserMessage):
    content: list[dict[str, Any]] = []
    for c in msg.content:
        if isinstance(c, TextContent):
            content.append({"type": "input_text", "text": c.text})
        elif isinstance(c, ImageContent):
            content.append({
                "type": "input_image",
                "image_url": f"data:{c.media_type};base64,{c.source}",
            })
    if content:
        api_input.append({"role": "user", "content": content})
```

Add `ImageContent` to the imports from `cubepi.providers.base`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/openai.py cubepi/providers/openai_responses.py tests/providers/test_openai.py
git commit -m "feat: add image content support to OpenAI providers"
```

---

## Task 9: Extract shared `emit_event` helper (eliminate duplicate `_emit`)

**Files:**
- Modify: `cubepi/agent/types.py`
- Modify: `cubepi/agent/loop.py`
- Modify: `cubepi/agent/tools.py`

- [ ] **Step 1: Add `emit_event` to `agent/types.py`**

At the end of `cubepi/agent/types.py`:

```python
async def emit_event(emit_fn: Callable, event: AgentEvent) -> None:
    result = emit_fn(event)
    if asyncio.iscoroutine(result):
        await result
```

- [ ] **Step 2: Replace `_emit` in `loop.py`**

Remove the local `_emit` definition (lines 28-31). Add `emit_event` to the imports from `cubepi.agent.types`. Replace all `_emit(` calls with `emit_event(` calls throughout the file.

- [ ] **Step 3: Replace `_emit` in `tools.py`**

Remove the local `_emit` definition (lines 70-73). Add `emit_event` to the imports from `cubepi.agent.types`. Replace all `_emit(` calls with `emit_event(` calls throughout the file.

- [ ] **Step 4: Run tests**

Run: `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add cubepi/agent/types.py cubepi/agent/loop.py cubepi/agent/tools.py
git commit -m "refactor: extract shared emit_event helper, remove duplicate _emit"
```

---

## Task 10: Fix fire-and-forget `asyncio.create_task` in `MessageStream`

**Files:**
- Modify: `cubepi/providers/base.py:168-194`
- Modify: `cubepi/providers/anthropic.py:155`
- Modify: `cubepi/providers/openai.py:247`
- Modify: `cubepi/providers/openai_responses.py:443`
- Modify: `cubepi/providers/faux.py:287`
- Test: `tests/providers/test_base.py`

- [ ] **Step 1: Write the failing test**

In `tests/providers/test_base.py`, add:

```python
class TestMessageStreamTaskTracking:
    async def test_attach_task_stores_reference(self):
        ms = MessageStream()

        async def dummy():
            ms.push(StreamEvent(type="done"))
            ms.set_result(AssistantMessage(content=[]))

        task = asyncio.create_task(dummy())
        ms.attach_task(task)
        assert ms._producer_task is task
        await task

    async def test_result_propagates_task_exception(self):
        ms = MessageStream()

        async def failing():
            raise RuntimeError("producer died before pushing error")

        task = asyncio.create_task(failing())
        ms.attach_task(task)

        import pytest
        with pytest.raises(RuntimeError, match="producer died"):
            await ms.result()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_base.py::TestMessageStreamTaskTracking -v`
Expected: FAIL — `attach_task` does not exist

- [ ] **Step 3: Add `attach_task` to `MessageStream`**

In `cubepi/providers/base.py`:

```python
class MessageStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._result_future: asyncio.Future[AssistantMessage] = (
            asyncio.get_running_loop().create_future()
        )
        self._producer_task: asyncio.Task | None = None

    def attach_task(self, task: asyncio.Task) -> None:
        self._producer_task = task
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc and not self._result_future.done():
            self._result_future.set_exception(exc)
            self._queue.put_nowait(None)

    # ... push, set_result, __aiter__, __anext__, result unchanged ...
```

- [ ] **Step 4: Update all providers to use `attach_task`**

In each provider's `stream()` method, change:
```python
asyncio.create_task(_produce())
```
to:
```python
ms.attach_task(asyncio.create_task(_produce()))
```

Files: `anthropic.py:155`, `openai.py:247`, `openai_responses.py:443`, `faux.py:287`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/base.py cubepi/providers/anthropic.py cubepi/providers/openai.py cubepi/providers/openai_responses.py cubepi/providers/faux.py tests/providers/test_base.py
git commit -m "fix: track producer task in MessageStream to prevent silent failures"
```

---

## Task 11: Clean up dead code and rename private helpers

**Files:**
- Modify: `cubepi/providers/openai_responses.py:78-86`
- Modify: `cubepi/providers/base.py:225-249`
- Modify: `cubepi/providers/anthropic.py:24-26`
- Modify: `cubepi/providers/openai.py:24-26`

- [ ] **Step 1: Remove dead code in OpenAI Responses system prompt handling**

In `cubepi/providers/openai_responses.py`, replace lines 78-86:

```python
        if system_prompt:
            role = "developer" if model.reasoning else "system"
            kwargs["instructions"] = system_prompt
            # The instructions param uses system/developer role implicitly.
            # For explicit role control, prepend to input instead.
            kwargs["input"] = [{"role": role, "content": system_prompt}] + api_input

            # Remove instructions since we use input-based system prompt
            del kwargs["instructions"]
```

With:

```python
        if system_prompt:
            role = "developer" if model.reasoning else "system"
            kwargs["input"] = [{"role": role, "content": system_prompt}] + api_input
```

- [ ] **Step 2: Rename `_invoke_on_payload` → `invoke_on_payload` and `_invoke_on_response` → `invoke_on_response`**

In `cubepi/providers/base.py`, rename both functions by removing the leading underscore.

- [ ] **Step 3: Update imports in providers**

In `cubepi/providers/anthropic.py`, change:
```python
_invoke_on_payload,
_invoke_on_response,
```
to:
```python
invoke_on_payload,
invoke_on_response,
```

And update the call sites (`_invoke_on_payload(` → `invoke_on_payload(`).

Same in `cubepi/providers/openai.py`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/ --ignore=tests/checkpointer/test_sqlite.py -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add cubepi/providers/base.py cubepi/providers/anthropic.py cubepi/providers/openai.py cubepi/providers/openai_responses.py
git commit -m "refactor: remove dead code, rename private helpers to public API"
```

---

## Summary

| Task | Spec Item | Priority | What |
|---|---|---|---|
| 1 | §2 | P0 | Add `content_index` to `StreamEvent` |
| 2 | §2 | P0 | Fill `content_index` in all 4 providers |
| 3 | §3 | P0 | Add `provider_id`/`model_id`/`response_id` to `AssistantMessage` |
| 4 | §1 | P0 | Replace `Any` with typed `Message` union |
| 5 | §5 | P1 | Add `details` to `ToolResultMessage` |
| 6 | §6 | P1 | Initial steering poll in `_run_loop` |
| 7 | §4 | P1 | Integrate checkpointer into Agent |
| 8 | §7 | P1 | OpenAI image support |
| 9 | §8 | P2 | Extract shared `emit_event` |
| 10 | §9 | P2 | Fix fire-and-forget task in `MessageStream` |
| 11 | §10+§11 | P2 | Dead code cleanup + rename private helpers |
