# cubepi Framework Design Spec

A Pythonic agent framework inspired by pi-agent-core, designed to replace langgraph in cubebox.

## Goals

- General-purpose Python agent framework, not cubebox-specific
- Built-in multi-provider LLM abstraction (Anthropic, OpenAI)
- Hook callbacks (like pi) + Middleware protocol (new) for extensibility
- Optional Checkpointer with built-in Memory and SQLite implementations
- Streaming-first, async-native

## Module Structure

```
cubepi/
├── __init__.py
├── providers/           # LLM provider abstraction
│   ├── __init__.py
│   ├── base.py          # Provider Protocol, Message types, StreamEvent, MessageStream
│   ├── anthropic.py     # Anthropic implementation
│   ├── openai.py        # OpenAI implementation
│   └── faux.py          # FauxProvider (testing utility, public API)
├── agent/               # Agent runtime core
│   ├── __init__.py
│   ├── agent.py         # Agent class (stateful wrapper)
│   ├── loop.py          # run_agent_loop (stateless core loop)
│   ├── types.py         # AgentEvent, AgentContext, AgentTool, hooks
│   └── tools.py         # Tool execution engine (sequential/parallel)
├── middleware/           # Middleware protocol and composition
│   ├── __init__.py
│   └── base.py          # Middleware Protocol + compose_middleware()
├── checkpointer/        # Optional state persistence
│   ├── __init__.py
│   ├── base.py          # Checkpointer Protocol
│   ├── memory.py        # MemoryCheckpointer (dev/test)
│   └── sqlite.py        # SQLiteCheckpointer (lightweight persistence)
└── py.typed             # PEP 561 marker
```

## Provider Layer

Corresponds to pi-ai. Each provider is a class implementing the `Provider` Protocol.

### Divergences from pi-ai

| pi-ai | cubepi | Reason |
|-------|--------|--------|
| `streamSimple` global function + `Model.api` dispatch | `Provider` instance per backend | More Pythonic, easier to test/mock |
| API key in `SimpleStreamOptions` per call | API key in Provider constructor | Config belongs at init, not per-request |
| TypeBox schema for tool params | Pydantic model → JSON Schema | Pydantic is the Python standard |
| `EventStream<T, R>` class | `MessageStream` (async iterator + `.result()`) | Python async generator pattern |

### Types

All types use Pydantic `BaseModel`.

```python
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

class ModelCost(BaseModel):
    input: float = 0        # per million tokens
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
    content: list[Content | ToolCall]
    stop_reason: str = "stop"  # "stop" | "tool_use" | "length" | "error" | "aborted"
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
```

### Provider Protocol

```python
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

### MessageStream

```python
class MessageStream:
    def __aiter__(self) -> AsyncIterator[StreamEvent]: ...
    async def result(self) -> AssistantMessage: ...
```

### StreamEvent Types

Identical to pi-ai's AssistantMessageEvent:

- `start` — initial partial message
- `text_start` / `text_delta` / `text_end`
- `thinking_start` / `thinking_delta` / `thinking_end`
- `toolcall_start` / `toolcall_delta` / `toolcall_end`
- `done` — normal completion
- `error` — error completion

### ToolDefinition

Schema-only description sent to the LLM (no execution logic):

```python
class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
```

`AgentTool` auto-generates `ToolDefinition` from its Pydantic parameters model.

## Agent Runtime

Corresponds to pi-agent-core's agent.ts + agent-loop.ts.

### AgentMessage Extension

**pi's approach**: TypeScript declaration merging extends `CustomAgentMessages`.

**cubepi's approach**: Generic type parameter.

```python
# Default: AgentMessage = Message
agent = Agent(...)

# Extended with custom messages:
CubeboxMessage = Message | NotificationMessage | ArtifactMessage
agent = Agent[CubeboxMessage](convert_to_llm=my_converter, ...)
```

More explicit than pi's global declaration merging, but safer (no cross-library pollution).

### AgentEvent

Identical to pi's 11 event types, snake_case naming:

```python
AgentEvent = (
    AgentStartEvent | AgentEndEvent |
    TurnStartEvent | TurnEndEvent |
    MessageStartEvent | MessageUpdateEvent | MessageEndEvent |
    ToolExecutionStartEvent | ToolExecutionUpdateEvent | ToolExecutionEndEvent
)
```

Each event is a Pydantic model with `type: Literal[...]` discriminator.

`MessageUpdateEvent.stream_event` corresponds to pi's `assistantMessageEvent` (renamed for clarity).

### AgentTool

```python
class AgentTool(Generic[TParams]):
    name: str
    description: str
    parameters: type[TParams]  # Pydantic model class
    label: str = ""
    execution_mode: Literal["sequential", "parallel"] | None = None

    async def execute(
        self,
        tool_call_id: str,
        params: TParams,
        *,
        signal: asyncio.Event | None = None,
        on_update: Callable[[ToolResult], None] | None = None,
    ) -> ToolResult: ...
```

**Divergences from pi**:
- pi uses TypeBox schema → cubepi uses Pydantic model class (auto JSON Schema generation)
- pi's `prepareArguments` compatibility shim omitted (YAGNI)
- Execution failure via `raise` (same as pi), framework catches and converts to error tool result

### Agent Class

```python
class Agent(Generic[TMessage]):
    def __init__(
        self,
        *,
        provider: Provider,
        model: Model,
        system_prompt: str = "",
        tools: list[AgentTool] | None = None,
        thinking: ThinkingLevel = "off",
        checkpointer: Checkpointer | None = None,
        thread_id: str | None = None,
        convert_to_llm: Callable | None = None,
        transform_context: Callable | None = None,
        before_tool_call: Callable | None = None,
        after_tool_call: Callable | None = None,
        should_stop_after_turn: Callable | None = None,
        steering_mode: Literal["all", "one-at-a-time"] = "one-at-a-time",
        follow_up_mode: Literal["all", "one-at-a-time"] = "one-at-a-time",
        tool_execution: Literal["sequential", "parallel"] = "parallel",
    ): ...

    @property
    def state(self) -> AgentState: ...

    def subscribe(self, listener: AgentListener) -> Callable[[], None]: ...

    async def prompt(self, message: str | TMessage | list[TMessage]) -> None: ...
    async def resume(self) -> None: ...  # pi: continue() — renamed, Python keyword
    def abort(self) -> None: ...
    async def wait_for_idle(self) -> None: ...
    def reset(self) -> None: ...

    def steer(self, message: TMessage) -> None: ...
    def follow_up(self, message: TMessage) -> None: ...
```

**Divergences from pi**:
- `continue()` → `resume()` (Python reserved word)
- pi: `new Agent({ initialState: { model, tools, systemPrompt }, streamFn })` → cubepi: explicit keyword args `Agent(provider=..., model=..., tools=...)`
- Cancel signal: `asyncio.Event` instead of `AbortSignal`
- `subscribe()` returns unsubscribe callable (same as pi)
- Two message queues (steering/follow-up) with configurable drain mode (same as pi)
- Checkpointer and thread_id accepted directly by Agent (pi has no persistence)

### Agent Loop

Stateless core function, identical nested-loop structure to pi's `runLoop`:

```
Outer loop: follow-up message processing
├─ Inner loop: tool execution + steering messages
│  ├─ Stream assistant response (Provider.stream)
│  ├─ Extract tool calls from response
│  ├─ Execute tools (sequential or parallel)
│  ├─ Poll steering messages
│  └─ Repeat if tool calls remain
├─ Check should_stop_after_turn
├─ Poll follow-up messages
└─ Continue or exit
```

Two-level API (same as pi):
- `Agent` class for most use cases
- `run_agent_loop()` / `run_agent_loop_continue()` for custom control

### Hooks

All hooks from pi's `AgentLoopConfig`, mapped 1:1:

| pi hook | cubepi hook | Notes |
|---------|-------------|-------|
| `convertToLlm` | `convert_to_llm` | Required when using custom message types |
| `transformContext` | `transform_context` | Optional, runs before convert_to_llm |
| `beforeToolCall` | `before_tool_call` | Return `BeforeToolCallResult(block=True)` to prevent execution |
| `afterToolCall` | `after_tool_call` | Override content/details/is_error/terminate |
| `shouldStopAfterTurn` | `should_stop_after_turn` | Graceful stop after current turn |
| `getSteeringMessages` | `get_steering_messages` | Inject messages mid-run |
| `getFollowUpMessages` | `get_follow_up_messages` | Inject messages when agent would stop |

Contract: hooks must not raise. Return safe fallback values on failure (same as pi).

## Middleware

New layer not present in pi. Added because cubebox has 11+ middleware classes that need clean composition.

### Middleware Protocol

```python
class Middleware(Protocol[TMessage]):
    async def transform_context(self, messages, signal=None) -> list: ...
    async def convert_to_llm(self, messages) -> list[Message]: ...
    async def before_tool_call(self, ctx, signal=None) -> BeforeToolCallResult | None: ...
    async def after_tool_call(self, ctx, signal=None) -> AfterToolCallResult | None: ...
    async def should_stop_after_turn(self, ctx) -> bool: ...
```

All methods optional. Implementations only define the hooks they need.

### Composition

```python
def compose_middleware(middlewares: list[Middleware]) -> dict[str, Callable]:
    """Compose multiple Middleware into hook callbacks for AgentLoopConfig.

    Composition semantics:
    - transform_context: chained (output of one is input to next)
    - convert_to_llm: last one providing an implementation wins
    - before_tool_call: sequential, any returning block=True stops execution
    - after_tool_call: sequential, later ones can override earlier results
    - should_stop_after_turn: any returning True stops the agent
    """
```

Simple cases use hooks directly. Complex cases (cubebox) use Middleware + compose.

## Checkpointer

New layer not present in pi. pi has no persistence; applications handle it themselves.

### Protocol

```python
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Any]) -> None: ...
    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None: ...

class CheckpointData:
    messages: list[Any]
    extra: dict[str, Any]
```

### Save Timing

The framework saves incrementally per step:

- `message_end` → `append([message])` for each new message
- `turn_end` → `save_extra(extra)` for mutable state (compaction, etc.)

This avoids the full-snapshot-per-step overhead of langgraph's Postgres checkpointer, which serializes the entire messages list on every channel version change.

### Why Not Full Snapshot

langgraph saves a full messages blob each time the messages channel changes. For a 1000-message conversation, that means serializing and writing ~1000 messages on every step. cubepi's append-based approach writes only the new message(s), keeping DB I/O constant regardless of conversation length.

### Built-in Implementations

| Implementation | Use case | Dependency |
|----------------|----------|------------|
| `MemoryCheckpointer` | Dev/test, in-memory dict | None |
| `SQLiteCheckpointer` | Lightweight local persistence | `aiosqlite` |

Postgres, Redis, etc. are implemented by applications (cubebox will provide `PostgresCheckpointer`).

### HITL Considerations

pi does not implement human-in-the-loop pause/resume. Its `beforeToolCall` with `block=True` immediately rejects the tool call with an error result — the LLM sees the rejection and can ask the user for confirmation in the next turn.

cubepi follows the same pattern. The tool call information is not lost:
1. `AssistantMessage` (containing the tool call) is already in messages
2. `ToolResultMessage` (with error: "blocked") is added to messages
3. Both are checkpointed via `append`
4. LLM naturally asks the user for confirmation
5. Next user message → LLM retries the tool call → `before_tool_call` allows it

No special pause/resume mechanism or `pending_tool_calls` field needed.

## Cancel Mechanism

**pi**: `AbortController` / `AbortSignal`, passed to all async operations.

**cubepi**: `asyncio.Event` as signal, matching `AbortSignal` semantics — a checkable, awaitable flag.

Tools check `signal.is_set()` during execution. The Agent's `abort()` method sets the event.

## Testing

### Framework

- `pytest` + `pytest-asyncio` for all tests
- Tests live in `tests/` mirroring the `cubepi/` module structure

### Faux Provider

`cubepi.providers.faux` — a public module, shipped with the package. Downstream projects (e.g., cubebox) can import and use it for their own tests.

A fake `Provider` implementation that:
- Accepts pre-configured responses (queued, consumed in order)
- Supports async response factories
- Streams realistic event sequences (text, thinking, tool call deltas)
- Simulates usage estimation from serialized context
- Supports abort mid-stream (text, thinking, tool call phases)
- Emits error/aborted terminal events

All agent loop tests and e2e tests depend on it.

### Test Coverage from pi-agent-core

All tests from pi's `packages/agent/test/` must be ported:

**Agent Loop tests** (from `agent-loop.test.ts`):
- Event emission with AgentMessage types
- Custom message types via `convert_to_llm`
- `transform_context` applied before `convert_to_llm`
- Tool call execution and results
- `before_tool_call` argument mutation (executed without revalidation)
- `prepare_arguments` for tool argument validation
- Parallel tool execution: `tool_execution_end` in completion order, tool results in source order
- Steering message injection after tool calls complete
- Per-tool `execution_mode` override (sequential forces sequential even under parallel config)
- Mixed execution modes (one sequential tool forces entire batch sequential)
- All-parallel execution when every tool opts in
- `should_stop_after_turn` hook
- Early termination when all tool results set `terminate=True`
- Partial termination (not all tools terminate → continue)
- `after_tool_call` marking batch as terminating

**Agent Loop Continue tests** (from `agent-loop.test.ts`):
- Raise when context has no messages
- Continue from existing context without emitting user message events
- Custom message types as last message

**Agent class tests** (from `agent.test.ts`):
- Default state creation
- Custom initial state
- Event subscription and unsubscription
- Full lifecycle events for thrown run failures
- Async subscriber awaiting before `prompt()` resolves
- `wait_for_idle` awaits async subscribers
- Abort signal passed to subscribers
- State mutation
- Steering message queue
- Follow-up message queue
- Abort handling
- Raise when `prompt()` called while streaming
- Raise when `resume()` called while streaming
- `resume()` processes queued follow-up messages after assistant turn
- `resume()` keeps one-at-a-time steering semantics from assistant tail

**E2E tests** (from `e2e.test.ts`, using FauxProvider):
- Basic text prompt
- Tool execution with pending tool call tracking
- Abort during streaming
- Lifecycle event emission during streaming
- Context maintained across multiple turns
- Thinking content block preservation
- `resume()` validation (no messages, last message is assistant)
- Continue from user message
- Continue from tool result

### Tests for cubepi-specific Features

**Middleware tests**:
- `compose_middleware` composition semantics:
  - `transform_context`: chained (output → input)
  - `convert_to_llm`: last implementation wins
  - `before_tool_call`: sequential, any `block=True` stops execution
  - `after_tool_call`: sequential, later overrides earlier
  - `should_stop_after_turn`: any `True` stops
- Empty middleware list
- Middleware with partial method implementations (only some hooks defined)
- Middleware ordering matters for chained hooks

**Checkpointer tests**:
- `MemoryCheckpointer`: load/append/save_extra round-trip, multiple threads, empty thread
- `SQLiteCheckpointer`: same as MemoryCheckpointer, plus persistence across instances, concurrent access
- Incremental append correctness (append N messages → load returns all N)
- `save_extra` merges with existing extra data
- Agent integration: checkpoint saved at `message_end` and `turn_end`

**Provider tests** (unit, no real API calls):
- `AnthropicProvider`: message format conversion, tool definition mapping, streaming event parsing
- `OpenAIProvider`: same as Anthropic
- Abort mid-stream for both providers
- Error handling (API errors → `AssistantMessage` with `stop_reason="error"`)

**Provider E2E tests** (require API keys, skipped in CI by default):
- Basic text streaming
- Tool call streaming
- Thinking/reasoning streaming
- Abort mid-stream
- Error recovery

## Dependencies

Core (required):
- `pydantic` — types, validation, JSON Schema generation

Provider implementations:
- `anthropic` — for AnthropicProvider
- `openai` — for OpenAIProvider

Checkpointer implementations:
- `aiosqlite` — for SQLiteCheckpointer (optional)

Dev/test:
- `pytest` + `pytest-asyncio` — test framework
- `pytest-cov` — coverage reporting

No dependency on langchain, langgraph, or any other agent framework.
