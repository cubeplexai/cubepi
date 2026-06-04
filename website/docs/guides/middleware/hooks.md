---
title: The 8 Hooks
description: "Reference for the 8 middleware hooks in CubePi — transform_context, convert_to_llm, before_tool_call, after_tool_call, on_run_end, and more."
---

# The 8 Hooks

`Middleware` is a class with up to eight optional async methods. Each
hook fires at a precise point in the agent loop. Implement only the
ones you need — CubePi only wires in the ones you override.

```python
from cubepi import Middleware

class MyMiddleware(Middleware):
    async def transform_context(self, messages, *, ctx, signal=None):
        ...
```

Pass instances to `Agent(middleware=[MyMiddleware(), …])`.

## `transform_context`

```python
async def transform_context(
    self,
    messages: list[Message],
    *,
    ctx: AgentContext,
    signal=None,
) -> list[Message]:
    ...
```

Fires **before each model call**, on the full message list. `ctx` is
the current `AgentContext`; use `ctx.extra` for per-run or
checkpointer-persisted middleware state. Use to:

- Truncate or summarise to fit context windows.
- Inject system reminders (better: use `transform_system_prompt`).
- Add or remove specific messages dynamically.

Return the (possibly new) list. **Don't mutate the input** — return a
new list so other code that holds the original isn't surprised.

Composition: chained — each middleware sees the previous one's
output.

## `convert_to_llm`

```python
async def convert_to_llm(
    self,
    messages: list[Message],
    *,
    ctx: AgentContext,
) -> list[Message]:
    ...
```

Fires **right before serialisation** to the provider. This is the last
chance to reshape what the LLM sees. Use for:

- Stripping tool results to text-only.
- Replacing image content with text descriptions for non-multimodal
  providers.
- Compacting long tool outputs.

Composition: **last implementation wins** (not chained). Use this when
multiple middlewares would conflict and you want a single owner.

## `transform_system_prompt`

```python
async def transform_system_prompt(
    self,
    system_prompt: str,
    *,
    ctx: AgentContext,
    signal=None,
) -> str:
    ...
```

Fires **before each model call**, on the system prompt string. Use to:

- Inject runtime info (current time, user role).
- Compose modular system-prompt fragments.
- A/B test prompt variants.

Composition: chained.

## `before_tool_call`

```python
async def before_tool_call(self, ctx: BeforeToolCallContext, *, signal=None) -> BeforeToolCallResult | None:
    ...
```

Fires **per tool call**, after argument validation, before
`tool.execute`. The context provides:

- `ctx.assistant_message` — the message that initiated the call.
- `ctx.tool_call` — the `ToolCall` block.
- `ctx.args` — the *validated* Pydantic instance.
- `ctx.context` — the full `AgentContext`.

Return `BeforeToolCallResult(block=True, reason="…")` to short-circuit
— CubePi feeds the reason back as the tool result with
`is_error=True`. Return `None` (or no return) to proceed.

Use for: permissions, rate limiting, dry-run modes, sandboxing,
human-in-the-loop confirmation (see [HITL guide](../hitl/overview)).

Composition: **first `block=True` short-circuits** the chain.

## `after_tool_call`

```python
async def after_tool_call(self, ctx: AfterToolCallContext, *, signal=None) -> AfterToolCallResult | None:
    ...
```

Fires **per tool call**, after `tool.execute` returns (or raises).
The context adds:

- `ctx.result` — the `AgentToolResult` from execute.
- `ctx.is_error` — whether the tool errored.

Return `AfterToolCallResult(content=…, details=…, is_error=…, terminate=…)`
to override individual fields of the result (any `None` field keeps
the original). Return `None` to pass through unchanged.

Use for: redaction, retries, logging, transforming results.

Composition: later overrides earlier (each non-`None` field in a
return value overrides the prior).

## `should_stop_after_turn`

```python
async def should_stop_after_turn(self, ctx: ShouldStopAfterTurnContext) -> bool:
    ...
```

Fires **at each turn boundary** (after any tool batch). Return `True`
to end the run without another model call.

Use for: max-turn limits, budget caps, application-defined stop
conditions.

Composition: **any `True` stops** (logical OR across the chain).

## `after_model_response`

```python
async def after_model_response(
    self,
    response: AssistantMessage,
    ctx: AgentContext,
    *,
    signal=None,
) -> TurnAction | None:
    ...
```

Fires **immediately after** the assistant message lands, **before**
`message_end` is emitted and **before** any tool calls dispatch. The
hook returns a `TurnAction`:

```python
from cubepi.middleware.base import TurnAction
from cubepi.providers.base import UserMessage, TextContent

TurnAction(
    response=modified_message,            # replace the message; None to keep
    inject_messages=[UserMessage(...)],   # extra messages to append before next turn
    decision="natural",                   # "natural" | "stop" | "loop_to_model"
)
```

Three control-flow knobs:

- `decision="natural"` (default) — proceed to tool execution / next
  turn as normal.
- `decision="stop"` — end the run after emitting `turn_end` and
  `agent_end`. No tools run, no more model calls.
- `decision="loop_to_model"` — skip tool execution and re-invoke the
  model immediately (use with `inject_messages` to add context first).

Use for: response moderation, structured-output validation with
re-prompts, conditional routing.

Composition: chain — each middleware sees the previous
middleware's `response`; `inject_messages` concatenate across the
chain; the last middleware's `decision` wins.

## `on_run_end`

```python
async def on_run_end(
    self,
    ctx: AgentContext,
    *,
    signal=None,
) -> list[Message] | None:
    ...
```

Fires **once per `prompt()` call**, after all turns and tool calls
complete, before `AgentEndEvent` is emitted. Return a non-empty
`list[Message]` to inject those messages into context and run **one
extra model turn** (the run-end pass). Return `None` or `[]` to do
nothing.

The extra turn fires exactly once — a `_reflection_fired` guard in the
loop prevents the injected turn from triggering another `on_run_end`.

**When it fires:**
- Normal completion (loop breaks naturally after all turns).
- `should_stop_after_turn` returns `True`.
- `after_model_response` returns `decision="stop"`.

**When it does NOT fire:**
- The run ends with `stop_reason` `"error"` or `"aborted"`.
- HITL interruption (`HitlDetached` / `HitlAborted`) — the run is
  paused, not finished.

Use for: post-run memory consolidation, conversation summarisation,
audit logging that needs the full turn context.

Composition: messages from all middleware **concatenate** into a single
list; all are injected together before the extra model turn.

## Anatomy of a middleware

A middleware doesn't have to implement every hook. Only override the
ones you need; the base class's unimplemented hooks raise
`NotImplementedError`, but `compose_middleware` skips them
automatically.

```python
from cubepi import Middleware

class MaxTurnsMiddleware(Middleware):
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        self.turns = 0

    async def should_stop_after_turn(self, ctx) -> bool:
        self.turns += 1
        return self.turns >= self.max_turns


agent = Agent(provider=…, model=…, middleware=[MaxTurnsMiddleware(5)])
```

## See also

- [Composition Rules](./composition) — exact semantics when multiple
  middlewares define the same hook.
- [Examples](./examples) — working middleware for rate limiting,
  logging, retries, sliding-window context, post-run memory.
