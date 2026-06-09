# Goal Middleware

Autonomous goal-driven agent runs. Inspired by Claude Code's `/goal` command:
set a completion condition, let the agent work until an independent evaluator
model confirms the condition is met.

## Core Idea: Dual-Model Evaluation

The worker model (e.g. Sonnet/Opus) does the actual work. A separate evaluator
model (e.g. Haiku) judges whether the goal condition has been achieved after
each run. The evaluator only reads the conversation transcript — it cannot call
tools or read files independently.

"The agent isn't grading its own homework."

## Prerequisites

### 1. `tool_choice` on Provider

CubePi's providers currently have no `tool_choice` support. This is a
fundamental provider capability needed by `generate_structured` (to force the
model to call the synthetic tool) and useful to users generally.

**Type definition** (added to `base.py`):

```python
ToolChoice = Literal["auto", "required", "none"] | str
```

- `"auto"` — model decides (default, equivalent to not passing tool_choice)
- `"required"` — must call some tool
- `"none"` — no tool calls allowed
- Any other `str` — force a specific tool by name

**Where it goes**: explicit parameter on `stream()` and `generate()`, NOT in
`StreamOptions`. Rationale: `StreamOptions` is a transparent bag for
observation/control (on_payload, signal). `tool_choice` is a semantic
parameter paired with `tools`, like `system_prompt`.

**Provider mapping**:

| CubePi | Anthropic | OpenAI / OpenAI Responses |
|--------|-----------|---------------------------|
| `"auto"` | `{"type": "auto"}` | `"auto"` |
| `"required"` | `{"type": "any"}` | `"required"` |
| `"none"` | (omit tools) | `"none"` |
| `"name"` | `{"type": "tool", "name": "name"}` | `{"type": "function", "function": {"name": "name"}}` |

**FauxProvider**: accepts the parameter but ignores it (test provider returns
scripted responses regardless).

**Agent loop**: unchanged — the loop doesn't need `tool_choice`. This is a
`BoundModel`-level / direct-caller feature.

**Files changed**: `base.py` (type + Protocol + BoundModel + BaseProvider),
`anthropic.py`, `openai.py`, `openai_responses.py`, `faux.py`.

### 2. Structured Output on BoundModel

CubePi's `BoundModel` currently has no structured output support. The
evaluator needs to return `{achieved: bool, reason: str}` reliably. Rather
than hack JSON parsing inside GoalMiddleware, we add a general-purpose
structured output method to `BoundModel` first.

#### Design (tool-based, inspired by pydantic-ai ToolOutput)

```python
from pydantic import BaseModel

class GoalResult(BaseModel):
    achieved: bool
    reason: str

result: GoalResult = await model.generate_structured(
    GoalResult,
    messages=[...],
    system_prompt="...",
)
```

**Implementation strategy**: inject a synthetic tool whose `parameters` is the
Pydantic model's JSON schema, call `generate()` with
`tools=[synthetic_tool], tool_choice=tool_name` to force the model to call it,
extract the `ToolCall.arguments` from the response, validate through
`Schema.model_validate(arguments)`.

This is the approach pydantic-ai uses by default (`ToolOutput` mode with
`final_result` tool name). It works across all providers because every LLM API
supports tool/function calling.

**Three layers of robustness**:

1. **`tool_choice`** — API-level forcing, model must call the tool
2. **Default system prompt** — prompt-level guidance ("You MUST respond by
   calling the tool")
3. **Validation + clear error** — if model somehow returns text or invalid
   JSON, `StructuredOutputError` is raised

**Notable divergence from pydantic-ai**: pydantic-ai defines output_type at
agent construction time and weaves it into the agent loop (retry on validation
failure, output validators, etc.). CubePi's `generate_structured` is a
standalone method on BoundModel — simpler, no agent coupling. Has its own
`max_retries` for validation failure (feeds the error back to the model).

**Method signature on BoundModel**:

```python
async def generate_structured(
    self,
    output_type: type[T],
    messages: list[Message],
    *,
    system_prompt: str = "",
    tool_name: str = "structured_output",
    tool_description: str = "Return the structured output",
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    max_retries: int = 1,
) -> T:
    ...
```

**Error handling**: if the model returns text instead of a tool call, or if
Pydantic validation fails after all retries, raise `StructuredOutputError`.

**Files**: added directly to `cubepi/providers/base.py` on `BoundModel`.
No new module needed.

## User API

```python
from cubepi.middleware.goal import GoalMiddleware

goal = GoalMiddleware(
    evaluator=provider.model("haiku"),
    max_evaluations=10,  # default 10; safety cap
)

agent = Agent(
    model=provider.model("sonnet"),
    middleware=[goal],
    tools=[...],
)

# /goal prefix activates goal mode; condition is everything after it
await agent.prompt("/goal all tests in tests/auth pass and ruff check is clean")

# Without /goal prefix — middleware is transparent, agent runs normally
await agent.prompt("fix the bug in auth.py")
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `evaluator` | `BoundModel` | required | Model used to evaluate the condition |
| `max_evaluations` | `int` | `10` | Max evaluator calls before forced stop |

## Activation

GoalMiddleware detects `/goal` from the user's message at runtime:

- `transform_context`: inspects the latest user message. If it starts with
  `/goal `, extracts the condition text, stores it as internal state, and
  rewrites the message to a work directive (strips the `/goal` prefix).
- Messages without `/goal` prefix: middleware is fully transparent (no-op).

## Evaluation Flow

Uses the `on_run_end` middleware hook. The outer loop calls `on_run_end`
after the worker finishes a complete run (all inner turns + tool calls
exhausted).

```
Worker completes a run → on_run_end fires
  → No active goal? Return None (no-op)
  → evaluator.generate_structured(GoalResult, ...) reads condition + recent messages
  → Returns GoalResult(achieved=bool, reason=str)

  achieved=True:
    → Write ctx.extra["goal"] = {status: "achieved", reason, evaluations: N}
    → Return None (loop ends naturally)

  achieved=False AND evaluations < max_evaluations:
    → Write ctx.extra["goal"] = {status: "active", reason, evaluations: N}
    → Return [UserMessage("Goal not yet met: {reason}. Continue working.")]
    → Loop continues, worker sees the feedback

  evaluations >= max_evaluations:
    → Write ctx.extra["goal"] = {status: "exhausted", reason, evaluations: N}
    → Return None (forced stop)
```

## Loop Change: `on_run_end` Multi-Fire

Current `_run_loop` in `loop.py` guards `on_run_end` with
`_reflection_fired`, making it single-fire per `prompt()` call. Goal
evaluation requires multiple fires.

**Change**: remove the `_reflection_fired` guard. `on_run_end` fires on
every outer-loop iteration. Existing users unaffected — an `on_run_end`
that returns `None` produces no loop iteration regardless.

Before:
```python
if on_run_end and not _reflection_fired:
    _reflection_fired = True
    inject = await on_run_end(current_context, signal=opts.signal)
```

After:
```python
if on_run_end:
    inject = await on_run_end(current_context, signal=opts.signal)
```

The `_reflection_fired` variable and its guard are removed entirely.

## Evaluator Prompt

Built-in, not user-configurable:

```
You are evaluating whether an agent has achieved its goal.

Goal condition:
{condition}

Recent conversation (last 20 messages):
{messages}

Has the goal been achieved? Use the structured_output tool to respond.
```

The message window is the last 20 messages (hardcoded) to keep evaluator
costs low. The evaluator has no tools (other than the synthetic output
tool) and cannot make further LLM calls.

## State & Observability

GoalMiddleware writes structured state to `AgentContext.extra["goal"]`:

```python
{
    "status": "achieved" | "active" | "exhausted",
    "condition": "all tests pass...",
    "evaluations": 3,
    "max_evaluations": 10,
    "last_reason": "2 tests still failing in test_auth.py",
}
```

No new event types. Callers read `agent.state.extra["goal"]` after the
run completes to check outcome.

## Tracing

GoalMiddleware declares its evaluator via `extra_llm_calls()` so the
tracing Recorder can subscribe to the evaluator's provider and attribute
evaluation spans correctly (not as the worker's spans).

## File Layout

```
cubepi/providers/base.py               # ToolChoice type, tool_choice on Protocol/BoundModel, generate_structured()
cubepi/providers/anthropic.py          # tool_choice → Anthropic wire format
cubepi/providers/openai.py             # tool_choice → OpenAI wire format
cubepi/providers/openai_responses.py   # tool_choice → OpenAI Responses wire format
cubepi/providers/faux.py               # accept tool_choice (ignored)
cubepi/middleware/goal.py              # GoalMiddleware implementation
tests/providers/test_tool_choice.py    # Tests for tool_choice across providers
tests/test_structured_output.py        # Tests for generate_structured
tests/test_goal.py                     # Tests for GoalMiddleware (FauxProvider as evaluator)
```

## What This Is NOT

- Not a persistent goal that survives across sessions (session-scoped only)
- Not a task tracker / todo list
- Not a scheduler or cron — the agent works continuously, not on intervals
- The evaluator cannot call tools — it only judges based on transcript
