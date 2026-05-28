---
title: Human-in-the-Loop (HITL)
description: "Add human-in-the-loop pauses to your CubePi agent with confirm, approve, and structured ask verbs."
sidebar_position: 10
---

# Human-in-the-Loop (HITL)

cubepi's HITL channel lets an agent **pause and ask a human** before proceeding.
It handles two recurring patterns with a single primitive:

1. **Sandbox tool confirmation** — a dangerous tool (bash, file writes, API
   mutations) needs approve / deny / edit from a human before running.
2. **Mid-run structured questions** — the agent needs a specific selection or
   form answer from the user before it can continue.

The channel is an `await`-able coroutine collaborator. Tool authors write
`await channel.ask(...)` and the channel handles the pause. Host code
(subscribers) renders the pending request and posts an answer. Two backends
cover the full spectrum:

- `InMemoryChannel` — CLI, notebooks, tests. Process dies, pending is lost.
- `CheckpointedChannel` — web services. Persists the pending request to a
  `Checkpointer` so a different process (or the same process after restart)
  can pick up and answer hours later.

## Architecture

```text
┌────────────────────────────────────────────────────────┐
│ Host (cubebox web service / CLI / TUI)                │
│                                                        │
│   subscribe to channel.pending  ◄──── answer / cancel  │
└──────────────────────┬─────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │     HitlChannel (Protocol) │
         │  confirm / approve / ask   │
         │  pending / answer / cancel │
         └──────┬──────────────┬──────┘
                │              │
     ┌──────────▼───┐   ┌──────▼────────────┐
     │InMemoryChannel│   │CheckpointedChannel│
     │ (Future+Queue)│   │  (Future + persi- │
     │               │   │   sts to Checkpo- │
     │               │   │   inter)          │
     └───────────────┘   └───────────────────┘
                │              │
                │  used by ────┤
                │              │
     ┌──────────▼──────────────▼────────────┐
     │  ask_user tool    ConfirmToolCallMW  │
     │  (structured form) (approve/deny/    │
     │                     edit per-tool)   │
     └──────────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │   cubepi Agent loop        │
         │   (BeforeToolCallResult    │
         │    carries hitl_trace)     │
         └────────────────────────────┘
```

## Channel types

### `InMemoryChannel`

For single-process use. Holds an `asyncio.Future` internally — the host
calls `channel.answer()` from the same event loop.

```python
from cubepi.hitl import InMemoryChannel

channel = InMemoryChannel(
    default_timeout=180.0,  # per-call timeout; None = no timeout (default)
)
```

### `CheckpointedChannel`

For cross-process use. On every `confirm / approve / ask`, the `HitlRequest`
is persisted via `Checkpointer.save_pending_request(thread_id, ...)`. On
`channel.answer()`, the pending is cleared. On `HitlDetached` (graceful
suspend), the pending stays so a later `Agent.respond()` can resume.

```python
from cubepi.hitl import CheckpointedChannel
from cubepi.checkpointer.sqlite import SQLiteCheckpointer  # or postgres / mysql

async with SQLiteCheckpointer("path/to.db") as cp:
    channel = CheckpointedChannel(
        checkpointer=cp,
        thread_id="conversation-42",
        default_timeout=None,              # cross-process: typical to disable
        allow_inside_custom_tool=False,    # safety gate (see Durable scope below)
    )
```

`CheckpointedChannel.__init__` validates the checkpointer implements
`save_pending_request` and `load_pending_request` — it fails early with
`HitlError` if not.

## The three verbs

### `confirm(prompt, *, details, timeout, signal) → bool`

A simple yes/no question. The host answers `True` or `False`.

```python
if await channel.confirm("Deploy to production?", details={"env": "prod"}):
    await deploy()
```

### `approve(tool_name, tool_call_id, args, *, details, timeout, signal) → ApproveAnswer`

The sandbox-confirm verb. Returns an `ApproveAnswer` with one of three decisions:

| Decision | Result |
|---|---|
| `"approve"` | Tool runs with the original args. |
| `"deny"` | Tool is blocked; `tool_result.is_error=True` with `details["hitl"]["decision"]="human_deny"`. |
| `"edit"` | Tool runs with the edited args (re-validated against the tool's pydantic parameter model). |

For `approve` requests, the envelope's `question_id` is set to the LLM's
`tool_call_id` — no separate UUID, so host code can correlate by the same
ID it already tracks from the tool stream.

### `ask(questions, *, timeout, signal) → dict[str, str | list[str]]`

A structured form with one or more `Question` objects. Each question can be:

- **Free-text** (`options=None`)
- **Single-select** (`options=[...]`, `multi_select=False`)
- **Multi-select** (`options=[...]`, `multi_select=True`)
- **"Other" with input** (option has `allow_input=True` — user types free text)

```python
from cubepi.hitl.types import Question, Option

answers = await channel.ask([
    Question(key="framework", prompt="Which framework?", options=[
        Option(label="React", value="react"),
        Option(label="Vue", value="vue"),
        Option(label="Other", value="other", allow_input=True),
    ]),
    Question(key="features", prompt="Enable:", multi_select=True, options=[
        Option(label="Auth", value="auth"),
        Option(label="Payments", value="payments"),
    ]),
])
# answers == {"framework": "react", "features": ["auth", "payments"]}
```

## Timeout

Both channels accept a `default_timeout` constructor arg and every verb
accepts a per-call `timeout` kwarg (per-call overrides default).

| Layer | API |
|---|---|
| Channel constructor | `InMemoryChannel(default_timeout=30.0)` |
| Per-call override | `channel.confirm("ok?", timeout=10.0)` |
| Per-call off | `channel.approve(..., timeout=None)` when channel default is set |

Timeout expiry raises `HitlTimedOut(BaseException)` from the agent-side
`await`. The surrounding tool or middleware translates it to
`tool_result.is_error=True` with `details["hitl"]["decision"]="timed_out"`,
so the model sees a clean denial and can react naturally. The envelope's
`HitlRequest.timeout_seconds` is filled automatically so the frontend can
render a countdown.

## Built-in middlewares

### `ConfirmToolCallMiddleware`

"Always ask the human for tool *names* in this set."

```python
from cubepi.hitl import ConfirmToolCallMiddleware

# Set-based matching — only ask for these tool names
agent = Agent(
    ...,
    middleware=[
        ConfirmToolCallMiddleware(
            channel,
            require_confirm={"bash", "write_file", "http_post"},
            timeout_seconds=180,
        ),
    ],
)
```

`require_confirm` options:

| Value | Behavior |
|---|---|
| `None` (default) | Confirm **every** tool. |
| `set[str]` | Confirm if `tool_call.name` is in the set. |
| `Callable[[BeforeToolCallContext], bool]` | Custom predicate — inspect args, context, etc. |

An optional `details_fn(ctx: BeforeToolCallContext) -> dict` enriches the
approve request payload with extra context the frontend can render (e.g.
matched rule name, impact preview, affected file list).

### `ApprovalPolicyMiddleware`

For hosts with a **policy engine** that classifies tool calls into three
tiers — auto-allow, hard-deny, or human-confirm.

```python
from cubepi.hitl import Approve, ApprovalPolicyMiddleware, AskUser, Deny

def my_policy(ctx):
    if ctx.tool_call.name == "read_file":
        return Approve()                               # passthrough
    if ctx.tool_call.name.startswith("dangerous_"):
        return Deny(reason="blocked by policy")        # hard block, no human asked
    return AskUser(timeout_seconds=180)                # human confirm

agent = Agent(
    ...,
    middleware=[ApprovalPolicyMiddleware(channel, policy=my_policy)],
)
```

The policy function can be sync or async (`await`-able). It returns one of:

| Return | Effect |
|---|---|
| `Approve()` | Tool runs; channel never invoked. |
| `Deny(reason)` | Tool blocked; `hitl_trace["decision"]="policy_deny"`. |
| `AskUser(timeout_seconds=..., details=...)` | Channel invoked; human chooses approve/deny/edit. |

Policy-deny and human-deny produce different `hitl_trace` keys (`policy_deny`
vs `human_deny`) so audit and trace can distinguish them.

## `ask_user` built-in tool

A tool the **model** invokes when it needs structured input from the user.
The factory returns an `AgentTool` named `"ask_user"` with
`execution_mode="sequential"` — it can't run in parallel with other tools.

```python
from cubepi.hitl import ask_user_tool

agent = Agent(
    ...,
    tools=[bash_tool, ask_user_tool(channel)],
)
```

The tool description explicitly steers the model away from using `ask_user`
for free-form clarification ("for free-form questions, end your turn with
text — the user's next message is your answer"). The model should only
invoke it when a **structured** answer is needed.

The `Parameters` prompt schema the model sees:

| Field | Type | Description |
|---|---|---|
| `questions` | array | One or more question objects. |
| `questions[].key` | string | Field name in the answer dict. |
| `questions[].prompt` | string | The question text. |
| `questions[].options` | array (optional) | Selection options. `None` = free text. |
| `questions[].options[].label` | string | Human-facing label. |
| `questions[].options[].value` | string | Value returned to agent. |
| `questions[].options[].allow_input` | bool (default `false`) | "Other / please specify." |
| `questions[].multi_select` | bool (default `false`) | Allow multiple selections. |
| `questions[].required` | bool (default `true`) | Can the user skip this? |

Cancel and timeout are surfaced as `tool_result.is_error=True` with
`details["hitl"]["outcome"]="cancelled"` / `"timed_out"` — the model sees
a clean error tool result and can react. Other HITL control exceptions
(HitlDetached, HitlAborted) propagate to the Agent layer, not the model.

## Cross-process (web service) flow

Full lifecycle of a suspend/resume cycle:

```python
import asyncio
from cubepi.agent.agent import Agent
from cubepi.checkpointer.postgres import PostgresCheckpointer
from cubepi.hitl import (
    ApproveAnswer, CheckpointedChannel, ConfirmToolCallMiddleware,
)

# ---------- Process 1: HTTP POST /chat ----------

async with PostgresCheckpointer("postgresql://...") as cp:
    channel = CheckpointedChannel(checkpointer=cp, thread_id="conv-42")

    agent = Agent(
        provider=..., model=...,
        tools=[bash_tool],
        middleware=[
            ConfirmToolCallMiddleware(channel, require_confirm={"bash"}),
        ],
        channel=channel,
        checkpointer=cp,
        thread_id="conv-42",
    )

    # prompt() blocks until the HITL request is answered OR the host
    # calls detach(). Same-process host coroutines can answer directly
    # via channel.answer(); cross-process hosts use respond().
    task = asyncio.create_task(agent.prompt("delete temp files"))

    # Poll for pending (or subscribe to channel for SSE push)
    for _ in range(1000):
        pending = channel.pending
        if pending is not None:
            break
        await asyncio.sleep(0.1)

    assert pending is not None
    # Render pending.payload to the frontend (an ApproveRequest:
    # tool_name="bash", args={"cmd":"rm /tmp/foo"}, ...)

    # Graceful suspend — persist the assistant message + unresolved
    # tool_calls, keep pending_request in DB, emit AgentSuspendedEvent.
    # The HTTP handler returns 200 { status: "awaiting_approval" }.
    await agent.detach()
    await task  # prompt() unwinds with HitlDetached


# ---------- Process 2: HTTP POST /respond ----------

async with PostgresCheckpointer("postgresql://...") as cp:
    channel = CheckpointedChannel(checkpointer=cp, thread_id="conv-42")

    agent = Agent(
        provider=..., model=...,
        tools=[bash_tool],
        middleware=[
            ConfirmToolCallMiddleware(channel, require_confirm={"bash"}),
        ],
        channel=channel,
        checkpointer=cp,
        thread_id="conv-42",
    )

    # Loads the persisted history + pending, validates the question_id
    # matches, attaches the answer to the channel, re-enters the loop
    # where the last assistant message had unresolved tool calls.
    await agent.respond(
        question_id=request.json["call_id"],
        answer=ApproveAnswer(decision="approve"),
    )
    # The bash tool executes, the model receives the tool_result and
    # produces the next assistant turn. Conversation continues normally.
```

**If the user closes the tab without answering:**

```python
await agent.abort_pending(reason="user closed tab")
# Phase 1: signals the in-flight HITL await (if any) to raise HitlAborted.
# Phase 2: appends synthetic deny ToolResultMessage(s) for unresolved
#   tool_calls, appends a terminal AssistantMessage(stop_reason="aborted"),
#   clears persisted pending, emits AgentAbortedEvent.
# No model call is made. The conversation is closed.
```

## Agent API

### Constructor

```python
Agent(
    ...,
    channel: HitlChannel | None = None,    # bind a channel to this agent
)
```

If a channel is bound, the Agent wires its `_process_event` as the channel's
emit callback so `HitlRequestEvent` / `HitlAnswerEvent` are dispatched to all
agent listeners. A `_run_lock` (`asyncio.Lock`) serialises `prompt`,
`resume`, and `respond` calls.

### Properties

| Property | Returns | Description |
|---|---|---|
| `agent.channel` | `HitlChannel \| None` | The bound channel or `None`. |
| `agent.in_flight_hitl_request` | `HitlRequest \| None` | The channel's current pending payload. Raises `HitlError` if no channel bound. |

### Methods

| Method | Signature | Description |
|---|---|---|
| `load_pending_hitl_request()` | `async → HitlRequest \| None` | Reads the pending from the checkpointer (even post-detach). |
| `detach()` | `async → None` | Emits `AgentSuspendedEvent(pending_request=...)` then triggers `HitlDetached` on the channel future. The loop exits silently; assistant message retains unresolved tool calls; `pending_request` stays persisted. |
| `respond(*, question_id=, answer=)` | `async → None` | Resumes a suspended run. Validates qid matches persistent pending, attaches answer to channel, re-enters the loop via `run_agent_loop_resume`. |
| `abort_pending(reason=)` | `async → None` | Closes the conversation. Two-phase: Phase 1 signals in-flight await (no lock). Phase 2 appends synthetic deny tool_results + terminal `stop_reason="aborted"` assistant (under lock). |

`in_flight_hitl_request` is a synchronous property (reads the in-memory
channel slot). `load_pending_hitl_request()` is async (reads from the
checkpointer; useful post-detach or in a fresh process).

## Events

Four new events are emitted on the agent's event stream:

| Event | When | Key fields |
|---|---|---|
| `HitlRequestEvent` | Channel receives a new `confirm/approve/ask`. | `request: HitlRequest` |
| `HitlAnswerEvent` | `channel.answer()` or `channel.cancel()` fires. | `question_id: str`, `answer: Any`, `cancelled: bool`, `timed_out: bool` |
| `AgentSuspendedEvent` | `agent.detach()` called while HITL was pending. | `pending_request: HitlRequest` |
| `AgentAbortedEvent` | `agent.abort_pending()` closes the conversation. | `reason: str` |

These are all included in the `AgentEvent` union, so typed listeners
automatically cover them. `HitlRequestEvent` and `HitlAnswerEvent` are
emitted by the channel through the agent's emit binding. `AgentSuspendedEvent`
and `AgentAbortedEvent` are emitted by the Agent layer (not the loop — the
Agent has the channel handle to populate the real `pending_request` payload).

## Trace spans

When the `cubepi[tracing]` extra is installed, each HITL await is wrapped
in an OpenTelemetry span:

| Span name | Attributes |
|---|---|
| `hitl.approve` | `hitl.tool_name`, `hitl.tool_call_id`, `hitl.outcome`, `hitl.from_resume`, `hitl.duration_seconds` |
| `hitl.confirm` | `hitl.question_id`, `hitl.outcome`, `hitl.duration_seconds` |
| `hitl.ask` | `hitl.question_id`, `hitl.outcome`, `hitl.duration_seconds` |

`hitl.outcome` is one of: `approved`, `denied`, `edited`, `answered`,
`cancelled`, `timed_out`, `aborted`, `detached`.

The tracing import is lazy — if `opentelemetry` is not installed, the
channel silently falls back to a no-op span.

## Error reference

| Exception | Base class | Meaning |
|---|---|---|
| `HitlCancelled(reason)` | `BaseException` | Host called `channel.cancel(qid)`. |
| `HitlTimedOut(seconds)` | `BaseException` | Per-call or channel-default timeout expired. |
| `HitlDetached` | `BaseException` | `agent.detach()` called during HITL await. |
| `HitlAborted` | `BaseException` | `agent.abort_pending()` signalled the agent. |
| `HitlConcurrencyError` | `Exception` | `confirm/approve/ask` called while channel already has a pending request. |
| `HitlStaleAnswer` | `Exception` | `channel.answer(qid)` with a `question_id` that doesn't match the current pending. |
| `HitlNoPendingRequest` | `Exception` | `agent.respond(…)` called but no `pending_request` on the thread. |
| `HitlDurabilityNotGuaranteed` | `Exception` | Custom tool called `CheckpointedChannel.ask()` without `allow_inside_custom_tool=True`. |

`HitlControlException` (the parent of the four `BaseException` subclasses)
is intentionally NOT caught by the existing broad `except Exception:` handlers
in `cubepi.agent.tools._prepare_tool_call` and `_execute_prepared` — this
mirrors `asyncio.CancelledError`'s pattern.

## When to use `ask_user` vs end of turn

| Goal | Use |
|---|---|
| Free-text follow-up question to user | Just end the turn with the question as text; the user's next message is your answer. |
| Structured selection (one of N) | `ask_user` tool with `options`. |
| Multi-select ("pick any of") | `ask_user` tool with `multi_select=True`. |
| "Other" with free-text input | `ask_user` tool option with `allow_input=True`. |
| Confirm/edit tool args before run | `ConfirmToolCallMiddleware` or `ApprovalPolicyMiddleware`. |

## Durable scope

Durable cross-process resume (survives process death) is supported at two
well-defined safe suspension points:

1. **`before_tool_call` approval gate** — the approval middleware calls
   `channel.approve(...)` *before* the tool's `execute()` body runs. No tool
   side effects exist yet. Resume re-enters the loop and either runs the
   (possibly edited) tool body or substitutes a synthetic deny tool_result.
2. **`ask_user` tool body** — whose entire `execute()` body is
   `return await channel.ask(...)`. Resume replays nothing because nothing
   else happened.

**Custom tools that mix HITL with other work inside `execute()` are NOT
durable cross-process by default.** If such a tool's process dies mid-execute,
anything that ran before the channel call would be lost. cubepi will raise
`HitlDurabilityNotGuaranteed` unless the `CheckpointedChannel` is constructed
with `allow_inside_custom_tool=True` — the caller must acknowledge the
idempotency contract (the tool body must be a pure HITL wait with no
preceding observable side effects).

## Testing helpers

```python
from cubepi.hitl.testing import ScriptedChannel, NoopChannel

# ScriptedChannel: pre-programmed answers, consumed in order.
ch = ScriptedChannel(answers=[
    ApproveAnswer(decision="approve"),
    {"color": "red"},                       # ask answer
    lambda req: ApproveAnswer(decision="deny", reason="test")  # callable
])
assert len(ch.history) == 3  # all HitlRequests ever seen

# NoopChannel: auto-approves everything. Useful for subagents.
ch = NoopChannel()
assert (await ch.approve("bash", "tc", {})).decision == "approve"
assert (await ch.confirm("?")) is True
assert await ch.ask([Question(key="k", prompt="p")]) == {"k": ""}
```

## Architecture notes

- **Single pending per thread.** The agent loop is sequential — at most one
  HITL request is outstanding per `thread_id`. Concurrent `confirm/approve/ask`
  raises `HitlConcurrencyError`.
- **Prompt-cache prefix invariant.** Between pause and resume, the messages
  list changes only by appending tool-result messages and the next assistant
  turn at the tail. No inserting, reordering, or mutating prior messages —
  that would invalidate the provider-side prompt cache.
- **`question_id == tool_call_id` for approve requests.** No aliasing or
  mapping needed — hosts that already track `call_id` from the tool stream
  pass it directly.
- **Resume does not replay.** It re-enters the loop with the answer pre-loaded
  into the channel. The last assistant message's unresolved tool calls dictate
  what executes next. No node-based replay semantics.
