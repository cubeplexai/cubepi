# HITL (Human-in-the-Loop) Channel — Design Spec

- **Status**: Draft, awaiting review
- **Date**: 2026-05-28
- **Branch / worktree**: `2026-05-28-hitl-channel` / `.worktrees/2026-05-28-hitl-channel`
- **Author**: brainstormed with the user, drafted by Claude

## 1. Motivation

Two recurring scenarios in cubepi-built agents are not first-class today:

1. **Sandbox tool confirmation** — a dangerous tool (e.g. `bash`, `write_file`) is about to run; a human must approve, deny, or edit the arguments before execution.
2. **Mid-run question from the agent** — the agent (model or middleware) needs a structured answer (a selection, a multi-question form) before it can continue.

Both can be hacked together today with custom `before_tool_call` middlewares and bespoke event plumbing, but every cubepi consumer ends up reinventing:

- A way for a coroutine deep inside a tool / middleware to **pause until a human answers**, with first-class support for the two-process case (agent suspends, web client posts an answer later via HTTP).
- A consistent **event/trace surface** so hosts (cubebox, custom TUIs, web UIs) can render pending requests uniformly.
- A way to **resume cleanly** without replaying side-effecting tool calls.

This spec introduces a single primitive — a **HITL channel** — together with two built-in clients (`ask_user` tool and `ConfirmToolCallMiddleware`) that cover both scenarios. The channel is a small protocol with two interchangeable implementations (in-memory and checkpointed).

## 2. Design Philosophy

> cubepi's HITL is not "a graph interrupt node" — it is "a conversation that paused and resumed". State is encoded by the message list; the channel is an awaitable collaborator; resume does not replay, it just notices that the last assistant message had unresolved tool calls and continues from there with the answer pre-loaded.

Concretely:

- **Tool / middleware author writes `await channel.ask(...)`** — they don't write two versions for in-process and cross-process modes.
- **Same protocol, two implementations.** `InMemoryChannel` for CLI/notebooks/tests; `CheckpointedChannel` for web services where the agent process may die between question and answer.
- **Single pending per thread.** cubepi's agent loop is sequential — at most one HITL request is outstanding per `thread_id`. This kills a whole class of correlation / concurrency complexity.
- **No replay.** Resume re-enters the loop with the answer pre-loaded into the channel. The last assistant message's unresolved tool calls dictate "what we were doing"; the answer flows into the natural tool-execution code path. No side-effects re-run.
- **Channel emits events** so hosts that prefer event-stream subscription (rather than synchronous coroutines) can also consume.

## 3. Surface Area

Two surfaces, both backed by the same channel.

### 3.1 `ask_user` built-in tool

A `cubepi.hitl.ask_user_tool(channel)` factory returns an `AgentTool` named `ask_user`. The model invokes it like any other tool to ask the user a *structured* question (one or more, each with optional single/multi-select options, optional "allow free-text input" per option).

The tool's `execution_mode="sequential"` — HITL cannot share a turn with other parallel tools. Tool description explicitly steers the model away from using `ask_user` for free-form clarification ("for free-form questions, end your turn with text — the user's next message is your answer").

### 3.2 `ConfirmToolCallMiddleware`

A middleware configured with either a set of tool names or a predicate. In `before_tool_call`, it calls `channel.approve(tool_name, tool_call_id, args)` and acts on the result:

- `approve` → returns `None` (no interception; tool runs normally)
- `deny` → returns `BeforeToolCallResult(block=True, deny_reason=...)`
- `edit` → returns `BeforeToolCallResult(edited_args=...)` so the loop re-validates and runs the tool with new args

Either decision carries a `hitl_trace: dict` field through to the resulting `ToolResultMessage.details["hitl"]` for audit and trace visibility (see §6.3).

### 3.3 Custom usage

Anyone can write their own tool or middleware that takes a `HitlChannel` and calls `confirm` / `approve` / `ask`. The two built-ins are the common-case packaging, not the only way.

## 4. Channel Protocol

### 4.1 Data types (`cubepi/hitl/types.py`)

```python
from typing import Literal, Any
from pydantic import BaseModel

class Option(BaseModel):
    label: str                           # human-facing
    value: str                           # returned to agent
    description: str | None = None
    allow_input: bool = False            # "Other / please specify" — user types custom text

class Question(BaseModel):
    key: str                             # form field name; key in answers dict
    prompt: str
    options: list[Option] | None = None  # None ⇒ free-text answer
    multi_select: bool = False
    required: bool = True

class ConfirmRequest(BaseModel):
    kind: Literal["confirm"] = "confirm"
    prompt: str
    details: dict[str, Any] | None = None

class ApproveRequest(BaseModel):
    kind: Literal["approve"] = "approve"
    tool_name: str
    tool_call_id: str
    args: dict[str, Any]
    details: dict[str, Any] | None = None

class AskRequest(BaseModel):
    kind: Literal["ask"] = "ask"
    questions: list[Question]

class HitlRequest(BaseModel):
    question_id: str                     # uuid4 string
    thread_id: str | None
    payload: ConfirmRequest | ApproveRequest | AskRequest
    created_at: float

class ApproveAnswer(BaseModel):
    decision: Literal["approve", "deny", "edit"]
    edited_args: dict[str, Any] | None = None  # only when decision == "edit"
    reason: str | None = None                  # only when decision == "deny"

# ask answer: dict[question.key, str | list[str]]
```

### 4.2 `HitlChannel` Protocol (`cubepi/hitl/channel.py`)

```python
from typing import Protocol, AsyncIterator, Any

class HitlChannel(Protocol):
    # ---- agent side ----
    async def confirm(self, prompt: str, *,
                      details: dict | None = None,
                      tool_call_id: str | None = None,
                      timeout: float | None = None) -> bool: ...

    async def approve(self, tool_name: str, tool_call_id: str, args: dict, *,
                      details: dict | None = None,
                      timeout: float | None = None) -> ApproveAnswer: ...

    async def ask(self, questions: list[Question], *,
                  timeout: float | None = None) -> dict[str, str | list[str]]: ...

    # ---- host side ----
    @property
    def pending(self) -> HitlRequest | None: ...
    def subscribe(self) -> AsyncIterator[HitlRequest]: ...
    async def answer(self, question_id: str, answer: Any) -> None: ...
    async def cancel(self, question_id: str, reason: str = "cancelled") -> None: ...

    # ---- resume support (used by Agent.resume) ----
    def attach_resume_answer(self, question_id: str, answer: Any) -> None: ...
```

#### Single-pending invariant

If `confirm/approve/ask` is called while `_pending is not None`, the channel raises `HitlConcurrencyError`. The agent loop's sequential execution makes this logically unreachable; the check exists to catch implementation bugs early.

#### Per-call vs channel-default timeout

Both `InMemoryChannel` and `CheckpointedChannel` accept a `default_timeout: float | None = None` constructor argument. Each `confirm/approve/ask` call may override via the `timeout` kwarg. Timeout expiry raises `HitlTimedOut` from the agent-side `await`, which the surrounding tool or middleware naturally surfaces as `tool_result.is_error=True, content="timed out after N seconds"` plus `hitl_trace` annotations.

Timeout is enforced in the **channel-hosting process only.** Cross-process pending requests do not have a wall-clock timeout reconstituted on resume — if the original process died, the host decides on resume whether to keep waiting, cancel, or answer.

### 4.3 `InMemoryChannel` implementation

In-memory state:

```
_pending: HitlRequest | None
_future: asyncio.Future[Any] | None
_subscribers: list[asyncio.Queue[HitlRequest]]
```

`confirm/approve/ask` flow:

1. Generate `question_id = uuid4()`.
2. Build `HitlRequest`, store as `_pending`.
3. Emit `HitlRequestEvent` (via agent's emit callback if attached) and put into each subscriber queue.
4. `await asyncio.wait_for(self._future, timeout=...)` (or no timeout if None).
5. On success: clear `_pending`, return answer.
6. On `asyncio.TimeoutError`: raise `HitlTimedOut`, clear `_pending`.
7. On `cancel()`: future raises `HitlCancelled`.
8. On signal abort: see §7.

`answer(qid, ans)`:

1. If `_pending is None` or `_pending.question_id != qid`: raise `HitlStaleAnswer`.
2. `self._future.set_result(ans)` — agent side wakes.

### 4.4 `CheckpointedChannel` implementation

Same API as InMemory, plus:

- On `confirm/approve/ask`: after building `HitlRequest` but **before** awaiting, persist via `checkpointer.save_pending_request(thread_id, request)`. Then await the future as before. While awaiting, the agent is still alive in this process — `channel.answer()` from the same process wakes it normally (the same-process happy path).
- On successful answer / cancel / timeout: `await checkpointer.save_pending_request(thread_id, None)` to clear.
- `attach_resume_answer(question_id, answer)`: stores `(qid, answer)` in a one-shot slot. The next `confirm/approve/ask` invocation, **if its newly-generated `question_id` would replay the persisted one** (see §5.2), pops the slot and returns immediately, **bypassing future-and-emit**. Trace span records `from_resume=True`.

The "still alive in this process" case requires no special handling — it behaves like InMemory plus a checkpoint write.

The "process died, web client posts answer hours later" case is handled by `Agent.resume()`, which loads the persisted pending, attaches the answer, and re-enters the loop (§5).

### 4.5 `Agent.detach()` (graceful suspend)

`Agent.detach()` causes any in-flight HITL `await` to raise `HitlDetached`, which the agent loop catches and treats like a clean stop (assistant message keeps its unresolved tool_calls; `pending_request` stays persisted; `Agent.run()` returns `AgentResult(state="suspended", pending_request=...)`).

Without `detach()`, a CheckpointedChannel agent simply blocks until an answer comes in via the same process or the process is killed externally. `detach()` is the explicit "I'm done waiting in this process" signal for hosts that want long-lived suspension across requests.

## 5. Suspend / Resume Protocol

### 5.1 Persisted state

`Checkpointer` (existing protocol) gains two optional methods:

```python
async def save_pending_request(self, thread_id: str,
                                request: HitlRequest | None) -> None: ...
async def load_pending_request(self, thread_id: str) -> HitlRequest | None: ...
```

- `MemoryCheckpointer`: a dict keyed by `thread_id`.
- `SQLiteCheckpointer` / `PostgresCheckpointer` / `MySQLCheckpointer`: new column `pending_request JSON NULL` (or equivalent) on the existing thread row. Each backend gets a migration step.

Backwards compat: existing data without the column reads as `None`; no behavior change for non-HITL flows.

### 5.2 Two resume paths

`Agent` already exposes `resume()` for steering / follow-up resumption. We **do not** overload it. Instead:

**Same-process path (no Agent API needed).** While `agent.run(...)` is awaiting an in-flight HITL call, another coroutine in the same process calls `await channel.answer(question_id, answer)`. The channel resolves its internal future; the awaiting tool / middleware returns; the loop continues. `agent.run()` returns when the conversation completes normally.

**Cross-process / post-detach path: new `Agent.respond(...)`.**

```python
async def respond(self, *, question_id: str | None = None, answer: Any) -> None:
    """Resume an agent whose previous run suspended on a pending HITL request.

    Required when the original channel's in-flight future no longer exists —
    i.e. the original process died, or detach() was called.
    """
    if not (self.thread_id and self.checkpointer):
        raise RuntimeError("respond() requires thread_id + checkpointer")
    if self._state.is_streaming:
        raise RuntimeError("Agent is already running")

    # Load history if not already loaded.
    if not self._state._messages:
        data = await self.checkpointer.load(self.thread_id)
        if data:
            self._state._messages = list(data.messages or [])
            self._extra = dict(data.extra or {})

    pending = await self.checkpointer.load_pending_request(self.thread_id)
    if pending is None:
        raise HitlNoPendingRequest("no pending request on this thread")
    if question_id is None:
        question_id = pending.question_id
    if question_id != pending.question_id:
        raise HitlStaleAnswer(f"answer for {question_id}, pending is {pending.question_id}")

    self._channel.attach_resume_answer(question_id, answer)
    await self.checkpointer.save_pending_request(self.thread_id, None)
    await self._run_hitl_resume()         # wraps run_agent_loop_resume; see §5.3
```

`agent.run(...)`, like today, returns `None`; persistent state lives in `agent.state` and the checkpointer, and the suspended state is observable by inspecting `_state._messages` (last message is an `AssistantMessage` with unresolved tool calls) and by calling `await checkpointer.load_pending_request(thread_id)`.

#### Detecting suspension from `agent.run(...)`

When `Agent.detach()` is called during a pending HITL request (see §4.5), the loop catches `HitlDetached`, exits cleanly, and emits a new `AgentSuspendedEvent(pending_request=...)` so listeners can react. The assistant message keeps its unresolved tool calls; the next `respond()` will pick up from there.

### 5.3 The resume code path

The resume path **never re-streams a model response that has already been streamed once.** Instead it inspects the last message and dispatches:

- **Last message is `AssistantMessage` with at least one `ToolCall` whose `tool_call_id` has no matching `ToolResultMessage` later in the list** — call `execute_tool_calls(...)` directly on that assistant message. The `ask_user` tool / `ConfirmToolCallMiddleware` calls `channel.{ask,approve,confirm}`, which pops the pre-loaded answer and returns immediately. Tool results flow into the normal loop; the next iteration re-streams a fresh model response with the tool results in context.
- **Last message is `ToolResultMessage`** — a HITL request was made *during* `after_tool_call` or another non-execute path (rare). Re-enter the normal `_run_loop`; the channel still has its pre-loaded answer for whatever code path will call `ask()` next.
- **Anything else** — `HitlInconsistentState`, since pending should only have been written from inside an active turn.

Implementation strategy: a new function `run_agent_loop_resume(...)` in `cubepi/agent/loop.py` that wraps `_run_loop` and pre-positions the "execute pending tool_calls" step. Most of `_run_loop` is reused; the only new logic is "skip the first `_stream_assistant_response` call if the last message is an unresolved-tool-call assistant message; jump straight to `execute_tool_calls`." It is exposed via `Agent._run_hitl_resume()` (called from `respond()`) which wires it into `_run_with_lifecycle` exactly like the existing `_run_prompt` / `_run_continuation` paths, so checkpointing on every `MessageEndEvent` continues to work.

### 5.4 Same-process suspend (no resume needed)

If the host stays in-process and `channel.answer()` is called while the agent is `await`ing, the future resolves and the loop continues without ever going through resume. The persisted `pending_request` is cleared on success. This is the fast path for short waits (seconds–minutes); resume is the slow path for long waits (hours–days) or process restarts.

## 6. Loop / Middleware Integration

### 6.1 `BeforeToolCallResult` extension (`cubepi/agent/types.py`)

```python
class BeforeToolCallResult(BaseModel):
    block: bool = False
    reason: str | None = None             # already exists
    edited_args: dict | None = None       # NEW: re-validate & run with these
    deny_reason: str | None = None        # NEW: distinct from generic `reason`
    hitl_trace: dict | None = None        # NEW: merged into tool_result.details["hitl"]
```

`reason` is the existing field — the message surfaced to the model when a tool call is blocked. `deny_reason` is new and is mirrored into `hitl_trace` for audit; we keep them distinct so the wording shown to the model and the wording stored for human audit can differ (the middleware fills both, usually with the same string).

### 6.2 `loop.py` changes

In `_prepare_tool_call`:

```python
if before_result := await before_tool_call(before_ctx, signal=signal):
    if before_result.block:
        return _ImmediateOutcome(
            result=_error_result(before_result.reason or "Tool execution was blocked"),
            is_error=True, blocked_by_hook=True,
            block_reason=before_result.deny_reason or before_result.reason,
            hitl_trace=before_result.hitl_trace,
        )
    if before_result.edited_args is not None:
        try:
            validated_args = tool.parameters.model_validate(before_result.edited_args)
        except ValidationError as exc:
            return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)
    # carry hitl_trace forward
    hitl_trace_for_finalize = before_result.hitl_trace
```

`_PreparedToolCall` and `_ImmediateOutcome`/`_FinalizedOutcome` gain a `hitl_trace: dict | None = None` field. `_make_tool_result_message` merges it:

```python
details = dict(finalized.result.details or {})
if finalized.hitl_trace:
    details["hitl"] = finalized.hitl_trace
return ToolResultMessage(..., details=details, ...)
```

### 6.3 `ConfirmToolCallMiddleware`

```python
class ConfirmToolCallMiddleware(Middleware):
    def __init__(
        self,
        channel: HitlChannel,
        *,
        require_confirm: Callable[[BeforeToolCallContext], bool] | set[str] | None = None,
        details_fn: Callable[[BeforeToolCallContext], dict] | None = None,
    ): ...

    async def before_tool_call(self, ctx, *, signal=None):
        if not self._needs_confirm(ctx):
            return None
        answer = await self._channel.approve(
            tool_name=ctx.tool_call.name,
            tool_call_id=ctx.tool_call.id,
            args=ctx.args.model_dump() if hasattr(ctx.args, "model_dump") else dict(ctx.args),
            details=self._details_fn(ctx) if self._details_fn else None,
        )
        if answer.decision == "approve":
            return None
        if answer.decision == "deny":
            return BeforeToolCallResult(
                block=True, deny_reason=answer.reason,
                hitl_trace={"decision": "deny", "reason": answer.reason},
            )
        if answer.decision == "edit":
            return BeforeToolCallResult(
                edited_args=answer.edited_args,
                hitl_trace={
                    "decision": "edit",
                    "original_args": ctx.args.model_dump() if hasattr(ctx.args, "model_dump") else dict(ctx.args),
                    "edited_args": answer.edited_args,
                },
            )
```

The middleware does **not** implement `after_tool_call` — `hitl_trace` is plumbed through `BeforeToolCallResult` and merged into `tool_result.details` by the loop (§6.2). This means no per-tool-call dict state in the middleware itself.

`require_confirm`:
- `None` ⇒ every tool requires confirm (rare; explicit opt-in)
- `set[str]` ⇒ only tools whose name is in the set
- `Callable[[BeforeToolCallContext], bool]` ⇒ caller-supplied predicate (e.g. inspect args for dangerous flags)

### 6.4 New events

In `cubepi/agent/types.py`:

```python
class HitlRequestEvent(AgentEvent):
    request: HitlRequest

class HitlAnswerEvent(AgentEvent):
    question_id: str
    answer: Any
    cancelled: bool = False
    timed_out: bool = False

class AgentSuspendedEvent(AgentEvent):
    """Emitted when detach() causes the loop to exit with a pending HITL request."""
    pending_request: HitlRequest
```

Channel implementations emit `HitlRequestEvent` / `HitlAnswerEvent` via the same `emit` callable used by `_run_loop` (wired in by Agent at construction). `AgentSuspendedEvent` is emitted by the loop itself when it catches `HitlDetached`.

### 6.5 Trace spans

`cubepi/hitl/_trace.py` opens an OTel span around the `await` inside each `confirm/approve/ask`. Span name: `hitl.confirm` / `hitl.approve` / `hitl.ask`. Attributes:

- `hitl.question_id`
- `hitl.tool_call_id` (approve only)
- `hitl.tool_name` (approve only)
- `hitl.from_resume` — `True` if the answer came from `attach_resume_answer`
- `hitl.outcome` — `approved` / `denied` / `edited` / `answered` / `cancelled` / `timed_out`
- `hitl.duration_seconds`

The tracing import is lazy (try/except ImportError → `_NullSpan`), matching the existing constraint that `cubepi.tracing` is an optional extra.

The trace CLI (`cubepi trace view`) renders these spans inline in the run tree, so an auditor can see "the bash tool was held for 47s waiting for human approval; user edited the command before approving".

## 7. Errors, Cancel, and Abort Semantics

| Scenario | Behavior |
|---|---|
| `channel.cancel(qid, reason)` | Pending future raises `HitlCancelled(reason)`; ask_user / ConfirmToolCallMiddleware lets it propagate; loop catches in tool execution → `tool_result.is_error=True, content="cancelled by user: <reason>"`, `details["hitl"]={"outcome":"cancelled","reason":...}`. `pending_request` cleared from checkpointer. |
| `timeout` exceeded | Same shape as cancel, but `HitlTimedOut`; `details["hitl"]={"outcome":"timed_out","seconds":N}`. |
| `signal.set()` during pending | Channel observes signal (it's passed in via tool's `execute(signal=...)` chain and `Middleware.before_tool_call(signal=...)`) and raises `asyncio.CancelledError`-equivalent; surrounding tool/MW lets it bubble; loop's existing abort path produces `AssistantMessage(stop_reason="aborted")`. `pending_request` is cleared. |
| `Agent.detach()` | Pending future raises `HitlDetached`; loop catches, exits cleanly with assistant message intact (tool_calls still unresolved); `pending_request` stays persisted; `Agent.run()` returns suspended result. |
| `answer(qid)` with unknown / stale qid | `HitlStaleAnswer`. Host code is expected to log / discard. |
| `resume(answer=...)` but no `pending_request` | `HitlNoPendingRequest`. |
| `resume()` (no answer) when there IS a pending | `HitlMissingAnswer`. |
| `confirm/approve/ask` while `_pending is not None` | `HitlConcurrencyError`. (Should be unreachable in practice; presence-check is a guardrail.) |
| Two processes concurrently call `resume()` on the same thread | Out of scope; the existing checkpointer concurrency story applies. Each backend's behavior is documented separately. |
| Resume but last message shape is unexpected | `HitlInconsistentState`. |

## 8. Subagents

Subagents (spawned via parent's tool calls; see existing subagent trace nesting in `dev/specs/2026-05-13-cubepi-cubebox-readiness-design.md` and related) **inherit the parent agent's channel by default**. A subagent's `ask`/`confirm`/`approve` surfaces to the same host. Subagent constructors may override by passing an explicit `channel=` — e.g. a subagent that should auto-approve without prompting can be given a `NoopChannel` that returns canned answers.

Single-pending semantics still hold per channel — if a subagent is asking, the parent loop is blocked in the subagent's `execute_tool` call, so no parallel HITL ever materializes.

## 9. Testing Strategy

Continue cubepi's pattern: `FauxProvider` + real channel + scripted host.

### 9.1 New test helper: `ScriptedChannel`

```python
# cubepi/hitl/testing.py
class ScriptedChannel(HitlChannel):
    """Pre-program answers in order. Tests don't need a separate UI coroutine.

    Implements the full HitlChannel Protocol (subscribe, pending,
    attach_resume_answer, etc.) — `subscribe()` yields recorded requests
    so even tests of the host-event-stream path can use it.
    """
    def __init__(self, answers: list[Any | Callable[[HitlRequest], Any]]): ...
    @property
    def history(self) -> list[HitlRequest]: ...
```

A `Callable` answer can inspect the request and dynamically produce a response (e.g. for testing edit semantics).

### 9.2 Test matrix

| Test | Purpose |
|---|---|
| `test_ask_user_tool_single_question` | Faux model emits ask_user toolcall, channel scripted to answer; tool_result content == answer |
| `test_ask_user_multi_question_form` | Multiple questions, multi_select returns list[str] |
| `test_ask_user_allow_input_option` | Selecting an `allow_input=True` option returns the typed string |
| `test_confirm_middleware_approve` | Dangerous tool, approve → tool runs unchanged |
| `test_confirm_middleware_deny` | deny → tool_result is_error=True, details.hitl.decision=="deny", reason present |
| `test_confirm_middleware_edit` | edit → tool runs with edited args; details.hitl has original and edited |
| `test_confirm_middleware_edit_revalidation_failure` | edited args fail pydantic validation → tool_result is_error |
| `test_in_memory_channel_subscribe_yields_pending` | host subscribe() yields HitlRequest when ask invoked |
| `test_cancel_propagates_as_tool_error` | cancel during pending → is_error="cancelled..." |
| `test_timeout_raises_in_tool` | `ask(..., timeout=0.1)` → is_error="timed out after 0.1s" |
| `test_channel_default_timeout` | InMemoryChannel(default_timeout=…) applies; per-call `timeout=None` disables |
| `test_signal_abort_during_pending` | signal.set() while waiting → AssistantMessage.stop_reason=="aborted" |
| `test_checkpointed_channel_persists_pending` | ask → checkpointer.load_pending_request returns the request; answer clears |
| `test_respond_with_ask_user` | suspend via detach → new Agent + `respond(answer=…)` → loop continues to next model turn |
| `test_respond_with_dangerous_tool_approve` | suspend on approve → `respond(answer=approve)` → tool executes; tool_result.details.hitl present |
| `test_respond_with_dangerous_tool_edit` | `respond` with edit → re-validation runs; tool executes new args |
| `test_respond_stale_answer_raises` | `respond(question_id=wrong)` → HitlStaleAnswer |
| `test_respond_no_pending_raises` | `respond(answer=…)` when nothing pending → HitlNoPendingRequest |
| `test_same_process_answer_no_respond` | host calls `channel.answer()` during `run()` → no `respond()` call needed; run() returns normally |
| `test_subagent_inherits_channel` | subagent's ask surfaces to parent's channel |
| `test_subagent_channel_override` | subagent constructed with explicit channel uses that instead |
| `test_concurrency_check_raises` | manually invoke confirm twice → HitlConcurrencyError |
| `test_trace_emits_hitl_spans` | with tracing extra installed, `cubepi trace view` shows `hitl.ask` span with correct attrs |
| `test_checkpointer_migrations` | (per backend) old schema upgrades to include pending_request column; reads/writes work for both old and new rows |
| `test_event_stream_emits_hitl_events` | agent's event listener receives HitlRequestEvent and HitlAnswerEvent |
| `test_detach_emits_suspended_event` | Agent.detach() during pending → `AgentSuspendedEvent` fires; `run()` returns; assistant message keeps unresolved tool_calls; pending_request remains in checkpointer |

All tests use `FauxProvider`. Resume tests use `MemoryCheckpointer`. Per-backend resume tests (SQLite/Postgres/MySQL) are E2E and gated like existing checkpointer tests.

## 10. Prior Art and Divergences

cubepi's spec process requires comparing major design decisions against established prior art. Here are the relevant systems and how cubepi diverges.

### 10.1 LangGraph (`interrupt()` + `Command(resume=...)`)

LangGraph's HITL is graph-node-based: a node function calls `interrupt(payload)` which raises `GraphInterrupt`. On `Command(resume=value)`, **the entire node function re-runs from the beginning** ("replay" semantics); when it hits `interrupt()` the second time, the call returns the resume value instead of raising.

**cubepi divergence:** we have no graph or nodes — the runtime is a flat loop. We do not replay. Resume re-enters the loop with the channel pre-loaded so the next `await channel.ask()` returns the answer immediately, but **no surrounding code re-runs**. This avoids the "node must be idempotent" caveat LangGraph users have to internalize, and matches cubepi's "the message list is the state" philosophy.

### 10.2 Anthropic Claude Code

Claude Code has two relevant primitives:

- **Permission prompts** for dangerous tool calls (bash, file edits): UI presents "approve / deny / edit"; on edit, user modifies the args and the tool re-runs with new args. Tool result reflects whatever was actually executed.
- **`AskUserQuestion`** tool: model invokes when it needs structured selection; supports per-question options with implicit "Other" free-text input, optional `multiSelect`.

**cubepi inheritance and divergence:**
- `ConfirmToolCallMiddleware` is a direct adaptation of permission prompts.
- `ask_user` tool is a direct adaptation of `AskUserQuestion`.
- Where cubepi diverges: Claude Code is one host (its own CLI/IDE) so it doesn't need an abstraction layer. cubepi is a library used by many hosts (cubebox web, custom TUIs, third parties), so we expose the channel as a protocol and let each host plug in its own surface — synchronous `await` for tool authors, event stream for hosts that prefer subscription.
- cubepi explicitly **does not** ship a `confirm_remember_seconds` / `commandHash` / `approvalTtlSeconds` story (see §10.3). Those are policy layered above the channel by hosts that want them.

### 10.3 craft-agents-oss / pi-agent-server

`packages/core/src/types/message.ts` defines a `permission_request` `AgentEvent` with `requestId`, `toolName`, `command`, `description`, `permissionType` (`bash` | `file_write` | `mcp_mutation` | `api_mutation` | `admin_approval`), plus three policy-ish fields:

- `commandHash` — binds the approval to a hash of the args; if the agent later tries a different command, the grant doesn't apply.
- `approvalTtlSeconds` — the approval is only valid for N seconds.
- `rememberForMinutes` — "yes, and don't ask again for this command for N minutes".

**cubepi divergence:**

- **Event-stream-only vs awaitable channel.** craft-agents-oss is event-stream-only: the agent emits the request and proceeds via some other resumption signal. cubepi offers both — `await channel.confirm/approve/ask` for the tool / middleware author (synchronous mental model), *and* `HitlRequestEvent` / `HitlAnswerEvent` so hosts can subscribe to a stream. Tool authors don't have to think about event-stream protocols.
- **No built-in `commandHash` / `approvalTtlSeconds` / `rememberForMinutes` / `PermissionRequestType`.** These are UX/policy concerns and are **deliberately not in the channel protocol**. Hosts that want them can layer above: cache approvals by `(tool_call_id, hash(args))`, gate by wall-clock, classify by `tool_name`. Keeping the channel minimal aligns with cubepi's "lean core" principle.
- **No fixed `permissionType` taxonomy.** The category is just the tool name; classification (bash vs file_write etc.) is host-side rendering policy.

### 10.4 Workflow engines (Temporal, etc.)

Durable workflow runtimes solve the "suspend across processes" problem in general, with workflow definitions, replay-based determinism, version pinning, and signal handlers. cubepi's HITL is a far simpler subset: one suspend point per thread, no replay, no workflow definitions, no determinism requirement on tool execution. We deliberately do **not** introduce workflow runtime concepts.

## 11. Open Questions / Out of Scope

- **Multi-host fanout** (same channel routed to multiple human approvers, M-of-N). Not supported; channel has a single delivery point per `question_id`. A future extension could subclass `HitlChannel` with consensus semantics, but it's not in this spec.
- **Approval caching / "don't ask again for N minutes."** Not in core channel. Hosts can layer.
- **Approval signing / commandHash binding.** Not in core channel. Hosts can layer.
- **`PermissionRequestType` taxonomy.** Not in core channel. Hosts classify by `tool_name` or `details`.
- **Replay-based determinism.** Out of scope — see §10.4.
- **Voice / non-text rendering hints in `Question`.** Out of scope; `details` is the extensibility point.

## 12. Documentation Deliverables

Per CLAUDE.md ("a feature without docs is not done"), the implementation PR ships:

- `website/docs/guides/hitl.md` — user-facing guide: motivation, when to use `ask_user` vs end-of-turn free text, when to use `ConfirmToolCallMiddleware`, channel implementations, suspend/resume protocol, cross-process recipe.
- `website/docs/recipes/sandbox-confirm.md` — recipe: wiring `ConfirmToolCallMiddleware` to gate `bash`/`write_file` in a cubebox-style web service.
- `website/docs/recipes/ask-user-form.md` — recipe: structured form with multi-select + "Other" free-text option.
- README "Architecture" tree update to mention `cubepi/hitl/`.

## 13. Build Sequence (preview — full plan lives in `dev/plans/`)

Rough phases, finalized in the writing-plans step:

1. Types + `HitlChannel` protocol + `InMemoryChannel` + tests (no agent integration yet).
2. `BeforeToolCallResult` extension + `loop.py` `hitl_trace` merge + tests.
3. `ConfirmToolCallMiddleware` + `ask_user_tool` + integration tests with `FauxProvider`.
4. New events (`HitlRequestEvent`, `HitlAnswerEvent`) + agent wiring of channel-to-emit.
5. `Checkpointer` `save_pending_request` / `load_pending_request` + per-backend migrations.
6. `CheckpointedChannel` + `Agent.detach()` + `Agent.resume()` resume path + tests.
7. Trace integration (lazy OTel) + trace CLI rendering tweaks if needed.
8. Subagent channel inheritance + tests.
9. Documentation (guide, recipes, README).

Each phase has its own test suite that must pass before moving on; codex local review per CLAUDE.md after each milestone.
