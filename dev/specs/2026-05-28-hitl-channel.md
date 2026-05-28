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
- **Prompt-cache prefix invariant (acceptance criterion).** Between pause and resume, the `messages` list must change **only by appending tool-result message(s) and the next assistant turn at the tail.** No inserting, reordering, mutating, or rewriting prior messages — that would invalidate the provider-side prompt cache and cost a fresh prefix re-tokenization on every resume. Every code path that touches `_state._messages` during HITL pause/resume must respect this; the resume tests assert it byte-exactly (see §9.2 `test_resume_preserves_cache_prefix`).

### 2.1 Scope of durable (cross-process) HITL

cubepi's pause/resume gives durable execution **only at two well-defined suspension points** where the cubepi loop can guarantee no replay of agent-side side effects:

1. **`before_tool_call` approval gate** — when `ApprovalPolicyMiddleware` (or any user middleware) calls `channel.approve(...)` *before* the tool's `execute()` body has run, no tool side effects exist yet. The loop can suspend cleanly and resume by either running the (possibly edited) tool body or substituting a synthetic deny tool_result.
2. **`ask_user` built-in tool** — whose entire `execute()` body is `return await channel.ask(...)`. The tool body is a pure HITL wait; resume replays nothing because nothing else happened.

**Custom tools that call `await channel.{ask,confirm,approve}(...)` from inside their `execute()` body alongside other work are explicitly NOT durable across processes.** If such a tool's process dies mid-execute, anything that ran before the channel call would be lost; if cubepi tried to re-enter the tool body on resume, side effects would repeat. cubepi will:

- Support such tools in the **same-process** mode (await blocks, host answers, loop resumes — no replay needed because the process never died).
- Reject them in the **cross-process** mode: a `CheckpointedChannel` started inside such a tool raises `HitlDurabilityNotGuaranteed` unless the channel is constructed with `allow_inside_custom_tool=True`, which the caller uses to acknowledge the idempotency contract (the tool body must be a pure HITL wait at that point, with no preceding observable side effects).

The two built-in suspension points are wired by cubepi itself and carry the `allow_inside_custom_tool` flag implicitly. Documentation will say plainly: **for durable HITL, use `ConfirmToolCallMiddleware` / `ApprovalPolicyMiddleware` or the `ask_user` tool. Anything else is best-effort same-process.**

## 3. Surface Area

Two surfaces, both backed by the same channel.

### 3.1 `ask_user` built-in tool

A `cubepi.hitl.ask_user_tool(channel)` factory returns an `AgentTool` named `ask_user`. The model invokes it like any other tool to ask the user a *structured* question (one or more, each with optional single/multi-select options, optional "allow free-text input" per option).

The tool's `execution_mode="sequential"` — HITL cannot share a turn with other parallel tools. Tool description explicitly steers the model away from using `ask_user` for free-form clarification ("for free-form questions, end your turn with text — the user's next message is your answer").

### 3.2 `ApprovalPolicyMiddleware` and `ConfirmToolCallMiddleware`

Two middlewares for gating tool calls — same internals, different ergonomics:

- **`ApprovalPolicyMiddleware(channel, policy)`** — the policy-driven variant for hosts with a rule engine (e.g. cubebox's command-rule catalog). `policy(ctx)` returns one of `Approve()` / `Deny(reason)` / `AskUser(...)`. `Deny` skips the channel entirely (host-side hard reject); `AskUser` triggers the channel.approve flow.
- **`ConfirmToolCallMiddleware(channel, require_confirm=..., timeout_seconds=...)`** — the simple "always ask the human for these tool names" wrapper. Internally a thin shim over `ApprovalPolicyMiddleware`.

The channel's three-state human response (`approve` / `deny` / `edit`) plus the two host-side outcomes (`policy_deny`, `timed_out`, `cancelled`) all flow as `hitl_trace: dict` through `BeforeToolCallResult` into the resulting `ToolResultMessage.details["hitl"]` for audit and trace visibility (see §6.3).

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
    question_id: str                     # For ApproveRequest: identical to payload.tool_call_id.
                                         # For ConfirmRequest / AskRequest: uuid4 generated by channel.
    thread_id: str | None
    payload: ConfirmRequest | ApproveRequest | AskRequest
    created_at: float
    timeout_seconds: float | None = None # effective timeout for this request
                                         # (per-call timeout if set, else channel default).
                                         # Embedded in the envelope so SSE consumers / UIs can
                                         # render countdowns without separate config.

class ApproveAnswer(BaseModel):
    decision: Literal["approve", "deny", "edit"]
    edited_args: dict[str, Any] | None = None  # only when decision == "edit"
    reason: str | None = None                  # only when decision == "deny"

# ask answer: dict[question.key, str | list[str]]
```

#### `tool_call_id` ↔ `question_id` for approve requests

For `ApproveRequest` (the dangerous-tool-confirm flow), the channel sets `question_id = tool_call_id` — they are **the same string**, not aliases of distinct UUIDs. The two field names exist for clarity at the call site (`question_id` in the channel/host APIs, `tool_call_id` in the inner payload), but no mapping or translation is needed anywhere. Hosts that already track `call_id` from the tool stream pass it directly to `channel.answer(question_id=call_id, ...)` and `agent.respond(question_id=call_id, ...)` — no alias kwarg required.

For `ConfirmRequest` and `AskRequest`, there is no associated tool call, so `question_id` is a freshly-generated uuid4.

### 4.2 `HitlChannel` Protocol (`cubepi/hitl/channel.py`)

```python
from typing import Protocol, AsyncIterator, Any
import asyncio

class HitlChannel(Protocol):
    # ---- agent side ----
    async def confirm(self, prompt: str, *,
                      details: dict | None = None,
                      tool_call_id: str | None = None,
                      timeout: float | None = None,
                      signal: asyncio.Event | None = None) -> bool: ...

    async def approve(self, tool_name: str, tool_call_id: str, args: dict, *,
                      details: dict | None = None,
                      timeout: float | None = None,
                      signal: asyncio.Event | None = None) -> ApproveAnswer: ...

    async def ask(self, questions: list[Question], *,
                  timeout: float | None = None,
                  signal: asyncio.Event | None = None) -> dict[str, str | list[str]]: ...

    # ---- host side ----
    @property
    def pending(self) -> HitlRequest | None: ...
    def subscribe(self) -> AsyncIterator[HitlRequest]: ...
    async def answer(self, question_id: str, answer: Any) -> None: ...
    async def cancel(self, question_id: str, reason: str = "cancelled") -> None: ...

    # ---- resume support (used by Agent.respond) ----
    def attach_resume_answer(self, question_id: str, answer: Any) -> None: ...
```

The `signal` kwarg lets channel calls participate in the agent's abort signal. Channels race the answer future against `signal.wait()`; if signal fires first, the channel raises `HitlAborted` (a subclass of `HitlControlException`, not `Exception`) and clears `_pending`. The built-in middleware / tool factories thread the current `signal` through automatically; custom callers pass it explicitly.

#### Exception hierarchy (`cubepi/hitl/exceptions.py`)

```python
class HitlControlException(BaseException):
    """Base for HITL-driven control-flow exceptions.

    Inherits BaseException, NOT Exception — so the existing broad
    `except Exception` catches in cubepi.agent.tools._prepare_tool_call
    and _execute_prepared do NOT swallow these. The agent loop catches
    HitlControlException explicitly at the right layer."""

class HitlCancelled(HitlControlException):
    def __init__(self, reason: str): ...

class HitlTimedOut(HitlControlException):
    def __init__(self, seconds: float): ...

class HitlDetached(HitlControlException): ...
class HitlAborted(HitlControlException): ...

# Regular Exceptions (caller-fixable misuse, not control flow):
class HitlError(Exception): ...
class HitlConcurrencyError(HitlError): ...
class HitlStaleAnswer(HitlError): ...
class HitlNoPendingRequest(HitlError): ...
class HitlMissingAnswer(HitlError): ...
class HitlInconsistentState(HitlError): ...
class HitlDurabilityNotGuaranteed(HitlError): ...
```

The `BaseException` choice is deliberate and matches the precedent set by `asyncio.CancelledError`. Loop changes (§6.2) explicitly catch `HitlControlException` *before* the broad `except Exception` handlers in `_prepare_tool_call` and `_execute_prepared`, so HITL control flow is never converted into a tool error.

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

`Agent.detach()` causes any in-flight HITL `await` to raise `HitlDetached`, which the agent loop catches and treats like a clean stop (assistant message keeps its unresolved tool_calls; `pending_request` stays persisted; `agent.prompt()` returns normally and an `AgentSuspendedEvent(pending_request=...)` fires on the event stream so listeners can react).

Without `detach()`, a CheckpointedChannel agent simply blocks until an answer comes in via the same process or the process is killed externally. `detach()` is the explicit "I'm done waiting in this process" signal for hosts that want long-lived suspension across requests.

## 5. Suspend / Resume Protocol

### 5.1 Persisted state — per-backend storage plan

`Checkpointer` (existing protocol) gains two optional methods (defined as default `None` returners in the base so existing checkpointers don't have to opt in to compile):

```python
async def save_pending_request(self, thread_id: str,
                                request: HitlRequest | None) -> None: ...
async def load_pending_request(self, thread_id: str) -> HitlRequest | None: ...
```

`HitlRequest` is serialized via `model_dump_json()` (pydantic) — discriminated union on `payload.kind` round-trips cleanly through any JSON storage.

#### `MemoryCheckpointer`

Add `_pending: dict[str, HitlRequest] = {}` alongside the existing in-memory message store. Trivial.

#### `SQLiteCheckpointer`

SQLite has no thread row today (only `messages` + `thread_extra`). Add a new table:

```sql
CREATE TABLE IF NOT EXISTS thread_pending_request (
    thread_id TEXT PRIMARY KEY,
    request_json TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (julianday('now'))
);
```

`save_pending_request(thread_id, request)`: `INSERT OR REPLACE`; if `request is None`, `DELETE WHERE thread_id = ?`. `load_pending_request(thread_id)`: `SELECT request_json …`. No migration needed — `CREATE TABLE IF NOT EXISTS` is idempotent on existing dbs.

#### `PostgresCheckpointer`

`cubepi_threads` has a row per thread. Add a column:

```sql
ALTER TABLE cubepi_threads ADD COLUMN pending_request JSONB NULL;
```

This requires a schema-version bump: `EXPECTED_SCHEMA_VERSION` goes from `1` to `2`. A new migration helper `migrate_v1_to_v2(connection)` adds the column. Backends that find `cubepi_schema_version.version == 1` on startup will refuse to start unless the caller runs the migration first (matching the existing strict version check). The migration is a single `ALTER TABLE … ADD COLUMN` with `NULL` default — non-blocking on Postgres.

`save_pending_request` / `load_pending_request` are `UPDATE cubepi_threads SET pending_request = … WHERE thread_id = …` and `SELECT pending_request FROM cubepi_threads WHERE thread_id = …`. Since the row is created lazily when the first message arrives (existing behavior), HITL writes assume the row exists by the time pending is being persisted (which is always true — `_pending` only fires inside an active turn that already produced an assistant message).

#### `MySQLCheckpointer`

Same as Postgres: add `pending_request JSON NULL` to `cubepi_threads`, bump `EXPECTED_SCHEMA_VERSION` to `2`, ship a `migrate_v1_to_v2()` helper. MySQL 8.0+ JSON type round-trips pydantic JSON cleanly.

#### Backwards compatibility

- SQLite: no migration needed (new independent table).
- Postgres / MySQL: explicit migration step. Documented in the release notes for this PR. The schema-version assertion fails loudly on old data, so users can't silently lose pending state by upgrading without migrating.

### 5.2 Two resume paths

`Agent` already exposes `resume()` for steering / follow-up resumption. We **do not** overload it. Instead:

**Same-process path (no Agent API needed).** While `agent.prompt(...)` is awaiting an in-flight HITL call, another coroutine in the same process calls `await channel.answer(question_id, answer)`. The channel resolves its internal future; the awaiting tool / middleware returns; the loop continues. `agent.prompt()` returns when the conversation completes normally.

**Cross-process / post-detach path: new `Agent.respond(...)`.**

```python
async def respond(
    self,
    *,
    question_id: str | None = None,
    answer: Any,
) -> None:
    """Resume an agent whose previous run suspended on a pending HITL request.

    Required when the original channel's in-flight future no longer exists —
    i.e. the original process died, or detach() was called.

    For approve-kind pending, question_id == tool_call_id (see §4.1); pass
    whichever the host already tracks.
    """
    if self._channel is None:
        raise HitlError("agent has no channel bound; pass channel= to Agent()")
    if not (self.thread_id and self.checkpointer):
        raise RuntimeError("respond() requires thread_id + checkpointer")

    async with self._run_lock:    # see "Concurrency guard" below
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
            raise HitlStaleAnswer(
                f"answer for {question_id}, pending is {pending.question_id}"
            )

        self._channel.attach_resume_answer(question_id, answer)
        # NOTE: pending_request is NOT cleared yet — see ordering rules below.
        await self._run_hitl_resume()         # wraps run_agent_loop_resume; see §5.3
```

#### Pending-clear ordering (crash safety)

`pending_request` must be cleared **only after** the resulting `ToolResultMessage` (or synthetic deny) has been appended and persisted. Otherwise a crash between clear-pending and tool-execute would leave the thread in a state with neither pending nor tool_result, and `respond()` re-runs would raise `HitlNoPendingRequest`. The implementation:

1. `attach_resume_answer(qid, answer)` — channel's resume-answer slot now holds the answer.
2. Call `_run_hitl_resume()` which dispatches into `run_agent_loop_resume(...)`.
3. Inside the loop, the gated tool runs (or synthetic deny is built) and the resulting `ToolResultMessage` is checkpointed via the existing `MessageEndEvent` → `checkpointer.append(...)` path.
4. **Immediately after** the `ToolResultMessage` is checkpointed, the resume code clears `pending_request` via `checkpointer.save_pending_request(thread_id, None)` in the same checkpointer transaction where possible. For SQLite this is two separate writes; for Postgres/MySQL we wrap both writes in a transaction so they're atomic.

If the process crashes before step 4: on the next `respond()` attempt, the resume-answer slot is gone (in-memory only) but the channel sees the pending is still persisted **and** the conversation messages already include the tool_result — `attach_resume_answer` is idempotent and the loop skips re-execution of the gated tool (it sees the tool_result is already there in messages list). `pending_request` is cleared on this second attempt. **The invariant is: `pending_request` cleared ⇔ corresponding tool_result message exists in checkpointer.**

#### Concurrency guard

`Agent` gains `self._run_lock: asyncio.Lock`. `prompt()`, `resume()` (existing), and `respond()` all acquire it. The existing `_state.is_streaming` flag becomes a debug-only signal — the lock is the source of truth. Two concurrent `respond()` calls from different coroutines on the same `Agent` instance now block on the lock instead of racing the `is_streaming` check.

Cross-process concurrency (two processes both calling `respond()` on the same `thread_id`) is NOT protected here — that's the existing checkpointer concurrency story. Hosts that worry about it use their checkpointer's locking primitives (e.g. Postgres advisory lock on `thread_id`).

#### Async semantics

`respond()` is `async` and **does not return immediately** — it runs the loop forward and returns when the next pause or `AgentEndEvent` fires. Hosts that want to free the request thread for an SSE stream wrap it in `asyncio.create_task(agent.respond(...))` and consume events via `agent.subscribe()` (existing API) — exactly the pattern used today for `agent.prompt(...)`.

`agent.prompt(...)`, like today, returns `None`; persistent state lives in `agent.state` and the checkpointer, and the suspended state is observable by inspecting `agent.state.messages` (last message is an `AssistantMessage` with unresolved tool calls) and by calling `await checkpointer.load_pending_request(thread_id)`.

#### Detecting suspension from `agent.prompt(...)`

When `Agent.detach()` is called during a pending HITL request (see §4.5), `Agent.detach()` itself emits `AgentSuspendedEvent(pending_request=self._channel.pending)` *before* triggering the exception, then the loop catches `HitlDetached` and exits silently. (Earlier drafts had the loop emit the event, but the loop has no channel handle — emitting with `pending_request=None` violated the event contract. The Agent layer has the real handle and emits with the proper payload.) The assistant message keeps its unresolved tool calls; the next `respond()` will pick up from there.

For hosts that need to probe pending state, `Agent` exposes **two** APIs — sync and async — because the in-memory channel slot is cheap to read but the checkpointer is not:

```python
@property
def in_flight_hitl_request(self) -> HitlRequest | None:
    """Synchronous read of the channel's in-memory pending slot.

    Returns the pending request iff a HITL `await` is currently outstanding
    in THIS process. Returns None when the agent has detached or the
    process is a fresh resume that hasn't called channel.{ask,approve,confirm}() yet.
    """

async def load_pending_hitl_request(self) -> HitlRequest | None:
    """Async lookup that consults the checkpointer.

    Returns the pending HITL request persisted for this thread, or None.
    Use this after detach() / in fresh resume processes where the in-flight
    slot may be empty but pending data lives in the checkpointer.
    """
```

A typical post-`prompt()` check:

```python
await agent.prompt(message)
pending = agent.in_flight_hitl_request or await agent.load_pending_hitl_request()
if pending is not None:
    # render pending.payload to the user; later call agent.respond(...)
    ...
```

#### Aborting a suspended thread

For "user closed the conversation" / "admin kill switch", a method beyond per-question `channel.cancel()` is needed. Semantics: **abort closes the conversation.** No further model turn is engaged; the thread is terminal until the host explicitly continues with a new `prompt()`.

```python
async def abort_pending(self, reason: str = "aborted by host") -> None:
    """Cancel any pending HITL request on this thread, fake-deny the gated tool call,
    and CLOSE the conversation (no further model call).

    - If a request is in-flight in the same process: channel.cancel(qid, reason)
      causes the awaiting tool / middleware to surface a deny tool_result via the
      normal path. The loop then sees the conversation has an aborted tool_result
      and exits via a synthesized terminal AssistantMessage(stop_reason="aborted").
    - If suspended cross-process (in-flight slot empty, checkpointer pending set):
      load checkpoint + pending; synthesize a ToolResultMessage with
      is_error=True, content=f"aborted: {reason}", details.hitl={"decision":"aborted","reason":...}
      and append it for the gated tool_call_id; then append a terminal
      AssistantMessage with empty content and stop_reason="aborted";
      clear pending_request; emit AgentSuspendedEvent or a new AgentAbortedEvent.

    No new model call is made during abort. The conversation history ends with
    a stop_reason="aborted" assistant turn, which any subsequent prompt() can
    follow naturally (the new user message simply appends after the aborted
    assistant turn — providers tolerate this).
    """
```

The synthetic deny path uses the same `hitl_trace` schema as policy/human deny so trace and audit semantics stay uniform. `AgentEndEvent` does not currently carry a `stop_reason`; we add a new `AgentAbortedEvent` (or extend `AgentEndEvent` with an optional `stop_reason` field) so listeners can distinguish normal completion from abort.

### 5.3 The resume code path

The resume path **never re-streams a model response that has already been streamed once.** The only durable suspension points (per §2.1) are before-tool-call approval and the `ask_user` tool body, both of which produce the same observable state: the last persisted message is an `AssistantMessage` containing at least one `ToolCall` whose `tool_call_id` has no matching `ToolResultMessage`. Resume thus has a single case:

- **Last message is `AssistantMessage` with one or more unresolved `ToolCall`s** — call `execute_tool_calls(...)` directly on that assistant message. The `ask_user` tool / `ApprovalPolicyMiddleware` calls `channel.{ask,approve,confirm}`, which pops the pre-loaded answer and returns immediately. Tool results flow into the normal loop; the next iteration re-streams a fresh model response with the tool results in context.
- **Anything else** — `HitlInconsistentState`. The only way to reach a state where pending is set but the last message is something other than an unresolved-tool-call AssistantMessage is data corruption or external mutation; we fail loudly.

(Earlier drafts allowed a "last message is `ToolResultMessage`" case for HITL invoked from `after_tool_call`. That's removed: `after_tool_call` runs *before* the `ToolResultMessage` is emitted in the current loop, so this case is unreachable.)

Implementation strategy: a new function `run_agent_loop_resume(...)` in `cubepi/agent/loop.py` that:

1. Asserts the last-message invariant (raise `HitlInconsistentState` otherwise).
2. Skips the initial `_stream_assistant_response` call.
3. Calls `execute_tool_calls(...)` on the last assistant message — the channel pops its pre-loaded answer, the tool executes (or synthetic deny is built), `ToolResultMessage`(s) are appended and checkpointed via the normal `MessageEndEvent` path.
4. Clears `pending_request` (see §5.2 ordering).
5. Emits `TurnEndEvent`.
6. Runs `should_stop_after_turn` if configured.
7. **Now** drains any queued steering messages (matching the existing `_run_loop` ordering — steering MUST come after tool_results to preserve tool_use/tool_result adjacency).
8. Falls through to the normal `_run_loop` for the next model call.

Step 7 ordering matters: appending steering before tool_results would break the Anthropic-strict adjacency invariant the existing loop already protects. The resume path inherits that constraint.

`run_agent_loop_resume` is exposed via `Agent._run_hitl_resume()` (called from `respond()`), which wires it into `_run_with_lifecycle` exactly like the existing `_run_prompt` / `_run_continuation` paths, so checkpointing on every `MessageEndEvent` continues to work and the `_run_lock` is held for the duration.

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

### 6.1.1 `Agent` channel wiring

`Agent.__init__` gains a new keyword argument:

```python
def __init__(
    self,
    *,
    provider, model, ...,
    channel: HitlChannel | None = None,    # NEW
    **kwargs,
):
    ...
    self._channel = channel
    if channel is not None:
        # Channel emits HitlRequestEvent / HitlAnswerEvent through this agent's
        # event listeners; bind during construction.
        channel._bind_emit(lambda e: self._process_event(e))
```

`agent.channel` (read-only property) returns the bound channel or `None`. Built-in factories pull from `agent.channel` by default:

```python
agent = Agent(provider=..., model=..., channel=InMemoryChannel())
agent.tools.append(ask_user_tool(agent.channel))
agent.middlewares.append(ConfirmToolCallMiddleware(agent.channel, require_confirm={"bash"}))
```

`Agent.respond()`, `Agent.detach()`, `Agent.abort_pending()`, `agent.in_flight_hitl_request` all require `self._channel is not None` — they raise `HitlError("agent has no channel bound")` if called on a non-HITL agent.

### 6.2 `loop.py` changes

In `_prepare_tool_call`, the existing broad `except Exception` becomes selective — `HitlControlException` (from §4.2) is allowed to propagate; only regular `Exception` is captured as a tool error:

```python
try:
    before_result = await before_tool_call(before_ctx, signal=signal)
except HitlControlException:
    raise   # HitlAborted / HitlDetached / etc bubble to _run_loop
except Exception as exc:
    return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)

if before_result:
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
    hitl_trace_for_finalize = before_result.hitl_trace
```

Same selective catch is added around `tool.execute(...)` in `_execute_prepared`.

At the `_run_loop` level, a new outer try-block wraps the inner `_run_loop` body to catch `HitlDetached` / `HitlAborted` and exit cleanly **without** emitting any extra event — the Agent caller (`Agent.detach()` / `Agent.abort_pending()`) is responsible for emitting `AgentSuspendedEvent` / `AgentAbortedEvent` with the real payload before triggering the exception.

`_PreparedToolCall` and `_ImmediateOutcome`/`_FinalizedOutcome` gain a `hitl_trace: dict | None = None` field. `_make_tool_result_message` merges it defensively (the existing `AgentToolResult.details` is typed `Any`, not `dict` — see SHOULD-FIX 13):

```python
def _merge_hitl_details(base: Any, hitl: dict | None) -> Any:
    if hitl is None:
        return base
    if base is None:
        return {"hitl": hitl}
    if isinstance(base, dict):
        merged = dict(base)
        merged["hitl"] = hitl
        return merged
    # base is some other type (rare); preserve under reserved key
    return {"_non_dict_details": base, "hitl": hitl}

details = _merge_hitl_details(finalized.result.details, finalized.hitl_trace)
return ToolResultMessage(..., details=details, ...)
```

### 6.3 Middlewares: `ConfirmToolCallMiddleware` and `ApprovalPolicyMiddleware`

Two host-facing middlewares ship in `cubepi.hitl`. They share an internal helper but differ in how policy is expressed.

#### 6.3.1 `ApprovalDecision` — the host policy contract

Hosts whose policy engine needs to express "absolutely deny without asking" (e.g. cubebox's command-rule engine that has an explicit-block tier) need a richer return type than a yes-or-no predicate. The canonical decision type:

```python
# cubepi/hitl/policy.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Approve: pass

@dataclass(frozen=True)
class Deny:
    reason: str

@dataclass(frozen=True)
class AskUser:
    prompt: str | None = None              # extra context shown to the human
    timeout_seconds: float | None = None   # overrides channel default for this call
    details: dict | None = None            # extra payload (e.g. matched rule, impact preview)

ApprovalDecision = Approve | Deny | AskUser
```

A policy function is `Callable[[BeforeToolCallContext], ApprovalDecision | Awaitable[ApprovalDecision]]`. Returning `Approve()` is the pass-through; `Deny(reason)` skips the channel entirely and goes straight to the block path (with `hitl_trace={"decision":"policy_deny", "reason": ...}` so audits can distinguish policy-deny from human-deny); `AskUser(...)` triggers the channel.

#### 6.3.2 `ApprovalPolicyMiddleware`

```python
class ApprovalPolicyMiddleware(Middleware):
    def __init__(self, channel: HitlChannel, policy: Callable[..., ApprovalDecision | Awaitable[ApprovalDecision]]):
        self._channel = channel
        self._policy = policy

    async def before_tool_call(self, ctx, *, signal=None):
        decision = self._policy(ctx)
        if inspect.isawaitable(decision):
            decision = await decision

        if isinstance(decision, Approve):
            return None
        if isinstance(decision, Deny):
            return BeforeToolCallResult(
                block=True, deny_reason=decision.reason,
                hitl_trace={"decision": "policy_deny", "reason": decision.reason},
            )
        if isinstance(decision, AskUser):
            return await self._ask_and_translate(ctx, decision)
        raise TypeError(f"policy returned unexpected {type(decision).__name__}")

    async def _ask_and_translate(self, ctx, ask: AskUser):
        try:
            answer = await self._channel.approve(
                tool_name=ctx.tool_call.name,
                tool_call_id=ctx.tool_call.id,
                args=_args_to_dict(ctx.args),
                details=ask.details,
                timeout=ask.timeout_seconds,
            )
        except HitlTimedOut:
            return BeforeToolCallResult(
                block=True, deny_reason="approval_timeout",
                hitl_trace={"decision": "timed_out"},
            )
        except HitlCancelled as exc:
            return BeforeToolCallResult(
                block=True, deny_reason=f"cancelled: {exc.reason}",
                hitl_trace={"decision": "cancelled", "reason": exc.reason},
            )
        if answer.decision == "approve":
            return None
        if answer.decision == "deny":
            return BeforeToolCallResult(
                block=True, deny_reason=answer.reason,
                hitl_trace={"decision": "human_deny", "reason": answer.reason},
            )
        if answer.decision == "edit":
            return BeforeToolCallResult(
                edited_args=answer.edited_args,
                hitl_trace={
                    "decision": "edit",
                    "original_args": _args_to_dict(ctx.args),
                    "edited_args": answer.edited_args,
                },
            )
```

Note the `HitlTimedOut` / `HitlCancelled` catch — they get translated into clean `deny_reason="approval_timeout"` / `"cancelled: …"` blocks rather than raw tool-execution errors (Patch 4 / **§7 timeout-as-deny semantics**).

#### 6.3.3 `ConfirmToolCallMiddleware` — "always ask, ship-and-go" variant

For users who don't have a policy engine and just want "ask the human for these tool names" out of the box:

```python
class ConfirmToolCallMiddleware(ApprovalPolicyMiddleware):
    def __init__(
        self,
        channel: HitlChannel,
        *,
        require_confirm: Callable[[BeforeToolCallContext], bool] | set[str] | None = None,
        details_fn: Callable[[BeforeToolCallContext], dict] | None = None,
        timeout_seconds: float | None = None,
    ):
        def policy(ctx) -> ApprovalDecision:
            if require_confirm is None or self._matches(require_confirm, ctx):
                return AskUser(
                    details=details_fn(ctx) if details_fn else None,
                    timeout_seconds=timeout_seconds,
                )
            return Approve()
        super().__init__(channel, policy=policy)
```

`require_confirm` semantics unchanged from the original sketch (set / predicate / None=all). This is now a thin wrapper around the policy middleware.

Neither middleware implements `after_tool_call` — `hitl_trace` flows through `BeforeToolCallResult` and is merged into `tool_result.details` by the loop (§6.2), so there is no per-tool-call dict state in the middleware itself.

### 6.3.4 `compose_middleware` redesign

The existing `compose_middleware()` in `cubepi/middleware/base.py` shortcuts `before_tool_call` results: it returns the first one whose `.block` is `True` and **discards** non-blocking results. That drops `edited_args` and `hitl_trace`, which we need. New behavior for `composed_before`:

```python
async def composed_before(ctx, *, signal=None):
    accumulated_hitl_trace: dict = {}
    edited_args: dict | None = None
    deny_reason: str | None = None
    block_reason: str | None = None
    blocked = False

    for mw in before_chain:
        # If a prior middleware edited args, the next middleware in the chain
        # sees the EDITED ctx (we rebuild ctx with new args so subsequent
        # middlewares can re-validate / re-deny against the edited form).
        cur_ctx = ctx if edited_args is None else _rebuild_ctx_with_args(ctx, edited_args)
        result = await mw.before_tool_call(cur_ctx, signal=signal)
        if result is None:
            continue
        if result.hitl_trace:
            # Last writer wins per key; earlier traces preserved under "_chain".
            if accumulated_hitl_trace:
                accumulated_hitl_trace.setdefault("_chain", []).append(dict(accumulated_hitl_trace))
            accumulated_hitl_trace.update(result.hitl_trace)
        if result.edited_args is not None:
            edited_args = result.edited_args
        if result.block:
            blocked = True
            block_reason = result.reason or block_reason
            deny_reason = result.deny_reason or deny_reason
            break    # short-circuit on block; later middlewares don't run

    if not blocked and edited_args is None and not accumulated_hitl_trace:
        return None
    return BeforeToolCallResult(
        block=blocked,
        reason=block_reason,
        deny_reason=deny_reason,
        edited_args=edited_args,
        hitl_trace=accumulated_hitl_trace or None,
    )
```

Conflict semantics:
- **edited_args**: last writer in the chain wins; subsequent middlewares see the edit.
- **block**: short-circuits — any blocking middleware stops the chain immediately. Edits accumulated before the block are discarded (since the tool won't run).
- **hitl_trace**: merge by key; last writer wins per key; previous values archived under `_chain` so audit can see the full sequence.

These semantics are documented inline in the `compose_middleware` source. Tests cover all three cases (`test_compose_before_edit_chain`, `test_compose_before_block_after_edit`, `test_compose_before_hitl_trace_merge`).

### 6.3.5 HITL-capable tools must be sequential

Tools that call `await channel.{ask,confirm,approve}(...)` from inside their `execute()` body MUST declare `execution_mode="sequential"`. The built-in `ask_user_tool(channel)` factory sets it automatically. When the loop assembles a turn's tool_calls and any HITL-capable tool is among them, it falls back to `_execute_sequential` (this is already the existing behavior — `has_sequential` in `execute_tool_calls`).

For `before_tool_call`-based HITL (the `ApprovalPolicyMiddleware` / `ConfirmToolCallMiddleware` path), the existing loop **already serializes** `_prepare_tool_call` calls — both `_execute_parallel` and `_execute_sequential` call `_prepare_tool_call` in their outer for-loop before kicking off tool tasks. So even with parallel tools, only one approval prompt is outstanding at a time. We add a regression test (`test_parallel_tools_serialize_approvals`) to lock this in.

Subagent tools are themselves tools from the parent loop's POV. If a subagent uses HITL, its containing tool must be sequential at the parent level — `subagent_tool(...)` factory will set `execution_mode="sequential"` when constructed with a channel.

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

Channel implementations emit `HitlRequestEvent` / `HitlAnswerEvent` via the same `emit` callable used by `_run_loop` (wired in by Agent at construction). `AgentSuspendedEvent` and `AgentAbortedEvent` are emitted by the **Agent layer** (`Agent.detach()` / `Agent.abort_pending()`), not the loop — the Agent has the channel handle needed to populate `pending_request` with the real payload.

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
| `channel.cancel(qid, reason)` in **ask_user tool** context | Channel raises `HitlCancelled(reason)` (subclass of `HitlControlException`, see §4.2); `_execute_prepared`'s selective handler lets it propagate to `_run_loop`, which catches and produces `tool_result.is_error=True, content="cancelled by user: <reason>"`, `details["hitl"]={"outcome":"cancelled","reason":...}`. `pending_request` cleared from checkpointer. |
| `channel.cancel(qid, reason)` in **approve middleware** context | `ApprovalPolicyMiddleware` catches `HitlCancelled` and returns `BeforeToolCallResult(block=True, deny_reason=f"cancelled: {reason}", hitl_trace={"decision":"cancelled", ...})` — the gated tool never executes, model sees a clean denial (not a raw error). |
| `timeout` exceeded in **ask_user tool** context | Same shape as cancel-in-ask, but `HitlTimedOut(seconds=N)`; `details["hitl"]={"outcome":"timed_out","seconds":N}`. |
| `timeout` exceeded in **approve middleware** context | Translated to `deny_reason="approval_timeout"` + `hitl_trace={"decision":"timed_out"}` — a clean fake-deny, *not* a tool error. Matches cubebox's "timeout → fake-deny so LLM keeps reasoning" requirement. |
| `signal.set()` during pending | Channel observes the `asyncio.Event` (passed in via tool's `execute(signal=...)` and `Middleware.before_tool_call(signal=...)`) racing it against the answer future; if signal wins, channel raises `HitlAborted`. `HitlAborted` propagates through the selective handlers up to `_run_loop`, which exits **silently** — the Agent caller (`Agent.abort_pending()`, which set the signal) is responsible for emitting `AgentAbortedEvent` with the proper payload. `pending_request` is cleared in `_on_pending_cleared` (HitlAborted ≠ HitlDetached). |
| `Agent.detach()` | `Agent.detach()` emits `AgentSuspendedEvent(pending_request=channel.pending)` then sets `HitlDetached` on the channel's future. Channel raises `HitlDetached`; `_run_loop` catches at the outer level and exits silently (no event from loop). Assistant message intact, tool_calls still unresolved, `pending_request` stays persisted; the next `respond()` picks up. |
| `answer(qid)` with unknown / stale qid | `HitlStaleAnswer` (`HitlError`, regular `Exception`). Host code is expected to log / discard. |
| `respond(answer=...)` but no `pending_request` | `HitlNoPendingRequest`. |
| `respond()` (no answer) when there IS a pending | `HitlMissingAnswer`. |
| `confirm/approve/ask` while `_pending is not None` | `HitlConcurrencyError`. (Should be unreachable in practice — see §6.3.5 sequential-execution guarantees; presence-check is a guardrail.) |
| Two processes concurrently call `respond()` on the same thread | Out of scope at the channel level; existing checkpointer concurrency story applies. Postgres/MySQL hosts can use advisory locks; SQLite hosts run a single writer by convention. |
| Resume but last message shape is not unresolved-tool-call AssistantMessage | `HitlInconsistentState` — unreachable in normal operation; indicates corruption or external mutation. |
| Custom tool starts `CheckpointedChannel.ask()` from inside `execute()` body | `HitlDurabilityNotGuaranteed` unless the channel was constructed with `allow_inside_custom_tool=True`. See §2.1. |

## 8. Subagents

cubepi today does not ship a built-in subagent tool — subagents are constructed by users wrapping an inner `Agent` instance inside a tool's `execute()` body. Since that pattern puts the subagent's run **inside a parent tool's execute()**, two constraints follow directly from §2.1 and §6.3.5:

1. **The parent tool wrapping a subagent that may use HITL must declare `execution_mode="sequential"`.** Otherwise multiple parallel subagents could all suspend at once and violate the single-pending invariant.
2. **The subagent's channel should be the parent's channel.** The natural pattern: the subagent-tool factory accepts `channel=parent.channel` from the constructor:

   ```python
   def subagent_tool(name, provider, model, *, channel=None, **kwargs):
       async def execute(call_id, args, *, signal, on_update):
           inner = Agent(provider=provider, model=model, channel=channel, ...)
           await inner.prompt(args.task)
           return AgentToolResult(content=[TextContent(text=inner.state.last_text)])
       return AgentTool(
           name=name,
           parameters=...,
           execute=execute,
           execution_mode="sequential" if channel else "parallel",
       )
   ```

3. **`HitlControlException`s bubble out of the inner `prompt()` call into the parent's tool execute.** The parent loop's selective handler in `_execute_prepared` (see §6.2) lets them propagate further up so the parent itself can be detached / aborted along with the subagent. The subagent's pending request lives in **the subagent's own thread_id** in the checkpointer (subagents that need durable resume must have their own `thread_id`); the parent's `pending_request` slot is unaffected.

For users who want a built-in "no-prompt always approve" channel for subagents (a common testing pattern), `cubepi.hitl.testing.NoopChannel` returns canned answers and never blocks.

Single-pending semantics still hold per channel — if a subagent is asking, the parent loop is blocked in the subagent's `execute_tool` call, so no parallel HITL ever materializes at the channel level.

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
| `test_pending_hitl_request_property` | After detach, `agent.pending_hitl_request` reflects checkpointer state; during in-flight ask, reflects channel state; otherwise None |
| `test_approval_policy_approve_passthrough` | Policy returns `Approve()` → channel never invoked → tool runs |
| `test_approval_policy_deny_skips_channel` | Policy returns `Deny(reason)` → channel never invoked → block path; `hitl_trace.decision == "policy_deny"` |
| `test_approval_policy_ask_user_round_trip` | Policy returns `AskUser(timeout=…)` → channel.approve invoked → human reply → tool runs or blocks accordingly |
| `test_approval_policy_timeout_becomes_deny` | `AskUser(timeout=0.05)` + no answer → block with `deny_reason="approval_timeout"`, `hitl_trace.decision == "timed_out"` (NOT a tool error) |
| `test_approval_policy_cancel_becomes_deny` | mid-ask `channel.cancel()` → block with `deny_reason="cancelled: …"`, `hitl_trace.decision == "cancelled"` |
| `test_abort_pending_in_flight` | abort_pending() while same-process await → channel.cancel happens, normal cancel path |
| `test_abort_pending_cross_process` | abort_pending() on a thread with checkpoint-only pending → synthetic deny tool_result appended; pending cleared; AgentEndEvent stop_reason="aborted"; no model call made |
| `test_hitl_request_envelope_carries_timeout` | `channel.approve(..., timeout=42)` → emitted `HitlRequest.timeout_seconds == 42`; channel default applies when per-call omitted |
| `test_resume_preserves_cache_prefix_memory` | (MemoryCheckpointer) Pause-and-resume on a dangerous-tool flow: serialize the messages list before pause and again before the model call after resume; assert the prefix (everything except the new tool_result + new assistant turn) is byte-identical |
| `test_resume_preserves_cache_prefix_sqlite` | Same, but using SQLiteCheckpointer so the messages round-trip through JSON storage. Asserts byte-identical prefix in `provider.stream(...)` calls captured by FauxProvider's recording mode |
| `test_resume_preserves_cache_prefix_postgres` | Same, gated on the Postgres E2E test env (mirrors existing checkpointer E2E patterns) |
| `test_question_id_equals_tool_call_id_for_approve` | For ApproveRequest, channel emits `HitlRequest.question_id == payload.tool_call_id` |
| `test_hitl_control_exceptions_not_swallowed_by_tool_handler` | Raise `HitlAborted` from inside a tool's execute; assert it propagates out of `_run_loop` (does NOT become a tool error result) |
| `test_compose_before_edit_chain` | Two middlewares: first edits args, second sees edited args and adds a hitl_trace; assert tool runs with second-edited args, details.hitl carries both traces |
| `test_compose_before_block_after_edit` | First middleware edits args; second middleware blocks. Assert tool does not run; tool_result.is_error=True; edited_args from first MW are discarded |
| `test_compose_before_hitl_trace_merge` | Two middlewares each add hitl_trace; assert merged dict contains keys from both, with earlier values preserved under `_chain` |
| `test_parallel_tools_serialize_approvals` | tool_execution="parallel" + ApprovalPolicyMiddleware on two dangerous tools → approvals issued sequentially, not concurrently; channel never sees concurrent ask |
| `test_hitl_tool_must_be_sequential` | ask_user_tool factory sets execution_mode="sequential"; manually constructing a parallel HITL tool raises at first invocation |
| `test_channel_signal_abort` | `signal.set()` while channel awaits → channel raises `HitlAborted`; `_on_pending_cleared` clears persisted pending; loop exits silently (Agent.abort_pending emits the AgentAbortedEvent separately) |
| `test_durability_not_guaranteed_for_custom_tool` | Custom tool calls CheckpointedChannel.ask() inside execute() → raises HitlDurabilityNotGuaranteed; can be bypassed with allow_inside_custom_tool=True |
| `test_respond_concurrent_calls_serialize` | Two concurrent respond() calls on same agent block on _run_lock; second one sees HitlNoPendingRequest after first completes |
| `test_respond_crash_recovery_idempotent` | Simulate crash between attach_resume_answer and tool execution → on retry, attach_resume_answer is idempotent and the loop sees pending still set with tool_result already in messages → completes correctly |
| `test_postgres_schema_v2_migration` | (Postgres E2E) Existing v1 schema + pending_request column added via migrate_v1_to_v2 → reads/writes work; old threads without pending_request behave as None |
| `test_mysql_schema_v2_migration` | (MySQL E2E) Same |
| `test_sqlite_pending_table_idempotent` | Re-running CREATE TABLE IF NOT EXISTS is safe; multiple connections to existing DB don't break |

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

- **Durable HITL from inside custom tool bodies.** See §2.1: only the `before_tool_call` approval gate and the `ask_user` built-in tool body are durable across processes. Custom tools that mix HITL with other work are same-process-only unless they opt in to `allow_inside_custom_tool` and accept the idempotency contract themselves.
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

1. Types + `HitlChannel` protocol + `HitlControlException` hierarchy + `InMemoryChannel` + tests (no agent integration yet).
2. `BeforeToolCallResult` extension + `compose_middleware` redesign + `loop.py` selective-exception + `hitl_trace` merge + sequential-HITL enforcement + tests.
3. `Agent.__init__(channel=...)` wiring + `agent.channel` property + `agent.in_flight_hitl_request` + emit binding.
4. `ApprovalDecision` types + `ApprovalPolicyMiddleware` + `ConfirmToolCallMiddleware` shim + `ask_user_tool` (sequential) + integration tests with `FauxProvider`.
5. New events (`HitlRequestEvent`, `HitlAnswerEvent`, `AgentSuspendedEvent`, `AgentAbortedEvent`).
6. `Checkpointer.save_pending_request` / `load_pending_request` per backend: Memory dict; SQLite new table; Postgres schema v1→v2 migration; MySQL schema v1→v2 migration; tests including v2 migration E2E for SQL backends.
7. `CheckpointedChannel` (with `allow_inside_custom_tool` guard) + `agent.load_pending_hitl_request()` + `Agent.detach()` + `Agent.respond()` (incl. `_run_lock` + pending-clear ordering) + `Agent.abort_pending()` + tests (including all `test_resume_preserves_cache_prefix_*` variants and `test_respond_crash_recovery_idempotent`).
8. Trace integration (lazy OTel) + trace CLI rendering tweaks if needed.
9. Subagent channel inheritance + sequential constraint at parent level + tests.
10. Documentation (guide, recipes, README).

Each phase has its own test suite that must pass before moving on; codex local review per CLAUDE.md after each milestone.
