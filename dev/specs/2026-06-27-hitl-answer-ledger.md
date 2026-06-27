# HITL Answer Ledger Design Spec

- Date: 2026-06-27
- Status: Draft for review
- Related issue: ROOT-A, parallel two-phase tool preparation plus single-slot HITL deadlock
- Companion plan: `dev/plans/2026-06-27-hitl-answer-ledger.md`

## Problem

cubepi's parallel tool executor intentionally uses two phases:

1. Prepare every requested tool call. Preparation includes middleware gates such
   as approval checks.
2. Only after every tool call is prepared, execute the prepared calls in
   parallel and append the resulting tool results.

That split prevents a bad replay failure mode. If tool A executed before tool B
finished preparation, and tool B then detached for human input, tool A could have
performed side effects without a durable `ToolResult`. Resuming the same turn
would then risk running tool A again.

The split is correct for side-effect safety, but it conflicts with the current
HITL resume channel. The channel has one visible pending request and one
single-use resume answer slot. With two parallel approval-gated tool calls,
`A` and `B`, the system can deadlock:

1. User approves `A`.
2. Resume prepares `A`, consumes the answer, then prepares `B`.
3. `B` has no answer, so preparation detaches before any tool body executes.
4. User approves `B`.
5. Resume starts over from the same assistant message. `A` now has no answer,
   so preparation detaches before `B` can be prepared.

The pending request flips between `A` and `B`. No tool results or idempotency
sentinels are written because execution never starts.

Marking approval-gated tools as sequential avoids the deadlock, but it leaks a
framework invariant into application policy. It also disables useful parallelism
for batches that are safe to run together after their gates have all been
answered.

## Goals

- Preserve the parallel executor's side-effect guarantee: no tool body in a
  parallel batch starts until every tool call in that batch has prepared
  successfully.
- Preserve one visible pending HITL request at a time, so existing hosts do not
  need to build a multi-prompt UI.
- Make answered HITL prompts durable and replayable across resume attempts.
- Keep application code out of the routing decision. Hosts should not need to
  mark approval-gated tools as sequential to avoid this deadlock.
- Support the existing approval middleware and `ask_user` flow without changing
  public tool definitions.
- Keep the design small and async-native, consistent with cubepi's current
  checkpointer and channel abstractions.

## Non-goals

- Showing multiple simultaneous pending prompts in the public HITL API.
- Executing an approved prefix of a parallel batch before the rest of the batch
  has been approved.
- Replacing cubepi's message-loop execution model with a graph runtime.
- Making arbitrary custom tool bodies crash-safe when they call HITL after
  performing side effects. `CheckpointedChannel` should continue to reject that
  case by default.
- Adding a new hard runtime dependency.

## Proposed Design

Add a durable HITL answer ledger. The ledger stores answered HITL requests by
thread, run, and question id. Preparing a gated call first checks the ledger. If
an answer already exists for that question id, preparation replays that answer
without publishing a new pending request and without deleting the answer.

This changes the deadlocked sequence into a progressive collection of answers:

1. User approves `A`.
2. Resume prepares `A` from the ledger, then detaches on `B`. No tool body runs.
3. User approves `B`.
4. Resume prepares `A` from the ledger and prepares `B` from the ledger.
5. All preparation succeeds, so the executor runs `A` and `B` in parallel.
6. Tool results are appended, pending state is cleared, and ledger entries for
   the completed run are deleted.

The answer ledger preserves two-phase execution. It does not execute `A` early;
it only remembers that `A`'s gate has already been satisfied.

### Ledger Key

The logical key is:

```text
(thread_id, run_id, question_id)
```

`run_id` is nullable for legacy in-memory or non-run hosts. First-party durable
checkpointers should normalize `None` to an empty string or equivalent storage
key so the database primary key remains stable.

`question_id` is the existing HITL question id. For approval middleware this is
the tool call id. For `ask_user`, it is the id already produced by the channel.

### Ledger Record

The record only needs enough data to replay the answer:

```python
@dataclass
class HitlAnswerRecord:
    question_id: str
    answer: StructuredValue
    run_id: str | None = None
    answered_at: float = field(default_factory=time.time)
```

Implementations may store a request snapshot for audit/debugging, but replay
must not depend on it. The pending row remains the source of truth for the
currently visible request.

### Checkpointer API

Extend the checkpointer protocol with answer ledger methods:

```python
async def save_hitl_answer(
    self,
    thread_id: str,
    question_id: str,
    answer: StructuredValue,
    *,
    run_id: str | None = None,
) -> None: ...

async def load_hitl_answer(
    self,
    thread_id: str,
    question_id: str,
    *,
    run_id: str | None = None,
) -> StructuredValue | None: ...

async def clear_hitl_answers(
    self,
    thread_id: str,
    question_ids: Iterable[str] | None = None,
    *,
    run_id: str | None = None,
) -> None: ...
```

First-party checkpointers must implement these methods:

- `MemoryCheckpointer`: an in-memory dict keyed by `(thread_id, run_key,
  question_id)`.
- `SQLiteCheckpointer`: a new table with `thread_id`, `run_id`,
  `question_id`, `answer_json`, and `answered_at`.
- `PostgresCheckpointer`: equivalent table with JSONB answer storage.
- `MySQLCheckpointer`: equivalent table with JSON answer storage.

`CheckpointedChannel` should validate these methods when it is used for
detached HITL. A missing method should fail early with an actionable `HitlError`,
the same way missing pending APIs are handled today.

### Channel Semantics

`_BaseChannel._await_answer()` should resolve answers in this order:

1. If a durable ledger lookup is available and contains `question_id`, return
   that answer immediately.
2. Otherwise, if the process-local resume slot matches `question_id`, return
   that answer. This keeps same-process behavior and legacy tests simple.
3. Otherwise, publish the request as the single pending request and detach or
   wait according to the current channel mode.

Returning a ledger answer must not delete it. Deletion before the whole batch is
durably resolved would recreate the same bug after a later detachment.

When an answer is supplied through `channel.answer()` in-process, the channel
should also save it to the ledger before waking the waiter. That keeps behavior
consistent between direct in-process HITL and `Agent.respond()`.

### Agent.respond Semantics

`Agent.respond()` should:

1. Load the pending request and recovered run id atomically through
   `load_pending()`.
2. Validate that the supplied `question_id` matches the visible pending request.
3. Validate the agent/channel run binding against the recovered run id.
4. Save the answer into the ledger before starting resume.
5. Optionally attach the process-local resume answer for fast in-process replay.
6. Resume the original tool cycle.

If resume detaches on another request, the just-saved answer remains in the
ledger. A later resume can replay it.

### Cleanup Semantics

Ledger answers are cleared only when they can no longer be needed for replay:

- After the suspended tool cycle completes successfully and tool results have
  been appended.
- When the active HITL request is aborted or cancelled by the host.
- When the agent clears pending state because the run has reached a terminal
  outcome.

Do not clear a ledger entry immediately after a single preparation step consumes
it. For parallel batches, preparation may still detach on a later tool call.

For simplicity and safety, successful completion can clear all ledger answers
for `(thread_id, run_id)` rather than trying to clear only the exact tool call
ids. This also covers `ask_user`, where the question id may not equal a tool
call id.

### User-visible Behavior

After approving one tool in a multi-gated parallel batch, the tool may still not
have executed. The approval means "this gate is answered"; execution begins only
when all gates in the current parallel preparation batch are answered.

Hosts can present this as:

```text
Approved. Waiting for remaining approvals before execution.
```

No API change is required for hosts that already call `Agent.respond()` against
the currently pending request.

## Compatibility

- Existing pending request APIs remain unchanged.
- Existing `execution_mode="sequential"` remains useful for genuinely ordered
  or non-concurrency-safe tools, but it is not required for approval-gated tools.
- Third-party checkpointers that use `CheckpointedChannel` for detached HITL
  need to implement the ledger methods. The failure mode should be explicit at
  channel construction or first detached response, not a silent fallback to the
  deadlocking behavior.
- In-process `InMemoryChannel` remains usable without a durable checkpointer,
  but should still keep an in-memory ledger so the same replay rules apply.

## Testing Requirements

Add tests covering:

- `MemoryCheckpointer`, `SQLiteCheckpointer`, `PostgresCheckpointer`, and
  `MySQLCheckpointer` answer ledger CRUD where those backends already have HITL
  tests.
- `CheckpointedChannel` replays a saved answer for the same `question_id`
  without publishing a new pending request.
- A parallel batch with two approval-gated tools no longer deadlocks:
  approving `A` records the answer and leaves pending `B`; approving `B`
  executes both tools exactly once and appends two tool results.
- Restart behavior: use a fresh `Agent` instance between approvals and verify
  replay still succeeds.
- Rejection/deny and edited approval decisions replay correctly.
- Existing regression coverage that a later preparation detach prevents earlier
  tool bodies from starting still passes.
- Ledger entries are cleared after successful completion and after abort.

## Documentation Requirements

Update the HITL user-facing docs under `website/docs/` to explain:

- Parallel approval-gated tool calls are safe.
- Approving one item in a parallel batch records that decision, but execution
  waits until all gates in the batch are satisfied.
- Application authors do not need to mark approval-gated tools as sequential
  solely for HITL safety.

## Risks

- Stale answers are dangerous if keyed too broadly. Use `run_id` when available
  and keep question ids unique within a turn.
- Clearing too early recreates the deadlock. Cleanup must happen after durable
  tool results are appended or after terminal abort/cancel.
- Third-party checkpointers need a clear migration path. The error should name
  the new ledger methods and explain that detached HITL requires them.
