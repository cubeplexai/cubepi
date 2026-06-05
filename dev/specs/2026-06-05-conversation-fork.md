# Conversation Fork — `Agent.fork()` + `Agent.fork_once()` (run-based)

- **Date**: 2026-06-05
- **Status**: Draft (v2 — run-based fork model)
- **Branch / worktree**: `2026-06-05-conversation-fork` in `.worktrees/2026-06-05-conversation-fork`
- **Drives**: cubebox "copy this conversation from this assistant reply" UI button;
  one-shot off-thread probes; data model foundation for future
  `Agent.delete_run()`.

## 1. Motivation

Three related needs are unaddressed today:

1. **Persistent fork.** Cubebox wants a per-assistant-message button that
   spawns a new conversation, pre-populated with the prior runs up to a
   chosen point. The user keeps both conversations and continues them
   independently.
2. **One-shot off-thread prompt.** Application code wants to ask the model a
   follow-up from the context of an existing thread without polluting the
   thread's persisted history. Reflection-runner probes, automated eval
   harnesses, scratch "what if I asked X instead?" queries.
3. **Future: per-run deletion.** Cubebox plans a "delete this run" UX
   (undo the last exchange). Not in this PR, but the data model laid down
   here MUST make it a one-line query later.

A "run" in cubepi is **one `Agent.prompt()` call from start to terminal
completion** — the user message that initiated it plus every tool_use /
tool_result / assistant message produced before the loop exits with a
terminal stop_reason. Multi-step assistant loops live inside a single run.
HITL pause/resume keep the same run_id across the suspension.

The Postgres and MySQL checkpointer schemas reserved `parent_thread_id`
and `forked_at_seq` columns for fork (see
`website/docs/migration/from-langgraph.md` and
`website/docs/guides/checkpointing/postgres.md`), but no API existed.
The Memory and SQLite backends have no fork hooks at all. **Run-as-a-unit
is a new concept**: today only `cubepi_threads.pending_request.run_id`
mentions run, and only for HITL recovery. This spec extends run identity
to every message and adds a per-run completion marker.

## 2. Goals / Non-goals

**Goals**

- Per-message `run_id` field on `Message` (all three variants), persisted
  by every backend.
- New `cubepi_runs` storage per backend recording per-run lifecycle:
  `(thread_id, run_id, claimed_at, completed_at, completion_seq)`,
  PK `(thread_id, run_id)`. Atomic claim on `Agent.prompt()` entry
  prevents same-run_id races (see §3.6.1).
- `Agent.prompt(message, *, run_id=None) -> str` (accept-or-generate;
  returns the run_id actually used).
- `Checkpointer.fork(src, new, *, after_run_id, metadata=None) -> None`
  — physical copy of all messages of completed runs up through
  `after_run_id`.
- `Checkpointer.snapshot(thread_id, *, after_run_id) -> list[Message]`
  — shared read primitive used by fork_once and tests.
- `Agent.fork(src, new, *, after_run_id, metadata=None) -> None` — thin
  wrapper over `Checkpointer.fork()`.
- `Agent.fork_once(src, message, *, after_run_id) -> ForkOnceResult` —
  in-memory single-turn continuation from a snapshot prefix.
- Implementations across Memory, SQLite, Postgres, MySQL.
- User-facing docs page under `website/docs/guides/checkpointing/`.
- Foundation that makes `Agent.delete_run(thread_id, run_id, …)` a small
  follow-up PR.

**Non-goals**

- `Agent.delete_run()` itself — separate follow-up spec.
- Fork at message granularity (the cubebox UX and forward design only
  ever fork at run boundaries; per-message cuts were considered and
  rejected with the user — they produce surprising states mid-tool-call
  and are not needed). The previously-discussed `message_count` /
  `Message.id` / `after_response_id` parameters are NOT in this spec.
- Copy-on-write storage (rejected for the same partitioning / mutation
  reasons documented in §3.1).
- Backfilling run identity into pre-existing messages. Legacy
  messages remain `run_id=NULL`; they are not deletable by run_id but
  ARE included in forks as the "legacy prefix" of a mixed thread —
  see §3.6.4.
- A `fork_into_agent()` convenience or `fork_and_switch`. Caller owns
  what to do with `new_thread_id`.
- Resuming a `fork_once()` session.
- A `cubepi fork` CLI subcommand.

## 3. Design

### 3.1 Storage semantics: physical copy

`fork()` physically copies messages of all completed runs up through
`after_run_id` to the new thread, then records lineage
(`parent_thread_id`, `forked_at_seq` for SQL backends) and copies the
relevant `cubepi_runs` rows so the new thread keeps its
run history.

Rejected: **logical pointer / COW**. Same rationale as v1:

- All four backends partition / shard reads per `thread_id`
  (HASH/KEY partitioning in Postgres/MySQL; per-thread dict slot in
  Memory). COW reads span parent and child partitions.
- The source thread is mutable (`agent.respond()` injects synthetic
  messages; future compaction may rewrite). Under COW the child
  silently drifts; physical copy freezes the child at fork time.
- Deletion semantics are clean under physical copy: parent and child
  independently deletable. Under COW, parent deletion breaks the child.
- The reserved `parent_thread_id` + `forked_at_seq` columns retain
  value as lineage metadata at no recurring read cost.
- LangGraph's `copy_thread` makes the same choice.

### 3.2 Run as the unit of fork

The **only** fork handle is `after_run_id: str`. The new thread contains:

1. Every message whose `run_id IS NULL` (legacy prefix — see §3.6.4),
   PLUS
2. Every message whose `run_id` is in `cubepi_runs` for
   `src_thread_id` with **`completion_seq IS NOT NULL` AND
   `completion_seq <= after_run_id.completion_seq`**.

Messages are inserted into the new thread in source seq order.

#### Why `completion_seq`, not `completed_at`

Wall-clock timestamps are not safe ordering keys:

- SQLite's `julianday('now')` can produce identical values for runs
  completed in the same tick.
- MySQL `TIMESTAMP` precision varies by version and configuration.
- Lex-tiebreaking by `run_id` is deterministic but **not the same as
  correct completion ordering** — a later-completing run with a
  lexicographically-earlier `run_id` would sort before an earlier
  completion.

`completion_seq` is a per-thread monotonic BIGINT allocated under the
same per-thread lock that serializes other writes. The allocation is
`(SELECT COALESCE(MAX(completion_seq), 0) + 1 FROM cubepi_runs WHERE
thread_id = ? AND completion_seq IS NOT NULL)`. Under the lock this
yields a strictly-monotonic per-thread sequence whose ordering matches
real completion ordering. `completed_at` is retained as audit
metadata only — never used for ordering or filtering.

#### Why "set of completed runs", not "seq <= last_seq_of(after_run_id)"

A naive seq-cut is unsafe under concurrent runs. Example: runs A and B
both active on the same thread; A writes seqs 1, 2, 5 and completes,
while B writes seqs 3, 4 and is still in flight. A seq-cut
"`seq <= 5`" would pull B's seqs 3, 4 into the fork — messages of an
unfinished run we have no right to copy. The set-based selection
solves this by construction: only messages tagged with a *completed*
run_id (or NULL for legacy) are eligible.

This selection is **row-correct** in the presence of:

- **In-flight runs**: their messages have a `run_id` whose row has
  `completion_seq IS NULL` → excluded.
- **Mixed legacy + post-upgrade threads**: NULL-run_id messages are
  preserved as a chronological prefix, post-upgrade completed runs
  appear after.

#### What "row-correct" does NOT cover: context consistency under interleaved runs

If two runs A and B run concurrently on the same thread, the agent
loop's lazy load (`load(thread_id)`) at the start of B will pick up
A's in-flight messages as context. If B then completes and A doesn't,
`fork(after_run_id=B)` copies B's messages but EXCLUDES A's — yet B's
outputs were conditioned on A's context. The fork is row-correct (no
orphan tool_use, no half-truncated run) but **semantically dangling**:
B's assistant reply may refer to information that no longer exists in
the forked thread's history.

The spec **does not solve this** at the cubepi layer. It is left as a
host responsibility:

- **Recommended pattern**: one active run per thread (cubebox's actual
  pattern — RunManager serializes per conversation; `Agent.prompt()`
  inside a single Agent instance is also single-flight via the
  existing `_run_lock`).
- **If hosts want multi-process / multi-Agent prompts on the same
  thread**: they must accept that fork can produce semantically
  dangling artifacts when runs overlap. This is documented as a
  known limitation; a future spec may add a per-thread run lease
  if a real workload needs the stronger guarantee.

#### Why a run is the right unit

- A run is the atomic transaction the user thinks in ("I asked X,
  agent did its thing, gave me an answer"). Cubebox's UI button maps
  directly to "fork after this exchange".
- A run boundary is by construction a clean cut. Inside a run there
  may be unresolved `tool_use` blocks awaiting `tool_result`s; once
  the run is marked complete (terminal stop_reason, no pending HITL),
  no unresolved tool calls remain. **The v1 spec's `ForkBoundaryError`
  / mid-tool-call invariants vanish structurally**: there is no API
  to ask for a cut mid-run, so no boundary check is needed.
- It generalizes to the future `delete_run()` cleanly:
  `DELETE FROM cubepi_messages WHERE thread_id=? AND run_id=?` is a
  one-line operation backed by the same `run_id` column the fork uses
  for lookup.

#### Single-flight per Agent instance (existing, enforced)

`Agent.prompt()` is single-flight on its own instance — the existing
`_run_lock` (see `cubepi/agent/agent.py:204`) guards `prompt()` and
`respond()` so two coroutines on the SAME Agent object serialize.
`Agent.state.active_run_id` is therefore safe from clobbering under
single-instance concurrency.

The unguarded case (and the context-consistency caveat above) is
**multiple Agent instances or multiple processes** on the same thread.
Spec does not protect that case.

### 3.3 Atomicity and concurrency

`fork()` is atomic per backend:

- **Memory**: `asyncio.Lock`; copy messages, completions row, and
  lineage under the lock. Memory backend is single-process only by
  definition (existing limitation, documented for fork too).
- **SQLite**: `BEGIN IMMEDIATE` (RESERVED lock) wraps the entire fork
  (validation, completion lookup, message copy, completions copy,
  thread_extra insert with lineage). The existing `append()` /
  `save_extra()` / `save_pending_request()` are also promoted to
  `BEGIN IMMEDIATE` so writer-vs-writer races are uniformly
  serialized, including across processes sharing the DB file.
  At connect time the checkpointer sets `PRAGMA busy_timeout = 5000`.
  If the 5 s window elapses without acquiring the lock, the
  `aiosqlite.OperationalError` propagates as
  `CheckpointerLockTimeoutError`.
- **Postgres**: single transaction. Takes
  `pg_advisory_xact_lock(hashtext($src_thread_id))` — the same
  per-thread advisory lock `append()` / `save_extra()` /
  `save_pending_request()` use — to fence racing appends on the source
  for the duration of the fork. Then `INSERT INTO cubepi_threads`,
  `INSERT INTO cubepi_messages … SELECT …`,
  `INSERT INTO cubepi_runs … SELECT …`. Commit releases
  the lock.
- **MySQL**: single transaction with `SELECT … FOR UPDATE` on the
  source `cubepi_threads` row. The existing `append()` already takes
  the same row lock; this spec confirms `save_extra` /
  `save_pending_request` follow suit. Then the same three INSERTs.

Concurrent forks from the same source serialize on the per-thread
lock/row-lock; identical or different message-set outcomes both
correct depending on append interleaving.

Error pre-checks happen inside the transaction so they see the same
world the copy will see:

- new thread already exists → `ThreadAlreadyExistsError`, nothing
  written
- source thread does not exist → `ThreadNotFoundError`
- `after_run_id` has no completion marker on the source thread →
  `RunNotCompletedError`

### 3.4 What gets copied

| Field / row | Copied? | Notes |
|---|---|---|
| `cubepi_messages` rows where `run_id IS NULL` OR `run_id IN {completed runs of src with completion_seq IS NOT NULL AND completion_seq <= after_run_id.completion_seq}` | yes | physical copy; PG/MySQL preserve source `seq` values for the copied range. SQLite copies the JSON payloads under fresh global `id`s (its `messages.id` is a global auto-increment, not a per-thread seq). Memory copies in-list order. Each copied row keeps its original `run_id` value (or NULL). Source seq order is preserved across the copy. |
| `cubepi_runs` rows for the copied runs | yes | so the new thread can be further forked / deleted by run |
| `extra` | yes | deep copy of the source JSON object |
| `parent_thread_id` | written (new) | set to `src_thread_id` on the new thread row |
| `forked_at_seq` | written (PG/MySQL only) | the `seq` of the last message in the copied set (highest copied seq). Memory and SQLite store no equivalent — those backends have no per-message seq column and lineage is recoverable from `parent_thread_id` + `cubepi_runs` alone. |
| `extra['fork']` | written (new) when `metadata` arg supplied | overwrites any pre-existing `extra['fork']` on source (lineage is recoverable via the `parent_thread_id` chain) |
| `pending_request` | **no** | new thread starts clean; HITL is run-state, not history |
| `cubepi_threads.run_id` (the host-side HITL marker, NOT a run on this thread) | **no** | new thread has no run in flight |
| `created_at` / `updated_at` | new | server-default to fork time |

### 3.5 Tracing for `fork_once()`

Unchanged from v1:

- Span name: `cubepi.agent.fork_once`
- Inherits the active OTel parent context if one is bound; otherwise
  becomes a true trace root. Does NOT attempt to attach to or replay
  spans from the source thread's prior runs.
- Attributes: `cubepi.fork.src_thread_id`, `cubepi.fork.after_run_id`,
  `cubepi.fork.src_seq` (PG/MySQL only — the seq of the last copied
  message), plus the standard cubepi tracing attributes.
- The in-process child Agent's spans nest under this span.

The persistent `fork()` does not need a special span; existing
checkpointer instrumentation covers it.

### 3.6 Run lifecycle in cubepi

#### 3.6.1 `Agent.prompt()` signature change

```python
class Agent(Generic[TMessage]):
    async def prompt(
        self,
        message: str | Message | list[Message],
        *,
        run_id: str | None = None,
    ) -> str: ...
```

`run_id` is accept-or-generate:

- If the caller supplies a string, cubepi uses it verbatim.
- If `None`, cubepi generates one via `uuid.uuid4().hex` (no host
  dependency on uuid7; cubebox can keep generating its own and pass it
  in — single source of truth).
- The return value is the run_id actually in effect, so the caller can
  store it / cross-check.

The active `run_id` is threaded through the agent loop to
`Checkpointer.append()` by stamping it on every `Message` instance
about to be appended. **`Checkpointer.append()` signature does not
change** — the run_id rides on the messages themselves
(`Message.run_id` field, §3.6.5).

`Agent.prompt()` calls
`self.checkpointer.claim_run(self.thread_id, run_id)` before any
append, gated by:

- **Skip when the checkpointer or thread_id is absent**: when
  `self.checkpointer is None` OR `self.thread_id is None` (e.g., the
  transient agent inside `fork_once()` — see §3.8), `prompt()` skips
  the claim entirely; it still stamps `run_id` on every in-memory
  Message but no `cubepi_runs` row is written.
  `mark_run_complete()` is similarly skipped.

- **Skip when the checkpointer is run-unaware (legacy degraded
  mode)**: when `self.checkpointer` exists but does NOT implement
  BOTH `claim_run` AND `mark_run_complete` (a third-party v3-only
  Protocol impl), `prompt()` skips both. Messages still get
  `run_id` stamped; no marker is ever written; `Agent.fork()` and
  `Agent.fork_once()` on this checkpointer raise
  `CheckpointerError`. The detection is one capability check at
  `prompt()` entry:
  ```python
  run_aware = (
      hasattr(self.checkpointer, "claim_run")
      and hasattr(self.checkpointer, "mark_run_complete")
  )
  ```
  Partial implementations (one but not the other) are treated as
  NOT run-aware — never call into a checkpointer that can claim
  but not complete, or the inverse. See §4 for the full
  compatibility story.

The claim INSERT atomically inserts a row into `cubepi_runs` with
`claimed_at = now()` and `completed_at = NULL`. The insert is the
synchronization point:

- PK conflict where the existing row has `completed_at IS NULL` →
  `RunAlreadyClaimedError` (another process is currently running
  this run_id; concurrent claim is rejected, the loser must retry
  with a different run_id)
- PK conflict where the existing row has `completed_at IS NOT NULL` →
  `RunAlreadyCompletedError` (this run_id has already finished;
  runs are append-only — use a new `run_id`)

This **eliminates** the v2 R1 "messages were appended before the
conflict was detected" half-state: no message is appended unless the
claim succeeded, and a successful claim guarantees no other process
can append messages tagged with the same run_id (they will hit
`RunAlreadyClaimedError` at their own claim step).

`Agent.state.active_run_id` is set to the claimed `run_id` for the
duration of the run, so observers (callbacks, listeners, debugging)
can see it. The same value is the return value of a successful
`prompt()` call.

If `Agent.prompt()` raises any error AFTER a successful claim, the
claim row is left with `completed_at IS NULL` (abandoned). The
abandoned row prevents accidental reuse of the same run_id (still
gets `RunAlreadyClaimedError` on retry). The future `delete_run()`
admin operation can clean up abandoned claims.

If `prompt()` raises BEFORE the claim (validation failure, no
checkpointer, etc.), no row is inserted.

#### 3.6.2 Completion — when written

Completion is recorded by `Checkpointer.mark_run_complete(thread_id,
run_id)`, which under the per-thread lock:

1. `SELECT completed_at FROM cubepi_runs WHERE thread_id=? AND
   run_id=?`.
2. **No row** → `RunNotClaimedError`. (Agent-loop logic bug: claim
   was never made, or its row was lost.)
3. **Row exists with `completed_at IS NOT NULL`** → idempotent
   success. Return without raising and without re-allocating
   `completion_seq`. Supports retry-after-lost-ack from
   `CompletionMarkerFailedError`.
4. **Row exists with `completed_at IS NULL`** →
   - Allocate `next_seq = (SELECT COALESCE(MAX(completion_seq), 0)
     + 1 FROM cubepi_runs WHERE thread_id=? AND completion_seq IS
     NOT NULL)` (per-thread monotonic, §3.2).
   - `UPDATE cubepi_runs SET completed_at = now(),
     completion_seq = $next_seq WHERE thread_id=? AND run_id=?`.

`mark_run_complete()` **never raises `RunAlreadyCompletedError`** —
that error is reserved for `claim_run()` collisions only (see §3.6.1).

`mark_run_complete()` is called by the agent loop AFTER the run's
final `append()` and BEFORE `Agent.prompt()` returns.

**The completion write is NOT atomic with the final append.** The
existing cubepi event loop (`cubepi/agent/loop.py`) persists each
message on its own `MessageEndEvent`; the loop only knows it has
finished when `AgentEndEvent` fires, *after* the final append.
Coupling them atomically would require an Agent-layer buffering
rewrite that is out of scope.

Two failure modes and their consequences:

- **Process crash between final append and `mark_run_complete()`**:
  messages of run R are persisted; the `cubepi_runs` row remains
  with `completed_at IS NULL` (abandoned claim). `fork(after_run_id=R)`
  raises `RunNotCompletedError`. The data is not corrupt; the run is
  in the same "abandoned" state as if the loop had been killed for
  any other reason. The future `delete_run()` admin operation can
  remove abandoned claims (and their messages) by querying
  `WHERE completed_at IS NULL`. The same operation can also
  manually-finalize via direct DB UPDATE if the host knows the run
  actually completed.
- **Transient checkpointer failure on `mark_run_complete()`**:
  `Agent.prompt()` raises `CompletionMarkerFailedError(run_id=R,
  cause=<underlying>)`. The exception **carries the run_id** so
  callers using `prompt(run_id=None)` (cubepi-generated id) still
  know the value and can:
  - retry `mark_run_complete(thread_id, R)` directly (idempotent —
    the row exists, only completed_at update is in flight); OR
  - abandon R; the messages remain under R but the run is not
    forkable. `delete_run(R)` cleans up later.

The recommended retry pattern: "retry `mark_run_complete()`, not the
whole `prompt()`" — re-running `prompt(run_id=R)` raises
`RunAlreadyClaimedError` because R's claim row already exists.

#### Trigger conditions for `mark_run_complete()` — enumerated by loop outcome

The current `cubepi/agent/loop.py` is **not** a "return-means-success"
machine. It catches provider/tool exceptions, appends a synthetic
assistant message with `stop_reason="error"` or `"aborted"`, and
returns normally (see `loop.py:485` early-exit branch). A naive
"call mark_run_complete after `_run_prompt()` returns"
implementation would mark FAILED runs complete. To prevent that, the
spec enumerates each loop outcome AND adds a structural pre-completion
invariant (below).

#### Pre-completion invariant: well-formed tool-use turns

**Before calling `mark_run_complete()`, the agent loop MUST verify
that every `AssistantMessage` with `ToolCall` blocks in this run is
immediately followed by a contiguous block of `ToolResultMessage`s
whose `tool_call_id`s exactly cover (set-equal to) that
assistant's `ToolCall.id`s, before any other `AssistantMessage` or
`UserMessage` appears in the run's message list.**

Concretely, for each `AssistantMessage` A in the run carrying
ToolCalls with ids `{c1, c2, …}`:

- The next K messages in the run (where K = number of A's ToolCalls)
  MUST all be `ToolResultMessage`s.
- Their `tool_call_id` set MUST equal `{c1, c2, …}` exactly (no
  missing, no extras, no duplicates).
- No `UserMessage` or `AssistantMessage` may appear within that
  K-message window.

A weaker "any matching tool_call_id later in the run" check is NOT
sufficient: cubepi's existing code documents that `tool_call_id`s
are not globally unique, so a later same-id result could falsely
satisfy an earlier orphan. Strict per-turn adjacency closes that
hole.

This catches two real loop-level escape hatches:

1. A custom `after_model_response` middleware returns
   `TurnAction(decision="stop")` after an assistant message
   containing `ToolCall` blocks — loop short-circuits without
   executing tools, prompt() returns with `stop_reason="tool_use"`
   and no `ToolResultMessage` ever appended for this turn.
2. A custom middleware injects intervening messages (a user message,
   a separate assistant message) between the tool-use assistant and
   what would be its tool_results — leaves the providers' strict
   adjacency expectation broken even when a matching tool_call_id
   eventually appears later.

The check is O(N) over the run's own messages (those tagged with
this `run_id` appended since `claim_run`). On violation the loop
treats the outcome as **"incomplete tool cycle"** — a row in the
outcome table below — and does NOT call `mark_run_complete`. The
cubepi_runs row remains with `completed_at IS NULL`; fork sees
`RunNotCompletedError`. Future `delete_run(R)` can clean up.

The check is the only remaining vestige of v1's boundary validation,
relocated from arbitrary fork-cut time to run-completion time (where
a single pass suffices). Because runs are by construction the only
fork unit, validating each run once at completion is enough — fork
never needs to re-validate.

#### Outcome table

| Loop outcome | Detection signal | `mark_run_complete()` called? |
|---|---|---|
| **Clean success** | Final `AssistantMessage` has `stop_reason in {"end_turn", "stop", "tool_use", …}` (any non-error / non-aborted stop_reason) AND `error_message is None` AND no pending HITL on the thread AND **the pre-completion invariant above passes** (no orphan `ToolCall`s in this run) | **YES** |
| **Incomplete tool cycle** | Pre-completion invariant (above) fails: some `AssistantMessage` in this run is not immediately followed by a contiguous block of `ToolResultMessage`s covering exactly its `ToolCall.id`s. Typical triggers: `after_model_response(decision="stop")` on a tool-use response; middleware injecting intervening user/assistant messages between the tool-use turn and its results | NO — same handling as a failed run; claim row remains; `delete_run()` can clean up |
| **HITL suspended** (normal pause) | `pending_request` row written this run; `prompt()` returns with the agent in suspended state | NO — `respond()` writes the marker on resume's clean exit |
| **HITL detached** (`HitlDetached` caught in `loop.py:196` / `loop.py:418`) | Exception caught internally; loop returns silently, NO `AgentEndEvent` emitted. `pending_request` may still be set so `respond()` can resume later. | NO — same as normal HITL pause; resume handles completion |
| **HITL aborted** (`HitlAborted`, via `Agent.abort_pending()`) | Exception caught internally; loop returns silently, NO `AgentEndEvent` emitted. The synthetic deny + terminal aborted assistant are appended by `abort_pending()` itself (see `agent.py:582-595`). | NO — claim row remains; the run is terminally abandoned; future `delete_run()` cleans up |
| **Provider / tool error** | Final assistant message has `stop_reason="error"` OR `error_message is not None` (set on the message by the loop's exception handler) | NO — claim row remains; future `delete_run()` cleans up |
| **Abort during streaming** (`agent.abort()`) | Final assistant message has `stop_reason="aborted"` | NO — claim row remains |
| **Cancellation** (`asyncio.CancelledError` propagates out — i.e. the caller's task is cancelled and no internal catch swallowed it) | Exception propagates out of `prompt()`; no `AgentEndEvent` emitted; `active_run_id` left set (§3.7) | NO — claim row remains |

The agent loop's existing `if message.stop_reason in ("error",
"aborted"): … return early` branch (`loop.py:485`) is the natural
gating point. After it, on the success path, the loop calls
`mark_run_complete()` once before final cleanup.

Tests MUST cover each row of this table independently (provider
exception, tool exception, abort, abort_pending, cancel,
HitlDetached) and assert the cubepi_runs row's `completed_at IS
NULL` for every non-success outcome.

#### 3.6.3 HITL pause / resume continuity

A run that pauses for HITL keeps its `run_id` across the suspension.

- `Agent.prompt(message, run_id=R)` runs partway, hits HITL → writes
  `pending_request` with `run_id=R` (existing schema v3 mechanism), no
  completion marker.
- `Agent.respond(question_id, answer)` recovers `run_id=R` from the
  same `pending_request` row, restores it as the active run_id in the
  agent loop, continues the same run, stamps every subsequent
  appended message with `run_id=R`, and calls `mark_run_complete()`
  on terminal exit.

**`respond()` does NOT call `claim_run()`.** The run was already
claimed by the original `prompt()` call; the `cubepi_runs` row
exists with `completed_at IS NULL`. Calling `claim_run(R)` again
would (correctly) raise `RunAlreadyClaimedError`. The agent loop
distinguishes "fresh prompt" (claim required) from "resume"
(claim skipped) by which entry point was called. A single end-to-end
HITL-bearing run calls `claim_run()` exactly once across all
`prompt()` + N × `respond()` invocations.

`Agent.respond()` signature does **not** change.

#### 3.6.3.1 HITL channel run_id binding constraint

`cubepi.hitl.channel.CheckpointedChannel.__init__` takes `run_id` as a
constructor argument (`self._run_id`) and never updates it per call.
The value is written to `pending_request.run_id` whenever the channel
pauses for HITL. This pre-dates the spec and is preserved.

The spec **adds a constraint** to make the channel's bound run_id
agree with the agent's active run_id, so HITL recovery on resume
finds the right value. The constraint must be checkable AND must
distinguish in-memory HITL (no persistence, no constraint) from
checkpointed HITL (writes pending_request.run_id, must match).

Structural change — single nested attribute replaces the prior
`requires_hitl` + `bound_hitl_run_id` pair:

```python
@dataclass(frozen=True)
class HitlBinding:
    """How a tool/middleware integrates with HITL."""
    checkpointed: bool              # True iff backed by CheckpointedChannel
    run_id: str | None              # the channel's bound run_id (str if
                                    # checkpointed; None for in-memory)

class AgentTool:                    # cubepi.agent.types
    ...
    hitl: HitlBinding | None = None # None = tool has no HITL involvement

class Middleware:                   # cubepi.middleware.base
    ...
    hitl: HitlBinding | None = None
```

`hitl is not None` replaces the v1 `requires_hitl=True` flag for
detection purposes (§3.8.2 fork_once ban).

- `cubepi.hitl.ask_user_tool(channel)` factory MUST set
  `tool.hitl = HitlBinding(checkpointed=isinstance(channel,
  CheckpointedChannel), run_id=getattr(channel, "_run_id", None))`.
- `cubepi.hitl.middleware.ApprovalPolicyMiddleware.__init__` does the
  same with its channel argument; `ConfirmToolCallMiddleware` inherits.
- Third-party HITL tools / middleware MUST set `hitl=HitlBinding(...)`
  if they bind a channel; documented in the user guide.

**Enforcement at `Agent.prompt()` start, before `claim_run()`:**

```python
bound: set[str] = set()
for elem in (*self.tools, *self.middleware):
    if elem.hitl is None:
        continue
    if not elem.hitl.checkpointed:
        continue  # in-memory HITL has no persistence; no run_id constraint
    # Checkpointed HITL: run_id MUST be a non-None string
    if elem.hitl.run_id is None:
        raise ValueError(
            f"Checkpointed HITL element {elem!r} has no run_id bound; "
            "construct CheckpointedChannel(run_id=...) before passing "
            "it to ask_user_tool/HITL middleware"
        )
    bound.add(elem.hitl.run_id)

if bound:
    if run_id is None:
        raise ValueError(
            "Agent has checkpointed HITL elements bound to "
            f"run_ids {bound!r}; prompt(run_id=...) must be "
            "explicitly supplied (generate-mode rejected)"
        )
    if any(b != run_id for b in bound):
        raise ValueError(
            f"prompt(run_id={run_id!r}) does not match "
            f"HITL-bound run_ids {bound!r}"
        )
```

- For non-HITL agents (no `hitl` attribute on any element), or for
  agents with only in-memory HITL (no checkpointed channels),
  `Agent.prompt` accepts `run_id=None` and generates; no constraint.

- For checkpointed HITL with `CheckpointedChannel(run_id=None)`
  (genuine config error — channel can't persist run_id because it
  has none), Agent raises immediately, before `claim_run`.

This avoids the alternative of refactoring `CheckpointedChannel` to
take run_id at call time (a backward-incompatible HITL API change
beyond this spec's scope) while still making the binding semantics
unambiguous AND mechanically checkable.

**Protocol change required to support this.** Today,
`Checkpointer.load_pending_request(thread_id) -> HitlRequest | None`
returns only the request. The run_id lives in a backend-specific
`load_pending_run_id(thread_id)` method that is NOT on the Protocol.
This spec adds it to the Protocol so `respond()` can recover the
run_id portably:

```python
class Checkpointer(Protocol):
    async def load_pending(
        self, thread_id: str
    ) -> tuple[HitlRequest, str | None] | None:
        """Load both the pending HITL request and the persisted run_id
        in a single call. Returns None when no pending request exists.
        The run_id may itself be None for legacy rows written before
        run_id was tracked.

        Backends MUST read both values from the same row in a single
        statement (no separate round-trips), so the pair is internally
        consistent.
        """
```

`load_pending_request()` is kept as an alias for the
request-only case (existing callers unchanged). New code uses
`load_pending()`.

Second-pause scenarios — a resumed run hits HITL *again* — work the
same way: `respond()` re-suspends with the SAME `run_id=R` stored
back to `pending_request`. A subsequent `respond()` call recovers R
and continues.

#### 3.6.4 Legacy data and mixed threads

Messages persisted before this spec carry `run_id = NULL`. No
completion markers exist for them. There are two cases:

**All-legacy thread** (no post-upgrade runs ever appended):

- `fork(src_thread_id, after_run_id=X)` raises `RunNotCompletedError`
  for every `X` because no markers exist for this thread at all.
- Future `delete_run(thread_id, run_id)` has nothing to target.
- `load()` works; the thread is readable. Only by-run operations
  are blocked.

**Mixed thread** (legacy NULL-run_id prefix + post-upgrade
completed runs):

- `fork(src_thread_id, after_run_id=R)` where R is a completed
  post-upgrade run on this thread:
  - Includes ALL legacy NULL-run_id messages (the chronological
    prefix the user has been chatting on top of)
  - PLUS messages of every completed run with
    `completion_seq <= R.completion_seq` (see §3.2 — `completion_seq`
    is the per-thread monotonic ordering key; `completed_at` is
    audit-only)
  - Source seq order preserved across the copy
- `delete_run(thread_id, R)` removes only R's messages (those with
  `run_id = R`); the legacy NULL prefix is untouched. To clear the
  legacy prefix the user must delete the whole thread.

This means a typical cubebox upgrade path is graceful: users
continue chatting on their existing conversations; their NEW exchanges
get run identity automatically; clicking "fork from this new
exchange" preserves the pre-upgrade history as expected.

No backfill is provided. (We cannot reliably reconstruct historical
run boundaries from messages alone — multi-step tool_use sequences
look like multiple runs.) Users who upgrade and then immediately try
to fork an all-legacy conversation see `RunNotCompletedError` because
that conversation has no run-marked exchange yet; the first new
prompt will create a forkable run.

#### 3.6.5 `Message.run_id` field

```python
class UserMessage(BaseModel):
    ...
    run_id: str | None = None     # NEW

class AssistantMessage(BaseModel):
    ...
    run_id: str | None = None     # NEW

class ToolResultMessage(BaseModel):
    ...
    run_id: str | None = None     # NEW
```

Default `None` preserves backward compatibility for callers that
construct `Message`s directly without going through `Agent.prompt()`.
The agent loop populates it from the active run before every append.

### 3.7 API surface

#### `cubepi.checkpointer.base`

```python
@dataclass
class CheckpointData:
    messages: list[Message] = field(default_factory=list)
    extra: JsonObject = field(default_factory=dict)
    parent_thread_id: str | None = None     # NEW (v1 had this too)


@runtime_checkable
class Checkpointer(Protocol):
    # existing: load, append, save_extra, save_pending_request, load_pending_request

    async def snapshot(
        self,
        thread_id: str,
        *,
        after_run_id: str,
    ) -> list[Message]:
        """Return all messages of completed runs up through `after_run_id`
        (inclusive), in source order.

        Raises `ThreadNotFoundError`, `RunNotCompletedError`.
        """

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        after_run_id: str,
        metadata: JsonObject | None = None,
    ) -> None:
        """Atomically create `new_thread_id` populated with the messages
        of every completed run on `src_thread_id` up through
        `after_run_id`. Copies `cubepi_runs` rows for the
        included runs. Records `parent_thread_id=src_thread_id` and
        (PG/MySQL only) `forked_at_seq`. Copies `extra` deeply. Writes
        `extra['fork'] = metadata` when `metadata` is supplied.

        Raises `ThreadNotFoundError`, `ThreadAlreadyExistsError`,
        `RunNotCompletedError`.
        """

    async def claim_run(
        self,
        thread_id: str,
        run_id: str,
    ) -> None:
        """Atomically insert a `cubepi_runs` row with `claimed_at =
        now()` and `completed_at = NULL`. PK conflict raises
        `RunAlreadyClaimedError` (existing row has completed_at IS
        NULL) or `RunAlreadyCompletedError` (existing row has
        completed_at IS NOT NULL). See §3.6.1.

        Called by `Agent.prompt()` before any `append()` of the run.
        """

    async def mark_run_complete(
        self,
        thread_id: str,
        run_id: str,
    ) -> None:
        """Allocate the next per-thread `completion_seq` under the
        per-thread lock and UPDATE the `cubepi_runs` row to set
        `completed_at = now()` and `completion_seq = <allocated>`.

        Called by the agent loop AFTER the run's final `append()` and
        BEFORE `Agent.prompt()` returns. NOT coupled to the final
        append — see §3.6.2.

        **Idempotent on already-completed rows.** If the row exists
        and already has `completed_at IS NOT NULL`, the call returns
        successfully (no-op) — does NOT raise
        `RunAlreadyCompletedError`. This makes the
        `CompletionMarkerFailedError` recovery story honest: a retry
        after the DB committed but the ack was lost succeeds cleanly.

        Raises `RunNotClaimedError` if no row exists for
        `(thread_id, run_id)` (genuine logic bug — claim_run never
        ran or its row was lost).
        """

    async def load_pending(
        self, thread_id: str
    ) -> tuple[HitlRequest, str | None] | None:
        """See §3.6.3. Returns (request, run_id) in one read, or None
        when no pending request exists."""
```

#### `cubepi.checkpointer.exceptions`

```python
class CheckpointerError(Exception):
    """Base class for cubepi checkpointer runtime errors.

    Separate from CubepiSchemaError (which is about DB-vs-library
    schema incompatibility). CheckpointerError is for runtime
    operation outcomes (missing thread, lock timeout, run state, etc.).
    """


class ThreadNotFoundError(CheckpointerError): ...
class ThreadAlreadyExistsError(CheckpointerError): ...


class RunNotCompletedError(CheckpointerError):
    """The cubepi_runs row for (thread_id, run_id) either does not
    exist, or exists with completed_at IS NULL (paused for HITL,
    abandoned, or in-flight)."""


class RunNotClaimedError(CheckpointerError):
    """mark_run_complete() called but no cubepi_runs row exists for
    (thread_id, run_id). Indicates a logic bug — claim_run() was
    not called or was not persisted."""


class RunAlreadyClaimedError(CheckpointerError):
    """claim_run() called but a cubepi_runs row already exists for
    (thread_id, run_id) with completed_at IS NULL. Another process
    is currently running this run_id (or the prior attempt was
    abandoned without cleanup). Retry with a different run_id."""


class RunAlreadyCompletedError(CheckpointerError):
    """claim_run() called but the cubepi_runs row already has
    completed_at IS NOT NULL. Runs are append-only; start a new run
    with a different run_id.

    NOT raised by mark_run_complete() — that path is idempotent on
    already-completed rows (see §3.6.2 retry semantics)."""


class CompletionMarkerFailedError(CheckpointerError):
    """mark_run_complete() failed AFTER the run's final append
    succeeded. The exception carries the `run_id` so callers using
    Agent.prompt(run_id=None) (cubepi-generated) can recover it and
    retry mark_run_complete() directly. See §3.6.2."""

    def __init__(self, *, thread_id: str, run_id: str, cause: BaseException):
        super().__init__(
            f"mark_run_complete failed for ({thread_id}, {run_id}): {cause}"
        )
        self.thread_id = thread_id
        self.run_id = run_id
        self.__cause__ = cause


class CheckpointerLockTimeoutError(CheckpointerError):
    """SQLite (or other locking backend) could not acquire the writer
    lock within the configured busy_timeout. See §3.3."""
```

#### `cubepi.agent.agent`

The existing `AgentState` dataclass gains one field:

```python
@dataclass
class AgentState:
    # ... existing fields ...
    active_run_id: str | None = None     # NEW
```

Set to the claimed run_id at the start of `prompt()` (after a
successful `claim_run()`). Cleared back to `None` ONLY on clean
return; on exception it is left set so callers reading
`agent.state.active_run_id` from the except block can still recover
it. For the specific `CompletionMarkerFailedError` case the run_id is
ALSO carried on the exception (`exc.run_id`) — that is the
recommended source of truth in `except` blocks because it survives
any subsequent `prompt()` invocation that would otherwise overwrite
`active_run_id`. `active_run_id` is also cleared back to None on
successful resume completion via `respond()`.

The single-flight `_run_lock` (see `cubepi/agent/agent.py:204`)
prevents two concurrent `prompt()` / `respond()` calls on the same
Agent instance from racing on `active_run_id`.

```python
class Agent(Generic[TMessage]):
    def __init__(
        self,
        *,
        # ... all existing args unchanged ...
        messages: Sequence[Message] | None = None,
    ):
        """`messages`: pre-seed initial history for ephemeral runs
        (used by fork_once). Deep-copies via `m.model_copy(deep=True)`.
        Raises `ValueError` if `messages` is combined with
        `thread_id` + `checkpointer` (pre-seed conflicts with lazy
        load). The exact validation of pre-seeded messages is
        backend-agnostic at this point (no §3.3-style invariants);
        callers passing arbitrary `messages` are on their own."""

    async def prompt(
        self,
        message: str | Message | list[Message],
        *,
        run_id: str | None = None,
    ) -> str:
        """See §3.6.1. Returns the run_id actually used. On any failure
        AFTER a successful claim, `self.state.active_run_id` remains
        set to the run_id so callers (and exception handlers) can
        recover it even when the return value is unavailable. On
        CompletionMarkerFailedError specifically, the run_id is also
        carried on the exception."""

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        after_run_id: str,
        metadata: JsonObject | None = None,
    ) -> None:
        """Persistent fork. Requires `self.checkpointer`. Delegates to
        `self.checkpointer.fork(...)`. Does NOT mutate `self`."""

    async def fork_once(
        self,
        src_thread_id: str,
        message: str | Message | list[Message],
        *,
        after_run_id: str,
    ) -> ForkOnceResult:
        """See §3.8."""
```

#### `cubepi.agent.types`

```python
@dataclass(frozen=True)
class ForkOnceResult:
    text: str
    messages: list[Message]   # new messages produced this turn only
    stop_reason: str
```

### 3.8 `Agent.fork_once()` execution detail

1. `self._require_checkpointer()` — `RuntimeError` if no checkpointer.
2. HITL pre-flight (§3.8.2) — `RuntimeError` if any inherited tool
   or middleware has `elem.hitl is not None`.
3. `snapshot = await self.checkpointer.snapshot(src_thread_id,
   after_run_id=...)` — propagates
   `ThreadNotFoundError` / `RunNotCompletedError`.
4. Build a transient `Agent` with this agent's `model`, `system_prompt`,
   `tools`, `middleware`, `convert_to_llm`, `messages=snapshot`,
   `checkpointer=None`, `thread_id=None`.
5. Start a `cubepi.agent.fork_once` span (inheriting the surrounding
   OTel context per §3.5). Capture the pre-seed length.
6. Cancellation is best-effort, not bounded — same caveats as
   `Agent.prompt()` (a tool that ignores abort can hold the task until
   it returns). Span is closed in `finally` with the appropriate
   status.
7. `await child.prompt(message, run_id=<fresh-uuid>)`. The fresh run_id
   is internal — never persisted (no checkpointer), but populated on
   the in-memory messages so observers see consistent metadata.
8. Read final assistant text + messages added after the pre-seed
   length from the child; close the span.
9. Return `ForkOnceResult(text, new_messages, stop_reason)`.

#### 3.8.1 Isolation contract

`fork_once()` guarantees ONE thing: **no message produced by the
transient run is written to the cubepi checkpointer**. The source
thread's persisted history is byte-identical before and after the
call.

`fork_once()` does NOT guarantee:

- That tools the transient agent invokes have no side effects.
  A tool that writes to an external DB / sends an email / calls a
  remote API will do so. A tool whose closure captures
  `self.thread_id` and writes to a side store keyed by it will
  contaminate the source.
- That middleware with internal mutable state doesn't change that
  state across the transient run.

This is by design. Cubepi cannot inspect closures or external
systems. The contract is narrowly: "cubepi's own message store is
untouched." Anything else is the caller's responsibility — pick the
tool/middleware set for the transient agent accordingly. The
recommended pattern when transient-safety matters: build a second
`Agent(...)` with only read-only / idempotent / fork-safe tools and
call `fork_once()` on that one.

The one tool/middleware shape cubepi DOES detect and reject is HITL,
because (a) HITL has no graceful "no-op" mode (the run blocks
forever) and (b) HITL channels write directly into the cubepi
checkpointer, which would silently violate the isolation contract
above. See §3.8.2.

#### 3.8.2 HITL is not supported inside `fork_once()`

Two reasons HITL cannot work in `fork_once()`:

1. **No persistence target.** HITL pending requests are written via
   `Checkpointer.save_pending_request(thread_id, ...)`. The transient
   agent has no checkpointer and no thread_id.
2. **Worse: inherited HITL channels write to the source thread.** A
   host typically constructs
   `CheckpointedChannel(checkpointer=cp, thread_id=conversation_id, …)`
   and binds it to `ask_user_tool(channel)`. The channel object holds
   the source `thread_id`. Reusing such a tool in `fork_once()` would
   persist a pending HITL request to the source thread — silent
   contamination.

Detection (structural, bypass-proof) — uses the same
`hitl: HitlBinding | None` attribute introduced in §3.6.3.1:

- Any element with `elem.hitl is not None` is rejected from
  fork_once. This covers both checkpointed channels (CheckpointedChannel
  side effects on source thread) and in-memory channels
  (InMemoryChannel blocks forever waiting for an unreachable host
  inside a transient agent).
- `cubepi.hitl.ask_user_tool(...)` and `ApprovalPolicyMiddleware`
  set `hitl` per §3.6.3.1; `ConfirmToolCallMiddleware` inherits.
- Third-party HITL tools / middleware MUST set `hitl` — same
  contract as §3.6.3.1.
- `fork_once()` scans `self.tools` and `self.middleware`; any
  element with `elem.hitl is not None` triggers `RuntimeError`
  BEFORE any snapshot is read.

### 3.9 Per-backend implementation sketch

#### Postgres / MySQL

Existing schema is at version 3. This spec bumps to **version 4** with
two additive changes (alembic migration in cubebox / cubepi-using apps):

- `ALTER TABLE cubepi_messages ADD COLUMN run_id VARCHAR/TEXT NULL`
  + index `(thread_id, run_id)`.
- New table `cubepi_runs`:
  - `thread_id TEXT/VARCHAR NOT NULL`
  - `run_id TEXT/VARCHAR NOT NULL`
  - `claimed_at TIMESTAMPTZ NOT NULL DEFAULT now()`
  - `completed_at TIMESTAMPTZ NULL`
  - `completion_seq BIGINT NULL` (allocated per-thread monotonic
    under the per-thread lock at mark_run_complete time;
    NULL means "claimed but not completed")
  - PRIMARY KEY `(thread_id, run_id)`
  - INDEX `(thread_id, completion_seq)` for fork's set-selection query
  - Postgres: `HASH (thread_id)` partitioned to match
    `cubepi_messages`;
    **`FOREIGN KEY (thread_id) REFERENCES cubepi_threads(thread_id)
    ON DELETE CASCADE`** so thread deletion cleans up runs (mirrors
    `cubepi_messages`).
  - MySQL: `KEY (thread_id)` partitioned, **no FK** (consistent with
    `cubepi_messages` — MySQL forbids FKs on partitioned tables).
    Hosts that delete threads MUST also DELETE the matching
    `cubepi_runs` rows in the same transaction, the same as they
    already do for `cubepi_messages`. Document this in the MySQL
    backend guide.

`fork()` in one transaction. **The thread row is created BEFORE
the message INSERT because `cubepi_messages.thread_id` FKs to
`cubepi_threads(thread_id)`. `forked_at_seq` is populated in a
trailing UPDATE since its value isn't known until messages have
been copied.**

1. Advisory lock / `FOR UPDATE` on the source thread.
2. `SELECT completion_seq AS cutoff FROM cubepi_runs WHERE
   thread_id = $src AND run_id = $after_run_id AND completion_seq IS
   NOT NULL` → if no row, `RunNotCompletedError`.
3. `INSERT INTO cubepi_threads (thread_id, parent_thread_id,
   forked_at_seq, extra, …)
   VALUES ($new, $src, NULL, $merged_extra, …)` —
   `ThreadAlreadyExistsError` on PK violation. `forked_at_seq` is
   NULL for now; populated in step 6.
4. Copy messages:
   ```sql
   INSERT INTO cubepi_messages (thread_id, seq, role, run_id,
                                metadata, payload)
   SELECT $new, seq, role, run_id, metadata, payload
   FROM cubepi_messages
   WHERE thread_id = $src
     AND (
       run_id IS NULL
       OR run_id IN (
         SELECT run_id FROM cubepi_runs
         WHERE thread_id = $src
           AND completion_seq IS NOT NULL
           AND completion_seq <= $cutoff
       )
     )
   ORDER BY seq;
   ```
5. `INSERT INTO cubepi_runs (thread_id, run_id, claimed_at,
   completed_at, completion_seq)
   SELECT $new, run_id, claimed_at, completed_at, completion_seq
   FROM cubepi_runs
   WHERE thread_id = $src
     AND completion_seq IS NOT NULL
     AND completion_seq <= $cutoff`.
6. `UPDATE cubepi_threads SET forked_at_seq = (SELECT MAX(seq)
   FROM cubepi_messages WHERE thread_id = $new) WHERE thread_id =
   $new`. The MAX subquery returns NULL for an empty-prefix fork
   (e.g., fork of a thread with only legacy NULL-run messages where
   the chosen run_id excluded everything) — NULL is the correct
   value in that edge case.
7. Commit.

`claim_run()`. **Must lazily ensure the `cubepi_threads` row exists
before inserting into `cubepi_runs`, because `cubepi_runs.thread_id`
FKs to it. This mirrors the existing `append()` lazy-creation
pattern: a brand-new thread's first writer creates the threads row
with defaults.**

- Take per-thread advisory lock / `FOR UPDATE`.
- `INSERT INTO cubepi_threads (thread_id) VALUES (?)
  ON CONFLICT (thread_id) DO NOTHING` — idempotently ensures the
  thread row exists with defaults (`parent_thread_id=NULL`,
  `forked_at_seq=NULL`, `extra='{}'`).
- `INSERT INTO cubepi_runs (thread_id, run_id) VALUES (?, ?)` →
  on PK conflict, `SELECT completed_at FROM cubepi_runs WHERE
  thread_id = ? AND run_id = ?` → if NULL,
  `RunAlreadyClaimedError`; else `RunAlreadyCompletedError`.
- Commit.

MySQL equivalent: `INSERT INTO cubepi_threads (thread_id) VALUES (?)
ON DUPLICATE KEY UPDATE thread_id = thread_id` — a no-op update on
duplicate that traps ONLY the dup-key case. **Do NOT use `INSERT
IGNORE`**: it silently swallows non-duplicate errors (type
coercion, FK violation if any) and would hide real bugs in the
lazy-create path. This matches the existing `append()` lazy-create
idiom in `cubepi/checkpointer/mysql/checkpointer.py`.

`mark_run_complete()`:

- Take per-thread advisory lock / `FOR UPDATE`.
- `SELECT completed_at FROM cubepi_runs WHERE thread_id=? AND run_id=?`.
  - No row → `RunNotClaimedError`.
  - Existing row with `completed_at IS NOT NULL` → **idempotent
    success**, commit, return.
  - Existing row with `completed_at IS NULL` → proceed to UPDATE.
- `next_seq = (SELECT COALESCE(MAX(completion_seq), 0) + 1 FROM
  cubepi_runs WHERE thread_id = ? AND completion_seq IS NOT NULL)`.
- `UPDATE cubepi_runs SET completed_at = now(), completion_seq =
  $next_seq WHERE thread_id = ? AND run_id = ?`.
- Commit.

`append()` (existing) is unchanged in surface but must persist
`message.run_id` into the new column.

#### SQLite

Schema additions at connect time using the existing PRAGMA-probe +
`ALTER TABLE` pattern (the file already does this for `run_id` on
`thread_pending_request`):

- `ALTER TABLE messages ADD COLUMN run_id TEXT NULL`.
- `CREATE TABLE IF NOT EXISTS runs (thread_id TEXT NOT NULL, run_id
  TEXT NOT NULL, claimed_at REAL NOT NULL DEFAULT (julianday('now')),
  completed_at REAL NULL, completion_seq INTEGER NULL, PRIMARY KEY
  (thread_id, run_id))`.
- `CREATE INDEX IF NOT EXISTS ix_runs_thread_completion ON runs
  (thread_id, completion_seq)`.
- `ALTER TABLE thread_extra ADD COLUMN parent_thread_id TEXT NULL`.

`fork()`:

1. `BEGIN IMMEDIATE`.
2. Validate source exists; validate `(src_thread_id, after_run_id)`
   has a `runs` row with `completion_seq IS NOT NULL` → else
   `RunNotCompletedError`.
3. Validate new thread does not exist (probe `messages` /
   `thread_extra` / `runs` for `new_thread_id`) →
   `ThreadAlreadyExistsError`.
4. Read cutoff: `SELECT completion_seq FROM runs WHERE thread_id =
   $src AND run_id = $after_run_id` (validated non-null in step 2).
5. `INSERT INTO messages (thread_id, run_id, message_json)
   SELECT $new, run_id, message_json FROM messages
   WHERE thread_id = $src
     AND (
       run_id IS NULL
       OR run_id IN (
         SELECT run_id FROM runs
         WHERE thread_id = $src
           AND completion_seq IS NOT NULL
           AND completion_seq <= $cutoff
       )
     )
   ORDER BY id`. New rows get fresh global `id`s (the `messages.id`
   column is a global auto-increment; identity is not preserved
   across the copy, but per-thread row order is).
6. `INSERT INTO runs (thread_id, run_id, claimed_at, completed_at,
   completion_seq)
   SELECT $new, run_id, claimed_at, completed_at, completion_seq
   FROM runs
   WHERE thread_id = $src
     AND completion_seq IS NOT NULL
     AND completion_seq <= $cutoff`.
7. `INSERT INTO thread_extra (thread_id, extra_json, parent_thread_id)
   VALUES ($new, $merged_extra_json, $src)`.
8. Commit.

`claim_run()`:

- `BEGIN IMMEDIATE`.
- `INSERT INTO runs (thread_id, run_id) VALUES (?, ?)` → on PK
  conflict, `SELECT completed_at FROM runs WHERE thread_id=? AND
  run_id=?`; NULL → `RunAlreadyClaimedError`, non-NULL →
  `RunAlreadyCompletedError`.
- Commit.

`mark_run_complete()`:

- `BEGIN IMMEDIATE`.
- `SELECT completed_at FROM runs WHERE thread_id=? AND run_id=?`.
  - No row → `RunNotClaimedError`.
  - `completed_at IS NOT NULL` → idempotent success, commit, return.
  - `completed_at IS NULL` → proceed.
- `next_seq = (SELECT COALESCE(MAX(completion_seq), 0) + 1 FROM
  runs WHERE thread_id=? AND completion_seq IS NOT NULL)`.
- `UPDATE runs SET completed_at = julianday('now'), completion_seq =
  $next_seq WHERE thread_id=? AND run_id=?`.
- Commit.

`append()` is also wrapped in `BEGIN IMMEDIATE` (uniform writer
discipline; see §3.3). `PRAGMA busy_timeout = 5000` set at connect.
Each appended `Message` carries its `run_id` value into the new
`messages.run_id` column.

No `forked_at_seq` column added — SQLite has no per-thread seq.

#### Memory

`MemoryCheckpointer` today is `dict[str, CheckpointData]`. This spec:

- Extends `CheckpointData` with `parent_thread_id: str | None = None`.
- Adds an internal `dict[str, dict[str, RunState]]` mapping
  `thread_id -> {run_id: RunState(claimed_at, completed_at,
  completion_seq)}` where `completed_at` / `completion_seq` are None
  until the run is marked complete. `completion_seq` is a per-thread
  monotonic int allocated in `mark_run_complete()` under the lock,
  mirroring the SQL backends.
- Adds a single shared `asyncio.Lock` that ALL write paths
  (`append`, `save_extra`, `save_pending_request`, `claim_run`,
  `mark_run_complete`, `fork`) take. Uniform locking removes
  check-then-write races.
- `Message.run_id` is just a field — Memory persists the Message via
  reference, no extra scaffolding.

`fork()` under the lock:

1. Source-exists check, new-thread-does-not-exist check.
2. Look up `runs[src][after_run_id]`; if missing or
   `completion_seq is None` → `RunNotCompletedError`.
3. Cutoff = `runs[src][after_run_id].completion_seq`.
4. Walk `src.messages` in order. Include each message if
   `m.run_id is None` OR `runs[src][m.run_id].completion_seq is not
   None AND runs[src][m.run_id].completion_seq <= cutoff`.
   Deep-copy via `model_copy(deep=True)`.
5. Filter `runs[src]` to entries with `completion_seq IS NOT NULL
   AND completion_seq <= cutoff`; deep-copy under the new thread_id.
6. Deep-copy `src.extra`; merge `extra['fork']=metadata` per §3.4.
7. Store `CheckpointData(messages=…, extra=…,
   parent_thread_id=src_thread_id)` under `new_thread_id`.

`claim_run()` under the lock:

- If `run_id in runs[thread_id]`: check
  `runs[thread_id][run_id].completed_at` → None →
  `RunAlreadyClaimedError`; non-None → `RunAlreadyCompletedError`.
- Else insert `RunState(claimed_at=monotonic_now(),
  completed_at=None, completion_seq=None)`.

`mark_run_complete()` under the lock:

- `state = runs[thread_id].get(run_id)`.
- If missing → `RunNotClaimedError`.
- If `state.completed_at is not None` → **idempotent success**, return.
- Else allocate `next_seq = max((s.completion_seq for s in
  runs[thread_id].values() if s.completion_seq is not None),
  default=0) + 1`; update state's completed_at + completion_seq.

`append()` under the lock:

- For each message with `m.run_id is not None`: if `m.run_id in
  runs[thread_id]` and `runs[thread_id][m.run_id].completed_at is
  not None` → raise `RunAlreadyCompletedError`. (Defense in depth;
  Agent.prompt() pre-claim makes this rare, but protects direct
  `Checkpointer.append()` callers.)
- Extend the messages list.

No `forked_at_seq` field — Memory has no seq.

### 3.10 Error semantics summary

| Situation | Raised |
|---|---|
| `self.checkpointer is None` (fork or fork_once) | `RuntimeError("fork requires a checkpointer")` |
| `fork_once()` finds HITL-bearing tool/middleware (§3.8.2) | `RuntimeError("fork_once() does not support HITL: <names>")` |
| `src_thread_id` does not exist | `ThreadNotFoundError(src_thread_id)` |
| `new_thread_id` already exists (fork) | `ThreadAlreadyExistsError(new_thread_id)` |
| `after_run_id`'s `cubepi_runs` row missing or has `completion_seq IS NULL` | `RunNotCompletedError(thread_id=src_thread_id, run_id=after_run_id)` |
| `prompt(run_id=R)` and `R` is currently claimed (completed_at IS NULL) on the thread | `RunAlreadyClaimedError(thread_id=..., run_id=R)` |
| `prompt(run_id=R)` and `R` has completed_at IS NOT NULL on the thread | `RunAlreadyCompletedError(thread_id=..., run_id=R)` |
| `mark_run_complete()` called without a prior `claim_run()` | `RunNotClaimedError(thread_id=..., run_id=...)` — indicates an agent-loop logic bug |
| `mark_run_complete()` called on a row that already has `completed_at IS NOT NULL` | **success (no-op)** — idempotent; supports retry-after-lost-ack from `CompletionMarkerFailedError` |
| `mark_run_complete()` fails AFTER the run's final append succeeded | `CompletionMarkerFailedError(thread_id=..., run_id=..., cause=...)` — `run_id` is recoverable even when `prompt(run_id=None)` was used |
| `Agent.fork_once()` child run errors | propagates (same surface as `Agent.prompt()`) |
| `Agent.fork_once()` is cancelled mid-turn | `asyncio.CancelledError` re-raises after transient agent abort completes (best-effort; see §3.8 step 6) |
| SQLite cannot acquire writer lock within `busy_timeout` | `CheckpointerLockTimeoutError` |
| `Agent(messages=..., thread_id=X, checkpointer=Y)` | `ValueError` — pre-seeding conflicts with lazy load |

## 4. Migration / Compatibility

- **Protocol change — full impact disclosure.** This is a v4
  Checkpointer contract change, not just a "fork API addition".
  Five new Protocol methods: `snapshot`, `fork`, `claim_run`,
  `mark_run_complete`, `load_pending`. Existing third-party
  checkpointers that implement only the v3 surface (`load`, `append`,
  `save_extra`, `save_pending_request`, `load_pending_request`) will
  fail **ordinary `Agent.prompt()` calls** once `claim_run()` is
  invoked at the start of every run — not just fork API calls.

  Mitigation: `Agent.prompt()` checks both `hasattr(self.checkpointer,
  "claim_run")` AND `hasattr(self.checkpointer, "mark_run_complete")`
  at the start. If EITHER is absent, `prompt()` runs in **legacy
  degraded mode**:

  - No `claim_run()` / `mark_run_complete()` call.
  - `Message.run_id` is still stamped on appended messages (purely
    informational — no marker exists to anchor it).
  - `Agent.fork()` / `Agent.fork_once()` on this checkpointer raises
    `CheckpointerError("backend does not support fork; missing
    claim_run / mark_run_complete")`.

  This keeps existing third-party Protocol-only impls working for
  vanilla `prompt()` while making the missing fork capability an
  explicit error rather than a silent corruption path. First-party
  backends (Memory, SQLite, Postgres, MySQL) always implement the
  new methods after this spec lands; the degraded mode is purely a
  third-party compatibility shim.

  `Checkpointer.load_pending_request` is kept as a thin alias for
  `load_pending()` returning only the request part — existing
  callers unchanged. The matching backend-specific
  `load_pending_run_id()` methods are deprecated in favor of
  `load_pending()` but left in place for one release.
- **`Agent.prompt()` signature**: adds an optional keyword `run_id`
  and changes return type from `None` to `str`. Returning a value that
  the caller previously ignored is **not** a breaking change for
  callers that wrote `await agent.prompt(msg)` — the return value can
  simply be discarded. Callers using `await agent.prompt(msg); …` keep
  working. Documenting the new return type in the migration page is
  enough.
- **`Agent.respond()` signature**: unchanged. Internal logic reads
  `run_id` from `pending_request`.
- **Storage**: Postgres / MySQL — schema v3 → v4 (additive: one
  column on `cubepi_messages`, one new table `cubepi_runs`,
  matching partition strategy). Alembic migration provided.
  SQLite — additive `ALTER TABLE` and `CREATE TABLE IF NOT EXISTS` at
  connect time. Memory — N/A.
- **Existing messages** (`run_id IS NULL`) remain readable; not
  forkable / not deletable (§3.6.4). No backfill.
- **Existing `cubepi_threads.pending_request.run_id`** semantics
  unchanged — it is the host-side run identifier for HITL recovery.
  The new `Message.run_id` is structurally the same string; the same
  value lives in both places during an active HITL pause, and that is
  intentional (the value passed to `Agent.prompt(run_id=…)` is the
  value written to `pending_request.run_id` and to every appended
  `Message.run_id`).

## 5. Testing

- **Unit / per-backend** (Memory + SQLite in-process; Postgres against
  the bundled docker fixture; MySQL against the live test server at
  `reference_mysql_test_server`):

  - `prompt()` accept-or-generate: caller-supplied run_id is used
    verbatim; None generates a uuid; return value matches what was
    persisted on appended messages
  - `prompt(run_id=R)` raises `RunAlreadyClaimedError` if R is
    currently claimed (completed_at IS NULL) on the thread; raises
    `RunAlreadyCompletedError` if R is already completed
  - `mark_run_complete()` is called AFTER the final append, NOT
    atomically with it. Inject a checkpointer that raises on
    `mark_run_complete()` and assert: messages of the run are
    persisted; the `cubepi_runs` row exists with `completed_at IS
    NULL`; `Agent.prompt()` raises
    `CompletionMarkerFailedError(run_id=…)`. Retrying
    `checkpointer.mark_run_complete(thread_id, exc.run_id)` succeeds
    and the run becomes forkable.
  - HITL pause does NOT call `mark_run_complete()`; `cubepi_runs`
    row remains with `completed_at IS NULL`. `respond()` calls
    `mark_run_complete()` once the resumed loop completes terminally.
  - fork happy path: source has 3 completed runs A, B, C →
    `fork(after_run_id=B)` produces a thread with messages of A+B
    (in source order), `runs` rows for A+B, and no
    pending_request
  - fork preserves `extra`; sets `parent_thread_id`; (PG/MySQL) sets
    `forked_at_seq` to the last copied seq
  - `forked_at_seq` is NOT stored for Memory/SQLite (no column/field)
  - `extra['fork']` overwrites any pre-existing `extra['fork']` on
    the source
  - `fork` does NOT copy `pending_request` / host-side run_id
  - `ThreadAlreadyExistsError` on collision; nothing written
  - `ThreadNotFoundError` on bad source
  - `RunNotCompletedError` when `after_run_id` does not exist, is
    from a different thread, or is paused / aborted (no marker)
  - source thread unaffected by fork (independence test, byte-equal
    before/after)
  - fork-of-fork lineage: A → B at run X; B → C at run Y (Y is one of
    B's runs). C's `parent_thread_id == B`; for PG/MySQL,
    `forked_at_seq` is B's seq for Y, not A's
  - subsequent `prompt()` on a forked thread starts a new run_id; new
    messages get the new run_id; completion writes its own marker
  - concurrent fork + mark_run_complete on source serialize correctly
  - **interleaved runs**: append run A's msgs (seq 1,2), append run
    B's msgs (seq 3,4), mark B complete, mark A complete (later than
    B). `fork(after_run_id=A)` copies A+B both (B completed earlier);
    `fork(after_run_id=B)` copies only B; messages of B's seqs 3,4
    are NOT pulled by `fork(after_run_id=A)` if A completed first
    (regression test for the v2 R1 CRITICAL finding)
  - **legacy + new mixed thread**: legacy NULL-run_id messages on a
    pre-spec thread + a new completed run R. `fork(after_run_id=R)`
    copies the legacy prefix AND R's messages, in source seq order;
    `delete_run(thread_id, R)` (when implemented) removes only R's
    messages, legacy prefix untouched
  - **`Agent.prompt(run_id=R)` atomic claim**: if R is already
    claimed (completed_at IS NULL) on the thread, raise
    `RunAlreadyClaimedError` BEFORE any append happens. If R is
    already completed, raise `RunAlreadyCompletedError`. In both
    cases assert the source message count is unchanged.
  - **Same-run_id concurrent claim race** (regression test for v2
    R2 HIGH #1): two coroutines call `Agent.prompt(thread, run_id=R)`
    concurrently. Exactly one claim succeeds and proceeds to
    completion; the other raises `RunAlreadyClaimedError` with ZERO
    appends. After both settle, `fork(after_run_id=R)` includes only
    the winning process's messages (none from the loser, because the
    loser never appended).
  - **`completion_seq` strict ordering** (regression test for v2 R2
    HIGH #2): mark runs A, B, C complete in that order; assert
    `runs[A].completion_seq < runs[B].completion_seq <
    runs[C].completion_seq` strictly, regardless of wall-clock
    timestamps. Force a wall-clock tie (mock now() to return the
    same value) and assert ordering is still strict and matches
    actual completion order.
  - **`CompletionMarkerFailedError` recovery** (regression test for
    v2 R2 HIGH #3): use `Agent.prompt(run_id=None)` to let cubepi
    generate the run_id. Inject a failure on `mark_run_complete()`.
    Assert the exception is `CompletionMarkerFailedError`, that
    `exc.run_id` matches `Agent.state.active_run_id` from
    immediately before, and that retrying
    `checkpointer.mark_run_complete(thread_id, exc.run_id)`
    successfully finishes the run.
  - **HITL resume run_id continuity**: prompt(run_id=R) pauses;
    pending_request stores R; respond() recovers R via
    `load_pending()` (single read), continues, every appended message
    post-resume carries `run_id=R`, `mark_run_complete()` writes R's
    marker on terminal exit. Second-pause variant: respond() pauses
    again with same R, third respond() resumes again with same R.
    **Single-claim invariant** (regression test for v2 R4 HIGH #2):
    `claim_run()` is called exactly once across the whole
    prompt + N × respond chain (assert via a checkpointer spy).
    `respond()` calling `claim_run()` would raise
    `RunAlreadyClaimedError` — if the test ever sees that, regression
    triggered.
  - **fork_once no checkpointer side effects** (regression test for
    v2 R4 HIGH #1): the transient agent inside `fork_once()` runs
    `prompt(message, run_id=<fresh>)` with `checkpointer=None`. No
    `claim_run()` or `mark_run_complete()` call is made (assert via
    spying on a checkpointer instance that would have been used by
    the parent — and observe no calls from the transient agent).
  - **`mark_run_complete()` idempotency** (regression test for v2 R7
    HIGH #3): call mark_run_complete twice on the same
    `(thread_id, run_id)`; second call returns success (no raise),
    `completion_seq` unchanged from the first call.
  - **HITL channel run_id binding** (regression test for v2 R7
    HIGH #2 + v2 R8 HIGH #2): construct Agent with
    `ask_user_tool(CheckpointedChannel(checkpointer=cp,
    thread_id=t, run_id="R1"))`. Assert the returned tool has
    `tool.bound_hitl_run_id == "R1"` (the factory MUST set this).
    Then `Agent.prompt(message, run_id=None)` → raises `ValueError`
    (generate-mode rejected, error message names the bound run_id).
    `Agent.prompt(message, run_id="R2")` (mismatch) → raises
    `ValueError`. `Agent.prompt(message, run_id="R1")` → succeeds.
    Same with `ApprovalPolicyMiddleware` instead of the tool. Non-HITL
    agents accept `run_id=None` unchanged.
  - **Loop-outcome completion enumeration** (regression test for v2
    R7 HIGH #5 + v2 R9 HIGH #2): exercise each row of the §3.6.2
    table — clean success, **incomplete tool cycle** (custom
    `after_model_response(decision="stop")` on a tool-use
    AssistantMessage), HITL suspended (normal pause), HITL detached
    (HitlDetached caught in loop), HITL aborted (abort_pending),
    provider exception, tool exception, abort during streaming,
    propagating cancellation — and assert
    `cubepi_runs.completed_at IS NULL` for every non-success outcome
    and `IS NOT NULL` for the clean-success outcome. HITL
    suspended → respond() resume → assert marker now NOT NULL.
    The incomplete-tool-cycle test specifically covers each
    shape rejected by the §3.6.2 invariant:
    - (i) `after_model_response(decision="stop")` on a tool-use
      response (no tool_results appended at all);
    - (ii) `after_model_response` injects a `UserMessage` between
      the tool-use assistant and what would be its tool_results
      (subsequent matching tool_call_id later in the run does NOT
      satisfy the strict-adjacency requirement);
    - (iii) `after_model_response` produces a partial cover (only
      some of the assistant's tool_call_ids have results in the
      adjacent block);
    - (iv) two assistants in the run with reused tool_call_ids —
      strict adjacency prevents the later assistant's results from
      satisfying the earlier assistant's calls.
    For each shape: assert (a) the pre-completion invariant
    rejected the run; (b) `fork(after_run_id=R)` raises
    `RunNotCompletedError`.
  - **Legacy-degraded mode** (regression test for v2 R7 HIGH #4 +
    v2 R8 HIGH #3): construct an Agent with a checkpointer that
    implements only the v3 surface (no `claim_run` /
    `mark_run_complete`). Vanilla `Agent.prompt(msg)` succeeds (no
    claim attempted; messages stamped with run_id but no marker
    written). Calling `Agent.fork(...)` or `Agent.fork_once(...)`
    raises `CheckpointerError`.

    Partial-implementation variant: checkpointer has `claim_run`
    but NOT `mark_run_complete` (or the inverse). Agent.prompt()
    treats it the same as fully-missing: no claim attempted, no
    marker attempted. This protects against the
    "claim-but-can't-complete" hazard where vanilla prompt() would
    leave orphan claim rows on every call.
  - SQLite cross-connection: two `SQLiteCheckpointer` instances on
    the same DB file, one `fork()` and one `mark_run_complete()`
    from different processes serialize via `BEGIN IMMEDIATE`
  - legacy data: thread with `run_id=NULL` messages and no completion
    markers raises `RunNotCompletedError` on `fork(...)`
  - `CheckpointData.parent_thread_id` round-trip via `load()`

- **`Agent.fork_once()` (FauxProvider)**:

  - simple text-only follow-up returns expected final text; source
    thread unchanged
  - turn with tool calls completes fully, returns new messages
  - raises `RuntimeError` when no checkpointer
  - raises `RuntimeError` when any tool in `self.tools` has
    `elem.hitl is not None` (exercised with the actual
    `ask_user_tool(...)` factory result, asserting it populated
    `tool.hitl` from the channel)
  - raises `RuntimeError` when `ApprovalPolicyMiddleware` (or its
    subclass `ConfirmToolCallMiddleware`) in `self.middleware`
  - both checkpointed AND in-memory HITL channels trip the rejection
    (HitlBinding.checkpointed=False is still rejected by fork_once
    because the transient agent cannot drive HITL to completion)
  - cancellation: `asyncio.wait_for(agent.fork_once(...), timeout=…)`
    raises `TimeoutError`; transient agent's abort fires
  - tracing span: emits `cubepi.agent.fork_once` with the documented
    attributes; nests under surrounding span if one exists; otherwise
    is a trace root

- **`Agent(messages=...)` constructor (§3.7)**:

  - happy path: pre-seeded messages reflected in the next `prompt()`
  - `Agent(messages=[...], thread_id="t", checkpointer=cp)` raises
    `ValueError`
  - deep-copy isolation: mutate every nested mutable field
    (`AssistantMessage.content`, `ToolCall.arguments`,
    `ToolResultMessage.content`, every `metadata` dict) on the
    ORIGINAL messages after construction; assert the agent's
    internals are unchanged. Mirror the mutation against the agent
    and assert the originals are unchanged.

- **Exception hierarchy** (`tests/checkpointer/test_exceptions.py`):
  every new error (`ThreadNotFoundError`, `ThreadAlreadyExistsError`,
  `RunNotCompletedError`, `RunAlreadyCompletedError`,
  `CheckpointerLockTimeoutError`) is catchable via
  `except CheckpointerError`. `CheckpointerError` is NOT a subclass
  of `CubepiSchemaError`.

## 6. Open questions

All closed during brainstorm:

- Storage model → physical copy
- Fork handle → `after_run_id` only (no message_count, no Message.id,
  no after_response_id)
- Run state ownership → cubepi (not cubebox)
- Run_id source → accept-or-generate from `Agent.prompt`
- Legacy data → not forkable / not deletable; no backfill
- Run completion atomicity → NOT coupled to final append; separate
  `mark_run_complete()` Protocol method called after the final
  append, before `Agent.prompt()` returns. Crash window between the
  two is acceptable (run looks unfinished, fork blocked, recoverable)
- HITL pause/resume → same run_id across suspension; marker written
  on resume completion
- fork_once HITL → banned via `requires_hitl` marker

## 7. Out of this PR (follow-ups)

- **`Agent.delete_run(thread_id, run_id, *, including_subsequent: bool
  = True)`** — separate spec. Data model laid down here makes it a
  small change: a `DELETE WHERE thread_id=? AND run_id=?` (single-run
  surgical delete, leaves a hole; documented caveat) or a
  `DELETE WHERE thread_id=? AND seq >= min_seq_of_run` (rollback,
  also drops subsequent runs).
- Cubebox wiring: new `Conversation.parent_conversation_id` column,
  `POST /conversations/{id}/fork` endpoint, per-assistant-message UI
  button. Lives in the cubebox repo.
- CLI sugar (`cubepi fork`) — only if a real user asks.
- Backfill heuristic for legacy threads (best-effort run-boundary
  inference) — only if a real workload needs it.
