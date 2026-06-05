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
- New `cubepi_run_completions` storage per backend recording which runs
  have finished cleanly.
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
relevant `cubepi_run_completions` rows so the new thread keeps its
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
2. Every message whose `run_id` belongs to the set of completed runs
   on `src_thread_id` whose `completed_at` is **at or before**
   `after_run_id`'s `completed_at`.

Messages are inserted into the new thread in source seq order.

#### Why "set of completed runs", not "seq <= last_seq_of(after_run_id)"

A naive seq-cut is unsafe under concurrent runs. Example: runs A and B
both active on the same thread; A writes seqs 1, 2, 5 and completes,
while B writes seqs 3, 4 and is still in flight. A seq-cut
"`seq <= 5`" would pull B's seqs 3, 4 into the fork — messages of an
unfinished run we have no right to copy. The set-based selection
solves this by construction: only messages tagged with a *completed*
run_id (or NULL for legacy) are eligible.

This selection is also robust to:

- **Interleaved runs**: each run's messages are addressed by tag, not
  by position. Gaps in seq are fine.
- **In-flight runs**: their messages have a `run_id` not yet in
  `cubepi_run_completions` → excluded.
- **Mixed legacy + post-upgrade threads**: NULL-run_id messages are
  preserved as a chronological prefix, post-upgrade completed runs
  appear after.
- **Tie in `completed_at`**: two completions at the same wall-clock
  tick are tiebroken by lexicographic `run_id` for deterministic
  output. (Rare in practice; documented for spec completeness.)

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

#### Recommendation (not enforced): one active run per thread

For UX clarity and predictable ordering, hosts SHOULD serialize runs
per thread (cubebox naturally does — one conversation = one in-flight
prompt at a time). The spec does NOT enforce this — the set-based fork
selection means interleaved runs are *correct*, just unusual. A future
spec may add a "one active run per thread" lease if a real workload
needs the stronger guarantee.

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
  `INSERT INTO cubepi_run_completions … SELECT …`. Commit releases
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
| `cubepi_messages` rows where `run_id IS NULL` OR `run_id IN {completed runs of src with completed_at <= after_run_id.completed_at}` | yes | physical copy; PG/MySQL preserve source `seq` values for the copied range. SQLite copies the JSON payloads under fresh global `id`s (its `messages.id` is a global auto-increment, not a per-thread seq). Memory copies in-list order. Each copied row keeps its original `run_id` value (or NULL). Source seq order is preserved across the copy. |
| `cubepi_run_completions` rows for the copied runs | yes | so the new thread can be further forked / deleted by run |
| `extra` | yes | deep copy of the source JSON object |
| `parent_thread_id` | written (new) | set to `src_thread_id` on the new thread row |
| `forked_at_seq` | written (PG/MySQL only) | the `seq` of the last message in the copied set (highest copied seq). Memory and SQLite store no equivalent — those backends have no per-message seq column and lineage is recoverable from `parent_thread_id` + `cubepi_run_completions` alone. |
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

If the caller supplies a `run_id` that already has a completion
marker on the source thread, `prompt()` raises
`RunAlreadyCompletedError` **before any append happens** — runs are
append-only; you cannot continue or re-run a completed run. (Use a
new `run_id` for a new exchange.) The pre-flight check is mandatory
to prevent "messages were written before the conflict was detected"
half-states.

There is a narrow race window: two processes both pre-check, both see
no marker for run_id R, both begin appending. The first to call
`mark_run_complete()` wins; the second raises
`RunAlreadyCompletedError` from the marker INSERT (unique PK on
`(thread_id, run_id)`). The losing process's intermediate appends
remain on the thread, tagged with R but with no marker → fork by R
excludes them, and the future `delete_run(thread_id, R)` cleans them
up. Callers that race must retry with a fresh `run_id`. Hosts that
serialize runs per thread (cubebox does) never hit this.

#### 3.6.2 Completion marker — when written

The marker `cubepi_run_completions(thread_id, run_id, completed_at)`
is written by a dedicated, separate Protocol method
`Checkpointer.mark_run_complete(thread_id, run_id) -> None` — a
single-row INSERT, called AFTER the run's final `append()` and
BEFORE `Agent.prompt()` returns to the caller.

**The marker write is NOT atomic with the final append.** The
existing cubepi event loop (`cubepi/agent/loop.py`) persists each
message on its own `MessageEndEvent`; the loop only knows it has
finished when `AgentEndEvent` fires, *after* the final append.
Trying to atomically couple the two would require an Agent-layer
buffering rewrite that is out of scope.

This is acceptable. Two failure modes and their consequences:

- **Process crash between final append and `mark_run_complete()`**:
  messages of run R are persisted but no completion marker exists.
  `fork(after_run_id=R)` raises `RunNotCompletedError`. The user
  sees "the run looks finished but I can't fork it yet." A future
  admin / recovery API can backfill the marker. The data is not
  corrupt — just in an unmarked state, same as an abandoned run.
- **Transient checkpointer failure on `mark_run_complete()`**:
  `prompt()` raises the underlying error so the caller knows. The
  messages are already persisted; retrying just `mark_run_complete()`
  with the same `(thread_id, run_id)` succeeds. Idempotency: if the
  caller retries `prompt(run_id=R)` instead, it gets
  `RunAlreadyCompletedError` from the pre-flight check (§3.6.1)
  EXCEPT — pre-flight checks for the marker; if the marker still
  isn't there because of the same persistent failure, the caller can
  resume by calling `mark_run_complete()` directly, or by abandoning
  R and starting fresh. The recommended pattern is "retry the marker
  call, not the whole prompt".

Trigger conditions for the agent loop to call `mark_run_complete()`:

- The loop exits cleanly with a terminal stop_reason
  (`end_turn` / `tool_use_completed`-then-`end_turn` / any
  non-suspended terminal state)
- AND no pending HITL request remains for this run

If `prompt()` returns because of:

- **HITL pause** (`pending_request` set) → marker NOT written; resumed
  by `respond()` (§3.6.3).
- **Exception / abort / cancellation** → marker NOT written; the run
  is abandoned. Its messages remain on the thread under the same
  `run_id`, but `fork(after_run_id=X)` raises `RunNotCompletedError`,
  and the future `delete_run(X)` can clean them up.

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

`Agent.respond()` signature does **not** change.

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
    `completed_at <= R.completed_at`
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
        `after_run_id`. Copies `cubepi_run_completions` rows for the
        included runs. Records `parent_thread_id=src_thread_id` and
        (PG/MySQL only) `forked_at_seq`. Copies `extra` deeply. Writes
        `extra['fork'] = metadata` when `metadata` is supplied.

        Raises `ThreadNotFoundError`, `ThreadAlreadyExistsError`,
        `RunNotCompletedError`.
        """

    async def mark_run_complete(
        self,
        thread_id: str,
        run_id: str,
    ) -> None:
        """Insert the `cubepi_run_completions` row for `(thread_id,
        run_id)`. Single-row write, NOT coupled to the final append
        (see §3.6.2 for the rationale).

        Called by the agent loop AFTER the run's final `append()` and
        BEFORE `Agent.prompt()` returns. PK conflict on
        `(thread_id, run_id)` raises `RunAlreadyCompletedError`.
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
    """No completion marker for (thread_id, run_id). The run either
    does not exist on this thread, was paused for HITL and never
    resumed, or failed / was aborted before terminal completion."""


class RunAlreadyCompletedError(CheckpointerError):
    """The run already has a completion marker. Runs are append-only;
    start a new run with a different run_id."""


class CheckpointerLockTimeoutError(CheckpointerError):
    """SQLite (or other locking backend) could not acquire the writer
    lock within the configured busy_timeout. See §3.3."""
```

#### `cubepi.agent.agent`

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
        """See §3.6.1. Returns the run_id actually used."""

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
2. HITL pre-flight (§3.8.2) — `RuntimeError` if inherited tools or
   middleware mark `requires_hitl=True`.
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

Detection (marker-based, bypass-proof):

- New attribute `requires_hitl: bool = False` on
  `cubepi.agent.types.AgentTool` and `cubepi.middleware.base.Middleware`.
- Set `True` on the `AgentTool` returned by
  `cubepi.hitl.ask_user_tool(...)` and on
  `cubepi.hitl.middleware.ApprovalPolicyMiddleware`
  (`ConfirmToolCallMiddleware` inherits via subclassing).
- Third-party HITL tools / middleware MUST set the same flag.
- `fork_once()` scans `self.tools` and `self.middleware`; any
  `requires_hitl=True` element triggers `RuntimeError` BEFORE any
  snapshot is read.

### 3.9 Per-backend implementation sketch

#### Postgres / MySQL

Existing schema is at version 3. This spec bumps to **version 4** with
two additive changes (alembic migration in cubebox / cubepi-using apps):

- `ALTER TABLE cubepi_messages ADD COLUMN run_id VARCHAR/TEXT NULL`
  + index `(thread_id, run_id)`.
- New table `cubepi_run_completions(thread_id TEXT/VARCHAR, run_id
  TEXT/VARCHAR, completed_at TIMESTAMPTZ, PRIMARY KEY (thread_id,
  run_id), FOREIGN KEY (thread_id) REFERENCES cubepi_threads)`.
  Postgres: `HASH (thread_id)` partitioned to match `cubepi_messages`.
  MySQL: `KEY (thread_id)` partitioned, no FK (consistent with
  `cubepi_messages`).

`fork()` in one transaction:

1. Advisory lock / `FOR UPDATE` on the source thread.
2. `SELECT completed_at AS cutoff FROM cubepi_run_completions WHERE
   thread_id = $src AND run_id = $after_run_id` → if no row,
   `RunNotCompletedError`.
3. `INSERT INTO cubepi_threads (thread_id, parent_thread_id,
   forked_at_seq, extra, …)
   VALUES ($new, $src, $last_copied_seq, $merged_extra, …)` —
   `ThreadAlreadyExistsError` on PK violation. `$last_copied_seq` is
   computed below in step 4 (or set with a CTE / temp value depending
   on implementation).
4. `INSERT INTO cubepi_messages (thread_id, seq, role, run_id,
   metadata, payload)
   SELECT $new, seq, role, run_id, metadata, payload
   FROM cubepi_messages
   WHERE thread_id = $src
     AND (
       run_id IS NULL                                   -- legacy prefix
       OR run_id IN (
         SELECT run_id FROM cubepi_run_completions
         WHERE thread_id = $src
           AND (completed_at, run_id) <= ($cutoff, $after_run_id)
       )
     )
   ORDER BY seq`.
   The composite `(completed_at, run_id)` comparison gives the
   deterministic tiebreak from §3.2.
5. `INSERT INTO cubepi_run_completions (thread_id, run_id,
   completed_at)
   SELECT $new, run_id, completed_at
   FROM cubepi_run_completions
   WHERE thread_id = $src
     AND (completed_at, run_id) <= ($cutoff, $after_run_id)`.
6. (Optional, audit) `UPDATE cubepi_threads SET forked_at_seq =
   (SELECT MAX(seq) FROM cubepi_messages WHERE thread_id = $new)
   WHERE thread_id = $new` — sets the lineage marker.
7. Commit.

`mark_run_complete()` (single statement, own transaction OR within
the existing append/save lock window — the spec doesn't mandate;
either is correct because the row is independently consistent):

- Take the per-thread advisory lock / `FOR UPDATE` to serialize
  vs. concurrent appends / forks.
- `INSERT INTO cubepi_run_completions (thread_id, run_id,
  completed_at) VALUES (?, ?, now())` —
  `RunAlreadyCompletedError` on PK violation.
- Commit.

`append()` (existing) is unchanged in surface but must persist
`message.run_id` into the new column.

#### SQLite

Schema additions at connect time using the existing PRAGMA-probe +
`ALTER TABLE` pattern (the file already does this for `run_id` on
`thread_pending_request`):

- `ALTER TABLE messages ADD COLUMN run_id TEXT NULL`.
- `CREATE TABLE IF NOT EXISTS run_completions (thread_id TEXT NOT
  NULL, run_id TEXT NOT NULL, completed_at REAL NOT NULL DEFAULT
  (julianday('now')), PRIMARY KEY (thread_id, run_id))`.
- `ALTER TABLE thread_extra ADD COLUMN parent_thread_id TEXT NULL`.

`fork()`:

1. `BEGIN IMMEDIATE`.
2. Validate source exists; validate `(src_thread_id, after_run_id)`
   has a `run_completions` row → else `RunNotCompletedError`.
3. Validate new thread does not exist (probe `messages` /
   `thread_extra` / `run_completions` for `new_thread_id`) →
   `ThreadAlreadyExistsError`.
4. Read cutoff: `SELECT completed_at FROM run_completions WHERE
   thread_id = $src AND run_id = $after_run_id` (already validated
   non-null in step 2).
5. `INSERT INTO messages (thread_id, run_id, message_json)
   SELECT $new, run_id, message_json FROM messages
   WHERE thread_id = $src
     AND (
       run_id IS NULL
       OR run_id IN (
         SELECT run_id FROM run_completions
         WHERE thread_id = $src
           AND (completed_at, run_id) <= ($cutoff, $after_run_id)
       )
     )
   ORDER BY id`. New rows get fresh global `id`s (the `messages.id`
   column is a global auto-increment; identity is not preserved
   across the copy, but per-thread row order is).
6. `INSERT INTO run_completions (thread_id, run_id, completed_at)
   SELECT $new, run_id, completed_at FROM run_completions
   WHERE thread_id = $src
     AND (completed_at, run_id) <= ($cutoff, $after_run_id)`.
7. `INSERT INTO thread_extra (thread_id, extra_json, parent_thread_id)
   VALUES ($new, $merged_extra_json, $src)`.
8. Commit.

`mark_run_complete()`:

- `BEGIN IMMEDIATE`.
- `INSERT INTO run_completions (thread_id, run_id) VALUES (?, ?)` →
  `RunAlreadyCompletedError` if `(thread_id, run_id)` PK conflict.
- Commit.

`append()` is also wrapped in `BEGIN IMMEDIATE` (uniform writer
discipline; see §3.3). `PRAGMA busy_timeout = 5000` set at connect.
Each appended `Message` carries its `run_id` value into the new
`messages.run_id` column.

No `forked_at_seq` column added — SQLite has no per-thread seq.

#### Memory

`MemoryCheckpointer` today is `dict[str, CheckpointData]`. This spec:

- Extends `CheckpointData` with `parent_thread_id: str | None = None`.
- Adds an internal `dict[str, dict[str, float]]` mapping
  `thread_id -> {run_id: completed_at_monotonic_index}` — the value
  is just an insertion-order counter (per-thread monotonic int)
  serving as the equivalent of `completed_at` for ordering. Memory
  does not have a real wall-clock requirement; the counter is
  sufficient for the `(completed_at, run_id) <= cutoff` comparison.
- Adds a single shared `asyncio.Lock` that ALL write paths
  (`append`, `save_extra`, `save_pending_request`,
  `mark_run_complete`, `fork`) take. The existing single-statement
  methods didn't strictly need one, but uniform locking removes the
  finding-#4 check-then-write race in this backend.
- `Message.run_id` is just a field — Memory persists the whole
  Message via reference, so no extra storage scaffolding.

`fork()` under the lock:

1. Source-exists check, new-thread-does-not-exist check.
2. Look up the completed run_ids map for `src_thread_id`; if
   `after_run_id` not in it → `RunNotCompletedError`.
3. Compute the cutoff index = `completions[src][after_run_id]`.
4. Walk `src.messages` in order. For each message: include it if
   `m.run_id is None` OR `(completions[src][m.run_id], m.run_id) <=
   (cutoff, after_run_id)`. Deep-copy via `model_copy(deep=True)`.
5. Subset of `src`'s completion map filtered by the same cutoff:
   carry under the new thread_id.
6. Deep-copy `src.extra`; merge `extra['fork']=metadata` per §3.4.
7. Store `CheckpointData(messages=…, extra=…,
   parent_thread_id=src_thread_id)` under `new_thread_id`.

`mark_run_complete()` under the lock:

- Check `run_id NOT IN completions[thread_id]` → else
  `RunAlreadyCompletedError`.
- Add `run_id` with the next monotonic counter value as the ordering
  key.

`append()` under the lock:

- Optional pre-check: if any `message.run_id is not None` and that
  run_id is already in `completions[thread_id]` → raise
  `RunAlreadyCompletedError` (defense in depth; agent loop already
  pre-flights at prompt() entry, but checking here protects direct
  `Checkpointer.append()` calls).
- Extend the messages list.

No `forked_at_seq` field — Memory has no seq.

### 3.10 Error semantics summary

| Situation | Raised |
|---|---|
| `self.checkpointer is None` (fork or fork_once) | `RuntimeError("fork requires a checkpointer")` |
| `fork_once()` finds HITL-bearing tool/middleware (§3.8.2) | `RuntimeError("fork_once() does not support HITL: <names>")` |
| `src_thread_id` does not exist | `ThreadNotFoundError(src_thread_id)` |
| `new_thread_id` already exists (fork) | `ThreadAlreadyExistsError(new_thread_id)` |
| `after_run_id` has no completion marker on the source thread | `RunNotCompletedError(thread_id=src_thread_id, run_id=after_run_id)` |
| `prompt(run_id=R)` and `R` already has a completion marker on the thread | `RunAlreadyCompletedError(thread_id=..., run_id=R)` |
| `Agent.fork_once()` child run errors | propagates (same surface as `Agent.prompt()`) |
| `Agent.fork_once()` is cancelled mid-turn | `asyncio.CancelledError` re-raises after transient agent abort completes (best-effort; see §3.8 step 6) |
| SQLite cannot acquire writer lock within `busy_timeout` | `CheckpointerLockTimeoutError` |
| `Agent(messages=..., thread_id=X, checkpointer=Y)` | `ValueError` — pre-seeding conflicts with lazy load |

## 4. Migration / Compatibility

- **Protocol change**: `Checkpointer.snapshot`, `Checkpointer.fork`,
  `Checkpointer.mark_run_complete`, `Checkpointer.load_pending`
  are new methods on the `runtime_checkable` Protocol. Existing
  user-implemented checkpointers keep type-checking; they only fail
  when the new methods are actually called.
  `Checkpointer.load_pending_request` is kept as a thin alias for
  `load_pending()` returning only the request part — existing callers
  unchanged.
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
  column on `cubepi_messages`, one new table `cubepi_run_completions`,
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
  - `prompt()` raises `RunAlreadyCompletedError` if caller passes a
    run_id that already has a completion marker
  - completion marker written atomically with the final append (test:
    crash between final append and marker write is structurally
    impossible — they're one transaction)
  - HITL pause does NOT write a completion marker; `respond()` resume
    DOES write it once the resumed loop completes
  - fork happy path: source has 3 completed runs A, B, C →
    `fork(after_run_id=B)` produces a thread with messages of A+B
    (in source order), `run_completions` rows for A+B, and no
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
  - **`Agent.prompt(run_id=R)` pre-flight check**: if R has a
    completion marker on the source thread, raise
    `RunAlreadyCompletedError` BEFORE any append happens (asserted
    by snapshotting message count, attempting prompt, asserting
    raise + unchanged count)
  - **HITL resume run_id continuity**: prompt(run_id=R) pauses;
    pending_request stores R; respond() recovers R via
    `load_pending()` (single read), continues, every appended message
    post-resume carries `run_id=R`, `mark_run_complete()` writes R's
    marker on terminal exit. Second-pause variant: respond() pauses
    again with same R, third respond() resumes again with same R
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
  - raises `RuntimeError` when `requires_hitl=True` tool in
    `self.tools` (exercised with the actual `ask_user_tool(...)`
    factory result)
  - raises `RuntimeError` when `ApprovalPolicyMiddleware` (or its
    subclass `ConfirmToolCallMiddleware`) in `self.middleware`
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
  also drops subsequent run_completions).
- Cubebox wiring: new `Conversation.parent_conversation_id` column,
  `POST /conversations/{id}/fork` endpoint, per-assistant-message UI
  button. Lives in the cubebox repo.
- CLI sugar (`cubepi fork`) — only if a real user asks.
- Backfill heuristic for legacy threads (best-effort run-boundary
  inference) — only if a real workload needs it.
