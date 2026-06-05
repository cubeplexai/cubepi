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
- Backfilling run identity into pre-existing messages. Legacy messages
  remain `run_id=NULL` and are not forkable / not deletable — see §3.6.4.
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

The **only** fork handle is `after_run_id: str`. The new thread contains
exactly the messages of every completed run on the source up through and
including `after_run_id` (in source seq order).

Why a run is the right unit:

- A run is the atomic transaction the user thinks in ("I asked X, agent
  did its thing, gave me an answer"). Cubebox's UI button maps directly
  to "fork after this exchange".
- A run boundary is by construction a clean cut. Inside a run there may
  be unresolved `tool_use` blocks awaiting `tool_result`s; once the run
  is marked complete (terminal stop_reason, no pending HITL), no
  unresolved tool calls remain. So **the v1 spec's `ForkBoundaryError`
  / mid-tool-call invariants vanish structurally**: there is no API to
  ask for a cut mid-run, so no boundary check is needed.
- It generalizes to the future `delete_run()` cleanly:
  `DELETE FROM cubepi_messages WHERE thread_id=? AND run_id=?` is a
  one-line operation backed by the same `run_id` column the fork uses
  for lookup.

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
| `cubepi_messages` rows with `seq <= last_seq_of(after_run_id)` | yes | physical copy; PG/MySQL preserve source `seq` values for the copied range. SQLite copies the JSON payloads under fresh global `id`s (its `messages.id` is a global auto-increment, not a per-thread seq). Memory copies in-list order. Each copied row keeps its original `run_id` value. |
| `cubepi_run_completions` rows for the copied runs | yes | so the new thread can be further forked / deleted by run |
| `extra` | yes | deep copy of the source JSON object |
| `parent_thread_id` | written (new) | set to `src_thread_id` on the new thread row |
| `forked_at_seq` | written (PG/MySQL only) | the `seq` of the last message of `after_run_id`. Memory and SQLite store no equivalent — those backends have no per-message seq column and lineage is recoverable from `parent_thread_id` + `cubepi_run_completions` alone. |
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

If the caller supplies a `run_id` that already has a completion marker
on the source thread, `prompt()` raises `RunAlreadyCompletedError` —
runs are append-only; you cannot continue or re-run a completed run.
(Use a new `run_id` for a new exchange.)

#### 3.6.2 Completion marker — when written

The marker `cubepi_run_completions(thread_id, run_id, completed_at)`
is written **inside the same transaction as the final `append()` of
the run** — atomic with the last message. The agent loop signals
"this is the last append of the run" via a flag on the checkpointer
call (or via a dedicated `Checkpointer.complete_run()` call made in
the same lock window — exact form decided in the implementation plan,
but the requirement is atomicity).

Trigger conditions for writing the marker:

- The agent loop exits with a terminal stop_reason (`end_turn`,
  `tool_use_completed` followed by `end_turn`, etc. — any
  non-suspended terminal state)
- AND no pending HITL request remains for this run

If `prompt()` returns because of:

- HITL pause (pending_request set) → marker NOT written; the run is
  resumed later by `respond()`.
- Exception / abort / cancellation → marker NOT written; the run is
  abandoned. Its messages remain on the thread under the same
  `run_id`, but `fork(after_run_id=X)` raises `RunNotCompletedError`,
  and the future `delete_run(X)` can clean them up.

#### 3.6.3 HITL pause / resume continuity

A run that pauses for HITL keeps its `run_id` across the suspension.

- `Agent.prompt(message, run_id=R)` runs partway, hits HITL → writes
  `pending_request` with `run_id=R` (existing schema v3 mechanism), no
  completion marker.
- `Agent.respond(question_id, answer)` recovers `run_id=R` from
  `pending_request` (it's already there), continues the same run,
  writes the completion marker when the resumed loop terminates
  cleanly.

`Agent.respond()` signature does **not** change. The run_id is
sourced from `pending_request`.

#### 3.6.4 Legacy data

Messages persisted before this spec carry `run_id = NULL`. No
completion markers exist for them. Consequence:

- `fork(src_thread_id, …, after_run_id=X)` on a legacy thread raises
  `RunNotCompletedError` for every `X` because no marker exists.
- Future `delete_run(thread_id, run_id)` will likewise have nothing
  to target.
- Legacy threads remain readable (`load()` works); only the
  by-run operations are blocked.

No backfill is provided. (We can not reliably reconstruct historical
run boundaries from messages alone — multi-step tool_use sequences
look like multiple runs.) Users who need to fork a legacy
conversation can manually start a new conversation; the loss is
limited to "no clone shortcut on threads that predate the upgrade".

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

    async def complete_run(
        self,
        thread_id: str,
        run_id: str,
        last_messages: list[Message],
    ) -> None:
        """Atomically append `last_messages` to the thread AND write the
        `cubepi_run_completions` row for `(thread_id, run_id)`.

        Called by the agent loop on terminal-state exit only. Replaces
        the final `append()` of the run — the agent loop normally uses
        `append()` for intermediate messages of a run and
        `complete_run()` for the last batch. Idempotent: a second call
        with the same `(thread_id, run_id)` is an error
        (`RunAlreadyCompletedError`) since runs are append-only.
        """
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
2. HITL pre-flight (§3.8.1) — `RuntimeError` if inherited tools or
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

#### 3.8.1 HITL is not supported inside `fork_once()`

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
2. `SELECT MAX(seq) AS last_seq FROM cubepi_messages WHERE thread_id =
   $src AND seq <= (SELECT MAX(seq) FROM cubepi_messages WHERE
   thread_id = $src AND run_id = $after_run_id)`
   AND `EXISTS (SELECT 1 FROM cubepi_run_completions WHERE thread_id =
   $src AND run_id = $after_run_id)` → if no row,
   `RunNotCompletedError`.
3. `INSERT INTO cubepi_threads (thread_id, parent_thread_id,
   forked_at_seq, extra, …) VALUES ($new, $src, $last_seq,
   $merged_extra, …)` — `ThreadAlreadyExistsError` on PK violation.
4. `INSERT INTO cubepi_messages (thread_id, seq, role, run_id,
   metadata, payload) SELECT $new, seq, role, run_id, metadata, payload
   FROM cubepi_messages WHERE thread_id = $src AND seq <= $last_seq
   ORDER BY seq`.
5. `INSERT INTO cubepi_run_completions (thread_id, run_id,
   completed_at) SELECT $new, run_id, completed_at FROM
   cubepi_run_completions WHERE thread_id = $src AND run_id IN
   (SELECT DISTINCT run_id FROM cubepi_messages WHERE thread_id = $src
   AND seq <= $last_seq)`.
6. Commit.

`complete_run()` in one transaction:

- Same advisory lock / `FOR UPDATE` on `thread_id`.
- `INSERT INTO cubepi_messages …` for `last_messages` (allocating seqs).
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
4. `INSERT INTO messages (thread_id, run_id, message_json) SELECT
   $new, run_id, message_json FROM messages WHERE thread_id = $src
   AND id <= (SELECT MAX(id) FROM messages WHERE thread_id = $src AND
   run_id = $after_run_id) ORDER BY id`. New rows get fresh global
   `id`s (the `messages.id` column is a global auto-increment;
   identity is not preserved across the copy, but per-thread row
   order is).
5. `INSERT INTO run_completions (thread_id, run_id, completed_at)
   SELECT $new, run_id, completed_at FROM run_completions WHERE
   thread_id = $src AND run_id IN (SELECT DISTINCT run_id FROM
   messages WHERE thread_id = $src AND id <= $cut_id)`.
6. `INSERT INTO thread_extra (thread_id, extra_json, parent_thread_id)
   VALUES ($new, $merged_extra_json, $src)`.
7. Commit.

`complete_run()`:

- `BEGIN IMMEDIATE`.
- `INSERT INTO messages …` for each `last_messages` element (each
  carrying its `run_id`).
- `INSERT INTO run_completions (thread_id, run_id) VALUES (?, ?)` →
  `RunAlreadyCompletedError` if `(thread_id, run_id)` PK conflict.
- Commit.

`append()` is also wrapped in `BEGIN IMMEDIATE` (uniform writer
discipline; see §3.3). `PRAGMA busy_timeout = 5000` set at connect.

No `forked_at_seq` column added — SQLite has no per-thread seq.

#### Memory

`MemoryCheckpointer` today is `dict[str, CheckpointData]`. This spec:

- Extends `CheckpointData` with `parent_thread_id: str | None = None`.
- Adds an internal `dict[str, set[str]]` mapping
  `thread_id -> set of completed run_ids`.
- Adds an `asyncio.Lock` for fork (existing single-statement methods
  do not strictly need one, but fork is multi-step).
- `Message.run_id` is just a field — Memory persists the whole
  Message via `model_dump`, so no extra storage scaffolding.

`fork()` under the lock:

1. Source-exists check, new-thread-does-not-exist check.
2. Look up the completed run_ids set; if `after_run_id` not there →
   `RunNotCompletedError`.
3. Walk `src.messages`, find the index of the last message with
   `run_id == after_run_id`. Take prefix `[0..idx+1)`. Deep-copy
   each message (`model_copy(deep=True)`).
4. Compute the subset of `src`'s completed run_ids that appear in the
   prefix; carry that set under the new thread_id.
5. Deep-copy `src.extra`; merge `extra['fork']=metadata` per §3.4.
6. Store `CheckpointData(messages=…, extra=…,
   parent_thread_id=src_thread_id)` under `new_thread_id`.

`complete_run()`:

- Append `last_messages` to `src.messages`.
- Add `run_id` to the completed-runs set; if already present →
  `RunAlreadyCompletedError`.

No `forked_at_seq` field — Memory has no seq.

### 3.10 Error semantics summary

| Situation | Raised |
|---|---|
| `self.checkpointer is None` (fork or fork_once) | `RuntimeError("fork requires a checkpointer")` |
| `fork_once()` finds HITL-bearing tool/middleware (§3.8.1) | `RuntimeError("fork_once() does not support HITL: <names>")` |
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
  `Checkpointer.complete_run` are new methods on the
  `runtime_checkable` Protocol. Existing user-implemented checkpointers
  keep type-checking; they only fail when the new methods are actually
  called.
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
  - concurrent fork + complete_run on source serialize correctly
  - SQLite cross-connection: two `SQLiteCheckpointer` instances on
    the same DB file, one `fork()` and one `complete_run()` from
    different processes serialize via `BEGIN IMMEDIATE`
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
- Run completion atomicity → same transaction as final append
  (`complete_run()` Protocol method)
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
