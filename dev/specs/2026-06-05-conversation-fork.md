# Conversation Fork — `Agent.fork()` + `Agent.fork_once()`

- **Date**: 2026-06-05
- **Status**: Draft
- **Branch / worktree**: `2026-06-05-conversation-fork` in `.worktrees/2026-06-05-conversation-fork`
- **Drives**: cubebox "copy this conversation from message N" UI button; future
  A/B exploration / reflection-runner side experiments.

## 1. Motivation

Two related needs are unaddressed today:

1. **Persistent fork.** Cubebox wants a per-assistant-message button that
   spawns a new conversation, pre-populated with all prior messages from a
   chosen point. The user keeps both conversations and can continue them
   independently (compare answers, try a different next question, save a
   branch before risky edits, etc.).
2. **One-shot off-thread prompt.** Application code wants to ask the model a
   follow-up question from the context of an existing thread without
   polluting that thread's persisted history. Reflection-runner-style
   probes, automated evaluation harnesses, scratch "what if I asked X
   instead?" queries.

The Postgres and MySQL checkpointer schemas reserved `parent_thread_id` and
`forked_at_seq` columns for this exact eventuality (see
`website/docs/migration/from-langgraph.md` and
`website/docs/guides/checkpointing/postgres.md`), but no API exists yet.
The Memory and SQLite backends have no fork hooks at all.

This spec adds the missing API across all four backends and exposes it on
`Agent` as `fork()` (persistent) and `fork_once()` (ephemeral one-shot).

## 2. Goals / Non-goals

**Goals**

- A `Checkpointer.fork()` operation that physically copies a prefix of a
  thread's messages under a caller-supplied new `thread_id`, records
  lineage, and is atomic.
- A `Checkpointer.snapshot()` operation that returns the same prefix as
  `list[Message]` without writing anything — the shared primitive both
  `fork()` and `fork_once()` build on.
- `Agent.fork()` — thin wrapper over `Checkpointer.fork()`.
- `Agent.fork_once()` — in-memory single-turn continuation from a snapshot
  prefix; not persisted; emits its own tracing span.
- Implementations across Memory, SQLite, Postgres, MySQL.
- Boundary validation that rejects cuts that would orphan a `tool_call`
  from its `tool_result`.
- User-facing docs page under `website/docs/guides/checkpointing/`.

**Non-goals**

- Copy-on-write / logical pointer storage (deliberately rejected — see §3.1).
- Forking subagent state, MCP session state, or external resources spun up
  during the parent run.
- Mutating the source thread (this is purely a read operation on the source).
- A `fork_into_agent()` convenience that hands back a ready-to-use `Agent`
  instance bound to the new thread. Caller decides what to do with the new
  `thread_id`; constructing the next `Agent` is one line of user code and
  keeping `Agent` state machinery out of `fork()` avoids confusing
  "which agent instance owns which thread" questions.
- Resuming a `fork_once()` session. By construction it is single-turn,
  in-memory, and discarded.
- Multi-turn ephemeral sessions. If they become a real need later they can
  ship as a separate `EphemeralAgent` handle returned from a future call;
  YAGNI for now.
- A `cubepi fork` CLI subcommand. Not needed by the cubebox driving use
  case; can be added later if it earns its keep.

## 3. Design

### 3.1 Storage semantics: physical copy

`fork()` physically copies the prefix `[0..message_count)` of source-thread
messages into the new thread, then records lineage metadata
(`parent_thread_id`, `forked_at_seq`) on the new thread row. The new
thread is fully independent — subsequent reads on either thread are
single-thread operations.

Considered and rejected: **logical pointer / copy-on-write** (store only
parent reference + new tail in the child). Reasons:

- All four backends are designed to keep a single thread's reads local.
  Postgres uses `HASH (thread_id)` partitioning on `cubepi_messages`;
  MySQL uses `KEY` partitioning by `thread_id`. COW reads would have to
  span parent and child partitions, defeating that invariant.
- The source thread is mutable in real cubepi usage: `agent.respond()`,
  HITL deny appends synthetic messages, future compaction may rewrite
  history. Under COW the child silently drifts when the parent changes;
  under physical copy the child is frozen at fork time.
- Deletion semantics: with physical copy the parent and child are
  independently deletable. Under COW, deleting the parent either breaks
  the child (FK violation) or wipes it (CASCADE) — neither is
  expressible cleanly across the four backends.
- The `parent_thread_id` + `forked_at_seq` columns retain value as pure
  lineage metadata (UI family tree, audit, debugging) under physical
  copy. They keep the cost of a few bytes per fork, not the cost of a
  recursive read.
- LangGraph's `copy_thread` uses physical copy for the same reasons.

Space cost (long conversations forked many times) is acknowledged. If a
real workload hits it we add a GC / compaction job later — that is much
cheaper to layer on physical copy than to retrofit COW.

### 3.2 Cut point: `message_count`, not `seq`

The user-facing cut is **`message_count`: include the first N messages**.
`message_count=None` copies everything currently in the thread.

`seq` is intentionally NOT exposed to callers:

- Cubepi's `Message` types (`UserMessage`, `AssistantMessage`,
  `ToolResultMessage`) carry no `seq` field. Only the
  Postgres/MySQL storage layer assigns seqs.
- Cubebox renders messages from `Checkpointer.load()` and identifies the
  click target by list index. "First N messages" lines up trivially with
  that.
- Memory and SQLite backends have no native seq column; `message_count`
  reduces to "len of the first N elements" there.

Storage-level `forked_at_seq` is still written for Postgres/MySQL
(it equals the seq of the last copied message — for the in-memory and
SQLite backends it equals `message_count` because seq == index+1 there).
It is metadata; callers do not pass or read it directly.

### 3.3 Boundary validation

Cubepi's message protocol requires every `AssistantMessage` that emits
`ToolCall` blocks to be followed (eventually) by `ToolResultMessage`s
for every call before the next assistant turn. Forking mid-tool-call
would leave the new thread in a state the provider rejects on the next
`prompt()` call ("`tool_use` block without matching `tool_result`").

`snapshot()` and `fork()` validate the **entire prefix**
`[0..message_count)`, not just the last message. The prefix is valid
iff all of the following hold:

1. Every `ToolCall.id` produced by an `AssistantMessage` in the prefix
   has exactly one `ToolResultMessage.tool_call_id` later in the
   prefix that matches it.
2. Every `ToolResultMessage.tool_call_id` in the prefix matches a
   `ToolCall.id` produced by some earlier `AssistantMessage` in the
   prefix.
3. Tool-result messages for the same producing assistant turn appear
   contiguously after that assistant turn (no other assistant turn
   interleaves).

Examples that MUST raise `ForkBoundaryError`:

- `[assistant(tc-1), user(...)]` — assistant emits `tc-1` but the
  prefix never carries its tool_result before another user turn.
- `[assistant(tc-1), tool_result(tc-1), assistant(tc-2)]` with
  `message_count=3` — the last assistant has an unresolved `tc-2`.
- `[tool_result(tc-99)]` — orphan tool_result; no producer.
- `[assistant(tc-1, tc-2), tool_result(tc-1)]` — partial close of a
  multi-call turn; `tc-2` is unresolved.
- `[assistant(tc-1), assistant(...)]` — two assistants in a row with
  `tc-1` unresolved between them.

The error payload lists the offending state (`unresolved_tool_call_ids`,
`orphan_tool_result_ids`) so the caller can re-pick a valid
`message_count`. `ForkBoundaryError` is raised before any write
happens.

The check runs in O(message_count) time and a single pass over the
prefix; it is performed by `snapshot()` and reused by `fork()`.

### 3.4 Atomicity and concurrency

`fork()` is atomic per backend:

- **Memory**: holds `asyncio.Lock` for the source thread (a single
  shared lock is acceptable for this backend's scale); copies under
  the lock. Memory backend is **single-process only by definition**;
  this is already documented for the existing methods and applies to
  fork unchanged. The fork doc page calls it out explicitly so users
  do not expect fork lineage to survive a process restart.
- **SQLite**: opens its transaction with `BEGIN IMMEDIATE` to take a
  RESERVED lock on the database for the duration of the fork. This
  blocks concurrent writers (`append`, `save_extra`,
  `save_pending_request`, other `fork`s) — including writers from
  other processes sharing the same DB file — until the fork commits
  or rolls back. Readers (`load`, `load_pending_request`, `snapshot`)
  continue under WAL. The existing `append()` does NOT currently use
  `BEGIN IMMEDIATE`; this spec promotes it to `BEGIN IMMEDIATE` as
  well so writer-vs-writer races are uniformly serialized. (The
  promotion is a strict subset of correctness — current behavior
  relies on aiosqlite's single-connection serialization within one
  process; the upgrade makes the contract explicit and
  cross-process-safe.)
- **Postgres**: single transaction. Inside it:
  1. `pg_advisory_xact_lock(hashtext($src_thread_id))` — the same
     per-thread advisory lock `append()` uses, but here taken on the
     SOURCE thread to fence racing appends to the source for the
     duration of the fork.
  2. `INSERT INTO cubepi_threads` for the new thread.
  3. `INSERT INTO cubepi_messages (...) SELECT ... FROM cubepi_messages
     WHERE thread_id=$src AND seq <= $cut_seq ORDER BY seq`.
  4. Commit (releases the advisory lock).

  `save_extra`, `save_pending_request`, and `compaction` on the source
  also take this same per-thread advisory lock, so they serialize
  against the fork.

- **MySQL**: single transaction with `SELECT ... FOR UPDATE` on the
  source thread row in `cubepi_threads` (MySQL has no advisory lock
  equivalent that fits the model). The existing `append()` already
  takes the same row lock; this spec confirms `save_extra`,
  `save_pending_request`, and compaction follow suit. Then the same
  thread-row INSERT + messages `INSERT…SELECT` as Postgres.

Concurrent forks from the same source (`fork(src=X, new=A)` and
`fork(src=X, new=B)`) serialize cleanly on the source-thread
lock/row-lock in every backend; the two new threads end up with
identical or differently-prefixed message sets depending on which
fork ran first relative to any concurrent append, and either ordering
is correct.

Error pre-checks: if the new thread already exists, `fork()` raises
`ThreadAlreadyExistsError` and writes nothing. If the source thread
does not exist, `fork()` raises `ThreadNotFoundError`. Both checks
happen inside the transaction so they see the same world the copy
will see.

### 3.5 What gets copied

| Field | Copied? | Notes |
|---|---|---|
| `messages` `[0..message_count)` | yes | physical copy; Postgres/MySQL preserve the source seq values for the copied range |
| `extra` | yes | deep copy of the source JSON object (`json.loads(json.dumps(extra))`) |
| `parent_thread_id` | written (new) | set to `src_thread_id` on new row |
| `forked_at_seq` | written (new, Postgres/MySQL only) | seq of the last copied message. NULL for Memory and SQLite — those backends have no per-message seq column and forging a value would invite false continuation logic. The column exists in the SQLite schema for parity but is always NULL there. |
| `extra['fork']` | written (new) when `metadata` is supplied | merge rule below |
| `pending_request` | **no** | new thread starts clean; HITL is run-state, not history |
| `run_id` | **no** | host-side run identifier; new thread has none |
| `created_at` / `updated_at` | new | server-default to fork time |

**`extra['fork']` merge rule.** The source's existing `extra` is
deep-copied first. Then, if the caller passes `metadata`, fork writes
`new_extra['fork'] = metadata` — unconditionally overwriting whatever
key was at `extra['fork']` in the source. Rationale: a fork's `fork`
key describes THIS fork, not the parent's fork ancestry. Lineage is
already recoverable from the `parent_thread_id` chain in the threads
table, so callers that want full ancestry walk the chain rather than
reading nested `fork` blobs. Callers who want to preserve the source's
`extra['fork']` under a different key are free to do so themselves via
a follow-up `save_extra()` call.

### 3.6 Tracing for `fork_once()`

`fork_once()` emits a span named `cubepi.agent.fork_once` that:

- **Inherits the active OTel parent context** if one is bound when
  `fork_once()` is called (e.g., the FastAPI request span, a
  cubebox stream span). This is standard OTel convention; "fork_once
  emits its own logical root" does NOT mean "detach from the
  surrounding trace_id". A `fork_once` invoked outside any active
  context becomes a true trace root.
- Does **not** attempt to attach to or replay spans from the source
  thread's prior runs. Those spans are completed and live in
  different traces; we only carry forward the source thread's
  identity via attributes.

Attributes on the span:

- `cubepi.fork.src_thread_id`
- `cubepi.fork.src_message_count`
- `cubepi.fork.src_seq` (Postgres/MySQL only — the storage seq of the
  last copied message; absent for Memory/SQLite for the same reason
  `forked_at_seq` is NULL there)
- The standard cubepi tracing attributes (`cubepi.model.id`, etc.)

The in-process child `Agent` it runs nests its agent / turn / tool
spans under the `cubepi.agent.fork_once` span as normal.

The persistent `fork()` does not need a special span; existing
checkpointer instrumentation (if any) covers it.

### 3.7 API

#### `cubepi.checkpointer.base`

```python
@runtime_checkable
class Checkpointer(Protocol):
    # existing: load, append, save_extra, save_pending_request, load_pending_request

    async def snapshot(
        self,
        thread_id: str,
        *,
        message_count: int | None = None,
    ) -> list[Message]:
        """Return messages [0..message_count) of `thread_id`.

        `message_count=None` returns all messages currently in the thread.
        Raises `ThreadNotFoundError`, `ForkBoundaryError`.
        """

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        message_count: int | None = None,
        metadata: JsonObject | None = None,
    ) -> None:
        """Atomically create `new_thread_id` with the first `message_count`
        messages of `src_thread_id`.

        Records `parent_thread_id=src_thread_id` and `forked_at_seq` on
        the new thread row. Copies `extra` deeply. Writes
        `extra['fork'] = metadata` when `metadata` is not None.

        Raises `ThreadNotFoundError`, `ThreadAlreadyExistsError`,
        `ForkBoundaryError`.
        """
```

#### `cubepi.checkpointer.exceptions`

Add:

```python
class ThreadNotFoundError(CheckpointerError): ...
class ThreadAlreadyExistsError(CheckpointerError): ...
class ForkBoundaryError(CheckpointerError):
    """message_count cuts across an unresolved tool_call/tool_result pair."""
    def __init__(self, message_count: int, unresolved_tool_call_ids: list[str]):
        ...
```

`CheckpointerError` is the existing base in `cubepi/checkpointer/exceptions.py`.

#### `cubepi.agent.agent`

```python
class Agent(Generic[TMessage]):

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        message_count: int | None = None,
        metadata: JsonObject | None = None,
    ) -> None:
        """Persistent fork. Requires `self.checkpointer`. Delegates to
        `self.checkpointer.fork(...)`. Does NOT mutate `self`.
        """

    async def fork_once(
        self,
        src_thread_id: str,
        message: str | Message | list[Message],
        *,
        message_count: int | None = None,
    ) -> ForkOnceResult:
        """One-shot continuation. Reads snapshot from `self.checkpointer`,
        constructs an ephemeral `Agent` with this agent's `model`,
        `tools`, `middleware`, `system_prompt`; runs one full turn
        (including any tool calls); returns the result. Never writes
        to the checkpointer.

        The `message` parameter accepts the same types as
        `Agent.prompt()` for consistency. The transient agent receives
        the snapshot as pre-seeded history, then `prompt(message)` runs
        the single turn.

        Raises `RuntimeError` if any inherited tool or middleware
        requires HITL (see §3.9.1).
        """
```

#### `cubepi.agent.types`

```python
@dataclass(frozen=True)
class ForkOnceResult:
    text: str
    messages: list[Message]   # new messages produced this turn only
    stop_reason: str
```

### 3.8 Per-backend implementation sketch

- **Memory** (`cubepi/checkpointer/memory.py`): add a `dict[str, dict]`
  of thread metadata (parent_thread_id, forked_at_seq, extra). `fork()`
  copies the first N messages from the source list under the new key.
  `snapshot()` slices the list. Boundary validation walks the prefix
  once.
- **SQLite** (`cubepi/checkpointer/sqlite.py`): a thread is currently
  one row keyed by `thread_id`. Schema needs an additive migration to
  add `parent_thread_id` (TEXT, NULL) and `forked_at_seq` (INTEGER,
  always NULL). The existing checkpointer initialises its tables via
  `CREATE TABLE IF NOT EXISTS`; the new columns are added at connect
  time using the existing pattern (`PRAGMA table_info` probe, then
  `ALTER TABLE … ADD COLUMN` if missing — exact form decided by the
  plan to match what the file already does for schema upgrades).
  `fork()` opens a transaction with `BEGIN IMMEDIATE` (RESERVED
  lock), re-serializes the prefix into a new row, writes the
  `parent_thread_id` reference, leaves `forked_at_seq` NULL, commits.
  `forked_at_seq` exists in the schema purely for parity with the SQL
  backends — SQLite stores the messages as a single serialized blob,
  there are no per-message seqs to record, and synthesising one would
  give callers a false invariant to depend on.
- **Postgres** (`cubepi/checkpointer/postgres/`): the columns already
  exist (schema version 3). `fork()` is a single transaction:
  - `INSERT INTO cubepi_threads (thread_id, parent_thread_id, forked_at_seq, extra, …) …`
  - `INSERT INTO cubepi_messages (thread_id, seq, role, metadata, payload)
     SELECT $new_thread_id, seq, role, metadata, payload
     FROM cubepi_messages
     WHERE thread_id=$src AND ($n IS NULL OR seq <= $cut_seq)
     ORDER BY seq`
  - Holds the per-thread advisory lock on `src_thread_id` so an
    in-flight `append()` cannot make the count drift mid-copy.
- **MySQL** (`cubepi/checkpointer/mysql/`): same idea, MySQL syntax.
  The `cubepi_messages` table is KEY-partitioned by `thread_id` and
  has no FK to `cubepi_threads`. Order is enforced by `ORDER BY seq`
  in the `INSERT…SELECT`. Uses the existing locking idiom.

Schema bump from v3 → v4 for Postgres/MySQL is **not** required —
the necessary columns are already there. SQLite needs a small
in-process migration (it has no formal schema_version table today;
the code already handles backfills on `CREATE TABLE … IF NOT EXISTS`,
the same pattern applies to the new columns).

### 3.9 `Agent.fork_once()` execution detail

1. `self._require_checkpointer()` — raises `RuntimeError` with a
   clear message when the agent has no checkpointer bound.
2. **HITL pre-flight check** (see §3.9.1) — raises before doing any
   work if the inherited toolset is HITL-capable.
3. `snapshot = await self.checkpointer.snapshot(src_thread_id, message_count=...)`
   — performs the full-prefix boundary validation in §3.3.
4. Build a transient `Agent` configured with this agent's `model`,
   `system_prompt`, `tools`, `middleware`, and `convert_to_llm`, with
   `checkpointer=None` and `thread_id=None`. Pre-seed its message
   history with `snapshot`.

   **This requires a new public Agent surface for seeding initial
   history.** The exact form (constructor arg `messages=...` vs. a
   dedicated `Agent.preload(messages)` method) is decided in the
   implementation plan, but it is in scope for THIS spec — `fork_once()`
   MUST NOT reach into private attributes of `Agent`. The plan SHOULD
   prefer a constructor arg because it keeps the agent's "ready to run"
   state machine single-phase; if the plan picks a separate method,
   document that the method must be called before any `prompt()`.

5. Start a `cubepi.agent.fork_once` span (inheriting the surrounding
   OTel context per §3.6). Capture the pre-seed length.
6. Cancellation is propagated: if the surrounding task is cancelled
   (or `asyncio.CancelledError` is raised), the transient agent's
   abort signal fires and the cancellation re-raises after the agent
   loop returns — same semantics as cancelling `Agent.prompt()`. The
   tracing span is closed in a `finally` block with the failure
   status set.
7. `await child.prompt(message)` — runs the full turn (any tool calls
   included).
8. Read final assistant text + the messages added after the pre-seed
   length from the child; close the span.
9. Return `ForkOnceResult(text, new_messages, stop_reason)`.

The transient `Agent` is local to the call frame and dropped on return.
No state leak to `self`.

#### 3.9.1 HITL is not supported inside `fork_once()`

Two reasons HITL cannot work in `fork_once()`:

1. **No persistence target.** HITL pending requests are written via
   `Checkpointer.save_pending_request(thread_id, ...)`. The transient
   agent has no checkpointer and no thread_id; there is nowhere to
   persist the pause.
2. **Worse: inherited HITL channels would write to the SOURCE thread.**
   A real cubepi host typically constructs a
   `CheckpointedChannel(checkpointer=cp, thread_id=conversation_id, …)`
   and binds it to `ask_user_tool(channel)`. The channel object holds
   the source `thread_id` by reference. Reusing such a tool inside
   `fork_once()` would persist the fork's pending HITL request to the
   SOURCE conversation, contaminating it. Silent failure mode.

`fork_once()` therefore pre-checks `self.tools` and `self.middleware`
for HITL involvement. Detection rule:

- Any tool whose `name` is `ask_user` (cubepi's built-in HITL tool name).
- Any middleware that is an instance of `cubepi.hitl.middleware.HitlMiddleware`
  (or any subclass).
- Any tool whose `name` matches a configurable host-supplied set —
  the spec defers the exact extension hook to the plan.

If any match, `fork_once()` raises:

```
RuntimeError(
    "fork_once() does not support HITL. Found HITL-bearing tools/"
    "middleware: <names>. Construct a different Agent without these "
    "for ephemeral probes."
)
```

The error is raised BEFORE the snapshot is read. The
recommended pattern for callers that need probes with most of the
host's tools but no HITL is to build a second `Agent` with a filtered
toolset and call `fork_once()` on that one.

This is documented in the user guide page as a known limitation.
Lifting it (e.g., supporting in-memory HITL via a special transient
channel) is explicit follow-up scope.

### 3.10 Error semantics summary

| Situation | Raised |
|---|---|
| `self.checkpointer is None` (fork or fork_once) | `RuntimeError("fork requires a checkpointer")` |
| `fork_once()` finds HITL-bearing tool/middleware (§3.9.1) | `RuntimeError("fork_once() does not support HITL: <names>")` |
| `src_thread_id` does not exist | `ThreadNotFoundError(src_thread_id)` |
| `new_thread_id` already exists (fork) | `ThreadAlreadyExistsError(new_thread_id)` |
| `message_count < 0` | `ValueError` |
| `message_count > len(messages)` | `ValueError` (caller asked for more than exists) |
| `message_count = 0` | **valid**: clones an empty starter thread. Boundary check passes vacuously. For Postgres/MySQL, `forked_at_seq IS NULL`. The new thread row is created with no messages and acts as if just-created with a `parent_thread_id` reference. |
| `message_count = None` | normalized to `len(messages_in_src)` — copy everything currently present |
| cut violates any §3.3 invariant (unresolved tool_call, orphan tool_result, partial multi-call close, two assistants in a row with unresolved calls) | `ForkBoundaryError(message_count, unresolved_tool_call_ids=[…], orphan_tool_result_ids=[…])` |
| `Agent.fork_once()` child run errors | propagates (same surface as `Agent.prompt()`) |
| `Agent.fork_once()` is cancelled mid-turn | `asyncio.CancelledError` re-raises after transient agent abort completes |

## 4. Migration / Compatibility

- **Protocol change**: `Checkpointer.snapshot` and `Checkpointer.fork` are
  new methods on the `runtime_checkable` Protocol. Existing
  user-implemented checkpointers will keep type-checking (Protocol
  membership is structural; missing methods only matter when called).
  Documenting the new optional surface in the checkpointer guide is
  sufficient.
- **Storage**: Postgres / MySQL — no migration. SQLite — additive
  columns, in-process backfill (`ALTER TABLE … ADD COLUMN`). Memory —
  N/A.
- **No public API changes** to existing methods. No deprecations.

## 5. Testing

- **Unit / per-backend** (Memory + SQLite in-process; Postgres
  against the bundled docker fixture; MySQL against the live test
  server documented at `reference_mysql_test_server`):
  - fork all messages → new thread reads back identically
  - fork prefix → new thread holds prefix, source still holds full
  - fork preserves `extra`; sets `parent_thread_id`
  - `forked_at_seq` assertions: equals last copied seq for
    Postgres/MySQL; IS NULL for Memory/SQLite
  - fork copies `extra['fork']` from `metadata` arg (and overwrites
    a pre-existing `extra['fork']` on the source — explicit test for
    the §3.5 merge rule)
  - fork does NOT copy `pending_request`: verified via
    `load_pending_request(new_thread_id) is None`
  - fork does NOT copy `run_id`
  - `ThreadAlreadyExistsError` on collision; nothing written
    (no partial thread row, no orphan messages)
  - `ThreadNotFoundError` on bad source
  - `ValueError` for `message_count < 0` and `message_count > len`
  - `message_count = 0` succeeds, produces an empty new thread with
    `parent_thread_id` set
  - `message_count = None` copies all current messages
  - `ForkBoundaryError` for each illegal-prefix case enumerated in
    §3.3: unresolved tool_call at end, orphan tool_result, partial
    multi-call close, two assistants in a row with unresolved calls.
    The error payload lists the offending ids.
  - source thread unaffected by fork (independence test —
    snapshot/load source before and after, byte-equal)
  - Postgres/MySQL only: subsequent `append()` to the new thread
    continues seq numbering from `forked_at_seq + 1`
  - concurrent fork + append on source serializes correctly (no
    half-copied state) — exercise with two concurrent tasks
  - fork-of-fork: fork A → B; fork B → C. C's `parent_thread_id`
    points to B (not A). Walking the chain via `parent_thread_id`
    reaches A.
  - fork after `respond()`-injected synthetic deny: the synthetic
    messages are in the snapshot and survive the fork
  - SQLite cross-connection: open two separate `SQLiteCheckpointer`
    instances against the same DB file, exercise `fork()` from one
    while `append()` runs from the other; `BEGIN IMMEDIATE`
    serializes the two writers correctly

- **`Agent.fork_once()` (FauxProvider)**:
  - simple text-only follow-up returns expected final text
  - turn with tool calls completes fully, returns new messages
  - source thread (and its checkpointer) is byte-for-byte unchanged
  - raises `RuntimeError` when no checkpointer bound
  - raises `RuntimeError` when an `ask_user` tool is in
    `self.tools` (HITL pre-flight, §3.9.1)
  - raises `RuntimeError` when `HitlMiddleware` is in
    `self.middleware`
  - cancellation: `asyncio.wait_for(agent.fork_once(...), timeout=…)`
    raises `TimeoutError` and the transient agent's abort signal
    fires (verified via FauxProvider abort hook)
  - tracing span: emits `cubepi.agent.fork_once` with the documented
    attributes. When called inside an existing span context, the
    fork_once span is a child of the surrounding span (parent
    propagation test). When called outside any span, it is a trace
    root.
  - The existing tracing test infrastructure
    (`tests/tracing/conftest.py` in-memory exporter, if present) is
    reused; if no such infrastructure exists, the plan stage
    introduces a minimal `InMemorySpanExporter` fixture as a
    prerequisite.

- **`Agent.fork()` (FauxProvider)**:
  - happy path returns None; checkpointer state is correct
  - error pass-through (`ThreadNotFoundError`, `ThreadAlreadyExistsError`,
    `ForkBoundaryError`, `RuntimeError` for missing checkpointer)
  - does NOT mutate `self.thread_id`

## 6. Open questions

None remaining — answers locked in during brainstorm:

- Storage model → physical copy
- Cut parameter → `message_count`
- Caller supplies `new_thread_id` (required, not auto-generated)
- Boundary on unresolved tool_calls → raise `ForkBoundaryError`
- `extra` copied; `pending_request` / `run_id` not copied
- `metadata` arg merged into `extra['fork']`
- Spec scope → both `fork()` and `fork_once()` together
- Both methods live on `Agent` (thin wrappers over checkpointer
  primitives for `fork`; runtime work for `fork_once`)

## 7. Out of this PR (follow-ups)

- Cubebox-side wiring: new `Conversation.parent_conversation_id` column,
  `POST /conversations/{id}/fork` endpoint, per-assistant-message UI
  button. Lives in the cubebox repo.
- CLI sugar (`cubepi fork`) — only if a real user asks.
- GC / size cap for heavily forked thread trees — only if a real
  workload hits the space cost.
