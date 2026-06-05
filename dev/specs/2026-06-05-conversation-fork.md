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

`snapshot()` and `fork()` both validate the cut: if message
`message_count - 1` is an `AssistantMessage` with unresolved
`ToolCall`s (i.e. one of its tool_call_ids has no corresponding
`ToolResultMessage` in the prefix), they raise `ForkBoundaryError`.
The error message lists the unresolved `tool_call_id`s so the caller
can re-pick a later message_count.

### 3.4 Atomicity

`fork()` is atomic per backend:

- **Memory**: holds `asyncio.Lock`; copies under the lock.
- **SQLite**: single `BEGIN…COMMIT`.
- **Postgres**: single transaction; thread row INSERT + messages
  `INSERT…SELECT WHERE thread_id=$1 AND seq <= $2`. Per-thread advisory
  lock on the source thread (the same lock `append()` already uses for
  monotonic seq allocation) prevents racing appends from being included
  half-and-half.
- **MySQL**: single transaction; thread row INSERT + `INSERT…SELECT`.
  Uses the same locking discipline already in `append()`.

If the new thread already exists, `fork()` raises
`ThreadAlreadyExistsError` and writes nothing.

If the source thread does not exist, `fork()` raises `ThreadNotFoundError`.

### 3.5 What gets copied

| Field | Copied? | Notes |
|---|---|---|
| `messages` `[0..message_count)` | yes | physical copy; new thread's seqs equal the copied parent seqs |
| `extra` | yes | deep copy of the source JSON object |
| `parent_thread_id` | written (new) | set to `src_thread_id` on new row |
| `forked_at_seq` | written (new) | the seq of message `message_count - 1` (Postgres/MySQL); equals `message_count` for Memory/SQLite |
| `extra['fork']` | written (new) | `= metadata` argument when caller supplies it |
| `pending_request` | **no** | new thread starts clean; HITL is run-state, not history |
| `run_id` | **no** | host-side run identifier; new thread has none |
| `created_at` / `updated_at` | new | server-default to fork time |

### 3.6 Tracing for `fork_once()`

`fork_once()` runs an in-memory `Agent` instance that is never persisted.
It emits its own root span (it does **not** nest under the source
thread's last run span — those spans are completed and unrelated):

- Span name: `cubepi.agent.fork_once`
- Attributes:
  - `cubepi.fork.src_thread_id`
  - `cubepi.fork.src_message_count`
  - `cubepi.fork.src_seq` (the storage seq of the last copied message,
    when the source backend has one)
  - `cubepi.model.id`, etc. — the standard cubepi tracing attributes
- The in-process child `Agent` it runs subscribes to the standard
  tracing middleware if the host has one bound; child events nest under
  this span as normal.
- The persistent `fork()` does not need a special span; existing
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
        prompt: str | list[Content],
        *,
        message_count: int | None = None,
    ) -> ForkOnceResult:
        """One-shot continuation. Reads snapshot from `self.checkpointer`,
        constructs an ephemeral `Agent` with this agent's `model`,
        `tools`, `middleware`, `system_prompt`; runs one full turn
        (including any tool calls); returns the result. Never writes
        to the checkpointer.
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
  one row keyed by `thread_id`. Schema needs a small migration to add
  `parent_thread_id`, `forked_at_seq` columns (NULL for existing
  rows). `fork()` re-serializes the prefix into a new row.
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

1. `self._require_checkpointer()` — raises with a clear message when
   the agent has no checkpointer bound.
2. `snapshot = await self.checkpointer.snapshot(src_thread_id, message_count=...)`
   — performs boundary validation.
3. Build a transient `Agent` configured with this agent's `model`,
   `system_prompt`, `tools`, `middleware`, and `convert_to_llm`, with
   no `checkpointer` and no `thread_id`. Pre-seed its message history
   with `snapshot`. The exact pre-seed hook (constructor arg vs.
   dedicated method) is left to the implementation plan — the
   constraint is that `fork_once()` must not reach into private
   attributes of `Agent`.
4. Start a `cubepi.agent.fork_once` root span. Attach a one-shot
   listener that captures the child's new messages (everything after
   the pre-seed length).
5. `await child.prompt(prompt)` — runs the full turn (any tool calls
   included).
6. Read final assistant text + new messages from the child; close the
   span.
7. Return `ForkOnceResult(text, new_messages, stop_reason)`.

The transient `Agent` is local to the call frame and dropped on return.
No state leak to `self`.

### 3.10 Error semantics summary

| Situation | Raised |
|---|---|
| `self.checkpointer is None` (fork or fork_once) | `RuntimeError("fork requires a checkpointer")` |
| `src_thread_id` does not exist | `ThreadNotFoundError(src_thread_id)` |
| `new_thread_id` already exists (fork) | `ThreadAlreadyExistsError(new_thread_id)` |
| `message_count < 0` | `ValueError` |
| `message_count > len(messages)` | `ValueError` (caller asked for more than exists) |
| cut would orphan a tool_call | `ForkBoundaryError(message_count, unresolved_ids)` |
| `Agent.fork_once()` child run errors | propagates (same surface as `Agent.prompt()`) |

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
  - fork preserves `extra`; sets `parent_thread_id` + `forked_at_seq`
  - fork copies `extra['fork']` from `metadata` arg
  - fork does NOT copy `pending_request` / `run_id`
  - `ThreadAlreadyExistsError` on collision; nothing written
  - `ThreadNotFoundError` on bad source
  - `ForkBoundaryError` when cutting mid-tool-call (and the error
    payload lists unresolved tool_call_ids)
  - source thread unaffected by fork (independence test)
  - subsequent `append()` to the new thread continues seq numbering
    from `forked_at_seq + 1`
  - concurrent fork + append on source serializes correctly (no
    half-copied state)

- **`Agent.fork_once()` (FauxProvider)**:
  - simple text-only follow-up returns expected final text
  - turn with tool calls completes fully, returns new messages
  - source thread (and its checkpointer) is byte-for-byte unchanged
  - raises when no checkpointer bound
  - emits a `cubepi.agent.fork_once` span with the documented
    attributes (using the in-memory tracing exporter helper)

- **`Agent.fork()` (FauxProvider)**:
  - happy path returns None; checkpointer state is correct
  - error pass-through (`ThreadNotFoundError`, etc.)

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
