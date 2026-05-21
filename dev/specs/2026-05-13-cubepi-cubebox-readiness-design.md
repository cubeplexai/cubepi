# cubepi cubebox-readiness — Design

Date: 2026-05-13
Status: Design
Companion spec: `cubebox/docs/superpowers/specs/2026-05-13-cubepi-main-agent-migration-design.md` (Spec B)

## Why this spec exists

cubebox plans to drop LangGraph entirely and run on cubepi (Spec B).
cubepi v0.2.0 lacks several pieces that cubebox needs. This spec
collects those gaps as upstream cubepi work so cubebox's migration
(Spec B) has a clean dependency surface. None of the work here is
cubebox-specific in shape — every addition is designed as general
public API for any cubepi user.

cubebox depends on cubepi via local path dependency (`uv add --editable
/home/chris/cubepi`) throughout development. No cubepi releases are
required for Spec B to start consuming these changes.

## Non-goals

- LangGraph compatibility shims. cubepi stays pure.
- Behavioral changes to existing cubepi APIs. All additions are new
  surface; existing users see no behavior change.
- Auto-migration tooling from langgraph checkpoint format. cubebox has
  no released data; greenfield write.
- cubepi.Responses API integration into cubebox — already shipped, this
  spec doesn't touch `OpenAIResponsesProvider`.

## Deliverables overview

| # | Item | Module / surface |
|---|---|---|
| D1 | Postgres checkpointer | `cubepi.checkpointer.postgres` (new) |
| D2 | MCP adapter (HTTP + stdio) | `cubepi.mcp` (new) |
| D3 | AnthropicProvider configurable cache marker policy | `cubepi.providers.anthropic` |
| D4 | OpenAIProvider OSS reasoning extraction + payload quirks | `cubepi.providers.openai` |
| D5 | `Message.metadata` field on all three Message types | `cubepi.providers.base` |
| D6 | `AgentContext.extra` mutable dict | `cubepi.agent.types` |
| D7 | `transform_system_prompt` middleware hook | `cubepi.middleware.base` |
| D8 | `after_model_response` middleware hook + `TurnAction` | `cubepi.middleware.base` |
| D9 | Packaging: `[postgres]`, `[mcp]` extras | `pyproject.toml` |

All deliverables are backward-compatible. Existing cubepi users get
no behavior change unless they opt in.

---

## D1 — Postgres checkpointer

### Scope

`cubepi.checkpointer.postgres.PostgresCheckpointer` implementing the
existing `Checkpointer` protocol (`load` / `append` / `save_extra`)
against PostgreSQL.

### Schema

cubepi defines table classes on a private `MetaData` instance (NOT on
`SQLModel.metadata`) so downstreams using any ORM (SQLModel, plain
SQLAlchemy, none) can compose it into their alembic.

```python
# cubepi/checkpointer/postgres/models.py
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime

EXPECTED_SCHEMA_VERSION = 1
PARTITION_COUNT = 64
cubepi_metadata = sa.MetaData()

class CubepiBase(DeclarativeBase):
    metadata = cubepi_metadata

class CubepiThread(CubepiBase):
    __tablename__ = "cubepi_threads"
    thread_id:        Mapped[str]          = mapped_column(sa.Text, primary_key=True)
    parent_thread_id: Mapped[str | None]   = mapped_column(
        sa.Text, sa.ForeignKey("cubepi_threads.thread_id"), nullable=True)
    forked_at_seq:    Mapped[int | None]   = mapped_column(sa.BigInteger, nullable=True)
    extra:            Mapped[dict]         = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"))
    created_at:       Mapped[datetime]     = mapped_column(
        sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()"))
    updated_at:       Mapped[datetime]     = mapped_column(
        sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()"))

class CubepiMessage(CubepiBase):
    __tablename__ = "cubepi_messages"
    __table_args__ = (
        sa.Index(
            "ix_cubepi_messages_metadata_gin", "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
        ),
        {"postgresql_partition_by": "HASH (thread_id)"},
    )

    thread_id:  Mapped[str]   = mapped_column(
        sa.Text,
        sa.ForeignKey("cubepi_threads.thread_id", ondelete="CASCADE"),
        primary_key=True,
    )
    seq:        Mapped[int]   = mapped_column(sa.BigInteger, primary_key=True)
    role:       Mapped[str]   = mapped_column(sa.Text, nullable=False)
    metadata:   Mapped[dict]  = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"))
    payload:    Mapped[bytes] = mapped_column(sa.LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()"))

class CubepiSchemaVersion(CubepiBase):
    __tablename__ = "cubepi_schema_version"
    version: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
```

### Why these shape choices

| Decision | Rationale |
|---|---|
| Private `cubepi_metadata` (not `SQLModel.metadata`) | Downstream ORM-agnostic. SQLAlchemy `target_metadata` accepts a list; SQLModel users add it alongside their own metadata. |
| Composite PK `(thread_id, seq)` | Append-only writes hit one B-tree path; `WHERE thread_id ORDER BY seq` is a PK range scan = sequential I/O. |
| `seq BIGINT` monotonic per-thread | Simpler than langgraph's `version: "123.random"` string format. cubepi has no snapshot semantics → no version dedup needed. |
| `payload BYTEA + msgpack` | ~30% smaller than JSONB serialization for messages; faster ser/deser; messages are opaque blobs. |
| `metadata JSONB` (separate column) | Queryable via `@>` for role-independent filters (cost, tool_call_id, source). Carrying snapshot data for memory middleware (Spec B). |
| GIN(metadata) `jsonb_path_ops` | Efficient `@>` queries; `jsonb_path_ops` is ~30% smaller than default GIN. |
| No separate blob dedup table | cubepi has no snapshot model → no value to dedup. langgraph's `checkpoint_blobs` is a fix for snapshot bloat we don't have. |
| No `checkpoint_ns` | cubepi doesn't model namespaces. Subagents use distinct `thread_id`s. |
| HASH(thread_id) PARTITION BY, 64 partitions | All hot queries filter by `thread_id` → partition pruning O(1). 64 partitions cover ~6B rows comfortably; PG hash partition count is fixed (changing requires full rewrite) so over-provision now. |
| `cubepi_threads` not partitioned | Metadata-table sized (one row per conversation); orders of magnitude smaller than messages. |
| FK `cubepi_messages.thread_id → cubepi_threads` with `ON DELETE CASCADE` | Integrity + simple cleanup. Lazy `INSERT ... ON CONFLICT DO NOTHING` on first append per thread. |
| `parent_thread_id` + `forked_at_seq` nullable | Schema-ready for future fork API; columns sit empty until cubepi protocol exposes fork. |

### Why partition from v1

PostgreSQL hash partition count is **static** — changing modulus
requires full data rewrite (`INSERT ... SELECT`) or logical
replication. Both require maintenance windows or significant ops
work. The cost of starting with 64 partitions is negligible
(~200-400MB metadata, no measurable query impact under PG 14+
pruning). The expected reward (never rehashing) is large. Over-
provision now.

### Schema version enforcement

cubepi runtime checks DB schema matches `EXPECTED_SCHEMA_VERSION` on
first connect:

```python
async def _verify_schema(self, conn) -> None:
    row = await conn.fetchrow("SELECT version FROM cubepi_schema_version LIMIT 1")
    if row is None:
        raise CubepiSchemaUninitialized(
            "cubepi tables not found. Run host application's alembic upgrade."
        )
    if row["version"] != EXPECTED_SCHEMA_VERSION:
        raise CubepiSchemaMismatch(
            expected=EXPECTED_SCHEMA_VERSION,
            actual=row["version"],
            hint="cubepi was upgraded but host alembic is behind. "
                 "Generate a new revision and apply.",
        )
```

When cubepi changes schema → bump `EXPECTED_SCHEMA_VERSION` → host
application gets a startup-time hard failure if their alembic is
behind. Downstream is forced to acknowledge the schema change.

### Alembic helpers

cubepi exports two SQL helpers for host alembic migrations:

```python
# cubepi/checkpointer/postgres/alembic_helpers.py
from .models import EXPECTED_SCHEMA_VERSION, PARTITION_COUNT

def create_message_partitions_op() -> str:
    """SQL DDL creating all 64 child partitions of cubepi_messages.
    Call inside an alembic upgrade() after the parent table is created."""
    return "\n".join(
        f"CREATE TABLE cubepi_messages_p{i:02d} "
        f"PARTITION OF cubepi_messages "
        f"FOR VALUES WITH (modulus {PARTITION_COUNT}, remainder {i});"
        for i in range(PARTITION_COUNT)
    )

def write_schema_version_op() -> str:
    """SQL to record the current schema version. Call inside upgrade()."""
    return (
        f"INSERT INTO cubepi_schema_version (version) "
        f"VALUES ({EXPECTED_SCHEMA_VERSION}) ON CONFLICT DO NOTHING;"
    )
```

Host application alembic env.py imports the metadata to enable
autogenerate:

```python
from cubepi.checkpointer.postgres.models import cubepi_metadata
target_metadata = [SQLModel.metadata, cubepi_metadata]
```

### Append path (seq allocation)

Concurrent `append()` calls to the same `thread_id` must produce
strictly monotonic `seq`. Strategy: per-thread advisory lock for the
duration of the append transaction.

```python
async def append(self, thread_id: str, messages: list[Message]) -> None:
    async with self._pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext($1))",
                thread_id,
            )
            await conn.execute(
                "INSERT INTO cubepi_threads (thread_id) "
                "VALUES ($1) ON CONFLICT DO NOTHING",
                thread_id,
            )
            last_seq = await conn.fetchval(
                "SELECT COALESCE(MAX(seq), 0) FROM cubepi_messages "
                "WHERE thread_id = $1",
                thread_id,
            )
            rows = [
                (thread_id, last_seq + i + 1, m.role, m.metadata,
                 msgpack.packb(m.model_dump(mode="json")))
                for i, m in enumerate(messages)
            ]
            await conn.executemany(
                "INSERT INTO cubepi_messages "
                "(thread_id, seq, role, metadata, payload) "
                "VALUES ($1, $2, $3, $4, $5)",
                rows,
            )
```

Advisory lock scope is the transaction; released automatically on
commit/rollback. No background cleanup needed.

### Load path

```sql
SELECT seq, role, metadata, payload
FROM cubepi_messages
WHERE thread_id = $1
ORDER BY seq;
```

Single query, PK range scan + partition pruning to a single
`cubepi_messages_pNN` child. `payload` deserialized via msgpack into
the appropriate `Message` subclass (dispatched on `role`).

### `save_extra` path

```sql
INSERT INTO cubepi_threads (thread_id, extra, updated_at)
VALUES ($1, $2, now())
ON CONFLICT (thread_id) DO UPDATE
SET extra = cubepi_threads.extra || EXCLUDED.extra,
    updated_at = now();
```

PostgreSQL `||` on JSONB does shallow merge — matches existing
`SQLiteCheckpointer.save_extra` semantics (caller-side merge in
SQLite, server-side merge here).

### Packaging

```toml
# pyproject.toml additions
[project.optional-dependencies]
postgres = ["sqlalchemy>=2.0", "asyncpg>=0.29", "msgpack>=1.0"]
```

### Out of scope (deferred)

- Application-layer thread archival / pruning
- Time-based partitioning (use case: log/audit data — doesn't fit
  conversation access patterns)
- Read replicas / multi-region
- Online schema migration tooling beyond schema version check
- Compression (TOAST + msgpack already covers common cases)

---

## D2 — MCP adapter

### Scope

`cubepi.mcp` module with `load_mcp_tools_http` and
`load_mcp_tools_stdio` returning `list[cubepi.AgentTool]`. Both
transports supported; cubebox today uses HTTP only, future
cubepi-coding-agent use cases (stdio servers like
`@modelcontextprotocol/server-filesystem`) need stdio.

### API

```python
# cubepi/mcp/__init__.py
from cubepi import AgentTool

async def load_mcp_tools_http(
    server_url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> list[AgentTool]:
    """Connect to an HTTP/SSE MCP server, discover tools,
    convert each to cubepi.AgentTool."""
    ...

async def load_mcp_tools_stdio(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float = 30.0,
) -> list[AgentTool]:
    """Spawn a stdio MCP server subprocess, discover tools,
    convert each to cubepi.AgentTool."""
    ...
```

Each returned `AgentTool.execute` invokes the corresponding MCP
`tools/call` RPC and translates the response into `AgentToolResult`.

### Packaging

```toml
[project.optional-dependencies]
mcp = ["mcp>=1.0"]
```

Single extra covers both transports; `mcp>=1.0` Python package
includes stdio + HTTP/SSE client.

### Testing

cubepi E2E:
- HTTP transport: spin up a fake MCP server (FastAPI) that advertises
  two tools; load via `load_mcp_tools_http`; invoke each; verify
  result translation.
- stdio transport: spawn a Python subprocess implementing minimal MCP
  stdio protocol; load via `load_mcp_tools_stdio`; same assertions.

### Out of scope

- MCP server-side helpers (cubepi consumes MCP, doesn't expose itself
  as an MCP server)
- OAuth flows for MCP servers (host application handles credentials,
  passes headers)
- Resource / prompt primitives (only tools supported; aligns with
  cubebox's use)

---

## D3 — Anthropic configurable cache marker policy

### Why

cubepi `AnthropicProvider` currently marks: system + last message +
last tool. cubebox needs: system + last completed AIMessage + tools.
The "last completed AIMessage" requires walking back the message
list. Hard-coded policy doesn't fit; needs to be caller-configurable.

### API

```python
# cubepi/providers/anthropic.py additions
from typing import Protocol

class CacheMarkerPolicy(Protocol):
    def mark_system(self) -> bool: ...
    def mark_last_tool(self) -> bool: ...
    def message_breakpoint_indices(
        self,
        messages: list[Message],
    ) -> list[int]: ...

class DefaultCacheMarkerPolicy:
    """Preserves current cubepi behavior: system + last message + last tool."""
    def mark_system(self) -> bool:
        return True
    def mark_last_tool(self) -> bool:
        return True
    def message_breakpoint_indices(self, messages):
        return [len(messages) - 1] if messages else []

class AnthropicProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_policy: CacheMarkerPolicy | None = None,
    ):
        ...
        self._cache_policy = cache_policy or DefaultCacheMarkerPolicy()
```

Provider applies marker policy at message conversion time. Existing
users who don't pass `cache_policy` see no behavior change.

### Out of scope

- Provider-side knowledge of "completed AIMessage" semantics. Policy
  callback is given the message list as-is; caller decides what
  "completed" means in their domain.
- OpenAI cache markers — OpenAI auto-caches on byte-identical prefix,
  no markers needed.

---

## D4 — OpenAI Chat Completions OSS reasoning + payload quirks

### Why

cubepi `OpenAIResponsesProvider` already handles OpenAI's official
Responses API reasoning. But cubebox uses Chat Completions endpoints
against openai-compatible servers (DeepSeek, DouBao, Qwen, vLLM,
MiniMax) which return reasoning via three different non-standard
fields:

- `reasoning_content` — DeepSeek, DouBao, Qwen
- `reasoning` — vLLM
- `reasoning_details: [{text: ...}, ...]` — MiniMax

`OpenAIProvider` currently ignores all of these.

### API

```python
# cubepi/providers/openai.py additions
from typing import Literal

class OpenAIProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        payload_quirks: list[Literal["max_completion_tokens_alias"]] | None = None,
    ):
        ...
        self._payload_quirks = set(payload_quirks or [])
```

Two changes inside `OpenAIProvider`:

1. **Streaming chunk handler** (and non-streaming `_create_chat_result`
   equivalent): after standard content/tool_call extraction, check
   `delta.reasoning_content` → `delta.reasoning` →
   `delta.reasoning_details[*].text` in that priority order. Emit
   existing `thinking_start` / `thinking_delta` / `thinking_end`
   events (no new event types needed).

2. **Payload preparation**: if `"max_completion_tokens_alias"` is in
   `payload_quirks`, rewrite `max_completion_tokens → max_tokens` in
   the outgoing request body (some older openai-compatible endpoints
   reject the newer param name).

### Testing

cubepi E2E:
- Fake OpenAI-compatible server emitting each of the three reasoning
  field variants in streaming + non-streaming responses; verify
  cubepi emits correct `thinking_*` events.
- Fake server requires `max_tokens`; verify `payload_quirks=
  ["max_completion_tokens_alias"]` produces correct rewrite.

### Out of scope

- Tool call timing metadata (cubebox-specific telemetry — host
  application's responsibility, not provider's)
- New event types — existing `thinking_*` already covers reasoning
  semantics across both OpenAI Responses and Chat Completions paths

---

## D5 — `Message.metadata` field

### Why

cubebox's memory system stores per-user-message immutable snapshots
of the relevance memory shown to the model at that turn. The cubebox
side persists them per message, not as a separate state channel.
cubepi needs to carry arbitrary metadata alongside each message
through ser/deser.

This is also useful for any cubepi user who needs to attach metadata
to messages (cost, source, tool_call_id linking, audit tags, etc.).

### API

```python
# cubepi/providers/base.py changes
from pydantic import BaseModel, Field

class UserMessage(BaseModel):
    content: list[Content]
    metadata: dict[str, Any] = Field(default_factory=dict)
    ...

class AssistantMessage(BaseModel):
    content: list[Content]
    metadata: dict[str, Any] = Field(default_factory=dict)
    ...

class ToolResultMessage(BaseModel):
    content: list[Content]
    metadata: dict[str, Any] = Field(default_factory=dict)
    ...
```

### Checkpointer integration

- `PostgresCheckpointer.append`: writes `message.metadata` to
  `cubepi_messages.metadata JSONB`.
- `PostgresCheckpointer.load`: hydrates `message.metadata` from the
  row.
- `SQLiteCheckpointer`: same; existing serialization to/from JSON
  already round-trips dicts.

### Backwards compatibility

Existing `UserMessage(content=[...])` calls work unchanged —
`metadata` defaults to `{}`. Existing serialized messages (without
the field) deserialize cleanly because pydantic defaults the field.

---

## D6 — `AgentContext.extra` mutable dict

### Why

Middleware needs access to per-thread non-message state (cubebox's
compaction summary, TodoListMiddleware's 6 planning channels, etc.).
The `Checkpointer` protocol already exposes `save_extra(thread_id,
extra)` / `load` returns `CheckpointData.extra`. But the
middleware hooks don't currently expose extra to the middleware
implementation.

Adding `extra` to `AgentContext` and threading it through hook
contexts lets middleware read/mutate per-thread state in any hook,
with the agent loop persisting at the right times.

### API

```python
# cubepi/agent/types.py changes
from dataclasses import dataclass, field

@dataclass
class AgentContext:
    system_prompt: str
    messages: list[Message]
    tools: list[AgentTool] | None = None
    extra: dict[str, Any] = field(default_factory=dict)   # NEW
```

`extra` is mutable. Middleware in any hook with access to
`AgentContext` (or a context type embedding it like
`BeforeToolCallContext.context`, `AfterToolCallContext.context`,
`ShouldStopAfterTurnContext.context`, and the new hook contexts
introduced in D7/D8) can read and mutate `ctx.context.extra` in
place.

### Agent loop persistence

After each tool execution turn (post `after_tool_call`) and after
each model response (post `after_model_response`), the agent loop
calls `checkpointer.save_extra(thread_id, agent_context.extra)`.
This is unconditional — cheap UPSERT on JSONB.

On load: `agent.prompt(...)` calls `checkpointer.load(thread_id)` and
initializes `AgentContext.extra` from `CheckpointData.extra`.

### Backwards compatibility

Existing middleware that doesn't access `ctx.extra` works unchanged.
Existing `AgentContext` constructors work — `extra` defaults to `{}`.

---

## D7 — `transform_system_prompt` middleware hook

### Why

cubebox's `SandboxMiddleware` and `SkillsMiddleware` dynamically
mutate the system prompt per turn (sandbox capability text injected
when sandbox is initialized; skill content appended after a `load_skill`
tool call). cubepi's `system_prompt` is currently set once at
`Agent` construction; middleware has no hook to modify it per call.

### API

```python
# cubepi/middleware/base.py additions
class Middleware:
    async def transform_system_prompt(
        self,
        system_prompt: str,
        *,
        signal: asyncio.Event | None = None,
    ) -> str:
        raise NotImplementedError
```

### Composition

Chain — each middleware receives the previous middleware's return
value. Identical to `transform_context`'s composition rule.

```python
def compose_middleware(middlewares: list[Middleware]) -> dict[str, Callable]:
    ...
    sp_chain = [m for m in middlewares if _has_method(m, "transform_system_prompt")]
    if sp_chain:
        async def composed_sp(system_prompt, *, signal=None):
            result = system_prompt
            for mw in sp_chain:
                result = await mw.transform_system_prompt(result, signal=signal)
            return result
        hooks["transform_system_prompt"] = composed_sp
```

### Agent loop integration

Before each provider call, agent loop applies `transform_context` to
messages AND `transform_system_prompt` to the system prompt, then
sends both to the provider's `stream(...)`.

### Backwards compatibility

Middleware that doesn't override `transform_system_prompt` has no
effect. Existing `Agent(system_prompt="...")` constructors work
unchanged.

---

## D8 — `after_model_response` middleware hook + `TurnAction`

### Why

cubebox uses LangChain's `after_model` hook for three purposes:

- **CostMiddleware**: read `AssistantMessage.usage`, record billing
- **TimestampMiddleware**: stamp turn-end time on response
- **TodoListMiddleware**: inspect response for write_todos usage,
  inject corrective SystemMessages, optionally force loop to model or
  force stop

cubepi today has `should_stop_after_turn` (bool, simple termination)
but no hook for response observation, response mutation, message
injection, or forced loop-back.

### API

```python
# cubepi/middleware/base.py additions
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class TurnAction:
    """Directs the agent loop's next step after a model response.

    Composition (chain): each middleware sees previous middleware's
    TurnAction. Last middleware's value wins for `response` and
    `decision`. `inject_messages` concatenates across the chain.
    """
    response: AssistantMessage | None = None
    inject_messages: list[Message] = field(default_factory=list)
    decision: Literal["natural", "stop", "loop_to_model"] = "natural"

class Middleware:
    async def after_model_response(
        self,
        response: AssistantMessage,
        ctx: AgentContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> TurnAction | None:
        raise NotImplementedError
```

### Semantics

| `decision` | Effect on agent loop |
|---|---|
| `natural` | Default flow — if response has tool calls, execute them; otherwise terminate the turn |
| `stop` | Force termination, override natural continuation |
| `loop_to_model` | Re-invoke the model (with `inject_messages` added to context); override natural termination |

Returning `None` is shorthand for `TurnAction(decision="natural")`.

### Composition

Chain. Each middleware receives the previous middleware's
`TurnAction` as input (via wrapping logic in `compose_middleware`).

```python
after_resp_chain = [
    m for m in middlewares if _has_method(m, "after_model_response")
]
if after_resp_chain:
    async def composed(response, ctx, *, signal=None):
        current_response = response
        all_inject: list[Message] = []
        last_decision: Literal["natural", "stop", "loop_to_model"] = "natural"
        for mw in after_resp_chain:
            result = await mw.after_model_response(
                current_response, ctx, signal=signal
            )
            if result is None:
                continue
            if result.response is not None:
                current_response = result.response
            if result.inject_messages:
                all_inject.extend(result.inject_messages)
            last_decision = result.decision
        return TurnAction(
            response=current_response,
            inject_messages=all_inject,
            decision=last_decision,
        )
    hooks["after_model_response"] = composed
```

### Relationship to `should_stop_after_turn`

`should_stop_after_turn` stays in cubepi (no change, no deprecation).
It remains the simple bool-returning termination hook. Cubebox
middleware uses `after_model_response` for the richer use case;
existing cubepi users keep using `should_stop_after_turn` for simple
stop conditions.

### Agent loop integration

After receiving the model's `AssistantMessage`:

1. Call composed `after_model_response(response, ctx)` → `TurnAction`
2. If `decision == "stop"`: terminate turn, return `TurnAction.response`
3. If `decision == "loop_to_model"`: append `inject_messages` to
   `ctx.messages`, jump back to model call
4. If `decision == "natural"`: proceed (execute tool calls or
   terminate)

### Backwards compatibility

Middleware that doesn't override `after_model_response` has no
effect. The agent loop's natural behavior (execute tools then loop,
or stop if no tools) is preserved when no middleware overrides.

---

## D9 — Packaging extras (consolidated)

```toml
# pyproject.toml
[project.optional-dependencies]
sqlite   = ["aiosqlite>=0.19"]                                 # existing
postgres = ["sqlalchemy>=2.0", "asyncpg>=0.29", "msgpack>=1.0"] # D1
mcp      = ["mcp>=1.0"]                                        # D2
```

Anthropic, OpenAI client deps stay as base requirements (already
there).

---

## Acceptance criteria

Spec A is complete when all of the following hold:

| # | Check | Verified by |
|---|---|---|
| A1 | `cubepi.checkpointer.postgres.PostgresCheckpointer` round-trips `append` / `load` / `save_extra` against real Postgres | cubepi E2E |
| A2 | Schema version check raises on uninitialized DB / on mismatch | cubepi E2E |
| A3 | 64 hash partitions created via alembic helpers; insert routes correctly; load reads correctly | cubepi E2E |
| A4 | `cubepi.mcp.load_mcp_tools_http` loads + executes tools against fake MCP server | cubepi E2E |
| A5 | `cubepi.mcp.load_mcp_tools_stdio` loads + executes tools against subprocess MCP server | cubepi E2E |
| A6 | `AnthropicProvider(cache_policy=CustomPolicy())` applies custom marker placement | cubepi unit + E2E |
| A7 | `AnthropicProvider()` without `cache_policy` preserves v0.2 behavior byte-for-byte | cubepi unit (golden fixture) |
| A8 | `OpenAIProvider` emits `thinking_*` events for all three OSS reasoning field variants | cubepi unit + E2E |
| A9 | `OpenAIProvider(payload_quirks=["max_completion_tokens_alias"])` rewrites payload correctly | cubepi unit |
| A10 | `Message.metadata` round-trips through `PostgresCheckpointer` + `SQLiteCheckpointer` byte-identical | cubepi unit |
| A11 | `AgentContext.extra` mutated in any hook persists via `save_extra` after the turn | cubepi unit |
| A12 | `transform_system_prompt` chain composition: 2 middlewares applied in order produce expected output | cubepi unit |
| A13 | `after_model_response` chain composition: response mutation + message injection + decision precedence per spec | cubepi unit |
| A14 | All 9 deliverables backward-compatible: existing v0.2 tests pass unchanged | cubepi full test suite |

cubepi may release as v0.3.0 once A1-A14 pass; cubebox path
dependency consumes earlier (during dev iteration).

---

## Timeline relationship to Spec B

cubebox Spec B's milestone M0 depends on D1 (Postgres checkpointer).
M1 / M3 depend on D3 / D5 / D6 / D7 / D8. M2 depends on D2.

Spec A deliverables can land in any order in cubepi as long as the
dependency relationships above hold. cubebox can start M0 as soon as
D1 + D5 are merged in cubepi.

## Out of scope (deferred to future cubepi work)

- Provider-level retry / rate-limit policies (host application's
  responsibility today)
- New hook types beyond D7 / D8 — add singly if cubebox's middleware
  port turns up a real gap
- Online schema migration tooling
- Multi-region / read replica support in `PostgresCheckpointer`
- Subagent persistence primitive (cubepi loop currently subagent-
  agnostic; subagents-as-tools pattern remains in host application)
- Time-based partitioning support
- cubepi-side observability (tracing, metrics) integration
