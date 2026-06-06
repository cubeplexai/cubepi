# Conversation Fork — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement run-based conversation fork (`Agent.fork()` + `Agent.fork_once()`) per `dev/specs/2026-06-05-conversation-fork.md` (V2, codex-clean at commit d27276d).

**Architecture:**
- Add per-message `run_id`, a `cubepi_runs` table per backend tracking
  `(thread_id, run_id, claimed_at, completed_at, completion_seq)`, and the
  Checkpointer Protocol methods `claim_run` / `mark_run_complete` /
  `snapshot` / `fork` / `load_pending`. Agent.prompt() claims at start,
  marks at clean terminal exit; fork is a set-based physical copy keyed by
  completed `run_id`s.
- Implementations across Memory / SQLite / Postgres / MySQL, with a
  legacy-degraded mode for third-party v3-only checkpointers.
- HITL channel run_id binding enforced at prompt() entry via a structural
  `HitlBinding` attribute on `AgentTool` / `Middleware`.

**Tech Stack:** Python 3.13, pydantic v2, asyncio, aiosqlite, asyncpg,
aiomysql, SQLAlchemy 2.0, pytest (asyncio_mode=auto), uv, ruff, mypy.

**Out of scope (separate spec):** `Agent.delete_run()`, cubebox UI wiring,
backfill of legacy threads.

---

## Phase 0 — Bootstrap & spec checkpoint

### Task 0: Pre-flight

**Files:** none modified

- [ ] **Step 1: Confirm worktree + branch**

Run:
```
git rev-parse --abbrev-ref HEAD
git status
```
Expected: branch `2026-06-05-conversation-fork`, working tree clean.

- [ ] **Step 2: Confirm spec is the codex-clean V2**

Run: `git log --oneline 3a1490e..HEAD | head -1`
Expected: most recent commit is `d27276d spec: address v2 R10 codex finding (tighten tool-cycle invariant)`.

- [ ] **Step 3: Install + baseline test run**

Run:
```
uv sync --all-extras --dev
uv run pytest tests/ -x -q
```
Expected: all green (existing tests, no impl changes yet).

### Fixture conventions for the rest of the plan

These names appear in test snippets throughout the plan.

| Plan name | Maps to | Returns |
|---|---|---|
| Postgres DSN (raw, empty DB) | `clean_db` (existing) | DSN string |
| MySQL DSN (raw, empty DB) | `clean_mysql_db` (existing) | DSN string |
| Postgres DSN with v4 schema | `pg_v4_dsn` (NEW — Task 14) | DSN string |
| MySQL DSN with v4 schema | `mysql_v4_dsn` (NEW — Task 18) | DSN string |
| In-memory tracing exporter | `in_memory_exporter` (NEW — Task 34) | OTel exporter |
| MemoryCheckpointer | (no fixture — instantiate inline `MemoryCheckpointer()`) | — |

**Schema setup matters.** `clean_db` / `clean_mysql_db` create empty
databases. `PostgresCheckpointer.__aenter__` runs schema-version
verification and raises `CubepiSchemaUninitialized` against an empty
DB. Tests that exercise a real checkpointer instance MUST use the
v4-applied fixtures (`pg_v4_dsn` / `mysql_v4_dsn`), which Tasks 14 and
18 create by extending the existing `_setup_schema(dsn)` helper at
`tests/checkpointer/test_postgres.py:177` (PG) and the equivalent
pattern in `test_mysql.py` (MySQL).

Both SQL checkpointers' constructors take a DSN string (PG signature:
`PostgresCheckpointer(dsn, *, min_pool_size=1, max_pool_size=10)` at
`cubepi/checkpointer/postgres/checkpointer.py:62`; MySQL similar).
Test snippets MUST construct them via `async with
PostgresCheckpointer(pg_v4_dsn) as cp:` — NOT by passing a pool object,
and NOT by passing raw `clean_db` if exercising the checkpointer.

### FauxProvider construction

`cubepi/providers/faux.py:156` defines:
```python
class FauxProvider(BaseProvider):
    def __init__(self, *, tokens_per_second=None, ..., provider_id=""): ...
    def set_responses(self, responses: list[AssistantMessage | FauxResponseFactory]): ...
```

It has **no `text=` / `error=` / `tool_error=` / `abort_mid_stream=` /
`sleep_seconds=` kwargs**. To produce a single "ok" reply, every test
file uses this helper:

```python
from cubepi.providers.base import AssistantMessage, TextContent
from cubepi.providers.faux import FauxProvider


def _ok_faux() -> FauxProvider:
    p = FauxProvider()
    p.set_responses([
        AssistantMessage(content=[TextContent(text="ok")], stop_reason="end_turn")
    ])
    return p
```

Plan snippets that say `_ok_faux()` rely on this local helper —
define it at the top of each test file (or share via
`tests/agent/_helpers.py`).

To inject provider-side errors / slow streams, subclass
`FauxProvider` and override `stream_message`:

```python
class _RaisingProvider(FauxProvider):
    async def stream(self, *args, **kwargs):
        raise RuntimeError("provider down")
```

---

## Phase 1 — Foundation types

Define the data shapes the rest of the plan builds on. No behavior yet.

### Task 1: Add `CheckpointerError` runtime base + fork-error types

**Files:**
- Modify: `cubepi/checkpointer/exceptions.py`
- Test: `tests/checkpointer/test_exceptions.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/checkpointer/test_exceptions.py`:
```python
import pytest

from cubepi.checkpointer.exceptions import (
    CheckpointerError,
    CheckpointerLockTimeoutError,
    CompletionMarkerFailedError,
    CubepiSchemaError,
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
    RunNotCompletedError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [
        ThreadNotFoundError,
        ThreadAlreadyExistsError,
        RunNotCompletedError,
        RunNotClaimedError,
        RunAlreadyClaimedError,
        RunAlreadyCompletedError,
        CheckpointerLockTimeoutError,
    ],
)
def test_runtime_errors_inherit_checkpointer_error(exc_cls):
    assert issubclass(exc_cls, CheckpointerError)
    assert issubclass(exc_cls, Exception)


def test_checkpointer_error_separate_from_schema_error():
    assert not issubclass(CheckpointerError, CubepiSchemaError)
    assert not issubclass(CubepiSchemaError, CheckpointerError)


def test_completion_marker_failed_error_carries_run_id():
    cause = RuntimeError("db timeout")
    exc = CompletionMarkerFailedError(
        thread_id="t1", run_id="r1", cause=cause
    )
    assert exc.thread_id == "t1"
    assert exc.run_id == "r1"
    assert exc.__cause__ is cause
    assert "t1" in str(exc) and "r1" in str(exc)


def test_runtime_errors_constructable_with_kwargs():
    e = ThreadNotFoundError("missing thread t1")
    assert "t1" in str(e)
    e = RunNotCompletedError("thread=t1 run=r1")
    assert "r1" in str(e)
```

- [ ] **Step 2: Run test → expect failure**

Run: `uv run pytest tests/checkpointer/test_exceptions.py -v`
Expected: `ImportError: cannot import name 'CheckpointerError' …`

- [ ] **Step 3: Extend `cubepi/checkpointer/exceptions.py`**

Append to the existing file:
```python
class CheckpointerError(Exception):
    """Base class for cubepi checkpointer runtime errors.

    Distinct from ``CubepiSchemaError`` (schema-vs-library incompatibility).
    ``CheckpointerError`` covers runtime operation outcomes — missing
    thread, lock timeout, run state, etc.
    """


class ThreadNotFoundError(CheckpointerError):
    pass


class ThreadAlreadyExistsError(CheckpointerError):
    pass


class RunNotCompletedError(CheckpointerError):
    """The cubepi_runs row for (thread_id, run_id) does not exist, or
    exists with completed_at IS NULL (paused, abandoned, or in flight)."""


class RunNotClaimedError(CheckpointerError):
    """mark_run_complete() called but no cubepi_runs row exists for
    (thread_id, run_id). Indicates an agent-loop logic bug."""


class RunAlreadyClaimedError(CheckpointerError):
    """claim_run() found an existing row with completed_at IS NULL.
    Another process is currently running this run_id; retry with a
    different run_id."""


class RunAlreadyCompletedError(CheckpointerError):
    """claim_run() found an existing row with completed_at IS NOT NULL.
    Runs are append-only; start a new run with a different run_id.

    NOT raised by mark_run_complete() — that path is idempotent on
    already-completed rows (spec §3.6.2).
    """


class CheckpointerLockTimeoutError(CheckpointerError):
    """Backend writer lock not acquired within the configured timeout
    (SQLite busy_timeout, etc.)."""


class CompletionMarkerFailedError(CheckpointerError):
    """mark_run_complete() failed AFTER the run's final append succeeded.
    Carries `run_id` so callers using prompt(run_id=None) can recover
    the cubepi-generated value (spec §3.6.2)."""

    def __init__(
        self,
        *,
        thread_id: str,
        run_id: str,
        cause: BaseException,
    ) -> None:
        super().__init__(
            f"mark_run_complete failed for ({thread_id}, {run_id}): {cause}"
        )
        self.thread_id = thread_id
        self.run_id = run_id
        self.__cause__ = cause
```

- [ ] **Step 4: Run test → expect pass**

Run: `uv run pytest tests/checkpointer/test_exceptions.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```
git add cubepi/checkpointer/exceptions.py tests/checkpointer/test_exceptions.py
git commit -m "feat(checkpointer): add CheckpointerError runtime hierarchy"
```

---

### Task 2: Add `parent_thread_id` to `CheckpointData`

**Files:**
- Modify: `cubepi/checkpointer/base.py`
- Test: `tests/checkpointer/test_base.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/checkpointer/test_base.py`:
```python
from cubepi.checkpointer.base import CheckpointData


def test_checkpoint_data_default_parent_is_none():
    cd = CheckpointData()
    assert cd.parent_thread_id is None
    assert cd.messages == []
    assert cd.extra == {}


def test_checkpoint_data_with_parent():
    cd = CheckpointData(parent_thread_id="src_thread")
    assert cd.parent_thread_id == "src_thread"


def test_checkpoint_data_keyword_construction_unchanged():
    cd = CheckpointData(messages=[], extra={"k": "v"})
    assert cd.extra == {"k": "v"}
    assert cd.parent_thread_id is None
```

- [ ] **Step 2: Run test → expect failure**

Run: `uv run pytest tests/checkpointer/test_base.py -v`
Expected: `AttributeError: 'CheckpointData' object has no attribute 'parent_thread_id'`.

- [ ] **Step 3: Extend `CheckpointData`**

In `cubepi/checkpointer/base.py`, modify the `CheckpointData` dataclass:
```python
@dataclass
class CheckpointData:
    messages: list[Message] = field(default_factory=list)
    extra: JsonObject = field(default_factory=dict)
    parent_thread_id: str | None = None
```

- [ ] **Step 4: Run test → expect pass**

Run: `uv run pytest tests/checkpointer/test_base.py -v`
Expected: 3 passed. Also run existing checkpointer tests: `uv run pytest tests/checkpointer/ -q` — all green.

- [ ] **Step 5: Commit**

```
git add cubepi/checkpointer/base.py tests/checkpointer/test_base.py
git commit -m "feat(checkpointer): add parent_thread_id field to CheckpointData"
```

---

### Task 3: Add `run_id` to all three Message variants

**Files:**
- Modify: `cubepi/providers/base.py`
- Test: `tests/providers/test_message_run_id.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/providers/test_message_run_id.py`:
```python
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)


def test_user_message_run_id_default_none():
    m = UserMessage(content=[TextContent(text="hi")])
    assert m.run_id is None


def test_assistant_message_run_id_default_none():
    m = AssistantMessage(content=[])
    assert m.run_id is None


def test_tool_result_message_run_id_default_none():
    m = ToolResultMessage(
        tool_call_id="tc1", tool_name="foo", content=[]
    )
    assert m.run_id is None


def test_run_id_round_trip_serialization():
    src = AssistantMessage(content=[], run_id="r-1")
    blob = src.model_dump_json()
    dst = AssistantMessage.model_validate_json(blob)
    assert dst.run_id == "r-1"


def test_run_id_is_keyword_only_in_practice():
    # All existing call sites pass content positionally / by keyword;
    # adding run_id as a defaulted field at the end is non-breaking.
    m = UserMessage(content=[TextContent(text="hi")], run_id="r-2")
    assert m.run_id == "r-2"
```

- [ ] **Step 2: Run test → expect failure**

Run: `uv run pytest tests/providers/test_message_run_id.py -v`
Expected: `ValidationError` or attribute errors on `run_id`.

- [ ] **Step 3: Add field to all three variants**

In `cubepi/providers/base.py`, add `run_id: str | None = None` to
`UserMessage`, `AssistantMessage`, and `ToolResultMessage` — last field
in each class so positional construction is unaffected:

```python
class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[Content]
    timestamp: float | None = None
    metadata: JsonObject = Field(default_factory=dict)
    run_id: str | None = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[Content | ThinkingContent | ToolCall]
    stop_reason: str = "stop"
    error_message: str | None = None
    usage: Usage | None = None
    timestamp: float | None = None
    provider_id: str = ""
    model_id: str = ""
    response_id: str | None = None
    metadata: JsonObject = Field(default_factory=dict)
    run_id: str | None = None


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[Content]
    details: StructuredValue = None
    is_error: bool = False
    timestamp: float | None = None
    metadata: JsonObject = Field(default_factory=dict)
    run_id: str | None = None
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/providers/test_message_run_id.py tests/providers/ -q`
Expected: all pass. Also `uv run mypy cubepi/providers/base.py` clean.

- [ ] **Step 5: Commit**

```
git add cubepi/providers/base.py tests/providers/test_message_run_id.py
git commit -m "feat(providers): add run_id field to Message variants"
```

---

### Task 4: Add `HitlBinding` type + attach to `AgentTool` and `Middleware`

**Files:**
- Modify: `cubepi/agent/types.py`
- Modify: `cubepi/middleware/base.py`
- Create: `cubepi/hitl/binding.py` (so importing it doesn't pull the agent module)
- Test: `tests/agent/test_hitl_binding.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/agent/test_hitl_binding.py`:
```python
from cubepi.agent.types import AgentTool
from cubepi.hitl.binding import HitlBinding
from cubepi.middleware.base import Middleware
from pydantic import BaseModel


class _NoArgs(BaseModel):
    pass


async def _noop(tool_call_id, args, *, signal=None, on_update=None):
    raise NotImplementedError


def test_agent_tool_hitl_default_none():
    t = AgentTool(
        name="t",
        description="d",
        parameters=_NoArgs,
        execute=_noop,
    )
    assert t.hitl is None


def test_agent_tool_hitl_can_be_set():
    binding = HitlBinding(checkpointed=True, run_id="r-1")
    t = AgentTool(
        name="t",
        description="d",
        parameters=_NoArgs,
        execute=_noop,
        hitl=binding,
    )
    assert t.hitl is binding
    assert t.hitl.checkpointed is True
    assert t.hitl.run_id == "r-1"


def test_middleware_hitl_default_none():
    class _Mw(Middleware):
        pass

    assert _Mw().hitl is None


def test_middleware_hitl_can_be_set_in_subclass():
    class _Mw(Middleware):
        def __init__(self) -> None:
            self.hitl = HitlBinding(checkpointed=False, run_id=None)

    mw = _Mw()
    assert mw.hitl is not None
    assert mw.hitl.checkpointed is False
    assert mw.hitl.run_id is None


def test_hitl_binding_is_frozen():
    b = HitlBinding(checkpointed=True, run_id="r-1")
    try:
        b.checkpointed = False  # type: ignore[misc]
    except Exception:
        return
    assert False, "HitlBinding should be frozen"
```

- [ ] **Step 2: Run test → expect failure**

Run: `uv run pytest tests/agent/test_hitl_binding.py -v`
Expected: `ImportError: cannot import name 'HitlBinding'`.

- [ ] **Step 3: Create `cubepi/hitl/binding.py`**

```python
"""HitlBinding — structural attribute on AgentTool and Middleware
declaring how a HITL element integrates with the checkpointer.

See dev/specs/2026-06-05-conversation-fork.md §3.6.3.1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HitlBinding:
    """How a tool/middleware integrates with HITL.

    Attributes:
        checkpointed: True iff backed by ``CheckpointedChannel`` (writes
            ``pending_request`` to the source thread on pause).
        run_id: The channel's bound run_id. For checkpointed HITL this
            MUST be a non-empty string; ``None`` is a configuration
            error and is rejected at ``Agent.prompt()`` entry. For
            in-memory HITL it is ``None``.
    """

    checkpointed: bool
    run_id: str | None
```

- [ ] **Step 4: Add `hitl: HitlBinding | None = None` to `AgentTool`**

In `cubepi/agent/types.py`, extend `AgentTool` (last field):
```python
from cubepi.hitl.binding import HitlBinding


@dataclass
class AgentTool(Generic[TParams]):
    name: str
    description: str
    parameters: type[TParams]
    execute: Callable[..., Awaitable[AgentToolResult]]
    label: str = ""
    execution_mode: Literal["sequential", "parallel"] | None = None
    hitl_builtin: bool = False
    hitl: HitlBinding | None = None
```

Note: `hitl_builtin` is the pre-existing flag for "this tool is one of
cubepi's built-in HITL tools" — leave it alone. The new `hitl`
attribute is independent and carries structural binding info.

- [ ] **Step 5: Add `hitl: HitlBinding | None = None` to `Middleware`**

In `cubepi/middleware/base.py`, add a class-level default on
`Middleware`:
```python
from cubepi.hitl.binding import HitlBinding


class Middleware:
    hitl: HitlBinding | None = None

    async def transform_context(self, ...): ...
    # rest unchanged
```

- [ ] **Step 6: Run tests → expect pass**

Run: `uv run pytest tests/agent/test_hitl_binding.py -v`
Expected: 5 passed. Also `uv run mypy cubepi/agent/types.py cubepi/middleware/base.py cubepi/hitl/binding.py` clean.

- [ ] **Step 7: Commit**

```
git add cubepi/hitl/binding.py cubepi/agent/types.py cubepi/middleware/base.py tests/agent/test_hitl_binding.py
git commit -m "feat(hitl): add HitlBinding attribute to AgentTool and Middleware"
```

---

### Task 5: Add `active_run_id` field to `AgentState`

**Files:**
- Modify: `cubepi/agent/agent.py`
- Test: `tests/agent/test_agent_state.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/agent/test_agent_state.py`:
```python
from cubepi.agent.agent import AgentState


def test_agent_state_default_active_run_id_none():
    s = AgentState()
    assert s.active_run_id is None


def test_agent_state_active_run_id_settable():
    s = AgentState()
    s.active_run_id = "r-1"
    assert s.active_run_id == "r-1"
    s.active_run_id = None
    assert s.active_run_id is None
```

- [ ] **Step 2: Run test → expect failure**

Run: `uv run pytest tests/agent/test_agent_state.py -v`
Expected: `AttributeError: 'AgentState' object has no attribute 'active_run_id'`.

- [ ] **Step 3: Add the field**

In `cubepi/agent/agent.py`, locate `class AgentState` (around line 92).
Add:
```python
class AgentState:
    # ... existing fields ...
    active_run_id: str | None = None
```

(Use the same dataclass/attrs style the existing class uses; just add
the new attribute as a sibling with a `None` default.)

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/agent/test_agent_state.py tests/agent/ -q`
Expected: all green.

- [ ] **Step 5: Commit**

```
git add cubepi/agent/agent.py tests/agent/test_agent_state.py
git commit -m "feat(agent): add active_run_id field to AgentState"
```

---

## Phase 2 — Checkpointer Protocol additions

### Task 6: Declare new Protocol methods on `Checkpointer`

**Files:**
- Modify: `cubepi/checkpointer/base.py`

This is a Protocol-only change. No structural runtime test — the methods
will be tested per-backend in Phases 3–6.

- [ ] **Step 1: Add the method signatures**

In `cubepi/checkpointer/base.py`, after the existing methods on the
`Checkpointer` Protocol, add:

```python
from cubepi.hitl.types import HitlRequest


@runtime_checkable
class Checkpointer(Protocol):
    # ... existing: load, append, save_extra, save_pending_request,
    # load_pending_request ...

    async def snapshot(
        self, thread_id: str, *, after_run_id: str
    ) -> list[Message]:
        """Return messages of completed runs of `thread_id` up through
        and including `after_run_id`, in source seq order. Raises
        ThreadNotFoundError or RunNotCompletedError."""
        ...

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        after_run_id: str,
        metadata: JsonObject | None = None,
    ) -> None:
        """Atomically physical-copy messages of completed runs up
        through `after_run_id` from src to new. See spec §3.2 / §3.4."""
        ...

    async def claim_run(
        self,
        thread_id: str,
        run_id: str,
    ) -> None:
        """Insert cubepi_runs row with claimed_at=now, completed_at=NULL.
        Lazily creates the threads row if needed. Raises
        RunAlreadyClaimedError or RunAlreadyCompletedError on PK
        conflict (distinguished by completed_at IS NULL/NOT NULL)."""
        ...

    async def mark_run_complete(
        self,
        thread_id: str,
        run_id: str,
    ) -> None:
        """Allocate next per-thread completion_seq; UPDATE the run row.
        Idempotent on already-completed rows (does NOT raise
        RunAlreadyCompletedError). Raises RunNotClaimedError when no
        row exists."""
        ...

    async def load_pending(
        self,
        thread_id: str,
    ) -> tuple[HitlRequest, str | None] | None:
        """Read (HitlRequest, run_id) atomically from the pending row,
        or None when no pending request exists."""
        ...
```

- [ ] **Step 2: Verify protocol signatures import cleanly**

Run: `uv run python -c "from cubepi.checkpointer.base import Checkpointer; print(Checkpointer.__annotations__)"`
Expected: prints attributes incl. the new methods. Also
`uv run mypy cubepi/checkpointer/base.py` clean.

- [ ] **Step 3: Commit**

```
git add cubepi/checkpointer/base.py
git commit -m "feat(checkpointer): declare snapshot/fork/claim_run/mark_run_complete/load_pending on Protocol"
```

---

## Phase 3 — Memory backend

Memory is the simplest backend; nail the semantics here before
tackling SQL.

### Task 7: Memory backend — shared lock + RunState dict

**Files:**
- Modify: `cubepi/checkpointer/memory.py`
- Test: `tests/checkpointer/test_memory_runs.py` (new)

- [ ] **Step 1: Write failing test (lock + state map exist)**

Create `tests/checkpointer/test_memory_runs.py`:
```python
import asyncio

import pytest

from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
)
from cubepi.checkpointer.memory import MemoryCheckpointer


@pytest.mark.asyncio
async def test_claim_then_complete_roundtrip():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "r1")
    await cp.mark_run_complete("t", "r1")
    # Idempotent: second mark is a no-op.
    await cp.mark_run_complete("t", "r1")


@pytest.mark.asyncio
async def test_claim_collision_in_flight_raises_claimed():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "r1")
    with pytest.raises(RunAlreadyClaimedError):
        await cp.claim_run("t", "r1")


@pytest.mark.asyncio
async def test_claim_collision_completed_raises_completed():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "r1")
    await cp.mark_run_complete("t", "r1")
    with pytest.raises(RunAlreadyCompletedError):
        await cp.claim_run("t", "r1")


@pytest.mark.asyncio
async def test_mark_without_claim_raises_not_claimed():
    cp = MemoryCheckpointer()
    with pytest.raises(RunNotClaimedError):
        await cp.mark_run_complete("t", "r1")


@pytest.mark.asyncio
async def test_completion_seq_monotonic_per_thread():
    cp = MemoryCheckpointer()
    for rid in ("A", "B", "C"):
        await cp.claim_run("t", rid)
        await cp.mark_run_complete("t", rid)
    # Internal inspection: completion_seq for A < B < C strictly.
    seqs = [cp._runs["t"][rid].completion_seq for rid in ("A", "B", "C")]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 3
```

- [ ] **Step 2: Run test → expect failure**

Run: `uv run pytest tests/checkpointer/test_memory_runs.py -v`
Expected: `AttributeError: 'MemoryCheckpointer' object has no attribute 'claim_run'`.

- [ ] **Step 3: Extend `MemoryCheckpointer`**

Replace `cubepi/checkpointer/memory.py` with:
```python
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from cubepi.checkpointer.base import CheckpointData
from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
    RunNotCompletedError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)
from cubepi.hitl.types import HitlRequest
from cubepi.providers.base import Message
from cubepi.types import JsonObject


@dataclass
class _RunState:
    claimed_at: float
    completed_at: float | None = None
    completion_seq: int | None = None


class MemoryCheckpointer:
    def __init__(self) -> None:
        self._store: dict[str, CheckpointData] = {}
        self._pending: dict[str, HitlRequest] = {}
        self._pending_run_id: dict[str, str | None] = {}
        self._runs: dict[str, dict[str, _RunState]] = {}
        self._lock = asyncio.Lock()

    async def load(self, thread_id: str) -> CheckpointData | None:
        return self._store.get(thread_id)

    async def append(self, thread_id: str, messages: list[Message]) -> None:
        async with self._lock:
            for m in messages:
                if m.run_id is None:
                    continue
                rs = self._runs.get(thread_id, {}).get(m.run_id)
                if rs is not None and rs.completed_at is not None:
                    raise RunAlreadyCompletedError(
                        f"append on completed run thread={thread_id} run={m.run_id}"
                    )
            if thread_id not in self._store:
                self._store[thread_id] = CheckpointData()
            self._store[thread_id].messages.extend(messages)

    async def save_extra(self, thread_id: str, extra: JsonObject) -> None:
        async with self._lock:
            if thread_id not in self._store:
                self._store[thread_id] = CheckpointData()
            self._store[thread_id].extra.update(extra)

    async def save_pending_request(
        self,
        thread_id: str,
        request: HitlRequest | None,
        *,
        run_id: str | None = None,
    ) -> None:
        async with self._lock:
            if request is None:
                self._pending.pop(thread_id, None)
                self._pending_run_id.pop(thread_id, None)
            else:
                self._pending[thread_id] = request
                self._pending_run_id[thread_id] = run_id

    async def load_pending_request(self, thread_id: str) -> HitlRequest | None:
        return self._pending.get(thread_id)

    async def load_pending_run_id(self, thread_id: str) -> str | None:
        return self._pending_run_id.get(thread_id)

    async def load_pending(
        self, thread_id: str
    ) -> tuple[HitlRequest, str | None] | None:
        req = self._pending.get(thread_id)
        if req is None:
            return None
        return req, self._pending_run_id.get(thread_id)

    async def claim_run(self, thread_id: str, run_id: str) -> None:
        async with self._lock:
            runs = self._runs.setdefault(thread_id, {})
            existing = runs.get(run_id)
            if existing is not None:
                if existing.completed_at is None:
                    raise RunAlreadyClaimedError(
                        f"thread={thread_id} run={run_id} in flight"
                    )
                raise RunAlreadyCompletedError(
                    f"thread={thread_id} run={run_id} already completed"
                )
            runs[run_id] = _RunState(claimed_at=time.time())

    async def mark_run_complete(self, thread_id: str, run_id: str) -> None:
        async with self._lock:
            runs = self._runs.get(thread_id) or {}
            state = runs.get(run_id)
            if state is None:
                raise RunNotClaimedError(
                    f"thread={thread_id} run={run_id} has no claim row"
                )
            if state.completed_at is not None:
                return  # idempotent
            existing_seqs = [
                s.completion_seq
                for s in runs.values()
                if s.completion_seq is not None
            ]
            next_seq = max(existing_seqs, default=0) + 1
            state.completed_at = time.time()
            state.completion_seq = next_seq
```

(Note: `snapshot` and `fork` are added in Task 9.)

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/checkpointer/test_memory_runs.py tests/checkpointer/test_memory.py -v`
Expected: new tests pass; existing memory tests still green.

- [ ] **Step 5: Commit**

```
git add cubepi/checkpointer/memory.py tests/checkpointer/test_memory_runs.py
git commit -m "feat(memory): claim_run + mark_run_complete with shared asyncio.Lock"
```

---

### Task 8: Memory backend — `append()` rejects completed run_ids; `load_pending` returns tuple

Already implemented in Task 7. Add explicit tests.

**Files:**
- Test: `tests/checkpointer/test_memory_runs.py` (extend)

- [ ] **Step 1: Append failing tests**

Append to `tests/checkpointer/test_memory_runs.py`:
```python
from cubepi.providers.base import TextContent, UserMessage


@pytest.mark.asyncio
async def test_append_on_completed_run_id_rejected():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "r1")
    await cp.mark_run_complete("t", "r1")
    msg = UserMessage(content=[TextContent(text="late")], run_id="r1")
    with pytest.raises(RunAlreadyCompletedError):
        await cp.append("t", [msg])


@pytest.mark.asyncio
async def test_append_in_flight_run_id_ok():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "r1")
    msg = UserMessage(content=[TextContent(text="ok")], run_id="r1")
    await cp.append("t", [msg])
    data = await cp.load("t")
    assert data is not None and len(data.messages) == 1


@pytest.mark.asyncio
async def test_load_pending_returns_tuple_with_run_id():
    from cubepi.hitl.types import HitlRequest

    cp = MemoryCheckpointer()
    req = HitlRequest(question_id="q1", question="hi", schema={}, choices=None)
    await cp.save_pending_request("t", req, run_id="r-1")
    res = await cp.load_pending("t")
    assert res is not None
    got_req, got_run_id = res
    assert got_req.question_id == "q1"
    assert got_run_id == "r-1"


@pytest.mark.asyncio
async def test_load_pending_returns_none_when_empty():
    cp = MemoryCheckpointer()
    assert await cp.load_pending("t") is None
```

- [ ] **Step 2: Run tests → expect pass**

Run: `uv run pytest tests/checkpointer/test_memory_runs.py -v`
Expected: 9 passed.

- [ ] **Step 3: Commit**

```
git add tests/checkpointer/test_memory_runs.py
git commit -m "test(memory): cover append-on-completed and load_pending tuple"
```

---

### Task 9: Memory backend — `snapshot()` + `fork()`

**Files:**
- Modify: `cubepi/checkpointer/memory.py`
- Test: `tests/checkpointer/test_memory_fork.py` (new)

- [ ] **Step 1: Write failing tests covering set-based selection**

Create `tests/checkpointer/test_memory_fork.py`:
```python
import pytest

from cubepi.checkpointer.exceptions import (
    RunNotCompletedError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import TextContent, UserMessage


def _msg(run_id: str | None, text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], run_id=run_id)


@pytest.mark.asyncio
async def test_fork_copies_completed_runs_only():
    cp = MemoryCheckpointer()
    # Two completed runs A and B.
    await cp.claim_run("src", "A")
    await cp.append("src", [_msg("A", "a1"), _msg("A", "a2")])
    await cp.mark_run_complete("src", "A")
    await cp.claim_run("src", "B")
    await cp.append("src", [_msg("B", "b1")])
    await cp.mark_run_complete("src", "B")
    # An in-flight run C — must be excluded.
    await cp.claim_run("src", "C")
    await cp.append("src", [_msg("C", "c1")])
    await cp.fork("src", "dst", after_run_id="B")
    loaded = await cp.load("dst")
    assert loaded is not None
    texts = [m.content[0].text for m in loaded.messages]
    assert texts == ["a1", "a2", "b1"]
    assert loaded.parent_thread_id == "src"


@pytest.mark.asyncio
async def test_fork_includes_legacy_null_run_id_prefix():
    cp = MemoryCheckpointer()
    # Legacy NULL-run_id message.
    await cp.append("src", [_msg(None, "legacy")])
    # One completed run.
    await cp.claim_run("src", "A")
    await cp.append("src", [_msg("A", "a1")])
    await cp.mark_run_complete("src", "A")
    await cp.fork("src", "dst", after_run_id="A")
    loaded = await cp.load("dst")
    assert [m.content[0].text for m in loaded.messages] == ["legacy", "a1"]


@pytest.mark.asyncio
async def test_fork_unknown_src_raises_thread_not_found():
    cp = MemoryCheckpointer()
    with pytest.raises(ThreadNotFoundError):
        await cp.fork("missing", "dst", after_run_id="X")


@pytest.mark.asyncio
async def test_fork_unknown_run_id_raises_not_completed():
    cp = MemoryCheckpointer()
    await cp.append("src", [_msg(None, "x")])
    with pytest.raises(RunNotCompletedError):
        await cp.fork("src", "dst", after_run_id="missing")


@pytest.mark.asyncio
async def test_fork_destination_collision_raises_already_exists():
    cp = MemoryCheckpointer()
    await cp.claim_run("src", "A")
    await cp.append("src", [_msg("A", "a1")])
    await cp.mark_run_complete("src", "A")
    await cp.fork("src", "dst", after_run_id="A")
    with pytest.raises(ThreadAlreadyExistsError):
        await cp.fork("src", "dst", after_run_id="A")


@pytest.mark.asyncio
async def test_fork_carries_extra_and_writes_metadata():
    cp = MemoryCheckpointer()
    await cp.save_extra("src", {"original": "x"})
    await cp.claim_run("src", "A")
    await cp.append("src", [_msg("A", "a1")])
    await cp.mark_run_complete("src", "A")
    await cp.fork("src", "dst", after_run_id="A", metadata={"source": "test"})
    loaded = await cp.load("dst")
    assert loaded.extra["original"] == "x"
    assert loaded.extra["fork"] == {"source": "test"}


@pytest.mark.asyncio
async def test_snapshot_matches_fork_messages():
    cp = MemoryCheckpointer()
    await cp.claim_run("src", "A")
    await cp.append("src", [_msg("A", "a1")])
    await cp.mark_run_complete("src", "A")
    msgs = await cp.snapshot("src", after_run_id="A")
    assert [m.content[0].text for m in msgs] == ["a1"]
```

- [ ] **Step 2: Run tests → expect failure**

Run: `uv run pytest tests/checkpointer/test_memory_fork.py -v`
Expected: `AttributeError: 'MemoryCheckpointer' object has no attribute 'fork'`.

- [ ] **Step 3: Implement `snapshot()` + `fork()` on `MemoryCheckpointer`**

Append to `cubepi/checkpointer/memory.py`:
```python
import copy


def _legible_message_copy(m):
    return m.model_copy(deep=True)


class MemoryCheckpointer:  # continue class
    async def snapshot(
        self, thread_id: str, *, after_run_id: str
    ) -> list[Message]:
        async with self._lock:
            data = self._store.get(thread_id)
            if data is None and thread_id not in self._runs:
                raise ThreadNotFoundError(f"thread={thread_id}")
            runs = self._runs.get(thread_id, {})
            cutoff_state = runs.get(after_run_id)
            if cutoff_state is None or cutoff_state.completion_seq is None:
                raise RunNotCompletedError(
                    f"thread={thread_id} run={after_run_id} not completed"
                )
            cutoff = cutoff_state.completion_seq
            selected: list[Message] = []
            for m in (data.messages if data else []):
                if m.run_id is None:
                    selected.append(_legible_message_copy(m))
                    continue
                rs = runs.get(m.run_id)
                if rs is None or rs.completion_seq is None:
                    continue
                if rs.completion_seq <= cutoff:
                    selected.append(_legible_message_copy(m))
            return selected

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        after_run_id: str,
        metadata: JsonObject | None = None,
    ) -> None:
        async with self._lock:
            if new_thread_id in self._store or new_thread_id in self._runs:
                raise ThreadAlreadyExistsError(f"thread={new_thread_id}")
            src_data = self._store.get(src_thread_id)
            src_runs = self._runs.get(src_thread_id, {})
            if src_data is None and not src_runs:
                raise ThreadNotFoundError(f"thread={src_thread_id}")
            cutoff_state = src_runs.get(after_run_id)
            if cutoff_state is None or cutoff_state.completion_seq is None:
                raise RunNotCompletedError(
                    f"thread={src_thread_id} run={after_run_id} not completed"
                )
            cutoff = cutoff_state.completion_seq
            # Select messages.
            new_messages: list[Message] = []
            for m in (src_data.messages if src_data else []):
                if m.run_id is None:
                    new_messages.append(_legible_message_copy(m))
                    continue
                rs = src_runs.get(m.run_id)
                if rs is None or rs.completion_seq is None:
                    continue
                if rs.completion_seq <= cutoff:
                    new_messages.append(_legible_message_copy(m))
            # Deep copy extra and merge metadata.
            base_extra = copy.deepcopy(
                src_data.extra if src_data else {}
            )
            if metadata is not None:
                base_extra["fork"] = copy.deepcopy(metadata)
            # Carry completed runs satisfying cutoff.
            new_runs: dict[str, _RunState] = {}
            for rid, state in src_runs.items():
                if state.completion_seq is None:
                    continue
                if state.completion_seq <= cutoff:
                    new_runs[rid] = _RunState(
                        claimed_at=state.claimed_at,
                        completed_at=state.completed_at,
                        completion_seq=state.completion_seq,
                    )
            self._store[new_thread_id] = CheckpointData(
                messages=new_messages,
                extra=base_extra,
                parent_thread_id=src_thread_id,
            )
            self._runs[new_thread_id] = new_runs
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/checkpointer/test_memory_fork.py tests/checkpointer/ -q`
Expected: all green.

- [ ] **Step 5: Commit**

```
git add cubepi/checkpointer/memory.py tests/checkpointer/test_memory_fork.py
git commit -m "feat(memory): snapshot + fork with set-based run_id selection"
```

---

## Phase 4 — SQLite backend

### Task 10: SQLite — schema additions + `BEGIN IMMEDIATE` + `busy_timeout`

**Files:**
- Modify: `cubepi/checkpointer/sqlite.py`
- Test: `tests/checkpointer/test_sqlite_schema.py` (new)

- [ ] **Step 1: Write failing schema test**

Create `tests/checkpointer/test_sqlite_schema.py`:
```python
import tempfile
from pathlib import Path

import pytest

from cubepi.checkpointer.sqlite import SQLiteCheckpointer


@pytest.mark.asyncio
async def test_runs_table_and_columns_exist_after_init():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.db"
        async with SQLiteCheckpointer(str(path)) as cp:
            cur = await cp._db.execute("PRAGMA table_info(runs)")
            cols = {row[1] for row in await cur.fetchall()}
            assert {
                "thread_id",
                "run_id",
                "claimed_at",
                "completed_at",
                "completion_seq",
            } <= cols

            cur = await cp._db.execute("PRAGMA table_info(messages)")
            cols = {row[1] for row in await cur.fetchall()}
            assert "run_id" in cols

            cur = await cp._db.execute("PRAGMA table_info(thread_extra)")
            cols = {row[1] for row in await cur.fetchall()}
            assert "parent_thread_id" in cols


@pytest.mark.asyncio
async def test_busy_timeout_is_set():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.db"
        async with SQLiteCheckpointer(str(path)) as cp:
            cur = await cp._db.execute("PRAGMA busy_timeout")
            (val,) = await cur.fetchone()
            assert val >= 5000
```

- [ ] **Step 2: Run test → expect failure**

Run: `uv run pytest tests/checkpointer/test_sqlite_schema.py -v`
Expected: `OperationalError: no such table: runs` (or busy_timeout = 0).

- [ ] **Step 3: Extend `SQLiteCheckpointer.__aenter__`**

In `cubepi/checkpointer/sqlite.py`, add the new DDL + `PRAGMA busy_timeout`
inside `__aenter__` after the existing tables are created. Use the
existing PRAGMA-probe pattern for ALTER TABLE on `messages` and
`thread_extra`:

```python
async def __aenter__(self) -> "SQLiteCheckpointer":
    self._db = await aiosqlite.connect(self._db_path)
    await self._db.execute("PRAGMA busy_timeout = 5000")
    await self._db.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  thread_id TEXT NOT NULL,"
        "  message_json TEXT NOT NULL,"
        "  created_at REAL NOT NULL DEFAULT (julianday('now'))"
        ")"
    )
    # ... thread_extra and thread_pending_request unchanged ...

    # New: runs table.
    await self._db.execute(
        "CREATE TABLE IF NOT EXISTS runs ("
        "  thread_id TEXT NOT NULL,"
        "  run_id TEXT NOT NULL,"
        "  claimed_at REAL NOT NULL DEFAULT (julianday('now')),"
        "  completed_at REAL,"
        "  completion_seq INTEGER,"
        "  PRIMARY KEY (thread_id, run_id)"
        ")"
    )
    await self._db.execute(
        "CREATE INDEX IF NOT EXISTS ix_runs_thread_completion "
        "ON runs (thread_id, completion_seq)"
    )

    # Add run_id column to messages if missing.
    cur = await self._db.execute("PRAGMA table_info(messages)")
    cols = {row[1] for row in await cur.fetchall()}
    if "run_id" not in cols:
        await self._db.execute(
            "ALTER TABLE messages ADD COLUMN run_id TEXT"
        )

    # Add parent_thread_id to thread_extra if missing.
    cur = await self._db.execute("PRAGMA table_info(thread_extra)")
    cols = {row[1] for row in await cur.fetchall()}
    if "parent_thread_id" not in cols:
        await self._db.execute(
            "ALTER TABLE thread_extra ADD COLUMN parent_thread_id TEXT"
        )

    # ... existing run_id backfill on thread_pending_request unchanged ...
    await self._db.commit()
    return self
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/checkpointer/test_sqlite_schema.py tests/checkpointer/test_sqlite.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```
git add cubepi/checkpointer/sqlite.py tests/checkpointer/test_sqlite_schema.py
git commit -m "feat(sqlite): add runs table, run_id + parent_thread_id columns, busy_timeout"
```

---

### Task 11: SQLite — promote all writers to `BEGIN IMMEDIATE` + surface lock timeout

**Files:**
- Modify: `cubepi/checkpointer/sqlite.py`
- Test: `tests/checkpointer/test_sqlite_concurrency.py` (new)

- [ ] **Step 1: Failing test for `CheckpointerLockTimeoutError`**

Create `tests/checkpointer/test_sqlite_concurrency.py`:
```python
import asyncio
import tempfile
from pathlib import Path

import aiosqlite
import pytest

from cubepi.checkpointer.exceptions import CheckpointerLockTimeoutError
from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.providers.base import TextContent, UserMessage


@pytest.mark.asyncio
async def test_lock_timeout_surfaces_as_typed_error(monkeypatch):
    """Hold a write lock from a second connection long enough that the
    busy_timeout expires; the checkpointer must raise
    CheckpointerLockTimeoutError, not the raw aiosqlite error."""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "x.db")
        async with SQLiteCheckpointer(path) as cp:
            # Make the timeout tiny so the test is fast.
            await cp._db.execute("PRAGMA busy_timeout = 100")
            other = await aiosqlite.connect(path)
            try:
                await other.execute("BEGIN IMMEDIATE")
                msg = UserMessage(content=[TextContent(text="x")])
                with pytest.raises(CheckpointerLockTimeoutError):
                    await cp.append("t", [msg])
            finally:
                await other.rollback()
                await other.close()
```

- [ ] **Step 2: Run test → expect failure**

Run: `uv run pytest tests/checkpointer/test_sqlite_concurrency.py -v`
Expected: raw `OperationalError` propagates.

- [ ] **Step 3: Add a writer-wrapper helper + use `BEGIN IMMEDIATE`**

In `cubepi/checkpointer/sqlite.py`, define a helper at module scope:

```python
from contextlib import asynccontextmanager

import aiosqlite

from cubepi.checkpointer.exceptions import CheckpointerLockTimeoutError


@asynccontextmanager
async def _writer_txn(db):
    """Wrap a writer transaction in BEGIN IMMEDIATE and surface
    SQLITE_BUSY as CheckpointerLockTimeoutError."""
    try:
        await db.execute("BEGIN IMMEDIATE")
    except aiosqlite.OperationalError as exc:
        if "lock" in str(exc).lower() or "busy" in str(exc).lower():
            raise CheckpointerLockTimeoutError(str(exc)) from exc
        raise
    try:
        yield
    except BaseException:
        await db.rollback()
        raise
    else:
        await db.commit()
```

Then update each writer method (`append`, `save_extra`,
`save_pending_request`) to use this helper INSTEAD OF the bare
`async with self._lock` + per-statement commits. Example for `append`:

```python
async def append(self, thread_id: str, messages: list[Message]) -> None:
    assert self._db is not None
    async with self._lock, _writer_txn(self._db):
        for msg in messages:
            msg_json = _serialize_message(msg)
            run_id = getattr(msg, "run_id", None)
            await self._db.execute(
                "INSERT INTO messages (thread_id, message_json, run_id) "
                "VALUES (?, ?, ?)",
                (thread_id, msg_json, run_id),
            )
```

Drop the inner `self._db.commit()` calls — `_writer_txn` commits on
clean exit. Do the same for `save_extra` and `save_pending_request`.

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/checkpointer/test_sqlite_concurrency.py tests/checkpointer/test_sqlite.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```
git add cubepi/checkpointer/sqlite.py tests/checkpointer/test_sqlite_concurrency.py
git commit -m "feat(sqlite): BEGIN IMMEDIATE writer txn + CheckpointerLockTimeoutError"
```

---

### Task 12: SQLite — `claim_run`, `mark_run_complete`, `load_pending`

**Files:**
- Modify: `cubepi/checkpointer/sqlite.py`
- Test: `tests/checkpointer/test_sqlite_runs.py` (new)

- [ ] **Step 1: Write tests (mirror the Memory test set)**

Create `tests/checkpointer/test_sqlite_runs.py` — copy the structure of
`tests/checkpointer/test_memory_runs.py`, replacing the factory with
SQLite via `tempfile.TemporaryDirectory`. Cover the same assertions:
claim+complete roundtrip; idempotent second mark; in-flight collision →
`RunAlreadyClaimedError`; completed collision → `RunAlreadyCompletedError`;
mark without claim → `RunNotClaimedError`; per-thread monotonic
`completion_seq`; append on completed run_id rejected; `load_pending`
tuple round-trip.

- [ ] **Step 2: Run test → expect failure**

Run: `uv run pytest tests/checkpointer/test_sqlite_runs.py -v`
Expected: missing methods.

- [ ] **Step 3: Implement the three methods**

Append to `cubepi/checkpointer/sqlite.py`:

```python
async def claim_run(self, thread_id: str, run_id: str) -> None:
    assert self._db is not None
    async with self._lock, _writer_txn(self._db):
        cur = await self._db.execute(
            "SELECT completed_at FROM runs "
            "WHERE thread_id = ? AND run_id = ?",
            (thread_id, run_id),
        )
        row = await cur.fetchone()
        if row is not None:
            completed_at = row[0]
            if completed_at is None:
                raise RunAlreadyClaimedError(
                    f"thread={thread_id} run={run_id} in flight"
                )
            raise RunAlreadyCompletedError(
                f"thread={thread_id} run={run_id} already completed"
            )
        await self._db.execute(
            "INSERT INTO runs (thread_id, run_id) VALUES (?, ?)",
            (thread_id, run_id),
        )

async def mark_run_complete(
    self, thread_id: str, run_id: str
) -> None:
    assert self._db is not None
    async with self._lock, _writer_txn(self._db):
        cur = await self._db.execute(
            "SELECT completed_at FROM runs "
            "WHERE thread_id = ? AND run_id = ?",
            (thread_id, run_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise RunNotClaimedError(
                f"thread={thread_id} run={run_id} has no claim row"
            )
        if row[0] is not None:
            return  # idempotent success
        cur = await self._db.execute(
            "SELECT COALESCE(MAX(completion_seq), 0) + 1 FROM runs "
            "WHERE thread_id = ? AND completion_seq IS NOT NULL",
            (thread_id,),
        )
        (next_seq,) = await cur.fetchone()
        await self._db.execute(
            "UPDATE runs SET completed_at = julianday('now'), "
            "completion_seq = ? WHERE thread_id = ? AND run_id = ?",
            (next_seq, thread_id, run_id),
        )

async def load_pending(
    self, thread_id: str
) -> tuple[HitlRequest, str | None] | None:
    assert self._db is not None
    async with self._lock:
        cur = await self._db.execute(
            "SELECT request_json, run_id FROM thread_pending_request "
            "WHERE thread_id = ?",
            (thread_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return HitlRequest.model_validate_json(row[0]), row[1]
```

Also update `append` to reject completed run_ids (defense in depth):

```python
async def append(self, thread_id: str, messages: list[Message]) -> None:
    assert self._db is not None
    run_ids = {m.run_id for m in messages if m.run_id is not None}
    async with self._lock, _writer_txn(self._db):
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            cur = await self._db.execute(
                f"SELECT run_id FROM runs WHERE thread_id = ? "
                f"AND run_id IN ({placeholders}) "
                f"AND completed_at IS NOT NULL",
                (thread_id, *run_ids),
            )
            done = await cur.fetchall()
            if done:
                bad = ", ".join(r[0] for r in done)
                raise RunAlreadyCompletedError(
                    f"append on completed run thread={thread_id} runs={bad}"
                )
        for msg in messages:
            msg_json = _serialize_message(msg)
            await self._db.execute(
                "INSERT INTO messages (thread_id, message_json, run_id) "
                "VALUES (?, ?, ?)",
                (thread_id, msg_json, getattr(msg, "run_id", None)),
            )
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/checkpointer/test_sqlite_runs.py tests/checkpointer/test_sqlite.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```
git add cubepi/checkpointer/sqlite.py tests/checkpointer/test_sqlite_runs.py
git commit -m "feat(sqlite): claim_run, mark_run_complete, load_pending"
```

---

### Task 13: SQLite — `snapshot()` + `fork()`

**Files:**
- Modify: `cubepi/checkpointer/sqlite.py`
- Test: `tests/checkpointer/test_sqlite_fork.py` (new)

- [ ] **Step 1: Write tests (mirror Memory fork test set)**

Create `tests/checkpointer/test_sqlite_fork.py` mirroring
`tests/checkpointer/test_memory_fork.py`, with the SQLite factory.

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/checkpointer/test_sqlite_fork.py -v`
Expected: `AttributeError: ... no attribute 'fork'`.

- [ ] **Step 3: Implement `snapshot()` + `fork()`**

Append to `cubepi/checkpointer/sqlite.py`:

```python
async def snapshot(
    self, thread_id: str, *, after_run_id: str
) -> list[Message]:
    assert self._db is not None
    async with self._lock:
        cur = await self._db.execute(
            "SELECT completion_seq FROM runs "
            "WHERE thread_id = ? AND run_id = ?",
            (thread_id, after_run_id),
        )
        row = await cur.fetchone()
        if row is None or row[0] is None:
            raise RunNotCompletedError(
                f"thread={thread_id} run={after_run_id} not completed"
            )
        cutoff = row[0]
        cur = await self._db.execute(
            "SELECT message_json FROM messages WHERE thread_id = ? "
            "AND (run_id IS NULL OR run_id IN ("
            "  SELECT run_id FROM runs WHERE thread_id = ? "
            "  AND completion_seq IS NOT NULL "
            "  AND completion_seq <= ?"
            ")) ORDER BY id",
            (thread_id, thread_id, cutoff),
        )
        rows = await cur.fetchall()
        return [_deserialize_message(json.loads(r[0])) for r in rows]

async def fork(
    self,
    src_thread_id: str,
    new_thread_id: str,
    *,
    after_run_id: str,
    metadata: JsonObject | None = None,
) -> None:
    assert self._db is not None
    async with self._lock, _writer_txn(self._db):
        # Source existence: messages OR thread_extra.
        cur = await self._db.execute(
            "SELECT 1 FROM messages WHERE thread_id = ? LIMIT 1",
            (src_thread_id,),
        )
        src_has_msg = await cur.fetchone() is not None
        cur = await self._db.execute(
            "SELECT 1 FROM thread_extra WHERE thread_id = ?",
            (src_thread_id,),
        )
        src_has_extra = await cur.fetchone() is not None
        if not (src_has_msg or src_has_extra):
            raise ThreadNotFoundError(f"thread={src_thread_id}")
        # Destination collision.
        cur = await self._db.execute(
            "SELECT 1 FROM messages WHERE thread_id = ? LIMIT 1",
            (new_thread_id,),
        )
        if await cur.fetchone():
            raise ThreadAlreadyExistsError(f"thread={new_thread_id}")
        cur = await self._db.execute(
            "SELECT 1 FROM thread_extra WHERE thread_id = ?",
            (new_thread_id,),
        )
        if await cur.fetchone():
            raise ThreadAlreadyExistsError(f"thread={new_thread_id}")
        # Cutoff.
        cur = await self._db.execute(
            "SELECT completion_seq FROM runs "
            "WHERE thread_id = ? AND run_id = ?",
            (src_thread_id, after_run_id),
        )
        row = await cur.fetchone()
        if row is None or row[0] is None:
            raise RunNotCompletedError(
                f"thread={src_thread_id} run={after_run_id} not completed"
            )
        cutoff = row[0]
        # Copy messages.
        await self._db.execute(
            "INSERT INTO messages (thread_id, run_id, message_json) "
            "SELECT ?, run_id, message_json FROM messages "
            "WHERE thread_id = ? AND ("
            "  run_id IS NULL OR run_id IN ("
            "    SELECT run_id FROM runs WHERE thread_id = ? "
            "    AND completion_seq IS NOT NULL "
            "    AND completion_seq <= ?"
            "  )"
            ") ORDER BY id",
            (new_thread_id, src_thread_id, src_thread_id, cutoff),
        )
        # Copy runs.
        await self._db.execute(
            "INSERT INTO runs (thread_id, run_id, claimed_at, "
            "completed_at, completion_seq) "
            "SELECT ?, run_id, claimed_at, completed_at, completion_seq "
            "FROM runs WHERE thread_id = ? "
            "AND completion_seq IS NOT NULL AND completion_seq <= ?",
            (new_thread_id, src_thread_id, cutoff),
        )
        # Build merged extra.
        cur = await self._db.execute(
            "SELECT extra_json FROM thread_extra WHERE thread_id = ?",
            (src_thread_id,),
        )
        row = await cur.fetchone()
        merged_extra = json.loads(row[0]) if row else {}
        if metadata is not None:
            merged_extra["fork"] = json.loads(json.dumps(metadata))
        await self._db.execute(
            "INSERT INTO thread_extra (thread_id, extra_json, "
            "parent_thread_id) VALUES (?, ?, ?)",
            (new_thread_id, json.dumps(merged_extra), src_thread_id),
        )
```

Also extend `load()` to populate `CheckpointData.parent_thread_id` from
`thread_extra.parent_thread_id`:

```python
async def load(self, thread_id: str) -> CheckpointData | None:
    # ... existing code that fetches messages and extra ...
    cur = await self._db.execute(
        "SELECT extra_json, parent_thread_id FROM thread_extra "
        "WHERE thread_id = ?",
        (thread_id,),
    )
    extra_row = await cur.fetchone()
    parent = extra_row[1] if extra_row else None
    # ... extra = json.loads(...) ...
    return CheckpointData(messages=messages, extra=extra, parent_thread_id=parent)
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/checkpointer/test_sqlite_fork.py tests/checkpointer/test_sqlite.py -q`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/checkpointer/sqlite.py tests/checkpointer/test_sqlite_fork.py
git commit -m "feat(sqlite): snapshot + fork with set-based selection"
```

---

## Phase 5 — Postgres backend

The Postgres tasks mirror SQLite with these differences:
- SQLAlchemy model declarations in `cubepi/checkpointer/postgres/models.py`
- Schema version bump v3 → v4 + alembic migration template
- Use `pg_advisory_xact_lock(hashtext($thread_id))` for per-thread locking
- Use `INSERT INTO cubepi_threads ON CONFLICT DO NOTHING` for lazy create
- `cubepi_runs` table is `HASH (thread_id)` partitioned, FK with `ON DELETE CASCADE`

> All Postgres tests run against the bundled docker fixture (existing
> `tests/checkpointer/conftest.py` infrastructure).

### Task 14: Postgres — extend models + bump schema to v4

**Files:**
- Modify: `cubepi/checkpointer/postgres/models.py`
- Modify: `cubepi/checkpointer/postgres/alembic_helpers.py` (add migration template)
- Test: `tests/checkpointer/test_postgres_schema.py` (new)

- [ ] **Step 1: Write schema-version + table-shape tests**

Create `tests/checkpointer/test_postgres_schema.py`:
```python
import pytest

from cubepi.checkpointer.postgres.models import EXPECTED_SCHEMA_VERSION


def test_expected_schema_version_is_4():
    assert EXPECTED_SCHEMA_VERSION == 4


@pytest.mark.asyncio
async def test_cubepi_runs_table_present(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        rows = await cp._pool.fetch(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'cubepi_runs' ORDER BY ordinal_position"
        )
        cols = {r["column_name"] for r in rows}
        assert {
            "thread_id",
            "run_id",
            "claimed_at",
            "completed_at",
            "completion_seq",
        } <= cols


@pytest.mark.asyncio
async def test_cubepi_messages_has_run_id_column(pg_v4_dsn):
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        row = await cp._pool.fetchrow(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'cubepi_messages' AND column_name = 'run_id'"
        )
        assert row is not None
```

(`clean_db` fixture from `tests/checkpointer/conftest.py` yields a DSN
string; `PostgresCheckpointer(dsn)` opens its own pool on `__aenter__`.)

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/checkpointer/test_postgres_schema.py -v`
Expected: schema version mismatch / missing column.

- [ ] **Step 3: Update `cubepi/checkpointer/postgres/models.py`**

- Bump `EXPECTED_SCHEMA_VERSION = 4`.
- Add `run_id` column to `CubepiMessage`:
  ```python
  run_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
  ```
  and add an index `Index("ix_cubepi_messages_thread_run", "thread_id", "run_id")`.
- Add new `CubepiRun` model:
  ```python
  class CubepiRun(CubepiBase):
      __tablename__ = "cubepi_runs"
      __table_args__ = (
          sa.Index("ix_cubepi_runs_thread_seq", "thread_id", "completion_seq"),
          {"postgresql_partition_by": "HASH (thread_id)"},
      )
      thread_id: Mapped[str] = mapped_column(
          sa.Text,
          sa.ForeignKey("cubepi_threads.thread_id", ondelete="CASCADE"),
          primary_key=True,
      )
      run_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
      claimed_at: Mapped[_dt.datetime] = mapped_column(
          sa.TIMESTAMP(timezone=True),
          nullable=False,
          server_default=sa.text("now()"),
      )
      completed_at: Mapped[_dt.datetime | None] = mapped_column(
          sa.TIMESTAMP(timezone=True), nullable=True
      )
      completion_seq: Mapped[int | None] = mapped_column(
          sa.BigInteger, nullable=True
      )
  ```

- [ ] **Step 4: Update alembic helper to emit the v3→v4 migration**

In `cubepi/checkpointer/postgres/alembic_helpers.py`, add a function
`upgrade_v3_to_v4(op)` that:
- adds `run_id TEXT NULL` to `cubepi_messages` (partitioned-table
  alter: hold a brief access-exclusive on parent + each partition);
- creates the index `(thread_id, run_id)`;
- creates the partitioned `cubepi_runs` parent + N partitions (use
  `PARTITION_COUNT` constant from `models.py`);
- bumps `cubepi_schema_version` to 4.

Provide a downgrade that drops the table + column.

- [ ] **Step 5: Extend `_setup_schema` to v4 + add `pg_v4_dsn` fixture**

In `tests/checkpointer/test_postgres.py`, the existing `_setup_schema`
helper (`test_postgres.py:177`) builds the v3 schema. **Edit it
in place** to ALSO apply the v4 changes from Task 14:
- `ALTER TABLE cubepi_messages ADD COLUMN run_id TEXT NULL`
- `CREATE INDEX ix_cubepi_messages_thread_run ON cubepi_messages (thread_id, run_id)`
- Create the partitioned `cubepi_runs` table + partitions (mirror
  `create_message_partitions_op` from `alembic_helpers.py`)
- Bump `cubepi_schema_version` from 3 to 4

**Do NOT move the helper** — leaving it in `test_postgres.py` means
all existing call sites in that file keep working. Update the
existing `def test_expected_schema_version` assertion in that file
from `== 3` to `== 4` (and the parallel one in `test_mysql.py`).

Then add the new fixture in `tests/checkpointer/conftest.py`:
```python
# tests/checkpointer/conftest.py
import pytest_asyncio
from tests.checkpointer.test_postgres import _setup_schema as _setup_pg_schema


@pytest_asyncio.fixture
async def pg_v4_dsn(clean_db: str):
    await _setup_pg_schema(clean_db)
    yield clean_db
```

(Importing from a sibling test module is unusual but acceptable for
this kind of shared schema helper; the alternative — duplicating the
~80 lines of DDL into conftest.py — is worse. If pytest complains
about the import at collection time, move the helper to
`tests/checkpointer/_schema.py` and import from there in both places.)

PG tests that exercise a PostgresCheckpointer instance MUST use
`pg_v4_dsn` instead of raw `clean_db`. Tests that only assert
information_schema shape can also use `pg_v4_dsn` (the schema is
already applied; opening the checkpointer is optional).

Update the table-shape tests at the top of Task 14 to use
`pg_v4_dsn`:
```python
async def test_cubepi_runs_table_present(pg_v4_dsn):
    conn = await asyncpg.connect(pg_v4_dsn)
    try:
        rows = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'cubepi_runs'"
        )
        cols = {r["column_name"] for r in rows}
        assert {
            "thread_id", "run_id", "claimed_at",
            "completed_at", "completion_seq",
        } <= cols
    finally:
        await conn.close()
```

(No `PostgresCheckpointer` needed here — we're only inspecting schema.)

- [ ] **Step 6: Run tests → expect pass**

Run: `uv run pytest tests/checkpointer/test_postgres_schema.py tests/checkpointer/test_postgres.py -q`
Expected: green.

- [ ] **Step 7: Commit**

```
git add cubepi/checkpointer/postgres/ tests/checkpointer/test_postgres_schema.py tests/checkpointer/conftest.py
git commit -m "feat(postgres): schema v3→v4 — add run_id column + cubepi_runs table"
```

---

### Task 15: Postgres — `claim_run` + persist `Message.run_id` in `append()`

**Files:**
- Modify: `cubepi/checkpointer/postgres/checkpointer.py`
- Test: `tests/checkpointer/test_postgres_runs.py` (new)

**CRITICAL:** Task 14 added the `run_id` column to `cubepi_messages`,
but the existing `PostgresCheckpointer.append()` SQL
(`postgres/checkpointer.py:201`) writes only `(thread_id, seq, role,
metadata, payload)` — it must be extended to write `run_id` too, or
fork's set-based selection will treat every new message as legacy
(`run_id IS NULL`).

- [ ] **Step 1: Failing tests (mirror SQLite run tests) + run_id persistence**

Create `tests/checkpointer/test_postgres_runs.py` with the same
scenarios as `tests/checkpointer/test_sqlite_runs.py`, plus these
Postgres-specific tests:

```python
@pytest.mark.asyncio
async def test_append_persists_run_id_into_column(pg_v4_dsn):
    """Regression for plan-review CRITICAL: append() MUST write
    Message.run_id into cubepi_messages.run_id."""
    from cubepi.providers.base import TextContent, UserMessage
    from cubepi.checkpointer.postgres.checkpointer import PostgresCheckpointer

    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        await cp.claim_run("t", "R1")
        msg = UserMessage(content=[TextContent(text="hi")], run_id="R1")
        await cp.append("t", [msg])
        row = await cp._pool.fetchrow(
            "SELECT run_id FROM cubepi_messages WHERE thread_id = $1",
            "t",
        )
        assert row["run_id"] == "R1"


@pytest.mark.asyncio
async def test_append_rejects_completed_run_id(pg_v4_dsn):
    from cubepi.checkpointer.exceptions import RunAlreadyCompletedError
    from cubepi.providers.base import TextContent, UserMessage
    from cubepi.checkpointer.postgres.checkpointer import PostgresCheckpointer

    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        await cp.claim_run("t", "R1")
        await cp.mark_run_complete("t", "R1")
        msg = UserMessage(content=[TextContent(text="late")], run_id="R1")
        with pytest.raises(RunAlreadyCompletedError):
            await cp.append("t", [msg])


@pytest.mark.asyncio
async def test_claim_run_creates_threads_row_lazily(pg_v4_dsn):
    from cubepi.checkpointer.postgres.checkpointer import PostgresCheckpointer

    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        await cp.claim_run("new_thread", "R1")
        row = await cp._pool.fetchrow(
            "SELECT thread_id FROM cubepi_threads WHERE thread_id = $1",
            "new_thread",
        )
        assert row is not None
```

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/checkpointer/test_postgres_runs.py -v`
Expected: method missing.

- [ ] **Step 3a: Update `append()` to write `run_id` and reject completed run_ids**

In `cubepi/checkpointer/postgres/checkpointer.py:201` (the existing
`INSERT INTO cubepi_messages` statement inside `append()`), extend the
column list and parameter tuple:

```python
# Before any insert, defend against append-on-completed.
run_ids = {m.run_id for m in messages if m.run_id is not None}
if run_ids:
    completed = await conn.fetch(
        "SELECT run_id FROM cubepi_runs WHERE thread_id = $1 "
        "AND run_id = ANY($2::text[]) AND completed_at IS NOT NULL",
        thread_id, list(run_ids),
    )
    if completed:
        bad = ", ".join(r["run_id"] for r in completed)
        raise RunAlreadyCompletedError(
            f"append on completed run thread={thread_id} runs={bad}"
        )

for i, msg in enumerate(messages):
    seq = last_seq + i + 1
    await conn.execute(
        "INSERT INTO cubepi_messages "
        "(thread_id, seq, role, run_id, metadata, payload) "
        "VALUES ($1, $2, $3, $4, $5, $6)",
        thread_id, seq, _role_of(msg),
        getattr(msg, "run_id", None),
        json.dumps(msg.metadata),
        msgpack.packb(msg.model_dump(), use_bin_type=True),
    )
```

Both `tests/checkpointer/test_postgres_runs.py::
test_append_persists_run_id_into_column` and
`::test_append_rejects_completed_run_id` exercise this.

- [ ] **Step 3b: Implement `claim_run` with lazy threads create**

Add to `PostgresCheckpointer`:
```python
import asyncpg

async def claim_run(self, thread_id: str, run_id: str) -> None:
    async with self._pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext($1))", thread_id
        )
        # Lazy create the threads row.
        await conn.execute(
            "INSERT INTO cubepi_threads (thread_id) VALUES ($1) "
            "ON CONFLICT (thread_id) DO NOTHING",
            thread_id,
        )
        try:
            await conn.execute(
                "INSERT INTO cubepi_runs (thread_id, run_id) "
                "VALUES ($1, $2)",
                thread_id, run_id,
            )
        except asyncpg.UniqueViolationError:
            row = await conn.fetchrow(
                "SELECT completed_at FROM cubepi_runs "
                "WHERE thread_id = $1 AND run_id = $2",
                thread_id, run_id,
            )
            if row is None or row["completed_at"] is None:
                raise RunAlreadyClaimedError(
                    f"thread={thread_id} run={run_id} in flight"
                )
            raise RunAlreadyCompletedError(
                f"thread={thread_id} run={run_id} already completed"
            )
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/checkpointer/test_postgres_runs.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/checkpointer/postgres/checkpointer.py tests/checkpointer/test_postgres_runs.py
git commit -m "feat(postgres): claim_run with lazy threads-row create"
```

---

### Task 16: Postgres — `mark_run_complete` + `load_pending`

**Files:**
- Modify: `cubepi/checkpointer/postgres/checkpointer.py`
- Test: extend `tests/checkpointer/test_postgres_runs.py` with idempotent-retry test (mirror SQLite Task 12)

- [ ] **Step 1: Add failing idempotency + load_pending tests**

In `tests/checkpointer/test_postgres_runs.py`, add tests for:
- `mark_run_complete` then second call returns success (no raise)
- `RunNotClaimedError` when no claim exists
- `completion_seq` strictly monotonic across runs on the same thread
- `load_pending` returns `(HitlRequest, run_id)` tuple atomically

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/checkpointer/test_postgres_runs.py -v`
Expected: missing methods.

- [ ] **Step 3: Implement methods**

Add to `PostgresCheckpointer`:
```python
async def mark_run_complete(self, thread_id: str, run_id: str) -> None:
    async with self._pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext($1))", thread_id
        )
        row = await conn.fetchrow(
            "SELECT completed_at FROM cubepi_runs "
            "WHERE thread_id = $1 AND run_id = $2",
            thread_id, run_id,
        )
        if row is None:
            raise RunNotClaimedError(
                f"thread={thread_id} run={run_id} has no claim row"
            )
        if row["completed_at"] is not None:
            return  # idempotent
        next_seq = await conn.fetchval(
            "SELECT COALESCE(MAX(completion_seq), 0) + 1 "
            "FROM cubepi_runs WHERE thread_id = $1 "
            "AND completion_seq IS NOT NULL",
            thread_id,
        )
        await conn.execute(
            "UPDATE cubepi_runs SET completed_at = now(), "
            "completion_seq = $3 "
            "WHERE thread_id = $1 AND run_id = $2",
            thread_id, run_id, next_seq,
        )

async def load_pending(
    self, thread_id: str
) -> tuple[HitlRequest, str | None] | None:
    async with self._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT pending_request, run_id FROM cubepi_threads "
            "WHERE thread_id = $1",
            thread_id,
        )
    if row is None or row["pending_request"] is None:
        return None
    req = HitlRequest.model_validate(row["pending_request"])
    return req, row["run_id"]
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/checkpointer/test_postgres_runs.py -q`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/checkpointer/postgres/checkpointer.py tests/checkpointer/test_postgres_runs.py
git commit -m "feat(postgres): mark_run_complete (idempotent) + load_pending"
```

---

### Task 17: Postgres — `snapshot()` + `fork()` (threads row first)

**Files:**
- Modify: `cubepi/checkpointer/postgres/checkpointer.py`
- Test: `tests/checkpointer/test_postgres_fork.py` (new)

- [ ] **Step 1: Tests mirror Memory/SQLite fork suite**

Create `tests/checkpointer/test_postgres_fork.py` covering the same
scenarios; add one Postgres-specific concurrency test:
```python
@pytest.mark.asyncio
async def test_fork_blocks_concurrent_appends_on_src(pg_v4_dsn):
    """Two coroutines: fork holds advisory lock; concurrent append
    on src serializes after fork commits. No half-copied messages."""
    # ... use asyncio.gather to interleave; assert ordering ...
```

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/checkpointer/test_postgres_fork.py -v`
Expected: missing fork method.

- [ ] **Step 3: Implement `snapshot()` + `fork()` per spec §3.9 order**

```python
async def snapshot(
    self, thread_id: str, *, after_run_id: str
) -> list[Message]:
    async with self._pool.acquire() as conn:
        # Spec §3.10: ThreadNotFoundError takes precedence over
        # RunNotCompletedError. Validate the source thread first.
        thread_exists = await conn.fetchval(
            "SELECT 1 FROM cubepi_threads WHERE thread_id = $1",
            thread_id,
        )
        if not thread_exists:
            raise ThreadNotFoundError(f"thread={thread_id}")
        cutoff = await conn.fetchval(
            "SELECT completion_seq FROM cubepi_runs "
            "WHERE thread_id = $1 AND run_id = $2",
            thread_id, after_run_id,
        )
        if cutoff is None:
            raise RunNotCompletedError(
                f"thread={thread_id} run={after_run_id} not completed"
            )
        rows = await conn.fetch(
            "SELECT role, metadata, payload, seq FROM cubepi_messages "
            "WHERE thread_id = $1 AND ("
            "  run_id IS NULL OR run_id IN ("
            "    SELECT run_id FROM cubepi_runs WHERE thread_id = $1 "
            "    AND completion_seq IS NOT NULL AND completion_seq <= $2"
            "  )"
            ") ORDER BY seq",
            thread_id, cutoff,
        )
        # Decode using the existing pattern from load() in this file
        # (msgpack unpack + role-to-class). Inline here rather than
        # invent a `_row_to_message` helper that doesn't exist.
        result: list[Message] = []
        for r in rows:
            cls = _ROLE_TO_CLS[r["role"]]
            data = msgpack.unpackb(r["payload"], raw=False)
            result.append(cls.model_validate(data))
        return result

async def fork(
    self,
    src_thread_id: str,
    new_thread_id: str,
    *,
    after_run_id: str,
    metadata: JsonObject | None = None,
) -> None:
    async with self._pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext($1))", src_thread_id
        )
        # Source existence FIRST (spec §3.10 error precedence).
        src_extra = await conn.fetchval(
            "SELECT extra FROM cubepi_threads WHERE thread_id = $1",
            src_thread_id,
        )
        if src_extra is None:
            raise ThreadNotFoundError(f"thread={src_thread_id}")
        # Cutoff + RunNotCompletedError.
        cutoff = await conn.fetchval(
            "SELECT completion_seq FROM cubepi_runs "
            "WHERE thread_id = $1 AND run_id = $2",
            src_thread_id, after_run_id,
        )
        if cutoff is None:
            raise RunNotCompletedError(
                f"thread={src_thread_id} run={after_run_id} not completed"
            )
        merged = json.loads(json.dumps(src_extra))
        if metadata is not None:
            merged["fork"] = json.loads(json.dumps(metadata))
        # INSERT thread row first (FK order).
        try:
            await conn.execute(
                "INSERT INTO cubepi_threads "
                "(thread_id, parent_thread_id, forked_at_seq, extra) "
                "VALUES ($1, $2, NULL, $3::jsonb)",
                new_thread_id, src_thread_id, json.dumps(merged),
            )
        except asyncpg.UniqueViolationError:
            raise ThreadAlreadyExistsError(f"thread={new_thread_id}")
        # Copy messages.
        await conn.execute(
            "INSERT INTO cubepi_messages "
            "(thread_id, seq, role, run_id, metadata, payload) "
            "SELECT $1, seq, role, run_id, metadata, payload "
            "FROM cubepi_messages WHERE thread_id = $2 AND ("
            "  run_id IS NULL OR run_id IN ("
            "    SELECT run_id FROM cubepi_runs WHERE thread_id = $2 "
            "    AND completion_seq IS NOT NULL AND completion_seq <= $3"
            "  )"
            ") ORDER BY seq",
            new_thread_id, src_thread_id, cutoff,
        )
        # Copy runs.
        await conn.execute(
            "INSERT INTO cubepi_runs "
            "(thread_id, run_id, claimed_at, completed_at, completion_seq) "
            "SELECT $1, run_id, claimed_at, completed_at, completion_seq "
            "FROM cubepi_runs WHERE thread_id = $2 "
            "AND completion_seq IS NOT NULL AND completion_seq <= $3",
            new_thread_id, src_thread_id, cutoff,
        )
        # Trailing forked_at_seq update.
        await conn.execute(
            "UPDATE cubepi_threads SET forked_at_seq = ("
            "  SELECT MAX(seq) FROM cubepi_messages WHERE thread_id = $1"
            ") WHERE thread_id = $1",
            new_thread_id,
        )
```

Also update `PostgresCheckpointer.load()` to populate
`CheckpointData.parent_thread_id` from `cubepi_threads.parent_thread_id`.

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/checkpointer/test_postgres_fork.py tests/checkpointer/test_postgres.py -q`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/checkpointer/postgres/checkpointer.py tests/checkpointer/test_postgres_fork.py
git commit -m "feat(postgres): snapshot + fork (threads-first, set-based)"
```

---

## Phase 6 — MySQL backend

MySQL mirrors Postgres with:
- `SELECT … FOR UPDATE` on `cubepi_threads` row instead of advisory lock
- `INSERT … ON DUPLICATE KEY UPDATE thread_id = thread_id` for lazy create
  (NOT `INSERT IGNORE` — spec §3.9 explicit)
- `cubepi_runs` is `KEY (thread_id)` partitioned, no FK
- `aiomysql.IntegrityError` for PK conflict detection

### Task 18: MySQL — extend models + bump schema to v4

**Files:**
- Modify: `cubepi/checkpointer/mysql/models.py`
- Modify: `cubepi/checkpointer/mysql/alembic_helpers.py`
- Modify: `tests/checkpointer/conftest.py` (add `mysql_v4_dsn` fixture)
- Test: `tests/checkpointer/test_mysql_schema.py` (new)

Steps parallel Task 14 with MySQL syntax:
- Bump `EXPECTED_SCHEMA_VERSION = 4`.
- Add `run_id VARCHAR(255) NULL` to `CubepiMessage` + composite index.
- Add `CubepiRun` model (KEY-partitioned by thread_id, no FK).
- Extend `_setup_schema(dsn)` in `test_mysql.py` (mirror of the
  Postgres helper at `test_postgres.py:177`) **in place** to apply
  the v4 DDL. Update the parallel `test_expected_schema_version`
  assertion in that file from `== 3` to `== 4`.

  Then add the `mysql_v4_dsn` fixture in
  `tests/checkpointer/conftest.py`, importing from the existing
  module (same pattern as `pg_v4_dsn` in Task 14):
  ```python
  from tests.checkpointer.test_mysql import _setup_schema as _setup_mysql_schema


  @pytest_asyncio.fixture
  async def mysql_v4_dsn(clean_mysql_db: str):
      await _setup_mysql_schema(clean_mysql_db)
      yield clean_mysql_db
  ```
- Provide v3→v4 alembic helper template (no PARTITION BY in
  SQLAlchemy declarative; emit raw DDL).

- [ ] Steps 1–7 follow Task 14 pattern. All test snippets use
  `mysql_v4_dsn` (not raw `clean_mysql_db`) when constructing a
  `MySQLCheckpointer`. Run `tests/checkpointer/test_mysql_*` for green.

Commit: `feat(mysql): schema v3→v4 — add run_id column + cubepi_runs table`.

---

### Task 19: MySQL — `claim_run`, `mark_run_complete`, `load_pending` + persist `Message.run_id`

**Files:**
- Modify: `cubepi/checkpointer/mysql/checkpointer.py`
- Test: `tests/checkpointer/test_mysql_runs.py` (new)

**CRITICAL — same as Task 15:** Existing
`MySQLCheckpointer.append()` SQL at
`mysql/checkpointer.py:241` writes `(thread_id, seq, role, metadata,
payload)` without `run_id`. Extend it. Add corresponding test
asserting `cubepi_messages.run_id` is populated, mirroring the
Postgres Task 15 Step 1 tests (`test_append_persists_run_id_into_column`,
`test_append_rejects_completed_run_id`).

Mirror Tasks 15–16 with MySQL syntax. Key differences:

```python
async def claim_run(self, thread_id: str, run_id: str) -> None:
    async with self._pool.acquire() as conn:
        await conn.begin()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO cubepi_threads (thread_id) VALUES (%s) "
                    "ON DUPLICATE KEY UPDATE thread_id = thread_id",
                    (thread_id,),
                )
                await cur.execute(
                    "SELECT thread_id FROM cubepi_threads "
                    "WHERE thread_id = %s FOR UPDATE",
                    (thread_id,),
                )
                try:
                    await cur.execute(
                        "INSERT INTO cubepi_runs (thread_id, run_id) "
                        "VALUES (%s, %s)",
                        (thread_id, run_id),
                    )
                except aiomysql.IntegrityError:
                    await cur.execute(
                        "SELECT completed_at FROM cubepi_runs "
                        "WHERE thread_id = %s AND run_id = %s",
                        (thread_id, run_id),
                    )
                    row = await cur.fetchone()
                    if row is None or row[0] is None:
                        await conn.rollback()
                        raise RunAlreadyClaimedError(
                            f"thread={thread_id} run={run_id} in flight"
                        )
                    await conn.rollback()
                    raise RunAlreadyCompletedError(
                        f"thread={thread_id} run={run_id} already completed"
                    )
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
```

`mark_run_complete` and `load_pending` follow the SQL forms shown in
Task 16 with MySQL placeholders.

Commit: `feat(mysql): claim_run, mark_run_complete, load_pending`.

---

### Task 20: MySQL — `snapshot()` + `fork()`

**Files:**
- Modify: `cubepi/checkpointer/mysql/checkpointer.py`
- Test: `tests/checkpointer/test_mysql_fork.py` (new)

Mirror Task 17 with MySQL syntax. Use `SELECT … FROM cubepi_threads
WHERE thread_id = %s FOR UPDATE` as the per-thread fence in `fork()`.

Commit: `feat(mysql): snapshot + fork (threads-first, set-based)`.

---

## Phase 7 — Agent integration

This is where the persistence primitives wire into the agent loop.

### Task 21: `Agent(messages=…)` constructor pre-seed with deep copy

**Files:**
- Modify: `cubepi/agent/agent.py`
- Test: `tests/agent/test_agent_messages_kw.py` (new)

- [ ] **Step 1: Failing tests for constructor + deep-copy isolation**

Create `tests/agent/test_agent_messages_kw.py`:
```python
import pytest

from cubepi.agent.agent import Agent
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider


def _agent(**kw):
    return Agent(model=FauxProvider().model("faux-model"), **kw)


def test_messages_kw_none_keeps_default():
    a = _agent()
    assert a.state.messages == []


def test_messages_kw_seeds_initial_history():
    msgs = [UserMessage(content=[TextContent(text="hi")])]
    a = _agent(messages=msgs)
    assert len(a.state.messages) == 1


def test_messages_kw_conflicts_with_thread_id_checkpointer():
    from cubepi.checkpointer.memory import MemoryCheckpointer

    msgs = [UserMessage(content=[TextContent(text="hi")])]
    with pytest.raises(ValueError):
        _agent(
            messages=msgs,
            thread_id="t",
            checkpointer=MemoryCheckpointer(),
        )


def test_messages_kw_deep_copies_all_three_variants():
    user = UserMessage(content=[TextContent(text="u")], metadata={"k": "v"})
    assistant = AssistantMessage(
        content=[ToolCall(id="c1", name="t", arguments={"k": [1, 2]})],
        metadata={},
    )
    tool = ToolResultMessage(
        tool_call_id="c1", tool_name="t",
        content=[TextContent(text="r")], metadata={"x": 1},
    )
    a = _agent(messages=[user, assistant, tool])
    # Mutate originals.
    user.metadata["k"] = "MUT"
    assistant.content[0].arguments["k"].append(99)
    tool.metadata["x"] = 999
    # Internal copies untouched.
    assert a.state.messages[0].metadata["k"] == "v"
    assert a.state.messages[1].content[0].arguments["k"] == [1, 2]
    assert a.state.messages[2].metadata["x"] == 1
    # Mutate agent's copies; originals untouched.
    a.state.messages[0].metadata["k"] = "AGENT"
    assert user.metadata["k"] == "MUT"
```

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/agent/test_agent_messages_kw.py -v`
Expected: `TypeError: __init__() got an unexpected keyword argument 'messages'`.

- [ ] **Step 3: Add the constructor arg + retain the model reference**

In `cubepi/agent/agent.py`, locate `Agent.__init__`. Add the keyword-only
parameter `messages: Sequence[Message] | None = None` at the end. Inside,
add the messages-handling block AND a one-line retention of the original
`model` argument (needed by `fork_once()` in Task 31):

```python
# Retain the original `model` argument so fork_once() can construct
# a transient Agent reusing the same model.
self._model = model

if messages is not None:
    if thread_id is not None and checkpointer is not None:
        raise ValueError(
            "Agent(messages=...) cannot be combined with "
            "thread_id + checkpointer (pre-seed conflicts with lazy "
            "load). Construct an ephemeral Agent without those for "
            "fork_once-style usage."
        )
    seeded = [m.model_copy(deep=True) for m in messages]
    # AgentState exposes `messages` as a property — use the public
    # setter (which delegates to `_messages`); don't poke `_messages`
    # directly. The setter is at agent.py:117 today.
    self._state.messages = list(seeded)
```

(Make sure to import `Sequence` from `collections.abc` if not already
imported, and `Message` from `cubepi.providers.base` likewise.)

**Note:** the spec §3.7a forbids touching private attributes from
fork_once. Going through the public `messages` setter on AgentState is
the supported path; do not reach into `_messages` directly.

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/agent/test_agent_messages_kw.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/agent/agent.py tests/agent/test_agent_messages_kw.py
git commit -m "feat(agent): add messages=... pre-seed kw with deep-copy isolation"
```

---

### Task 22: `Agent.prompt()` accepts `run_id`, returns `str`, sets `active_run_id`

**Files:**
- Modify: `cubepi/agent/agent.py`
- Test: `tests/agent/test_agent_run_id.py` (new)

- [ ] **Step 1: Failing tests**

Create `tests/agent/test_agent_run_id.py`:
```python
import uuid

import pytest

from cubepi.agent.agent import Agent
from cubepi.providers.faux import FauxProvider


def _agent(**kw):
    return Agent(model=_ok_faux().model("faux-model"), **kw)


@pytest.mark.asyncio
async def test_prompt_returns_supplied_run_id():
    a = _agent()
    got = await a.prompt("hello", run_id="R1")
    assert got == "R1"


@pytest.mark.asyncio
async def test_prompt_generates_run_id_when_none():
    a = _agent()
    got = await a.prompt("hello")
    assert isinstance(got, str) and len(got) >= 8


@pytest.mark.asyncio
async def test_prompt_sets_then_clears_active_run_id_on_clean_return():
    a = _agent()
    assert a.state.active_run_id is None
    await a.prompt("hello", run_id="R1")
    assert a.state.active_run_id is None  # cleared on clean return


@pytest.mark.asyncio
async def test_prompt_leaves_active_run_id_set_on_raise(monkeypatch):
    """Spec §3.7 + Task 22: active_run_id must be LEFT SET on any
    propagating failure after claim. Provider exceptions are caught
    by Agent._run_with_lifecycle and synthesized into error messages
    — they do NOT propagate. To exercise the propagation path at
    THIS task (before Task 26 introduces CompletionMarkerFailedError),
    monkeypatch self._run_prompt to raise after entry."""
    a = Agent(model=_ok_faux().model("faux-model"))

    async def _boom(*args, **kwargs):
        # Raises AFTER prompt() sets active_run_id and BEFORE the
        # `else:` clear runs.
        raise RuntimeError("boom")

    monkeypatch.setattr(a, "_run_prompt", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        await a.prompt("hello", run_id="R1")
    # Spec §3.7: left set so except-block readers can recover it.
    assert a.state.active_run_id == "R1"
```

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/agent/test_agent_run_id.py -v`
Expected: prompt returns None or TypeError on `run_id`.

- [ ] **Step 3: Modify `Agent.prompt`**

In `cubepi/agent/agent.py`, edit `prompt`:
```python
async def prompt(
    self,
    message: str | Message | list[Message],
    *,
    run_id: str | None = None,
) -> str:
    if self._run_lock.locked() or self._state.is_streaming:
        raise RuntimeError("Agent is already running")
    effective_run_id = run_id or uuid.uuid4().hex
    self._state.active_run_id = effective_run_id
    try:
        async with self._run_lock:
            # ... existing body, with run_id threaded through ...
            await self._run_prompt(message, run_id=effective_run_id)
    except BaseException:
        # Spec §3.7: leave active_run_id SET on failure so except-block
        # readers can recover it. CompletionMarkerFailedError.run_id is
        # still the recommended source in handlers (see Task 26).
        raise
    else:
        self._state.active_run_id = None
        return effective_run_id
```

(Note: the actual edit shape depends on the existing prompt() body —
preserve everything that's there; only thread the run_id through and
add the active_run_id set + try/except/else clear pattern. Internal
helpers that perform `append()` will pass run_id into the Message
construction in subsequent tasks.)

**Per spec §3.7 — do NOT clear `active_run_id` in `finally`.** Clearing
in `finally` would wipe the value before `except` handlers can read
it, defeating the recovery contract for `CompletionMarkerFailedError`
and any other post-claim failure mode.

Also import `uuid` at the top of the file.

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/agent/test_agent_run_id.py tests/agent/ -q`
Expected: green. (Test `tests/agent/test_agent.py` may need a small
update if it asserted `prompt()` returns None — change to `await
agent.prompt(...)` only without assigning, or update to assert `str`.)

- [ ] **Step 5: Commit**

```
git add cubepi/agent/agent.py tests/agent/test_agent_run_id.py
git commit -m "feat(agent): prompt accepts run_id, returns it, exposes via active_run_id"
```

---

### Task 23: Stamp `run_id` on every appended Message via `Agent._process_event`

**Files:**
- Modify: `cubepi/agent/agent.py` (central stamping point)
- Test: extend `tests/agent/test_agent_run_id.py`

**Why `_process_event` is the right location.** `cubepi/agent/loop.py`
does NOT directly construct most of the Message objects that get
persisted — the loop emits events, and `Agent._process_event`
(`cubepi/agent/agent.py:~329`) is the single dispatch site where
`MessageEndEvent` triggers the actual `checkpointer.append([msg])`
call. The same `_process_event` is also called by `abort_pending()`
when it injects synthetic deny + terminal aborted messages, and by
`respond()` resume path. Stamping at this single chokepoint covers
every append.

- [ ] **Step 1: Failing test asserts persisted messages carry the run_id**

Append to `tests/agent/test_agent_run_id.py`:
```python
@pytest.mark.asyncio
async def test_appended_messages_carry_run_id():
    from cubepi.checkpointer.memory import MemoryCheckpointer

    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="t",
    )
    await a.prompt("hello", run_id="R1")
    data = await cp.load("t")
    for m in data.messages:
        assert m.run_id == "R1"


@pytest.mark.asyncio
async def test_prompt_rejects_mismatched_run_id_before_claim():
    """Caller pre-stamps a Message with a different run_id than the
    one supplied to prompt(). Reject BEFORE claim_run so no row is
    written and the run_id is still reusable."""
    from cubepi.checkpointer.memory import MemoryCheckpointer
    from cubepi.providers.base import TextContent, UserMessage

    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="t",
    )
    bad_msg = UserMessage(
        content=[TextContent(text="hi")], run_id="WRONG"
    )
    with pytest.raises(ValueError, match="does not match"):
        await a.prompt(bad_msg, run_id="R1")
    # No claim row written — "R1" still freely claimable.
    assert "R1" not in cp._runs.get("t", {})
    # ... and a second prompt with the same run_id succeeds:
    await a.prompt("hi", run_id="R1")
    assert "R1" in cp._runs["t"]
```

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/agent/test_agent_run_id.py::test_appended_messages_carry_run_id -v`
Expected: `m.run_id` is None.

- [ ] **Step 3: Stamp in `Agent._process_event` (single chokepoint)**

**Pre-flight in `Agent.prompt()` (BEFORE `claim_run`).** Validating
caller-supplied `Message.run_id` mismatches only at `_process_event`
time is too late — by then `claim_run` has already written a row.
Walk the prompt input first:

```python
def _validate_input_run_ids(
    self, message, effective_run_id: str
) -> None:
    if isinstance(message, str):
        return
    if isinstance(message, list):
        candidates = message
    else:
        candidates = [message]
    for m in candidates:
        if getattr(m, "run_id", None) is not None and m.run_id != effective_run_id:
            raise ValueError(
                f"message.run_id={m.run_id!r} does not match "
                f"prompt(run_id={effective_run_id!r})"
            )
```

Call `self._validate_input_run_ids(message, effective_run_id)`
immediately after `effective_run_id` is decided, BEFORE the HITL
binding check, BEFORE `claim_run`.

**Defense-in-depth in `_process_event`** (still keep this — the
agent loop's own streaming may construct Messages elsewhere):

In `cubepi/agent/agent.py`, locate `_process_event(self, event)` and
the `MessageEndEvent` branch. Stamp `run_id` from
`self._state.active_run_id` onto the message before it's appended:

```python
elif isinstance(event, MessageEndEvent):
    msg = event.message
    active = self._state.active_run_id
    if active is not None:
        if msg.run_id is None:
            msg = msg.model_copy(update={"run_id": active})
            event = event.model_copy(update={"message": msg})
        elif msg.run_id != active:
            # Should be impossible after the pre-flight in prompt(),
            # but defend against future code paths that construct
            # Messages with stale run_ids.
            raise ValueError(
                f"message.run_id={msg.run_id!r} does not match "
                f"active run_id={active!r}"
            )
    # ... existing append + state-mirror code, now operating on the
    # stamped `msg` / `event` ...
```

This is the single dispatch site every appended message flows through
(loop streaming, abort_pending's synthetic deny + terminal aborted at
`agent.py:582-595`, respond resume). One edit covers all paths.

Also stamp the input message constructed from a `str` argument inside
`Agent.prompt()` before passing to `_run_prompt` (locate the
`UserMessage(content=[TextContent(text=message)])` construction):

```python
if isinstance(message, str):
    message = UserMessage(
        content=[TextContent(text=message)],
        run_id=effective_run_id,
    )
```

If `message` is already a `Message` / `list[Message]`, leave caller's
value alone — `_process_event` will stamp at append time as a
backstop.

Loop.py and tools.py: NO changes required for this task — they emit
events that flow through `_process_event` and inherit the stamping.

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/agent/ -q`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/agent/loop.py cubepi/agent/agent.py tests/agent/test_agent_run_id.py
git commit -m "feat(agent): thread active run_id through loop into appended messages"
```

---

### Task 24: HITL channel run_id binding enforcement at `prompt()` entry

**Files:**
- Modify: `cubepi/agent/agent.py`
- Test: `tests/agent/test_hitl_binding_enforcement.py` (new)

- [ ] **Step 1: Failing tests**

Create `tests/agent/test_hitl_binding_enforcement.py`:
```python
import pytest

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool
from cubepi.hitl.binding import HitlBinding
from cubepi.middleware.base import Middleware
from cubepi.providers.faux import FauxProvider
from pydantic import BaseModel


class _NoArgs(BaseModel):
    pass


async def _noop(tool_call_id, args, *, signal=None, on_update=None):
    raise NotImplementedError


def _tool_with_hitl(binding):
    return AgentTool(
        name="ask_user",
        description="d",
        parameters=_NoArgs,
        execute=_noop,
        hitl=binding,
    )


def _agent(tools=None, middleware=None):
    return Agent(
        model=_ok_faux().model("faux-model"),
        tools=tools or [],
        middleware=middleware or [],
    )


@pytest.mark.asyncio
async def test_checkpointed_hitl_with_none_run_id_raises():
    tool = _tool_with_hitl(HitlBinding(checkpointed=True, run_id=None))
    a = _agent(tools=[tool])
    with pytest.raises(ValueError, match="no run_id bound"):
        await a.prompt("hi", run_id="R1")


@pytest.mark.asyncio
async def test_checkpointed_hitl_requires_explicit_run_id():
    tool = _tool_with_hitl(HitlBinding(checkpointed=True, run_id="R1"))
    a = _agent(tools=[tool])
    with pytest.raises(ValueError, match="generate-mode rejected"):
        await a.prompt("hi", run_id=None)


@pytest.mark.asyncio
async def test_checkpointed_hitl_run_id_mismatch_raises():
    tool = _tool_with_hitl(HitlBinding(checkpointed=True, run_id="R1"))
    a = _agent(tools=[tool])
    with pytest.raises(ValueError, match="does not match"):
        await a.prompt("hi", run_id="R2")


@pytest.mark.asyncio
async def test_checkpointed_hitl_run_id_match_succeeds():
    tool = _tool_with_hitl(HitlBinding(checkpointed=True, run_id="R1"))
    a = _agent(tools=[tool])
    got = await a.prompt("hi", run_id="R1")
    assert got == "R1"


@pytest.mark.asyncio
async def test_in_memory_hitl_no_constraint():
    tool = _tool_with_hitl(HitlBinding(checkpointed=False, run_id=None))
    a = _agent(tools=[tool])
    got = await a.prompt("hi")  # generate-mode allowed
    assert isinstance(got, str)
```

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/agent/test_hitl_binding_enforcement.py -v`
Expected: ValueError not raised.

- [ ] **Step 3: Add the enforcement to `Agent.prompt()` start**

The real `Agent` attribute names are `self._state.tools` (a public
property over `_tools`) and `self._middleware` (raw list saved at
`agent.py:169`). Use those, not `self.tools` / `self.middleware`.

Just before the `try:` block that sets `active_run_id`, add:
```python
bound: set[str] = set()
for elem in (*self._state.tools, *self._middleware):
    binding = getattr(elem, "hitl", None)
    if binding is None or not binding.checkpointed:
        continue
    if binding.run_id is None:
        raise ValueError(
            f"Checkpointed HITL element {elem!r} has no run_id bound; "
            "construct CheckpointedChannel(run_id=...) before passing "
            "it to ask_user_tool/HITL middleware"
        )
    bound.add(binding.run_id)
if bound:
    if run_id is None:
        raise ValueError(
            f"Agent has checkpointed HITL elements bound to "
            f"run_ids {sorted(bound)!r}; prompt(run_id=...) "
            "must be explicitly supplied (generate-mode rejected)"
        )
    if any(b != run_id for b in bound):
        raise ValueError(
            f"prompt(run_id={run_id!r}) does not match "
            f"HITL-bound run_ids {sorted(bound)!r}"
        )
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/agent/test_hitl_binding_enforcement.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```
git add cubepi/agent/agent.py tests/agent/test_hitl_binding_enforcement.py
git commit -m "feat(agent): enforce HITL channel run_id binding at prompt() entry"
```

---

### Task 25: `claim_run` pre-flight + degraded-mode capability check

**Files:**
- Modify: `cubepi/agent/agent.py`
- Test: `tests/agent/test_agent_claim_run.py` (new)

- [ ] **Step 1: Failing tests**

Create `tests/agent/test_agent_claim_run.py`:
```python
import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
)
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.faux import FauxProvider


def _agent(**kw):
    return Agent(model=_ok_faux().model("faux-model"), **kw)


@pytest.mark.asyncio
async def test_prompt_calls_claim_run_before_append():
    cp = MemoryCheckpointer()
    a = _agent(checkpointer=cp, thread_id="t")
    await a.prompt("hi", run_id="R1")
    # Run row exists.
    assert "R1" in cp._runs["t"]


@pytest.mark.asyncio
async def test_prompt_rejects_already_claimed_run_id():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "R1")
    a = _agent(checkpointer=cp, thread_id="t")
    with pytest.raises(RunAlreadyClaimedError):
        await a.prompt("hi", run_id="R1")


@pytest.mark.asyncio
async def test_prompt_rejects_already_completed_run_id():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "R1")
    await cp.mark_run_complete("t", "R1")
    a = _agent(checkpointer=cp, thread_id="t")
    with pytest.raises(RunAlreadyCompletedError):
        await a.prompt("hi", run_id="R1")


@pytest.mark.asyncio
async def test_no_checkpointer_no_claim_call():
    a = _agent()  # checkpointer=None
    got = await a.prompt("hi")
    assert isinstance(got, str)  # works fine; no claim attempted


@pytest.mark.asyncio
async def test_degraded_mode_v3_only_checkpointer():
    class _V3Only:
        async def load(self, thread_id): return None
        async def append(self, thread_id, msgs): pass
        async def save_extra(self, thread_id, extra): pass
        async def save_pending_request(self, thread_id, req, *, run_id=None): pass
        async def load_pending_request(self, thread_id): return None
        # No claim_run / mark_run_complete.
    a = _agent(checkpointer=_V3Only(), thread_id="t")
    # Vanilla prompt works (degraded mode skips claim).
    got = await a.prompt("hi")
    assert isinstance(got, str)
```

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/agent/test_agent_claim_run.py -v`
Expected: claim not called / errors not surfaced.

- [ ] **Step 3: Add the capability check + claim call**

In `Agent.__init__` (or as a `@property`), compute:
```python
self._run_aware = (
    self.checkpointer is not None
    and hasattr(self.checkpointer, "claim_run")
    and hasattr(self.checkpointer, "mark_run_complete")
)
```

In `Agent.prompt()`, after the HITL binding check but BEFORE setting
`active_run_id`, add the claim:
```python
if self._run_aware and self.thread_id is not None:
    await self.checkpointer.claim_run(self.thread_id, effective_run_id)
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/agent/test_agent_claim_run.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/agent/agent.py tests/agent/test_agent_claim_run.py
git commit -m "feat(agent): claim_run pre-flight + legacy degraded-mode fallback"
```

---

### Task 26: `mark_run_complete` after terminal success + outcome enumeration

**Files:**
- Create: `cubepi/agent/_outcome.py` (the `RunOutcome` literal + helpers)
- Modify: `cubepi/agent/loop.py` — `run_agent_loop`, `_run_loop`,
  `_run_loop_inner` to RETURN a `RunOutcome` instead of `None`
- Modify: `cubepi/agent/agent.py` — `_run_prompt` and `_run_hitl_resume`
  to propagate the outcome; `prompt()` and `respond()` dispatch on it
- Test: `tests/agent/test_agent_completion.py` (new)

**Why this is a deliberate refactor, not a drop-in change.** The current
`run_agent_loop`, `_run_loop`, and `_run_loop_inner` return `None` (or
a `list[Message]`). The loop catches `HitlDetached` / `HitlAborted`
silently (`loop.py:196` and `loop.py:418`) and returns normally. There
is no first-class "what happened" return value today. This task
introduces one.

- [ ] **Step 1: Failing tests covering every row of spec §3.6.2 table**

**FauxProvider API reminder.** `FauxProvider`
(`cubepi/providers/faux.py:156`) has no `text=` / `error=` /
`tool_error=` / `abort_mid_stream=` / `sleep_seconds=` kwargs. Its
real API: construct empty, then call `set_responses([...])` with a
list of `AssistantMessage` instances (or `FauxResponseFactory`
callables). To simulate provider failures, subclass `FauxProvider`
inside the test file.

Create `tests/agent/test_agent_completion.py`:

```python
import asyncio

import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.exceptions import CompletionMarkerFailedError
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import AssistantMessage, TextContent
from cubepi.providers.faux import FauxProvider


def _ok_provider() -> FauxProvider:
    p = FauxProvider()
    p.set_responses([
        AssistantMessage(content=[TextContent(text="ok")], stop_reason="end_turn")
    ])
    return p


def _agent(provider, **kw):
    cp = MemoryCheckpointer()
    a = Agent(model=provider.model("faux-model"), checkpointer=cp, thread_id="t", **kw)
    return a, cp


@pytest.mark.asyncio
async def test_clean_success_marks_complete():
    a, cp = _agent(_ok_provider())
    await a.prompt("hi", run_id="R1")
    rs = cp._runs["t"]["R1"]
    assert rs.completed_at is not None
    assert rs.completion_seq is not None


@pytest.mark.asyncio
async def test_provider_error_does_not_mark():
    """A provider exception is caught by the agent lifecycle, which
    appends a synthetic assistant with stop_reason='error' and
    RETURNS NORMALLY (does not re-raise). state.last_outcome is set
    to "abandoned" via the loop.py:485 branch, dispatch sees
    not-complete, mark_run_complete is not called."""
    class _RaisingProvider(FauxProvider):
        async def stream(self, *args, **kwargs):
            raise RuntimeError("provider down")
    a, cp = _agent(_RaisingProvider())
    await a.prompt("hi", run_id="R1")  # returns normally, no raise
    rs = cp._runs["t"]["R1"]
    assert rs.completed_at is None
    # The synthetic error assistant is persisted.
    data = await cp.load("t")
    assert any(
        getattr(m, "stop_reason", None) == "error"
        for m in data.messages
    )


@pytest.mark.asyncio
async def test_hitl_detached_outcome_suspended_no_mark():
    """Provider returns an ask_user tool_call → loop pauses → channel
    detach → loop catch sets state.last_outcome="suspended" → no mark.
    Then respond() resumes; provider finishes; marker IS written.
    """
    from cubepi.checkpointer.memory import MemoryCheckpointer
    from cubepi.hitl.ask_user import ask_user_tool
    from cubepi.hitl.channel import CheckpointedChannel
    from cubepi.providers.base import AssistantMessage, TextContent, ToolCall

    cp = MemoryCheckpointer()
    p = FauxProvider()
    p.set_responses([
        # Turn 1: ask_user tool_call.
        AssistantMessage(
            content=[ToolCall(id="tc-1", name="ask_user",
                              arguments={"questions": [{"key": "ans", "prompt": "?"}]})],
            stop_reason="tool_use",
        ),
        # Turn 2 (after respond appends the answer as tool_result):
        # final assistant.
        AssistantMessage(
            content=[TextContent(text="done")], stop_reason="end_turn"
        ),
    ])
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool = ask_user_tool(ch)
    a = Agent(
        model=p.model("faux-model"),
        tools=[tool],
        checkpointer=cp,
        thread_id="t",
        channel=ch,
    )
    # Drive the run inside a Task so we can detach mid-pause.
    task = asyncio.create_task(a.prompt("hi", run_id="R1"))
    # Wait until pending_request row exists (loop has reached the
    # ask_user pause).
    while True:
        if (await cp.load_pending("t")) is not None:
            break
        await asyncio.sleep(0.01)
    # CheckpointedChannel has no .detach(); the Agent does
    # (agent.py:405). a.detach() raises HitlDetached in the loop.
    await a.detach()
    await task  # returns normally; state.last_outcome == "suspended"
    assert cp._runs["t"]["R1"].completed_at is None

    # NOTE: The resume-writes-marker assertion lives in Task 28
    # (test_respond_resume_writes_marker), because run_id recovery
    # via load_pending() is implemented there. At Task 26 we only
    # assert the suspended-state-no-marker portion.


@pytest.mark.asyncio
async def test_hitl_aborted_via_abort_pending_does_not_mark():
    """Provider returns an ask_user tool_call → loop pauses →
    abort_pending raises HitlAborted in the loop → catch sets
    state.last_outcome="abandoned" (NOT "suspended"). agent.py:582-595
    appends the synthetic deny + terminal aborted assistant.
    """
    from cubepi.checkpointer.memory import MemoryCheckpointer
    from cubepi.hitl.ask_user import ask_user_tool
    from cubepi.hitl.channel import CheckpointedChannel
    from cubepi.providers.base import AssistantMessage, ToolCall

    cp = MemoryCheckpointer()
    p = FauxProvider()
    p.set_responses([
        AssistantMessage(
            content=[ToolCall(id="tc-1", name="ask_user",
                              arguments={"questions": [{"key": "ans", "prompt": "?"}]})],
            stop_reason="tool_use",
        ),
    ])
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool = ask_user_tool(ch)
    a = Agent(
        model=p.model("faux-model"),
        tools=[tool],
        checkpointer=cp,
        thread_id="t",
        channel=ch,
    )
    task = asyncio.create_task(a.prompt("hi", run_id="R1"))
    while (await cp.load_pending("t")) is None:
        await asyncio.sleep(0.01)
    await a.abort_pending(reason="user cancelled")
    await task

    rs = cp._runs["t"]["R1"]
    assert rs.completed_at is None
    data = await cp.load("t")
    assert any(
        getattr(m, "stop_reason", None) == "aborted"
        for m in data.messages
    )


# Note: test_incomplete_tool_cycle_does_not_mark is implemented in
# Task 27 alongside the _tool_cycle.py helper that demotes
# "complete" → "incomplete". The outcome table row above is covered
# by that task's tests; we do not stub it here at Task 26 because
# the check_tool_cycle helper doesn't exist yet.


@pytest.mark.asyncio
async def test_propagating_cancel_does_not_mark():
    class _SlowProvider(FauxProvider):
        async def stream(self, *args, **kwargs):
            await asyncio.sleep(10)
            return  # unreachable
    a, cp = _agent(_SlowProvider())
    task = asyncio.create_task(a.prompt("hi", run_id="R1"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    rs = cp._runs["t"]["R1"]
    assert rs.completed_at is None


@pytest.mark.asyncio
async def test_completion_marker_failed_carries_run_id_when_generated():
    """prompt(run_id=None) generates; mark_run_complete fails;
    exception carries the generated run_id; active_run_id is left set."""
    class _BrokenMark(MemoryCheckpointer):
        async def mark_run_complete(self, thread_id, run_id):
            raise RuntimeError("db down")

    cp = _BrokenMark()
    a = Agent(
        model=_ok_provider().model("faux-model"),
        checkpointer=cp,
        thread_id="t",
    )
    with pytest.raises(CompletionMarkerFailedError) as exc_info:
        await a.prompt("hi")  # run_id=None → generate
    assert exc_info.value.run_id is not None
    assert exc_info.value.run_id == a.state.active_run_id
```

(The exact FauxProvider injection API may differ; use whatever mechanism
the test suite already uses to inject errors / stop_reasons. The
assertion shape — "mark called" vs "mark not called" — is what matters.)

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/agent/test_agent_completion.py -v`
Expected: mark not called even on clean success.

- [ ] **Step 3: Introduce `RunOutcome` and propagate it**

Create `cubepi/agent/_outcome.py`:
```python
from __future__ import annotations
from typing import Literal

RunOutcome = Literal["complete", "suspended", "incomplete", "abandoned"]
```

Modify `cubepi/agent/loop.py`. **The existing public helpers
`run_agent_loop`, `run_agent_loop_continue`, `run_agent_loop_resume`
return `list[Message]` and are exported from `cubepi`.** Do NOT change
their public return type. But `loop.py` has no `AgentState` reference
(it receives an `AgentContext` snapshot), so the outcome can't be
written to state from inside the loop. Use an **outcome sink callback**.

Add a new field on `AgentState` (Task 5 introduced `active_run_id`;
this adds the sibling):
```python
class AgentState:
    # ... existing fields ...
    active_run_id: str | None = None
    last_outcome: RunOutcome | None = None   # NEW
```

Add an optional callback parameter to each public loop helper
(`run_agent_loop`, `run_agent_loop_continue`, `run_agent_loop_resume`)
AND the private `_run_loop` / `_run_loop_inner`:

```python
async def run_agent_loop(
    *,
    # ... all existing params unchanged ...
    set_outcome: Callable[[str], None] | None = None,
) -> list[Message]:
    ...
```

External callers omit `set_outcome` (default `None`) — completely
non-breaking. Tests in `tests/agent/test_loop.py` keep working without
changes.

Inside the loop helpers, at each exit path call
`set_outcome("<literal>")` BEFORE the existing `return new_messages`
(guarded by `set_outcome is not None`). Pass `set_outcome` down through
`_run_loop` / `_run_loop_inner` the same way other callables are
threaded today.

The literals:
  - After clean `AgentEndEvent` with terminal stop_reason and no
    pending HITL → `if set_outcome: set_outcome("complete")` then
    `return new_messages`
  - The existing combined `except (HitlDetached, HitlAborted)` blocks
    at `loop.py:196` and `loop.py:418` must be **split by exception
    type**:
    ```python
    except HitlDetached:
        if set_outcome is not None:
            set_outcome("suspended")
        return new_messages
    except HitlAborted:
        if set_outcome is not None:
            set_outcome("abandoned")
        return new_messages
    ```
    `HitlDetached` = channel detached, resumable via respond().
    `HitlAborted` = abort_pending called; synthetic deny + terminal
    aborted assistant already appended by `agent.py:582-595`.
  - After the early-exit branch on `stop_reason in ("error", "aborted")`
    at `loop.py:485` → `if set_outcome: set_outcome("abandoned")`
    then `return new_messages`
  - When `check_tool_cycle(...)` (from Task 27) raises
    `ToolCycleViolation` immediately before declaring `complete`,
    catch it and `return "incomplete"` instead
  - On propagating `asyncio.CancelledError` → DO NOT catch; the
    function never returns; `prompt()`'s outer try/except handles it
- Run `_run_loop_inner`'s `_run_with_lifecycle` (if it wraps anything)
  must also forward the outcome.

Modify `cubepi/agent/agent.py`:
- `_run_prompt` and `_run_hitl_resume` return `list[Message]` (no
  signature change) and internally pass `set_outcome=` to
  `run_agent_loop` / `run_agent_loop_resume`. The sink writes into
  `self._state.last_outcome`:
  ```python
  def _outcome_sink(self):
      def _sink(value: str) -> None:
          self._state.last_outcome = value
      return _sink
  ```
  Each call site passes `set_outcome=self._outcome_sink()`.
- In `Agent.prompt()` (replacing the placeholder line `await
  self._run_prompt(...)`):
  ```python
  self._state.last_outcome = None
  await self._run_prompt(message, run_id=effective_run_id)
  outcome = self._state.last_outcome or "abandoned"
  await self._dispatch_outcome(outcome, effective_run_id)
  ```
- `Agent.respond()` dispatch wiring is **not** part of Task 26 —
  it needs the `recovered_run_id` from `load_pending()` which is
  introduced in Task 28 (Step 3). Task 28 adds the matching
  ```python
  self._state.last_outcome = None
  await self._run_hitl_resume(...)
  outcome = self._state.last_outcome or "abandoned"
  await self._dispatch_outcome(outcome, recovered_run_id)
  ```
  block. Tests for respond's marker-on-clean-resume therefore live
  in Task 28.

`tests/agent/test_loop.py` does NOT need updates — public return
type AND default behavior are unchanged (sink is optional). The
breaking surface stays limited to `Agent.prompt()` return type
(`None` → `str`), already in the spec's documented breaking-changes
list.
- New private helper `_dispatch_outcome`:
  ```python
  async def _dispatch_outcome(
      self, outcome: RunOutcome, run_id: str
  ) -> None:
      if outcome != "complete":
          return
      if not (self._run_aware and self.thread_id):
          return
      try:
          await self.checkpointer.mark_run_complete(
              self.thread_id, run_id
          )
      except Exception as exc:
          raise CompletionMarkerFailedError(
              thread_id=self.thread_id,
              run_id=run_id,
              cause=exc,
          ) from exc
  ```

Mapping from spec §3.6.2 outcome table to literal:

| §3.6.2 row | Returned outcome | Marker called? |
|---|---|---|
| Clean success | `"complete"` | yes |
| HITL suspended (normal pause) | `"suspended"` | no |
| HITL detached (silent catch) | `"suspended"` | no — respond() resumes |
| HITL aborted (abort_pending) | `"abandoned"` | no |
| Incomplete tool cycle (Task 27) | `"incomplete"` | no |
| Provider / tool error | `"abandoned"` | no |
| Abort during streaming | `"abandoned"` | no |
| Propagating cancellation | (raises, no return) | no |

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/agent/test_agent_completion.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/agent/loop.py cubepi/agent/agent.py tests/agent/test_agent_completion.py
git commit -m "feat(agent): mark_run_complete only on clean terminal outcome"
```

---

### Task 27: Pre-completion tool-cycle invariant

**Files:**
- Create: `cubepi/agent/_tool_cycle.py` (helper, kept small + testable)
- Modify: `cubepi/agent/loop.py` to call it before declaring `complete`
- Test: `tests/agent/test_tool_cycle_invariant.py` (new)

- [ ] **Step 1: Failing tests covering the four spec shapes**

Create `tests/agent/test_tool_cycle_invariant.py`:
```python
import asyncio

import pytest

from cubepi.agent._tool_cycle import (
    ToolCycleViolation,
    check_tool_cycle,
)
from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl.ask_user import ask_user_tool
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.middleware.base import TurnAction
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider


def _asst(call_ids, run_id="R"):
    return AssistantMessage(
        content=[
            ToolCall(id=cid, name="t", arguments={}) for cid in call_ids
        ],
        run_id=run_id,
    )


def _res(call_id, run_id="R"):
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name="t",
        content=[TextContent(text="r")],
        run_id=run_id,
    )


def test_no_tool_calls_ok():
    check_tool_cycle([
        UserMessage(content=[TextContent(text="hi")], run_id="R"),
        AssistantMessage(content=[TextContent(text="hi back")], run_id="R"),
    ])


def test_complete_cycle_ok():
    check_tool_cycle([
        _asst(["c1", "c2"]),
        _res("c1"),
        _res("c2"),
        AssistantMessage(content=[TextContent(text="done")], run_id="R"),
    ])


def test_no_results_at_all_violation():
    try:
        check_tool_cycle([_asst(["c1"])])
    except ToolCycleViolation:
        return
    assert False, "expected ToolCycleViolation"


def test_intervening_user_message_violation():
    try:
        check_tool_cycle([
            _asst(["c1"]),
            UserMessage(content=[TextContent(text="hi")], run_id="R"),
            _res("c1"),
        ])
    except ToolCycleViolation:
        return
    assert False


def test_partial_cover_violation():
    try:
        check_tool_cycle([_asst(["c1", "c2"]), _res("c1")])
    except ToolCycleViolation:
        return
    assert False


def test_duplicate_ids_across_turns_violation():
    try:
        check_tool_cycle([
            _asst(["c1"]),
            _asst(["c1"]),  # second assistant reuses id
            _res("c1"),
        ])
    except ToolCycleViolation:
        return
    assert False


def test_multiset_mismatch_within_window_violation():
    """Assistant emits {c1, c2}; window has [c1, c1] — set-equality
    would have failed, but the bug is multiset-specific: window length
    matches K=2, and only the multiset check catches that c2 is missing
    while c1 is duplicated. (A trailing extra tool_result AFTER the
    K-window is a separate concern handled by the NEXT assistant turn's
    adjacency check; it's not what this test covers.)"""
    try:
        check_tool_cycle([
            _asst(["c1", "c2"]),
            _res("c1"),
            _res("c1"),  # duplicate of c1; c2 missing
        ])
    except ToolCycleViolation:
        return
    assert False
```

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/agent/test_tool_cycle_invariant.py -v`
Expected: import error.

- [ ] **Step 3: Implement `_tool_cycle.py`**

Create `cubepi/agent/_tool_cycle.py`:
```python
"""Pre-completion tool-cycle invariant — spec §3.6.2.

For each AssistantMessage with ToolCall blocks {c1..cK}, the K-message
window immediately following MUST be all ToolResultMessages whose
tool_call_id MULTISET equals {c1..cK} — no extras, no missing, no
duplicates beyond what the assistant emitted, and no other
AssistantMessage or UserMessage may appear in that window.
"""

from __future__ import annotations

from collections import Counter

from cubepi.providers.base import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
)


class ToolCycleViolation(ValueError):
    def __init__(
        self,
        *,
        kind: str,
        assistant_index: int,
        expected: Counter,
        got: Counter,
    ) -> None:
        super().__init__(
            f"tool-cycle violation [{kind}] at assistant index "
            f"{assistant_index}: expected {dict(expected)}, "
            f"got {dict(got)}"
        )
        self.kind = kind
        self.assistant_index = assistant_index
        self.expected = expected
        self.got = got


def check_tool_cycle(messages: list[Message]) -> None:
    for i, m in enumerate(messages):
        if not isinstance(m, AssistantMessage):
            continue
        call_ids = [c.id for c in m.content if isinstance(c, ToolCall)]
        if not call_ids:
            continue
        expected = Counter(call_ids)
        k = len(call_ids)
        window = messages[i + 1 : i + 1 + k]
        if len(window) < k:
            raise ToolCycleViolation(
                kind="incomplete-window",
                assistant_index=i,
                expected=expected,
                got=Counter(),
            )
        for w in window:
            if not isinstance(w, ToolResultMessage):
                raise ToolCycleViolation(
                    kind="non-tool-result-in-window",
                    assistant_index=i,
                    expected=expected,
                    got=Counter(),
                )
        got = Counter(w.tool_call_id for w in window)
        # Multiset equality — catches duplicates the spec rejects.
        if got != expected:
            raise ToolCycleViolation(
                kind="multiset-mismatch",
                assistant_index=i,
                expected=expected,
                got=got,
            )
```

- [ ] **Step 4: Wire into the Agent-layer completion dispatch**

`loop.py` has no `run_id` parameter — the active run_id lives on
`AgentState.active_run_id` (Task 5). Apply the invariant from `Agent`,
NOT `loop.py`:

In `cubepi/agent/agent.py`, modify the `_dispatch_outcome` helper
introduced in Task 26 Step 3. BEFORE writing the mark, when
`outcome == "complete"`, scan the run's messages and demote on
violation:

```python
from cubepi.agent._tool_cycle import ToolCycleViolation, check_tool_cycle


async def _dispatch_outcome(
    self, outcome: RunOutcome, run_id: str
) -> None:
    if outcome == "complete":
        run_messages = [
            m for m in self._state.messages if m.run_id == run_id
        ]
        try:
            check_tool_cycle(run_messages)
        except ToolCycleViolation:
            outcome = "incomplete"
    if outcome != "complete":
        return
    if not (self._run_aware and self.thread_id):
        return
    try:
        await self.checkpointer.mark_run_complete(
            self.thread_id, run_id
        )
    except Exception as exc:
        raise CompletionMarkerFailedError(
            thread_id=self.thread_id, run_id=run_id, cause=exc
        ) from exc
```

**On HITL resume specifically.** When `respond()` resumes a run, the
pre-suspend tool-use assistant is still in `state.messages`
(loaded from the checkpointer at lazy-load time and carrying the same
`run_id`). Filtering by `m.run_id == run_id` picks up the full
cross-suspend run. The invariant therefore catches a tool-use that
remained unresolved across the pause.

Add TWO tests to `tests/agent/test_tool_cycle_invariant.py`. The
first is the straight `after_model_response(decision="stop")` case
that was moved out of Task 26 because the helper this task
introduces is what catches it:

```python
@pytest.mark.asyncio
async def test_incomplete_tool_cycle_does_not_mark():
    """after_model_response(decision='stop') on a tool-use response
    leaves an unresolved tool_call. _dispatch_outcome filters
    state.messages by run_id and demotes 'complete' to 'incomplete'
    via check_tool_cycle. Marker not written."""
    p = FauxProvider()
    p.set_responses([
        AssistantMessage(
            content=[ToolCall(id="c1", name="t", arguments={})],
            stop_reason="tool_use",
        ),
    ])

    async def _stop_after(response, ctx, *, signal=None):
        return TurnAction(decision="stop")

    cp = MemoryCheckpointer()
    a = Agent(
        model=p.model("faux-model"),
        checkpointer=cp,
        thread_id="t",
        after_model_response=_stop_after,
    )
    await a.prompt("hi", run_id="R1")
    assert cp._runs["t"]["R1"].completed_at is None


@pytest.mark.asyncio
async def test_tool_cycle_invariant_spans_hitl_resume():
    """Pause mid-tool-use (ask_user). Resume; provider then emits an
    assistant carrying an UNRELATED unresolved tool_call (no matching
    ToolResultMessage will follow). The invariant filters
    state.messages by m.run_id == 'R1' — sees the unresolved c1 →
    outcome demoted from 'complete' to 'incomplete' → marker NOT
    written."""
    cp = MemoryCheckpointer()
    p = FauxProvider()
    p.set_responses([
        # Turn 1: ask_user → pause.
        AssistantMessage(
            content=[ToolCall(id="ask-1", name="ask_user",
                              arguments={"questions": [{"key": "ans", "prompt": "?"}]})],
            stop_reason="tool_use",
        ),
        # Turn 2 (resume): assistant with an unresolved tool_call
        # that NO ToolResultMessage will satisfy. The agent loop's
        # after_model_response forces a stop on this turn so the
        # call never gets executed.
        AssistantMessage(
            content=[
                ToolCall(id="orphan-1", name="lookup", arguments={}),
            ],
            stop_reason="tool_use",
        ),
    ])

    async def _stop_after(response, ctx, *, signal=None):
        # Stop after the second assistant turn so the orphan tool_call
        # is NEVER executed.
        if any(
            getattr(c, "id", None) == "orphan-1" for c in response.content
        ):
            return TurnAction(decision="stop")
        return None

    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool = ask_user_tool(ch)
    a = Agent(
        model=p.model("faux-model"),
        tools=[tool],
        checkpointer=cp,
        thread_id="t",
        channel=ch,
        after_model_response=_stop_after,
    )
    task = asyncio.create_task(a.prompt("hi", run_id="R1"))
    while (await cp.load_pending("t")) is None:
        await asyncio.sleep(0.01)
    # prompt() blocks until detach / abort / answer. Detach to
    # surface HitlDetached → loop returns with last_outcome="suspended".
    await a.detach()
    await task

    # Resume with a fresh Agent. The second turn emits orphan-1;
    # the after_model_response hook stops the loop. Pre-completion
    # invariant scans state.messages filtered by run_id=='R1' and
    # finds the unresolved orphan-1 → outcome 'incomplete' → no mark.
    ch2 = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool2 = ask_user_tool(ch2)
    a2 = Agent(
        model=p.model("faux-model"),
        tools=[tool2],
        checkpointer=cp,
        thread_id="t",
        channel=ch2,
        after_model_response=_stop_after,
    )
    pending = await cp.load_pending("t")
    qid = pending[0].question_id
    await a2.respond(question_id=qid, answer="yes")

    assert cp._runs["t"]["R1"].completed_at is None
```

- [ ] **Step 5: Run tests → expect pass**

Run: `uv run pytest tests/agent/test_tool_cycle_invariant.py tests/agent/test_agent_completion.py -v`
Expected: green.

- [ ] **Step 6: Commit**

```
git add cubepi/agent/_tool_cycle.py cubepi/agent/loop.py tests/agent/test_tool_cycle_invariant.py
git commit -m "feat(agent): pre-completion tool-cycle strict-adjacency invariant"
```

---

### Task 28: `respond()` resumes run_id from `load_pending`; does NOT call `claim_run`

**Files:**
- Modify: `cubepi/agent/agent.py`
- Test: `tests/agent/test_respond_resume_run_id.py` (new)

- [ ] **Step 1: Failing test asserting single-claim invariant**

Create `tests/agent/test_respond_resume_run_id.py` that:
- Mocks/spies on `MemoryCheckpointer.claim_run` to count calls
- Triggers a HITL pause (using cubepi's testing helpers from
  `cubepi/hitl/testing.py`)
- Calls `respond()` to resume
- Asserts:
  - `claim_run` was called exactly ONCE (during the initial prompt)
  - Messages appended after resume carry the same `run_id`
  - `mark_run_complete` fires on terminal exit of the resumed loop

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/agent/test_respond_resume_run_id.py -v`
Expected: `respond()` either reclaims or doesn't propagate run_id.

- [ ] **Step 3: Modify `Agent.respond()`**

**Edit the existing `respond()` body — DO NOT replace it.** Preserve
the existing `HitlNoPendingRequest` / `HitlStaleAnswer` raises, the
lazy history reload, and the `attach_resume_answer` call. The
respond body today is at `agent.py:431-469`. Make these three
changes:

1. Replace `load_pending = getattr(self.checkpointer,
   "load_pending_request", None)` (and the matching `await
   load_pending(self.thread_id)` call) with the new
   `load_pending()` Protocol method (Task 6) that returns
   `(HitlRequest, run_id | None)`. Keep the same fallback error
   when the method is missing.
2. Set `self._state.active_run_id` from the recovered run_id.
3. Wrap the `_run_hitl_resume()` call in try/except/else for
   outcome dispatch, mirroring `prompt()` (Task 26).

Concretely, the new `respond()` body looks like:

```python
async def respond(
    self, *, question_id: str | None = None, answer: StructuredValue
) -> None:
    from cubepi.hitl.exceptions import (
        HitlNoPendingRequest,
        HitlStaleAnswer,
    )

    if self._channel is None:
        raise HitlError("agent has no channel bound")
    if not (self.thread_id and self.checkpointer):
        raise RuntimeError("respond() requires thread_id + checkpointer")

    load_pending = getattr(self.checkpointer, "load_pending", None)
    if load_pending is None:
        raise HitlError(
            "respond() requires a checkpointer that implements "
            "load_pending (added in checkpointer v4)"
        )

    async with self._run_lock:
        if not self._state._messages:
            data = await self.checkpointer.load(self.thread_id)
            if data:
                self._state._messages = list(data.messages or [])
                self._extra = dict(data.extra or {})

        loaded = await load_pending(self.thread_id)
        if loaded is None:
            raise HitlNoPendingRequest("no pending request on this thread")
        pending, recovered_run_id = loaded
        if question_id is None:
            question_id = pending.question_id
        if question_id != pending.question_id:
            raise HitlStaleAnswer(
                f"answer for {question_id}, pending is {pending.question_id}"
            )

        # NEW: thread the recovered run_id into agent state.
        if recovered_run_id is not None:
            self._state.active_run_id = recovered_run_id
        self._state.last_outcome = None

        self._channel.attach_resume_answer(question_id, answer)
        try:
            await self._run_hitl_resume()
        except BaseException:
            # Spec §3.7: leave active_run_id SET on raise.
            raise
        else:
            # Legacy guard: pending rows written by pre-spec code
            # may carry run_id=None. Skip dispatch entirely — the
            # caller didn't opt into run tracking for this run.
            if recovered_run_id is not None:
                outcome = self._state.last_outcome or "abandoned"
                await self._dispatch_outcome(outcome, recovered_run_id)
            # Clear on successful resume completion (legacy or not).
            self._state.active_run_id = None
```

Add a regression test mirroring the legacy case:

```python
@pytest.mark.asyncio
async def test_respond_resume_with_legacy_pending_does_not_crash():
    """A pre-spec pending row was written without run_id. respond()
    must resume cleanly: dispatch is skipped (no mark), no crash.

    Setup nuance: the initial prompt DOES claim R1 (Task 25's
    pre-flight). The legacy-ness we're simulating is on the
    pending row only — `save_pending_request(run_id=None)`. The
    assertion is therefore "completed_at remains None" for R1 (the
    resume saw a None recovered_run_id and skipped dispatch),
    NOT "no R1 row at all".
    """
    cp = MemoryCheckpointer()
    p = _pause_then_finish_provider()
    a, task = await _drive_pause(cp, lambda: p.model("faux-model"))
    await a.detach()
    await task
    # Forge legacy state: clear the pending row's run_id (simulating
    # the pre-spec on-disk shape).
    pending_req = (await cp.load_pending("t"))[0]
    await cp.save_pending_request("t", pending_req, run_id=None)

    ch2 = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id=None)
    tool2 = ask_user_tool(ch2)
    a2 = Agent(
        model=p.model("faux-model"),
        tools=[tool2],
        checkpointer=cp,
        thread_id="t",
        channel=ch2,
    )
    pending = await cp.load_pending("t")
    qid = pending[0].question_id
    # Must not raise.
    await a2.respond(question_id=qid, answer="yes")
    # R1's claim row still exists from the initial prompt, but no
    # marker was written because dispatch was skipped on the
    # None-run_id resume.
    assert cp._runs["t"]["R1"].completed_at is None
```

**Do not call `claim_run` here.** All subsequent append calls use
`self._state.active_run_id` (the same chokepoint as prompt — see
Task 23). At terminal clean exit, `_dispatch_outcome` (from Task 26)
fires `mark_run_complete`.

Add tests in `tests/agent/test_respond_resume_run_id.py`:

```python
import asyncio

import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl.ask_user import ask_user_tool
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall
from cubepi.providers.faux import FauxProvider


def _pause_then_finish_provider() -> FauxProvider:
    p = FauxProvider()
    p.set_responses([
        # Turn 1: ask_user → pause.
        AssistantMessage(
            content=[ToolCall(
                id="tc-1", name="ask_user",
                arguments={"questions": [{"key": "ans", "prompt": "?"}]},
            )],
            stop_reason="tool_use",
        ),
        # Turn 2 (resume): final assistant.
        AssistantMessage(
            content=[TextContent(text="done")],
            stop_reason="end_turn",
        ),
    ])
    return p


async def _drive_pause(cp, model_factory):
    """Helper: start a prompt that pauses for ask_user. Returns
    (agent, task) — caller awaits task after detach/abort/respond."""
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool = ask_user_tool(ch)
    a = Agent(
        model=model_factory(),
        tools=[tool],
        checkpointer=cp,
        thread_id="t",
        channel=ch,
    )
    task = asyncio.create_task(a.prompt("hi", run_id="R1"))
    while (await cp.load_pending("t")) is None:
        await asyncio.sleep(0.01)
    return a, task


@pytest.mark.asyncio
async def test_respond_clears_active_run_id_on_clean_resume():
    cp = MemoryCheckpointer()
    p = _pause_then_finish_provider()
    a, task = await _drive_pause(cp, lambda: p.model("faux-model"))
    await a.detach()
    await task
    # Resume with a fresh Agent on a fresh channel.
    ch2 = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool2 = ask_user_tool(ch2)
    a2 = Agent(
        model=p.model("faux-model"),
        tools=[tool2],
        checkpointer=cp,
        thread_id="t",
        channel=ch2,
    )
    pending = await cp.load_pending("t")
    qid = pending[0].question_id
    await a2.respond(question_id=qid, answer="yes")
    assert a2.state.active_run_id is None


@pytest.mark.asyncio
async def test_respond_resume_writes_marker():
    """Cross-reference: Task 26's
    test_hitl_detached_outcome_suspended_no_mark stops at the
    suspended-state assertion. This test continues to the resume
    path and asserts cp._runs[t][R1].completed_at IS NOT None
    after respond()."""
    cp = MemoryCheckpointer()
    p = _pause_then_finish_provider()
    a, task = await _drive_pause(cp, lambda: p.model("faux-model"))
    await a.detach()
    await task
    assert cp._runs["t"]["R1"].completed_at is None  # not yet marked

    ch2 = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool2 = ask_user_tool(ch2)
    a2 = Agent(
        model=p.model("faux-model"),
        tools=[tool2],
        checkpointer=cp,
        thread_id="t",
        channel=ch2,
    )
    pending = await cp.load_pending("t")
    qid = pending[0].question_id
    await a2.respond(question_id=qid, answer="yes")
    assert cp._runs["t"]["R1"].completed_at is not None
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/agent/test_respond_resume_run_id.py tests/agent/ -q`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/agent/agent.py tests/agent/test_respond_resume_run_id.py
git commit -m "feat(agent): respond() resumes run_id via load_pending; never reclaims"
```

---

## Phase 8 — Public fork API

### Task 29: `Agent.fork()`

**Files:**
- Modify: `cubepi/agent/agent.py`
- Test: `tests/agent/test_agent_fork.py` (new)

- [ ] **Step 1: Failing test**

```python
import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.faux import FauxProvider


@pytest.mark.asyncio
async def test_agent_fork_delegates_to_checkpointer():
    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await a.prompt("hello", run_id="R1")  # creates src + R1 marker
    await a.fork("src", "dst", after_run_id="R1")
    loaded = await cp.load("dst")
    assert loaded.parent_thread_id == "src"


@pytest.mark.asyncio
async def test_agent_fork_no_checkpointer_raises():
    a = Agent(model=_ok_faux().model("faux-model"))
    with pytest.raises(RuntimeError, match="checkpointer"):
        await a.fork("src", "dst", after_run_id="R1")


@pytest.mark.asyncio
async def test_agent_fork_does_not_mutate_self_thread_id():
    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await a.prompt("hello", run_id="R1")
    await a.fork("src", "dst", after_run_id="R1")
    assert a.thread_id == "src"
```

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/agent/test_agent_fork.py -v`
Expected: `AttributeError: 'Agent' object has no attribute 'fork'`.

- [ ] **Step 3: Add the method**

In `cubepi/agent/agent.py`:
```python
async def fork(
    self,
    src_thread_id: str,
    new_thread_id: str,
    *,
    after_run_id: str,
    metadata: JsonObject | None = None,
) -> None:
    if self.checkpointer is None:
        raise RuntimeError("fork requires a checkpointer")
    if not self._run_aware:
        from cubepi.checkpointer.exceptions import CheckpointerError
        raise CheckpointerError(
            "backend does not support fork; missing claim_run / "
            "mark_run_complete"
        )
    await self.checkpointer.fork(
        src_thread_id,
        new_thread_id,
        after_run_id=after_run_id,
        metadata=metadata,
    )
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/agent/test_agent_fork.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/agent/agent.py tests/agent/test_agent_fork.py
git commit -m "feat(agent): add Agent.fork() public API"
```

---

### Task 30: `ForkOnceResult` dataclass

**Files:**
- Modify: `cubepi/agent/types.py`

- [ ] **Step 1: Add the dataclass**

```python
@dataclass(frozen=True)
class ForkOnceResult:
    text: str
    messages: list[Message]
    stop_reason: str
```

Add a test asserting the type imports cleanly and is frozen.

- [ ] **Step 2: Commit**

```
git add cubepi/agent/types.py
git commit -m "feat(agent): add ForkOnceResult dataclass"
```

---

### Task 31: `Agent.fork_once()` — HITL ban + transient agent + tracing span

**Files:**
- Modify: `cubepi/agent/agent.py`
- Test: `tests/agent/test_agent_fork_once.py` (new)

- [ ] **Step 1: Failing tests covering all §3.8 behavior**

Create `tests/agent/test_agent_fork_once.py` testing:
- Simple text-only follow-up returns expected final text; source thread
  unchanged
- `RuntimeError` when no checkpointer
- `CheckpointerError` when checkpointer is a v3-only stub lacking
  `snapshot` / `claim_run` / `mark_run_complete` (degraded-mode test
  mirroring Task 25's coverage but for fork_once)
- `RuntimeError` when any `tool.hitl is not None` (checkpointed OR
  in-memory)
- `RuntimeError` when middleware has `hitl is not None`
- Cancellation: `asyncio.wait_for(..., timeout=0.01)` raises
  `TimeoutError`
- Source thread byte-identical before/after

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/agent/test_agent_fork_once.py -v`
Expected: missing method.

- [ ] **Step 3: Implement `fork_once`**

```python
async def fork_once(
    self,
    src_thread_id: str,
    message: str | Message | list[Message],
    *,
    after_run_id: str,
) -> ForkOnceResult:
    if self.checkpointer is None:
        raise RuntimeError("fork_once requires a checkpointer")
    # Degraded-mode guard — same shape as Agent.fork (Task 29).
    if not self._run_aware or not hasattr(self.checkpointer, "snapshot"):
        from cubepi.checkpointer.exceptions import CheckpointerError
        raise CheckpointerError(
            "backend does not support fork_once; missing snapshot / "
            "claim_run / mark_run_complete"
        )
    # HITL pre-flight. Real attributes: self._state.tools, self._middleware.
    hitl_offenders = [
        elem for elem in (*self._state.tools, *self._middleware)
        if getattr(elem, "hitl", None) is not None
    ]
    if hitl_offenders:
        names = ", ".join(
            getattr(e, "name", type(e).__name__) for e in hitl_offenders
        )
        raise RuntimeError(
            f"fork_once() does not support HITL. Found HITL-bearing "
            f"tools/middleware: {names}. Construct a different Agent "
            "without these for ephemeral probes."
        )
    snapshot = await self.checkpointer.snapshot(
        src_thread_id, after_run_id=after_run_id
    )
    # Build transient agent. The constructor takes the original `model`
    # arg (a BoundModel-like with `.provider` + `.spec`). Task 22 must
    # also have stored the constructor's `model` argument on
    # `self._model` (or equivalent retention) so fork_once can reuse it.
    # If that retention isn't already present, add it as a one-line
    # `self._model = model` in __init__ as part of this task.
    child = Agent(
        model=self._model,
        system_prompt=self._state.system_prompt,
        tools=list(self._state.tools),
        middleware=list(self._middleware),
        convert_to_llm=self.convert_to_llm,
        messages=snapshot,
        # checkpointer=None, thread_id=None — disable persistence.
    )
    pre_len = len(child.state.messages)
    fresh_run_id = uuid.uuid4().hex
    # Tracing — see Phase 10. For now just call prompt.
    with self._fork_once_span(
        src_thread_id=src_thread_id, after_run_id=after_run_id
    ):
        await child.prompt(message, run_id=fresh_run_id)
    new_messages = child.state.messages[pre_len:]
    # Extract final assistant text and stop_reason.
    final_text = ""
    stop_reason = "stop"
    for m in reversed(new_messages):
        from cubepi.providers.base import AssistantMessage, TextContent
        if isinstance(m, AssistantMessage):
            final_text = "".join(
                c.text for c in m.content if isinstance(c, TextContent)
            )
            stop_reason = m.stop_reason
            break
    return ForkOnceResult(
        text=final_text, messages=new_messages, stop_reason=stop_reason
    )

def _fork_once_span(self, *, src_thread_id, after_run_id):
    """Replaced by a real OTel span in Phase 10. Placeholder no-op."""
    from contextlib import nullcontext
    return nullcontext()
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/agent/test_agent_fork_once.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/agent/agent.py cubepi/agent/types.py tests/agent/test_agent_fork_once.py
git commit -m "feat(agent): add Agent.fork_once() with HITL ban + transient agent"
```

---

## Phase 9 — HITL integration

### Task 32: `ask_user_tool()` populates `tool.hitl` from channel

**Files:**
- Modify: `cubepi/hitl/ask_user.py`
- Test: `tests/hitl/test_ask_user_binding.py` (new)

- [ ] **Step 1: Failing test**

```python
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl.ask_user import ask_user_tool
from cubepi.hitl.channel import CheckpointedChannel, InMemoryChannel


def test_ask_user_tool_with_checkpointed_channel_sets_binding():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool = ask_user_tool(ch)
    assert tool.hitl is not None
    assert tool.hitl.checkpointed is True
    assert tool.hitl.run_id == "R1"


def test_ask_user_tool_with_in_memory_channel_sets_binding():
    ch = InMemoryChannel(thread_id="t")
    tool = ask_user_tool(ch)
    assert tool.hitl is not None
    assert tool.hitl.checkpointed is False
    assert tool.hitl.run_id is None
```

- [ ] **Step 2: Run → expect failure**

Run: `uv run pytest tests/hitl/test_ask_user_binding.py -v`
Expected: `tool.hitl is None`.

- [ ] **Step 3: Patch the factory**

In `cubepi/hitl/ask_user.py` (the `ask_user_tool` factory), at the
return site:
```python
from cubepi.hitl.binding import HitlBinding
from cubepi.hitl.channel import CheckpointedChannel

checkpointed = isinstance(channel, CheckpointedChannel)
binding = HitlBinding(
    checkpointed=checkpointed,
    run_id=getattr(channel, "_run_id", None) if checkpointed else None,
)
tool = AgentTool(
    name="ask_user",
    description=...,
    parameters=...,
    execute=...,
    hitl_builtin=True,
    hitl=binding,
)
return tool
```

- [ ] **Step 4: Run tests → expect pass**

Run: `uv run pytest tests/hitl/test_ask_user_binding.py tests/hitl/ -q`
Expected: green.

- [ ] **Step 5: Commit**

```
git add cubepi/hitl/ask_user.py tests/hitl/test_ask_user_binding.py
git commit -m "feat(hitl): ask_user_tool sets tool.hitl HitlBinding from channel"
```

---

### Task 33: `ApprovalPolicyMiddleware` populates `self.hitl`

**Files:**
- Modify: `cubepi/hitl/middleware.py`
- Test: `tests/hitl/test_approval_policy_binding.py` (new)

Same pattern as Task 32 applied to `ApprovalPolicyMiddleware.__init__`
and (transitively via subclassing) `ConfirmToolCallMiddleware`. Add
assertion test for both.

Commit: `feat(hitl): ApprovalPolicyMiddleware populates self.hitl`.

---

## Phase 10 — Tracing

### Task 34: `cubepi.agent.fork_once` OTel span

**Files:**
- Modify: `cubepi/agent/agent.py` (replace nullcontext placeholder)
- Modify: `cubepi/tracing/schema.py` (add span name constant)
- Test: `tests/tracing/test_fork_once_span.py` (new)

- [ ] **Step 1: Create `tests/tracing/conftest.py` with an in-memory exporter fixture if missing**

First check: `ls tests/tracing/`. If `conftest.py` exists with an
`in_memory_exporter` fixture already, skip. Otherwise create:

```python
# tests/tracing/conftest.py
from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry import trace


@pytest.fixture
def in_memory_exporter():
    """Configure the global tracer provider with an in-memory exporter
    for the duration of one test. Returns the exporter so tests can
    call get_finished_spans()."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
```

If `tests/tracing/` doesn't exist as a directory yet, create it with an
empty `__init__.py` first.

- [ ] **Step 2: Failing test using the in-memory exporter**

```python
# tests/tracing/test_fork_once_span.py
import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.faux import FauxProvider


@pytest.mark.asyncio
async def test_fork_once_emits_named_span(in_memory_exporter):
    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await a.prompt("hello", run_id="R1")
    await a.fork_once("src", "follow up?", after_run_id="R1")
    spans = in_memory_exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "cubepi.agent.fork_once" in names
    span = next(s for s in spans if s.name == "cubepi.agent.fork_once")
    attrs = dict(span.attributes)
    assert attrs["cubepi.fork.src_thread_id"] == "src"
    assert attrs["cubepi.fork.after_run_id"] == "R1"
```

- [ ] **Step 3: Run → expect failure**

Run: `uv run pytest tests/tracing/test_fork_once_span.py -v`
Expected: no span named `cubepi.agent.fork_once`.

- [ ] **Step 4: Replace `_fork_once_span` placeholder with a real span**

```python
def _fork_once_span(self, *, src_thread_id, after_run_id):
    try:
        from opentelemetry import trace
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()
    tracer = trace.get_tracer("cubepi.agent")
    return tracer.start_as_current_span(
        "cubepi.agent.fork_once",
        attributes={
            "cubepi.fork.src_thread_id": src_thread_id,
            "cubepi.fork.after_run_id": after_run_id,
        },
    )
```

- [ ] **Step 5: Run tests → expect pass**

Run: `uv run pytest tests/tracing/test_fork_once_span.py -v`
Expected: green.

- [ ] **Step 6: Commit**

```
git add cubepi/agent/agent.py tests/tracing/conftest.py tests/tracing/__init__.py tests/tracing/test_fork_once_span.py
git commit -m "feat(tracing): emit cubepi.agent.fork_once span"
```

---

## Phase 11 — Documentation

### Task 35: User-facing fork guide

**Files:**
- Create: `website/docs/guides/checkpointing/forking.md`

Write a guide covering:
- What a fork is + persistent vs ephemeral
- The cubebox copy-button UX
- `Agent.prompt()` accept-or-generate `run_id`
- `Agent.fork(after_run_id=…)`
- `Agent.fork_once()` + isolation contract (only checkpointer writes
  isolated; tools may have side effects; HITL banned)
- HITL binding requirement: build `CheckpointedChannel(run_id=…)` with
  the same value as `Agent.prompt(run_id=…)`
- Schema v3→v4 migration note pointing to alembic helper
- Legacy data behavior (NULL run_id messages remain readable; mixed
  threads forkable; all-legacy threads not forkable)
- Known limitation: cross-process interleaved runs on the same thread
  produce row-correct but semantically dangling forks

Cross-link from `website/docs/guides/checkpointing/postgres.md`,
`mysql.md`, `sqlite.md` (each gets a short "fork" subsection pointing
to the main guide).

Commit: `docs(guides): add conversation-fork guide`.

---

### Task 36: Migration notes in per-backend docs

**Files:**
- Modify: `website/docs/guides/checkpointing/postgres.md`
- Modify: `website/docs/guides/checkpointing/mysql.md`
- Modify: `website/docs/guides/checkpointing/sqlite.md` (if exists)

Each gets a "Schema v3 → v4 migration" subsection. For PG/MySQL,
include the alembic helper invocation. For SQLite, point at the
auto-migration in `__aenter__`.

Commit: `docs(guides): backend-specific v3→v4 migration notes`.

---

### Task 37: API reference + CHANGELOG

**Files:**
- Modify: `website/docs/api/cubepi-agent.mdx` (add new Agent methods)
- Modify: `website/docs/api/cubepi-checkpointer.mdx` (add new Protocol methods, errors)
- Modify: `CHANGELOG.md`

CHANGELOG entry:
```markdown
## 0.8.0 — TBD

### Added
- `Agent.fork(src, new, *, after_run_id, metadata=None)` — physical-copy fork at a completed-run boundary.
- `Agent.fork_once(src, message, *, after_run_id) -> ForkOnceResult` — single-turn ephemeral continuation.
- `Agent.prompt(message, *, run_id=None) -> str` now accept-or-generates the run_id and returns it.
- `Agent.state.active_run_id` exposes the in-flight run_id.
- `Agent(messages=...)` constructor arg for ephemeral pre-seeded history.
- `Checkpointer.snapshot`, `fork`, `claim_run`, `mark_run_complete`, `load_pending` Protocol methods.
- `cubepi_runs` table per backend (PG/MySQL schema v3 → v4).
- `Message.run_id: str | None` field on all three Message variants.
- `HitlBinding` attribute on `AgentTool` / `Middleware`; `ask_user_tool` and `ApprovalPolicyMiddleware` populate it.

### Breaking
- `Agent.prompt()` return type changed from `None` to `str`. Callers ignoring the return value keep working.
- `Checkpointer` Protocol gained 5 new methods. Third-party v3-only checkpointers continue to work for vanilla `prompt()` via degraded mode; fork APIs raise `CheckpointerError` on such backends.

### Migration
- Postgres / MySQL: run the new alembic helper (see backend guides).
- SQLite: auto-migration at connect time.
- Legacy `run_id=NULL` messages remain readable; threads with only such messages are not forkable.
```

Commit: `docs: API ref + CHANGELOG for fork feature`.

---

## Phase 12 — End-to-end tests

### Task 38: Cross-backend happy path

**Files:**
- Create: `tests/e2e/test_fork_e2e.py`

- [ ] **Step 1: Write parameterized test across all four backends**

```python
import pytest

from cubepi.agent.agent import Agent
from cubepi.providers.faux import FauxProvider

BACKENDS = ["memory", "sqlite", "postgres", "mysql"]


@pytest.fixture(params=BACKENDS)
async def checkpointer(request, tmp_path, pg_v4_dsn, mysql_v4_dsn):
    # Use v4-applied DSN fixtures; constructors open their own pools.
    if request.param == "memory":
        from cubepi.checkpointer.memory import MemoryCheckpointer
        yield MemoryCheckpointer()
    elif request.param == "sqlite":
        from cubepi.checkpointer.sqlite import SQLiteCheckpointer
        async with SQLiteCheckpointer(str(tmp_path / "x.db")) as cp:
            yield cp
    elif request.param == "postgres":
        from cubepi.checkpointer.postgres.checkpointer import PostgresCheckpointer
        async with PostgresCheckpointer(pg_v4_dsn) as cp:
            yield cp
    elif request.param == "mysql":
        from cubepi.checkpointer.mysql.checkpointer import MySQLCheckpointer
        async with MySQLCheckpointer(mysql_v4_dsn) as cp:
            yield cp


@pytest.mark.asyncio
async def test_fork_e2e_happy_path(checkpointer):
    cp = checkpointer
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    r1 = await a.prompt("first", run_id="R1")
    assert r1 == "R1"
    r2 = await a.prompt("second", run_id="R2")
    assert r2 == "R2"
    await a.fork("src", "dst", after_run_id="R1", metadata={"label": "branch"})
    loaded = await cp.load("dst")
    assert loaded.parent_thread_id == "src"
    assert loaded.extra["fork"] == {"label": "branch"}
    # dst contains R1's messages only, not R2's.
    run_ids = {m.run_id for m in loaded.messages if m.run_id}
    assert run_ids == {"R1"}
```

- [ ] **Step 2: Run on all backends**

Run: `uv run pytest tests/e2e/test_fork_e2e.py -v`
Expected: 4 backends × 1 test = 4 passes (skip PG / MySQL if test
infra isn't available in CI for those — gate appropriately).

- [ ] **Step 3: Commit**

```
git add tests/e2e/test_fork_e2e.py
git commit -m "test(e2e): cross-backend fork happy path"
```

---

### Task 39: HITL pause+resume+fork chain

**Files:**
- Create: `tests/e2e/test_fork_hitl_e2e.py`

- [ ] **Step 1: Test**

```python
@pytest.mark.asyncio
async def test_fork_after_hitl_resume():
    from cubepi.checkpointer.memory import MemoryCheckpointer
    cp = MemoryCheckpointer()
    """prompt pauses for HITL → respond resumes → marker written →
    fork succeeds and copies the resumed run."""
    ...
```

Use `cubepi/hitl/testing.py` helpers to script the HITL response.

- [ ] **Step 2: Commit**

```
git add tests/e2e/test_fork_hitl_e2e.py
git commit -m "test(e2e): HITL pause+resume followed by fork"
```

---

### Task 40: Concurrent fork race on Postgres

**Files:**
- Create: `tests/e2e/test_fork_concurrency_pg.py`

- [ ] **Step 1: Test**

Two concurrent forks of the same source thread (different
`new_thread_id`s); both succeed; outputs consistent (one fork's
messages are a prefix of the other's iff completion order is observed).

Also: concurrent claim of the same run_id from two coroutines via two
PostgresCheckpointer instances pointing at the same DB → exactly one
succeeds, the other raises `RunAlreadyClaimedError` with zero messages
appended.

- [ ] **Step 2: Commit**

```
git add tests/e2e/test_fork_concurrency_pg.py
git commit -m "test(e2e): concurrent fork + claim races on Postgres"
```

---

## Final checklist

- [ ] **All tests pass**

Run: `uv run pytest tests/ -q`
Expected: full suite green.

- [ ] **Type check clean**

Run: `uv run mypy cubepi`
Expected: no errors.

- [ ] **Lint clean**

Run: `uv run ruff check cubepi/ tests/ && uv run ruff format --check cubepi/ tests/`
Expected: no errors.

- [ ] **Coverage on new code**

Run: `uv run pytest tests/ --cov=cubepi --cov-report=term-missing`
Spot-check: `cubepi/agent/_tool_cycle.py`, `cubepi/checkpointer/memory.py`,
`cubepi/checkpointer/sqlite.py`, `cubepi/hitl/binding.py` are all
≥ 90 %.

- [ ] **Spec coverage walk**

Open `dev/specs/2026-06-05-conversation-fork.md` side-by-side and
confirm each section maps to a task:
- §3.1 storage → Phase 3–6
- §3.2 set-based selection → Tasks 9, 13, 17, 20
- §3.3 atomicity → Tasks 7, 10–13, 14–17, 18–20
- §3.4 copy table → Tasks 9, 13, 17, 20
- §3.5 tracing → Task 34
- §3.6 run lifecycle → Tasks 22–28
- §3.6.3.1 HITL binding → Tasks 4, 24, 32, 33
- §3.7 API surface → Tasks 1, 2, 4, 5, 6, 21, 22, 29, 30, 31
- §3.8 fork_once → Tasks 31, 34
- §3.9 per-backend → Phases 3–6
- §3.10 errors → Task 1
- §4 migration → Tasks 14, 18, 36, 37
- §5 testing → tests in every task + Phase 12

- [ ] **PR description draft**

Once everything is green, draft the PR body covering:
- One-paragraph summary
- Schema changes (PG/MySQL v3→v4; SQLite ALTER)
- Public API changes + return-type change on `Agent.prompt()`
- Migration story for third-party checkpointers (legacy degraded mode)
- Link to `dev/specs/2026-06-05-conversation-fork.md`
- Link to this plan

Open the PR. Drive the codex PR review loop per CLAUDE.md §5 (poll
~2 min, fix, reply `@codex review again` until clean).
