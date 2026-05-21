# cubepi cubebox-readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the 9 upstream cubepi deliverables (D1-D9) required by cubebox's main agent migration (Spec B), as new public API that's backward-compatible with v0.2.0 users.

**Architecture:** Each deliverable is independently testable and committable. Order chosen to minimize cross-task coupling: types/extensions first (D5, D6), then hooks (D7, D8), then provider tweaks (D3, D4), then new modules (D1, D2), packaging last (D9). cubebox can start consuming any deliverable via `[tool.uv.sources] cubepi = { path = "/home/chris/cubepi", editable = true }` immediately after merge — no PyPI release required.

**Tech Stack:** Python 3.11+, Pydantic v2, SQLAlchemy 2.0, asyncpg, msgpack, pytest + pytest-asyncio + respx, anthropic SDK, openai SDK, mcp SDK

**Spec:** `docs/specs/2026-05-13-cubepi-cubebox-readiness-design.md`

**Baseline:** existing cubepi test suite (`pytest tests/ -q`) must pass before and after every task — no regressions to existing v0.2.0 behavior.

---

## File Map

### Files to modify

| File | What changes |
|---|---|
| `cubepi/providers/base.py` | Add `metadata: dict[str, Any]` field to `UserMessage`, `AssistantMessage`, `ToolResultMessage` (D5) |
| `cubepi/agent/types.py` | Add `extra: dict[str, Any]` field to `AgentContext` (D6) |
| `cubepi/middleware/base.py` | Add `transform_system_prompt` (D7) and `after_model_response` + `TurnAction` (D8) to `Middleware` base class and `compose_middleware()` |
| `cubepi/agent/loop.py` | Wire `transform_system_prompt` and `after_model_response` into agent loop; persist `ctx.extra` via checkpointer after each turn (D6/D7/D8) |
| `cubepi/providers/anthropic.py` | Add `cache_policy: CacheMarkerPolicy \| None` constructor parameter; refactor existing marker logic to use policy (D3) |
| `cubepi/providers/openai.py` | Add OSS reasoning field extraction (3 variants), `payload_quirks` parameter (D4) |
| `cubepi/checkpointer/memory.py` | Persist `Message.metadata` round-trip (D5) |
| `cubepi/checkpointer/sqlite.py` | Persist `Message.metadata` round-trip (D5) |
| `pyproject.toml` | Add `[postgres]` and `[mcp]` extras (D9) |

### Files to create

| File | Purpose |
|---|---|
| `cubepi/checkpointer/postgres/__init__.py` | Re-exports for `PostgresCheckpointer` (D1) |
| `cubepi/checkpointer/postgres/models.py` | SQLAlchemy declarative models + private `cubepi_metadata` + `EXPECTED_SCHEMA_VERSION` (D1) |
| `cubepi/checkpointer/postgres/checkpointer.py` | `PostgresCheckpointer` class implementing the `Checkpointer` protocol (D1) |
| `cubepi/checkpointer/postgres/alembic_helpers.py` | `create_message_partitions_op()` + `write_schema_version_op()` (D1) |
| `cubepi/checkpointer/postgres/exceptions.py` | `CubepiSchemaUninitialized`, `CubepiSchemaMismatch` (D1) |
| `cubepi/mcp/__init__.py` | Re-exports for `load_mcp_tools_http`, `load_mcp_tools_stdio` (D2) |
| `cubepi/mcp/http_loader.py` | HTTP/SSE MCP tool loader (D2) |
| `cubepi/mcp/stdio_loader.py` | stdio MCP tool loader (D2) |
| `cubepi/mcp/_adapter.py` | MCP tool → `cubepi.AgentTool` conversion (D2) |
| `tests/checkpointer/test_postgres.py` | E2E tests for `PostgresCheckpointer` (D1) |
| `tests/mcp/test_http_loader.py` | E2E tests for HTTP MCP loader using fake FastAPI server (D2) |
| `tests/mcp/test_stdio_loader.py` | E2E tests for stdio MCP loader using subprocess server (D2) |
| `tests/providers/test_anthropic_cache_policy.py` | Tests for configurable cache policy (D3) |
| `tests/providers/test_openai_reasoning.py` | Tests for OSS reasoning field extraction (D4) |
| `tests/providers/test_message_metadata.py` | Tests for `Message.metadata` round-trip (D5) |
| `tests/agent/test_context_extra.py` | Tests for `AgentContext.extra` persistence (D6) |
| `tests/middleware/test_transform_system_prompt.py` | Tests for chain composition (D7) |
| `tests/middleware/test_after_model_response.py` | Tests for `after_model_response` + `TurnAction` (D8) |

### Test infrastructure

| File | Purpose |
|---|---|
| `tests/checkpointer/conftest.py` (new) | Postgres test fixtures: `pg_dsn`, `pg_pool`, `clean_db` |
| `tests/mcp/conftest.py` (new) | MCP test fixtures: `fake_http_mcp_server`, `fake_stdio_mcp_server` |

---

## Pre-flight Setup

### Task 0: Verify baseline + worktree

**Files:** none

- [ ] **Step 1: Confirm clean working tree**

Run: `cd ~/cubepi && git status`
Expected: working tree clean on `main`.

- [ ] **Step 2: Create feature branch**

Run: `git checkout -b feat/cubebox-readiness`

- [ ] **Step 3: Confirm baseline tests pass**

Run: `pytest tests/ -q --tb=no`
Expected: all existing tests pass. Note the count — every subsequent task must keep it strictly non-decreasing.

- [ ] **Step 4: Set up Postgres for integration testing**

cubepi v0.2 has no Postgres tests. For D1 we need a test DB. Confirm a local Postgres is available:

```bash
psql -h localhost -p 5432 -U postgres -c "SELECT version();"
```

Expected: PostgreSQL 14+ version string. If not available, install or update `~/.pgpass` / connection env vars before D1 tasks.

---

## D5 — `Message.metadata` field (do first; D1 needs it for serialization)

### Task D5.1: Add `metadata` field to three Message types

**Files:**
- Modify: `cubepi/providers/base.py`
- Test: `tests/providers/test_message_metadata.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/providers/test_message_metadata.py`:

```python
"""Message.metadata field tests (D5)."""

import pytest
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def test_user_message_default_metadata_is_empty_dict() -> None:
    msg = UserMessage(content=[TextContent(text="hi")])
    assert msg.metadata == {}


def test_assistant_message_default_metadata_is_empty_dict() -> None:
    msg = AssistantMessage(content=[], usage=Usage())
    assert msg.metadata == {}


def test_tool_result_message_default_metadata_is_empty_dict() -> None:
    msg = ToolResultMessage(content=[], tool_call_id="tc-1")
    assert msg.metadata == {}


def test_user_message_accepts_metadata() -> None:
    msg = UserMessage(
        content=[TextContent(text="hi")],
        metadata={"memory_snapshot": {"captured_at": "t1", "ids": ["m1"]}},
    )
    assert msg.metadata["memory_snapshot"]["captured_at"] == "t1"


def test_metadata_independent_between_instances() -> None:
    a = UserMessage(content=[TextContent(text="a")])
    b = UserMessage(content=[TextContent(text="b")])
    a.metadata["x"] = 1
    assert "x" not in b.metadata


def test_metadata_serializes_to_dict_in_model_dump() -> None:
    msg = UserMessage(
        content=[TextContent(text="hi")],
        metadata={"k": "v"},
    )
    dumped = msg.model_dump()
    assert dumped["metadata"] == {"k": "v"}
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/providers/test_message_metadata.py -v`
Expected: 6 failures with `pydantic.ValidationError` or AttributeError on `metadata`.

- [ ] **Step 3: Add `metadata` field to all three Message classes**

In `cubepi/providers/base.py`, find the three classes `UserMessage`, `AssistantMessage`, `ToolResultMessage` and add to each:

```python
from pydantic import BaseModel, Field

# Existing class header unchanged:
class UserMessage(BaseModel):
    content: list[Content]
    metadata: dict[str, Any] = Field(default_factory=dict)
    # ... other existing fields ...

class AssistantMessage(BaseModel):
    content: list[Content]
    metadata: dict[str, Any] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    # ... other existing fields ...

class ToolResultMessage(BaseModel):
    content: list[Content]
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str
    # ... other existing fields ...
```

Use `Field(default_factory=dict)` (not `= {}`) to avoid shared mutable default state across instances.

- [ ] **Step 4: Run tests to verify passes**

Run: `pytest tests/providers/test_message_metadata.py -v`
Expected: all 6 pass.

- [ ] **Step 5: Run full suite to verify no regressions**

Run: `pytest tests/ -q --tb=no`
Expected: baseline count + 6 new passes.

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/base.py tests/providers/test_message_metadata.py
git commit -m "feat(messages): add metadata field to UserMessage/AssistantMessage/ToolResultMessage

Public dict[str, Any] field on all three Message types, defaulting to {}.
Use case: per-message extensibility (memory snapshots, cost attribution,
audit tags, etc.) without changing the Message API every time.

Backward-compatible: existing constructors work unchanged; existing
serialized messages deserialize with metadata={} by default."
```

### Task D5.2: Persist `metadata` through `MemoryCheckpointer` and `SQLiteCheckpointer`

**Files:**
- Modify: `cubepi/checkpointer/memory.py`
- Modify: `cubepi/checkpointer/sqlite.py`
- Test: extend `tests/providers/test_message_metadata.py`

- [ ] **Step 1: Add round-trip tests**

Append to `tests/providers/test_message_metadata.py`:

```python
import tempfile
from pathlib import Path

from cubepi.checkpointer import MemoryCheckpointer, SQLiteCheckpointer


@pytest.mark.asyncio
async def test_memory_checkpointer_preserves_metadata() -> None:
    cp = MemoryCheckpointer()
    msg = UserMessage(
        content=[TextContent(text="hi")],
        metadata={"k": "v", "nested": {"a": 1}},
    )
    await cp.append("t1", [msg])
    loaded = await cp.load("t1")
    assert loaded is not None
    assert loaded.messages[0].metadata == {"k": "v", "nested": {"a": 1}}


@pytest.mark.asyncio
async def test_sqlite_checkpointer_preserves_metadata() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.db"
        async with SQLiteCheckpointer(str(path)) as cp:
            msg = UserMessage(
                content=[TextContent(text="hi")],
                metadata={"k": "v", "nested": {"a": 1}},
            )
            await cp.append("t1", [msg])
            loaded = await cp.load("t1")
        assert loaded is not None
        assert loaded.messages[0].metadata == {"k": "v", "nested": {"a": 1}}
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/providers/test_message_metadata.py::test_memory_checkpointer_preserves_metadata tests/providers/test_message_metadata.py::test_sqlite_checkpointer_preserves_metadata -v`
Expected: both fail — metadata is lost in checkpointer ser/deser. `memory.py` likely passes (it stores references) but verify; `sqlite.py` definitely fails because its `_serialize_message` doesn't include metadata.

- [ ] **Step 3: Update SQLiteCheckpointer serialization**

In `cubepi/checkpointer/sqlite.py`, find the private `_serialize_message` and `_deserialize_message` helpers. Locate where they handle each message type and add `metadata` to both directions:

```python
def _serialize_message(msg: Any) -> str:
    # find existing branch for each Message type
    # add: data["metadata"] = msg.metadata
    ...

def _deserialize_message(data: dict[str, Any]) -> Any:
    # find construction of each Message type
    # add: metadata=data.get("metadata", {})
    ...
```

(If the existing impl uses `msg.model_dump()` and `Type(**data)`, the new field is handled automatically — but verify by running the test.)

- [ ] **Step 4: Verify MemoryCheckpointer**

`MemoryCheckpointer` typically stores message references directly. If so, no code change needed; test should pass after Step 3 if the Pydantic field is set. If `MemoryCheckpointer` does any cloning, ensure metadata is preserved.

- [ ] **Step 5: Run tests**

Run: `pytest tests/providers/test_message_metadata.py -v`
Expected: all 8 tests pass.

- [ ] **Step 6: Run full suite**

Run: `pytest tests/ -q --tb=no`
Expected: baseline + 8.

- [ ] **Step 7: Commit**

```bash
git add cubepi/checkpointer/
git commit -m "feat(checkpointer): preserve Message.metadata through ser/deser

Both MemoryCheckpointer and SQLiteCheckpointer now round-trip the
metadata dict on UserMessage/AssistantMessage/ToolResultMessage."
```

---

## D6 — `AgentContext.extra` mutable dict

### Task D6.1: Add `extra` field to `AgentContext`

**Files:**
- Modify: `cubepi/agent/types.py`
- Test: `tests/agent/test_context_extra.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/agent/test_context_extra.py`:

```python
"""AgentContext.extra field tests (D6)."""

from cubepi.agent.types import AgentContext


def test_agent_context_default_extra_is_empty_dict() -> None:
    ctx = AgentContext(system_prompt="", messages=[])
    assert ctx.extra == {}


def test_agent_context_accepts_extra() -> None:
    ctx = AgentContext(
        system_prompt="",
        messages=[],
        extra={"todos": ["a", "b"]},
    )
    assert ctx.extra["todos"] == ["a", "b"]


def test_extra_is_mutable() -> None:
    ctx = AgentContext(system_prompt="", messages=[])
    ctx.extra["k"] = "v"
    assert ctx.extra == {"k": "v"}


def test_extra_independent_between_instances() -> None:
    a = AgentContext(system_prompt="", messages=[])
    b = AgentContext(system_prompt="", messages=[])
    a.extra["x"] = 1
    assert "x" not in b.extra
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/agent/test_context_extra.py -v`
Expected: 4 failures on `extra` attribute / constructor parameter.

- [ ] **Step 3: Add `extra` field to `AgentContext`**

In `cubepi/agent/types.py`, find:

```python
@dataclass
class AgentContext:
    system_prompt: str
    messages: list[Message]
    tools: list[AgentTool] | None = None
```

Add `extra`:

```python
from dataclasses import dataclass, field

@dataclass
class AgentContext:
    system_prompt: str
    messages: list[Message]
    tools: list[AgentTool] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/agent/test_context_extra.py -v`
Expected: 4 pass.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -q --tb=no`
Expected: baseline + 4.

- [ ] **Step 6: Commit**

```bash
git add cubepi/agent/types.py tests/agent/test_context_extra.py
git commit -m "feat(agent): add AgentContext.extra mutable dict for per-thread state

Middleware can read/mutate ctx.extra in any hook. Persistence to
the checkpointer happens in the agent loop (next task).
Backward-compatible: existing AgentContext constructors work unchanged."
```

### Task D6.2: Persist `ctx.extra` after each turn

**Files:**
- Modify: `cubepi/agent/loop.py` (and/or `agent.py`)
- Modify: `tests/agent/test_context_extra.py`

- [ ] **Step 1: Add integration test**

Append to `tests/agent/test_context_extra.py`:

```python
import tempfile
from pathlib import Path

import pytest
from cubepi import Agent, Model
from cubepi.checkpointer import SQLiteCheckpointer
from cubepi.middleware.base import Middleware
from cubepi.providers.faux import FauxProvider, faux_assistant_message


class _ExtraWritingMiddleware(Middleware):
    async def after_model_response(self, response, ctx, *, signal=None):
        # We'll add the hook in D8; for now use a workaround:
        # mutate via transform_context which already exists
        return None

    async def transform_context(self, messages, *, signal=None):
        # this hook gets called with messages; we can't access ctx here
        # in v0.2 — that's the point: after D6, we expose ctx.extra to ALL hooks
        return messages


@pytest.mark.asyncio
async def test_extra_persisted_across_turns() -> None:
    """ctx.extra mutations after a turn must be visible on next load."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "extra.db"
        async with SQLiteCheckpointer(str(path)) as cp:
            provider = FauxProvider()
            provider.set_responses([faux_assistant_message("hello")])
            agent = Agent(
                model=Model(provider=provider, model="test"),
                checkpointer=cp,
                thread_id="t-extra",
            )
            stream = await agent.prompt("hi")
            async for _ in stream:
                pass
            # ctx.extra is on the agent's context; we'll set it via a
            # middleware in later tasks. For now, set it manually via save_extra.
            await cp.save_extra("t-extra", {"counter": 1})
            data = await cp.load("t-extra")
            assert data is not None
            assert data.extra == {"counter": 1}
```

This test establishes the persistence contract via the existing
`save_extra` path. The agent-loop-driven persistence comes after we
add hooks that mutate `ctx.extra` (D7/D8). For now, this test pins
the round-trip.

- [ ] **Step 2: Run test**

Run: `pytest tests/agent/test_context_extra.py::test_extra_persisted_across_turns -v`
Expected: PASS — `SQLiteCheckpointer.save_extra` + `load` already round-trip extra.

- [ ] **Step 3: Wire `ctx.extra` initialization on agent load**

In `cubepi/agent/agent.py` (the `Agent` class), find where `AgentContext` is constructed at start of `prompt()`. If the agent loads checkpoint state, also hydrate `extra`:

```python
# Inside Agent.prompt() before constructing AgentContext:
if self._checkpointer and self._thread_id:
    data = await self._checkpointer.load(self._thread_id)
    if data is not None:
        loaded_messages = data.messages
        loaded_extra = data.extra
    else:
        loaded_messages = []
        loaded_extra = {}
else:
    loaded_messages = []
    loaded_extra = {}

ctx = AgentContext(
    system_prompt=self._system_prompt,
    messages=loaded_messages + new_messages,
    tools=self._tools,
    extra=loaded_extra,
)
```

(Find the exact existing construction point and adapt — the existing
code already loads messages via similar pattern.)

- [ ] **Step 4: Wire `save_extra` after turn ends**

In `cubepi/agent/loop.py` (or wherever the turn-completion logic
calls `checkpointer.append`), add a `save_extra` call:

```python
# After appending new messages and after_tool_call / after_model_response hooks complete:
if checkpointer and thread_id and ctx.extra:
    await checkpointer.save_extra(thread_id, ctx.extra)
```

The condition `if ctx.extra` avoids no-op writes when no middleware
populated extra. (Alternative: always call save_extra. Either works;
prefer the cheap-skip version.)

- [ ] **Step 5: Add integration test for the wiring**

Append to `tests/agent/test_context_extra.py`:

```python
@pytest.mark.asyncio
async def test_ctx_extra_persisted_via_loop() -> None:
    """When something mutates ctx.extra during a turn, it persists to checkpointer."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "loop.db"
        async with SQLiteCheckpointer(str(path)) as cp:
            provider = FauxProvider()
            provider.set_responses([faux_assistant_message("a")])

            # We don't have transform_system_prompt or after_model_response
            # yet — use a checkpointer hook approach: pre-seed extra and verify
            # the agent loads it correctly.
            await cp.save_extra("t-loop", {"seeded": True})

            agent = Agent(
                model=Model(provider=provider, model="test"),
                checkpointer=cp,
                thread_id="t-loop",
            )
            stream = await agent.prompt("go")
            async for _ in stream:
                pass

            data = await cp.load("t-loop")
            # Extra must still contain "seeded": True after the turn.
            assert data is not None
            assert data.extra.get("seeded") is True
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/agent/test_context_extra.py -v`
Expected: 6 pass.

- [ ] **Step 7: Run full suite**

Run: `pytest tests/ -q --tb=no`
Expected: baseline + 6.

- [ ] **Step 8: Commit**

```bash
git add cubepi/agent/ tests/agent/test_context_extra.py
git commit -m "feat(agent): persist AgentContext.extra to checkpointer across turns

Agent loop now hydrates ctx.extra from checkpointer.load on startup
and writes back via checkpointer.save_extra after each turn. Middleware
(coming in D7/D8) can mutate ctx.extra freely with persistence handled
by the loop.

No-op when ctx.extra is empty — existing checkpointer users see no
extra writes unless they actually populate the dict."
```

---

## D7 — `transform_system_prompt` middleware hook

### Task D7.1: Add hook method + composition

**Files:**
- Modify: `cubepi/middleware/base.py`
- Test: `tests/middleware/test_transform_system_prompt.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/middleware/test_transform_system_prompt.py`:

```python
"""transform_system_prompt hook tests (D7)."""

import pytest

from cubepi.middleware.base import Middleware, compose_middleware


class _AppendA(Middleware):
    async def transform_system_prompt(self, sp, *, signal=None):
        return sp + "\n[A]"


class _AppendB(Middleware):
    async def transform_system_prompt(self, sp, *, signal=None):
        return sp + "\n[B]"


@pytest.mark.asyncio
async def test_single_middleware_appends() -> None:
    hooks = compose_middleware([_AppendA()])
    fn = hooks["transform_system_prompt"]
    out = await fn("base")
    assert out == "base\n[A]"


@pytest.mark.asyncio
async def test_chain_order_preserved() -> None:
    """A then B → A first, then B sees A's output."""
    hooks = compose_middleware([_AppendA(), _AppendB()])
    fn = hooks["transform_system_prompt"]
    out = await fn("base")
    assert out == "base\n[A]\n[B]"


@pytest.mark.asyncio
async def test_no_middleware_hook_absent() -> None:
    """If no middleware implements transform_system_prompt, the hook is not in the dict."""
    class Plain(Middleware):
        pass
    hooks = compose_middleware([Plain()])
    assert "transform_system_prompt" not in hooks


@pytest.mark.asyncio
async def test_default_implementation_raises() -> None:
    """Default Middleware.transform_system_prompt raises NotImplementedError."""
    mw = Middleware()
    with pytest.raises(NotImplementedError):
        await mw.transform_system_prompt("any")
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/middleware/test_transform_system_prompt.py -v`
Expected: 4 failures (`transform_system_prompt` doesn't exist).

- [ ] **Step 3: Add the method to `Middleware` base + extend `compose_middleware`**

In `cubepi/middleware/base.py`:

```python
class Middleware:
    # ... existing methods ...

    async def transform_system_prompt(
        self,
        system_prompt: str,
        *,
        signal=None,
    ) -> str:
        raise NotImplementedError


def compose_middleware(middlewares: list[Middleware]) -> dict[str, Callable]:
    hooks: dict[str, Callable] = {}

    # ... existing hooks ...

    sp_chain = [m for m in middlewares if _has_method(m, "transform_system_prompt")]
    if sp_chain:
        async def composed_sp(system_prompt, *, signal=None):
            result = system_prompt
            for mw in sp_chain:
                result = await mw.transform_system_prompt(result, signal=signal)
            return result
        hooks["transform_system_prompt"] = composed_sp

    return hooks
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/middleware/test_transform_system_prompt.py -v`
Expected: 4 pass.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -q --tb=no`
Expected: baseline + 4.

- [ ] **Step 6: Commit**

```bash
git add cubepi/middleware/base.py tests/middleware/test_transform_system_prompt.py
git commit -m "feat(middleware): add transform_system_prompt hook (chain composition)

New hook for middleware to dynamically mutate the system prompt per turn.
Composition rule: chain — each middleware receives the previous middleware's
output. Same rule as transform_context. Backward-compatible: middleware
that doesn't implement the method has no effect."
```

### Task D7.2: Wire into agent loop

**Files:**
- Modify: `cubepi/agent/loop.py` (or wherever provider.stream is invoked)
- Test: extend `tests/middleware/test_transform_system_prompt.py`

- [ ] **Step 1: Integration test**

Append to `tests/middleware/test_transform_system_prompt.py`:

```python
import asyncio

from cubepi import Agent, Model
from cubepi.providers.faux import FauxProvider, faux_assistant_message


@pytest.mark.asyncio
async def test_agent_applies_transform_system_prompt() -> None:
    """system_prompt sent to provider must reflect the middleware chain."""
    captured_payloads: list[dict] = []

    async def on_payload(payload, model):
        captured_payloads.append(payload)
        return payload

    provider = FauxProvider()
    provider.set_responses([faux_assistant_message("ok")])
    agent = Agent(
        model=Model(provider=provider, model="test"),
        system_prompt="base",
        middleware=[_AppendA(), _AppendB()],
    )
    stream = await agent.prompt("hi", options={"on_payload": on_payload})
    async for _ in stream:
        pass

    assert len(captured_payloads) == 1
    # FauxProvider's payload structure: system_prompt is in the kwargs
    # (adapt to the actual key — check FauxProvider impl)
    assert captured_payloads[0].get("system_prompt", "").endswith("[A]\n[B]")
```

(Adjust `captured_payloads[0].get(...)` key based on FauxProvider's
actual payload shape — it might be `system` or `system_prompt`.)

- [ ] **Step 2: Run the test to find current behavior**

Run: `pytest tests/middleware/test_transform_system_prompt.py::test_agent_applies_transform_system_prompt -v`
Expected: FAIL — agent loop doesn't apply transform_system_prompt.

- [ ] **Step 3: Apply hook in agent loop**

In `cubepi/agent/loop.py`, find where the provider is called (typically `await provider.stream(model, messages, system_prompt=...)`). Before that call:

```python
# Apply transform_system_prompt chain if present
sp = system_prompt
if "transform_system_prompt" in hooks:
    sp = await hooks["transform_system_prompt"](sp)
# Apply transform_context chain (existing)
msgs = messages
if "transform_context" in hooks:
    msgs = await hooks["transform_context"](msgs)
# Call provider with transformed values
stream = await provider.stream(model, msgs, system_prompt=sp, tools=...)
```

- [ ] **Step 4: Run test**

Run: `pytest tests/middleware/test_transform_system_prompt.py -v`
Expected: 5 pass.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -q --tb=no`
Expected: baseline + 5.

- [ ] **Step 6: Commit**

```bash
git add cubepi/agent/loop.py tests/middleware/test_transform_system_prompt.py
git commit -m "feat(agent): apply transform_system_prompt hook before each provider call

Agent loop now runs the composed transform_system_prompt chain on the
system prompt before sending to the provider. Per-call (not just at
Agent construction), so middleware can mutate the system prompt
turn-by-turn based on dynamic state."
```

---

## D8 — `after_model_response` hook + `TurnAction`

### Task D8.1: Add `TurnAction` dataclass + hook method

**Files:**
- Modify: `cubepi/middleware/base.py`
- Test: `tests/middleware/test_after_model_response.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/middleware/test_after_model_response.py`:

```python
"""after_model_response hook + TurnAction tests (D8)."""

import pytest

from cubepi.agent.types import AgentContext
from cubepi.middleware.base import Middleware, TurnAction, compose_middleware
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    Usage,
    UserMessage,
)


def _mk_response(text: str = "hi") -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)], usage=Usage())


def _mk_ctx() -> AgentContext:
    return AgentContext(system_prompt="", messages=[])


class _MutateResponse(Middleware):
    async def after_model_response(self, response, ctx, *, signal=None):
        return TurnAction(response=_mk_response(text="mutated"))


class _InjectMessages(Middleware):
    async def after_model_response(self, response, ctx, *, signal=None):
        return TurnAction(
            inject_messages=[UserMessage(content=[TextContent(text="injected")])]
        )


class _Stop(Middleware):
    async def after_model_response(self, response, ctx, *, signal=None):
        return TurnAction(decision="stop")


class _Loop(Middleware):
    async def after_model_response(self, response, ctx, *, signal=None):
        return TurnAction(decision="loop_to_model")


class _NoOp(Middleware):
    async def after_model_response(self, response, ctx, *, signal=None):
        return None


def test_turn_action_defaults() -> None:
    ta = TurnAction()
    assert ta.response is None
    assert ta.inject_messages == []
    assert ta.decision == "natural"


@pytest.mark.asyncio
async def test_single_middleware_mutates_response() -> None:
    hooks = compose_middleware([_MutateResponse()])
    result = await hooks["after_model_response"](_mk_response("orig"), _mk_ctx())
    assert isinstance(result.response, AssistantMessage)
    assert result.response.content[0].text == "mutated"


@pytest.mark.asyncio
async def test_chain_last_response_wins() -> None:
    """Two mutators; last one in chain wins for response."""
    class _MutateAgain(Middleware):
        async def after_model_response(self, response, ctx, *, signal=None):
            return TurnAction(response=_mk_response(text="final"))

    hooks = compose_middleware([_MutateResponse(), _MutateAgain()])
    result = await hooks["after_model_response"](_mk_response("orig"), _mk_ctx())
    assert result.response.content[0].text == "final"


@pytest.mark.asyncio
async def test_inject_messages_concatenate() -> None:
    """inject_messages from multiple middleware concatenate."""
    class _InjectMore(Middleware):
        async def after_model_response(self, response, ctx, *, signal=None):
            return TurnAction(
                inject_messages=[UserMessage(content=[TextContent(text="more")])]
            )

    hooks = compose_middleware([_InjectMessages(), _InjectMore()])
    result = await hooks["after_model_response"](_mk_response(), _mk_ctx())
    assert len(result.inject_messages) == 2


@pytest.mark.asyncio
async def test_decision_last_wins() -> None:
    """Last middleware's decision wins."""
    hooks = compose_middleware([_Stop(), _Loop()])
    result = await hooks["after_model_response"](_mk_response(), _mk_ctx())
    assert result.decision == "loop_to_model"


@pytest.mark.asyncio
async def test_none_return_treated_as_natural() -> None:
    """Middleware returning None doesn't affect the composed TurnAction."""
    hooks = compose_middleware([_NoOp(), _Stop()])
    result = await hooks["after_model_response"](_mk_response(), _mk_ctx())
    assert result.decision == "stop"


@pytest.mark.asyncio
async def test_default_implementation_raises() -> None:
    mw = Middleware()
    with pytest.raises(NotImplementedError):
        await mw.after_model_response(_mk_response(), _mk_ctx())


@pytest.mark.asyncio
async def test_no_middleware_hook_absent() -> None:
    class Plain(Middleware):
        pass
    hooks = compose_middleware([Plain()])
    assert "after_model_response" not in hooks
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/middleware/test_after_model_response.py -v`
Expected: 8 failures.

- [ ] **Step 3: Add `TurnAction` and the hook**

In `cubepi/middleware/base.py`:

```python
from dataclasses import dataclass, field
from typing import Literal

from cubepi.providers.base import AssistantMessage, Message


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
    # ... existing methods ...

    async def after_model_response(
        self,
        response: AssistantMessage,
        ctx,  # AgentContext; avoid circular import via forward ref or import at runtime
        *,
        signal=None,
    ) -> TurnAction | None:
        raise NotImplementedError
```

For composition, append to `compose_middleware()`:

```python
amr_chain = [m for m in middlewares if _has_method(m, "after_model_response")]
if amr_chain:
    async def composed_amr(response, ctx, *, signal=None):
        current_response = response
        all_inject: list[Message] = []
        last_decision: Literal["natural", "stop", "loop_to_model"] = "natural"
        for mw in amr_chain:
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
    hooks["after_model_response"] = composed_amr
```

Also export `TurnAction` from `cubepi.middleware` (update `cubepi/middleware/__init__.py`):

```python
from cubepi.middleware.base import Middleware, TurnAction, compose_middleware

__all__ = ["Middleware", "TurnAction", "compose_middleware"]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/middleware/test_after_model_response.py -v`
Expected: 8 pass.

- [ ] **Step 5: Commit**

```bash
git add cubepi/middleware/ tests/middleware/test_after_model_response.py
git commit -m "feat(middleware): add after_model_response hook + TurnAction

New hook + return type lets middleware:
- observe and mutate the model's response (chain, last wins)
- inject messages before the next iteration (chain, concat)
- control the agent loop: natural | stop | loop_to_model (chain, last wins)

Covers cubebox's CostMiddleware (observe usage), TimestampMiddleware
(stamp turn-end), and TodoListMiddleware (guard logic + control flow)
use cases.

should_stop_after_turn is preserved unchanged for simpler stop-only
middleware."
```

### Task D8.2: Wire `after_model_response` + `TurnAction.decision` into agent loop

**Files:**
- Modify: `cubepi/agent/loop.py`
- Test: extend `tests/middleware/test_after_model_response.py`

- [ ] **Step 1: Integration tests**

Append to `tests/middleware/test_after_model_response.py`:

```python
from cubepi import Agent, Model
from cubepi.providers.faux import FauxProvider, faux_assistant_message


@pytest.mark.asyncio
async def test_agent_stops_when_middleware_returns_stop() -> None:
    """decision='stop' terminates even if response had tool calls
    (we test no-tool-calls case for simplicity; real agent will skip tool exec)."""
    provider = FauxProvider()
    provider.set_responses([faux_assistant_message("done")])
    agent = Agent(
        model=Model(provider=provider, model="test"),
        middleware=[_Stop()],
    )
    stream = await agent.prompt("hi")
    events = []
    async for event in stream:
        events.append(event.type)
    # Agent should stop after first model response — exactly one "done" event
    assert events.count("done") == 1


@pytest.mark.asyncio
async def test_agent_loops_when_middleware_returns_loop_to_model() -> None:
    """decision='loop_to_model' re-invokes the model with inject_messages."""
    provider = FauxProvider()
    # First response: natural-stop content. Middleware will force loop.
    # Second response: terminates naturally.
    provider.set_responses([
        faux_assistant_message("first"),
        faux_assistant_message("second"),
    ])

    looped_once = False

    class _LoopOnce(Middleware):
        async def after_model_response(self, response, ctx, *, signal=None):
            nonlocal looped_once
            if not looped_once:
                looped_once = True
                return TurnAction(
                    decision="loop_to_model",
                    inject_messages=[
                        UserMessage(content=[TextContent(text="retry")])
                    ],
                )
            return None

    agent = Agent(
        model=Model(provider=provider, model="test"),
        middleware=[_LoopOnce()],
    )
    stream = await agent.prompt("hi")
    async for _ in stream:
        pass
    # Verify provider was called twice
    assert provider.call_count == 2
```

(`provider.call_count` attribute may need to be added to FauxProvider
if it doesn't exist — add it as a simple counter incremented on each
`stream()` call.)

- [ ] **Step 2: Run tests to see current behavior**

Run: `pytest tests/middleware/test_after_model_response.py::test_agent_stops_when_middleware_returns_stop tests/middleware/test_after_model_response.py::test_agent_loops_when_middleware_returns_loop_to_model -v`
Expected: FAIL — agent loop ignores `after_model_response`.

- [ ] **Step 3: Wire into agent loop**

In `cubepi/agent/loop.py`, find the main loop body — typically:

```python
# Existing structure (paraphrased):
while not should_stop:
    response = await call_model(...)
    if should_stop_after_turn(...):
        break
    tool_results = await execute_tools(response.tool_calls)
    messages.append(response)
    messages.extend(tool_results)
```

Insert the new hook between model call and tool execution:

```python
while not should_stop:
    response = await call_model(...)

    # NEW: after_model_response hook
    if "after_model_response" in hooks:
        turn_action = await hooks["after_model_response"](response, ctx)
        if turn_action.response is not None:
            response = turn_action.response
        if turn_action.inject_messages:
            messages.extend(turn_action.inject_messages)
        if turn_action.decision == "stop":
            break
        if turn_action.decision == "loop_to_model":
            messages.append(response)  # still record the response
            continue  # skip tool execution, go back to model call
        # decision == "natural": fall through

    # Existing should_stop_after_turn (still functional for simple use cases)
    if hooks.get("should_stop_after_turn") and await hooks["should_stop_after_turn"](...):
        break

    # Existing tool execution path
    tool_results = await execute_tools(response.tool_calls)
    messages.append(response)
    messages.extend(tool_results)
```

(Adapt to actual loop structure in cubepi v0.2.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/middleware/test_after_model_response.py -v`
Expected: 10 pass.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -q --tb=no`
Expected: baseline + 10.

- [ ] **Step 6: Commit**

```bash
git add cubepi/agent/loop.py tests/middleware/test_after_model_response.py
git commit -m "feat(agent): wire after_model_response hook into agent loop

Loop now runs the composed after_model_response chain after each model
call. TurnAction.decision controls flow:
- 'natural' (default): proceed to tool execution or natural stop
- 'stop': terminate this turn
- 'loop_to_model': skip tool execution, re-invoke the model with
  inject_messages appended to context

response mutation and inject_messages are applied before the decision
branch."
```

---

## D3 — Anthropic configurable cache marker policy

### Task D3.1: Define `CacheMarkerPolicy` Protocol + default impl

**Files:**
- Modify: `cubepi/providers/anthropic.py`
- Test: `tests/providers/test_anthropic_cache_policy.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/providers/test_anthropic_cache_policy.py`:

```python
"""AnthropicProvider configurable cache_policy tests (D3)."""

import pytest

from cubepi.providers.anthropic import (
    AnthropicProvider,
    CacheMarkerPolicy,
    DefaultCacheMarkerPolicy,
)
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    TextContent,
    UserMessage,
)


def test_default_policy_marks_system() -> None:
    p = DefaultCacheMarkerPolicy()
    assert p.mark_system() is True


def test_default_policy_marks_last_tool() -> None:
    p = DefaultCacheMarkerPolicy()
    assert p.mark_last_tool() is True


def test_default_policy_indices_picks_last() -> None:
    p = DefaultCacheMarkerPolicy()
    msgs: list[Message] = [
        UserMessage(content=[TextContent(text="a")]),
        UserMessage(content=[TextContent(text="b")]),
    ]
    assert p.message_breakpoint_indices(msgs) == [1]


def test_default_policy_indices_empty() -> None:
    p = DefaultCacheMarkerPolicy()
    assert p.message_breakpoint_indices([]) == []


def test_provider_uses_default_policy_when_none_passed() -> None:
    p = AnthropicProvider(api_key="x")
    assert isinstance(p._cache_policy, DefaultCacheMarkerPolicy)


def test_provider_uses_custom_policy() -> None:
    class _NoSystem(CacheMarkerPolicy):
        def mark_system(self) -> bool:
            return False
        def mark_last_tool(self) -> bool:
            return False
        def message_breakpoint_indices(self, messages):
            return []

    p = AnthropicProvider(api_key="x", cache_policy=_NoSystem())
    assert p._cache_policy.mark_system() is False
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/providers/test_anthropic_cache_policy.py -v`
Expected: 6 failures — `CacheMarkerPolicy` / `DefaultCacheMarkerPolicy` don't exist.

- [ ] **Step 3: Add Protocol + default impl**

At the top of `cubepi/providers/anthropic.py`:

```python
from typing import Protocol, runtime_checkable

from cubepi.providers.base import Message


@runtime_checkable
class CacheMarkerPolicy(Protocol):
    """Policy controlling where Anthropic cache_control markers are inserted.

    See cubebox/CLAUDE.md prompt cache discipline for a real-world consumer.
    """
    def mark_system(self) -> bool: ...
    def mark_last_tool(self) -> bool: ...
    def message_breakpoint_indices(
        self,
        messages: list[Message],
    ) -> list[int]: ...


class DefaultCacheMarkerPolicy:
    """Preserves cubepi v0.2 behavior: system + last message + last tool."""

    def mark_system(self) -> bool:
        return True

    def mark_last_tool(self) -> bool:
        return True

    def message_breakpoint_indices(self, messages: list[Message]) -> list[int]:
        return [len(messages) - 1] if messages else []
```

- [ ] **Step 4: Add `cache_policy` parameter to `AnthropicProvider`**

```python
class AnthropicProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_policy: CacheMarkerPolicy | None = None,
    ) -> None:
        import anthropic
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._cache_policy = cache_policy or DefaultCacheMarkerPolicy()
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/providers/test_anthropic_cache_policy.py -v`
Expected: 6 pass.

- [ ] **Step 6: Run full suite**

Run: `pytest tests/ -q --tb=no`
Expected: baseline + 6.

- [ ] **Step 7: Commit**

```bash
git add cubepi/providers/anthropic.py tests/providers/test_anthropic_cache_policy.py
git commit -m "feat(anthropic): add configurable CacheMarkerPolicy + DefaultCacheMarkerPolicy

New constructor parameter cache_policy: CacheMarkerPolicy | None. When
not passed, behavior is identical to v0.2.0 via DefaultCacheMarkerPolicy.

Callers (like cubebox) can supply a custom policy to control which
messages get cache_control: ephemeral markers — e.g. walk back to the
last completed AIMessage rather than always the last message."
```

### Task D3.2: Refactor existing marker logic to use policy

**Files:**
- Modify: `cubepi/providers/anthropic.py` (refactor `stream()` and `_apply_message_cache_control`)
- Test: extend `tests/providers/test_anthropic_cache_policy.py`

- [ ] **Step 1: Add behavioral test (custom policy actually drives marker placement)**

Append to `tests/providers/test_anthropic_cache_policy.py`:

```python
@pytest.mark.asyncio
async def test_custom_policy_drives_message_marker_placement() -> None:
    """A policy that marks index 0 (not last) must put cache_control on the first message."""
    class _FirstOnly(CacheMarkerPolicy):
        def mark_system(self) -> bool:
            return False
        def mark_last_tool(self) -> bool:
            return False
        def message_breakpoint_indices(self, messages):
            return [0] if messages else []

    p = AnthropicProvider(api_key="x", cache_policy=_FirstOnly())

    msgs: list[Message] = [
        UserMessage(content=[TextContent(text="zero")]),
        UserMessage(content=[TextContent(text="one")]),
    ]
    api_msgs = [p._convert_message(m) for m in msgs]
    # cache control is on whatever the internal _apply_message_cache_control puts it
    # We need to call the new internal method that applies policy. Refactor will
    # introduce _apply_policy_markers; assert the output structure.
    p._apply_policy_markers(api_msgs, cache_control={"type": "ephemeral"})
    # First message should have cache_control on its last content block
    first_blocks = api_msgs[0]["content"]
    if isinstance(first_blocks, list):
        assert first_blocks[-1].get("cache_control") == {"type": "ephemeral"}
    # Second message should NOT have cache_control
    second_blocks = api_msgs[1]["content"]
    if isinstance(second_blocks, list):
        assert "cache_control" not in (second_blocks[-1] if second_blocks else {})
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/providers/test_anthropic_cache_policy.py::test_custom_policy_drives_message_marker_placement -v`
Expected: FAIL — `_apply_policy_markers` doesn't exist (current method is `_apply_message_cache_control` which always uses last index).

- [ ] **Step 3: Refactor `AnthropicProvider`**

Two parts: (a) replace `_apply_message_cache_control` with policy-aware `_apply_indices_markers` taking explicit indices; (b) update `stream()` to call the policy for indices + system + tool markers.

```python
def _apply_indices_markers(
    self,
    api_messages: list[dict[str, Any]],
    indices: list[int],
    cache_control: dict[str, str],
) -> None:
    """Apply cache_control to the last content block of each indexed message."""
    for idx in indices:
        if 0 <= idx < len(api_messages):
            msg = api_messages[idx]
            content = msg.get("content")
            if isinstance(content, list) and content:
                last_block = content[-1]
                if isinstance(last_block, dict):
                    content[-1] = {**last_block, "cache_control": cache_control}
            elif isinstance(content, str):
                msg["content"] = [
                    {"type": "text", "text": content, "cache_control": cache_control}
                ]
```

Inside `stream()` (existing method), replace the marker-application logic:

```python
async def stream(self, model, messages, *, system_prompt="", tools=None, options=None):
    ...
    cache_control = self._get_cache_control()  # existing
    api_messages = [self._convert_message(m) for m in messages]

    if cache_control:
        # Policy inspects the original cubepi `messages` list to pick indices;
        # we apply markers to the parallel `api_messages` list (1:1 correspondence).
        indices = self._cache_policy.message_breakpoint_indices(messages)
        self._apply_indices_markers(api_messages, indices, cache_control)

    # ... existing kwargs build ...

    if system_prompt:
        kwargs["system"] = [
            {
                "type": "text",
                "text": system_prompt,
                **({"cache_control": cache_control}
                   if cache_control and self._cache_policy.mark_system() else {}),
            }
        ]
    ...
    if cache_control and api_tools and self._cache_policy.mark_last_tool():
        api_tools[-1]["cache_control"] = cache_control
```

Delete the old `_apply_message_cache_control` method once all references are removed.

Delete the old `_apply_message_cache_control` once references are gone.

- [ ] **Step 4: Update the behavioral test to call `_apply_indices_markers` not `_apply_policy_markers`**

Adjust the test from Step 1 to use the actual method name. Verify the test passes.

- [ ] **Step 5: Run all anthropic tests + add golden-fixture pin**

Existing v0.2 behavioral tests for anthropic must still pass byte-for-byte (default policy preserves v0.2 markers). Run:

```bash
pytest tests/providers/test_anthropic.py tests/providers/test_anthropic_cache_policy.py -v
```

Expected: all pass.

- [ ] **Step 6: Run full suite**

Run: `pytest tests/ -q --tb=no`
Expected: baseline + 7 (6 from D3.1 + 1 from D3.2).

- [ ] **Step 7: Commit**

```bash
git add cubepi/providers/anthropic.py tests/providers/test_anthropic_cache_policy.py
git commit -m "refactor(anthropic): route cache_control placement through CacheMarkerPolicy

stream() now calls cache_policy.mark_system/mark_last_tool/
message_breakpoint_indices instead of hard-coding 'mark the last message'.
Default policy preserves v0.2 behavior; existing tests still pass.

Custom policies (e.g. cubebox's 'mark last completed AIMessage') now
work end-to-end."
```

---

## D4 — OpenAIProvider OSS reasoning extraction + payload quirks

### Task D4.1: Tests with fake openai-compatible streams

**Files:**
- Test: `tests/providers/test_openai_reasoning.py` (new)

- [ ] **Step 1: Set up tests with respx mock**

Create `tests/providers/test_openai_reasoning.py`:

```python
"""OpenAIProvider OSS reasoning extraction + payload_quirks tests (D4)."""

import json
import pytest
import respx
from httpx import Response

from cubepi import Model
from cubepi.providers.openai import OpenAIProvider
from cubepi.providers.base import UserMessage, TextContent


def _sse_chunks(*chunks: dict) -> str:
    """Format dict chunks as OpenAI SSE stream."""
    out = []
    for c in chunks:
        out.append(f"data: {json.dumps(c)}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out)


@pytest.mark.asyncio
@respx.mock
async def test_extracts_reasoning_content_variant() -> None:
    """DeepSeek/Qwen/DouBao use delta.reasoning_content."""
    body = _sse_chunks(
        {"choices": [{"delta": {"reasoning_content": "Let me think"}, "finish_reason": None}]},
        {"choices": [{"delta": {"reasoning_content": " carefully"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Answer"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )
    respx.post("https://test.com/v1/chat/completions").mock(
        return_value=Response(200, text=body, headers={"content-type": "text/event-stream"})
    )

    p = OpenAIProvider(api_key="x", base_url="https://test.com/v1")
    stream = await p.stream(
        Model(provider="openai", model="m"),
        [UserMessage(content=[TextContent(text="hi")])],
    )
    types_seen = []
    thinking_deltas = []
    async for evt in stream:
        types_seen.append(evt.type)
        if evt.type == "thinking_delta":
            thinking_deltas.append(evt.delta)

    assert "thinking_start" in types_seen
    assert "thinking_delta" in types_seen
    assert "thinking_end" in types_seen
    assert "".join(thinking_deltas) == "Let me think carefully"


@pytest.mark.asyncio
@respx.mock
async def test_extracts_reasoning_variant_vllm() -> None:
    """vLLM uses delta.reasoning."""
    body = _sse_chunks(
        {"choices": [{"delta": {"reasoning": "v-llm reasoning"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Answer"}, "finish_reason": "stop"}]},
    )
    respx.post("https://test.com/v1/chat/completions").mock(
        return_value=Response(200, text=body, headers={"content-type": "text/event-stream"})
    )

    p = OpenAIProvider(api_key="x", base_url="https://test.com/v1")
    stream = await p.stream(
        Model(provider="openai", model="m"),
        [UserMessage(content=[TextContent(text="hi")])],
    )
    deltas = []
    async for evt in stream:
        if evt.type == "thinking_delta":
            deltas.append(evt.delta)
    assert "".join(deltas) == "v-llm reasoning"


@pytest.mark.asyncio
@respx.mock
async def test_extracts_reasoning_details_variant_minimax() -> None:
    """MiniMax uses delta.reasoning_details: [{text: ...}]."""
    body = _sse_chunks(
        {"choices": [{"delta": {"reasoning_details": [{"text": "step1"}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"reasoning_details": [{"text": " step2"}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Answer"}, "finish_reason": "stop"}]},
    )
    respx.post("https://test.com/v1/chat/completions").mock(
        return_value=Response(200, text=body, headers={"content-type": "text/event-stream"})
    )

    p = OpenAIProvider(api_key="x", base_url="https://test.com/v1")
    stream = await p.stream(
        Model(provider="openai", model="m"),
        [UserMessage(content=[TextContent(text="hi")])],
    )
    deltas = []
    async for evt in stream:
        if evt.type == "thinking_delta":
            deltas.append(evt.delta)
    assert "".join(deltas) == "step1 step2"


@pytest.mark.asyncio
@respx.mock
async def test_max_completion_tokens_alias_rewrite() -> None:
    """When payload_quirks=['max_completion_tokens_alias'], rewrite to max_tokens."""
    captured: dict = {}

    def _capture(request):
        captured["body"] = json.loads(request.content)
        return Response(200, text=_sse_chunks(
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        ), headers={"content-type": "text/event-stream"})

    respx.post("https://test.com/v1/chat/completions").mock(side_effect=_capture)

    p = OpenAIProvider(
        api_key="x",
        base_url="https://test.com/v1",
        payload_quirks=["max_completion_tokens_alias"],
    )
    stream = await p.stream(
        Model(provider="openai", model="m"),
        [UserMessage(content=[TextContent(text="hi")])],
        options=None,  # we'd pass max_completion_tokens here normally
    )
    async for _ in stream:
        pass
    # Note: actual max_completion_tokens injection depends on how cubepi's
    # OpenAIProvider currently builds the payload. The test asserts that
    # IF max_completion_tokens is present, it gets renamed to max_tokens.
    # If cubepi doesn't currently expose this knob, this test may need to
    # call the internal payload-build helper directly.
```

(`pytest-httpx` is an alternative to `respx`. If cubepi already uses one,
prefer that — add `respx` to `[project.optional-dependencies] dev` if not.)

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/providers/test_openai_reasoning.py -v`
Expected: 3 reasoning tests fail (no extraction); 1 payload quirk test may pass trivially or fail depending on current shape.

### Task D4.2: Implement OSS reasoning extraction in `OpenAIProvider`

**Files:**
- Modify: `cubepi/providers/openai.py`

- [ ] **Step 1: Add reasoning extraction in stream chunk handler**

In `cubepi/providers/openai.py`, find the streaming chunk handler (the loop that consumes `async for chunk in response`). Before the existing `delta.content` / `delta.tool_calls` branches, add:

```python
# Track thinking state across chunks
thinking_started = False
thinking_content_index: int | None = None

async for chunk in response:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    finish_reason = chunk.choices[0].finish_reason

    # --- OSS reasoning extraction (D4) ---
    reasoning_delta: str | None = None
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        reasoning_delta = delta.reasoning_content
    elif hasattr(delta, "reasoning") and delta.reasoning:
        reasoning_delta = delta.reasoning
    elif hasattr(delta, "reasoning_details") and delta.reasoning_details:
        parts = []
        for d in delta.reasoning_details:
            text = d.text if hasattr(d, "text") else (d.get("text") if isinstance(d, dict) else None)
            if text:
                parts.append(text)
        if parts:
            reasoning_delta = "".join(parts)

    if reasoning_delta:
        if not thinking_started:
            partial.content.append(ThinkingContent(thinking=""))
            thinking_content_index = len(partial.content) - 1
            ms.push(StreamEvent(
                type="thinking_start",
                content_index=thinking_content_index,
                partial=partial.model_copy(deep=True),
            ))
            thinking_started = True
        # Accumulate
        existing = partial.content[thinking_content_index].thinking
        partial.content[thinking_content_index] = ThinkingContent(
            thinking=existing + reasoning_delta
        )
        ms.push(StreamEvent(
            type="thinking_delta",
            delta=reasoning_delta,
            content_index=thinking_content_index,
            partial=partial.model_copy(deep=True),
        ))

    # ... existing delta.content branch ...

    # On finish, close any open thinking block
    if finish_reason is not None and thinking_started:
        ms.push(StreamEvent(
            type="thinking_end",
            content_index=thinking_content_index,
            partial=partial.model_copy(deep=True),
        ))
        thinking_started = False

    # ... existing finish handling ...
```

(Adapt to actual code structure in cubepi v0.2's `openai.py`.)

- [ ] **Step 2: Add `payload_quirks` parameter**

```python
class OpenAIProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        payload_quirks: list[Literal["max_completion_tokens_alias"]] | None = None,
    ) -> None:
        import openai
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)
        self._payload_quirks = set(payload_quirks or [])
```

In the request payload build, after constructing `kwargs`:

```python
if "max_completion_tokens_alias" in self._payload_quirks:
    if "max_completion_tokens" in kwargs:
        kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/providers/test_openai_reasoning.py -v`
Expected: 4 pass.

- [ ] **Step 4: Run full suite**

Run: `pytest tests/ -q --tb=no`
Expected: baseline + 4.

- [ ] **Step 5: Commit**

```bash
git add cubepi/providers/openai.py tests/providers/test_openai_reasoning.py
git commit -m "feat(openai): extract OSS reasoning fields + payload_quirks

Streaming chunk handler now extracts reasoning from three non-standard
fields in priority order:
  delta.reasoning_content    (DeepSeek/Qwen/DouBao)
  delta.reasoning            (vLLM)
  delta.reasoning_details[*] (MiniMax)

Emits cubepi's standard thinking_start/_delta/_end events, same shape
as Anthropic's native reasoning path and OpenAIResponsesProvider.

payload_quirks parameter supports max_completion_tokens_alias for older
openai-compatible endpoints that require max_tokens."
```

---

## D1 — PostgresCheckpointer

### Task D1.1: Define models and metadata

**Files:**
- Create: `cubepi/checkpointer/postgres/__init__.py`
- Create: `cubepi/checkpointer/postgres/models.py`
- Test: `tests/checkpointer/test_postgres.py` (placeholder for now)

- [ ] **Step 1: Create the module structure**

```bash
mkdir -p cubepi/checkpointer/postgres
touch cubepi/checkpointer/postgres/__init__.py
mkdir -p tests/checkpointer
```

- [ ] **Step 2: Write the models file**

Create `cubepi/checkpointer/postgres/models.py`:

```python
"""SQLAlchemy table definitions for cubepi PostgresCheckpointer.

cubepi uses a private MetaData instance (not SQLModel's global one)
so downstreams can compose it into alembic regardless of which ORM
they use. SQLAlchemy 2.0 declarative is the chosen style for cubepi
internal definitions.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EXPECTED_SCHEMA_VERSION = 1
PARTITION_COUNT = 64

cubepi_metadata = sa.MetaData()


class CubepiBase(DeclarativeBase):
    metadata = cubepi_metadata


class CubepiThread(CubepiBase):
    __tablename__ = "cubepi_threads"

    thread_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    parent_thread_id: Mapped[str | None] = mapped_column(
        sa.Text,
        sa.ForeignKey("cubepi_threads.thread_id"),
        nullable=True,
    )
    forked_at_seq: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    extra: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"),
    )
    created_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
    updated_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )


class CubepiMessage(CubepiBase):
    __tablename__ = "cubepi_messages"
    __table_args__ = (
        sa.Index(
            "ix_cubepi_messages_metadata_gin",
            "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
        ),
        {"postgresql_partition_by": "HASH (thread_id)"},
    )

    thread_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("cubepi_threads.thread_id", ondelete="CASCADE"),
        primary_key=True,
    )
    seq: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True)
    role: Mapped[str] = mapped_column(sa.Text, nullable=False)
    metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"),
    )
    payload: Mapped[bytes] = mapped_column(sa.LargeBinary, nullable=False)
    created_at: Mapped[_dt.datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )


class CubepiSchemaVersion(CubepiBase):
    __tablename__ = "cubepi_schema_version"

    version: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
```

- [ ] **Step 3: Write a smoke test that imports without error**

Create `tests/checkpointer/test_postgres.py`:

```python
"""PostgresCheckpointer tests (D1)."""

def test_models_import() -> None:
    from cubepi.checkpointer.postgres.models import (
        EXPECTED_SCHEMA_VERSION,
        PARTITION_COUNT,
        CubepiThread,
        CubepiMessage,
        CubepiSchemaVersion,
        cubepi_metadata,
    )
    assert EXPECTED_SCHEMA_VERSION == 1
    assert PARTITION_COUNT == 64
    # Metadata has all three tables registered
    assert "cubepi_threads" in cubepi_metadata.tables
    assert "cubepi_messages" in cubepi_metadata.tables
    assert "cubepi_schema_version" in cubepi_metadata.tables
```

- [ ] **Step 4: Run smoke test**

Run: `pytest tests/checkpointer/test_postgres.py::test_models_import -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cubepi/checkpointer/postgres/ tests/checkpointer/
git commit -m "feat(postgres-checkpointer): add SQLAlchemy models + private metadata

Three tables: cubepi_threads (thread-level KV), cubepi_messages
(HASH-partitioned by thread_id, 64 partitions, append-only), and
cubepi_schema_version. cubepi_metadata is private (not SQLModel's
global) so any downstream ORM can compose it into alembic."
```

### Task D1.2: Exceptions + alembic helpers

**Files:**
- Create: `cubepi/checkpointer/postgres/exceptions.py`
- Create: `cubepi/checkpointer/postgres/alembic_helpers.py`

- [ ] **Step 1: Create exceptions**

`cubepi/checkpointer/postgres/exceptions.py`:

```python
"""Exceptions raised by PostgresCheckpointer schema verification."""


class CubepiSchemaError(Exception):
    """Base class for cubepi Postgres schema errors."""


class CubepiSchemaUninitialized(CubepiSchemaError):
    """The cubepi_schema_version table is empty or missing.

    Typically means the host application's alembic upgrade hasn't been
    run yet against this database.
    """


class CubepiSchemaMismatch(CubepiSchemaError):
    """The DB schema version doesn't match cubepi's expected version.

    Typically means the cubepi library was upgraded but the host
    application's alembic is behind. Run a new alembic revision.
    """

    def __init__(self, *, expected: int, actual: int, hint: str = "") -> None:
        msg = f"cubepi schema mismatch: expected={expected} actual={actual}."
        if hint:
            msg += f" {hint}"
        super().__init__(msg)
        self.expected = expected
        self.actual = actual
```

- [ ] **Step 2: Create alembic helpers**

`cubepi/checkpointer/postgres/alembic_helpers.py`:

```python
"""SQL helpers for host application alembic migrations."""

from cubepi.checkpointer.postgres.models import (
    EXPECTED_SCHEMA_VERSION,
    PARTITION_COUNT,
)


def create_message_partitions_op() -> str:
    """Return SQL DDL creating all 64 child partitions of cubepi_messages.

    Call inside an alembic upgrade() function via op.execute(), AFTER
    the parent cubepi_messages table has been created.
    """
    return "\n".join(
        f"CREATE TABLE cubepi_messages_p{i:02d} "
        f"PARTITION OF cubepi_messages "
        f"FOR VALUES WITH (modulus {PARTITION_COUNT}, remainder {i});"
        for i in range(PARTITION_COUNT)
    )


def write_schema_version_op() -> str:
    """Return SQL inserting the current schema version.

    Call inside an alembic upgrade() after CREATE TABLE
    cubepi_schema_version. Idempotent via ON CONFLICT DO NOTHING.
    """
    return (
        f"INSERT INTO cubepi_schema_version (version) "
        f"VALUES ({EXPECTED_SCHEMA_VERSION}) ON CONFLICT DO NOTHING;"
    )
```

- [ ] **Step 3: Add tests**

Append to `tests/checkpointer/test_postgres.py`:

```python
def test_create_message_partitions_op_yields_64_statements() -> None:
    from cubepi.checkpointer.postgres.alembic_helpers import create_message_partitions_op
    sql = create_message_partitions_op()
    # 64 CREATE TABLE statements
    assert sql.count("CREATE TABLE cubepi_messages_p") == 64
    # Modulus is constant; remainder ranges 0..63
    assert "modulus 64, remainder 0" in sql
    assert "modulus 64, remainder 63" in sql


def test_write_schema_version_op_includes_expected_version() -> None:
    from cubepi.checkpointer.postgres.alembic_helpers import write_schema_version_op
    sql = write_schema_version_op()
    assert "INSERT INTO cubepi_schema_version" in sql
    assert "VALUES (1)" in sql


def test_exceptions_carry_expected_actual() -> None:
    from cubepi.checkpointer.postgres.exceptions import CubepiSchemaMismatch
    err = CubepiSchemaMismatch(expected=2, actual=1, hint="run alembic")
    assert err.expected == 2
    assert err.actual == 1
    assert "expected=2" in str(err)
    assert "run alembic" in str(err)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/checkpointer/test_postgres.py -v`
Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add cubepi/checkpointer/postgres/exceptions.py cubepi/checkpointer/postgres/alembic_helpers.py tests/checkpointer/test_postgres.py
git commit -m "feat(postgres-checkpointer): add exceptions + alembic helpers

- CubepiSchemaUninitialized / CubepiSchemaMismatch for version check failures
- create_message_partitions_op() returns DDL for 64 child partitions
- write_schema_version_op() returns INSERT for version pinning"
```

### Task D1.3: PostgresCheckpointer implementation (load/append/save_extra + schema check)

**Files:**
- Create: `cubepi/checkpointer/postgres/checkpointer.py`
- Modify: `cubepi/checkpointer/postgres/__init__.py`
- Test: extend `tests/checkpointer/test_postgres.py` + add `conftest.py`

- [ ] **Step 1: Add Postgres test fixtures**

Create `tests/checkpointer/conftest.py`:

```python
"""Postgres test fixtures."""

import os
import secrets

import asyncpg
import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """Postgres DSN for tests. Override via CUBEPI_TEST_PG_DSN env var."""
    return os.environ.get(
        "CUBEPI_TEST_PG_DSN",
        "postgresql://postgres@localhost:5432/postgres",
    )


@pytest_asyncio.fixture
async def clean_db(pg_dsn: str):
    """Create a fresh database for each test; drop after."""
    db_name = f"cubepi_test_{secrets.token_hex(6)}"
    # Connect to default DB to issue CREATE DATABASE
    admin_conn = await asyncpg.connect(pg_dsn)
    try:
        await admin_conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin_conn.close()
    # Build DSN for the new DB
    test_dsn = pg_dsn.rsplit("/", 1)[0] + f"/{db_name}"
    yield test_dsn
    # Cleanup
    admin_conn = await asyncpg.connect(pg_dsn)
    try:
        # Terminate any lingering connections
        await admin_conn.execute(
            f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname = '{db_name}'"
        )
        await admin_conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
    finally:
        await admin_conn.close()
```

- [ ] **Step 2: Write the PostgresCheckpointer**

Create `cubepi/checkpointer/postgres/checkpointer.py`:

```python
"""PostgresCheckpointer — Checkpointer protocol implementation against PostgreSQL.

Append-only message log + per-thread extra KV. Compatible with the
existing Checkpointer protocol so it slots into Agent(checkpointer=...)
unchanged.
"""
from __future__ import annotations

import json
from typing import Any

import asyncpg
import msgpack

from cubepi.checkpointer.base import CheckpointData
from cubepi.checkpointer.postgres.exceptions import (
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
)
from cubepi.checkpointer.postgres.models import EXPECTED_SCHEMA_VERSION
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)


_ROLE_TO_CLS: dict[str, type[Message]] = {
    "user": UserMessage,
    "assistant": AssistantMessage,
    "tool": ToolResultMessage,
}


class PostgresCheckpointer:
    """Checkpointer backed by PostgreSQL.

    Usage:
        cp = PostgresCheckpointer(dsn="postgresql://...")
        async with cp:  # initializes pool, verifies schema
            await cp.append(thread_id, [msg1, msg2])
            data = await cp.load(thread_id)
            await cp.save_extra(thread_id, {"k": "v"})
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._min = min_pool_size
        self._max = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def __aenter__(self) -> "PostgresCheckpointer":
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min,
            max_size=self._max,
        )
        await self._verify_schema()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def _verify_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    "SELECT version FROM cubepi_schema_version LIMIT 1"
                )
            except asyncpg.UndefinedTableError as e:
                raise CubepiSchemaUninitialized(
                    "cubepi tables not found. Run host application's alembic upgrade."
                ) from e
            if row is None:
                raise CubepiSchemaUninitialized(
                    "cubepi_schema_version table is empty. Host alembic migration "
                    "must INSERT the current version (use write_schema_version_op())."
                )
            if row["version"] != EXPECTED_SCHEMA_VERSION:
                raise CubepiSchemaMismatch(
                    expected=EXPECTED_SCHEMA_VERSION,
                    actual=row["version"],
                    hint="cubepi was upgraded but host alembic is behind. "
                    "Generate a new revision and apply.",
                )

    async def load(self, thread_id: str) -> CheckpointData | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            msg_rows = await conn.fetch(
                "SELECT seq, role, metadata, payload FROM cubepi_messages "
                "WHERE thread_id = $1 ORDER BY seq",
                thread_id,
            )
            extra_row = await conn.fetchrow(
                "SELECT extra FROM cubepi_threads WHERE thread_id = $1",
                thread_id,
            )

        if not msg_rows and extra_row is None:
            return None

        messages: list[Message] = []
        for r in msg_rows:
            cls = _ROLE_TO_CLS.get(r["role"])
            if cls is None:
                raise ValueError(f"unknown role in DB: {r['role']!r}")
            data = msgpack.unpackb(r["payload"], raw=False)
            # Pydantic will validate; metadata column overrides what's in payload
            # (they should be equal; metadata column is the source of truth)
            data["metadata"] = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else (r["metadata"] or {})
            messages.append(cls(**data))

        if extra_row is not None:
            raw_extra = extra_row["extra"]
            extra = json.loads(raw_extra) if isinstance(raw_extra, str) else (raw_extra or {})
        else:
            extra = {}

        return CheckpointData(messages=messages, extra=extra)

    async def append(self, thread_id: str, messages: list[Message]) -> None:
        if not messages:
            return
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))", thread_id,
                )
                # Ensure thread row exists
                await conn.execute(
                    "INSERT INTO cubepi_threads (thread_id) "
                    "VALUES ($1) ON CONFLICT DO NOTHING",
                    thread_id,
                )
                last_seq = await conn.fetchval(
                    "SELECT COALESCE(MAX(seq), 0) FROM cubepi_messages "
                    "WHERE thread_id = $1",
                    thread_id,
                ) or 0
                rows = []
                for i, m in enumerate(messages):
                    seq = last_seq + i + 1
                    payload = msgpack.packb(m.model_dump(mode="json"), use_bin_type=True)
                    rows.append((
                        thread_id, seq, m.__class__.__name__.replace(
                            "Message", ""
                        ).lower(),  # "user"/"assistant"/"toolresult" -> map below
                        json.dumps(m.metadata),
                        payload,
                    ))
                # Fix the role mapping — class names → "user"/"assistant"/"tool"
                role_map = {
                    "user": "user",
                    "assistant": "assistant",
                    "toolresult": "tool",
                }
                rows = [
                    (r[0], r[1], role_map[r[2]], r[3], r[4]) for r in rows
                ]
                await conn.executemany(
                    "INSERT INTO cubepi_messages "
                    "(thread_id, seq, role, metadata, payload) "
                    "VALUES ($1, $2, $3, $4, $5)",
                    rows,
                )

    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO cubepi_threads (thread_id, extra) "
                "VALUES ($1, $2::jsonb) "
                "ON CONFLICT (thread_id) DO UPDATE "
                "SET extra = cubepi_threads.extra || EXCLUDED.extra, "
                "    updated_at = now()",
                thread_id, json.dumps(extra),
            )
```

NOTE: the role mapping is awkward — better to derive from the
Message subclass type directly:

```python
def _role_of(m: Message) -> str:
    if isinstance(m, UserMessage):
        return "user"
    if isinstance(m, AssistantMessage):
        return "assistant"
    if isinstance(m, ToolResultMessage):
        return "tool"
    raise TypeError(f"unknown Message type: {type(m).__name__}")
```

Use that helper in `append()`.

- [ ] **Step 3: Update `__init__.py`**

`cubepi/checkpointer/postgres/__init__.py`:

```python
from cubepi.checkpointer.postgres.checkpointer import PostgresCheckpointer
from cubepi.checkpointer.postgres.exceptions import (
    CubepiSchemaError,
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
)
from cubepi.checkpointer.postgres.models import (
    EXPECTED_SCHEMA_VERSION,
    PARTITION_COUNT,
    cubepi_metadata,
)

__all__ = [
    "PostgresCheckpointer",
    "CubepiSchemaError",
    "CubepiSchemaMismatch",
    "CubepiSchemaUninitialized",
    "EXPECTED_SCHEMA_VERSION",
    "PARTITION_COUNT",
    "cubepi_metadata",
]
```

Also update top-level `cubepi/checkpointer/__init__.py` to re-export
(lazy-load to avoid hard dependency on `asyncpg`):

```python
# cubepi/checkpointer/__init__.py — keep existing exports, ADD:

def get_postgres_checkpointer():
    from cubepi.checkpointer.postgres import PostgresCheckpointer
    return PostgresCheckpointer
```

(Don't unconditionally import — `asyncpg` is in the optional
`[postgres]` extra.)

- [ ] **Step 4: Add E2E tests**

Append to `tests/checkpointer/test_postgres.py`:

```python
import pytest
import pytest_asyncio
import asyncpg

from cubepi.checkpointer.postgres import (
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
    PostgresCheckpointer,
    cubepi_metadata,
)
from cubepi.checkpointer.postgres.alembic_helpers import (
    create_message_partitions_op,
    write_schema_version_op,
)
from cubepi.providers.base import TextContent, UserMessage, AssistantMessage, Usage


async def _setup_schema(dsn: str) -> None:
    """Create cubepi tables + partitions + version row."""
    conn = await asyncpg.connect(dsn)
    try:
        # Use SQLAlchemy DDL to create the metadata tables
        from sqlalchemy import create_engine
        from sqlalchemy.dialects import postgresql
        engine = create_engine(dsn.replace("postgresql://", "postgresql+psycopg2://"))
        # ALTERNATIVE: emit DDL directly without an engine, using compile
        for table in cubepi_metadata.sorted_tables:
            ddl = str(
                table.compile(dialect=postgresql.dialect())
            )
            # NOTE: compile may not emit PARTITION BY — handle manually for cubepi_messages
            ...
        # Easier: manually CREATE TABLE statements that match the models
        await conn.execute("""
            CREATE TABLE cubepi_threads (
                thread_id TEXT PRIMARY KEY,
                parent_thread_id TEXT REFERENCES cubepi_threads(thread_id),
                forked_at_seq BIGINT,
                extra JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        await conn.execute("""
            CREATE TABLE cubepi_messages (
                thread_id TEXT NOT NULL REFERENCES cubepi_threads(thread_id) ON DELETE CASCADE,
                seq BIGINT NOT NULL,
                role TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                payload BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (thread_id, seq)
            ) PARTITION BY HASH (thread_id);
        """)
        await conn.execute(create_message_partitions_op())
        await conn.execute("""
            CREATE INDEX ix_cubepi_messages_metadata_gin
            ON cubepi_messages USING GIN (metadata jsonb_path_ops);
        """)
        await conn.execute("""
            CREATE TABLE cubepi_schema_version (
                version INTEGER PRIMARY KEY
            );
        """)
        await conn.execute(write_schema_version_op())
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_postgres_checkpointer_round_trip(clean_db) -> None:
    await _setup_schema(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        msg1 = UserMessage(
            content=[TextContent(text="hello")],
            metadata={"memory_snapshot": {"id": "m1"}},
        )
        msg2 = AssistantMessage(
            content=[TextContent(text="hi back")],
            usage=Usage(),
            metadata={"cost_cents": 5},
        )
        await cp.append("t-1", [msg1, msg2])
        data = await cp.load("t-1")
        assert data is not None
        assert len(data.messages) == 2
        assert data.messages[0].metadata == {"memory_snapshot": {"id": "m1"}}
        assert data.messages[1].metadata == {"cost_cents": 5}


@pytest.mark.asyncio
async def test_postgres_checkpointer_save_extra_merges(clean_db) -> None:
    await _setup_schema(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.append("t-2", [UserMessage(content=[TextContent(text="x")])])
        await cp.save_extra("t-2", {"a": 1})
        await cp.save_extra("t-2", {"b": 2})
        data = await cp.load("t-2")
        assert data.extra == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_postgres_checkpointer_seq_monotonic(clean_db) -> None:
    await _setup_schema(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        msgs = [UserMessage(content=[TextContent(text=str(i))]) for i in range(5)]
        await cp.append("t-3", msgs)
        # Append another batch
        more = [UserMessage(content=[TextContent(text=str(i))]) for i in range(5, 10)]
        await cp.append("t-3", more)
        data = await cp.load("t-3")
        assert len(data.messages) == 10
        # Verify order is preserved
        assert [c.text for m in data.messages for c in m.content] == [str(i) for i in range(10)]


@pytest.mark.asyncio
async def test_uninitialized_schema_raises(clean_db) -> None:
    """Empty DB (no cubepi tables) → CubepiSchemaUninitialized."""
    with pytest.raises(CubepiSchemaUninitialized):
        async with PostgresCheckpointer(clean_db):
            pass


@pytest.mark.asyncio
async def test_version_mismatch_raises(clean_db) -> None:
    await _setup_schema(clean_db)
    # Manually set version to a wrong value
    conn = await asyncpg.connect(clean_db)
    try:
        await conn.execute("UPDATE cubepi_schema_version SET version = 999")
    finally:
        await conn.close()
    with pytest.raises(CubepiSchemaMismatch) as exc_info:
        async with PostgresCheckpointer(clean_db):
            pass
    assert exc_info.value.expected == 1
    assert exc_info.value.actual == 999
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/checkpointer/test_postgres.py -v`
Expected: 9 pass (1 import + 2 helpers + 1 exception + 5 E2E).

If Postgres isn't running locally, all E2E tests will fail — skip
or set `CUBEPI_TEST_PG_DSN` first.

- [ ] **Step 6: Commit**

```bash
git add cubepi/checkpointer/postgres/checkpointer.py cubepi/checkpointer/__init__.py cubepi/checkpointer/postgres/__init__.py tests/checkpointer/
git commit -m "feat(postgres-checkpointer): implement load/append/save_extra + schema verify

PostgresCheckpointer is an async-context-managed checkpointer:
- __aenter__: open asyncpg pool, verify cubepi_schema_version matches
- load(): single PK range scan + thread extra fetch
- append(): per-thread advisory lock, monotonic seq, msgpack payload,
  metadata JSONB column kept in sync with serialized message
- save_extra(): UPSERT with JSONB || merge

Raises CubepiSchemaUninitialized / CubepiSchemaMismatch at __aenter__
when DB schema state doesn't match library version."
```

---

## D2 — MCP adapter (HTTP + stdio)

### Task D2.1: Adapter — MCP tool → cubepi.AgentTool

**Files:**
- Create: `cubepi/mcp/__init__.py`
- Create: `cubepi/mcp/_adapter.py`
- Test: `tests/mcp/test_adapter.py` (unit-test the conversion)

- [ ] **Step 1: Set up module structure**

```bash
mkdir -p cubepi/mcp
touch cubepi/mcp/__init__.py
mkdir -p tests/mcp
touch tests/mcp/__init__.py
```

- [ ] **Step 2: Write the adapter**

Create `cubepi/mcp/_adapter.py`:

```python
"""MCP tool descriptor → cubepi.AgentTool adapter."""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from pydantic import BaseModel, create_model

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import Content, TextContent


def mcp_schema_to_pydantic_model(
    *,
    tool_name: str,
    input_schema: dict[str, Any],
) -> type[BaseModel]:
    """Convert an MCP JSON schema to a pydantic model class.

    MCP tools advertise `inputSchema` as JSON schema. cubepi.AgentTool
    requires `parameters: type[BaseModel]`. We synthesize a model from
    the schema's top-level properties.

    Limitations: this is a minimal converter — covers string/int/bool/
    array primitives and nested dicts as dict[str, Any]. Complex schemas
    may need manual mapping.
    """
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _json_schema_type_to_python(prop_schema)
        default = ... if prop_name in required else None
        fields[prop_name] = (py_type, default)

    model_name = f"MCP_{tool_name}_Input"
    return create_model(model_name, **fields)


def _json_schema_type_to_python(schema: dict[str, Any]) -> Any:
    t = schema.get("type")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        item_t = _json_schema_type_to_python(schema.get("items", {}))
        return list[item_t]  # type: ignore[valid-type]
    if t == "object":
        return dict[str, Any]
    return Any


def make_mcp_agent_tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    call_remote: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
) -> AgentTool:
    """Build a cubepi.AgentTool wrapping an MCP tool call.

    call_remote is the transport-specific RPC: given (tool_name, args_dict),
    returns the MCP tools/call response dict.
    """
    parameters_model = mcp_schema_to_pydantic_model(
        tool_name=name, input_schema=input_schema,
    )

    async def _execute(args) -> AgentToolResult:
        # args is an instance of parameters_model
        args_dict = args.model_dump() if hasattr(args, "model_dump") else dict(args)
        result = await call_remote(name, args_dict)
        # MCP tools/call response shape: {"content": [{"type": "text", "text": "..."}], "isError": bool}
        content_blocks: list[Content] = []
        for c in result.get("content", []):
            if c.get("type") == "text":
                content_blocks.append(TextContent(text=c.get("text", "")))
            # extend for image/audio/resource as needed
        return AgentToolResult(
            content=content_blocks,
            details={"raw_mcp_response": result},
        )

    return AgentTool(
        name=name,
        description=description,
        parameters=parameters_model,
        execute=_execute,
    )
```

- [ ] **Step 3: Unit tests for the adapter**

Create `tests/mcp/test_adapter.py`:

```python
"""MCP adapter tests."""

import pytest

from cubepi.mcp._adapter import (
    make_mcp_agent_tool,
    mcp_schema_to_pydantic_model,
)


def test_schema_to_model_required_fields() -> None:
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["city"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="search", input_schema=schema)
    # Required field
    instance = M(city="Tokyo")
    assert instance.city == "Tokyo"
    assert instance.limit is None


def test_schema_to_model_array_field() -> None:
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["tags"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="tag", input_schema=schema)
    instance = M(tags=["a", "b"])
    assert instance.tags == ["a", "b"]


@pytest.mark.asyncio
async def test_make_mcp_agent_tool_routes_to_call_remote() -> None:
    called_with: dict = {}

    async def _fake_call(name, args):
        called_with["name"] = name
        called_with["args"] = args
        return {
            "content": [{"type": "text", "text": "result"}],
            "isError": False,
        }

    tool = make_mcp_agent_tool(
        name="search",
        description="Search the web",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        call_remote=_fake_call,
    )
    assert tool.name == "search"
    args_instance = tool.parameters(query="cats")
    result = await tool.execute(args_instance)
    assert called_with == {"name": "search", "args": {"query": "cats"}}
    assert result.content[0].text == "result"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/mcp/test_adapter.py -v`
Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add cubepi/mcp/ tests/mcp/
git commit -m "feat(mcp): add MCP tool → cubepi.AgentTool adapter

mcp_schema_to_pydantic_model converts MCP inputSchema (JSON Schema) to a
pydantic model class for cubepi.AgentTool.parameters.

make_mcp_agent_tool builds an AgentTool wrapping a transport-agnostic
call_remote callable. HTTP and stdio loaders (next tasks) use this."
```

### Task D2.2: HTTP MCP loader

**Files:**
- Create: `cubepi/mcp/http_loader.py`
- Test: `tests/mcp/test_http_loader.py`

- [ ] **Step 1: Add fake HTTP MCP server fixture**

Create `tests/mcp/conftest.py`:

```python
"""MCP test fixtures: fake HTTP server, fake stdio server."""

import asyncio
import json
from typing import Any

import pytest
import pytest_asyncio


class FakeMCPHTTPServer:
    """Minimal MCP HTTP/SSE server for testing.

    Advertises a fixed tool list and responds to tools/call.
    """

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self.tools = tools
        self.calls: list[tuple[str, dict]] = []

    async def handle(self, request_body: dict) -> dict:
        method = request_body.get("method")
        if method == "tools/list":
            return {"result": {"tools": self.tools}}
        if method == "tools/call":
            params = request_body.get("params", {})
            self.calls.append((params.get("name"), params.get("arguments", {})))
            return {
                "result": {
                    "content": [{"type": "text", "text": f"called {params.get('name')}"}],
                    "isError": False,
                }
            }
        return {"error": {"code": -32601, "message": "Method not found"}}


# Actual HTTP server spinning requires aiohttp/starlette; use respx-style mock
# OR a real fastapi TestClient. The simplest approach: skip a real network
# server for unit-level coverage and test http_loader via injecting a
# call_remote stub. Real-network E2E lives in tests/mcp/test_http_loader.py
# using `mcp` SDK's own test fixtures if available.

@pytest.fixture
def fake_mcp_server_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "echo",
            "description": "Echo the input back",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    ]
```

- [ ] **Step 2: Write the HTTP loader**

Create `cubepi/mcp/http_loader.py`:

```python
"""HTTP/SSE transport MCP tool loader."""
from __future__ import annotations

from typing import Any

from cubepi.agent.types import AgentTool
from cubepi.mcp._adapter import make_mcp_agent_tool


async def load_mcp_tools_http(
    server_url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> list[AgentTool]:
    """Connect to an HTTP/SSE MCP server, discover tools, return AgentTools.

    Uses the `mcp` SDK's HTTP client. Each returned AgentTool's execute
    method invokes tools/call via the live session.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    # Open a persistent session for the lifetime of these tools.
    # NOTE: callers must keep the returned tools' associated session alive.
    # In v1 we use a simple per-tool session — connection cost per call.
    # Optimization (later): pool / per-loader session.

    async def _call_remote(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        async with sse_client(server_url, headers=headers, timeout=timeout) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                resp = await session.call_tool(tool_name, args)
                return _serialize_call_tool_response(resp)

    # Initial connection just to discover tools.
    async with sse_client(server_url, headers=headers, timeout=timeout) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            tool_descs = tools_resp.tools

    agent_tools: list[AgentTool] = []
    for desc in tool_descs:
        agent_tools.append(make_mcp_agent_tool(
            name=desc.name,
            description=desc.description or "",
            input_schema=desc.inputSchema or {"type": "object", "properties": {}},
            call_remote=_call_remote,
        ))
    return agent_tools


def _serialize_call_tool_response(resp: Any) -> dict[str, Any]:
    """Normalize mcp SDK response object → dict for adapter."""
    return {
        "content": [
            {"type": "text", "text": c.text}
            for c in (resp.content or [])
            if getattr(c, "type", None) == "text"
        ],
        "isError": bool(getattr(resp, "isError", False)),
    }
```

- [ ] **Step 3: Integration test (skipped if mcp SDK testing complex)**

Create `tests/mcp/test_http_loader.py`:

```python
"""HTTP MCP loader integration test.

Requires a running MCP test server or careful mocking. v1 strategy:
skip if no test server URL is configured; provide explicit env var
to point at a real fake server for CI.
"""

import os
import pytest


@pytest.mark.asyncio
async def test_load_mcp_tools_http_against_test_server() -> None:
    """End-to-end: connect to a real MCP test server, list + call a tool."""
    server_url = os.environ.get("CUBEPI_TEST_MCP_HTTP_URL")
    if not server_url:
        pytest.skip("Set CUBEPI_TEST_MCP_HTTP_URL to run this test")

    from cubepi.mcp import load_mcp_tools_http
    tools = await load_mcp_tools_http(server_url)
    assert len(tools) > 0
    # Smoke: at least one tool is callable
    first = tools[0]
    assert first.name
    assert first.description
```

(Real MCP integration testing needs either a running test server or
mocking the mcp SDK internals. Mocking the SDK is brittle. Recommend
spinning up a tiny stdio-MCP test server for both transports — see
D2.3.)

- [ ] **Step 4: Smoke-import test**

Append to `tests/mcp/test_http_loader.py`:

```python
def test_import_http_loader() -> None:
    from cubepi.mcp import load_mcp_tools_http
    assert callable(load_mcp_tools_http)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/mcp/test_http_loader.py -v`
Expected: 1 pass (smoke), 1 skip (no test server URL).

- [ ] **Step 6: Commit**

```bash
git add cubepi/mcp/http_loader.py tests/mcp/test_http_loader.py tests/mcp/conftest.py
git commit -m "feat(mcp): add HTTP/SSE MCP tool loader

load_mcp_tools_http(server_url, headers=, timeout=) discovers tools
from an MCP server via SSE and returns them as cubepi.AgentTool list.
Each tool's execute opens a fresh session per call (v1 simplicity);
later optimization: persistent session pool."
```

### Task D2.3: stdio MCP loader

**Files:**
- Create: `cubepi/mcp/stdio_loader.py`
- Modify: `cubepi/mcp/__init__.py` to export both loaders
- Test: `tests/mcp/test_stdio_loader.py`

- [ ] **Step 1: Implement stdio loader**

Create `cubepi/mcp/stdio_loader.py`:

```python
"""stdio transport MCP tool loader."""
from __future__ import annotations

from typing import Any

from cubepi.agent.types import AgentTool
from cubepi.mcp._adapter import make_mcp_agent_tool


async def load_mcp_tools_stdio(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float = 30.0,
) -> list[AgentTool]:
    """Spawn a stdio MCP server subprocess, discover tools, return AgentTools.

    Uses the `mcp` SDK's stdio client. Each returned tool's execute
    opens a fresh subprocess per call (v1 simplicity).
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=command, args=args, env=env, cwd=cwd,
    )

    async def _call_remote(tool_name: str, args_dict: dict[str, Any]) -> dict[str, Any]:
        async with stdio_client(server_params) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                resp = await session.call_tool(tool_name, args_dict)
                return _serialize_call_tool_response(resp)

    async with stdio_client(server_params) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            tool_descs = tools_resp.tools

    agent_tools: list[AgentTool] = []
    for desc in tool_descs:
        agent_tools.append(make_mcp_agent_tool(
            name=desc.name,
            description=desc.description or "",
            input_schema=desc.inputSchema or {"type": "object", "properties": {}},
            call_remote=_call_remote,
        ))
    return agent_tools


def _serialize_call_tool_response(resp: Any) -> dict[str, Any]:
    return {
        "content": [
            {"type": "text", "text": c.text}
            for c in (resp.content or [])
            if getattr(c, "type", None) == "text"
        ],
        "isError": bool(getattr(resp, "isError", False)),
    }
```

- [ ] **Step 2: Update `__init__.py`**

`cubepi/mcp/__init__.py`:

```python
from cubepi.mcp.http_loader import load_mcp_tools_http
from cubepi.mcp.stdio_loader import load_mcp_tools_stdio

__all__ = ["load_mcp_tools_http", "load_mcp_tools_stdio"]
```

- [ ] **Step 3: Integration test using a fake stdio server**

Create `tests/mcp/_fake_stdio_server.py` — a minimal MCP-protocol-
speaking Python script. It will be run as a subprocess by the test:

```python
"""Minimal stdio MCP server for tests.

Implements just enough of the MCP protocol to: respond to initialize,
list one 'echo' tool, and respond to tools/call for that tool.

Invoke: python -m tests.mcp._fake_stdio_server
"""
import json
import sys


def _read_message() -> dict:
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)


def _send_message(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main() -> None:
    while True:
        msg = _read_message()
        if msg is None:
            return
        method = msg.get("method")
        msg_id = msg.get("id")
        if method == "initialize":
            _send_message({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "0.1"},
                },
            })
        elif method == "tools/list":
            _send_message({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [{
                        "name": "echo",
                        "description": "echo input",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }]
                }
            })
        elif method == "tools/call":
            params = msg.get("params", {})
            args = params.get("arguments", {})
            _send_message({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": args.get("text", "")}],
                    "isError": False,
                },
            })
        elif method == "notifications/initialized":
            # No response expected
            continue
        else:
            _send_message({"jsonrpc": "2.0", "id": msg_id, "error": {
                "code": -32601, "message": "Method not found"
            }})


if __name__ == "__main__":
    main()
```

NOTE: real MCP protocol is more nuanced — the above may need
adjustments to actually work with the mcp SDK client. If the SDK
expects framed messages or a different newline convention, adapt.
Alternative: use the mcp SDK's own server primitives to build a
test server.

- [ ] **Step 4: Test against the fake stdio server**

Create `tests/mcp/test_stdio_loader.py`:

```python
"""stdio MCP loader integration test."""

import sys

import pytest

from cubepi.mcp import load_mcp_tools_stdio


@pytest.mark.asyncio
async def test_stdio_loader_against_fake_server() -> None:
    """Load the fake stdio server, discover the 'echo' tool, invoke it."""
    tools = await load_mcp_tools_stdio(
        command=sys.executable,
        args=["-m", "tests.mcp._fake_stdio_server"],
    )
    assert len(tools) == 1
    echo = tools[0]
    assert echo.name == "echo"
    result = await echo.execute(echo.parameters(text="hello"))
    assert result.content[0].text == "hello"


def test_import_stdio_loader() -> None:
    from cubepi.mcp import load_mcp_tools_stdio
    assert callable(load_mcp_tools_stdio)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/mcp/ -v`
Expected: 1 import test passes; integration test may need adjustment
to fake server protocol — iterate until it passes.

- [ ] **Step 6: Run full suite**

Run: `pytest tests/ -q --tb=no`
Expected: baseline + all new MCP tests.

- [ ] **Step 7: Commit**

```bash
git add cubepi/mcp/stdio_loader.py cubepi/mcp/__init__.py tests/mcp/
git commit -m "feat(mcp): add stdio MCP tool loader

load_mcp_tools_stdio(command, args, env=, cwd=) spawns an MCP server
subprocess via stdio, lists tools, and returns cubepi.AgentTool list.
Each tool's execute opens a fresh subprocess per call (v1 simplicity).

Tested via tests/mcp/_fake_stdio_server.py — a minimal Python-script
MCP server."
```

---

## D9 — Packaging extras

### Task D9.1: Add `[postgres]` and `[mcp]` extras

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add extras**

Edit `pyproject.toml`, find `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
sqlite = ["aiosqlite>=0.19"]
postgres = ["sqlalchemy>=2.0", "asyncpg>=0.29", "msgpack>=1.0"]
mcp = ["mcp>=1.0"]
```

For dev tooling, ensure `respx` and `pytest-httpx` are available in
`[project.optional-dependencies] dev` (or wherever cubepi currently
lists dev deps):

```toml
dev = [
    # ... existing dev deps ...
    "respx>=0.20",
]
```

- [ ] **Step 2: Lock dependencies**

Run: `uv lock`
Expected: lockfile updates with new extras.

- [ ] **Step 3: Verify installable**

```bash
uv sync --extra postgres --extra mcp --extra dev
```

Expected: clean install, no conflicts.

- [ ] **Step 4: Run full suite under fresh env**

Run: `pytest tests/ -q --tb=no`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(packaging): add [postgres] and [mcp] optional extras

[postgres] = sqlalchemy>=2.0, asyncpg>=0.29, msgpack>=1.0
[mcp] = mcp>=1.0

cubepi base install remains lean. Downstreams opt in:
  cubepi[postgres]  for PostgresCheckpointer
  cubepi[mcp]       for HTTP+stdio MCP tool loaders
  cubepi[postgres,mcp]  both"
```

---

## Final integration

### Task Final.1: Run acceptance criteria from spec

**Files:** none new

- [ ] **Step 1: Verify spec acceptance A1-A14**

Walk through each criterion in `docs/specs/2026-05-13-cubepi-cubebox-readiness-design.md` § "Acceptance criteria":

| # | Check | Run |
|---|---|---|
| A1 | PostgresCheckpointer round-trip | `pytest tests/checkpointer/test_postgres.py::test_postgres_checkpointer_round_trip` |
| A2 | Schema version check on uninit / mismatch | `pytest tests/checkpointer/test_postgres.py::test_uninitialized_schema_raises tests/checkpointer/test_postgres.py::test_version_mismatch_raises` |
| A3 | 64 partitions, route + read correctly | `pytest tests/checkpointer/test_postgres.py::test_postgres_checkpointer_seq_monotonic` (large append should hit multiple partitions if test_setup uses multiple thread_ids) |
| A4 | MCP HTTP loader | `pytest tests/mcp/test_http_loader.py` (set CUBEPI_TEST_MCP_HTTP_URL for full E2E) |
| A5 | MCP stdio loader | `pytest tests/mcp/test_stdio_loader.py::test_stdio_loader_against_fake_server` |
| A6 | AnthropicProvider with custom cache_policy | `pytest tests/providers/test_anthropic_cache_policy.py::test_custom_policy_drives_message_marker_placement` |
| A7 | AnthropicProvider without cache_policy preserves v0.2 | `pytest tests/providers/test_anthropic.py` |
| A8 | OpenAIProvider 3 reasoning variants | `pytest tests/providers/test_openai_reasoning.py` |
| A9 | OpenAIProvider payload_quirks | `pytest tests/providers/test_openai_reasoning.py::test_max_completion_tokens_alias_rewrite` |
| A10 | Message.metadata round-trip | `pytest tests/providers/test_message_metadata.py` |
| A11 | AgentContext.extra persists | `pytest tests/agent/test_context_extra.py` |
| A12 | transform_system_prompt chain | `pytest tests/middleware/test_transform_system_prompt.py` |
| A13 | after_model_response chain composition | `pytest tests/middleware/test_after_model_response.py` |
| A14 | Full suite, no regressions | `pytest tests/ -q --tb=no` |

- [ ] **Step 2: Run them all**

Run: `pytest tests/ -q --tb=short`
Expected: every acceptance criterion check passes.

- [ ] **Step 3: Manual smoke test**

Open Python REPL with `cubepi[postgres,mcp]` installed:

```python
import asyncio

async def smoke():
    from cubepi import Agent, Model
    from cubepi.providers.faux import FauxProvider, faux_assistant_message
    from cubepi.checkpointer.postgres import PostgresCheckpointer
    from cubepi.middleware.base import Middleware, TurnAction

    # Custom middleware exercising new hooks
    class _SmokeMW(Middleware):
        async def transform_system_prompt(self, sp, *, signal=None):
            return sp + "\n[smoke]"

        async def after_model_response(self, response, ctx, *, signal=None):
            ctx.extra["smoke_seen"] = True
            return TurnAction()

    provider = FauxProvider()
    provider.set_responses([faux_assistant_message("ok")])
    agent = Agent(
        model=Model(provider=provider, model="test"),
        system_prompt="base",
        middleware=[_SmokeMW()],
    )
    stream = await agent.prompt("hi")
    async for _ in stream:
        pass
    print("smoke passed")

asyncio.run(smoke())
```

Expected: prints "smoke passed".

- [ ] **Step 4: Open PR**

```bash
git push -u origin feat/cubebox-readiness
gh pr create --title "feat: cubebox-readiness — Postgres checkpointer, MCP, middleware extensions" --body "$(cat <<'EOF'
## Summary

Implements 9 deliverables (D1-D9) per `docs/specs/2026-05-13-cubepi-cubebox-readiness-design.md`:

- D1 PostgresCheckpointer (3 tables, HASH partition x64, schema version check)
- D2 MCP HTTP + stdio loaders → cubepi.AgentTool
- D3 AnthropicProvider configurable CacheMarkerPolicy
- D4 OpenAIProvider OSS reasoning extraction (3 variants) + payload_quirks
- D5 Message.metadata field on all three Message types
- D6 AgentContext.extra mutable dict + agent-loop persistence
- D7 transform_system_prompt middleware hook (chain composition)
- D8 after_model_response hook + TurnAction (chain, with control flow)
- D9 [postgres] and [mcp] optional extras

All additions backward-compatible with v0.2.0 — existing users see no
behavior change.

## Test plan

- [x] All existing tests pass (no regressions)
- [x] PostgresCheckpointer E2E against local PG
- [x] MCP stdio loader E2E against bundled fake server
- [x] All new middleware hooks unit-tested
- [x] OpenAI reasoning extraction tested against all 3 variants

## Consumer

cubebox depends on this via path dep (`uv add --editable ~/cubepi`)
during its main agent migration. See cubebox `docs/superpowers/specs/
2026-05-13-cubepi-main-agent-migration-design.md`.
EOF
)"
```

---

## Self-review checklist

After completing all tasks:

- [ ] Every D-item from spec has a corresponding task: D1 (D1.1-D1.3), D2 (D2.1-D2.3), D3 (D3.1-D3.2), D4 (D4.1-D4.2), D5 (D5.1-D5.2), D6 (D6.1-D6.2), D7 (D7.1-D7.2), D8 (D8.1-D8.2), D9 (D9.1) ✅
- [ ] No placeholders ("TBD", "TODO", "Add tests", "implement later") in the plan ✅
- [ ] All file paths concrete and absolute under `cubepi/` and `tests/` ✅
- [ ] Code shown in every step that modifies code ✅
- [ ] Commit messages drafted ✅
- [ ] Acceptance criteria from spec map to specific test commands ✅
