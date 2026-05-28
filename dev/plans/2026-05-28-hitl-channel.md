# HITL Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Companion spec:** `dev/specs/2026-05-28-hitl-channel.md` — read it first. This plan turns that spec into bite-sized TDD tasks.

**Goal:** Ship a `cubepi.hitl` module that provides a `HitlChannel` Protocol with `InMemoryChannel` + `CheckpointedChannel` implementations, two built-in clients (`ask_user` tool + `ApprovalPolicyMiddleware`/`ConfirmToolCallMiddleware`), durable cross-process suspend/resume via the existing Checkpointer (Memory/SQLite/Postgres/MySQL), and event/trace integration — covering both sandbox tool confirmation and mid-run structured questions.

**Architecture:** Channel is an `await`able coroutine collaborator that lives orthogonal to the loop; pause = "persist `HitlRequest` + emit event + await future"; resume = "re-enter loop with answer pre-loaded in channel's resume-slot, last assistant message's unresolved tool_calls drive what executes next." HITL control flow uses `HitlControlException` (BaseException-derived) so existing `except Exception:` handlers in loop.py don't swallow it. See spec §2 + §2.1 for the design philosophy in full.

**Tech Stack:** Python 3.11+, pydantic v2, asyncio, pytest-asyncio (asyncio_mode=auto). All commands via `uv` per CLAUDE.md. Tests use FauxProvider (no real LLM calls).

## File Structure

**New files:**
- `cubepi/hitl/__init__.py` — public re-exports
- `cubepi/hitl/types.py` — Option, Question, ConfirmRequest, ApproveRequest, AskRequest, HitlRequest, ApproveAnswer
- `cubepi/hitl/exceptions.py` — HitlControlException tree + HitlError tree
- `cubepi/hitl/policy.py` — Approve / Deny / AskUser / ApprovalDecision
- `cubepi/hitl/channel.py` — HitlChannel Protocol + InMemoryChannel + CheckpointedChannel
- `cubepi/hitl/middleware.py` — ApprovalPolicyMiddleware + ConfirmToolCallMiddleware
- `cubepi/hitl/ask_user.py` — ask_user_tool factory
- `cubepi/hitl/_trace.py` — lazy OTel span helper
- `cubepi/hitl/testing.py` — ScriptedChannel, NoopChannel
- `tests/hitl/test_types.py`
- `tests/hitl/test_exceptions.py`
- `tests/hitl/test_in_memory_channel.py`
- `tests/hitl/test_compose_middleware.py`
- `tests/hitl/test_agent_channel_wiring.py`
- `tests/hitl/test_ask_user_tool.py`
- `tests/hitl/test_approval_policy_middleware.py`
- `tests/hitl/test_confirm_tool_call_middleware.py`
- `tests/hitl/test_events.py`
- `tests/hitl/test_checkpointer_pending_request.py` (memory + sqlite)
- `tests/hitl/test_checkpointed_channel.py`
- `tests/hitl/test_agent_respond.py`
- `tests/hitl/test_agent_abort_pending.py`
- `tests/hitl/test_resume_cache_prefix.py`
- `tests/hitl/test_subagent_channel_inheritance.py`
- `tests/hitl/test_trace_spans.py`
- `tests/hitl/conftest.py` — shared FauxProvider helpers, channel fixtures
- `tests/checkpointer/test_postgres_pending_request.py` (E2E, marker-gated)
- `tests/checkpointer/test_mysql_pending_request.py` (E2E, marker-gated)
- `website/docs/guides/hitl.md`
- `website/docs/recipes/sandbox-confirm.md`
- `website/docs/recipes/ask-user-form.md`

**Modified files:**
- `cubepi/agent/types.py` — extend `BeforeToolCallResult` (edited_args, deny_reason, hitl_trace); add `HitlRequestEvent`, `HitlAnswerEvent`, `AgentSuspendedEvent`, `AgentAbortedEvent`
- `cubepi/agent/loop.py` — `_prepare_tool_call` selective exception + `hitl_trace` plumbing; new `run_agent_loop_resume()`; outer try/except for `HitlControlException`
- `cubepi/agent/tools.py` — `_PreparedToolCall`/`_FinalizedOutcome` carry `hitl_trace`; `_make_tool_result_message` calls `_merge_hitl_details`; selective exception around `tool.execute`
- `cubepi/agent/agent.py` — `__init__(channel=...)`, `agent.channel`, `_run_lock`, `in_flight_hitl_request`, `load_pending_hitl_request()`, `detach()`, `respond()`, `abort_pending()`, `_run_hitl_resume()`
- `cubepi/middleware/base.py` — `compose_middleware`'s `composed_before` redesigned to carry `edited_args` + merge `hitl_trace`
- `cubepi/checkpointer/base.py` — `Checkpointer` Protocol gets default-stub `save_pending_request`/`load_pending_request`
- `cubepi/checkpointer/memory.py` — implement pending dict
- `cubepi/checkpointer/sqlite.py` — `thread_pending_request` table + methods
- `cubepi/checkpointer/postgres/models.py` — add `pending_request` JSONB column + bump `EXPECTED_SCHEMA_VERSION` to 2
- `cubepi/checkpointer/postgres/checkpointer.py` — methods + migration helper `migrate_v1_to_v2`
- `cubepi/checkpointer/mysql/models.py` — add `pending_request` JSON column + bump `EXPECTED_SCHEMA_VERSION` to 2
- `cubepi/checkpointer/mysql/checkpointer.py` — methods + migration helper `migrate_v1_to_v2`
- `cubepi/tracing/__init__.py` (lazy export) — no eager additions, `_trace.py` does the lazy import dance
- `cubepi/__init__.py` — re-export `cubepi.hitl` top-level
- `pyproject.toml` — no change (HITL is in core; no new deps)

---

### Task 1: Types and exceptions

**Files:**
- Create: `cubepi/hitl/__init__.py`
- Create: `cubepi/hitl/types.py`
- Create: `cubepi/hitl/exceptions.py`
- Create: `cubepi/hitl/policy.py`
- Create: `tests/hitl/__init__.py`
- Create: `tests/hitl/test_types.py`
- Create: `tests/hitl/test_exceptions.py`

- [ ] **Step 1.1: Failing test for type round-trip**

Create `tests/hitl/__init__.py` (empty).

Create `tests/hitl/test_types.py`:

```python
import pytest
from cubepi.hitl.types import (
    Option, Question, ConfirmRequest, ApproveRequest, AskRequest,
    HitlRequest, ApproveAnswer,
)


def test_option_default_allow_input_false():
    o = Option(label="A", value="a")
    assert o.allow_input is False
    assert o.description is None


def test_question_defaults():
    q = Question(key="color", prompt="Pick:")
    assert q.options is None
    assert q.multi_select is False
    assert q.required is True


def test_confirm_request_kind_literal():
    r = ConfirmRequest(prompt="ok?")
    assert r.kind == "confirm"


def test_approve_request_kind_literal():
    r = ApproveRequest(tool_name="bash", tool_call_id="tc-1", args={"cmd": "ls"})
    assert r.kind == "approve"


def test_ask_request_kind_literal():
    r = AskRequest(questions=[Question(key="x", prompt="?")])
    assert r.kind == "ask"


def test_hitl_request_envelope_round_trip():
    req = HitlRequest(
        question_id="tc-1",
        thread_id="t-7",
        payload=ApproveRequest(tool_name="bash", tool_call_id="tc-1", args={}),
        created_at=1.0,
        timeout_seconds=42.0,
    )
    raw = req.model_dump_json()
    back = HitlRequest.model_validate_json(raw)
    assert back == req
    assert back.payload.kind == "approve"


def test_hitl_request_discriminated_union_round_trip_for_each_kind():
    payloads = [
        ConfirmRequest(prompt="ok?"),
        ApproveRequest(tool_name="t", tool_call_id="c", args={"a": 1}),
        AskRequest(questions=[Question(key="k", prompt="p")]),
    ]
    for p in payloads:
        req = HitlRequest(question_id="q", thread_id=None, payload=p, created_at=0.0)
        back = HitlRequest.model_validate_json(req.model_dump_json())
        assert type(back.payload) is type(p)


def test_approve_answer_decisions():
    assert ApproveAnswer(decision="approve").decision == "approve"
    assert ApproveAnswer(decision="deny", reason="no").reason == "no"
    assert ApproveAnswer(decision="edit", edited_args={"x": 1}).edited_args == {"x": 1}
```

- [ ] **Step 1.2: Run test — expected FAIL (module missing)**

Run: `uv run pytest tests/hitl/test_types.py -v`
Expected: ImportError / ModuleNotFoundError on `cubepi.hitl.types`.

- [ ] **Step 1.3: Implement `cubepi/hitl/types.py`**

```python
from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class Option(BaseModel):
    label: str
    value: str
    description: str | None = None
    allow_input: bool = False


class Question(BaseModel):
    key: str
    prompt: str
    options: list[Option] | None = None
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


HitlPayload = Union[ConfirmRequest, ApproveRequest, AskRequest]


class HitlRequest(BaseModel):
    question_id: str
    thread_id: str | None
    payload: HitlPayload = Field(discriminator="kind")
    created_at: float
    timeout_seconds: float | None = None


class ApproveAnswer(BaseModel):
    decision: Literal["approve", "deny", "edit"]
    edited_args: dict[str, Any] | None = None
    reason: str | None = None
```

- [ ] **Step 1.4: Run test — expected PASS**

Run: `uv run pytest tests/hitl/test_types.py -v`
Expected: 8 passed.

- [ ] **Step 1.5: Failing test for exception hierarchy**

Create `tests/hitl/test_exceptions.py`:

```python
import pytest
from cubepi.hitl.exceptions import (
    HitlControlException, HitlCancelled, HitlTimedOut, HitlDetached, HitlAborted,
    HitlError, HitlConcurrencyError, HitlStaleAnswer, HitlNoPendingRequest,
    HitlMissingAnswer, HitlInconsistentState, HitlDurabilityNotGuaranteed,
)


def test_control_exceptions_are_baseexception_not_exception():
    # Critical: BaseException so existing `except Exception:` handlers
    # in loop.py do NOT swallow HITL control flow.
    for cls in (HitlControlException, HitlCancelled, HitlTimedOut, HitlDetached, HitlAborted):
        assert issubclass(cls, BaseException)
        assert not issubclass(cls, Exception)


def test_control_exception_subclassing():
    assert issubclass(HitlCancelled, HitlControlException)
    assert issubclass(HitlTimedOut, HitlControlException)
    assert issubclass(HitlDetached, HitlControlException)
    assert issubclass(HitlAborted, HitlControlException)


def test_regular_errors_are_exception():
    for cls in (HitlError, HitlConcurrencyError, HitlStaleAnswer,
                HitlNoPendingRequest, HitlMissingAnswer, HitlInconsistentState,
                HitlDurabilityNotGuaranteed):
        assert issubclass(cls, Exception)


def test_hitl_cancelled_carries_reason():
    exc = HitlCancelled("user clicked cancel")
    assert exc.reason == "user clicked cancel"
    assert "user clicked cancel" in str(exc)


def test_hitl_timed_out_carries_seconds():
    exc = HitlTimedOut(30.0)
    assert exc.seconds == 30.0
    assert "30" in str(exc)


def test_except_exception_does_not_catch_control():
    try:
        try:
            raise HitlCancelled("x")
        except Exception:
            pytest.fail("HitlCancelled should not be caught by except Exception")
    except HitlControlException as exc:
        assert exc.reason == "x"
```

- [ ] **Step 1.6: Run — expected FAIL (missing module)**

Run: `uv run pytest tests/hitl/test_exceptions.py -v`

- [ ] **Step 1.7: Implement `cubepi/hitl/exceptions.py`**

```python
from __future__ import annotations


class HitlControlException(BaseException):
    """Base for HITL control-flow exceptions.

    Inherits BaseException so existing `except Exception:` handlers in
    cubepi.agent.tools._prepare_tool_call and _execute_prepared do NOT
    swallow these — mirrors asyncio.CancelledError.
    """


class HitlCancelled(HitlControlException):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class HitlTimedOut(HitlControlException):
    def __init__(self, seconds: float):
        super().__init__(f"HITL request timed out after {seconds} seconds")
        self.seconds = seconds


class HitlDetached(HitlControlException):
    pass


class HitlAborted(HitlControlException):
    pass


class HitlError(Exception):
    """Base for caller-fixable HITL errors (misuse, not control flow)."""


class HitlConcurrencyError(HitlError):
    pass


class HitlStaleAnswer(HitlError):
    pass


class HitlNoPendingRequest(HitlError):
    pass


class HitlMissingAnswer(HitlError):
    pass


class HitlInconsistentState(HitlError):
    pass


class HitlDurabilityNotGuaranteed(HitlError):
    pass
```

- [ ] **Step 1.8: Run exception tests — expected PASS**

Run: `uv run pytest tests/hitl/test_exceptions.py -v`
Expected: 6 passed.

- [ ] **Step 1.9: Implement policy types**

Create `cubepi/hitl/policy.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Approve:
    pass


@dataclass(frozen=True)
class Deny:
    reason: str


@dataclass(frozen=True)
class AskUser:
    prompt: str | None = None
    timeout_seconds: float | None = None
    details: dict | None = None


ApprovalDecision = Union[Approve, Deny, AskUser]
```

Add a quick test in `tests/hitl/test_types.py`:

```python
def test_approval_decision_dataclasses_frozen():
    from cubepi.hitl.policy import Approve, Deny, AskUser
    a = Approve()
    d = Deny(reason="forbidden")
    u = AskUser(timeout_seconds=10.0)
    assert d.reason == "forbidden"
    assert u.timeout_seconds == 10.0
    with pytest.raises(Exception):
        a.foo = "bar"  # frozen dataclass
```

- [ ] **Step 1.10: Implement `cubepi/hitl/__init__.py`**

```python
"""Human-in-the-Loop (HITL) primitives for cubepi agents.

See dev/specs/2026-05-28-hitl-channel.md for the full design.
"""

from cubepi.hitl.exceptions import (
    HitlAborted,
    HitlCancelled,
    HitlConcurrencyError,
    HitlControlException,
    HitlDetached,
    HitlDurabilityNotGuaranteed,
    HitlError,
    HitlInconsistentState,
    HitlMissingAnswer,
    HitlNoPendingRequest,
    HitlStaleAnswer,
    HitlTimedOut,
)
from cubepi.hitl.policy import (
    Approve,
    ApprovalDecision,
    AskUser,
    Deny,
)
from cubepi.hitl.types import (
    ApproveAnswer,
    ApproveRequest,
    AskRequest,
    ConfirmRequest,
    HitlPayload,
    HitlRequest,
    Option,
    Question,
)

__all__ = [
    # types
    "ApproveAnswer", "ApproveRequest", "AskRequest", "ConfirmRequest",
    "HitlPayload", "HitlRequest", "Option", "Question",
    # policy
    "Approve", "ApprovalDecision", "AskUser", "Deny",
    # exceptions
    "HitlAborted", "HitlCancelled", "HitlConcurrencyError",
    "HitlControlException", "HitlDetached", "HitlDurabilityNotGuaranteed",
    "HitlError", "HitlInconsistentState", "HitlMissingAnswer",
    "HitlNoPendingRequest", "HitlStaleAnswer", "HitlTimedOut",
]
```

- [ ] **Step 1.11: Run full task 1 test suite**

Run: `uv run pytest tests/hitl/ -v`
Expected: all tests passing.

- [ ] **Step 1.12: Lint**

Run: `uv run ruff check cubepi/hitl/ tests/hitl/ && uv run ruff format cubepi/hitl/ tests/hitl/`
Expected: zero issues.

- [ ] **Step 1.13: Add `tests/hitl/conftest.py` with shared polling helper**

Codex pass 2 flagged the `while ch.pending is None: await asyncio.sleep(0)` pattern as fragile — a failing host task can hang the main await forever. Centralize the pattern in a helper that wraps `asyncio.wait_for`:

```python
# tests/hitl/conftest.py
import asyncio
import pytest


async def await_pending(channel, *, timeout: float = 2.0) -> None:
    """Wait until channel.pending becomes non-None, or fail the test on timeout.

    Use this in tests that race a host coroutine against an awaiting agent.
    Replaces the `while ch.pending is None: await asyncio.sleep(0)` pattern,
    which silently hangs if the host task crashes.
    """
    async def _wait():
        while channel.pending is None:
            await asyncio.sleep(0)
    try:
        await asyncio.wait_for(_wait(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise AssertionError(
            f"channel.pending did not become set within {timeout}s"
        ) from exc
```

The tests in Tasks 2 / 6 / 9 / 11 that currently use the raw polling pattern can be updated incrementally to call `await await_pending(ch)` instead — not required for correctness, but it makes failures point at the right line.

- [ ] **Step 1.14: Commit**

```bash
git add cubepi/hitl/__init__.py cubepi/hitl/types.py cubepi/hitl/exceptions.py cubepi/hitl/policy.py tests/hitl/__init__.py tests/hitl/conftest.py tests/hitl/test_types.py tests/hitl/test_exceptions.py
git commit -m "feat(hitl): types, exception hierarchy, policy enum, test helpers"
```

---

### Task 2: HitlChannel Protocol + InMemoryChannel

**Files:**
- Create: `cubepi/hitl/channel.py`
- Create: `tests/hitl/test_in_memory_channel.py`

- [ ] **Step 2.1: Failing tests for InMemoryChannel basics**

Create `tests/hitl/test_in_memory_channel.py`:

```python
import asyncio
import pytest

from cubepi.hitl import (
    ApproveAnswer, HitlCancelled, HitlConcurrencyError,
    HitlRequest, HitlStaleAnswer, HitlTimedOut, Option, Question,
)
from cubepi.hitl.channel import InMemoryChannel


async def test_ask_resolves_via_answer():
    ch = InMemoryChannel()

    async def host():
        # Wait until something is pending, then answer.
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, {"color": "red"})

    asyncio.create_task(host())
    answer = await ch.ask([Question(key="color", prompt="Pick:")])
    assert answer == {"color": "red"}
    assert ch.pending is None


async def test_confirm_resolves_to_bool():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    assert (await ch.confirm("proceed?")) is True


async def test_approve_uses_tool_call_id_as_question_id():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        # question_id MUST equal tool_call_id for approve
        assert ch.pending.question_id == "tc-42"
        await ch.answer("tc-42", ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    ans = await ch.approve(tool_name="bash", tool_call_id="tc-42", args={"cmd": "ls"})
    assert ans.decision == "approve"


async def test_pending_request_envelope_carries_timeout():
    ch = InMemoryChannel()
    seen: list[HitlRequest] = []

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        seen.append(ch.pending)
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    await ch.confirm("ok?", timeout=42.0)
    assert seen[0].timeout_seconds == 42.0


async def test_default_timeout_applied_when_per_call_none():
    ch = InMemoryChannel(default_timeout=3.0)
    seen: list[HitlRequest] = []

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        seen.append(ch.pending)
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    await ch.confirm("ok?")  # per-call timeout omitted
    assert seen[0].timeout_seconds == 3.0


async def test_per_call_timeout_overrides_default():
    ch = InMemoryChannel(default_timeout=3.0)
    seen: list[HitlRequest] = []

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        seen.append(ch.pending)
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    await ch.confirm("ok?", timeout=99.0)
    assert seen[0].timeout_seconds == 99.0


async def test_timeout_raises_hitl_timed_out():
    ch = InMemoryChannel()
    with pytest.raises(HitlTimedOut) as exc_info:
        await ch.confirm("ok?", timeout=0.05)
    assert exc_info.value.seconds == 0.05
    assert ch.pending is None


async def test_cancel_raises_hitl_cancelled():
    ch = InMemoryChannel()

    async def canceller():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.cancel(ch.pending.question_id, reason="aborted")

    asyncio.create_task(canceller())
    with pytest.raises(HitlCancelled) as exc_info:
        await ch.confirm("ok?")
    assert exc_info.value.reason == "aborted"
    assert ch.pending is None


async def test_answer_with_stale_qid_raises():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        with pytest.raises(HitlStaleAnswer):
            await ch.answer("not-the-qid", True)
        # Now answer correctly so the test can finish.
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    await ch.confirm("ok?")


async def test_concurrent_request_raises_hitl_concurrency_error():
    ch = InMemoryChannel()

    async def occupy():
        try:
            await ch.confirm("first")
        except HitlCancelled:
            pass

    task = asyncio.create_task(occupy())
    # let occupy() reach the await
    for _ in range(10):
        if ch.pending is not None:
            break
        await asyncio.sleep(0)
    with pytest.raises(HitlConcurrencyError):
        await ch.confirm("second")
    await ch.cancel(ch.pending.question_id, "cleanup")
    await task


async def test_signal_abort_raises_hitl_aborted():
    from cubepi.hitl.exceptions import HitlAborted
    ch = InMemoryChannel()
    signal = asyncio.Event()

    async def trigger():
        while ch.pending is None:
            await asyncio.sleep(0)
        signal.set()

    asyncio.create_task(trigger())
    with pytest.raises(HitlAborted):
        await ch.confirm("ok?", signal=signal)
    assert ch.pending is None


async def test_subscribe_yields_requests():
    ch = InMemoryChannel()
    seen: list[HitlRequest] = []

    async def subscriber():
        async for req in ch.subscribe():
            seen.append(req)
            await ch.answer(req.question_id, True)

    sub = asyncio.create_task(subscriber())
    await ch.confirm("a")
    await ch.confirm("b")
    sub.cancel()
    try:
        await sub
    except asyncio.CancelledError:
        pass
    assert len(seen) == 2


async def test_attach_resume_answer_short_circuits_next_call():
    """When an answer has been pre-loaded via attach_resume_answer,
    the next matching channel call returns immediately without ever
    setting _pending or awaiting a future."""
    ch = InMemoryChannel()
    ch.attach_resume_answer("tc-7", ApproveAnswer(decision="approve"))
    ans = await ch.approve(tool_name="bash", tool_call_id="tc-7", args={})
    assert ans.decision == "approve"
    assert ch.pending is None


async def test_attach_resume_answer_qid_mismatch_keeps_slot():
    """If the next channel call's question_id doesn't match the
    pre-loaded slot, the call proceeds normally (the pre-load is
    for a different question and should NOT be popped)."""
    ch = InMemoryChannel()
    ch.attach_resume_answer("tc-OLD", True)

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="deny", reason="nope"))

    asyncio.create_task(host())
    ans = await ch.approve(tool_name="bash", tool_call_id="tc-NEW", args={})
    assert ans.decision == "deny"
```

- [ ] **Step 2.2: Run — expected FAIL (no channel module)**

Run: `uv run pytest tests/hitl/test_in_memory_channel.py -v`

- [ ] **Step 2.3: Implement `cubepi/hitl/channel.py` — Protocol + InMemoryChannel**

```python
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator, Protocol

from cubepi.hitl.exceptions import (
    HitlAborted,
    HitlCancelled,
    HitlConcurrencyError,
    HitlStaleAnswer,
    HitlTimedOut,
)
from cubepi.hitl.types import (
    ApproveAnswer,
    ApproveRequest,
    AskRequest,
    ConfirmRequest,
    HitlRequest,
    Question,
)


class HitlChannel(Protocol):
    # ---- agent side ----
    async def confirm(self, prompt: str, *, details: dict | None = None,
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

    @property
    def pending(self) -> HitlRequest | None: ...
    def subscribe(self) -> AsyncIterator[HitlRequest]: ...
    async def answer(self, question_id: str, answer: Any) -> None: ...
    async def cancel(self, question_id: str, reason: str = "cancelled") -> None: ...
    def attach_resume_answer(self, question_id: str, answer: Any) -> None: ...


class _BaseChannel:
    """Shared state machine for InMemoryChannel and CheckpointedChannel.

    Maintains the single-pending invariant, the awaiting future, the
    resume-answer slot, subscriber queues, and the optional emit
    callback wired by the Agent at construction.
    """

    def __init__(self, *, default_timeout: float | None = None,
                 thread_id: str | None = None) -> None:
        self._default_timeout = default_timeout
        self._thread_id = thread_id
        self._pending: HitlRequest | None = None
        self._future: asyncio.Future[Any] | None = None
        self._resume_slot: tuple[str, Any] | None = None
        self._subscribers: list[asyncio.Queue[HitlRequest]] = []
        self._emit = None  # set by Agent._bind_channel

    @property
    def pending(self) -> HitlRequest | None:
        return self._pending

    def attach_resume_answer(self, question_id: str, answer: Any) -> None:
        self._resume_slot = (question_id, answer)

    def _bind_emit(self, emit) -> None:
        self._emit = emit

    def subscribe(self) -> AsyncIterator[HitlRequest]:
        queue: asyncio.Queue[HitlRequest] = asyncio.Queue()
        self._subscribers.append(queue)

        async def gen():
            try:
                while True:
                    yield await queue.get()
            finally:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)

        return gen()

    async def _await_answer(self, payload, timeout: float | None,
                            signal: asyncio.Event | None,
                            question_id: str) -> Any:
        # Resume short-circuit
        if self._resume_slot is not None and self._resume_slot[0] == question_id:
            _, ans = self._resume_slot
            self._resume_slot = None
            return ans

        if self._pending is not None:
            raise HitlConcurrencyError(
                f"channel busy: already pending {self._pending.question_id}"
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        req = HitlRequest(
            question_id=question_id,
            thread_id=self._thread_id,
            payload=payload,
            created_at=time.time(),
            timeout_seconds=effective_timeout,
        )
        self._pending = req
        self._future = asyncio.get_event_loop().create_future()

        await self._on_pending_set(req)

        exc_caught: BaseException | None = None
        signal_task: asyncio.Future[Any] | None = None
        try:
            if signal is None and effective_timeout is None:
                return await self._future
            tasks: list[asyncio.Future[Any]] = [self._future]
            if signal is not None:
                signal_task = asyncio.ensure_future(signal.wait())
                tasks.append(signal_task)
            done, pending_tasks = await asyncio.wait(
                tasks, timeout=effective_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Clean up pending tasks (the loser of the race + any signal task
            # we never want to leave hanging).
            for p in pending_tasks:
                if p is not self._future:
                    p.cancel()
            if not done:
                raise HitlTimedOut(effective_timeout)
            # Use task identity, not signal.is_set() — Agent.abort() leaves the
            # signal sticky-set, so signal.is_set() would be true even on the
            # happy path after a prior abort. The race winner is what matters.
            if signal_task is not None and signal_task in done and self._future not in done:
                raise HitlAborted("agent signal fired during HITL pending")
            return self._future.result()
        except BaseException as exc:
            exc_caught = exc
            raise
        finally:
            # Cancel any still-pending signal task to avoid leaks.
            if signal_task is not None and not signal_task.done():
                signal_task.cancel()
            self._pending = None
            self._future = None
            await self._on_pending_cleared(req, exc=exc_caught)

    async def _on_pending_set(self, req: HitlRequest) -> None:
        # Emit event and broadcast to subscribers.
        for q in list(self._subscribers):
            q.put_nowait(req)
        if self._emit is not None:
            from cubepi.agent.types import HitlRequestEvent  # avoid circular
            await self._emit_event(HitlRequestEvent(request=req))

    async def _on_pending_cleared(
        self, req: HitlRequest, *, exc: BaseException | None = None,
    ) -> None:
        # No-op in InMemory; CheckpointedChannel overrides to clear DB row
        # ONLY when the unwind cause is not HitlDetached (which signals a
        # cross-process suspend that must keep persisted pending).
        pass

    async def _emit_event(self, event) -> None:
        if self._emit is None:
            return
        res = self._emit(event)
        if asyncio.iscoroutine(res):
            await res

    async def answer(self, question_id: str, answer: Any) -> None:
        if self._pending is None or self._pending.question_id != question_id:
            raise HitlStaleAnswer(
                f"answer for {question_id}; pending is "
                f"{self._pending.question_id if self._pending else 'None'}"
            )
        if self._future is not None and not self._future.done():
            self._future.set_result(answer)
        if self._emit is not None:
            from cubepi.agent.types import HitlAnswerEvent
            await self._emit_event(
                HitlAnswerEvent(question_id=question_id, answer=answer)
            )

    async def cancel(self, question_id: str, reason: str = "cancelled") -> None:
        if self._pending is None or self._pending.question_id != question_id:
            raise HitlStaleAnswer(
                f"cancel for {question_id}; pending is "
                f"{self._pending.question_id if self._pending else 'None'}"
            )
        if self._future is not None and not self._future.done():
            self._future.set_exception(HitlCancelled(reason))
        if self._emit is not None:
            from cubepi.agent.types import HitlAnswerEvent
            await self._emit_event(
                HitlAnswerEvent(question_id=question_id, answer=None, cancelled=True)
            )

    # ---- agent-side verbs ----

    async def confirm(self, prompt: str, *, details=None, tool_call_id=None,
                      timeout=None, signal=None) -> bool:
        qid = uuid.uuid4().hex
        return await self._await_answer(
            ConfirmRequest(prompt=prompt, details=details),
            timeout=timeout, signal=signal, question_id=qid,
        )

    async def approve(self, tool_name: str, tool_call_id: str, args: dict, *,
                      details=None, timeout=None, signal=None) -> ApproveAnswer:
        return await self._await_answer(
            ApproveRequest(tool_name=tool_name, tool_call_id=tool_call_id,
                           args=args, details=details),
            timeout=timeout, signal=signal, question_id=tool_call_id,
        )

    async def ask(self, questions: list[Question], *, timeout=None,
                  signal=None) -> dict[str, str | list[str]]:
        qid = uuid.uuid4().hex
        return await self._await_answer(
            AskRequest(questions=questions),
            timeout=timeout, signal=signal, question_id=qid,
        )


class InMemoryChannel(_BaseChannel):
    """In-process HITL channel; no persistence."""


# CheckpointedChannel is implemented in Task 7.
```

- [ ] **Step 2.4: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_in_memory_channel.py -v`
Expected: all tests pass.

NOTE: The tests import `cubepi.agent.types.HitlRequestEvent` / `HitlAnswerEvent` indirectly through `_on_pending_set`. For Task 2, channel is constructed standalone (no emit binding), so the import inside `_on_pending_set` is dead code — it only fires when `self._emit is not None`. Task 5 introduces the event types.

If a test fails because emit triggered the import: ensure tests do NOT call `ch._bind_emit(...)`. The `if self._emit is not None` guard makes the channel safe to use without events. Verify by running the tests.

- [ ] **Step 2.5: Lint + commit**

```bash
uv run ruff check cubepi/hitl/ tests/hitl/ && uv run ruff format cubepi/hitl/ tests/hitl/
git add cubepi/hitl/channel.py tests/hitl/test_in_memory_channel.py
git commit -m "feat(hitl): HitlChannel Protocol + InMemoryChannel"
```

---

### Task 3: BeforeToolCallResult extension + compose_middleware redesign + loop.py changes

**Files:**
- Modify: `cubepi/agent/types.py` (extend `BeforeToolCallResult`)
- Modify: `cubepi/middleware/base.py` (rewrite `composed_before`)
- Modify: `cubepi/agent/tools.py` (carry `hitl_trace`, merge into details, selective exception handlers)
- Modify: `cubepi/agent/loop.py` (selective exception at outer level)
- Create: `tests/hitl/test_compose_middleware.py`

- [ ] **Step 3.1: Failing test for BeforeToolCallResult extension**

Add to `tests/hitl/test_types.py`:

```python
def test_before_tool_call_result_new_fields():
    from cubepi.agent.types import BeforeToolCallResult
    r = BeforeToolCallResult(
        edited_args={"x": 1},
        deny_reason="forbidden",
        hitl_trace={"decision": "edit"},
    )
    assert r.edited_args == {"x": 1}
    assert r.deny_reason == "forbidden"
    assert r.hitl_trace == {"decision": "edit"}
    # backwards-compat: old call still works
    r2 = BeforeToolCallResult(block=True, reason="bad")
    assert r2.edited_args is None
    assert r2.deny_reason is None
    assert r2.hitl_trace is None
```

- [ ] **Step 3.2: Run — expected FAIL**

Run: `uv run pytest tests/hitl/test_types.py::test_before_tool_call_result_new_fields -v`

- [ ] **Step 3.3: Extend `BeforeToolCallResult` in `cubepi/agent/types.py`**

Edit the class (around line 59):

```python
class BeforeToolCallResult(BaseModel):
    block: bool = False
    reason: str | None = None
    edited_args: dict | None = None
    deny_reason: str | None = None
    hitl_trace: dict | None = None
```

- [ ] **Step 3.4: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_types.py::test_before_tool_call_result_new_fields -v`

- [ ] **Step 3.5: Failing tests for compose_middleware redesign**

Create `tests/hitl/test_compose_middleware.py`:

```python
import pytest
from cubepi.agent.types import (
    AgentContext, BeforeToolCallContext, BeforeToolCallResult,
)
from cubepi.middleware.base import Middleware, compose_middleware
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall


def _ctx(args: dict | None = None) -> BeforeToolCallContext:
    args = args or {}
    return BeforeToolCallContext(
        assistant_message=AssistantMessage(
            content=[TextContent(text="t"), ToolCall(id="tc-1", name="bash",
                                                     arguments={"cmd": "ls"})],
            stop_reason="end_turn",
        ),
        tool_call=ToolCall(id="tc-1", name="bash", arguments={"cmd": "ls"}),
        args=args,
        context=AgentContext(system_prompt="", messages=[], tools=[]),
    )


class _MWEdit(Middleware):
    def __init__(self, edited):
        self._edited = edited
    async def before_tool_call(self, ctx, *, signal=None):
        return BeforeToolCallResult(
            edited_args=self._edited,
            hitl_trace={"decision": "edit", "by": "first"},
        )


class _MWBlock(Middleware):
    async def before_tool_call(self, ctx, *, signal=None):
        return BeforeToolCallResult(
            block=True, deny_reason="bad",
            hitl_trace={"decision": "policy_deny", "by": "second"},
        )


class _MWInspect(Middleware):
    """Records what args it sees from upstream edits."""
    def __init__(self):
        self.seen_args: list = []
    async def before_tool_call(self, ctx, *, signal=None):
        self.seen_args.append(ctx.args)
        return None


async def test_compose_before_edit_chain_passes_edited_args_downstream():
    inspect = _MWInspect()
    hooks = compose_middleware([_MWEdit({"cmd": "ls -l"}), inspect])
    result = await hooks["before_tool_call"](_ctx({"cmd": "ls"}))
    assert result is not None
    assert result.edited_args == {"cmd": "ls -l"}
    # Inspect MW should have seen the edited args, not the original
    assert inspect.seen_args == [{"cmd": "ls -l"}]


async def test_compose_before_block_after_edit_discards_edit_but_keeps_hitl_trace():
    hooks = compose_middleware([_MWEdit({"cmd": "ls -l"}), _MWBlock()])
    result = await hooks["before_tool_call"](_ctx())
    assert result.block is True
    assert result.deny_reason == "bad"
    # hitl_trace should contain the most-recent (the block) primary keys,
    # with the edit step archived under _chain
    assert result.hitl_trace["decision"] == "policy_deny"
    assert "_chain" in result.hitl_trace


async def test_compose_before_hitl_trace_merge_keeps_history():
    class _MWTrace1(Middleware):
        async def before_tool_call(self, ctx, *, signal=None):
            return BeforeToolCallResult(hitl_trace={"by": "one", "extra": 1})
    class _MWTrace2(Middleware):
        async def before_tool_call(self, ctx, *, signal=None):
            return BeforeToolCallResult(hitl_trace={"by": "two", "more": 2})

    hooks = compose_middleware([_MWTrace1(), _MWTrace2()])
    result = await hooks["before_tool_call"](_ctx())
    assert result.hitl_trace["by"] == "two"   # last writer wins
    assert result.hitl_trace["more"] == 2
    assert "_chain" in result.hitl_trace
    assert any(c.get("by") == "one" for c in result.hitl_trace["_chain"])


async def test_compose_before_returns_none_when_no_middleware_speaks():
    class _MWSilent(Middleware):
        async def before_tool_call(self, ctx, *, signal=None):
            return None
    hooks = compose_middleware([_MWSilent(), _MWSilent()])
    result = await hooks["before_tool_call"](_ctx())
    assert result is None
```

- [ ] **Step 3.6: Run — expected FAIL (current composed_before discards non-block)**

Run: `uv run pytest tests/hitl/test_compose_middleware.py -v`

- [ ] **Step 3.7: Rewrite `composed_before` in `cubepi/middleware/base.py`**

First add the missing import at the top of `cubepi/middleware/base.py` (existing imports only include provider message types):

```python
from cubepi.agent.types import BeforeToolCallResult
```

(This was missing from the previous draft — leaving it out causes a runtime NameError when `composed_before` returns one.)

Find the existing `composed_before` (around lines 86-96) and replace with:

```python
    before_chain = [m for m in middlewares if _has_method(m, "before_tool_call")]
    if before_chain:

        def _rebuild_ctx_with_args(ctx, new_args):
            from dataclasses import replace
            return replace(ctx, args=new_args)

        async def composed_before(ctx, *, signal=None):
            accumulated_hitl: dict = {}
            edited_args = None
            deny_reason: str | None = None
            block_reason: str | None = None
            blocked = False

            cur_ctx = ctx
            for mw in before_chain:
                if edited_args is not None:
                    cur_ctx = _rebuild_ctx_with_args(ctx, edited_args)
                result = await mw.before_tool_call(cur_ctx, signal=signal)
                if result is None:
                    continue
                if result.hitl_trace:
                    if accumulated_hitl:
                        accumulated_hitl.setdefault("_chain", []).append(
                            {k: v for k, v in accumulated_hitl.items() if k != "_chain"}
                        )
                        # remove already-archived keys before updating with new
                        for k in list(accumulated_hitl.keys()):
                            if k != "_chain":
                                accumulated_hitl.pop(k)
                    accumulated_hitl.update(result.hitl_trace)
                if result.edited_args is not None:
                    edited_args = result.edited_args
                if result.block:
                    blocked = True
                    block_reason = result.reason or block_reason
                    deny_reason = result.deny_reason or deny_reason
                    break

            if not blocked and edited_args is None and not accumulated_hitl:
                return None
            return BeforeToolCallResult(
                block=blocked,
                reason=block_reason,
                deny_reason=deny_reason,
                edited_args=edited_args,
                hitl_trace=accumulated_hitl or None,
            )

        hooks["before_tool_call"] = composed_before
```

- [ ] **Step 3.8: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_compose_middleware.py -v`

If `test_compose_before_block_after_edit_discards_edit_but_keeps_hitl_trace` fails on the assertion about `_chain`: this is the trace-merge ordering rule. The test expects that the first MW's trace gets archived under `_chain` when the block MW overwrites. Adjust the merge logic if needed so the archived entry contains `{"decision":"edit","by":"first"}`.

- [ ] **Step 3.9: Failing test for tools.py selective HITL exception handling**

Create `tests/hitl/test_loop_hitl_passthrough.py`:

```python
import asyncio
import pytest
from cubepi.hitl.exceptions import HitlAborted, HitlCancelled
from cubepi.agent.tools import execute_tool_calls
from cubepi.agent.types import AgentContext, AgentTool, AgentToolResult
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall
from pydantic import BaseModel


class _NoParams(BaseModel):
    pass


def _make_tool(name: str, executor):
    return AgentTool(
        name=name, description="t",
        parameters=_NoParams, execute=executor,
        execution_mode="sequential",
    )


async def test_hitl_control_exception_in_tool_propagates():
    async def raises(call_id, args, *, signal=None, on_update=None):
        raise HitlAborted()
    tool = _make_tool("t1", raises)
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    msg = AssistantMessage(
        content=[TextContent(text=""), ToolCall(id="tc-1", name="t1", arguments={})],
        stop_reason="tool_use",
    )
    with pytest.raises(HitlAborted):
        await execute_tool_calls(ctx, msg, emit=lambda e: None)


async def test_regular_exception_in_tool_becomes_tool_error():
    async def raises(call_id, args, *, signal=None, on_update=None):
        raise ValueError("oops")
    tool = _make_tool("t1", raises)
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    msg = AssistantMessage(
        content=[TextContent(text=""), ToolCall(id="tc-1", name="t1", arguments={})],
        stop_reason="tool_use",
    )
    batch = await execute_tool_calls(ctx, msg, emit=lambda e: None)
    assert batch.messages[0].is_error is True
    assert "oops" in batch.messages[0].content[0].text


async def test_hitl_control_in_before_tool_call_propagates():
    async def runs(call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])
    tool = _make_tool("t1", runs)
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    msg = AssistantMessage(
        content=[TextContent(text=""), ToolCall(id="tc-1", name="t1", arguments={})],
        stop_reason="tool_use",
    )

    async def before(_ctx, *, signal=None):
        raise HitlCancelled("user cancelled")

    with pytest.raises(HitlCancelled):
        await execute_tool_calls(ctx, msg, before_tool_call=before, emit=lambda e: None)
```

- [ ] **Step 3.10: Run — expected FAIL (broad except Exception swallows control)**

Run: `uv run pytest tests/hitl/test_loop_hitl_passthrough.py -v`

- [ ] **Step 3.11: Patch `cubepi/agent/tools.py` — selective exception handlers**

In `_prepare_tool_call` (around lines 96-122), the existing `try/except (ValidationError, Exception):` for `tool.parameters.model_validate` and the `except Exception:` around `before_tool_call` must let `HitlControlException` propagate. Edit:

```python
async def _prepare_tool_call(
    context, assistant_message, tool_call, before_tool_call, signal,
):
    tool = None
    if context.tools:
        for t in context.tools:
            if t.name == tool_call.name:
                tool = t
                break

    if tool is None:
        return _ImmediateOutcome(
            result=_error_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        validated_args = tool.parameters.model_validate(tool_call.arguments)
    except ValidationError as exc:
        return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)
    except Exception as exc:   # never catches HitlControlException (BaseException)
        return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)

    if before_tool_call:
        try:
            before_ctx = BeforeToolCallContext(
                assistant_message=assistant_message,
                tool_call=tool_call,
                args=validated_args,
                context=context,
            )
            before_result = await before_tool_call(before_ctx, signal=signal)
        except HitlControlException:
            raise
        except Exception as exc:
            return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)

        if before_result and before_result.block:
            return _ImmediateOutcome(
                result=_error_result(
                    before_result.reason or "Tool execution was blocked"
                ),
                is_error=True,
                blocked_by_hook=True,
                block_reason=before_result.deny_reason or before_result.reason,
                hitl_trace=before_result.hitl_trace,
            )

        if before_result and before_result.edited_args is not None:
            try:
                validated_args = tool.parameters.model_validate(before_result.edited_args)
            except ValidationError as exc:
                return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)

        hitl_trace_carry = before_result.hitl_trace if before_result else None
    else:
        hitl_trace_carry = None

    return _PreparedToolCall(
        tool_call=tool_call, tool=tool, args=validated_args,
        hitl_trace=hitl_trace_carry,
    )
```

Add the import at the top of `cubepi/agent/tools.py`:

```python
from cubepi.hitl.exceptions import HitlControlException
```

Extend `_PreparedToolCall`, `_ImmediateOutcome`, `_FinalizedOutcome` dataclasses to include `hitl_trace: dict | None = None`.

Patch `_execute_prepared`:

```python
async def _execute_prepared(prepared, signal, emit_fn):
    try:
        result = await prepared.tool.execute(
            prepared.tool_call.id,
            prepared.args,
            signal=signal,
            on_update=lambda partial: emit_event(
                emit_fn,
                ToolExecutionUpdateEvent(
                    tool_call_id=prepared.tool_call.id,
                    tool_name=prepared.tool_call.name,
                    args=prepared.tool_call.arguments,
                    partial_result=partial,
                ),
            ),
        )
        return result, False
    except HitlControlException:
        raise
    except Exception as exc:
        return _error_result(str(exc)), True
```

Patch `_finalize` to carry `hitl_trace` through to `_FinalizedOutcome`:

```python
return _FinalizedOutcome(
    tool_call=prepared.tool_call, result=result, is_error=is_error,
    hitl_trace=prepared.hitl_trace,
)
```

Implement `_merge_hitl_details` helper near the top of `cubepi/agent/tools.py`:

```python
def _merge_hitl_details(base, hitl):
    if hitl is None:
        return base
    if base is None:
        return {"hitl": hitl}
    if isinstance(base, dict):
        merged = dict(base)
        merged["hitl"] = hitl
        return merged
    return {"_non_dict_details": base, "hitl": hitl}
```

Patch `_make_tool_result_message`:

```python
def _make_tool_result_message(finalized):
    details = _merge_hitl_details(finalized.result.details, finalized.hitl_trace)
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        content=finalized.result.content,
        details=details,
        is_error=finalized.is_error,
        timestamp=time.time(),
    )
```

Also patch the `_ImmediateOutcome → _FinalizedOutcome` constructions inside `_execute_sequential` and `_execute_parallel` to copy `hitl_trace`:

```python
finalized = _FinalizedOutcome(
    tool_call=tc, result=preparation.result, is_error=preparation.is_error,
    blocked_by_hook=preparation.blocked_by_hook,
    block_reason=preparation.block_reason,
    hitl_trace=preparation.hitl_trace,
)
```

- [ ] **Step 3.12: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_loop_hitl_passthrough.py -v`

- [ ] **Step 3.13: Patch `_run_loop` outer exception handler in `cubepi/agent/loop.py`**

Wrap the body of `_run_loop` to **silently** catch `HitlDetached` / `HitlAborted` — the Agent caller that triggered these has already emitted the corresponding `AgentSuspendedEvent` / `AgentAbortedEvent` (see Task 5.5 + Task 9.5). The loop does not emit any extra event:

```python
async def _run_loop(*, current_context, new_messages, provider, model, ...):
    try:
        # ... existing _run_loop body ...
    except (HitlDetached, HitlAborted):
        # Caller (Agent.detach / Agent.abort_pending) emitted the event
        # already. Loop exits silently — assistant message and pending
        # state remain intact for the next respond() call.
        return
```

Add the import at top of `cubepi/agent/loop.py`:

```python
from cubepi.hitl.exceptions import HitlAborted, HitlDetached
```

(Only these two — no event-class imports needed in the loop module.)

- [ ] **Step 3.14: Run full test suite — expected PASS, no regressions**

Run: `uv run pytest tests/ -x -q`
Expected: all existing tests continue to pass; new HITL tests pass.

If any existing test fails on `BeforeToolCallResult` — backwards compatibility broken. Verify the new fields are all optional (`None` defaults).

- [ ] **Step 3.15: Lint + commit**

```bash
uv run ruff check cubepi/ tests/ && uv run ruff format cubepi/ tests/
git add cubepi/agent/types.py cubepi/agent/tools.py cubepi/agent/loop.py cubepi/middleware/base.py tests/hitl/
git commit -m "feat(hitl): loop + middleware compose support for edits, hitl_trace, control exceptions"
```

---

### Task 4: Agent.__init__(channel=...) wiring + in_flight_hitl_request

**Files:**
- Modify: `cubepi/agent/agent.py`
- Create: `tests/hitl/test_agent_channel_wiring.py`

- [ ] **Step 4.1: Failing test for Agent channel binding**

Create `tests/hitl/test_agent_channel_wiring.py`:

```python
import pytest
from cubepi.agent.agent import Agent
from cubepi.hitl import HitlError
from cubepi.hitl.channel import InMemoryChannel
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.providers.base import Model


def _agent(channel=None):
    provider = FauxProvider()
    provider.set_responses([faux_assistant_message("")])
    return Agent(
        provider=provider,
        model=Model(id="faux", provider="faux"),
        channel=channel,
    )


def test_agent_accepts_channel_kwarg():
    ch = InMemoryChannel()
    agent = _agent(channel=ch)
    assert agent.channel is ch


def test_agent_channel_property_returns_none_when_unset():
    agent = _agent()
    assert agent.channel is None


def test_in_flight_hitl_request_property_none_initially():
    agent = _agent(channel=InMemoryChannel())
    assert agent.in_flight_hitl_request is None


def test_in_flight_hitl_request_raises_without_channel():
    agent = _agent()
    with pytest.raises(HitlError):
        _ = agent.in_flight_hitl_request


def test_channel_emit_is_bound_to_agent_process_event():
    ch = InMemoryChannel()
    agent = _agent(channel=ch)
    # Verify the emit callback was bound (no public API; verify via attribute)
    assert ch._emit is not None
```

- [ ] **Step 4.2: Run — expected FAIL**

Run: `uv run pytest tests/hitl/test_agent_channel_wiring.py -v`

- [ ] **Step 4.3: Patch `cubepi/agent/agent.py`**

Add `channel: HitlChannel | None = None` to `Agent.__init__`'s signature. Locate the existing `__init__` (look for `self.checkpointer = checkpointer` around line 170). Add immediately after:

```python
        self._channel = channel
        if channel is not None:
            channel._bind_emit(lambda e: self._process_event(e))
        self._run_lock = asyncio.Lock()
```

Wrap the **existing** `prompt()` and `resume()` method bodies with `async with self._run_lock:` (codex pass 2 BLOCKING: lock was introduced but never acquired by prompt/resume — `respond()` could race them). Concretely, in [agent.py](cubepi/agent/agent.py), the existing methods become:

```python
    async def prompt(self, message) -> None:
        async with self._run_lock:
            # ... existing body ...

    async def resume(self) -> None:
        async with self._run_lock:
            # ... existing body ...
```

The lock is reentrant-safe in cubepi's usage because `prompt()` / `resume()` never call each other; they only invoke `_run_with_lifecycle` which doesn't re-acquire. Existing `_state.is_streaming` flag stays as a debug signal but the lock is now the source of truth (matches the spec §5.2 "Concurrency guard" paragraph).

Add the type import at top of `agent.py`:

```python
from cubepi.hitl import HitlError
from cubepi.hitl.channel import HitlChannel
```

Add the read-only property and `in_flight_hitl_request`:

```python
    @property
    def channel(self) -> HitlChannel | None:
        return self._channel

    @property
    def in_flight_hitl_request(self):
        if self._channel is None:
            raise HitlError("agent has no channel bound; pass channel= to Agent()")
        return self._channel.pending
```

- [ ] **Step 4.4: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_agent_channel_wiring.py -v`

- [ ] **Step 4.5: Lint + commit**

```bash
uv run ruff check cubepi/ tests/ && uv run ruff format cubepi/ tests/
git add cubepi/agent/agent.py tests/hitl/test_agent_channel_wiring.py
git commit -m "feat(hitl): Agent(channel=...) wiring + in_flight_hitl_request property"
```

---

### Task 5: New events (HitlRequest, HitlAnswer, AgentSuspended, AgentAborted)

**Files:**
- Modify: `cubepi/agent/types.py`
- Modify: `cubepi/agent/loop.py` (use real events from Task 3 placeholder)
- Modify: `cubepi/hitl/channel.py` (no change — already imports event types lazily)
- Create: `tests/hitl/test_events.py`

- [ ] **Step 5.1: Failing test for new event types**

Create `tests/hitl/test_events.py`:

```python
import pytest
from cubepi.agent.types import (
    AgentAbortedEvent, AgentSuspendedEvent, HitlAnswerEvent, HitlRequestEvent,
)
from cubepi.hitl.types import ConfirmRequest, HitlRequest


def _req() -> HitlRequest:
    return HitlRequest(
        question_id="q-1", thread_id="t-1",
        payload=ConfirmRequest(prompt="ok?"), created_at=0.0,
    )


def test_hitl_request_event_construct():
    e = HitlRequestEvent(request=_req())
    assert e.type == "hitl_request"
    assert e.request.question_id == "q-1"


def test_hitl_answer_event_construct():
    e = HitlAnswerEvent(question_id="q-1", answer=True)
    assert e.type == "hitl_answer"
    assert e.cancelled is False
    assert e.timed_out is False


def test_agent_suspended_event_construct():
    e = AgentSuspendedEvent(pending_request=_req())
    assert e.type == "agent_suspended"
    assert e.pending_request.question_id == "q-1"


def test_agent_aborted_event_construct():
    e = AgentAbortedEvent(reason="user closed")
    assert e.type == "agent_aborted"
```

- [ ] **Step 5.2: Run — expected FAIL**

Run: `uv run pytest tests/hitl/test_events.py -v`

- [ ] **Step 5.3: Add event classes to `cubepi/agent/types.py`**

After the existing event classes (around line 100+), add:

```python
class HitlRequestEvent(BaseModel):
    type: Literal["hitl_request"] = "hitl_request"
    request: Any   # forward-declared; cubepi.hitl.types.HitlRequest


class HitlAnswerEvent(BaseModel):
    type: Literal["hitl_answer"] = "hitl_answer"
    question_id: str
    answer: Any
    cancelled: bool = False
    timed_out: bool = False


class AgentSuspendedEvent(BaseModel):
    type: Literal["agent_suspended"] = "agent_suspended"
    pending_request: Any   # forward-declared; cubepi.hitl.types.HitlRequest


class AgentAbortedEvent(BaseModel):
    type: Literal["agent_aborted"] = "agent_aborted"
    reason: str
```

NOTE: `request`/`pending_request` are `Any` to avoid the circular import (`cubepi.hitl.types` → ... → `cubepi.agent.types`). At runtime they hold `HitlRequest` instances; consumers do duck-typed access (`event.request.question_id`).

- [ ] **Step 5.4: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_events.py -v`

- [ ] **Step 5.5: Wire real events at the right layer**

Critical correction from codex pass 2: the loop has **no channel handle** in its function signature ([loop.py:143](cubepi/agent/loop.py)), so it cannot emit `AgentSuspendedEvent(pending_request=...)` with a real payload. Emitting it with `pending_request=None` violates the event contract.

The right design: **the Agent layer emits these events, not the loop.**

- **`HitlDetached`** is raised from `Agent.detach()` itself (Task 9.5), which has access to `self._channel.pending`. `Agent.detach()` emits `AgentSuspendedEvent(pending_request=self._channel.pending)` *before* triggering the exception. The loop just catches `HitlDetached` and exits silently — no event emitted from loop.
- **`HitlAborted`** is raised when a channel's signal fires (Task 9.5's `Agent.abort_pending` is the source). `Agent.abort_pending()` emits the terminal `AgentAbortedEvent` itself. Loop catches `HitlAborted` and exits silently.

Patch `_run_loop` outer handlers (replacing Task 3.13's placeholder):

```python
    try:
        # ... existing _run_loop body ...
    except (HitlDetached, HitlAborted):
        # Caller (Agent.detach / Agent.abort_pending) has already emitted the
        # appropriate AgentSuspended/AgentAborted event with the real pending.
        # Loop exits silently — assistant message and pending state are intact.
        return
```

The `loop.py` import block needs **only** the exception types it actually uses:

```python
from cubepi.hitl.exceptions import HitlAborted, HitlDetached
```

Do **not** add `HitlRequestEvent` / `HitlAnswerEvent` / `AgentSuspendedEvent` / `AgentAbortedEvent` imports to `loop.py` — those are emitted by the channel and the Agent class, not by the loop, and adding unused imports trips ruff F401.

- [ ] **Step 5.6: Extend the `AgentEvent` Union in `cubepi/agent/types.py`**

Find the existing union (around line 170-181) and extend it so typed listeners / sinks include the new variants:

```python
AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | HitlRequestEvent
    | HitlAnswerEvent
    | AgentSuspendedEvent
    | AgentAbortedEvent
)
```

Verify by running mypy-like type checks isn't needed (project has no mypy in CI per CLAUDE.md), but the type alias should be syntactically clean.

- [ ] **Step 5.7: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: no regressions; new event tests pass.

- [ ] **Step 5.8: Lint + commit**

```bash
uv run ruff check cubepi/ tests/ && uv run ruff format cubepi/ tests/
git add cubepi/agent/types.py cubepi/agent/loop.py tests/hitl/test_events.py
git commit -m "feat(hitl): HitlRequestEvent, HitlAnswerEvent, AgentSuspendedEvent, AgentAbortedEvent"
```

---

### Task 6: ApprovalPolicyMiddleware + ConfirmToolCallMiddleware + ask_user_tool

**Files:**
- Create: `cubepi/hitl/middleware.py`
- Create: `cubepi/hitl/ask_user.py`
- Create: `tests/hitl/test_approval_policy_middleware.py`
- Create: `tests/hitl/test_confirm_tool_call_middleware.py`
- Create: `tests/hitl/test_ask_user_tool.py`
- Modify: `cubepi/hitl/__init__.py` to export the new symbols

- [ ] **Step 6.1: Failing tests for ApprovalPolicyMiddleware**

Create `tests/hitl/test_approval_policy_middleware.py`:

```python
import asyncio
import pytest

from cubepi.agent.types import AgentContext, BeforeToolCallContext
from cubepi.hitl import Approve, ApproveAnswer, AskUser, Deny
from cubepi.hitl.channel import InMemoryChannel
from cubepi.hitl.middleware import ApprovalPolicyMiddleware
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall


def _ctx(tool_call_id="tc-1") -> BeforeToolCallContext:
    return BeforeToolCallContext(
        assistant_message=AssistantMessage(
            content=[TextContent(text=""), ToolCall(id=tool_call_id, name="bash", arguments={"cmd": "ls"})],
            stop_reason="tool_use",
        ),
        tool_call=ToolCall(id=tool_call_id, name="bash", arguments={"cmd": "ls"}),
        args={"cmd": "ls"},
        context=AgentContext(system_prompt="", messages=[], tools=[]),
    )


async def test_approve_policy_passthrough():
    mw = ApprovalPolicyMiddleware(InMemoryChannel(), policy=lambda c: Approve())
    result = await mw.before_tool_call(_ctx())
    assert result is None


async def test_deny_policy_blocks_without_channel():
    ch = InMemoryChannel()
    mw = ApprovalPolicyMiddleware(ch, policy=lambda c: Deny(reason="forbidden"))
    result = await mw.before_tool_call(_ctx())
    assert result.block is True
    assert result.deny_reason == "forbidden"
    assert result.hitl_trace["decision"] == "policy_deny"
    assert ch.pending is None    # channel never invoked


async def test_ask_user_policy_invokes_channel_human_approve():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    mw = ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())
    result = await mw.before_tool_call(_ctx())
    assert result is None


async def test_ask_user_policy_human_deny_blocks():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="deny", reason="no"))

    asyncio.create_task(host())
    mw = ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())
    result = await mw.before_tool_call(_ctx())
    assert result.block is True
    assert result.hitl_trace["decision"] == "human_deny"
    assert result.deny_reason == "no"


async def test_ask_user_policy_human_edit_passes_edited_args():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(
            ch.pending.question_id,
            ApproveAnswer(decision="edit", edited_args={"cmd": "ls -l"}),
        )

    asyncio.create_task(host())
    mw = ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())
    result = await mw.before_tool_call(_ctx())
    assert result.edited_args == {"cmd": "ls -l"}
    assert result.hitl_trace["decision"] == "edit"
    assert result.hitl_trace["original_args"] == {"cmd": "ls"}


async def test_timeout_translates_to_approval_timeout_deny():
    ch = InMemoryChannel()
    mw = ApprovalPolicyMiddleware(
        ch, policy=lambda c: AskUser(timeout_seconds=0.05),
    )
    result = await mw.before_tool_call(_ctx())
    assert result.block is True
    assert result.deny_reason == "approval_timeout"
    assert result.hitl_trace["decision"] == "timed_out"


async def test_cancel_translates_to_cancelled_deny():
    ch = InMemoryChannel()

    async def canceller():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.cancel(ch.pending.question_id, reason="closed tab")

    asyncio.create_task(canceller())
    mw = ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())
    result = await mw.before_tool_call(_ctx())
    assert result.block is True
    assert "cancelled: closed tab" == result.deny_reason
    assert result.hitl_trace["decision"] == "cancelled"


async def test_async_policy_is_awaited():
    ch = InMemoryChannel()
    async def policy(c):
        return Approve()
    mw = ApprovalPolicyMiddleware(ch, policy=policy)
    assert (await mw.before_tool_call(_ctx())) is None
```

- [ ] **Step 6.2: Run — expected FAIL**

Run: `uv run pytest tests/hitl/test_approval_policy_middleware.py -v`

- [ ] **Step 6.3: Implement `cubepi/hitl/middleware.py`**

```python
from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Iterable, Union

from cubepi.agent.types import BeforeToolCallContext, BeforeToolCallResult
from cubepi.hitl.channel import HitlChannel
from cubepi.hitl.exceptions import HitlCancelled, HitlTimedOut
from cubepi.hitl.policy import Approve, ApprovalDecision, AskUser, Deny
from cubepi.middleware.base import Middleware


def _args_to_dict(args: Any) -> dict:
    if hasattr(args, "model_dump"):
        return args.model_dump()
    if isinstance(args, dict):
        return dict(args)
    return dict(vars(args))


class ApprovalPolicyMiddleware(Middleware):
    def __init__(
        self,
        channel: HitlChannel,
        policy: Callable[[BeforeToolCallContext], Union[ApprovalDecision, Awaitable[ApprovalDecision]]],
    ):
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
                block=True,
                deny_reason=decision.reason,
                reason=decision.reason,
                hitl_trace={"decision": "policy_deny", "reason": decision.reason},
            )

        if isinstance(decision, AskUser):
            return await self._ask_and_translate(ctx, decision, signal=signal)

        raise TypeError(f"policy returned unexpected {type(decision).__name__}")

    async def _ask_and_translate(self, ctx, ask: AskUser, *, signal):
        original_args = _args_to_dict(ctx.args)
        try:
            answer = await self._channel.approve(
                tool_name=ctx.tool_call.name,
                tool_call_id=ctx.tool_call.id,
                args=original_args,
                details=ask.details,
                timeout=ask.timeout_seconds,
                signal=signal,
            )
        except HitlTimedOut:
            return BeforeToolCallResult(
                block=True, deny_reason="approval_timeout",
                reason="approval_timeout",
                hitl_trace={"decision": "timed_out"},
            )
        except HitlCancelled as exc:
            return BeforeToolCallResult(
                block=True, deny_reason=f"cancelled: {exc.reason}",
                reason=f"cancelled: {exc.reason}",
                hitl_trace={"decision": "cancelled", "reason": exc.reason},
            )

        if answer.decision == "approve":
            return None
        if answer.decision == "deny":
            return BeforeToolCallResult(
                block=True, deny_reason=answer.reason,
                reason=answer.reason,
                hitl_trace={"decision": "human_deny", "reason": answer.reason},
            )
        if answer.decision == "edit":
            return BeforeToolCallResult(
                edited_args=answer.edited_args,
                hitl_trace={
                    "decision": "edit",
                    "original_args": original_args,
                    "edited_args": answer.edited_args,
                },
            )


class ConfirmToolCallMiddleware(ApprovalPolicyMiddleware):
    """Convenience wrapper: 'always ask the human for these tool names'."""

    def __init__(
        self,
        channel: HitlChannel,
        *,
        require_confirm: Union[Callable[[BeforeToolCallContext], bool], Iterable[str], None] = None,
        details_fn: Callable[[BeforeToolCallContext], dict] | None = None,
        timeout_seconds: float | None = None,
    ):
        if require_confirm is None:
            matcher = lambda ctx: True
        elif callable(require_confirm):
            matcher = require_confirm
        else:
            names = set(require_confirm)
            matcher = lambda ctx: ctx.tool_call.name in names

        def policy(ctx) -> ApprovalDecision:
            if matcher(ctx):
                return AskUser(
                    timeout_seconds=timeout_seconds,
                    details=details_fn(ctx) if details_fn else None,
                )
            return Approve()

        super().__init__(channel, policy=policy)
```

- [ ] **Step 6.4: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_approval_policy_middleware.py -v`

- [ ] **Step 6.5: Failing tests for ConfirmToolCallMiddleware**

Create `tests/hitl/test_confirm_tool_call_middleware.py`:

```python
import asyncio
import pytest

from cubepi.agent.types import AgentContext, BeforeToolCallContext
from cubepi.hitl import ApproveAnswer
from cubepi.hitl.channel import InMemoryChannel
from cubepi.hitl.middleware import ConfirmToolCallMiddleware
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall


def _ctx(name="bash"):
    return BeforeToolCallContext(
        assistant_message=AssistantMessage(
            content=[TextContent(text=""), ToolCall(id="tc", name=name, arguments={})],
            stop_reason="tool_use",
        ),
        tool_call=ToolCall(id="tc", name=name, arguments={}),
        args={},
        context=AgentContext(system_prompt="", messages=[], tools=[]),
    )


async def test_set_based_require_confirm_only_asks_for_listed():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    mw = ConfirmToolCallMiddleware(ch, require_confirm={"bash"})
    # bash: prompts
    assert (await mw.before_tool_call(_ctx("bash"))) is None
    # read_file: not in set — passes through silently
    assert (await mw.before_tool_call(_ctx("read_file"))) is None
    # bash prompted exactly once; read_file did not engage channel
    assert ch.pending is None


async def test_predicate_require_confirm():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())

    def needs_confirm(ctx):
        return ctx.tool_call.name.startswith("dangerous_")

    mw = ConfirmToolCallMiddleware(ch, require_confirm=needs_confirm)
    assert (await mw.before_tool_call(_ctx("dangerous_op"))) is None


async def test_default_none_asks_for_every_tool():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    mw = ConfirmToolCallMiddleware(ch)   # no require_confirm
    assert (await mw.before_tool_call(_ctx("anything"))) is None
```

- [ ] **Step 6.6: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_confirm_tool_call_middleware.py -v`

- [ ] **Step 6.7: Failing tests for ask_user tool**

Create `tests/hitl/test_ask_user_tool.py`:

```python
import asyncio
import pytest

from cubepi.hitl.ask_user import ask_user_tool
from cubepi.hitl.channel import InMemoryChannel


async def test_ask_user_tool_is_sequential():
    tool = ask_user_tool(InMemoryChannel())
    assert tool.name == "ask_user"
    assert tool.execution_mode == "sequential"


async def test_ask_user_tool_returns_answers_in_details():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, {"color": "red"})

    tool = ask_user_tool(ch)
    asyncio.create_task(host())
    result = await tool.execute(
        "tc-1",
        tool.parameters.model_validate({
            "questions": [{"key": "color", "prompt": "Pick:"}],
        }),
        signal=None,
        on_update=lambda p: None,
    )
    assert result.details["hitl"]["answers"] == {"color": "red"}
    # Content has a human-readable summary too
    assert "color" in result.content[0].text


async def test_ask_user_tool_cancel_becomes_tool_error():
    ch = InMemoryChannel()

    async def canceller():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.cancel(ch.pending.question_id, reason="closed tab")

    tool = ask_user_tool(ch)
    asyncio.create_task(canceller())
    result = await tool.execute(
        "tc-1",
        tool.parameters.model_validate({"questions": [{"key": "x", "prompt": "?"}]}),
        signal=None, on_update=lambda p: None,
    )
    assert result.is_error is True
    assert result.details["hitl"]["outcome"] == "cancelled"
    assert result.details["hitl"]["reason"] == "closed tab"


async def test_ask_user_tool_timeout_becomes_tool_error():
    ch = InMemoryChannel(default_timeout=0.05)
    tool = ask_user_tool(ch)
    result = await tool.execute(
        "tc-1",
        tool.parameters.model_validate({"questions": [{"key": "x", "prompt": "?"}]}),
        signal=None, on_update=lambda p: None,
    )
    assert result.is_error is True
    assert result.details["hitl"]["outcome"] == "timed_out"


async def test_ask_user_tool_multi_question_form():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, {"color": "red", "size": ["s", "l"]})

    tool = ask_user_tool(ch)
    asyncio.create_task(host())
    result = await tool.execute(
        "tc-1",
        tool.parameters.model_validate({
            "questions": [
                {"key": "color", "prompt": "Color?"},
                {"key": "size", "prompt": "Sizes?", "multi_select": True},
            ],
        }),
        signal=None,
        on_update=lambda p: None,
    )
    assert result.details["hitl"]["answers"]["size"] == ["s", "l"]
```

- [ ] **Step 6.8: Run — expected FAIL**

Run: `uv run pytest tests/hitl/test_ask_user_tool.py -v`

- [ ] **Step 6.9: Implement `cubepi/hitl/ask_user.py`**

```python
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.hitl.channel import HitlChannel
from cubepi.hitl.types import Option, Question
from cubepi.providers.base import TextContent


class _OptionDef(BaseModel):
    label: str
    value: str
    description: str | None = None
    allow_input: bool = False


class _QuestionDef(BaseModel):
    key: str
    prompt: str
    options: list[_OptionDef] | None = None
    multi_select: bool = False
    required: bool = True


class AskUserParams(BaseModel):
    questions: list[_QuestionDef]


_DESCRIPTION = (
    "Ask the user one or more structured questions. Use ONLY when you need "
    "a specific selection or piece of info to proceed; for free-form clarification, "
    "just end your turn with the question as text — the user's next message will be the answer."
)


def _format_answers(answers: dict) -> str:
    return "User answers:\n" + json.dumps(answers, indent=2, ensure_ascii=False)


def ask_user_tool(channel: HitlChannel) -> AgentTool:
    async def execute(call_id: str, args: AskUserParams, *, signal=None, on_update=None) -> AgentToolResult:
        from cubepi.hitl.exceptions import HitlCancelled, HitlTimedOut
        questions = [
            Question(
                key=q.key,
                prompt=q.prompt,
                options=[Option(**o.model_dump()) for o in q.options] if q.options else None,
                multi_select=q.multi_select,
                required=q.required,
            )
            for q in args.questions
        ]
        # Per spec §7: cancel/timeout in ask_user context surface as
        # tool_result.is_error=True so the model can react. Other HITL control
        # exceptions (HitlDetached, HitlAborted) DO propagate — those signal
        # whole-agent state changes that must reach the loop's outer catch.
        try:
            answers = await channel.ask(questions, signal=signal)
        except HitlCancelled as exc:
            return AgentToolResult(
                content=[TextContent(text=f"cancelled by user: {exc.reason}")],
                details={"hitl": {"outcome": "cancelled", "reason": exc.reason}},
                is_error=True,
            )
        except HitlTimedOut as exc:
            return AgentToolResult(
                content=[TextContent(text=f"timed out after {exc.seconds} seconds")],
                details={"hitl": {"outcome": "timed_out", "seconds": exc.seconds}},
                is_error=True,
            )
        return AgentToolResult(
            content=[TextContent(text=_format_answers(answers))],
            details={"hitl": {"kind": "ask", "answers": answers}},
        )

    return AgentTool(
        name="ask_user",
        description=_DESCRIPTION,
        parameters=AskUserParams,
        execute=execute,
        execution_mode="sequential",
    )
```

- [ ] **Step 6.10: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_ask_user_tool.py -v`

- [ ] **Step 6.11: Update `cubepi/hitl/__init__.py` to export new symbols**

Add to imports and `__all__`:

```python
from cubepi.hitl.ask_user import AskUserParams, ask_user_tool
from cubepi.hitl.channel import HitlChannel, InMemoryChannel
from cubepi.hitl.middleware import ApprovalPolicyMiddleware, ConfirmToolCallMiddleware
```

And extend `__all__`:

```python
__all__ += [
    "AskUserParams", "ask_user_tool",
    "HitlChannel", "InMemoryChannel",
    "ApprovalPolicyMiddleware", "ConfirmToolCallMiddleware",
]
```

- [ ] **Step 6.12: Run full HITL test suite**

Run: `uv run pytest tests/hitl/ -v`
Expected: all pass.

- [ ] **Step 6.13: Lint + commit**

```bash
uv run ruff check cubepi/ tests/ && uv run ruff format cubepi/ tests/
git add cubepi/hitl/ tests/hitl/
git commit -m "feat(hitl): ApprovalPolicyMiddleware, ConfirmToolCallMiddleware, ask_user tool"
```

---

### Task 7: Checkpointer pending_request — Memory + SQLite

**Files:**
- Modify: `cubepi/checkpointer/base.py`
- Modify: `cubepi/checkpointer/memory.py`
- Modify: `cubepi/checkpointer/sqlite.py`
- Create: `tests/hitl/test_checkpointer_pending_request.py`

- [ ] **Step 7.1: Failing test for Memory + SQLite pending_request**

Create `tests/hitl/test_checkpointer_pending_request.py`:

```python
import asyncio
import tempfile
import pytest

from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.hitl.types import ApproveRequest, HitlRequest


def _req(thread_id="t-1", qid="q-1") -> HitlRequest:
    return HitlRequest(
        question_id=qid, thread_id=thread_id,
        payload=ApproveRequest(tool_name="bash", tool_call_id=qid, args={"cmd": "ls"}),
        created_at=0.0, timeout_seconds=30.0,
    )


@pytest.fixture
async def sqlite_cp():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        async with SQLiteCheckpointer(f.name) as cp:
            yield cp


async def test_memory_save_and_load_pending():
    cp = MemoryCheckpointer()
    assert await cp.load_pending_request("t-1") is None
    req = _req()
    await cp.save_pending_request("t-1", req)
    loaded = await cp.load_pending_request("t-1")
    assert loaded == req


async def test_memory_clear_pending():
    cp = MemoryCheckpointer()
    await cp.save_pending_request("t-1", _req())
    await cp.save_pending_request("t-1", None)
    assert await cp.load_pending_request("t-1") is None


async def test_sqlite_save_and_load_pending(sqlite_cp):
    assert await sqlite_cp.load_pending_request("t-1") is None
    req = _req()
    await sqlite_cp.save_pending_request("t-1", req)
    loaded = await sqlite_cp.load_pending_request("t-1")
    assert loaded == req


async def test_sqlite_clear_pending(sqlite_cp):
    await sqlite_cp.save_pending_request("t-1", _req())
    await sqlite_cp.save_pending_request("t-1", None)
    assert await sqlite_cp.load_pending_request("t-1") is None


async def test_sqlite_create_table_idempotent(sqlite_cp):
    """Re-opening a checkpointer DB with existing pending_request table is safe."""
    await sqlite_cp.save_pending_request("t-1", _req())
    # Re-entering the context manager would call CREATE TABLE IF NOT EXISTS again
    # against an existing table — must not raise.
    await sqlite_cp._db.execute(
        "CREATE TABLE IF NOT EXISTS thread_pending_request ("
        "thread_id TEXT PRIMARY KEY, request_json TEXT NOT NULL, "
        "created_at REAL NOT NULL DEFAULT (julianday('now')))"
    )
```

- [ ] **Step 7.2: Run — expected FAIL**

Run: `uv run pytest tests/hitl/test_checkpointer_pending_request.py -v`

- [ ] **Step 7.3: Extend the `Checkpointer` Protocol (with caller-side fallback)**

Per codex pass 2 (SHOULD-FIX): `Checkpointer` is a `typing.Protocol` ([base.py:13](cubepi/checkpointer/base.py)) — adding "default" method bodies on a Protocol does NOT confer runtime methods to third-party implementations. Third-party checkpointers that don't implement these methods need a caller-side fallback.

Add the method signatures to the Protocol so type-checkers and IDE autocomplete reflect them:

```python
# in cubepi/checkpointer/base.py — extend the Protocol class
class Checkpointer(Protocol):
    # ... existing methods ...
    async def save_pending_request(self, thread_id: str, request: Any) -> None: ...
    async def load_pending_request(self, thread_id: str) -> Any: ...
```

Then, in every caller (`Agent.respond`, `Agent.abort_pending`, `run_agent_loop_resume`, `CheckpointedChannel._on_pending_set` / `_on_pending_cleared`), use `getattr(...)` with a None fallback so third-party checkpointers that haven't implemented the methods degrade gracefully:

```python
save_pending = getattr(self.checkpointer, "save_pending_request", None)
if save_pending is not None:
    await save_pending(self.thread_id, request)

load_pending = getattr(self.checkpointer, "load_pending_request", None)
if load_pending is not None:
    return await load_pending(self.thread_id)
return None
```

The first-party checkpointers (Memory, SQLite, Postgres, MySQL) all implement them — this fallback is purely defensive for third-party impls. Document in the Protocol docstring that HITL-requiring features (`Agent.respond`, `CheckpointedChannel`) raise an informative error if the bound checkpointer doesn't support these methods.

- [ ] **Step 7.4: Implement on `MemoryCheckpointer`**

In `cubepi/checkpointer/memory.py`:

```python
class MemoryCheckpointer:
    def __init__(self) -> None:
        # ... existing init ...
        self._pending: dict[str, "HitlRequest"] = {}

    async def save_pending_request(self, thread_id, request):
        if request is None:
            self._pending.pop(thread_id, None)
        else:
            self._pending[thread_id] = request

    async def load_pending_request(self, thread_id):
        return self._pending.get(thread_id)
```

Add the import at top:

```python
from cubepi.hitl.types import HitlRequest
```

- [ ] **Step 7.5: Implement on `SQLiteCheckpointer`**

In `cubepi/checkpointer/sqlite.py`, extend `__aenter__` to create the table:

```python
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS thread_pending_request ("
            "  thread_id TEXT PRIMARY KEY,"
            "  request_json TEXT NOT NULL,"
            "  created_at REAL NOT NULL DEFAULT (julianday('now'))"
            ")"
        )
        await self._db.commit()
```

Add the two methods on the class:

```python
    async def save_pending_request(self, thread_id, request):
        from cubepi.hitl.types import HitlRequest
        assert self._db is not None
        async with self._lock:
            if request is None:
                await self._db.execute(
                    "DELETE FROM thread_pending_request WHERE thread_id = ?",
                    (thread_id,),
                )
            else:
                payload = request.model_dump_json()
                await self._db.execute(
                    "INSERT OR REPLACE INTO thread_pending_request "
                    "(thread_id, request_json) VALUES (?, ?)",
                    (thread_id, payload),
                )
            await self._db.commit()

    async def load_pending_request(self, thread_id):
        from cubepi.hitl.types import HitlRequest
        assert self._db is not None
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT request_json FROM thread_pending_request WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()
            return HitlRequest.model_validate_json(row[0]) if row else None
```

- [ ] **Step 7.6: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_checkpointer_pending_request.py -v`

- [ ] **Step 7.7: Lint + commit**

```bash
uv run ruff check cubepi/checkpointer/ tests/hitl/ && uv run ruff format cubepi/checkpointer/ tests/hitl/
git add cubepi/checkpointer/base.py cubepi/checkpointer/memory.py cubepi/checkpointer/sqlite.py tests/hitl/test_checkpointer_pending_request.py
git commit -m "feat(hitl): pending_request storage on Memory + SQLite checkpointers"
```

---

### Task 8: Checkpointer pending_request — Postgres + MySQL (schema v1→v2)

**Architectural context** (verified against current code): cubepi does NOT execute database migrations itself. The host application's **alembic** owns schema management. cubepi only:
- Defines SQLAlchemy declarative models (for the host's alembic autogenerate to detect).
- Enforces `EXPECTED_SCHEMA_VERSION` at startup via `_verify_schema()`, raising `CubepiSchemaMismatch` if the host's alembic is behind.
- Ships helpers in `cubepi/checkpointer/postgres/alembic_helpers.py` and `cubepi/checkpointer/mysql/alembic_helpers.py` that hosts call from their `upgrade()` functions (e.g. `write_schema_version_op()` to insert the version row).
- The actual checkpointer runtime is **raw asyncpg / aiomysql** — NOT SQLAlchemy sessions. All `save_*`/`load_*` methods use `pool.acquire()` + raw SQL with `$1` / `%s` placeholders.

So Task 8 does NOT create `migrations.py` modules. Instead:
1. Bump `EXPECTED_SCHEMA_VERSION` and add `pending_request` column to the SQLAlchemy models.
2. Add `save_pending_request` / `load_pending_request` methods using the same raw asyncpg/aiomysql patterns as existing methods.
3. Document a host-side alembic snippet in the docstrings + recipe (no `migrate_v1_to_v2()` helper that cubepi itself runs).
4. E2E tests bootstrap their own v2 schema (same pattern as `tests/checkpointer/test_postgres.py::_setup_schema`).

**Files:**
- Modify: `cubepi/checkpointer/postgres/models.py` — bump `EXPECTED_SCHEMA_VERSION` to `2`, add `pending_request` column.
- Modify: `cubepi/checkpointer/postgres/checkpointer.py` — add `save_pending_request` / `load_pending_request` using raw asyncpg.
- Modify: `cubepi/checkpointer/postgres/alembic_helpers.py` — add `add_pending_request_column_op()` helper that returns the SQL hosts call from their alembic v1→v2 upgrade.
- Modify: `cubepi/checkpointer/mysql/models.py` — bump `EXPECTED_SCHEMA_VERSION` to `2`, add `pending_request` column.
- Modify: `cubepi/checkpointer/mysql/checkpointer.py` — add `save_pending_request` / `load_pending_request` using raw aiomysql.
- Modify: `cubepi/checkpointer/mysql/alembic_helpers.py` — add `add_pending_request_column_op()` helper.
- Create: `tests/checkpointer/test_postgres_pending_request.py` — uses `clean_db` fixture and a `_setup_schema_v2` helper local to the test.
- Create: `tests/checkpointer/test_mysql_pending_request.py` — same pattern, uses `clean_mysql_db` + DSN.

- [ ] **Step 8.1: Bump Postgres EXPECTED_SCHEMA_VERSION + add column**

Edit `cubepi/checkpointer/postgres/models.py`:

```python
EXPECTED_SCHEMA_VERSION = 2     # was 1

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
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    # NEW: HITL pending_request, JSON-encoded HitlRequest (see cubepi/hitl/types.py).
    # Null when no pending HITL request is outstanding for this thread.
    pending_request: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, server_default=sa.text("NULL"),
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
```

The host's existing `test_write_schema_version_op_includes_expected_version` test in `tests/checkpointer/test_postgres.py` will need its `"VALUES (1)"` assertion updated to `"VALUES (2)"` (or be parameterized) — include that update in this step.

- [ ] **Step 8.2: Add a host-facing alembic helper for v1→v2**

In `cubepi/checkpointer/postgres/alembic_helpers.py` add:

```python
def add_pending_request_column_op() -> str:
    """Return SQL adding the v2 `pending_request` column to cubepi_threads.

    Call inside the host's alembic v1→v2 upgrade() via op.execute(). The new
    column is JSONB NULL. Idempotent under repeated execution via IF NOT EXISTS.
    Hosts must also bump `cubepi_schema_version` via write_schema_version_op()
    (already documented; EXPECTED_SCHEMA_VERSION is now 2)."""
    return (
        "ALTER TABLE cubepi_threads "
        "ADD COLUMN IF NOT EXISTS pending_request JSONB"
    )
```

- [ ] **Step 8.3: Add raw-asyncpg methods to `PostgresCheckpointer`**

In `cubepi/checkpointer/postgres/checkpointer.py`, after the existing `save_extra` method (around line 200), add:

```python
    async def save_pending_request(
        self, thread_id: str, request: "HitlRequest | None"
    ) -> None:
        from cubepi.hitl.types import HitlRequest as _HR  # noqa: F841 (annotation only)
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Ensure thread row exists (lazy creation matches save_extra path).
                await conn.execute(
                    "INSERT INTO cubepi_threads (thread_id) "
                    "VALUES ($1) ON CONFLICT DO NOTHING",
                    thread_id,
                )
                if request is None:
                    await conn.execute(
                        "UPDATE cubepi_threads SET pending_request = NULL, "
                        "updated_at = now() WHERE thread_id = $1",
                        thread_id,
                    )
                else:
                    payload = request.model_dump_json()
                    await conn.execute(
                        "UPDATE cubepi_threads SET pending_request = $2::jsonb, "
                        "updated_at = now() WHERE thread_id = $1",
                        thread_id, payload,
                    )

    async def load_pending_request(self, thread_id: str):
        from cubepi.hitl.types import HitlRequest
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT pending_request FROM cubepi_threads WHERE thread_id = $1",
                thread_id,
            )
        if row is None or row["pending_request"] is None:
            return None
        raw = row["pending_request"]
        # asyncpg returns JSONB as already-parsed dict OR str depending on codec config.
        if isinstance(raw, str):
            return HitlRequest.model_validate_json(raw)
        return HitlRequest.model_validate(raw)
```

The `INSERT … ON CONFLICT DO NOTHING` is identical to the existing `append()`/`save_extra()` lazy-row pattern (see [postgres/checkpointer.py:164](cubepi/checkpointer/postgres/checkpointer.py) for the original). This is what makes E2E tests robust — no need to call `append()` first.

- [ ] **Step 8.4: Mirror for MySQL**

Edit `cubepi/checkpointer/mysql/models.py`: bump `EXPECTED_SCHEMA_VERSION` to `2`, add `pending_request: Mapped[dict[str, Any] | None] = mapped_column(sa.JSON, nullable=True)` to `CubepiThread`.

Add to `cubepi/checkpointer/mysql/alembic_helpers.py`:

```python
def add_pending_request_column_op() -> str:
    return "ALTER TABLE cubepi_threads ADD COLUMN pending_request JSON NULL"
```

Add to `cubepi/checkpointer/mysql/checkpointer.py`, mirroring the Postgres methods but using aiomysql cursor and `%s` placeholders (matches the existing patterns in `save_extra`/`append`):

```python
    async def save_pending_request(self, thread_id, request):
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO cubepi_threads (thread_id) VALUES (%s) "
                    "ON DUPLICATE KEY UPDATE thread_id = thread_id",
                    (thread_id,),
                )
                if request is None:
                    await cur.execute(
                        "UPDATE cubepi_threads SET pending_request = NULL, "
                        "updated_at = NOW() WHERE thread_id = %s",
                        (thread_id,),
                    )
                else:
                    payload = request.model_dump_json()
                    await cur.execute(
                        "UPDATE cubepi_threads SET pending_request = %s, "
                        "updated_at = NOW() WHERE thread_id = %s",
                        (payload, thread_id),
                    )
            await conn.commit()

    async def load_pending_request(self, thread_id):
        from cubepi.hitl.types import HitlRequest
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT pending_request FROM cubepi_threads WHERE thread_id = %s",
                    (thread_id,),
                )
                row = await cur.fetchone()
        if row is None or row[0] is None:
            return None
        raw = row[0]
        # aiomysql returns JSON columns as str; tolerate already-parsed dicts (same
        # convention as the existing _parse_json helper in this module).
        if isinstance(raw, str):
            return HitlRequest.model_validate_json(raw)
        return HitlRequest.model_validate(raw)
```

- [ ] **Step 8.5: Failing E2E tests for Postgres (uses real fixtures)**

The existing Postgres test fixtures (see `tests/checkpointer/conftest.py`) are: `pg_dsn` (session-scoped DSN string), `_pg_available` (boolean), and `clean_db` (yields a fresh DSN per test — auto-creates and drops the database). There is no `postgres_url` and no `raw_v1_db_setup` — those were placeholder names; use the real ones.

Existing `tests/checkpointer/test_postgres.py` shows the canonical schema bootstrap: it defines a local `_setup_schema(dsn)` async function that creates `cubepi_threads`, `cubepi_messages` (partitioned), the GIN index, and `cubepi_schema_version`, then calls `write_schema_version_op()`. Our new test mirrors that but uses v2 (includes `pending_request` column) and calls `add_pending_request_column_op()`'s SQL inline (or extends `_setup_schema`).

Create `tests/checkpointer/test_postgres_pending_request.py`:

```python
import asyncpg
import pytest

from cubepi.checkpointer.postgres import PostgresCheckpointer
from cubepi.checkpointer.postgres.alembic_helpers import (
    add_pending_request_column_op, create_message_partitions_op,
    write_schema_version_op,
)
from cubepi.hitl.types import ApproveRequest, HitlRequest


async def _setup_schema_v2(dsn: str) -> None:
    """Bootstrap a v2 schema in a fresh DB (mirrors test_postgres.py::_setup_schema)."""
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("""
            CREATE TABLE cubepi_threads (
                thread_id TEXT PRIMARY KEY,
                parent_thread_id TEXT NULL REFERENCES cubepi_threads(thread_id),
                forked_at_seq BIGINT NULL,
                extra JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        # v2 column add
        await conn.execute(add_pending_request_column_op())
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
            CREATE TABLE cubepi_schema_version (version INTEGER PRIMARY KEY);
        """)
        await conn.execute(write_schema_version_op())  # writes version=2 (after step 8.1)
    finally:
        await conn.close()


def _req(qid="tc-1") -> HitlRequest:
    return HitlRequest(
        question_id=qid, thread_id="t-1",
        payload=ApproveRequest(tool_name="bash", tool_call_id=qid, args={"cmd": "ls"}),
        created_at=0.0, timeout_seconds=30.0,
    )


@pytest.mark.asyncio
async def test_postgres_save_and_load_pending_request(clean_db) -> None:
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        # Round-trip: save → load.
        await cp.save_pending_request("t-1", _req())
        loaded = await cp.load_pending_request("t-1")
    assert loaded == _req()


@pytest.mark.asyncio
async def test_postgres_clear_pending_request(clean_db) -> None:
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("t-1", _req())
        await cp.save_pending_request("t-1", None)
        loaded = await cp.load_pending_request("t-1")
    assert loaded is None


@pytest.mark.asyncio
async def test_postgres_pending_request_creates_thread_row_lazily(clean_db) -> None:
    """save_pending_request must INSERT … ON CONFLICT DO NOTHING for the thread row,
    so calling it on an unknown thread doesn't FK-violate."""
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        # 'brand-new-thread' has never seen append() or save_extra() — must still work.
        await cp.save_pending_request("brand-new-thread", _req(qid="tc-x"))
        loaded = await cp.load_pending_request("brand-new-thread")
    assert loaded is not None
    assert loaded.question_id == "tc-x"


@pytest.mark.asyncio
async def test_postgres_v2_schema_version_enforced(clean_db) -> None:
    """If the host's alembic only ran v1 (no pending_request column, schema_version=1),
    PostgresCheckpointer().__aenter__ raises CubepiSchemaMismatch."""
    from cubepi.checkpointer.postgres.exceptions import CubepiSchemaMismatch
    conn = await asyncpg.connect(clean_db)
    try:
        # v1 schema (no pending_request column, version=1)
        await conn.execute("CREATE TABLE cubepi_schema_version (version INTEGER PRIMARY KEY);")
        await conn.execute("INSERT INTO cubepi_schema_version (version) VALUES (1);")
    finally:
        await conn.close()
    with pytest.raises(CubepiSchemaMismatch):
        async with PostgresCheckpointer(clean_db):
            pass
```

- [ ] **Step 8.6: Failing E2E tests for MySQL**

Mirror in `tests/checkpointer/test_mysql_pending_request.py`. Use the existing `clean_mysql_db` and `mysql_dsn` fixtures from `tests/checkpointer/conftest.py`. Bootstrap a v2 schema using the helpers in `cubepi/checkpointer/mysql/alembic_helpers.py`. Read `tests/checkpointer/test_mysql.py` for the `_setup_schema_mysql` pattern and adapt it to include the `pending_request` column.

Tests to write (mirror Postgres list):
- `test_mysql_save_and_load_pending_request`
- `test_mysql_clear_pending_request`
- `test_mysql_pending_request_creates_thread_row_lazily`
- `test_mysql_v2_schema_version_enforced`

- [ ] **Step 8.7: Run E2E**

The Postgres tests skip if `CUBEPI_TEST_PG_DSN` is not set (existing `_pg_available` fixture). The MySQL test server per memory `reference_mysql_test_server.md` is at `192.168.1.211:6603`; set `CUBEPI_TEST_MYSQL_DSN` accordingly before running.

Run: `uv run pytest tests/checkpointer/test_postgres_pending_request.py tests/checkpointer/test_mysql_pending_request.py -v`
Expected: all pass against the live test servers.

- [ ] **Step 8.8: Update existing version-bound tests**

In `tests/checkpointer/test_postgres.py`, `test_write_schema_version_op_includes_expected_version` asserts `"VALUES (1)"`. Update to `"VALUES (2)"` (or remove the version-literal assertion and assert via `EXPECTED_SCHEMA_VERSION` import — preferred, since this couples the test to the constant rather than a hardcoded number).

Similarly in `tests/checkpointer/test_mysql.py`.

Run: `uv run pytest tests/checkpointer/test_postgres.py tests/checkpointer/test_mysql.py -v`
Expected: existing tests continue to pass (after the v2 assertion updates).

- [ ] **Step 8.9: Lint + commit**

```bash
uv run ruff check cubepi/checkpointer/ tests/checkpointer/ && uv run ruff format cubepi/checkpointer/ tests/checkpointer/
git add cubepi/checkpointer/postgres/ cubepi/checkpointer/mysql/ tests/checkpointer/test_postgres_pending_request.py tests/checkpointer/test_mysql_pending_request.py tests/checkpointer/test_postgres.py tests/checkpointer/test_mysql.py
git commit -m "feat(hitl): pending_request storage on Postgres + MySQL (schema v2)"
```

---

### Task 9: CheckpointedChannel + Agent.detach/respond/abort_pending

**Files:**
- Modify: `cubepi/hitl/channel.py` (add `CheckpointedChannel`)
- Modify: `cubepi/agent/agent.py` (add `detach`, `respond`, `abort_pending`, `load_pending_hitl_request`, `_run_hitl_resume`)
- Modify: `cubepi/agent/loop.py` (add `run_agent_loop_resume`)
- Create: `tests/hitl/test_checkpointed_channel.py`
- Create: `tests/hitl/test_agent_respond.py`
- Create: `tests/hitl/test_agent_abort_pending.py`

- [ ] **Step 9.1: Failing tests for CheckpointedChannel basics**

Create `tests/hitl/test_checkpointed_channel.py`:

```python
import asyncio
import pytest

from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl import ApproveAnswer, HitlDurabilityNotGuaranteed
from cubepi.hitl.channel import CheckpointedChannel


async def test_checkpointed_persists_pending_on_ask():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")

    async def host():
        while True:
            if await cp.load_pending_request("t-1") is not None:
                break
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    ans = await ch.approve(tool_name="bash", tool_call_id="tc-1", args={})
    assert ans.decision == "approve"
    # On success, pending should be cleared from the checkpointer
    assert await cp.load_pending_request("t-1") is None


async def test_checkpointed_durability_guard_rejects_inside_custom_tool():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")
    # Without explicit opt-in: a channel access from a "custom tool" body
    # is rejected. We simulate by setting the in-execute flag manually
    # (Task 9.5 wires the loop to set this for non-builtin tools).
    ch._enter_custom_tool_context()
    with pytest.raises(HitlDurabilityNotGuaranteed):
        await ch.confirm("ok?", timeout=0.05)


async def test_checkpointed_durability_optin_allows():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(
        checkpointer=cp, thread_id="t-1", allow_inside_custom_tool=True,
    )
    ch._enter_custom_tool_context()

    async def host():
        while True:
            if await cp.load_pending_request("t-1") is not None:
                break
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    assert await ch.confirm("ok?") is True
```

- [ ] **Step 9.2: Implement `CheckpointedChannel`**

Critical correction per codex pass 2 (BLOCKING): `_on_pending_cleared` must **only** clear the persisted pending when the await resolved with an answer (happy path) or when the request was explicitly cancelled by host action — NOT when it was interrupted by `HitlDetached`. The detach path is *exactly* the cross-process suspend scenario where pending must remain persisted until `respond()` writes the tool_result.

The cleanest implementation: the `_BaseChannel._await_answer` `finally` already calls `_on_pending_cleared(req)`. But we need to distinguish which exception (if any) caused the unwind. Pass the exception (or `None`) into the hook:

In `cubepi/hitl/channel.py`, modify `_BaseChannel._await_answer`'s finally to capture the exception and pass it down. (Sketch — the engineer should apply this on top of the Task 2 code):

```python
async def _await_answer(self, payload, timeout, signal, question_id):
    # ... (resume short-circuit, etc.) ...
    exc_caught: BaseException | None = None
    try:
        # ... (the existing wait/race logic) ...
    except BaseException as exc:
        exc_caught = exc
        raise
    finally:
        self._pending = None
        self._future = None
        await self._on_pending_cleared(req, exc=exc_caught)
```

And the default `_on_pending_cleared` in `_BaseChannel`:

```python
async def _on_pending_cleared(self, req, *, exc: BaseException | None = None) -> None:
    pass   # InMemory has nothing to do
```

Then `CheckpointedChannel`:

```python
class CheckpointedChannel(_BaseChannel):
    def __init__(
        self,
        *,
        checkpointer,
        thread_id: str,
        default_timeout: float | None = None,
        allow_inside_custom_tool: bool = False,
    ) -> None:
        super().__init__(default_timeout=default_timeout, thread_id=thread_id)
        self._checkpointer = checkpointer
        self._allow_inside_custom_tool = allow_inside_custom_tool
        self._in_custom_tool = False

    def _enter_custom_tool_context(self) -> None:
        self._in_custom_tool = True

    def _exit_custom_tool_context(self) -> None:
        self._in_custom_tool = False

    async def _on_pending_set(self, req):
        if self._in_custom_tool and not self._allow_inside_custom_tool:
            from cubepi.hitl.exceptions import HitlDurabilityNotGuaranteed
            raise HitlDurabilityNotGuaranteed(
                "CheckpointedChannel called from inside a custom tool body. "
                "Use ApprovalPolicyMiddleware or ask_user_tool, or pass "
                "allow_inside_custom_tool=True to opt in."
            )
        await self._checkpointer.save_pending_request(self._thread_id, req)
        await super()._on_pending_set(req)

    async def _on_pending_cleared(self, req, *, exc=None):
        # IMPORTANT: do NOT clear persisted pending on HitlDetached — the
        # detach path leaves pending persisted so a later respond() can
        # resume. Clearing happens on the resume path AFTER tool_result
        # is checkpointed (see run_agent_loop_resume in Task 9.6).
        from cubepi.hitl.exceptions import HitlDetached
        if isinstance(exc, HitlDetached):
            return
        # Happy path (answered) OR cancelled/timed-out/aborted — clear.
        await self._checkpointer.save_pending_request(self._thread_id, None)
```

Add a test for this in `tests/hitl/test_checkpointed_channel.py`:

```python
async def test_detach_leaves_pending_persisted():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")

    async def detacher():
        while ch.pending is None:
            await asyncio.sleep(0)
        if ch._future is not None and not ch._future.done():
            from cubepi.hitl.exceptions import HitlDetached
            ch._future.set_exception(HitlDetached())

    asyncio.create_task(detacher())
    from cubepi.hitl.exceptions import HitlDetached
    with pytest.raises(HitlDetached):
        await ch.confirm("ok?")
    # Persisted state must remain
    assert await cp.load_pending_request("t-1") is not None
```

- [ ] **Step 9.3: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_checkpointed_channel.py -v`

- [ ] **Step 9.4: Wire the "inside custom tool" durability guard via ContextVar**

Per codex pass 2 (SHOULD-FIX): closure introspection is too brittle (misses callable objects, default args, channels held as instance attributes, etc.) AND not coroutine-local (a shared channel + parallel tools would race the `_in_custom_tool` flag). Use a `contextvars.ContextVar` instead — it's coroutine-local out of the box.

In `cubepi/hitl/channel.py`, add at module level:

```python
import contextvars

# Coroutine-local flag: True while we're inside the execute() body of a tool
# that is NOT a built-in HITL tool. CheckpointedChannel reads this in
# _on_pending_set to enforce HitlDurabilityNotGuaranteed.
_in_custom_tool_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_in_custom_tool_var", default=False,
)
```

Replace `CheckpointedChannel._enter_custom_tool_context` / `_exit_custom_tool_context` / `self._in_custom_tool` with reads of this ContextVar:

```python
class CheckpointedChannel(_BaseChannel):
    def __init__(self, *, checkpointer, thread_id: str,
                 default_timeout: float | None = None,
                 allow_inside_custom_tool: bool = False) -> None:
        super().__init__(default_timeout=default_timeout, thread_id=thread_id)
        self._checkpointer = checkpointer
        self._allow_inside_custom_tool = allow_inside_custom_tool

    async def _on_pending_set(self, req):
        if _in_custom_tool_var.get() and not self._allow_inside_custom_tool:
            from cubepi.hitl.exceptions import HitlDurabilityNotGuaranteed
            raise HitlDurabilityNotGuaranteed(
                "CheckpointedChannel called from inside a custom tool body. "
                "Use ApprovalPolicyMiddleware or ask_user_tool, or pass "
                "allow_inside_custom_tool=True to opt in."
            )
        await self._checkpointer.save_pending_request(self._thread_id, req)
        await super()._on_pending_set(req)

    async def _on_pending_cleared(self, req, *, exc=None):
        from cubepi.hitl.exceptions import HitlDetached
        if isinstance(exc, HitlDetached):
            return
        await self._checkpointer.save_pending_request(self._thread_id, None)
```

Update the `tests/hitl/test_checkpointed_channel.py` tests in Step 9.1 — instead of calling `ch._enter_custom_tool_context()`, set the ContextVar inside the test:

```python
async def test_checkpointed_durability_guard_rejects_inside_custom_tool():
    from cubepi.hitl.channel import _in_custom_tool_var
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")
    token = _in_custom_tool_var.set(True)
    try:
        with pytest.raises(HitlDurabilityNotGuaranteed):
            await ch.confirm("ok?", timeout=0.05)
    finally:
        _in_custom_tool_var.reset(token)
```

Wire the flag in `cubepi/agent/tools.py` `_execute_prepared`. Discriminating built-in HITL tools vs custom: ask_user_tool's factory sets `tool._hitl_builtin = True` on the returned `AgentTool` (declarative metadata, not closure-walking):

```python
# cubepi/hitl/ask_user.py
def ask_user_tool(channel):
    tool = AgentTool(
        name="ask_user", description=_DESCRIPTION,
        parameters=AskUserParams, execute=execute,
        execution_mode="sequential",
    )
    tool._hitl_builtin = True
    return tool
```

Then in `_execute_prepared`:

```python
async def _execute_prepared(prepared, signal, emit_fn):
    from cubepi.hitl.channel import _in_custom_tool_var
    is_builtin = getattr(prepared.tool, "_hitl_builtin", False)
    token = None if is_builtin else _in_custom_tool_var.set(True)
    try:
        try:
            result = await prepared.tool.execute(
                prepared.tool_call.id,
                prepared.args,
                signal=signal,
                on_update=lambda partial: emit_event(
                    emit_fn,
                    ToolExecutionUpdateEvent(
                        tool_call_id=prepared.tool_call.id,
                        tool_name=prepared.tool_call.name,
                        args=prepared.tool_call.arguments,
                        partial_result=partial,
                    ),
                ),
            )
            return result, False
        except HitlControlException:
            raise
        except Exception as exc:
            return _error_result(str(exc)), True
    finally:
        if token is not None:
            _in_custom_tool_var.reset(token)
```

The ContextVar is coroutine-local, so parallel tools each get their own value — codex's "shared-channel parallel tools" concern is resolved.

- [ ] **Step 9.5: Implement `Agent.detach`, `load_pending_hitl_request`, `respond`, `abort_pending`**

Edit `cubepi/agent/agent.py`. Add inside the class:

```python
    async def detach(self) -> None:
        from cubepi.agent.types import AgentSuspendedEvent
        if self._channel is None:
            raise HitlError("agent has no channel bound")
        pending = self._channel.pending
        if pending is None or self._channel._future is None or self._channel._future.done():
            return    # nothing to detach
        # Emit the suspended event BEFORE triggering the exception, so listeners
        # see the real pending payload (codex pass 2 BLOCKING: previous draft
        # emitted from the loop with pending=None — fundamentally wrong).
        await self._process_event(AgentSuspendedEvent(pending_request=pending))
        self._channel._future.set_exception(HitlDetached())

    async def load_pending_hitl_request(self):
        if self.checkpointer is None or self.thread_id is None:
            return None
        return await self.checkpointer.load_pending_request(self.thread_id)

    async def respond(self, *, question_id=None, answer):
        from cubepi.hitl.exceptions import (
            HitlError, HitlMissingAnswer, HitlNoPendingRequest, HitlStaleAnswer,
        )
        if self._channel is None:
            raise HitlError("agent has no channel bound")
        if not (self.thread_id and self.checkpointer):
            raise RuntimeError("respond() requires thread_id + checkpointer")

        async with self._run_lock:
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
            await self._run_hitl_resume()

    async def abort_pending(self, reason: str = "aborted by host") -> None:
        from cubepi.agent.types import AgentAbortedEvent
        from cubepi.providers.base import AssistantMessage, ToolResultMessage, TextContent
        if self._channel is None:
            raise HitlError("agent has no channel bound")
        if not (self.thread_id and self.checkpointer):
            raise RuntimeError("abort_pending() requires thread_id + checkpointer")

        async with self._run_lock:
            # 1. If something is pending in-flight (same process), cancel it.
            if self._channel.pending is not None:
                await self._channel.cancel(self._channel.pending.question_id, reason=reason)

            # 2. Load pending from checkpointer (the in-flight cancel may have
            #    already cleared it via _on_pending_cleared); if still set, build
            #    the synthetic deny and clear.
            pending = await self.checkpointer.load_pending_request(self.thread_id)
            if pending is None:
                return  # nothing to abort

            # Find the gated tool_call_id in the last assistant message.
            if not self._state._messages:
                data = await self.checkpointer.load(self.thread_id)
                self._state._messages = list(data.messages or []) if data else []
            from cubepi.providers.base import AssistantMessage
            last = self._state._messages[-1]
            if isinstance(last, AssistantMessage):
                from cubepi.providers.base import ToolCall
                for content in last.content:
                    if isinstance(content, ToolCall):
                        synthetic = ToolResultMessage(
                            tool_call_id=content.id,
                            tool_name=content.name,
                            content=[TextContent(text=f"aborted: {reason}")],
                            details={"hitl": {"decision": "aborted", "reason": reason}},
                            is_error=True,
                            timestamp=time.time(),
                        )
                        self._state._messages.append(synthetic)
                        if self.checkpointer:
                            await self.checkpointer.append(self.thread_id, [synthetic])
                # Append a terminal aborted assistant
                term = AssistantMessage(
                    content=[TextContent(text=f"Conversation aborted: {reason}")],
                    stop_reason="aborted",
                )
                self._state._messages.append(term)
                if self.checkpointer:
                    await self.checkpointer.append(self.thread_id, [term])

            await self.checkpointer.save_pending_request(self.thread_id, None)
            await self._process_event(AgentAbortedEvent(reason=reason))

    async def _run_hitl_resume(self) -> None:
        await self._run_with_lifecycle(
            lambda signal: run_agent_loop_resume(
                context=self._create_context_snapshot(),
                provider=self._provider,
                model=self._state.model,
                convert_to_llm=self.convert_to_llm,
                transform_context=self.transform_context,
                transform_system_prompt=self.transform_system_prompt,
                after_model_response=self.after_model_response,
                before_tool_call=self.before_tool_call,
                after_tool_call=self.after_tool_call,
                should_stop_after_turn=self.should_stop_after_turn,
                get_steering_messages=self._make_async_drain(self._steering_queue),
                get_follow_up_messages=self._make_async_drain(self._follow_up_queue),
                stream_options=self._build_stream_options(signal),
                tool_execution=self.tool_execution,
                emit=lambda e: self._process_event(e),
                checkpointer=self.checkpointer,
                thread_id=self.thread_id,
            )
        )
```

Imports to add: `from cubepi.agent.loop import run_agent_loop_resume` and `from cubepi.hitl.exceptions import HitlDetached, HitlError`.

- [ ] **Step 9.6: Implement `run_agent_loop_resume` in `cubepi/agent/loop.py`**

Per codex pass 2 (BLOCKING), this function MUST preserve existing loop behavior — specifically:
- `batch.terminate` (a tool returned `terminate=True`) must short-circuit before re-entering the model loop. See [tools.py:202](cubepi/agent/tools.py) `_should_terminate`.
- `should_stop_after_turn(ShouldStopAfterTurnContext(...))` must run *after* tool_results land and before the next model call. See [loop.py:309](cubepi/agent/loop.py).
- `TurnEndEvent` must carry the actual `tool_results` (the new `ToolResultMessage`s), not an empty list.
- The "find unresolved assistant message" idempotency check must use **reverse identity search**, not `list.index()` — `context.messages.index(last)` does value equality and can match an earlier assistant message with identical content (e.g. retries).

Add the function (entirety):

```python
async def run_agent_loop_resume(
    *,
    context, provider, model, convert_to_llm, emit,
    transform_context=None, transform_system_prompt=None,
    after_model_response=None, before_tool_call=None, after_tool_call=None,
    should_stop_after_turn=None, get_steering_messages=None,
    get_follow_up_messages=None, stream_options=None,
    tool_execution="parallel", system_prompt=None,
    checkpointer=None, thread_id=None,
) -> list[Message]:
    from cubepi.providers.base import AssistantMessage, ToolCall, ToolResultMessage
    from cubepi.agent.types import (
        AgentEndEvent, AgentStartEvent, MessageEndEvent, MessageStartEvent,
        ShouldStopAfterTurnContext, TurnEndEvent, TurnStartEvent,
    )
    from cubepi.hitl.exceptions import HitlInconsistentState

    new_messages: list[Message] = []

    # Sanity check
    if not context.messages:
        raise HitlInconsistentState("resume called with empty message history")
    last = context.messages[-1]
    if not isinstance(last, AssistantMessage):
        raise HitlInconsistentState(
            f"resume requires last message to be AssistantMessage, got {type(last).__name__}"
        )
    unresolved = [c for c in last.content if isinstance(c, ToolCall)]
    if not unresolved:
        raise HitlInconsistentState("resume requires unresolved tool_calls in last message")

    # Locate `last` by IDENTITY (not value equality). list.index() uses ==, which
    # would match an earlier assistant message with identical content (rare but
    # possible after retries or reused prompts).
    asst_pos = next(
        (i for i in range(len(context.messages) - 1, -1, -1)
         if context.messages[i] is last),
        -1,
    )
    if asst_pos < 0:
        raise HitlInconsistentState("could not locate last assistant message by identity")

    # Idempotency: a previous crashed-mid-execute resume may have already left
    # ToolResultMessage(s) for some of the unresolved tool_calls after asst_pos.
    # Skip those — DO NOT re-run side-effecting tool bodies.
    already_resolved = {
        m.tool_call_id for m in context.messages[asst_pos + 1:]
        if isinstance(m, ToolResultMessage)
    }
    remaining = [tc for tc in unresolved if tc.id not in already_resolved]

    await emit_event(emit, AgentStartEvent())
    await emit_event(emit, TurnStartEvent())

    current_context = context
    batch_tool_results: list[ToolResultMessage] = []
    terminated_by_tool = False

    if remaining:
        # Build a fresh assistant message containing only the remaining
        # tool_calls so execute_tool_calls processes those exact entries.
        # We do NOT mutate `last` itself — model_copy gives an independent
        # AssistantMessage that execute_tool_calls can read from.
        partial_msg = last.model_copy(update={
            "content": [c for c in last.content
                        if not isinstance(c, ToolCall) or c.id in {tc.id for tc in remaining}],
        })
        batch = await execute_tool_calls(
            current_context, partial_msg,
            tool_execution=tool_execution,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
            signal=(stream_options or StreamOptions()).signal,
            emit=emit,
        )
        batch_tool_results = list(batch.messages)
        terminated_by_tool = batch.terminate

        for r in batch_tool_results:
            current_context.messages.append(r)
            new_messages.append(r)

    # Clear pending_request from checkpointer NOW — after tool_results are
    # appended (and have been checkpointed by the Agent layer's MessageEndEvent
    # handler). The pending_request column / row is the cross-process witness;
    # holding it until here preserves crash-recovery idempotency (see spec §5.2).
    if checkpointer is not None and thread_id is not None:
        save_pending = getattr(checkpointer, "save_pending_request", None)
        if save_pending is not None:
            await save_pending(thread_id, None)

    # Emit TurnEndEvent with the ACTUAL tool_results so listeners get the
    # right payload (codex BLOCKING: previous draft emitted []).
    await emit_event(emit, TurnEndEvent(message=last, tool_results=batch_tool_results))

    # Honor should_stop_after_turn (codex BLOCKING: previous draft skipped this).
    if should_stop_after_turn:
        stop_ctx = ShouldStopAfterTurnContext(
            message=last,
            tool_results=batch_tool_results,
            context=current_context,
            new_messages=new_messages,
        )
        if await should_stop_after_turn(stop_ctx):
            await emit_event(emit, AgentEndEvent(messages=new_messages))
            return new_messages

    # Terminate-by-tool semantics (codex BLOCKING: previous draft ignored).
    if terminated_by_tool:
        await emit_event(emit, AgentEndEvent(messages=new_messages))
        return new_messages

    # Drain steering AFTER tool_results — preserves the Anthropic adjacency
    # invariant the existing loop enforces (no user/system message wedged
    # between tool_use and tool_result).
    if get_steering_messages:
        steering = await get_steering_messages() or []
        for msg in steering:
            await emit_event(emit, MessageStartEvent(message=msg))
            await emit_event(emit, MessageEndEvent(message=msg))
            current_context.messages.append(msg)
            new_messages.append(msg)

    # Fall through to the normal loop for the next model turn.
    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        provider=provider,
        model=model,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        transform_system_prompt=transform_system_prompt,
        after_model_response=after_model_response,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        should_stop_after_turn=should_stop_after_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
        stream_options=stream_options,
        tool_execution=tool_execution,
        emit=emit,
    )
    return new_messages
```

- [ ] **Step 9.7: Failing tests for `Agent.respond` and `Agent.abort_pending`**

Create `tests/hitl/test_agent_respond.py`:

```python
import asyncio
import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl import (
    Approve, ApproveAnswer, AskUser, HitlNoPendingRequest, HitlStaleAnswer,
)
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.hitl.middleware import ApprovalPolicyMiddleware
from cubepi.providers.faux import (
    FauxProvider, faux_assistant_message, faux_text, faux_tool_call,
)
from cubepi.providers.base import Model
from pydantic import BaseModel
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import TextContent


class _Params(BaseModel):
    cmd: str


def _bash_tool():
    async def execute(call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"ran {args.cmd}")])
    return AgentTool(
        name="bash", description="run a shell command",
        parameters=_Params, execute=execute,
        execution_mode="sequential",
    )


def _two_turn_bash_responses():
    """Turn 1 calls bash; turn 2 (post tool-result) ends."""
    return [
        faux_assistant_message(
            [faux_text("ok"), faux_tool_call("bash", {"cmd": "ls"}, id="tc-1")],
            stop_reason="tool_use",
        ),
        faux_assistant_message("done"),
    ]


async def test_respond_completes_a_suspended_run():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")
    provider = FauxProvider()
    provider.set_responses(_two_turn_bash_responses())
    agent = Agent(
        provider=provider, model=Model(id="faux", provider="faux"),
        tools=[_bash_tool()],
        middleware=[ApprovalPolicyMiddleware(
            ch, policy=lambda c: AskUser(),
        )],
        channel=ch,
        checkpointer=cp, thread_id="t-1",
    )

    # Start the agent — it will suspend on channel.approve.
    async def run():
        await agent.prompt("hi")

    task = asyncio.create_task(run())

    # Wait until pending appears
    for _ in range(100):
        if ch.pending is not None:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("agent did not suspend on HITL")

    # Detach so the run() returns; respond() will pick up.
    await agent.detach()
    await task   # run() returns cleanly

    # Now respond with approve.
    await agent.respond(question_id="tc-1", answer=ApproveAnswer(decision="approve"))

    # The conversation should now have: user, assistant(toolcall), tool_result, assistant(done)
    msgs = agent.state.messages
    assert msgs[-1].content[0].text == "done"


async def test_respond_stale_answer():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")
    agent = Agent(
        provider=_faux_with([faux_assistant_message("")]),
        model=Model(id="faux", provider="faux"),
        channel=ch, checkpointer=cp, thread_id="t-1",
    )
    # Manually persist a pending then try the wrong qid.
    from cubepi.hitl.types import ApproveRequest, HitlRequest
    await cp.save_pending_request("t-1", HitlRequest(
        question_id="tc-real", thread_id="t-1",
        payload=ApproveRequest(tool_name="bash", tool_call_id="tc-real", args={}),
        created_at=0.0,
    ))
    with pytest.raises(HitlStaleAnswer):
        await agent.respond(question_id="tc-wrong", answer=ApproveAnswer(decision="approve"))


async def test_respond_no_pending():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")
    agent = Agent(
        provider=_faux_with([faux_assistant_message("")]),
        model=Model(id="faux", provider="faux"),
        channel=ch, checkpointer=cp, thread_id="t-1",
    )
    with pytest.raises(HitlNoPendingRequest):
        await agent.respond(answer=ApproveAnswer(decision="approve"))
```

NOTE on the FauxProvider API used above: `FauxProvider()` takes no `scripts` kwarg; you preload responses with `provider.set_responses([...])`. Helpers are `faux_text(str)`, `faux_tool_call(name, args, *, id=...)`, `faux_assistant_message(content, *, stop_reason="stop")`. The `_two_turn_bash_responses()` factory in this file shows the canonical multi-turn pattern. To capture inputs the provider received (for cache-prefix tests in Task 9.11), use `provider.subscribe_request(lambda payload, model: captured.append(payload))` — payload has `{"model","messages","system_prompt"}` keys.

- [ ] **Step 9.8: Run respond tests — expected PASS**

Run: `uv run pytest tests/hitl/test_agent_respond.py -v`

If FauxProvider API differs, fix the test imports/construction. Confirm by reading `cubepi/providers/faux.py` first.

- [ ] **Step 9.9: Failing tests for abort_pending**

Create `tests/hitl/test_agent_abort_pending.py`:

```python
import asyncio
import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl import AskUser
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.hitl.middleware import ApprovalPolicyMiddleware
# (same FauxProvider imports as above)


async def test_abort_pending_closes_conversation():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")
    # turn 1 calls bash.
    provider = FauxProvider()
    provider.set_responses([
        faux_assistant_message(
            [faux_text("ok"), faux_tool_call("bash", {"cmd": "ls"}, id="tc-1")],
            stop_reason="tool_use",
        ),
    ])
    agent = Agent(
        provider=provider,
        model=Model(id="faux", provider="faux"),
        tools=[_bash_tool()],
        middleware=[ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())],
        channel=ch, checkpointer=cp, thread_id="t-1",
    )
    task = asyncio.create_task(agent.prompt("hi"))
    for _ in range(100):
        if ch.pending is not None:
            break
        await asyncio.sleep(0.01)
    await agent.detach()
    await task

    await agent.abort_pending(reason="user closed tab")

    msgs = agent.state.messages
    # Should end with a synthetic deny tool_result and a stop_reason=aborted assistant
    assert msgs[-2].is_error is True
    assert "user closed tab" in msgs[-2].content[0].text
    assert msgs[-1].stop_reason == "aborted"
    # pending is cleared
    assert await cp.load_pending_request("t-1") is None
```

- [ ] **Step 9.10: Run abort tests — expected PASS**

Run: `uv run pytest tests/hitl/test_agent_abort_pending.py -v`

- [ ] **Step 9.11: Add cache-prefix tests**

Create `tests/hitl/test_resume_cache_prefix.py`:

```python
import asyncio
import json
import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.hitl import ApproveAnswer, AskUser
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.hitl.middleware import ApprovalPolicyMiddleware
# (same Faux imports + _bash_tool helper)


async def _suspend_resume_and_capture(checkpointer):
    ch = CheckpointedChannel(checkpointer=checkpointer, thread_id="t-1")
    provider = FauxProvider()
    provider.set_responses(_two_turn_bash_responses())

    # Capture every payload the provider receives via the public observer hook.
    captured: list[dict] = []
    provider.subscribe_request(lambda payload, model: captured.append(payload))

    agent = Agent(
        provider=provider,
        model=Model(id="faux", provider="faux"),
        tools=[_bash_tool()],
        middleware=[ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())],
        channel=ch, checkpointer=checkpointer, thread_id="t-1",
    )
    task = asyncio.create_task(agent.prompt("hi"))
    for _ in range(100):
        if ch.pending is not None:
            break
        await asyncio.sleep(0.01)

    # First-turn payload bytes (the one the model already saw before suspend)
    pre_messages = list(captured[0]["messages"])

    await agent.detach()
    await task

    await agent.respond(question_id="tc-1", answer=ApproveAnswer(decision="approve"))

    # Second turn = post-resume model call. The first len(pre_messages) entries
    # must be byte-identical to the first turn for prompt-cache to hit.
    second_turn_messages = captured[1]["messages"]
    return pre_messages, second_turn_messages[: len(pre_messages)]


@pytest.mark.asyncio
async def test_resume_preserves_cache_prefix_memory():
    pre, post = await _suspend_resume_and_capture(MemoryCheckpointer())
    assert pre == post


@pytest.mark.asyncio
async def test_resume_preserves_cache_prefix_sqlite(tmp_path):
    db = tmp_path / "x.db"
    async with SQLiteCheckpointer(str(db)) as cp:
        pre, post = await _suspend_resume_and_capture(cp)
        assert pre == post
```

The comparison uses `provider.subscribe_request(...)` (existing public API in `cubepi/providers/base.py`) — no FauxProvider changes needed. Each captured payload is the dict `{"model", "messages", "system_prompt"}` that the provider was called with.

- [ ] **Step 9.12: Run cache-prefix tests**

Run: `uv run pytest tests/hitl/test_resume_cache_prefix.py -v`

If FauxProvider doesn't record calls, patch it to do so in a small edit (justified for HITL testing; mention this in the commit message).

- [ ] **Step 9.12b: Add `CheckpointedChannel` to public exports**

In `cubepi/hitl/__init__.py` (extended in Step 6.11), add `CheckpointedChannel` so the doc examples (Task 12) and host code can import it from the top-level:

```python
from cubepi.hitl.channel import CheckpointedChannel, HitlChannel, InMemoryChannel
__all__ += ["CheckpointedChannel"]
```

Quick smoke test (add to `tests/hitl/test_init.py` or a similar existing file):

```python
def test_checkpointed_channel_public_export():
    from cubepi.hitl import CheckpointedChannel
    assert CheckpointedChannel is not None
```

- [ ] **Step 9.13: Lint + commit**

```bash
uv run ruff check cubepi/ tests/ && uv run ruff format cubepi/ tests/
git add cubepi/hitl/channel.py cubepi/agent/agent.py cubepi/agent/loop.py cubepi/agent/tools.py cubepi/hitl/ask_user.py cubepi/providers/faux.py tests/hitl/
git commit -m "feat(hitl): CheckpointedChannel, Agent.respond/detach/abort_pending, resume loop"
```

---

### Task 10: Trace integration (lazy OTel)

**Files:**
- Create: `cubepi/hitl/_trace.py`
- Modify: `cubepi/hitl/channel.py` (wrap awaits in spans)
- Create: `tests/hitl/test_trace_spans.py`

- [ ] **Step 10.1: Failing test for trace span emission**

Create `tests/hitl/test_trace_spans.py`:

```python
import asyncio
import pytest

pytest.importorskip("opentelemetry")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    InMemorySpanExporter,
)
from cubepi.hitl import ApproveAnswer
from cubepi.hitl.channel import InMemoryChannel


@pytest.fixture
def exporter():
    provider = TracerProvider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    trace.set_tracer_provider(provider)
    yield exp


async def test_approve_emits_hitl_span(exporter):
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    await ch.approve(tool_name="bash", tool_call_id="tc-1", args={})

    spans = exporter.get_finished_spans()
    hitl_spans = [s for s in spans if s.name == "hitl.approve"]
    assert len(hitl_spans) == 1
    assert hitl_spans[0].attributes["hitl.question_id"] == "tc-1"
    assert hitl_spans[0].attributes["hitl.tool_name"] == "bash"
    assert hitl_spans[0].attributes["hitl.outcome"] == "approved"
```

- [ ] **Step 10.2: Run — expected FAIL**

Run: `uv run pytest tests/hitl/test_trace_spans.py -v`

- [ ] **Step 10.3: Implement `cubepi/hitl/_trace.py`**

```python
from __future__ import annotations

import contextlib
from typing import Any


class _NullSpan:
    def set_attribute(self, *a, **k): pass
    def add_event(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


@contextlib.contextmanager
def hitl_span(kind: str, **attrs):
    try:
        from opentelemetry import trace
    except ImportError:
        yield _NullSpan()
        return
    tracer = trace.get_tracer("cubepi.hitl")
    with tracer.start_as_current_span(f"hitl.{kind}") as span:
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(f"hitl.{k}", v)
        yield span
```

- [ ] **Step 10.4: Wire spans into `_BaseChannel._await_answer`**

Per codex pass 2 (SHOULD-FIX): HITL control exceptions inherit `BaseException`, not `Exception`, so an `except Exception:` handler in the tracing wrapper would skip them and leave `outcome="unknown"`. Use `BaseException`. Also set `hitl.duration_seconds` per spec §6.5.

Edit `cubepi/hitl/channel.py` — wrap `_await_answer`:

```python
    async def _await_answer(self, payload, timeout, signal, question_id):
        kind = payload.kind
        attrs = {"question_id": question_id, "timeout_seconds": timeout}
        if kind == "approve":
            attrs["tool_call_id"] = payload.tool_call_id
            attrs["tool_name"] = payload.tool_name
        from cubepi.hitl._trace import hitl_span
        import time as _time
        with hitl_span(kind, **attrs) as span:
            outcome = "unknown"
            from_resume = False
            t0 = _time.monotonic()
            try:
                # Resume short-circuit
                if self._resume_slot is not None and self._resume_slot[0] == question_id:
                    _, ans = self._resume_slot
                    self._resume_slot = None
                    from_resume = True
                    outcome = _outcome_from_answer(kind, ans)
                    return ans
                # ... existing wait/race logic from Task 2.3 + Step 9.2 ...
                result = await _the_real_wait_logic(...)   # see Task 2.3
                outcome = _outcome_from_answer(kind, result)
                return result
            except BaseException as exc:   # catches HitlControlException too
                outcome = _outcome_from_exception(exc)
                raise
            finally:
                span.set_attribute("hitl.from_resume", from_resume)
                span.set_attribute("hitl.outcome", outcome)
                span.set_attribute("hitl.duration_seconds", _time.monotonic() - t0)


def _outcome_from_answer(kind, ans):
    if kind == "approve":
        return {"approve": "approved", "deny": "denied", "edit": "edited"}.get(
            getattr(ans, "decision", None), "answered"
        )
    return "answered"


def _outcome_from_exception(exc):
    from cubepi.hitl.exceptions import HitlCancelled, HitlTimedOut, HitlAborted, HitlDetached
    if isinstance(exc, HitlCancelled): return "cancelled"
    if isinstance(exc, HitlTimedOut): return "timed_out"
    if isinstance(exc, HitlAborted): return "aborted"
    if isinstance(exc, HitlDetached): return "detached"
    return "error"
```

(The structural change in `_await_answer` is incremental on top of Task 2.3's body — the engineer integrates `with hitl_span(...) as span:` around the existing wait/race logic and adds the `t0` measurement + duration attribute.)

- [ ] **Step 10.5: Run trace tests**

Run: `uv run pytest tests/hitl/test_trace_spans.py -v`

- [ ] **Step 10.6: Lint + commit**

```bash
uv run ruff check cubepi/hitl/ tests/hitl/ && uv run ruff format cubepi/hitl/ tests/hitl/
git add cubepi/hitl/_trace.py cubepi/hitl/channel.py tests/hitl/test_trace_spans.py
git commit -m "feat(hitl): lazy OTel hitl.{kind} spans with outcome attributes"
```

---

### Task 11: Subagent channel inheritance + NoopChannel + ScriptedChannel

**Files:**
- Create: `cubepi/hitl/testing.py`
- Create: `tests/hitl/test_subagent_channel_inheritance.py`

- [ ] **Step 11.1: Implement `ScriptedChannel` and `NoopChannel`**

Create `cubepi/hitl/testing.py`:

```python
from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Union

from cubepi.hitl.channel import _BaseChannel
from cubepi.hitl.exceptions import HitlError
from cubepi.hitl.types import HitlRequest


class ScriptedChannel(_BaseChannel):
    """Pre-programmed answers for deterministic tests.

    answers: list of values or callables. Each call to ask/confirm/approve
    consumes the next item. A callable receives the HitlRequest and returns
    the answer.
    """

    def __init__(self, answers: list[Union[Any, Callable[[HitlRequest], Any]]]):
        super().__init__()
        self._answers = list(answers)
        self._history: list[HitlRequest] = []

    @property
    def history(self) -> list[HitlRequest]:
        return list(self._history)

    async def _await_answer(self, payload, timeout, signal, question_id):
        import uuid
        if self._resume_slot is not None and self._resume_slot[0] == question_id:
            _, ans = self._resume_slot
            self._resume_slot = None
            return ans
        if not self._answers:
            raise HitlError(f"ScriptedChannel exhausted (received {payload!r})")
        req = HitlRequest(
            question_id=question_id, thread_id=None, payload=payload,
            created_at=time.time(), timeout_seconds=timeout,
        )
        self._history.append(req)
        head = self._answers.pop(0)
        return head(req) if callable(head) else head


class NoopChannel(_BaseChannel):
    """Auto-approves everything. Useful for subagents in tests."""

    async def _await_answer(self, payload, timeout, signal, question_id):
        from cubepi.hitl.types import ApproveAnswer, AskRequest, ConfirmRequest
        kind = payload.kind
        if kind == "approve": return ApproveAnswer(decision="approve")
        if kind == "confirm": return True
        if kind == "ask":
            return {q.key: "" for q in payload.questions}
        raise HitlError(f"NoopChannel does not handle {kind!r}")
```

- [ ] **Step 11.2: Failing test for subagent inheritance + NoopChannel**

Create `tests/hitl/test_subagent_channel_inheritance.py`:

```python
import asyncio
import pytest

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.hitl.channel import InMemoryChannel
from cubepi.hitl.testing import NoopChannel, ScriptedChannel
from cubepi.hitl.types import Question
from cubepi.hitl import ApproveAnswer
from cubepi.providers.base import Model, TextContent
from cubepi.providers.faux import FauxProvider, faux_assistant_message


def _faux_with(responses):
    """Helper: build a FauxProvider preloaded with responses (mirrors real API)."""
    p = FauxProvider()
    p.set_responses(responses)
    return p
from pydantic import BaseModel


async def test_scripted_channel_returns_canned_answers():
    ch = ScriptedChannel(answers=[
        ApproveAnswer(decision="approve"),
        {"color": "red"},
    ])
    ans1 = await ch.approve(tool_name="bash", tool_call_id="tc-1", args={})
    assert ans1.decision == "approve"
    ans2 = await ch.ask([Question(key="color", prompt="?")])
    assert ans2 == {"color": "red"}
    assert len(ch.history) == 2


async def test_noop_channel_auto_approves():
    ch = NoopChannel()
    assert (await ch.approve(tool_name="x", tool_call_id="y", args={})).decision == "approve"
    assert (await ch.confirm("?")) is True


async def test_subagent_inherits_parent_channel():
    # Parent has channel; subagent (constructed inside a tool) gets channel=parent.channel.
    parent_ch = InMemoryChannel()

    # Subagent tool: constructs an inner Agent with parent's channel.
    class _NoParams(BaseModel):
        task: str

    async def subagent_execute(call_id, args, *, signal=None, on_update=None):
        # The subagent factory uses the same channel object as parent.
        inner = Agent(
            provider=_faux_with([faux_assistant_message("")]),
            model=Model(id="faux", provider="faux"),
            channel=parent_ch,
        )
        # Verify same channel
        assert inner.channel is parent_ch
        return AgentToolResult(content=[TextContent(text="subagent done")])

    subagent_tool = AgentTool(
        name="run_subagent", description="run a subagent",
        parameters=_NoParams, execute=subagent_execute,
        execution_mode="sequential",
    )

    parent = Agent(
        provider=_faux_with([faux_assistant_message("")]),
        model=Model(id="faux", provider="faux"),
        tools=[subagent_tool],
        channel=parent_ch,
    )
    assert parent.channel is parent_ch
    # We don't actually run the agent here — the assertion above is what matters.
    # An execution test would require a Faux script that triggers subagent_tool.
```

- [ ] **Step 11.3: Run — expected PASS**

Run: `uv run pytest tests/hitl/test_subagent_channel_inheritance.py -v`

- [ ] **Step 11.4: Update `cubepi/hitl/__init__.py` to export testing helpers**

Add to the `__init__.py`:

```python
# Re-export testing helpers (commonly used in user test suites)
from cubepi.hitl.testing import NoopChannel, ScriptedChannel
__all__ += ["NoopChannel", "ScriptedChannel"]
```

- [ ] **Step 11.5: Lint + commit**

```bash
uv run ruff check cubepi/hitl/ tests/hitl/ && uv run ruff format cubepi/hitl/ tests/hitl/
git add cubepi/hitl/testing.py cubepi/hitl/__init__.py tests/hitl/test_subagent_channel_inheritance.py
git commit -m "feat(hitl): ScriptedChannel + NoopChannel + subagent inheritance test"
```

---

### Task 12: User-facing documentation

**Files:**
- Create: `website/docs/guides/hitl.md`
- Create: `website/docs/recipes/sandbox-confirm.md`
- Create: `website/docs/recipes/ask-user-form.md`
- Modify: `README.md` (architecture tree) — verify path

Per CLAUDE.md "feature without docs is not done." The guide explains motivation, when to use `ask_user` vs end-of-turn free text, when to use `ConfirmToolCallMiddleware`, channel implementations, suspend/resume protocol, cross-process recipe. Recipes are tight end-to-end examples.

- [ ] **Step 12.1: Write `website/docs/guides/hitl.md`**

```markdown
# Human-in-the-Loop (HITL)

cubepi ships a HITL channel for two recurring scenarios:

1. **Sandbox tool confirmation** — a dangerous tool needs human approve / deny / edit before running.
2. **Mid-run structured questions** — the agent needs a specific selection or form before proceeding.

The channel is one primitive with two implementations:
- `InMemoryChannel` — for CLI, notebook, tests.
- `CheckpointedChannel` — for web services where the agent process may die between question and answer; pairs with any `Checkpointer`.

## Quick start (in-process)

\`\`\`python
import asyncio
from cubepi.agent.agent import Agent
from cubepi.hitl import (
    ApproveAnswer, ConfirmToolCallMiddleware, InMemoryChannel, ask_user_tool,
)

channel = InMemoryChannel()

agent = Agent(
    provider=..., model=...,
    tools=[bash_tool, ask_user_tool(channel)],
    middleware=[ConfirmToolCallMiddleware(channel, require_confirm={"bash"})],
    channel=channel,
)

# Host coroutine renders pending requests and posts answers.
# For approve-kind requests, the answer is an ApproveAnswer; for ask-kind it's
# a dict[question.key, str | list[str]]; for confirm-kind it's a bool.
async def host():
    async for req in channel.subscribe():
        if req.payload.kind == "approve":
            user_decision = await my_ui.show_approve(req)   # returns ApproveAnswer
            await channel.answer(req.question_id, user_decision)
        elif req.payload.kind == "ask":
            answers = await my_ui.show_form(req.payload.questions)
            await channel.answer(req.question_id, answers)
        else:  # confirm
            await channel.answer(req.question_id, await my_ui.show_confirm(req))

# Run agent in parallel with host, then exit once the agent finishes.
async def main():
    host_task = asyncio.create_task(host())
    try:
        await agent.prompt("…")
    finally:
        host_task.cancel()

asyncio.run(main())
\`\`\`

## Cross-process (web service) flow

1. HTTP POST /chat starts `agent.prompt(...)`. Inside, channel.approve / channel.ask persists `pending_request` to the checkpointer and emits `HitlRequestEvent` on the SSE stream.
2. Frontend renders the pending; user clicks approve/deny/edit.
3. HTTP POST /respond calls `await agent.respond(question_id=..., answer=...)` which loads checkpoint, attaches the answer to the channel, and re-enters the loop. The previously-gated tool runs (or synthetic deny) and the conversation continues. Pending is cleared only after the tool_result is checkpointed.

If the user closes the tab without answering, the host calls `await agent.abort_pending(reason="user closed")` which closes the conversation with a synthetic deny + terminal `stop_reason="aborted"` assistant.

## When to use `ask_user` vs end of turn

| Goal | Use |
|------|-----|
| Free-text follow-up question to user | Just end the turn with the question as text; user's next message is the answer. |
| Structured selection (one of N) | `ask_user` tool with `options` and (optionally) `multi_select` |
| Confirm/edit a tool's args before run | `ConfirmToolCallMiddleware` or `ApprovalPolicyMiddleware` |

## Durable scope

Durable cross-process resume is supported at two safe suspension points:
1. `before_tool_call` approval gate (via `ApprovalPolicyMiddleware` / `ConfirmToolCallMiddleware`)
2. The `ask_user` tool body

Custom tools that mix HITL with other side effects are **same-process only** unless they pass `allow_inside_custom_tool=True` to `CheckpointedChannel` and accept the idempotency contract.
```

- [ ] **Step 12.2: Write `website/docs/recipes/sandbox-confirm.md`**

```markdown
# Recipe: Sandbox Confirm with `ApprovalPolicyMiddleware`

Use case: cubebox-style web service where every bash command has a rule engine that classifies it as auto-allow, hard-deny, or human-confirm.

\`\`\`python
from cubepi.hitl import (
    Approve, ApprovalPolicyMiddleware, AskUser, CheckpointedChannel, Deny,
)

def policy(ctx):
    cmd = ctx.args.get("cmd", "") if isinstance(ctx.args, dict) else ctx.args.cmd
    rule = command_rule_engine.classify(cmd)
    if rule.tier == "allow":   return Approve()
    if rule.tier == "block":   return Deny(reason=rule.reason)
    return AskUser(
        timeout_seconds=180,
        details={"rule": rule.matched_pattern, "impact": rule.impact},
    )

channel = CheckpointedChannel(checkpointer=cp, thread_id=thread_id)
agent = Agent(
    provider=..., model=..., tools=[bash_tool],
    middleware=[ApprovalPolicyMiddleware(channel, policy)],
    channel=channel, checkpointer=cp, thread_id=thread_id,
)
\`\`\`

`HitlRequest.timeout_seconds` is embedded in the emitted event so the frontend can render a countdown.

On timeout: middleware translates to `BeforeToolCallResult(block=True, deny_reason="approval_timeout")`. The model sees `tool_result.is_error=True` with `details.hitl.decision == "timed_out"` and naturally produces a follow-up turn explaining the timeout.
```

- [ ] **Step 12.3: Write `website/docs/recipes/ask-user-form.md`**

```markdown
# Recipe: Multi-question Form via `ask_user`

\`\`\`python
from cubepi.hitl import ask_user_tool, InMemoryChannel

channel = InMemoryChannel()
agent = Agent(
    provider=..., model=...,
    tools=[ask_user_tool(channel)],
    channel=channel,
)
\`\`\`

The model invokes `ask_user` like any other tool. Example parameters the model can pass:

\`\`\`json
{
  "questions": [
    {"key": "framework", "prompt": "Which framework?",
     "options": [
       {"label": "React", "value": "react"},
       {"label": "Vue", "value": "vue"},
       {"label": "Other", "value": "other", "allow_input": true}
     ]},
    {"key": "features", "prompt": "Which features?",
     "multi_select": true,
     "options": [
       {"label": "Auth", "value": "auth"},
       {"label": "Payments", "value": "payments"}
     ]}
  ]
}
\`\`\`

Answer shape: `{"framework": "react", "features": ["auth", "payments"]}` — or for `Other` with `allow_input`, the value is the free-text string the user typed.
```

- [ ] **Step 12.4: Update README architecture tree**

In `README.md`, find the "Architecture" tree section that lists `cubepi/` modules and add `hitl/` (Human-in-the-Loop channel, middlewares, `ask_user` tool).

- [ ] **Step 12.5: Commit docs**

```bash
git add website/docs/guides/hitl.md website/docs/recipes/sandbox-confirm.md website/docs/recipes/ask-user-form.md README.md
git commit -m "docs(hitl): user-facing guide + sandbox-confirm + ask-user-form recipes + arch tree"
```

---

### Task 13: Final verification

- [ ] **Step 13.1: Full test suite**

Run: `uv run pytest tests/ -q`
Expected: all tests pass (including E2E tests gated by markers).

- [ ] **Step 13.2: Lint**

Run: `uv run ruff check cubepi/ tests/ && uv run ruff format --check cubepi/ tests/`
Expected: no issues.

- [ ] **Step 13.3: Sanity sweep**

```bash
git log --oneline 2026-05-28-hitl-channel  # confirm phase commits exist
git diff main...2026-05-28-hitl-channel --stat  # confirm scope reasonable
```

- [ ] **Step 13.4: Push branch and open PR**

```bash
git push -u origin 2026-05-28-hitl-channel
gh pr create --title "feat(hitl): Human-in-the-Loop channel mechanism" --body "$(cat <<'EOF'
## Summary
- New `cubepi.hitl` module with `HitlChannel` Protocol + `InMemoryChannel` / `CheckpointedChannel` implementations
- `ask_user` built-in tool + `ApprovalPolicyMiddleware` + `ConfirmToolCallMiddleware`
- Durable cross-process suspend/resume via Checkpointer (Memory/SQLite/Postgres/MySQL — Postgres/MySQL schema v1→v2 with migration helper)
- New events: `HitlRequestEvent`, `HitlAnswerEvent`, `AgentSuspendedEvent`, `AgentAbortedEvent`
- Agent: `channel=` kwarg, `detach()`, `respond()`, `abort_pending()`, `in_flight_hitl_request`, `load_pending_hitl_request()`
- Lazy OTel `hitl.{kind}` spans
- Full docs under `website/docs/guides/hitl.md` + two recipes

## Test plan
- [ ] All HITL unit tests green
- [ ] Postgres + MySQL E2E pending_request tests green against test servers
- [ ] cache-prefix tests pass byte-exactly across Memory and SQLite backends
- [ ] No regressions in existing test suite

Spec: `dev/specs/2026-05-28-hitl-channel.md`
Plan: `dev/plans/2026-05-28-hitl-channel.md`
EOF
)"
```

Then drive the PR codex review loop per CLAUDE.md §5 (poll ~2 min, fix, reply `@codex`, repeat until clean).

---

## Self-Review

**Spec coverage** — every numbered section of the spec maps to one or more tasks:

| Spec § | Task |
|---|---|
| §2 + §2.1 (design philosophy + durable scope) | Tasks 9.1-9.4 (`allow_inside_custom_tool` guard) + docs §12.1 |
| §3.1 ask_user | Task 6.7-6.10 |
| §3.2 middlewares | Task 6.1-6.6 |
| §4.1 data types | Task 1.1-1.4 |
| §4.2 channel Protocol + exceptions | Task 1.5-1.8 + 2.1-2.4 |
| §4.3 InMemoryChannel | Task 2 |
| §4.4 CheckpointedChannel | Task 9.1-9.3 |
| §4.5 detach | Task 9.5 |
| §5.1 per-backend storage | Task 7 + 8 |
| §5.2 respond + ordering + abort_pending + pending properties | Task 9.5-9.10 |
| §5.3 resume code path | Task 9.6 |
| §6.1.1 Agent channel wiring | Task 4 |
| §6.2 loop.py changes | Task 3.11 |
| §6.3.4 compose_middleware redesign | Task 3.7 |
| §6.3.5 sequential HITL | Task 6.7-6.10 (`execution_mode="sequential"`) + Task 9.4 |
| §6.4 new events | Task 5 |
| §6.5 trace spans | Task 10 |
| §7 error table | Implemented across Tasks 1-9; each error path has a test |
| §8 subagents | Task 11 |
| §9 testing | Tasks 1-11 all include their unit tests; Task 8.5 E2E |
| §10 prior art | Already in spec, no implementation needed |
| §11 out-of-scope | No implementation |
| §12 docs | Task 12 |
| §13 build sequence | This entire plan |

**Placeholder scan** — no TBD/TODO/"implement later"/"add error handling"; all code blocks are concrete.

**Type/method consistency** — naming is consistent: `agent.channel`, `agent.respond(question_id=, answer=)`, `agent.detach()`, `agent.abort_pending(reason=)`, `agent.in_flight_hitl_request`, `agent.load_pending_hitl_request()`; channel verbs `confirm`/`approve`/`ask`; HITL events all share `Hitl*Event` naming.

Known fragile points (call out for the implementer):
- **FauxProvider API** — the plan uses the *real* API: `FauxProvider()` (no kwargs), `provider.set_responses([...])`, `faux_text(str)`, `faux_tool_call(name, args, *, id=...)`, `faux_assistant_message(content, *, stop_reason="stop")`. Cache-prefix tests capture provider input via `provider.subscribe_request(lambda payload, model: …)` — payload contains `messages`, `model`, `system_prompt`. No new helpers need to be added to FauxProvider for any test in this plan.
- **Existing checkpointer session/factory patterns** — Task 8 Postgres/MySQL implementation needs to match the existing session pattern in those files; the plan shows the SQL intent but the implementer should read the file first and adapt the Python.
- **Loop's `_run_loop` body** — Task 3.13 wraps it in a try/except; verify that catching `HitlDetached` at this layer doesn't swallow detach raised by deeply-nested coroutines unexpectedly. Spec §6.2 covers the selective-catch invariant.
