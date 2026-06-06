"""Task 26 tests — outcome enumeration + mark_run_complete dispatch.

The contract under test: ``Agent.prompt()`` writes the completion marker
(via ``Checkpointer.mark_run_complete``) IFF the loop reached a terminal
``AgentEndEvent`` with a non-error/aborted stop_reason. Every other exit
(provider error, HITL detach, HITL abort, propagating cancel) must NOT
mark the run as complete, leaving the row resumable.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.checkpointer.exceptions import (
    CompletionMarkerFailedError,
    RunNotClaimedError,
)
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl import AskUser
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.hitl.middleware import ApprovalPolicyMiddleware
from cubepi.providers.base import AssistantMessage, TextContent
from cubepi.providers.faux import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)


class _Params(BaseModel):
    cmd: str


def _bash_tool() -> AgentTool:
    async def execute(call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"ran {args.cmd}")])

    return AgentTool(
        name="bash",
        description="run a shell command",
        parameters=_Params,
        execute=execute,
        execution_mode="sequential",
    )


def _ok_faux() -> FauxProvider:
    p = FauxProvider(provider_id="faux")
    p.set_responses(
        [AssistantMessage(content=[TextContent(text="ok")], stop_reason="end_turn")]
    )
    return p


# ---------------------------------------------------------------------------
# Clean success path → mark_run_complete IS called.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_success_marks_complete():
    cp = MemoryCheckpointer()
    a = Agent(model=_ok_faux().model("faux-m"), checkpointer=cp, thread_id="t")
    got = await a.prompt("hi", run_id="R1")
    assert got == "R1"
    assert a.state.last_outcome == "complete"
    assert a.state.active_run_id is None
    # mark_run_complete actually ran — row has completion_seq set.
    state = cp._runs["t"]["R1"]
    assert state.completed_at is not None
    assert state.completion_seq is not None


# ---------------------------------------------------------------------------
# Provider error → loop emits stop_reason="error" → outcome="abandoned" → no mark.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_error_does_not_mark():
    """FauxProvider with no queued responses surfaces stop_reason=error."""
    cp = MemoryCheckpointer()
    provider = FauxProvider(provider_id="faux")
    # Empty response queue → faux returns AssistantMessage(stop_reason="error").
    a = Agent(model=provider.model("faux-m"), checkpointer=cp, thread_id="t")
    got = await a.prompt("hi", run_id="R1")  # _handle_run_failure swallows
    assert got == "R1"
    assert a.state.last_outcome == "abandoned"
    # Run row claimed but NOT completed — still resumable.
    state = cp._runs["t"]["R1"]
    assert state.completed_at is None


# ---------------------------------------------------------------------------
# HITL detach → outcome="suspended" → no mark.
# ---------------------------------------------------------------------------


def _two_turn_bash_responses():
    return [
        faux_assistant_message(
            [faux_text("ok"), faux_tool_call("bash", {"cmd": "ls"}, id="tc-1")],
            stop_reason="tool_use",
        ),
        faux_assistant_message("done"),
    ]


@pytest.mark.asyncio
async def test_hitl_detached_outcome_suspended_no_mark():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(_two_turn_bash_responses())
    a = Agent(
        model=provider.model("faux-m"),
        tools=[_bash_tool()],
        middleware=[ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())],
        channel=ch,
        checkpointer=cp,
        thread_id="t",
    )

    task = asyncio.create_task(a.prompt("hi", run_id="R1"))
    # Wait for pending HITL request.
    for _ in range(200):
        if ch.pending is not None:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("agent did not suspend on HITL")

    await a.detach()
    await task

    assert a.state.last_outcome == "suspended"
    # Row claimed, NOT completed.
    state = cp._runs["t"]["R1"]
    assert state.completed_at is None
    # active_run_id cleared on clean else: branch (suspended is NOT an error).
    assert a.state.active_run_id is None


# ---------------------------------------------------------------------------
# HITL abort_pending → outcome="abandoned" → no mark.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hitl_aborted_via_abort_pending_does_not_mark():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(_two_turn_bash_responses())
    a = Agent(
        model=provider.model("faux-m"),
        tools=[_bash_tool()],
        middleware=[ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())],
        channel=ch,
        checkpointer=cp,
        thread_id="t",
    )

    task = asyncio.create_task(a.prompt("hi", run_id="R1"))
    for _ in range(200):
        if ch.pending is not None:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("agent did not suspend on HITL")

    # abort_pending: Phase 1 sets the agent signal — _BaseChannel._await_answer
    # raises HitlAborted, which surfaces from the loop as outcome="abandoned".
    await a.abort_pending(reason="user closed")
    await task

    assert a.state.last_outcome == "abandoned"
    # Row claimed, NOT completed.
    state = cp._runs["t"]["R1"]
    assert state.completed_at is None


# ---------------------------------------------------------------------------
# Propagating asyncio.CancelledError → outcome stays None → no mark.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propagating_cancel_does_not_mark():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(_two_turn_bash_responses())
    a = Agent(
        model=provider.model("faux-m"),
        tools=[_bash_tool()],
        middleware=[ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())],
        channel=ch,
        checkpointer=cp,
        thread_id="t",
    )

    task = asyncio.create_task(a.prompt("hi", run_id="R1"))
    for _ in range(200):
        if ch.pending is not None:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("agent did not suspend on HITL")

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Cancel propagates through prompt()'s `except BaseException: raise` —
    # active_run_id stays SET; no mark_run_complete call.
    assert a.state.active_run_id == "R1"
    state = cp._runs["t"]["R1"]
    assert state.completed_at is None
    # last_outcome may remain None (no loop terminal path fired).
    assert a.state.last_outcome != "complete"


# ---------------------------------------------------------------------------
# CompletionMarkerFailedError path: when mark_run_complete raises, the wrapped
# error propagates UP through prompt() carrying the generated run_id, and
# active_run_id stays SET so the host can introspect it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_marker_failed_carries_run_id_when_generated():
    cp = MemoryCheckpointer()

    boom_raised: dict[str, bool] = {}

    async def _boom_mark(thread_id: str, run_id: str) -> None:
        boom_raised["called"] = True
        boom_raised["run_id"] = run_id
        raise RunNotClaimedError("simulated DB hiccup")

    # Monkey-patch mark_run_complete on this checkpointer instance.
    cp.mark_run_complete = _boom_mark  # type: ignore[method-assign]

    a = Agent(model=_ok_faux().model("faux-m"), checkpointer=cp, thread_id="t")
    # run_id=None → cubepi generates one. Test that the generated value is
    # carried on the raised CompletionMarkerFailedError.
    with pytest.raises(CompletionMarkerFailedError) as excinfo:
        await a.prompt("hi")

    assert boom_raised.get("called") is True
    err = excinfo.value
    assert err.thread_id == "t"
    assert err.run_id == boom_raised["run_id"]
    # active_run_id stays SET — the clear line is unreachable past the raise.
    assert a.state.active_run_id == err.run_id
    # Outcome was "complete" before dispatch failed.
    assert a.state.last_outcome == "complete"
