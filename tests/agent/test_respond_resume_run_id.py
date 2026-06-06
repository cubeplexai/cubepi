"""Task 28: respond() resumes run_id via load_pending; never reclaims.

The new respond() body uses the unified ``load_pending`` Protocol method,
which returns ``(HitlRequest, run_id | None)``. On clean resume it
propagates the recovered run_id into AgentState.active_run_id, then runs
the resume loop. After the loop returns successfully it dispatches the
outcome (mark_run_complete on "complete") and clears active_run_id.

A legacy guard skips dispatch when the recovered run_id is None (pending
saved by an older save_pending_request call that didn't carry a run_id).
"""

from __future__ import annotations

import asyncio

import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl.ask_user import ask_user_tool
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
)
from cubepi.providers.faux import FauxProvider


def _pause_then_finish_provider() -> FauxProvider:
    """Two-turn script: ask_user (pause) → final assistant on resume."""
    p = FauxProvider()
    p.set_responses(
        [
            AssistantMessage(
                content=[
                    ToolCall(
                        id="ask-1",
                        name="ask_user",
                        arguments={"questions": [{"key": "ans", "prompt": "?"}]},
                    )
                ],
                stop_reason="tool_use",
            ),
            AssistantMessage(
                content=[TextContent(text="done")],
                stop_reason="end_turn",
            ),
        ]
    )
    return p


async def _drive_pause(
    cp: MemoryCheckpointer,
    model_factory,
) -> None:
    """Build an Agent, kick a prompt(), wait until pending is persisted."""
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
    # Wait for pending to be persisted.
    for _ in range(500):
        if await cp.load_pending("t") is not None:
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover — defensive
        task.cancel()
        raise AssertionError("pending never appeared")
    await a.detach()
    await task


@pytest.mark.asyncio
async def test_respond_clears_active_run_id_on_clean_resume() -> None:
    cp = MemoryCheckpointer()
    p = _pause_then_finish_provider()

    def _model():
        return p.model("faux-model")

    await _drive_pause(cp, _model)

    # Fresh Agent — emulates a new host process picking up the run.
    ch2 = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool2 = ask_user_tool(ch2)
    a2 = Agent(
        model=_model(),
        tools=[tool2],
        checkpointer=cp,
        thread_id="t",
        channel=ch2,
    )
    loaded = await cp.load_pending("t")
    assert loaded is not None
    qid = loaded[0].question_id
    await a2.respond(question_id=qid, answer="yes")

    assert a2.state.active_run_id is None


@pytest.mark.asyncio
async def test_respond_resume_writes_marker() -> None:
    cp = MemoryCheckpointer()
    p = _pause_then_finish_provider()

    def _model():
        return p.model("faux-model")

    await _drive_pause(cp, _model)

    # Marker not yet written — prompt() suspended, not completed.
    assert cp._runs["t"]["R1"].completed_at is None

    ch2 = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool2 = ask_user_tool(ch2)
    a2 = Agent(
        model=_model(),
        tools=[tool2],
        checkpointer=cp,
        thread_id="t",
        channel=ch2,
    )
    loaded = await cp.load_pending("t")
    assert loaded is not None
    qid = loaded[0].question_id
    await a2.respond(question_id=qid, answer="yes")

    # Clean resume → outcome "complete" → mark_run_complete fired.
    assert cp._runs["t"]["R1"].completed_at is not None


@pytest.mark.asyncio
async def test_respond_resume_with_legacy_pending_does_not_crash() -> None:
    """Legacy guard: pending persisted without a run_id (run_id=None)
    must NOT crash respond() and must NOT trigger dispatch."""
    cp = MemoryCheckpointer()
    p = _pause_then_finish_provider()

    def _model():
        return p.model("faux-model")

    await _drive_pause(cp, _model)

    # Forge legacy: re-save the pending request with run_id=None so the
    # checkpointer "forgets" the run_id binding.
    loaded = await cp.load_pending("t")
    assert loaded is not None
    pending_req, _ = loaded
    await cp.save_pending_request("t", pending_req, run_id=None)

    ch2 = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool2 = ask_user_tool(ch2)
    a2 = Agent(
        model=_model(),
        tools=[tool2],
        checkpointer=cp,
        thread_id="t",
        channel=ch2,
    )
    qid = pending_req.question_id
    # Must not raise.
    await a2.respond(question_id=qid, answer="yes")

    # Legacy guard skipped dispatch — marker remains unwritten.
    assert cp._runs["t"]["R1"].completed_at is None
