"""Task 39: HITL pause + resume + fork chain E2E.

Exercises the full host-facing flow:

1. Provider script: turn 1 = ``ask_user`` tool_call (pauses), turn 2 = final
   assistant text.
2. Driver: ``prompt()`` pauses for HITL → fresh Agent ``respond()`` resumes →
   completion marker for R1 is written.
3. Fork on a third Agent with ``after_run_id="R1"`` → new thread carries R1's
   messages and the fork metadata.

Memory backend is sufficient — cross-backend coverage is provided by Task 38.
"""

from __future__ import annotations

import asyncio

import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl.ask_user import ask_user_tool
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall
from cubepi.providers.faux import FauxProvider


def _ask_then_finish_provider() -> FauxProvider:
    """Turn 1 pauses via ask_user; turn 2 emits the final assistant text."""
    p = FauxProvider()
    p.set_responses(
        [
            AssistantMessage(
                content=[
                    ToolCall(
                        id="tc-1",
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


@pytest.mark.asyncio
async def test_fork_after_hitl_resume():
    """prompt pauses for HITL → respond resumes → marker written →
    fork succeeds and copies the resumed run."""
    cp = MemoryCheckpointer()
    p = _ask_then_finish_provider()

    # ---- Phase 1: drive the pause. ----
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool = ask_user_tool(ch)
    a = Agent(
        model=p.model("faux-model"),
        tools=[tool],
        checkpointer=cp,
        thread_id="t",
        channel=ch,
    )
    task = asyncio.create_task(a.prompt("hello", run_id="R1"))
    # Wait until pending is persisted.
    for _ in range(500):
        if (await cp.load_pending("t")) is not None:
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover — defensive
        task.cancel()
        raise AssertionError("pending never appeared")
    await a.detach()
    await task

    # Marker not yet written — prompt() suspended, not completed.
    assert cp._runs["t"]["R1"].completed_at is None

    # ---- Phase 2: resume on a fresh Agent. ----
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
    assert pending is not None
    pending_req, _recovered_run_id = pending
    await a2.respond(question_id=pending_req.question_id, answer="yes")

    # Clean resume → outcome "complete" → mark_run_complete fired.
    assert cp._runs["t"]["R1"].completed_at is not None

    # ---- Phase 3: fork on a fresh Agent ----
    a3 = Agent(
        model=p.model("faux-model"),  # not actually invoked
        checkpointer=cp,
        thread_id="t",
    )
    await a3.fork("t", "fork_t", after_run_id="R1", metadata={"reason": "branch"})

    loaded = await cp.load("fork_t")
    assert loaded is not None
    assert loaded.parent_thread_id == "t"
    assert loaded.extra["fork"] == {"reason": "branch"}
    # The forked thread contains the full resumed conversation:
    # user message + assistant ask_user + tool_result + final assistant.
    run_ids = {m.run_id for m in loaded.messages if m.run_id}
    assert "R1" in run_ids
    # Sanity: we copied more than just the user prompt.
    assert len(loaded.messages) >= 3
