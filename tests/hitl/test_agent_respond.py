from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl import (
    ApproveAnswer,
    AskUser,
    HitlNoPendingRequest,
    HitlStaleAnswer,
)
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.hitl.middleware import ApprovalPolicyMiddleware
from cubepi.providers.base import TextContent
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


def _two_turn_bash_responses():
    """Turn 1 calls bash; turn 2 (post tool-result) ends."""
    return [
        faux_assistant_message(
            [faux_text("ok"), faux_tool_call("bash", {"cmd": "ls"}, id="tc-1")],
            stop_reason="tool_use",
        ),
        faux_assistant_message("done"),
    ]


def _faux_with(responses) -> FauxProvider:
    p = FauxProvider(provider_id="faux")
    p.set_responses(responses)
    return p


async def test_respond_completes_a_suspended_run():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(_two_turn_bash_responses())
    agent = Agent(
        model=provider.model("faux"),
        tools=[_bash_tool()],
        middleware=[
            ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser()),
        ],
        channel=ch,
        checkpointer=cp,
        thread_id="t-1",
    )

    # Start the agent — it will suspend on channel.approve.
    async def run():
        await agent.prompt("hi")

    task = asyncio.create_task(run())

    # Wait until pending appears.
    for _ in range(200):
        if ch.pending is not None:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("agent did not suspend on HITL")

    # Detach so the run() returns; respond() will pick up.
    await agent.detach()
    await task  # run() returns cleanly

    # Now respond with approve.
    await agent.respond(question_id="tc-1", answer=ApproveAnswer(decision="approve"))

    msgs = agent.state.messages
    assert msgs[-1].content[0].text == "done"


async def test_respond_stale_answer():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")
    provider = _faux_with([faux_assistant_message("")])
    agent = Agent(
        model=provider.model("faux"),
        channel=ch,
        checkpointer=cp,
        thread_id="t-1",
    )
    # Manually persist a pending then try the wrong qid.
    from cubepi.hitl.types import ApproveRequest, HitlRequest

    await cp.save_pending_request(
        "t-1",
        HitlRequest(
            question_id="tc-real",
            thread_id="t-1",
            payload=ApproveRequest(tool_name="bash", tool_call_id="tc-real", args={}),
            created_at=0.0,
        ),
    )
    with pytest.raises(HitlStaleAnswer):
        await agent.respond(
            question_id="tc-wrong", answer=ApproveAnswer(decision="approve")
        )


async def test_respond_no_pending():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1")
    provider = _faux_with([faux_assistant_message("")])
    agent = Agent(
        model=provider.model("faux"),
        channel=ch,
        checkpointer=cp,
        thread_id="t-1",
    )
    with pytest.raises(HitlNoPendingRequest):
        await agent.respond(answer=ApproveAnswer(decision="approve"))
