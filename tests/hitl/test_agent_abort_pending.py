from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl import AskUser
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


async def test_abort_pending_closes_conversation():
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1", run_id="R1")
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [
            faux_assistant_message(
                [faux_text("ok"), faux_tool_call("bash", {"cmd": "ls"}, id="tc-1")],
                stop_reason="tool_use",
            ),
        ]
    )
    agent = Agent(
        model=provider.model("faux"),
        tools=[_bash_tool()],
        middleware=[ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())],
        channel=ch,
        checkpointer=cp,
        thread_id="t-1",
    )
    task = asyncio.create_task(agent.prompt("hi", run_id="R1"))
    for _ in range(200):
        if ch.pending is not None:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("agent did not suspend on HITL")
    await agent.detach()
    await task

    await agent.abort_pending(reason="user closed tab")

    msgs = agent.state.messages
    # Should end with a synthetic deny tool_result and a stop_reason=aborted
    # assistant.
    assert msgs[-2].is_error is True
    assert "user closed tab" in msgs[-2].content[0].text
    assert msgs[-1].stop_reason == "aborted"
    # pending is cleared
    assert await cp.load_pending_request("t-1") is None
    # Regression for codex R2 P1: synthetic deny + terminal abort must
    # carry the originating assistant's run_id so a later fork query
    # (which treats run_id IS NULL as legacy / always-copy) cannot copy
    # these aborted-run artifacts as if they were pre-v4 history.
    assert msgs[-2].run_id == "R1"
    assert msgs[-1].run_id == "R1"
