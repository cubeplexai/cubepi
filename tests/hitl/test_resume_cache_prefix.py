from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.hitl import ApproveAnswer, AskUser
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.hitl.middleware import ApprovalPolicyMiddleware
from cubepi.providers.base import Model, TextContent
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
    return [
        faux_assistant_message(
            [faux_text("ok"), faux_tool_call("bash", {"cmd": "ls"}, id="tc-1")],
            stop_reason="tool_use",
        ),
        faux_assistant_message("done"),
    ]


async def _suspend_resume_and_capture(checkpointer):
    ch = CheckpointedChannel(checkpointer=checkpointer, thread_id="t-1")
    provider = FauxProvider()
    provider.set_responses(_two_turn_bash_responses())

    captured: list[dict] = []
    provider.subscribe_request(lambda payload, model: captured.append(payload))

    agent = Agent(
        provider=provider,
        model=Model(id="faux", provider="faux"),
        tools=[_bash_tool()],
        middleware=[ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())],
        channel=ch,
        checkpointer=checkpointer,
        thread_id="t-1",
    )
    task = asyncio.create_task(agent.prompt("hi"))
    for _ in range(200):
        if ch.pending is not None:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("agent did not suspend on HITL")

    # First-turn payload bytes (the one the model already saw before suspend)
    pre_messages = list(captured[0]["messages"])

    await agent.detach()
    await task

    await agent.respond(question_id="tc-1", answer=ApproveAnswer(decision="approve"))

    # Second turn = post-resume model call. The first len(pre_messages) entries
    # must be byte-identical to the first turn for prompt-cache to hit.
    second_turn_messages = captured[1]["messages"]
    return pre_messages, second_turn_messages[: len(pre_messages)]


async def test_resume_preserves_cache_prefix_memory():
    pre, post = await _suspend_resume_and_capture(MemoryCheckpointer())
    assert pre == post


async def test_resume_preserves_cache_prefix_sqlite(tmp_path):
    db = tmp_path / "x.db"
    async with SQLiteCheckpointer(str(db)) as cp:
        pre, post = await _suspend_resume_and_capture(cp)
        assert pre == post
