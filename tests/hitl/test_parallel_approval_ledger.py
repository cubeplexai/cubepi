from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl import ApproveAnswer, AskUser
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.hitl.middleware import ApprovalPolicyMiddleware
from cubepi.providers.base import TextContent, ToolResultMessage
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_tool_call


class _NoParams(BaseModel):
    pass


def _make_parallel_tool(name: str, executions: list[str]) -> AgentTool:
    async def execute(call_id, args, *, signal=None, on_update=None):
        executions.append(call_id)
        return AgentToolResult(content=[TextContent(text=f"{name} ran")])

    return AgentTool(
        name=name,
        description=name,
        parameters=_NoParams,
        execute=execute,
        execution_mode="parallel",
    )


def _provider() -> FauxProvider:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("solar", {}, id="tc-a"),
                    faux_tool_call("semiconductor", {}, id="tc-b"),
                ],
                stop_reason="tool_use",
            ),
            faux_assistant_message("done"),
        ]
    )
    return provider


def _agent(
    *,
    cp: MemoryCheckpointer,
    provider: FauxProvider,
    executions: list[str],
) -> Agent:
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-root-a", run_id="R1")
    return Agent(
        model=provider.model("faux"),
        tools=[
            _make_parallel_tool("solar", executions),
            _make_parallel_tool("semiconductor", executions),
        ],
        middleware=[ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())],
        channel=ch,
        checkpointer=cp,
        thread_id="t-root-a",
    )


async def _wait_for_pending(cp: MemoryCheckpointer, expected_qid: str) -> None:
    for _ in range(200):
        loaded = await cp.load_pending("t-root-a")
        if loaded is not None and loaded[0].question_id == expected_qid:
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"pending request did not become {expected_qid!r}")


async def test_parallel_approval_answers_are_replayed_until_batch_can_execute():
    cp = MemoryCheckpointer()
    provider = _provider()
    executions: list[str] = []

    initial_agent = _agent(cp=cp, provider=provider, executions=executions)
    prompt_task = asyncio.create_task(initial_agent.prompt("run both", run_id="R1"))
    await _wait_for_pending(cp, "tc-a")
    await initial_agent.detach()
    await prompt_task

    first_resume = _agent(cp=cp, provider=provider, executions=executions)
    first_resume_task = asyncio.create_task(
        first_resume.respond(
            question_id="tc-a", answer=ApproveAnswer(decision="approve")
        )
    )
    await _wait_for_pending(cp, "tc-b")
    await first_resume.detach()
    await first_resume_task

    assert executions == []
    loaded = await cp.load_pending("t-root-a")
    assert loaded is not None
    assert loaded[0].question_id == "tc-b"

    second_resume = _agent(cp=cp, provider=provider, executions=executions)
    await second_resume.respond(
        question_id="tc-b", answer=ApproveAnswer(decision="approve")
    )

    assert sorted(executions) == ["tc-a", "tc-b"]
    assert await cp.load_pending("t-root-a") is None
    assert await cp.load_hitl_answer("t-root-a", "tc-a", run_id="R1") is None
    assert await cp.load_hitl_answer("t-root-a", "tc-b", run_id="R1") is None
    tool_results = [
        msg
        for msg in second_resume.state.messages
        if isinstance(msg, ToolResultMessage)
    ]
    assert [msg.tool_call_id for msg in tool_results] == ["tc-a", "tc-b"]
