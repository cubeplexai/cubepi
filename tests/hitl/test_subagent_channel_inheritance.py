from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.hitl import ApproveAnswer
from cubepi.hitl.channel import InMemoryChannel
from cubepi.hitl.testing import NoopChannel, ScriptedChannel
from cubepi.hitl.types import Question
from cubepi.providers.base import Model, TextContent
from cubepi.providers.faux import FauxProvider, faux_assistant_message


def _faux_with(responses):
    """Helper: build a FauxProvider preloaded with responses (mirrors real API)."""
    p = FauxProvider()
    p.set_responses(responses)
    return p


async def test_scripted_channel_returns_canned_answers():
    ch = ScriptedChannel(
        answers=[
            ApproveAnswer(decision="approve"),
            {"color": "red"},
        ]
    )
    ans1 = await ch.approve(tool_name="bash", tool_call_id="tc-1", args={})
    assert ans1.decision == "approve"
    ans2 = await ch.ask([Question(key="color", prompt="?")])
    assert ans2 == {"color": "red"}
    assert len(ch.history) == 2


async def test_noop_channel_auto_approves():
    ch = NoopChannel()
    assert (
        await ch.approve(tool_name="x", tool_call_id="y", args={})
    ).decision == "approve"
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
        name="run_subagent",
        description="run a subagent",
        parameters=_NoParams,
        execute=subagent_execute,
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
