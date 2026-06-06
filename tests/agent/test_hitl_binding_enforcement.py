import pytest

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool
from cubepi.hitl.binding import HitlBinding
from cubepi.providers.base import AssistantMessage, TextContent
from cubepi.providers.faux import FauxProvider
from pydantic import BaseModel


def _ok_faux() -> FauxProvider:
    p = FauxProvider()
    p.set_responses(
        [AssistantMessage(content=[TextContent(text="ok")], stop_reason="end_turn")]
    )
    return p


class _NoArgs(BaseModel):
    pass


async def _noop(tool_call_id, args, *, signal=None, on_update=None):
    raise NotImplementedError


def _tool_with_hitl(binding):
    return AgentTool(
        name="ask_user",
        description="d",
        parameters=_NoArgs,
        execute=_noop,
        hitl=binding,
    )


def _agent(tools=None, middleware=None):
    return Agent(
        model=_ok_faux().model("faux-model"),
        tools=tools or [],
        middleware=middleware or [],
    )


@pytest.mark.asyncio
async def test_checkpointed_hitl_with_none_run_id_raises():
    tool = _tool_with_hitl(HitlBinding(checkpointed=True, run_id=None))
    a = _agent(tools=[tool])
    with pytest.raises(ValueError, match="no run_id bound"):
        await a.prompt("hi", run_id="R1")


@pytest.mark.asyncio
async def test_checkpointed_hitl_requires_explicit_run_id():
    tool = _tool_with_hitl(HitlBinding(checkpointed=True, run_id="R1"))
    a = _agent(tools=[tool])
    with pytest.raises(ValueError, match="generate-mode rejected"):
        await a.prompt("hi", run_id=None)


@pytest.mark.asyncio
async def test_checkpointed_hitl_run_id_mismatch_raises():
    tool = _tool_with_hitl(HitlBinding(checkpointed=True, run_id="R1"))
    a = _agent(tools=[tool])
    with pytest.raises(ValueError, match="does not match"):
        await a.prompt("hi", run_id="R2")


@pytest.mark.asyncio
async def test_checkpointed_hitl_run_id_match_succeeds():
    tool = _tool_with_hitl(HitlBinding(checkpointed=True, run_id="R1"))
    a = _agent(tools=[tool])
    got = await a.prompt("hi", run_id="R1")
    assert got == "R1"


@pytest.mark.asyncio
async def test_in_memory_hitl_no_constraint():
    tool = _tool_with_hitl(HitlBinding(checkpointed=False, run_id=None))
    a = _agent(tools=[tool])
    got = await a.prompt("hi")  # generate-mode allowed
    assert isinstance(got, str)
