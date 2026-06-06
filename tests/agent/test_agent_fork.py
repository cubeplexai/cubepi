import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import AssistantMessage, TextContent
from cubepi.providers.faux import FauxProvider


def _ok_faux() -> FauxProvider:
    p = FauxProvider()
    p.set_responses(
        [AssistantMessage(content=[TextContent(text="ok")], stop_reason="end_turn")]
    )
    return p


@pytest.mark.asyncio
async def test_agent_fork_delegates_to_checkpointer():
    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await a.prompt("hello", run_id="R1")  # creates src + R1 marker
    await a.fork("src", "dst", after_run_id="R1")
    loaded = await cp.load("dst")
    assert loaded.parent_thread_id == "src"


@pytest.mark.asyncio
async def test_agent_fork_no_checkpointer_raises():
    a = Agent(model=_ok_faux().model("faux-model"))
    with pytest.raises(RuntimeError, match="checkpointer"):
        await a.fork("src", "dst", after_run_id="R1")


@pytest.mark.asyncio
async def test_agent_fork_does_not_mutate_self_thread_id():
    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await a.prompt("hello", run_id="R1")
    await a.fork("src", "dst", after_run_id="R1")
    assert a.thread_id == "src"
