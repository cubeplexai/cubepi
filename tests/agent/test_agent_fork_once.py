import asyncio

import pytest
from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, ForkOnceResult
from cubepi.checkpointer.exceptions import CheckpointerError
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl.binding import HitlBinding
from cubepi.middleware.base import Middleware
from cubepi.providers.base import AssistantMessage, TextContent
from cubepi.providers.faux import FauxProvider


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


@pytest.mark.asyncio
async def test_fork_once_simple_text_returns_final_text():
    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await a.prompt("hello", run_id="R1")
    before = await cp.load("src")
    before_msgs = list(before.messages)
    # New faux for the fork_once probe.
    a2 = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    result = await a2.fork_once("src", "what next?", after_run_id="R1")
    assert isinstance(result, ForkOnceResult)
    assert result.text == "ok"
    after = await cp.load("src")
    assert list(after.messages) == before_msgs


@pytest.mark.asyncio
async def test_fork_once_no_checkpointer_raises():
    a = Agent(model=_ok_faux().model("faux-model"))
    with pytest.raises(RuntimeError, match="checkpointer"):
        await a.fork_once("src", "msg", after_run_id="R1")


@pytest.mark.asyncio
async def test_fork_once_v3_only_checkpointer_raises_CheckpointerError():
    class _V3Only:
        async def load(self, thread_id):
            return None

        async def append(self, thread_id, msgs):
            pass

        async def save_extra(self, thread_id, extra):
            pass

        async def save_pending_request(self, thread_id, req, *, run_id=None):
            pass

        async def load_pending_request(self, thread_id):
            return None

    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=_V3Only(),
        thread_id="src",
    )
    with pytest.raises(CheckpointerError):
        await a.fork_once("src", "msg", after_run_id="R1")


@pytest.mark.asyncio
async def test_fork_once_rejects_checkpointed_hitl_tool():
    cp = MemoryCheckpointer()
    hitl_tool = AgentTool(
        name="ask_user",
        description="ask user",
        parameters=_NoArgs,
        execute=_noop,
        hitl=HitlBinding(checkpointed=True, run_id="R1"),
    )
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
        tools=[hitl_tool],
    )
    # Seed a completed run so snapshot would succeed if we reached it.
    await cp.claim_run("src", "R1")
    await cp.mark_run_complete("src", "R1")
    with pytest.raises(RuntimeError, match="does not support HITL"):
        await a.fork_once("src", "msg", after_run_id="R1")


@pytest.mark.asyncio
async def test_fork_once_rejects_in_memory_hitl_tool():
    cp = MemoryCheckpointer()
    hitl_tool = AgentTool(
        name="ask_user",
        description="ask user",
        parameters=_NoArgs,
        execute=_noop,
        hitl=HitlBinding(checkpointed=False, run_id=None),
    )
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
        tools=[hitl_tool],
    )
    await cp.claim_run("src", "R1")
    await cp.mark_run_complete("src", "R1")
    with pytest.raises(RuntimeError, match="does not support HITL"):
        await a.fork_once("src", "msg", after_run_id="R1")


@pytest.mark.asyncio
async def test_fork_once_rejects_hitl_middleware():
    cp = MemoryCheckpointer()

    class _HitlMw(Middleware):
        def __init__(self) -> None:
            self.hitl = HitlBinding(checkpointed=True, run_id="R1")

    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
        middleware=[_HitlMw()],
    )
    await cp.claim_run("src", "R1")
    await cp.mark_run_complete("src", "R1")
    with pytest.raises(RuntimeError, match="does not support HITL"):
        await a.fork_once("src", "msg", after_run_id="R1")


@pytest.mark.asyncio
async def test_fork_once_cancellation_propagates():
    cp = MemoryCheckpointer()
    # Seed src with a quick run.
    seed_agent = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await seed_agent.prompt("hello", run_id="R1")

    # Build a slow FauxProvider for the probe.
    async def _slow_factory(messages, model, system_prompt, tools):
        await asyncio.sleep(5)
        return AssistantMessage(
            content=[TextContent(text="never")], stop_reason="end_turn"
        )

    slow_p = FauxProvider()
    slow_p.set_responses([_slow_factory])

    a = Agent(
        model=slow_p.model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            a.fork_once("src", "msg", after_run_id="R1"), timeout=0.05
        )


@pytest.mark.asyncio
async def test_fork_once_source_thread_byte_identical():
    cp = MemoryCheckpointer()
    seed_agent = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await seed_agent.prompt("hello", run_id="R1")
    before = await cp.load("src")
    before_dump = [m.model_dump() for m in before.messages]
    before_extra = dict(before.extra)
    before_runs = dict(cp._runs.get("src", {}))

    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await a.fork_once("src", "probe", after_run_id="R1")

    after = await cp.load("src")
    after_dump = [m.model_dump() for m in after.messages]
    assert after_dump == before_dump
    assert dict(after.extra) == before_extra
    # Source thread's run state untouched.
    assert dict(cp._runs.get("src", {})) == before_runs
