import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
)
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import AssistantMessage, TextContent
from cubepi.providers.faux import FauxProvider


def _ok_faux() -> FauxProvider:
    p = FauxProvider()
    p.set_responses(
        [AssistantMessage(content=[TextContent(text="ok")], stop_reason="end_turn")]
    )
    return p


def _agent(**kw):
    return Agent(model=_ok_faux().model("faux-model"), **kw)


@pytest.mark.asyncio
async def test_prompt_calls_claim_run_before_append():
    cp = MemoryCheckpointer()
    a = _agent(checkpointer=cp, thread_id="t")
    await a.prompt("hi", run_id="R1")
    # Run row exists.
    assert "R1" in cp._runs["t"]


@pytest.mark.asyncio
async def test_prompt_rejects_already_claimed_run_id():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "R1")
    a = _agent(checkpointer=cp, thread_id="t")
    with pytest.raises(RunAlreadyClaimedError):
        await a.prompt("hi", run_id="R1")


@pytest.mark.asyncio
async def test_prompt_rejects_already_completed_run_id():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "R1")
    await cp.mark_run_complete("t", "R1")
    a = _agent(checkpointer=cp, thread_id="t")
    with pytest.raises(RunAlreadyCompletedError):
        await a.prompt("hi", run_id="R1")


@pytest.mark.asyncio
async def test_no_checkpointer_no_claim_call():
    a = _agent()  # checkpointer=None
    got = await a.prompt("hi")
    assert isinstance(got, str)  # works fine; no claim attempted


@pytest.mark.asyncio
async def test_degraded_mode_v3_only_checkpointer():
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

        # No claim_run / mark_run_complete.

    a = _agent(checkpointer=_V3Only(), thread_id="t")
    # Vanilla prompt works (degraded mode skips claim).
    got = await a.prompt("hi")
    assert isinstance(got, str)
