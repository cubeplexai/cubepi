import pytest

from cubepi.agent.agent import Agent
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
async def test_prompt_returns_supplied_run_id():
    a = _agent()
    got = await a.prompt("hello", run_id="R1")
    assert got == "R1"


@pytest.mark.asyncio
async def test_prompt_generates_run_id_when_none():
    a = _agent()
    got = await a.prompt("hello")
    assert isinstance(got, str) and len(got) >= 8


@pytest.mark.asyncio
async def test_prompt_sets_then_clears_active_run_id_on_clean_return():
    a = _agent()
    assert a.state.active_run_id is None
    await a.prompt("hello", run_id="R1")
    assert a.state.active_run_id is None  # cleared on clean return


@pytest.mark.asyncio
async def test_prompt_leaves_active_run_id_set_on_raise(monkeypatch):
    """Spec §3.7 + Task 22: active_run_id must be LEFT SET on any
    propagating failure after claim."""
    a = Agent(model=_ok_faux().model("faux-model"))

    async def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(a, "_run_prompt", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        await a.prompt("hello", run_id="R1")
    assert a.state.active_run_id == "R1"
