"""AgentContext.extra field tests (D6)."""

import tempfile
from pathlib import Path

import pytest

from cubepi.agent.types import AgentContext


def test_agent_context_default_extra_is_empty_dict() -> None:
    ctx = AgentContext(system_prompt="", messages=[])
    assert ctx.extra == {}


def test_agent_context_accepts_extra() -> None:
    ctx = AgentContext(
        system_prompt="",
        messages=[],
        extra={"todos": ["a", "b"]},
    )
    assert ctx.extra["todos"] == ["a", "b"]


def test_extra_is_mutable() -> None:
    ctx = AgentContext(system_prompt="", messages=[])
    ctx.extra["k"] = "v"
    assert ctx.extra == {"k": "v"}


def test_extra_independent_between_instances() -> None:
    a = AgentContext(system_prompt="", messages=[])
    b = AgentContext(system_prompt="", messages=[])
    a.extra["x"] = 1
    assert "x" not in b.extra


@pytest.mark.asyncio
async def test_save_extra_round_trip_via_checkpointer() -> None:
    """save_extra + load round-trip works (sanity check for the checkpointer)."""
    from cubepi.checkpointer import SQLiteCheckpointer

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "extra.db"
        async with SQLiteCheckpointer(str(path)) as cp:
            await cp.save_extra("t-extra", {"counter": 1})
            data = await cp.load("t-extra")
            assert data is not None
            assert data.extra == {"counter": 1}


@pytest.mark.asyncio
async def test_agent_hydrates_ctx_extra_from_checkpointer() -> None:
    """When agent loads a thread with pre-existing extra, the extra is preserved
    through the turn and written back via save_extra."""
    from cubepi.agent.agent import Agent
    from cubepi.checkpointer import SQLiteCheckpointer
    from cubepi.providers.base import Model
    from cubepi.providers.faux import FauxProvider, faux_assistant_message

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "hydrate.db"
        async with SQLiteCheckpointer(str(path)) as cp:
            await cp.save_extra("t-hyd", {"seeded": True})

            provider = FauxProvider()
            provider.set_responses([faux_assistant_message("ok")])
            agent = Agent(
                provider=provider,
                model=Model(id="test", provider="faux"),
                checkpointer=cp,
                thread_id="t-hyd",
            )
            await agent.prompt("hi")

            # After the turn, the seeded extra must still be there
            # (we haven't mutated it; loop must preserve it across save_extra).
            data = await cp.load("t-hyd")
            assert data is not None
            assert data.extra.get("seeded") is True


@pytest.mark.asyncio
async def test_agent_persists_ctx_extra_mutation_after_turn() -> None:
    """Pre-seeded extra is round-tripped: hydrated on load, persisted via
    save_extra after the turn completes."""
    from cubepi.agent.agent import Agent
    from cubepi.checkpointer import SQLiteCheckpointer
    from cubepi.providers.base import Model
    from cubepi.providers.faux import FauxProvider, faux_assistant_message

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "persist.db"
        async with SQLiteCheckpointer(str(path)) as cp:
            provider = FauxProvider()
            provider.set_responses([faux_assistant_message("ok")])

            agent = Agent(
                provider=provider,
                model=Model(id="test", provider="faux"),
                checkpointer=cp,
                thread_id="t-pst",
            )
            # Pre-seed via the checkpointer before the agent runs.
            await cp.save_extra("t-pst", {"initial": 1})

            await agent.prompt("hi")

            data = await cp.load("t-pst")
            assert data is not None
            assert data.extra.get("initial") == 1
