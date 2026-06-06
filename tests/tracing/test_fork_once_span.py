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
async def test_fork_once_emits_named_span(in_memory_exporter):
    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await a.prompt("hello", run_id="R1")
    # Fresh agent for the fork_once probe (mirrors existing fork_once tests).
    a2 = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await a2.fork_once("src", "follow up?", after_run_id="R1")
    spans = in_memory_exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "cubepi.agent.fork_once" in names
    span = next(s for s in spans if s.name == "cubepi.agent.fork_once")
    attrs = dict(span.attributes)
    assert attrs["cubepi.fork.src_thread_id"] == "src"
    assert attrs["cubepi.fork.after_run_id"] == "R1"
