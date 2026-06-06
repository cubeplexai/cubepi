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


@pytest.mark.asyncio
async def test_fork_once_span_falls_back_to_nullcontext_when_otel_missing(
    monkeypatch,
):
    """When the optional `tracing` extra isn't installed, the span helper
    must fall back to contextlib.nullcontext so fork_once still works."""
    import builtins

    real_import = builtins.__import__

    def _no_otel(name, *args, **kwargs):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError("simulated: opentelemetry not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_otel)

    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    await a.prompt("hello", run_id="R1")
    a2 = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="src",
    )
    # Must not raise — the helper falls back to a no-op nullcontext.
    result = await a2.fork_once("src", "follow up?", after_run_id="R1")
    assert result is not None
