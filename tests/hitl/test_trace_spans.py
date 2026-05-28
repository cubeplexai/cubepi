import asyncio
import pytest

pytest.importorskip("opentelemetry")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from cubepi.hitl import ApproveAnswer
from cubepi.hitl.channel import InMemoryChannel


@pytest.fixture
def exporter():
    provider = TracerProvider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    trace.set_tracer_provider(provider)
    yield exp


async def test_approve_emits_hitl_span(exporter):
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    await ch.approve(tool_name="bash", tool_call_id="tc-1", args={})

    spans = exporter.get_finished_spans()
    hitl_spans = [s for s in spans if s.name == "hitl.approve"]
    assert len(hitl_spans) == 1
    assert hitl_spans[0].attributes["hitl.question_id"] == "tc-1"
    assert hitl_spans[0].attributes["hitl.tool_name"] == "bash"
    assert hitl_spans[0].attributes["hitl.outcome"] == "approved"
