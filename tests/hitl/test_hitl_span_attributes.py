"""Tests for the HITL trace span ``hitl.run_id`` + ``hitl.detached`` attrs.

``hitl.run_id`` should appear on spans emitted by a CheckpointedChannel
constructed with ``run_id=...``. ``hitl.detached`` should be True when
the unwind cause was HitlDetached (cross-process suspend), and False for
all other outcomes.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl import ApproveAnswer
from cubepi.hitl.channel import CheckpointedChannel, InMemoryChannel
from cubepi.hitl.exceptions import HitlDetached


@pytest.fixture
def exporter():
    # OTel forbids replacing the global TracerProvider once set, so attach a
    # fresh InMemorySpanExporter to whatever provider is already installed
    # (or install one if none is). Per-test exporter keeps tests independent.
    current = trace.get_tracer_provider()
    if not isinstance(current, TracerProvider):
        current = TracerProvider()
        trace.set_tracer_provider(current)
    exp = InMemorySpanExporter()
    current.add_span_processor(SimpleSpanProcessor(exp))
    yield exp


async def test_checkpointed_channel_emits_run_id_attribute(exporter):
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-1", run_id="r-trace")

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    await ch.approve(tool_name="bash", tool_call_id="tc-1", args={})

    spans = [s for s in exporter.get_finished_spans() if s.name == "hitl.approve"]
    assert len(spans) == 1
    assert spans[0].attributes["hitl.run_id"] == "r-trace"
    assert spans[0].attributes["hitl.detached"] is False


async def test_in_memory_channel_omits_run_id_attribute(exporter):
    """InMemoryChannel has no run_id concept — the attribute must NOT be set
    (hitl_span skips None-valued attrs)."""
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    await ch.approve(tool_name="bash", tool_call_id="tc-2", args={})

    spans = [s for s in exporter.get_finished_spans() if s.name == "hitl.approve"]
    assert len(spans) == 1
    assert "hitl.run_id" not in spans[0].attributes
    assert spans[0].attributes["hitl.detached"] is False


async def test_detached_outcome_marked_on_span(exporter):
    cp = MemoryCheckpointer()
    ch = CheckpointedChannel(checkpointer=cp, thread_id="t-2", run_id="r-detach")

    async def detacher():
        while ch.pending is None:
            await asyncio.sleep(0)
        if ch._future is not None and not ch._future.done():
            ch._future.set_exception(HitlDetached())

    asyncio.create_task(detacher())
    with pytest.raises(HitlDetached):
        await ch.confirm("ok?")

    spans = [s for s in exporter.get_finished_spans() if s.name == "hitl.confirm"]
    assert len(spans) == 1
    assert spans[0].attributes["hitl.detached"] is True
    assert spans[0].attributes["hitl.outcome"] == "detached"
    assert spans[0].attributes["hitl.run_id"] == "r-detach"
