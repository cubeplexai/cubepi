"""Tests for Tracer.oneshot() — instrumented one-shot LLM calls."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from cubepi.providers.base import Model, TextContent, UserMessage
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.tracing import Tracer
from cubepi.tracing.tracer import _OneShotSession


MODEL = Model(id="faux-1", provider="faux")


class InMemoryExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Any) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _make_tracer(*, record_content: bool = False) -> tuple[Tracer, InMemoryExporter]:
    exporter = InMemoryExporter()
    tracer = Tracer(
        exporters=[exporter],
        record_content=record_content,
        atexit_flush=False,
    )
    return tracer, exporter


def _spans_by_name(spans: list[ReadableSpan]) -> dict[str, ReadableSpan]:
    result: dict[str, ReadableSpan] = {}
    for s in spans:
        result[s.name] = s
    return result


@pytest.mark.asyncio
async def test_oneshot_session_type() -> None:
    provider = FauxProvider()
    tracer, _ = _make_tracer()
    async with tracer.oneshot(provider=provider, model=MODEL) as session:
        assert isinstance(session, _OneShotSession)
    await tracer.shutdown()


@pytest.mark.asyncio
async def test_oneshot_produces_root_and_chat_spans() -> None:
    provider = FauxProvider()
    provider.append_responses([faux_assistant_message("hello world")])
    tracer, exporter = _make_tracer()

    async with tracer.oneshot(
        provider=provider,
        model=MODEL,
        operation="test_op",
        metadata={"conversation_id": "conv-123", "user_id": "usr-456"},
    ) as session:
        text = await session.generate(
            system="You are helpful.",
            messages=[UserMessage(content=[TextContent(text="hi")])],
            max_output_tokens=100,
        )

    await tracer.force_flush()
    await tracer.shutdown()

    assert text == "hello world"

    by_name = _spans_by_name(exporter.spans)
    assert "invoke_agent" in by_name, f"spans: {[s.name for s in exporter.spans]}"
    assert f"chat {MODEL.id}" in by_name

    root = by_name["invoke_agent"]
    attrs = dict(root.attributes or {})
    assert attrs.get("gen_ai.operation.name") == "invoke_agent"
    assert attrs.get("cubepi.oneshot.operation") == "test_op"
    assert attrs.get("cubepi.metadata.conversation_id") == "conv-123"
    assert attrs.get("cubepi.metadata.user_id") == "usr-456"
    assert "cubepi.run_id" in attrs

    chat = by_name[f"chat {MODEL.id}"]
    # chat span must be a child of root
    assert chat.parent is not None
    assert chat.parent.span_id == root.context.span_id

    # token counts recorded
    chat_attrs = dict(chat.attributes or {})
    assert "gen_ai.usage.input_tokens" in chat_attrs
    assert "gen_ai.usage.output_tokens" in chat_attrs


@pytest.mark.asyncio
async def test_oneshot_metadata_on_root_span() -> None:
    provider = FauxProvider()
    provider.append_responses([faux_assistant_message("ok")])
    tracer, exporter = _make_tracer()

    async with tracer.oneshot(
        provider=provider,
        model=MODEL,
        operation="consolidate_memory",
        metadata={"conversation_id": "conv-abc"},
    ) as session:
        await session.generate(
            system="sys",
            messages=[UserMessage(content=[TextContent(text="x")])],
            max_output_tokens=50,
        )

    await tracer.force_flush()
    await tracer.shutdown()

    roots = [s for s in exporter.spans if s.name == "invoke_agent"]
    assert len(roots) == 1
    attrs = dict(roots[0].attributes or {})
    assert attrs["cubepi.oneshot.operation"] == "consolidate_memory"
    assert attrs["cubepi.metadata.conversation_id"] == "conv-abc"


@pytest.mark.asyncio
async def test_oneshot_no_metadata_ok() -> None:
    provider = FauxProvider()
    provider.append_responses([faux_assistant_message("result")])
    tracer, exporter = _make_tracer()

    async with tracer.oneshot(provider=provider, model=MODEL) as session:
        text = await session.generate(
            system="sys",
            messages=[UserMessage(content=[TextContent(text="q")])],
            max_output_tokens=10,
        )

    await tracer.force_flush()
    await tracer.shutdown()

    assert text == "result"
    assert any(s.name == "invoke_agent" for s in exporter.spans)


@pytest.mark.asyncio
async def test_oneshot_record_content_captures_messages() -> None:
    provider = FauxProvider()
    provider.append_responses([faux_assistant_message("answer")])
    tracer, exporter = _make_tracer(record_content=True)

    async with tracer.oneshot(
        provider=provider,
        model=MODEL,
        record_content=True,
    ) as session:
        await session.generate(
            system="be helpful",
            messages=[UserMessage(content=[TextContent(text="question")])],
            max_output_tokens=50,
        )

    await tracer.force_flush()
    await tracer.shutdown()

    chat_spans = [s for s in exporter.spans if s.name.startswith("chat")]
    assert chat_spans, "expected a chat span"
    attrs = dict(chat_spans[0].attributes or {})
    # With record_content=True the system instructions are recorded on the chat span
    assert "gen_ai.system_instructions" in attrs


@pytest.mark.asyncio
async def test_oneshot_generate_error_event_raises() -> None:
    """generate() propagates RuntimeError when the stream emits an error event."""
    from unittest.mock import AsyncMock, MagicMock, patch

    provider = FauxProvider()
    tracer, _ = _make_tracer()

    # Patch provider.stream to yield an error event
    error_stream = MagicMock()

    async def _gen():
        yield MagicMock(type="error", error_message="boom", delta=None)

    error_stream.__aiter__ = lambda self: _gen()
    with patch.object(provider, "stream", new=AsyncMock(return_value=error_stream)):
        with pytest.raises(RuntimeError, match="boom"):
            async with tracer.oneshot(provider=provider, model=MODEL) as session:
                await session.generate(
                    system="sys",
                    messages=[UserMessage(content=[TextContent(text="q")])],
                    max_output_tokens=10,
                )

    await tracer.shutdown()


@pytest.mark.asyncio
async def test_oneshot_subscribe_failure_raises_and_ends_root_span() -> None:
    """If provider subscription raises, oneshot re-raises and ends the root span."""
    from unittest.mock import patch

    provider = FauxProvider()
    tracer, exporter = _make_tracer()

    with patch.object(
        provider, "subscribe_request", side_effect=RuntimeError("subscribe failed")
    ):
        with pytest.raises(RuntimeError, match="subscribe failed"):
            async with tracer.oneshot(provider=provider, model=MODEL):
                pass  # pragma: no cover — never reached

    await tracer.force_flush()
    await tracer.shutdown()

    # Root span must have been ended even though subscribe raised
    roots = [s for s in exporter.spans if s.name == "invoke_agent"]
    assert len(roots) == 1


@pytest.mark.asyncio
async def test_oneshot_detacher_exception_is_swallowed() -> None:
    """Detach errors during cleanup must not propagate to the caller."""
    from unittest.mock import patch

    provider = FauxProvider()
    provider.append_responses([faux_assistant_message("ok")])
    tracer, _ = _make_tracer()

    # Make subscribe_request return a detacher that raises on call
    original_subscribe = provider.subscribe_request

    def patched_subscribe(cb):
        unsub = original_subscribe(cb)

        def raising_detach():
            raise RuntimeError("detach failed")

        return raising_detach

    with patch.object(provider, "subscribe_request", side_effect=patched_subscribe):
        async with tracer.oneshot(provider=provider, model=MODEL) as session:
            text = await session.generate(
                system="sys",
                messages=[UserMessage(content=[TextContent(text="q")])],
                max_output_tokens=10,
            )

    await tracer.shutdown()
    assert text == "ok"


@pytest.mark.asyncio
async def test_oneshot_does_not_interfere_with_concurrent_agent() -> None:
    """Oneshot's active-run gate must not bleed into a concurrent Agent run."""
    import asyncio

    from cubepi.agent.agent import Agent

    provider = FauxProvider()
    # Push responses for both agent and oneshot
    provider.append_responses([faux_assistant_message("agent reply")])
    provider.append_responses([faux_assistant_message("oneshot reply")])

    tracer, exporter = _make_tracer()
    agent = Agent(provider=provider, model=MODEL, system_prompt="sys")
    detach = tracer.attach(agent)

    # Run agent and oneshot concurrently
    async def run_agent() -> str:
        await agent.prompt("agent question")
        return str(agent.state.messages[-1].content[0].text)  # type: ignore[index]

    async def run_oneshot() -> str:
        async with tracer.oneshot(provider=provider, model=MODEL) as session:
            return await session.generate(
                system="sys",
                messages=[UserMessage(content=[TextContent(text="oneshot q")])],
                max_output_tokens=50,
            )

    agent_result, oneshot_result = await asyncio.gather(run_agent(), run_oneshot())

    result = detach()
    if result is not None:
        await result
    await tracer.force_flush()
    await tracer.shutdown()

    assert "agent reply" in agent_result or "oneshot reply" in agent_result
    assert "agent reply" in oneshot_result or "oneshot reply" in oneshot_result

    # Both runs should produce an invoke_agent span
    roots = [s for s in exporter.spans if s.name == "invoke_agent"]
    assert len(roots) == 2
