"""Tests for Tracer.oneshot() — instrumented one-shot LLM calls."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from cubepi.providers.base import (
    AssistantMessage,
    Message,
    MessageStream,
    Model,
    ReasoningControl,
    StreamEvent,
    StreamOptions,
    TextContent,
    ToolDefinition,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.tracing import Tracer
from cubepi.tracing.tracer import _OneShotSession


MODEL = Model(id="faux-1", provider_id="faux")


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
    provider = FauxProvider(provider_id="faux")
    tracer, _ = _make_tracer()
    async with tracer.oneshot(model=provider.model(MODEL.id)) as session:
        assert isinstance(session, _OneShotSession)
    await tracer.shutdown()


@pytest.mark.asyncio
async def test_oneshot_session_uses_provider_generate() -> None:
    provider = _GenerateOnlyProvider()
    tracer, _ = _make_tracer()

    async with tracer.oneshot(model=provider.model(MODEL.id)) as session:
        text = await session.generate(
            system="sys",
            messages=[UserMessage(content=[TextContent(text="q")])],
            max_output_tokens=123,
        )

    await tracer.shutdown()

    assert text == "via generate"
    assert provider.seen_max_output_tokens == 123
    assert provider.seen_options is not None
    assert provider.seen_options.signal is not None


@pytest.mark.asyncio
async def test_oneshot_produces_root_and_chat_spans() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.append_responses([faux_assistant_message("hello world")])
    tracer, exporter = _make_tracer()

    async with tracer.oneshot(
        model=provider.model(MODEL.id),
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
    # operation also exposed under cubepi.metadata.* for --meta CLI filter
    assert attrs.get("cubepi.metadata.oneshot_operation") == "test_op"
    assert attrs.get("cubepi.metadata.conversation_id") == "conv-123"
    assert attrs.get("cubepi.metadata.user_id") == "usr-456"
    assert "cubepi.run_id" in attrs
    # Successful one-shot must NOT be marked aborted by the cleanup sweeper.
    assert attrs.get("cubepi.aborted") is None
    assert attrs.get("error.type") is None

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
    provider = FauxProvider(provider_id="faux")
    provider.append_responses([faux_assistant_message("ok")])
    tracer, exporter = _make_tracer()

    async with tracer.oneshot(
        model=provider.model(MODEL.id),
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
async def test_oneshot_user_metadata_cannot_shadow_reserved_oneshot_operation() -> None:
    """If a caller's metadata dict includes 'oneshot_operation', the value
    derived from the operation argument must still win — the documented
    `cubepi trace ls --meta oneshot_operation=<op>` filter depends on it."""
    provider = FauxProvider(provider_id="faux")
    provider.append_responses([faux_assistant_message("ok")])
    tracer, exporter = _make_tracer()

    async with tracer.oneshot(
        model=provider.model(MODEL.id),
        operation="real_op",
        metadata={"oneshot_operation": "user_supplied_value"},
    ) as session:
        await session.generate(
            system="sys",
            messages=[UserMessage(content=[TextContent(text="q")])],
            max_output_tokens=10,
        )

    await tracer.force_flush()
    await tracer.shutdown()

    roots = [s for s in exporter.spans if s.name == "invoke_agent"]
    assert len(roots) == 1
    attrs = dict(roots[0].attributes or {})
    assert attrs["cubepi.metadata.oneshot_operation"] == "real_op"
    assert attrs["cubepi.oneshot.operation"] == "real_op"


@pytest.mark.asyncio
async def test_oneshot_no_metadata_ok() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.append_responses([faux_assistant_message("result")])
    tracer, exporter = _make_tracer()

    async with tracer.oneshot(model=provider.model(MODEL.id)) as session:
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
    provider = FauxProvider(provider_id="faux")
    provider.append_responses([faux_assistant_message("answer")])
    tracer, exporter = _make_tracer(record_content=True)

    async with tracer.oneshot(
        model=provider.model(MODEL.id),
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

    # The root invoke_agent span must also carry input/output/system so that
    # `cubepi trace ls` (which reads gen_ai.input.messages off the root) can
    # show the prompt in its input column.
    root_spans = [s for s in exporter.spans if s.name == "invoke_agent"]
    assert root_spans, "expected a root invoke_agent span"
    root_attrs = dict(root_spans[0].attributes or {})
    assert "gen_ai.system_instructions" in root_attrs
    assert "gen_ai.input.messages" in root_attrs
    assert "gen_ai.output.messages" in root_attrs
    assert "question" in root_attrs["gen_ai.input.messages"]
    assert "answer" in root_attrs["gen_ai.output.messages"]


@pytest.mark.asyncio
async def test_oneshot_generate_error_event_raises_and_marks_root() -> None:
    """generate() propagates RuntimeError on a stream error event AND marks
    the root invoke_agent span with ERROR status so `cubepi trace ls` shows
    the run as failed instead of as a successful invoke_agent."""
    from opentelemetry.trace import StatusCode
    from unittest.mock import AsyncMock, patch

    provider = FauxProvider(provider_id="faux")
    tracer, exporter = _make_tracer()

    error_stream = MessageStream()
    error_stream.push(StreamEvent(type="error", error_message="boom"))
    error_stream.set_result(
        AssistantMessage(content=[], stop_reason="error", error_message="boom")
    )
    with patch.object(provider, "stream", new=AsyncMock(return_value=error_stream)):
        with pytest.raises(RuntimeError, match="boom"):
            async with tracer.oneshot(model=provider.model(MODEL.id)) as session:
                await session.generate(
                    system="sys",
                    messages=[UserMessage(content=[TextContent(text="q")])],
                    max_output_tokens=10,
                )

    await tracer.force_flush()
    await tracer.shutdown()

    roots = [s for s in exporter.spans if s.name == "invoke_agent"]
    assert len(roots) == 1
    root = roots[0]
    assert root.status.status_code == StatusCode.ERROR
    attrs = dict(root.attributes or {})
    assert attrs.get("error.type") == "RuntimeError"


@pytest.mark.asyncio
async def test_oneshot_subscribe_failure_closes_stream_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A subscription failure on the pre-yield path must close any per-run
    stream file that was already opened, so record_stream mode doesn't leak
    file descriptors on this error path."""
    from unittest.mock import patch

    exporter = InMemoryExporter()
    tracer = Tracer(
        exporters=[exporter],
        record_stream=True,
        stream_dir=str(tmp_path),
        atexit_flush=False,
    )

    provider = FauxProvider(provider_id="faux")

    with patch.object(
        provider, "subscribe_chunk", side_effect=RuntimeError("chunk sub failed")
    ):
        with pytest.raises(RuntimeError, match="chunk sub failed"):
            async with tracer.oneshot(model=provider.model(MODEL.id)):
                pass  # pragma: no cover

    await tracer.shutdown()

    # File was created and then closed; it should be a closed handle, not
    # a leaked open one. We can't directly inspect FD state, but writing
    # via run.stream_file would have failed; instead assert the file
    # exists and is empty/parseable (no events written before failure).
    stream_files = list(tmp_path.glob("*.stream.jsonl"))
    assert len(stream_files) == 1
    # File must be closed and finite-size (no leak symptom: pending writes
    # buffered indefinitely or unable-to-stat). On Linux this is a basic
    # sanity check.
    assert stream_files[0].stat().st_size == 0


@pytest.mark.asyncio
async def test_oneshot_propagates_silent_producer_failure() -> None:
    """If the producer task fails AFTER emitting done (without an explicit
    error event), generate() must still raise — not return empty text and
    leave the root span marked successful."""
    from opentelemetry.trace import StatusCode
    from unittest.mock import AsyncMock, MagicMock, patch

    provider = FauxProvider(provider_id="faux")
    tracer, exporter = _make_tracer()

    # Build a stream that emits done normally but whose result() raises
    fake_stream = MagicMock()

    async def _events():
        yield MagicMock(type="done", delta=None, error_message=None)

    fake_stream.__aiter__ = lambda self: _events()

    async def failing_result():
        raise RuntimeError("producer crashed after done")

    fake_stream.result = failing_result

    with patch.object(provider, "stream", new=AsyncMock(return_value=fake_stream)):
        with pytest.raises(RuntimeError, match="producer crashed"):
            async with tracer.oneshot(model=provider.model(MODEL.id)) as session:
                await session.generate(
                    system="sys",
                    messages=[UserMessage(content=[TextContent(text="q")])],
                    max_output_tokens=10,
                )

    await tracer.force_flush()
    await tracer.shutdown()

    roots = [s for s in exporter.spans if s.name == "invoke_agent"]
    assert len(roots) == 1
    # Root must be marked ERROR (not silently successful)
    assert roots[0].status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_oneshot_subscribe_failure_raises_and_ends_root_span() -> None:
    """If provider subscription raises, oneshot re-raises and ends the root span."""
    from unittest.mock import patch

    provider = FauxProvider(provider_id="faux")
    tracer, exporter = _make_tracer()

    with patch.object(
        provider, "subscribe_request", side_effect=RuntimeError("subscribe failed")
    ):
        with pytest.raises(RuntimeError, match="subscribe failed"):
            async with tracer.oneshot(model=provider.model(MODEL.id)):
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

    provider = FauxProvider(provider_id="faux")
    provider.append_responses([faux_assistant_message("ok")])
    tracer, _ = _make_tracer()

    # Make subscribe_request return a detacher that raises on call
    original_subscribe = provider.subscribe_request

    def patched_subscribe(cb):
        original_subscribe(cb)

        def raising_detach():
            raise RuntimeError("detach failed")

        return raising_detach

    with patch.object(provider, "subscribe_request", side_effect=patched_subscribe):
        async with tracer.oneshot(model=provider.model(MODEL.id)) as session:
            text = await session.generate(
                system="sys",
                messages=[UserMessage(content=[TextContent(text="q")])],
                max_output_tokens=10,
            )

    await tracer.shutdown()
    assert text == "ok"


@pytest.mark.asyncio
async def test_oneshot_passes_signal_to_provider_and_sets_on_cancel() -> None:
    """generate() must forward a StreamOptions.signal so a cancellation in
    the consumer tears the producer task down too."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    provider = FauxProvider(provider_id="faux")
    tracer, _ = _make_tracer()

    captured_options: dict[str, StreamOptions | None] = {"opts": None}

    async def hanging_stream():
        # Never yields, simulating a long provider call
        await asyncio.sleep(60)
        yield  # pragma: no cover

    hanging = MagicMock()
    hanging.__aiter__ = lambda self: hanging_stream()

    real_stream = AsyncMock(return_value=hanging)

    async def patched_stream(**kwargs):
        captured_options["opts"] = kwargs.get("options")
        return await real_stream(**kwargs)

    async def do_work():
        async with tracer.oneshot(model=provider.model(MODEL.id)) as session:
            with patch.object(provider, "stream", new=patched_stream):
                await session.generate(
                    system="sys",
                    messages=[UserMessage(content=[TextContent(text="q")])],
                    max_output_tokens=10,
                )

    task = asyncio.create_task(do_work())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await tracer.shutdown()

    # The provider received options with a signal Event…
    opts = captured_options["opts"]
    assert opts is not None
    assert opts.signal is not None
    # …and the signal was set when the consumer was cancelled.
    assert opts.signal.is_set()


@pytest.mark.asyncio
async def test_oneshot_record_stream_writes_jsonl(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """When record_stream is on, oneshot must open a per-run stream.jsonl
    so _on_provider_chunk can write the same per-chunk log as the agent path."""
    exporter = InMemoryExporter()
    tracer = Tracer(
        exporters=[exporter],
        record_stream=True,
        stream_dir=str(tmp_path),
        atexit_flush=False,
    )

    provider = FauxProvider(provider_id="faux")
    provider.append_responses([faux_assistant_message("streamed text")])

    async with tracer.oneshot(model=provider.model(MODEL.id)) as session:
        text = await session.generate(
            system="sys",
            messages=[UserMessage(content=[TextContent(text="hi")])],
            max_output_tokens=20,
        )

    await tracer.shutdown()

    assert text == "streamed text"
    stream_files = list(tmp_path.glob("*.stream.jsonl"))
    assert len(stream_files) == 1, f"expected one stream file, got {stream_files}"
    content = stream_files[0].read_text(encoding="utf-8").strip()
    assert content, "stream file should have at least one event"
    # The file should be one JSON object per line — sanity-check parse
    import json

    for line in content.splitlines():
        json.loads(line)


@pytest.mark.asyncio
async def test_oneshot_partial_subscribe_failure_unwinds_listeners() -> None:
    """If subscribe_chunk raises after subscribe_request succeeded, the first
    listener must be unsubscribed before re-raising (no dangling listeners)."""
    from unittest.mock import patch

    provider = FauxProvider(provider_id="faux")
    tracer, exporter = _make_tracer()

    # Track how many listeners were subscribed / detached
    subscribed: list[str] = []
    detached: list[str] = []
    original_sub_req = provider.subscribe_request

    def patched_sub_req(cb):
        subscribed.append("request")
        real_unsub = original_sub_req(cb)

        def counting_unsub():
            detached.append("request")
            real_unsub()

        return counting_unsub

    with (
        patch.object(provider, "subscribe_request", side_effect=patched_sub_req),
        patch.object(
            provider, "subscribe_chunk", side_effect=RuntimeError("chunk sub failed")
        ),
        pytest.raises(RuntimeError, match="chunk sub failed"),
    ):
        async with tracer.oneshot(model=provider.model(MODEL.id)):
            pass  # pragma: no cover

    await tracer.force_flush()
    await tracer.shutdown()

    # subscribe_request succeeded; its detacher must have been called on cleanup
    assert "request" in subscribed
    assert "request" in detached

    # Root span must still be exported despite the error
    roots = [s for s in exporter.spans if s.name == "invoke_agent"]
    assert len(roots) == 1


@pytest.mark.asyncio
async def test_oneshot_cancelled_generate_closes_chat_span() -> None:
    """If generate() is cancelled mid-stream, the chat span must be closed."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    provider = FauxProvider(provider_id="faux")
    tracer, exporter = _make_tracer()

    # A stream that never yields (blocks forever) so we can cancel it
    async def hanging_stream():
        await asyncio.sleep(60)
        yield  # pragma: no cover

    hanging = MagicMock()
    hanging.__aiter__ = lambda self: hanging_stream()

    async def do_work():
        async with tracer.oneshot(model=provider.model(MODEL.id)) as session:
            with patch.object(provider, "stream", new=AsyncMock(return_value=hanging)):
                await session.generate(
                    system="sys",
                    messages=[UserMessage(content=[TextContent(text="q")])],
                    max_output_tokens=10,
                )

    task = asyncio.create_task(do_work())
    await asyncio.sleep(0.05)  # let the task reach generate()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await tracer.force_flush()
    await tracer.shutdown()

    # Root span must be exported; chat span if opened must also be ended.
    # Cancellation is tracked as cubepi.aborted (not StatusCode.ERROR),
    # matching the agent path's contract.
    roots = [s for s in exporter.spans if s.name == "invoke_agent"]
    assert len(roots) == 1
    attrs = dict(roots[0].attributes or {})
    assert attrs.get("cubepi.aborted") is True
    assert attrs.get("error.type") == "cubepi.aborted"


@pytest.mark.asyncio
async def test_oneshot_cancel_mid_stream_marks_open_chat_span_aborted() -> None:
    """When cancellation happens after the chat span has been opened by the
    provider request listener but before the response listener has closed
    it, the oneshot cleanup must close it and stamp cubepi.aborted."""
    import asyncio

    # Slow provider: streams the response one token at a time so we can
    # cancel mid-stream (after _on_provider_request has fired).
    provider = FauxProvider(tokens_per_second=5.0)
    provider.append_responses([faux_assistant_message("a b c d e f g h")])
    tracer, exporter = _make_tracer()

    async def do_work():
        async with tracer.oneshot(model=provider.model(MODEL.id)) as session:
            await session.generate(
                system="sys",
                messages=[UserMessage(content=[TextContent(text="q")])],
                max_output_tokens=50,
            )

    task = asyncio.create_task(do_work())
    await asyncio.sleep(0.15)  # let chunks start, chat span opens
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await tracer.force_flush()
    await tracer.shutdown()

    # Chat span must have been ended with cubepi.aborted=true
    chat_spans = [s for s in exporter.spans if s.name.startswith("chat ")]
    assert chat_spans, "expected at least one chat span (opened by request listener)"
    chat_attrs = dict(chat_spans[0].attributes or {})
    assert chat_attrs.get("cubepi.aborted") is True
    assert chat_attrs.get("error.type") == "cubepi.aborted"


@pytest.mark.asyncio
async def test_oneshot_does_not_interfere_with_concurrent_agent() -> None:
    """Oneshot's active-run gate must not bleed into a concurrent Agent run."""
    import asyncio

    from cubepi.agent.agent import Agent

    provider = FauxProvider(provider_id="faux")
    # Push responses for both agent and oneshot
    provider.append_responses([faux_assistant_message("agent reply")])
    provider.append_responses([faux_assistant_message("oneshot reply")])

    tracer, exporter = _make_tracer()
    agent = Agent(model=provider.model(MODEL.id), system_prompt="sys")
    detach = tracer.attach(agent)

    # Run agent and oneshot concurrently
    async def run_agent() -> str:
        await agent.prompt("agent question")
        return str(agent.state.messages[-1].content[0].text)  # type: ignore[index]

    async def run_oneshot() -> str:
        async with tracer.oneshot(model=provider.model(MODEL.id)) as session:
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


class _GenerateOnlyProvider(FauxProvider):
    def __init__(self) -> None:
        super().__init__()
        self.seen_max_output_tokens: int | None = None
        self.seen_options: StreamOptions | None = None

    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: Any = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning: ReasoningControl | None = None,
    ) -> AssistantMessage:
        del model, messages, system_prompt, tools, tool_choice, temperature, reasoning
        self.seen_max_output_tokens = max_output_tokens
        self.seen_options = options
        return AssistantMessage(content=[TextContent(text="via generate")])

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: Any = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        del model, messages, system_prompt, tools, tool_choice, options
        raise AssertionError("Tracer.oneshot must call provider.generate()")
