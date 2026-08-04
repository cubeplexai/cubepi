"""Tracing coverage for durable HITL suspension."""

from __future__ import annotations

import asyncio
import gc
from contextlib import suppress
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.hitl.ask_user import ask_user_tool
from cubepi.hitl.channel import CheckpointedChannel
from cubepi.mcp import _tracing as mcp_tracing
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall
from cubepi.providers.faux import FauxProvider
from cubepi.tracing import Tracer
from cubepi.tracing import recorder as recorder_module


class InMemoryExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans):  # noqa: ANN001
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _build_suspending_agent(
    thread_id: str,
) -> tuple[Agent, CheckpointedChannel, InMemoryExporter, Tracer]:
    cp = MemoryCheckpointer()
    channel = CheckpointedChannel(
        checkpointer=cp,
        thread_id=thread_id,
        run_id="R1",
    )
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [
            AssistantMessage(
                content=[
                    ToolCall(
                        id="ask-1",
                        name="ask_user",
                        arguments={"questions": [{"key": "answer", "prompt": "?"}]},
                    )
                ],
                stop_reason="tool_use",
            ),
            AssistantMessage(
                content=[TextContent(text="done")],
                stop_reason="end_turn",
            ),
        ]
    )
    agent = Agent(
        model=provider.model("faux-model"),
        tools=[ask_user_tool(channel)],
        checkpointer=cp,
        thread_id=thread_id,
        channel=channel,
    )
    exporter = InMemoryExporter()
    tracer = Tracer(service_name="test", agent_name="test-agent", exporters=[exporter])
    return agent, channel, exporter, tracer


async def _wait_for_pending(channel: CheckpointedChannel) -> None:
    for _ in range(200):
        if channel.pending is not None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("agent did not suspend on HITL")


async def _detach_tracing(detach: Any, tracer: Tracer) -> None:
    flush_task = detach()
    if flush_task is not None:
        await flush_task
    await tracer.shutdown()


async def test_suspended_outcome_without_snapshot_emits_no_event() -> None:
    agent, _channel, _exporter, tracer = _build_suspending_agent("thread-no-snapshot")
    observed: list[str] = []

    async def observe(event, signal=None):  # noqa: ANN001, ANN202
        del signal
        observed.append(event.type)

    unsubscribe = agent.subscribe(observe)
    await agent._emit_suspended_event("suspended")
    unsubscribe()
    await tracer.shutdown()

    assert observed == []


async def test_recorder_suspension_without_active_run_is_noop() -> None:
    from cubepi.tracing.recorder import Recorder

    tracer = Tracer(service_name="test", exporters=[])
    recorder = Recorder(tracer)

    recorder._on_agent_suspended()
    await tracer.shutdown()

    assert recorder._run is None


async def test_suspended_run_exports_suspended_trace_without_abort_or_leaks() -> None:
    agent, channel, exporter, tracer = _build_suspending_agent("thread-1")
    provider_stack_baseline = len(mcp_tracing._provider_stack)
    active_entries_baseline = len(mcp_tracing._active_entries)
    detach_tracing = tracer.attach(agent)

    prompt_task = asyncio.create_task(agent.prompt("hi", run_id="R1"))
    await _wait_for_pending(channel)
    await agent.detach()
    await prompt_task
    await _detach_tracing(detach_tracing, tracer)

    roots = [span for span in exporter.spans if span.name == "invoke_agent"]
    assert len(roots) == 1
    assert roots[0].attributes.get("cubepi.output.messages.count") == 1
    suspended_spans = [
        span
        for span in exporter.spans
        if span.name in {"invoke_agent", "cubepi.turn"}
        or span.name.startswith("execute_tool ")
    ]
    assert {span.name for span in suspended_spans} == {
        "invoke_agent",
        "cubepi.turn",
        "execute_tool ask_user",
    }
    assert all(
        span.attributes.get("cubepi.run.outcome") == "suspended"
        for span in suspended_spans
    )
    assert not [
        span.name
        for span in exporter.spans
        if span.attributes.get("cubepi.aborted") is True
    ]
    assert len(mcp_tracing._provider_stack) == provider_stack_baseline
    assert len(mcp_tracing._active_entries) == active_entries_baseline


async def test_recorder_suspension_finalizes_all_open_resources() -> None:
    from cubepi.tracing.recorder import Recorder, _RunState

    class _Span:
        def __init__(self) -> None:
            self.attributes: dict[str, Any] = {}
            self.ended = False

        def set_attribute(self, key: str, value: Any) -> None:
            self.attributes[key] = value

        def end(self) -> None:
            self.ended = True

    class _Stream:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    tracer = Tracer(service_name="test", exporters=[])
    recorder = Recorder(tracer)
    agent_span = _Span()
    turn_span = _Span()
    chat_span = _Span()
    tool_span = _Span()
    stream = _Stream()
    run = _RunState(
        run_id="run-open-resources",
        agent_span=agent_span,
        turn_span=turn_span,
        chat_span=chat_span,
        tool_spans={"tool-1": tool_span},
        stream_file=stream,
    )
    recorder._run = run

    recorder._on_agent_suspended()
    await tracer.shutdown()

    for span in (agent_span, turn_span, chat_span, tool_span):
        assert span.attributes["cubepi.run.outcome"] == "suspended"
        assert span.ended is True
    assert stream.closed is True
    assert run.tool_spans == {}
    assert run.chat_span is None
    assert run.turn_span is None
    assert run.stream_file is None
    assert recorder._run is None


async def test_recorder_suspension_cleanup_continues_after_resource_errors() -> None:
    from cubepi.tracing.recorder import Recorder, _RunState

    class _Span:
        def __init__(self) -> None:
            self.attributes: dict[str, Any] = {}
            self.ended = False

        def set_attribute(self, key: str, value: Any) -> None:
            self.attributes[key] = value

        def end(self) -> None:
            self.ended = True

    class _BoomSpan(_Span):
        def end(self) -> None:
            raise RuntimeError("span end failed")

    class _BoomStream:
        def close(self) -> None:
            raise OSError("stream close failed")

    tracer = Tracer(service_name="test", exporters=[])
    recorder = Recorder(tracer)
    agent_span = _Span()
    run = _RunState(
        run_id="run-resource-errors",
        agent_span=agent_span,
        turn_span=_BoomSpan(),
        chat_span=_BoomSpan(),
        tool_spans={"tool-1": _BoomSpan()},
        stream_file=_BoomStream(),
    )
    recorder._run = run

    recorder._on_agent_suspended()
    await tracer.shutdown()

    assert agent_span.attributes["cubepi.run.outcome"] == "suspended"
    assert agent_span.ended is True
    assert run.tool_spans == {}
    assert run.chat_span is None
    assert run.turn_span is None
    assert run.stream_file is None
    assert recorder._run is None


async def test_suspension_event_observes_committed_runtime_state() -> None:
    agent, channel, _exporter, tracer = _build_suspending_agent("thread-2")
    detach_tracing = tracer.attach(agent)
    observed: list[tuple[str | None, str | None]] = []

    async def fail_after_observing(event, signal=None):  # noqa: ANN001, ANN202
        del signal
        if event.type == "agent_suspended":
            observed.append((agent.state.last_outcome, agent.state.active_run_id))
            raise RuntimeError("listener failed after observing suspension")

    unsubscribe = agent.subscribe(fail_after_observing)
    prompt_task = asyncio.create_task(agent.prompt("hi", run_id="R1"))
    await _wait_for_pending(channel)

    with suppress(RuntimeError):
        await agent.detach()
    timed_out = False
    try:
        await asyncio.wait_for(asyncio.shield(prompt_task), timeout=0.5)
    except RuntimeError:
        pass
    except TimeoutError:
        timed_out = True

    unsubscribe()
    if not prompt_task.done():
        await agent.detach()
        await prompt_task
    await _detach_tracing(detach_tracing, tracer)

    assert timed_out is False
    assert observed == [("suspended", None)]


async def test_earlier_listener_failure_cannot_hide_suspension_from_tracer() -> None:
    agent, channel, exporter, tracer = _build_suspending_agent("thread-early-error")

    async def fail_on_suspension(event, signal=None):  # noqa: ANN001, ANN202
        del signal
        if event.type == "agent_suspended":
            raise RuntimeError("earlier listener failed")

    unsubscribe = agent.subscribe(fail_on_suspension)
    detach_tracing = tracer.attach(agent)
    prompt_task = asyncio.create_task(agent.prompt("hi", run_id="R1"))
    await _wait_for_pending(channel)
    await agent.detach()
    with suppress(RuntimeError):
        await prompt_task
    unsubscribe()
    await _detach_tracing(detach_tracing, tracer)

    roots = [span for span in exporter.spans if span.name == "invoke_agent"]
    assert len(roots) == 1
    assert roots[0].attributes.get("cubepi.run.outcome") == "suspended"
    assert roots[0].attributes.get("cubepi.aborted") is not True


async def test_cancelled_listener_cannot_hide_suspension_from_tracer() -> None:
    agent, channel, exporter, tracer = _build_suspending_agent(
        "thread-cancelled-listener"
    )

    async def cancel_on_suspension(event, signal=None):  # noqa: ANN001, ANN202
        del signal
        if event.type == "agent_suspended":
            raise asyncio.CancelledError

    unsubscribe = agent.subscribe(cancel_on_suspension)
    detach_tracing = tracer.attach(agent)
    prompt_task = asyncio.create_task(agent.prompt("hi", run_id="R1"))
    await _wait_for_pending(channel)
    await agent.detach()
    with suppress(asyncio.CancelledError):
        await prompt_task
    unsubscribe()
    await _detach_tracing(detach_tracing, tracer)

    assert agent.state.last_outcome == "suspended"
    assert agent.state.active_run_id is None
    roots = [span for span in exporter.spans if span.name == "invoke_agent"]
    assert len(roots) == 1
    assert roots[0].attributes.get("cubepi.run.outcome") == "suspended"
    assert roots[0].attributes.get("cubepi.aborted") is not True


async def test_listener_cancellation_takes_priority_after_terminal_fanout() -> None:
    agent, channel, exporter, tracer = _build_suspending_agent(
        "thread-listener-priority"
    )

    async def fail_on_suspension(event, signal=None):  # noqa: ANN001, ANN202
        del signal
        if event.type == "agent_suspended":
            raise RuntimeError("regular listener failure")

    async def cancel_on_suspension(event, signal=None):  # noqa: ANN001, ANN202
        del signal
        if event.type == "agent_suspended":
            raise asyncio.CancelledError

    unsubscribe_fail = agent.subscribe(fail_on_suspension)
    unsubscribe_cancel = agent.subscribe(cancel_on_suspension)
    detach_tracing = tracer.attach(agent)
    prompt_task = asyncio.create_task(agent.prompt("hi", run_id="R1"))
    await _wait_for_pending(channel)
    await agent.detach()
    raised: BaseException | None = None
    try:
        await prompt_task
    except BaseException as exc:
        raised = exc
    unsubscribe_fail()
    unsubscribe_cancel()
    await _detach_tracing(detach_tracing, tracer)

    assert isinstance(raised, asyncio.CancelledError)
    roots = [span for span in exporter.spans if span.name == "invoke_agent"]
    assert len(roots) == 1
    assert roots[0].attributes.get("cubepi.run.outcome") == "suspended"


async def test_detach_then_owner_cancel_clears_suspension_snapshot() -> None:
    agent, channel, exporter, tracer = _build_suspending_agent("thread-owner-cancel")
    detach_tracing = tracer.attach(agent)
    prompt_task = asyncio.create_task(agent.prompt("hi", run_id="R1"))
    await _wait_for_pending(channel)
    loop = asyncio.get_running_loop()
    old_exception_handler = loop.get_exception_handler()
    unhandled: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    try:
        await agent.detach()
        prompt_task.cancel()
        with suppress(asyncio.CancelledError):
            await prompt_task
        await _detach_tracing(detach_tracing, tracer)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(old_exception_handler)

    assert agent._pending_suspension_event is None
    assert unhandled == []
    roots = [span for span in exporter.spans if span.name == "invoke_agent"]
    assert len(roots) == 1
    assert roots[0].attributes.get("cubepi.run.outcome") is None
    assert roots[0].attributes.get("cubepi.aborted") is True


async def test_reset_before_owner_resumes_does_not_drop_suspension_snapshot() -> None:
    agent, channel, exporter, tracer = _build_suspending_agent("thread-reset-race")
    detach_tracing = tracer.attach(agent)
    prompt_task = asyncio.create_task(agent.prompt("hi", run_id="R1"))
    await _wait_for_pending(channel)

    await agent.detach()
    agent.reset()
    await prompt_task
    persisted_pending = await agent.load_pending_hitl_request()
    await _detach_tracing(detach_tracing, tracer)

    assert agent.state.last_outcome == "suspended"
    assert persisted_pending is not None
    roots = [span for span in exporter.spans if span.name == "invoke_agent"]
    assert len(roots) == 1
    assert roots[0].attributes.get("cubepi.run.outcome") == "suspended"
    assert roots[0].attributes.get("cubepi.aborted") is not True


async def test_respond_after_suspension_opens_a_new_activation_trace() -> None:
    agent, channel, exporter, tracer = _build_suspending_agent("thread-respond")
    detach_tracing = tracer.attach(agent)

    prompt_task = asyncio.create_task(agent.prompt("hi", run_id="R1"))
    await _wait_for_pending(channel)
    await agent.detach()
    await prompt_task
    pending = await agent.load_pending_hitl_request()
    assert pending is not None
    await agent.respond(question_id=pending.question_id, answer="yes")
    await _detach_tracing(detach_tracing, tracer)

    roots = [span for span in exporter.spans if span.name == "invoke_agent"]
    assert len(roots) == 2
    assert roots[0].attributes.get("cubepi.run.outcome") == "suspended"
    assert roots[1].attributes.get("cubepi.run.outcome") is None
    assert roots[1].attributes.get("cubepi.aborted") is not True
    assert roots[0].context.trace_id != roots[1].context.trace_id


async def test_suspension_clears_active_run_in_owning_prompt_task() -> None:
    agent, channel, _exporter, tracer = _build_suspending_agent("thread-3")
    detach_tracing = tracer.attach(agent)

    async def prompt_and_read_active_run():
        await agent.prompt("hi", run_id="R1")
        return recorder_module._active_run.get()

    prompt_task = asyncio.create_task(prompt_and_read_active_run())
    await _wait_for_pending(channel)
    await agent.detach()
    active_run_after_prompt = await prompt_task
    await _detach_tracing(detach_tracing, tracer)

    assert active_run_after_prompt is None
