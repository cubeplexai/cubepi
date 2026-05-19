"""End-to-end tracing tests against FauxProvider.

Each test attaches a Tracer + an in-memory exporter to a fresh Agent,
runs a scenario, then inspects the captured spans.
"""

from __future__ import annotations

import asyncio
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind, StatusCode

from cubepi.agent.agent import Agent
from cubepi.agent.types import (
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
)
from cubepi.providers.base import Model, TextContent, ToolCall
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.tracing import Tracer


MODEL = Model(id="faux-1", provider="faux")


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


async def _build(
    *,
    system_prompt: str = "test prompt",
    tools: list[AgentTool] | None = None,
    before_tool_call=None,
) -> tuple[Agent, FauxProvider, InMemoryExporter, Tracer]:
    provider = FauxProvider()
    agent = Agent(
        provider=provider,
        model=MODEL,
        system_prompt=system_prompt,
        tools=tools,
        before_tool_call=before_tool_call,
    )
    exporter = InMemoryExporter()
    tracer = Tracer(
        service_name="test-svc",
        agent_name="test-agent",
        exporters=[exporter],
    )
    tracer.attach(agent)
    return agent, provider, exporter, tracer


def _spans_by_name(exporter: InMemoryExporter) -> dict[str, list[ReadableSpan]]:
    out: dict[str, list[ReadableSpan]] = {}
    for s in exporter.spans:
        out.setdefault(s.name, []).append(s)
    return out


def _find_attached_recorder(provider):
    """Walk a provider's request_listeners to return the cubepi
    Recorder instance — used by tests that need to drive recorder
    callbacks directly with synthetic provider events."""
    from cubepi.tracing.recorder import Recorder

    for cb in getattr(provider, "_request_listeners", []):
        if hasattr(cb, "__self__") and isinstance(cb.__self__, Recorder):
            return cb.__self__
    return None


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


class TestSpanTreeBasic:
    async def test_simple_run_emits_three_spans(self):
        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("hello")])

        await agent.prompt("hi")
        await agent.wait_for_idle()
        await tracer.shutdown()

        names = sorted(s.name for s in exporter.spans)
        assert "invoke_agent" in names
        assert "cubepi.turn" in names
        assert any(n.startswith("chat ") for n in names)
        assert len(exporter.spans) == 3

    async def test_root_invoke_agent_attrs(self):
        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("ok")])

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        roots = [s for s in exporter.spans if s.name == "invoke_agent"]
        assert len(roots) == 1
        attrs = _attrs(roots[0])
        assert attrs["gen_ai.operation.name"] == "invoke_agent"
        assert attrs["gen_ai.provider.name"] == "faux"
        assert "cubepi.run_id" in attrs
        assert "cubepi.agent.system_prompt.sha256" in attrs
        assert roots[0].kind == SpanKind.INTERNAL


class TestChatSpan:
    async def test_chat_span_is_client_and_carries_gen_ai_attrs(self):
        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("hi")])

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        chats = [s for s in exporter.spans if s.name.startswith("chat ")]
        assert len(chats) == 1
        chat = chats[0]
        assert chat.kind == SpanKind.CLIENT
        attrs = _attrs(chat)
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.provider.name"] == "faux"
        assert attrs["gen_ai.request.model"] == MODEL.id
        assert attrs["gen_ai.request.stream"] is True
        assert "gen_ai.usage.input_tokens" in attrs
        assert "gen_ai.usage.output_tokens" in attrs
        assert attrs["gen_ai.response.finish_reasons"] == ("stop",)


class TestTurnSpan:
    async def test_turn_span_no_operation_name(self):
        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("ok")])

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        turn = [s for s in exporter.spans if s.name == "cubepi.turn"][0]
        attrs = _attrs(turn)
        assert "gen_ai.operation.name" not in attrs
        assert "gen_ai.workflow.name" not in attrs
        assert attrs["cubepi.turn.index"] == 0
        assert attrs["cubepi.turn.stop_reason"] == "stop"

    async def test_multi_turn_indexes_increment(self):
        from pydantic import BaseModel

        class P(BaseModel):
            pass

        async def echo(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="done")])

        tool = AgentTool(name="echo", description="echo", parameters=P, execute=echo)

        agent, provider, exporter, tracer = await _build(tools=[tool])
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="t1", name="echo", arguments={})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        turns = sorted(
            [s for s in exporter.spans if s.name == "cubepi.turn"],
            key=lambda s: s.start_time or 0,
        )
        assert len(turns) == 2
        assert _attrs(turns[0])["cubepi.turn.index"] == 0
        assert _attrs(turns[1])["cubepi.turn.index"] == 1


class TestExecuteToolSpan:
    async def test_tool_span_attrs(self):
        from pydantic import BaseModel

        class P(BaseModel):
            pass

        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="done")])

        tool = AgentTool(
            name="echo",
            description="echo a thing",
            parameters=P,
            execute=run,
        )

        agent, provider, exporter, tracer = await _build(tools=[tool])
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="t1", name="echo", arguments={})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        tool_spans = [s for s in exporter.spans if s.name.startswith("execute_tool ")]
        assert len(tool_spans) == 1
        t = tool_spans[0]
        assert t.kind == SpanKind.INTERNAL
        attrs = _attrs(t)
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert attrs["gen_ai.tool.name"] == "echo"
        assert attrs["gen_ai.tool.call.id"] == "t1"
        assert attrs["gen_ai.tool.type"] == "function"
        # description and execution_mode come from the AgentTool — the
        # Recorder reads them off the agent's tool registry at exec
        # start.
        assert attrs["gen_ai.tool.description"] == "echo a thing"
        assert attrs["cubepi.tool.execution_mode"] in {"parallel", "sequential"}
        assert attrs["cubepi.tool.is_error"] is False
        assert "cubepi.tool.terminate" not in attrs

    async def test_tool_terminate_flag_propagates(self):
        from pydantic import BaseModel

        class P(BaseModel):
            pass

        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="done")], terminate=True)

        tool = AgentTool(name="t", description="t", parameters=P, execute=run)

        agent, provider, exporter, tracer = await _build(tools=[tool])
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="t1", name="t", arguments={})],
                    stop_reason="tool_use",
                ),
            ]
        )

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        t_span = [s for s in exporter.spans if s.name.startswith("execute_tool ")][0]
        assert _attrs(t_span)["cubepi.tool.terminate"] is True
        turn = [s for s in exporter.spans if s.name == "cubepi.turn"][0]
        assert _attrs(turn)["cubepi.turn.terminated_by_tool"] is True

    async def test_tool_blocked_by_hook(self):
        from pydantic import BaseModel

        class P(BaseModel):
            pass

        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="never")])

        async def before(ctx, *, signal=None):
            return BeforeToolCallResult(block=True, reason="no thanks")

        tool = AgentTool(name="g", description="g", parameters=P, execute=run)

        agent, provider, exporter, tracer = await _build(
            tools=[tool], before_tool_call=before
        )
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="t1", name="g", arguments={})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("acknowledged"),
            ]
        )

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        t_span = [s for s in exporter.spans if s.name.startswith("execute_tool ")][0]
        attrs = _attrs(t_span)
        assert attrs["cubepi.tool.is_error"] is True
        assert attrs["cubepi.tool.blocked_by_hook"] is True
        assert attrs["cubepi.tool.block_reason"] == "no thanks"
        assert t_span.status.status_code == StatusCode.ERROR
        assert attrs["error.type"] == "cubepi.tool.blocked_by_hook"


class TestParentChild:
    async def test_chat_and_tool_share_turn_parent(self):
        from pydantic import BaseModel

        class P(BaseModel):
            pass

        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="done")])

        tool = AgentTool(name="t", description="t", parameters=P, execute=run)

        agent, provider, exporter, tracer = await _build(tools=[tool])
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="t1", name="t", arguments={})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        trace_ids = {s.context.trace_id for s in exporter.spans}
        assert len(trace_ids) == 1

        by_name = _spans_by_name(exporter)
        root = by_name["invoke_agent"][0]
        assert root.parent is None
        for turn in by_name["cubepi.turn"]:
            assert turn.parent is not None
            assert turn.parent.span_id == root.context.span_id
        for chat in [s for s in exporter.spans if s.name.startswith("chat ")]:
            assert chat.parent is not None
            assert chat.parent.span_id in {
                t.context.span_id for t in by_name["cubepi.turn"]
            }


class TestErrorAndAbort:
    async def test_chat_span_records_provider_exception(self):
        agent, provider, exporter, tracer = await _build()

        async def boom(messages, model, system_prompt, tools):
            raise RuntimeError("provider boom")

        provider.append_responses([boom])

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        chats = [s for s in exporter.spans if s.name.startswith("chat ")]
        assert len(chats) == 1
        chat = chats[0]
        assert chat.status.status_code == StatusCode.ERROR
        attrs = _attrs(chat)
        assert "error.type" in attrs
        evnames = [e.name for e in chat.events]
        assert "gen_ai.client.operation.exception" in evnames

    async def test_agent_aborted_marks_cubepi_aborted(self):
        # Slow chunking so the abort signal lands MID-stream.
        provider = FauxProvider(tokens_per_second=10.0)
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = InMemoryExporter()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        tracer.attach(agent)

        provider.append_responses([faux_assistant_message("x" * 600)])

        # ``agent.prompt`` blocks until the run finishes, so kick it off
        # as a background task; abort while it's still streaming.
        run = asyncio.create_task(agent.prompt("hi"))
        await asyncio.sleep(0.1)
        agent.abort()
        await run
        await tracer.shutdown()

        root = [s for s in exporter.spans if s.name == "invoke_agent"][0]
        attrs = _attrs(root)
        assert attrs.get("cubepi.aborted") is True

    async def test_chat_span_marks_aborted_when_provider_returns_no_body(self):
        """Real Anthropic/OpenAI/OpenAI-Responses abort branches return
        from the stream before assembling a body, so the response
        listener is called as ``(None, model, None)``. Without an
        explicit case for that the chat span would close UNSET — out
        of sync with the turn/root which TurnEnd marks aborted (codex
        P2 round on PR #82).

        The (None, None) shape is only treated as abort when the
        agent's ``_active_signal`` is set — otherwise it's a benign
        provider-side fallback (e.g. OpenAIResponsesProvider
        finalizing an incomplete stream). See the companion
        ``test_no_body_without_signal_keeps_chat_unset`` below.

        Drive the recorder's listeners directly with the (None, None)
        shape — reproducing real provider behaviour without needing
        the provider to actually cooperate."""
        import asyncio as _asyncio

        provider = FauxProvider()
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = InMemoryExporter()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        tracer.attach(agent)

        # First, do a real run so a turn span is open via normal flow.
        provider.append_responses([faux_assistant_message("warmup")])
        await agent.prompt("warmup")

        recorder = _find_attached_recorder(provider)
        assert recorder is not None
        from cubepi.agent.types import AgentStartEvent, TurnStartEvent

        await recorder._on_agent_event(AgentStartEvent())
        await recorder._on_agent_event(TurnStartEvent())
        recorder._on_provider_request({"messages": []}, MODEL)
        # Simulate that the user called agent.abort() before the
        # response listener fires.
        agent._active_signal = _asyncio.Event()
        agent._active_signal.set()
        # Provider abort branch -> (body=None, exc=None).
        recorder._on_provider_response(None, MODEL, None)

        await tracer.shutdown()
        chats = [s for s in exporter.spans if s.name.startswith("chat ")]
        assert len(chats) >= 2
        aborted_chat = chats[-1]
        attrs = _attrs(aborted_chat)
        assert attrs.get("cubepi.aborted") is True
        assert attrs.get("error.type") == "cubepi.aborted"
        assert aborted_chat.status.status_code == StatusCode.UNSET

    async def test_no_body_without_signal_keeps_chat_unset(self):
        """``OpenAIResponsesProvider`` finalizes a stream that ends
        without ``response.completed`` by firing response listeners
        with ``(body=None, exc=None)`` — a successful (if incomplete)
        completion. The chat span MUST NOT be marked aborted in that
        case; only a (None, None) shape coinciding with the agent's
        abort signal counts as a real abort (codex P2 follow-up on
        PR #87)."""
        provider = FauxProvider()
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = InMemoryExporter()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        tracer.attach(agent)

        provider.append_responses([faux_assistant_message("warmup")])
        await agent.prompt("warmup")

        recorder = _find_attached_recorder(provider)
        assert recorder is not None
        from cubepi.agent.types import AgentStartEvent, TurnStartEvent

        await recorder._on_agent_event(AgentStartEvent())
        await recorder._on_agent_event(TurnStartEvent())
        recorder._on_provider_request({"messages": []}, MODEL)
        # Provider's non-abort fallback path — no signal set.
        agent._active_signal = None
        recorder._on_provider_response(None, MODEL, None)

        await tracer.shutdown()
        chats = [s for s in exporter.spans if s.name.startswith("chat ")]
        assert len(chats) >= 2
        fallback_chat = chats[-1]
        attrs = _attrs(fallback_chat)
        assert "cubepi.aborted" not in attrs, (
            "non-abort (None, None) fallback must not be marked aborted"
        )
        assert attrs.get("error.type") != "cubepi.aborted"
        assert fallback_chat.status.status_code == StatusCode.UNSET


class TestSafeToolName:
    """``_safe_tool_name`` must handle all three tool payload shapes:
    Anthropic/cubepi top-level ``name``, OpenAI Responses top-level
    ``name``, and OpenAI Chat's nested ``{type: function, function:
    {name: ...}}``. Without the nested branch the root span's tool
    list was filled with ``[""]`` for OpenAI Chat (codex
    overall-review MINOR)."""

    def test_top_level_name(self):
        from cubepi.tracing.recorder import _safe_tool_name

        assert _safe_tool_name({"name": "search"}) == "search"

    def test_openai_chat_nested_function_shape(self):
        from cubepi.tracing.recorder import _safe_tool_name

        assert (
            _safe_tool_name({"type": "function", "function": {"name": "fetch"}})
            == "fetch"
        )

    def test_object_attribute(self):
        from cubepi.tracing.recorder import _safe_tool_name

        class _T:
            name = "calc"

        assert _safe_tool_name(_T()) == "calc"

    def test_missing_name_returns_empty(self):
        from cubepi.tracing.recorder import _safe_tool_name

        assert _safe_tool_name({}) == ""
        assert _safe_tool_name({"type": "function"}) == ""


class TestRequestMaxTokensCrossProvider:
    """OpenAI Responses uses ``max_output_tokens`` while
    chat-completions / Anthropic use ``max_tokens``. The recorder
    must capture either into ``gen_ai.request.max_tokens`` so the
    attribute is consistent across providers (codex overall-review
    MINOR)."""

    async def test_max_output_tokens_lands_in_request_max_tokens(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = InMemoryExporter()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        tracer.attach(agent)

        recorder = _find_attached_recorder(provider)
        assert recorder is not None
        from cubepi.agent.types import AgentStartEvent, TurnStartEvent

        await recorder._on_agent_event(AgentStartEvent())
        await recorder._on_agent_event(TurnStartEvent())
        # Simulate an OpenAI Responses request payload shape.
        recorder._on_provider_request(
            {"messages": [], "max_output_tokens": 4096},
            MODEL,
        )
        recorder._on_provider_response({"model": "faux-1"}, MODEL, None)
        await tracer.shutdown()

        chats = [s for s in exporter.spans if s.name.startswith("chat ")]
        assert chats, "no chat span captured"
        # The most recent chat span carries the value.
        attrs = _attrs(chats[-1])
        assert attrs.get("gen_ai.request.max_tokens") == 4096


class TestDetachFlushGuarantee:
    """``Tracer.attach()``'s ``detach()`` must let callers await the
    flush so buffered spans land before they proceed — previously the
    flush was scheduled fire-and-forget and the caller could exit
    asyncio.run before BatchSpanProcessor drained (codex overall-
    review MAJOR)."""

    async def test_detach_returns_awaitable_task(self):
        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("ok")])
        await agent.prompt("x")
        await agent.wait_for_idle()

        # Replicate Tracer.attach's detach: it should return a Task.
        detach = tracer.attach(agent)
        # `attach()` already happened in _build, so detach has been
        # consumed. Build a fresh attach here.
        await agent.prompt("y")
        await agent.wait_for_idle()
        result = detach()
        assert result is not None, (
            "detach() inside running loop must return a flush Task; got None"
        )
        assert asyncio.isfuture(result) or asyncio.iscoroutine(result), (
            f"detach() must return an awaitable; got {type(result).__name__}"
        )
        # Awaiting it must complete the flush.
        flushed = await result
        # force_flush returns a bool — exporter should have spans.
        del flushed
        await tracer.shutdown()
        assert exporter.spans, "no spans landed after awaited detach"

    def test_detach_outside_loop_returns_none(self):
        # Build a Tracer + Agent in sync context (no running loop).
        provider = FauxProvider()
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        tracer = Tracer(service_name="t", agent_name="a", exporters=[])
        detach = tracer.attach(agent)
        # No running loop → flush is the caller's responsibility.
        assert detach() is None


class TestCancellationExportsSpans:
    """``asyncio.CancelledError`` bypasses cubepi's agent-loop
    ``except Exception``, so no AgentEnd / TurnEnd / ToolExecutionEnd
    event is emitted on cancel. Without an explicit close path the
    open spans never reach ``span.end()`` and ``BatchSpanProcessor``
    drops them — cancelled runs simply disappear from the backend
    (codex overall-review BLOCKING)."""

    async def test_cancelled_run_still_exports_invoke_agent_span(self):
        provider = FauxProvider(tokens_per_second=10.0)
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = InMemoryExporter()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        detach = tracer.attach(agent)

        provider.append_responses([faux_assistant_message("x" * 600)])

        run = asyncio.create_task(agent.prompt("hi"))
        await asyncio.sleep(0.05)
        run.cancel()
        try:
            await run
        except asyncio.CancelledError:
            pass

        detach()
        await tracer.shutdown()

        # invoke_agent + cubepi.turn + chat must all have been ended
        # and exported, even though no AgentEnd/TurnEnd fired.
        names = {s.name for s in exporter.spans}
        assert "invoke_agent" in names, (
            f"cancelled run's invoke_agent span not exported; got {names}"
        )
        assert "cubepi.turn" in names
        assert any(n.startswith("chat ") for n in names)

        # Each must carry cubepi.aborted so the backend sees the
        # interruption rather than thinking the run completed.
        for span in exporter.spans:
            if span.name in ("invoke_agent", "cubepi.turn") or span.name.startswith(
                "chat "
            ):
                attrs = _attrs(span)
                assert attrs.get("cubepi.aborted") is True, (
                    f"{span.name} missing cubepi.aborted after cancellation"
                )


class TestCloseOpenSpansDefensive:
    """``_close_open_spans`` swallows any exception raised while
    marking a span aborted — the cleanup must continue across all
    open spans even if one raises. Pin those defensive branches
    (codex overall-review BLOCKING follow-up: pure coverage)."""

    def test_close_open_spans_swallows_set_attribute_errors(self):
        from cubepi.tracing.recorder import Recorder, _RunState

        tracer = Tracer(service_name="t", exporters=[])
        recorder = Recorder(tracer)

        class _BoomSpan:
            def set_attribute(self, *_a, **_kw):
                raise RuntimeError("set_attribute boom")

            def end(self):
                raise RuntimeError("end boom")

        run = _RunState(run_id="r", agent_span=_BoomSpan())
        run.tool_spans = {"tc1": _BoomSpan(), "tc2": _BoomSpan()}
        run.chat_span = _BoomSpan()
        run.turn_span = _BoomSpan()

        # All four branches' exception handlers must fire without
        # propagating. Post-condition: tool_spans cleared, others
        # nulled, no exception raised.
        recorder._close_open_spans(run)
        assert run.tool_spans == {}
        assert run.chat_span is None
        assert run.turn_span is None


class TestAgentSignalHelper:
    """Unit-level coverage for the defensive branches of
    ``Recorder._agent_signal_is_set`` introduced in PR #87. The
    main-path branch (signal is set / not set) is already exercised
    by the abort-handling chat-span tests; here we pin the
    no-agent and signal-raises edges so codecov accepts the patch."""

    def test_returns_false_when_no_agent_attached(self):
        from cubepi.tracing.recorder import Recorder

        tracer = Tracer(service_name="t", exporters=[])
        recorder = Recorder(tracer)
        # Default: no agent attached.
        assert recorder._agent_signal_is_set() is False

    def test_returns_false_when_agent_has_no_signal(self):
        from cubepi.tracing.recorder import Recorder

        tracer = Tracer(service_name="t", exporters=[])
        recorder = Recorder(tracer)

        class _Bare:
            pass  # no _active_signal attribute

        recorder._agent = _Bare()
        assert recorder._agent_signal_is_set() is False

    def test_returns_false_when_signal_is_set_raises(self):
        from cubepi.tracing.recorder import Recorder

        tracer = Tracer(service_name="t", exporters=[])
        recorder = Recorder(tracer)

        class _Boom:
            def is_set(self):
                raise RuntimeError("broken signal")

        class _Agent:
            _active_signal = _Boom()

        recorder._agent = _Agent()
        # Exception is swallowed; helper degrades to False rather than
        # crashing the response-listener callback chain.
        assert recorder._agent_signal_is_set() is False


class TestTranscriptSeedingDefensiveBranches:
    """The transcript-seeding path on ``_on_agent_start`` reads
    ``agent.state.messages`` defensively — pin the no-state and
    raising-state branches so the patch is fully covered."""

    async def test_no_seed_when_agent_has_no_state(self):
        from cubepi.tracing.recorder import Recorder
        from cubepi.agent.types import AgentStartEvent

        tracer = Tracer(service_name="t", exporters=[])
        recorder = Recorder(tracer)

        class _Bare:
            pass  # no `state` attribute

        recorder._agent = _Bare()
        await recorder._on_agent_event(AgentStartEvent())
        # _RunState was created (otherwise other handlers crash) but
        # transcript stays empty.
        assert recorder._run is not None
        assert recorder._run.transcript == []

    async def test_seed_handles_exception_in_state_messages(self):
        from cubepi.tracing.recorder import Recorder
        from cubepi.agent.types import AgentStartEvent

        tracer = Tracer(service_name="t", exporters=[])
        recorder = Recorder(tracer)

        class _BoomState:
            @property
            def messages(self):
                raise RuntimeError("state broken")

        class _Agent:
            state = _BoomState()

        recorder._agent = _Agent()
        await recorder._on_agent_event(AgentStartEvent())
        assert recorder._run is not None
        assert recorder._run.transcript == []


class TestAttachedContextManager:
    """``Tracer.attached(agent)`` is the RAII wrapper around
    ``attach`` / ``detach``. Pin: the body runs with the recorder
    attached, the exit path runs the same cleanup as a manual
    ``detach()`` call (closes any cancelled-run spans, schedules and
    awaits the flush), and the next attach on the same agent doesn't
    pile up handlers."""

    async def test_basic_usage(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = InMemoryExporter()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        provider.append_responses([faux_assistant_message("ok")])

        async with tracer.attached(agent):
            await agent.prompt("hi")
            await agent.wait_for_idle()

        # After the block, spans have been flushed.
        names = {s.name for s in exporter.spans}
        assert "invoke_agent" in names
        assert "cubepi.turn" in names
        assert any(n.startswith("chat ") for n in names)

    async def test_cancellation_inside_block_still_closes_spans(self):
        provider = FauxProvider(tokens_per_second=10.0)
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = InMemoryExporter()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        provider.append_responses([faux_assistant_message("x" * 400)])

        async with tracer.attached(agent):
            run = asyncio.create_task(agent.prompt("hi"))
            await asyncio.sleep(0.05)
            run.cancel()
            try:
                await run
            except asyncio.CancelledError:
                pass
        # The async exit ran detach + awaited flush.
        names = {s.name for s in exporter.spans}
        assert "invoke_agent" in names, (
            f"cancelled run's invoke_agent span did not export; got {names}"
        )
        # Each open-at-cancel span carries cubepi.aborted.
        for span in exporter.spans:
            if span.name == "invoke_agent" or span.name == "cubepi.turn":
                attrs = _attrs(span)
                assert attrs.get("cubepi.aborted") is True

    async def test_exception_inside_block_still_detaches(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        tracer = Tracer(service_name="t", agent_name="a", exporters=[])

        try:
            async with tracer.attached(agent):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        # Exception propagated; the recorder must be detached so it
        # can be attached again to a different agent without piling up
        # subscriptions on the original.
        async with tracer.attached(agent):
            pass  # no-op; should not error or warn

    async def test_combined_with_tracer_async_with(self):
        """The intended top-level idiom — Tracer + attached in one
        ``async with`` line.
        """
        provider = FauxProvider()
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        provider.append_responses([faux_assistant_message("ok")])
        exporter = InMemoryExporter()

        async with (
            Tracer(service_name="t", agent_name="a", exporters=[exporter]) as tracer,
            tracer.attached(agent),
        ):
            await agent.prompt("x")
            await agent.wait_for_idle()
        # Tracer is shut down; further use raises.
        assert tracer._shutdown is True
        assert exporter.spans


class TestLifecycle:
    async def test_shutdown_is_idempotent(self):
        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("ok")])

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()
        await tracer.shutdown()

    async def test_record_content_true_is_accepted(self):
        # Phase 2: record_content=True is supported. No exception.
        tracer = Tracer(service_name="s", record_content=True, exporters=[])
        assert tracer is not None


class TestJsonlExporter:
    async def test_writes_jsonl_files(self, tmp_path):
        from cubepi.tracing.exporters import JsonlSpanExporter

        provider = FauxProvider()
        provider.append_responses([faux_assistant_message("ok")])
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = JsonlSpanExporter(directory=tmp_path)
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        tracer.attach(agent)

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        files = list(tmp_path.rglob("*.jsonl"))
        assert files, "expected at least one jsonl file"
        lines = []
        for f in files:
            lines.extend(line for line in f.read_text().splitlines() if line.strip())
        import json as _json

        for line in lines:
            d = _json.loads(line)
            assert "name" in d
            assert "context" in d
