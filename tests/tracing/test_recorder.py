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
from cubepi.providers.base import BoundModel, Model, TextContent, ToolCall
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.tracing import Tracer


MODEL = Model(id="faux-1", provider_id="faux")


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
    provider = FauxProvider(provider_id="faux")
    agent = Agent(
        model=provider.model(MODEL.id),
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
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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

        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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


class TestMiddlewareProviders:
    """Recorder must also wire listeners for any provider a middleware drives
    directly (e.g. CompactionMiddleware.summary_provider) — without this the
    summarizer LLM call is invisible to the trace, even though the rest of the
    turn is recorded."""

    async def test_compaction_summary_provider_emits_chat_span(self):
        from cubepi.checkpointer.memory import MemoryCheckpointer
        from cubepi.middleware.compaction import CompactionMiddleware
        from cubepi.providers.base import UserMessage

        main = FauxProvider()
        summarizer = FauxProvider()
        # Cap at near-zero so transform_context triggers immediately.
        # Distinct ``provider`` labels on the two Models so the test can also
        # verify the root ``invoke_agent`` span stays attributed to the agent's
        # own provider, not the summarizer that fires first.
        summary_model = Model(id="summary-1", provider_id="faux-summary")
        mw = CompactionMiddleware(
            summary_model=BoundModel(provider=summarizer, spec=summary_model),
            max_tokens_before_compact=1,
            keep_recent_messages=1,
            max_summary_tokens=128,
            min_compact_messages=2,
        )
        # Pre-seed history via a checkpointer so transform_context sees a
        # message list above the compaction boundary on the first prompt.
        cp = MemoryCheckpointer()
        thread_id = "t-compaction-trace"
        await cp.append(
            thread_id,
            [
                UserMessage(content=[TextContent(text="first")]),
                faux_assistant_message("first reply"),
                UserMessage(content=[TextContent(text="second")]),
                faux_assistant_message("second reply"),
            ],
        )

        agent = Agent(
            model=BoundModel(provider=main, spec=MODEL),
            system_prompt="agent system prompt — kept on root",
            middleware=[mw],
            checkpointer=cp,
            thread_id=thread_id,
        )
        exporter = InMemoryExporter()
        tracer = Tracer(
            service_name="test-svc",
            agent_name="test-agent",
            exporters=[exporter],
        )
        tracer.attach(agent)

        summarizer.append_responses([faux_assistant_message("a brief summary")])
        main.append_responses([faux_assistant_message("ok")])

        await agent.prompt("third")
        await agent.wait_for_idle()
        await tracer.shutdown()

        chats = [s for s in exporter.spans if s.name.startswith("chat ")]
        # One chat from the summarizer, one from the agent's main turn —
        # both must be present so the run trace shows the summarizer call,
        # not just the agent's own chat.
        chat_models = sorted(_attrs(c).get("gen_ai.request.model") for c in chats)
        assert chat_models == ["faux-1", "summary-1"], chat_models

        # Both chat spans share the agent run's trace_id — the summarizer
        # call must be part of the same trace, not a sibling root.
        turn = [s for s in exporter.spans if s.name == "cubepi.turn"][0]
        for chat in chats:
            assert chat.context.trace_id == turn.context.trace_id

        # Root ``invoke_agent`` attribution must reflect the agent's own
        # provider, not the summarizer that fires first during
        # ``transform_context``. Without the per-listener gate the root
        # would carry "faux-summary" + the summarizer's system prompt.
        root = [s for s in exporter.spans if s.name.startswith("invoke_agent")][0]
        assert _attrs(root)["gen_ai.provider.name"] == "faux"

    async def test_shared_provider_still_protected_by_model_gate(self):
        # When ``summary_provider`` IS the agent's main provider (a common
        # "reuse the client, swap the model" pattern), listener-identity
        # dedupe would route the summarizer through the main listener and
        # let it clobber the root attribution. The fix gates on the model,
        # not the listener — this regression test pins that contract.
        from cubepi.checkpointer.memory import MemoryCheckpointer
        from cubepi.middleware.compaction import CompactionMiddleware
        from cubepi.providers.base import UserMessage

        shared = FauxProvider()
        agent_model = Model(id="agent-1", provider_id="faux-main")
        summary_model = Model(id="summary-1", provider_id="faux-summary")
        mw = CompactionMiddleware(
            summary_model=BoundModel(provider=shared, spec=summary_model),
            max_tokens_before_compact=1,
            keep_recent_messages=1,
            max_summary_tokens=128,
            min_compact_messages=2,
        )
        cp = MemoryCheckpointer()
        thread_id = "t-shared-provider"
        await cp.append(
            thread_id,
            [
                UserMessage(content=[TextContent(text="first")]),
                faux_assistant_message("first reply"),
                UserMessage(content=[TextContent(text="second")]),
                faux_assistant_message("second reply"),
            ],
        )

        agent = Agent(
            model=BoundModel(provider=shared, spec=agent_model),
            system_prompt="agent system prompt — kept on root",
            middleware=[mw],
            checkpointer=cp,
            thread_id=thread_id,
        )
        exporter = InMemoryExporter()
        tracer = Tracer(
            service_name="test-svc",
            agent_name="test-agent",
            exporters=[exporter],
        )
        tracer.attach(agent)

        shared.append_responses(
            [
                faux_assistant_message("a brief summary"),  # summarizer first
                faux_assistant_message("ok"),  # then agent
            ]
        )

        await agent.prompt("third")
        await agent.wait_for_idle()
        await tracer.shutdown()

        root = [s for s in exporter.spans if s.name.startswith("invoke_agent")][0]
        # Despite the summarizer firing first on the shared listener, the
        # root must stay attributed to the agent's own provider (the
        # ``unknown:`` prefix is just ``map_provider_name`` canonicalising
        # the unfamiliar test label — what matters is it's the agent's
        # ``faux-main``, not the summarizer's ``faux-summary``).
        assert _attrs(root)["gen_ai.provider.name"].endswith("faux-main")

    def test_middleware_extra_llm_calls_default_is_empty(self):
        # ``Middleware.extra_llm_calls()`` default must return an empty
        # iterable so middlewares that don't drive any LLM are zero-cost on
        # Recorder.attach.
        from cubepi.middleware.base import Middleware

        assert list(Middleware().extra_llm_calls()) == []

    async def test_degenerate_same_model_falls_back_to_first_call_wins(self):
        # Edge case: middleware declares the same (provider, model) as the
        # agent — model-based gating would otherwise skip root attribution
        # for the agent's own call too and leave the root span with the
        # placeholder ``cubepi`` provider. The recorder must exclude these
        # degenerate keys from the extra set so the run still gets a
        # concrete root attribution (even if it ends up reflecting whichever
        # call fired first).
        from cubepi.middleware.base import Middleware

        class _SameModelMiddleware(Middleware):
            def __init__(self, provider, model):
                self._p = provider
                self._m = model

            def extra_llm_calls(self):
                return [BoundModel(provider=self._p, spec=self._m)]

        provider = FauxProvider()
        agent = Agent(
            model=BoundModel(provider=provider, spec=MODEL),
            system_prompt="test",
            middleware=[_SameModelMiddleware(provider, MODEL)],
        )
        exporter = InMemoryExporter()
        tracer = Tracer(
            service_name="test-svc",
            agent_name="test-agent",
            exporters=[exporter],
        )
        tracer.attach(agent)
        provider.append_responses([faux_assistant_message("ok")])
        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        root = [s for s in exporter.spans if s.name.startswith("invoke_agent")][0]
        # Root must end up with a real provider name, not the cubepi placeholder.
        assert _attrs(root)["gen_ai.provider.name"] != "cubepi"

    async def test_middleware_extra_provider_not_baseprovider_is_skipped(self):
        # A middleware that hands the recorder a provider not derived from
        # ``BaseProvider`` (no listener registry) is skipped by
        # ``_subscribe`` — its calls simply won't be observable, but attach
        # must not error. Covers the early-return branch in ``_subscribe``
        # that codecov flagged.
        from cubepi.middleware.base import Middleware

        class _DuckProvider:
            pass

        class _DuckMiddleware(Middleware):
            def __init__(self, model):
                self._m = model

            def extra_llm_calls(self):
                return [BoundModel(provider=_DuckProvider(), spec=self._m)]

        provider = FauxProvider()
        agent = Agent(
            model=BoundModel(provider=provider, spec=MODEL),
            system_prompt="test",
            middleware=[_DuckMiddleware(Model(id="dm-1", provider_id="duck"))],
        )
        exporter = InMemoryExporter()
        tracer = Tracer(
            service_name="test-svc",
            agent_name="test-agent",
            exporters=[exporter],
        )
        tracer.attach(agent)
        provider.append_responses([faux_assistant_message("ok")])
        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()
        assert any(s.name.startswith("chat ") for s in exporter.spans)

    async def test_middleware_extra_llm_calls_legacy_tuple_is_skipped(self, caplog):
        # A third-party middleware that hasn't migrated from the
        # pre-BoundModel ``(provider, model)`` tuple contract must not kill
        # tracing attach — the recorder skips the bad entry, logs a
        # warning, and continues. Same defensive philosophy as the raising
        # and duck-provider cases below.
        from cubepi.middleware.base import Middleware

        legacy_provider = FauxProvider()
        legacy_model = Model(id="legacy-m", provider_id="legacy")

        class _LegacyTupleMiddleware(Middleware):
            def extra_llm_calls(self):
                return [(legacy_provider, legacy_model)]

        provider = FauxProvider()
        agent = Agent(
            model=BoundModel(provider=provider, spec=MODEL),
            system_prompt="test",
            middleware=[_LegacyTupleMiddleware()],
        )
        exporter = InMemoryExporter()
        tracer = Tracer(
            service_name="test-svc",
            agent_name="test-agent",
            exporters=[exporter],
        )
        with caplog.at_level("WARNING", logger="cubepi.tracing"):
            tracer.attach(agent)
        provider.append_responses([faux_assistant_message("ok")])
        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()
        assert any(s.name.startswith("chat ") for s in exporter.spans)
        assert any("non-BoundModel" in record.message for record in caplog.records), (
            "expected a warning about the legacy tuple entry"
        )

    async def test_middleware_extra_llm_calls_raising_is_swallowed(self):
        # If a middleware's ``extra_llm_calls()`` raises during attach the
        # recorder must keep going — the agent's own provider is the
        # load-bearing subscription, a buggy middleware mustn't break tracing.
        from cubepi.middleware.base import Middleware

        class _BoomMiddleware(Middleware):
            def extra_llm_calls(self):
                raise RuntimeError("boom")

        provider = FauxProvider()
        agent = Agent(
            model=BoundModel(provider=provider, spec=MODEL),
            system_prompt="test",
            middleware=[_BoomMiddleware()],
        )
        exporter = InMemoryExporter()
        tracer = Tracer(
            service_name="test-svc",
            agent_name="test-agent",
            exporters=[exporter],
        )
        # Should not raise.
        tracer.attach(agent)
        provider.append_responses([faux_assistant_message("ok")])
        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()
        # Trace still emits.
        assert any(s.name.startswith("chat ") for s in exporter.spans)


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
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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

    async def test_flush_exception_surfaces_when_body_ok(self):
        """If the post-block flush fails and the body itself did NOT
        raise, the exception must surface to the caller — same as
        what ``await detach()`` would do manually. Otherwise users
        continue past the block thinking spans landed (codex P2 on
        PR #90)."""
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        tracer = Tracer(service_name="t", agent_name="a", exporters=[])
        provider.append_responses([faux_assistant_message("ok")])

        sentinel = RuntimeError("flush exploded")

        async def _bad_flush(*_a, **_kw):
            raise sentinel

        # Patch the underlying force_flush to fail.
        import cubepi.tracing.tracer as _t_mod

        original = tracer.force_flush
        tracer.force_flush = _bad_flush  # type: ignore[method-assign]
        try:
            with __import__("pytest").raises(RuntimeError, match="flush exploded"):
                async with tracer.attached(agent):
                    await agent.prompt("hi")
                    await agent.wait_for_idle()
        finally:
            tracer.force_flush = original  # type: ignore[method-assign]
            del _t_mod

    async def test_flush_exception_suppressed_when_body_raises(self):
        """When the body raised, the flush failure must NOT mask the
        original exception — the body's exception is the real
        problem. Matches the standard contextlib pattern."""
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        tracer = Tracer(service_name="t", agent_name="a", exporters=[])

        async def _bad_flush(*_a, **_kw):
            raise RuntimeError("flush exploded")

        tracer.force_flush = _bad_flush  # type: ignore[method-assign]
        try:
            with __import__("pytest").raises(ValueError, match="body"):
                async with tracer.attached(agent):
                    raise ValueError("body")
        finally:
            pass

    async def test_combined_with_tracer_async_with(self):
        """The intended top-level idiom — Tracer + attached in one
        ``async with`` line.
        """
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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


class TestTracingContext:
    """``cubepi.tracing.tracing_context`` sets per-task tags +
    metadata that the recorder stamps onto the invoke_agent span."""

    async def test_tags_land_on_invoke_agent_span(self):
        from cubepi.tracing import tracing_context

        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("ok")])

        with tracing_context(tags=["beta-arm", "test-suite"]):
            await agent.prompt("hi")
            await agent.wait_for_idle()
        await tracer.shutdown()

        root = next(s for s in exporter.spans if s.name == "invoke_agent")
        attrs = _attrs(root)
        assert attrs.get("cubepi.tags") == ("beta-arm", "test-suite")

    async def test_metadata_keys_land_with_prefix(self):
        from cubepi.tracing import tracing_context

        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("ok")])

        with tracing_context(metadata={"user_id": "u-42", "ab_arm": "control"}):
            await agent.prompt("hi")
            await agent.wait_for_idle()
        await tracer.shutdown()

        root = next(s for s in exporter.spans if s.name == "invoke_agent")
        attrs = _attrs(root)
        assert attrs.get("cubepi.metadata.user_id") == "u-42"
        assert attrs.get("cubepi.metadata.ab_arm") == "control"

    async def test_no_context_means_no_tag_attr(self):
        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("ok")])
        await agent.prompt("hi")
        await agent.wait_for_idle()
        await tracer.shutdown()

        root = next(s for s in exporter.spans if s.name == "invoke_agent")
        attrs = _attrs(root)
        assert "cubepi.tags" not in attrs

    async def test_context_does_not_leak_across_runs(self):
        """The contextvar resets on block exit — the next run started
        outside the block must NOT carry the previous tags."""
        from cubepi.tracing import tracing_context

        agent, provider, exporter, tracer = await _build()
        provider.append_responses(
            [
                faux_assistant_message("first"),
                faux_assistant_message("second"),
            ]
        )

        with tracing_context(tags=["scoped"]):
            await agent.prompt("first")
            await agent.wait_for_idle()
        await agent.prompt("second")
        await agent.wait_for_idle()
        await tracer.shutdown()

        roots = sorted(
            [s for s in exporter.spans if s.name == "invoke_agent"],
            key=lambda s: s.start_time or 0,
        )
        assert len(roots) == 2
        assert _attrs(roots[0]).get("cubepi.tags") == ("scoped",)
        assert "cubepi.tags" not in _attrs(roots[1])

    async def test_nested_contexts_merge_additively(self):
        from cubepi.tracing import tracing_context

        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("ok")])

        with tracing_context(tags=["outer"], metadata={"k": "outer-val"}):
            with tracing_context(
                tags=["inner"], metadata={"k": "inner-val", "x": "new"}
            ):
                await agent.prompt("hi")
                await agent.wait_for_idle()
        await tracer.shutdown()

        root = next(s for s in exporter.spans if s.name == "invoke_agent")
        attrs = _attrs(root)
        assert attrs.get("cubepi.tags") == ("outer", "inner")
        assert attrs.get("cubepi.metadata.k") == "inner-val"
        assert attrs.get("cubepi.metadata.x") == "new"

    async def test_unsupported_metadata_value_is_dropped(self):
        """OTel attributes can't hold dicts or arbitrary objects."""
        from cubepi.tracing import tracing_context

        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("ok")])

        with tracing_context(
            metadata={
                "good_str": "yes",
                "good_int": 42,
                "bad_dict": {"nested": 1},
            }
        ):
            await agent.prompt("hi")
            await agent.wait_for_idle()
        await tracer.shutdown()

        root = next(s for s in exporter.spans if s.name == "invoke_agent")
        attrs = _attrs(root)
        assert attrs.get("cubepi.metadata.good_str") == "yes"
        assert attrs.get("cubepi.metadata.good_int") == 42
        assert "cubepi.metadata.bad_dict" not in attrs

    async def test_metadata_set_attribute_typeerror_is_swallowed(self, monkeypatch):
        """OTel SDK silently drops most invalid attribute values, but
        a non-conforming type could in principle raise TypeError /
        ValueError from ``set_attribute``. The recorder must swallow
        per-key so one bad metadata entry can't crash the whole span
        (covers the defensive ``except (TypeError, ValueError)``
        branch)."""
        from cubepi.tracing import tracing_context
        from opentelemetry.sdk.trace import Span as _SdkSpan

        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("ok")])

        # Patch set_attribute to raise on a specific cubepi.metadata.*
        # key while letting every other attribute go through. The
        # recorder writes many cubepi.* / gen_ai.* attrs at agent
        # start, so we need to be surgical.
        original = _SdkSpan.set_attribute

        def _selective_set_attribute(self, key, value):
            if key == "cubepi.metadata.boom":
                raise TypeError("simulated OTel reject")
            return original(self, key, value)

        monkeypatch.setattr(_SdkSpan, "set_attribute", _selective_set_attribute)

        with tracing_context(metadata={"boom": object(), "good": "yes"}):
            await agent.prompt("hi")
            await agent.wait_for_idle()
        await tracer.shutdown()

        root = next(s for s in exporter.spans if s.name == "invoke_agent")
        attrs = _attrs(root)
        # bad key swallowed; good key landed.
        assert "cubepi.metadata.boom" not in attrs
        assert attrs.get("cubepi.metadata.good") == "yes"

    async def test_metadata_cannot_clobber_reserved_cubepi_attrs(self):
        """User-supplied metadata keys must not be able to overwrite
        recorder-owned schema attributes like ``cubepi.run_id`` —
        the JSONL exporter shards spans by that key, so an
        invoke_agent span with a clobbered ``run_id`` ends up in a
        different file than its turn/chat/tool spans (codex P2 on
        PR #92). The fix is the ``cubepi.metadata.*`` sub-namespace."""
        from cubepi.tracing import tracing_context

        agent, provider, exporter, tracer = await _build()
        provider.append_responses([faux_assistant_message("ok")])

        with tracing_context(metadata={"run_id": "hijacked", "turn.index": "x"}):
            await agent.prompt("hi")
            await agent.wait_for_idle()
        await tracer.shutdown()

        root = next(s for s in exporter.spans if s.name == "invoke_agent")
        attrs = _attrs(root)
        # The genuine recorder-owned value is unchanged…
        assert attrs.get("cubepi.run_id") and attrs["cubepi.run_id"] != "hijacked"
        # …and the user-supplied values land under the safe namespace.
        assert attrs.get("cubepi.metadata.run_id") == "hijacked"
        assert attrs.get("cubepi.metadata.turn.index") == "x"


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


class TestAtexitFlush:
    """``Tracer(atexit_flush=True)`` registers a process-exit hook
    that sync-flushes any buffered spans through BatchSpanProcessor.
    Safety net for callers who forget ``await tracer.shutdown()`` —
    matches the Traceloop SDK pattern. atexit doesn't run on
    SIGKILL / os._exit but covers normal Ctrl-C / unhandled
    exception / sys.exit paths."""

    def test_registers_atexit_when_enabled(self, monkeypatch):
        registered: list = []
        unregistered: list = []
        import atexit as _atexit

        monkeypatch.setattr(_atexit, "register", lambda f: registered.append(f))
        monkeypatch.setattr(_atexit, "unregister", lambda f: unregistered.append(f))

        tracer = Tracer(service_name="t", exporters=[])
        assert tracer._atexit_flush in registered
        assert tracer._atexit_unregister is not None
        # Sanity: the unregister callback removes the same function.
        tracer._atexit_unregister()
        assert tracer._atexit_flush in unregistered

    def test_disabled_when_opted_out(self, monkeypatch):
        registered: list = []
        import atexit as _atexit

        monkeypatch.setattr(_atexit, "register", lambda f: registered.append(f))
        tracer = Tracer(service_name="t", exporters=[], atexit_flush=False)
        assert tracer._atexit_flush not in registered
        assert tracer._atexit_unregister is None

    async def test_atexit_flush_calls_provider_force_flush(self):
        flushed: list = []

        class _FakeProvider:
            def force_flush(self, timeout_millis: int = 30_000) -> bool:
                flushed.append(timeout_millis)
                return True

            def shutdown(self) -> None:
                pass

        tracer = Tracer(service_name="t", exporters=[], atexit_flush=False)
        tracer._provider = _FakeProvider()  # type: ignore[assignment]
        tracer._atexit_flush_timeout_ms = 5000
        tracer._atexit_flush()
        assert flushed == [5000]

    async def test_atexit_flush_noop_after_shutdown(self):
        flushed: list = []

        class _FakeProvider:
            def force_flush(self, timeout_millis: int = 30_000) -> bool:
                flushed.append(timeout_millis)
                return True

            def shutdown(self) -> None:
                pass

        tracer = Tracer(service_name="t", exporters=[], atexit_flush=False)
        tracer._provider = _FakeProvider()  # type: ignore[assignment]
        await tracer.shutdown()
        flushed.clear()
        tracer._atexit_flush()
        assert flushed == [], "atexit hook must no-op after explicit shutdown"

    def test_atexit_flush_swallows_exceptions(self):
        """atexit handlers must NEVER raise — would corrupt
        interpreter shutdown for every other registered handler."""

        class _BoomProvider:
            def force_flush(self, timeout_millis: int = 30_000) -> bool:
                raise RuntimeError("flush boom")

            def shutdown(self) -> None:
                pass

        tracer = Tracer(service_name="t", exporters=[], atexit_flush=False)
        tracer._provider = _BoomProvider()  # type: ignore[assignment]
        # Must NOT raise.
        tracer._atexit_flush()

    async def test_shutdown_swallows_unregister_failure(self, monkeypatch):
        """If ``atexit.unregister`` raises (e.g. mocked away or Python
        version edge case), ``shutdown()`` must still complete — the
        unregister is best-effort cleanup."""
        import atexit as _atexit

        def _bad_unregister(_f):
            raise RuntimeError("unregister boom")

        monkeypatch.setattr(_atexit, "register", lambda f: None)
        monkeypatch.setattr(_atexit, "unregister", _bad_unregister)
        tracer = Tracer(service_name="t", exporters=[])
        # Replace the unregister callback with one that hits the
        # patched atexit.unregister.
        tracer._atexit_unregister = lambda: _atexit.unregister(tracer._atexit_flush)
        # Must NOT raise.
        await tracer.shutdown()

    async def test_shutdown_unregisters_atexit_hook(self, monkeypatch):
        unregistered: list = []
        import atexit as _atexit

        monkeypatch.setattr(_atexit, "register", lambda f: None)
        monkeypatch.setattr(_atexit, "unregister", lambda f: unregistered.append(f))

        tracer = Tracer(service_name="t", exporters=[])
        hook = tracer._atexit_flush
        await tracer.shutdown()
        assert hook in unregistered, (
            "shutdown() must unregister the atexit hook to keep the "
            "atexit table from growing across many Tracer instances"
        )
        # Idempotent: a second shutdown does not double-unregister.
        unregistered.clear()
        await tracer.shutdown()
        assert unregistered == []


class TestJsonlExporter:
    async def test_writes_jsonl_files(self, tmp_path):
        from cubepi.tracing.exporters import JsonlSpanExporter

        provider = FauxProvider(provider_id="faux")
        provider.append_responses([faux_assistant_message("ok")])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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
