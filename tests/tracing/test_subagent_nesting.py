"""Subagent nesting: a provider shared by a parent and an inner agent
must not let the parent recorder's chat-span listener fire for the
inner agent's LLM calls.

Both agents share ONE FauxProvider instance, and both attach the SAME
Tracer. Provider listeners are per-provider-instance, so without the
per-task active-run gate the parent recorder's ``_on_provider_request``
fires for the inner agent's call too — minting a duplicate chat span
under the parent's turn. These tests pin that the gate routes each
listener only to the run that owns the calling asyncio task.

The harness mirrors ``test_recorder.py``: a real ``Agent`` + ``Tracer``
+ in-memory exporter, driven through ``agent.prompt`` against a
``FauxProvider`` whose queued responses are consumed FIFO across BOTH
agents in cross-agent call order.
"""

from __future__ import annotations

from asyncio import CancelledError

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
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


class _Empty(BaseModel):
    pass


def _make_agent(provider: FauxProvider, tools: list[AgentTool] | None) -> Agent:
    return Agent(
        provider=provider,
        model=MODEL,
        system_prompt="test prompt",
        tools=tools,
    )


async def test_shared_provider_does_not_double_mint_inner_chat():
    provider = FauxProvider()
    exporter = InMemoryExporter()
    tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])

    # Cross-agent CALL order: parent tool-call to `spawn`, inner-final,
    # parent-final. Consumed FIFO across both agents.
    provider.append_responses(
        [
            faux_assistant_message(
                [ToolCall(id="tc1", name="spawn", arguments={})],
                stop_reason="tool_use",
            ),
            faux_assistant_message("inner-final"),
            faux_assistant_message("parent-final"),
        ]
    )

    async def _spawn(tool_call_id, args, *, signal=None, on_update=None):
        inner = _make_agent(provider, tools=[])
        detach = tracer.attach(inner)
        try:
            await inner.prompt("inner task")
            await inner.wait_for_idle()
        finally:
            res = detach()
            if res is not None:
                await res
        return AgentToolResult(content=[TextContent(text="ok")])

    spawn = AgentTool(
        name="spawn", description="spawn inner", parameters=_Empty, execute=_spawn
    )
    parent = _make_agent(provider, tools=[spawn])
    detach_parent = tracer.attach(parent)
    try:
        await parent.prompt("go")
        await parent.wait_for_idle()
    finally:
        res = detach_parent()
        if res is not None:
            await res
    await tracer.shutdown()

    chat_spans = [
        s
        for s in exporter.spans
        if (s.attributes or {}).get("gen_ai.operation.name") == "chat"
    ]
    assert len(chat_spans) == 3, f"expected 3 chat spans, got {len(chat_spans)}"


async def test_cancelled_inner_gate_released_for_parents_next_turn():
    """A cancelled inner run must release the active-run gate so the parent's
    next turn is recorded. The gate-reset path that matters here is `detach()`
    -> `_close_open_spans` -> `_reset_active_run` (the inner run may skip
    AgentEndEvent on cancellation). Asserting the parent run has 2 chat spans
    proves both that the gate was released AND that no inner chat leaked onto
    the parent run.
    """
    provider = FauxProvider()
    exporter = InMemoryExporter()
    tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])

    def _raise_cancelled(messages, model, system_prompt, tools):
        raise CancelledError()

    # parent tool-call, inner provider call raises CancelledError, parent-final.
    provider.append_responses(
        [
            faux_assistant_message(
                [ToolCall(id="tc1", name="spawn", arguments={})],
                stop_reason="tool_use",
            ),
            _raise_cancelled,
            faux_assistant_message("parent-final"),
        ]
    )

    async def _spawn(tool_call_id, args, *, signal=None, on_update=None):
        inner = _make_agent(provider, tools=[])
        detach = tracer.attach(inner)
        try:
            await inner.prompt("inner task")
            await inner.wait_for_idle()
        except CancelledError:
            pass
        finally:
            res = detach()
            if res is not None:
                await res
        return AgentToolResult(content=[TextContent(text="[cancelled]")], is_error=True)

    # Sequential mode so the inner run shares the parent/loop task.
    spawn = AgentTool(
        name="spawn",
        description="spawn inner",
        parameters=_Empty,
        execute=_spawn,
        execution_mode="sequential",
    )
    parent = _make_agent(provider, tools=[spawn])
    detach_parent = tracer.attach(parent)
    try:
        await parent.prompt("go")
        await parent.wait_for_idle()
    finally:
        res = detach_parent()
        if res is not None:
            await res
    await tracer.shutdown()

    # Identify the parent run unambiguously via the execute_tool span's run_id.
    tool_span = next(s for s in exporter.spans if s.name == "execute_tool spawn")
    parent_run_id = (tool_span.attributes or {}).get("cubepi.run_id")
    parent_chats = [
        s
        for s in exporter.spans
        if (s.attributes or {}).get("gen_ai.operation.name") == "chat"
        and (s.attributes or {}).get("cubepi.run_id") == parent_run_id
    ]
    assert len(parent_chats) == 2, (
        f"expected 2 parent-run chat spans, got {len(parent_chats)}"
    )


async def test_inner_run_nests_under_active_tool_span():
    """When invoke_agent opens while an execute_tool span is active for the
    task (a subagent running inside a tool body), the inner agent's root
    invoke_agent span must nest under that tool span — inheriting trace_id
    and setting parent_span_id — instead of starting a new root.
    """
    from cubepi.mcp import _tracing as mcp_tracing

    provider = FauxProvider()
    exporter = InMemoryExporter()
    tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])

    provider.append_responses([faux_assistant_message("inner-final")])

    parent_span = tracer.otel_tracer.start_span("execute_tool subagent")
    token = mcp_tracing.register_tool_span("tc1", parent_span, provider=None)
    try:
        inner = _make_agent(provider, tools=[])
        detach = tracer.attach(inner)
        try:
            await inner.prompt("hi")
            await inner.wait_for_idle()
        finally:
            res = detach()
            if res is not None:
                await res
    finally:
        mcp_tracing.unregister_tool_span(token)
        parent_span.end()
    await tracer.shutdown()

    root = next(
        s
        for s in exporter.spans
        if (s.attributes or {}).get("gen_ai.operation.name") == "invoke_agent"
    )
    pctx = parent_span.get_span_context()
    assert root.parent is not None, "invoke_agent opened as a new root"
    assert root.parent.span_id == pctx.span_id, (
        "invoke_agent not nested under tool span"
    )
    assert root.context.trace_id == pctx.trace_id, "trace_id not inherited"


async def test_full_nested_subtree_via_real_tool():
    """End-to-end composition of Tasks 1+2: a tool whose body attaches and runs
    an inner agent (which itself calls a tool) yields the full nested subtree
    ``execute_tool spawn -> invoke_agent -> cubepi.turn -> {chat,
    execute_tool inner_tool}`` — the inner chat and inner tool span both descend
    from the inner run's root, and the inner root nests under the tool span.
    """
    provider = FauxProvider()
    exporter = InMemoryExporter()
    tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])

    # FIFO across both agents in CALL order: parent->spawn, inner->inner_tool,
    # inner-final, parent-final.
    provider.append_responses(
        [
            faux_assistant_message(
                [ToolCall(id="tc1", name="spawn", arguments={})],
                stop_reason="tool_use",
            ),
            faux_assistant_message(
                [ToolCall(id="tc2", name="inner_tool", arguments={})],
                stop_reason="tool_use",
            ),
            faux_assistant_message("inner-final"),
            faux_assistant_message("parent-final"),
        ]
    )

    async def _inner_tool(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="inner-tool-ok")])

    inner_tool = AgentTool(
        name="inner_tool",
        description="inner tool",
        parameters=_Empty,
        execute=_inner_tool,
    )

    async def _spawn(tool_call_id, args, *, signal=None, on_update=None):
        inner = _make_agent(provider, tools=[inner_tool])
        detach = tracer.attach(inner)
        try:
            await inner.prompt("do the thing")
            await inner.wait_for_idle()
        finally:
            res = detach()
            if res is not None:
                await res
        return AgentToolResult(content=[TextContent(text="ok")])

    spawn = AgentTool(
        name="spawn", description="spawn inner", parameters=_Empty, execute=_spawn
    )
    parent = _make_agent(provider, tools=[spawn])
    detach_parent = tracer.attach(parent)
    try:
        await parent.prompt("go")
        await parent.wait_for_idle()
    finally:
        res = detach_parent()
        if res is not None:
            await res
    await tracer.shutdown()

    spans = exporter.spans
    by_id = {s.context.span_id: s for s in spans}
    tool_span = next(s for s in spans if s.name == "execute_tool spawn")
    inner_root = next(
        s
        for s in spans
        if (s.attributes or {}).get("gen_ai.operation.name") == "invoke_agent"
        and s.parent is not None
        and s.parent.span_id == tool_span.context.span_id
    )

    def _descends_from(span, ancestor_id):
        cur = span
        while cur is not None and cur.parent is not None:
            if cur.parent.span_id == ancestor_id:
                return True
            cur = by_id.get(cur.parent.span_id)
        return False

    inner_chats = [
        s
        for s in spans
        if (s.attributes or {}).get("gen_ai.operation.name") == "chat"
        and _descends_from(s, inner_root.context.span_id)
    ]
    assert inner_chats, "inner chat span did not nest under the inner run"
    inner_tool_spans = [
        s
        for s in spans
        if s.name == "execute_tool inner_tool"
        and _descends_from(s, inner_root.context.span_id)
    ]
    assert inner_tool_spans, (
        "inner execute_tool span missing or not nested under the inner run"
    )
