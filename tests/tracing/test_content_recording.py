"""Phase 2: pin record_content=True attribute emissions + redaction hook.

When ``Tracer(record_content=True)``, the recorder emits opt-in content
attrs on each span layer per the OTel GenAI semconv:

- ``invoke_agent`` (root): gen_ai.input.messages / gen_ai.output.messages
  / gen_ai.system_instructions
- ``cubepi.turn``: gen_ai.input.messages / gen_ai.output.messages
- ``chat``: gen_ai.system_instructions / gen_ai.input.messages /
  gen_ai.tool.definitions / cubepi.llm.raw_request / cubepi.llm.raw_response
- ``execute_tool``: gen_ai.tool.call.arguments / gen_ai.tool.call.result

Plus a ``redact`` hook on the Tracer for per-attribute filtering.
"""

from __future__ import annotations

import json
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import Model, TextContent, ToolCall
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.tracing import Tracer


MODEL = Model(id="faux-1", provider="faux")


class _Capture(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans):  # noqa: ANN001
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _json_attr(span: ReadableSpan, key: str) -> Any:
    raw = _attrs(span).get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def _build(*, record_content: bool, redact=None, tools=None):
    provider = FauxProvider()
    agent = Agent(
        provider=provider,
        model=MODEL,
        system_prompt="be helpful, be careful",
        tools=tools,
    )
    exporter = _Capture()
    tracer = Tracer(
        service_name="t",
        agent_name="a",
        exporters=[exporter],
        record_content=record_content,
        redact=redact,
    )
    tracer.attach(agent)
    return agent, provider, exporter, tracer


class TestContentDisabled:
    async def test_no_content_attrs_when_record_content_false(self):
        agent, provider, exporter, tracer = await _build(record_content=False)
        provider.append_responses([faux_assistant_message("hello")])

        await agent.prompt("hi")
        await agent.wait_for_idle()
        await tracer.shutdown()

        for span in exporter.spans:
            attrs = _attrs(span)
            for forbidden in (
                "gen_ai.input.messages",
                "gen_ai.output.messages",
                "gen_ai.system_instructions",
                "gen_ai.tool.definitions",
                "gen_ai.tool.call.arguments",
                "gen_ai.tool.call.result",
                "cubepi.llm.raw_request",
                "cubepi.llm.raw_response",
            ):
                assert forbidden not in attrs, (
                    f"{span.name} should not carry {forbidden} with record_content=False"
                )


class TestRootContent:
    async def test_invoke_agent_records_input_output_system(self):
        agent, provider, exporter, tracer = await _build(record_content=True)
        provider.append_responses([faux_assistant_message("hi back")])

        await agent.prompt("hello there")
        await agent.wait_for_idle()
        await tracer.shutdown()

        root = [s for s in exporter.spans if s.name == "invoke_agent"][0]
        sys_msgs = _json_attr(root, "gen_ai.system_instructions")
        assert sys_msgs == [
            {
                "role": "system",
                "parts": [{"type": "text", "content": "be helpful, be careful"}],
            }
        ]
        inp = _json_attr(root, "gen_ai.input.messages")
        assert isinstance(inp, list)
        assert inp[0]["role"] == "user"
        assert inp[0]["parts"][0] == {"type": "text", "content": "hello there"}
        out = _json_attr(root, "gen_ai.output.messages")
        assert isinstance(out, list)
        assert any(
            m.get("role") == "assistant"
            and any(p.get("content") == "hi back" for p in m.get("parts", []))
            for m in out
        )


class TestChatContent:
    async def test_chat_span_carries_system_input_and_raw_payload(self):
        agent, provider, exporter, tracer = await _build(record_content=True)
        provider.append_responses([faux_assistant_message("hi")])

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        chat = next(s for s in exporter.spans if s.name.startswith("chat "))
        attrs = _attrs(chat)
        assert "gen_ai.system_instructions" in attrs
        assert "gen_ai.input.messages" in attrs
        raw = json.loads(attrs["cubepi.llm.raw_request"])
        assert raw["model"] == MODEL.id
        assert "messages" in raw
        resp = json.loads(attrs["cubepi.llm.raw_response"])
        assert resp["id"] == "faux-1"
        assert resp["role"] == "assistant"


class TestToolContent:
    async def test_execute_tool_carries_arguments_and_result(self):
        class Params(BaseModel):
            value: str

        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(
                content=[TextContent(text=f"echoed: {params.value}")]
            )

        tool = AgentTool(
            name="echo", description="echo a thing", parameters=Params, execute=run
        )

        agent, provider, exporter, tracer = await _build(
            record_content=True, tools=[tool]
        )
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="t1", name="echo", arguments={"value": "X"})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        t_span = next(s for s in exporter.spans if s.name.startswith("execute_tool "))
        attrs = _attrs(t_span)
        args = json.loads(attrs["gen_ai.tool.call.arguments"])
        assert args == {"value": "X"}
        result = json.loads(attrs["gen_ai.tool.call.result"])
        assert "content" in result
        assert any(
            isinstance(c, dict) and c.get("text") == "echoed: X"
            for c in result.get("content", [])
        )


class TestTurnContent:
    async def test_turn_records_per_turn_input_and_output(self):
        agent, provider, exporter, tracer = await _build(record_content=True)
        provider.append_responses([faux_assistant_message("hello back")])

        await agent.prompt("hi")
        await agent.wait_for_idle()
        await tracer.shutdown()

        turn = next(s for s in exporter.spans if s.name == "cubepi.turn")
        inp = _json_attr(turn, "gen_ai.input.messages")
        assert inp[0]["parts"][0]["content"] == "hi"
        out = _json_attr(turn, "gen_ai.output.messages")
        assert out[0]["role"] == "assistant"


class TestRedaction:
    async def test_redact_can_substitute(self):
        seen_keys: list[str] = []

        def redact(key: str, value: Any) -> Any:
            seen_keys.append(key)
            if key == "gen_ai.input.messages":
                return [
                    {
                        "role": "user",
                        "parts": [{"type": "text", "content": "<REDACTED>"}],
                    }
                ]
            return value

        agent, provider, exporter, tracer = await _build(
            record_content=True, redact=redact
        )
        provider.append_responses([faux_assistant_message("ok")])
        await agent.prompt("secret prompt")
        await agent.wait_for_idle()
        await tracer.shutdown()

        assert "gen_ai.input.messages" in seen_keys
        chat = next(s for s in exporter.spans if s.name.startswith("chat "))
        inp = _json_attr(chat, "gen_ai.input.messages")
        assert inp == [
            {"role": "user", "parts": [{"type": "text", "content": "<REDACTED>"}]}
        ]

    async def test_redact_can_drop_attribute(self):
        def redact(key: str, value: Any) -> Any:
            if key == "cubepi.llm.raw_request":
                return None
            return value

        agent, provider, exporter, tracer = await _build(
            record_content=True, redact=redact
        )
        provider.append_responses([faux_assistant_message("ok")])
        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        chat = next(s for s in exporter.spans if s.name.startswith("chat "))
        assert "cubepi.llm.raw_request" not in _attrs(chat)
        assert "gen_ai.input.messages" in _attrs(chat)

    async def test_redact_exception_is_swallowed(self):
        def redact(key: str, value: Any) -> Any:
            raise RuntimeError("bad redactor")

        agent, provider, exporter, tracer = await _build(
            record_content=True, redact=redact
        )
        provider.append_responses([faux_assistant_message("ok")])
        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        assert any(s.name == "invoke_agent" for s in exporter.spans)
        chat = next(s for s in exporter.spans if s.name.startswith("chat "))
        assert "gen_ai.input.messages" not in _attrs(chat)


class TestContentHelpers:
    def test_messages_to_semconv_handles_all_block_types(self):
        from cubepi.providers.base import (
            AssistantMessage,
            ThinkingContent,
        )
        from cubepi.providers.base import ToolCall as _ToolCall
        from cubepi.providers.base import (
            ToolResultMessage,
            UserMessage,
        )
        from cubepi.tracing.content import messages_to_semconv

        msgs = [
            UserMessage(content=[TextContent(text="hi")]),
            AssistantMessage(
                content=[
                    TextContent(text="ok"),
                    ThinkingContent(thinking="planning..."),
                    _ToolCall(id="c1", name="t", arguments={"a": 1}),
                ],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="c1",
                tool_name="t",
                content=[TextContent(text="done")],
            ),
        ]
        out = messages_to_semconv(msgs)
        assert out[0]["role"] == "user"
        assert out[1]["role"] == "assistant"
        assert out[1]["parts"][0] == {"type": "text", "content": "ok"}
        assert out[1]["parts"][1] == {
            "type": "reasoning",
            "content": "planning...",
        }
        assert out[1]["parts"][2] == {
            "type": "tool_call",
            "id": "c1",
            "name": "t",
            "arguments": {"a": 1},
        }
        assert out[2]["role"] == "tool"
        assert out[2]["parts"][0] == {
            "type": "tool_call_response",
            "id": "c1",
            "result": "done",
        }

    def test_tool_definitions_anthropic_shape(self):
        from cubepi.tracing.content import tool_definitions_to_semconv

        payload = {
            "tools": [
                {
                    "name": "fetch",
                    "description": "fetch a url",
                    "input_schema": {"type": "object"},
                }
            ]
        }
        out = tool_definitions_to_semconv(payload)
        assert out == [
            {
                "name": "fetch",
                "description": "fetch a url",
                "parameters": {"type": "object"},
            }
        ]

    def test_tool_definitions_openai_chat_shape(self):
        from cubepi.tracing.content import tool_definitions_to_semconv

        payload = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "fetch",
                        "description": "fetch a url",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        }
        out = tool_definitions_to_semconv(payload)
        assert out == [
            {
                "name": "fetch",
                "description": "fetch a url",
                "parameters": {"type": "object"},
            }
        ]

    def test_serialize_for_attribute_fallback(self):
        from cubepi.tracing.content import serialize_for_attribute

        class NotJsonable:
            pass

        out = serialize_for_attribute(NotJsonable())
        assert isinstance(out, str)
