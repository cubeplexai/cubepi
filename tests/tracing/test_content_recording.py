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


class TestRootOutputIsOnlyGenerated:
    async def test_root_output_excludes_user_prompts(self):
        """The root invoke_agent's gen_ai.output.messages must be model
        output only (assistant + tool_result), NOT include the caller's
        user prompts — codex round-1 finding on PR #83."""
        agent, provider, exporter, tracer = await _build(record_content=True)
        provider.append_responses([faux_assistant_message("hello back")])
        await agent.prompt("hi there")
        await agent.wait_for_idle()
        await tracer.shutdown()

        root = next(s for s in exporter.spans if s.name == "invoke_agent")
        out = _json_attr(root, "gen_ai.output.messages")
        # No user-role messages must appear in output.
        for m in out:
            assert m["role"] != "user", (
                f"root gen_ai.output.messages must not include user prompts; got {m}"
            )
        # An assistant message with the model output IS expected.
        assert any(
            m["role"] == "assistant"
            and any(p.get("content") == "hello back" for p in m.get("parts", []))
            for m in out
        )


class TestChatInputIncludesPriorAssistant:
    async def test_second_chat_input_includes_prior_assistant_turn(self):
        """In a tool-using multi-turn run, the second chat span's
        gen_ai.input.messages must include the prior assistant
        (tool-call) message + the tool_result so consumers can
        reconstruct the prompt — codex round-1 finding on PR #83."""

        class P(BaseModel):
            pass

        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="tool-output")])

        tool = AgentTool(name="t", description="t", parameters=P, execute=run)

        agent, provider, exporter, tracer = await _build(
            record_content=True, tools=[tool]
        )
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="c1", name="t", arguments={})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("final answer"),
            ]
        )

        await agent.prompt("kick off")
        await agent.wait_for_idle()
        await tracer.shutdown()

        chats = sorted(
            [s for s in exporter.spans if s.name.startswith("chat ")],
            key=lambda s: s.start_time or 0,
        )
        assert len(chats) == 2
        second = chats[1]
        inp = _json_attr(second, "gen_ai.input.messages")
        roles = [m["role"] for m in inp]
        # Must include the original user prompt, the assistant tool-call
        # message, and the tool result — in that order.
        assert roles.count("user") == 1
        assert "assistant" in roles
        assert "tool" in roles
        # The assistant message must carry the tool_call part.
        asst = next(m for m in inp if m["role"] == "assistant")
        assert any(p.get("type") == "tool_call" for p in asst.get("parts", []))
        # Pin the round-83 codex fix: tool_result must appear exactly
        # once (it was previously appended both by _on_message_start
        # and _on_message_end, producing two identical ``tool`` entries
        # even though the provider context contains only one).
        assert roles.count("tool") == 1


class TestChatInputForContinuedHistory:
    """For ``Agent.resume()`` and other continuation paths, the cubepi
    loop intentionally does NOT emit ``MessageStartEvent`` for the
    pre-existing ``context.messages`` — so without seeding, the
    recorder's transcript starts empty and the first chat span's
    ``gen_ai.input.messages`` omits the conversation history that the
    provider request actually carries.

    Codex P2 finding on PR #83: the recorder must seed the transcript
    from the agent's existing messages at agent_start.
    """

    async def test_resume_chat_input_includes_prior_history(self):
        from cubepi.providers.base import UserMessage as _U
        from cubepi.providers.base import AssistantMessage as _A
        from cubepi.providers.base import ToolResultMessage as _TR

        agent, provider, exporter, tracer = await _build(record_content=True)
        # Pre-load the agent with conversation history.
        agent.messages = [
            _U(content=[TextContent(text="earlier prompt")]),
            _A(
                content=[TextContent(text="ok, working on it")],
                stop_reason="end_turn",
            ),
        ]
        # Now drive a new prompt and check that the FIRST chat span
        # includes the prior turns in its input. (resume() doesn't
        # add a new prompt; using prompt() here is equivalent for the
        # transcript-seeding contract since both code paths leave the
        # pre-existing history un-emitted.)
        provider.append_responses([faux_assistant_message("acknowledged")])
        await agent.prompt("now continue")
        await agent.wait_for_idle()
        await tracer.shutdown()

        chats = sorted(
            [s for s in exporter.spans if s.name.startswith("chat ")],
            key=lambda s: s.start_time or 0,
        )
        assert len(chats) == 1
        inp = _json_attr(chats[0], "gen_ai.input.messages")
        roles = [m["role"] for m in inp]
        # Must carry: earlier user + earlier assistant + new user prompt.
        assert roles == ["user", "assistant", "user"], (
            f"first chat span input must include seeded history; got {roles}"
        )
        # Earlier user content survived.
        assert any("earlier prompt" in p.get("content", "") for p in inp[0]["parts"])
        # Earlier assistant content survived.
        assert any("ok, working on it" in p.get("content", "") for p in inp[1]["parts"])
        # New prompt is last.
        assert any("now continue" in p.get("content", "") for p in inp[2]["parts"])

        # Silence the unused-import linter on the ToolResultMessage
        # alias — it documents the message shapes this test cares about
        # even though only User+Assistant are exercised inline.
        del _TR


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
