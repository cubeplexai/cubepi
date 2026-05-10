import asyncio

import pytest

from cubepi.providers.base import (
    AssistantMessage,
    ImageContent,
    MessageStream,
    Model,
    ModelCost,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class TestMessageTypes:
    def test_text_content_defaults(self):
        tc = TextContent()
        assert tc.type == "text"
        assert tc.text == ""

    def test_text_content_with_value(self):
        tc = TextContent(text="hello")
        assert tc.text == "hello"

    def test_image_content(self):
        ic = ImageContent(source="base64data", media_type="image/png")
        assert ic.type == "image"
        assert ic.source == "base64data"
        assert ic.media_type == "image/png"

    def test_thinking_content(self):
        tc = ThinkingContent(thinking="step by step")
        assert tc.type == "thinking"
        assert tc.thinking == "step by step"

    def test_tool_call(self):
        tc = ToolCall(id="tc-1", name="search", arguments={"query": "hello"})
        assert tc.type == "tool_call"
        assert tc.id == "tc-1"
        assert tc.name == "search"
        assert tc.arguments == {"query": "hello"}

    def test_user_message(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        assert msg.role == "user"
        assert msg.timestamp is None

    def test_assistant_message_defaults(self):
        msg = AssistantMessage(content=[TextContent(text="hello")])
        assert msg.role == "assistant"
        assert msg.stop_reason == "stop"
        assert msg.error_message is None
        assert msg.usage is None

    def test_assistant_message_with_tool_calls(self):
        msg = AssistantMessage(
            content=[
                TextContent(text="Let me search."),
                ToolCall(id="tc-1", name="search", arguments={"q": "test"}),
            ],
            stop_reason="tool_use",
        )
        assert len(msg.content) == 2
        assert msg.content[1].type == "tool_call"

    def test_tool_result_message(self):
        msg = ToolResultMessage(
            tool_call_id="tc-1",
            tool_name="search",
            content=[TextContent(text="result")],
        )
        assert msg.role == "tool_result"
        assert msg.is_error is False

    def test_model_defaults(self):
        m = Model(id="gpt-4o", provider="openai")
        assert m.context_window == 200_000
        assert m.max_tokens == 8192
        assert m.reasoning is False
        assert m.cost is None

    def test_usage(self):
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0

    def test_model_cost(self):
        c = ModelCost(input=3.0, output=15.0)
        assert c.cache_read == 0

    def test_tool_definition(self):
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert td.name == "search"


class TestStreamEvent:
    def test_content_index_default_none(self):
        event = StreamEvent(type="text_delta", delta="hi")
        assert event.content_index is None

    def test_content_index_set(self):
        event = StreamEvent(type="text_start", content_index=0)
        assert event.content_index == 0


class TestMessageStream:
    async def test_stream_iteration_and_result(self):
        stream = MessageStream()
        msg = AssistantMessage(content=[TextContent(text="hello")])

        async def produce():
            await asyncio.sleep(0)
            stream.push(StreamEvent(type="text_delta", delta="hello"))
            stream.push(StreamEvent(type="done"))
            stream.set_result(msg)

        asyncio.create_task(produce())

        events = []
        async for event in stream:
            events.append(event)

        assert len(events) == 2
        assert events[0].type == "text_delta"
        assert events[1].type == "done"

        result = await stream.result()
        assert result.content[0].text == "hello"

    async def test_stream_error_event(self):
        stream = MessageStream()
        error_msg = AssistantMessage(
            content=[],
            stop_reason="error",
            error_message="API error",
        )

        async def produce():
            await asyncio.sleep(0)
            stream.push(StreamEvent(type="error", error_message="API error"))
            stream.set_result(error_msg)

        asyncio.create_task(produce())

        events = []
        async for event in stream:
            events.append(event)

        result = await stream.result()
        assert result.stop_reason == "error"


class TestMessageStreamTaskTracking:
    async def test_attach_task_stores_reference(self):
        ms = MessageStream()

        async def dummy():
            ms.push(StreamEvent(type="done"))
            ms.set_result(AssistantMessage(content=[]))

        task = asyncio.create_task(dummy())
        ms.attach_task(task)
        assert ms._producer_task is task
        await task

    async def test_result_propagates_task_exception(self):
        ms = MessageStream()

        async def failing():
            raise RuntimeError("producer died before pushing error")

        task = asyncio.create_task(failing())
        ms.attach_task(task)
        with pytest.raises(RuntimeError, match="producer died"):
            await ms.result()


class TestAssistantMessageMetadata:
    def test_default_metadata_fields(self):
        msg = AssistantMessage(content=[])
        assert msg.provider_id == ""
        assert msg.model_id == ""
        assert msg.response_id is None

    def test_metadata_fields_set(self):
        msg = AssistantMessage(
            content=[],
            provider_id="anthropic",
            model_id="claude-sonnet-4-20250514",
            response_id="msg_abc123",
        )
        assert msg.provider_id == "anthropic"
        assert msg.model_id == "claude-sonnet-4-20250514"
        assert msg.response_id == "msg_abc123"
