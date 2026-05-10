from cubepi.providers.openai import OpenAIProvider
from cubepi.providers.base import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)


class TestOpenAIMessageConversion:
    def test_convert_user_message(self):
        msg = UserMessage(content=[TextContent(text="hello")])
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "hello"

    def test_convert_assistant_message(self):
        msg = AssistantMessage(content=[TextContent(text="hi")])
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "hi"

    def test_convert_assistant_with_tool_calls(self):
        msg = AssistantMessage(
            content=[ToolCall(id="tc-1", name="search", arguments={"q": "test"})],
            stop_reason="tool_use",
        )
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "tc-1"
        assert result["tool_calls"][0]["function"]["name"] == "search"

    def test_convert_tool_result(self):
        msg = ToolResultMessage(
            tool_call_id="tc-1",
            tool_name="search",
            content=[TextContent(text="result")],
        )
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "tc-1"
        assert result["content"] == "result"


class TestOpenAIImageConversion:
    def test_user_message_with_image(self):
        msg = UserMessage(content=[
            TextContent(text="What's in this image?"),
            ImageContent(source="base64data", media_type="image/png"),
        ])
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "What's in this image?"}
        assert result["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,base64data"},
        }

    def test_user_message_text_only_stays_simple(self):
        msg = UserMessage(content=[TextContent(text="hello")])
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "hello"


class TestOpenAIToolConversion:
    def test_convert_tool_definition(self):
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        result = OpenAIProvider._convert_tool(td)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["parameters"]["type"] == "object"
