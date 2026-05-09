import pytest

from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.base import (
    Model,
    TextContent,
    ToolCall,
    ToolDefinition,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
)


class TestAnthropicMessageConversion:
    def test_convert_user_message(self):
        msg = UserMessage(content=[TextContent(text="hello")])
        result = AnthropicProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "hello"

    def test_convert_assistant_message(self):
        msg = AssistantMessage(content=[TextContent(text="hi")])
        result = AnthropicProvider._convert_message(msg)
        assert result["role"] == "assistant"

    def test_convert_assistant_with_tool_call(self):
        msg = AssistantMessage(
            content=[ToolCall(id="tc-1", name="search", arguments={"q": "test"})],
            stop_reason="tool_use",
        )
        result = AnthropicProvider._convert_message(msg)
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["id"] == "tc-1"
        assert result["content"][0]["name"] == "search"
        assert result["content"][0]["input"] == {"q": "test"}

    def test_convert_tool_result(self):
        msg = ToolResultMessage(
            tool_call_id="tc-1",
            tool_name="search",
            content=[TextContent(text="result")],
        )
        result = AnthropicProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "tc-1"


class TestAnthropicToolConversion:
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
        result = AnthropicProvider._convert_tool(td)
        assert result["name"] == "search"
        assert result["description"] == "Search the web"
        assert result["input_schema"]["type"] == "object"
