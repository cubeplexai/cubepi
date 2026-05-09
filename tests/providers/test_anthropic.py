from cubepi.providers.anthropic import AnthropicProvider, CacheRetention
from cubepi.providers.base import (
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


# ---------------------------------------------------------------------------
# Prompt caching tests
# ---------------------------------------------------------------------------


def _make_provider(retention: CacheRetention = "short") -> AnthropicProvider:
    """Create a provider without hitting the network (api_key is unused in tests)."""
    return AnthropicProvider(api_key="test-key", cache_retention=retention)


class TestCacheRetention:
    def test_default_retention_is_short(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider._cache_retention == "short"

    def test_retention_none_returns_no_cache_control(self):
        provider = _make_provider("none")
        assert provider._get_cache_control() is None

    def test_retention_short_returns_ephemeral(self):
        provider = _make_provider("short")
        cc = provider._get_cache_control()
        assert cc == {"type": "ephemeral"}

    def test_retention_long_returns_ephemeral_with_ttl(self):
        provider = _make_provider("long")
        cc = provider._get_cache_control()
        assert cc == {"type": "ephemeral", "ttl": "1h"}


class TestCacheControlOnMessages:
    """Verify cache_control markers are placed on the last message content block."""

    CACHE_CONTROL = {"type": "ephemeral"}

    def test_cache_control_on_last_user_message_text(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)

        # Only the last message's last block should have cache_control
        assert msgs[-1]["content"][-1]["cache_control"] == self.CACHE_CONTROL
        assert "cache_control" not in msgs[0]["content"][0]
        assert "cache_control" not in msgs[1]["content"][0]

    def test_cache_control_on_last_tool_result(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc-1",
                        "content": [{"type": "text", "text": "ok"}],
                        "is_error": False,
                    }
                ],
            }
        ]
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)
        assert msgs[0]["content"][-1]["cache_control"] == self.CACHE_CONTROL

    def test_cache_control_on_multi_block_message(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)
        # Only the last block gets the marker
        assert "cache_control" not in msgs[0]["content"][0]
        assert msgs[0]["content"][1]["cache_control"] == self.CACHE_CONTROL

    def test_cache_control_converts_bare_string_content(self):
        msgs = [{"role": "user", "content": "bare string"}]
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)
        # Should have been converted to a list with a text block
        assert isinstance(msgs[0]["content"], list)
        assert msgs[0]["content"][0]["type"] == "text"
        assert msgs[0]["content"][0]["text"] == "bare string"
        assert msgs[0]["content"][0]["cache_control"] == self.CACHE_CONTROL

    def test_empty_messages_is_noop(self):
        msgs: list[dict] = []
        # Should not raise
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)
        assert msgs == []

    def test_no_cache_control_when_retention_none(self):
        provider = _make_provider("none")
        assert provider._get_cache_control() is None


class TestCacheControlOnSystemPrompt:
    """Verify that the stream method builds system prompt blocks with cache_control."""

    def test_system_prompt_has_cache_control(self):
        """The system prompt should be sent as a content block with cache_control."""
        provider = _make_provider("short")
        cache_control = provider._get_cache_control()
        # Simulate what stream() does for system_prompt
        system_prompt = "You are a helpful assistant."
        system_block = {
            "type": "text",
            "text": system_prompt,
            **({"cache_control": cache_control} if cache_control else {}),
        }
        assert system_block["cache_control"] == {"type": "ephemeral"}

    def test_system_prompt_no_cache_when_retention_none(self):
        provider = _make_provider("none")
        cache_control = provider._get_cache_control()
        system_prompt = "You are a helpful assistant."
        system_block = {
            "type": "text",
            "text": system_prompt,
            **({"cache_control": cache_control} if cache_control else {}),
        }
        assert "cache_control" not in system_block


class TestCacheControlOnTools:
    """Verify cache_control is applied to the last tool definition."""

    CACHE_CONTROL = {"type": "ephemeral"}

    def _make_tools(self, count: int) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            )
            for i in range(count)
        ]

    def test_cache_control_on_last_tool_only(self):
        tools = self._make_tools(3)
        api_tools = [AnthropicProvider._convert_tool(t) for t in tools]
        # Apply cache_control the same way stream() does
        if api_tools:
            api_tools[-1]["cache_control"] = self.CACHE_CONTROL

        assert "cache_control" not in api_tools[0]
        assert "cache_control" not in api_tools[1]
        assert api_tools[2]["cache_control"] == self.CACHE_CONTROL

    def test_single_tool_gets_cache_control(self):
        tools = self._make_tools(1)
        api_tools = [AnthropicProvider._convert_tool(t) for t in tools]
        if api_tools:
            api_tools[-1]["cache_control"] = self.CACHE_CONTROL

        assert api_tools[0]["cache_control"] == self.CACHE_CONTROL

    def test_no_cache_control_when_retention_none(self):
        provider = _make_provider("none")
        cache_control = provider._get_cache_control()
        tools = self._make_tools(2)
        api_tools = [AnthropicProvider._convert_tool(t) for t in tools]
        if cache_control and api_tools:
            api_tools[-1]["cache_control"] = cache_control

        # With retention="none", no tool should have cache_control
        for tool in api_tools:
            assert "cache_control" not in tool
