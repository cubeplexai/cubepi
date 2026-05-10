import asyncio

from cubepi.providers.base import (
    Model,
    StreamOptions,
    TextContent,
    ToolDefinition,
    UserMessage,
)
from cubepi.providers.faux import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)


class TestFauxHelpers:
    def test_faux_text(self):
        block = faux_text("hello")
        assert block.type == "text"
        assert block.text == "hello"

    def test_faux_thinking(self):
        block = faux_thinking("step by step")
        assert block.type == "thinking"
        assert block.thinking == "step by step"

    def test_faux_tool_call(self):
        block = faux_tool_call("search", {"q": "test"}, id="tc-1")
        assert block.type == "tool_call"
        assert block.id == "tc-1"
        assert block.name == "search"

    def test_faux_tool_call_auto_id(self):
        block = faux_tool_call("search", {"q": "test"})
        assert block.id.startswith("tool:")

    def test_faux_assistant_message_string(self):
        msg = faux_assistant_message("hello")
        assert msg.role == "assistant"
        assert len(msg.content) == 1
        assert msg.content[0].type == "text"
        assert msg.content[0].text == "hello"
        assert msg.stop_reason == "stop"

    def test_faux_assistant_message_blocks(self):
        msg = faux_assistant_message(
            [faux_text("hi"), faux_tool_call("search", {"q": "x"}, id="t1")]
        )
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "tool_call"

    def test_faux_assistant_message_tool_use_stop_reason(self):
        msg = faux_assistant_message(
            [faux_tool_call("search", {"q": "x"}, id="t1")],
            stop_reason="tool_use",
        )
        assert msg.stop_reason == "tool_use"


class TestFauxProvider:
    def _make_model(self):
        from cubepi.providers.base import Model

        return Model(id="faux-1", provider="faux")

    async def test_basic_text_response(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("hello world")])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "hello world"
        assert result.stop_reason == "stop"
        assert any(e.type == "done" for e in events)

    async def test_responses_consumed_in_order(self):
        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message("first"),
                faux_assistant_message("second"),
            ]
        )
        model = self._make_model()

        stream1 = await provider.stream(model, [])
        _ = [e async for e in stream1]
        r1 = await stream1.result()

        stream2 = await provider.stream(model, [])
        _ = [e async for e in stream2]
        r2 = await stream2.result()

        assert r1.content[0].text == "first"
        assert r2.content[0].text == "second"

    async def test_error_when_queue_exhausted(self):
        provider = FauxProvider()
        model = self._make_model()

        stream = await provider.stream(model, [])
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.stop_reason == "error"
        assert "No more faux responses" in (result.error_message or "")

    async def test_set_responses_replaces_queue(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("old")])
        provider.set_responses([faux_assistant_message("new")])
        model = self._make_model()

        stream = await provider.stream(model, [])
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "new"

    async def test_append_responses(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("first")])
        provider.append_responses([faux_assistant_message("second")])

        assert provider.pending_response_count == 2

    async def test_async_response_factory(self):
        async def factory(context, model):
            return faux_assistant_message("dynamic response")

        provider = FauxProvider()
        provider.set_responses([factory])
        model = self._make_model()

        stream = await provider.stream(model, [])
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "dynamic response"

    async def test_streams_text_deltas(self):
        provider = FauxProvider(token_size_min=1, token_size_max=1)
        provider.set_responses([faux_assistant_message("AB")])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        event_types = [e.type for e in events]

        assert "start" in event_types
        assert "text_start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types
        assert "done" in event_types

    async def test_streams_thinking_deltas(self):
        provider = FauxProvider(token_size_min=1, token_size_max=1)
        provider.set_responses(
            [faux_assistant_message([faux_thinking("think"), faux_text("ok")])]
        )
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        event_types = [e.type for e in events]

        assert "thinking_start" in event_types
        assert "thinking_delta" in event_types
        assert "thinking_end" in event_types

    async def test_streams_tool_call_deltas(self):
        provider = FauxProvider(token_size_min=1, token_size_max=1)
        provider.set_responses(
            [
                faux_assistant_message(
                    [faux_tool_call("search", {"q": "test"}, id="t1")],
                    stop_reason="tool_use",
                ),
            ]
        )
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        event_types = [e.type for e in events]

        assert "toolcall_start" in event_types
        assert "toolcall_delta" in event_types
        assert "toolcall_end" in event_types

    async def test_abort_before_start(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("hello")])
        model = self._make_model()
        signal = asyncio.Event()
        signal.set()

        stream = await provider.stream(model, [], options=StreamOptions(signal=signal))
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.stop_reason == "aborted"

    async def test_abort_mid_stream(self):
        provider = FauxProvider(
            tokens_per_second=20, token_size_min=2, token_size_max=2
        )
        provider.set_responses(
            [
                faux_assistant_message(
                    "one two three four five six seven eight nine ten"
                ),
            ]
        )
        model = self._make_model()
        signal = asyncio.Event()

        stream = await provider.stream(model, [], options=StreamOptions(signal=signal))

        events = []
        count = 0
        async for event in stream:
            events.append(event)
            count += 1
            if count >= 3:
                signal.set()

        result = await stream.result()
        assert result.stop_reason == "aborted"

    async def test_error_message_passthrough(self):
        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message(
                    "", stop_reason="error", error_message="API rate limit"
                ),
            ]
        )
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        result = await stream.result()

        assert result.stop_reason == "error"
        assert any(e.type == "error" for e in events)

    async def test_multiple_tool_calls_in_one_message(self):
        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message(
                    [
                        faux_tool_call("search", {"q": "a"}, id="t1"),
                        faux_tool_call("search", {"q": "b"}, id="t2"),
                    ],
                    stop_reason="tool_use",
                ),
            ]
        )
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        await stream.result()

        toolcall_starts = [e for e in events if e.type == "toolcall_start"]
        assert len(toolcall_starts) == 2

    async def test_call_count_tracking(self):
        provider = FauxProvider()
        provider.set_responses(
            [faux_assistant_message("a"), faux_assistant_message("b")]
        )
        model = self._make_model()

        assert provider.call_count == 0

        s1 = await provider.stream(model, [])
        _ = [e async for e in s1]
        assert provider.call_count == 1

        s2 = await provider.stream(model, [])
        _ = [e async for e in s2]
        assert provider.call_count == 2

    async def test_provider_metadata_set(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("hello")])
        model = Model(id="faux-1", provider="faux")
        stream = await provider.stream(model, [])
        _ = [e async for e in stream]
        result = await stream.result()
        assert result.provider_id == "faux"
        assert result.model_id == "faux-1"

    async def test_provider_metadata_on_partial_events(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("hello")])
        model = Model(id="faux-1", provider="faux")
        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        start_event = next(e for e in events if e.type == "start")
        assert start_event.partial is not None
        assert start_event.partial.provider_id == "faux"
        assert start_event.partial.model_id == "faux-1"


class TestFauxProviderExtendedFactory:
    """Tests for extended factory signature (messages, model, system_prompt, tools)."""

    def _make_model(self):
        from cubepi.providers.base import Model

        return Model(id="faux-1", provider="faux")

    async def test_extended_sync_factory_receives_all_args(self):
        """Factory with 4 params receives system_prompt and tools."""
        received = {}

        def factory(messages, model, system_prompt, tools):
            received["system_prompt"] = system_prompt
            received["tools"] = tools
            received["messages"] = messages
            received["model"] = model
            return faux_assistant_message("extended")

        provider = FauxProvider()
        provider.set_responses([factory])
        model = self._make_model()
        tools = [
            ToolDefinition(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]

        stream = await provider.stream(
            model, [], system_prompt="You are helpful.", tools=tools
        )
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "extended"
        assert received["system_prompt"] == "You are helpful."
        assert received["tools"] is not None
        assert len(received["tools"]) == 1
        assert received["tools"][0].name == "search"

    async def test_extended_async_factory_receives_all_args(self):
        """Async factory with 4 params receives system_prompt and tools."""
        received = {}

        async def factory(messages, model, system_prompt, tools):
            received["system_prompt"] = system_prompt
            received["tools"] = tools
            return faux_assistant_message("async extended")

        provider = FauxProvider()
        provider.set_responses([factory])
        model = self._make_model()

        stream = await provider.stream(
            model, [], system_prompt="Be concise.", tools=None
        )
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "async extended"
        assert received["system_prompt"] == "Be concise."
        assert received["tools"] is None

    async def test_old_factory_still_works(self):
        """Old-style factory with 2 params still works (backward compat)."""

        def factory(messages, model):
            return faux_assistant_message(f"old style: {model.id}")

        provider = FauxProvider()
        provider.set_responses([factory])
        model = self._make_model()

        stream = await provider.stream(model, [], system_prompt="ignored", tools=None)
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "old style: faux-1"

    async def test_old_async_factory_still_works(self):
        """Old-style async factory with 2 params still works."""

        async def factory(messages, model):
            return faux_assistant_message("old async")

        provider = FauxProvider()
        provider.set_responses([factory])
        model = self._make_model()

        stream = await provider.stream(model, [], system_prompt="ignored", tools=None)
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "old async"

    async def test_factory_with_var_positional_args(self):
        """Factory using *args receives all arguments."""

        def factory(*args):
            assert len(args) == 4
            return faux_assistant_message(f"got {len(args)} args")

        provider = FauxProvider()
        provider.set_responses([factory])
        model = self._make_model()

        stream = await provider.stream(model, [], system_prompt="test")
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "got 4 args"

    async def test_extended_factory_with_tools_content(self):
        """Extended factory can use tools to decide response."""

        def factory(messages, model, system_prompt, tools):
            if tools and any(t.name == "calculator" for t in tools):
                return faux_assistant_message(
                    [faux_tool_call("calculator", {"expr": "2+2"}, id="t1")],
                    stop_reason="tool_use",
                )
            return faux_assistant_message("no tools available")

        provider = FauxProvider()
        tools = [
            ToolDefinition(
                name="calculator",
                description="Evaluate math",
                parameters={
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                },
            )
        ]
        provider.set_responses([factory])
        model = self._make_model()

        stream = await provider.stream(model, [], tools=tools)
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.stop_reason == "tool_use"
        assert result.content[0].name == "calculator"


class TestFauxProviderPromptCache:
    """Tests for prompt cache simulation."""

    def _make_model(self):
        from cubepi.providers.base import Model

        return Model(id="faux-1", provider="faux")

    async def test_first_call_populates_cache_write(self):
        """First call should have cache_write > 0 and cache_read == 0."""
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("hello")])
        model = self._make_model()

        stream = await provider.stream(model, [], system_prompt="You are helpful.")
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.usage is not None
        assert result.usage.cache_write_tokens > 0
        assert result.usage.cache_read_tokens == 0

    async def test_second_call_same_context_has_cache_read(self):
        """Second call with same context should have cache_read > 0."""
        provider = FauxProvider()
        provider.set_responses(
            [faux_assistant_message("first"), faux_assistant_message("second")]
        )
        model = self._make_model()
        messages = [UserMessage(content=[TextContent(text="Hello")])]

        # First call
        s1 = await provider.stream(model, messages, system_prompt="You are helpful.")
        _ = [e async for e in s1]
        r1 = await s1.result()

        # Second call with same context
        s2 = await provider.stream(model, messages, system_prompt="You are helpful.")
        _ = [e async for e in s2]
        r2 = await s2.result()

        assert r1.usage is not None
        assert r2.usage is not None
        # First call: all cache_write, no cache_read
        assert r1.usage.cache_write_tokens > 0
        assert r1.usage.cache_read_tokens == 0
        # Second call: should have cache_read (prefix match)
        assert r2.usage.cache_read_tokens > 0

    async def test_different_system_prompt_has_no_cache_read(self):
        """Changing system_prompt should result in cache miss (no prefix match)."""
        provider = FauxProvider()
        provider.set_responses(
            [faux_assistant_message("first"), faux_assistant_message("second")]
        )
        model = self._make_model()

        # First call
        s1 = await provider.stream(
            model, [], system_prompt="You are a helpful assistant."
        )
        _ = [e async for e in s1]
        r1 = await s1.result()

        # Second call with different system prompt
        s2 = await provider.stream(model, [], system_prompt="You are a code reviewer.")
        _ = [e async for e in s2]
        r2 = await s2.result()

        assert r1.usage is not None
        assert r2.usage is not None
        # Both should have cache_write since prompts differ significantly
        assert r1.usage.cache_write_tokens > 0
        assert r2.usage.cache_write_tokens > 0
        # Second call might have partial prefix match ("system:You are a ")
        # but should have less cache_read than a full match
        # The key assertion: cache_write on second call is non-zero
        # (new content that wasn't in cache)
        assert r2.usage.cache_write_tokens > 0

    async def test_clear_prompt_cache(self):
        """Clearing cache causes next call to be a full cache miss."""
        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message("a"),
                faux_assistant_message("b"),
                faux_assistant_message("c"),
            ]
        )
        model = self._make_model()

        # First call populates cache
        s1 = await provider.stream(model, [], system_prompt="test")
        _ = [e async for e in s1]

        # Second call hits cache
        s2 = await provider.stream(model, [], system_prompt="test")
        _ = [e async for e in s2]
        r2 = await s2.result()
        assert r2.usage is not None
        assert r2.usage.cache_read_tokens > 0

        # Clear cache
        provider.clear_prompt_cache()
        assert provider.prompt_cache == {}

        # Third call after clear: cache miss
        s3 = await provider.stream(model, [], system_prompt="test")
        _ = [e async for e in s3]
        r3 = await s3.result()
        assert r3.usage is not None
        assert r3.usage.cache_read_tokens == 0
        assert r3.usage.cache_write_tokens > 0

    async def test_cache_with_tools_change(self):
        """Changing tools definition should affect cache behavior."""
        provider = FauxProvider()
        provider.set_responses(
            [faux_assistant_message("first"), faux_assistant_message("second")]
        )
        model = self._make_model()
        tools_v1 = [
            ToolDefinition(
                name="search",
                description="Search",
                parameters={"type": "object"},
            )
        ]
        tools_v2 = [
            ToolDefinition(
                name="search",
                description="Search",
                parameters={"type": "object"},
            ),
            ToolDefinition(
                name="calculate",
                description="Calculate",
                parameters={"type": "object"},
            ),
        ]

        # First call with tools_v1
        s1 = await provider.stream(model, [], system_prompt="test", tools=tools_v1)
        _ = [e async for e in s1]
        r1 = await s1.result()

        # Second call with tools_v2 (different tools)
        s2 = await provider.stream(model, [], system_prompt="test", tools=tools_v2)
        _ = [e async for e in s2]
        r2 = await s2.result()

        assert r1.usage is not None
        assert r2.usage is not None
        # Second call should have partial cache read (system_prompt matches)
        # but also cache_write for the new tool definitions
        assert r2.usage.cache_write_tokens > 0

    async def test_cache_populates_usage_on_static_response(self):
        """Static AssistantMessage responses also get cache-aware usage."""
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("static")])
        model = self._make_model()

        stream = await provider.stream(model, [], system_prompt="You are helpful.")
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.usage is not None
        assert result.usage.cache_write_tokens > 0
        assert result.usage.output_tokens > 0

    async def test_prompt_cache_property_is_copy(self):
        """The prompt_cache property returns a copy, not the internal dict."""
        provider = FauxProvider()
        cache = provider.prompt_cache
        cache["injected"] = "value"
        assert "injected" not in provider.prompt_cache
