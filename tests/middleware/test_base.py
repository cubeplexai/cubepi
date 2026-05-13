from cubepi.middleware.base import Middleware, compose_middleware
from cubepi.agent.types import (
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    AfterToolCallContext,
    ShouldStopAfterTurnContext,
    AgentContext,
    AgentToolResult,
)
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
    UserMessage,
)


class TestComposeMiddlewareEmpty:
    def test_empty_list_returns_empty_dict(self):
        hooks = compose_middleware([])
        assert hooks == {}


class TestTransformContext:
    async def test_chained_transform_context(self):
        class AddPrefix(Middleware):
            async def transform_context(self, messages, *, signal=None):
                return [UserMessage(content=[TextContent(text="PREFIX")])] + list(
                    messages
                )

        class AddSuffix(Middleware):
            async def transform_context(self, messages, *, signal=None):
                return list(messages) + [
                    UserMessage(content=[TextContent(text="SUFFIX")])
                ]

        hooks = compose_middleware([AddPrefix(), AddSuffix()])
        result = await hooks["transform_context"](
            [UserMessage(content=[TextContent(text="middle")])], signal=None
        )

        assert len(result) == 3
        assert result[0].content[0].text == "PREFIX"
        assert result[1].content[0].text == "middle"
        assert result[2].content[0].text == "SUFFIX"


class TestConvertToLlm:
    async def test_last_implementation_wins(self):
        class First(Middleware):
            async def convert_to_llm(self, messages):
                return [UserMessage(content=[TextContent(text="first")])]

        class Second(Middleware):
            async def convert_to_llm(self, messages):
                return [UserMessage(content=[TextContent(text="second")])]

        hooks = compose_middleware([First(), Second()])
        result = await hooks["convert_to_llm"]([])

        assert len(result) == 1
        assert result[0].content[0].text == "second"


class TestBeforeToolCall:
    async def test_any_block_stops_execution(self):
        class Allower(Middleware):
            async def before_tool_call(self, ctx, *, signal=None):
                return None

        class Blocker(Middleware):
            async def before_tool_call(self, ctx, *, signal=None):
                return BeforeToolCallResult(block=True, reason="Blocked")

        hooks = compose_middleware([Allower(), Blocker()])

        ctx = BeforeToolCallContext(
            assistant_message=AssistantMessage(content=[]),
            tool_call=ToolCall(id="t1", name="test", arguments={}),
            args={},
            context=AgentContext(system_prompt="", messages=[]),
        )
        result = await hooks["before_tool_call"](ctx, signal=None)

        assert result is not None
        assert result.block is True
        assert result.reason == "Blocked"

    async def test_no_block_returns_none(self):
        class Allower(Middleware):
            async def before_tool_call(self, ctx, *, signal=None):
                return None

        hooks = compose_middleware([Allower()])
        ctx = BeforeToolCallContext(
            assistant_message=AssistantMessage(content=[]),
            tool_call=ToolCall(id="t1", name="test", arguments={}),
            args={},
            context=AgentContext(system_prompt="", messages=[]),
        )
        result = await hooks["before_tool_call"](ctx, signal=None)
        assert result is None


class TestAfterToolCall:
    async def test_later_overrides_earlier(self):
        class First(Middleware):
            async def after_tool_call(self, ctx, *, signal=None):
                return AfterToolCallResult(content=[TextContent(text="first")])

        class Second(Middleware):
            async def after_tool_call(self, ctx, *, signal=None):
                return AfterToolCallResult(content=[TextContent(text="second")])

        hooks = compose_middleware([First(), Second()])

        ctx = AfterToolCallContext(
            assistant_message=AssistantMessage(content=[]),
            tool_call=ToolCall(id="t1", name="test", arguments={}),
            args={},
            result=AgentToolResult(content=[TextContent(text="original")]),
            is_error=False,
            context=AgentContext(system_prompt="", messages=[]),
        )
        result = await hooks["after_tool_call"](ctx, signal=None)

        assert result.content[0].text == "second"


class TestShouldStopAfterTurn:
    async def test_any_true_stops(self):
        class NoStop(Middleware):
            async def should_stop_after_turn(self, ctx):
                return False

        class YesStop(Middleware):
            async def should_stop_after_turn(self, ctx):
                return True

        hooks = compose_middleware([NoStop(), YesStop()])

        ctx = ShouldStopAfterTurnContext(
            message=AssistantMessage(content=[]),
            tool_results=[],
            context=AgentContext(system_prompt="", messages=[]),
            new_messages=[],
        )
        result = await hooks["should_stop_after_turn"](ctx)
        assert result is True

    async def test_all_false_continues(self):
        class NoStop(Middleware):
            async def should_stop_after_turn(self, ctx):
                return False

        hooks = compose_middleware([NoStop()])
        ctx = ShouldStopAfterTurnContext(
            message=AssistantMessage(content=[]),
            tool_results=[],
            context=AgentContext(system_prompt="", messages=[]),
            new_messages=[],
        )
        result = await hooks["should_stop_after_turn"](ctx)
        assert result is False


class TestPartialMiddleware:
    async def test_middleware_with_only_some_hooks(self):
        class OnlyTransform(Middleware):
            async def transform_context(self, messages, *, signal=None):
                return messages

        hooks = compose_middleware([OnlyTransform()])
        assert "transform_context" in hooks
        assert "convert_to_llm" not in hooks
        assert "before_tool_call" not in hooks


class TestBaseMiddlewareDefaults:
    """Each base Middleware method raises NotImplementedError if invoked directly.

    These default bodies are never reached through compose_middleware (it
    filters them out via _has_method), so we exercise them explicitly to
    guarantee the contract: subclasses must override what they declare.
    """

    async def test_transform_context_raises(self):
        import pytest

        with pytest.raises(NotImplementedError):
            await Middleware().transform_context([], signal=None)

    async def test_convert_to_llm_raises(self):
        import pytest

        with pytest.raises(NotImplementedError):
            await Middleware().convert_to_llm([])

    async def test_before_tool_call_raises(self):
        import pytest

        with pytest.raises(NotImplementedError):
            await Middleware().before_tool_call(None, signal=None)

    async def test_after_tool_call_raises(self):
        import pytest

        with pytest.raises(NotImplementedError):
            await Middleware().after_tool_call(None, signal=None)

    async def test_transform_system_prompt_raises(self):
        import pytest

        with pytest.raises(NotImplementedError):
            await Middleware().transform_system_prompt("hello", signal=None)

    async def test_should_stop_after_turn_raises(self):
        import pytest

        with pytest.raises(NotImplementedError):
            await Middleware().should_stop_after_turn(None)

    async def test_after_model_response_raises(self):
        import pytest

        with pytest.raises(NotImplementedError):
            await Middleware().after_model_response(
                AssistantMessage(content=[]), None, signal=None
            )
