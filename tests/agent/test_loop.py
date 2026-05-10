from typing import Any

from pydantic import BaseModel

from cubepi.agent.loop import run_agent_loop, run_agent_loop_continue
from cubepi.agent.types import (
    AgentContext,
    AgentEvent,
    AgentTool,
    AgentToolResult,
)
from cubepi.providers.base import (
    Message,
    Model,
    TextContent,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_tool_call


def make_model() -> Model:
    return Model(id="faux-1", provider="faux")


def make_user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)])


def identity_converter(messages: list[Any]) -> list[Message]:
    return [
        m
        for m in messages
        if hasattr(m, "role") and m.role in ("user", "assistant", "tool_result")
    ]


class EchoParams(BaseModel):
    value: str


def make_echo_tool(*, execution_mode=None, execute_fn=None) -> AgentTool:
    async def default_execute(tool_call_id, params, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echoed: {params.value}")])

    return AgentTool(
        name="echo",
        description="Echo tool",
        parameters=EchoParams,
        execute=execute_fn or default_execute,
        execution_mode=execution_mode,
    )


class TestAgentLoop:
    async def test_emit_events_with_agent_message_types(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Hi there!")])
        context = AgentContext(system_prompt="You are helpful.", messages=[], tools=[])
        user_prompt = make_user_message("Hello")

        events: list[AgentEvent] = []
        messages = await run_agent_loop(
            prompts=[user_prompt],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            emit=lambda e: events.append(e),
        )

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "turn_start" in event_types
        assert "message_start" in event_types
        assert "message_end" in event_types
        assert "turn_end" in event_types
        assert "agent_end" in event_types

    async def test_custom_message_types_via_convert_to_llm(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Response")])

        notification = {"role": "notification", "text": "info"}
        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[notification],
            tools=[],
        )
        user_prompt = make_user_message("Hello")

        converted: list[Message] = []

        def converter(messages):
            result = [
                m
                for m in messages
                if hasattr(m, "role") and m.role in ("user", "assistant", "tool_result")
            ]
            converted.extend(result)
            return result

        await run_agent_loop(
            prompts=[user_prompt],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=converter,
            emit=lambda e: None,
        )

        assert len(converted) == 1
        assert converted[0].role == "user"

    async def test_transform_context_before_convert_to_llm(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Response")])

        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[
                make_user_message("old 1"),
                faux_assistant_message("old resp 1"),
                make_user_message("old 2"),
                faux_assistant_message("old resp 2"),
            ],
            tools=[],
        )

        transformed_len = []
        converted_len = []

        async def transform(messages, *, signal=None):
            result = messages[-2:]
            transformed_len.append(len(result))
            return result

        def converter(messages):
            converted_len.append(len(messages))
            return identity_converter(messages)

        await run_agent_loop(
            prompts=[make_user_message("new")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=converter,
            transform_context=transform,
            emit=lambda e: None,
        )

        assert transformed_len[0] == 2
        assert converted_len[0] == 2

    async def test_tool_calls_and_results(self):
        executed = []

        async def echo_execute(tool_call_id, params, *, signal=None, on_update=None):
            executed.append(params.value)
            return AgentToolResult(
                content=[TextContent(text=f"echoed: {params.value}")]
            )

        tool = make_echo_tool(execute_fn=echo_execute)
        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message(
                    [faux_tool_call("echo", {"value": "hello"}, id="tool-1")],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )

        context = AgentContext(system_prompt="", messages=[], tools=[tool])
        events: list[AgentEvent] = []

        await run_agent_loop(
            prompts=[make_user_message("echo something")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            emit=lambda e: events.append(e),
        )

        assert executed == ["hello"]
        event_types = [e.type for e in events]
        assert "tool_execution_start" in event_types
        assert "tool_execution_end" in event_types

    async def test_should_stop_after_turn(self):
        tool = make_echo_tool()
        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message(
                    [faux_tool_call("echo", {"value": "hello"}, id="tool-1")],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("should not run"),
            ]
        )

        context = AgentContext(system_prompt="", messages=[], tools=[tool])
        stop_called = []

        async def should_stop(ctx):
            stop_called.append(True)
            return True

        events: list[AgentEvent] = []
        messages = await run_agent_loop(
            prompts=[make_user_message("echo something")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            should_stop_after_turn=should_stop,
            emit=lambda e: events.append(e),
        )

        assert len(stop_called) == 1
        assert provider.call_count == 1
        roles = [m.role for m in messages]
        assert roles == ["user", "assistant", "tool_result"]

    async def test_steering_messages_injected_after_tool_calls(self):
        executed = []

        async def echo_execute(tool_call_id, params, *, signal=None, on_update=None):
            executed.append(params.value)
            return AgentToolResult(content=[TextContent(text=f"ok:{params.value}")])

        tool = make_echo_tool(execute_fn=echo_execute)
        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message(
                    [
                        faux_tool_call("echo", {"value": "first"}, id="tool-1"),
                        faux_tool_call("echo", {"value": "second"}, id="tool-2"),
                    ],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )

        context = AgentContext(system_prompt="", messages=[], tools=[tool])
        steering_delivered = False

        async def get_steering():
            nonlocal steering_delivered
            if len(executed) >= 1 and not steering_delivered:
                steering_delivered = True
                return [make_user_message("interrupt")]
            return []

        events: list[AgentEvent] = []
        await run_agent_loop(
            prompts=[make_user_message("start")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            get_steering_messages=get_steering,
            tool_execution="sequential",
            emit=lambda e: events.append(e),
        )

        assert executed == ["first", "second"]

    async def test_terminate_when_all_tool_results_terminate(self):
        async def term_execute(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="done")], terminate=True)

        tool = make_echo_tool(execute_fn=term_execute)
        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message(
                    [faux_tool_call("echo", {"value": "hello"}, id="tool-1")],
                    stop_reason="tool_use",
                ),
            ]
        )

        context = AgentContext(system_prompt="", messages=[], tools=[tool])
        messages = await run_agent_loop(
            prompts=[make_user_message("echo something")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            emit=lambda e: None,
        )

        assert provider.call_count == 1
        roles = [m.role for m in messages]
        assert roles == ["user", "assistant", "tool_result"]

    async def test_steering_messages_polled_before_first_turn(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Got it")])

        context = AgentContext(system_prompt="", messages=[], tools=[])
        call_count = 0

        async def get_steering():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [make_user_message("steering hint")]
            return []

        events: list[AgentEvent] = []
        messages = await run_agent_loop(
            prompts=[make_user_message("Hello")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            get_steering_messages=get_steering,
            emit=lambda e: events.append(e),
        )

        # Messages should be: initial prompt, steering message, assistant response
        roles = [m.role for m in messages]
        assert roles == ["user", "user", "assistant"]

        # Verify event ordering: message_end events should reflect
        # initial prompt -> steering message -> assistant response
        message_ends = [e for e in events if e.type == "message_end"]
        assert len(message_ends) == 3
        assert message_ends[0].message.role == "user"  # initial prompt
        assert message_ends[1].message.role == "user"  # steering message
        assert message_ends[2].message.role == "assistant"  # response

    async def test_error_stop_reason_ends_loop(self):
        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message(
                    "", stop_reason="error", error_message="API error"
                ),
            ]
        )

        context = AgentContext(system_prompt="", messages=[], tools=[])
        events: list[AgentEvent] = []

        messages = await run_agent_loop(
            prompts=[make_user_message("hello")],
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            emit=lambda e: events.append(e),
        )

        event_types = [e.type for e in events]
        assert "agent_end" in event_types
        assert messages[-1].role == "assistant"
        assert messages[-1].stop_reason == "error"


class TestAgentLoopContinue:
    async def test_raises_when_no_messages(self):
        provider = FauxProvider()
        context = AgentContext(system_prompt="", messages=[], tools=[])

        try:
            await run_agent_loop_continue(
                context=context,
                provider=provider,
                model=make_model(),
                convert_to_llm=identity_converter,
                emit=lambda e: None,
            )
            assert False, "Should have raised"
        except ValueError as e:
            assert "no messages" in str(e).lower()

    async def test_continue_without_user_message_events(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Response")])

        context = AgentContext(
            system_prompt="",
            messages=[make_user_message("Hello")],
            tools=[],
        )

        events: list[AgentEvent] = []
        messages = await run_agent_loop_continue(
            context=context,
            provider=provider,
            model=make_model(),
            convert_to_llm=identity_converter,
            emit=lambda e: events.append(e),
        )

        assert len(messages) == 1
        assert messages[0].role == "assistant"

        message_ends = [e for e in events if e.type == "message_end"]
        assert len(message_ends) == 1
        assert message_ends[0].message.role == "assistant"
