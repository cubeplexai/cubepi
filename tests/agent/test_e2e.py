import asyncio

from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import (
    AssistantMessage,
    Model,
    ReasoningControl,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from cubepi.providers.faux import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)


def make_model() -> Model:
    return Model(id="faux-1", provider_id="faux")


class CalculateParams(BaseModel):
    expression: str


def make_calculate_tool() -> AgentTool:
    async def execute(tool_call_id, params, *, signal=None, on_update=None):
        try:
            result = eval(params.expression)
            return AgentToolResult(
                content=[TextContent(text=f"{params.expression} = {result}")]
            )
        except Exception as e:
            raise RuntimeError(str(e))

    return AgentTool(
        name="calculate",
        description="Calculate a math expression",
        parameters=CalculateParams,
        execute=execute,
    )


class TestE2EBasic:
    async def test_basic_text_prompt(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses([faux_assistant_message("4")])
        agent = Agent(
            model=provider.model("faux-1"),
            system_prompt="You are a helpful assistant.",
        )

        await agent.prompt("What is 2+2?")

        assert agent.state.is_streaming is False
        assert len(agent.state.messages) == 2
        assert agent.state.messages[0].role == "user"
        assert agent.state.messages[1].role == "assistant"
        assert "4" in agent.state.messages[1].content[0].text

    async def test_tool_execution_with_pending_tracking(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses(
            [
                faux_assistant_message(
                    [
                        faux_text("Let me calculate that."),
                        faux_tool_call(
                            "calculate", {"expression": "123 * 456"}, id="calc-1"
                        ),
                    ],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("The result is 56088."),
            ]
        )
        agent = Agent(
            model=provider.model("faux-1"),
            system_prompt="Always use the calculator tool for math.",
            tools=[make_calculate_tool()],
        )

        pending_events = []
        agent.subscribe(
            lambda e, s=None: (
                pending_events.append(
                    {"type": e.type, "ids": list(agent.state.pending_tool_calls)}
                )
                if e.type in ("tool_execution_start", "tool_execution_end")
                else None
            )
        )

        await agent.prompt("Calculate 123 * 456")

        assert agent.state.is_streaming is False
        assert len(agent.state.messages) >= 4
        tool_result = next(m for m in agent.state.messages if m.role == "tool_result")
        assert "56088" in tool_result.content[0].text
        assert agent.state.pending_tool_calls == set()

    async def test_abort_during_streaming(self):
        provider = FauxProvider(
            tokens_per_second=20, token_size_min=2, token_size_max=2
        )
        provider.set_responses(
            [
                faux_assistant_message(
                    "one two three four five six seven eight nine ten eleven twelve thirteen"
                ),
            ]
        )
        agent = Agent(model=provider.model("faux-1"))

        prompt_task = asyncio.create_task(agent.prompt("Count"))
        await asyncio.sleep(0.03)
        agent.abort()
        await prompt_task

        assert agent.state.is_streaming is False
        last_msg = agent.state.messages[-1]
        assert last_msg.role == "assistant"
        assert last_msg.stop_reason == "aborted"

    async def test_lifecycle_events_during_streaming(self):
        provider = FauxProvider(token_size_min=1, token_size_max=1)
        provider.set_responses([faux_assistant_message("1 2 3 4 5")])
        agent = Agent(model=provider.model("faux-1"))

        events = []
        agent.subscribe(lambda e, s=None: events.append(e.type))

        await agent.prompt("Count from 1 to 5")

        assert "agent_start" in events
        assert "message_start" in events
        assert "message_update" in events
        assert "message_end" in events
        assert "agent_end" in events

    async def test_context_across_multiple_turns(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses(
            [
                faux_assistant_message("Nice to meet you, Alice."),
                lambda msgs, model: faux_assistant_message(
                    "Your name is Alice."
                    if any(
                        hasattr(m, "content")
                        and isinstance(m.content, list)
                        and any(
                            hasattr(c, "text") and "Alice" in c.text for c in m.content
                        )
                        for m in msgs
                        if hasattr(m, "role") and m.role == "user"
                    )
                    else "I don't know your name."
                ),
            ]
        )
        agent = Agent(model=provider.model("faux-1"))

        await agent.prompt("My name is Alice.")
        assert len(agent.state.messages) == 2

        await agent.prompt("What is my name?")
        assert len(agent.state.messages) == 4
        assert "alice" in agent.state.messages[3].content[0].text.lower()

    async def test_thinking_content_preserved(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses(
            [
                faux_assistant_message([faux_thinking("step by step"), faux_text("4")]),
            ]
        )
        agent = Agent(
            model=provider.model("faux-reasoning", reasoning=True),
            reasoning=ReasoningControl(mode="on", effort="low"),
        )

        await agent.prompt("What is 2+2?")

        assistant_msg = agent.state.messages[1]
        assert assistant_msg.content[0].type == "thinking"
        assert assistant_msg.content[0].thinking == "step by step"
        assert assistant_msg.content[1].type == "text"
        assert assistant_msg.content[1].text == "4"


class TestE2EResume:
    async def test_raises_when_no_messages(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))

        try:
            await agent.resume()
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "no messages" in str(e).lower()

    async def test_continue_from_user_message(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses([faux_assistant_message("HELLO WORLD")])
        agent = Agent(model=provider.model("faux-1"))

        agent.state.messages = [
            UserMessage(content=[TextContent(text="Say HELLO WORLD")]),
        ]

        await agent.resume()

        assert agent.state.is_streaming is False
        assert len(agent.state.messages) == 2
        assert agent.state.messages[1].role == "assistant"

    async def test_continue_from_tool_result(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses([faux_assistant_message("The answer is 8.")])
        agent = Agent(
            model=provider.model("faux-1"),
            tools=[make_calculate_tool()],
        )

        agent.state.messages = [
            UserMessage(content=[TextContent(text="What is 5 + 3?")]),
            AssistantMessage(
                content=[
                    TextContent(text="Let me calculate."),
                    ToolCall(
                        id="calc-1", name="calculate", arguments={"expression": "5 + 3"}
                    ),
                ],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="calc-1",
                tool_name="calculate",
                content=[TextContent(text="5 + 3 = 8")],
            ),
        ]

        await agent.resume()

        assert len(agent.state.messages) >= 4
        assert agent.state.messages[-1].role == "assistant"
        assert "8" in agent.state.messages[-1].content[0].text
