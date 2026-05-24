import asyncio

import pytest
from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import (
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from cubepi.providers.faux import (
    FauxProvider,
    faux_assistant_message,
    faux_tool_call,
)


def make_model() -> Model:
    return Model(id="faux-1", provider="faux")


class TestCheckpointerIntegration:
    async def test_messages_persisted_on_message_end(self):
        checkpointer = MemoryCheckpointer()
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Hello!")])
        agent = Agent(
            provider=provider,
            model=make_model(),
            checkpointer=checkpointer,
            thread_id="thread-1",
        )
        await agent.prompt("Hi")
        data = await checkpointer.load("thread-1")
        assert data is not None
        assert len(data.messages) == 2  # user + assistant

    async def test_history_restored_on_prompt(self):
        checkpointer = MemoryCheckpointer()
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("First reply")])
        agent1 = Agent(
            provider=provider,
            model=make_model(),
            checkpointer=checkpointer,
            thread_id="thread-1",
        )
        await agent1.prompt("First message")

        provider.set_responses([faux_assistant_message("Second reply")])
        agent2 = Agent(
            provider=provider,
            model=make_model(),
            checkpointer=checkpointer,
            thread_id="thread-1",
        )
        await agent2.prompt("Second message")
        assert len(agent2.state.messages) == 4  # 2 from first + 2 from second

    async def test_no_checkpointer_works_as_before(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Hi")])
        agent = Agent(provider=provider, model=make_model())
        await agent.prompt("Hello")
        assert len(agent.state.messages) == 2

    async def test_tool_use_messages_persisted(self):
        """Checkpointer persists the full tool-use conversation:
        user, assistant (tool call), tool result, final assistant."""

        class EchoParams(BaseModel):
            text: str

        async def echo_execute(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text=f"echo: {params.text}")])

        echo_tool = AgentTool(
            name="echo",
            description="Echo the input text",
            parameters=EchoParams,
            execute=echo_execute,
        )

        checkpointer = MemoryCheckpointer()
        provider = FauxProvider()
        # First response: assistant calls the echo tool
        # Second response: assistant gives a final text answer
        provider.set_responses(
            [
                faux_assistant_message(
                    faux_tool_call("echo", {"text": "hello"}, id="tc-1"),
                    stop_reason="tool_use",
                ),
                faux_assistant_message("Done! The echo said hello."),
            ]
        )
        agent = Agent(
            provider=provider,
            model=make_model(),
            tools=[echo_tool],
            checkpointer=checkpointer,
            thread_id="thread-tool",
        )

        await agent.prompt("Please echo hello")

        data = await checkpointer.load("thread-tool")
        assert data is not None
        # user + assistant(tool_call) + tool_result + final assistant = 4
        assert len(data.messages) == 4

        # Verify message roles in order
        assert data.messages[0].role == "user"
        assert data.messages[1].role == "assistant"
        assert data.messages[2].role == "tool_result"
        assert data.messages[3].role == "assistant"

        # Verify tool result content
        tool_result = data.messages[2]
        assert isinstance(tool_result, ToolResultMessage)
        assert tool_result.tool_call_id == "tc-1"
        assert tool_result.tool_name == "echo"
        assert any(
            hasattr(c, "text") and "echo: hello" in c.text for c in tool_result.content
        )

    async def test_tool_use_history_restored(self):
        """A second Agent session restores tool-use history and continues."""

        class EchoParams(BaseModel):
            text: str

        async def echo_execute(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text=f"echo: {params.text}")])

        echo_tool = AgentTool(
            name="echo",
            description="Echo the input text",
            parameters=EchoParams,
            execute=echo_execute,
        )

        checkpointer = MemoryCheckpointer()
        provider = FauxProvider()

        # --- First session: tool-use conversation ---
        provider.set_responses(
            [
                faux_assistant_message(
                    faux_tool_call("echo", {"text": "hi"}, id="tc-1"),
                    stop_reason="tool_use",
                ),
                faux_assistant_message("The echo returned hi."),
            ]
        )
        agent1 = Agent(
            provider=provider,
            model=make_model(),
            tools=[echo_tool],
            checkpointer=checkpointer,
            thread_id="thread-tool-restore",
        )
        await agent1.prompt("Echo hi")
        # 4 messages after first session
        assert len(agent1.state.messages) == 4

        # --- Second session: same thread, new Agent ---
        provider.set_responses([faux_assistant_message("Sure, continuing.")])
        agent2 = Agent(
            provider=provider,
            model=make_model(),
            tools=[echo_tool],
            checkpointer=checkpointer,
            thread_id="thread-tool-restore",
        )
        await agent2.prompt("Continue please")
        # 4 restored + 2 new (user + assistant) = 6
        assert len(agent2.state.messages) == 6

        # Verify the restored history kept tool-use messages
        assert agent2.state.messages[0].role == "user"
        assert agent2.state.messages[1].role == "assistant"
        assert agent2.state.messages[2].role == "tool_result"
        assert agent2.state.messages[3].role == "assistant"
        # New messages
        assert agent2.state.messages[4].role == "user"
        assert agent2.state.messages[5].role == "assistant"

    async def test_cancel_mid_tool_backfills_tool_result(self):
        """A cancel during tool execution must not leave an orphan tool_call.

        The assistant message (with the tool_call) is checkpointed before the
        tool runs; if the tool execution is cancelled, the loop backfills a
        synthetic tool_result so the persisted thread stays resumable.
        """

        class BlockParams(BaseModel):
            pass

        async def cancel_execute(tool_call_id, params, *, signal=None, on_update=None):
            raise asyncio.CancelledError()

        block_tool = AgentTool(
            name="block",
            description="Never returns; simulates a cancelled tool call",
            parameters=BlockParams,
            execute=cancel_execute,
        )

        checkpointer = MemoryCheckpointer()
        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message(
                    faux_tool_call("block", {}, id="tc-cancel"),
                    stop_reason="tool_use",
                ),
            ]
        )
        agent = Agent(
            provider=provider,
            model=make_model(),
            tools=[block_tool],
            checkpointer=checkpointer,
            thread_id="thread-cancel",
        )

        with pytest.raises(asyncio.CancelledError):
            await agent.prompt("Run the blocking tool")

        data = await checkpointer.load("thread-cancel")
        assert data is not None
        # user + assistant(tool_call) + synthetic tool_result = 3
        assert [m.role for m in data.messages] == ["user", "assistant", "tool_result"]
        tool_result = data.messages[2]
        assert isinstance(tool_result, ToolResultMessage)
        assert tool_result.tool_call_id == "tc-cancel"
        assert tool_result.is_error is True

    async def test_cancel_backfill_handles_reused_tool_call_id(self):
        """A reused tool_call_id from an earlier (answered) turn must not
        suppress the backfill for the current cancelled turn.

        tool_call ids carry no global-uniqueness guarantee, so 'answered'
        must be scoped to results after the last assistant message.
        """

        class BlockParams(BaseModel):
            pass

        async def cancel_execute(tool_call_id, params, *, signal=None, on_update=None):
            raise asyncio.CancelledError()

        block_tool = AgentTool(
            name="block",
            description="Cancels on execution",
            parameters=BlockParams,
            execute=cancel_execute,
        )

        checkpointer = MemoryCheckpointer()
        # Pre-seed an earlier, fully-answered turn that already used id "dup".
        await checkpointer.append(
            "thread-reuse",
            [
                UserMessage(content=[TextContent(text="first")]),
                AssistantMessage(
                    content=[ToolCall(id="dup", name="block", arguments={})],
                    stop_reason="tool_use",
                ),
                ToolResultMessage(
                    tool_call_id="dup",
                    tool_name="block",
                    content=[TextContent(text="ok")],
                ),
            ],
        )

        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message(
                    faux_tool_call("block", {}, id="dup"),
                    stop_reason="tool_use",
                ),
            ]
        )
        agent = Agent(
            provider=provider,
            model=make_model(),
            tools=[block_tool],
            checkpointer=checkpointer,
            thread_id="thread-reuse",
        )

        with pytest.raises(asyncio.CancelledError):
            await agent.prompt("second")

        data = await checkpointer.load("thread-reuse")
        assert data is not None
        # Earlier turn (3) + new user + assistant + synthetic tool_result = 6
        assert [m.role for m in data.messages] == [
            "user",
            "assistant",
            "tool_result",
            "user",
            "assistant",
            "tool_result",
        ]
        backfilled = data.messages[5]
        assert isinstance(backfilled, ToolResultMessage)
        assert backfilled.tool_call_id == "dup"
        assert backfilled.is_error is True
