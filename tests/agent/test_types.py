from pydantic import BaseModel

from cubepi.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
    AfterToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from cubepi.providers.base import (
    AssistantMessage,
    StreamEvent,
    TextContent,
    ToolResultMessage,
    UserMessage,
)


class TestAgentEvents:
    def test_agent_start_event(self):
        e = AgentStartEvent()
        assert e.type == "agent_start"

    def test_agent_end_event(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        e = AgentEndEvent(messages=[msg])
        assert e.type == "agent_end"
        assert len(e.messages) == 1

    def test_turn_start_event(self):
        e = TurnStartEvent()
        assert e.type == "turn_start"

    def test_turn_end_event(self):
        msg = AssistantMessage(content=[TextContent(text="hi")])
        e = TurnEndEvent(message=msg, tool_results=[])
        assert e.type == "turn_end"

    def test_message_start_event(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        e = MessageStartEvent(message=msg)
        assert e.type == "message_start"

    def test_message_update_event(self):
        msg = AssistantMessage(content=[TextContent(text="h")])
        se = StreamEvent(type="text_delta", delta="h")
        e = MessageUpdateEvent(message=msg, stream_event=se)
        assert e.type == "message_update"

    def test_message_end_event(self):
        msg = AssistantMessage(content=[TextContent(text="hello")])
        e = MessageEndEvent(message=msg)
        assert e.type == "message_end"

    def test_tool_execution_events(self):
        start = ToolExecutionStartEvent(
            tool_call_id="t1", tool_name="search", args={"q": "test"}
        )
        assert start.type == "tool_execution_start"

        update = ToolExecutionUpdateEvent(
            tool_call_id="t1",
            tool_name="search",
            args={"q": "test"},
            partial_result=AgentToolResult(content=[TextContent(text="partial")]),
        )
        assert update.type == "tool_execution_update"

        end = ToolExecutionEndEvent(
            tool_call_id="t1",
            tool_name="search",
            result=AgentToolResult(content=[TextContent(text="done")]),
            is_error=False,
        )
        assert end.type == "tool_execution_end"


class TestAgentTool:
    async def test_tool_definition_generation(self):
        class SearchParams(BaseModel):
            query: str
            limit: int = 10

        async def execute(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text=f"found: {params.query}")])

        tool = AgentTool(
            name="search",
            description="Search the web",
            parameters=SearchParams,
            execute=execute,
        )

        defn = tool.to_definition()
        assert defn.name == "search"
        assert defn.description == "Search the web"
        assert "query" in defn.parameters.get("properties", {})
        assert "limit" in defn.parameters.get("properties", {})

    async def test_tool_execution(self):
        class EchoParams(BaseModel):
            text: str

        async def execute(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text=params.text)])

        tool = AgentTool(
            name="echo",
            description="Echo text",
            parameters=EchoParams,
            execute=execute,
        )

        result = await tool.execute("t1", EchoParams(text="hello"))
        assert result.content[0].text == "hello"


class TestAgentContext:
    def test_context_creation(self):
        ctx = AgentContext(system_prompt="You are helpful.", messages=[], tools=[])
        assert ctx.system_prompt == "You are helpful."
        assert ctx.messages == []


class TestHookTypes:
    def test_before_tool_call_result_defaults(self):
        r = BeforeToolCallResult()
        assert r.block is False
        assert r.reason is None

    def test_before_tool_call_result_block(self):
        r = BeforeToolCallResult(block=True, reason="Not allowed")
        assert r.block is True
        assert r.reason == "Not allowed"

    def test_after_tool_call_result_partial_override(self):
        r = AfterToolCallResult(terminate=True)
        assert r.terminate is True
        assert r.content is None
        assert r.is_error is None


class TestTypedMessages:
    def test_agent_context_accepts_message_union(self):
        ctx = AgentContext(
            system_prompt="test",
            messages=[
                UserMessage(content=[TextContent(text="hi")]),
                AssistantMessage(content=[TextContent(text="hello")]),
            ],
        )
        assert len(ctx.messages) == 2

    def test_agent_end_event_typed_messages(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        event = AgentEndEvent(messages=[msg])
        assert event.messages[0].role == "user"

    def test_turn_end_event_typed_message(self):
        msg = AssistantMessage(content=[TextContent(text="done")])
        event = TurnEndEvent(message=msg, tool_results=[])
        assert event.message.role == "assistant"

    def test_message_start_event_typed(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        event = MessageStartEvent(message=msg)
        assert event.message.role == "user"
