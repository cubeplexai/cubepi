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

    def test_turn_end_event_typed_tool_results(self):
        msg = AssistantMessage(content=[TextContent(text="done")])
        tr = ToolResultMessage(
            tool_call_id="t1",
            tool_name="search",
            content=[TextContent(text="result")],
        )
        event = TurnEndEvent(message=msg, tool_results=[tr])
        assert len(event.tool_results) == 1
        assert event.tool_results[0].role == "tool_result"


class TestStructuredValueSerialization:
    """Regression: fields typed ``StructuredValue`` must dump the runtime
    subclass, not the declared ``BaseModel`` base. Without
    ``SerializeAsAny`` on the union branch, pydantic silently emits ``{}``
    for any concrete BaseModel held in such a field — which previously lost
    ``ToolResultMessage.details`` on checkpointer save and compaction
    message-ref hashing.
    """

    def _inner(self) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(text="inner")],
            details={"kind": "quote_result", "chip_metrics": {"price": 9}},
        )

    def test_tool_execution_end_event_preserves_basemodel_result(self):
        ev = ToolExecutionEndEvent(
            tool_call_id="t1", tool_name="quote", result=self._inner()
        )
        dumped = ev.model_dump()
        assert dumped["result"] != {}
        assert dumped["result"]["content"] == [{"type": "text", "text": "inner"}]
        assert dumped["result"]["details"] == {
            "kind": "quote_result",
            "chip_metrics": {"price": 9},
        }
        # json mode must agree with python mode
        assert ev.model_dump(mode="json")["result"] == dumped["result"]

    def test_tool_result_message_preserves_basemodel_details(self):
        msg = ToolResultMessage(
            tool_call_id="t1",
            tool_name="quote",
            content=[TextContent(text="hi")],
            details=self._inner(),
        )
        # Checkpointer path: plain model_dump()
        dumped = msg.model_dump()
        assert dumped["details"] != {}
        assert dumped["details"]["details"] == {
            "kind": "quote_result",
            "chip_metrics": {"price": 9},
        }
        # Compaction path: mode="json", exclude_none=True
        compact = msg.model_dump(mode="json", exclude_none=True)
        assert compact["details"]["content"] == [{"type": "text", "text": "inner"}]
        assert compact["content"] == [{"type": "text", "text": "hi"}]

    def test_after_tool_call_result_preserves_basemodel_details(self):
        r = AfterToolCallResult(details=self._inner())
        dumped = r.model_dump()
        assert dumped["details"] != {}
        assert dumped["details"]["content"] == [{"type": "text", "text": "inner"}]

    def test_basemodel_nested_in_list_and_dict_is_polymorphic(self):
        # list branch
        ev1 = ToolExecutionEndEvent(
            tool_call_id="t1", tool_name="x", result=[self._inner()]
        )
        assert ev1.model_dump()["result"][0]["details"] == {
            "kind": "quote_result",
            "chip_metrics": {"price": 9},
        }
        # dict branch
        ev2 = ToolExecutionEndEvent(
            tool_call_id="t1", tool_name="x", result={"wrap": self._inner()}
        )
        assert ev2.model_dump()["result"]["wrap"]["content"] == [
            {"type": "text", "text": "inner"}
        ]

    def test_plain_dict_value_still_roundtrips(self):
        # Non-BaseModel payloads must still validate and dump cleanly.
        msg = ToolResultMessage(
            tool_call_id="t1",
            tool_name="x",
            content=[TextContent(text="hi")],
            details={"k": [1, 2, {"nested": True}]},
        )
        dumped = msg.model_dump()
        restored = ToolResultMessage.model_validate(dumped)
        assert restored.details == {"k": [1, 2, {"nested": True}]}
