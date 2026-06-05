import asyncio

import pytest
from pydantic import Field

from cubepi import AgentTool, AgentToolResult, TextContent, tool
from cubepi.agent.tools import execute_tool_calls
from cubepi.agent.types import AgentContext
from cubepi.providers.base import AssistantMessage, ToolCall


def _ctx(tools: list[AgentTool]) -> AgentContext:
    return AgentContext(system_prompt="", messages=[], tools=tools)


def _assistant(tool_calls: list[ToolCall]) -> AssistantMessage:
    return AssistantMessage(content=tool_calls, stop_reason="tool_use")


class TestBuild:
    def test_bare_decorator_builds_agent_tool(self):
        @tool
        async def get_weather(city: str) -> str:
            "Get the current weather for a city."
            return f"sunny in {city}"

        assert isinstance(get_weather, AgentTool)
        assert get_weather.name == "get_weather"
        assert get_weather.description == "Get the current weather for a city."

    def test_schema_generated_from_signature(self):
        @tool
        async def search(query: str, limit: int = 10) -> str:
            "Search."
            return query

        schema = search.to_definition().parameters
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["limit"]["type"] == "integer"
        assert schema["properties"]["limit"]["default"] == 10
        # query is required, limit (with a default) is not
        assert schema["required"] == ["query"]

    def test_field_metadata_is_honoured(self):
        @tool
        async def search(query: str = Field(..., description="the query")) -> str:
            "Search."
            return query

        schema = search.to_definition().parameters
        assert schema["properties"]["query"]["description"] == "the query"

    def test_options_form(self):
        @tool(name="weather", description="override", execution_mode="sequential")
        async def get_weather(city: str) -> str:
            "docstring ignored when description given"
            return city

        assert get_weather.name == "weather"
        assert get_weather.description == "override"
        assert get_weather.execution_mode == "sequential"

    def test_no_params_tool(self):
        @tool
        async def ping() -> str:
            "Ping."
            return "pong"

        schema = ping.to_definition().parameters
        assert schema.get("properties", {}) == {}


class TestExecute:
    async def _run(self, t: AgentTool, **kwargs) -> AgentToolResult:
        params = t.parameters(**kwargs)
        return await t.execute("call-1", params, signal=None, on_update=None)

    async def test_str_return_is_wrapped(self):
        @tool
        async def echo(value: str) -> str:
            "Echo."
            return f"got {value}"

        result = await self._run(echo, value="hi")
        assert isinstance(result, AgentToolResult)
        assert result.content[0].text == "got hi"

    async def test_agent_tool_result_passthrough(self):
        @tool
        async def echo(value: str) -> AgentToolResult:
            "Echo."
            return AgentToolResult(content=[TextContent(text=value)], details={"k": 1})

        result = await self._run(echo, value="hi")
        assert result.details == {"k": 1}

    async def test_content_and_list_returns(self):
        @tool
        async def one(value: str) -> TextContent:
            "One."
            return TextContent(text=value)

        @tool
        async def many(value: str) -> list:
            "Many."
            return [TextContent(text=value), TextContent(text=value)]

        assert (await self._run(one, value="x")).content[0].text == "x"
        assert len((await self._run(many, value="x")).content) == 2

    async def test_injected_params_excluded_from_schema_but_passed(self):
        seen = {}

        @tool
        async def t(value: str, *, tool_call_id, signal=None, on_update=None) -> str:
            "Tool that uses injected args."
            seen["tool_call_id"] = tool_call_id
            seen["signal"] = signal
            seen["on_update"] = on_update
            return value

        # reserved names are not in the input schema
        props = t.to_definition().parameters["properties"]
        assert set(props) == {"value"}

        sig = asyncio.Event()
        cb = lambda _p: None  # noqa: E731
        params = t.parameters(value="hi")
        await t.execute("call-9", params, signal=sig, on_update=cb)
        assert seen == {"tool_call_id": "call-9", "signal": sig, "on_update": cb}


class TestErrors:
    def test_sync_function_rejected(self):
        with pytest.raises(TypeError, match="async function"):

            @tool
            def not_async(x: str) -> str:  # type: ignore[misc]
                return x

    def test_missing_annotation_rejected(self):
        with pytest.raises(TypeError, match="needs a type"):

            @tool
            async def bad(x) -> str:  # type: ignore[no-untyped-def]
                return x

    def test_var_args_rejected(self):
        with pytest.raises(TypeError, match="cannot use"):

            @tool
            async def bad(*args: str) -> str:
                return ""

    def test_positional_only_param_rejected(self):
        with pytest.raises(TypeError, match="positional-only"):

            @tool
            async def bad(id: str, /) -> str:
                "Positional-only is not callable with **kwargs."
                return id

    async def test_bad_return_type_rejected(self):
        @tool
        async def bad(value: str) -> int:  # type: ignore[return-value]
            "Bad."
            return 123  # type: ignore[return-value]

        params = bad.parameters(value="x")
        with pytest.raises(TypeError, match="returned int"):
            await bad.execute("c", params, signal=None, on_update=None)


class TestEngineIntegration:
    async def test_decorated_tool_runs_through_engine(self):
        """The generated execute signature must match what the loop calls."""

        @tool
        async def add(a: int, b: int) -> str:
            "Add two numbers."
            return str(a + b)

        ctx = _ctx([add])
        msg = _assistant([ToolCall(id="t1", name="add", arguments={"a": 2, "b": 3})])
        events: list = []
        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential", emit=lambda e: events.append(e)
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].tool_call_id == "t1"
        assert not batch.messages[0].is_error
        assert batch.messages[0].content[0].text == "5"

    async def test_is_error_result_surfaces_through_engine(self):
        """A tool that returns is_error=True (without raising) is an error."""

        @tool
        async def flaky(value: str) -> AgentToolResult:
            "Return an explicit error result."
            return AgentToolResult(content=[TextContent(text="nope")], is_error=True)

        ctx = _ctx([flaky])
        msg = _assistant([ToolCall(id="t1", name="flaky", arguments={"value": "x"})])
        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential", emit=lambda e: None
        )

        assert batch.messages[0].is_error
