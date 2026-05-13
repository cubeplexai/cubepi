"""MCP adapter unit tests (D2.1)."""

import pytest

from cubepi.mcp._adapter import (
    make_mcp_agent_tool,
    mcp_schema_to_pydantic_model,
)


def test_schema_to_model_required_field() -> None:
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["city"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="search", input_schema=schema)
    instance = M(city="Tokyo")
    assert instance.city == "Tokyo"
    assert instance.limit is None


def test_schema_to_model_array_field() -> None:
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["tags"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="tag", input_schema=schema)
    instance = M(tags=["a", "b"])
    assert instance.tags == ["a", "b"]


def test_schema_to_model_boolean_and_number() -> None:
    schema = {
        "type": "object",
        "properties": {
            "active": {"type": "boolean"},
            "rate": {"type": "number"},
        },
        "required": ["active", "rate"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="x", input_schema=schema)
    instance = M(active=True, rate=1.5)
    assert instance.active is True
    assert instance.rate == 1.5


def test_schema_to_model_unknown_type_becomes_any() -> None:
    """An unrecognized JSON Schema type falls back to typing.Any."""
    from typing import Any

    schema = {
        "type": "object",
        "properties": {
            # Neither a known scalar nor array/object — must hit the Any fallback.
            "weird": {"type": "totally-not-a-real-type"},
        },
        "required": [],
    }
    M = mcp_schema_to_pydantic_model(tool_name="x", input_schema=schema)
    # Field accepts arbitrary values because its type is Any.
    instance = M(weird={"anything": [1, 2, 3]})
    assert instance.weird == {"anything": [1, 2, 3]}
    annotations = M.model_fields["weird"].annotation
    assert annotations is Any


def test_schema_to_model_object_field_becomes_dict() -> None:
    schema = {
        "type": "object",
        "properties": {
            "config": {"type": "object"},
        },
        "required": [],
    }
    M = mcp_schema_to_pydantic_model(tool_name="x", input_schema=schema)
    instance = M(config={"a": 1})
    assert instance.config == {"a": 1}


@pytest.mark.asyncio
async def test_make_mcp_agent_tool_routes_to_call_remote() -> None:
    called: dict = {}

    async def _fake_call(name, args):
        called["name"] = name
        called["args"] = args
        return {
            "content": [{"type": "text", "text": "result"}],
            "isError": False,
        }

    tool = make_mcp_agent_tool(
        name="search",
        description="Search the web",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        call_remote=_fake_call,
    )
    assert tool.name == "search"
    assert tool.description == "Search the web"

    args = tool.parameters(query="cats")
    # Use the production signature: tool_call_id positional, then args, then keyword-only
    result = await tool.execute("test-call-id-1", args, signal=None, on_update=None)
    assert called == {"name": "search", "args": {"query": "cats"}}
    assert len(result.content) == 1
    from cubepi.providers.base import TextContent

    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "result"


@pytest.mark.asyncio
async def test_make_mcp_agent_tool_carries_raw_response_in_details() -> None:
    async def _fake_call(name, args):
        return {"content": [], "isError": True, "errorMessage": "boom"}

    tool = make_mcp_agent_tool(
        name="bad",
        description="",
        input_schema={"type": "object", "properties": {}, "required": []},
        call_remote=_fake_call,
    )
    args = tool.parameters()
    result = await tool.execute("test-call-id-2", args, signal=None, on_update=None)
    assert result.details == {
        "raw_mcp_response": {"content": [], "isError": True, "errorMessage": "boom"}
    }
    # isError from MCP response must be reflected on AgentToolResult
    assert result.is_error is True


@pytest.mark.asyncio
async def test_make_mcp_agent_tool_omits_none_optional_args() -> None:
    """Optional schema field absent from args → omitted from call (not sent as null)."""
    captured: dict = {}

    async def _fake_call(name, args):
        captured["args"] = args
        return {"content": [], "isError": False}

    tool = make_mcp_agent_tool(
        name="search",
        description="",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        call_remote=_fake_call,
    )
    args = tool.parameters(query="cats")  # limit not passed
    await tool.execute("tc-3", args, signal=None, on_update=None)
    # Critical: 'limit' must NOT appear in args (not even as null)
    assert "limit" not in captured["args"], (
        f"Optional field not provided should be omitted, got: {captured['args']!r}"
    )
    assert captured["args"] == {"query": "cats"}
