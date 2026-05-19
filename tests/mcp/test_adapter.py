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


def test_schema_to_model_preserves_enum() -> None:
    """enum becomes Literal — invalid values rejected by Pydantic."""
    schema = {
        "type": "object",
        "properties": {
            "unit": {"type": "string", "enum": ["c", "f"]},
        },
        "required": ["unit"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="weather", input_schema=schema)
    assert M(unit="c").unit == "c"
    with pytest.raises(Exception):  # noqa: BLE001 - Pydantic ValidationError
        M(unit="kelvin")


def test_schema_to_model_preserves_string_constraints() -> None:
    schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "pattern": "^[A-Z]{3}$",
                "minLength": 3,
                "maxLength": 3,
                "description": "ISO airport code",
            },
        },
        "required": ["code"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="ap", input_schema=schema)
    assert M(code="SFO").code == "SFO"
    with pytest.raises(Exception):
        M(code="sfo")  # lowercase fails pattern
    with pytest.raises(Exception):
        M(code="TOOLONG")  # exceeds maxLength
    assert M.model_fields["code"].description == "ISO airport code"


def test_schema_to_model_preserves_numeric_bounds() -> None:
    schema = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "exclusiveMinimum": 0,
            },
        },
        "required": ["limit"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="pg", input_schema=schema)
    assert M(limit=50).limit == 50
    with pytest.raises(Exception):
        M(limit=0)
    with pytest.raises(Exception):
        M(limit=101)


def test_schema_to_model_preserves_array_size() -> None:
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 3,
            },
        },
        "required": ["tags"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="tag", input_schema=schema)
    assert M(tags=["a", "b"]).tags == ["a", "b"]
    with pytest.raises(Exception):
        M(tags=[])
    with pytest.raises(Exception):
        M(tags=["a", "b", "c", "d"])


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


async def test_signal_abort_cancels_in_flight_mcp_call() -> None:
    """``agent.abort()`` only sets a cooperative ``asyncio.Event``;
    it does NOT cancel the running task. An in-flight MCP
    ``tools/call`` would block until the response or transport
    timeout regardless of abort. The adapter must watch the signal
    and bail out with ``CancelledError`` so cancellation latency
    isn't bounded by MCP timeout (codex overall-review MAJOR)."""
    import asyncio as _asyncio

    from cubepi.mcp._adapter import make_mcp_agent_tool

    started = _asyncio.Event()
    finished = _asyncio.Event()

    async def _slow_call(_name: str, _args: dict) -> dict:
        started.set()
        try:
            await _asyncio.sleep(10.0)  # would block well past test budget
        except _asyncio.CancelledError:
            finished.set()
            raise
        return {"content": [], "isError": False}

    tool = make_mcp_agent_tool(
        name="slow",
        description="slow tool",
        input_schema={"type": "object", "properties": {}, "required": []},
        call_remote=_slow_call,
    )
    signal = _asyncio.Event()
    args = tool.parameters()
    task = _asyncio.create_task(
        tool.execute("tc-abort", args, signal=signal, on_update=None)
    )
    await started.wait()
    # Cooperative abort: set the signal but do NOT cancel the task.
    signal.set()
    # Adapter must propagate as CancelledError.
    try:
        await _asyncio.wait_for(task, timeout=1.0)
    except _asyncio.CancelledError:
        pass
    except _asyncio.TimeoutError:
        task.cancel()
        raise AssertionError(
            "MCP adapter ignored abort signal; tools/call did not bail out"
        )
    # The inner call_remote should have been cancelled too.
    assert finished.is_set(), "call_remote was not cancelled when signal fired"


async def test_signal_already_set_before_call_aborts_immediately() -> None:
    import asyncio as _asyncio

    from cubepi.mcp._adapter import make_mcp_agent_tool

    called = False

    async def _call(_name: str, _args: dict) -> dict:
        nonlocal called
        called = True
        return {"content": [], "isError": False}

    tool = make_mcp_agent_tool(
        name="t",
        description="t",
        input_schema={"type": "object", "properties": {}, "required": []},
        call_remote=_call,
    )
    signal = _asyncio.Event()
    signal.set()  # already aborted before _execute even starts
    try:
        await tool.execute("tc-pre", tool.parameters(), signal=signal, on_update=None)
    except _asyncio.CancelledError:
        pass
    else:
        raise AssertionError(
            "pre-set signal should have raised CancelledError before call_remote"
        )
