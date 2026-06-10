from __future__ import annotations

from cubepi.deferred._expand_tool import (
    TOOL_NAME,
    LoadToolsInput,
    LoadToolsOutput,
    _make_load_tools,
)
from cubepi.agent.types import AgentTool


class TestLoadToolsInput:
    def test_group_id_only(self) -> None:
        inp = LoadToolsInput(group_id="mcp:github")
        assert inp.group_id == "mcp:github"
        assert inp.tool_names is None

    def test_group_id_with_tool_names(self) -> None:
        inp = LoadToolsInput(group_id="mcp:github", tool_names=["create_issue"])
        assert inp.tool_names == ["create_issue"]


class TestLoadToolsOutput:
    def test_success_output(self) -> None:
        out = LoadToolsOutput(
            group_id="mcp:github",
            expanded=True,
            tool_names=["create_issue"],
            remaining=5,
        )
        assert out.expanded is True
        assert out.error is None

    def test_error_output(self) -> None:
        out = LoadToolsOutput(
            group_id="bad:id",
            expanded=False,
            tool_names=[],
            remaining=0,
            error="Unknown group_id: bad:id",
        )
        assert out.expanded is False
        assert out.error is not None


class TestMakeLoadTools:
    def test_returns_agent_tool(self) -> None:
        tool = _make_load_tools(load_callback=_noop_callback)
        assert isinstance(tool, AgentTool)
        assert tool.name == TOOL_NAME

    def test_schema_has_group_id_and_tool_names(self) -> None:
        tool = _make_load_tools(load_callback=_noop_callback)
        defn = tool.to_definition()
        props = defn.parameters.get("properties", {})
        assert "group_id" in props
        assert "tool_names" in props

    async def test_execute_calls_callback(self) -> None:
        calls: list[tuple[str, list[str] | None]] = []

        async def _callback(
            group_id: str, tool_names: list[str] | None
        ) -> LoadToolsOutput:
            calls.append((group_id, tool_names))
            return LoadToolsOutput(
                group_id=group_id,
                expanded=True,
                tool_names=["t1"],
                remaining=0,
            )

        tool = _make_load_tools(load_callback=_callback)
        result = await tool.execute("call-1", LoadToolsInput(group_id="mcp:github"))
        assert len(calls) == 1
        assert calls[0] == ("mcp:github", None)
        assert result.is_error is None or result.is_error is False

    async def test_execute_with_tool_names(self) -> None:
        calls: list[tuple[str, list[str] | None]] = []

        async def _callback(
            group_id: str, tool_names: list[str] | None
        ) -> LoadToolsOutput:
            calls.append((group_id, tool_names))
            return LoadToolsOutput(
                group_id=group_id,
                expanded=True,
                tool_names=tool_names or [],
                remaining=0,
            )

        tool = _make_load_tools(load_callback=_callback)
        await tool.execute(
            "call-2",
            LoadToolsInput(group_id="mcp:github", tool_names=["create_issue"]),
        )
        assert calls[0] == ("mcp:github", ["create_issue"])

    async def test_execute_error_sets_is_error(self) -> None:
        async def _err_callback(
            group_id: str, tool_names: list[str] | None
        ) -> LoadToolsOutput:
            return LoadToolsOutput(
                group_id=group_id,
                expanded=False,
                tool_names=[],
                remaining=0,
                error="Unknown group_id: bad",
            )

        tool = _make_load_tools(load_callback=_err_callback)
        result = await tool.execute("call-3", LoadToolsInput(group_id="bad"))
        assert result.is_error is True


async def _noop_callback(
    group_id: str, tool_names: list[str] | None
) -> LoadToolsOutput:
    return LoadToolsOutput(group_id=group_id, expanded=True, tool_names=[], remaining=0)


def test_load_tools_output_carries_schemas() -> None:
    out = LoadToolsOutput(
        group_id="g",
        expanded=True,
        tool_names=["t"],
        remaining=0,
        schemas=[{"name": "t", "description": "d", "parameters": {}}],
    )
    dumped = out.model_dump()
    assert dumped["schemas"][0]["name"] == "t"
