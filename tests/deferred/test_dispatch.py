from __future__ import annotations

from pydantic import BaseModel

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import TextContent


class _Empty(BaseModel):
    pass


def _dummy_tool(name: str, *, expose: bool = True) -> AgentTool:
    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    return AgentTool(
        name=name,
        description="dummy",
        parameters=_Empty,
        execute=_exec,
        expose_to_model=expose,
    )


class TestExposeToModel:
    def test_default_is_true(self) -> None:
        async def _exec(tool_call_id, args, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = AgentTool(name="t", description="d", parameters=_Empty, execute=_exec)
        assert tool.expose_to_model is True

    def test_hidden_tool_excluded_from_payload_filter(self) -> None:
        tools = [_dummy_tool("visible"), _dummy_tool("hidden", expose=False)]
        visible = [t.to_definition() for t in tools if t.expose_to_model]
        assert [d.name for d in visible] == ["visible"]
