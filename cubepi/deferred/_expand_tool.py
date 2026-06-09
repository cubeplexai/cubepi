from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import TextContent
from cubepi.types import StructuredValue

TOOL_NAME = "load_tools"


class LoadToolsInput(BaseModel):
    group_id: str = Field(
        description="The group_id from your 'Deferred tool groups' catalog.",
    )
    tool_names: list[str] | None = Field(
        default=None,
        description="Specific tools to load. Omit to load all tools in the group.",
    )


class LoadToolsOutput(BaseModel):
    group_id: str
    expanded: bool
    tool_names: list[str]
    remaining: int
    error: str | None = None


LoadCallback = Callable[
    [str, list[str] | None],
    Awaitable[LoadToolsOutput],
]


def _make_load_tools(
    *,
    load_callback: LoadCallback,
) -> AgentTool[LoadToolsInput]:
    async def _execute(
        tool_call_id: str,
        args: LoadToolsInput,
        *,
        signal: asyncio.Event | None = None,
        on_update: Callable[[StructuredValue], None] | None = None,
    ) -> AgentToolResult:
        del signal, on_update
        output = await load_callback(args.group_id, args.tool_names)
        text = json.dumps(output.model_dump(), ensure_ascii=False)
        return AgentToolResult(
            content=[TextContent(text=text)],
            is_error=not output.expanded,
        )

    return AgentTool(
        name=TOOL_NAME,
        description=(
            "Load a deferred tool group to make its tools available. "
            "Call with a group_id from the 'Deferred tool groups' catalog. "
            "Optionally pass tool_names to load specific tools only."
        ),
        parameters=LoadToolsInput,
        execute=_execute,
    )
