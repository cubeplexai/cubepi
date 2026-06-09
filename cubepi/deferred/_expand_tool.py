from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import TextContent
from cubepi.types import StructuredValue


class ExpandToolsInput(BaseModel):
    group_id: str = Field(
        description="The group_id from your 'Deferred tool groups' catalog.",
    )
    tool_names: list[str] | None = Field(
        default=None,
        description="Specific tools to expand. Omit to expand all tools in the group.",
    )


class ExpandToolsOutput(BaseModel):
    group_id: str
    expanded: bool
    tool_names: list[str]
    remaining: int
    error: str | None = None


ExpandCallback = Callable[
    [str, list[str] | None],
    Awaitable[ExpandToolsOutput],
]


def _make_expand_tools(
    *,
    expand_callback: ExpandCallback,
) -> AgentTool[ExpandToolsInput]:
    async def _execute(
        tool_call_id: str,
        args: ExpandToolsInput,
        *,
        signal: asyncio.Event | None = None,
        on_update: Callable[[StructuredValue], None] | None = None,
    ) -> AgentToolResult:
        del signal, on_update
        output = await expand_callback(args.group_id, args.tool_names)
        text = json.dumps(output.model_dump(), ensure_ascii=False)
        return AgentToolResult(
            content=[TextContent(text=text)],
            is_error=bool(output.error),
        )

    return AgentTool(
        name="expand_tools",
        description=(
            "Expand a deferred tool group to make its tools available. "
            "Call with a group_id from the 'Deferred tool groups' catalog. "
            "Optionally pass tool_names to expand specific tools only."
        ),
        parameters=ExpandToolsInput,
        execute=_execute,
    )
