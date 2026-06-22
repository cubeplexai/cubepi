from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import TextContent
from cubepi.types import StructuredValue

DISPATCH_TOOL_NAME = "deferred_tool_call"


class DeferredToolCallInput(BaseModel):
    tool_name: str = Field(
        description=(
            "Name of a deferred tool, from the 'Deferred tool groups' catalog "
            "or a load_tools result."
        ),
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool, matching its schema.",
    )


def _make_deferred_tool_call(
    *,
    known_tool_names: Callable[[], list[str]],
) -> AgentTool[DeferredToolCallInput]:
    """Build the dispatcher builtin.

    Its ``execute`` only runs when the middleware resolver declined to rewrite
    the call (unknown tool name) — it is the structured-error fallback. Known
    names are rewritten by ``resolve_tool_call`` before the pipeline and never
    reach this body.
    """

    async def _execute(
        tool_call_id: str,
        args: DeferredToolCallInput,
        *,
        signal: asyncio.Event | None = None,
        on_update: Callable[[StructuredValue], None] | None = None,
    ) -> AgentToolResult:
        del signal, on_update
        names = known_tool_names()
        return AgentToolResult(
            content=[
                TextContent(
                    text=(
                        f"Unknown deferred tool: {args.tool_name!r}. "
                        f"Valid names: {', '.join(sorted(names))}. "
                        "Call load_tools(group_id=...) to list a group's "
                        "full schemas."
                    )
                )
            ],
            is_error=True,
        )

    return AgentTool(
        name=DISPATCH_TOOL_NAME,
        description=(
            "Invoke a deferred tool by name. Place `tool_name` before "
            "`arguments` in the call's JSON so streaming clients can resolve "
            "the target tool before its args finish arriving. Use schemas "
            "from load_tools results to shape `arguments`. Tools load on "
            "demand if needed."
        ),
        parameters=DeferredToolCallInput,
        execute=_execute,
    )
