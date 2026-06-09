from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from cubepi.agent.types import AgentTool


@dataclass
class DeferredToolGroup:
    """A group of tools that starts collapsed and expands on demand.

    ``loader`` is called exactly once per group per agent run — the middleware
    caches the result and filters by ``tool_names`` on selective expansions.
    """

    group_id: str
    display_name: str
    description: str
    tool_names: list[str]
    loader: Callable[[], Awaitable[list[AgentTool]]]
