from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from cubepi.agent.types import AgentTool

# "dispatch" (default): static tools array + system prompt; schemas delivered
# via load_tools results; calls routed through the deferred_tool_call
# dispatcher. Zero prompt-cache invalidation.
# "inject": v1 behavior — loaded tools join the model-visible tools array
# (native tool calling, one full cache re-read per expansion).
DeferredStrategy = Literal["dispatch", "inject"]


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
