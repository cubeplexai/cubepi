"""DeferredToolsMiddleware — progressive tool disclosure for large tool sets.

Two-phase expansion:
1. ``_expand_callback`` runs inside ``expand_tools`` execute (no AgentContext).
   Validates, invokes the loader (cached), updates state, queues pending tools.
2. ``after_tool_call`` fires after execute and injects pending tools into
   ``ctx.context.tools`` so the next model loop iteration sees them.

``_expand`` combines both phases for testing without the full agent loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cubepi.agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentTool,
)
from cubepi.deferred._catalog import (
    DEFAULT_CATALOG_HEADER,
    ToolSchema,
    render_catalog,
    render_expanded_schemas,
)
from cubepi.deferred._expand_tool import (
    ExpandToolsOutput,
    _make_expand_tools,
)
from cubepi.deferred.types import DeferredToolGroup
from cubepi.middleware.base import Middleware


@dataclass
class ResumedState:
    """Pre-loaded tools and remaining groups for cross-run replay."""

    pre_loaded_tools: list[AgentTool]
    remaining_groups: list[DeferredToolGroup]


class DeferredToolsMiddleware(Middleware):
    """Middleware that manages deferred tool groups via progressive disclosure.

    Attributes:
        tools: ``[expand_tools]`` — merged into agent state at construction.
    """

    def __init__(
        self,
        *,
        groups: list[DeferredToolGroup],
        extra_ref: Callable[[], dict[str, Any]],
        catalog_header: str = DEFAULT_CATALOG_HEADER,
    ) -> None:
        self._groups: dict[str, DeferredToolGroup] = {
            g.group_id: g for g in groups
        }
        self._extra_ref = extra_ref
        self._catalog_header = catalog_header

        # Loader called exactly once per group per run.
        self._loader_cache: dict[str, list[AgentTool]] = {}
        # Append-only, expansion order for cache stability.
        self._expanded_schemas: list[tuple[str, list[ToolSchema]]] = []
        # Staging area: expand_callback queues, after_tool_call drains.
        self._pending_injection: list[AgentTool] = []

        self.tools: list[AgentTool] = [
            _make_expand_tools(expand_callback=self._expand_callback)
        ]

    # ------------------------------------------------------------------
    # Phase 1: expand callback (called from expand_tools execute)
    # ------------------------------------------------------------------

    async def _expand_callback(
        self,
        group_id: str,
        tool_names: list[str] | None,
    ) -> ExpandToolsOutput:
        extra = self._extra_ref()
        group = self._groups.get(group_id)
        if group is None:
            return ExpandToolsOutput(
                group_id=group_id,
                expanded=False,
                tool_names=[],
                remaining=0,
                error=(
                    f"Unknown group_id: {group_id}. "
                    f"Available: {', '.join(sorted(self._groups))}"
                ),
            )

        # Load once per group per run.
        try:
            if group_id not in self._loader_cache:
                self._loader_cache[group_id] = await group.loader()
        except Exception as exc:
            return ExpandToolsOutput(
                group_id=group_id,
                expanded=False,
                tool_names=[],
                remaining=len(group.tool_names),
                error=f"Loader failed: {exc}",
            )

        all_loaded = self._loader_cache[group_id]

        # Determine what is already expanded.
        expanded_groups: dict[str, list[str] | None] = extra.get(
            "expanded_groups", {},
        )
        already_val = expanded_groups.get(group_id)
        if group_id in expanded_groups and already_val is None:
            # Fully expanded (None sentinel) — everything already injected.
            already_set: set[str] = {t.name for t in all_loaded}
        elif isinstance(already_val, list):
            already_set = set(already_val)
        else:
            already_set = set()

        # Filter to requested tools.
        if tool_names is not None:
            requested_set = set(tool_names)
            requested = [t for t in all_loaded if t.name in requested_set]
        else:
            requested = list(all_loaded)

        newly_expanded = [t for t in requested if t.name not in already_set]
        expanded_names = [t.name for t in requested]

        # Update expansion state.
        if tool_names is None:
            expanded_groups[group_id] = None
        else:
            prev = expanded_groups.get(group_id)
            if prev is None and group_id in expanded_groups:
                pass  # already fully expanded, no-op
            else:
                merged = list(prev) if isinstance(prev, list) else []
                merged_set = set(merged)
                for name in expanded_names:
                    if name not in merged_set:
                        merged.append(name)
                        merged_set.add(name)
                expanded_groups[group_id] = merged
        extra["expanded_groups"] = expanded_groups

        # Record schemas for expanded tools (append-only).
        if newly_expanded:
            new_schemas = [
                t.to_definition().model_dump() for t in newly_expanded
            ]
            existing_idx = next(
                (
                    i
                    for i, (gid, _) in enumerate(self._expanded_schemas)
                    if gid == group_id
                ),
                None,
            )
            if existing_idx is not None:
                prev_schemas = self._expanded_schemas[existing_idx][1]
                self._expanded_schemas[existing_idx] = (
                    group_id,
                    prev_schemas + new_schemas,
                )
            else:
                self._expanded_schemas.append((group_id, new_schemas))

        # Stage newly expanded tools for injection by after_tool_call.
        self._pending_injection.extend(newly_expanded)

        # Calculate remaining count.
        current_expanded = expanded_groups.get(group_id)
        if current_expanded is None and group_id in expanded_groups:
            remaining = 0
        elif isinstance(current_expanded, list):
            remaining = len(group.tool_names) - len(current_expanded)
        else:
            remaining = len(group.tool_names)

        return ExpandToolsOutput(
            group_id=group_id,
            expanded=True,
            tool_names=expanded_names,
            remaining=max(remaining, 0),
        )

    # ------------------------------------------------------------------
    # Combined expand + inject (for testing without the full agent loop)
    # ------------------------------------------------------------------

    async def _expand(
        self,
        *,
        group_id: str,
        tool_names: list[str] | None,
        context: AgentContext,
    ) -> ExpandToolsOutput:
        output = await self._expand_callback(group_id, tool_names)
        if output.expanded and self._pending_injection:
            newly_expanded = list(self._pending_injection)
            self._pending_injection.clear()
            if context.tools is not None:
                existing_names = {t.name for t in context.tools}
                for tool in newly_expanded:
                    if tool.name not in existing_names:
                        context.tools.append(tool)
        return output

    # ------------------------------------------------------------------
    # Phase 2: after_tool_call hook (injects tools into live context)
    # ------------------------------------------------------------------

    async def after_tool_call(
        self,
        ctx: AfterToolCallContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> AfterToolCallResult | None:
        del signal
        if ctx.tool_call.name != "expand_tools":
            return None
        if ctx.is_error:
            return None

        if self._pending_injection:
            newly_expanded = list(self._pending_injection)
            self._pending_injection.clear()
            if ctx.context.tools is not None:
                existing_names = {t.name for t in ctx.context.tools}
                for tool in newly_expanded:
                    if tool.name not in existing_names:
                        ctx.context.tools.append(tool)

        return None

    # ------------------------------------------------------------------
    # System prompt: catalog + expanded schemas
    # ------------------------------------------------------------------

    async def transform_system_prompt(
        self,
        system_prompt: str,
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> str:
        del signal
        extra = self._extra_ref()
        expanded: dict[str, list[str] | None] = extra.get(
            "expanded_groups", {},
        )

        catalog = render_catalog(
            groups=list(self._groups.values()),
            expanded=expanded,
            header=self._catalog_header,
        )

        schemas = render_expanded_schemas(
            expanded_schemas=self._expanded_schemas,
        )

        parts = [system_prompt]
        if catalog:
            parts.append(catalog)
        if schemas:
            parts.append(schemas)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Cross-run replay
    # ------------------------------------------------------------------

    @staticmethod
    async def prepare_resumed_state(
        groups: list[DeferredToolGroup],
        expanded: dict[str, list[str] | None],
    ) -> ResumedState:
        """Replay expansion state from a previous run.

        Fully expanded groups (``None``) get all tools pre-loaded and are
        removed from the remaining set.  Partially expanded groups get the
        selected tools pre-loaded but stay in remaining (still deferrable).
        Unexpanded groups pass through untouched.
        """
        pre_loaded: list[AgentTool] = []
        remaining: list[DeferredToolGroup] = []

        for group in groups:
            exp = expanded.get(group.group_id)
            if exp is None and group.group_id not in expanded:
                # Never expanded — stays deferred.
                remaining.append(group)
                continue

            loaded = await group.loader()
            if exp is None:
                # Fully expanded (None sentinel) — load all, drop from remaining.
                pre_loaded.extend(loaded)
            else:
                # Partially expanded — load selected, keep in remaining.
                name_set = set(exp)
                selected = [t for t in loaded if t.name in name_set]
                pre_loaded.extend(selected)
                remaining.append(group)

        return ResumedState(
            pre_loaded_tools=pre_loaded,
            remaining_groups=remaining,
        )
