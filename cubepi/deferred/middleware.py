"""DeferredToolsMiddleware — progressive tool disclosure for large tool sets.

Two strategies:

* ``dispatch`` (default) — the tools array and system prompt are byte-stable
  for the whole run (zero prompt-cache invalidation). ``load_tools`` returns
  full schemas in its tool result; calls route through the
  ``deferred_tool_call`` dispatcher, which the engine's ``resolve_tool_call``
  hook rewrites to the real tool before validation/hooks/tracing. Loaded
  tools live in ``context.tools`` with ``expose_to_model=False``.
* ``inject`` — v1 behavior: loaded tools join the model-visible tools array
  (native tool calling; each expansion re-reads the prompt cache).

Two-phase expansion (both strategies):
1. ``_expand_callback`` runs inside ``load_tools`` execute (no AgentContext).
   Validates, invokes the loader (cached), updates state, queues pending tools.
2. ``after_tool_call`` fires after execute and injects pending tools into
   ``ctx.context.tools`` so the next model loop iteration sees them.

``_expand`` combines both phases for testing without the full agent loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from cubepi.agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentTool,
)
from cubepi.deferred._catalog import (
    DEFAULT_CATALOG_HEADER,
    DEFAULT_DISPATCH_CATALOG_HEADER,
    render_catalog,
    render_static_catalog,
)
from cubepi.deferred._dispatch_tool import (
    DISPATCH_TOOL_NAME,
    _make_deferred_tool_call,
)
from cubepi.deferred._expand_tool import (
    TOOL_NAME,
    LoadToolsOutput,
    _make_load_tools,
)
from cubepi.deferred.types import DeferredStrategy, DeferredToolGroup
from cubepi.middleware.base import Middleware
from cubepi.providers.base import ToolCall


@dataclass
class ResumedState:
    """Pre-loaded tools and remaining groups for cross-run replay."""

    pre_loaded_tools: list[AgentTool]
    remaining_groups: list[DeferredToolGroup]
    loader_cache: dict[str, list[AgentTool]]


class DeferredToolsMiddleware(Middleware):
    """Middleware that manages deferred tool groups via progressive disclosure.

    Attributes:
        tools: ``[load_tools]`` (+ ``[deferred_tool_call]`` in dispatch mode)
            — merged into agent state at construction.
    """

    def __init__(
        self,
        *,
        groups: list[DeferredToolGroup],
        extra_ref: Callable[[], dict[str, Any]],
        strategy: DeferredStrategy = "dispatch",
        catalog_header: str | None = None,
        resumed_loader_cache: dict[str, list[AgentTool]] | None = None,
        on_tools_expanded: Callable[[list[AgentTool]], None] | None = None,
    ) -> None:
        self._groups: dict[str, DeferredToolGroup] = {g.group_id: g for g in groups}
        self._extra_ref = extra_ref
        self._strategy: DeferredStrategy = strategy
        self._catalog_header = catalog_header or (
            DEFAULT_DISPATCH_CATALOG_HEADER
            if strategy == "dispatch"
            else DEFAULT_CATALOG_HEADER
        )
        self._on_tools_expanded = on_tools_expanded
        self._tool_to_group: dict[str, str] = {
            name: g.group_id for g in groups for name in g.tool_names
        }

        # Loader called exactly once per group per run.
        self._loader_cache: dict[str, list[AgentTool]] = (
            dict(resumed_loader_cache) if resumed_loader_cache else {}
        )
        # Staging area: expand_callback queues, after_tool_call drains.
        self._pending_injection: list[AgentTool] = []
        # Per-group lock to prevent concurrent loader invocations.
        self._loader_locks: dict[str, asyncio.Lock] = {}

        self.tools: list[AgentTool] = [
            _make_load_tools(load_callback=self._expand_callback)
        ]
        if strategy == "dispatch":
            self.tools.append(
                _make_deferred_tool_call(
                    known_tool_names=lambda: list(self._tool_to_group)
                )
            )

        # Dispatch catalog is expansion-independent — render once, serve
        # byte-identical every turn.
        self._static_catalog: str = (
            render_static_catalog(
                groups=list(self._groups.values()),
                header=self._catalog_header,
            )
            if strategy == "dispatch"
            else ""
        )

    # ------------------------------------------------------------------
    # Phase 1: expand callback (called from load_tools execute)
    # ------------------------------------------------------------------

    async def _expand_callback(
        self,
        group_id: str,
        tool_names: list[str] | None,
    ) -> LoadToolsOutput:
        extra = self._extra_ref()
        group = self._groups.get(group_id)
        if group is None:
            return LoadToolsOutput(
                group_id=group_id,
                expanded=False,
                tool_names=[],
                remaining=0,
                error=(
                    f"Unknown group_id: {group_id}. "
                    f"Available: {', '.join(sorted(self._groups))}"
                ),
            )

        # Load once per group per run (lock prevents duplicate under parallel).
        lock = self._loader_locks.setdefault(group_id, asyncio.Lock())
        async with lock:
            try:
                if group_id not in self._loader_cache:
                    self._loader_cache[group_id] = await group.loader()
            except Exception as exc:
                return LoadToolsOutput(
                    group_id=group_id,
                    expanded=False,
                    tool_names=[],
                    remaining=len(group.tool_names),
                    error=f"Loader failed: {exc}",
                )

        all_loaded = self._loader_cache[group_id]

        # Use the shared dict in extra — never replace it, only mutate in place.
        if "expanded_groups" not in extra:
            extra["expanded_groups"] = {}
        expanded_groups: dict[str, list[str] | None] = extra["expanded_groups"]
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

        # Stage newly expanded tools for injection by after_tool_call.
        # Dispatch mode hides them from the provider payload — the engine
        # still resolves them by name.
        staged = newly_expanded
        if self._strategy == "dispatch":
            staged = [replace(t, expose_to_model=False) for t in newly_expanded]
        self._pending_injection.extend(staged)
        # Persist to the agent's canonical tool list for cross-prompt() use.
        if staged and self._on_tools_expanded:
            self._on_tools_expanded(staged)

        # Calculate remaining count.
        current_expanded = expanded_groups.get(group_id)
        if current_expanded is None and group_id in expanded_groups:
            remaining = 0
        else:
            assert isinstance(current_expanded, list)
            remaining = len(group.tool_names) - len(current_expanded)

        schemas: list[dict[str, object]] | None = None
        if self._strategy == "dispatch":
            # Full set for the request — idempotent across repeat calls
            # (compaction self-rescue: re-calling re-serves the schemas).
            schemas = [t.to_definition().model_dump() for t in requested]

        return LoadToolsOutput(
            group_id=group_id,
            expanded=True,
            tool_names=expanded_names,
            remaining=max(remaining, 0),
            schemas=schemas,
        )

    # ------------------------------------------------------------------
    # Drain pending injection into a tools list (shared by _expand and
    # after_tool_call so the dedup logic lives in one place).
    # ------------------------------------------------------------------

    def _drain_pending(self, tools: list[AgentTool]) -> None:
        if not self._pending_injection:
            return
        pending = list(self._pending_injection)
        self._pending_injection.clear()
        existing_names = {t.name for t in tools}
        for tool in pending:
            if tool.name not in existing_names:
                tools.append(tool)
                existing_names.add(tool.name)

    # ------------------------------------------------------------------
    # Combined expand + inject (for testing without the full agent loop)
    # ------------------------------------------------------------------

    async def _expand(
        self,
        *,
        group_id: str,
        tool_names: list[str] | None,
        context: AgentContext,
    ) -> LoadToolsOutput:
        output = await self._expand_callback(group_id, tool_names)
        if output.expanded and context.tools is not None:
            self._drain_pending(context.tools)
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
        if ctx.tool_call.name != TOOL_NAME:
            return None
        if ctx.is_error:
            return None
        if ctx.context.tools is not None:
            self._drain_pending(ctx.context.tools)
        return None

    # ------------------------------------------------------------------
    # Dispatch: resolve deferred_tool_call to the real tool
    # ------------------------------------------------------------------

    async def resolve_tool_call(
        self,
        tool_call: ToolCall,
        *,
        context: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> ToolCall | None:
        del signal
        if self._strategy != "dispatch" or tool_call.name != DISPATCH_TOOL_NAME:
            return None
        args = tool_call.arguments
        name = args.get("tool_name") if isinstance(args, dict) else None
        if not isinstance(name, str) or name not in self._tool_to_group:
            return None  # falls through to the dispatcher's error fallback
        already_loaded = context.tools is not None and any(
            t.name == name for t in context.tools
        )
        if not already_loaded:
            await self._ensure_loaded(self._tool_to_group[name], [name], context)
        inner = args.get("arguments")
        return ToolCall(
            id=tool_call.id,
            name=name,
            arguments=inner if isinstance(inner, dict) else {},
        )

    async def _ensure_loaded(
        self,
        group_id: str,
        tool_names: list[str],
        context: AgentContext,
    ) -> None:
        """Implicit load for dispatched calls: reuse the load path, then drain
        staged tools into the live context immediately (no after_tool_call
        fires on the dispatcher's behalf for a rewritten call)."""
        await self._expand_callback(group_id, tool_names)
        if context.tools is not None:
            self._drain_pending(context.tools)

    # ------------------------------------------------------------------
    # System prompt: catalog (static in dispatch mode)
    # ------------------------------------------------------------------

    async def transform_system_prompt(
        self,
        system_prompt: str,
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> str:
        del signal
        if self._strategy == "dispatch":
            catalog = self._static_catalog
            return f"{system_prompt}\n\n{catalog}" if catalog else system_prompt

        extra = self._extra_ref()
        expanded: dict[str, list[str] | None] = extra.get(
            "expanded_groups",
            {},
        )
        catalog = render_catalog(
            groups=list(self._groups.values()),
            expanded=expanded,
            header=self._catalog_header,
        )
        return f"{system_prompt}\n\n{catalog}" if catalog else system_prompt

    # ------------------------------------------------------------------
    # Cross-run replay
    # ------------------------------------------------------------------

    @staticmethod
    async def prepare_resumed_state(
        groups: list[DeferredToolGroup],
        expanded: dict[str, list[str] | None],
        *,
        strategy: DeferredStrategy,
    ) -> ResumedState:
        """Replay expansion state from a previous run.

        ``strategy`` is required and must match the middleware's strategy —
        a default would let an inject-mode host silently resume with hidden
        tools (invisible to the model, with no dispatcher to reach them).

        Dispatch mode: schemas live in message history (the checkpointer
        brings them back) — only the loader cache and hidden tool objects
        need rebuilding, and every loaded group stays in ``remaining`` so
        ``load_tools`` can re-serve schemas after compaction. Inject mode:
        tools come back model-visible; fully expanded groups leave the
        catalog as in v1.
        """
        hidden = strategy == "dispatch"
        remaining: list[DeferredToolGroup] = []
        to_load: list[tuple[DeferredToolGroup, list[str] | None]] = []
        for group in groups:
            exp = expanded.get(group.group_id)
            if exp is None and group.group_id not in expanded:
                remaining.append(group)
            else:
                to_load.append((group, exp))

        loaded_results: list[list[AgentTool]] = await asyncio.gather(
            *(g.loader() for g, _ in to_load)
        )

        pre_loaded: list[AgentTool] = []
        cache: dict[str, list[AgentTool]] = {}
        for (group, exp), loaded in zip(to_load, loaded_results):
            cache[group.group_id] = loaded
            if exp is None:
                selected = loaded
                if hidden:
                    remaining.append(group)
            else:
                name_set = set(exp)
                selected = [t for t in loaded if t.name in name_set]
                remaining.append(group)
            pre_loaded.extend(
                replace(t, expose_to_model=False) if hidden else t for t in selected
            )

        return ResumedState(
            pre_loaded_tools=pre_loaded,
            remaining_groups=remaining,
            loader_cache=cache,
        )
