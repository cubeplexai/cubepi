# DeferredToolGroup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `DeferredToolGroup` primitive to cubepi that lets host applications register groups of tools that start collapsed (catalog only) and expand on demand mid-run, reducing context bloat and improving tool selection accuracy.

**Architecture:** A new `cubepi/deferred/` package provides `DeferredToolGroup` (dataclass) and `DeferredToolsMiddleware` (middleware). The middleware injects an `expand_tools` builtin, renders a catalog in the system prompt, and handles mid-run tool injection via the `after_tool_call` hook. `Agent.__init__` gains a `deferred_tool_groups` parameter that auto-creates the middleware. Zero changes to `loop.py`.

**Tech Stack:** Python 3.11+, Pydantic v2 (tool input/output schemas), pytest with `asyncio_mode=auto`, mypy strict.

---

## File Structure

- **Create** `cubepi/deferred/__init__.py` — public exports: `DeferredToolGroup`, `DeferredToolsMiddleware`, `ResumedState`.
- **Create** `cubepi/deferred/types.py` — `DeferredToolGroup` dataclass.
- **Create** `cubepi/deferred/_catalog.py` — pure functions: `render_catalog()`, `render_expanded_schemas()`.
- **Create** `cubepi/deferred/_expand_tool.py` — `expand_tools` builtin factory (`_make_expand_tools`), `ExpandToolsInput`, `ExpandToolsOutput`.
- **Create** `cubepi/deferred/middleware.py` — `DeferredToolsMiddleware`, `ResumedState`, `prepare_resumed_state`.
- **Modify** `cubepi/agent/agent.py:141-166` — add `deferred_tool_groups` parameter, auto-create middleware.
- **Create** `tests/deferred/__init__.py` — empty.
- **Create** `tests/deferred/test_catalog.py` — catalog rendering tests.
- **Create** `tests/deferred/test_expand_tool.py` — expand_tools builtin tests.
- **Create** `tests/deferred/test_middleware.py` — middleware integration tests.
- **Create** `tests/deferred/test_agent_wiring.py` — Agent-level `deferred_tool_groups` wiring tests.

---

## Task 1: `DeferredToolGroup` dataclass

**Files:**
- Create: `cubepi/deferred/__init__.py`
- Create: `cubepi/deferred/types.py`
- Create: `tests/deferred/__init__.py`
- Test: `tests/deferred/test_catalog.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/deferred/test_catalog.py
from __future__ import annotations

import pytest

from cubepi.deferred.types import DeferredToolGroup


class TestDeferredToolGroup:
    def test_basic_construction(self) -> None:
        async def _loader():
            return []

        group = DeferredToolGroup(
            group_id="mcp:github",
            display_name="GitHub",
            description="Code hosting: issues, PRs, repos",
            tool_names=["create_issue", "search_repos"],
            loader=_loader,
        )
        assert group.group_id == "mcp:github"
        assert group.display_name == "GitHub"
        assert group.description == "Code hosting: issues, PRs, repos"
        assert group.tool_names == ["create_issue", "search_repos"]
        assert group.loader is _loader
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/test_catalog.py::TestDeferredToolGroup::test_basic_construction -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cubepi.deferred'`

- [ ] **Step 3: Create the package and dataclass**

```python
# cubepi/deferred/__init__.py
"""Deferred tool groups — progressive tool disclosure primitive."""

from cubepi.deferred.types import DeferredToolGroup

__all__ = ["DeferredToolGroup"]
```

```python
# cubepi/deferred/types.py
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
```

```python
# tests/deferred/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/test_catalog.py::TestDeferredToolGroup::test_basic_construction -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && git add cubepi/deferred/__init__.py cubepi/deferred/types.py tests/deferred/__init__.py tests/deferred/test_catalog.py && git commit -m "feat(deferred): add DeferredToolGroup dataclass (#166)"
```

---

## Task 2: Catalog rendering — pure functions

**Files:**
- Create: `cubepi/deferred/_catalog.py`
- Test: `tests/deferred/test_catalog.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/deferred/test_catalog.py`:

```python
from cubepi.deferred._catalog import render_catalog, render_expanded_schemas


def _make_group(
    group_id: str,
    display_name: str,
    description: str,
    tool_names: list[str],
) -> DeferredToolGroup:
    async def _noop_loader():
        return []

    return DeferredToolGroup(
        group_id=group_id,
        display_name=display_name,
        description=description,
        tool_names=tool_names,
        loader=_noop_loader,
    )


class TestRenderCatalog:
    def test_no_groups_returns_empty(self) -> None:
        result = render_catalog(groups=[], expanded={})
        assert result == ""

    def test_single_group_no_expansion(self) -> None:
        groups = [_make_group("mcp:github", "GitHub", "Code hosting", ["create_issue", "search_repos"])]
        result = render_catalog(groups=groups, expanded={})
        assert "mcp:github" in result
        assert "GitHub" in result
        assert "Code hosting" in result
        assert "2 tools" in result
        assert "create_issue" in result
        assert "search_repos" in result

    def test_sorted_by_group_id(self) -> None:
        groups = [
            _make_group("z:last", "Last", "desc", ["t1"]),
            _make_group("a:first", "First", "desc", ["t2"]),
        ]
        result = render_catalog(groups=groups, expanded={})
        a_pos = result.index("a:first")
        z_pos = result.index("z:last")
        assert a_pos < z_pos

    def test_byte_stable_across_input_orderings(self) -> None:
        g1 = _make_group("mcp:a", "A", "desc", ["t1"])
        g2 = _make_group("mcp:b", "B", "desc", ["t2"])
        result_ab = render_catalog(groups=[g1, g2], expanded={})
        result_ba = render_catalog(groups=[g2, g1], expanded={})
        assert result_ab == result_ba

    def test_fully_expanded_group_omitted(self) -> None:
        groups = [
            _make_group("mcp:github", "GitHub", "Code hosting", ["create_issue", "search_repos"]),
            _make_group("mcp:linear", "Linear", "Issues", ["create_issue"]),
        ]
        expanded: dict[str, list[str] | None] = {"mcp:github": None}
        result = render_catalog(groups=groups, expanded=expanded)
        assert "mcp:github" not in result
        assert "mcp:linear" in result

    def test_partially_expanded_shows_remaining(self) -> None:
        groups = [_make_group("mcp:github", "GitHub", "Code hosting", ["create_issue", "search_repos", "create_pr"])]
        expanded: dict[str, list[str] | None] = {"mcp:github": ["create_issue"]}
        result = render_catalog(groups=groups, expanded=expanded)
        assert "2 remaining tools" in result
        assert "create_issue" not in result
        assert "search_repos" in result
        assert "create_pr" in result

    def test_all_groups_fully_expanded_returns_empty(self) -> None:
        groups = [_make_group("mcp:github", "GitHub", "desc", ["t1"])]
        expanded: dict[str, list[str] | None] = {"mcp:github": None}
        result = render_catalog(groups=groups, expanded=expanded)
        assert result == ""

    def test_custom_header(self) -> None:
        groups = [_make_group("mcp:a", "A", "desc", ["t1"])]
        result = render_catalog(groups=groups, expanded={}, header="Custom header text")
        assert "Custom header text" in result


class TestRenderExpandedSchemas:
    def test_no_expansions_returns_empty(self) -> None:
        result = render_expanded_schemas(expanded_schemas=[])
        assert result == ""

    def test_single_expansion(self) -> None:
        schemas = [("mcp:github", [{"name": "create_issue", "description": "Create an issue", "parameters": {"type": "object", "properties": {}}}])]
        result = render_expanded_schemas(expanded_schemas=schemas)
        assert "mcp:github" in result
        assert "create_issue" in result
        assert "Create an issue" in result

    def test_expansion_order_preserved(self) -> None:
        schemas = [
            ("mcp:linear", [{"name": "t1", "description": "d1", "parameters": {}}]),
            ("mcp:github", [{"name": "t2", "description": "d2", "parameters": {}}]),
        ]
        result = render_expanded_schemas(expanded_schemas=schemas)
        linear_pos = result.index("mcp:linear")
        github_pos = result.index("mcp:github")
        assert linear_pos < github_pos

    def test_append_only_prefix_stable(self) -> None:
        schemas_v1 = [("mcp:linear", [{"name": "t1", "description": "d1", "parameters": {}}])]
        schemas_v2 = [
            ("mcp:linear", [{"name": "t1", "description": "d1", "parameters": {}}]),
            ("mcp:github", [{"name": "t2", "description": "d2", "parameters": {}}]),
        ]
        result_v1 = render_expanded_schemas(expanded_schemas=schemas_v1)
        result_v2 = render_expanded_schemas(expanded_schemas=schemas_v2)
        assert result_v2.startswith(result_v1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/test_catalog.py -v -k "not test_basic_construction"`
Expected: FAIL with `ModuleNotFoundError: No module named 'cubepi.deferred._catalog'`

- [ ] **Step 3: Implement catalog rendering**

```python
# cubepi/deferred/_catalog.py
from __future__ import annotations

import json

from cubepi.deferred.types import DeferredToolGroup

DEFAULT_CATALOG_HEADER = (
    "# Deferred tool groups\n"
    "\n"
    "These tool groups are available but not yet loaded. Call `expand_tools(group_id)`\n"
    "to load a group's tools for the rest of this conversation.\n"
    "You can also call `expand_tools(group_id, tool_names=[...])` to load specific tools only."
)


ToolSchema = dict[str, object]


def render_catalog(
    *,
    groups: list[DeferredToolGroup],
    expanded: dict[str, list[str] | None],
    header: str = DEFAULT_CATALOG_HEADER,
) -> str:
    lines: list[str] = []

    for group in sorted(groups, key=lambda g: g.group_id):
        expanded_names = expanded.get(group.group_id)

        if expanded_names is None and group.group_id in expanded:
            continue

        if expanded_names is not None:
            remaining = [n for n in group.tool_names if n not in set(expanded_names)]
        else:
            remaining = list(group.tool_names)

        if not remaining:
            continue

        count = len(remaining)
        count_label = f"{count} remaining tools" if group.group_id in expanded else f"{count} tools"
        lines.append(f"- `{group.group_id}` — {group.display_name}: {group.description} ({count_label})")
        lines.append(f"  {', '.join(remaining)}")

    if not lines:
        return ""

    return header + "\n\n" + "\n".join(lines)


def render_expanded_schemas(
    *,
    expanded_schemas: list[tuple[str, list[ToolSchema]]],
) -> str:
    if not expanded_schemas:
        return ""

    sections: list[str] = []
    for group_id, tool_defs in expanded_schemas:
        tool_lines: list[str] = []
        for td in tool_defs:
            name = td.get("name", "")
            desc = td.get("description", "")
            params = td.get("parameters", {})
            params_json = json.dumps(params, sort_keys=True, ensure_ascii=False)
            tool_lines.append(f"- **{name}**: {desc}")
            tool_lines.append(f"  Parameters: {params_json}")
        sections.append(f"## {group_id}\n\n" + "\n".join(tool_lines))

    return "# Expanded tool groups\n\n" + "\n\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/test_catalog.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && git add cubepi/deferred/_catalog.py tests/deferred/test_catalog.py && git commit -m "feat(deferred): catalog + expanded schema rendering (#166)"
```

---

## Task 3: `expand_tools` builtin

**Files:**
- Create: `cubepi/deferred/_expand_tool.py`
- Test: `tests/deferred/test_expand_tool.py`

The `expand_tools` builtin is an `AgentTool` whose `execute` delegates expansion logic to the middleware (via a callback). The tool itself handles input validation and returns structured output. The actual loader invocation and tool injection happen in `after_tool_call` (Task 4).

- [ ] **Step 1: Write the failing tests**

```python
# tests/deferred/test_expand_tool.py
from __future__ import annotations

import pytest

from cubepi.deferred._expand_tool import (
    ExpandToolsInput,
    ExpandToolsOutput,
    _make_expand_tools,
)
from cubepi.agent.types import AgentTool


class TestExpandToolsInput:
    def test_group_id_only(self) -> None:
        inp = ExpandToolsInput(group_id="mcp:github")
        assert inp.group_id == "mcp:github"
        assert inp.tool_names is None

    def test_group_id_with_tool_names(self) -> None:
        inp = ExpandToolsInput(group_id="mcp:github", tool_names=["create_issue"])
        assert inp.tool_names == ["create_issue"]


class TestExpandToolsOutput:
    def test_success_output(self) -> None:
        out = ExpandToolsOutput(
            group_id="mcp:github",
            expanded=True,
            tool_names=["create_issue"],
            remaining=5,
        )
        assert out.expanded is True
        assert out.error is None

    def test_error_output(self) -> None:
        out = ExpandToolsOutput(
            group_id="bad:id",
            expanded=False,
            tool_names=[],
            remaining=0,
            error="Unknown group_id: bad:id",
        )
        assert out.expanded is False
        assert out.error is not None


class TestMakeExpandTools:
    def test_returns_agent_tool(self) -> None:
        tool = _make_expand_tools(expand_callback=_noop_callback)
        assert isinstance(tool, AgentTool)
        assert tool.name == "expand_tools"

    def test_schema_has_group_id_and_tool_names(self) -> None:
        tool = _make_expand_tools(expand_callback=_noop_callback)
        defn = tool.to_definition()
        props = defn.parameters.get("properties", {})
        assert "group_id" in props
        assert "tool_names" in props

    async def test_execute_calls_callback(self) -> None:
        calls: list[tuple[str, list[str] | None]] = []

        async def _callback(group_id: str, tool_names: list[str] | None) -> ExpandToolsOutput:
            calls.append((group_id, tool_names))
            return ExpandToolsOutput(
                group_id=group_id,
                expanded=True,
                tool_names=["t1"],
                remaining=0,
            )

        tool = _make_expand_tools(expand_callback=_callback)
        result = await tool.execute("call-1", ExpandToolsInput(group_id="mcp:github"))
        assert len(calls) == 1
        assert calls[0] == ("mcp:github", None)
        assert result.is_error is None or result.is_error is False

    async def test_execute_with_tool_names(self) -> None:
        calls: list[tuple[str, list[str] | None]] = []

        async def _callback(group_id: str, tool_names: list[str] | None) -> ExpandToolsOutput:
            calls.append((group_id, tool_names))
            return ExpandToolsOutput(
                group_id=group_id,
                expanded=True,
                tool_names=tool_names or [],
                remaining=0,
            )

        tool = _make_expand_tools(expand_callback=_callback)
        result = await tool.execute(
            "call-2",
            ExpandToolsInput(group_id="mcp:github", tool_names=["create_issue"]),
        )
        assert calls[0] == ("mcp:github", ["create_issue"])

    async def test_execute_error_sets_is_error(self) -> None:
        async def _err_callback(group_id: str, tool_names: list[str] | None) -> ExpandToolsOutput:
            return ExpandToolsOutput(
                group_id=group_id,
                expanded=False,
                tool_names=[],
                remaining=0,
                error="Unknown group_id: bad",
            )

        tool = _make_expand_tools(expand_callback=_err_callback)
        result = await tool.execute("call-3", ExpandToolsInput(group_id="bad"))
        assert result.is_error is True


async def _noop_callback(group_id: str, tool_names: list[str] | None) -> ExpandToolsOutput:
    return ExpandToolsOutput(group_id=group_id, expanded=True, tool_names=[], remaining=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/test_expand_tool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cubepi.deferred._expand_tool'`

- [ ] **Step 3: Implement the expand_tools builtin**

```python
# cubepi/deferred/_expand_tool.py
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/test_expand_tool.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && git add cubepi/deferred/_expand_tool.py tests/deferred/test_expand_tool.py && git commit -m "feat(deferred): expand_tools builtin tool (#166)"
```

---

## Task 4: `DeferredToolsMiddleware` — core middleware

**Files:**
- Create: `cubepi/deferred/middleware.py`
- Modify: `cubepi/deferred/__init__.py`
- Test: `tests/deferred/test_middleware.py`

This is the largest task. The middleware ties together catalog rendering, the expand_tools builtin, loader caching, mid-run tool injection, and expansion state persistence.

- [ ] **Step 1: Write the failing tests**

```python
# tests/deferred/test_middleware.py
from __future__ import annotations

import json

import pytest

from cubepi.agent.types import AgentContext, AgentTool, AgentToolResult, AfterToolCallContext
from cubepi.deferred.middleware import DeferredToolsMiddleware, ResumedState
from cubepi.deferred.types import DeferredToolGroup
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_tool(name: str, description: str = "dummy") -> AgentTool:
    from pydantic import BaseModel

    class _Empty(BaseModel):
        pass

    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    return AgentTool(name=name, description=description, parameters=_Empty, execute=_exec)


def _make_group(
    group_id: str,
    tool_names: list[str],
    *,
    display_name: str = "Test",
    description: str = "desc",
    loader_tools: list[AgentTool] | None = None,
    loader_call_count: list[int] | None = None,
) -> DeferredToolGroup:
    tools = loader_tools or [_dummy_tool(n) for n in tool_names]
    call_count = loader_call_count if loader_call_count is not None else [0]

    async def _loader() -> list[AgentTool]:
        call_count[0] += 1
        return list(tools)

    return DeferredToolGroup(
        group_id=group_id,
        display_name=display_name,
        description=description,
        tool_names=tool_names,
        loader=_loader,
    )


def _make_expand_result(group_id: str, tool_names: list[str] | None = None) -> AgentToolResult:
    """Simulate what expand_tools returns on success."""
    payload = {
        "group_id": group_id,
        "expanded": True,
        "tool_names": tool_names or [],
        "remaining": 0,
        "error": None,
    }
    return AgentToolResult(content=[TextContent(text=json.dumps(payload))])


def _make_after_tool_call_ctx(
    tool_name: str,
    args: dict,
    result: AgentToolResult,
    context: AgentContext,
    is_error: bool = False,
) -> AfterToolCallContext:
    tc = ToolCall(id="tc-1", name=tool_name, arguments=args)
    msg = AssistantMessage(content=[tc])
    from pydantic import BaseModel

    class _Empty(BaseModel):
        pass

    return AfterToolCallContext(
        assistant_message=msg,
        tool_call=tc,
        args=_Empty(),
        result=result,
        is_error=is_error,
        context=context,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMiddlewareConstruction:
    def test_tools_attribute_contains_expand_tools(self) -> None:
        extra: dict = {}
        group = _make_group("mcp:a", ["t1"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        assert len(mw.tools) == 1
        assert mw.tools[0].name == "expand_tools"


class TestTransformSystemPrompt:
    async def test_appends_catalog(self) -> None:
        extra: dict = {}
        group = _make_group("mcp:github", ["create_issue", "search_repos"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        ctx = AgentContext(system_prompt="base", messages=[])
        result = await mw.transform_system_prompt("base prompt", ctx=ctx)
        assert "mcp:github" in result
        assert "create_issue" in result
        assert "base prompt" in result

    async def test_fully_expanded_group_omitted_from_catalog(self) -> None:
        extra: dict = {"expanded_groups": {"mcp:github": None}}
        group = _make_group("mcp:github", ["t1"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        ctx = AgentContext(system_prompt="", messages=[])
        result = await mw.transform_system_prompt("base", ctx=ctx)
        assert "mcp:github" not in result or "Expanded tool groups" in result

    async def test_expanded_schemas_appended_after_catalog(self) -> None:
        extra: dict = {"expanded_groups": {"mcp:a": None}}
        group = _make_group("mcp:a", ["t1"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        # Simulate having expanded schemas stored
        mw._expanded_schemas.append(
            ("mcp:a", [{"name": "t1", "description": "Tool 1", "parameters": {}}])
        )
        ctx = AgentContext(system_prompt="", messages=[])
        result = await mw.transform_system_prompt("base", ctx=ctx)
        assert "Expanded tool groups" in result
        assert "t1" in result


class TestAfterToolCallExpansion:
    async def test_expand_all_injects_tools(self) -> None:
        extra: dict = {}
        group = _make_group("mcp:github", ["create_issue", "search_repos"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)

        context_tools: list[AgentTool] = [mw.tools[0]]
        ctx = AgentContext(system_prompt="", messages=[], tools=context_tools, extra=extra)

        # Simulate calling expand_tools
        output = await mw._expand(group_id="mcp:github", tool_names=None, context=ctx)
        assert output.expanded is True
        assert len(output.tool_names) == 2
        assert output.remaining == 0
        # Tools injected into context
        assert len(context_tools) == 3  # expand_tools + 2 new
        assert extra["expanded_groups"] == {"mcp:github": None}

    async def test_expand_selective_injects_only_requested(self) -> None:
        extra: dict = {}
        group = _make_group("mcp:github", ["create_issue", "search_repos", "create_pr"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)

        context_tools: list[AgentTool] = [mw.tools[0]]
        ctx = AgentContext(system_prompt="", messages=[], tools=context_tools, extra=extra)

        output = await mw._expand(group_id="mcp:github", tool_names=["create_issue"], context=ctx)
        assert output.expanded is True
        assert output.tool_names == ["create_issue"]
        assert output.remaining == 2
        assert len(context_tools) == 2  # expand_tools + 1 new
        assert extra["expanded_groups"] == {"mcp:github": ["create_issue"]}

    async def test_incremental_expand_same_group(self) -> None:
        extra: dict = {}
        call_count = [0]
        group = _make_group("mcp:github", ["t1", "t2", "t3"], loader_call_count=call_count)
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)

        context_tools: list[AgentTool] = [mw.tools[0]]
        ctx = AgentContext(system_prompt="", messages=[], tools=context_tools, extra=extra)

        # First expand
        await mw._expand(group_id="mcp:github", tool_names=["t1"], context=ctx)
        assert len(context_tools) == 2
        assert extra["expanded_groups"] == {"mcp:github": ["t1"]}

        # Second expand
        await mw._expand(group_id="mcp:github", tool_names=["t2"], context=ctx)
        assert len(context_tools) == 3
        assert extra["expanded_groups"] == {"mcp:github": ["t1", "t2"]}

        # Loader called only once
        assert call_count[0] == 1

    async def test_expand_unknown_group_returns_error(self) -> None:
        extra: dict = {}
        mw = DeferredToolsMiddleware(groups=[], extra_ref=lambda: extra)
        ctx = AgentContext(system_prompt="", messages=[], tools=[], extra=extra)

        output = await mw._expand(group_id="bad:id", tool_names=None, context=ctx)
        assert output.expanded is False
        assert output.error is not None
        assert "expanded_groups" not in extra

    async def test_expand_idempotent_no_duplicate(self) -> None:
        extra: dict = {}
        group = _make_group("mcp:github", ["t1"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)

        context_tools: list[AgentTool] = [mw.tools[0]]
        ctx = AgentContext(system_prompt="", messages=[], tools=context_tools, extra=extra)

        await mw._expand(group_id="mcp:github", tool_names=None, context=ctx)
        assert len(context_tools) == 2

        # Expand again — same tool, no duplicate
        await mw._expand(group_id="mcp:github", tool_names=None, context=ctx)
        assert len(context_tools) == 2

    async def test_loader_failure_returns_error(self) -> None:
        async def _failing_loader() -> list[AgentTool]:
            raise RuntimeError("connection refused")

        group = DeferredToolGroup(
            group_id="mcp:broken",
            display_name="Broken",
            description="desc",
            tool_names=["t1"],
            loader=_failing_loader,
        )
        extra: dict = {}
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        ctx = AgentContext(system_prompt="", messages=[], tools=[], extra=extra)

        output = await mw._expand(group_id="mcp:broken", tool_names=None, context=ctx)
        assert output.expanded is False
        assert "connection refused" in (output.error or "")
        assert "expanded_groups" not in extra


class TestExpansionOrderPreserved:
    async def test_expansion_order_in_schemas(self) -> None:
        extra: dict = {}
        g1 = _make_group("mcp:z", ["tz"])
        g2 = _make_group("mcp:a", ["ta"])
        mw = DeferredToolsMiddleware(groups=[g1, g2], extra_ref=lambda: extra)

        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools), extra=extra)

        # Expand z first, then a
        await mw._expand(group_id="mcp:z", tool_names=None, context=ctx)
        await mw._expand(group_id="mcp:a", tool_names=None, context=ctx)

        # Schema order should be z then a (expansion order), not a then z (sorted)
        assert list(extra["expanded_groups"].keys()) == ["mcp:z", "mcp:a"]
        assert len(mw._expanded_schemas) == 2
        assert mw._expanded_schemas[0][0] == "mcp:z"
        assert mw._expanded_schemas[1][0] == "mcp:a"


class TestPrepareResumedState:
    async def test_fully_expanded_group(self) -> None:
        group = _make_group("mcp:github", ["t1", "t2"])
        expanded: dict[str, list[str] | None] = {"mcp:github": None}
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group], expanded=expanded,
        )
        assert len(resumed.pre_loaded_tools) == 2
        assert len(resumed.remaining_groups) == 0

    async def test_partially_expanded_group(self) -> None:
        group = _make_group("mcp:github", ["t1", "t2", "t3"])
        expanded: dict[str, list[str] | None] = {"mcp:github": ["t1"]}
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group], expanded=expanded,
        )
        assert len(resumed.pre_loaded_tools) == 1
        assert resumed.pre_loaded_tools[0].name == "t1"
        assert len(resumed.remaining_groups) == 1

    async def test_unexpanded_group_stays_deferred(self) -> None:
        group = _make_group("mcp:github", ["t1"])
        expanded: dict[str, list[str] | None] = {}
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group], expanded=expanded,
        )
        assert len(resumed.pre_loaded_tools) == 0
        assert len(resumed.remaining_groups) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/test_middleware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cubepi.deferred.middleware'`

- [ ] **Step 3: Implement the middleware**

```python
# cubepi/deferred/middleware.py
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
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
    pre_loaded_tools: list[AgentTool]
    remaining_groups: list[DeferredToolGroup]


class DeferredToolsMiddleware(Middleware):
    def __init__(
        self,
        *,
        groups: list[DeferredToolGroup],
        extra_ref: Callable[[], dict[str, Any]],
        catalog_header: str = DEFAULT_CATALOG_HEADER,
    ) -> None:
        self._groups: dict[str, DeferredToolGroup] = {g.group_id: g for g in groups}
        self._extra_ref = extra_ref
        self._catalog_header = catalog_header
        self._loader_cache: dict[str, list[AgentTool]] = {}
        self._expanded_schemas: list[tuple[str, list[ToolSchema]]] = []

        self.tools: list[AgentTool] = [
            _make_expand_tools(expand_callback=self._expand_callback)
        ]

    async def _expand_callback(
        self, group_id: str, tool_names: list[str] | None
    ) -> ExpandToolsOutput:
        # Called from expand_tools execute(). Does validation, loader
        # invocation, state bookkeeping, and schema recording. Stores
        # newly loaded tools in _pending_injection for after_tool_call
        # to inject into context.tools (execute() has no AgentContext).
        extra = self._extra_ref()
        group = self._groups.get(group_id)
        if group is None:
            return ExpandToolsOutput(
                group_id=group_id,
                expanded=False,
                tool_names=[],
                remaining=0,
                error=f"Unknown group_id: {group_id}. Available: {', '.join(sorted(self._groups))}",
            )

        # Load tools (cached per group)
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
        expanded_groups: dict[str, list[str] | None] = extra.get("expanded_groups", {})
        already_expanded = expanded_groups.get(group_id)
        already_set: set[str] = set(already_expanded) if isinstance(already_expanded, list) else (
            set() if already_expanded is None and group_id not in expanded_groups else
            {t.name for t in all_loaded}
        )

        # Determine which tools to expand this call
        if tool_names is not None:
            requested = [t for t in all_loaded if t.name in set(tool_names)]
        else:
            requested = list(all_loaded)

        newly_expanded = [t for t in requested if t.name not in already_set]
        expanded_names = [t.name for t in requested]

        # Update expansion state
        if tool_names is None:
            expanded_groups[group_id] = None
        else:
            prev = expanded_groups.get(group_id)
            if prev is None and group_id in expanded_groups:
                pass  # already fully expanded
            else:
                merged = list(prev) if isinstance(prev, list) else []
                for name in expanded_names:
                    if name not in set(merged):
                        merged.append(name)
                expanded_groups[group_id] = merged

        extra["expanded_groups"] = expanded_groups

        # Store schema info for expanded tools (append-only for cache stability)
        if newly_expanded:
            new_schemas = [t.to_definition().model_dump() for t in newly_expanded]
            # Check if this group already has a schemas entry
            existing_idx = next(
                (i for i, (gid, _) in enumerate(self._expanded_schemas) if gid == group_id),
                None,
            )
            if existing_idx is not None:
                prev_schemas = self._expanded_schemas[existing_idx][1]
                self._expanded_schemas[existing_idx] = (group_id, prev_schemas + new_schemas)
            else:
                self._expanded_schemas.append((group_id, new_schemas))

        # Store pending injection for after_tool_call
        self._pending_injection = newly_expanded

        # Calculate remaining
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

    async def _expand(
        self,
        *,
        group_id: str,
        tool_names: list[str] | None,
        context: AgentContext,
    ) -> ExpandToolsOutput:
        """Expand a group — called directly in tests, via callback + after_tool_call in production."""
        output = await self._expand_callback(group_id, tool_names)
        if output.expanded and hasattr(self, "_pending_injection"):
            newly_expanded = self._pending_injection
            self._pending_injection = []
            if context.tools is not None:
                existing_names = {t.name for t in context.tools}
                for tool in newly_expanded:
                    if tool.name not in existing_names:
                        context.tools.append(tool)
        return output

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

        # Inject pending tools into context
        if hasattr(self, "_pending_injection") and self._pending_injection:
            newly_expanded = self._pending_injection
            self._pending_injection = []
            if ctx.context.tools is not None:
                existing_names = {t.name for t in ctx.context.tools}
                for tool in newly_expanded:
                    if tool.name not in existing_names:
                        ctx.context.tools.append(tool)

        return None

    async def transform_system_prompt(
        self,
        system_prompt: str,
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> str:
        del signal
        extra = self._extra_ref()
        expanded: dict[str, list[str] | None] = extra.get("expanded_groups", {})

        catalog = render_catalog(
            groups=list(self._groups.values()),
            expanded=expanded,
            header=self._catalog_header,
        )

        schemas = render_expanded_schemas(expanded_schemas=self._expanded_schemas)

        parts = [system_prompt]
        if catalog:
            parts.append(catalog)
        if schemas:
            parts.append(schemas)

        return "\n\n".join(parts)

    @staticmethod
    async def prepare_resumed_state(
        groups: list[DeferredToolGroup],
        expanded: dict[str, list[str] | None],
    ) -> ResumedState:
        pre_loaded: list[AgentTool] = []
        remaining: list[DeferredToolGroup] = []

        for group in groups:
            exp = expanded.get(group.group_id)
            if exp is None and group.group_id not in expanded:
                remaining.append(group)
                continue

            loaded = await group.loader()
            if exp is None:
                pre_loaded.extend(loaded)
            else:
                name_set = set(exp)
                selected = [t for t in loaded if t.name in name_set]
                pre_loaded.extend(selected)
                remaining.append(group)

        return ResumedState(
            pre_loaded_tools=pre_loaded,
            remaining_groups=remaining,
        )
```

- [ ] **Step 4: Update `__init__.py` exports**

```python
# cubepi/deferred/__init__.py
"""Deferred tool groups — progressive tool disclosure primitive."""

from cubepi.deferred.middleware import DeferredToolsMiddleware, ResumedState
from cubepi.deferred.types import DeferredToolGroup

__all__ = ["DeferredToolGroup", "DeferredToolsMiddleware", "ResumedState"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/test_middleware.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run all deferred tests together**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/ -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && git add cubepi/deferred/middleware.py cubepi/deferred/__init__.py tests/deferred/test_middleware.py && git commit -m "feat(deferred): DeferredToolsMiddleware with expansion, catalog, injection (#166)"
```

---

## Task 5: Agent-level `deferred_tool_groups` parameter

**Files:**
- Modify: `cubepi/agent/agent.py:141-166`
- Test: `tests/deferred/test_agent_wiring.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/deferred/test_agent_wiring.py
from __future__ import annotations

import pytest

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.deferred import DeferredToolGroup, DeferredToolsMiddleware
from cubepi.providers.base import TextContent
from pydantic import BaseModel


class _Empty(BaseModel):
    pass


def _dummy_tool(name: str) -> AgentTool:
    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    return AgentTool(name=name, description="dummy", parameters=_Empty, execute=_exec)


def _make_group(group_id: str, tool_names: list[str]) -> DeferredToolGroup:
    async def _loader():
        return [_dummy_tool(n) for n in tool_names]

    return DeferredToolGroup(
        group_id=group_id,
        display_name="Test",
        description="desc",
        tool_names=tool_names,
        loader=_loader,
    )


def _make_faux_model():
    """Create a minimal BoundModel for Agent construction."""
    from cubepi.providers.faux import FauxProvider
    provider = FauxProvider()
    return provider.model("faux")


class TestAgentDeferredToolGroups:
    def test_agent_accepts_deferred_tool_groups(self) -> None:
        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1", "t2"])
        agent = Agent(
            model=model,
            tools=[_dummy_tool("builtin")],
            deferred_tool_groups=[group],
        )
        # expand_tools should be in the tool set
        tool_names = [t.name for t in agent._state.tools]
        assert "builtin" in tool_names
        assert "expand_tools" in tool_names

    def test_middleware_auto_created(self) -> None:
        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1"])
        agent = Agent(
            model=model,
            deferred_tool_groups=[group],
        )
        assert any(isinstance(mw, DeferredToolsMiddleware) for mw in agent._middleware)

    def test_extra_ref_bound_to_agent_extra(self) -> None:
        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1"])
        agent = Agent(
            model=model,
            deferred_tool_groups=[group],
        )
        deferred_mw = next(mw for mw in agent._middleware if isinstance(mw, DeferredToolsMiddleware))
        # The extra_ref should point to agent._extra
        assert deferred_mw._extra_ref() is agent._extra

    def test_no_deferred_groups_no_middleware(self) -> None:
        model = _make_faux_model()
        agent = Agent(model=model, tools=[_dummy_tool("t1")])
        assert not any(isinstance(mw, DeferredToolsMiddleware) for mw in agent._middleware)

    def test_explicit_middleware_still_works(self) -> None:
        """User can still pass DeferredToolsMiddleware directly."""
        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1"])
        extra: dict = {}
        mw = DeferredToolsMiddleware(
            groups=[group],
            extra_ref=lambda: extra,
            catalog_header="Custom",
        )
        agent = Agent(
            model=model,
            middleware=[mw],
        )
        assert any(isinstance(m, DeferredToolsMiddleware) for m in agent._middleware)
        tool_names = [t.name for t in agent._state.tools]
        assert "expand_tools" in tool_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/test_agent_wiring.py -v`
Expected: FAIL with `TypeError: Agent.__init__() got an unexpected keyword argument 'deferred_tool_groups'`

- [ ] **Step 3: Add `deferred_tool_groups` parameter to Agent**

In `cubepi/agent/agent.py`, modify `Agent.__init__`:

1. Add the parameter to the signature (after `middleware`):

```python
        deferred_tool_groups: list[DeferredToolGroup] | None = None,
```

2. Add the import at the top of the file (with the other type imports):

```python
from __future__ import annotations
from typing import TYPE_CHECKING
# ... existing imports ...
if TYPE_CHECKING:
    from cubepi.deferred.types import DeferredToolGroup
```

3. Add the auto-wiring logic right before the existing `middleware = middleware or []` line (around line 185):

```python
        if deferred_tool_groups:
            from cubepi.deferred.middleware import DeferredToolsMiddleware

            deferred_mw = DeferredToolsMiddleware(
                groups=deferred_tool_groups,
                extra_ref=lambda: self._extra,
            )
            middleware = [*(middleware or []), deferred_mw]
```

The exact edit: the current line 185 reads `middleware = middleware or []`. Replace it with the deferred_tool_groups logic followed by the existing line. The conditional must run before `middleware = middleware or []` because it needs to append to the list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/test_agent_wiring.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run all existing agent tests to verify no regression**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/agent/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && git add cubepi/agent/agent.py tests/deferred/test_agent_wiring.py && git commit -m "feat(deferred): Agent(deferred_tool_groups=...) parameter (#166)"
```

---

## Task 6: Type checking and full test sweep

**Files:**
- All files from Tasks 1-5

- [ ] **Step 1: Run mypy on the new package**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run mypy cubepi/deferred/`
Expected: PASS (no errors)

- [ ] **Step 2: Run mypy on agent.py**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run mypy cubepi/agent/agent.py`
Expected: PASS (no errors). Fix any type issues found.

- [ ] **Step 3: Run all deferred tests**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/deferred/ -v`
Expected: ALL PASS

- [ ] **Step 4: Run full test suite**

Run: `cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && uv run pytest tests/ -v --ignore=tests/e2e`
Expected: ALL PASS. Fix any regressions.

- [ ] **Step 5: Commit any fixes**

Only if Step 1-4 required changes:

```bash
cd /home/chris/cubepi/.worktrees/feat-deferred-tool-group && git add -u && git commit -m "fix(deferred): type and test fixes (#166)"
```

---

## Summary

| Task | What it builds | Key files |
|------|---------------|-----------|
| 1 | `DeferredToolGroup` dataclass | `types.py` |
| 2 | Catalog + expanded schema rendering | `_catalog.py` |
| 3 | `expand_tools` builtin tool | `_expand_tool.py` |
| 4 | `DeferredToolsMiddleware` (core) | `middleware.py` |
| 5 | `Agent(deferred_tool_groups=...)` | `agent.py` |
| 6 | Type checking + full sweep | all |
