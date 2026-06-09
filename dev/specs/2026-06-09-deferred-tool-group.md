# DeferredToolGroup — Progressive Tool Disclosure Primitive

Date: 2026-06-09
Branch: `feat/deferred-tool-group`
Issue: cubepi#166

## Problem

When a cubepi agent has many tools registered (MCP servers, plugins, host-provided builtins), their
full JSON schemas are sent to the model on every turn via `tools=`. This creates three scaling
problems:

- **Context bloat.** Tool definitions are part of the cached prefix. A handful of rich servers can
  push the tool block into tens of thousands of tokens before the user types anything.
- **Cache cost.** That block is billed at cache-write rate on changes and cache-read rate every
  turn. More tools = a bigger fixed tax per turn.
- **Worse tool selection.** Published benchmarks show accuracy drops as the toolset grows; the model
  has to scan dozens of similar schemas to pick one.

Host applications (cubebox, standalone cubepi users) need a way to register tool groups that are
hidden by default and expanded on demand, without building the entire disclosure mechanism
themselves.

## Goals

- Provide a `DeferredToolGroup` dataclass that host apps use to register groups of tools that start
  collapsed.
- Provide a `DeferredToolsMiddleware` that handles everything: catalog rendering in the system
  prompt, an `expand_tools` builtin, mid-run tool injection on expand, expansion-order tracking.
- **Zero changes to Agent or loop.py.** The mechanism works entirely through the existing middleware
  system.
- Expansion-order-preserving, append-only system-prompt growth for prompt-cache stability.
- Tool-source-agnostic: cubepi knows about "groups with loaders", not about MCP or any specific
  tool source.

## Non-goals

- No MCP-specific integration in v1 (convenience bridge from `MCPDiscoveryResult` →
  `DeferredToolGroup` is a later recipe, not core API).
- No per-tool (sub-group) disclosure — the unit is a whole group.
- No semantic / embedding retrieval — the catalog is designed to be small enough for the model to
  read directly.
- No "code mode" (tools as a callable API the model writes code against).
- No changes to the existing MCP loader module (`cubepi/mcp/`).

## Prior art

- **hermes-agent Tool Search** (`tools/tool_search.py`, 735 lines): per-tool granularity, three
  bridge tools (search/describe/call), BM25 retrieval, stateless catalog rebuilt every turn,
  context-window-percentage threshold. Key lessons: (1) core tools must never defer, (2) catalog
  drift silently drops tools (OpenClaw #84141), (3) transparent unwrap so hooks see real tool names.
- **Anthropic Tool Search Tool** (GA Feb 2026): provider-side `defer_loading: true`, regex/BM25
  search, ~85% token reduction. Provider-specific; cubepi must work provider-agnostically.
- **cubebox skills system** (`SkillsMiddleware` + `load_skill`): the direct analog in cubebox.
  Catalog in system prompt, model calls builtin to expand, middleware injects content into prompt
  suffix. DeferredToolGroup generalizes this pattern for arbitrary tool sources.
- **cubepi TodoListMiddleware**: the in-repo pattern to follow — `tools` attribute for builtin
  injection, `extra_ref` for state persistence, `transform_system_prompt` for prompt augmentation.

## Current cubepi architecture (relevant to this design)

### Agent tool lifecycle

1. **Construction.** `Agent.__init__` receives `tools: list[AgentTool]` and `middleware: list`.
   Middleware `tools` attributes are extracted and appended to `agent.state.tools` (agent.py:191-195).
2. **Context snapshot.** `_create_context_snapshot()` (agent.py:713-719) builds `AgentContext` with
   `tools=list(self._state._tools)`. This is the snapshot the loop runs against.
3. **Loop reads.** `run_agent_loop` creates `current_context` with `tools=context.tools`
   (loop.py:61) — **a reference, not a deep copy**. Each iteration calls
   `_stream_assistant_response(context=current_context)` which re-reads `context.tools` at
   loop.py:705-706 to build `tools_defs`.

**Key insight:** because `current_context.tools` is a reference to the same list object, appending
to it from an `after_tool_call` hook makes new tools visible to the model on the **next iteration
of the same run**. No changes to Agent or loop.py are needed for mid-run tool injection.

### Middleware hooks (used by this feature)

- `tools: list[AgentTool]` — class attribute; extracted at Agent construction and merged into
  `agent.state.tools`. Used to inject the `expand_tools` builtin.
- `transform_system_prompt(system_prompt, *, ctx, signal) → str` — chained sequentially across
  middlewares. Used to append catalog text and expanded schema text.
- `after_tool_call(ctx, *, signal) → AfterToolCallResult | None` — fires after every tool
  execution. Used to detect `expand_tools` calls and inject the expanded group's tools.

### State persistence

`Agent._extra` is a `JsonObject` dict shared with `AgentContext.extra`. Middlewares write state
into it (e.g. TodoListMiddleware writes todos). Host applications persist it via checkpointers.
DeferredToolGroup uses this for expansion state (`extra["expanded_groups"]`).

## Design

### Data types

```python
@dataclass
class DeferredToolGroup:
    group_id: str                                       # "mcp:github", "plugin:kanban"
    display_name: str                                   # "GitHub"
    description: str                                    # "Code hosting: issues, PRs"
    tool_names: list[str]                               # for catalog display only
    loader: Callable[[], Awaitable[list[AgentTool]]]    # invoked on expand
```

`loader` is an async callback that returns callable `AgentTool`s. It is invoked exactly once per
expansion (not per turn). The host application owns what happens inside: cubebox calls
`load_workspace_mcp_tools_for_cubepi` filtered to one server; a standalone user might call
`load_mcp_tools_http` directly.

`tool_names` is for catalog display only — these names appear in the system prompt so the model
can judge which group to expand. They are never used to construct tools or validate schemas.

### `DeferredToolsMiddleware`

A single middleware that encapsulates the entire feature. Follows TodoListMiddleware patterns.

```python
class DeferredToolsMiddleware(Middleware):
    def __init__(
        self,
        *,
        groups: list[DeferredToolGroup],
        extra_ref: Callable[[], dict[str, Any]],
        catalog_header: str = DEFAULT_CATALOG_HEADER,
    ) -> None: ...
```

**Construction-time:**
- Stores the deferred groups indexed by `group_id`.
- Creates the `expand_tools` builtin as `self.tools = [_make_expand_tools(...)]`.
- `extra_ref` works exactly like TodoListMiddleware: a closure returning `agent._extra` so the
  tool's `execute` can write expansion state into the persisted dict.

**`tools` attribute** — exposes `[expand_tools_agent_tool]`. Agent construction merges this into
`agent.state.tools` automatically (agent.py:191-195).

**`transform_system_prompt`:**
1. Append **catalog section** — sorted by `group_id` for byte-stability. For each group, shows only
   the **not-yet-expanded** tool names. A fully expanded group is omitted entirely; a partially
   expanded group shows only remaining tools with an updated count.
2. Append **expanded schema section** — for each expanded group, **in expansion order** (not
   sorted), append the **expanded tools'** full definitions (name + description + parameters JSON).
   When a group is partially expanded, only the selected tools' schemas appear. Expansion order is
   append-only: a newly expanded group (or newly expanded tools within a group) always lands after
   every already-rendered block, preserving earlier cache segments byte-identical.

Why expansion order, not sorted? Groups expand incrementally mid-conversation. Sorting could insert
a later expansion before an already-cached block and invalidate the prompt-cache prefix.

**`after_tool_call`:**
- Fires on every tool call. Checks: is the tool name `expand_tools`? Did it succeed?
- If yes: parse the result to get the `group_id` and optional `tool_names`. Look up the group.
  On first access to a group, call `loader()` and **cache the full result** (the loader is invoked
  exactly once per group per run regardless of how many selective expansions follow). Filter the
  cached tools by `tool_names` (or take all if `tool_names` is None). Append the filtered
  `AgentTool`s to `ctx.context.tools` (the live list reference — next iteration sees them).
  Record the expansion in `extra["expanded_groups"]`.
- Store the loaded tools' definitions for `transform_system_prompt` to render in the expanded
  schema section.

### `expand_tools` builtin

```python
class ExpandToolsInput(BaseModel):
    group_id: str = Field(
        description="The group_id from your 'Deferred tool groups' catalog."
    )
    tool_names: list[str] | None = Field(
        default=None,
        description="Specific tools to expand. Omit to expand all tools in the group.",
    )

class ExpandToolsOutput(BaseModel):
    group_id: str
    expanded: bool
    tool_names: list[str]       # the tools that were actually expanded this call
    remaining: int              # tools still deferred in this group
    error: str | None = None
```

Behavior:
- Valid group_id, no tool_names → expand all tools, return full list.
- Valid group_id + tool_names → expand only those tools, return them. Unknown names within the
  list are silently ignored (the rest still expand).
- Unknown group_id → return `is_error=True` + error message.
- Already-expanded tools → idempotent (no re-inject, no duplicate in context.tools). The response
  still lists them as expanded.

**Incremental expansion.** The model can call `expand_tools` multiple times for the same group with
different `tool_names`. Each call adds only the newly requested tools. Example flow:
```
expand_tools("mcp:github", tool_names=["create_issue"])       → 1 tool loaded, 11 remaining
expand_tools("mcp:github", tool_names=["search_repos"])       → 1 more loaded, 10 remaining
expand_tools("mcp:github")                                     → remaining 10 loaded, 0 remaining
```

The tool result contains tool names + descriptions only, **not** full schemas. The middleware
injects schema text into the system-prompt suffix (matching the skills pattern where content goes
into the prompt, not the tool result).

### Catalog rendering

Example output appended to system prompt:

```
# Deferred tool groups

These tool groups are available but not yet loaded. Call `expand_tools(group_id)`
to load a group's tools for the rest of this conversation.

- `mcp:linear` — Issue tracking (8 tools)
  create_issue, update_issue, search_issues, get_issue, create_project,
  list_projects, create_cycle, list_cycles
- `mcp:gdrive` — Google Drive (5 tools)
  search_files, read_file, list_folders, get_file_metadata, create_file
```

Properties:
- Sorted by `group_id` → byte-identical every turn (for the non-expanded portion).
- Fully expanded groups are omitted. Partially expanded groups show only the remaining
  (not-yet-expanded) tool names with an updated count.
- Tool names listed without descriptions or schemas — tool names in MCP/plugin conventions
  (`verb_noun`) are self-descriptive. ~40 tokens per group of 12 tools.
- `catalog_header` is customizable via constructor for host apps with different wording needs.

Example after partial expansion of `mcp:github` (`create_issue` already expanded):
```
- `mcp:github` — Code hosting (11 remaining tools)
  search_repos, create_pr, list_pull_requests, merge_pull_request,
  search_code, list_commits, create_comment, ...
```

### Expanded schema rendering

Appended after the catalog, in expansion order:

```
# Expanded tool groups

## mcp:linear

- **create_issue**: Open a new issue
  Parameters: {"type": "object", "properties": {"title": {"type": "string"}, ...}}
- **update_issue**: Update an existing issue
  Parameters: {...}
...

## mcp:gdrive

...
```

Properties:
- Expansion order preserved (append-only, never re-sorted).
- Schema text derived from the loaded `AgentTool.to_definition()` at expansion time.
- Stored on the middleware instance (not re-computed from tools list) so it's stable across turns.

### Mid-run tool injection

When `after_tool_call` fires for a successful `expand_tools`:

```python
# First access to this group: call loader, cache the full result
if group_id not in self._loader_cache:
    self._loader_cache[group_id] = await group.loader()

all_tools = self._loader_cache[group_id]

# Filter by requested tool_names (or take all)
selected = [t for t in all_tools if t.name in requested_names] if requested_names else all_tools

# Only inject tools not already in context (idempotent)
existing_names = {t.name for t in ctx.context.tools}
new_tools = [t for t in selected if t.name not in existing_names]
ctx.context.tools.extend(new_tools)   # visible next iteration
```

This works because `current_context.tools` in `_run_loop` is a reference to the same list — the
loop re-reads `context.tools` at each iteration (loop.py:705). No Agent or loop changes needed.

The expanded tools are real `AgentTool`s in `tools=` — the model can call them directly. The
expanded schema section in the system prompt is supplementary context (descriptions, parameter
docs), not the mechanism that makes tools callable.

### Expansion state persistence

```python
extra["expanded_groups"]
# Ordered dict: group_id → list[str] | None
# None = all tools expanded; list = specific tool names
# Ordering = expansion order (first group expanded first)
#
# Example:
# {
#     "mcp:github": ["create_issue", "search_repos"],  # partial
#     "mcp:linear": null,                                # all expanded
# }
```

- Written by `after_tool_call` on each new expansion (append new group, or extend existing
  group's tool list).
- Read by `transform_system_prompt` to decide what to render in catalog vs expanded sections.
- Persisted by the host application's checkpointer (same mechanism as TodoListMiddleware's todos).
- **Must be serialized as an ordered dict.** If a checkpointer deserializes it as an unordered map,
  the expansion-order invariant breaks and the prompt-cache prefix becomes unstable across turns.
  Python `dict` preserves insertion order since 3.7, so standard JSON round-tripping is safe.

### Replay on subsequent runs

When the host application creates a new run (next user turn), it should:

1. Read persisted `extra["expanded_groups"]` from the checkpointer (an ordered dict of
   `group_id → list[str] | None`).
2. Pre-load the expanded tools: for each group with expanded tools, call loader and filter to the
   expanded tool names (or take all if `None`).
3. Pass the pre-loaded tools as regular `tools` to `Agent(tools=[...builtins, ...pre_loaded])`.
4. Construct `DeferredToolsMiddleware` with the original groups list — the middleware reads
   `extra["expanded_groups"]` to know which tools are already expanded and adjusts the catalog
   accordingly.

This is the host application's responsibility, not the middleware's — the middleware only handles
within-a-single-run expansion. cubepi provides a helper for step 2-3:

```python
@dataclass
class ResumedState:
    pre_loaded_tools: list[AgentTool]       # tools to pass as Agent(tools=...)
    remaining_groups: list[DeferredToolGroup]  # groups still fully/partially deferred

@staticmethod
async def prepare_resumed_state(
    groups: list[DeferredToolGroup],
    expanded: dict[str, list[str] | None],
) -> ResumedState:
    """Call loaders for expanded groups, filter to expanded tools, return split."""
    ...
```

## File layout

```
cubepi/
  deferred/
    __init__.py          # public exports
    types.py             # DeferredToolGroup dataclass
    middleware.py         # DeferredToolsMiddleware
    _catalog.py          # catalog + expanded schema rendering (pure functions)
    _expand_tool.py      # expand_tools builtin (AgentTool factory)
tests/
  test_deferred/
    test_catalog.py      # catalog rendering determinism
    test_middleware.py    # middleware integration (after_tool_call, transform_system_prompt)
    test_expand_tool.py  # expand_tools builtin (valid/invalid/idempotent)
    test_injection.py    # mid-run tool injection via context.tools mutation
```

New `cubepi/deferred/` package — does not import from `cubepi/mcp/` (tool-source-agnostic).

## Public API surface

```python
# cubepi.deferred
from cubepi.deferred import DeferredToolGroup, DeferredToolsMiddleware

# Construction
groups = [
    DeferredToolGroup(
        group_id="mcp:github",
        display_name="GitHub",
        description="Code hosting: issues, PRs, repos",
        tool_names=["create_issue", "search_repos", "create_pr", ...],
        loader=lambda: load_my_github_tools(),
    ),
]

middleware = DeferredToolsMiddleware(
    groups=groups,
    extra_ref=lambda: agent._extra,
)

agent = Agent(
    model=model,
    system_prompt="...",
    tools=[...builtins],
    middleware=[middleware, ...other_middleware],
)
```

Two exports. `DeferredToolGroup` is a plain dataclass, `DeferredToolsMiddleware` is the middleware.
No registration method on Agent — follows the existing pattern where middlewares are passed to
`Agent(middleware=[...])`.

## Testing strategy

Unit tests (no real LLM, no MCP server):

1. **Catalog rendering** — sorted by group_id, byte-stable across input orderings, no schemas,
   fully expanded groups omitted, partially expanded groups show only remaining tools.
2. **Expanded schema rendering** — expansion order preserved, append-only (second expansion's text
   starts with first expansion's text), JSON keys sorted for byte stability. Partial expansion
   renders only the selected tools' schemas.
3. **expand_tools builtin** — valid group → expanded=True + tool names; unknown group → error;
   already-expanded → idempotent; with tool_names → selective expansion.
4. **Incremental expansion** — expand one tool, then another from same group; loader called only
   once; both tools in context; catalog shows remaining.
5. **Mid-run injection** — after expand_tools fires, `ctx.context.tools` contains the new tools;
   simulate a second `_stream_assistant_response` call and verify `tools_defs` includes them.
6. **Expansion state** — after_tool_call writes to extra["expanded_groups"] correctly; duplicate
   tool names not added twice; transform_system_prompt uses the stored order.
7. **Loader failure** — if loader raises, expand_tools returns is_error=True, no tools injected,
   no state change.
8. **Loader caching** — two selective expands on the same group call loader exactly once.

Integration test (with Agent, mock provider):

9. **Full round-trip** — Agent with DeferredToolsMiddleware, mock provider that calls expand_tools
   then uses an expanded tool. Assert: first model call sees expand_tools but not deferred tools;
   after expansion, next iteration sees deferred tools in tools_defs.

## Open questions

- **Catalog in system prompt vs. in tool description.** Current design puts the catalog in the
  system prompt (via `transform_system_prompt`). Alternative: put it in the `expand_tools` tool's
  description field. Pro: shorter system prompt. Con: tool descriptions are less visible to models
  than system prompt text, and the catalog changes as groups expand (dynamic tool descriptions are
  unusual). Recommend: system prompt (current design).
- **Loader caching.** Should the middleware cache loader results so a re-expand (or replay) doesn't
  re-invoke the loader? v1: no — the host application manages caching if needed. The middleware
  calls loader exactly once per group per run.
- **Max groups / token budget.** Should the middleware have a hard limit on how many groups can be
  registered, or a token-budget gate like hermes-agent's threshold_pct? v1: no — the host
  application decides what to defer. A future version could add an `auto` mode that estimates
  token cost and only defers when it's worth the indirection overhead.

## v1 scope vs later

**v1:**
- `DeferredToolGroup` dataclass.
- `DeferredToolsMiddleware` with catalog rendering, `expand_tools` builtin, mid-run injection,
  expansion-order tracking, expanded schema rendering.
- `prepare_resumed_state` helper for cross-run replay.
- Unit + integration tests.

**Later:**
- MCP convenience bridge (`deferred_group_from_mcp_server`).
- Auto-mode threshold (context-window-percentage gate).
- Unify with skills disclosure under a shared primitive.
- Per-tool (sub-group) disclosure.
- Semantic / embedding retrieval for very large catalogs.
