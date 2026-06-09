---
title: Deferred Tool Groups
description: "Hide MCP tool schemas from the model by default, expanding them on demand to reduce context bloat."
---

# Deferred Tool Groups

When an agent connects to many MCP servers, their combined tool schemas
can consume thousands of tokens of context on every turn — even if the
model only needs one or two groups for the current task.
`DeferredToolGroup` solves this by replacing full schemas with a compact
catalog, letting the model expand groups on demand.

## How it works

1. At construction time, the agent's system prompt includes a short
   catalog — one line per group with a description and tool list.
2. The model sees a built-in `load_tools` tool it can call to load a
   group (or specific tools within a group).
3. On expansion, the loader runs once, the tools are injected into the
   live tool set, and their schemas are appended to the system prompt.

```
# Deferred tool groups

These tool groups are available but not yet loaded. Call `load_tools(group_id)`
to load a group's tools for the rest of this conversation.

- `mcp:github` — GitHub: Issues, PRs, repos, code search (4 tools)
  create_issue, search_repos, create_pr, list_comments
- `mcp:linear` — Linear: Project management and issue tracking (6 tools)
  create_issue, update_issue, list_projects, ...
```

## Basic setup

Pass `deferred_tool_groups` to `Agent`. The middleware is created
automatically — no manual wiring needed:

```python
from cubepi import Agent
from cubepi.deferred import DeferredToolGroup

github_group = DeferredToolGroup(
    group_id="mcp:github",
    display_name="GitHub",
    description="Issues, PRs, repos, code search",
    tool_names=["create_issue", "search_repos", "create_pr", "list_comments"],
    loader=github_mcp.load_tools,  # async () -> list[AgentTool]
)

linear_group = DeferredToolGroup(
    group_id="mcp:linear",
    display_name="Linear",
    description="Project management and issue tracking",
    tool_names=["create_issue", "update_issue", "list_projects"],
    loader=linear_mcp.load_tools,
)

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    tools=[search_tool, calculator],              # always-available tools
    deferred_tool_groups=[github_group, linear_group],
)
```

### `DeferredToolGroup` fields

| Field | Type | Description |
|---|---|---|
| `group_id` | `str` | Unique identifier the model uses in `load_tools` calls (e.g. `"mcp:github"`) |
| `display_name` | `str` | Human-readable label shown in the catalog |
| `description` | `str` | One-line summary of the group's capabilities |
| `tool_names` | `list[str]` | Tool names shown in the catalog |
| `loader` | `async () -> list[AgentTool]` | Callback that returns the full tool set for this group |

## The `load_tools` tool

The model calls `load_tools` to load a group's tools. Two modes:

```
# Expand everything in the group
load_tools(group_id="mcp:github")

# Expand specific tools only
load_tools(group_id="mcp:github", tool_names=["create_issue", "search_repos"])
```

The tool returns a structured result:

```json
{
  "group_id": "mcp:github",
  "expanded": true,
  "tool_names": ["create_issue", "search_repos", "create_pr", "list_comments"],
  "remaining": 0
}
```

After expansion, the tools are immediately available for the model to
call in the same turn (via the `after_tool_call` hook).

### Selective expansion

The model can expand a group incrementally — requesting one or two tools
now and more later:

```
load_tools(group_id="mcp:github", tool_names=["create_issue"])
# → remaining: 3

# later...
load_tools(group_id="mcp:github", tool_names=["search_repos"])
# → remaining: 2
```

Already-expanded tools are idempotent — re-requesting them is a no-op.

### Loader caching

The `loader` callback is invoked exactly **once per group per run**.
The first `load_tools` call triggers it; subsequent selective
expansions filter from the cached result. If the loader fails, the
error is returned to the model and the group remains unexpanded.

## Prompt-cache stability

The system prompt is designed for prompt-cache prefix stability:

- **Catalog** is sorted by `group_id` alphabetically — input order
  doesn't matter, the rendered text is byte-stable.
- **Expanded schemas** are appended in expansion order (the order the
  model called `load_tools`), never reordered. Each new expansion
  appends to the end, preserving the existing prefix.

This means the LLM API's prompt cache remains valid across turns: the
system prompt only grows, and only at the end.

## Expansion state

The middleware tracks which groups are expanded in `ctx.extra`:

```python
ctx.extra["expanded_groups"] = {
    "mcp:github": None,                    # fully expanded (None = all tools)
    "mcp:linear": ["create_issue"],        # partially expanded
    # mcp:slack not present = unexpanded
}
```

This state survives checkpointing and can be used for cross-run replay
(see below).

## Cross-run replay

When resuming a conversation from a previous run, you need to restore
the expansion state so the model has the same tools available.
`prepare_resumed_state` handles this:

```python
from cubepi.deferred import DeferredToolsMiddleware

# saved_extra is the persisted ctx.extra from the previous run
resumed = await DeferredToolsMiddleware.prepare_resumed_state(
    groups=all_groups,
    expanded=saved_extra["expanded_groups"],
)

agent = Agent(
    model=model,
    tools=[*builtin_tools, *resumed.pre_loaded_tools],
    deferred_tool_groups=resumed.remaining_groups,
)
```

`prepare_resumed_state` returns a `ResumedState` with:

| Field | Description |
|---|---|
| `pre_loaded_tools` | Tools from previously-expanded groups, ready to use |
| `remaining_groups` | Groups that were never expanded or only partially expanded |
| `expanded_schemas` | Schema data for the system prompt (pass to `resumed_schemas` for advanced use) |
| `loader_cache` | Pre-loaded tool cache (pass to `resumed_loader_cache` to avoid redundant loader calls) |

Fully expanded groups are loaded and removed from the deferred set.
Partially expanded groups load the selected tools but stay deferrable
(the model can still expand the rest).

### Restoring schema text

The `Agent(deferred_tool_groups=...)` shorthand handles the common case.
For full prompt-cache continuity — where the resumed run's system prompt
must match the previous run's final state byte-for-byte — construct the
middleware directly with `resumed_schemas`:

```python
mw = DeferredToolsMiddleware(
    groups=resumed.remaining_groups,
    extra_ref=lambda: agent_extra,
    resumed_schemas=resumed.expanded_schemas,
    resumed_loader_cache=resumed.loader_cache,
)

agent = Agent(
    model=model,
    tools=[*builtin_tools, *resumed.pre_loaded_tools],
    middleware=[mw],
)
```

## Advanced: constructing the middleware directly

For full control over the catalog header, cross-run schema seeding, or
other middleware options, construct `DeferredToolsMiddleware` yourself:

```python
from cubepi.deferred import DeferredToolsMiddleware

mw = DeferredToolsMiddleware(
    groups=[github_group, linear_group],
    extra_ref=lambda: agent_extra,
    catalog_header="# Available integrations\n\nExpand with load_tools().",
    resumed_schemas=None,  # or pass schemas from a previous run
)

agent = Agent(
    model=model,
    tools=[search_tool],
    middleware=[mw],
)
```

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `groups` | `list[DeferredToolGroup]` | required | Groups to defer |
| `extra_ref` | `() -> dict` | required | Returns the live `ctx.extra` dict |
| `catalog_header` | `str` | *(built-in)* | Header text for the catalog section |
| `resumed_schemas` | `list[tuple[str, list[dict]]] \| None` | `None` | Schema data to seed from a previous run |
| `resumed_loader_cache` | `dict[str, list[AgentTool]] \| None` | `None` | Pre-loaded tool cache from a previous run (avoids re-calling loaders on resume) |
| `on_tools_expanded` | `(list[AgentTool]) -> None \| None` | `None` | Called after new tools are expanded (used internally for cross-turn persistence) |

When using the `Agent(deferred_tool_groups=...)` shorthand, `extra_ref`
is automatically bound to `self._extra`.

## When to use it

**Good fit:**

- Agent has access to 5+ MCP servers but typically uses 1–2 per conversation.
- Tool schemas are large (many parameters, long descriptions).
- You want to keep prompt-cache hit rates high across turns.

**Skip it when:**

- The agent has only a few tools — the overhead of the catalog and
  `load_tools` call isn't worth it.
- All tools are needed on every turn — deferring just adds a round trip.
- Tool schemas are small — the context savings are minimal.

## See also

- [Loading MCP Tools](../mcp/loading) — how to get `AgentTool` lists from
  MCP servers.
- [The 8 Hooks](./hooks) — the middleware hooks that power deferred tools
  (`transform_system_prompt`, `after_tool_call`).
- [Composition](./composition) — how middleware composes when stacked with
  other middleware.
