---
title: Tool Use & Parallel Execution
---

# Tool Use & Parallel Execution

Tools are how an agent acts on the world. CubePi turns each `AgentTool`
into a JSON Schema for the model, validates arguments with Pydantic,
runs the work, and feeds the result back as a `ToolResultMessage`. By
default tools run in parallel when the model calls more than one in a
single turn.

## Anatomy of a tool

```python
from pydantic import BaseModel, Field
from cubepi import AgentTool, AgentToolResult, TextContent


class SearchParams(BaseModel):
    query: str = Field(..., description="The natural-language query")
    limit: int = Field(10, ge=1, le=100)


async def search(tool_call_id, params: SearchParams, *, signal=None, on_update=None):
    results = await my_search_backend(params.query, params.limit)
    return AgentToolResult(
        content=[TextContent(text="\n".join(results))],
        details={"raw_results": results},   # passes through to ToolResultMessage.details
    )


search_tool = AgentTool(
    name="search",
    description="Search the internal knowledge base.",
    parameters=SearchParams,
    execute=search,
)
```

The `description` is shown to the model verbatim — write it for the
model, not for humans. Pydantic `Field(description=...)` propagates
into the JSON Schema and helps the model understand each parameter.

## Parallel by default

When the model emits multiple tool calls in one assistant message,
CubePi schedules them on `asyncio.create_task()` and gathers them.
That's almost always what you want.

```python
agent = Agent(
    provider=provider,
    model=model,
    tools=[search_tool, fetch_url_tool, summarise_tool],
)
```

The event stream emits `tool_execution_start` for all of them up
front, interleaved `tool_execution_update` events as each tool reports
progress, then `tool_execution_end` per tool in completion order.

## Forcing sequential execution

There are two ways to opt out of parallel mode:

1. **Per agent** — `Agent(tool_execution="sequential")`. All tool
   batches run one-by-one in the order the model emitted them.

2. **Per tool** — set `execution_mode="sequential"` on the
   `AgentTool`. If *any* tool in the current batch is sequential, the
   whole batch falls back to sequential.

    ```python
    write_db_tool = AgentTool(
        name="write_db",
        description="Persist a record.",
        parameters=WriteDbParams,
        execute=write_db,
        execution_mode="sequential",   # opt out of parallelism for safety
    )
    ```

The built-in `ask_user` HITL tool (see [HITL guide](../hitl)) uses
`execution_mode="sequential"` — it pauses the agent for human input and
cannot share a turn with other tools.

Use sequential mode when tools mutate shared state (a DB, a counter)
and you want a deterministic order.

## Streaming tool progress

Long-running tools can stream partial updates that show up as
`tool_execution_update` events on the agent stream:

```python
async def slow_search(tool_call_id, params, *, signal=None, on_update=None):
    for i, page in enumerate(await fetch_pages(params.query)):
        if signal and signal.is_set():
            break
        if on_update:
            on_update({"progress": i, "total": len(pages), "url": page.url})
        await process(page)
    return AgentToolResult(content=[TextContent(text="done")])
```

`partial_result` in the event is whatever object you handed to
`on_update`. Use a small dict; it doesn't end up in the model's
context.

## Cancelling in-flight tools

The `signal` argument is the same `asyncio.Event` that
`agent.abort()` sets. Check it in any loop:

```python
async def long_running(tool_call_id, params, *, signal=None, on_update=None):
    for chunk in big_dataset:
        if signal and signal.is_set():
            return AgentToolResult(content=[TextContent(text="cancelled")])
        await process_chunk(chunk)
```

If your work is one big `await`, wrap it in
`asyncio.wait_for(..., timeout=…)` or use the abort callbacks of the
underlying library.

## Returning errors

Two ways:

1. **Raise an exception.** CubePi catches it, turns it into an
   `AgentToolResult` with `is_error=True` and the exception string as
   `TextContent`.
2. **Return `is_error=True` explicitly.** Useful when you want a
   structured error body:

    ```python
    return AgentToolResult(
        content=[TextContent(text="Rate limit exceeded; try again in 60s")],
        is_error=True,
    )
    ```

Either way, the model receives a tool result that's clearly marked as
an error and usually adapts (retries with different args, asks the
user, etc.).

## Ending the turn from a tool: `terminate`

A tool can declare *"after this, stop looping — don't ask the model
again."* Set `terminate=True`:

```python
async def submit_final_answer(tool_call_id, params, *, signal=None, on_update=None):
    save_answer(params.answer)
    return AgentToolResult(
        content=[TextContent(text="submitted")],
        terminate=True,
    )
```

CubePi only terminates if *every* tool result in the current batch is
`terminate=True`. The agent loop emits `turn_end`, then `agent_end`,
and exits.

## Common pitfalls

- **Forgetting the keyword-only args** — `execute(tool_call_id,
  params)` will work in dev but crash when the framework passes
  `signal=`. Keep `*, signal=None, on_update=None` in the signature.
- **Heavy `details` payloads** — `details` is preserved through the
  agent event but is *not* shown to the model. Don't pack massive blobs
  there unless you have a downstream consumer.
- **Pydantic strictness surprises** — `Field(..., min_length=1)` lets
  the model see the constraint via JSON Schema; constraints help, but
  remember the model still sometimes sends bad JSON. CubePi turns the
  `ValidationError` into a tool error result; you don't need to wrap
  validation yourself.
- **`tools=[]` and the model still asks for a tool** — Usually means
  the prompt mentions one. Either remove the suggestion from the
  system prompt or pass the tool.

## See also

- [Streaming Events](./streaming) — how `tool_execution_*` events fit
  the larger event taxonomy.
- [Middleware → before_tool_call](../middleware/hooks#before_tool_call)
  and [after_tool_call](../middleware/hooks#after_tool_call) —
  interception, policy, retries.
- [Recipes → Weather Agent](../../recipes/weather-agent) — a real
  HTTP-calling tool, end-to-end.
- [MCP Loading](../mcp/loading) — pull an entire toolset off an MCP
  server in one call.
