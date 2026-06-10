# Deferred Tools v2 — Dispatch Strategy Implementation Plan

> **For agentic workers:** Execute task-by-task with TDD. Steps use checkbox (`- [ ]`) syntax
> for tracking. Spec: `dev/specs/2026-06-10-deferred-dispatcher.md`.

**Goal:** Add a zero-cache-invalidation `dispatch` strategy (default) to deferred tool groups:
static tools array + static system prompt, schemas delivered via `load_tools` results, calls
routed through a `deferred_tool_call` dispatcher that the engine unwraps to real tool names.

**Architecture:** A new `resolve_tool_call` hook at the top of the tool-execution pipeline
rewrites dispatcher calls before validation/hooks/tracing, so everything downstream sees the
real tool. Loaded tools live in `context.tools` with a new `AgentTool.expose_to_model=False`
flag that keeps them out of the provider payload. The middleware grows a `strategy` parameter;
inject mode survives slimmed (no system-prompt schema section).

**Tech stack:** Python 3.13, pydantic, pytest (asyncio_mode=auto), FauxProvider for all tests.
Commands always via `uv`.

**Working directory:** `.worktrees/2026-06-10-deferred-dispatcher` (branch
`2026-06-10-deferred-dispatcher`). All paths below are relative to the worktree root.

---

## File map

| File | Change |
|---|---|
| `cubepi/agent/types.py` | `AgentTool.expose_to_model: bool = True` |
| `cubepi/agent/loop.py` | Filter hidden tools from provider payload; thread `resolve_tool_call` |
| `cubepi/agent/tools.py` | `resolve_tool_call` hook in `_prepare_tool_call`; schema suffix on resolved-call validation errors |
| `cubepi/middleware/base.py` | `Middleware.resolve_tool_call` + first-non-None composition |
| `cubepi/agent/agent.py` | Compose/thread the hook; `deferred_tool_strategy` param |
| `cubepi/deferred/types.py` | `DeferredStrategy` literal |
| `cubepi/deferred/_dispatch_tool.py` | **new** — `deferred_tool_call` builtin (fallback-error execute) |
| `cubepi/deferred/_expand_tool.py` | `LoadToolsOutput.schemas` + `usage` fields |
| `cubepi/deferred/_catalog.py` | `render_static_catalog`; delete `render_expanded_schemas` |
| `cubepi/deferred/middleware.py` | `strategy` param; resolver + implicit load; inject slimming; resume simplification |
| `cubepi/deferred/__init__.py` | Export `DeferredStrategy` |
| `tests/agent/test_resolve_tool_call.py` | **new** — engine hook unit tests |
| `tests/deferred/test_dispatch.py` | **new** — dispatch strategy + byte-stability tests |
| `tests/deferred/test_catalog.py`, `test_middleware.py`, `test_agent_wiring.py` | Update for slimming/strategy |
| `website/docs/guides/middleware/deferred-tools.md` | Document both strategies, cache table |
| `CHANGELOG.md` | Breaking-change entry with one-line opt-out |

---

### Task 1: `AgentTool.expose_to_model` + provider-payload filter

**Files:**
- Modify: `cubepi/agent/types.py:34-50` (AgentTool dataclass)
- Modify: `cubepi/agent/loop.py:705-707` (`_stream_assistant_response`)
- Test: `tests/deferred/test_dispatch.py` (new file)

- [ ] **Step 1: Write the failing tests**

Create `tests/deferred/test_dispatch.py`:

```python
from __future__ import annotations

from pydantic import BaseModel

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import TextContent


class _Empty(BaseModel):
    pass


def _dummy_tool(name: str, *, expose: bool = True) -> AgentTool:
    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    return AgentTool(
        name=name,
        description="dummy",
        parameters=_Empty,
        execute=_exec,
        expose_to_model=expose,
    )


class TestExposeToModel:
    def test_default_is_true(self) -> None:
        async def _exec(tool_call_id, args, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = AgentTool(
            name="t", description="d", parameters=_Empty, execute=_exec
        )
        assert tool.expose_to_model is True

    def test_hidden_tool_excluded_from_payload_filter(self) -> None:
        tools = [_dummy_tool("visible"), _dummy_tool("hidden", expose=False)]
        visible = [t.to_definition() for t in tools if t.expose_to_model]
        assert [d.name for d in visible] == ["visible"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/deferred/test_dispatch.py -v`
Expected: FAIL — `TypeError: AgentTool.__init__() got an unexpected keyword argument 'expose_to_model'`

- [ ] **Step 3: Add the field**

In `cubepi/agent/types.py`, append to the `AgentTool` dataclass after `hitl`:

```python
@dataclass
class AgentTool(Generic[TParams]):
    name: str
    description: str
    parameters: type[TParams]
    execute: Callable[..., Awaitable[AgentToolResult]]
    label: str = ""
    execution_mode: Literal["sequential", "parallel"] | None = None
    hitl_builtin: bool = False
    hitl: HitlBinding | None = None
    # When False the tool is resolvable/executable by the engine but its
    # definition is never sent to the provider (deferred dispatch mode).
    expose_to_model: bool = True
```

- [ ] **Step 4: Apply the payload filter in the loop**

In `cubepi/agent/loop.py` (`_stream_assistant_response`, currently lines 705-707), replace:

```python
    tools_defs = None
    if context.tools:
        tools_defs = [t.to_definition() for t in context.tools]
```

with:

```python
    tools_defs = None
    if context.tools:
        visible = [t for t in context.tools if t.expose_to_model]
        if visible:
            tools_defs = [t.to_definition() for t in visible]
```

- [ ] **Step 5: Run tests + full suite slice**

Run: `uv run pytest tests/deferred/ tests/agent/ -q`
Expected: PASS (the new tests pass; nothing else regresses — the filter is a no-op while
every existing tool defaults to `expose_to_model=True`).

- [ ] **Step 6: Commit**

```bash
git add cubepi/agent/types.py cubepi/agent/loop.py tests/deferred/test_dispatch.py
git commit -m "feat(agent): AgentTool.expose_to_model controls provider payload visibility"
```

---

### Task 2: `resolve_tool_call` engine hook

**Files:**
- Modify: `cubepi/agent/tools.py` (`_prepare_tool_call` at :142, `execute_tool_calls` at :318, `_execute_sequential` at :359, `_execute_parallel` at :425)
- Modify: `cubepi/agent/loop.py` (both `execute_tool_calls` call sites, :306 resume and :590 main loop; add the parameter to every loop entry function that already takes `before_tool_call`)
- Modify: `cubepi/middleware/base.py` (`Middleware` base method + `compose_middleware`)
- Modify: `cubepi/agent/agent.py` (`__init__` param + `self.resolve_tool_call` + pass-through at :590/:723/:746)
- Test: `tests/agent/test_resolve_tool_call.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/agent/test_resolve_tool_call.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, Field

from cubepi.agent.tools import execute_tool_calls
from cubepi.agent.types import AgentContext, AgentTool, AgentToolResult
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
)
from cubepi.providers.faux import faux_assistant_message


class _EchoArgs(BaseModel):
    value: str = Field(description="echoed back")


def _echo_tool(name: str, *, expose: bool = True) -> AgentTool:
    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echo:{args.value}")])

    return AgentTool(
        name=name,
        description="echo",
        parameters=_EchoArgs,
        execute=_exec,
        expose_to_model=expose,
    )


def _assistant_with(call: ToolCall) -> AssistantMessage:
    # faux_assistant_message fills usage/timestamp with valid defaults.
    return faux_assistant_message(call, stop_reason="tool_use")


def _noop_emit(event) -> None:
    return None


async def test_resolver_rewrites_call_before_pipeline() -> None:
    real = _echo_tool("real_tool", expose=False)
    ctx = AgentContext(system_prompt="", messages=[], tools=[real])
    call = ToolCall(
        id="tc-1",
        name="deferred_tool_call",
        arguments={"tool_name": "real_tool", "arguments": {"value": "hi"}},
    )
    seen_before: list[str] = []

    async def resolver(tool_call, *, context, signal=None):
        if tool_call.name != "deferred_tool_call":
            return None
        return ToolCall(
            id=tool_call.id,
            name=tool_call.arguments["tool_name"],
            arguments=tool_call.arguments["arguments"],
        )

    async def before(hook_ctx, *, signal=None):
        seen_before.append(hook_ctx.tool_call.name)
        return None

    batch = await execute_tool_calls(
        ctx,
        _assistant_with(call),
        before_tool_call=before,
        resolve_tool_call=resolver,
        emit=_noop_emit,
    )
    # Hook saw the real name, not the dispatcher envelope.
    assert seen_before == ["real_tool"]
    # Result keyed to the ORIGINAL tool_use id, carrying the real name.
    msg = batch.messages[0]
    assert msg.tool_call_id == "tc-1"
    assert msg.tool_name == "real_tool"
    assert msg.content[0].text == "echo:hi"


async def test_resolver_none_is_passthrough() -> None:
    tool = _echo_tool("plain")
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    call = ToolCall(id="tc-2", name="plain", arguments={"value": "x"})

    async def resolver(tool_call, *, context, signal=None):
        return None

    batch = await execute_tool_calls(
        ctx, _assistant_with(call), resolve_tool_call=resolver, emit=_noop_emit
    )
    assert batch.messages[0].content[0].text == "echo:x"


async def test_resolver_exception_becomes_error_result() -> None:
    tool = _echo_tool("plain")
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    call = ToolCall(id="tc-3", name="plain", arguments={"value": "x"})

    async def resolver(tool_call, *, context, signal=None):
        raise RuntimeError("resolver blew up")

    batch = await execute_tool_calls(
        ctx, _assistant_with(call), resolve_tool_call=resolver, emit=_noop_emit
    )
    msg = batch.messages[0]
    assert msg.is_error is True
    assert "resolver blew up" in msg.content[0].text


async def test_resolved_call_validation_error_includes_schema() -> None:
    real = _echo_tool("real_tool", expose=False)
    ctx = AgentContext(system_prompt="", messages=[], tools=[real])
    call = ToolCall(
        id="tc-4",
        name="deferred_tool_call",
        arguments={"tool_name": "real_tool", "arguments": {"wrong_field": 1}},
    )

    async def resolver(tool_call, *, context, signal=None):
        return ToolCall(
            id=tool_call.id,
            name="real_tool",
            arguments=tool_call.arguments["arguments"],
        )

    batch = await execute_tool_calls(
        ctx, _assistant_with(call), resolve_tool_call=resolver, emit=_noop_emit
    )
    msg = batch.messages[0]
    assert msg.is_error is True
    text = msg.content[0].text
    assert "Invalid arguments for tool 'real_tool'" in text
    # Full schema appended so the model can self-correct in one round trip.
    assert '"value"' in text


async def test_unresolved_call_validation_error_has_no_schema() -> None:
    tool = _echo_tool("plain")
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    call = ToolCall(id="tc-5", name="plain", arguments={"wrong": 1})

    batch = await execute_tool_calls(ctx, _assistant_with(call), emit=_noop_emit)
    text = batch.messages[0].content[0].text
    assert "Invalid arguments" in text
    assert "Full schema" not in text
```

(The two validation-error tests belong to Task 3's implementation; writing them now is fine —
they stay red until Task 3.)

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/agent/test_resolve_tool_call.py -v`
Expected: FAIL — `TypeError: execute_tool_calls() got an unexpected keyword argument 'resolve_tool_call'`

- [ ] **Step 3: Thread the hook through `cubepi/agent/tools.py`**

Add the parameter to `execute_tool_calls`, `_execute_sequential`, `_execute_parallel` (mirror
the existing `before_tool_call` plumbing — same position, default `None`), and pass it into
`_prepare_tool_call`. Then in `_prepare_tool_call`, before the tool lookup loop:

```python
async def _prepare_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    before_tool_call: Callable | None,
    resolve_tool_call: Callable | None,
    signal: asyncio.Event | None,
) -> _PreparedToolCall | _ImmediateOutcome:
    resolved = False
    if resolve_tool_call:
        try:
            rewritten = await resolve_tool_call(
                tool_call, context=context, signal=signal
            )
        except HitlControlException:
            raise
        except Exception as exc:
            return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)
        if rewritten is not None:
            tool_call = rewritten
            resolved = True

    tool = None
    ...
```

`resolved` is consumed in Task 3 — fold Task 3 Step 1's validation-branch change in
immediately while editing this function (avoids an assigned-but-unused lint window); Task 3
then only runs its tests.

**Event naming (review finding).** `ToolExecutionStartEvent` / `ToolExecutionEndEvent` must
carry the **resolved** name. Check where `_execute_sequential` / `_execute_parallel` emit
them: if the start event is emitted from the raw `tool_call` before `_prepare_tool_call`
runs, move the emission after preparation (or emit from `prepared.tool_call`) so tracing's
`_find_tool(event.tool_name)` (cubepi/tracing/recorder.py ~:832) resolves the real tool.
Extend `test_resolver_rewrites_call_before_pipeline` with an emit recorder:

```python
    events: list = []
    batch = await execute_tool_calls(
        ctx,
        _assistant_with(call),
        before_tool_call=before,
        resolve_tool_call=resolver,
        emit=events.append,
    )
    from cubepi.agent.types import ToolExecutionStartEvent

    start_names = [e.tool_name for e in events if isinstance(e, ToolExecutionStartEvent)]
    assert start_names == ["real_tool"]
```

(Replace the `emit=_noop_emit` in that one test with the recorder above.)

- [ ] **Step 4: Thread through `cubepi/agent/loop.py`**

Add `resolve_tool_call: Callable | None = None` as a parameter to every function in `loop.py`
that already has a `before_tool_call` parameter (the loop entry points and the resume path),
and pass `resolve_tool_call=resolve_tool_call` at both `execute_tool_calls` call sites
(currently :306 and :590).

- [ ] **Step 5: Middleware base + composition**

In `cubepi/middleware/base.py`, add to the `Middleware` base class (next to
`before_tool_call` at :51), matching its def-but-not-abstract style:

```python
    async def resolve_tool_call(
        self,
        tool_call: ToolCall,
        *,
        context: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> ToolCall | None:
        """Rewrite a tool call before validation/hooks. Return None to pass through.

        The returned ToolCall MUST keep the original ``id`` — the result
        message is keyed by it on the wire.

        Composition is FIRST-NON-NONE-WINS: the first middleware to return a
        rewritten call short-circuits the chain. This differs from
        ``before_tool_call``, which chains sequentially. A resolver never sees
        another resolver's output.
        """
        return None
```

In `compose_middleware`, add a first-non-None-wins chain (after the `before_chain` block):

```python
    resolve_chain = [m for m in middlewares if _has_method(m, "resolve_tool_call")]
    if resolve_chain:

        async def composed_resolve(tool_call, *, context, signal=None):
            for mw in resolve_chain:
                result = await mw.resolve_tool_call(
                    tool_call, context=context, signal=signal
                )
                if result is not None:
                    return result
            return None

        hooks["resolve_tool_call"] = composed_resolve
```

Import `ToolCall` from `cubepi.providers.base` if not already imported.

- [ ] **Step 6: Agent wiring in `cubepi/agent/agent.py`**

Mirror `before_tool_call` exactly: add `resolve_tool_call: Callable | None = None` to
`__init__` (next to `before_tool_call` at :180), set
`self.resolve_tool_call = resolve_tool_call or _mw_hooks.get("resolve_tool_call")` (next to
:248), and pass `resolve_tool_call=self.resolve_tool_call` at the three loop invocation sites
(:590, :723, :746).

**Fork wiring (review finding — regression otherwise).** The fork construction in
`agent.py` (~:581-599) builds the child with `middleware=[]` but forwards hook callables
explicitly (`before_tool_call` is already passed). Forward the resolver the same way:
`resolve_tool_call=self.resolve_tool_call` in the fork's `Agent(...)` kwargs. Without this,
dispatch-mode forks can never invoke loaded deferred tools (the dispatcher's execute is
denied by `_deny_in_fork`, hidden tools are invisible, and no resolver exists) — a regression
vs v1 where expanded tools remained usable in forks. Add a test in
`tests/deferred/test_dispatch.py` that drives a fork the same way
`tests/agent/test_agent_fork_once.py` does and asserts a `deferred_tool_call` to a
pre-loaded tool succeeds inside the fork.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/agent/test_resolve_tool_call.py -v`
Expected: the first three tests PASS; the two validation-schema tests still FAIL (Task 3).

Run: `uv run pytest tests/ -q`
Expected: no regressions outside the two intentionally-red tests.

- [ ] **Step 8: Commit**

```bash
git add cubepi/agent/tools.py cubepi/agent/loop.py cubepi/middleware/base.py \
        cubepi/agent/agent.py tests/agent/test_resolve_tool_call.py
git commit -m "feat(agent): resolve_tool_call hook rewrites tool calls before the pipeline"
```

---

### Task 3: Schema suffix on resolved-call validation errors

**Files:**
- Modify: `cubepi/agent/tools.py` (`_prepare_tool_call` validation branch, currently :162-170)
- Test: `tests/agent/test_resolve_tool_call.py` (already written in Task 2)

- [ ] **Step 1: Implement**

In `_prepare_tool_call`, replace the `ValidationError` branch:

```python
    try:
        validated_args = tool.parameters.model_validate(tool_call.arguments)
    except ValidationError as exc:
        message = _format_validation_error(exc, tool.name)
        if resolved:
            schema = json.dumps(
                tool.parameters.model_json_schema(),
                sort_keys=True,
                ensure_ascii=False,
            )
            message = (
                f"{message}\n\nFull schema for '{tool.name}':\n{schema}"
            )
        return _ImmediateOutcome(result=_error_result(message), is_error=True)
```

Add `import json` to the module imports.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/agent/test_resolve_tool_call.py -v`
Expected: all 5 PASS.

- [ ] **Step 3: Commit**

```bash
git add cubepi/agent/tools.py
git commit -m "feat(agent): resolved-call validation errors append the full tool schema"
```

---

### Task 4: Static catalog + schemas in `LoadToolsOutput`

**Files:**
- Modify: `cubepi/deferred/types.py` (add `DeferredStrategy`)
- Modify: `cubepi/deferred/_catalog.py` (add `render_static_catalog`; `render_expanded_schemas` is deleted in Task 6)
- Modify: `cubepi/deferred/_expand_tool.py` (`LoadToolsOutput` gains `schemas`, `usage`)
- Test: `tests/deferred/test_catalog.py`, `tests/deferred/test_expand_tool.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/deferred/test_catalog.py` (reuse its existing `_make_group`-style helpers if
present; otherwise construct `DeferredToolGroup` inline as below):

```python
class TestStaticCatalog:
    def _group(self, gid: str, names: list[str]) -> DeferredToolGroup:
        async def _loader():
            return []

        return DeferredToolGroup(
            group_id=gid,
            display_name="GitHub",
            description="GitHub tools",
            tool_names=names,
            loader=_loader,
        )

    def test_lists_all_tools_sorted_by_group_id(self) -> None:
        out = render_static_catalog(
            groups=[self._group("b", ["t2"]), self._group("a", ["t1"])],
            header="HDR",
        )
        assert out.index("`a`") < out.index("`b`")
        assert "t1" in out and "t2" in out

    def test_no_remaining_counts(self) -> None:
        out = render_static_catalog(
            groups=[self._group("a", ["t1", "t2"])], header="HDR"
        )
        assert "remaining" not in out

    def test_deterministic(self) -> None:
        groups = [self._group("a", ["t1"]), self._group("b", ["t2"])]
        assert render_static_catalog(
            groups=groups, header="HDR"
        ) == render_static_catalog(groups=list(reversed(groups)), header="HDR")
```

Append to `tests/deferred/test_expand_tool.py`:

```python
def test_load_tools_output_carries_schemas() -> None:
    out = LoadToolsOutput(
        group_id="g",
        expanded=True,
        tool_names=["t"],
        remaining=0,
        schemas=[{"name": "t", "description": "d", "parameters": {}}],
    )
    dumped = out.model_dump()
    assert dumped["schemas"][0]["name"] == "t"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/deferred/test_catalog.py tests/deferred/test_expand_tool.py -v`
Expected: FAIL — `ImportError: cannot import name 'render_static_catalog'` / pydantic
`ValidationError: Unexpected keyword argument` for `schemas`.

- [ ] **Step 3: Implement**

`cubepi/deferred/types.py` — add at module level:

```python
from typing import Literal

DeferredStrategy = Literal["dispatch", "inject"]
```

`cubepi/deferred/_catalog.py` — add (keep `render_catalog` as-is for inject mode):

```python
DEFAULT_DISPATCH_CATALOG_HEADER = (
    "# Deferred tool groups\n"
    "\n"
    "These tool groups are available but not yet loaded. Call `load_tools(group_id)`\n"
    "to get their full schemas, then invoke them via\n"
    "`deferred_tool_call(tool_name=..., arguments=...)`.\n"
    "If you already know the right arguments from the names below, you may call\n"
    "`deferred_tool_call` directly — the tool loads on demand."
)


def render_static_catalog(
    *,
    groups: list[DeferredToolGroup],
    header: str = DEFAULT_DISPATCH_CATALOG_HEADER,
) -> str:
    """Dispatch-mode catalog: byte-stable, independent of expansion state."""
    lines: list[str] = []
    for group in sorted(groups, key=lambda g: g.group_id):
        count = len(group.tool_names)
        lines.append(
            f"- `{group.group_id}` — {group.display_name}: "
            f"{group.description} ({count} tools)"
        )
        lines.append(f"  {', '.join(group.tool_names)}")
    if not lines:
        return ""
    return header + "\n\n" + "\n".join(lines)
```

`cubepi/deferred/_expand_tool.py` — extend the output model:

```python
class LoadToolsOutput(BaseModel):
    group_id: str
    expanded: bool
    tool_names: list[str]
    remaining: int
    error: str | None = None
    # Dispatch mode only: full schemas, delivered in the tool result so they
    # live in message history (append-only, cache-safe). The calling
    # convention lives once in the static catalog header and the dispatcher
    # description — not repeated per result.
    schemas: list[dict[str, object]] | None = None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/deferred/test_catalog.py tests/deferred/test_expand_tool.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cubepi/deferred/types.py cubepi/deferred/_catalog.py cubepi/deferred/_expand_tool.py \
        tests/deferred/test_catalog.py tests/deferred/test_expand_tool.py
git commit -m "feat(deferred): static dispatch catalog + schema-bearing LoadToolsOutput"
```

---

### Task 5: Dispatcher builtin + middleware dispatch strategy

**Files:**
- Create: `cubepi/deferred/_dispatch_tool.py`
- Modify: `cubepi/deferred/middleware.py`
- Modify: `cubepi/deferred/__init__.py` (export `DeferredStrategy`)
- Test: `tests/deferred/test_dispatch.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/deferred/test_dispatch.py` (extend the Task 1 header imports):

```python
import json

from cubepi.agent.types import AgentContext
from cubepi.deferred import DeferredToolGroup, DeferredToolsMiddleware
from cubepi.providers.base import ToolCall


class _EchoArgs(BaseModel):
    value: str


def _echo_tool(name: str) -> AgentTool:
    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echo:{args.value}")])

    return AgentTool(name=name, description="echo", parameters=_EchoArgs, execute=_exec)


def _make_group(gid: str, names: list[str]) -> DeferredToolGroup:
    async def _loader():
        return [_echo_tool(n) for n in names]

    return DeferredToolGroup(
        group_id=gid,
        display_name="Test",
        description="desc",
        tool_names=names,
        loader=_loader,
    )


def _mw(groups, *, strategy="dispatch", extra=None):
    extra = extra if extra is not None else {}
    return DeferredToolsMiddleware(
        groups=groups, extra_ref=lambda: extra, strategy=strategy
    )


class TestDispatchStrategy:
    def test_tools_attr_has_both_builtins(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        names = [t.name for t in mw.tools]
        assert names == ["load_tools", "deferred_tool_call"]

    def test_inject_strategy_has_only_load_tools(self) -> None:
        mw = _mw([_make_group("g", ["t1"])], strategy="inject")
        assert [t.name for t in mw.tools] == ["load_tools"]

    async def test_system_prompt_static_across_expansion(self) -> None:
        mw = _mw([_make_group("g", ["t1", "t2"])])
        ctx = AgentContext(system_prompt="base", messages=[], tools=list(mw.tools))
        before = await mw.transform_system_prompt("base", ctx=ctx)
        await mw._expand_callback("g", None)
        after = await mw.transform_system_prompt("base", ctx=ctx)
        assert before == after  # byte-identical — the headline property

    async def test_load_tools_result_carries_schemas(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        out = await mw._expand_callback("g", None)
        assert out.expanded is True
        assert out.schemas is not None
        assert out.schemas[0]["name"] == "t1"

    async def test_load_tools_idempotent_and_deterministic(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        first = await mw._expand_callback("g", None)
        second = await mw._expand_callback("g", None)
        # Compaction self-rescue: repeat calls serialize byte-identically.
        assert json.dumps(first.schemas, sort_keys=True) == json.dumps(
            second.schemas, sort_keys=True
        )

    async def test_loaded_tools_enter_context_hidden(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        await mw._expand(group_id="g", tool_names=None, context=ctx)
        loaded = next(t for t in ctx.tools if t.name == "t1")
        assert loaded.expose_to_model is False

    async def test_resolver_dispatches_with_implicit_load(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        call = ToolCall(
            id="tc-1",
            name="deferred_tool_call",
            arguments={"tool_name": "t1", "arguments": {"value": "hi"}},
        )
        rewritten = await mw.resolve_tool_call(call, context=ctx)
        assert rewritten is not None
        assert rewritten.id == "tc-1"
        assert rewritten.name == "t1"
        assert rewritten.arguments == {"value": "hi"}
        # Implicit load appended the hidden tool so the pipeline can find it.
        assert any(t.name == "t1" and not t.expose_to_model for t in ctx.tools)

    async def test_resolver_ignores_other_tools(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        call = ToolCall(id="tc-2", name="load_tools", arguments={"group_id": "g"})
        assert await mw.resolve_tool_call(call, context=ctx) is None

    async def test_resolver_unknown_tool_returns_none(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        call = ToolCall(
            id="tc-3",
            name="deferred_tool_call",
            arguments={"tool_name": "nope", "arguments": {}},
        )
        assert await mw.resolve_tool_call(call, context=ctx) is None

    async def test_dispatcher_execute_is_error_fallback(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        dispatcher = next(t for t in mw.tools if t.name == "deferred_tool_call")
        args = dispatcher.parameters.model_validate(
            {"tool_name": "nope", "arguments": {}}
        )
        result = await dispatcher.execute("tc-4", args)
        assert result.is_error is True
        assert "t1" in result.content[0].text  # lists valid names

    async def test_direct_native_call_to_hidden_tool_executes(self) -> None:
        """If the model hallucinates a direct tool_use with the real name,
        the engine resolves it from context.tools despite expose_to_model=False."""
        from cubepi.agent.tools import execute_tool_calls
        from cubepi.providers.faux import faux_assistant_message

        mw = _mw([_make_group("g", ["t1"])])
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        await mw._expand(group_id="g", tool_names=None, context=ctx)
        call = ToolCall(id="tc-direct", name="t1", arguments={"value": "hi"})
        batch = await execute_tool_calls(
            ctx,
            faux_assistant_message(call, stop_reason="tool_use"),
            emit=lambda e: None,
        )
        assert batch.messages[0].content[0].text == "echo:hi"

    async def test_inject_strategy_resolver_is_noop(self) -> None:
        mw = _mw([_make_group("g", ["t1"])], strategy="inject")
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        call = ToolCall(
            id="tc-5",
            name="deferred_tool_call",
            arguments={"tool_name": "t1", "arguments": {"value": "x"}},
        )
        assert await mw.resolve_tool_call(call, context=ctx) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/deferred/test_dispatch.py -v`
Expected: FAIL — `TypeError: DeferredToolsMiddleware.__init__() got an unexpected keyword argument 'strategy'`

- [ ] **Step 3: Create `cubepi/deferred/_dispatch_tool.py`**

```python
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
            "Invoke a deferred tool by name. Use schemas from load_tools "
            "results to shape `arguments`. Tools load on demand if needed."
        ),
        parameters=DeferredToolCallInput,
        execute=_execute,
    )
```

- [ ] **Step 4: Rework `cubepi/deferred/middleware.py`**

Changes (full method bodies below; everything not mentioned stays as on main):

4a. Imports: add `from dataclasses import dataclass, replace`,
`from cubepi.deferred._catalog import render_static_catalog, DEFAULT_DISPATCH_CATALOG_HEADER`,
`from cubepi.deferred._dispatch_tool import DISPATCH_TOOL_NAME, _make_deferred_tool_call`,
`from cubepi.deferred.types import DeferredStrategy, DeferredToolGroup`,
`from cubepi.providers.base import ToolCall`.

4b. Constructor — add `strategy`, build the tool list per strategy, precompute a
name→group index for the resolver:

```python
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

        self._loader_cache: dict[str, list[AgentTool]] = (
            dict(resumed_loader_cache) if resumed_loader_cache else {}
        )
        self._pending_injection: list[AgentTool] = []
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
```

(Note: `resumed_schemas` and `self._expanded_schemas` are gone — Task 6/7 remove their
remaining uses; do it in one edit here and fix fallout in those tasks' tests.)

4c. `_expand_callback` — same load/lock/state logic as v1, but: tools staged for injection
are marked hidden in dispatch mode, the schema bookkeeping is dropped, and the dispatch
result carries schemas:

```python
        # (after computing newly_expanded / expanded_names / remaining as today)
        staged = newly_expanded
        if self._strategy == "dispatch":
            staged = [replace(t, expose_to_model=False) for t in newly_expanded]
        self._pending_injection.extend(staged)
        if staged and self._on_tools_expanded:
            self._on_tools_expanded(staged)

        schemas: list[dict[str, object]] | None = None
        if self._strategy == "dispatch":
            # Full set for the request — idempotent across repeat calls.
            schemas = [t.to_definition().model_dump() for t in requested]

        return LoadToolsOutput(
            group_id=group_id,
            expanded=True,
            tool_names=expanded_names,
            remaining=max(remaining, 0),
            schemas=schemas,
        )
```

(Idempotency: `requested` is computed from the loader cache before the `already_set` filter,
so repeat calls return the same `schemas` even when `newly_expanded` is empty.)

4d. New `resolve_tool_call` + `_ensure_loaded`:

```python
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
        fires for a rewritten dispatcher call on the dispatcher's behalf)."""
        await self._expand_callback(group_id, tool_names)
        if context.tools is not None:
            self._drain_pending(context.tools)
```

4e. `transform_system_prompt` — strategy split:

```python
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
        expanded: dict[str, list[str] | None] = extra.get("expanded_groups", {})
        catalog = render_catalog(
            groups=list(self._groups.values()),
            expanded=expanded,
            header=self._catalog_header,
        )
        return f"{system_prompt}\n\n{catalog}" if catalog else system_prompt
```

4f. Export: `cubepi/deferred/__init__.py` adds `DeferredStrategy` to imports/`__all__`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/deferred/test_dispatch.py -v`
Expected: PASS.

Run: `uv run pytest tests/deferred/ -q`
Expected: failures ONLY in tests that assert the removed schema section /
`resumed_schemas` / `_expanded_schemas` (fixed in Tasks 6-7). List them; nothing else.

- [ ] **Step 6: Commit**

```bash
git add cubepi/deferred/ tests/deferred/test_dispatch.py
git commit -m "feat(deferred): dispatch strategy — static prompt, dispatcher with implicit load"
```

---

### Task 6: Inject slimming — delete the system-prompt schema section

**Files:**
- Modify: `cubepi/deferred/_catalog.py` (delete `render_expanded_schemas`)
- Modify: `cubepi/deferred/middleware.py` (already done structurally in Task 5 — this task fixes the remaining references and tests)
- Test: `tests/deferred/test_catalog.py`, `tests/deferred/test_middleware.py`

- [ ] **Step 1: Delete `render_expanded_schemas` and the `ToolSchema` alias's remaining uses**

Remove the function from `_catalog.py`. Grep for stragglers:

Run: `grep -rn "render_expanded_schemas\|_expanded_schemas\|ToolSchema" cubepi/ tests/`
Fix every hit in `cubepi/` (there must be none left after Task 5; this is a verification
gate). Keep `ToolSchema` only if `LoadToolsOutput.schemas` typing reuses it; otherwise delete.

- [ ] **Step 2: Update inject-mode tests**

In `tests/deferred/test_catalog.py`: delete tests of `render_expanded_schemas`.
In `tests/deferred/test_middleware.py`: existing inject tests constructing the middleware
must pass `strategy="inject"` (v1 behavior is now opt-in); assertions that the system prompt
contains "# Expanded tool groups" or parameter JSON flip to asserting absence:

```python
async def test_inject_system_prompt_has_no_schema_section(self) -> None:
    # (inside the existing middleware-test class, using its helpers)
    mw = DeferredToolsMiddleware(
        groups=[_make_group("g", ["t1"])],
        extra_ref=lambda: extra,
        strategy="inject",
    )
    ctx = AgentContext(system_prompt="base", messages=[], tools=list(mw.tools))
    await mw._expand(group_id="g", tool_names=None, context=ctx)
    prompt = await mw.transform_system_prompt("base", ctx=ctx)
    assert "Expanded tool groups" not in prompt
    assert '"parameters"' not in prompt
    # Injection itself still works — schemas live in the tools array.
    assert any(t.name == "t1" and t.expose_to_model for t in ctx.tools)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/deferred/ -q`
Expected: only resume-related failures remain (Task 7).

- [ ] **Step 4: Commit**

```bash
git add cubepi/deferred/_catalog.py tests/deferred/
git commit -m "feat(deferred)!: drop system-prompt schema rendering in inject mode"
```

---

### Task 7: Resume simplification

**Files:**
- Modify: `cubepi/deferred/middleware.py` (`ResumedState`, `prepare_resumed_state`)
- Test: `tests/deferred/test_middleware.py`, `tests/deferred/test_dispatch.py`

- [ ] **Step 1: Write/adjust the tests**

Append to `tests/deferred/test_dispatch.py`:

```python
class TestDispatchResume:
    async def test_resume_restores_loader_cache_and_hidden_tools(self) -> None:
        group = _make_group("g", ["t1", "t2"])
        state = await DeferredToolsMiddleware.prepare_resumed_state(
            [group], {"g": ["t1"]}, strategy="dispatch"
        )
        assert [t.name for t in state.pre_loaded_tools] == ["t1"]
        assert all(not t.expose_to_model for t in state.pre_loaded_tools)
        assert "g" in state.loader_cache
        # Partially expanded groups remain deferrable.
        assert [g.group_id for g in state.remaining_groups] == ["g"]

        extra: dict = {"expanded_groups": {"g": ["t1"]}}
        mw = DeferredToolsMiddleware(
            groups=state.remaining_groups,
            extra_ref=lambda: extra,
            strategy="dispatch",
            resumed_loader_cache=state.loader_cache,
        )
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=[*mw.tools, *state.pre_loaded_tools],
        )
        call = ToolCall(
            id="tc-r1",
            name="deferred_tool_call",
            arguments={"tool_name": "t1", "arguments": {"value": "hi"}},
        )
        rewritten = await mw.resolve_tool_call(call, context=ctx)
        assert rewritten is not None and rewritten.name == "t1"

    async def test_resume_inject_marks_tools_visible(self) -> None:
        group = _make_group("g", ["t1"])
        state = await DeferredToolsMiddleware.prepare_resumed_state(
            [group], {"g": None}, strategy="inject"
        )
        assert all(t.expose_to_model for t in state.pre_loaded_tools)
```

Update `tests/deferred/test_middleware.py` resume tests: drop every reference to
`ResumedState.expanded_schemas` and the `resumed_schemas` constructor argument; add the new
required `strategy` argument where those tests construct resumed middleware.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/deferred/test_dispatch.py::TestDispatchResume -v`
Expected: FAIL — `prepare_resumed_state() got an unexpected keyword argument 'strategy'`.

- [ ] **Step 3: Implement**

In `cubepi/deferred/middleware.py`:

```python
@dataclass
class ResumedState:
    """Pre-loaded tools and remaining groups for cross-run replay."""

    pre_loaded_tools: list[AgentTool]
    remaining_groups: list[DeferredToolGroup]
    loader_cache: dict[str, list[AgentTool]]
```

`prepare_resumed_state` **stays a `@staticmethod` on `DeferredToolsMiddleware`** (as in v1 —
tests call `DeferredToolsMiddleware.prepare_resumed_state`):

```python
    @staticmethod
    async def prepare_resumed_state(
        groups: list[DeferredToolGroup],
        expanded: dict[str, list[str] | None],
        *,
        strategy: DeferredStrategy,  # REQUIRED — must match the middleware's
        # strategy. A default would let an inject-mode host silently resume
        # with hidden tools (invisible to the model, no dispatcher to reach
        # them). Forcing the kwarg makes the mismatch impossible.
    ) -> ResumedState:
        """Replay expansion state from a previous run.

        Dispatch mode: schemas live in message history (the checkpointer
        brings them back) — only the loader cache and hidden tool objects
        need rebuilding. Inject mode: tools come back model-visible.
        """
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

        hidden = strategy == "dispatch"
        pre_loaded: list[AgentTool] = []
        cache: dict[str, list[AgentTool]] = {}
        for (group, exp), loaded in zip(to_load, loaded_results):
            cache[group.group_id] = loaded
            if exp is None:
                selected = loaded
            else:
                name_set = set(exp)
                selected = [t for t in loaded if t.name in name_set]
                remaining.append(group)
            pre_loaded.extend(
                replace(t, expose_to_model=False) if hidden else t
                for t in selected
            )

        return ResumedState(
            pre_loaded_tools=pre_loaded,
            remaining_groups=remaining,
            loader_cache=cache,
        )
```

(Note: in dispatch mode a fully expanded group should arguably stay in `remaining` too so
`load_tools` can re-serve schemas after compaction — `_expand_callback` already serves any
group in `self._groups`, and `prepare_resumed_state`'s `remaining` feeds the constructor's
`groups`. So in dispatch mode, append `group` to `remaining` for **both** branches:
`if exp is None: remaining.append(group) if hidden else None`. Implement as:

```python
            if exp is None:
                selected = loaded
                if hidden:
                    remaining.append(group)
            else:
                ...
```

This keeps fully-expanded groups re-loadable in dispatch mode while preserving v1 inject
semantics, and it keeps the static catalog complete — which is required anyway, because the
dispatch catalog is expansion-independent.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/deferred/ -q`
Expected: PASS — all deferred tests green.

- [ ] **Step 5: Commit**

```bash
git add cubepi/deferred/middleware.py tests/deferred/
git commit -m "feat(deferred)!: simplify resume — drop expanded-schema replay"
```

---

### Task 8: Agent wiring — `deferred_tool_strategy`

**Files:**
- Modify: `cubepi/agent/agent.py:192-224`
- Test: `tests/deferred/test_agent_wiring.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/deferred/test_agent_wiring.py`:

```python
class TestDeferredStrategyParam:
    def test_default_strategy_is_dispatch(self) -> None:
        model = _make_faux_model()
        agent = Agent(
            model=model,
            deferred_tool_groups=[_make_group("g", ["t1"])],
        )
        names = [t.name for t in agent._state.tools]
        assert "deferred_tool_call" in names

    def test_inject_strategy_opt_in(self) -> None:
        model = _make_faux_model()
        agent = Agent(
            model=model,
            deferred_tool_groups=[_make_group("g", ["t1"])],
            deferred_tool_strategy="inject",
        )
        names = [t.name for t in agent._state.tools]
        assert "load_tools" in names
        assert "deferred_tool_call" not in names

    def test_resolve_hook_composed(self) -> None:
        model = _make_faux_model()
        agent = Agent(
            model=model,
            deferred_tool_groups=[_make_group("g", ["t1"])],
        )
        assert agent.resolve_tool_call is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/deferred/test_agent_wiring.py -v`
Expected: FAIL — `TypeError: Agent.__init__() got an unexpected keyword argument 'deferred_tool_strategy'`

- [ ] **Step 3: Implement**

In `cubepi/agent/agent.py`: add the parameter next to `deferred_tool_groups` (:192):

```python
        deferred_tool_groups: list[DeferredToolGroup] | None = None,
        deferred_tool_strategy: DeferredStrategy = "dispatch",
```

(Type import under the existing `if TYPE_CHECKING` block where `DeferredToolGroup` is
imported at :54; runtime default is a plain string literal so no runtime import is needed.)

Pass it through at :217:

```python
            deferred_mw = DeferredToolsMiddleware(
                groups=deferred_tool_groups,
                extra_ref=lambda: self._extra,
                strategy=deferred_tool_strategy,
                on_tools_expanded=lambda new: self._state._tools.extend(
                    t for t in new if t.name not in {e.name for e in self._state._tools}
                ),
            )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/deferred/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cubepi/agent/agent.py tests/deferred/test_agent_wiring.py
git commit -m "feat(agent)!: deferred_tool_strategy param, default dispatch"
```

---

### Task 9: End-to-end byte-stability test

**Files:**
- Test: `tests/deferred/test_dispatch.py`

- [ ] **Step 1: Write the test**

A capturing FauxProvider subclass records every request's `(system_prompt, tools)`; a
scripted three-turn run (load → dispatch → done) must keep them byte-identical:

```python
from cubepi.providers.base import (
    Message,
    MessageStream,
    Model,
    StreamOptions,
    ToolChoice,
    ToolDefinition,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_tool_call


class _CapturingFaux(FauxProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured: list[tuple[str, list[dict] | None]] = []

    async def stream(
        self,
        model,
        messages,
        *,
        system_prompt="",
        tools=None,
        tool_choice=None,
        options=None,
    ):
        self.captured.append(
            (
                system_prompt,
                [t.model_dump() for t in tools] if tools else None,
            )
        )
        return await super().stream(
            model,
            messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            options=options,
        )


class TestByteStability:
    async def test_prefix_static_across_load_and_dispatch(self) -> None:
        from cubepi.agent.agent import Agent

        provider = _CapturingFaux()
        provider.set_responses(
            [
                faux_assistant_message(
                    faux_tool_call("load_tools", {"group_id": "g"}, id="tc-1"),
                    stop_reason="tool_use",
                ),
                faux_assistant_message(
                    faux_tool_call(
                        "deferred_tool_call",
                        {"tool_name": "t1", "arguments": {"value": "hi"}},
                        id="tc-2",
                    ),
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )
        agent = Agent(
            model=provider.model("faux"),
            system_prompt="base prompt",
            deferred_tool_groups=[_make_group("g", ["t1", "t2"])],
        )
        await agent.prompt("go")

        assert len(provider.captured) == 3
        systems = {c[0] for c in provider.captured}
        assert len(systems) == 1  # system prompt byte-identical every turn
        import json as _json

        tool_payloads = {
            _json.dumps(c[1], sort_keys=True) for c in provider.captured
        }
        assert len(tool_payloads) == 1  # tools param byte-identical every turn
        # And the dispatched tool actually ran:
        result_texts = [
            m.content[0].text
            for m in agent._state.messages
            if getattr(m, "tool_call_id", None) == "tc-2"
        ]
        assert result_texts == ["echo:hi"]
```

(If `agent.prompt` has a different awaitable shape — e.g. returns an async iterator — adapt
the invocation to the idiom used in `tests/agent/test_agent.py`; the assertions stand.)

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/deferred/test_dispatch.py::TestByteStability -v`
Expected: PASS. If the system prompts differ, diff the two strings in the failure output —
any drift is a real cache-invalidation bug in the dispatch path.

- [ ] **Step 3: Commit**

```bash
git add tests/deferred/test_dispatch.py
git commit -m "test(deferred): end-to-end byte-stability for dispatch strategy"
```

---

### Task 10: Docs

**Files:**
- Modify: `website/docs/guides/middleware/deferred-tools.md`

- [ ] **Step 1: Update the guide**

Read the existing page first; restructure to cover (keeping its frontmatter and tone):

1. Strategy choice up front: `dispatch` (default, zero cache invalidation) vs `inject`
   (native tool calling, per-expansion cache cost). Include the spec's cache-behavior table.
2. Dispatch walkthrough: catalog → `load_tools` returns schemas → `deferred_tool_call`;
   note implicit load and that hooks/tracing see real tool names.
3. `Agent(deferred_tool_groups=[...], deferred_tool_strategy=...)` and the middleware-level
   API with `strategy=`.
4. Migration note from 0.10: behavior change + `deferred_tool_strategy="inject"` opt-out;
   `resumed_schemas` removal; inject mode no longer renders schemas into the system prompt.
5. CubePi capitalization in prose ("CubePi"), lowercase only in code.

- [ ] **Step 2: Build check (if docs site builds locally)**

Run: `ls website/package.json && echo has-site`
If the site builds in CI only, skip local build; otherwise follow the repo's docs build
command from `website/README.md` if present.

- [ ] **Step 3: Commit**

```bash
git add website/docs/guides/middleware/deferred-tools.md
git commit -m "docs(deferred): document dispatch vs inject strategies"
```

---

### Task 11: CHANGELOG + full verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: CHANGELOG entry**

Add under an Unreleased/next-version heading (match the file's existing format):

```markdown
### Breaking

- **Deferred tool groups default to the new `dispatch` strategy.** Tool schemas are
  delivered through `load_tools` results and invoked via `deferred_tool_call`; the tools
  array and system prompt stay byte-stable, so expansions no longer invalidate the prompt
  cache. Restore the v0.10 behavior with
  `Agent(deferred_tool_strategy="inject")` / `DeferredToolsMiddleware(strategy="inject")`.
- `DeferredToolsMiddleware(resumed_schemas=...)` and `ResumedState.expanded_schemas` are
  removed; `prepare_resumed_state` takes a `strategy` keyword.
- Inject mode no longer renders expanded schemas into the system prompt (they were already
  in the tools array; this removes double billing).

### Added

- `AgentTool.expose_to_model` — engine-resolvable tools hidden from the provider payload.
- `resolve_tool_call` middleware hook — rewrite tool calls before validation/hooks/tracing.
```

- [ ] **Step 2: Full verification gates**

```bash
uv run pytest tests/ -q
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/
uv run mypy cubepi
```

Expected: all green. Capture output to a file before claiming success
(`uv run pytest tests/ -q 2>&1 | tail -20`).

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for deferred dispatch strategy"
```

---

## Self-review checklist (run after writing, before code review)

- Spec coverage: every spec section maps to a task — strategy param (T5/T8), static surfaces
  (T4/T5), resolver+implicit load (T5), unwrap (T2), schema-on-error (T3), expose_to_model
  (T1), inject slimming (T6), resume (T7), byte-stability test (T9), docs (T10),
  breaking-change notes (T11).
- Spec items intentionally deferred: `"native"` strategy (future spec), compaction preserve
  rules (non-goal).
- Naming consistency: `deferred_tool_call` / `DISPATCH_TOOL_NAME`, `expose_to_model`,
  `resolve_tool_call`, `deferred_tool_strategy`, `DeferredStrategy` — used identically across
  tasks.
