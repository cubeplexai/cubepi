# Deferred Tools v2 — Dispatch Strategy (Zero Cache Invalidation)

Date: 2026-06-10
Branch: `2026-06-10-deferred-dispatcher`
Supersedes (partially): `2026-06-09-deferred-tool-group.md` (v1, shipped in v0.10.0)

## Problem

Deferred tool groups v1 (inject strategy) solves context bloat but pays a heavy prompt-cache
price. On every `load_tools` expansion:

1. **Tools-array append.** Expanded tools are injected into `ctx.context.tools` and serialized
   into the provider `tools` parameter on the next iteration. Tools render at position 0 of the
   prompt — appending inserts bytes *before* system + messages, so the entire conversation
   history is re-read uncached. One full-cost re-read of system + history per expansion event.
2. **System-prompt mutation.** The catalog entry is rewritten (remaining counts) and an
   "Expanded tool groups" section is appended — a second, independent invalidator.
3. **Double billing.** Expanded schemas are rendered in *both* the tools array and the system
   prompt ("supplementary context" per the v1 spec). Every expanded tool's parameter JSON is
   billed twice per turn, at cache-read rate, forever.

(1) is inherent to client-side injection on every prefix-cached provider (Anthropic, OpenAI):
any tool that must appear in the native `tools` parameter sits in the cached prefix. The only
way to reach zero invalidation provider-agnostically is to never mutate the tools parameter or
the system prompt after the first request.

## Goals

- A **dispatch strategy** with *zero* prompt-cache invalidation: static tools array, static
  system prompt, schemas delivered through tool results (append-only message suffix).
- `strategy` parameter on `DeferredToolsMiddleware` and `Agent(deferred_tool_strategy=...)`;
  **default `"dispatch"`**.
- **Hook transparency**: middleware hooks, HITL/permissions, and tracing see the *real* tool
  name and validated args for dispatched calls — never the dispatcher envelope.
- **Implicit load**: dispatching a not-yet-loaded tool loads it on the fly and validates;
  a validation failure returns the full schema so the model can self-correct in one round trip.
- Keep the inject strategy for hosts that prioritize native tool-calling quality, and fix its
  double-billing (system-prompt schema section removed).

## Non-goals

- **Anthropic native Tool Search integration** (`defer_loading` + `tool_reference` expansion).
  Recorded as the future `"native"` strategy below; out of scope here because it requires all
  deferred schemas to be available at request time (conflicts with lazy `loader()`) and
  provider-capability plumbing.
- Compaction preserve rules for schema-bearing tool results. The idempotent-`load_tools`
  self-rescue path (below) is the v2 mitigation; transcript-preservation rules are a possible
  later enhancement.
- BM25 / embedding retrieval over the catalog. The catalog stays small enough to read.

## Prior art

Researched 2026-06-10 (Anthropic docs, pi-agent-core, langgraph, Claude Code):

- **Anthropic native Tool Search Tool** (GA 2026-02): tools carry `defer_loading: true` but
  their *full schemas still ship in the `tools` request param every turn* — the API merely
  withholds them from the rendered prompt. Discovery returns `tool_reference` markers that the
  **server expands inline into message history**, re-applied across the whole conversation on
  every request; invocation stays a native, strict-mode-validated `tool_use`. Documented as
  cache-preserving. A client tool may return `tool_reference` blocks in a normal `tool_result`
  and get the same server-side expansion — this escape hatch is what makes the future `"native"`
  strategy cheap to add.
- **Claude Code ToolSearch**: deferred tools announced name-only in a system reminder; the
  ToolSearch *tool result* carries full JSONSchema definitions; calling an unloaded tool fails
  fast. Rides the native `tool_reference` expansion under the hood; permissions see real tool
  calls.
- **langgraph `bigtool`**: RAG-over-tools; retrieved tool ids accumulate in graph state and the
  tools array is **rebound per request** — cache-hostile, undocumented as such. One idea worth
  stealing: "which tools are loaded" persists as durable agent state, not as an artifact of
  surviving messages.
- **pi-agent-core**: no deferral at all; tools change by reassigning `agent.state.tools`
  (rebind pattern). Hooks always see first-class tool calls.
- **hermes-agent** (v1 spec prior art; `tools/tool_search.py`): the direct precedent for the
  dispatch strategy — three bridge tools (`search`/`describe`/`call`) where `call` is exactly a
  generic dispatcher. Per-tool granularity, BM25 retrieval, stateless catalog rebuilt every
  turn. Its production lessons map onto this design: (1) core tools must never defer → cubepi
  only defers what hosts explicitly group; (2) catalog drift silently drops tools (OpenClaw
  #84141) → the dispatch-mode catalog is *static*, immune to drift by construction; (3)
  transparent unwrap so hooks see real tool names → the `resolve_tool_call` engine hook below.

**Divergence callout** — the surveyed systems split into two camps. Anthropic native, Claude
Code, langgraph, and pi-agent-core all keep invocation as a native, provider-validated
`tool_use` (paying for it with either a server-side Anthropic-only feature or cache-hostile
rebinding). hermes-agent — and now cubepi — instead route deferred calls through a generic
dispatcher, trading provider-side arg constraint for provider-agnostic zero invalidation.
cubepi differs from hermes in granularity (group-level `load_tools` vs per-tool
`search`/`describe`) and in keeping the catalog static rather than rebuilt per turn. The known
costs of dispatching and their mitigations:

| Cost of dispatching | Mitigation |
|---|---|
| No provider-side arg constraint/strict mode for inner args | Framework-side pydantic validation (same `model_validate` path as native calls); validation failure returns the full schema |
| Hooks/permissions/tracing would see the envelope | Engine-level unwrap *before* the hook layer (below) |
| Schemas in message history can be compacted away | `load_tools` is idempotent and re-callable; loaded-state is durable in `extra` (bigtool lesson) |
| Slightly degraded calling ergonomics vs native | Acknowledged trade-off; hosts that care choose `strategy="inject"` |

## Settled decisions

1. **Two strategies, default `dispatch`.** Zero cache invalidation is the right default for the
   large-toolset use case this feature exists for. Hosts that need native calling quality opt
   into `inject`.
2. **Engine-level unwrap to real names.** All observability and control surfaces operate on the
   real tool call (hermes-agent lesson #3; consistent with every surveyed system).
3. **Static system prompt in dispatch mode.** Catalog lists groups + tool names only — no
   remaining counts, no schema section. Load state is communicated through `load_tools` results.
4. **Implicit load + validate.** A dispatched call to an unloaded tool auto-loads its group
   (loader cache + per-group lock reused from v1), then validates. The model can skip the
   explicit `load_tools` round trip when it can already produce correct args from catalog names.
5. **Compaction self-rescue via idempotent `load_tools`.** Re-calling returns the same schemas.
6. **Inject mode slimmed.** The "Expanded tool groups" system-prompt section is removed
   entirely — full schemas already live in the tools array. The catalog (with remaining counts)
   stays, since inject mode's per-expansion invalidation is unavoidable anyway and the counts
   carry signal.

## Design

### API surface

```python
DeferredStrategy = Literal["dispatch", "inject"]   # "native" reserved for future

class DeferredToolsMiddleware(Middleware):
    def __init__(
        self,
        *,
        groups: list[DeferredToolGroup],
        extra_ref: Callable[[], dict[str, Any]],
        strategy: DeferredStrategy = "dispatch",
        catalog_header: str | None = None,   # default depends on strategy
        ...,
    ) -> None: ...

# Agent-level convenience (mirrors v1):
Agent(deferred_tool_groups=[...], deferred_tool_strategy="dispatch")
```

`DeferredToolGroup` is unchanged.

### Dispatch strategy mechanics

**Static surfaces (never mutated after construction):**

- `tools` (model-visible): regular agent tools + `load_tools` + `deferred_tool_call`. The
  middleware's `tools` attribute is `[load_tools, deferred_tool_call]`.
- System prompt: catalog header + one line per group (`group_id`, display name, description,
  full tool-name list, sorted by `group_id`). Rendered once; byte-identical every turn.

**`load_tools(group_id, tool_names=None)`** — same input schema as v1. The tool result now
carries the schemas instead of the middleware mutating prompt state:

```json
{
  "group_id": "mcp:github",
  "loaded": ["create_issue", "list_issues"],
  "schemas": [
    {"name": "create_issue", "description": "...", "parameters": {...}},
    {"name": "list_issues", "description": "...", "parameters": {...}}
  ],
  "usage": "Call these via deferred_tool_call(tool_name=..., arguments=...)."
}
```

- `parameters` serialized with `sort_keys=True` (determinism — the result becomes part of the
  cached prefix on subsequent turns).
- Idempotent: repeat calls return the same payload (compaction self-rescue). Loader still runs
  exactly once per group per run (v1 loader cache + per-group `asyncio.Lock` reused).

**`deferred_tool_call(tool_name: str, arguments: dict)`** — the dispatcher. Its `execute` never
runs; calls are rewritten by the engine before the standard pipeline (next section).

**Internal tool registry.** Loaded `AgentTool`s are appended to `context.tools` with a new
`AgentTool.expose_to_model: bool = False` field. `loop.py` filters them out of the provider
payload:

```python
# loop.py — the only loop change
tools_defs = [t.to_definition() for t in context.tools if t.expose_to_model]
```

Keeping loaded tools in `context.tools` (rather than a middleware-private dict) means the
existing `_prepare_tool_call` name lookup, fork snapshotting, and subagent context inheritance
all work unchanged. Bonus robustness: if the model hallucinates a *direct* native call to a
loaded tool's real name, the engine resolves it anyway.

### Engine-level unwrap: `resolve_tool_call`

A new optional hook threaded through `execute_tool_calls`, symmetric with
`before_tool_call`/`after_tool_call`:

```python
ResolveToolCall = Callable[[ToolCall, AgentContext], Awaitable[ToolCall | None]]
```

Called at the top of `_prepare_tool_call`. Returning a `ToolCall` (same `id`, rewritten
`name`/`arguments`) replaces the original for the *entire* downstream pipeline — pydantic
validation, `before_tool_call`, execution, `after_tool_call`, `ToolExecutionUpdateEvent`,
tracing all see the real name. The tool result is keyed by the unchanged `tool_call.id`, so the
wire protocol (assistant `tool_use` named `deferred_tool_call` ↔ `tool_result` by id) stays
consistent for the provider.

The middleware's resolver:

1. `tool_call.name != "deferred_tool_call"` → return `None` (no-op for everything else).
2. Parse `{tool_name, arguments}`. Unknown `tool_name` (not in any group) → leave the call
   unresolved with a structured error result listing valid names.
3. **Implicit load**: if `tool_name`'s group isn't loaded, run the loader (cache + lock), append
   the group's tools to `context.tools` (`expose_to_model=False`), record load state in
   `extra["expanded_groups"]`.
4. Return `ToolCall(id=同, name=tool_name, arguments=arguments)`.

**Validation failure includes the schema.** When a dispatch-resolved call fails
`model_validate`, the error result appends the tool's full schema JSON, so the model retries
with correct args in one round trip. (Mechanism: the resolver marks the rewritten call;
`_format_validation_error` gains an optional schema suffix. Exact plumbing settled in the plan.)

### State persistence & resume

`extra["expanded_groups"]` keeps the v1 shape (`dict[group_id, list[str] | None]`) — durable
load state outside the transcript. Dispatch-mode resume becomes trivial:

- Schemas need no rebuilding — `load_tools` results live in message history and come back via
  the checkpointer.
- `prepare_resumed_state` only restores the loader cache and re-appends loaded tools
  (`expose_to_model=False`) so dispatched calls resolve immediately. No ordering invariant to
  preserve (nothing renders into the prefix), which retires v1's resume order-fidelity caveat.

### Inject strategy changes (v1 fix)

- `transform_system_prompt` renders the **catalog only**. The "Expanded tool groups" schema
  section, `render_expanded_schemas`, and the `_expanded_schemas` append-only bookkeeping are
  deleted — schemas exist solely in the tools array.
- Everything else (injection via `after_tool_call`, expansion-order tool append, loader cache,
  locks) is unchanged.

### Cache behavior comparison

| | tools param | system prompt | per-expansion cache cost | calling path |
|---|---|---|---|---|
| inject (v1 → v2) | grows per expansion | catalog counts change (v2: schema section removed) | full re-read of system + history | native `tool_use` |
| **dispatch (v2 default)** | **static** | **static** | **zero** (schemas are message-suffix appends) | dispatcher, engine-unwrapped |
| native (future) | static bytes (all schemas always present, `defer_loading: true`) | static | zero (server-side `tool_reference` expansion) | native `tool_use` |

## Breaking changes (v0.10 → next minor)

Per project policy: break loudly, no compatibility shims.

1. **Default strategy is `dispatch`.** Hosts upgrading from v0.10 get a different catalog
   wording, a new `deferred_tool_call` builtin, and no tools-array growth. Restoring v1
   behavior is `deferred_tool_strategy="inject"`.
2. **Inject mode system prompt slims** — the expanded-schema section disappears (pure token
   saving; behavior-neutral for the model since schemas remain in the tools array).
3. `DeferredToolsMiddleware` constructor: `resumed_schemas` parameter removed (dispatch resume
   doesn't need it; inject resume rebuilds nothing now); `strategy` added.

CHANGELOG must call out (1) explicitly with the one-line opt-out.

## Testing

Unit (FauxProvider, no API calls):

1. **Byte-stability** (the headline property): render two consecutive request payloads across a
   `load_tools` + `deferred_tool_call` sequence in dispatch mode; assert `tools` and `system`
   are byte-identical, and message history is append-only.
2. Resolver unwrap: hooks (`before_tool_call`/`after_tool_call`) and emitted events receive the
   real tool name and validated args; result keyed to the dispatcher's `tool_use_id`.
3. Implicit load: dispatching an unloaded tool loads exactly once (concurrent dispatches to the
   same group race-safe via the per-group lock), then executes.
4. Validation failure returns `is_error=True` with the full schema in the result.
5. Unknown `tool_name` → structured error listing valid names; no load attempted.
6. `load_tools` idempotency: repeat call returns identical schemas payload.
7. Direct native call to a loaded `expose_to_model=False` tool resolves and executes.
8. Inject mode: system prompt contains catalog only; schemas absent; injection still works.
9. Resume (dispatch): loader cache + loaded set restored; dispatched call works without a
   fresh `load_tools`.
10. `expose_to_model` filtering: provider payload excludes hidden tools in both strategies.

## Open questions

- **Dispatcher tool name**: `deferred_tool_call` vs `call_tool`. Leaning `deferred_tool_call`
  (self-describing, low collision risk with host tools).
- Should `load_tools` in dispatch mode also accept `group_id=None` to dump all schemas of all
  groups? Leaning no (defeats progressive disclosure).
- Whether the `"native"` strategy lands as a third `strategy` value or as an
  `AnthropicProvider` capability that the middleware queries. Deferred to its own spec.
