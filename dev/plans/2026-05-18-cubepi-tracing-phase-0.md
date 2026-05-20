# CubePi Tracing — Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the two cubepi-internal extensions that the tracing module depends on, with no `cubepi/tracing/` code introduced yet. After Phase 0, the existing agent loop and providers behave identically for users who haven't subscribed any listener, and a third party can attach observers to capture wire payloads, per-chunk timing, assembled response body, and richer tool-execution context.

**Architecture:** Two independent, additive extensions to cubepi:

1. **Provider listener registry** (`cubepi/providers/`) — keeps the existing `Provider` `runtime_checkable` Protocol **unchanged** (preserves the documented duck-typing contract for user providers; see Task 2 rationale). Introduces a new concrete base class `BaseProvider` carrying a multi-subscriber listener registry. Adds three new persistent subscription methods (`subscribe_request`, `subscribe_chunk`, `subscribe_response`) and three new callback types. Existing per-call `StreamOptions.on_payload` / `on_response` slots retain their current semantics. Built-in providers (`anthropic.py`, `openai.py`, `openai_responses.py`, `faux.py`) inherit from `BaseProvider` and call a single `_fire_listeners` helper at three points in `stream()`: after the request payload is finalized, for each `StreamEvent` pushed onto the message stream, and exactly once in a `finally` block after the stream terminates (normal completion, exception, or `asyncio.CancelledError`). Duck-typed user providers continue to work; the tracer detects listener support via `isinstance(provider, BaseProvider)` and skips request/chunk/response capture for providers that don't opt in.

2. **`ToolExecutionEndEvent` extension** (`cubepi/agent/`) — adds three additive fields (`terminate`, `blocked_by_hook`, `block_reason`) to the event. The tool-execution layer (`cubepi/agent/tools.py`) already has all the information internally; it just needs to thread three more values through the existing `_ImmediateOutcome` / `_FinalizedOutcome` dataclasses and the three `ToolExecutionEndEvent` emission sites (sequential path, parallel path's immediate-outcome branch, and parallel path's async-execution branch). Backward compatible — defaults are conservative.

Both extensions are additive: no existing tests should change, no public API names change, no behavior changes for callers that don't opt in. The PRs can land in either order.

**Tech Stack:** Python 3.10+, `pydantic>=2`, `pytest`, `pytest-asyncio`. No new runtime dependencies.

**Spec:** `docs/specs/2026-05-18-cubepi-tracing-design.md` §3.4

---

## Phase 0a — Provider listener registry

### Task 1: Add new callback types and helper to `providers/base.py`

**Files:**
- Modify: `cubepi/providers/base.py`

**Goal:** Expose the three new callback type aliases and a single `_fire_listeners` helper that all providers will use. No behavior change yet.

- [ ] **Step 1: Add callback type aliases**

In `cubepi/providers/base.py`, after the existing `OnPayloadCallback` / `OnResponseCallback` declarations, add:

```python
OnRequestCallback = Callable[[dict, Model], Awaitable[None] | None]
"""Persistent observer. Fires just before HTTP send, after any per-call
StreamOptions.on_payload mutation has been applied. Receives the final wire
payload dict and the Model. Return value is ignored."""

OnChunkCallback = Callable[["StreamEvent", Model], Awaitable[None] | None]
"""Persistent observer. Fires for every StreamEvent pushed onto the stream
(start, text_delta, thinking_delta, toolcall_delta, done, error, ...).
Heavy listeners should early-return on irrelevant event types — this hook
fires hot. Return value is ignored."""

OnResponseBodyCallback = Callable[
    [dict | None, Model, BaseException | None],
    Awaitable[None] | None,
]
"""Persistent observer. Fires exactly once per stream() call, in a finally
block, after the stream terminates.
  - body: assembled provider response as a dict (same shape a non-streaming
    call to the provider would have returned), or None if the stream failed
    before a response could be assembled.
  - exc: the exception that ended the stream (including asyncio.CancelledError),
    or None on normal completion.
Return value is ignored."""
```

- [ ] **Step 2: Add `_fire_listeners` helper**

Below the new callback types, add a single async helper that invokes every listener in registration order and swallows their exceptions:

```python
async def _fire_listeners(
    listeners: "list[Callable]",
    *args: Any,
) -> None:
    """Invoke each listener with *args. Listener return values and exceptions
    are ignored — a buggy listener must never crash the stream. Exceptions
    are logged via the loguru logger if available, otherwise via stdlib
    logging.warning.

    Iterates a snapshot (``tuple(listeners)``) so a listener that detaches
    itself mid-iteration does not silently skip subsequent listeners.
    Caller is responsible for the hot-path guard — see Step 4 below."""
    if not listeners:
        return
    for cb in tuple(listeners):
        try:
            result = cb(*args)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            _log_listener_exception(cb, exc)


def _log_listener_exception(cb: Callable, exc: BaseException) -> None:
    try:
        from loguru import logger
        logger.opt(exception=exc).warning(
            "cubepi provider listener {} raised; swallowed", cb
        )
    except ImportError:
        import logging
        logging.getLogger("cubepi.providers").warning(
            "cubepi provider listener %r raised; swallowed", cb, exc_info=exc
        )
```

- [ ] **Step 3: Run `uv run pytest tests/` to confirm no regression**

No tests should fail or change. This task is purely additive.

- [ ] **Step 4: (Documentation) Note the per-site hot-path guard pattern**

Every provider call site that invokes `_fire_listeners` for chunks (which can fire hundreds of times per stream) MUST guard with a synchronous `if` to avoid `await` overhead when no listeners are registered:

```python
# at every chunk emission site:
if self._chunk_listeners:
    await _fire_listeners(self._chunk_listeners, event, model)
```

For the request and response sites (one call each per stream), the guard is optional but recommended for symmetry. Tasks 3–5 use this guard at every call site.

---

### Task 2: Introduce `BaseProvider` alongside the existing `Provider` Protocol

**Files:**
- Modify: `cubepi/providers/base.py`
- Modify: `cubepi/providers/__init__.py`
- Modify: `cubepi/__init__.py`

**Goal:** Preserve backward compatibility for duck-typed user providers (`tests/agent/test_loop.py:512` `_NoFinalEventProvider` and `:567` `_NoFinalEventNoPartialProvider` are existing duck-typed providers; `website/docs/guides/providers/custom.md` documents the duck-typing contract publicly). Adding `subscribe_*` methods to the existing `Protocol` would silently break any caller relying on the documented "any class with a `stream()` method works" promise — `isinstance(x, Provider)` would return False for them after the change.

Strategy: leave `Provider` as the unchanged `runtime_checkable` Protocol; introduce a **new** concrete base class `BaseProvider` that built-in providers (`AnthropicProvider`, `OpenAIProvider`, `OpenAIResponsesProvider`, `FauxProvider`) inherit from. The Protocol covers everyone; the base class adds the listener registry for built-ins.

The tracer (Phase 1) discovers listener support via `isinstance(provider, BaseProvider)` or `hasattr(provider, "subscribe_request")`; for duck-typed providers without the registry, the tracer emits high-level spans (from `agent.subscribe`) but no `chat` span attributes for request/chunk/response — and logs an INFO line about this.

- [ ] **Step 1: Leave the existing `Provider` Protocol untouched**

Do **not** modify `class Provider(Protocol)`. It stays as today:

```python
@runtime_checkable
class Provider(Protocol):
    async def stream(...) -> MessageStream: ...
```

- [ ] **Step 2: Add a new `BaseProvider` concrete class to `cubepi/providers/base.py`**

Place it immediately after the `Provider` Protocol declaration:

```python
class BaseProvider:
    """Concrete base class for built-in cubepi providers.

    Built-in providers (Anthropic, OpenAI, FauxProvider) inherit from this
    class to gain the persistent listener registry used by ``cubepi.tracing``
    and other observers. User-defined providers may also inherit from
    ``BaseProvider`` to opt in, or remain duck-typed against the
    ``Provider`` Protocol (which only requires ``stream()``).

    Concrete subclasses must implement ``stream()`` and call
    ``_fire_listeners`` at three points: after the request payload is
    finalized, for each ``StreamEvent`` pushed onto the stream, and exactly
    once in a ``finally`` block after the stream terminates.

    Per-call mutators (``StreamOptions.on_payload``, ``StreamOptions.on_response``)
    retain their existing single-slot semantics and fire independently of
    the persistent listener registry below.
    """

    def __init__(self) -> None:
        self._request_listeners: list[OnRequestCallback] = []
        self._chunk_listeners: list[OnChunkCallback] = []
        self._response_listeners: list[OnResponseBodyCallback] = []

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        raise NotImplementedError

    def subscribe_request(self, cb: OnRequestCallback) -> Callable[[], None]:
        """Register a persistent observer for request payloads.
        Returns a detach callable that removes this specific subscription."""
        self._request_listeners.append(cb)
        return lambda: _detach(self._request_listeners, cb)

    def subscribe_chunk(self, cb: OnChunkCallback) -> Callable[[], None]:
        """Register a persistent observer for stream chunks.
        Returns a detach callable."""
        self._chunk_listeners.append(cb)
        return lambda: _detach(self._chunk_listeners, cb)

    def subscribe_response(self, cb: OnResponseBodyCallback) -> Callable[[], None]:
        """Register a persistent observer for assembled responses.
        Returns a detach callable."""
        self._response_listeners.append(cb)
        return lambda: _detach(self._response_listeners, cb)


def _detach(listeners: list, cb: Callable) -> None:
    try:
        listeners.remove(cb)
    except ValueError:
        pass
```

- [ ] **Step 3: Export `BaseProvider` from the package**

In `cubepi/providers/__init__.py`, add `BaseProvider` (alongside the existing `Provider`) to imports and `__all__`. In `cubepi/__init__.py`, do the same so users can `from cubepi import BaseProvider`.

- [ ] **Step 4: Run `uv run pytest tests/`**

All existing tests pass — duck-typed test providers still work because the `Provider` Protocol is unchanged.

---

### Task 3: Update `AnthropicProvider` to inherit and fire listeners

**Files:**
- Modify: `cubepi/providers/anthropic.py`

**Goal:** Make `AnthropicProvider` inherit from `BaseProvider`, call `_fire_listeners` at three points in `stream()`, and guarantee `on_response` fires exactly once in a `finally` block.

**Codebase anchors** (verified against current `anthropic.py`):
- Class at `anthropic.py:60` is plain `class AnthropicProvider:` — no base class today.
- `invoke_on_payload` is called at `anthropic.py:143`.
- The internal `MessageStream` variable inside the producer is named **`ms`**, not `stream`.
- Direct `ms.push(...)` sites at lines 165, 177, 190, 201.
- A synchronous helper `_handle_event(event, partial, ms)` defined at `anthropic.py:334` makes **9 internal `ms.push(...)` calls** (lines 343, 352, 363, 378, 391, 400, 413, 421, 429). Caller at line 186: `self._handle_event(event, partial, ms)`.
- The producer body lives inside `stream()` and is what `MessageStream.attach_task` runs.

- [ ] **Step 1: Inherit from `BaseProvider`**

Find:

```python
class AnthropicProvider:
```

Change to:

```python
class AnthropicProvider(BaseProvider):
```

If an `__init__` exists, add `super().__init__()` as its first statement. If not, add one:

```python
def __init__(self) -> None:
    super().__init__()
```

- [ ] **Step 2: Fire `_request_listeners` after `invoke_on_payload`**

After the existing line at `anthropic.py:143` (`kwargs = await invoke_on_payload(opts.on_payload, kwargs, model)`), append:

```python
if self._request_listeners:
    await _fire_listeners(self._request_listeners, kwargs, model)
```

- [ ] **Step 3: Refactor `_handle_event` to async + add an `_emit` helper**

`_handle_event` (sync, line 334) cannot `await` listener firings. Two coupled changes:

  a) Add a private async helper to the class:

  ```python
  async def _emit(self, ms: MessageStream, event: StreamEvent, model: Model) -> None:
      ms.push(event)
      if self._chunk_listeners:
          await _fire_listeners(self._chunk_listeners, event, model)
  ```

  b) Refactor `_handle_event` from `def _handle_event(self, event, partial, ms)` to `async def _handle_event(self, event, partial, ms, model)`. Update its call site at `anthropic.py:186` to `await self._handle_event(event, partial, ms, model)`.

  c) Inside the refactored `_handle_event`, replace every `ms.push(...)` (the 9 sites at lines 343, 352, 363, 378, 391, 400, 413, 421, 429) with `await self._emit(ms, ..., model)`.

  d) Replace the 4 direct `ms.push(...)` sites in `stream()` itself (lines 165, 177, 190, 201) with `await self._emit(ms, ..., model)`.

- [ ] **Step 4: Wrap the producer body in `try / except BaseException / finally`**

Locate the producer coroutine body inside `stream()` — the section that does HTTP streaming, pushes events, and concludes with `ms.push(StreamEvent(type="done"))`. The wrap is around **that** body, not the `stream()` setup that synchronously returns a `MessageStream`. Structure:

```python
body: dict | None = None
exc: BaseException | None = None
try:
    # existing producer body — HTTP stream open, event loop, accumulating
    # `partial` AssistantMessage and any provider-side state needed to
    # assemble `body` below.
    # On normal completion, just before pushing the final "done" event:
    body = self._assemble_response(partial, response_id, response_model, ...)
except BaseException as e:
    exc = e
    raise
finally:
    if self._response_listeners:
        await _fire_listeners(self._response_listeners, body, model, exc)
```

`asyncio.CancelledError` is a `BaseException`, not an `Exception`. Catching `BaseException` is intentional; the exception is re-raised after listeners fire.

- [ ] **Step 5: Add `_assemble_response()`**

Returns a dict shaped like Anthropic's non-streaming `messages.create()` response. Source every field from state the streaming loop already accumulates — **do not** issue a second HTTP call.

Required fields, with sources:

| Key | Source |
|---|---|
| `id` | the `message_start` event's `message.id` (already captured) |
| `type` | constant `"message"` |
| `role` | constant `"assistant"` |
| `model` | the `message_start` event's `message.model` |
| `content` | accumulated `partial.content` translated back to Anthropic's content-block shape: `TextContent → {"type": "text", "text": ...}`, `ThinkingContent → {"type": "thinking", "thinking": ...}`, `ToolCall → {"type": "tool_use", "id": ..., "name": ..., "input": ...}` |
| `stop_reason` | raw Anthropic stop reason from `message_delta` (e.g. `"end_turn"`, `"tool_use"`, `"max_tokens"`, `"stop_sequence"`) — **NOT** the normalized cubepi value |
| `stop_sequence` | from `message_delta`, or `None` |
| `usage.input_tokens` | from `message_delta.usage.input_tokens` |
| `usage.output_tokens` | from `message_delta.usage.output_tokens` |
| `usage.cache_creation_input_tokens` | from `message_delta.usage.cache_creation_input_tokens`, or `0` |
| `usage.cache_read_input_tokens` | from `message_delta.usage.cache_read_input_tokens`, or `0` |

If the stream aborts before any of these are available, that field is omitted (not set to `None`). The recorder treats absent fields as "provider didn't return this" and skips the corresponding span attribute.

- [ ] **Step 6: Run `uv run pytest tests/providers/test_anthropic*.py`**

Existing tests pass — listener-related code is no-op when registries are empty.

---

### Task 4: Update OpenAI providers

**Files:**
- Modify: `cubepi/providers/openai.py`
- Modify: `cubepi/providers/openai_responses.py`

**Goal:** Same listener wiring as Task 3, applied to both OpenAI providers. The two providers have different stream shapes and different assembly paths — handled separately below.

**Note on the absence of `_handle_event`:** unlike `anthropic.py`, both OpenAI providers call `ms.push(...)` inline within their stream loop — no synchronous helper exists. Each call site can be inlined with `await self._emit(ms, ..., model)` directly (define `_emit` as a method on each provider class).

#### 4A. `openai.py` (`OpenAIProvider`, chat completions API)

**Codebase anchors:**
- Class at `openai.py:31` is plain `class OpenAIProvider:`.
- 16+ inline `ms.push(...)` sites between lines 132 and 383 (none in a helper function).
- `invoke_on_response` is called around line 119 with HTTP metadata; `invoke_on_payload` follows the same pattern as Anthropic — find it in the same file.
- The streaming loop accumulates `response_id` (line 169), `usage` (`u` variable at line 148), and tool-call deltas; finish reason is captured at line 326.

- [ ] **Step 1: Inherit from `BaseProvider`** — same pattern as Task 3 Step 1.

- [ ] **Step 2: Add `_emit` helper method on the class.** Identical signature/body to the Anthropic helper (Task 3 Step 3a).

- [ ] **Step 3: Fire `_request_listeners` after `invoke_on_payload`** — find the call site, append the guarded fire as in Task 3 Step 2.

- [ ] **Step 4: Replace every `ms.push(...)` with `await self._emit(ms, ..., model)`** at all 16+ sites. (Search/replace, then visually verify each.)

- [ ] **Step 5: Wrap producer body in `try / except BaseException / finally`** — same pattern as Task 3 Step 4.

- [ ] **Step 6: Implement `_assemble_response()`** — returns OpenAI's `chat.completion` non-streaming response shape:

| Key | Source |
|---|---|
| `id` | `response_id` captured at line 169 |
| `object` | constant `"chat.completion"` |
| `created` | first chunk's `created` (capture during stream) |
| `model` | first chunk's `model` (capture during stream) |
| `system_fingerprint` | first chunk's `system_fingerprint`, or absent |
| `service_tier` | first chunk's `service_tier`, or absent |
| `choices` | one-element array: `[{"index": 0, "message": {"role": "assistant", "content": <accumulated text>, "tool_calls": <accumulated tool_calls in OpenAI's format>}, "finish_reason": <captured at line 326>, "logprobs": null}]` |
| `usage` | accumulated `Usage` dict: `{"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details": {"cached_tokens": ...}}` — sourced from the `u` variable around line 148 |

Tool-call shape inside `choices[0].message.tool_calls` follows OpenAI's spec: `[{"id": ..., "type": "function", "function": {"name": ..., "arguments": <json string>}}]`. Accumulate from the `toolcall_*` `StreamEvent`s during streaming.

#### 4B. `openai_responses.py` (`OpenAIResponsesProvider`, Responses API)

**Codebase anchors:**
- Class at `openai_responses.py:42` is plain `class OpenAIResponsesProvider:`.
- Many inline `ms.push(...)` sites between lines 131 and 479.
- **Key shortcut:** at line 410, the streaming code already binds `resp = event.response` — this is the OpenAI SDK's `Response` object containing the fully-assembled response. No manual accumulation needed; `resp.model_dump()` (or `dict(resp)`) gives the canonical shape.

- [ ] **Step 1: Inherit from `BaseProvider`.**

- [ ] **Step 2: Add `_emit` helper method.**

- [ ] **Step 3: Fire `_request_listeners` after `invoke_on_payload`.**

- [ ] **Step 4: Replace every `ms.push(...)` with `await self._emit(ms, ..., model)`.**

- [ ] **Step 5: Wrap producer body in `try / except BaseException / finally`.**

- [ ] **Step 6: Implement `_assemble_response()`** — leverage the SDK's already-assembled response object. Capture `resp` at line 410 (and the analogous line 450 for the failure path) into an instance/local variable accessible from the `finally` block, then:

  ```python
  body = resp.model_dump(mode="json") if resp is not None else None
  ```

  This yields the canonical Responses-API shape. No field-by-field assembly required.

- [ ] **Step 7: Run `uv run pytest tests/providers/test_openai*.py`**

---

### Task 5: Update `FauxProvider`

**Files:**
- Modify: `cubepi/providers/faux.py`

**Goal:** Same listener wiring as Task 3, applied to the test/dev faux provider. **This is the most important provider** for testing the listener registry itself (Task 6 pins contract via faux). Tests must be deterministic — `faux.py` currently uses `random.randbytes`, `random.randint`, and `time.time()` (see `_random_id` at line 36, the token-size sampling at line 81, `timestamp` at lines 76, 241, 285, 476). The faux response assembly must NOT inherit that nondeterminism.

**Codebase anchors:**
- Class at `faux.py:151` is `class FauxProvider:`.
- `_random_id(prefix)` at line 36 builds non-deterministic IDs from `time.time()` + `random.randbytes`.

- [ ] **Step 1: Inherit from `BaseProvider`** — same pattern as Task 3.

- [ ] **Step 2: Add `_emit` helper method and replace every `ms.push(...)`** — same pattern.

- [ ] **Step 3: Fire `_request_listeners` after `invoke_on_payload`** — same pattern.

- [ ] **Step 4: Wrap producer body in `try / except BaseException / finally`** — same pattern.

- [ ] **Step 5: Implement `_assemble_response()` with a pinned schema**

Define a per-instance monotonic counter (`self._response_seq`, initialized in `__init__`, incremented at the start of each `stream()` call) so IDs are deterministic per FauxProvider instance:

```python
def __init__(self, ...) -> None:
    super().__init__()
    self._response_seq = 0
    # ... existing init

# inside stream() producer:
self._response_seq += 1
seq = self._response_seq
```

Pin the dict schema exactly:

```python
def _assemble_response(
    self,
    *,
    seq: int,
    model: Model,
    content_blocks: list[dict],
    stop_reason: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    return {
        "id": f"faux-{seq}",
        "model": model.id,
        "role": "assistant",
        "content": content_blocks,                  # list of {"type", "text"|...}
        "stop_reason": stop_reason,                 # "stop" / "tool_use" / "length" / "error"
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
```

Tests in Task 6 will assert against this exact shape — no extra keys, no missing keys, no random IDs.

- [ ] **Step 6: Run `uv run pytest tests/providers/test_faux*.py`**

---

### Task 6: Listener registry tests

**Files:**
- Create: `tests/providers/test_listener_registry.py`

**Goal:** Pin the listener registry contract with tests that don't rely on any real provider.

- [ ] **Step 1: Test that listeners can be subscribed and called**

Use `FauxProvider` (subclassing `BaseProvider` after Task 5). Subscribe one listener of each type. Run a streamed call. Assert each listener was invoked at least once, with the documented argument shapes.

- [ ] **Step 2: Test detach()**

Subscribe a listener; detach; run another call. Listener must not be invoked again.

- [ ] **Step 3: Test multiple subscribers, in registration order**

Subscribe two listeners on `subscribe_request`. Assert both fire, in registration order.

- [ ] **Step 4: Test that a raising listener does not crash the stream**

Subscribe a listener that raises `RuntimeError`. Subscribe a second listener after it. Run a call. Assert: (a) the call completes successfully, (b) the second listener still fires.

- [ ] **Step 5: Test response listener fires exactly once on normal completion**

- [ ] **Step 6: Test response listener fires exactly once on exception**

Make `FauxProvider.stream()` raise mid-stream (via a test-only knob or by injecting an error). Assert `response_listener` was called once with `(None, model, exc)` (or with a partial body and `exc`, depending on how faux assembles).

- [ ] **Step 7: Test response listener fires exactly once on `asyncio.CancelledError`**

Start a stream, cancel the task before it completes. Assert `response_listener` was called once with `(body_or_None, model, isinstance=CancelledError)`. The `CancelledError` should propagate out.

- [ ] **Step 8: Test that `StreamOptions.on_payload` mutation runs before `subscribe_request` listeners see the payload**

Provide a per-call `on_payload` that mutates the payload. Subscribe a `subscribe_request` listener. Assert the listener sees the mutated payload, not the pre-mutation one.

- [ ] **Step 9: Test concurrent streams on the same Provider**

Subscribe one chunk listener. Launch two `faux.stream(...)` calls concurrently via `asyncio.gather`. Assert the listener receives events from both streams (use the `Model` argument to distinguish, since each call passes its own model) and that response_listener fires exactly twice — once per call.

- [ ] **Step 10: Test that a listener can detach itself mid-iteration**

Subscribe two listeners on `subscribe_chunk`. The first one calls its own detach callable on its first invocation. The second listener must still fire on the same and subsequent chunks. (This is the `tuple(listeners)` snapshot semantics — verify it.)

- [ ] **Step 11: Test that a listener subscribed mid-stream begins firing on the next call (not retroactively)**

Subscribe a `subscribe_response` listener. Inside its body, subscribe a second `subscribe_response` listener. Assert: on the same stream, only the first listener fired; on a follow-up stream, both fire.

- [ ] **Step 12: Test that a slow async listener serializes the stream**

Subscribe a chunk listener that `await asyncio.sleep(0.05)`. Time a faux stream end-to-end. Assert the wall time is at least `0.05 * chunk_count`. This documents the contract: listeners run **inline** in the producer coroutine and block subsequent chunks. The Phase 1 tracer's listeners are designed to be cheap; a slow listener is the listener-author's problem, but the contract must be visible in tests.

- [ ] **Step 13: Run `uv run pytest tests/providers/test_listener_registry.py -v`**

All tests pass.

---

## Phase 0b — `ToolExecutionEndEvent` extension

These tasks are independent of Phase 0a and can land in a separate PR.

### Task 7: Extend `ToolExecutionEndEvent`

**Files:**
- Modify: `cubepi/agent/types.py`

**Goal:** Add three additive fields with conservative defaults.

- [ ] **Step 1: Update the `ToolExecutionEndEvent` class**

Find:

```python
class ToolExecutionEndEvent(BaseModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False
```

Replace with:

```python
class ToolExecutionEndEvent(BaseModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False
    terminate: bool = False
    """True iff the tool's AgentToolResult.terminate was True (or the
    after_tool_call hook set terminate=True). Recorders use this to mark
    the turn as terminated-by-tool."""
    blocked_by_hook: bool = False
    """True iff the tool call was blocked by a before_tool_call hook
    returning block=True. Distinguishes hook-blocks from other immediate
    errors (tool-not-found, arg-validation)."""
    block_reason: str | None = None
    """When blocked_by_hook is True, the reason string from
    BeforeToolCallResult.reason (or None if the hook supplied no reason)."""
```

- [ ] **Step 2: Run `uv run pytest tests/`**

Tests pass. No existing test should be checking these fields yet; defaults mean nothing changes.

---

### Task 8: Track block reason in `_ImmediateOutcome`

**Files:**
- Modify: `cubepi/agent/tools.py`

**Goal:** Internally distinguish "blocked by hook" from other immediate errors, and carry the reason string.

- [ ] **Step 1: Add fields to `_ImmediateOutcome`**

Find:

```python
@dataclass
class _ImmediateOutcome:
    result: AgentToolResult
    is_error: bool
```

Replace with:

```python
@dataclass
class _ImmediateOutcome:
    result: AgentToolResult
    is_error: bool
    blocked_by_hook: bool = False
    block_reason: str | None = None
```

- [ ] **Step 2: Set the fields in `_prepare_tool_call` when `before_tool_call` blocks**

In `_prepare_tool_call`, find the block-handling branch:

```python
if before_result and before_result.block:
    return _ImmediateOutcome(
        result=_error_result(
            before_result.reason or "Tool execution was blocked"
        ),
        is_error=True,
    )
```

Change to:

```python
if before_result and before_result.block:
    return _ImmediateOutcome(
        result=_error_result(
            before_result.reason or "Tool execution was blocked"
        ),
        is_error=True,
        blocked_by_hook=True,
        block_reason=before_result.reason,
    )
```

Leave all other `_ImmediateOutcome` constructions unchanged — they keep `blocked_by_hook=False`, `block_reason=None` by default.

---

### Task 9: Thread `terminate` / `blocked_by_hook` / `block_reason` through `_FinalizedOutcome`

**Files:**
- Modify: `cubepi/agent/tools.py`

**Goal:** Carry the three values from `_ImmediateOutcome` (when applicable) and `AgentToolResult.terminate` through to the event emission sites.

- [ ] **Step 1: Add fields to `_FinalizedOutcome`**

Find:

```python
@dataclass
class _FinalizedOutcome:
    tool_call: ToolCall
    result: AgentToolResult
    is_error: bool
```

Replace with:

```python
@dataclass
class _FinalizedOutcome:
    tool_call: ToolCall
    result: AgentToolResult
    is_error: bool
    blocked_by_hook: bool = False
    block_reason: str | None = None
```

- [ ] **Step 2: Propagate blocked-by-hook info when building `_FinalizedOutcome` from an immediate-blocked outcome**

In both `_execute_sequential` and `_execute_parallel`, locate the point that creates a `_FinalizedOutcome` from an `_ImmediateOutcome`. Update it to copy the new fields:

```python
finalized = _FinalizedOutcome(
    tool_call=tc,
    result=preparation.result,
    is_error=preparation.is_error,
    blocked_by_hook=preparation.blocked_by_hook,
    block_reason=preparation.block_reason,
)
```

All other `_FinalizedOutcome` construction sites stay at default (`False`, `None`).

---

### Task 10: Emit new fields at all three `ToolExecutionEndEvent` sites

**Files:**
- Modify: `cubepi/agent/tools.py`

**Goal:** Populate the three new event fields in both the sequential and parallel emission paths.

- [ ] **Step 1: Sequential path**

Find the `ToolExecutionEndEvent` emission in `_execute_sequential` (around line 283):

```python
ToolExecutionEndEvent(
    tool_call_id=tc.id,
    tool_name=tc.name,
    result=finalized.result,
    is_error=finalized.is_error,
),
```

Replace with:

```python
ToolExecutionEndEvent(
    tool_call_id=tc.id,
    tool_name=tc.name,
    result=finalized.result,
    is_error=finalized.is_error,
    terminate=bool(finalized.result.terminate),
    blocked_by_hook=finalized.blocked_by_hook,
    block_reason=finalized.block_reason,
),
```

- [ ] **Step 2: Parallel path — immediate-outcome branch**

Same pattern at the equivalent emission site in `_execute_parallel`:

```python
ToolExecutionEndEvent(
    tool_call_id=tc.id,
    tool_name=tc.name,
    result=finalized.result,
    is_error=finalized.is_error,
    terminate=bool(finalized.result.terminate),
    blocked_by_hook=finalized.blocked_by_hook,
    block_reason=finalized.block_reason,
),
```

- [ ] **Step 3: Parallel path — async-execution branch**

There's a third emission inside the `_run` inner function (around line 355). Update the same way; `fin` is the `_FinalizedOutcome`:

```python
ToolExecutionEndEvent(
    tool_call_id=prep.tool_call.id,
    tool_name=prep.tool_call.name,
    result=fin.result,
    is_error=fin.is_error,
    terminate=bool(fin.result.terminate),
    blocked_by_hook=fin.blocked_by_hook,
    block_reason=fin.block_reason,
),
```

- [ ] **Step 4: Run `uv run pytest tests/agent/`**

All existing tests pass.

---

### Task 11: Event extension tests

**Files:**
- Create: `tests/agent/test_tool_event_extension.py`

**Goal:** Pin the three new field behaviors.

- [ ] **Step 1: Test `terminate=True` propagates**

Build an `AgentTool` whose execute returns `AgentToolResult(content=[...], terminate=True)`. Run the agent. Capture `ToolExecutionEndEvent` via `agent.subscribe(...)`. Assert `event.terminate is True`.

- [ ] **Step 2: Test `terminate=False` for normal tools**

Same setup with a tool that returns `terminate=None` or unset. Assert `event.terminate is False`.

- [ ] **Step 3: Test `blocked_by_hook=True` and `block_reason`**

Define a `before_tool_call` hook that returns `BeforeToolCallResult(block=True, reason="not allowed")`. Run the agent so the tool gets blocked. Assert the emitted event has `blocked_by_hook=True`, `block_reason="not allowed"`, `is_error=True`.

- [ ] **Step 4: Test other immediate errors don't set `blocked_by_hook`**

Force a tool-not-found case (call a tool the agent doesn't know). Assert the emitted event has `is_error=True` but `blocked_by_hook=False` and `block_reason is None`.

- [ ] **Step 5: Test arg-validation error doesn't set `blocked_by_hook`**

Pass invalid arguments. Same assertion as Step 4.

- [ ] **Step 6: Test parallel-execution path also populates the fields**

Run with `tool_execution="parallel"` and two tool calls, one of which the hook blocks. Both events should fire with correct fields.

- [ ] **Step 7: Run `uv run pytest tests/agent/test_tool_event_extension.py -v`**

---

## Cross-cutting

### Task 12: Version bump

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: (skipped — no CHANGELOG.md in repo)**

- [ ] **Step 2: Bump version**

Version lives in `pyproject.toml:3` (`version = "0.3.0"`). cubepi does **not** expose a `cubepi.__version__` attribute on the package; do not add one as part of this plan unless other work depends on it.

Bump `pyproject.toml` to `0.4.0` — additive but introduces new public API surface (`BaseProvider.subscribe_*`, three new `ToolExecutionEndEvent` fields).

There is no `CHANGELOG.md` in the repo today. Skip Step 1 (changelog) entirely; capture the release note in the PR description instead.

---

### Task 13: PR breakdown

These two phases land in **two separate PRs** on **two separate branches**:

- [ ] **PR 1:** `feat/provider-listener-registry` — Tasks 1–6 + Task 12 (version bump to 0.4.0)
- [ ] **PR 2:** `feat/tool-event-extension` — Tasks 7–11 (no separate version bump — PR 1's 0.4.0 covers both)

Order does not matter; they are independent. Phase 1 (the `cubepi/tracing/` module) waits on both.

---

## Acceptance Criteria

After Phase 0:

- [ ] `uv run pytest tests/` is green
- [ ] `uv run ruff check cubepi/ tests/` is clean
- [ ] `uv run ruff format --check cubepi/ tests/` is clean
- [ ] `Provider` remains an unchanged `runtime_checkable` Protocol; duck-typed user providers still satisfy `isinstance(x, Provider)`
- [ ] `BaseProvider` is a concrete class; all four built-in providers inherit from it
- [ ] `BaseProvider.subscribe_request` / `subscribe_chunk` / `subscribe_response` exist, return detach callables, support multiple subscribers
- [ ] Each provider's `stream()` invokes `_fire_listeners` at the three documented points
- [ ] `_response_listeners` fires exactly once per `stream()` call, including under `asyncio.CancelledError`
- [ ] A listener that raises does not crash the stream and does not prevent other listeners from firing
- [ ] `ToolExecutionEndEvent` has `terminate`, `blocked_by_hook`, `block_reason` fields with conservative defaults
- [ ] `cubepi/agent/tools.py` populates these fields correctly in both sequential and parallel paths
- [ ] No existing test is modified to pass (i.e., backwards compatible)
- [ ] No new runtime dependency added
- [ ] Version bumped to reflect new public API surface
