# FallbackBoundModel Design

**Date:** 2026-06-08
**Status:** Approved

## Problem

cubebox stores a `fallback_models` chain in config/DB, but the execution layer
never acts on it. The TODO in `run_manager.py` (PR #84 review) has been open
since cubepi had no equivalent of LangChain's `with_fallbacks()`. PR #157
threaded `BoundModel` through the agent loop, making this the right moment to
fill the gap at the `BoundModel` level rather than the `Provider` level.

## Decision

Add `FallbackBoundModel` to `cubepi/providers/fallback.py`. It is a standalone
dataclass (not a `BoundModel` subclass) that wraps an ordered chain of
`BoundModel` instances and implements the same `stream()`/`generate()` interface.
When the primary model fails with a retriable error, it transparently tries the
next model in the chain.

## Why BoundModel level, not Provider level

`Provider.stream(model, messages, ...)` takes the model spec as a parameter —
callers decide which spec to pass. A fallback chain contains heterogeneous
`(provider, spec)` pairs; there is no single "model spec" to pass from outside.
`BoundModel.stream(messages, ...)` owns both provider and spec internally,
making it the natural unit for a multi-model chain. The agent loop (post #157)
already works at the `BoundModel` level.

## Data Structure

```python
# cubepi/providers/fallback.py

DEFAULT_TRIGGER_ERRORS: frozenset[type[ProviderError]] = frozenset({
    RateLimited,
    ProviderUnavailable,
    ContextLengthExceeded,
})

@dataclass(frozen=True)
class FallbackBoundModel:
    chain: tuple[BoundModel, ...]          # chain[0] = primary
    trigger_errors: frozenset[type[ProviderError]] = DEFAULT_TRIGGER_ERRORS
    on_failover: Callable[[BoundModel, BoundModel, BaseException | str],
                           Awaitable[None] | None] | None = None

    @property
    def provider(self) -> Provider:
        return self.chain[0].provider      # primary provider, for tracing/billing

    @property
    def spec(self) -> Model:
        return self.chain[0].spec          # primary model spec
```

`chain` is the single source of truth. All elements are symmetric — there is no
implicit "primary stored separately in parent fields" distinction. `provider` and
`spec` properties expose primary metadata so tracing/billing code that reads
`agent._model.provider` and `agent._model.spec` continues to work without
changes.

## Type Compatibility

`Agent.__init__` currently types `model: BoundModel`. Since `FallbackBoundModel`
satisfies the same structural interface (`provider`, `spec`, `stream()`,
`generate()`), the type hint is widened to `BoundModel | FallbackBoundModel`.
A `ModelHandle` Protocol is explicitly out of scope for this change — the union
is sufficient and avoids a larger public API surface change.

## `stream()` Behaviour

Two error paths, handled differently:

**Path 1 — exception from `await bound.stream(...)`** (stream never started):
- Exception type is in `trigger_errors` → log warning + call `on_failover` +
  try next model in chain.
- Exception type is NOT in `trigger_errors` → re-raise immediately, do not try
  further models.

**Path 2 — first `StreamEvent` has `type == "error"`** (stream started but
provider errored before emitting content):
- Always fall over, regardless of error string content. At this point no content
  has been delivered to the consumer, so switching is safe. Error events carry
  only a string message — no structured type — so filtering by type is not
  possible.

**Once a non-error first event is received**, the rest of the stream is forwarded
as-is. Mid-stream errors (errors after the first non-error event) are forwarded
to the consumer unchanged — no retry (approach A).

**All models exhausted** → raise `ProviderUnavailable("all providers exhausted;
last error: ...")`.

```
for bound in chain:
    try:
        inner = await bound.stream(messages, ...)
    except trigger_errors as exc:
        _log_and_notify(failed=bound, next=next_bound, error=exc)
        continue
    except Exception:
        raise

    first = await peek_first_event(inner)
    if first is None or first.type == "error":
        _log_and_notify(failed=bound, next=next_bound, error=first.error_message)
        continue

    outer = MessageStream()
    outer.attach_task(asyncio.create_task(_forward(first, inner, outer)))
    return outer

raise ProviderUnavailable("all providers exhausted; last error: ...")
```

## `generate()` Behaviour

Simpler — no streaming peek needed:

```
for bound in chain:
    try:
        return await bound.generate(messages, ...)
    except trigger_errors as exc:
        _log_and_notify(failed=bound, next=next_bound, error=exc)
        continue
    except Exception:
        raise

raise ProviderUnavailable("all providers exhausted; last error: ...")
```

## Observability

**Structured log (built-in):** Every fallover emits a `WARNING` via loguru with
`failed`, `next`, `error`, and `attempt` fields:

```
WARNING cubepi.providers.fallback: failover triggered
  failed=anthropic/claude-opus-4-8  →  next=openai/gpt-4o
  reason=RateLimited: 429 Too Many Requests
  attempt=1/3
```

**`on_failover` callback (optional):** Called after the log, before trying the
next model. Signature: `(failed: BoundModel, next: BoundModel, error:
BaseException | str) -> Awaitable[None] | None`. Both sync and async callbacks
are accepted. Exceptions raised inside the callback are logged and swallowed —
a buggy callback must not abort the failover.

cubebox wires `billing.record_fallback_failure()` here. cubepi itself has no
knowledge of billing.

**`AssistantMessage.provider_id` / `model_id`:** Already set by the provider that
actually responds, so traces naturally show which model handled the request.

## Error Type Policy

Default `trigger_errors = {RateLimited, ProviderUnavailable, ContextLengthExceeded}`:

| Error | Default | Rationale |
|---|---|---|
| `RateLimited` | ✓ trigger | Quota exhausted; another provider can serve |
| `ProviderUnavailable` | ✓ trigger | Transient outage; switching is correct |
| `ContextLengthExceeded` | ✓ trigger | Fallback may have larger context window |
| `ProviderAuthFailed` | ✗ not triggered | Auth failures are config problems; silently falling over hides broken keys and drains fallback budget |
| `ProviderBadRequest` | ✗ not triggered | Bad request structure will fail on all models; falling over wastes the full chain and produces a less useful error |

Callers override by passing `trigger_errors=frozenset({...})`.

## File Layout

```
cubepi/providers/fallback.py     # FallbackBoundModel + DEFAULT_TRIGGER_ERRORS
tests/providers/test_fallback.py # unit tests (FauxProvider-based)
```

`FallbackBoundModel` and `DEFAULT_TRIGGER_ERRORS` are exported from
`cubepi/providers/__init__.py` and the top-level `cubepi/__init__.py`.

## Tests

All tests use `FauxProvider` (already in `tests/`), no real API calls.

| # | Scenario | Assert |
|---|---|---|
| 1 | Primary succeeds | Returns primary's stream; no fallover |
| 2 | Primary raises `RateLimited`, fallback succeeds | Returns fallback's stream; `on_failover` called with correct args |
| 3 | Primary raises `ProviderBadRequest` (not in trigger_errors) | Re-raises immediately; fallback not tried |
| 4 | Primary first event is `error`, fallback succeeds | Returns fallback's stream |
| 5 | All models exhausted | Raises `ProviderUnavailable` |
| 6 | `on_failover` callback raises | Swallowed; failover still completes |
| 7 | `generate()` — primary raises `RateLimited`, fallback succeeds | Returns fallback's `AssistantMessage` |
| 8 | Custom `trigger_errors` includes `ProviderAuthFailed` | Auth failure triggers fallover |

## Out of Scope

- Mid-stream retry (approach A decision: only first-event failover).
- `ModelHandle` Protocol (deferred; union type is sufficient for now).
- cubebox integration (separate cubebox task: wire `FallbackBoundModel` in
  `LLMFactory`, remove the TODO in `run_manager.py`).
- Retry-with-backoff on `RateLimited.retry_after` (separate concern).
