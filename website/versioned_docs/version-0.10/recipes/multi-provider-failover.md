---
title: Multi-Provider Failover
description: "Automatic failover between LLM providers using FallbackBoundModel."
---

# Recipe: Multi-Provider Failover

When the primary provider is rate-limited, unavailable, or has hit its context
limit, fall over to the next one automatically — without crashing the agent.
CubePi ships `FallbackBoundModel` for this out of the box.

**Time to read:** 5 minutes.
**Deps:** `cubepi`, API keys for two providers.

## The built-in: `FallbackBoundModel`

```python
import os
from cubepi import Agent, FallbackBoundModel
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.openai import OpenAIProvider

anthropic = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
openai = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])

model = FallbackBoundModel(
    chain=(
        anthropic.model("claude-opus-4-8"),   # primary
        openai.model("gpt-5"),                # fallback
    )
)

agent = Agent(model=model, system_prompt="You answer concisely.")
await agent.prompt("Capital of Mongolia?")
```

`FallbackBoundModel` peeks at the first stream event from each provider. If it
is a `type="error"` event, or if `stream()` raises a retriable error, the next
model in the chain is tried. Once a non-error first event arrives the stream is
forwarded as-is — mid-stream errors are not retried.

## Default trigger conditions

By default failover is triggered on:

| Error | Why |
|---|---|
| `RateLimited` | Quota hit; another provider can serve |
| `ProviderUnavailable` | 5xx / timeout / connection failure |
| `ContextLengthExceeded` | Fallback may have a larger context window |

Auth failures (`ProviderAuthFailed`) and bad requests (`ProviderBadRequest`)
are **not** triggered by default — a bad key or a malformed request will fail
the same way on every provider in the chain.

## Custom trigger conditions

Pass `trigger_errors` to override:

```python
from cubepi import FallbackBoundModel
from cubepi.errors import ProviderAuthFailed, ProviderUnavailable, RateLimited

model = FallbackBoundModel(
    chain=(primary, fallback),
    trigger_errors=frozenset({RateLimited, ProviderUnavailable, ProviderAuthFailed}),
)
```

## Monitoring failovers

Pass `on_failover` to hook into billing or alerting:

```python
import logging

log = logging.getLogger(__name__)

async def record_failover(failed, next_model, error):
    log.warning(
        "provider failover: %s/%s → %s/%s (%s)",
        failed.spec.provider_id, failed.spec.id,
        next_model.spec.provider_id if next_model else "none",
        next_model.spec.id if next_model else "—",
        error,
    )
    # e.g. await billing.record_fallback_failure(failed.spec, error)

model = FallbackBoundModel(
    chain=(primary, fallback),
    on_failover=record_failover,
)
```

`on_failover` receives `(failed: BoundModel, next_model: BoundModel | None,
error: BaseException | str)`. Both sync and async callables are accepted.
Exceptions raised inside the callback are logged and swallowed — a broken
callback never aborts the failover.

## `provider` and `spec` always reflect the primary

`FallbackBoundModel.provider` and `.spec` proxy `chain[0]`. Tracing and
billing code that reads `agent._model.provider` or `agent._model.spec` sees
the primary — which is the intended provider. The `AssistantMessage` returned
by the actual call carries `provider_id` and `model_id` of whichever model
really responded.

## Common pitfalls

- **Different tool schemas across providers** — Both built-in providers accept
  the same `ToolDefinition`, but vendor-specific extras (e.g. OpenAI
  `parallel_tool_calls=False`) won't carry to Anthropic. Keep cross-provider
  behaviour in `transform_context` middleware, not in `extra_body`.
- **Different cost** — Failover changes per-token cost. Track which provider
  answered via `AssistantMessage.provider_id` and bill accordingly; the
  `on_failover` callback is the right place to record the switch.
- **Mid-stream errors aren't retried** — Once a healthy first event arrives,
  `FallbackBoundModel` commits to that provider. Errors during the rest of
  the stream are forwarded to the agent as-is.
- **`ContextLengthExceeded` is only useful if the fallback is larger** — If
  both providers have the same context window the failover will fail the same
  way. Consider pairing a standard-window primary with a large-context fallback.

## Run the example

A runnable version is in the repository:

```bash
git clone https://github.com/cubeplexai/cubepi && cd cubepi
uv sync

export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY [+ OPENAI_BASE_URL]
uv run python examples/multi_provider_failover.py
```

The example deliberately uses a bad key for the primary to trigger failover,
then answers correctly via the real fallback.

## See also

- [Providers Overview](../guides/providers/overview) — provider setup and `CapabilityDescriptor`.
- [Providers / Anthropic](../guides/providers/anthropic) and [OpenAI](../guides/providers/openai) — provider-specific details.
- [Writing a Custom Provider](../guides/providers/custom) — when neither built-in provider fits.
