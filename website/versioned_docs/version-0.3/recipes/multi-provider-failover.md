---
title: Multi-Provider Failover
---

# Recipe: Multi-Provider Failover

When Anthropic is rate-limited or down, fail over to OpenAI without
crashing the agent. We'll wrap both providers behind a single
`Provider` adapter that does its own retry/failover logic.

**Time to run:** 10 minutes.
**Deps:** `cubepi`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.

## The wrapper provider

```python title="failover.py"
import asyncio
import logging
from typing import Sequence

from cubepi.providers.base import (
    Message,
    MessageStream,
    Model,
    Provider,
    StreamOptions,
    ToolDefinition,
)
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.openai import OpenAIProvider

log = logging.getLogger(__name__)


class FailoverProvider:
    """Try a primary provider; fall back to others on retryable errors."""

    def __init__(self, primary_pair: tuple[Provider, Model], *fallbacks: tuple[Provider, Model]) -> None:
        self._chain: list[tuple[Provider, Model]] = [primary_pair, *fallbacks]

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        # `model` is the agent's selected Model — we ignore it and use the
        # one paired with each provider in the chain.
        last_exc: BaseException | None = None
        for provider, mapped_model in self._chain:
            try:
                stream = await provider.stream(
                    mapped_model,
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    options=options,
                )
                # Peek at the first event to validate the stream actually started.
                # If the provider produces a stream object but errors on first chunk,
                # we want to fall through.
                return stream
            except Exception as e:
                log.warning("provider %s failed: %s — trying fallback", mapped_model.provider, e)
                last_exc = e
                continue

        raise RuntimeError(f"all providers exhausted; last error: {last_exc!r}")
```

## Use it

```python title="main.py"
import asyncio
import os

from cubepi import Agent, Model
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.openai import OpenAIProvider
from failover import FailoverProvider


async def main():
    failover = FailoverProvider(
        (
            AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"]),
            Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
        ),
        (
            OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]),
            Model(id="gpt-5", provider="openai"),
        ),
    )

    # The model passed here is overridden inside FailoverProvider; pass any
    # placeholder. We use the primary's so usage tracking labels match the
    # happy path.
    agent = Agent(
        provider=failover,
        model=Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
        system_prompt="You answer concisely.",
    )
    agent.subscribe(lambda e, s=None: None)
    await agent.prompt("Capital of Mongolia?")
    last = agent.state.messages[-1]
    print(last.content[0].text)


asyncio.run(main())
```

## What about smarter failover policies?

The example above falls back on **any** exception. That's the right
behaviour for `RateLimitError`, `APIConnectionError`, or 5xx — but
arguably wrong for `BadRequestError` (your code is wrong; the next
provider will fail the same way).

Tighten the catch:

```python
import anthropic, openai

RETRYABLE = (
    anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError,
    openai.RateLimitError, openai.APIConnectionError, openai.APIStatusError,
)

# Inside the loop:
except RETRYABLE as e:
    ...
except Exception:
    raise   # not retryable
```

## Adding circuit breaking

Don't keep retrying a provider that's clearly down. A simple counter:

```python
import time

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 60) -> None:
        self._failures = 0
        self._opened_at: float | None = None
        self._threshold = failure_threshold
        self._recovery = recovery_seconds

    def can_attempt(self) -> bool:
        if self._opened_at and time.monotonic() - self._opened_at < self._recovery:
            return False
        if self._opened_at:
            self._opened_at = None   # half-open
        return True

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()
            self._failures = 0

    def record_success(self) -> None:
        self._failures = 0
```

Hold one `CircuitBreaker` per provider in the `FailoverProvider`,
skip if `can_attempt()` is False.

## Per-tool failover doesn't apply

This recipe handles **provider** failures. Tool failures are different
— see [Middleware → Retries](../guides/middleware/examples#retries-with-backoff)
for that pattern.

## Common pitfalls

- **Different tool schemas across providers** — Both built-in
  providers accept the same `ToolDefinition`, but extra-body
  customisations (e.g. OpenAI `parallel_tool_calls=False`) won't carry
  to Anthropic. Keep cross-provider behaviour in
  [`transform_context`](../guides/middleware/hooks#transform_context),
  not in `extra_body`.
- **Different cost** — Failover from Anthropic to OpenAI changes
  per-token cost. Track which provider answered (via `on_response` or
  `AssistantMessage.provider_id`) and bill accordingly.
- **Streaming consistency** — The wrapper passes streams through
  unchanged, so consumers see the same `StreamEvent` shape regardless
  of which provider answered.

## See also

- [Providers / Anthropic](../guides/providers/anthropic) and
  [OpenAI](../guides/providers/openai) — provider-specific details.
- [Writing a Custom Provider](../guides/providers/custom) — the same
  Protocol used by this wrapper.
