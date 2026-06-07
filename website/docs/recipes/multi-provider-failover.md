---
title: Multi-Provider Failover
description: "Configure multi-provider failover in CubePi for resilience — automatic fallback between providers."
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
import time
from typing import Sequence

from cubepi.providers.base import (
    AssistantMessage,
    BaseProvider,
    BoundModel,
    Message,
    MessageStream,
    Model,
    StreamEvent,
    StreamOptions,
    ToolDefinition,
    Usage,
)
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.openai import OpenAIProvider

log = logging.getLogger(__name__)


class FailoverProvider(BaseProvider):
    """Try providers in order; fall over on construction or first-event errors.

    Built-in providers swallow API/network errors and surface them as
    `StreamEvent(type="error")` on the returned stream — never as exceptions
    out of `provider.stream()`. So we peek at the first event from each
    inner stream and only commit to it once we see a non-error event.

    Limitation: errors that arrive *after* the first event (e.g. mid-stream
    rate limit, server disconnect) are forwarded to the agent as-is.
    Fully replaying a half-streamed turn against a fallback provider would
    require buffering the whole turn — out of scope here.
    """

    def __init__(self, primary: BoundModel, *fallbacks: BoundModel) -> None:
        super().__init__(provider_id=primary.spec.provider_id)
        self._chain: list[BoundModel] = [primary, *fallbacks]

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        last_error: str | None = None

        for bound_model in self._chain:
            provider = bound_model.provider
            mapped_model = bound_model.spec
            # Construction-time failures (rare — most stay inside the producer task).
            try:
                inner = await provider.stream(
                    mapped_model,
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    options=options,
                )
            except Exception as e:
                log.warning("provider %s failed at construction: %s", mapped_model.provider_id, e)
                last_error = repr(e)
                continue

            # Peek at the first event to learn whether the stream is healthy.
            iterator = inner.__aiter__()
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                last_error = "stream ended before producing any events"
                continue

            if first.type == "error":
                log.warning("provider %s errored on first event: %s",
                            mapped_model.provider_id, first.error_message)
                last_error = first.error_message or "stream error"
                continue

            # Healthy — commit to this provider. Forward `first` plus the rest
            # through a fresh outer MessageStream so the caller sees a complete
            # stream starting at the start event.
            outer = MessageStream()

            async def _forward(first_event=first, src=iterator, src_stream=inner):
                try:
                    outer.push(first_event)
                    async for ev in src:
                        outer.push(ev)
                    final = await src_stream.result()
                    outer.set_result(final)
                except Exception as exc:
                    fallback_msg = AssistantMessage(
                        content=[],
                        stop_reason="error",
                        error_message=str(exc),
                        usage=Usage(),
                        timestamp=time.time(),
                    )
                    outer.push(StreamEvent(type="error", error_message=str(exc)))
                    outer.set_result(fallback_msg)

            outer.attach_task(asyncio.create_task(_forward()))
            return outer

        raise RuntimeError(f"all providers exhausted; last error: {last_error!r}")
```

## Use it

```python title="main.py"
import asyncio
import os

from cubepi import Agent
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.openai import OpenAIProvider
from failover import FailoverProvider


async def main():
    anthropic = AnthropicProvider(
        provider_id="anthropic",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    openai = OpenAIProvider(
        provider_id="openai",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    failover = FailoverProvider(
        anthropic.model("claude-sonnet-4-6"),
        openai.model("gpt-5"),
    )

    # The model passed here is overridden inside FailoverProvider; pass any
    # placeholder. We use the primary's so usage tracking labels match the
    # happy path.
    agent = Agent(
        model=failover.model("claude-sonnet-4-6"),
        system_prompt="You answer concisely.",
    )
    agent.subscribe(lambda e, s=None: None)
    await agent.prompt("Capital of Mongolia?")
    last = agent.state.messages[-1]
    print(last.content[0].text)


asyncio.run(main())
```

## What about smarter failover policies?

The example above falls back on **any** error event. That's fine for
`RateLimitError`, `APIConnectionError`, or 5xx — but arguably wrong for
`BadRequestError` (your code is wrong; the next provider will fail the
same way).

The first-event `error_message` comes from `str(exc)` on the
underlying SDK exception. Filter on substrings, or — better — wrap each
provider's `_produce` to tag the error category:

```python
NON_RETRYABLE_HINTS = ("bad request", "invalid_request_error", "401", "403")

if first.type == "error":
    msg = (first.error_message or "").lower()
    if any(h in msg for h in NON_RETRYABLE_HINTS):
        raise RuntimeError(f"non-retryable error from {mapped_model.provider_id}: {msg}")
    last_error = first.error_message
    continue
```

A more robust approach is to fork the built-in providers and re-raise
specific SDK exceptions from `_produce` so they reach `provider.stream()`
as real Python exceptions — but that's a larger change against
CubePi itself.

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
- **Streaming consistency** — The wrapper forwards events through a
  fresh `MessageStream`, so consumers see the same `StreamEvent` shape
  regardless of which provider answered. The original `start` event
  comes from the inner provider unchanged.
- **Mid-stream errors aren't recovered** — Once we've seen a healthy
  first event, the wrapper commits to that provider. If it errors
  halfway through a long response, the agent sees the error. Full
  mid-stream replay would require buffering — out of scope here.

## See also

- [Providers / Anthropic](../guides/providers/anthropic) and
  [OpenAI](../guides/providers/openai) — provider-specific details.
- [Writing a Custom Provider](../guides/providers/custom) — the same
  Protocol used by this wrapper.
