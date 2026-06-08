"""Multi-provider failover — recipe example.

Demonstrates FallbackBoundModel: the primary provider is given a bad key so its
first stream event is an auth error, which triggers transparent failover to the
real provider.

    uv run python examples/multi_provider_failover.py

Set ANTHROPIC_API_KEY or OPENAI_API_KEY (and optionally MODEL) before running.
See _provider.py for details.
"""

from __future__ import annotations

import asyncio
import os

from cubepi import Agent, FallbackBoundModel
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.base import BoundModel
from cubepi.providers.openai import OpenAIProvider

from _provider import MODEL_ID, provider


def _bad_key_primary() -> BoundModel:
    """Build a bound model with a bad API key to force a first-event auth error."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
        return AnthropicProvider(api_key="bad-key", base_url=base_url).model(MODEL_ID)
    else:
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        return OpenAIProvider(api_key="bad-key", base_url=base_url).model(MODEL_ID)


async def main() -> None:
    # Primary has a bad key → first stream event will be type="error" → failover.
    # Fallback uses the real provider from _provider.py.
    model = FallbackBoundModel(
        chain=(
            _bad_key_primary(),
            provider.model(MODEL_ID),
        ),
        on_failover=lambda failed, nxt, err: print(
            f"[failover] {failed.spec.provider_id}/{failed.spec.id}"
            f" → {nxt.spec.provider_id}/{nxt.spec.id if nxt else '—'}: {err}"
        ),
    )

    agent = Agent(
        model=model,
        system_prompt="You answer concisely.",
    )

    collected: list[str] = []

    def on_event(event, signal=None):
        if event.type == "message_update" and event.stream_event.type == "text_delta":
            collected.append(event.stream_event.delta)

    agent.subscribe(on_event)

    print("Sending request (primary has bad key — expecting failover to secondary)...")
    await agent.prompt("Capital of Mongolia?")
    print("Answer:", "".join(collected))


if __name__ == "__main__":
    asyncio.run(main())
