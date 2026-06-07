"""Multi-provider failover — recipe example.

Demonstrates a FailoverProvider wrapper that peeks at the first stream event
from each inner provider and falls over to the next on error.

    uv run python examples/multi_provider_failover.py

The example constructs a primary provider with a bad key to force a failover to
the real provider, proving the mechanism works end-to-end.

Set ANTHROPIC_API_KEY or OPENAI_API_KEY (and optionally MODEL) before running.
See _provider.py for details.
"""

import asyncio
import logging
import os
import time

from cubepi import Agent
from cubepi.providers.openai import OpenAIProvider
from cubepi.providers.anthropic import AnthropicProvider
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

from _provider import MODEL_ID, provider

log = logging.getLogger(__name__)


class FailoverProvider(BaseProvider):
    """Try providers in order; fall over on construction or first-event errors.

    Built-in providers swallow API/network errors and surface them as
    StreamEvent(type="error") on the returned stream. We peek at the first
    event from each inner stream and only commit once we see a non-error event.

    Limitation: errors that arrive *after* the first event are forwarded as-is.
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
            inner_provider = bound_model.provider
            mapped_model = bound_model.spec
            try:
                inner = await inner_provider.stream(
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

            iterator = inner.__aiter__()
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                last_error = "stream ended before producing any events"
                continue

            if first.type == "error":
                log.warning(
                    "provider %s errored on first event: %s",
                    mapped_model.provider_id,
                    first.error_message,
                )
                last_error = first.error_message or "stream error"
                continue

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


def _bad_key_provider() -> tuple[BaseProvider, str]:
    """Build a provider identical to the real one but with an invalid key."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
        return AnthropicProvider(api_key="bad-key", base_url=base_url), MODEL_ID
    else:
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        return OpenAIProvider(api_key="bad-key", base_url=base_url), MODEL_ID


async def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    bad_provider, model_id = _bad_key_provider()

    # Primary: bad key → errors on first event → triggers failover.
    # Fallback: real provider from _provider.py.
    failover = FailoverProvider(
        bad_provider.model(model_id),
        provider.model(MODEL_ID),
    )

    agent = Agent(
        model=failover.model(MODEL_ID),
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
