---
title: Writing a Custom Provider
description: "Write a custom provider for CubePi by implementing the Provider protocol."
---

# Writing a Custom Provider

A provider is any class with one method:

```python
class Provider(Protocol):
    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream: ...
```

That's the whole interface. Implement it and `Agent(provider=…)`
accepts your class.

This page covers two scenarios:

1. **A new real provider** — Bedrock, Vertex, Replicate, an internal
   LLM gateway, …
2. **`FauxProvider`** — the built-in deterministic provider that's
   essential for unit tests.

## A minimal real provider

The pattern: create a `MessageStream`, kick off a producer task that
pushes events into it, return the stream immediately.

```python
import asyncio
import time
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    MessageStream,
    Model,
    StreamEvent,
    StreamOptions,
    TextContent,
    ToolDefinition,
    Usage,
)


class MyProvider:
    def __init__(self, *, api_key: str) -> None:
        self._api_key = api_key

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        opts = options or StreamOptions()
        ms = MessageStream()

        async def _produce():
            try:
                partial = AssistantMessage(
                    content=[TextContent(text="")],
                    usage=Usage(),
                    timestamp=time.time(),
                    provider_id=model.provider,
                    model_id=model.id,
                )
                ms.push(StreamEvent(type="start", partial=partial.model_copy(deep=True)))

                # Call your backend. Stream tokens:
                async for token in call_my_backend(messages, model.id, signal=opts.signal):
                    if opts.signal and opts.signal.is_set():
                        ms.push(StreamEvent(type="error", error_message="aborted"))
                        ms.set_result(partial.model_copy(update={"stop_reason": "aborted"}))
                        return
                    partial.content[-1] = TextContent(text=partial.content[-1].text + token)
                    ms.push(StreamEvent(
                        type="text_delta",
                        delta=token,
                        partial=partial.model_copy(deep=True),
                    ))

                ms.push(StreamEvent(type="done"))
                ms.set_result(partial)

            except Exception as exc:
                error_msg = AssistantMessage(
                    content=[],
                    stop_reason="error",
                    error_message=str(exc),
                    usage=Usage(),
                    timestamp=time.time(),
                )
                ms.push(StreamEvent(type="error", error_message=str(exc)))
                ms.set_result(error_msg)

        ms.attach_task(asyncio.create_task(_produce()))
        return ms
```

Things to get right:

1. **Always push `start` first.** Subscribers count on it for UI
   bootstrapping.
2. **Always end with `done` or `error`.** The agent loop waits on the
   `MessageStream` until one of those arrives.
3. **Always call `ms.set_result(...)`** so `await stream.result()` can
   complete. Even on error.
4. **`ms.attach_task(...)` is required** if the producer is its own
   task — it wires the task's exception state into the stream so a
   crash surfaces as `error`, not a hang.
5. **Respect `opts.signal`.** Check it inside your read loop; emit an
   `aborted` stop_reason so the agent can shut down cleanly.

## Supporting tool calls

If your model emits tool calls, append `ToolCall` blocks to
`partial.content` as they stream in and emit `toolcall_start` /
`toolcall_delta` / `toolcall_end`:

```python
from cubepi.providers.base import ToolCall

tc = ToolCall(id=block_id, name=tool_name, arguments={})
partial.content.append(tc)
ms.push(StreamEvent(type="toolcall_start", content_index=len(partial.content)-1,
                    partial=partial.model_copy(deep=True)))
# ... as JSON args arrive:
ms.push(StreamEvent(type="toolcall_delta", delta=partial_json_chunk, …))
# ... on completion:
# replace tc.arguments with the parsed dict, push toolcall_end
```

CubePi's agent loop will dispatch the tool calls automatically once
`done` is emitted.

## Hooking `on_payload` / `on_response`

If your provider sends an HTTP request, call the helpers in
`cubepi.providers.base`:

```python
from cubepi.providers.base import (
    ProviderResponse,
    invoke_on_payload,
    invoke_on_response,
)

payload = await invoke_on_payload(opts.on_payload, payload, model)
http_resp = await self._client.post(..., json=payload)
await invoke_on_response(
    opts.on_response,
    ProviderResponse(status=http_resp.status_code, headers=dict(http_resp.headers)),
    model,
)
```

This mirrors what built-in providers do; your users get the same
inspection points for free.

## Using `FauxProvider` in tests

CubePi ships `FauxProvider` for deterministic tests — no network, no
flakiness, real streaming events:

```python
from cubepi import Agent, Model
from cubepi.providers import FauxProvider, faux_assistant_message, faux_text, faux_tool_call


def test_my_agent():
    provider = FauxProvider()
    provider.set_responses([
        faux_assistant_message([
            faux_tool_call("search", {"query": "python"}),
        ]),
        faux_assistant_message("Here are the results: …"),
    ])

    agent = Agent(
        provider=provider,
        model=Model(id="test", provider="faux"),
        tools=[my_search_tool],
    )
    events = []
    agent.subscribe(lambda e, signal=None: events.append(e))
    await agent.prompt("Search for python")

    assert any(e.type == "tool_execution_start" for e in events)
    assert events[-1].type == "agent_end"
```

`set_responses` is a FIFO queue: each model call pops one. The faux
provider replays it with realistic deltas (token-by-token), so your
streaming code paths actually exercise.

Helpers:

- `faux_text("Hello!")` — wraps a string into a `TextContent` block.
- `faux_thinking("Pondering…")` — a `ThinkingContent` block.
- `faux_tool_call("name", {"arg": …})` — a `ToolCall` block.
- `faux_assistant_message(content_or_text)` — builds a complete
  `AssistantMessage`.

## Common pitfalls

- **Missing `start` event** — Subscribers don't see the partial
  message. Always push it first.
- **Forgot `ms.set_result(...)`** — `await agent.prompt()` hangs
  forever. Set the result on both happy and error paths.
- **Synchronous `produce`** — `stream()` must return *immediately* —
  do the work inside the task. If you `await` your backend before
  `return ms`, you've blocked the caller.
- **Modifying `partial` after pushing** — Push `deep=True` copies; the
  caller iterates events asynchronously, and a mutation during
  iteration creates very hard-to-debug aliasing.

## See also

- [Capabilities & Preset Catalog](./capability-and-presets) — before
  writing a class from scratch, check whether your backend is just an
  OpenAI/Anthropic-compatible endpoint a `CapabilityDescriptor` or
  bundled preset already covers.
- [API Reference → providers/base](../../api/cubepi-providers) — the
  full type list.
- [Anthropic Provider source](https://github.com/cubeplexai/cubepi/blob/main/cubepi/providers/anthropic.py)
  — a real, complete example.
- [`FauxProvider` source](https://github.com/cubeplexai/cubepi/blob/main/cubepi/providers/faux.py)
  — the testing primitive, including stream-realism details.
