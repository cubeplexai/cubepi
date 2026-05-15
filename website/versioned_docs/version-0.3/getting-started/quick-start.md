---
title: Quick Start
---

# Quick Start

Ship a streaming, tool-using agent in five minutes. We'll build a
weather agent that calls a Python function as a tool, streams Claude's
response token-by-token, and exits cleanly.

## Prerequisites

- Python 3.11+
- `cubepi` installed (`pip install cubepi`)
- An `ANTHROPIC_API_KEY` in your environment

## The full script

Save this as `weather_agent.py`:

```python title="weather_agent.py"
import asyncio
import os
from pydantic import BaseModel

from cubepi import Agent, AgentTool, AgentToolResult, Model, TextContent
from cubepi.providers.anthropic import AnthropicProvider


class GetWeatherParams(BaseModel):
    city: str


async def get_weather(tool_call_id, params: GetWeatherParams, *, signal=None, on_update=None):
    # In a real app: call an HTTP weather API. Here we hard-code a reply.
    return AgentToolResult(
        content=[TextContent(text=f"72°F and sunny in {params.city}")]
    )


async def main():
    provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])

    agent = Agent(
        provider=provider,
        model=Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
        system_prompt="You are a concise weather assistant.",
        tools=[
            AgentTool(
                name="get_weather",
                description="Get current weather for a city",
                parameters=GetWeatherParams,
                execute=get_weather,
            ),
        ],
    )

    # Subscribe BEFORE prompt() — that's how you see streaming events.
    def on_event(event, signal=None):
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
        elif event.type == "agent_end":
            print()  # final newline

    agent.subscribe(on_event)
    await agent.prompt("What's the weather in Tokyo?")


asyncio.run(main())
```

Run it:

```bash
python weather_agent.py
```

You should see Claude stream a sentence like
*"The weather in Tokyo is currently 72°F and sunny…"*, with the
tool result threaded through.

## What just happened

CubePi ran a loop that looks (conceptually) like this:

1. `agent.prompt("What's the weather in Tokyo?")` enqueued a
   `UserMessage` and called the model.
2. The model decided to invoke `get_weather(city="Tokyo")` — CubePi
   parsed the JSON args with your Pydantic model, called your `async
   def`, and fed the result back as a `ToolResultMessage`.
3. The model produced a final assistant response, streamed back as
   `text_delta` events.
4. The loop emitted `agent_end` and returned.

`agent.subscribe(...)` registered a callback for every event the
runtime emits: `agent_start`, `turn_start`, `message_start`,
`text_delta`, `tool_execution_start`, `tool_execution_end`,
`message_end`, `turn_end`, `agent_end`. The script only cared about
`text_delta`, but you can hang any UI off the rest.

## Sources of confusion (read before you debug)

- **Subscribe before you prompt.** Listeners only receive events
  emitted *after* `subscribe` was called. Calling `prompt` first means
  the early events are gone.
- **`Model(id=…, provider=…)` is required.** The agent uses
  `model.provider` to clamp thinking levels and route metadata. The id
  must match a model the provider supports.
- **The `execute` signature is fixed.** Keep `(tool_call_id, params,
  *, signal, on_update)` even if you don't use the keyword-only
  arguments. CubePi always passes them.
- **`agent.prompt()` can only run one prompt at a time.** While it's
  running, use [`agent.steer()`](../guides/agents/multi-turn) to
  inject a follow-up, or `agent.follow_up()` to queue one for after
  the current run.

## Next steps

- [Core Concepts](./core-concepts) — the mental model behind `Agent`,
  `Tool`, `Provider`, `Stream`, `Middleware`, `Checkpointer`.
- [Building Your First Agent](../guides/agents/first-agent) — full
  walkthrough of tools, streaming, and error handling.
- [Tool Use & Parallel Execution](../guides/agents/tool-use) — make
  the agent fan out across multiple tools at once.
- [Recipes → Weather Agent](../recipes/weather-agent) — a slightly
  beefier version of this script with real HTTP calls.
