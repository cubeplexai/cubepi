"""Weather agent — recipe example.

A complete, runnable agent that calls a real weather API as a tool.
Demonstrates HTTP-calling tools, error handling, streaming UI, and cancellation.

    uv run python examples/weather_agent.py

Requires: httpx (pip install httpx)
Uses the Ark coding endpoint by default — override via CUBEPI_* env vars (see _provider.py).
"""

import asyncio
from typing import Annotated

import httpx
from pydantic import Field

from cubepi import Agent, AgentToolResult, TextContent, tool

from _provider import MODEL_ID, provider


@tool
async def get_weather(
    city: Annotated[str, Field(description="The city to look up weather for")],
    units: Annotated[str, Field(pattern="^(metric|imperial)$")] = "metric",
) -> str | AgentToolResult:
    "Get current weather for a city. Returns a short text summary."
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en"},
            )
            geo.raise_for_status()
            results = geo.json().get("results")
            if not results:
                return AgentToolResult(
                    content=[TextContent(text=f"Couldn't find a city named {city!r}.")],
                    is_error=True,
                )
            lat, lon = results[0]["latitude"], results[0]["longitude"]

            wx = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": True,
                    "temperature_unit": "celsius" if units == "metric" else "fahrenheit",
                },
            )
            wx.raise_for_status()
            cw = wx.json()["current_weather"]
            unit = "°C" if units == "metric" else "°F"
            return f"{cw['temperature']}{unit}, wind {cw['windspeed']} km/h in {city}."
        except httpx.HTTPError as e:
            return AgentToolResult(
                content=[TextContent(text=f"Weather API error: {e}")],
                is_error=True,
            )


async def main() -> None:
    agent = Agent(
        model=provider.model(MODEL_ID),
        system_prompt="You are a concise weather assistant. Always use the tool; don't guess.",
        tools=[get_weather],
    )

    def on_event(event, signal=None):
        if event.type == "message_update" and event.stream_event.type == "text_delta":
            print(event.stream_event.delta, end="", flush=True)
        elif event.type == "tool_execution_start":
            print(f"\n[calling {event.tool_name}({event.args})]")
        elif event.type == "agent_end":
            print()

    agent.subscribe(on_event)

    task = asyncio.create_task(agent.prompt("Weather in Tokyo and São Paulo, please."))
    try:
        await task
    except KeyboardInterrupt:
        agent.abort()
        await agent.wait_for_idle()


if __name__ == "__main__":
    asyncio.run(main())
