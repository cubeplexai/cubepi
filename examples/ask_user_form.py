"""Ask-user form — recipe example.

Demonstrates the ask_user HITL tool: the agent pauses and asks the user a
structured multi-question form before proceeding.

    uv run python examples/ask_user_form.py

The host loop answers the form programmatically (simulating a UI). In a real
app you'd render each question to a frontend and collect the response.

Uses the Ark coding endpoint by default — override via CUBEPI_* env vars (see _provider.py).
"""

import asyncio

from cubepi import Agent
from cubepi.hitl import InMemoryChannel, ask_user_tool

from _provider import MODEL_ID, provider


async def host(channel: InMemoryChannel) -> None:
    """Simulate a UI that answers the agent's form questions."""
    async for req in channel.subscribe():
        if req.payload.kind == "ask":
            print("\n[Form received from agent]")
            answers: dict[str, object] = {}
            for q in req.payload.questions:
                if q.options is None:
                    # Free-text: just echo the key as a placeholder answer.
                    answers[q.key] = f"demo-{q.key}"
                    print(f"  {q.prompt!r} → {answers[q.key]!r}")
                elif q.multi_select:
                    # Multi-select: pick first two options.
                    picks = [o.value for o in q.options[:2]]
                    answers[q.key] = picks
                    print(f"  {q.prompt!r} → {picks}")
                else:
                    # Single-select: pick first option.
                    answers[q.key] = q.options[0].value
                    print(f"  {q.prompt!r} → {q.options[0].label!r}")
            print()
            await channel.answer(req.question_id, answers)


async def main() -> None:
    channel = InMemoryChannel()

    agent = Agent(
        model=provider.model(MODEL_ID),
        system_prompt=(
            "When you need structured input from the user, use the ask_user tool. "
            "Ask for project type, framework, desired features, and project name "
            "before scaffolding anything."
        ),
        tools=[ask_user_tool(channel)],
        channel=channel,
    )

    collected: list[str] = []

    def on_event(event, signal=None):
        if event.type == "message_update" and event.stream_event.type == "text_delta":
            collected.append(event.stream_event.delta)
        elif event.type == "tool_execution_start":
            print(f"[tool: {event.tool_name}]")
        elif event.type == "agent_end":
            print()

    agent.subscribe(on_event)

    host_task = asyncio.create_task(host(channel))
    try:
        await agent.prompt("Scaffold a new web project for me.")
        print("Agent response:", "".join(collected))
    finally:
        host_task.cancel()
        try:
            await host_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
