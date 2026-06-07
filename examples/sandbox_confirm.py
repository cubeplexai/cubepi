"""Sandbox confirm — recipe example.

Demonstrates ApprovalPolicyMiddleware: a policy function classifies each
tool call as auto-allow, hard-deny, or human-confirm. The host loop handles
human-confirm requests by auto-approving them (simulating a UI).

    uv run python examples/sandbox_confirm.py

Set ANTHROPIC_API_KEY or OPENAI_API_KEY before running (see _provider.py).
"""

import asyncio
from pydantic import BaseModel

from cubepi import Agent, AgentToolResult, TextContent, tool
from cubepi.hitl import (
    Approve,
    AskUser,
    Deny,
    ApprovalPolicyMiddleware,
    InMemoryChannel,
    ApproveAnswer,
)

from _provider import MODEL_ID, provider


# --- Simulated bash tool ------------------------------------------------

class BashInput(BaseModel):
    cmd: str


@tool
async def bash(cmd: str) -> AgentToolResult:
    "Run a shell command (simulated — not a real shell)."
    print(f"  [bash executing: {cmd!r}]")
    return AgentToolResult(content=[TextContent(text=f"$ {cmd}\nok")])


# --- Policy function ----------------------------------------------------

def sandbox_policy(ctx):
    cmd: str = ctx.args.cmd
    if cmd.startswith(("ls", "cat", "head", "grep", "find", "echo")):
        print(f"  [policy: auto-allow {cmd!r}]")
        return Approve()
    if "rm -rf /" in cmd or cmd.startswith("dd "):
        print(f"  [policy: hard-deny {cmd!r}]")
        return Deny(reason="destructive I/O blocked by policy")
    print(f"  [policy: ask-user for {cmd!r}]")
    return AskUser(timeout_seconds=60, details={"cmd": cmd})


# --- Host loop ----------------------------------------------------------

async def host(channel: InMemoryChannel) -> None:
    async for req in channel.subscribe():
        if req.payload.kind == "approve":
            print(f"  [host: auto-approving tool={req.payload.tool_name}]")
            await channel.answer(
                req.question_id,
                ApproveAnswer(decision="approve"),
            )


# --- Main ---------------------------------------------------------------

async def main() -> None:
    channel = InMemoryChannel()

    agent = Agent(
        model=provider.model(MODEL_ID),
        system_prompt=(
            "You have access to a bash tool. "
            "Run: echo hello, then ls /tmp, then rm -rf /important (to show deny), "
            "then cat /etc/hostname. Each as a separate tool call."
        ),
        tools=[bash],
        middleware=[ApprovalPolicyMiddleware(channel, policy=sandbox_policy)],
        channel=channel,
    )

    def on_event(event, signal=None):
        if event.type == "message_update" and event.stream_event.type == "text_delta":
            print(event.stream_event.delta, end="", flush=True)
        elif event.type == "tool_execution_start":
            print(f"\n[tool: {event.tool_name}]")
        elif event.type == "agent_end":
            print()

    agent.subscribe(on_event)

    host_task = asyncio.create_task(host(channel))
    try:
        await agent.prompt("Run all four commands and summarize the results.")
    finally:
        host_task.cancel()
        try:
            await host_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
