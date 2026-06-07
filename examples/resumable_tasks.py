"""Resumable tasks — recipe example.

Demonstrates crash-resilient long tasks with checkpointing. Tools are
idempotent (backed by a file-based job store) so re-running after a crash
skips already-completed work.

    # Start a job:
    uv run python examples/resumable_tasks.py job-1 start

    # Kill mid-flight with Ctrl-C, then resume:
    uv run python examples/resumable_tasks.py job-1

Requires: cubepi[sqlite]
Set ANTHROPIC_API_KEY or OPENAI_API_KEY before running (see _provider.py).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from cubepi import Agent, AgentToolResult, TextContent, tool
from cubepi.checkpointer import SQLiteCheckpointer

from _provider import MODEL_ID, provider

JOB_DIR = Path(os.environ.get("JOB_DIR", "/tmp/cubepi-jobs"))
JOB_DIR.mkdir(parents=True, exist_ok=True)


@tool(execution_mode="sequential")
async def process_item(item_id: str, *, signal=None) -> AgentToolResult:
    "Process a work item. Idempotent — safe to retry after a crash."
    job_file = JOB_DIR / f"item-{item_id}.json"

    if job_file.exists():
        state = json.loads(job_file.read_text())
        return AgentToolResult(
            content=[TextContent(text=f"Item {item_id} already done: {state['result']}.")],
        )

    # Simulate slow work (1 second per item).
    print(f"  [processing item {item_id}...]")
    await asyncio.sleep(1)

    result = f"processed-{item_id}"
    job_file.write_text(json.dumps({"item_id": item_id, "result": result}))

    return AgentToolResult(
        content=[TextContent(text=f"Item {item_id} done: {result}.")],
    )


async def main(thread_id: str, start: bool) -> None:
    async with SQLiteCheckpointer("resumable_jobs.db") as cp:
        agent = Agent(
            model=provider.model(MODEL_ID),
            system_prompt=(
                "You orchestrate batch processing jobs. "
                "Process all items one by one using the process_item tool, "
                "then summarize the results."
            ),
            tools=[process_item],
            checkpointer=cp,
            thread_id=thread_id,
        )

        def on_event(event, signal=None):
            if event.type == "message_update" and event.stream_event.type == "text_delta":
                print(event.stream_event.delta, end="", flush=True)
            elif event.type == "tool_execution_start":
                print(f"\n[tool: {event.tool_name}({event.args})]")
            elif event.type == "agent_end":
                print()

        agent.subscribe(on_event)

        if start:
            await agent.prompt("Process items A, B, C, and D.")
        else:
            # Resume: hydrate state from checkpointer then resume.
            data = await cp.load(thread_id)
            if data is None:
                print(f"No saved state for thread {thread_id!r}. Use 'start' to begin.")
                return
            agent.state.messages = list(data.messages)
            last = agent.state.messages[-1] if agent.state.messages else None
            last_type = type(last).__name__ if last else "none"
            print(f"Resuming thread {thread_id!r} ({len(agent.state.messages)} messages, last={last_type})")
            if last_type == "AssistantMessage":
                from cubepi.providers.base import ToolCall
                has_pending_tools = any(isinstance(c, ToolCall) for c in last.content)
                if not has_pending_tools:
                    # Run completed normally — no pending tool calls to execute.
                    # In a real workflow you'd ask the user for the next prompt.
                    print("Run already completed. Nothing to resume.")
                    return
                # Has unresolved tool calls — resume() will execute them.
            await agent.resume()


if __name__ == "__main__":
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "job-1"
    start = len(sys.argv) > 2 and sys.argv[2] == "start"
    asyncio.run(main(thread_id, start))
