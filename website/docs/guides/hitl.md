---
title: Human-in-the-Loop (HITL)
sidebar_position: 10
---

# Human-in-the-Loop (HITL)

cubepi ships a HITL channel for two recurring scenarios:

1. **Sandbox tool confirmation** — a dangerous tool needs human approve / deny / edit before running.
2. **Mid-run structured questions** — the agent needs a specific selection or form before proceeding.

The channel is one primitive with two implementations:
- `InMemoryChannel` — for CLI, notebook, tests.
- `CheckpointedChannel` — for web services where the agent process may die between question and answer; pairs with any `Checkpointer`.

## Quick start (in-process)

```python
import asyncio
from cubepi.agent.agent import Agent
from cubepi.hitl import (
    ApproveAnswer, ConfirmToolCallMiddleware, InMemoryChannel, ask_user_tool,
)

channel = InMemoryChannel()

agent = Agent(
    provider=..., model=...,
    tools=[bash_tool, ask_user_tool(channel)],
    middleware=[ConfirmToolCallMiddleware(channel, require_confirm={"bash"})],
    channel=channel,
)

# Host coroutine renders pending requests and posts answers.
# For approve-kind requests, the answer is an ApproveAnswer; for ask-kind it's
# a dict[question.key, str | list[str]]; for confirm-kind it's a bool.
async def host():
    async for req in channel.subscribe():
        if req.payload.kind == "approve":
            user_decision = await my_ui.show_approve(req)   # returns ApproveAnswer
            await channel.answer(req.question_id, user_decision)
        elif req.payload.kind == "ask":
            answers = await my_ui.show_form(req.payload.questions)
            await channel.answer(req.question_id, answers)
        else:  # confirm
            await channel.answer(req.question_id, await my_ui.show_confirm(req))

# Run agent in parallel with host, then exit once the agent finishes.
async def main():
    host_task = asyncio.create_task(host())
    try:
        await agent.prompt("…")
    finally:
        host_task.cancel()

asyncio.run(main())
```

## Cross-process (web service) flow

1. HTTP POST /chat starts `agent.prompt(...)`. Inside, `channel.approve` / `channel.ask` persists `pending_request` to the checkpointer and emits `HitlRequestEvent` on the SSE stream.
2. Frontend renders the pending; user clicks approve/deny/edit.
3. HTTP POST /respond calls `await agent.respond(question_id=..., answer=...)` which loads checkpoint, attaches the answer to the channel, and re-enters the loop. The previously-gated tool runs (or synthetic deny) and the conversation continues. Pending is cleared only after the tool_result is checkpointed.

If the user closes the tab without answering, the host calls `await agent.abort_pending(reason="user closed")` which closes the conversation with a synthetic deny + terminal `stop_reason="aborted"` assistant.

## When to use `ask_user` vs end of turn

| Goal | Use |
|------|-----|
| Free-text follow-up question to user | Just end the turn with the question as text; user's next message is the answer. |
| Structured selection (one of N) | `ask_user` tool with `options` and (optionally) `multi_select` |
| Confirm/edit a tool's args before run | `ConfirmToolCallMiddleware` or `ApprovalPolicyMiddleware` |

## Durable scope

Durable cross-process resume is supported at two safe suspension points:
1. `before_tool_call` approval gate (via `ApprovalPolicyMiddleware` / `ConfirmToolCallMiddleware`)
2. The `ask_user` tool body

Custom tools that mix HITL with other side effects are **same-process only** unless they pass `allow_inside_custom_tool=True` to `CheckpointedChannel` and accept the idempotency contract.
