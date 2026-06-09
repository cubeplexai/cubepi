"""GoalMiddleware — autonomous goal-driven agent runs.

Hooks used:
    transform_context — detect /goal prefix, extract condition, rewrite message
    on_run_end        — evaluate condition via evaluator model, inject feedback
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from cubepi.agent.types import AgentContext
from cubepi.middleware.base import Middleware
from cubepi.providers.base import (
    BoundModel,
    Message,
    TextContent,
    UserMessage,
)

_GOAL_PREFIX = "/goal "

_MESSAGE_WINDOW = 20


class _GoalResult(BaseModel):
    achieved: bool
    reason: str


_EVALUATOR_PROMPT = """\
You are evaluating whether an agent has achieved its goal.

Goal condition:
{condition}

Recent conversation (last {window} messages):
{messages}

Has the goal been achieved? Use the structured_output tool to respond."""


def _format_messages_for_eval(messages: list[Message], window: int) -> str:
    tail = messages[-window:]
    parts: list[str] = []
    for msg in tail:
        role = getattr(msg, "role", "unknown")
        texts: list[str] = []
        for block in getattr(msg, "content", []):
            if isinstance(block, TextContent):
                texts.append(block.text)
        if texts:
            parts.append(f"[{role}] {' '.join(texts)}")
    return "\n".join(parts)


class GoalMiddleware(Middleware):
    def __init__(
        self,
        evaluator: BoundModel,
        max_evaluations: int = 10,
    ) -> None:
        self._evaluator = evaluator
        self._max_evaluations = max_evaluations
        self._condition: str | None = None
        self._evaluations = 0

    def extra_llm_calls(self) -> Iterable[BoundModel]:
        return (self._evaluator,)

    async def transform_context(
        self,
        messages: list[Message],
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> list[Message]:
        if not messages:
            return messages
        last = messages[-1]
        if not isinstance(last, UserMessage):
            return messages
        if not last.content:
            return messages
        first_block = last.content[0]
        if not isinstance(first_block, TextContent):
            return messages
        if not first_block.text.startswith(_GOAL_PREFIX):
            return messages

        condition = first_block.text[len(_GOAL_PREFIX) :].strip()
        if not condition:
            return messages
        self._condition = condition
        self._evaluations = 0

        rewritten = last.model_copy(update={"content": [TextContent(text=condition)]})
        # Mutate ctx.messages so the stripped form is what gets stored in
        # agent state — the /goal prefix is a one-way control signal that
        # should not persist in the conversation history.
        ctx.messages[-1] = rewritten
        return [*messages[:-1], rewritten]

    async def on_run_end(
        self,
        ctx: AgentContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> list[Message] | None:
        if self._condition is None:
            saved = ctx.extra.get("goal")
            if not isinstance(saved, dict) or saved.get("status") != "active":
                return None
            self._condition = saved["condition"]
            self._evaluations = saved.get("evaluations", 0)

        self._evaluations += 1

        transcript = _format_messages_for_eval(ctx.messages, _MESSAGE_WINDOW)
        prompt = _EVALUATOR_PROMPT.format(
            condition=self._condition,
            window=_MESSAGE_WINDOW,
            messages=transcript,
        )

        result = await self._evaluator.generate_structured(
            _GoalResult,
            messages=[UserMessage(content=[TextContent(text=prompt)])],
            system_prompt="",
            temperature=0.0,
        )

        goal_state: dict[str, Any] = {
            "condition": self._condition,
            "evaluations": self._evaluations,
            "max_evaluations": self._max_evaluations,
            "last_reason": result.reason,
        }

        if result.achieved:
            goal_state["status"] = "achieved"
            ctx.extra["goal"] = goal_state
            return None

        if self._evaluations >= self._max_evaluations:
            goal_state["status"] = "exhausted"
            ctx.extra["goal"] = goal_state
            return None

        goal_state["status"] = "active"
        ctx.extra["goal"] = goal_state
        return [
            UserMessage(
                content=[
                    TextContent(
                        text=f"Goal not yet met: {result.reason}. Continue working."
                    )
                ]
            )
        ]
