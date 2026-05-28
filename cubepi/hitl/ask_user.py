from __future__ import annotations

import json

from pydantic import BaseModel

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.hitl.channel import HitlChannel
from cubepi.hitl.types import Option, Question
from cubepi.providers.base import TextContent


class _OptionDef(BaseModel):
    label: str
    value: str
    description: str | None = None
    allow_input: bool = False


class _QuestionDef(BaseModel):
    key: str
    prompt: str
    options: list[_OptionDef] | None = None
    multi_select: bool = False
    required: bool = True


class AskUserParams(BaseModel):
    questions: list[_QuestionDef]


_DESCRIPTION = (
    "Ask the user one or more structured questions. Use ONLY when you need "
    "a specific selection or piece of info to proceed; for free-form clarification, "
    "just end your turn with the question as text — the user's next message will be the answer."
)


def _format_answers(answers: dict) -> str:
    return "User answers:\n" + json.dumps(answers, indent=2, ensure_ascii=False)


def ask_user_tool(channel: HitlChannel) -> AgentTool:
    async def execute(
        call_id: str, args: AskUserParams, *, signal=None, on_update=None
    ) -> AgentToolResult:
        from cubepi.hitl.exceptions import HitlCancelled, HitlTimedOut

        questions = [
            Question(
                key=q.key,
                prompt=q.prompt,
                options=[Option(**o.model_dump()) for o in q.options]
                if q.options
                else None,
                multi_select=q.multi_select,
                required=q.required,
            )
            for q in args.questions
        ]
        # Per spec §7: cancel/timeout in ask_user context surface as
        # tool_result.is_error=True so the model can react. Other HITL control
        # exceptions (HitlDetached, HitlAborted) DO propagate — those signal
        # whole-agent state changes that must reach the loop's outer catch.
        try:
            answers = await channel.ask(questions, signal=signal)
        except HitlCancelled as exc:
            return AgentToolResult(
                content=[TextContent(text=f"cancelled by user: {exc.reason}")],
                details={"hitl": {"outcome": "cancelled", "reason": exc.reason}},
                is_error=True,
            )
        except HitlTimedOut as exc:
            return AgentToolResult(
                content=[TextContent(text=f"timed out after {exc.seconds} seconds")],
                details={"hitl": {"outcome": "timed_out", "seconds": exc.seconds}},
                is_error=True,
            )
        return AgentToolResult(
            content=[TextContent(text=_format_answers(answers))],
            details={"hitl": {"kind": "ask", "answers": answers}},
        )

    tool = AgentTool(
        name="ask_user",
        description=_DESCRIPTION,
        parameters=AskUserParams,
        execute=execute,
        execution_mode="sequential",
    )
    # Signal to _execute_prepared that this is a built-in HITL tool so the
    # ContextVar durability guard is NOT set on entry — only custom tool
    # bodies trigger CheckpointedChannel's HitlDurabilityNotGuaranteed.
    tool._hitl_builtin = True
    return tool
