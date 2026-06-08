from __future__ import annotations

import logging
import asyncio
from typing import Any

from pydantic import ValidationError

from cubepi.agent.types import AgentContext
from cubepi.middleware.base import Middleware
from cubepi.middleware.compaction.boundary import safe_boundary
from cubepi.middleware.compaction.state import CompactionState, message_refs
from cubepi.middleware.compaction.summarizer import summarize
from cubepi.middleware.compaction.tokens import approx_tokens
from cubepi.providers.base import (
    BoundModel,
    Message,
    TextContent,
    UserMessage,
)

SUMMARY_PREFIX = "[Conversation summary so far]\n"
logger = logging.getLogger(__name__)


def _compressed_view(
    messages: list[Message],
    state: CompactionState | None,
    boundary: int | None,
) -> list[Message]:
    if state and boundary and boundary > 0:
        summary = UserMessage(
            content=[TextContent(text=SUMMARY_PREFIX + state.summary)],
        )
        return [summary, *messages[boundary:]]
    return list(messages)


def _load_state(value: Any) -> CompactionState | None:
    if value is None:
        return None
    if isinstance(value, CompactionState):
        return value
    if isinstance(value, dict):
        try:
            return CompactionState.model_validate(value)
        except ValidationError:
            return None
    return None


def _clear_state(ctx: AgentContext) -> None:
    ctx.extra.pop("compaction", None)
    ctx.extra.pop("compaction_until_msg_index", None)


def _state_matches_history(
    messages: list[Message],
    state: CompactionState | None,
    boundary: int,
) -> bool:
    if state is None or boundary <= 0:
        return True
    refs = state.summarized_message_refs
    if len(refs) != boundary:
        return False
    return refs == message_refs(messages[:boundary])


class CompactionMiddleware(Middleware):
    """Keep long histories within context by summarizing older turns."""

    def __init__(
        self,
        *,
        summary_model: BoundModel,
        max_tokens_before_compact: int,
        keep_recent_messages: int = 8,
        max_summary_tokens: int = 1024,
        min_compact_messages: int = 4,
    ) -> None:
        self._summary_model = summary_model
        self._max_tokens_before = max_tokens_before_compact
        self._keep_recent = keep_recent_messages
        self._max_summary_tokens = max_summary_tokens
        self._min_compact = min_compact_messages

    async def transform_context(
        self,
        messages: list[Message],
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> list[Message]:
        state = _load_state(ctx.extra.get("compaction"))
        raw_boundary = ctx.extra.get("compaction_until_msg_index")
        boundary = (
            int(raw_boundary) if isinstance(raw_boundary, (int, float, str)) else 0
        )
        if state is None and ("compaction" in ctx.extra or boundary > 0):
            boundary = 0
            _clear_state(ctx)
        if boundary >= len(messages) or not _state_matches_history(
            messages, state, boundary
        ):
            boundary = 0
            state = None
            _clear_state(ctx)
        compressed = _compressed_view(messages, state, boundary)

        if approx_tokens(compressed) < self._max_tokens_before:
            return compressed

        new_boundary = safe_boundary(
            messages,
            keep_recent=self._keep_recent,
            min_compact=max(self._min_compact, boundary + 1),
        )
        if new_boundary is None or new_boundary <= boundary:
            return compressed

        try:
            new_state = await summarize(
                model=self._summary_model,
                messages_to_summarize=messages[boundary:new_boundary],
                existing=state,
                max_summary_tokens=self._max_summary_tokens,
                abort_signal=signal,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CompactionMiddleware summarizer failed, skipping: %s", exc)
            return compressed

        ctx.extra["compaction"] = new_state.model_dump()
        ctx.extra["compaction_until_msg_index"] = new_boundary
        return _compressed_view(messages, new_state, new_boundary)

    def extra_llm_calls(self) -> tuple[BoundModel, ...]:
        # Surface the bound summary model so ``cubepi.tracing.Recorder`` can
        # both subscribe its listeners (the summarizer's chat span lands in
        # the trace) AND identify the summary call by spec — important when
        # the summary model's provider is the same instance as the agent's
        # main provider, the common "reuse the client, swap the model"
        # pattern.
        return (self._summary_model,)


__all__ = [
    "CompactionMiddleware",
    "CompactionState",
]
