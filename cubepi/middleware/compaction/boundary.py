from __future__ import annotations

from cubepi.providers.base import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def safe_boundary(
    messages: list[Message],
    *,
    keep_recent: int,
    min_compact: int = 1,
) -> int | None:
    """Return a message index that can be summarized safely."""
    if len(messages) <= keep_recent:
        return None

    candidate = len(messages) - keep_recent
    while candidate > 0:
        if not isinstance(messages[candidate], UserMessage):
            candidate -= 1
            continue
        if not _suffix_is_self_contained(messages[candidate:]):
            candidate -= 1
            continue
        if candidate < min_compact:
            return None
        return candidate

    return None


def _suffix_is_self_contained(suffix: list[Message]) -> bool:
    available_call_ids: set[str] = set()
    for message in suffix:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolCall) and block.id:
                    available_call_ids.add(block.id)
        elif isinstance(message, ToolResultMessage):
            if message.tool_call_id and message.tool_call_id not in available_call_ids:
                return False
    return True
