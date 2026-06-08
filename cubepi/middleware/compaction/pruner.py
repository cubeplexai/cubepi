from __future__ import annotations

from cubepi.providers.base import Message, TextContent, ToolResultMessage

_PRUNE_KEEP_CHARS = 120


def prune_tool_results(messages: list[Message], *, tail_start: int) -> list[Message]:
    """Replace old ToolResultMessage content with a compact one-liner.

    Messages at indices ``>= tail_start`` are the protected tail and are left
    intact. Among ``messages[:tail_start]``, results whose text content is
    already short (<= ``_PRUNE_KEEP_CHARS`` chars) are also kept as-is —
    pruning them would not save tokens worth the loss of context.

    Input is never mutated; new messages are produced via ``model_copy``.
    """
    if tail_start <= 0:
        return list(messages)

    result: list[Message] = []
    for i, msg in enumerate(messages):
        if i >= tail_start or not isinstance(msg, ToolResultMessage):
            result.append(msg)
            continue

        text = _extract_text(msg)
        if len(text) <= _PRUNE_KEEP_CHARS:
            result.append(msg)
            continue

        summary = f"[{msg.tool_name}] {len(text)} chars"
        result.append(msg.model_copy(update={"content": [TextContent(text=summary)]}))

    return result


def _extract_text(msg: ToolResultMessage) -> str:
    parts: list[str] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "\n".join(parts)
