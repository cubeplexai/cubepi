from __future__ import annotations

from collections.abc import Callable

from cubepi.providers.base import Message, TextContent, ToolResultMessage

_PRUNE_KEEP_CHARS = 120

ToolResultCompressor = Callable[[ToolResultMessage], str | None]


def prune_tool_results(
    messages: list[Message],
    *,
    tail_start: int,
    compressor: ToolResultCompressor | None = None,
) -> tuple[list[Message], dict[int, str]]:
    """Replace old ToolResultMessage content with a compact one-liner.

    Messages at indices ``>= tail_start`` are the protected tail and are left
    intact. Among ``messages[:tail_start]``, results whose text content is
    already short (<= ``_PRUNE_KEEP_CHARS`` chars) are also kept as-is —
    pruning them would not save tokens worth the loss of context.

    When *compressor* is provided it is called for each candidate message.
    Returning a ``str`` marks that tool result as **preserved** — the text
    will be attached verbatim to the compaction summary (handled by the
    caller). Returning ``None`` falls through to the default pruning logic.

    Returns:
        A ``(pruned_messages, preserved)`` tuple.  ``preserved`` maps the
        original message index to the text returned by the compressor.

    Input is never mutated; new messages are produced via ``model_copy``.
    """
    if tail_start <= 0:
        return list(messages), {}

    result: list[Message] = []
    preserved: dict[int, str] = {}
    for i, msg in enumerate(messages):
        if i >= tail_start or not isinstance(msg, ToolResultMessage):
            result.append(msg)
            continue

        if compressor is not None:
            action = compressor(msg)
            if action is not None:
                preserved[i] = action
                summary = f"[{msg.tool_name}] preserved"
                result.append(
                    msg.model_copy(update={"content": [TextContent(text=summary)]})
                )
                continue

        text = _extract_text(msg)
        if len(text) <= _PRUNE_KEEP_CHARS:
            result.append(msg)
            continue

        summary = f"[{msg.tool_name}] {len(text)} chars"
        result.append(msg.model_copy(update={"content": [TextContent(text=summary)]}))

    return result, preserved


def _extract_text(msg: ToolResultMessage) -> str:
    parts: list[str] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "\n".join(parts)
