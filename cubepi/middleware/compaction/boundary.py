from __future__ import annotations

from cubepi.middleware.compaction.tokens import approx_tokens
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def tail_start_by_tokens(messages: list[Message], budget: int) -> int:
    """Walk backward accumulating token estimates; return where the tail starts.

    Contract:
    - Empty input → return 0.
    - Non-empty input → return an index in ``[0, len(messages) - 1]``.
    - Walk backward, summing ``approx_tokens([msg])``. Return the first index
      whose inclusion would push the accumulated total *strictly over* budget,
      *provided* at least one message is already in the tail. (Equal-to-budget
      is acceptable.) If the last message alone exceeds budget, it is still
      included — the tail always contains ≥ 1 message for non-empty input.
    """
    if not messages:
        return 0
    accumulated = 0
    for i in range(len(messages) - 1, -1, -1):
        msg_tokens = approx_tokens([messages[i]])
        if accumulated + msg_tokens > budget and accumulated > 0:
            return i + 1
        accumulated += msg_tokens
    return 0


def safe_boundary(
    messages: list[Message],
    *,
    tail_start: int,
    min_compact: int = 1,
) -> int | None:
    """Return a message index that can be summarised safely.

    ``tail_start`` is the precomputed protection boundary (call
    :func:`tail_start_by_tokens` first). Messages at ``[tail_start, end)``
    are the protected tail; this function searches the prefix for the latest
    ``UserMessage`` whose suffix has no orphaned tool-call/result pairs.

    Returns ``None`` when:
    - ``tail_start`` is out of range (negative, zero, or beyond ``messages``)
    - no candidate satisfies the tool-call self-containment rule
    - the candidate would be smaller than ``min_compact``
    """
    if tail_start <= 0 or tail_start > len(messages):
        return None

    # Start the search at the first message inside the tail (we want to
    # split BEFORE the tail begins).
    candidate = tail_start
    if candidate == len(messages):
        candidate -= 1

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
