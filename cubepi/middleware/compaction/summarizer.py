from __future__ import annotations

import asyncio
import json

from cubepi.middleware.compaction.state import CompactionState, message_refs
from cubepi.providers.base import (
    BoundModel,
    Message,
    StreamOptions,
    TextContent,
    ToolCall,
    UserMessage,
)

_ARG_VALUE_CHARS = 200
_ARG_REPR_MAX = 500


SUMMARIZER_SYSTEM_PROMPT = """\
You compress a chat transcript into a brief, faithful narrative for an AI assistant
that is continuing the conversation. Rules:

1. Preserve facts, user goals, decisions made, and unresolved questions.
2. Preserve every citation marker verbatim. Do not renumber, merge, or drop them.
3. Do not quote long tool outputs. Reference them by their citation markers instead.
4. Keep the language of the original conversation.
5. Output the summary directly. No preamble, no JSON, no markdown headers.
"""

EXISTING_SUMMARY_SUFFIX = """\
A previous summary already covers earlier turns:

<previous_summary>
{prev}
</previous_summary>

Merge it with the new turns below. Output the updated summary."""


def _shrink_strings(obj: object) -> object:
    """Recursively shrink long string leaves; preserve everything else."""
    if isinstance(obj, str):
        if len(obj) <= _ARG_VALUE_CHARS:
            return obj
        return obj[:_ARG_VALUE_CHARS] + "...[truncated]"
    if isinstance(obj, dict):
        return {k: _shrink_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shrink_strings(v) for v in obj]
    return obj


def _format_arguments(arguments: object) -> str:
    """Render a ToolCall arguments dict as a compact JSON-ish suffix.

    Long string field values are shrunk individually so short fields (file
    paths, command names, query strings) survive intact. The full repr is
    additionally capped at ``_ARG_REPR_MAX`` chars as a last-line defence
    against pathological inputs (e.g. thousands of small fields).
    """
    if not arguments:
        return ""
    try:
        shrunk = _shrink_strings(arguments)
        serialised = json.dumps(shrunk, ensure_ascii=False)
    except (TypeError, ValueError):
        serialised = str(arguments)
    if len(serialised) > _ARG_REPR_MAX:
        serialised = serialised[:_ARG_REPR_MAX] + "..."
    return " " + serialised


def _format_message_for_summary(message: Message) -> str:
    role = message.__class__.__name__.removesuffix("Message").lower() or "message"
    parts: list[str] = []
    for block in getattr(message, "content", []):
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolCall):
            parts.append(f"[tool_call:{block.name}]{_format_arguments(block.arguments)}")
        elif hasattr(block, "text"):
            parts.append(str(getattr(block, "text", "")))
    return f"[{role}] " + " ".join(parts)


def _format_transcript(messages: list[Message]) -> str:
    return "\n\n".join(_format_message_for_summary(message) for message in messages)


async def summarize(
    *,
    model: BoundModel,
    messages_to_summarize: list[Message],
    existing: CompactionState | None,
    max_summary_tokens: int = 1024,
    abort_signal: asyncio.Event | None = None,
) -> CompactionState:
    system_prompt = SUMMARIZER_SYSTEM_PROMPT
    if existing and existing.summary:
        system_prompt += "\n\n" + EXISTING_SUMMARY_SUFFIX.format(prev=existing.summary)

    response = await model.generate(
        messages=[
            UserMessage(
                content=[TextContent(text=_format_transcript(messages_to_summarize))]
            )
        ],
        system_prompt=system_prompt,
        options=StreamOptions(signal=abort_signal),
        max_output_tokens=max_summary_tokens,
        temperature=0.0,
        thinking="off",
    )

    text = "".join(
        block.text for block in response.content if isinstance(block, TextContent)
    )
    if response.error_message is not None:
        raise RuntimeError(response.error_message)

    new_ids = [
        str(getattr(message, "id", "") or "") for message in messages_to_summarize
    ]
    new_ids = [message_id for message_id in new_ids if message_id]
    prior_ids = list(existing.summarized_message_ids) if existing else []
    prior_refs = list(existing.summarized_message_refs) if existing else []
    last_id = (
        new_ids[-1]
        if new_ids
        else (existing.last_summarized_message_id if existing else None)
    )

    return CompactionState(
        summary=text.strip(),
        summarized_message_ids=prior_ids + new_ids,
        summarized_message_refs=prior_refs + message_refs(messages_to_summarize),
        last_summarized_message_id=last_id,
    )
