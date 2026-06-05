from __future__ import annotations

import asyncio

from cubepi.middleware.compaction.state import CompactionState, message_refs
from cubepi.providers.base import (
    Message,
    Model,
    Provider,
    StreamOptions,
    TextContent,
    ToolCall,
    UserMessage,
)


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


def _format_message_for_summary(message: Message) -> str:
    role = message.__class__.__name__.removesuffix("Message").lower() or "message"
    parts: list[str] = []
    for block in getattr(message, "content", []):
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolCall):
            parts.append(f"[tool_call:{block.name}]")
        elif hasattr(block, "text"):
            parts.append(str(getattr(block, "text", "")))
    return f"[{role}] " + " ".join(parts)


def _format_transcript(messages: list[Message]) -> str:
    return "\n\n".join(_format_message_for_summary(message) for message in messages)


async def summarize(
    *,
    provider: Provider,
    model: Model,
    messages_to_summarize: list[Message],
    existing: CompactionState | None,
    max_summary_tokens: int = 1024,
    abort_signal: asyncio.Event | None = None,
) -> CompactionState:
    system_prompt = SUMMARIZER_SYSTEM_PROMPT
    if existing and existing.summary:
        system_prompt += "\n\n" + EXISTING_SUMMARY_SUFFIX.format(prev=existing.summary)

    response = await provider.generate(
        model=model,
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
