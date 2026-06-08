from __future__ import annotations

import asyncio
import json

from cubepi.middleware.compaction.state import CompactionState, message_refs
from cubepi.middleware.compaction.tokens import approx_tokens
from cubepi.providers.base import (
    AssistantMessage,
    BoundModel,
    Message,
    StreamOptions,
    TextContent,
    ToolCall,
    UserMessage,
)

_ARG_VALUE_CHARS = 200
_ARG_REPR_MAX = 500

_SUMMARY_MIN = 1024  # matches prior fixed default — never regress below it
_SUMMARY_RATIO = 0.15
_SUMMARY_MAX = 4096


def _dynamic_summary_budget(messages: list[Message]) -> int:
    """Compute a token budget for the summariser proportional to input size.

    Floor is ``_SUMMARY_MIN = 1024`` — the previous fixed default. No
    conversation ever gets a smaller budget than today; only larger ones
    get more headroom.
    """
    content_tokens = approx_tokens(messages)
    scaled = int(content_tokens * _SUMMARY_RATIO)
    return max(_SUMMARY_MIN, min(scaled, _SUMMARY_MAX))


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
            parts.append(
                f"[tool_call:{block.name}]{_format_arguments(block.arguments)}"
            )
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
    ref_messages: list[Message] | None = None,
    max_summary_tokens: int | None = None,
    abort_signal: asyncio.Event | None = None,
) -> CompactionState:
    """Run the LLM summariser and produce a new ``CompactionState``.

    ``messages_to_summarize`` is the source for the transcript fed to the
    LLM. ``ref_messages`` (when supplied) overrides the source for
    ID/SHA256-ref extraction so the persisted state matches the *original*
    message list — required when the transcript was built from pre-pruned
    content (the pruner rewrites tool result text, which would otherwise
    cause ``_state_matches_history`` to clear state on the next turn).

    ``max_summary_tokens``: when ``None`` (default), the budget is computed
    dynamically from ``messages_to_summarize`` size (floor 1024, ceiling
    4096). When provided, that exact value is used.
    """
    ref_source = ref_messages if ref_messages is not None else messages_to_summarize
    budget = (
        max_summary_tokens
        if max_summary_tokens is not None
        else _dynamic_summary_budget(messages_to_summarize)
    )

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
        max_output_tokens=budget,
        temperature=0.0,
        thinking="off",
    )

    text = "".join(
        block.text for block in response.content if isinstance(block, TextContent)
    )
    if response.error_message is not None:
        raise RuntimeError(response.error_message)

    new_ids = [str(getattr(message, "id", "") or "") for message in ref_source]
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
        summarized_message_refs=prior_refs + message_refs(ref_source),
        last_summarized_message_id=last_id,
    )


def build_fallback_summary(
    messages_to_summarize: list[Message],
    *,
    existing: CompactionState | None,
    ref_messages: list[Message] | None = None,
) -> CompactionState:
    """Deterministic fallback when the LLM summariser is unavailable.

    Builds a low-fidelity but structured handoff from the message list itself:
    first lines of up to 5 user requests, plus distinct tool names invoked.
    Lets compaction proceed (shrinking context) so the agent isn't stuck
    over-limit on every subsequent turn when the summariser model is down.

    ``ref_messages`` overrides the source for ID/SHA256-ref extraction —
    mirrors the contract of :func:`summarize`. ``is_fallback=True`` is set
    on the returned state.
    """
    ref_source = ref_messages if ref_messages is not None else messages_to_summarize

    user_lines: list[str] = []
    tool_names: list[str] = []

    for msg in messages_to_summarize:
        if isinstance(msg, UserMessage):
            if len(user_lines) >= 5:
                continue
            for user_block in msg.content:
                if isinstance(user_block, TextContent) and user_block.text.strip():
                    first_line = user_block.text.strip().splitlines()[0][:120]
                    user_lines.append(first_line)
                    break
        elif isinstance(msg, AssistantMessage):
            for asst_block in msg.content:
                if (
                    isinstance(asst_block, ToolCall)
                    and asst_block.name not in tool_names
                ):
                    tool_names.append(asst_block.name)

    parts: list[str] = ["[Compaction fallback — LLM summariser unavailable]"]
    if existing and existing.summary:
        parts.append(f"Prior context: {existing.summary}")
    if user_lines:
        parts.append("User requests: " + "; ".join(user_lines))
    if tool_names:
        parts.append("Tool calls: " + ", ".join(sorted(tool_names)))

    summary = "\n".join(parts)

    prior_ids = list(existing.summarized_message_ids) if existing else []
    prior_refs = list(existing.summarized_message_refs) if existing else []
    new_ids = [str(getattr(m, "id", "") or "") for m in ref_source]
    new_ids = [mid for mid in new_ids if mid]
    last_id = (
        new_ids[-1]
        if new_ids
        else (existing.last_summarized_message_id if existing else None)
    )

    return CompactionState(
        summary=summary,
        summarized_message_ids=prior_ids + new_ids,
        summarized_message_refs=prior_refs + message_refs(ref_source),
        last_summarized_message_id=last_id,
        is_fallback=True,
    )
