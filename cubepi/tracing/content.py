"""Convert cubepi messages and tool definitions into the OpenTelemetry
GenAI semconv ``gen_ai.input.messages`` / ``gen_ai.output.messages`` /
``gen_ai.tool.definitions`` JSON shapes.

The OTel SDK only accepts simple attribute types (str, bool, int, float
and sequences thereof) on :meth:`Span.set_attribute`, so structured
content is serialized to a JSON **string** here. Backends that parse
``gen_ai.input.messages`` follow this convention.
"""

from __future__ import annotations

import json
from typing import Any

from cubepi.providers.base import (
    AssistantMessage,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def messages_to_semconv(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate a list of cubepi messages into the GenAI semconv shape.

    Returns a list of ``{"role": ..., "parts": [...]}`` dicts. Note
    that ``"reasoning"`` is a cubepi extension to the standard semconv
    part-type set; see the spec.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, UserMessage):
            out.append(
                {"role": "user", "parts": _user_or_tool_content_to_parts(msg.content)}
            )
        elif isinstance(msg, AssistantMessage):
            out.append(
                {"role": "assistant", "parts": _assistant_content_to_parts(msg.content)}
            )
        elif isinstance(msg, ToolResultMessage):
            out.append(
                {
                    "role": "tool",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": msg.tool_call_id,
                            "result": _text_of(msg.content),
                        }
                    ],
                }
            )
    return out


def system_instructions_to_semconv(system_prompt: str | None) -> list[dict[str, Any]]:
    """Wrap a system prompt string in the semconv messages-array shape.

    Per the spec §10.5, ``gen_ai.system_instructions`` follows the same
    structure as ``gen_ai.input.messages`` but with a single
    ``role: "system"`` entry.
    """
    if not system_prompt:
        return []
    return [
        {
            "role": "system",
            "parts": [{"type": "text", "content": system_prompt}],
        }
    ]


def tool_definitions_to_semconv(payload: dict) -> list[dict[str, Any]]:
    """Pull tool schemas out of the provider's wire payload.

    Each provider shapes ``tools`` differently:
    - Anthropic: ``[{"name", "description", "input_schema"}]``
    - OpenAI chat: ``[{"type": "function", "function": {"name", "description", "parameters"}}]``
    - OpenAI Responses: same as chat but flattened
    - Faux: doesn't emit a tools key (handled by absence)

    Normalize all to ``[{"name", "description", "parameters"}]`` so a
    single backend renderer can show them consistently.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):
            fn = t["function"]
            out.append(
                {
                    "name": fn.get("name") or "",
                    "description": fn.get("description") or "",
                    "parameters": fn.get("parameters") or {},
                }
            )
        else:
            out.append(
                {
                    "name": t.get("name") or "",
                    "description": t.get("description") or "",
                    "parameters": t.get("input_schema") or t.get("parameters") or {},
                }
            )
    return out


def serialize_for_attribute(value: Any) -> str:
    """JSON-encode an arbitrary value for placement on a span attribute.

    Falls back to ``str(value)`` for objects without a JSON
    representation (rare, since we control the inputs).
    """
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _user_or_tool_content_to_parts(content: list[Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "content": block.text})
        else:
            # Other UserMessage content types (e.g. ImageContent) — record
            # the type marker so a downstream backend can display a
            # placeholder; binary payloads are NOT emitted.
            type_marker = getattr(block, "type", "unknown")
            parts.append({"type": str(type_marker)})
    return parts


def _assistant_content_to_parts(content: list[Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "content": block.text})
        elif isinstance(block, ThinkingContent):
            # ``reasoning`` is a cubepi extension to the semconv parts
            # vocabulary; see docs/specs/2026-05-18-cubepi-tracing-design.md §10.5.
            parts.append({"type": "reasoning", "content": block.thinking})
        elif isinstance(block, ToolCall):
            parts.append(
                {
                    "type": "tool_call",
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.arguments,
                }
            )
        else:
            type_marker = getattr(block, "type", "unknown")
            parts.append({"type": str(type_marker)})
    return parts


def _text_of(content: list[Any]) -> str:
    """Concatenate text-typed content blocks into a single string."""
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "".join(parts)
