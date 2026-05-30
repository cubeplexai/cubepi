"""cubepi trace convert — reconstruct an API request body from a recorded chat span."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cubepi.cli.trace import loader
from cubepi.tracing import schema


def cmd_convert(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    try:
        files = loader.resolve_run(args.run, directory)
    except loader.RunResolutionError as exc:
        print(str(exc))
        return 1

    spans, _skipped = loader.load_run(files)
    chat_spans = [s for s in spans if s.is_chat]
    if not chat_spans:
        print("no chat spans found — was record_content=True when tracing?")
        return 1

    target = None
    if args.span:
        prefix = args.span
        matches = [s for s in chat_spans if (s.span_id or "").startswith(prefix)]
        if not matches:
            print(f"no chat span with id prefix {prefix!r}")
            return 1
        if len(matches) > 1:
            ids = ", ".join(s.span_id or "" for s in matches)
            print(f"ambiguous span prefix {prefix!r}: {ids}")
            return 1
        target = matches[0]
    elif args.turn is not None:
        idx = args.turn - 1
        if idx < 0 or idx >= len(chat_spans):
            print(f"--turn {args.turn} out of range (1..{len(chat_spans)})")
            return 1
        target = chat_spans[idx]
    else:
        target = chat_spans[-1]

    attrs = target.attributes
    model_id = str(attrs.get(schema.GEN_AI_REQUEST_MODEL, ""))
    max_tokens = attrs.get(schema.GEN_AI_REQUEST_MAX_TOKENS)
    temperature = attrs.get(schema.GEN_AI_REQUEST_TEMPERATURE)

    msgs: list[dict[str, Any]] = _parse_json_attr(
        attrs.get(schema.GEN_AI_INPUT_MESSAGES)
    )
    sys_instructions: list[dict[str, Any]] = _parse_json_attr(
        attrs.get(schema.GEN_AI_SYSTEM_INSTRUCTIONS)
    )
    tool_defs: list[dict[str, Any]] = _parse_json_attr(
        attrs.get(schema.GEN_AI_TOOL_DEFINITIONS)
    )

    fmt = args.format
    if fmt == "anthropic":
        body = _build_anthropic(model_id, msgs, sys_instructions, tool_defs, max_tokens)
        print(json.dumps(body, indent=2, ensure_ascii=False))
    elif fmt == "curl":
        print(
            _build_curl(
                model_id, msgs, sys_instructions, tool_defs, max_tokens, temperature
            )
        )
    else:
        body = _build_openai(
            model_id, msgs, sys_instructions, tool_defs, max_tokens, temperature
        )
        print(json.dumps(body, indent=2, ensure_ascii=False))
    return 0


def _parse_json_attr(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _system_text(instructions: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in instructions:
        role = block.get("role", "")
        if role == "system":
            for part in block.get("parts", []) or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("content") or part.get("text") or ""
                    if text:
                        parts.append(str(text))
    return "\n\n".join(parts)


def messages_to_openai(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in msgs:
        role = msg.get("role", "")
        parts = msg.get("parts") or []
        if role == "user":
            texts = [
                str(p.get("content") or p.get("text") or "")
                for p in parts
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            out.append({"role": "user", "content": " ".join(t for t in texts if t)})
        elif role == "assistant":
            tool_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []
            for p in parts:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "tool_call":
                    args = p.get("arguments", {})
                    tool_calls.append(
                        {
                            "id": str(p.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": str(p.get("name", "")),
                                "arguments": (
                                    json.dumps(args)
                                    if isinstance(args, dict)
                                    else str(args)
                                ),
                            },
                        }
                    )
                elif p.get("type") == "text":
                    t = str(p.get("content") or p.get("text") or "")
                    if t:
                        text_parts.append(t)
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": " ".join(text_parts) if text_parts else None,
            }
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        elif role == "tool":
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "tool_call_response":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(p.get("id", "")),
                            "content": str(p.get("result", "")),
                        }
                    )
    return out


def messages_to_anthropic(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []
    for msg in msgs:
        role = msg.get("role", "")
        parts = msg.get("parts") or []
        if role == "tool":
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "tool_call_response":
                    pending_tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": str(p.get("id", "")),
                            "content": str(p.get("result", "")),
                        }
                    )
        elif role == "user":
            if pending_tool_results:
                out.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results = []
            content_blocks: list[dict[str, Any]] = []
            for p in parts:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    text = str(p.get("content") or p.get("text") or "")
                    if text:
                        content_blocks.append({"type": "text", "text": text})
            if content_blocks:
                out.append({"role": "user", "content": content_blocks})
        elif role == "assistant":
            if pending_tool_results:
                out.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results = []
            content_blocks = []
            for p in parts:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    text = str(p.get("content") or p.get("text") or "")
                    if text:
                        content_blocks.append({"type": "text", "text": text})
                elif p.get("type") == "tool_call":
                    args = p.get("arguments", {})
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(p.get("id", "")),
                            "name": str(p.get("name", "")),
                            "input": args if isinstance(args, dict) else {},
                        }
                    )
            if content_blocks:
                out.append({"role": "assistant", "content": content_blocks})
    if pending_tool_results:
        out.append({"role": "user", "content": pending_tool_results})
    return out


def _build_openai(
    model_id: str,
    msgs: list[dict[str, Any]],
    sys_instructions: list[dict[str, Any]],
    tool_defs: list[dict[str, Any]],
    max_tokens: Any,
    temperature: Any,
) -> dict[str, Any]:
    openai_msgs: list[dict[str, Any]] = []
    sys_text = _system_text(sys_instructions)
    if sys_text:
        openai_msgs.append({"role": "system", "content": sys_text})
    openai_msgs.extend(messages_to_openai(msgs))
    body: dict[str, Any] = {"model": model_id, "messages": openai_msgs, "stream": True}
    if max_tokens is not None:
        body["max_tokens"] = int(max_tokens)
    if temperature is not None:
        body["temperature"] = float(temperature)
    if tool_defs:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": str(t.get("name", "")),
                    "description": str(t.get("description", "")),
                    "parameters": t.get("parameters") or {},
                },
            }
            for t in tool_defs
        ]
    return body


def _build_anthropic(
    model_id: str,
    msgs: list[dict[str, Any]],
    sys_instructions: list[dict[str, Any]],
    tool_defs: list[dict[str, Any]],
    max_tokens: Any,
) -> dict[str, Any]:
    sys_text = _system_text(sys_instructions)
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages_to_anthropic(msgs),
        "max_tokens": int(max_tokens) if max_tokens is not None else 4096,
    }
    if sys_text:
        body["system"] = sys_text
    if tool_defs:
        body["tools"] = [
            {
                "name": str(t.get("name", "")),
                "description": str(t.get("description", "")),
                "input_schema": t.get("parameters") or {},
            }
            for t in tool_defs
        ]
    return body


def _build_curl(
    model_id: str,
    msgs: list[dict[str, Any]],
    sys_instructions: list[dict[str, Any]],
    tool_defs: list[dict[str, Any]],
    max_tokens: Any,
    temperature: Any,
) -> str:
    body = _build_openai(
        model_id, msgs, sys_instructions, tool_defs, max_tokens, temperature
    )
    body_json = json.dumps(body, ensure_ascii=False)
    escaped = body_json.replace("'", "'\"'\"'")
    return (
        'curl -s -X POST "${BASE_URL}/v1/chat/completions" \\\n'
        '  -H "Authorization: Bearer ${API_KEY}" \\\n'
        '  -H "Content-Type: application/json" \\\n'
        f"  -d '{escaped}'"
    )
