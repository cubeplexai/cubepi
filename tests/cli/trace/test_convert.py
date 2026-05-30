"""Tests for `cubepi trace convert` — reconstruct API request bodies from recorded spans."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cubepi.cli.__main__ import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MESSAGES = json.dumps(
    [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]},
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "hi there"},
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "search",
                    "arguments": {"q": "weather"},
                },
            ],
        },
        {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call_1",
                    "result": "sunny",
                }
            ],
        },
        {"role": "user", "parts": [{"type": "text", "content": "thanks"}]},
    ]
)

_SYSTEM = json.dumps(
    [{"role": "system", "parts": [{"type": "text", "content": "be helpful"}]}]
)

_TOOLS = json.dumps(
    [
        {
            "name": "search",
            "description": "search the web",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]
)


def _write_trace(directory: Path, *, two_turns: bool = False) -> None:
    f = directory / "2026-05-20" / "run1.jsonl"
    f.parent.mkdir(parents=True)
    rows = [
        {
            "name": "invoke_agent",
            "context": {"trace_id": "0xt", "span_id": "0x1"},
            "parent_id": None,
            "start_time": "2026-05-20T00:00:00Z",
            "end_time": "2026-05-20T00:00:10Z",
            "status": {"status_code": "UNSET"},
            "attributes": {
                "cubepi.run_id": "run1",
                "gen_ai.operation.name": "invoke_agent",
            },
        },
        {
            "name": "chat gpt-test",
            "context": {"trace_id": "0xt", "span_id": "0xabc123"},
            "parent_id": "0x1",
            "start_time": "2026-05-20T00:00:01Z",
            "end_time": "2026-05-20T00:00:05Z",
            "status": {"status_code": "UNSET"},
            "attributes": {
                "cubepi.run_id": "run1",
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "gpt-test",
                "gen_ai.request.max_tokens": 4096,
                "gen_ai.request.temperature": 0.7,
                "gen_ai.input.messages": _MESSAGES,
                "gen_ai.system_instructions": _SYSTEM,
                "gen_ai.tool.definitions": _TOOLS,
            },
        },
    ]
    if two_turns:
        rows.append(
            {
                "name": "chat gpt-test",
                "context": {"trace_id": "0xt", "span_id": "0xdef456"},
                "parent_id": "0x1",
                "start_time": "2026-05-20T00:00:06Z",
                "end_time": "2026-05-20T00:00:09Z",
                "status": {"status_code": "UNSET"},
                "attributes": {
                    "cubepi.run_id": "run1",
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": "gpt-test",
                    "gen_ai.input.messages": _MESSAGES,
                },
            }
        )
    with f.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _write_trace_no_content(directory: Path) -> None:
    f = directory / "2026-05-20" / "run_nc.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "name": "invoke_agent",
            "context": {"trace_id": "0xt2", "span_id": "0x1"},
            "parent_id": None,
            "start_time": "2026-05-20T00:00:00Z",
            "end_time": "2026-05-20T00:00:02Z",
            "status": {"status_code": "UNSET"},
            "attributes": {
                "cubepi.run_id": "run_nc",
                "gen_ai.operation.name": "invoke_agent",
            },
        },
    ]
    with f.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Basic output: default (OpenAI) format
# ---------------------------------------------------------------------------


class TestConvertDefaultFormat:
    def test_outputs_valid_json(self, tmp_path, capsys):
        _write_trace(tmp_path)
        rc = main(["trace", "convert", "run1", "--dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        body = json.loads(out)
        assert body["model"] == "gpt-test"
        assert body["stream"] is True

    def test_includes_system_message(self, tmp_path, capsys):
        _write_trace(tmp_path)
        main(["trace", "convert", "run1", "--dir", str(tmp_path)])
        out = capsys.readouterr().out
        body = json.loads(out)
        system_msgs = [m for m in body["messages"] if m["role"] == "system"]
        assert system_msgs
        assert "be helpful" in system_msgs[0]["content"]

    def test_includes_tools(self, tmp_path, capsys):
        _write_trace(tmp_path)
        main(["trace", "convert", "run1", "--dir", str(tmp_path)])
        out = capsys.readouterr().out
        body = json.loads(out)
        assert "tools" in body
        assert body["tools"][0]["function"]["name"] == "search"

    def test_max_tokens_and_temperature(self, tmp_path, capsys):
        _write_trace(tmp_path)
        main(["trace", "convert", "run1", "--dir", str(tmp_path)])
        out = capsys.readouterr().out
        body = json.loads(out)
        assert body["max_tokens"] == 4096
        assert abs(body["temperature"] - 0.7) < 1e-6

    def test_tool_call_in_assistant_message(self, tmp_path, capsys):
        _write_trace(tmp_path)
        main(["trace", "convert", "run1", "--dir", str(tmp_path)])
        out = capsys.readouterr().out
        body = json.loads(out)
        assistant_msgs = [m for m in body["messages"] if m["role"] == "assistant"]
        assert assistant_msgs
        tool_calls = assistant_msgs[0].get("tool_calls", [])
        assert any(tc["function"]["name"] == "search" for tc in tool_calls)

    def test_tool_result_message(self, tmp_path, capsys):
        _write_trace(tmp_path)
        main(["trace", "convert", "run1", "--dir", str(tmp_path)])
        out = capsys.readouterr().out
        body = json.loads(out)
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        assert tool_msgs
        assert tool_msgs[0]["content"] == "sunny"


# ---------------------------------------------------------------------------
# Anthropic format
# ---------------------------------------------------------------------------


class TestConvertAnthropicFormat:
    def test_anthropic_format_valid_json(self, tmp_path, capsys):
        _write_trace(tmp_path)
        rc = main(
            ["trace", "convert", "run1", "--dir", str(tmp_path), "--format", "anthropic"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        body = json.loads(out)
        assert body["model"] == "gpt-test"
        assert "messages" in body

    def test_anthropic_system_at_top_level(self, tmp_path, capsys):
        _write_trace(tmp_path)
        main(
            ["trace", "convert", "run1", "--dir", str(tmp_path), "--format", "anthropic"]
        )
        out = capsys.readouterr().out
        body = json.loads(out)
        assert "system" in body
        assert "be helpful" in body["system"]

    def test_anthropic_tool_use_blocks(self, tmp_path, capsys):
        _write_trace(tmp_path)
        main(
            ["trace", "convert", "run1", "--dir", str(tmp_path), "--format", "anthropic"]
        )
        out = capsys.readouterr().out
        body = json.loads(out)
        all_content = [
            block
            for msg in body["messages"]
            for block in (msg.get("content") or [])
            if isinstance(block, dict)
        ]
        tool_use = [b for b in all_content if b.get("type") == "tool_use"]
        assert tool_use
        assert tool_use[0]["name"] == "search"

    def test_anthropic_tool_result_as_user_message(self, tmp_path, capsys):
        _write_trace(tmp_path)
        main(
            ["trace", "convert", "run1", "--dir", str(tmp_path), "--format", "anthropic"]
        )
        out = capsys.readouterr().out
        body = json.loads(out)
        tool_results = [
            block
            for msg in body["messages"]
            if msg["role"] == "user"
            for block in (msg.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert tool_results

    def test_anthropic_tools_use_input_schema(self, tmp_path, capsys):
        _write_trace(tmp_path)
        main(
            ["trace", "convert", "run1", "--dir", str(tmp_path), "--format", "anthropic"]
        )
        out = capsys.readouterr().out
        body = json.loads(out)
        assert "tools" in body
        assert "input_schema" in body["tools"][0]


# ---------------------------------------------------------------------------
# curl format
# ---------------------------------------------------------------------------


class TestConvertCurlFormat:
    def test_curl_format_is_shell_command(self, tmp_path, capsys):
        _write_trace(tmp_path)
        rc = main(
            ["trace", "convert", "run1", "--dir", str(tmp_path), "--format", "curl"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "curl" in out
        assert "Authorization" in out
        assert "Content-Type" in out

    def test_curl_contains_model(self, tmp_path, capsys):
        _write_trace(tmp_path)
        main(["trace", "convert", "run1", "--dir", str(tmp_path), "--format", "curl"])
        out = capsys.readouterr().out
        assert "gpt-test" in out


# ---------------------------------------------------------------------------
# Span / turn selection
# ---------------------------------------------------------------------------


class TestConvertSpanSelection:
    def test_span_prefix_selects_correct_span(self, tmp_path, capsys):
        _write_trace(tmp_path, two_turns=True)
        # First chat span is 0xabc123; last is 0xdef456.
        rc = main(
            ["trace", "convert", "run1", "--dir", str(tmp_path), "--span", "0xabc"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        # First span has temperature; second does not.
        body = json.loads(out)
        assert "temperature" in body

    def test_span_prefix_no_match_returns_1(self, tmp_path, capsys):
        _write_trace(tmp_path)
        rc = main(
            ["trace", "convert", "run1", "--dir", str(tmp_path), "--span", "0xzzz"]
        )
        err = capsys.readouterr()
        assert rc == 1

    def test_turn_1_selects_first_chat_span(self, tmp_path, capsys):
        _write_trace(tmp_path, two_turns=True)
        rc = main(
            ["trace", "convert", "run1", "--dir", str(tmp_path), "--turn", "1"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        body = json.loads(out)
        assert "temperature" in body  # first span has temperature

    def test_turn_2_selects_second_chat_span(self, tmp_path, capsys):
        _write_trace(tmp_path, two_turns=True)
        rc = main(
            ["trace", "convert", "run1", "--dir", str(tmp_path), "--turn", "2"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        body = json.loads(out)
        assert "temperature" not in body  # second span has no temperature

    def test_turn_out_of_range_returns_1(self, tmp_path, capsys):
        _write_trace(tmp_path)
        rc = main(
            ["trace", "convert", "run1", "--dir", str(tmp_path), "--turn", "99"]
        )
        capsys.readouterr()
        assert rc == 1

    def test_default_selects_last_chat_span(self, tmp_path, capsys):
        _write_trace(tmp_path, two_turns=True)
        rc = main(["trace", "convert", "run1", "--dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        body = json.loads(out)
        # Last span has no temperature attr → key absent
        assert "temperature" not in body


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestConvertErrors:
    def test_run_not_found_returns_1(self, tmp_path, capsys):
        _write_trace(tmp_path)
        rc = main(["trace", "convert", "missing", "--dir", str(tmp_path)])
        capsys.readouterr()
        assert rc == 1

    def test_no_chat_spans_returns_1(self, tmp_path, capsys):
        _write_trace_no_content(tmp_path)
        rc = main(["trace", "convert", "run_nc", "--dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 1
        assert "no chat spans" in out


# ---------------------------------------------------------------------------
# Unit tests for message-conversion helpers
# ---------------------------------------------------------------------------


class TestMessagesToOpenAI:
    def test_user_message(self):
        from cubepi.cli.trace.convert import messages_to_openai

        msgs = [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
        out = messages_to_openai(msgs)
        assert out == [{"role": "user", "content": "hi"}]

    def test_assistant_text(self):
        from cubepi.cli.trace.convert import messages_to_openai

        msgs = [
            {"role": "assistant", "parts": [{"type": "text", "content": "hello"}]}
        ]
        out = messages_to_openai(msgs)
        assert len(out) == 1
        assert out[0]["role"] == "assistant"
        assert out[0]["content"] == "hello"

    def test_assistant_tool_call(self):
        from cubepi.cli.trace.convert import messages_to_openai

        msgs = [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "c1",
                        "name": "fn",
                        "arguments": {"x": 1},
                    }
                ],
            }
        ]
        out = messages_to_openai(msgs)
        tcs = out[0]["tool_calls"]
        assert tcs[0]["id"] == "c1"
        assert tcs[0]["function"]["name"] == "fn"
        parsed = json.loads(tcs[0]["function"]["arguments"])
        assert parsed == {"x": 1}

    def test_tool_result(self):
        from cubepi.cli.trace.convert import messages_to_openai

        msgs = [
            {
                "role": "tool",
                "parts": [{"type": "tool_call_response", "id": "c1", "result": "ok"}],
            }
        ]
        out = messages_to_openai(msgs)
        assert out == [{"role": "tool", "tool_call_id": "c1", "content": "ok"}]

    def test_unknown_role_skipped(self):
        from cubepi.cli.trace.convert import messages_to_openai

        msgs = [{"role": "system", "parts": [{"type": "text", "content": "x"}]}]
        out = messages_to_openai(msgs)
        assert out == []


class TestMessagesToAnthropic:
    def test_user_message(self):
        from cubepi.cli.trace.convert import messages_to_anthropic

        msgs = [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
        out = messages_to_anthropic(msgs)
        assert out[0]["role"] == "user"
        assert out[0]["content"][0]["text"] == "hi"

    def test_tool_result_batched_into_user_message(self):
        from cubepi.cli.trace.convert import messages_to_anthropic

        msgs = [
            {
                "role": "tool",
                "parts": [{"type": "tool_call_response", "id": "c1", "result": "x"}],
            },
            {"role": "user", "parts": [{"type": "text", "content": "follow-up"}]},
        ]
        out = messages_to_anthropic(msgs)
        # tool result should be flushed into a user message before the text user msg
        user_msgs = [m for m in out if m["role"] == "user"]
        all_content = [b for m in user_msgs for b in (m.get("content") or [])]
        tool_results = [b for b in all_content if isinstance(b, dict) and b.get("type") == "tool_result"]
        assert tool_results

    def test_assistant_tool_use_block(self):
        from cubepi.cli.trace.convert import messages_to_anthropic

        msgs = [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "c2",
                        "name": "do_thing",
                        "arguments": {"a": 1},
                    }
                ],
            }
        ]
        out = messages_to_anthropic(msgs)
        blocks = out[0]["content"]
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["name"] == "do_thing"
        assert blocks[0]["input"] == {"a": 1}

    def test_trailing_tool_results_flushed(self):
        from cubepi.cli.trace.convert import messages_to_anthropic

        msgs = [
            {
                "role": "tool",
                "parts": [{"type": "tool_call_response", "id": "c3", "result": "y"}],
            }
        ]
        out = messages_to_anthropic(msgs)
        assert out
        assert out[-1]["role"] == "user"
