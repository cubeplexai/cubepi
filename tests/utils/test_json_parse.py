"""Tests for cubepi.utils.json_parse — repair_json and parse_streaming_json."""

from __future__ import annotations

import json

import pytest

from cubepi.utils.json_parse import (
    _close_partial_json,
    parse_streaming_json,
    repair_json,
)


# ---------------------------------------------------------------------------
# repair_json
# ---------------------------------------------------------------------------


class TestRepairJson:
    """Tests for repair_json()."""

    def test_valid_json_unchanged(self) -> None:
        text = '{"key": "value", "num": 42}'
        assert repair_json(text) == text

    def test_escape_raw_control_chars_in_string(self) -> None:
        # Embed a NUL and BEL inside a JSON string value.
        raw = '{"msg": "hello\x00world\x07"}'
        repaired = repair_json(raw)
        assert "\\u0000" in repaired
        assert "\\u0007" in repaired
        # Repaired text must be valid JSON.
        parsed = json.loads(repaired)
        assert parsed["msg"] == "hello\x00world\x07"

    def test_tab_newline_carriage_return_escaped(self) -> None:
        raw = '{"msg": "a\tb\nc\rd"}'
        repaired = repair_json(raw)
        assert "\\t" in repaired
        assert "\\n" in repaired
        assert "\\r" in repaired
        parsed = json.loads(repaired)
        assert parsed["msg"] == "a\tb\nc\rd"

    def test_invalid_escape_sequence_fixed(self) -> None:
        # \x is not a valid JSON escape — should become \\x
        raw = r'{"path": "C:\xampp\htdocs"}'
        repaired = repair_json(raw)
        parsed = json.loads(repaired)
        assert "xampp" in parsed["path"]
        assert "htdocs" in parsed["path"]

    def test_valid_escape_sequences_preserved(self) -> None:
        raw = r'{"msg": "line1\nline2\ttab\\backslash\"quote"}'
        repaired = repair_json(raw)
        assert repaired == raw

    def test_valid_unicode_escape_preserved(self) -> None:
        raw = r'{"emoji": "AB"}'
        repaired = repair_json(raw)
        parsed = json.loads(repaired)
        assert parsed["emoji"] == "AB"

    def test_trailing_backslash(self) -> None:
        raw = '{"val": "trailing\\'
        repaired = repair_json(raw)
        # Should double the trailing backslash.
        assert repaired.endswith("\\\\")

    def test_control_chars_outside_string_unchanged(self) -> None:
        # Control chars outside strings are passed through unchanged
        # (invalid JSON anyway, but repair_json only touches strings).
        raw = '{\n"key":\t"val"\n}'
        assert repair_json(raw) == raw

    def test_multiple_strings(self) -> None:
        raw = '{"a": "x\x01y", "b": "normal"}'
        repaired = repair_json(raw)
        parsed = json.loads(repaired)
        assert parsed["a"] == "x\x01y"
        assert parsed["b"] == "normal"

    def test_empty_string(self) -> None:
        assert repair_json("") == ""

    def test_no_strings(self) -> None:
        raw = "[1, 2, 3]"
        assert repair_json(raw) == raw

    def test_backspace_and_formfeed(self) -> None:
        raw = '{"msg": "a\x08b\x0c"}'
        repaired = repair_json(raw)
        assert "\\b" in repaired
        assert "\\f" in repaired
        parsed = json.loads(repaired)
        assert parsed["msg"] == "a\x08b\x0c"

    def test_incomplete_unicode_escape(self) -> None:
        # \u followed by fewer than 4 hex digits — backslash should be doubled.
        raw = r'{"val": "\u00G"}'
        repaired = repair_json(raw)
        # Should be parseable — the \u was treated as invalid and backslash doubled.
        parsed = json.loads(repaired)
        assert "u00G" in parsed["val"]


# ---------------------------------------------------------------------------
# _close_partial_json
# ---------------------------------------------------------------------------


class TestClosePartialJson:
    """Tests for the internal _close_partial_json helper."""

    def test_complete_json_unchanged(self) -> None:
        text = '{"key": "val"}'
        assert _close_partial_json(text) == text

    def test_truncated_object(self) -> None:
        text = '{"key": "val"'
        closed = _close_partial_json(text)
        parsed = json.loads(closed)
        assert parsed == {"key": "val"}

    def test_truncated_nested_object(self) -> None:
        text = '{"a": {"b": 1'
        closed = _close_partial_json(text)
        parsed = json.loads(closed)
        assert parsed == {"a": {"b": 1}}

    def test_truncated_array(self) -> None:
        text = '{"items": [1, 2'
        closed = _close_partial_json(text)
        parsed = json.loads(closed)
        assert parsed == {"items": [1, 2]}

    def test_truncated_string(self) -> None:
        text = '{"key": "incom'
        closed = _close_partial_json(text)
        parsed = json.loads(closed)
        assert parsed["key"] == "incom"

    def test_truncated_deeply_nested(self) -> None:
        text = '{"a": {"b": [{"c": "d'
        closed = _close_partial_json(text)
        parsed = json.loads(closed)
        assert parsed["a"]["b"][0]["c"] == "d"

    def test_empty_string(self) -> None:
        assert _close_partial_json("") == ""

    def test_escaped_quote_in_string(self) -> None:
        text = r'{"key": "val\"ue'
        closed = _close_partial_json(text)
        parsed = json.loads(closed)
        assert 'val"ue' in parsed["key"]


# ---------------------------------------------------------------------------
# parse_streaming_json
# ---------------------------------------------------------------------------


class TestParseStreamingJson:
    """Tests for parse_streaming_json()."""

    def test_valid_json(self) -> None:
        assert parse_streaming_json('{"a": 1}') == {"a": 1}

    def test_none_input(self) -> None:
        assert parse_streaming_json(None) == {}

    def test_empty_string(self) -> None:
        assert parse_streaming_json("") == {}

    def test_whitespace_only(self) -> None:
        assert parse_streaming_json("   ") == {}

    def test_truncated_json(self) -> None:
        result = parse_streaming_json('{"name": "hello", "count": 42')
        assert result == {"name": "hello", "count": 42}

    def test_truncated_string_value(self) -> None:
        result = parse_streaming_json('{"cmd": "echo hel')
        assert result["cmd"] == "echo hel"

    def test_control_chars_in_value(self) -> None:
        raw = '{"msg": "line\x01end"}'
        result = parse_streaming_json(raw)
        assert result["msg"] == "line\x01end"

    def test_invalid_escape_in_value(self) -> None:
        raw = r'{"path": "C:\windows\system32"}'
        result = parse_streaming_json(raw)
        assert "windows" in result["path"]
        assert "system32" in result["path"]

    def test_completely_broken_returns_empty(self) -> None:
        assert parse_streaming_json("not json at all") == {}

    def test_non_dict_returns_empty(self) -> None:
        # parse_streaming_json should only return dicts.
        assert parse_streaming_json("[1, 2, 3]") == {}
        assert parse_streaming_json('"just a string"') == {}
        assert parse_streaming_json("42") == {}

    def test_complex_nested_truncated(self) -> None:
        text = '{"tool": "bash", "args": {"cmd": "ls -la", "opts": {"verbose": true'
        result = parse_streaming_json(text)
        assert result["tool"] == "bash"
        assert result["args"]["cmd"] == "ls -la"
        assert result["args"]["opts"]["verbose"] is True

    def test_truncated_after_key(self) -> None:
        # Truncated right after a colon — partial parse should still work
        # or return {} gracefully.
        result = parse_streaming_json('{"key":')
        # This is acceptable as either {} or a partial result
        assert isinstance(result, dict)

    def test_repair_then_partial(self) -> None:
        # Control char + truncation — needs both repair and partial parse.
        raw = '{"msg": "bad\x02char", "extra": "trunc'
        result = parse_streaming_json(raw)
        assert result["msg"] == "bad\x02char"
        assert result["extra"] == "trunc"

    def test_multiple_tool_calls_pattern(self) -> None:
        # Simulates a typical streaming tool-call argument pattern.
        text = '{"command": "grep -r TODO .", "timeout": 30}'
        result = parse_streaming_json(text)
        assert result["command"] == "grep -r TODO ."
        assert result["timeout"] == 30

    def test_boolean_and_null_values(self) -> None:
        text = '{"flag": true, "other": null, "num": 3.14}'
        result = parse_streaming_json(text)
        assert result["flag"] is True
        assert result["other"] is None
        assert result["num"] == pytest.approx(3.14)

    def test_truncated_boolean(self) -> None:
        # "tru" is not valid JSON — should fallback gracefully.
        result = parse_streaming_json('{"flag": tru')
        assert isinstance(result, dict)

    def test_unicode_content(self) -> None:
        text = '{"text": "Hello \\u4e16\\u754c"}'
        result = parse_streaming_json(text)
        assert result["text"] == "Hello 世界"
