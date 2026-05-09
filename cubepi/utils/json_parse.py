"""JSON repair and partial-parse utilities for streaming tool call arguments.

Provides a 3-tier fallback strategy mirroring pi-agent-core's approach:
  1. ``json.loads(text)`` — fast path for well-formed JSON.
  2. ``json.loads(repair_json(text))`` — fix control chars / bad escapes.
  3. Partial-parse (close open braces/brackets/strings) for truncated JSON.
  4. Return ``{}`` as the last resort.
"""

from __future__ import annotations

import json
import re

# Escapes that the JSON spec allows after a backslash.
_VALID_ESCAPES = frozenset('"\\bfnrtu/')


def _is_control_char(ch: str) -> bool:
    """Return True for ASCII control characters (0x00-0x1F)."""
    cp = ord(ch)
    return 0x00 <= cp <= 0x1F


def _escape_control_char(ch: str) -> str:
    """Convert an ASCII control character to its JSON escape sequence."""
    mapping = {
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }
    if ch in mapping:
        return mapping[ch]
    return f"\\u{ord(ch):04x}"


def repair_json(text: str) -> str:
    """Repair malformed JSON string literals.

    Operates character-by-character, tracking whether the cursor is inside a
    JSON string value.  Inside strings it:

    * Escapes raw control characters (0x00-0x1F) that are not already
      escaped (``\\t``, ``\\n``, ``\\r`` are kept when already escaped).
    * Doubles a backslash before an invalid escape character
      (e.g. ``\\x`` becomes ``\\\\x``), making the output parseable.
    * Handles truncated backslash at end-of-string.
    """
    repaired: list[str] = []
    in_string = False
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]

        # Outside a string literal — pass through, toggle on opening quote.
        if not in_string:
            repaired.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        # Inside a string literal.
        if ch == '"':
            repaired.append(ch)
            in_string = False
            i += 1
            continue

        if ch == "\\":
            # Look ahead for the escape character.
            if i + 1 >= length:
                # Trailing backslash — escape it.
                repaired.append("\\\\")
                i += 1
                continue

            next_ch = text[i + 1]

            # Valid \uXXXX?
            if next_ch == "u":
                hex_digits = text[i + 2 : i + 6]
                if len(hex_digits) == 4 and re.fullmatch(r"[0-9a-fA-F]{4}", hex_digits):
                    repaired.append(f"\\u{hex_digits}")
                    i += 6
                    continue
                # Invalid \u sequence — double the backslash and also
                # emit 'u' so we don't re-process it as an escape.
                repaired.append("\\\\u")
                i += 2
                continue

            if next_ch in _VALID_ESCAPES:
                repaired.append(f"\\{next_ch}")
                i += 2
                continue

            # Invalid escape — double the backslash so the original char is
            # preserved as a literal backslash in the output.
            repaired.append("\\\\")
            i += 1
            continue

        # Raw control character inside string — replace with escape.
        if _is_control_char(ch):
            repaired.append(_escape_control_char(ch))
        else:
            repaired.append(ch)
        i += 1

    return "".join(repaired)


def _close_partial_json(text: str) -> str:
    """Attempt to close truncated JSON by balancing braces, brackets, and strings.

    This is intentionally simple: it scans through *text* tracking nesting
    depth for ``{}``, ``[]``, and string literals, then appends closing
    tokens in reverse order.  It does **not** try to be a full parser — just
    good enough for the common streaming case where JSON is cut off mid-value.
    """
    # Stack of open tokens we need to close.
    stack: list[str] = []
    in_string = False
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]

        if in_string:
            if ch == "\\":
                # Skip escaped character.
                i += 2
                continue
            if ch == '"':
                in_string = False
                stack.pop()  # Remove the '"' we pushed.
            i += 1
            continue

        if ch == '"':
            in_string = True
            stack.append('"')
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch == "}":
            # Pop matching '{' closer.
            if stack and stack[-1] == "}":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "]":
                stack.pop()

        i += 1

    # Close everything that's still open, innermost first.
    closing = "".join(reversed(stack))
    return text + closing


def parse_streaming_json(text: str | None) -> dict:
    """Parse potentially incomplete or malformed JSON from streaming.

    Uses a 3-tier fallback:
      1. ``json.loads(text)``
      2. ``json.loads(repair_json(text))``
      3. Partial-parse: repair + close open braces/brackets/strings
      4. ``{}`` as last resort

    Always returns a *dict* (or at minimum ``{}``) so callers never need
    to handle parse failures.
    """
    if not text or not text.strip():
        return {}

    # Tier 1: direct parse.
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, ValueError):
        pass

    # Tier 2: repair then parse.
    repaired = repair_json(text)
    if repaired != text:
        try:
            result = json.loads(repaired)
            return result if isinstance(result, dict) else {}
        except (json.JSONDecodeError, ValueError):
            pass

    # Tier 3: partial parse (repair + close).
    try:
        closed = _close_partial_json(repaired)
        result = json.loads(closed)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, ValueError):
        pass

    # Tier 4: give up.
    return {}
