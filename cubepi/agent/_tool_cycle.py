"""Pre-completion tool-cycle invariant — spec §3.6.2.

For each AssistantMessage with ToolCall blocks {c1..cK}, the K-message
window immediately following MUST be all ToolResultMessages whose
tool_call_id MULTISET equals {c1..cK} — no extras, no missing, no
duplicates beyond what the assistant emitted, and no other
AssistantMessage or UserMessage may appear in that window.
"""

from __future__ import annotations

from collections import Counter

from cubepi.providers.base import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
)


class ToolCycleViolation(ValueError):
    def __init__(
        self,
        *,
        kind: str,
        assistant_index: int,
        expected: Counter,
        got: Counter,
    ) -> None:
        super().__init__(
            f"tool-cycle violation [{kind}] at assistant index "
            f"{assistant_index}: expected {dict(expected)}, "
            f"got {dict(got)}"
        )
        self.kind = kind
        self.assistant_index = assistant_index
        self.expected = expected
        self.got = got


def check_tool_cycle(messages: list[Message]) -> None:
    for i, m in enumerate(messages):
        if not isinstance(m, AssistantMessage):
            continue
        call_ids = [c.id for c in m.content if isinstance(c, ToolCall)]
        if not call_ids:
            continue
        expected = Counter(call_ids)
        k = len(call_ids)
        window = messages[i + 1 : i + 1 + k]
        if len(window) < k:
            raise ToolCycleViolation(
                kind="incomplete-window",
                assistant_index=i,
                expected=expected,
                got=Counter(),
            )
        for w in window:
            if not isinstance(w, ToolResultMessage):
                raise ToolCycleViolation(
                    kind="non-tool-result-in-window",
                    assistant_index=i,
                    expected=expected,
                    got=Counter(),
                )
        got = Counter(
            w.tool_call_id for w in window if isinstance(w, ToolResultMessage)
        )
        # Multiset equality — catches duplicates the spec rejects.
        if got != expected:
            raise ToolCycleViolation(
                kind="multiset-mismatch",
                assistant_index=i,
                expected=expected,
                got=got,
            )
