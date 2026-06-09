from __future__ import annotations

import asyncio
from typing import Any

from cubepi.middleware.compaction import CompactionState
from cubepi.middleware.compaction.summarizer import (
    _format_message_for_summary,
    summarize,
)
from cubepi.providers.base import (
    AssistantMessage,
    BoundModel,
    Message,
    Model,
    StreamOptions,
    TextContent,
    ToolCall,
    ToolDefinition,
    UserMessage,
)


class _FakeProvider:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: Any = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking=None,
        thinking_budgets=None,
    ) -> AssistantMessage:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "options": options,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "thinking": thinking,
                "thinking_budgets": thinking_budgets,
            }
        )
        return AssistantMessage(content=[TextContent(text=self.reply)])


async def test_summarize_uses_provider_generate_with_common_overrides() -> None:
    provider = _FakeProvider(" Compressed summary. ")
    model = Model(id="summary-model", provider_id="faux")
    signal = asyncio.Event()

    result = await summarize(
        model=BoundModel(provider=provider, spec=model),
        messages_to_summarize=[
            UserMessage(content=[TextContent(text="hello")]),
            AssistantMessage(content=[TextContent(text="hi")]),
        ],
        existing=None,
        max_summary_tokens=512,
        abort_signal=signal,
    )

    assert isinstance(result, CompactionState)
    assert result.summary == "Compressed summary."
    assert len(result.summarized_message_refs) == 2
    assert provider.calls[0]["max_output_tokens"] == 512
    assert provider.calls[0]["temperature"] == 0.0
    assert provider.calls[0]["thinking"] == "off"
    assert provider.calls[0]["options"].signal is signal


async def test_summarize_merges_existing_state() -> None:
    provider = _FakeProvider("Merged summary.")
    existing = CompactionState(summary="Older context.")

    result = await summarize(
        model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        messages_to_summarize=[UserMessage(content=[TextContent(text="new")])],
        existing=existing,
    )

    assert "Older context." in provider.calls[0]["system_prompt"]
    assert result.summary == "Merged summary."


async def test_summarize_raises_on_provider_error_message() -> None:
    class _ErrorProvider(_FakeProvider):
        async def generate(
            self,
            model: Model,
            messages: list[Message],
            *,
            system_prompt: str = "",
            tools: list[ToolDefinition] | None = None,
            tool_choice: Any = None,
            options: StreamOptions | None = None,
            max_output_tokens: int | None = None,
            temperature: float | None = None,
            thinking=None,
            thinking_budgets=None,
        ) -> AssistantMessage:
            del model, messages, system_prompt, tools, tool_choice, options
            del max_output_tokens, temperature, thinking, thinking_budgets
            return AssistantMessage(
                content=[],
                stop_reason="error",
                error_message="summary failed",
            )

    try:
        await summarize(
            model=BoundModel(
                provider=_ErrorProvider(""),
                spec=Model(id="summary-model", provider_id="faux"),
            ),
            messages_to_summarize=[UserMessage(content=[TextContent(text="new")])],
            existing=None,
        )
    except RuntimeError as exc:
        assert str(exc) == "summary failed"
    else:  # pragma: no cover
        raise AssertionError("provider error was not raised")


def test_format_message_for_summary_includes_tool_calls_and_text_like_blocks() -> None:
    class _TextLike:
        text = "extra text"

    class _Transcript:
        content = [
            TextContent(text="checking"),
            ToolCall(id="t1", name="lookup", arguments={"q": "x"}),
            _TextLike(),
        ]

    message = _Transcript()

    formatted = _format_message_for_summary(message)  # type: ignore[arg-type]

    assert "[_transcript]" in formatted
    assert "checking" in formatted
    assert "[tool_call:lookup]" in formatted
    assert "extra text" in formatted


def test_tool_call_arguments_included() -> None:
    msg = AssistantMessage(
        content=[
            ToolCall(
                id="c1",
                name="read_file",
                arguments={"path": "/home/user/config.py"},
            ),
        ]
    )
    formatted = _format_message_for_summary(msg)
    assert "read_file" in formatted
    assert "/home/user/config.py" in formatted


def test_tool_call_long_string_value_truncated() -> None:
    big_content = "x" * 1000
    msg = AssistantMessage(
        content=[
            ToolCall(
                id="c1",
                name="write_file",
                arguments={"path": "out.py", "content": big_content},
            ),
        ]
    )
    formatted = _format_message_for_summary(msg)
    # Short field survives intact
    assert "out.py" in formatted
    # Long field gets truncated
    assert big_content not in formatted
    assert "truncated" in formatted


def test_tool_call_short_arguments_kept_intact() -> None:
    msg = AssistantMessage(
        content=[
            ToolCall(id="c1", name="bash", arguments={"command": "ls -la"}),
        ]
    )
    formatted = _format_message_for_summary(msg)
    assert "bash" in formatted
    assert "ls -la" in formatted


def test_tool_call_repr_max_chars_enforced() -> None:
    # Many small fields — each individually under the per-field limit, but
    # the total serialised JSON would balloon. Cap at _ARG_REPR_MAX.
    msg = AssistantMessage(
        content=[
            ToolCall(
                id="c1",
                name="search",
                arguments={f"k{i}": f"v{i}" * 20 for i in range(40)},
            ),
        ]
    )
    formatted = _format_message_for_summary(msg)
    # Total formatted message must stay bounded.
    assert len(formatted) < 1000


def test_tool_call_empty_arguments() -> None:
    msg = AssistantMessage(content=[ToolCall(id="c1", name="ping", arguments={})])
    formatted = _format_message_for_summary(msg)
    assert "[tool_call:ping]" in formatted


# --- dynamic summary budget ---


def test_dynamic_budget_floor_for_small_content() -> None:
    from cubepi.middleware.compaction.summarizer import _dynamic_summary_budget

    small = [UserMessage(content=[TextContent(text="hi")])]
    assert _dynamic_summary_budget(small) == 1024


def test_dynamic_budget_scales_with_content() -> None:
    from cubepi.middleware.compaction.summarizer import _dynamic_summary_budget

    # 40 000 chars ≈ 20 000 tokens → budget = 20 000 * 0.15 = 3 000
    large = [UserMessage(content=[TextContent(text="x" * 40_000)])]
    budget = _dynamic_summary_budget(large)
    assert budget > 1024
    assert budget <= 4096


def test_dynamic_budget_empty_input_floor() -> None:
    from cubepi.middleware.compaction.summarizer import _dynamic_summary_budget

    assert _dynamic_summary_budget([]) == 1024


def test_dynamic_budget_ceiling() -> None:
    from cubepi.middleware.compaction.summarizer import _dynamic_summary_budget

    huge = [UserMessage(content=[TextContent(text="x" * 200_000)])]
    assert _dynamic_summary_budget(huge) == 4096


async def test_summarize_uses_dynamic_budget_when_none() -> None:
    provider = _FakeProvider("Summary.")
    # max_summary_tokens omitted (None default) → use dynamic
    await summarize(
        model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        messages_to_summarize=[UserMessage(content=[TextContent(text="x" * 40_000)])],
        existing=None,
    )
    # 20 000 tokens * 0.15 = 3 000
    captured = provider.calls[0]["max_output_tokens"]
    assert captured > 1024
    assert captured <= 4096


async def test_summarize_explicit_override_used_verbatim() -> None:
    provider = _FakeProvider("Summary.")
    await summarize(
        model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        messages_to_summarize=[UserMessage(content=[TextContent(text="x" * 40_000)])],
        existing=None,
        max_summary_tokens=777,
    )
    assert provider.calls[0]["max_output_tokens"] == 777


async def test_summarize_ref_messages_used_for_refs() -> None:
    """When ref_messages is supplied, refs are taken from it (not from
    messages_to_summarize). Needed for the pre-pruning case where the
    transcript is built from pruned content but state must reflect originals."""
    provider = _FakeProvider("Summary.")
    transcript = [UserMessage(content=[TextContent(text="pruned content")])]
    original = [UserMessage(content=[TextContent(text="original full content")])]

    state = await summarize(
        model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        messages_to_summarize=transcript,
        ref_messages=original,
        existing=None,
        max_summary_tokens=512,
    )

    # The transcript sent to the LLM contains "pruned content"
    transcript_text = provider.calls[0]["messages"][0].content[0].text
    assert "pruned content" in transcript_text
    # But refs are computed from the ORIGINAL messages
    from cubepi.middleware.compaction.state import message_refs

    assert state.summarized_message_refs == message_refs(original)


# --- static fallback summary ---


def test_fallback_summary_includes_user_requests() -> None:
    from cubepi.middleware.compaction.summarizer import build_fallback_summary

    msgs: list[Message] = [
        UserMessage(content=[TextContent(text="Please write a hello world script")]),
        AssistantMessage(content=[TextContent(text="Sure")]),
    ]
    state = build_fallback_summary(msgs, existing=None)
    assert state.is_fallback is True
    assert "Please write a hello world script" in state.summary


def test_fallback_summary_includes_tool_names() -> None:
    from cubepi.middleware.compaction.summarizer import build_fallback_summary
    from cubepi.providers.base import ToolResultMessage

    msgs: list[Message] = [
        UserMessage(content=[TextContent(text="run the tests")]),
        AssistantMessage(
            content=[ToolCall(id="c1", name="bash", arguments={"command": "pytest"})]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="bash",
            content=[TextContent(text="3 passed")],
        ),
    ]
    state = build_fallback_summary(msgs, existing=None)
    assert "bash" in state.summary
    assert state.is_fallback is True


def test_fallback_summary_merges_existing() -> None:
    from cubepi.middleware.compaction.state import CompactionState
    from cubepi.middleware.compaction.summarizer import build_fallback_summary

    existing = CompactionState(summary="prior context", is_fallback=False)
    msgs: list[Message] = [UserMessage(content=[TextContent(text="new task")])]
    state = build_fallback_summary(msgs, existing=existing)
    assert "prior context" in state.summary
    assert state.is_fallback is True


def test_fallback_summary_caps_user_requests_at_five() -> None:
    from cubepi.middleware.compaction.summarizer import build_fallback_summary

    msgs: list[Message] = [
        UserMessage(content=[TextContent(text=f"request {i}")]) for i in range(10)
    ]
    state = build_fallback_summary(msgs, existing=None)
    assert "request 0" in state.summary
    assert "request 4" in state.summary
    assert "request 5" not in state.summary  # capped at 5


def test_fallback_summary_uses_ref_messages_for_refs() -> None:
    from cubepi.middleware.compaction.state import message_refs
    from cubepi.middleware.compaction.summarizer import build_fallback_summary

    transcript = [UserMessage(content=[TextContent(text="pruned content")])]
    original = [UserMessage(content=[TextContent(text="full original")])]
    state = build_fallback_summary(transcript, ref_messages=original, existing=None)
    assert state.summarized_message_refs == message_refs(original)


# --- Task 7: structured prompt + override hooks ---


def test_summary_has_eight_sections() -> None:
    from cubepi.middleware.compaction.summarizer import SUMMARIZER_SYSTEM_PROMPT

    for section in (
        "Goal",
        "Constraints & preferences",
        "Completed actions",
        "Key decisions",
        "Resolved",
        "Pending",
        "Relevant artifacts",
        "Remaining work",
    ):
        assert f"## {section}" in SUMMARIZER_SYSTEM_PROMPT


def test_system_prompt_marks_output_as_non_instruction() -> None:
    from cubepi.middleware.compaction.summarizer import SUMMARIZER_SYSTEM_PROMPT

    # Collapse whitespace so line breaks don't fail the substring assertions.
    text = " ".join(SUMMARIZER_SYSTEM_PROMPT.lower().split())
    assert "reference material" in text
    assert "user message wins" in text


async def test_summarize_uses_system_prompt_override() -> None:
    provider = _FakeProvider("CUSTOM")
    await summarize(
        model=BoundModel(
            provider=provider,
            spec=Model(id="m", provider_id="faux"),
        ),
        messages_to_summarize=[UserMessage(content=[TextContent(text="hi")])],
        existing=None,
        system_prompt_override="CUSTOM PROMPT BODY",
    )
    assert provider.calls[0]["system_prompt"] == "CUSTOM PROMPT BODY"


async def test_summarize_uses_existing_summary_suffix_override() -> None:
    provider = _FakeProvider("merged")
    existing = CompactionState(summary="prior")
    await summarize(
        model=BoundModel(
            provider=provider,
            spec=Model(id="m", provider_id="faux"),
        ),
        messages_to_summarize=[UserMessage(content=[TextContent(text="hi")])],
        existing=existing,
        system_prompt_override="BASE",
        existing_summary_suffix="MERGE THIS: {prev}",
    )
    captured = provider.calls[0]["system_prompt"]
    assert captured.startswith("BASE")
    assert "MERGE THIS: prior" in captured


# --- coverage: _shrink_strings list + fall-through; _format_arguments non-JSON ---


def test_shrink_strings_recurses_into_lists() -> None:
    """The list branch of _shrink_strings shrinks string leaves inside lists,
    leaves non-string leaves intact."""
    from cubepi.middleware.compaction.summarizer import _shrink_strings

    long_text = "x" * 500
    obj = ["short", long_text, 42, True, None, [long_text]]
    shrunk = _shrink_strings(obj)
    assert isinstance(shrunk, list)
    assert shrunk[0] == "short"
    assert shrunk[1].startswith("x" * 200) and "truncated" in shrunk[1]
    assert shrunk[2] == 42
    assert shrunk[3] is True
    assert shrunk[4] is None
    # Nested list also recurses.
    assert "truncated" in shrunk[5][0]


def test_format_arguments_non_json_serialisable_falls_back_to_str() -> None:
    """A value json.dumps can't handle (e.g. a custom object) falls back to
    str() instead of raising — defensive against odd backend payloads."""
    from cubepi.middleware.compaction.summarizer import _format_arguments

    class _NotJson:
        def __repr__(self) -> str:
            return "<not-json>"

    # Pass a dict where one VALUE is non-serialisable. _shrink_strings does
    # NOT recurse into custom objects (returns them as-is), so json.dumps
    # raises TypeError → fallback to str().
    result = _format_arguments({"x": _NotJson()})
    assert "<not-json>" in result


# --- codex round 3: fallback does not recurse ---


def test_fallback_summary_does_not_embed_prior_fallback() -> None:
    """A fallback that follows another fallback must NOT embed the prior
    fallback text under 'Prior context:'. Otherwise each LLM-outage turn
    doubles the summary size — the chain compounds verbatim across runs."""
    from cubepi.middleware.compaction.summarizer import build_fallback_summary

    # Run 1: clean fallback.
    msgs_1: list[Message] = [
        UserMessage(content=[TextContent(text="task A")]),
        AssistantMessage(
            content=[ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})]
        ),
    ]
    state_1 = build_fallback_summary(msgs_1, existing=None)
    assert state_1.is_fallback is True
    assert "task A" in state_1.summary

    # Run 2: NEW fallback whose existing is run 1's fallback.
    msgs_2: list[Message] = [
        UserMessage(content=[TextContent(text="task B")]),
        AssistantMessage(
            content=[ToolCall(id="c2", name="grep", arguments={"q": "x"})]
        ),
    ]
    state_2 = build_fallback_summary(msgs_2, existing=state_1)

    # The prior fallback text MUST NOT appear verbatim as a "Prior context:"
    # line — that's the unbounded-growth pattern.
    assert "Prior context:" not in state_2.summary
    # But the structured fields from the prior fallback ARE preserved by
    # merging into the new fallback's user_lines / tool_names.
    assert "task A" in state_2.summary
    assert "task B" in state_2.summary
    assert "bash" in state_2.summary
    assert "grep" in state_2.summary
    # Tool names are deduplicated, not appended.
    assert state_2.summary.count("bash") == 1


def test_fallback_summary_user_lines_capped_after_merging() -> None:
    """Cap of 5 user lines holds across the merge with a prior fallback."""
    from cubepi.middleware.compaction.summarizer import build_fallback_summary

    msgs_1 = [UserMessage(content=[TextContent(text=f"older {i}")]) for i in range(4)]
    state_1 = build_fallback_summary(msgs_1, existing=None)

    msgs_2 = [UserMessage(content=[TextContent(text=f"newer {i}")]) for i in range(4)]
    state_2 = build_fallback_summary(msgs_2, existing=state_1)

    # 4 prior + 4 new = 8 candidates, capped at 5.
    user_line_section = [
        ln for ln in state_2.summary.splitlines() if ln.startswith("User requests:")
    ][0]
    items = [
        s.strip() for s in user_line_section.removeprefix("User requests: ").split(";")
    ]
    assert len(items) == 5


def test_fallback_summary_after_real_summary_still_embeds_prior() -> None:
    """When the prior was a REAL summary (is_fallback=False), the new
    fallback still embeds it under 'Prior context:' — only fallback-after-
    fallback is the unbounded-growth path."""
    from cubepi.middleware.compaction.state import CompactionState
    from cubepi.middleware.compaction.summarizer import build_fallback_summary

    real_prior = CompactionState(summary="## Goal\nbuild the thing", is_fallback=False)
    msgs = [UserMessage(content=[TextContent(text="next task")])]
    state = build_fallback_summary(msgs, existing=real_prior)
    assert "Prior context:" in state.summary
    assert "build the thing" in state.summary


def test_fallback_preserves_real_prior_across_multiple_outage_turns() -> None:
    """Codex P2: real summary → fallback → fallback. The 2nd fallback must
    still carry the real summary's prior context, otherwise an outage of
    more than one compaction cycle drops everything summarised before it."""
    from cubepi.middleware.compaction.state import CompactionState
    from cubepi.middleware.compaction.summarizer import build_fallback_summary

    # Step 1: a real LLM summary covering "early work".
    real_summary = CompactionState(
        summary="## Goal\nbuild the thing\n## Decisions\nused approach X",
        is_fallback=False,
    )

    # Step 2: LLM goes down → first fallback, which embeds the real summary.
    msgs_fb1 = [
        UserMessage(content=[TextContent(text="task during outage 1")]),
        AssistantMessage(
            content=[ToolCall(id="c1", name="bash", arguments={"q": "x"})]
        ),
    ]
    fb1 = build_fallback_summary(msgs_fb1, existing=real_summary)
    assert "build the thing" in fb1.summary  # real summary embedded
    assert "task during outage 1" in fb1.summary

    # Step 3: LLM STILL down → second fallback chained to first fallback.
    msgs_fb2 = [
        UserMessage(content=[TextContent(text="task during outage 2")]),
        AssistantMessage(
            content=[ToolCall(id="c2", name="grep", arguments={"q": "y"})]
        ),
    ]
    fb2 = build_fallback_summary(msgs_fb2, existing=fb1)

    # Critical: the real prior context (## Goal / ## Decisions) must
    # still be reachable in the 2nd fallback. Otherwise multi-turn
    # outages erase everything pre-outage.
    assert "build the thing" in fb2.summary
    assert "used approach X" in fb2.summary
    # User-line merging still works.
    assert "task during outage 1" in fb2.summary
    assert "task during outage 2" in fb2.summary
    # Tool-name merging still works (dedup + ordering).
    assert "bash" in fb2.summary
    assert "grep" in fb2.summary
