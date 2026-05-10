from cubepi.providers.base import Model
from cubepi.providers.faux import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)


def _make_model() -> Model:
    return Model(id="faux-1", provider="faux")


class TestContentIndexTextOnly:
    """FauxProvider text-only stream: all text events have content_index=0."""

    async def test_text_only_content_index(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("hello world")])

        stream = await provider.stream(_make_model(), [])
        events = [e async for e in stream]

        # "start" event should have content_index=None
        start_events = [e for e in events if e.type == "start"]
        assert len(start_events) == 1
        assert start_events[0].content_index is None

        # All text_start, text_delta, text_end events should have content_index=0
        text_events = [
            e for e in events if e.type in ("text_start", "text_delta", "text_end")
        ]
        assert len(text_events) >= 3  # at least start + 1 delta + end
        for e in text_events:
            assert e.content_index == 0, f"{e.type} had content_index={e.content_index}"


class TestContentIndexThinkingAndText:
    """FauxProvider thinking+text: thinking events index 0, text events index 1."""

    async def test_thinking_then_text_content_index(self):
        provider = FauxProvider()
        provider.set_responses(
            [faux_assistant_message([faux_thinking("let me think"), faux_text("answer")])]
        )

        stream = await provider.stream(_make_model(), [])
        events = [e async for e in stream]

        thinking_events = [
            e
            for e in events
            if e.type in ("thinking_start", "thinking_delta", "thinking_end")
        ]
        assert len(thinking_events) >= 3
        for e in thinking_events:
            assert (
                e.content_index == 0
            ), f"{e.type} had content_index={e.content_index}"

        text_events = [
            e for e in events if e.type in ("text_start", "text_delta", "text_end")
        ]
        assert len(text_events) >= 3
        for e in text_events:
            assert (
                e.content_index == 1
            ), f"{e.type} had content_index={e.content_index}"


class TestContentIndexTextAndToolCall:
    """FauxProvider text+tool_call: text events index 0, tool events index 1."""

    async def test_text_then_tool_content_index(self):
        provider = FauxProvider()
        provider.set_responses(
            [
                faux_assistant_message(
                    [faux_text("searching"), faux_tool_call("search", {"q": "test"}, id="tc-1")],
                    stop_reason="tool_use",
                )
            ]
        )

        stream = await provider.stream(_make_model(), [])
        events = [e async for e in stream]

        text_events = [
            e for e in events if e.type in ("text_start", "text_delta", "text_end")
        ]
        assert len(text_events) >= 3
        for e in text_events:
            assert (
                e.content_index == 0
            ), f"{e.type} had content_index={e.content_index}"

        tool_events = [
            e
            for e in events
            if e.type in ("toolcall_start", "toolcall_delta", "toolcall_end")
        ]
        assert len(tool_events) >= 3
        for e in tool_events:
            assert (
                e.content_index == 1
            ), f"{e.type} had content_index={e.content_index}"
