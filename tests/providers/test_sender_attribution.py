"""Tests for sender_attribution helpers used by group chat."""

from cubepi.providers.base import (
    ImageContent,
    TextContent,
    UserMessage,
    apply_sender_attribution,
    get_sender_display_name,
)


class TestGetSenderDisplayName:
    def test_returns_name_when_present(self):
        msg = UserMessage(
            content=[TextContent(text="hi")],
            metadata={"sender_display_name": "Alice"},
        )
        assert get_sender_display_name(msg) == "Alice"

    def test_returns_none_when_missing(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        assert get_sender_display_name(msg) is None

    def test_returns_none_for_empty_string(self):
        msg = UserMessage(
            content=[TextContent(text="hi")],
            metadata={"sender_display_name": ""},
        )
        assert get_sender_display_name(msg) is None

    def test_returns_none_for_non_string(self):
        msg = UserMessage(
            content=[TextContent(text="hi")],
            metadata={"sender_display_name": 42},
        )
        assert get_sender_display_name(msg) is None


class TestApplySenderAttribution:
    def test_no_metadata_returns_content_unchanged(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        out = apply_sender_attribution(msg, msg.content)
        assert out == msg.content

    def test_prefixes_first_text_block(self):
        msg = UserMessage(
            content=[TextContent(text="hello world")],
            metadata={"sender_display_name": "Alice"},
        )
        out = apply_sender_attribution(msg, msg.content)
        assert len(out) == 1
        assert isinstance(out[0], TextContent)
        assert out[0].text == "[Alice]: hello world"

    def test_only_first_text_block_gets_prefix(self):
        msg = UserMessage(
            content=[
                TextContent(text="first"),
                TextContent(text="second"),
            ],
            metadata={"sender_display_name": "Alice"},
        )
        out = apply_sender_attribution(msg, msg.content)
        assert isinstance(out[0], TextContent)
        assert isinstance(out[1], TextContent)
        assert out[0].text == "[Alice]: first"
        assert out[1].text == "second"

    def test_skips_empty_text_block_when_prefixing(self):
        msg = UserMessage(
            content=[
                TextContent(text=""),
                TextContent(text="real content"),
            ],
            metadata={"sender_display_name": "Alice"},
        )
        out = apply_sender_attribution(msg, msg.content)
        assert isinstance(out[0], TextContent)
        assert isinstance(out[1], TextContent)
        assert out[0].text == ""
        assert out[1].text == "[Alice]: real content"

    def test_attachment_only_message_gets_synthetic_prefix(self):
        msg = UserMessage(
            content=[
                ImageContent(media_type="image/png", source="base64data"),
            ],
            metadata={"sender_display_name": "Alice"},
        )
        out = apply_sender_attribution(msg, msg.content)
        assert len(out) == 2
        assert isinstance(out[0], TextContent)
        assert out[0].text == "[Alice] sent:"
        assert isinstance(out[1], ImageContent)

    def test_does_not_mutate_input(self):
        msg = UserMessage(
            content=[TextContent(text="hi")],
            metadata={"sender_display_name": "Alice"},
        )
        original_text = msg.content[0].text
        apply_sender_attribution(msg, msg.content)
        assert msg.content[0].text == original_text

    def test_mixed_text_and_image_only_text_prefixed(self):
        msg = UserMessage(
            content=[
                TextContent(text="look at this"),
                ImageContent(media_type="image/png", source="base64data"),
            ],
            metadata={"sender_display_name": "Alice"},
        )
        out = apply_sender_attribution(msg, msg.content)
        assert isinstance(out[0], TextContent)
        assert out[0].text == "[Alice]: look at this"
        assert isinstance(out[1], ImageContent)
