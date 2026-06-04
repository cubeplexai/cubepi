"""Tests for cubepi.errors — classifier heuristics and typed error taxonomy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cubepi.errors import (
    ContextLengthExceeded,
    ProviderAuthFailed,
    ProviderBadRequest,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
    _estimate_input_tokens,
    _retry_after_from,
    _status_of,
    classify_and_raise,
)
from cubepi.providers.base import Model, TextContent, UserMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(
    provider: str = "openai",
    model_id: str = "gpt-4o",
    context_window: int = 128_000,
) -> Model:
    return Model(id=model_id, provider=provider, context_window=context_window)


def _messages(text: str = "hello") -> list:
    return [UserMessage(content=[TextContent(text=text)])]


class _FakeExc(Exception):
    """Fake SDK exception with configurable status_code, message, and headers.

    ``status_code`` sets both the direct attr AND ``.response.status_code`` so
    existing classifier tests work. Pass ``response_status_code`` instead to
    put the code only on ``.response`` (for testing the fallback path).
    """

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        headers: dict | None = None,
        response_status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        response_code = (
            response_status_code if response_status_code is not None else status_code
        )
        if response_code is not None or headers is not None:
            self.response = SimpleNamespace(
                status_code=response_code,
                headers=headers or {},
            )


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------


class TestClassifyContextLengthExceeded:
    def test_explicit_context_length_message(self) -> None:
        exc = _FakeExc("This model's maximum context length is 4096 tokens.")
        with pytest.raises(ContextLengthExceeded) as ei:
            classify_and_raise(exc, model=_model())
        assert ei.value.__cause__ is exc

    def test_exceed_pattern(self) -> None:
        exc = _FakeExc("context length exceeded: reduce your input")
        with pytest.raises(ContextLengthExceeded):
            classify_and_raise(exc, model=_model())

    def test_too_many_tokens_pattern(self) -> None:
        exc = _FakeExc("too many tokens in input")
        with pytest.raises(ContextLengthExceeded):
            classify_and_raise(exc, model=_model())

    def test_prompt_is_too_long_pattern(self) -> None:
        exc = _FakeExc("prompt is too long")
        with pytest.raises(ContextLengthExceeded):
            classify_and_raise(exc, model=_model())

    def test_reduce_messages_pattern(self) -> None:
        exc = _FakeExc("Please reduce the messages sent in your request.")
        with pytest.raises(ContextLengthExceeded):
            classify_and_raise(exc, model=_model())

    def test_carries_provider_and_model_fields(self) -> None:
        exc = _FakeExc("maximum context length exceeded", status_code=400)
        with pytest.raises(ContextLengthExceeded) as ei:
            classify_and_raise(exc, model=_model(provider="openai", model_id="gpt-4o"))
        err = ei.value
        assert err.provider == "openai"
        assert err.model == "gpt-4o"
        assert err.status_code == 400

    def test_volcano_invalid_parameter_400_with_oversize_tokens(self) -> None:
        """Volcano ARK sends opaque 400 InvalidParameter when over-context."""
        # Build a message whose chars/4 estimate is >= 95% of context_window (100k).
        # context_window=100_000; 95% threshold = 95_000 tokens = 380_000 chars.
        big_text = "x" * 400_000
        msgs = [UserMessage(content=[TextContent(text=big_text)])]
        exc = _FakeExc("InvalidParameter", status_code=400)
        with pytest.raises(ContextLengthExceeded) as ei:
            classify_and_raise(exc, model=_model(context_window=100_000), messages=msgs)
        assert ei.value.tokens_in is not None
        assert ei.value.context_window == 100_000

    def test_400_below_95_percent_threshold_not_context_length(self) -> None:
        """400 with small messages should NOT be classified as ContextLengthExceeded."""
        msgs = _messages("short prompt")
        exc = _FakeExc("InvalidParameter", status_code=400)
        with pytest.raises(ProviderBadRequest):
            classify_and_raise(exc, model=_model(context_window=100_000), messages=msgs)


class TestClassifyRateLimited:
    def test_429_raises_rate_limited(self) -> None:
        exc = _FakeExc("Too Many Requests", status_code=429)
        with pytest.raises(RateLimited) as ei:
            classify_and_raise(exc, model=_model())
        assert ei.value.status_code == 429

    def test_retry_after_parsed_from_headers(self) -> None:
        exc = _FakeExc(
            "rate limit exceeded",
            status_code=429,
            headers={"retry-after": "42"},
        )
        with pytest.raises(RateLimited) as ei:
            classify_and_raise(exc, model=_model())
        assert ei.value.retry_after == 42.0

    def test_quota_wording_triggers_rate_limited(self) -> None:
        # Pattern: "quota (?:exceed|exhaust|limit|reach)" — matches "quota exceeded"
        exc = _FakeExc("quota exceeded for your account", status_code=200)
        with pytest.raises(RateLimited):
            classify_and_raise(exc, model=_model())

    def test_403_with_quota_wording_raises_rate_limited(self) -> None:
        """Anthropic-style 403 with 'quota limit' text → RateLimited, not AuthFailed."""
        exc = _FakeExc(
            "Your account has reached its quota limit. Please upgrade.",
            status_code=403,
        )
        with pytest.raises(RateLimited):
            classify_and_raise(exc, model=_model())


class TestClassifyProviderAuthFailed:
    def test_401_raises_provider_auth_failed(self) -> None:
        exc = _FakeExc("Unauthorized", status_code=401)
        with pytest.raises(ProviderAuthFailed) as ei:
            classify_and_raise(exc, model=_model())
        assert ei.value.status_code == 401

    def test_403_without_quota_wording_raises_provider_auth_failed(self) -> None:
        exc = _FakeExc("Forbidden", status_code=403)
        with pytest.raises(ProviderAuthFailed):
            classify_and_raise(exc, model=_model())

    def test_carries_provider_context(self) -> None:
        exc = _FakeExc("Invalid API key", status_code=401)
        with pytest.raises(ProviderAuthFailed) as ei:
            classify_and_raise(
                exc, model=_model(provider="anthropic", model_id="claude-3-5-sonnet")
            )
        assert ei.value.provider == "anthropic"


class TestClassifyProviderUnavailable:
    def test_5xx_raises_provider_unavailable(self) -> None:
        exc = _FakeExc("Internal Server Error", status_code=500)
        with pytest.raises(ProviderUnavailable) as ei:
            classify_and_raise(exc, model=_model())
        assert ei.value.status_code == 500

    def test_503_raises_provider_unavailable(self) -> None:
        exc = _FakeExc("Service Unavailable", status_code=503)
        with pytest.raises(ProviderUnavailable):
            classify_and_raise(exc, model=_model())

    def test_timeout_error_raises_provider_unavailable(self) -> None:
        exc = TimeoutError("request timed out")
        with pytest.raises(ProviderUnavailable):
            classify_and_raise(exc, model=_model())

    def test_connection_error_raises_provider_unavailable(self) -> None:
        exc = ConnectionError("failed to connect")
        with pytest.raises(ProviderUnavailable):
            classify_and_raise(exc, model=_model())


class TestClassifyProviderBadRequest:
    def test_generic_400_raises_provider_bad_request(self) -> None:
        exc = _FakeExc("Bad Request", status_code=400)
        with pytest.raises(ProviderBadRequest) as ei:
            classify_and_raise(exc, model=_model())
        assert ei.value.status_code == 400

    def test_404_raises_provider_bad_request(self) -> None:
        exc = _FakeExc("model not found", status_code=404)
        with pytest.raises(ProviderBadRequest):
            classify_and_raise(exc, model=_model())


class TestClassifyUnknown:
    def test_unknown_exception_re_raised_as_is(self) -> None:
        """A plain RuntimeError with no status code propagates unchanged."""
        exc = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom") as ei:
            classify_and_raise(exc, model=_model())
        assert ei.value is exc

    def test_already_typed_error_re_raised_unchanged(self) -> None:
        """A ProviderError passed in is re-raised without double-wrapping."""
        original = ContextLengthExceeded(
            "already typed", provider="openai", model="gpt-4o"
        )
        with pytest.raises(ContextLengthExceeded) as ei:
            classify_and_raise(original, model=_model())
        assert ei.value is original


class TestProviderErrorInheritance:
    def test_all_subclasses_are_provider_error(self) -> None:
        for cls in (
            ContextLengthExceeded,
            RateLimited,
            ProviderAuthFailed,
            ProviderUnavailable,
            ProviderBadRequest,
        ):
            err = cls("msg", provider="p", model="m")
            assert isinstance(err, ProviderError)
            assert isinstance(err, Exception)

    def test_context_length_carries_token_fields(self) -> None:
        err = ContextLengthExceeded(
            "too long",
            provider="openai",
            model="gpt-4o",
            tokens_in=5000,
            context_window=4096,
        )
        assert err.tokens_in == 5000
        assert err.context_window == 4096

    def test_rate_limited_carries_retry_after(self) -> None:
        err = RateLimited(
            "slow down", provider="openai", model="gpt-4o", retry_after=30.0
        )
        assert err.retry_after == 30.0


# ---------------------------------------------------------------------------
# Coverage fillers — internal helpers exercised directly.
# ---------------------------------------------------------------------------


class TestStatusOf:
    def test_direct_status_code(self) -> None:
        exc = _FakeExc("msg", status_code=400)
        assert _status_of(exc) == 400

    def test_response_fallback(self) -> None:
        """exc has no .status_code but .response.status_code — fallback path."""
        exc = _FakeExc("msg", response_status_code=503)
        assert getattr(exc, "status_code", None) is None
        assert _status_of(exc) == 503

    def test_no_status(self) -> None:
        exc = RuntimeError("no status anywhere")
        assert _status_of(exc) is None


class TestEstimateInputTokens:
    def test_none_returns_none(self) -> None:
        assert _estimate_input_tokens(None) is None

    def test_empty_returns_none(self) -> None:
        assert _estimate_input_tokens([]) is None

    def test_user_message_with_text_content(self) -> None:
        msgs = _messages("hello world")
        est = _estimate_input_tokens(msgs)
        assert est >= 2

    def test_block_with_content_not_text_attr(self) -> None:
        """Block that has .content instead of .text — non-TextContent blocks."""
        block = SimpleNamespace(text=None, content="tool result text")
        msg = type("_Msg", (), {"content": [block]})()
        est = _estimate_input_tokens([msg])
        assert est is not None and est >= 3

    def test_string_content(self) -> None:
        """Message with list[str] content blocks is NOT expected but is handled."""

        # Actually test the elif isinstance(content, str) branch via a mock.
        class _StrMsg:
            content = "plain string content for token estimation"

        est = _estimate_input_tokens([_StrMsg])  # type: ignore[list-item]
        assert est is not None and est >= 7

    def test_zero_total_returns_none(self) -> None:
        """Empty strings produce zero total."""
        msgs = [UserMessage(content=[TextContent(text="")])]
        assert _estimate_input_tokens(msgs) is None


class TestRetryAfterFrom:
    def test_lowercase_retry_after_header(self) -> None:
        exc = _FakeExc("rate limited", status_code=429, headers={"retry-after": "60"})
        assert _retry_after_from(exc) == 60.0

    def test_title_case_retry_after_header(self) -> None:
        exc = _FakeExc("rate limited", status_code=429, headers={"Retry-After": "30"})
        assert _retry_after_from(exc) == 30.0

    def test_no_headers_returns_none(self) -> None:
        exc = _FakeExc("rate limited", status_code=429)
        assert _retry_after_from(exc) is None

    def test_no_retry_after_key_returns_none(self) -> None:
        exc = _FakeExc("rate limited", status_code=429, headers={"x-other": "1"})
        assert _retry_after_from(exc) is None

    def test_non_numeric_value_returns_none(self) -> None:
        exc = _FakeExc(
            "rate limited", status_code=429, headers={"retry-after": "not-a-number"}
        )
        assert _retry_after_from(exc) is None
