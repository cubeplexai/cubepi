import pytest

from cubepi.errors import (
    ContextLengthExceeded,
    ProviderAuthFailed,
    ProviderBadRequest,
    ProviderUnavailable,
    RateLimited,
    classify_and_raise,
)
from cubepi.providers.images.types import ImagesModel


def _img_model() -> ImagesModel:
    return ImagesModel(id="gpt-image-1", provider_id="openai", api="openai-images")


class _StatusErr(Exception):
    def __init__(self, msg: str, status: int) -> None:
        super().__init__(msg)
        self.status_code = status


def test_rate_limit_via_status_429():
    with pytest.raises(RateLimited):
        classify_and_raise(_StatusErr("too many", 429), model=_img_model())


def test_auth_via_status_401():
    with pytest.raises(ProviderAuthFailed):
        classify_and_raise(_StatusErr("nope", 401), model=_img_model())


def test_auth_via_status_403():
    with pytest.raises(ProviderAuthFailed):
        classify_and_raise(_StatusErr("forbidden", 403), model=_img_model())


def test_unavailable_via_status_503():
    with pytest.raises(ProviderUnavailable):
        classify_and_raise(_StatusErr("down", 503), model=_img_model())


def test_bad_request_via_status_400():
    with pytest.raises(ProviderBadRequest):
        classify_and_raise(_StatusErr("bad", 400), model=_img_model())


def test_context_length_pattern_match():
    # Pattern wording wins regardless of status; just ensures images-side
    # model lookup doesn't blow up reading context_window.
    with pytest.raises(ContextLengthExceeded):
        classify_and_raise(
            _StatusErr("Request exceeds maximum context length", 400),
            model=_img_model(),
        )


def test_already_typed_passthrough():
    err = RateLimited("manual", provider="openai", model="gpt-image-1")
    with pytest.raises(RateLimited) as exc:
        classify_and_raise(err, model=_img_model())
    assert exc.value is err


def test_unknown_exception_falls_through_reraise():
    class _Weird(Exception): ...

    weird = _Weird("unknown")
    with pytest.raises(_Weird):
        classify_and_raise(weird, model=_img_model())


def test_chat_model_still_works():
    """Widening must not break the existing chat-side call path."""
    from cubepi.providers.base import Model

    chat_model = Model(
        id="claude-sonnet-4-6", provider_id="anthropic", context_window=200_000
    )
    with pytest.raises(RateLimited):
        classify_and_raise(_StatusErr("limit", 429), model=chat_model)


def test_uses_default_context_window_when_attribute_absent():
    """ImagesModel has no context_window; the function must not raise AttributeError."""
    assert not hasattr(_img_model(), "context_window")
    # Status 400 without context-length wording → ProviderBadRequest (not Context).
    with pytest.raises(ProviderBadRequest):
        classify_and_raise(_StatusErr("plain bad request", 400), model=_img_model())
