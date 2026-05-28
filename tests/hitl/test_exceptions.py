import pytest
from cubepi.hitl.exceptions import (
    HitlControlException,
    HitlCancelled,
    HitlTimedOut,
    HitlDetached,
    HitlAborted,
    HitlError,
    HitlConcurrencyError,
    HitlStaleAnswer,
    HitlNoPendingRequest,
    HitlMissingAnswer,
    HitlInconsistentState,
    HitlDurabilityNotGuaranteed,
)


def test_control_exceptions_are_baseexception_not_exception():
    for cls in (
        HitlControlException,
        HitlCancelled,
        HitlTimedOut,
        HitlDetached,
        HitlAborted,
    ):
        assert issubclass(cls, BaseException)
        assert not issubclass(cls, Exception)


def test_control_exception_subclassing():
    assert issubclass(HitlCancelled, HitlControlException)
    assert issubclass(HitlTimedOut, HitlControlException)
    assert issubclass(HitlDetached, HitlControlException)
    assert issubclass(HitlAborted, HitlControlException)


def test_regular_errors_are_exception():
    for cls in (
        HitlError,
        HitlConcurrencyError,
        HitlStaleAnswer,
        HitlNoPendingRequest,
        HitlMissingAnswer,
        HitlInconsistentState,
        HitlDurabilityNotGuaranteed,
    ):
        assert issubclass(cls, Exception)


def test_hitl_cancelled_carries_reason():
    exc = HitlCancelled("user clicked cancel")
    assert exc.reason == "user clicked cancel"
    assert "user clicked cancel" in str(exc)


def test_hitl_timed_out_carries_seconds():
    exc = HitlTimedOut(30.0)
    assert exc.seconds == 30.0
    assert "30" in str(exc)


def test_except_exception_does_not_catch_control():
    try:
        try:
            raise HitlCancelled("x")
        except Exception:
            pytest.fail("HitlCancelled should not be caught by except Exception")
    except HitlControlException as exc:
        assert exc.reason == "x"
