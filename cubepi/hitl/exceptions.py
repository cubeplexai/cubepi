from __future__ import annotations


class HitlControlException(BaseException):
    """Base for HITL control-flow exceptions.

    Inherits BaseException so existing `except Exception:` handlers in
    cubepi.agent.tools._prepare_tool_call and _execute_prepared do NOT
    swallow these — mirrors asyncio.CancelledError.
    """


class HitlCancelled(HitlControlException):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class HitlTimedOut(HitlControlException):
    def __init__(self, seconds: float):
        super().__init__(f"HITL request timed out after {seconds} seconds")
        self.seconds = seconds


class HitlDetached(HitlControlException):
    pass


class HitlAborted(HitlControlException):
    pass


class HitlError(Exception):
    """Base for caller-fixable HITL errors (misuse, not control flow)."""


class HitlConcurrencyError(HitlError):
    pass


class HitlStaleAnswer(HitlError):
    pass


class HitlNoPendingRequest(HitlError):
    pass


class HitlMissingAnswer(HitlError):
    pass


class HitlInconsistentState(HitlError):
    pass


class HitlDurabilityNotGuaranteed(HitlError):
    pass
