"""Human-in-the-Loop (HITL) primitives for cubepi agents.

See dev/specs/2026-05-28-hitl-channel.md for the full design.
"""

from cubepi.hitl.exceptions import (
    HitlAborted,
    HitlCancelled,
    HitlConcurrencyError,
    HitlControlException,
    HitlDetached,
    HitlDurabilityNotGuaranteed,
    HitlError,
    HitlInconsistentState,
    HitlMissingAnswer,
    HitlNoPendingRequest,
    HitlStaleAnswer,
    HitlTimedOut,
)
from cubepi.hitl.policy import (
    Approve,
    ApprovalDecision,
    AskUser,
    Deny,
)
from cubepi.hitl.ask_user import AskUserParams, ask_user_tool
from cubepi.hitl.channel import HitlChannel, InMemoryChannel
from cubepi.hitl.middleware import ApprovalPolicyMiddleware, ConfirmToolCallMiddleware
from cubepi.hitl.types import (
    ApproveAnswer,
    ApproveRequest,
    AskRequest,
    ConfirmRequest,
    HitlPayload,
    HitlRequest,
    Option,
    Question,
)

__all__ = [
    # types
    "ApproveAnswer",
    "ApproveRequest",
    "AskRequest",
    "ConfirmRequest",
    "HitlPayload",
    "HitlRequest",
    "Option",
    "Question",
    # policy
    "Approve",
    "ApprovalDecision",
    "AskUser",
    "Deny",
    # exceptions
    "HitlAborted",
    "HitlCancelled",
    "HitlConcurrencyError",
    "HitlControlException",
    "HitlDetached",
    "HitlDurabilityNotGuaranteed",
    "HitlError",
    "HitlInconsistentState",
    "HitlMissingAnswer",
    "HitlNoPendingRequest",
    "HitlStaleAnswer",
    "HitlTimedOut",
    # ask_user
    "AskUserParams",
    "ask_user_tool",
    # channel
    "HitlChannel",
    "InMemoryChannel",
    # middleware
    "ApprovalPolicyMiddleware",
    "ConfirmToolCallMiddleware",
]
