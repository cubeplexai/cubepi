"""Human-in-the-Loop (HITL) primitives for cubepi agents.

See dev/specs/2026-05-28-hitl-channel.md for the full design.
"""

from cubeloop.hitl.exceptions import (
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
from cubeloop.hitl.policy import (
    Approve,
    ApprovalDecision,
    AskUser,
    Deny,
)
from cubeloop.hitl.ask_user import AskUserParams, ask_user_tool
from cubeloop.hitl.channel import CheckpointedChannel, HitlChannel, InMemoryChannel
from cubeloop.hitl.testing import NoopChannel, ScriptedChannel
from cubeloop.hitl.middleware import ApprovalPolicyMiddleware, ConfirmToolCallMiddleware
from cubeloop.hitl.types import (
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
    "CheckpointedChannel",
    "HitlChannel",
    "InMemoryChannel",
    # middleware
    "ApprovalPolicyMiddleware",
    "ConfirmToolCallMiddleware",
    # testing
    "NoopChannel",
    "ScriptedChannel",
]
