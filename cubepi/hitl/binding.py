"""HitlBinding — structural attribute on AgentTool and Middleware
declaring how a HITL element integrates with the checkpointer.

See dev/specs/2026-06-05-conversation-fork.md §3.6.3.1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HitlBinding:
    """How a tool/middleware integrates with HITL.

    Attributes:
        checkpointed: True iff backed by ``CheckpointedChannel`` (writes
            ``pending_request`` to the source thread on pause).
        run_id: The channel's bound run_id. For checkpointed HITL this
            MUST be a non-empty string; ``None`` is a configuration
            error and is rejected at ``Agent.prompt()`` entry. For
            in-memory HITL it is ``None``.
    """

    checkpointed: bool
    run_id: str | None
