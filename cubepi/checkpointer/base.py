from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class CheckpointData:
    messages: list[Any] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Any]) -> None: ...
    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None: ...

    async def save_pending_request(self, thread_id: str, request: Any) -> None:
        """Persist (or clear, if request is None) the pending HITL request for a thread.

        First-party implementations (Memory, SQLite, Postgres, MySQL) all implement this.
        HITL-requiring features (Agent.respond, CheckpointedChannel) use
        ``getattr(checkpointer, "save_pending_request", None)`` for graceful degradation.
        """
        ...

    async def load_pending_request(self, thread_id: str) -> Any:
        """Load the persisted pending HITL request for a thread, or None.

        Returns a ``HitlRequest`` instance or ``None``.
        """
        ...
