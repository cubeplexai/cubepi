from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cubepi.hitl.types import HitlRequest
from cubepi.providers.base import Message
from cubepi.types import JsonObject


@dataclass
class CheckpointData:
    messages: list[Message] = field(default_factory=list)
    extra: JsonObject = field(default_factory=dict)
    parent_thread_id: str | None = None


@runtime_checkable
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Message]) -> None: ...
    async def save_extra(self, thread_id: str, extra: JsonObject) -> None: ...

    async def save_pending_request(
        self,
        thread_id: str,
        request: HitlRequest | None,
        *,
        run_id: str | None = None,
    ) -> None:
        """Persist (or clear, if request is None) the pending HITL request for a thread.

        First-party implementations (Memory, SQLite, Postgres, MySQL) all implement this.
        HITL-requiring features (Agent.respond, CheckpointedChannel) use
        ``getattr(checkpointer, "save_pending_request", None)`` for graceful degradation.
        """
        ...

    async def load_pending_request(self, thread_id: str) -> HitlRequest | None:
        """Load the persisted pending HITL request for a thread, or None.

        Returns a ``HitlRequest`` instance or ``None``.
        """
        ...
