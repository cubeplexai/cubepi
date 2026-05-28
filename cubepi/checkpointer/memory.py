from __future__ import annotations

from typing import Any

from cubepi.checkpointer.base import CheckpointData
from cubepi.hitl.types import HitlRequest


class MemoryCheckpointer:
    def __init__(self) -> None:
        self._store: dict[str, CheckpointData] = {}
        self._pending: dict[str, HitlRequest] = {}

    async def load(self, thread_id: str) -> CheckpointData | None:
        return self._store.get(thread_id)

    async def append(self, thread_id: str, messages: list[Any]) -> None:
        if thread_id not in self._store:
            self._store[thread_id] = CheckpointData()
        self._store[thread_id].messages.extend(messages)

    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None:
        if thread_id not in self._store:
            self._store[thread_id] = CheckpointData()
        self._store[thread_id].extra.update(extra)

    async def save_pending_request(self, thread_id: str, request: Any) -> None:
        if request is None:
            self._pending.pop(thread_id, None)
        else:
            self._pending[thread_id] = request

    async def load_pending_request(self, thread_id: str) -> Any:
        return self._pending.get(thread_id)
