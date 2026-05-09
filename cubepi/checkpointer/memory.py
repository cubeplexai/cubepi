from __future__ import annotations

from typing import Any

from cubepi.checkpointer.base import CheckpointData


class MemoryCheckpointer:
    def __init__(self) -> None:
        self._store: dict[str, CheckpointData] = {}

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
