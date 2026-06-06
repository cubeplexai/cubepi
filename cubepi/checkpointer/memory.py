from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from cubepi.checkpointer.base import CheckpointData
from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
)
from cubepi.hitl.types import HitlRequest
from cubepi.providers.base import Message
from cubepi.types import JsonObject


@dataclass
class _RunState:
    claimed_at: float
    completed_at: float | None = None
    completion_seq: int | None = None


class MemoryCheckpointer:
    def __init__(self) -> None:
        self._store: dict[str, CheckpointData] = {}
        self._pending: dict[str, HitlRequest] = {}
        self._pending_run_id: dict[str, str | None] = {}
        self._runs: dict[str, dict[str, _RunState]] = {}
        self._lock = asyncio.Lock()

    async def load(self, thread_id: str) -> CheckpointData | None:
        return self._store.get(thread_id)

    async def append(self, thread_id: str, messages: list[Message]) -> None:
        async with self._lock:
            for m in messages:
                if m.run_id is None:
                    continue
                rs = self._runs.get(thread_id, {}).get(m.run_id)
                if rs is not None and rs.completed_at is not None:
                    raise RunAlreadyCompletedError(
                        f"append on completed run thread={thread_id} run={m.run_id}"
                    )
            if thread_id not in self._store:
                self._store[thread_id] = CheckpointData()
            self._store[thread_id].messages.extend(messages)

    async def save_extra(self, thread_id: str, extra: JsonObject) -> None:
        async with self._lock:
            if thread_id not in self._store:
                self._store[thread_id] = CheckpointData()
            self._store[thread_id].extra.update(extra)

    async def save_pending_request(
        self,
        thread_id: str,
        request: HitlRequest | None,
        *,
        run_id: str | None = None,
    ) -> None:
        async with self._lock:
            if request is None:
                self._pending.pop(thread_id, None)
                self._pending_run_id.pop(thread_id, None)
            else:
                self._pending[thread_id] = request
                self._pending_run_id[thread_id] = run_id

    async def load_pending_request(self, thread_id: str) -> HitlRequest | None:
        return self._pending.get(thread_id)

    async def load_pending_run_id(self, thread_id: str) -> str | None:
        return self._pending_run_id.get(thread_id)

    async def load_pending(
        self, thread_id: str
    ) -> tuple[HitlRequest, str | None] | None:
        req = self._pending.get(thread_id)
        if req is None:
            return None
        return req, self._pending_run_id.get(thread_id)

    async def claim_run(self, thread_id: str, run_id: str) -> None:
        async with self._lock:
            runs = self._runs.setdefault(thread_id, {})
            existing = runs.get(run_id)
            if existing is not None:
                if existing.completed_at is None:
                    raise RunAlreadyClaimedError(
                        f"thread={thread_id} run={run_id} in flight"
                    )
                raise RunAlreadyCompletedError(
                    f"thread={thread_id} run={run_id} already completed"
                )
            runs[run_id] = _RunState(claimed_at=time.time())

    async def mark_run_complete(self, thread_id: str, run_id: str) -> None:
        async with self._lock:
            runs = self._runs.get(thread_id) or {}
            state = runs.get(run_id)
            if state is None:
                raise RunNotClaimedError(
                    f"thread={thread_id} run={run_id} has no claim row"
                )
            if state.completed_at is not None:
                return  # idempotent
            existing_seqs = [
                s.completion_seq for s in runs.values() if s.completion_seq is not None
            ]
            next_seq = max(existing_seqs, default=0) + 1
            state.completed_at = time.time()
            state.completion_seq = next_seq
