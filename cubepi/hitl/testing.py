from __future__ import annotations

import time
from typing import Any, Callable, Union

from cubepi.hitl.channel import _BaseChannel
from cubepi.hitl.exceptions import HitlError
from cubepi.hitl.types import HitlRequest


class ScriptedChannel(_BaseChannel):
    """Pre-programmed answers for deterministic tests.

    answers: list of values or callables. Each call to ask/confirm/approve
    consumes the next item. A callable receives the HitlRequest and returns
    the answer.
    """

    def __init__(self, answers: list[Union[Any, Callable[[HitlRequest], Any]]]):
        super().__init__()
        self._answers = list(answers)
        self._history: list[HitlRequest] = []

    @property
    def history(self) -> list[HitlRequest]:
        return list(self._history)

    async def _await_answer(self, payload, timeout, signal, question_id):
        if self._resume_slot is not None and self._resume_slot[0] == question_id:
            _, ans = self._resume_slot
            self._resume_slot = None
            return ans
        if not self._answers:
            raise HitlError(f"ScriptedChannel exhausted (received {payload!r})")
        req = HitlRequest(
            question_id=question_id,
            thread_id=None,
            payload=payload,
            created_at=time.time(),
            timeout_seconds=timeout,
        )
        self._history.append(req)
        head = self._answers.pop(0)
        return head(req) if callable(head) else head


class NoopChannel(_BaseChannel):
    """Auto-approves everything. Useful for subagents in tests."""

    async def _await_answer(self, payload, timeout, signal, question_id):
        from cubepi.hitl.types import ApproveAnswer

        kind = payload.kind
        if kind == "approve":
            return ApproveAnswer(decision="approve")
        if kind == "confirm":
            return True
        if kind == "ask":
            return {q.key: "" for q in payload.questions}
        raise HitlError(
            f"NoopChannel does not handle {kind!r}"
        )  # pragma: no cover — defensive
