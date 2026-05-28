from __future__ import annotations

import asyncio
import contextvars
import time
import uuid
from typing import Any, AsyncIterator, Protocol

from cubepi.hitl.exceptions import (
    HitlAborted,
    HitlCancelled,
    HitlConcurrencyError,
    HitlStaleAnswer,
    HitlTimedOut,
)
from cubepi.hitl.types import (
    ApproveAnswer,
    ApproveRequest,
    AskRequest,
    ConfirmRequest,
    HitlRequest,
    Question,
)

# ContextVar used by CheckpointedChannel to enforce the "do not ask HITL
# from inside a custom tool body" durability guard. Set by
# cubepi.agent.tools._execute_prepared for non-builtin tools; the guard fires
# in CheckpointedChannel._on_pending_set when the var is True and
# allow_inside_custom_tool=False.
_in_custom_tool_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_in_custom_tool_var",
    default=False,
)


class HitlChannel(Protocol):
    # ---- agent side ----
    async def confirm(
        self,
        prompt: str,
        *,
        details: dict | None = None,
        tool_call_id: str | None = None,
        timeout: float | None = None,
        signal: asyncio.Event | None = None,
    ) -> bool: ...

    async def approve(
        self,
        tool_name: str,
        tool_call_id: str,
        args: dict,
        *,
        details: dict | None = None,
        timeout: float | None = None,
        signal: asyncio.Event | None = None,
    ) -> ApproveAnswer: ...

    async def ask(
        self,
        questions: list[Question],
        *,
        timeout: float | None = None,
        signal: asyncio.Event | None = None,
    ) -> dict[str, str | list[str]]: ...

    @property
    def pending(self) -> HitlRequest | None: ...

    def subscribe(self) -> AsyncIterator[HitlRequest]: ...

    async def answer(self, question_id: str, answer: Any) -> None: ...

    async def cancel(self, question_id: str, reason: str = "cancelled") -> None: ...

    def attach_resume_answer(self, question_id: str, answer: Any) -> None: ...


class _BaseChannel:
    """Shared state machine for InMemoryChannel and CheckpointedChannel.

    Maintains the single-pending invariant, the awaiting future, the
    resume-answer slot, subscriber queues, and the optional emit
    callback wired by the Agent at construction.
    """

    def __init__(
        self,
        *,
        default_timeout: float | None = None,
        thread_id: str | None = None,
    ) -> None:
        self._default_timeout = default_timeout
        self._thread_id = thread_id
        self._pending: HitlRequest | None = None
        self._future: asyncio.Future[Any] | None = None
        self._resume_slot: tuple[str, Any] | None = None
        self._subscribers: list[asyncio.Queue[HitlRequest]] = []
        self._emit = None  # set by Agent._bind_channel

    @property
    def pending(self) -> HitlRequest | None:
        return self._pending

    def attach_resume_answer(self, question_id: str, answer: Any) -> None:
        self._resume_slot = (question_id, answer)

    def _bind_emit(self, emit) -> None:
        self._emit = emit

    def subscribe(self) -> AsyncIterator[HitlRequest]:
        queue: asyncio.Queue[HitlRequest] = asyncio.Queue()
        self._subscribers.append(queue)

        async def gen():
            try:
                while True:
                    yield await queue.get()
            finally:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)

        return gen()

    async def _await_answer(
        self,
        payload: Any,
        timeout: float | None,
        signal: asyncio.Event | None,
        question_id: str,
    ) -> Any:
        from cubepi.hitl._trace import hitl_span

        kind = payload.kind
        attrs: dict[str, Any] = {"question_id": question_id, "timeout_seconds": timeout}
        if kind == "approve":
            attrs["tool_call_id"] = payload.tool_call_id
            attrs["tool_name"] = payload.tool_name

        outcome = "unknown"
        from_resume = False
        t0 = time.monotonic()
        with hitl_span(kind, **attrs) as span:
            try:
                # Resume short-circuit — return immediately, do NOT set _pending.
                if (
                    self._resume_slot is not None
                    and self._resume_slot[0] == question_id
                ):
                    _, ans = self._resume_slot
                    self._resume_slot = None
                    from_resume = True
                    outcome = _outcome_from_answer(kind, ans)
                    return ans

                if self._pending is not None:
                    raise HitlConcurrencyError(
                        f"channel busy: already pending {self._pending.question_id}"
                    )

                effective_timeout = (
                    timeout if timeout is not None else self._default_timeout
                )
                req = HitlRequest(
                    question_id=question_id,
                    thread_id=self._thread_id,
                    payload=payload,
                    created_at=time.time(),
                    timeout_seconds=effective_timeout,
                )
                self._pending = req
                self._future = asyncio.get_running_loop().create_future()

                exc_caught: BaseException | None = None
                signal_task: asyncio.Future[Any] | None = None
                # CRITICAL: _on_pending_set is called INSIDE the try/finally so that
                # if it raises (e.g. CheckpointedChannel's HitlDurabilityNotGuaranteed
                # guard), the finally still clears _pending/_future. Otherwise the
                # channel would be permanently wedged.
                try:
                    await self._on_pending_set(req)
                    if signal is None and effective_timeout is None:
                        result = await self._future
                        outcome = _outcome_from_answer(kind, result)
                        return result
                    tasks: list[asyncio.Future[Any]] = [self._future]
                    if signal is not None:
                        signal_task = asyncio.ensure_future(signal.wait())
                        tasks.append(signal_task)
                    done, pending_tasks = await asyncio.wait(
                        tasks,
                        timeout=effective_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Clean up pending tasks (the loser of the race + any signal task
                    # we never want to leave hanging).
                    for p in pending_tasks:
                        if p is not self._future:
                            p.cancel()
                    if not done:
                        raise HitlTimedOut(effective_timeout)
                    # Use task identity, not signal.is_set() — Agent.abort() leaves the
                    # signal sticky-set, so signal.is_set() would be true even on the
                    # happy path after a prior abort. The race winner is what matters.
                    if (
                        signal_task is not None
                        and signal_task in done
                        and self._future not in done
                    ):
                        raise HitlAborted("agent signal fired during HITL pending")
                    result = self._future.result()
                    outcome = _outcome_from_answer(kind, result)
                    return result
                except BaseException as exc:
                    exc_caught = exc
                    raise
                finally:
                    # Cancel any still-pending signal task to avoid leaks.
                    if signal_task is not None and not signal_task.done():
                        signal_task.cancel()
                    # Keep the local _pending/_future slot occupied WHILE
                    # _on_pending_cleared runs — CheckpointedChannel's hook
                    # may do an `await save_pending_request(thread_id, None)`,
                    # and during that await another caller could otherwise
                    # see _pending == None, persist a new pending, then have
                    # it wiped by this prior request's late clear. Holding
                    # the local slot forces concurrent calls to bounce on
                    # HitlConcurrencyError instead.
                    try:
                        await self._on_pending_cleared(req, exc=exc_caught)
                    finally:
                        self._pending = None
                        self._future = None
            except BaseException as exc:
                outcome = _outcome_from_exception(exc)
                raise
            finally:
                span.set_attribute("hitl.from_resume", from_resume)
                span.set_attribute("hitl.outcome", outcome)
                span.set_attribute("hitl.duration_seconds", time.monotonic() - t0)

    async def _on_pending_set(self, req: HitlRequest) -> None:
        # Broadcast to subscribers.
        for q in list(self._subscribers):
            q.put_nowait(req)
        # Emit event — guard avoids importing not-yet-defined event types.
        if self._emit is not None:  # pragma: no cover — integration tested
            from cubepi.agent.types import HitlRequestEvent  # avoid circular

            await self._emit_event(HitlRequestEvent(request=req))

    async def _on_pending_cleared(
        self,
        req: HitlRequest,
        *,
        exc: BaseException | None = None,
    ) -> None:
        # No-op in InMemory; CheckpointedChannel overrides to clear DB row
        # ONLY when the unwind cause is not HitlDetached (which signals a
        # cross-process suspend that must keep persisted pending).
        pass

    async def _emit_event(self, event: Any) -> None:
        if self._emit is None:  # pragma: no cover — caller already guards this
            return
        res = self._emit(event)
        if asyncio.iscoroutine(res):
            await res

    async def answer(self, question_id: str, answer: Any) -> None:
        if self._pending is None or self._pending.question_id != question_id:
            raise HitlStaleAnswer(
                f"answer for {question_id}; pending is "
                f"{self._pending.question_id if self._pending else 'None'}"
            )
        if self._future is not None and not self._future.done():
            self._future.set_result(answer)
        if self._emit is not None:  # pragma: no cover — integration tested
            from cubepi.agent.types import HitlAnswerEvent  # avoid circular

            await self._emit_event(
                HitlAnswerEvent(question_id=question_id, answer=answer)
            )

    async def cancel(self, question_id: str, reason: str = "cancelled") -> None:
        if self._pending is None or self._pending.question_id != question_id:
            raise HitlStaleAnswer(
                f"cancel for {question_id}; pending is "
                f"{self._pending.question_id if self._pending else 'None'}"
            )
        if self._future is not None and not self._future.done():
            self._future.set_exception(HitlCancelled(reason))
        if self._emit is not None:  # pragma: no cover — integration tested
            from cubepi.agent.types import HitlAnswerEvent  # avoid circular

            await self._emit_event(
                HitlAnswerEvent(question_id=question_id, answer=None, cancelled=True)
            )

    # ---- agent-side verbs ----

    async def confirm(
        self,
        prompt: str,
        *,
        details: dict | None = None,
        tool_call_id: str | None = None,
        timeout: float | None = None,
        signal: asyncio.Event | None = None,
    ) -> bool:
        qid = uuid.uuid4().hex
        return await self._await_answer(
            ConfirmRequest(prompt=prompt, details=details),
            timeout=timeout,
            signal=signal,
            question_id=qid,
        )

    async def approve(
        self,
        tool_name: str,
        tool_call_id: str,
        args: dict,
        *,
        details: dict | None = None,
        timeout: float | None = None,
        signal: asyncio.Event | None = None,
    ) -> ApproveAnswer:
        return await self._await_answer(
            ApproveRequest(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                args=args,
                details=details,
            ),
            timeout=timeout,
            signal=signal,
            question_id=tool_call_id,
        )

    async def ask(
        self,
        questions: list[Question],
        *,
        timeout: float | None = None,
        signal: asyncio.Event | None = None,
    ) -> dict[str, str | list[str]]:
        qid = uuid.uuid4().hex
        return await self._await_answer(
            AskRequest(questions=questions),
            timeout=timeout,
            signal=signal,
            question_id=qid,
        )


def _outcome_from_answer(kind: str, ans: Any) -> str:
    if kind == "approve":
        return {"approve": "approved", "deny": "denied", "edit": "edited"}.get(
            getattr(ans, "decision", None), "answered"
        )
    return "answered"


def _outcome_from_exception(exc: BaseException) -> str:
    from cubepi.hitl.exceptions import (
        HitlAborted,
        HitlCancelled,
        HitlDetached,
        HitlTimedOut,
    )

    if isinstance(exc, HitlCancelled):
        return "cancelled"
    if isinstance(exc, HitlTimedOut):
        return "timed_out"
    if isinstance(exc, HitlAborted):
        return "aborted"
    if isinstance(exc, HitlDetached):
        return "detached"
    return "error"


class InMemoryChannel(_BaseChannel):
    """In-process HITL channel; no persistence."""


class CheckpointedChannel(_BaseChannel):
    """Cross-process HITL channel — persists the pending request via a
    checkpointer so a separate process can call ``Agent.respond()`` after
    detach.

    The checkpointer MUST implement ``save_pending_request`` and
    ``load_pending_request`` (first-party checkpointers do; third-party
    Protocol-only impls may not).
    """

    def __init__(
        self,
        *,
        checkpointer: Any,
        thread_id: str,
        default_timeout: float | None = None,
        allow_inside_custom_tool: bool = False,
    ) -> None:
        # Validate the checkpointer has the HITL methods early — better to
        # fail at construction than at first ask/approve/confirm (codex pass 3).
        if not (
            hasattr(checkpointer, "save_pending_request")
            and hasattr(checkpointer, "load_pending_request")
        ):
            from cubepi.hitl.exceptions import HitlError

            raise HitlError(
                "CheckpointedChannel requires a checkpointer with "
                "save_pending_request and load_pending_request methods. "
                "First-party checkpointers (Memory/SQLite/Postgres/MySQL) "
                "implement these; third-party Protocol-only impls may not."
            )
        super().__init__(default_timeout=default_timeout, thread_id=thread_id)
        self._checkpointer = checkpointer
        self._allow_inside_custom_tool = allow_inside_custom_tool

    async def _on_pending_set(self, req: HitlRequest) -> None:
        if _in_custom_tool_var.get() and not self._allow_inside_custom_tool:
            from cubepi.hitl.exceptions import HitlDurabilityNotGuaranteed

            raise HitlDurabilityNotGuaranteed(
                "CheckpointedChannel called from inside a custom tool body. "
                "Use ApprovalPolicyMiddleware or ask_user_tool, or pass "
                "allow_inside_custom_tool=True to opt in."
            )
        await self._checkpointer.save_pending_request(self._thread_id, req)
        await super()._on_pending_set(req)

    async def _on_pending_cleared(
        self,
        req: HitlRequest,
        *,
        exc: BaseException | None = None,
    ) -> None:
        from cubepi.hitl.exceptions import HitlDetached

        if isinstance(exc, HitlDetached):
            return
        await self._checkpointer.save_pending_request(self._thread_id, None)
