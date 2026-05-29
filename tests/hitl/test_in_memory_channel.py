import asyncio
import pytest

from cubepi.hitl import (
    ApproveAnswer,
    HitlCancelled,
    HitlConcurrencyError,
    HitlRequest,
    HitlStaleAnswer,
    HitlTimedOut,
    Question,
)
from cubepi.hitl.channel import InMemoryChannel


async def test_ask_resolves_via_answer():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, {"color": "red"})

    asyncio.create_task(host())
    answer = await ch.ask([Question(key="color", prompt="Pick:")])
    assert answer == {"color": "red"}
    assert ch.pending is None


async def test_confirm_resolves_to_bool():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    assert (await ch.confirm("proceed?")) is True


async def test_approve_uses_tool_call_id_as_question_id():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        # question_id MUST equal tool_call_id for approve
        assert ch.pending.question_id == "tc-42"
        await ch.answer("tc-42", ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    ans = await ch.approve(tool_name="bash", tool_call_id="tc-42", args={"cmd": "ls"})
    assert ans.decision == "approve"


async def test_pending_request_envelope_carries_timeout():
    ch = InMemoryChannel()
    seen: list[HitlRequest] = []

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        seen.append(ch.pending)
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    await ch.confirm("ok?", timeout=42.0)
    assert seen[0].timeout_seconds == 42.0


async def test_default_timeout_applied_when_per_call_none():
    ch = InMemoryChannel(default_timeout=3.0)
    seen: list[HitlRequest] = []

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        seen.append(ch.pending)
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    await ch.confirm("ok?")  # per-call timeout omitted
    assert seen[0].timeout_seconds == 3.0


async def test_per_call_timeout_overrides_default():
    ch = InMemoryChannel(default_timeout=3.0)
    seen: list[HitlRequest] = []

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        seen.append(ch.pending)
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    await ch.confirm("ok?", timeout=99.0)
    assert seen[0].timeout_seconds == 99.0


async def test_timeout_raises_hitl_timed_out():
    ch = InMemoryChannel()
    with pytest.raises(HitlTimedOut) as exc_info:
        await ch.confirm("ok?", timeout=0.05)
    assert exc_info.value.seconds == 0.05
    assert ch.pending is None


async def test_cancel_raises_hitl_cancelled():
    ch = InMemoryChannel()

    async def canceller():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.cancel(ch.pending.question_id, reason="aborted")

    asyncio.create_task(canceller())
    with pytest.raises(HitlCancelled) as exc_info:
        await ch.confirm("ok?")
    assert exc_info.value.reason == "aborted"
    assert ch.pending is None


async def test_answer_with_stale_qid_raises():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        with pytest.raises(HitlStaleAnswer):
            await ch.answer("not-the-qid", True)
        # Now answer correctly so the test can finish.
        await ch.answer(ch.pending.question_id, True)

    asyncio.create_task(host())
    await ch.confirm("ok?")


async def test_concurrent_request_raises_hitl_concurrency_error():
    ch = InMemoryChannel()

    async def occupy():
        try:
            await ch.confirm("first")
        except HitlCancelled:
            pass

    task = asyncio.create_task(occupy())
    # let occupy() reach the await
    for _ in range(10):
        if ch.pending is not None:
            break
        await asyncio.sleep(0)
    with pytest.raises(HitlConcurrencyError):
        await ch.confirm("second")
    await ch.cancel(ch.pending.question_id, "cleanup")
    await task


async def test_signal_abort_raises_hitl_aborted():
    from cubepi.hitl.exceptions import HitlAborted

    ch = InMemoryChannel()
    signal = asyncio.Event()

    async def trigger():
        while ch.pending is None:
            await asyncio.sleep(0)
        signal.set()

    asyncio.create_task(trigger())
    with pytest.raises(HitlAborted):
        await ch.confirm("ok?", signal=signal)
    assert ch.pending is None


async def test_subscribe_yields_requests():
    ch = InMemoryChannel()
    seen: list[HitlRequest] = []

    async def subscriber():
        async for req in ch.subscribe():
            seen.append(req)
            await ch.answer(req.question_id, True)

    sub = asyncio.create_task(subscriber())
    # Yield once so subscriber() runs subscribe() and registers its queue
    # before we start broadcasting.
    await asyncio.sleep(0)
    await ch.confirm("a")
    await ch.confirm("b")
    sub.cancel()
    try:
        await sub
    except asyncio.CancelledError:
        pass
    assert len(seen) == 2


async def test_attach_resume_answer_short_circuits_next_call():
    """When an answer has been pre-loaded via attach_resume_answer,
    the next matching channel call returns immediately without ever
    setting _pending or awaiting a future."""
    ch = InMemoryChannel()
    ch.attach_resume_answer("tc-7", ApproveAnswer(decision="approve"))
    ans = await ch.approve(tool_name="bash", tool_call_id="tc-7", args={})
    assert ans.decision == "approve"
    assert ch.pending is None


async def test_attach_resume_answer_qid_mismatch_keeps_slot():
    """If the next channel call's question_id doesn't match the
    pre-loaded slot, the call proceeds normally (the pre-load is
    for a different question and should NOT be popped)."""
    ch = InMemoryChannel()
    ch.attach_resume_answer("tc-OLD", True)
    assert ch._resume_slot == ("tc-OLD", True)  # baseline

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(
            ch.pending.question_id, ApproveAnswer(decision="deny", reason="nope")
        )

    asyncio.create_task(host())
    ans = await ch.approve(tool_name="bash", tool_call_id="tc-NEW", args={})
    assert ans.decision == "deny"
    # Slot for the OLD qid is still pre-loaded (the implementation should
    # NOT pop it just because a different qid came through).
    assert ch._resume_slot == ("tc-OLD", True)


async def test_ask_resume_slot_not_consumed_by_different_prompt():
    """Regression: when a tool body asks Q_A then Q_B and detaches with Q_B
    pending, respond() pre-loads Q_B's answer. The tool body re-runs from
    the top — Q_A's ask must NOT consume Q_B's resume slot (otherwise Q_A
    gets Q_B's answer and Q_B is never re-asked).

    Codex PR #127 review feedback (P2 channel.py).
    """
    ch = InMemoryChannel()
    questions_b = [Question(key="color", prompt="What colour for Q_B?")]
    questions_a = [Question(key="shape", prompt="What shape for Q_A?")]

    # Compute what Q_B's persisted qid would be via the public ask()
    # path (run a Q_B request, harvest qid, then drop it from pending).
    async def first_run_to_get_qid_b():
        async def host():
            while ch.pending is None:
                await asyncio.sleep(0)
            qid = ch.pending.question_id
            await ch.answer(qid, {"color": "blue"})
            return qid

        host_task = asyncio.create_task(host())
        await ch.ask(questions_b)
        return await host_task

    qid_b = await first_run_to_get_qid_b()

    # Now simulate Agent.respond(): re-prime the slot with Q_B's answer.
    ch.attach_resume_answer(qid_b, {"color": "blue"})

    # Tool body reruns from top and asks Q_A first — must NOT short-circuit
    # to Q_B's answer. With a tight per-call timeout, ask(Q_A) should time
    # out because no host is answering.
    with pytest.raises(HitlTimedOut):
        await ch.ask(questions_a, timeout=0.05)

    # Q_B's slot must still be there for the eventual ask(Q_B) re-call.
    assert ch._resume_slot == (qid_b, {"color": "blue"})


async def test_confirm_resume_slot_not_consumed_by_different_prompt():
    """Same regression as test_ask_resume_slot_..., for confirm()."""
    ch = InMemoryChannel()

    async def first_run_to_get_qid_b():
        async def host():
            while ch.pending is None:
                await asyncio.sleep(0)
            qid = ch.pending.question_id
            await ch.answer(qid, True)
            return qid

        host_task = asyncio.create_task(host())
        await ch.confirm("Delete database B?")
        return await host_task

    qid_b = await first_run_to_get_qid_b()
    ch.attach_resume_answer(qid_b, True)

    with pytest.raises(HitlTimedOut):
        await ch.confirm("Format disk A?", timeout=0.05)

    assert ch._resume_slot == (qid_b, True)
