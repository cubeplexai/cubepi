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


async def test_resume_short_circuit_emits_hitl_answer_event():
    """Resume short-circuit must emit HitlAnswerEvent so subscribers
    (e.g. IM outbound tailers) learn the question was answered."""
    from cubepi.agent.types import HitlAnswerEvent

    emitted: list[object] = []
    ch = InMemoryChannel()
    ch._bind_emit(lambda e: emitted.append(e))
    ch.attach_resume_answer("tc-7", ApproveAnswer(decision="approve"))
    ans = await ch.approve(tool_name="bash", tool_call_id="tc-7", args={})
    assert ans.decision == "approve"
    answer_events = [e for e in emitted if isinstance(e, HitlAnswerEvent)]
    assert len(answer_events) == 1
    assert answer_events[0].question_id == "tc-7"


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


async def _grab_qid_and_answer(ch: InMemoryChannel, value):
    """Host helper: wait for the next pending request, answer it, and
    return the qid that was set."""
    while ch.pending is None:
        await asyncio.sleep(0)
    qid = ch.pending.question_id
    await ch.answer(qid, value)
    return qid


async def test_repeat_confirm_gets_distinct_qids():
    """Regression: two consecutive `confirm()` calls with identical prompt
    and details must produce distinct question_ids — otherwise a stale
    answer from the first call (delivered late, after timeout/cancel) could
    erroneously match the second call's pending request.

    Codex PR #127 review feedback (P2 follow-up to qid derivation).
    """
    ch = InMemoryChannel()

    h1 = asyncio.create_task(_grab_qid_and_answer(ch, True))
    r1 = await ch.confirm("are you sure?")
    qid1 = await h1

    h2 = asyncio.create_task(_grab_qid_and_answer(ch, False))
    r2 = await ch.confirm("are you sure?")
    qid2 = await h2

    assert r1 is True
    assert r2 is False
    assert qid1 != qid2, (
        f"both confirm() calls produced qid={qid1!r} — a stale answer for "
        f"the first would erroneously match the second's pending request"
    )


async def test_same_process_resume_reuses_persisted_qid_despite_counter():
    """Regression: on a same-process resume, `_next_qid` must reuse the qid
    staged in `_resume_slot` instead of advancing its per-content counter.

    The counter is per-channel-instance. When the SAME channel resumes,
    the pre-detach call already advanced the counter, so a naive replay
    would derive `hash.1` while the persisted slot holds `hash.0` — the
    resume short-circuit misses and the tool re-asks an already-answered
    prompt (or hangs). Codex PR #127 review feedback (P1).
    """
    ch = InMemoryChannel()

    captured: dict[str, str] = {}

    async def host(value):
        while ch.pending is None:
            await asyncio.sleep(0)
        captured["qid"] = ch.pending.question_id
        await ch.answer(ch.pending.question_id, value)

    # First call mints hash.0 and advances the per-content counter to 1.
    h = asyncio.create_task(host(True))
    await ch.confirm("are you sure?")
    await h
    original_qid = captured["qid"]
    assert original_qid.endswith(".0")

    # Simulate Agent.respond() on the SAME channel: stage the persisted
    # answer under the original qid.
    ch.attach_resume_answer(original_qid, "RESUMED")

    # Tool body reruns and re-asks the identical prompt. No host answers
    # this time — it must short-circuit to the staged answer. With the bug
    # the counter mints hash.1, the slot (hash.0) never matches, and the
    # call times out instead of returning "RESUMED".
    result = await ch.confirm("are you sure?", timeout=0.5)
    assert result == "RESUMED"
    assert ch.pending is None


async def test_resume_replays_repeated_identical_ask_sequence():
    """Regression: a tool body that asks the SAME questions twice and detaches
    on the second ask must, on resume, replay the first (already-answered) ask
    AND consume the persisted answer for the second.

    This is codex's repeated-identical-ask case. The first run mints hash.0
    then hash.1 (persisted). On resume the counter is reset, so the replay
    re-mints hash.0 (must be served from prior history, not the slot) then
    hash.1 (must match the persisted slot). We model the persisted answer for
    the SECOND ask via _resume_slot and serve the FIRST ask from a host that
    answers it immediately, mirroring how respond() replays.
    """
    questions = [Question(key="color", prompt="Pick:")]

    # --- original run: two identical asks; capture both qids ---
    run1 = InMemoryChannel()
    qids: list[str] = []

    async def answer_each(value):
        while run1.pending is None:
            await asyncio.sleep(0)
        qids.append(run1.pending.question_id)
        await run1.answer(run1.pending.question_id, value)

    h1 = asyncio.create_task(answer_each({"color": "red"}))
    a1 = await run1.ask(questions)
    await h1
    h2 = asyncio.create_task(answer_each({"color": "blue"}))
    a2 = await run1.ask(questions)
    await h2

    assert a1 == {"color": "red"}
    assert a2 == {"color": "blue"}
    first_qid, second_qid = qids
    assert first_qid != second_qid  # hash.0, hash.1
    assert first_qid.endswith(".0")
    assert second_qid.endswith(".1")

    # --- cross-process resume: fresh channel, counter empty ---
    # respond() stages the persisted answer for the SECOND ask (the one that
    # was pending at detach). The replay's first ask is served by a host (it
    # was already answered/recorded in the original run); the second must
    # short-circuit to the staged answer.
    run2 = InMemoryChannel()
    run2.attach_resume_answer(second_qid, {"color": "STAGED"})

    async def answer_first(value):
        while run2.pending is None:
            await asyncio.sleep(0)
        await run2.answer(run2.pending.question_id, value)

    hf = asyncio.create_task(answer_first({"color": "replayed-first"}))
    r1 = await run2.ask(questions)  # mints hash.0 → host answers
    await hf
    r2 = await run2.ask(questions, timeout=0.5)  # mints hash.1 → matches slot

    assert r1 == {"color": "replayed-first"}
    assert r2 == {"color": "STAGED"}


async def test_repeat_ask_gets_distinct_qids():
    """Same regression as test_repeat_confirm_gets_distinct_qids for ask()."""
    ch = InMemoryChannel()
    questions = [Question(key="color", prompt="Pick:")]

    h1 = asyncio.create_task(_grab_qid_and_answer(ch, {"color": "red"}))
    r1 = await ch.ask(questions)
    qid1 = await h1

    h2 = asyncio.create_task(_grab_qid_and_answer(ch, {"color": "blue"}))
    r2 = await ch.ask(questions)
    qid2 = await h2

    assert r1 == {"color": "red"}
    assert r2 == {"color": "blue"}
    assert qid1 != qid2


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
