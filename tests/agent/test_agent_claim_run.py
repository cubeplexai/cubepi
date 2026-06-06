import asyncio

import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
)
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import AssistantMessage, TextContent
from cubepi.providers.faux import FauxProvider


def _ok_faux() -> FauxProvider:
    p = FauxProvider()
    p.set_responses(
        [AssistantMessage(content=[TextContent(text="ok")], stop_reason="end_turn")]
    )
    return p


def _agent(**kw):
    return Agent(model=_ok_faux().model("faux-model"), **kw)


@pytest.mark.asyncio
async def test_prompt_calls_claim_run_before_append():
    cp = MemoryCheckpointer()
    a = _agent(checkpointer=cp, thread_id="t")
    await a.prompt("hi", run_id="R1")
    # Run row exists.
    assert "R1" in cp._runs["t"]


@pytest.mark.asyncio
async def test_prompt_rejects_already_claimed_run_id():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "R1")
    a = _agent(checkpointer=cp, thread_id="t")
    with pytest.raises(RunAlreadyClaimedError):
        await a.prompt("hi", run_id="R1")


@pytest.mark.asyncio
async def test_prompt_rejects_already_completed_run_id():
    cp = MemoryCheckpointer()
    await cp.claim_run("t", "R1")
    await cp.mark_run_complete("t", "R1")
    a = _agent(checkpointer=cp, thread_id="t")
    with pytest.raises(RunAlreadyCompletedError):
        await a.prompt("hi", run_id="R1")


@pytest.mark.asyncio
async def test_no_checkpointer_no_claim_call():
    a = _agent()  # checkpointer=None
    got = await a.prompt("hi")
    assert isinstance(got, str)  # works fine; no claim attempted


@pytest.mark.asyncio
async def test_degraded_mode_v3_only_checkpointer():
    class _V3Only:
        async def load(self, thread_id):
            return None

        async def append(self, thread_id, msgs):
            pass

        async def save_extra(self, thread_id, extra):
            pass

        async def save_pending_request(self, thread_id, req, *, run_id=None):
            pass

        async def load_pending_request(self, thread_id):
            return None

        # No claim_run / mark_run_complete.

    a = _agent(checkpointer=_V3Only(), thread_id="t")
    # Vanilla prompt works (degraded mode skips claim).
    got = await a.prompt("hi")
    assert isinstance(got, str)


@pytest.mark.asyncio
async def test_resume_claims_run_and_stamps_messages():
    """Regression for codex R3 P1: `resume()` must claim a run, set
    `active_run_id`, and dispatch completion — otherwise messages emitted
    by the resumed turn land with `run_id=None` and fork queries (which
    treat NULL as legacy / always-copy) would copy them into forks even
    if the resumed work is still in flight or later abandoned.
    """
    from cubepi.providers.base import TextContent, UserMessage

    cp = MemoryCheckpointer()
    a = _agent(checkpointer=cp, thread_id="t")
    # Hydrate state from a persisted UserMessage (the documented
    # checkpointed-resume pattern: host reloads from checkpointer, then
    # calls resume()).
    a._state._messages = [
        UserMessage(content=[TextContent(text="hi")], run_id="R_seed"),
    ]

    effective = await a.resume(run_id="R_resume")
    assert effective == "R_resume"
    # The run was claimed AND completed.
    assert "R_resume" in cp._runs["t"]
    assert cp._runs["t"]["R_resume"].completed_at is not None
    # Messages emitted by the resumed turn carry the resume run_id.
    appended = a.state.messages[1:]
    assert appended  # the faux provider yielded an assistant message
    for m in appended:
        assert m.run_id == "R_resume", (
            f"message emitted by resume() left with run_id={m.run_id!r} "
            "(would be treated as legacy by fork queries)"
        )


@pytest.mark.asyncio
async def test_resume_auto_generates_run_id_when_not_supplied():
    """resume() without an explicit run_id auto-generates one."""
    from cubepi.providers.base import TextContent, UserMessage

    cp = MemoryCheckpointer()
    a = _agent(checkpointer=cp, thread_id="t")
    a._state._messages = [
        UserMessage(content=[TextContent(text="hi")], run_id="R_seed"),
    ]
    effective = await a.resume()
    assert isinstance(effective, str) and len(effective) > 0
    assert effective in cp._runs["t"]
    assert cp._runs["t"][effective].completed_at is not None


@pytest.mark.asyncio
async def test_resume_precondition_failure_does_not_claim():
    """Precondition failure ('Cannot continue from message role: assistant')
    must NOT call claim_run — otherwise a dangling claimed run would
    block re-resume with the same run_id.
    """
    from cubepi.providers.base import AssistantMessage, TextContent

    cp = MemoryCheckpointer()
    a = _agent(checkpointer=cp, thread_id="t")
    # Last message is AssistantMessage with no queued steering/follow-up.
    a._state._messages = [
        AssistantMessage(
            content=[TextContent(text="done")],
            stop_reason="end_turn",
            run_id="R_prev",
        ),
    ]
    with pytest.raises(RuntimeError, match="Cannot continue"):
        await a.resume(run_id="R_new")
    # R_new was NOT claimed — the precondition raised before the claim.
    assert "R_new" not in cp._runs.get("t", {})


@pytest.mark.asyncio
async def test_concurrent_cold_prompts_second_raises_fast():
    """Two concurrent cold prompt() calls — the second must raise immediately
    instead of serializing on the run-lock behind the first's claim_run I/O.

    Regression for codex P1: claim_run used to run BEFORE the lock was
    acquired, so two callers could both pass the fail-fast `.locked()` check
    while the first was awaiting claim_run.
    """

    class _SlowClaimCheckpointer(MemoryCheckpointer):
        async def claim_run(self, thread_id, run_id):  # type: ignore[override]
            await asyncio.sleep(0.05)
            return await super().claim_run(thread_id, run_id)

    cp = _SlowClaimCheckpointer()
    a = _agent(checkpointer=cp, thread_id="t")

    results = await asyncio.gather(
        a.prompt("first", run_id="R1"),
        a.prompt("second", run_id="R2"),
        return_exceptions=True,
    )
    # Exactly one succeeded; the other raised "already processing".
    runtime_errs = [r for r in results if isinstance(r, RuntimeError)]
    successes = [r for r in results if isinstance(r, str)]
    assert len(runtime_errs) == 1
    assert len(successes) == 1
    assert "already processing" in str(runtime_errs[0]).lower()
    # The loser's run_id was NOT claimed (claim_run only runs inside the lock).
    claimed = set(cp._runs.get("t", {}).keys())
    assert claimed == {successes[0]}
