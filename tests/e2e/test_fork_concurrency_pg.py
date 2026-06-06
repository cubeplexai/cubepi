"""Task 40: concurrent fork + claim races on Postgres E2E.

Two scenarios exercise the advisory-lock serialization on the real Postgres
backend (skipped automatically when ``CUBEPI_TEST_PG_DSN`` is unreachable):

1. **Concurrent forks of the same source thread.** Two parallel ``fork(src=X)``
   calls (different ``new_thread_id``s) both succeed; the per-thread advisory
   lock serializes the snapshot writes.
2. **Concurrent claim of the same run_id.** Two independent
   ``PostgresCheckpointer`` instances pointing at the same DB both try to claim
   ``run_id="R1"``. Exactly one wins; the loser raises
   ``RunAlreadyClaimedError`` (or ``RunAlreadyCompletedError`` if the winner
   finished while the loser was waiting on the lock).
"""

from __future__ import annotations

import asyncio

import pytest

from cubepi.agent.agent import Agent
from cubepi.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
)
from cubepi.checkpointer.postgres.checkpointer import PostgresCheckpointer
from cubepi.providers.base import AssistantMessage, TextContent
from cubepi.providers.faux import FauxProvider


def _ok_faux() -> FauxProvider:
    """Faux provider with plenty of one-shot end_turn responses."""
    p = FauxProvider()
    p.set_responses(
        [
            AssistantMessage(
                content=[TextContent(text=f"r{i}")], stop_reason="end_turn"
            )
            for i in range(10)
        ]
    )
    return p


@pytest.mark.asyncio
async def test_concurrent_forks_of_same_src(pg_v4_dsn):
    """Two parallel forks of the same source thread.

    The per-thread advisory lock serializes the writes; both forks
    succeed with identical message sets and the same parent_thread_id.
    """
    async with PostgresCheckpointer(pg_v4_dsn) as cp:
        a = Agent(
            model=_ok_faux().model("faux-model"),
            checkpointer=cp,
            thread_id="src",
        )
        await a.prompt("hello", run_id="R1")

        async def fork_to(new_id: str) -> None:
            await a.fork("src", new_id, after_run_id="R1")

        await asyncio.gather(fork_to("dst_A"), fork_to("dst_B"))

        for new_id in ("dst_A", "dst_B"):
            loaded = await cp.load(new_id)
            assert loaded is not None
            assert loaded.parent_thread_id == "src"
            assert {m.run_id for m in loaded.messages if m.run_id} == {"R1"}


@pytest.mark.asyncio
async def test_concurrent_claim_of_same_run_id_raises_one(pg_v4_dsn):
    """Two PostgresCheckpointer instances on the same DB both try to
    claim the same run_id.

    Exactly one must succeed; the other raises either
    ``RunAlreadyClaimedError`` (loser arrived during the winner's run) or
    ``RunAlreadyCompletedError`` (loser arrived after the winner finished).
    Both are acceptable outcomes — what matters is that we never get two
    "ok"s on the same run_id.
    """
    async with (
        PostgresCheckpointer(pg_v4_dsn) as cp1,
        PostgresCheckpointer(pg_v4_dsn) as cp2,
    ):
        a1 = Agent(
            model=_ok_faux().model("faux-model"),
            checkpointer=cp1,
            thread_id="t",
        )
        a2 = Agent(
            model=_ok_faux().model("faux-model"),
            checkpointer=cp2,
            thread_id="t",
        )

        async def run_a(agent: Agent, run_id: str) -> str:
            try:
                await agent.prompt("hi", run_id=run_id)
                return "ok"
            except (RunAlreadyClaimedError, RunAlreadyCompletedError) as e:
                return f"claimed:{e}"

        results = await asyncio.gather(
            run_a(a1, "R1"),
            run_a(a2, "R1"),
            return_exceptions=False,
        )

        ok_count = sum(1 for r in results if r == "ok")
        claimed_count = sum(1 for r in results if r.startswith("claimed:"))
        assert ok_count >= 1, f"At least one claim should have succeeded; got {results}"
        assert ok_count + claimed_count == 2, (
            f"Every result must be ok-or-claimed; got {results}"
        )
        # Strictly speaking, only one should succeed — but two "ok"s would
        # only be possible if both happened sequentially under the same
        # lock without re-checking the marker. Guard against that.
        assert ok_count == 1 or claimed_count == 1, (
            f"Exactly one of the two outcomes should differ; got {results}"
        )
