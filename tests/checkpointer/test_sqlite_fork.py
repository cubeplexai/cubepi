import tempfile
from pathlib import Path

import pytest

from cubepi.checkpointer.exceptions import (
    RunNotCompletedError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)
from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.providers.base import TextContent, UserMessage


def _msg(run_id: str | None, text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], run_id=run_id)


@pytest.mark.asyncio
async def test_fork_copies_completed_runs_only():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            await cp.claim_run("src", "A")
            await cp.append("src", [_msg("A", "a1"), _msg("A", "a2")])
            await cp.mark_run_complete("src", "A")
            await cp.claim_run("src", "B")
            await cp.append("src", [_msg("B", "b1")])
            await cp.mark_run_complete("src", "B")
            # An in-flight run C — must be excluded.
            await cp.claim_run("src", "C")
            await cp.append("src", [_msg("C", "c1")])
            await cp.fork("src", "dst", after_run_id="B")
            loaded = await cp.load("dst")
            assert loaded is not None
            texts = [m.content[0].text for m in loaded.messages]
            assert texts == ["a1", "a2", "b1"]
            assert loaded.parent_thread_id == "src"


@pytest.mark.asyncio
async def test_fork_includes_legacy_null_run_id_prefix():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            await cp.append("src", [_msg(None, "legacy")])
            await cp.claim_run("src", "A")
            await cp.append("src", [_msg("A", "a1")])
            await cp.mark_run_complete("src", "A")
            await cp.fork("src", "dst", after_run_id="A")
            loaded = await cp.load("dst")
            assert [m.content[0].text for m in loaded.messages] == ["legacy", "a1"]


@pytest.mark.asyncio
async def test_fork_unknown_src_raises_thread_not_found():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            with pytest.raises(ThreadNotFoundError):
                await cp.fork("missing", "dst", after_run_id="X")


@pytest.mark.asyncio
async def test_fork_unknown_run_id_raises_not_completed():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            await cp.append("src", [_msg(None, "x")])
            with pytest.raises(RunNotCompletedError):
                await cp.fork("src", "dst", after_run_id="missing")


@pytest.mark.asyncio
async def test_fork_destination_collision_raises_already_exists():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            await cp.claim_run("src", "A")
            await cp.append("src", [_msg("A", "a1")])
            await cp.mark_run_complete("src", "A")
            await cp.fork("src", "dst", after_run_id="A")
            with pytest.raises(ThreadAlreadyExistsError):
                await cp.fork("src", "dst", after_run_id="A")


@pytest.mark.asyncio
async def test_fork_carries_extra_and_writes_metadata():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            await cp.save_extra("src", {"original": "x"})
            await cp.claim_run("src", "A")
            await cp.append("src", [_msg("A", "a1")])
            await cp.mark_run_complete("src", "A")
            await cp.fork("src", "dst", after_run_id="A", metadata={"source": "test"})
            loaded = await cp.load("dst")
            assert loaded.extra["original"] == "x"
            assert loaded.extra["fork"] == {"source": "test"}


@pytest.mark.asyncio
async def test_snapshot_matches_fork_messages():
    with tempfile.TemporaryDirectory() as d:
        async with SQLiteCheckpointer(str(Path(d) / "x.db")) as cp:
            await cp.claim_run("src", "A")
            await cp.append("src", [_msg("A", "a1")])
            await cp.mark_run_complete("src", "A")
            msgs = await cp.snapshot("src", after_run_id="A")
            assert [m.content[0].text for m in msgs] == ["a1"]
