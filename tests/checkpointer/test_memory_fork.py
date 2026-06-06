import pytest

from cubepi.checkpointer.exceptions import (
    RunNotCompletedError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import TextContent, UserMessage


def _msg(run_id: str | None, text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], run_id=run_id)


@pytest.mark.asyncio
async def test_fork_copies_completed_runs_only():
    cp = MemoryCheckpointer()
    # Two completed runs A and B.
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
    cp = MemoryCheckpointer()
    # Legacy NULL-run_id message.
    await cp.append("src", [_msg(None, "legacy")])
    # One completed run.
    await cp.claim_run("src", "A")
    await cp.append("src", [_msg("A", "a1")])
    await cp.mark_run_complete("src", "A")
    await cp.fork("src", "dst", after_run_id="A")
    loaded = await cp.load("dst")
    assert [m.content[0].text for m in loaded.messages] == ["legacy", "a1"]


@pytest.mark.asyncio
async def test_fork_unknown_src_raises_thread_not_found():
    cp = MemoryCheckpointer()
    with pytest.raises(ThreadNotFoundError):
        await cp.fork("missing", "dst", after_run_id="X")


@pytest.mark.asyncio
async def test_fork_unknown_run_id_raises_not_completed():
    cp = MemoryCheckpointer()
    await cp.append("src", [_msg(None, "x")])
    with pytest.raises(RunNotCompletedError):
        await cp.fork("src", "dst", after_run_id="missing")


@pytest.mark.asyncio
async def test_fork_destination_collision_raises_already_exists():
    cp = MemoryCheckpointer()
    await cp.claim_run("src", "A")
    await cp.append("src", [_msg("A", "a1")])
    await cp.mark_run_complete("src", "A")
    await cp.fork("src", "dst", after_run_id="A")
    with pytest.raises(ThreadAlreadyExistsError):
        await cp.fork("src", "dst", after_run_id="A")


@pytest.mark.asyncio
async def test_fork_carries_extra_and_writes_metadata():
    cp = MemoryCheckpointer()
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
    cp = MemoryCheckpointer()
    await cp.claim_run("src", "A")
    await cp.append("src", [_msg("A", "a1")])
    await cp.mark_run_complete("src", "A")
    msgs = await cp.snapshot("src", after_run_id="A")
    assert [m.content[0].text for m in msgs] == ["a1"]


@pytest.mark.asyncio
async def test_snapshot_unknown_thread_raises_thread_not_found():
    cp = MemoryCheckpointer()
    with pytest.raises(ThreadNotFoundError):
        await cp.snapshot("missing", after_run_id="A")


@pytest.mark.asyncio
async def test_snapshot_uncompleted_run_raises_not_completed():
    cp = MemoryCheckpointer()
    await cp.claim_run("src", "A")
    await cp.append("src", [_msg("A", "a1")])
    # NOT calling mark_run_complete — run is claimed but uncompleted.
    with pytest.raises(RunNotCompletedError):
        await cp.snapshot("src", after_run_id="A")


@pytest.mark.asyncio
async def test_snapshot_includes_legacy_null_run_id_and_skips_uncompleted():
    """Snapshot must:
    - copy messages with run_id=None as legacy (pre-v4) history
    - skip messages whose run_id belongs to an uncompleted later run
    """
    cp = MemoryCheckpointer()
    # Pre-v4 legacy message (no run_id).
    await cp.append("src", [_msg(None, "legacy")])
    # Completed run A.
    await cp.claim_run("src", "A")
    await cp.append("src", [_msg("A", "a1")])
    await cp.mark_run_complete("src", "A")
    # Later uncompleted run B (claimed but not completed).
    await cp.claim_run("src", "B")
    await cp.append("src", [_msg("B", "b1-in-flight")])
    # Snapshot after A should include legacy + A but NOT B.
    msgs = await cp.snapshot("src", after_run_id="A")
    texts = [m.content[0].text for m in msgs]
    assert "legacy" in texts
    assert "a1" in texts
    assert "b1-in-flight" not in texts
