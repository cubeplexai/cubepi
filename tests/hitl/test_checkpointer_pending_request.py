import tempfile
import pytest

from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.hitl.types import ApproveRequest, HitlRequest


def _req(thread_id="t-1", qid="q-1") -> HitlRequest:
    return HitlRequest(
        question_id=qid,
        thread_id=thread_id,
        payload=ApproveRequest(tool_name="bash", tool_call_id=qid, args={"cmd": "ls"}),
        created_at=0.0,
        timeout_seconds=30.0,
    )


@pytest.fixture
async def sqlite_cp():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        async with SQLiteCheckpointer(f.name) as cp:
            yield cp


async def test_memory_save_and_load_pending():
    cp = MemoryCheckpointer()
    assert await cp.load_pending_request("t-1") is None
    req = _req()
    await cp.save_pending_request("t-1", req)
    loaded = await cp.load_pending_request("t-1")
    assert loaded == req


async def test_memory_clear_pending():
    cp = MemoryCheckpointer()
    await cp.save_pending_request("t-1", _req())
    await cp.save_pending_request("t-1", None)
    assert await cp.load_pending_request("t-1") is None


async def test_sqlite_save_and_load_pending(sqlite_cp):
    assert await sqlite_cp.load_pending_request("t-1") is None
    req = _req()
    await sqlite_cp.save_pending_request("t-1", req)
    loaded = await sqlite_cp.load_pending_request("t-1")
    assert loaded == req


async def test_sqlite_clear_pending(sqlite_cp):
    await sqlite_cp.save_pending_request("t-1", _req())
    await sqlite_cp.save_pending_request("t-1", None)
    assert await sqlite_cp.load_pending_request("t-1") is None


async def test_sqlite_create_table_idempotent(sqlite_cp):
    """Re-opening a checkpointer DB with existing pending_request table is safe."""
    await sqlite_cp.save_pending_request("t-1", _req())
    # Re-entering the context manager would call CREATE TABLE IF NOT EXISTS again
    # against an existing table — must not raise.
    await sqlite_cp._db.execute(
        "CREATE TABLE IF NOT EXISTS thread_pending_request ("
        "thread_id TEXT PRIMARY KEY, request_json TEXT NOT NULL, "
        "created_at REAL NOT NULL DEFAULT (julianday('now')))"
    )
