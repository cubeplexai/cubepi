import tempfile
from pathlib import Path

import aiosqlite
import pytest

from cubepi.checkpointer.exceptions import CheckpointerLockTimeoutError
from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.providers.base import TextContent, UserMessage


@pytest.mark.asyncio
async def test_lock_timeout_surfaces_as_typed_error(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "x.db")
        async with SQLiteCheckpointer(path) as cp:
            await cp._db.execute("PRAGMA busy_timeout = 100")
            other = await aiosqlite.connect(path)
            try:
                await other.execute("BEGIN IMMEDIATE")
                msg = UserMessage(content=[TextContent(text="x")])
                with pytest.raises(CheckpointerLockTimeoutError):
                    await cp.append("t", [msg])
            finally:
                await other.rollback()
                await other.close()
