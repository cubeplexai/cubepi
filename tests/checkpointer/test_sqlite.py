import os
import tempfile

import pytest

from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.providers.base import TextContent, UserMessage


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


class TestSQLiteCheckpointer:
    async def test_load_empty_thread(self, db_path):
        async with SQLiteCheckpointer(db_path) as cp:
            data = await cp.load("thread-1")
            assert data is None

    async def test_append_and_load(self, db_path):
        async with SQLiteCheckpointer(db_path) as cp:
            msg1 = UserMessage(content=[TextContent(text="hello")])
            msg2 = UserMessage(content=[TextContent(text="world")])
            await cp.append("thread-1", [msg1])
            await cp.append("thread-1", [msg2])

            data = await cp.load("thread-1")
            assert data is not None
            assert len(data.messages) == 2
            assert data.messages[0].content[0].text == "hello"

    async def test_save_extra(self, db_path):
        async with SQLiteCheckpointer(db_path) as cp:
            await cp.append("thread-1", [UserMessage(content=[TextContent(text="hi")])])
            await cp.save_extra("thread-1", {"index": 42})

            data = await cp.load("thread-1")
            assert data.extra["index"] == 42

    async def test_persistence_across_instances(self, db_path):
        async with SQLiteCheckpointer(db_path) as cp:
            await cp.append(
                "thread-1", [UserMessage(content=[TextContent(text="persist")])]
            )

        async with SQLiteCheckpointer(db_path) as cp:
            data = await cp.load("thread-1")
            assert data is not None
            assert data.messages[0].content[0].text == "persist"

    async def test_multiple_threads(self, db_path):
        async with SQLiteCheckpointer(db_path) as cp:
            await cp.append("t1", [UserMessage(content=[TextContent(text="t1")])])
            await cp.append("t2", [UserMessage(content=[TextContent(text="t2")])])

            d1 = await cp.load("t1")
            d2 = await cp.load("t2")
            assert d1.messages[0].content[0].text == "t1"
            assert d2.messages[0].content[0].text == "t2"

    async def test_save_extra_update_merges(self, db_path):
        """Second save_extra call should merge into existing extra, not replace it."""
        async with SQLiteCheckpointer(db_path) as cp:
            await cp.save_extra("thread-1", {"a": 1, "b": 2})
            await cp.save_extra("thread-1", {"b": 99, "c": 3})

            data = await cp.load("thread-1")
            assert data is not None
            assert data.extra == {"a": 1, "b": 99, "c": 3}

    async def test_save_extra_creates_thread(self, db_path):
        """save_extra on a thread that has no messages should still be loadable."""
        async with SQLiteCheckpointer(db_path) as cp:
            await cp.save_extra("new-thread", {"key": "value"})

            data = await cp.load("new-thread")
            assert data is not None
            assert data.messages == []
            assert data.extra == {"key": "value"}

    async def test_deserialize_unknown_role(self, db_path):
        """A message with an unknown role should be returned as a plain dict."""
        async with SQLiteCheckpointer(db_path) as cp:
            raw_msg = {"role": "custom", "data": "test"}
            await cp.append("thread-1", [raw_msg])

            data = await cp.load("thread-1")
            assert data is not None
            assert len(data.messages) == 1
            assert data.messages[0] == {"role": "custom", "data": "test"}
