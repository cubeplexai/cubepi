import os
import tempfile

import pytest

from cubepi.checkpointer.sqlite import SQLiteCheckpointer
from cubepi.providers.base import TextContent, ToolResultMessage, UserMessage


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

    async def test_round_trip_tool_result_message(self, db_path):
        """ToolResultMessage round-trips through the tool_result deserializer branch."""
        async with SQLiteCheckpointer(db_path) as cp:
            tool_msg = ToolResultMessage(
                tool_call_id="tc-1",
                tool_name="search",
                content=[TextContent(text="result")],
            )
            await cp.append("thread-1", [tool_msg])

            data = await cp.load("thread-1")
            assert data is not None
            assert len(data.messages) == 1
            loaded = data.messages[0]
            assert isinstance(loaded, ToolResultMessage)
            assert loaded.tool_call_id == "tc-1"
            assert loaded.content[0].text == "result"

    async def test_deserialize_unknown_role(self, db_path):
        """Unknown roles are corruption, matching the postgres/mysql
        backends — the old silent raw-dict passthrough let bad data flow
        into the message list and fail far from the cause."""
        from cubepi.checkpointer.exceptions import CheckpointCorruptionError

        async with SQLiteCheckpointer(db_path) as cp:
            raw_msg = {"role": "custom", "data": "test"}
            await cp.append("thread-1", [raw_msg])

            with pytest.raises(CheckpointCorruptionError):
                await cp.load("thread-1")


class TestCheckpointCorruption:
    async def test_corrupt_json_row_raises_typed(self, db_path):
        from cubepi.checkpointer.exceptions import CheckpointCorruptionError

        async with SQLiteCheckpointer(db_path) as cp:
            await cp.append(
                "thread-1",
                [
                    UserMessage(content=[TextContent(text="ok")]),
                    UserMessage(content=[TextContent(text="will corrupt")]),
                ],
            )
            await cp._db.execute(
                'UPDATE messages SET message_json = \'{"role": "user"\' '
                "WHERE id = (SELECT max(id) FROM messages WHERE thread_id = ?)",
                ("thread-1",),
            )
            await cp._db.commit()

            with pytest.raises(CheckpointCorruptionError) as excinfo:
                await cp.load("thread-1")

        err = excinfo.value
        assert err.thread_id == "thread-1"
        assert err.backend == "sqlite"
        assert err.row_ref.startswith("messages.id=")
        assert err.__cause__ is not None

    async def test_unknown_role_raises_typed(self, db_path):
        from cubepi.checkpointer.exceptions import CheckpointCorruptionError

        async with SQLiteCheckpointer(db_path) as cp:
            await cp.append("thread-1", [UserMessage(content=[TextContent(text="ok")])])
            await cp._db.execute(
                "UPDATE messages SET message_json = "
                '\'{"role": "alien", "content": []}\' WHERE thread_id = ?',
                ("thread-1",),
            )
            await cp._db.commit()

            with pytest.raises(CheckpointCorruptionError) as excinfo:
                await cp.load("thread-1")

        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "alien" in str(excinfo.value)
