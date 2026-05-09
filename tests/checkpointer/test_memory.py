from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import TextContent, UserMessage


class TestMemoryCheckpointer:
    async def test_load_empty_thread(self):
        cp = MemoryCheckpointer()
        data = await cp.load("thread-1")
        assert data is None

    async def test_append_and_load(self):
        cp = MemoryCheckpointer()
        msg1 = UserMessage(content=[TextContent(text="hello")])
        msg2 = UserMessage(content=[TextContent(text="world")])

        await cp.append("thread-1", [msg1])
        await cp.append("thread-1", [msg2])

        data = await cp.load("thread-1")
        assert data is not None
        assert len(data.messages) == 2
        assert data.messages[0].content[0].text == "hello"
        assert data.messages[1].content[0].text == "world"

    async def test_save_extra(self):
        cp = MemoryCheckpointer()
        await cp.append("thread-1", [UserMessage(content=[TextContent(text="hi")])])
        await cp.save_extra("thread-1", {"compaction_index": 5})

        data = await cp.load("thread-1")
        assert data is not None
        assert data.extra["compaction_index"] == 5

    async def test_save_extra_merges(self):
        cp = MemoryCheckpointer()
        await cp.append("thread-1", [UserMessage(content=[TextContent(text="hi")])])
        await cp.save_extra("thread-1", {"a": 1})
        await cp.save_extra("thread-1", {"b": 2})

        data = await cp.load("thread-1")
        assert data.extra == {"a": 1, "b": 2}

    async def test_multiple_threads(self):
        cp = MemoryCheckpointer()
        await cp.append("t1", [UserMessage(content=[TextContent(text="t1")])])
        await cp.append("t2", [UserMessage(content=[TextContent(text="t2")])])

        d1 = await cp.load("t1")
        d2 = await cp.load("t2")

        assert d1.messages[0].content[0].text == "t1"
        assert d2.messages[0].content[0].text == "t2"
