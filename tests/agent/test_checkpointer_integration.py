from cubepi.agent.agent import Agent
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.providers.base import Model
from cubepi.providers.faux import FauxProvider, faux_assistant_message


def make_model() -> Model:
    return Model(id="faux-1", provider="faux")


class TestCheckpointerIntegration:
    async def test_messages_persisted_on_message_end(self):
        checkpointer = MemoryCheckpointer()
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Hello!")])
        agent = Agent(
            provider=provider,
            model=make_model(),
            checkpointer=checkpointer,
            thread_id="thread-1",
        )
        await agent.prompt("Hi")
        data = await checkpointer.load("thread-1")
        assert data is not None
        assert len(data.messages) == 2  # user + assistant

    async def test_history_restored_on_prompt(self):
        checkpointer = MemoryCheckpointer()
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("First reply")])
        agent1 = Agent(
            provider=provider,
            model=make_model(),
            checkpointer=checkpointer,
            thread_id="thread-1",
        )
        await agent1.prompt("First message")

        provider.set_responses([faux_assistant_message("Second reply")])
        agent2 = Agent(
            provider=provider,
            model=make_model(),
            checkpointer=checkpointer,
            thread_id="thread-1",
        )
        await agent2.prompt("Second message")
        assert len(agent2.state.messages) == 4  # 2 from first + 2 from second

    async def test_no_checkpointer_works_as_before(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("Hi")])
        agent = Agent(provider=provider, model=make_model())
        await agent.prompt("Hello")
        assert len(agent.state.messages) == 2
