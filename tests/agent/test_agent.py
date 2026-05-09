import asyncio

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentEvent, AgentTool, AgentToolResult
from cubepi.providers.base import (
    AssistantMessage,
    Model,
    TextContent,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message


def make_model() -> Model:
    return Model(id="faux-1", provider="faux")


class TestAgentInit:
    def test_default_state(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        assert agent.state.system_prompt == ""
        assert agent.state.thinking == "off"
        assert agent.state.tools == []
        assert agent.state.messages == []
        assert agent.state.is_streaming is False
        assert agent.state.streaming_message is None
        assert agent.state.pending_tool_calls == set()
        assert agent.state.error_message is None

    def test_custom_initial_state(self):
        provider = FauxProvider()
        agent = Agent(
            provider=provider,
            model=make_model(),
            system_prompt="You are a helpful assistant.",
            thinking="low",
        )

        assert agent.state.system_prompt == "You are a helpful assistant."
        assert agent.state.thinking == "low"


class TestAgentSubscribe:
    def test_subscribe_and_unsubscribe(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        count = 0

        def listener(event, signal=None):
            nonlocal count
            count += 1

        unsub = agent.subscribe(listener)
        assert count == 0

        unsub()

    async def test_events_emitted_on_prompt(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("ok")])
        agent = Agent(provider=provider, model=make_model())

        events = []
        agent.subscribe(lambda e, s=None: events.append(e.type))

        await agent.prompt("hello")

        assert "agent_start" in events
        assert "message_start" in events
        assert "message_end" in events
        assert "agent_end" in events

    async def test_full_lifecycle_events_for_thrown_run_failures(self):
        async def bad_stream(*args, **kwargs):
            raise RuntimeError("provider exploded")

        provider = FauxProvider()
        provider.stream = bad_stream
        agent = Agent(provider=provider, model=make_model())

        events = []
        agent.subscribe(lambda e, s=None: events.append(e.type))

        await agent.prompt("hello")

        assert events == [
            "agent_start", "turn_start",
            "message_start", "message_end",
            "message_start", "message_end",
            "turn_end", "agent_end",
        ]
        last_msg = agent.state.messages[-1]
        assert last_msg.role == "assistant"
        assert last_msg.stop_reason == "error"
        assert last_msg.error_message == "provider exploded"
        assert agent.state.error_message == "provider exploded"

    async def test_await_async_subscribers_before_prompt_resolves(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("ok")])
        agent = Agent(provider=provider, model=make_model())

        barrier = asyncio.Event()
        listener_finished = False

        async def listener(event, signal=None):
            nonlocal listener_finished
            if event.type == "agent_end":
                await barrier.wait()
                listener_finished = True

        agent.subscribe(listener)

        prompt_resolved = False

        async def run_prompt():
            nonlocal prompt_resolved
            await agent.prompt("hello")
            prompt_resolved = True

        task = asyncio.create_task(run_prompt())
        await asyncio.sleep(0.05)

        assert not prompt_resolved
        assert not listener_finished
        assert agent.state.is_streaming is True

        barrier.set()
        await task

        assert listener_finished
        assert prompt_resolved
        assert agent.state.is_streaming is False


class TestAgentState:
    def test_tools_are_copied(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        tools = [AgentTool(name="t", description="t", parameters=type("P", (object,), {"model_json_schema": classmethod(lambda cls: {})}), execute=lambda *a, **k: None)]
        agent.state.tools = tools
        assert agent.state.tools is not tools

    def test_messages_are_copied(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        messages = [UserMessage(content=[TextContent(text="hi")])]
        agent.state.messages = messages
        assert agent.state.messages is not messages


class TestAgentQueues:
    def test_steer_queues_message(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        msg = UserMessage(content=[TextContent(text="steering")])
        agent.steer(msg)
        assert msg not in agent.state.messages

    def test_follow_up_queues_message(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())

        msg = UserMessage(content=[TextContent(text="follow-up")])
        agent.follow_up(msg)
        assert msg not in agent.state.messages


class TestAgentAbort:
    def test_abort_without_active_run_does_not_throw(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=make_model())
        agent.abort()


class TestAgentPromptGuards:
    async def test_raises_when_prompt_called_while_streaming(self):
        barrier = asyncio.Event()
        provider = FauxProvider()

        async def slow_stream(*args, **kwargs):
            from cubepi.providers.base import MessageStream, StreamEvent
            ms = MessageStream()

            async def produce():
                await barrier.wait()
                msg = faux_assistant_message("ok")
                ms.push(StreamEvent(type="done"))
                ms.set_result(msg)

            asyncio.create_task(produce())
            return ms

        provider.stream = slow_stream
        agent = Agent(provider=provider, model=make_model())

        task = asyncio.create_task(agent.prompt("first"))
        await asyncio.sleep(0.02)
        assert agent.state.is_streaming is True

        try:
            await agent.prompt("second")
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "already processing" in str(e).lower()

        barrier.set()
        await task

    async def test_raises_when_resume_called_while_streaming(self):
        barrier = asyncio.Event()
        provider = FauxProvider()

        async def slow_stream(*args, **kwargs):
            from cubepi.providers.base import MessageStream, StreamEvent
            ms = MessageStream()

            async def produce():
                await barrier.wait()
                msg = faux_assistant_message("ok")
                ms.push(StreamEvent(type="done"))
                ms.set_result(msg)

            asyncio.create_task(produce())
            return ms

        provider.stream = slow_stream
        agent = Agent(provider=provider, model=make_model())

        task = asyncio.create_task(agent.prompt("first"))
        await asyncio.sleep(0.02)

        try:
            await agent.resume()
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "already processing" in str(e).lower()

        barrier.set()
        await task


class TestAgentResume:
    async def test_resume_processes_follow_up_messages(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message("Initial response"),
            faux_assistant_message("Processed"),
        ])
        agent = Agent(provider=provider, model=make_model())

        await agent.prompt("Initial")
        agent.follow_up(UserMessage(content=[TextContent(text="follow-up")]))
        await agent.resume()

        has_follow_up = any(
            hasattr(m, "content") and hasattr(m, "role") and m.role == "user"
            and any(
                hasattr(c, "text") and c.text == "follow-up"
                for c in (m.content if isinstance(m.content, list) else [])
            )
            for m in agent.state.messages
        )
        assert has_follow_up
        assert agent.state.messages[-1].role == "assistant"
