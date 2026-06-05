import pytest
from cubepi.agent.agent import Agent
from cubepi.hitl import HitlError
from cubepi.hitl.channel import InMemoryChannel
from cubepi.providers.faux import FauxProvider, faux_assistant_message


def _agent(channel=None):
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("")])
    return Agent(
        model=provider.model("faux"),
        channel=channel,
    )


def test_agent_accepts_channel_kwarg():
    ch = InMemoryChannel()
    agent = _agent(channel=ch)
    assert agent.channel is ch


def test_agent_channel_property_returns_none_when_unset():
    agent = _agent()
    assert agent.channel is None


def test_in_flight_hitl_request_property_none_initially():
    agent = _agent(channel=InMemoryChannel())
    assert agent.in_flight_hitl_request is None


def test_in_flight_hitl_request_raises_without_channel():
    agent = _agent()
    with pytest.raises(HitlError):
        _ = agent.in_flight_hitl_request


def test_channel_emit_is_bound_to_agent_process_event():
    ch = InMemoryChannel()
    _agent(channel=ch)
    # Verify the emit callback was bound (no public API; verify via attribute)
    assert ch._emit is not None


def test_agent_accepts_protocol_only_channel_without_bind_emit():
    """Regression: Agent must not require the private _bind_emit hook on the
    channel — it is a _BaseChannel internal, not part of the HitlChannel
    protocol. Third-party channels that only implement the public protocol
    should construct cleanly.

    Codex PR #127 review feedback (P2, agent/agent.py).
    """

    class ProtocolOnlyChannel:
        """Implements the HitlChannel protocol surface; no _BaseChannel
        inheritance, no _bind_emit method."""

        pending = None

        async def confirm(self, *args, **kwargs):  # pragma: no cover - unused here
            return True

        async def approve(self, *args, **kwargs):  # pragma: no cover - unused here
            raise NotImplementedError

        async def ask(self, *args, **kwargs):  # pragma: no cover - unused here
            return {}

        def subscribe(self):  # pragma: no cover - unused here
            async def gen():
                if False:
                    yield None

            return gen()

        async def answer(self, *args, **kwargs):  # pragma: no cover - unused here
            return None

        async def cancel(self, *args, **kwargs):  # pragma: no cover - unused here
            return None

        def attach_resume_answer(self, *args, **kwargs):  # pragma: no cover
            return None

    ch = ProtocolOnlyChannel()
    agent = _agent(channel=ch)
    assert agent.channel is ch
