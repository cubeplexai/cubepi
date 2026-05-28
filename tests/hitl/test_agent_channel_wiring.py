import pytest
from cubepi.agent.agent import Agent
from cubepi.hitl import HitlError
from cubepi.hitl.channel import InMemoryChannel
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.providers.base import Model


def _agent(channel=None):
    provider = FauxProvider()
    provider.set_responses([faux_assistant_message("")])
    return Agent(
        provider=provider,
        model=Model(id="faux", provider="faux"),
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
