from cubepi.agent.agent import AgentState


def test_agent_state_default_active_run_id_none():
    s = AgentState()
    assert s.active_run_id is None


def test_agent_state_active_run_id_settable():
    s = AgentState()
    s.active_run_id = "r-1"
    assert s.active_run_id == "r-1"
    s.active_run_id = None
    assert s.active_run_id is None
