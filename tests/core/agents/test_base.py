import pytest
from unittest.mock import MagicMock
from core.agents.base import BaseAgent, AgentState

class DummyTool:
    def __init__(self, name, return_value=None, raise_exc=False):
        self.name = name
        self.return_value = return_value
        self.raise_exc = raise_exc
    def invoke(self, args):
        if self.raise_exc:
            raise Exception("Tool error")
        return self.return_value if self.return_value is not None else args

class DummyModel:
    def __init__(self, output=None):
        self.output = output or MagicMock(tool_calls=[])
        self.tools = None
    def bind_tools(self, tools):
        self.tools = tools
        return self
    def invoke(self, messages):
        return self.output

@pytest.mark.parametrize("messages", [[], [{}], [{'tool_calls': []}]])
def test_agentstate_structure(messages):
    state = AgentState(messages=messages)
    assert isinstance(state['messages'], list)

@pytest.mark.parametrize("tools,system", [([], None), ([DummyTool('t1')], "sys"), ([DummyTool('t1'), DummyTool('t2')], None)])
def test_baseagent_initialization(tools, system):
    model = DummyModel()
    agent = BaseAgent(model, tools, system or "")
    assert agent.system == (system or "")
    for t in tools:
        assert t.name in agent.tools

@pytest.mark.parametrize("tool_calls,result", [([], False), ([{'name': 't1', 'id': '1', 'args': {}}], True)])
def test_exists_action(tool_calls, result):
    model = DummyModel()
    agent = BaseAgent(model, [DummyTool('t1')])
    state = AgentState(messages=[MagicMock(tool_calls=tool_calls)])
    assert agent.exists_action(state) == result
