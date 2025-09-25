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

@pytest.mark.parametrize("system", [None, "sys"])
def test_call_model(system):
    model = DummyModel(output=MagicMock(tool_calls=[]))
    agent = BaseAgent(model, [DummyTool('t1')], system or "")
    state = AgentState(messages=[MagicMock(tool_calls=[])])
    result = agent.call_model(state)
    assert 'messages' in result
    assert isinstance(result['messages'], list)

@pytest.mark.parametrize("tool_calls,tool_names,expected", [
    ([{'name': 't1', 'id': '1', 'args': {}}], ['t1'], ['{}']),
    ([{'name': 'bad', 'id': '2', 'args': {}}], ['t1'], ['bad tool name, retry']),
    ([{'name': 't1', 'id': '3', 'args': {}}], ['t1'], ['{}']),
])
def test_take_action(tool_calls, tool_names, expected):
    tools = [DummyTool(name) for name in tool_names]
    model = DummyModel()
    agent = BaseAgent(model, tools)
    state = AgentState(messages=[MagicMock(tool_calls=tool_calls)])
    result = agent.take_action(state)
    assert 'messages' in result
    assert [m.content for m in result['messages']] == expected

# Exception in tool
def test_take_action_tool_exception():
    tools = [DummyTool('t1', raise_exc=True)]
    model = DummyModel()
    agent = BaseAgent(model, tools)
    state = AgentState(messages=[MagicMock(tool_calls=[{'name': 't1', 'id': '1', 'args': {}}])])
    # Should not raise, but return exception as string
    result = agent.take_action(state)
    assert 'messages' in result
    assert any('Exception' in m.content or 'error' in m.content.lower() or isinstance(m.content, str) for m in result['messages'])

# Non-string output from tool
def test_take_action_non_string_output():
    tools = [DummyTool('t1', return_value=123)]
    model = DummyModel()
    agent = BaseAgent(model, tools)
    state = AgentState(messages=[MagicMock(tool_calls=[{'name': 't1', 'id': '1', 'args': {}}])])
    result = agent.take_action(state)
    assert 'messages' in result
    assert all(isinstance(m.content, str) for m in result['messages'])

