import pytest
from unittest.mock import MagicMock
from core.agents import web_search
from core.agents.base import AgentState

class DummyModel:
    def __init__(self, output=None):
        self.output = output or MagicMock(tool_calls=[])
    def bind_tools(self, tools):
        return self
    def invoke(self, messages):
        return self.output

from unittest.mock import patch

@patch("core.agents.web_search.get_google_search", autospec=True)
def test_search_agent(mock_get_google_search):
    class DummyTool:
        name = "dummy_search"
        def __call__(self, *a, **kw):
            return "dummy search result"
    mock_get_google_search.return_value = DummyTool()
    model = DummyModel()
    agent = web_search.SearchAgent(model)
    state = AgentState(messages=[MagicMock(tool_calls=[])])
    assert hasattr(agent, "graph")
    assert isinstance(agent.call_model(state), dict)
