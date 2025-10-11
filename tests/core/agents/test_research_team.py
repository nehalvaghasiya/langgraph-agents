import pytest
from unittest.mock import MagicMock, patch
from core.agents import research_team

class DummyModel:
    def __init__(self, output=None):
        self.output = output or MagicMock(tool_calls=[])
    def bind_tools(self, tools):
        return self
    def invoke(self, messages):
        return self.output

@patch("core.agents.web_search.get_google_search", autospec=True)
def test_research_team_agent(mock_get_google_search):
    class DummyTool:
        name = "dummy_search"
        def __call__(self, *a, **kw):
            return "dummy search result"
    mock_get_google_search.return_value = DummyTool()
    model = DummyModel()
    agent = research_team.ResearchTeamAgent(model)
    assert hasattr(agent, "graph")
    state = MagicMock()
    assert hasattr(agent, "supervisor_node")
    assert hasattr(agent, "search_node")
    assert hasattr(agent, "web_scraper_node")
