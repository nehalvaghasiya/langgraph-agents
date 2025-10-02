import pytest
from unittest.mock import MagicMock
from core.agents import research_team

class DummyModel:
    def __init__(self, output=None):
        self.output = output or MagicMock(tool_calls=[])
    def bind_tools(self, tools):
        return self
    def invoke(self, messages):
        return self.output

def test_research_team_agent():
    model = DummyModel()
    agent = research_team.ResearchTeamAgent(model)
    assert hasattr(agent, "graph")
    state = MagicMock()
    assert hasattr(agent, "supervisor_node")
    assert hasattr(agent, "search_node")
    assert hasattr(agent, "web_scraper_node")
