import pytest
from unittest.mock import MagicMock
from core.agents import paper_writing

class DummyModel:
    def __init__(self, output=None):
        self.output = output or MagicMock(tool_calls=[])
    def bind_tools(self, tools):
        return self
    def invoke(self, messages):
        return self.output

def test_paper_writing_team_agent():
    model = DummyModel()
    agent = paper_writing.PaperWritingTeamAgent(model)
    assert hasattr(agent, "graph")
    state = MagicMock()
    agent.members = ["doc_writer", "note_taker", "chart_generator"]
    assert hasattr(agent, "supervisor_node")
    assert hasattr(agent, "doc_writer_node")
    assert hasattr(agent, "note_taker_node")
    assert hasattr(agent, "chart_generator_node")
