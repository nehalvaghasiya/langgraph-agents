import pytest
from unittest.mock import MagicMock
from core.agents import rag
from core.agents.base import AgentState

class DummyModel:
    def __init__(self, output=None):
        self.output = output or MagicMock(tool_calls=[])
    def bind_tools(self, tools):
        return self
    def invoke(self, messages):
        return self.output

def test_rag_agent():
    model = DummyModel()
    doc_splits = []
    agent = rag.RagAgent(model, doc_splits)
    state = AgentState(messages=[MagicMock(tool_calls=[])])
    assert hasattr(agent, "graph")
    assert isinstance(agent.generate_query_or_respond(state), dict)
