import pytest
from unittest.mock import MagicMock
from core.agents import doc_writer
from core.agents.base import AgentState

class DummyModel:
    def __init__(self, output=None):
        self.output = output or MagicMock(tool_calls=[])
    def bind_tools(self, tools):
        return self
    def invoke(self, messages):
        return self.output

def test_doc_writer_agent():
    model = DummyModel()
    agent = doc_writer.DocWriterAgent(model)
    state = AgentState(messages=[MagicMock(tool_calls=[])])
    assert hasattr(agent, "graph")
    assert isinstance(agent.call_model(state), dict)
