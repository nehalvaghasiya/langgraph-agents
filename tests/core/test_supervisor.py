import pytest
from types import SimpleNamespace
from core.supervisor import State, make_supervisor_node
from langgraph.types import Command

class DummyLLM:
    def __init__(self, next_value=None, raise_exc=False, missing_next=False):
        self.next_value = next_value
        self.raise_exc = raise_exc
        self.missing_next = missing_next

    def with_structured_output(self, Router):
        return self
    
    def invoke(self, messages):
        if self.raise_exc:
            raise Exception("LLM error")
        if self.missing_next:
            return SimpleNamespace()  # No 'next' attribute
        return SimpleNamespace(next=self.next_value)

# State class
def test_state_instantiation():
    s1 = State(messages=[])
    assert isinstance(s1, dict)
    s2 = State(messages=[], next="worker1")
    assert s2["next"] == "worker1"
    assert "messages" in s2

# Make_supervisor_node returns callable
def test_make_supervisor_node_returns_callable():
    llm = DummyLLM(next_value="worker1")
    node = make_supervisor_node(llm, ["worker1"])
    assert callable(node)

# Handles empty members
@pytest.mark.parametrize("members", [[], ["worker1"], ["worker1", "worker2"]])
def test_make_supervisor_node_members(members):
    llm = DummyLLM(next_value="worker1")
    node = make_supervisor_node(llm, members)
    state = State(messages=[])
    result = node(state)
    assert isinstance(result, Command)
    assert hasattr(result, "goto")
    assert hasattr(result, "update")

# Handles state with empty and populated messages
@pytest.mark.parametrize("messages", [[], [{"role": "user", "content": "hi"}]])
def test_supervisor_node_messages(messages):
    llm = DummyLLM(next_value="worker1")
    node = make_supervisor_node(llm, ["worker1"])
    state = State(messages=messages)
    result = node(state)
    assert isinstance(result, Command)
