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

