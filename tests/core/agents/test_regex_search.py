"""Tests for Regex Search Agent."""

import pytest
from unittest.mock import MagicMock
from core.agents import regex_search
from core.agents.base import AgentState


class DummyModel:
    """Mock LLM model for testing."""

    def __init__(self, output=None):
        self.output = output or MagicMock(tool_calls=[])

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.output


@pytest.fixture
def regex_agent():
    """Create a RegexSearchAgent instance with mock model."""
    model = DummyModel()
    return regex_search.RegexSearchAgent(model)


def test_regex_agent_initialization(regex_agent):
    """Test that RegexSearchAgent initializes correctly."""
    assert regex_agent is not None
    assert regex_agent.graph is not None
    assert regex_agent.model is not None


def test_regex_agent_has_required_tools(regex_agent):
    """Test that RegexSearchAgent has the required tools."""
    assert len(regex_agent.tools) == 6  # 6 tools
    tool_names = set(regex_agent.tools.keys())
    expected_tools = {
        "validate_and_explain_pattern",
        "compile_regex_pattern",
        "search_files_by_pattern",
        "search_text_in_file",
        "extract_pattern_matches",
        "replace_pattern_in_file",
    }
    assert tool_names == expected_tools


def test_regex_agent_system_message(regex_agent):
    """Test that RegexSearchAgent has appropriate system message."""
    assert "regex" in regex_agent.system.lower()
    assert "pattern" in regex_agent.system.lower()
    assert "search" in regex_agent.system.lower()


def test_regex_agent_can_invoke(regex_agent):
    """Test that regex agent can be invoked."""
    state = AgentState(messages=[MagicMock(tool_calls=[])])
    result = regex_agent.call_model(state)
    assert result is not None
    assert "messages" in result


def test_regex_agent_with_search_task(regex_agent):
    """Test regex agent with a search task."""
    state = AgentState(messages=[MagicMock(tool_calls=[])])
    result = regex_agent.call_model(state)
    assert result is not None
    assert isinstance(result, dict)
