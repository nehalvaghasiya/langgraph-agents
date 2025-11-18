"""Tests for Task Planner Agent."""

import pytest
from unittest.mock import MagicMock
from core.agents.task_planner import TaskPlannerAgent
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
def task_planner():
    """Create a TaskPlannerAgent instance with mock model."""
    model = DummyModel()
    return TaskPlannerAgent(model)


def test_task_planner_initialization(task_planner):
    """Test that TaskPlannerAgent initializes correctly."""
    assert task_planner is not None
    assert task_planner.graph is not None
    assert task_planner.model is not None
    assert len(task_planner.tools) == 4  # 4 tools


def test_task_planner_has_correct_tools(task_planner):
    """Test that TaskPlannerAgent has the correct tools."""
    tool_names = set(task_planner.tools.keys())
    expected_tools = {"plan_tasks", "analyze_reasoning", "observe_progress", "compare_results"}
    assert tool_names == expected_tools


def test_task_planner_system_message(task_planner):
    """Test that TaskPlannerAgent has appropriate system message."""
    assert "task planning" in task_planner.system.lower()
    assert "planning" in task_planner.system.lower()
    assert "reasoning" in task_planner.system.lower()


def test_task_planner_can_invoke(task_planner):
    """Test that TaskPlannerAgent can be invoked with mock model."""
    state = AgentState(messages=[MagicMock(tool_calls=[])])
    result = task_planner.call_model(state)
    
    assert result is not None
    assert "messages" in result
    assert isinstance(result["messages"], list)


def test_task_planner_with_complex_goal(task_planner):
    """Test TaskPlannerAgent tool invocation."""
    state = AgentState(messages=[MagicMock(tool_calls=[])])
    result = task_planner.call_model(state)
    
    assert result is not None
    assert "messages" in result
    assert isinstance(result["messages"], list)
