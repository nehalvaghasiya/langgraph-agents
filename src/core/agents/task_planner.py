"""Task Planner Agent that plans, reasons about, executes and observes task completion."""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent
from core.tools.planning import analyze_reasoning, compare_results, observe_progress, plan_tasks


class TaskPlannerAgent(BaseAgent):
    """Agent that plans tasks hierarchically with reasoning and observation.

    Implements the Planning-Reasoning-Acting-Observing loop:
    - Planning: Creates structured task breakdown
    - Reasoning: Analyzes task dependencies and priorities
    - Acting: Refines and validates the plan
    - Observing: Tracks completion and adjusts as needed
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[Callable] | None = None,
        system: str = "",
    ):
        """Initialize TaskPlannerAgent.

        Args:
            model: Language model to use
            tools: Optional custom tools
            system: Optional custom system prompt
        """
        system = (
            system
            or """You are a task planning expert. Your role is to:
1. PLAN: Break down goals into structured subtasks with clear dependencies
2. REASON: Analyze task relationships, priorities, and critical paths
3. ACT: Create and refine detailed execution plans
4. OBSERVE: Monitor progress and adapt plans based on completion rates

For each goal:
- Use plan_tasks to create initial structure
- Use analyze_reasoning to evaluate task dependencies
- Use observe_progress to track completion
- Use compare_results to validate against objectives

Provide clear, actionable task breakdowns with priorities and dependencies."""
        )
        tools = [plan_tasks, analyze_reasoning, observe_progress, compare_results]
        super().__init__(model, tools, system)
