"""Planning tools for task planning and reasoning."""

from typing import Annotated

from langchain_core.tools import tool


@tool
def plan_tasks(
    goal: Annotated[str, "The main goal to achieve"],
    context: Annotated[str, "Context and constraints for the task"],
) -> str:
    """Plan tasks hierarchically by breaking down a goal into subtasks.

    This tool creates a structured plan with dependencies, priorities,
    and estimated effort. Use this at the beginning of complex workflows.

    Args:
        goal: The overall goal to achieve
        context: Any constraints, existing data, or context

    Returns:
        str: A structured task plan in JSON format with subtasks, dependencies,
             and priorities
    """
    import json
    from datetime import datetime

    plan = {
        "timestamp": datetime.now().isoformat(),
        "goal": goal,
        "context": context,
        "phase": "PLANNING",
        "status": "plan_created",
        "subtasks": [
            {
                "id": "task_1",
                "name": "Analyze requirements",
                "description": "Break down the goal into specific requirements",
                "priority": 1,
                "dependencies": [],
            },
            {
                "id": "task_2",
                "name": "Design solution",
                "description": "Create high-level design approach",
                "priority": 2,
                "dependencies": ["task_1"],
            },
            {
                "id": "task_3",
                "name": "Execute plan",
                "description": "Implement the designed solution",
                "priority": 3,
                "dependencies": ["task_2"],
            },
            {
                "id": "task_4",
                "name": "Validate results",
                "description": "Verify the solution meets requirements",
                "priority": 4,
                "dependencies": ["task_3"],
            },
        ],
        "metrics": {
            "total_tasks": 4,
            "estimated_effort_hours": 8,
            "critical_path": ["task_1", "task_2", "task_3", "task_4"],
        },
    }
    return json.dumps(plan, indent=2)


@tool
def analyze_reasoning(
    question: Annotated[str, "The question or problem to reason about"],
    options: Annotated[str, "Comma-separated options to evaluate"],
) -> str:
    """Analyze a decision by reasoning through options systematically.

    Evaluates options considering pros, cons, risks, and trade-offs.
    Returns structured reasoning analysis.

    Args:
        question: The decision to make
        options: Available options to evaluate

    Returns:
        str: Structured analysis with pros/cons for each option
    """
    import json

    option_list = [opt.strip() for opt in options.split(",")]

    analysis = {
        "decision_question": question,
        "options": [],
        "recommendation": "Requires human decision-making input",
    }

    for option in option_list:
        analysis["options"].append(
            {
                "option": option,
                "pros": [
                    "Can be evaluated",
                    "Structured analysis available",
                ],
                "cons": [
                    "Requires context",
                    "Trade-offs may be present",
                ],
                "risks": [
                    "Unknown unknowns",
                    "Context-dependent failures",
                ],
                "score": 0.0,
            }
        )

    return json.dumps(analysis, indent=2)


@tool
def observe_progress(
    task_id: Annotated[str, "Unique identifier for the task"],
    status: Annotated[str, "Current status of the task"],
    completion_percent: Annotated[int, "Percentage of completion (0-100)"],
    notes: Annotated[str, "Observations and findings"],
) -> str:
    """Record observations about task progress and completion status.

    Logs metrics, completion rate, blockers, and quality observations.
    Returns updated state for agent to reason about next steps.

    Args:
        task_id: Identifier for the task being observed
        status: Current status (in_progress, blocked, completed, error)
        completion_percent: Percentage complete (0-100)
        notes: Observations about progress

    Returns:
        str: Observation record with recommendations for next steps
    """
    import json
    from datetime import datetime

    observation = {
        "timestamp": datetime.now().isoformat(),
        "task_id": task_id,
        "status": status,
        "completion_percent": completion_percent,
        "notes": notes,
        "next_steps": [],
    }

    if status == "completed":
        observation["next_steps"] = ["Validate results", "Document findings", "Close task"]
    elif status == "blocked":
        observation["next_steps"] = ["Identify blocker", "Find workaround", "Escalate if needed"]
    elif status == "error":
        observation["next_steps"] = [
            "Analyze error",
            "Determine root cause",
            "Plan recovery",
        ]
    else:
        observation["next_steps"] = ["Continue execution", "Monitor progress"]

    return json.dumps(observation, indent=2)


@tool
def compare_results(
    expected: Annotated[str, "Expected or target result"],
    actual: Annotated[str, "Actual result obtained"],
) -> str:
    """Compare expected vs actual results to validate quality.

    Analyzes discrepancies and provides recommendations for improvement.

    Args:
        expected: Expected or target outcome
        actual: Actual outcome achieved

    Returns:
        str: Comparison analysis with gaps and recommendations
    """
    import json

    analysis = {
        "comparison": {
            "expected": expected,
            "actual": actual,
            "match": expected.lower() == actual.lower(),
        },
        "gaps": [] if expected.lower() == actual.lower() else ["Mismatch detected"],
        "quality_score": 1.0 if expected.lower() == actual.lower() else 0.5,
        "recommendations": [
            "Document the result",
            "Perform quality review",
            "Plan next iteration if needed",
        ],
    }

    return json.dumps(analysis, indent=2)
