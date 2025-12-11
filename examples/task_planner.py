"""Example usage of Task Planner Agent.

Demonstrates planning, reasoning, acting, and observing in task breakdown.
"""

from infra.llm_clients.openai import get_llm
from core.agents.task_planner import TaskPlannerAgent
from langchain_core.messages import HumanMessage


def main():
    """Run task planner example."""
    llm = get_llm()
    agent = TaskPlannerAgent(llm)

    # Example: Planning a complex project
    query = """Plan the breakdown for "Building a real-time data analytics dashboard":
    
    Requirements:
    - Real-time data ingestion from multiple sources
    - Interactive visualization with filtering capabilities
    - Performance monitoring and optimization
    - Mobile-responsive design
    
    Please:
    1. Create a detailed task breakdown with dependencies
    2. Identify critical path and priorities
    3. Estimate effort for each phase
    4. Analyze dependencies between tasks
    """

    result = agent.graph.invoke({"messages": [HumanMessage(content=query)]})
    print("Task Plan Generated:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
