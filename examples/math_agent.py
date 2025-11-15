"""Example demonstrating the MathAgent for performing arithmetic operations.

This script shows how to use the MathAgent to solve mathematical problems
using the add_numbers and multiply_numbers tools.

Usage:
    export GROQ_API_KEY="gsk-..."
    PYTHONPATH=src python3 examples/math_agent.py
"""

from infra.llm_clients.groq import get_llm
from core.agents.math_agent import MathAgent
from langchain_core.messages import HumanMessage


def main():
    """Run the MathAgent example."""
    # Get LLM instance
    llm = get_llm()

    # Initialize the MathAgent
    math_agent = MathAgent(llm)

    # Example 1: Simple arithmetic problem
    print("=" * 60)
    print("Example 1: Simple Arithmetic Problem")
    print("=" * 60)
    query1 = "What is 45 plus 23?"
    print(f"Query: {query1}\n")
    result1 = math_agent.graph.invoke({"messages": [HumanMessage(content=query1)]})
    print(f"Answer: {result1['messages'][-1].content}\n")

    # Example 2: Multiplication problem
    print("=" * 60)
    print("Example 2: Multiplication Problem")
    print("=" * 60)
    query2 = "Calculate 12 multiplied by 8."
    print(f"Query: {query2}\n")
    result2 = math_agent.graph.invoke({"messages": [HumanMessage(content=query2)]})
    print(f"Answer: {result2['messages'][-1].content}\n")

    # Example 3: Combined arithmetic problem
    print("=" * 60)
    print("Example 3: Combined Arithmetic Problem")
    print("=" * 60)
    query3 = "First, add 50 and 30. Then multiply the result by 2. What's the final answer?"
    print(f"Query: {query3}\n")
    result3 = math_agent.graph.invoke({"messages": [HumanMessage(content=query3)]})
    print(f"Answer: {result3['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
