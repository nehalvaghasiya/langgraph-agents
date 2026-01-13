"""Example demonstrating the MathAgent for performing arithmetic operations.

This script shows how to use the MathAgent to solve various mathematical problems
using arithmetic, algebraic, trigonometric, and statistical tools.

Usage:
    export GROQ_API_KEY="gsk-..."
    PYTHONPATH=src python3 examples/math_agent.py
"""

from infra.llm_clients.openai import get_llm
from core.agents.math_agent import MathAgent
from langchain_core.messages import HumanMessage


def main():
    """Run the MathAgent example."""
    # Get LLM instance
    llm = get_llm()

    # Initialize the MathAgent
    math_agent = MathAgent(llm)

    # Example 1: Basic arithmetic
    print("Example 1: Basic Arithmetic")
    query1 = "What is 45 plus 23 minus 10?"
    print(f"Query: {query1}\n")
    result1 = math_agent.graph.invoke({"messages": [HumanMessage(content=query1)]})
    print(f"Answer: {result1['messages'][-1].content}\n")

    # Example 2: Power and roots
    print("Example 2: Powers and Square Roots")
    query2 = "Calculate 2 to the power of 8 and then find the square root of 256."
    print(f"Query: {query2}\n")
    result2 = math_agent.graph.invoke({"messages": [HumanMessage(content=query2)]})
    print(f"Answer: {result2['messages'][-1].content}\n")

    # Example 3: Percentage calculations
    print("Example 3: Percentage Calculations")
    query3 = "If a product costs 100 and there's a 20% discount, what's the final price? Also calculate the percentage savings."
    print(f"Query: {query3}\n")
    result3 = math_agent.graph.invoke({"messages": [HumanMessage(content=query3)]})
    print(f"Answer: {result3['messages'][-1].content}\n")

    # Example 4: Trigonometry
    print("Example 4: Trigonometry")
    query4 = "What is the sine of 30 degrees and cosine of 60 degrees?"
    print(f"Query: {query4}\n")
    result4 = math_agent.graph.invoke({"messages": [HumanMessage(content=query4)]})
    print(f"Answer: {result4['messages'][-1].content}\n")

    # Example 5: Advanced calculations
    print("Example 5: Advanced Calculations")
    query5 = "Find the factorial of 5 and the greatest common divisor of 48 and 18."
    print(f"Query: {query5}\n")
    result5 = math_agent.graph.invoke({"messages": [HumanMessage(content=query5)]})
    print(f"Answer: {result5['messages'][-1].content}\n")

    # Example 6: Statistical operations
    print("Example 6: Statistical Operations")
    query6 = "Calculate the average of these numbers: 10, 20, 30, 40, 50."
    print(f"Query: {query6}\n")
    result6 = math_agent.graph.invoke({"messages": [HumanMessage(content=query6)]})
    print(f"Answer: {result6['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
