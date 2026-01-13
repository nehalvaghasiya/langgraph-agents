"""Example demonstrating the CodeExecutorAgent for running and testing Python code.

This script shows how to use the CodeExecutorAgent to execute Python code,
debug issues, and validate code snippets.

Usage:
    export GROQ_API_KEY="gsk-..."
    PYTHONPATH=src python3 examples/code_executor.py
"""

from infra.llm_clients.openai import get_llm
from core.agents.code_executor import CodeExecutorAgent
from langchain_core.messages import HumanMessage


def main():
    """Run the CodeExecutorAgent example."""
    # Get LLM instance
    llm = get_llm()

    # Initialize the CodeExecutorAgent
    code_executor = CodeExecutorAgent(llm)

    # Example 1: Simple Python execution
    print("Example 1: Simple Python Execution")
    query1 = "Write a Python program that creates a list of numbers from 1 to 10 and prints their squares."
    print(f"Query: {query1}\n")
    result1 = code_executor.graph.invoke({"messages": [HumanMessage(content=query1)]})
    print(f"Response: {result1['messages'][-1].content}\n")

    # Example 2: String manipulation
    print("Example 2: String Manipulation")
    query2 = "Write Python code to reverse a string and check if it's a palindrome. Test with 'racecar'."
    print(f"Query: {query2}\n")
    result2 = code_executor.graph.invoke({"messages": [HumanMessage(content=query2)]})
    print(f"Response: {result2['messages'][-1].content}\n")

    # Example 3: Debugging code
    print("Example 3: Code Debugging")
    query3 = "Fix this Python code that should find the sum of even numbers: " \
             "numbers = [1, 2, 3, 4, 5, 6]; sum = 0; for n in numbers if n % 2 == 0: sum += n; print(sum)"
    print(f"Query: {query3}\n")
    result3 = code_executor.graph.invoke({"messages": [HumanMessage(content=query3)]})
    print(f"Response: {result3['messages'][-1].content}\n")

    # Example 4: Data structure operations
    print("Example 4: Dictionary Operations")
    query4 = "Write Python code to count the frequency of each character in the string 'hello world'."
    print(f"Query: {query4}\n")
    result4 = code_executor.graph.invoke({"messages": [HumanMessage(content=query4)]})
    print(f"Response: {result4['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
