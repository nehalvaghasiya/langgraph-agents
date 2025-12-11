"""Example usage of Regex Search Agent.

Demonstrates searching for files, extracting patterns from documents,
and pattern validation with an AI agent.
"""

from infra.llm_clients.openai import get_llm
from core.agents.regex_search import RegexSearchAgent
from langchain_core.messages import HumanMessage


def main():
    """Run regex search agent examples."""
    print("\n" + "=" * 80)
    print("REGEX SEARCH AGENT EXAMPLES")
    print("=" * 80)

    # Initialize the agent
    llm = get_llm()
    agent = RegexSearchAgent(llm)

    # Example 1: Find Python files
    print("\n" + "=" * 80)
    print("Example 1: Search for Python Files")
    print("=" * 80)
    query1 = (
        "Find all Python files (.py extension) in the src directory. "
        "Use a regex pattern to match files ending with .py"
    )
    print(f"Query: {query1}\n")
    result1 = agent.graph.invoke({"messages": [HumanMessage(content=query1)]})
    print(f"Result: {result1['messages'][-1].content}")

    # Example 2: Extract function definitions
    print("\n" + "=" * 80)
    print("Example 2: Extract Python Function Definitions from Code")
    print("=" * 80)
    query2 = (
        "Create a regex pattern that matches Python function definitions (def function_name...) "
        "and explain what the pattern matches. Test it with some examples."
    )
    print(f"Query: {query2}\n")
    result2 = agent.graph.invoke({"messages": [HumanMessage(content=query2)]})
    print(f"Result: {result2['messages'][-1].content}")

    # Example 3: Find imports in Python files
    print("\n" + "=" * 80)
    print("Example 3: Find Import Statements")
    print("=" * 80)
    query3 = (
        "Search for import statements in the file src/core/tools/regex.py. "
        "Use a regex pattern to match lines starting with 'import' or 'from'."
    )
    print(f"Query: {query3}\n")
    result3 = agent.graph.invoke({"messages": [HumanMessage(content=query3)]})
    print(f"Result: {result3['messages'][-1].content[:500]}...\n")

    # Example 4: Extract email addresses
    print("\n" + "=" * 80)
    print("Example 4: Extract Email Addresses from Text")
    print("=" * 80)
    query4 = (
        "Extract all email addresses from this text: "
        "'Contact us at support@example.com or info@company.org. For urgent issues, email urgent@example.com'. "
        "Create a regex pattern that matches email addresses and explain the pattern components."
    )
    print(f"Query: {query4}\n")
    result4 = agent.graph.invoke({"messages": [HumanMessage(content=query4)]})
    print(f"Result: {result4['messages'][-1].content}")

    # Example 5: Find test files
    print("\n" + "=" * 80)
    print("Example 5: Find Test Files")
    print("=" * 80)
    query5 = (
        "Find all test files that start with 'test_' and end with '.py' in the tests directory. "
        "Create the appropriate regex pattern and explain what it matches."
    )
    print(f"Query: {query5}\n")
    result5 = agent.graph.invoke({"messages": [HumanMessage(content=query5)]})
    print(f"Result: {result5['messages'][-1].content[:500]}...\n")

    # Example 6: Extract URLs
    print("\n" + "=" * 80)
    print("Example 6: Extract URLs from Text")
    print("=" * 80)
    query6 = (
        "Create a regex pattern to match URLs. Test it with examples like "
        "'https://example.com', 'http://test.org/path', 'ftp://files.server.com'. "
        "Explain which examples match and why."
    )
    print(f"Query: {query6}\n")
    result6 = agent.graph.invoke({"messages": [HumanMessage(content=query6)]})
    print(f"Result: {result6['messages'][-1].content}")

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
