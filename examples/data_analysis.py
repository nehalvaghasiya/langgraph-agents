"""Example demonstrating the DataAnalysisAgent for data processing and analysis.

This script shows how to use the DataAnalysisAgent to analyze datasets,
compute statistics, and create visualizations.

Usage:
    export GROQ_API_KEY="gsk-..."
    PYTHONPATH=src python3 examples/data_analysis.py
"""

from infra.llm_clients.groq import get_llm
from core.agents.data_analysis import DataAnalysisAgent
from langchain_core.messages import HumanMessage


def main():
    """Run the DataAnalysisAgent example."""
    # Get LLM instance
    llm = get_llm()

    # Initialize the DataAnalysisAgent
    data_analyzer = DataAnalysisAgent(llm)

    # Example 1: Basic statistics
    print("=" * 60)
    print("Example 1: Calculate Statistics")
    print("=" * 60)
    query1 = "Analyze this dataset: [23, 45, 56, 78, 34, 56, 89, 12, 34, 56]. " \
             "Calculate the average, find the maximum, minimum, and count occurrences."
    print(f"Query: {query1}\n")
    result1 = data_analyzer.graph.invoke({"messages": [HumanMessage(content=query1)]})
    print(f"Response: {result1['messages'][-1].content}\n")

    # Example 2: Data transformation
    print("=" * 60)
    print("Example 2: Data Transformation")
    print("=" * 60)
    query2 = "I have sales data for 5 months: [1000, 1500, 1200, 1800, 2000]. " \
             "Calculate the month-to-month growth percentages and the overall average."
    print(f"Query: {query2}\n")
    result2 = data_analyzer.graph.invoke({"messages": [HumanMessage(content=query2)]})
    print(f"Response: {result2['messages'][-1].content}\n")

    # Example 3: Complex calculations
    print("=" * 60)
    print("Example 3: Complex Analysis")
    print("=" * 60)
    query3 = "Analyze temperature data for a week: [72, 75, 73, 78, 80, 76, 74] degrees. " \
             "Calculate average, highest, lowest, and identify the temperature range."
    print(f"Query: {query3}\n")
    result3 = data_analyzer.graph.invoke({"messages": [HumanMessage(content=query3)]})
    print(f"Response: {result3['messages'][-1].content}\n")

    # Example 4: Financial analysis
    print("=" * 60)
    print("Example 4: Financial Calculations")
    print("=" * 60)
    query4 = "I invested $5000 and earned $1250 in returns. Calculate my ROI percentage " \
             "and round it to 2 decimal places."
    print(f"Query: {query4}\n")
    result4 = data_analyzer.graph.invoke({"messages": [HumanMessage(content=query4)]})
    print(f"Response: {result4['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
