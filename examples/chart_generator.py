"""Example: Using ChartGeneratorAgent to create charts."""

from langchain_core.messages import HumanMessage
from loguru import logger

from core.agents.chart_generator import ChartGeneratorAgent
from infra.llm_clients.groq import get_llm


def main():
    """Example of using ChartGeneratorAgent to generate charts."""
    logger.info("Initializing ChartGeneratorAgent example")
    
    # Get LLM instance
    llm = get_llm()
    logger.debug("LLM instance created")

    # Create Chart Generator agent
    chart_generator = ChartGeneratorAgent(llm)
    logger.debug("ChartGeneratorAgent initialized")
    
    # Define chart request
    request = "Create a chart showing the growth of artificial intelligence adoption over the past 5 years with data points for 2019, 2020, 2021, 2022, and 2023."
    logger.info(f"Chart Request: {request}")
    
    # Invoke the agent
    logger.info("Invoking ChartGeneratorAgent")
    result = chart_generator.graph.invoke(
        {
            "messages": [
                HumanMessage(content=request)
            ]
        }
    )
    
    # Log and print results
    logger.debug("Agent execution completed")
    logger.info(f"Result: {result}")
    print("\n" + "="*80)
    print("CHART GENERATOR AGENT RESULT")
    print("="*80)
    print(result)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
