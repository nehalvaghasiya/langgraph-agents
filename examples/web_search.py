"""Example: Using SearchAgent for web searches."""

from langchain_core.messages import HumanMessage
from loguru import logger

from core.agents.web_search import SearchAgent
from infra.llm_clients.groq import get_llm


def main():
    """Example of using SearchAgent to search the web."""
    logger.info("Initializing SearchAgent example")
    
    # Get LLM instance
    llm = get_llm()
    logger.debug("LLM instance created")

    # Create Search agent with explicit tools (demonstrating configuration)
    from infra.api.google_search import get_google_search
    search_agent = SearchAgent(llm, tools=[get_google_search()])
    logger.debug("SearchAgent initialized")
    
    # Define search query
    query = "What are the latest breakthroughs in artificial intelligence?"
    logger.info(f"Search Query: {query}")
    
    # Invoke the agent
    logger.info("Invoking SearchAgent")
    result = search_agent.graph.invoke(
        {
            "messages": [
                HumanMessage(content=query)
            ]
        }
    )
    
    # Log and print results
    logger.debug("Agent execution completed")
    logger.info(f"Result: {result}")
    print("\n" + "="*80)
    print("SEARCH AGENT RESULT")
    print("="*80)
    print(result)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
