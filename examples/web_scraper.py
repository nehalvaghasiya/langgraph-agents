"""Example: Using WebScraperAgent to scrape web content."""

from langchain_core.messages import HumanMessage
from loguru import logger

from core.agents.web_scraper import WebScraperAgent
from infra.llm_clients.openai import get_llm


def main():
    """Example of using WebScraperAgent to scrape web content."""
    logger.info("Initializing WebScraperAgent example")
    
    # Get LLM instance
    llm = get_llm()
    logger.debug("LLM instance created")

    # Create Web Scraper agent
    web_scraper = WebScraperAgent(llm)
    logger.debug("WebScraperAgent initialized")
    
    # Define scraping task
    task = "Scrape the content from https://example.com and extract key information."
    logger.info(f"Scraping Task: {task}")
    
    # Invoke the agent
    logger.info("Invoking WebScraperAgent")
    result = web_scraper.graph.invoke(
        {
            "messages": [
                HumanMessage(content=task)
            ]
        }
    )
    
    # Log and print results
    logger.debug("Agent execution completed")
    logger.info(f"Result: {result}")
    print("WEB SCRAPER AGENT RESULT")
    print(result)


if __name__ == "__main__":
    main()
