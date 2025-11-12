"""Example: Using NoteTakerAgent to take notes."""

from langchain_core.messages import HumanMessage
from loguru import logger

from core.agents.note_taker import NoteTakerAgent
from infra.llm_clients.groq import get_llm


def main():
    """Example of using NoteTakerAgent to take structured notes."""
    logger.info("Initializing NoteTakerAgent example")
    
    # Get LLM instance
    llm = get_llm()
    logger.debug("LLM instance created")

    # Create Note Taker agent
    note_taker = NoteTakerAgent(llm)
    logger.debug("NoteTakerAgent initialized")
    
    # Define content to take notes on
    content = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. There are three main types: supervised learning, unsupervised learning, and reinforcement learning."
    logger.info(f"Content: {content}")
    
    # Invoke the agent
    logger.info("Invoking NoteTakerAgent")
    result = note_taker.graph.invoke(
        {
            "messages": [
                HumanMessage(content=content)
            ]
        }
    )
    
    # Log and print results
    logger.debug("Agent execution completed")
    logger.info(f"Result: {result}")
    print("\n" + "="*80)
    print("NOTE TAKER AGENT RESULT")
    print("="*80)
    print(result)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
