"""Example: Using DocWriterAgent to generate documents."""

from langchain_core.messages import HumanMessage
from loguru import logger

from core.agents.doc_writer import DocWriterAgent
from infra.llm_clients.groq import get_llm


def main():
    """Example of using DocWriterAgent to write documents."""
    logger.info("Initializing DocWriterAgent example")
    
    # Get LLM instance
    llm = get_llm()
    logger.debug("LLM instance created")

    # Create Doc Writer agent
    doc_writer = DocWriterAgent(llm)
    logger.debug("DocWriterAgent initialized")
    
    # Define query
    query = "Write a comprehensive document about the history and characteristics of cats."
    logger.info(f"Query: {query}")
    
    # Invoke the agent
    logger.info("Invoking DocWriterAgent")
    result = doc_writer.graph.invoke({"messages": [HumanMessage(content=query)]})
    
    # Log and print results
    logger.debug("Agent execution completed")
    logger.info(f"Result: {result}")
    print("\n" + "="*80)
    print("DOC WRITER AGENT RESULT")
    print("="*80)
    print(result)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
