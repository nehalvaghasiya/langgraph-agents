"""Example: Using ResearchTeamAgent to research topics."""

from langchain_core.messages import HumanMessage
from loguru import logger

from core.agents.research_team import ResearchTeamAgent
from infra.llm_clients.groq import get_llm


def main():
    """Example of using ResearchTeamAgent to research a topic."""
    logger.info("Initializing ResearchTeamAgent example")
    
    # Get LLM instance
    llm = get_llm()
    logger.debug("LLM instance created")

    # Create Research Team agent
    research_team = ResearchTeamAgent(llm)
    logger.debug("ResearchTeamAgent initialized")
    
    # Define research query
    query = "Research the impact of artificial intelligence on modern society."
    logger.info(f"Research Query: {query}")
    
    # Invoke the agent
    logger.info("Invoking ResearchTeamAgent")
    result = research_team.graph.invoke(
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
    print("RESEARCH TEAM AGENT RESULT")
    print("="*80)
    print(result)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
