"""Example: Using PaperWritingTeamAgent for collaborative writing tasks."""

from langchain_core.messages import HumanMessage
from loguru import logger

from core.agents.paper_writing import PaperWritingTeamAgent
from infra.llm_clients.openai import get_llm


def main():
    """Example of using PaperWritingTeamAgent to write poems and related content."""
    logger.info("Initializing PaperWritingTeamAgent example")
    
    # Get LLM instance
    llm = get_llm()
    logger.debug("LLM instance created")

    # Create Paper Writing Team agent
    paper_team = PaperWritingTeamAgent(llm)
    logger.debug("PaperWritingTeamAgent initialized")
    
    # Define the task
    task = "Write an outline for a poem about cats and then write the poem to disk."
    logger.info(f"Task: {task}")
    
    # Invoke the agent
    logger.info("Invoking PaperWritingTeamAgent")
    team_result = paper_team.graph.invoke(
        {
            "messages": [
                HumanMessage(content=task)
            ]
        }
    )
    
    # Log and print results
    logger.debug("Agent execution completed")
    logger.info(f"Result: {team_result}")
    print("\n" + "="*80)
    print("PAPER WRITING TEAM AGENT RESULT")
    print("="*80)
    print(team_result)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
