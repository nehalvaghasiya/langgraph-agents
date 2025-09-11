from core.agents.base import BaseAgent
from infra.api.google_search import get_google_search


class SearchAgent(BaseAgent):
    """Search Agent class to search on Internet."""
    def __init__(self, model, tools=[get_google_search()], system=""):
        """Initialize search agent."""
        system = system or "You are a search assistant. Use the web search tool to look up information. Don't ask follow-up questions."
        super().__init__(model, tools, system)