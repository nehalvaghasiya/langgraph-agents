from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent
from infra.api.google_search import get_google_search


class SearchAgent(BaseAgent):
    """Search Agent class to search on Internet."""

    def __init__(self, model: BaseChatModel, tools: list[Callable] | None = None, system: str = ""):
        """Initialize search agent."""
        system = (
            system
            or "You are a search assistant. Use the web search tool to look up information. Don't ask follow-up questions."
        )
        tools = [get_google_search()]
        super().__init__(model, tools, system)
