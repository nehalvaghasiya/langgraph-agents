from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent
from core.tools.scrape import scrape_webpages


class WebScraperAgent(BaseAgent):
    """Create Agent for Web scraping."""

    def __init__(self, model: BaseChatModel, tools: list[Callable] | None = None, system: str = ""):
        """Initialize WebScraperAgent."""
        system = (
            system
            or "You are a web scraper. Use tools to scrape webpages. Don't ask follow-up questions."
        )
        tools = [scrape_webpages]
        super().__init__(model, tools, system)
