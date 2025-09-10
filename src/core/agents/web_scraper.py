from core.agents.base import BaseAgent
from core.tools.scrape import scrape_webpages


class WebScraperAgent(BaseAgent):
    """Create Agent for Web scraping."""
    def __init__(self, model, tools=[scrape_webpages], system=""):
        """Initialize WebScraperAgent."""
        system = system or "You are a web scraper. Use tools to scrape webpages. Don't ask follow-up questions."
        super().__init__(model, tools, system)