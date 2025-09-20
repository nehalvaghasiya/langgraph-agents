from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent
from core.tools.document_io import read_document
from core.tools.python_repl import python_repl_tool


class ChartGeneratorAgent(BaseAgent):
    """Agent for generating Charts."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[Callable] | None = None,
        system: str = "",
    ):
        """Initialize Chart Generator Agent."""
        system = (
            system
            or "You are a chart generator. Use tools to read documents and generate charts with Python REPL. Don't ask follow-up questions."
        )
        tools = [read_document, python_repl_tool]
        super().__init__(model, tools, system)
