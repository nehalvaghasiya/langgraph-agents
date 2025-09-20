from typing import Callable
from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent
from core.tools.document_io import edit_document, read_document, write_document


class DocWriterAgent(BaseAgent):
    """Agent class for writing Document."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[Callable] | None = None,
        system: str = "",
    ):
        """Initialize DocWriterAgent."""
        system = (
            system
            or "You can read, write and edit documents based on note-taker's outlines. Don't ask follow-up questions."
        )
        tools = [write_document, edit_document, read_document]
        super().__init__(model, tools, system)
