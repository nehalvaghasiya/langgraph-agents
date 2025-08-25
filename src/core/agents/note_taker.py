from core.agents.base import BaseAgent
from core.tools.document_io import create_outline, read_document


class NoteTakerAgent(BaseAgent):
    """Class for Note taking Agent."""
    def __init__(self, model, tools=[create_outline, read_document], system=""):
        """Initialize NoteTakerAgent."""
        system = system or "You can read documents and create outlines for the document writer. Don't ask follow-up questions."
        super().__init__(model, tools, system)