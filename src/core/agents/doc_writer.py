from core.agents.base import BaseAgent
from core.tools.document_io import edit_document, read_document, write_document


class DocWriterAgent(BaseAgent):
    def __init__(self, model, tools=[write_document, edit_document, read_document], system=""):
        system = system or "You can read, write and edit documents based on note-taker's outlines. Don't ask follow-up questions."
        super().__init__(model, tools, system)