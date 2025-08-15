from core.agents.base import BaseAgent
from core.tools.document_io import read_document
from core.tools.python_repl import python_repl_tool


class ChartGeneratorAgent(BaseAgent):
    def __init__(self, model, tools=[read_document, python_repl_tool], system=""):
        system = system or "You are a chart generator. Use tools to read documents and generate charts with Python REPL. Don't ask follow-up questions."
        super().__init__(model, tools, system)