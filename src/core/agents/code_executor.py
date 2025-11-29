from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent
from core.tools.python_repl import create_python_repl_tool


class CodeExecutorAgent(BaseAgent):
    """Agent for executing and testing Python code.

    This agent specializes in running Python code, debugging, and validating
    code snippets using a persistent REPL environment.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[Callable] | None = None,
        system: str = "",
    ):
        """Initialize CodeExecutorAgent.

        Args:
            model (BaseChatModel): The language model to use.
            tools (list[Callable] | None): Optional list of tools (uses defaults if None).
            system (str): Optional system message override.
        """
        system = (
            system
            or "You are a Python code execution assistant. Use the Python REPL tool to execute and test code. "
            "Provide clear explanations of what the code does and its output. "
            "Debug code issues and suggest improvements when needed."
        )
        tools = [create_python_repl_tool()]
        super().__init__(model, tools, system)
