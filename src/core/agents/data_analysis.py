from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent
from core.tools.math import (
    add_numbers,
    average,
    divide_numbers,
    multiply_numbers,
    round_number,
    subtract_numbers,
)
from core.tools.python_repl import create_python_repl_tool


class DataAnalysisAgent(BaseAgent):
    """Agent for data analysis and processing.

    This agent combines Python execution and mathematical operations
    to analyze datasets, compute statistics, and visualize data.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[Callable] | None = None,
        system: str = "",
    ):
        """Initialize DataAnalysisAgent.

        Args:
            model (BaseChatModel): The language model to use.
            tools (list[Callable] | None): Optional list of tools (uses defaults if None).
            system (str): Optional system message override.
        """
        system = (
            system
            or "You are a data analysis expert. Use Python for data processing and math tools for calculations. "
            "Break down complex analyses into steps and provide clear insights from the data. "
            "Use visualizations and statistics to support your analysis."
        )
        tools = [
            create_python_repl_tool(),
            add_numbers,
            subtract_numbers,
            multiply_numbers,
            divide_numbers,
            average,
            round_number,
        ]
        super().__init__(model, tools, system)
