from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent
from core.tools.math import add_numbers, multiply_numbers


class MathAgent(BaseAgent):
    """Agent class for performing mathematical operations.

    This agent leverages mathematical tools to solve arithmetic problems,
    including addition and multiplication operations.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[Callable] | None = None,
        system: str = "",
    ):
        """Initialize MathAgent.

        Args:
            model (BaseChatModel): The language model to use.
            tools (list[Callable] | None): Optional list of tools (uses defaults if None).
            system (str): Optional system message override.
        """
        system = (
            system
            or "You are a mathematical assistant. Use the available tools to perform arithmetic operations. "
            "Break down complex calculations into steps and explain your reasoning clearly."
        )
        tools = [add_numbers, multiply_numbers]
        super().__init__(model, tools, system)
