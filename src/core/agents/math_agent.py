from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent
from core.tools.math import (
    add_numbers,
    absolute_value,
    average,
    cosine,
    divide_numbers,
    factorial,
    greatest_common_divisor,
    least_common_multiple,
    logarithm,
    multiply_numbers,
    percentage,
    percentage_increase,
    power,
    round_number,
    sine,
    square_root,
    subtract_numbers,
    tangent,
)


class MathAgent(BaseAgent):
    """Agent class for performing mathematical operations.

    This agent leverages comprehensive mathematical tools to solve arithmetic,
    algebraic, trigonometric, and statistical problems.
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
            or "You are a mathematical assistant with expertise in arithmetic, algebra, trigonometry, "
            "and statistics. Use the available tools to perform calculations and solve problems. "
            "Break down complex calculations into steps and explain your reasoning clearly. "
            "Always provide the final answer in a clear and concise manner."
        )
        tools = [
            add_numbers,
            subtract_numbers,
            multiply_numbers,
            divide_numbers,
            power,
            square_root,
            absolute_value,
            percentage,
            percentage_increase,
            average,
            factorial,
            round_number,
            greatest_common_divisor,
            least_common_multiple,
            logarithm,
            sine,
            cosine,
            tangent,
        ]
        super().__init__(model, tools, system)
