from langchain_core.tools import tool


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a (float): The first number to be added.
        b (float): The second number to be added.

    Returns:
        float: The sum of the two input numbers (a + b).
    """
    result = a + b
    print(f"Adding {a} + {b} = {result}")
    return result


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together.

    Performs multiplication of two floating-point numbers and prints the operation
    to the console for verification and debugging purposes.

    Args:
        a (float): The first number to be multiplied.
        b (float): The second number to be multiplied.

    Returns:
        float: The product of the two input numbers (a * b).
    """
    result = a * b
    print(f"Multiplying {a} * {b} = {result}")
    return result