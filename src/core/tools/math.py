import math

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
def subtract_numbers(a: float, b: float) -> float:
    """Subtract two numbers.

    Args:
        a (float): The first number (minuend).
        b (float): The second number (subtrahend).

    Returns:
        float: The difference of the two input numbers (a - b).
    """
    result = a - b
    print(f"Subtracting {a} - {b} = {result}")
    return result


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together.

    Args:
        a (float): The first number to be multiplied.
        b (float): The second number to be multiplied.

    Returns:
        float: The product of the two input numbers (a * b).
    """
    result = a * b
    print(f"Multiplying {a} * {b} = {result}")
    return result


@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers.

    Args:
        a (float): The dividend (numerator).
        b (float): The divisor (denominator).

    Returns:
        float: The quotient of the two input numbers (a / b).

    Raises:
        ValueError: If attempting to divide by zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    result = a / b
    print(f"Dividing {a} / {b} = {result}")
    return result


@tool
def power(base: float, exponent: float) -> float:
    """Raise a number to a power.

    Args:
        base (float): The base number.
        exponent (float): The exponent to raise the base to.

    Returns:
        float: The result of base raised to the exponent (base ** exponent).
    """
    result = base**exponent
    print(f"Power {base} ** {exponent} = {result}")
    return result


@tool
def square_root(n: float) -> float:
    """Calculate the square root of a number.

    Args:
        n (float): The number to find the square root of.

    Returns:
        float: The square root of the input number.

    Raises:
        ValueError: If the input number is negative.
    """
    if n < 0:
        raise ValueError("Cannot calculate square root of negative number")
    result = math.sqrt(n)
    print(f"Square root of {n} = {result}")
    return result


@tool
def absolute_value(n: float) -> float:
    """Calculate the absolute value of a number.

    Args:
        n (float): The input number.

    Returns:
        float: The absolute value of the input number.
    """
    result = abs(n)
    print(f"Absolute value of {n} = {result}")
    return result


@tool
def percentage(value: float, percent: float) -> float:
    """Calculate a percentage of a value.

    Args:
        value (float): The base value.
        percent (float): The percentage to calculate (0-100).

    Returns:
        float: The percentage of the value.
    """
    result = (value * percent) / 100
    print(f"{percent}% of {value} = {result}")
    return result


@tool
def percentage_increase(original: float, new: float) -> float:
    """Calculate the percentage increase from an original value to a new value.

    Args:
        original (float): The original value.
        new (float): The new value.

    Returns:
        float: The percentage increase.
    """
    if original == 0:
        raise ValueError("Original value cannot be zero")
    result = ((new - original) / original) * 100
    print(f"Percentage increase from {original} to {new} = {result}%")
    return result


@tool
def average(numbers: list[float]) -> float:
    """Calculate the average (mean) of a list of numbers.

    Args:
        numbers (list[float]): A list of numbers to average.

    Returns:
        float: The average of the input numbers.

    Raises:
        ValueError: If the list is empty.
    """
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    result = sum(numbers) / len(numbers)
    print(f"Average of {numbers} = {result}")
    return result


@tool
def factorial(n: int) -> int:
    """Calculate the factorial of a number.

    Args:
        n (int): The non-negative integer to calculate factorial for.

    Returns:
        int: The factorial of n (n!).

    Raises:
        ValueError: If the input is negative.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    result = math.factorial(n)
    print(f"Factorial of {n} = {result}")
    return result


@tool
def round_number(n: float, decimal_places: int = 0) -> float:
    """Round a number to a specified number of decimal places.

    Args:
        n (float): The number to round.
        decimal_places (int): The number of decimal places to round to. Defaults to 0.

    Returns:
        float: The rounded number.
    """
    result = round(n, decimal_places)
    print(f"Rounding {n} to {decimal_places} decimal places = {result}")
    return result


@tool
def greatest_common_divisor(a: int, b: int) -> int:
    """Calculate the greatest common divisor (GCD) of two numbers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The greatest common divisor of a and b.
    """
    result = math.gcd(a, b)
    print(f"GCD of {a} and {b} = {result}")
    return result


@tool
def least_common_multiple(a: int, b: int) -> int:
    """Calculate the least common multiple (LCM) of two numbers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The least common multiple of a and b.
    """
    result = (a * b) // math.gcd(a, b)
    print(f"LCM of {a} and {b} = {result}")
    return result


@tool
def logarithm(n: float, base: float = math.e) -> float:
    """Calculate the logarithm of a number.

    Args:
        n (float): The number to calculate the logarithm of.
        base (float): The base of the logarithm. Defaults to e (natural logarithm).

    Returns:
        float: The logarithm of n with the specified base.

    Raises:
        ValueError: If n is not positive.
    """
    if n <= 0:
        raise ValueError("Logarithm is only defined for positive numbers")
    if base <= 0 or base == 1:
        raise ValueError("Logarithm base must be positive and not equal to 1")
    result = math.log(n, base)
    print(f"Log base {base} of {n} = {result}")
    return result


@tool
def sine(angle_degrees: float) -> float:
    """Calculate the sine of an angle in degrees.

    Args:
        angle_degrees (float): The angle in degrees.

    Returns:
        float: The sine of the angle.
    """
    angle_radians = math.radians(angle_degrees)
    result = math.sin(angle_radians)
    print(f"sin({angle_degrees}°) = {result}")
    return result


@tool
def cosine(angle_degrees: float) -> float:
    """Calculate the cosine of an angle in degrees.

    Args:
        angle_degrees (float): The angle in degrees.

    Returns:
        float: The cosine of the angle.
    """
    angle_radians = math.radians(angle_degrees)
    result = math.cos(angle_radians)
    print(f"cos({angle_degrees}°) = {result}")
    return result


@tool
def tangent(angle_degrees: float) -> float:
    """Calculate the tangent of an angle in degrees.

    Args:
        angle_degrees (float): The angle in degrees.

    Returns:
        float: The tangent of the angle.
    """
    angle_radians = math.radians(angle_degrees)
    result = math.tan(angle_radians)
    print(f"tan({angle_degrees}°) = {result}")
    return result
