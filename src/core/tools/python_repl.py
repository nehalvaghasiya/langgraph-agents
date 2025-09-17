from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL


# Instance of PythonREPL
repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.

    Executes Python code using a persistent REPL environment. The code execution
    maintains state between calls, allowing for iterative development and variable
    persistence across multiple executions.

    Args:
        code (Annotated[str]): The Python code to execute. Should be valid Python
                              syntax. Use print() statements to display output
                              that will be visible to the user.

    Returns:
        str: A formatted string containing the execution result. On success, returns
             the executed code block and stdout output. On failure, returns an error
             message with exception details.

    Note:
        - The REPL environment persists state between function calls
        - Variables and imports remain available for subsequent executions
        - Use print() statements to make output visible to the user
        - Execution errors are caught and returned as formatted error messages
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"