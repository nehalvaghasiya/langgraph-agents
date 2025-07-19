from dotenv import load_dotenv
import os

load_dotenv()

def get_env(key: str, default: str = "") -> str:
    """Get Env Variables.

    Args:
        key (str): Name of env.
        default (str, optional): Key. Defaults to "".

    Returns:
        str: Return value of Variable.
    """
    return os.getenv(key, default)