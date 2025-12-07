from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from core.utils.env import get_env


def get_llm() -> ChatOpenAI:
    """Return instance of langchain ChatOpenAI.

    Creates and configures a ChatOpenAI instance. The function sets up the model
    with specific parameters for temperature, token limits, and API configuration.

    Returns:
        ChatOpenAI: A configured ChatOpenAI instance ready for text generation
                   and conversation. The instance uses the LLM
                   via OpenAI-compatible API endpoint.

    Raises:
        KeyError: If the OPENAI_API_KEY environment variable is not found.
        ValueError: If the provided API key is invalid or malformed.
        ConnectionError: If unable to connect to the Groq API endpoint.

    Example:
        >>> llm = get_llm()
        >>> response = llm.invoke("What is machine learning?")
        >>> print(response.content)
    """
    # Read model and base URL from environment variables (OpenAI-compatible names)
    model = get_env("OPENAI_MODEL", "moonshotai/kimi-k2-instruct-0905")
    base_url = get_env("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

    return ChatOpenAI(
        model=model,
        temperature=0.7,
        max_completion_tokens=4096,
        api_key=SecretStr(get_env("OPENAI_API_KEY")),  # Wrap in SecretStr
        base_url=base_url,
    )
