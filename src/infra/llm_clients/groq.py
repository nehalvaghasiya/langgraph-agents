from langchain_openai import ChatOpenAI
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

    Note:
        - Requires OPENAI_API_KEY environment variable with a valid Groq API key
        - Uses Groq's OpenAI-compatible endpoint (https://api.groq.com/openai/v1)
        - Configured with temperature=0.7 for balanced creativity and consistency
        - Limited to 4096 max tokens per response
        - Model: moonshotai/kimi-k2-instruct for instruction-following capabilities

    Example:
        >>> llm = get_llm()
        >>> response = llm.invoke("What is machine learning?")
        >>> print(response.content)
    """
    return ChatOpenAI(
        model_name="moonshotai/kimi-k2-instruct-0905",
        temperature=0.7,
        max_tokens=4096,
        openai_api_key=get_env("OPENAI_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
    )