from langchain_openai import ChatOpenAI
from core.utils.env import get_env

def get_llm():
    return ChatOpenAI(
        model_name="moonshotai/kimi-k2-instruct",
        temperature=0.7,
        max_tokens=4096,
        openai_api_key=get_env("OPENAI_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
    )