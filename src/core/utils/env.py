from dotenv import load_dotenv
import os

load_dotenv()

def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)