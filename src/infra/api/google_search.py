from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun
from core.utils.env import get_env

def get_google_search():
    """Get Instance of Google Search."""
    wrapper = GoogleSearchAPIWrapper(
        google_api_key=get_env("SERPAPI_API_KEY"),
        google_cse_id=get_env("GOOGLE_CSE_ID")
    )
    return GoogleSearchRun(api_wrapper=wrapper)