from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun

from core.utils.env import get_env


def get_google_search():
    """Get Instance of Google Search API wrapper.

    Creates and configures a Google Search API wrapper instance using environment
    variables for authentication. The function initializes a GoogleSearchAPIWrapper
    with the required API credentials and returns a GoogleSearchRun instance
    ready for performing web searches.

    Returns:
        GoogleSearchRun: A configured Google Search API instance that can be used
                        to perform web searches through Google's Custom Search API.

    Raises:
        KeyError: If required environment variables (SERPAPI_API_KEY or GOOGLE_CSE_ID)
                 are not found.
        ValueError: If the provided API key or CSE ID are invalid.

    Note:
        - Requires SERPAPI_API_KEY environment variable with a valid SerpAPI key
        - Requires GOOGLE_CSE_ID environment variable with a valid Google Custom Search Engine ID
        - The API wrapper uses Google's Custom Search JSON API for web searches
        - Ensure proper API quotas and billing are configured for the Google account

    Example:
        >>> search_tool = get_google_search()
        >>> results = search_tool.run("LangChain documentation")
    """
    wrapper = GoogleSearchAPIWrapper(
        google_api_key=get_env("SERPAPI_API_KEY"), google_cse_id=get_env("GOOGLE_CSE_ID")
    )
    return GoogleSearchRun(api_wrapper=wrapper)
