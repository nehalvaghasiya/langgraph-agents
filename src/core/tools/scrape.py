from collections.abc import Callable

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool


def create_scraper_tool(requests_kwargs: dict | None = None) -> Callable:
    """Create a web scraping tool with configurable requests options.

    Args:
        requests_kwargs (dict | None): Optional keyword arguments to pass to the
                                     underlying requests session (e.g., headers,
                                     verify, timeout).

    Returns:
        Callable: The configured tool function.
    """

    @tool
    def scrape_webpages(urls: list[str]) -> str:
        """Use requests and bs4 to scrape the provided web pages for detailed information.

        Scrapes content from multiple web pages using LangChain's WebBaseLoader, which
        internally uses the requests library and BeautifulSoup4 for HTML parsing.
        The function loads each URL, extracts the page content and metadata, then
        formats the results into structured document blocks.

        Args:
            urls (List[str]): A list of web page URLs to scrape. Each URL should be
                             a valid HTTP or HTTPS web address.

        Returns:
            str: A formatted string containing all scraped documents. Each document
                 is wrapped in XML-like tags with the format:
                 '<Document name="title">page_content</Document>'
                 Multiple documents are separated by double newlines.

        Note:
            - The function relies on LangChain's WebBaseLoader for the actual scraping
            - Document titles are extracted from metadata when available
            - If a page cannot be loaded or parsed, it may be excluded from results
            - The returned format uses XML-like document tags for easy parsing
        """
        loader = WebBaseLoader(urls, requests_kwargs=requests_kwargs)
        docs = loader.load()
        return "\n\n".join(
            [
                f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
                for doc in docs
            ]
        )

    return scrape_webpages
