from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool


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
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )
