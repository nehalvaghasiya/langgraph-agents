"""Example: Using RagAgent for retrieval-augmented generation."""

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from core.agents.rag import RagAgent
from infra.llm_clients.openai import get_llm


def main():
    """Example of using RagAgent with web-based documents."""
    logger.info("Initializing RagAgent example")
    
    # Get LLM instance
    llm = get_llm()
    logger.debug("LLM instance created")

    # Load documents from URLs
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]
    
    logger.info(f"Loading documents from {len(urls)} URLs")
    
    try:
        # Load and flatten documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        
        logger.info(f"Loaded {len(docs_list)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs_list)
        
        logger.info(f"Split documents into {len(splits)} chunks")

        # Create RAG agent with configuration
        rag_agent = RagAgent(
            model=llm, 
            doc_splits=splits,
            search_kwargs={"k": 3}
        )
        logger.debug("RagAgent initialized")
        
        # Print summary
        print("RAG AGENT EXAMPLE")
        print(f"Loaded {len(docs_list)} documents from {len(urls)} URLs")
        print(f"Created {len(splits)} document chunks for RAG")
        
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
