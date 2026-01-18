"""Summarization tools including chunking and document analysis utilities.

This module provides tools for the SummarizationAgent including:
- SmartChunker: Strategy-aware text chunking
- Token counting and document metadata extraction
- LangChain @tool decorated functions for agent use
"""

from typing import Annotated, Literal

from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Strategy type literals
SummarizationStrategy = Literal["MAP_REDUCE", "REFINE", "HIERARCHICAL"]
ContentType = Literal["narrative", "informational", "massive_dataset"]

# Constants for strategy selection
TOKEN_THRESHOLD_MASSIVE = 50000  # Above this -> HIERARCHICAL
TOKEN_THRESHOLD_LONG = 2000  # Above this (but below massive) -> contextual selection
MAX_REVISION_COUNT = 2
PREVIEW_TOKEN_COUNT = 1000

# Chunking constants per strategy
CHUNK_CONFIG = {
    "MAP_REDUCE": {"chunk_size": 1500, "chunk_overlap": 100},
    "REFINE": {"chunk_size": 4000, "chunk_overlap": 400},
    "HIERARCHICAL": {"chunk_size": 800, "chunk_overlap": 50},
}


def count_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple character-based estimation. For production use,
    consider using tiktoken or the model's actual tokenizer.

    Args:
        text: The text to count tokens for.
        chars_per_token: Average characters per token (default 4.0).

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return int(len(text) / chars_per_token)


def get_doc_metadata(text: str) -> dict:
    """Extract document metadata for analysis.

    Args:
        text: The document text.

    Returns:
        Dictionary with document statistics including char_count,
        estimated_tokens, line_count, and paragraph_count.
    """
    if not text:
        return {
            "char_count": 0,
            "estimated_tokens": 0,
            "line_count": 0,
            "paragraph_count": 0,
        }

    lines = text.split("\n")
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    return {
        "char_count": len(text),
        "estimated_tokens": count_tokens(text),
        "line_count": len(lines),
        "paragraph_count": len(paragraphs),
    }


class SmartChunker:
    """Strategy-aware text chunker.

    Provides different chunking configurations based on the selected
    summarization strategy:
    - MAP_REDUCE: Standard chunks for parallel processing
    - REFINE: Larger chunks with more overlap for sequential processing
    - HIERARCHICAL: Smaller granular chunks for tree-based processing

    Attributes:
        separators: List of separators for recursive splitting.

    Example:
        >>> chunker = SmartChunker()
        >>> chunks = chunker.chunk("Long text...", "MAP_REDUCE")
        >>> print(f"Created {len(chunks)} chunks")
    """

    def __init__(
        self,
        separators: list[str] | None = None,
    ):
        """Initialize the SmartChunker.

        Args:
            separators: Custom separators for text splitting.
                       Defaults to [paragraph, newline, sentence, word, char].
        """
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(
        self,
        text: str,
        strategy: SummarizationStrategy,
        custom_chunk_size: int | None = None,
        custom_overlap: int | None = None,
    ) -> list[str]:
        """Split text into chunks based on strategy.

        Args:
            text: The text to split.
            strategy: The summarization strategy determining chunk config.
            custom_chunk_size: Override the default chunk size.
            custom_overlap: Override the default overlap.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        config = CHUNK_CONFIG.get(strategy, CHUNK_CONFIG["MAP_REDUCE"])
        chunk_size = custom_chunk_size or config["chunk_size"]
        chunk_overlap = custom_overlap or config["chunk_overlap"]

        # Ensure overlap is less than chunk size
        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size // 4

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

        return splitter.split_text(text)

    def chunk_for_hierarchical(
        self,
        text: str,
        group_size: int = 5,
    ) -> list[list[str]]:
        """Create hierarchical chunk groups for tree-based summarization.

        First creates granular leaf chunks, then groups them for
        hierarchical processing.

        Args:
            text: The text to split.
            group_size: Number of chunks per group at each level.

        Returns:
            List of chunk groups (each group is a list of strings).
        """
        # Get leaf-level chunks
        leaf_chunks = self.chunk(text, "HIERARCHICAL")

        if not leaf_chunks:
            return []

        # Group chunks for hierarchical processing
        groups = []
        for i in range(0, len(leaf_chunks), group_size):
            group = leaf_chunks[i : i + group_size]
            groups.append(group)

        return groups


@tool
def analyze_document(
    text: Annotated[str, "The document text to analyze"],
) -> str:
    """Analyze a document and return metadata statistics.

    This tool extracts key statistics about a document including
    character count, estimated token count, line count, and paragraph count.
    Useful for understanding document size before summarization.

    Args:
        text: The document text to analyze.

    Returns:
        A formatted string with document statistics.
    """
    if not text or not text.strip():
        return "Error: Empty or whitespace-only text provided."

    metadata = get_doc_metadata(text)

    return f"""Document Analysis:
- Character count: {metadata["char_count"]:,}
- Estimated tokens: {metadata["estimated_tokens"]:,}
- Line count: {metadata["line_count"]:,}
- Paragraph count: {metadata["paragraph_count"]:,}

Recommended strategy:
- Under {TOKEN_THRESHOLD_LONG:,} tokens: Direct summarization
- {TOKEN_THRESHOLD_LONG:,} - {TOKEN_THRESHOLD_MASSIVE:,} tokens: MAP_REDUCE or REFINE
- Over {TOKEN_THRESHOLD_MASSIVE:,} tokens: HIERARCHICAL"""


@tool
def chunk_text(
    text: Annotated[str, "The text to split into chunks"],
    strategy: Annotated[
        str,
        "Chunking strategy: 'MAP_REDUCE' (standard), 'REFINE' (larger), or 'HIERARCHICAL' (smaller)",
    ] = "MAP_REDUCE",
) -> str:
    """Split text into chunks based on a summarization strategy.

    Different strategies use different chunk sizes:
    - MAP_REDUCE: ~1500 chars - good for parallel processing
    - REFINE: ~4000 chars - good for sequential refinement
    - HIERARCHICAL: ~800 chars - good for tree-based summarization

    Args:
        text: The text to split into chunks.
        strategy: The chunking strategy to use.

    Returns:
        A formatted string showing chunk count and preview of each chunk.
    """
    if not text or not text.strip():
        return "Error: Empty or whitespace-only text provided."

    valid_strategies = ["MAP_REDUCE", "REFINE", "HIERARCHICAL"]
    strategy_upper = strategy.upper()
    if strategy_upper not in valid_strategies:
        return f"Error: Invalid strategy '{strategy}'. Use one of: {valid_strategies}"

    chunker = SmartChunker()
    chunks = chunker.chunk(text, strategy_upper)  # type: ignore

    if not chunks:
        return "Text is too short to chunk."

    result = f"Split into {len(chunks)} chunks using {strategy_upper} strategy:\n\n"
    for i, chunk in enumerate(chunks, 1):
        preview = chunk[:150].replace("\n", " ")
        if len(chunk) > 150:
            preview += "..."
        result += f"Chunk {i} ({len(chunk)} chars): {preview}\n\n"

    return result


@tool
def estimate_tokens(
    text: Annotated[str, "The text to estimate token count for"],
) -> str:
    """Estimate the number of tokens in a text.

    Uses a character-based estimation (approximately 4 characters per token).
    This is useful for determining if a document needs chunking or
    which summarization strategy to use.

    Args:
        text: The text to estimate tokens for.

    Returns:
        A string with the estimated token count and recommendations.
    """
    if not text:
        return "Token count: 0"

    tokens = count_tokens(text)

    recommendation = ""
    if tokens < TOKEN_THRESHOLD_LONG:
        recommendation = "Short document - direct summarization recommended."
    elif tokens < TOKEN_THRESHOLD_MASSIVE:
        recommendation = "Long document - MAP_REDUCE or REFINE strategy recommended."
    else:
        recommendation = "Massive document - HIERARCHICAL strategy required."

    return f"""Estimated tokens: {tokens:,}

{recommendation}

Thresholds:
- Short (direct): < {TOKEN_THRESHOLD_LONG:,} tokens
- Long (chunked): {TOKEN_THRESHOLD_LONG:,} - {TOKEN_THRESHOLD_MASSIVE:,} tokens
- Massive (hierarchical): > {TOKEN_THRESHOLD_MASSIVE:,} tokens"""
