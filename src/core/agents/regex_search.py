"""Regex Search Agent for pattern matching and text extraction."""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent
from core.tools.regex import (
    compile_regex_pattern,
    extract_pattern_matches,
    replace_pattern_in_file,
    search_files_by_pattern,
    search_text_in_file,
    validate_and_explain_pattern,
)


class RegexSearchAgent(BaseAgent):
    """Agent for regex-based searching and pattern matching.

    This agent specializes in searching for files by pattern, extracting text
    matching specific patterns from files and documents, and validating regex
    patterns for correctness.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[Callable] | None = None,
        system: str = "",
    ):
        """Initialize RegexSearchAgent.

        Args:
            model (BaseChatModel): The language model to use.
            tools (list[Callable] | None): Optional list of tools (uses defaults if None).
            system (str): Optional system message override.
        """
        system = (
            system
            or """You are a regex pattern expert and search specialist. Your role is to:
1. PLAN: Understand the search requirements and design appropriate regex patterns
2. REASON: Validate patterns and explain what they match
3. ACT: Execute searches on files and documents using regex patterns
4. OBSERVE: Return results and verify they match expectations

For each search task:
- Use validate_and_explain_pattern to understand and verify patterns
- Use compile_regex_pattern to test pattern validity
- Use search_files_by_pattern to find files matching patterns
- Use search_text_in_file to find content within specific files
- Use extract_pattern_matches to extract data from text
- Use replace_pattern_in_file for pattern-based replacements (with dry-run first)

Be precise with regex patterns and provide clear explanations of what each pattern does."""
        )
        tools = [
            validate_and_explain_pattern,
            compile_regex_pattern,
            search_files_by_pattern,
            search_text_in_file,
            extract_pattern_matches,
            replace_pattern_in_file,
        ]
        super().__init__(model, tools, system)
