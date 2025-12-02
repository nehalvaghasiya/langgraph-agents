"""Regex-based searching tools for files and documents.

This module provides a comprehensive set of regex tools for pattern matching,
file searching, text extraction, and validation. Tools include pattern
compilation, file system searches, content extraction, and safe file
modifications with preview capability.
"""

import os
import re
from typing import Annotated

from langchain_core.tools import tool


@tool
def compile_regex_pattern(
    pattern: Annotated[str, "The regex pattern to compile (e.g., r'.*\\.py$')"],
    flags: Annotated[
        str, "Optional regex flags: IGNORECASE, MULTILINE, DOTALL, VERBOSE (comma-separated)"
    ] = "",
) -> str:
    """Compile and validate a regex pattern.

    This tool compiles a regex pattern and validates it. Returns a summary of
    the compiled pattern with details about what it matches.

    Args:
        pattern (str): The regex pattern string to compile.
        flags (str): Optional flags to apply (e.g., 'IGNORECASE,MULTILINE').

    Returns:
        str: Status message indicating if pattern compiled successfully with
            pattern details.
    """
    try:
        # Parse flags
        flag_value = 0
        if flags:
            flag_names = [f.strip().upper() for f in flags.split(",")]
            flag_map = {
                "IGNORECASE": re.IGNORECASE,
                "MULTILINE": re.MULTILINE,
                "DOTALL": re.DOTALL,
                "VERBOSE": re.VERBOSE,
            }
            for flag_name in flag_names:
                if flag_name in flag_map:
                    flag_value |= flag_map[flag_name]

        # Compile the pattern
        re.compile(pattern, flag_value)

        return f"""✓ Regex pattern compiled successfully!
Pattern: {pattern}
Flags: {flags if flags else "None"}
Description: Pattern will match strings according to the provided regex rules."""

    except re.error as e:
        return f"✗ Regex compilation failed: {str(e)}"
    except Exception as e:
        return f"✗ Error: {str(e)}"


@tool
def search_files_by_pattern(
    pattern: Annotated[str, "The regex pattern to match filenames (e.g., r'.*\\.py$')"],
    search_path: Annotated[
        str, "The directory path to search in (default: current directory)"
    ] = ".",
    recursive: Annotated[bool, "Whether to search recursively in subdirectories"] = True,
    max_results: Annotated[int, "Maximum number of results to return (default: 100)"] = 100,
) -> str:
    """Search for files matching a regex pattern.

    Searches the filesystem starting from the specified path and returns all
    files matching the provided regex pattern. Useful for finding files by
    extension, naming pattern, or path structure.

    Args:
        pattern (str): The regex pattern to match against file paths.
        search_path (str): Directory to start searching from (uses absolute path).
        recursive (bool): If True, searches all subdirectories.
        max_results (int): Maximum number of results to return.

    Returns:
        str: Formatted list of matching file paths with count.
    """
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(search_path)

        if not os.path.exists(abs_path):
            return f"✗ Search path does not exist: {abs_path}"

        # Compile regex pattern
        compiled_pattern = re.compile(pattern)
        matching_files = []

        if recursive:
            # Walk through all directories
            for root, _dirs, files in os.walk(abs_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Match against the full path
                    if compiled_pattern.search(file_path):
                        matching_files.append(file_path)
                        if len(matching_files) >= max_results:
                            break
                if len(matching_files) >= max_results:
                    break
        else:
            # Search only in the specified directory
            if os.path.isdir(abs_path):
                for file in os.listdir(abs_path):
                    file_path = os.path.join(abs_path, file)
                    if os.path.isfile(file_path):
                        if compiled_pattern.search(file_path):
                            matching_files.append(file_path)
                            if len(matching_files) >= max_results:
                                break

        if not matching_files:
            return f"No files found matching pattern: {pattern}"

        results = f"Found {len(matching_files)} file(s) matching pattern: {pattern}\n"
        results += "=" * 70 + "\n"
        for i, file_path in enumerate(matching_files, 1):
            results += f"{i}. {file_path}\n"

        return results

    except re.error as e:
        return f"✗ Invalid regex pattern: {str(e)}"
    except Exception as e:
        return f"✗ Error during file search: {str(e)}"


@tool
def search_text_in_file(
    file_path: Annotated[str, "Path to the file to search in"],
    pattern: Annotated[str, "The regex pattern to search for"],
    context_lines: Annotated[
        int, "Number of context lines to show before and after match (default: 2)"
    ] = 2,
) -> str:
    """Search for regex pattern matches within a file.

    Finds all occurrences of a regex pattern in a file and returns the matches
    with context. Useful for finding specific content within documents or code
    files.

    Args:
        file_path (str): Path to the file to search in.
        pattern (str): The regex pattern to search for.
        context_lines (int): Number of lines to show around each match.

    Returns:
        str: Formatted results showing matches with context.
    """
    try:
        abs_path = os.path.abspath(file_path)

        if not os.path.exists(abs_path):
            return f"✗ File not found: {abs_path}"

        if not os.path.isfile(abs_path):
            return f"✗ Path is not a file: {abs_path}"

        # Read file
        with open(abs_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # Compile pattern
        compiled_pattern = re.compile(pattern)

        # Find matches
        matches = []
        for line_num, line in enumerate(lines, 1):
            if compiled_pattern.search(line):
                matches.append((line_num, line.rstrip()))

        if not matches:
            return f"No matches found for pattern: {pattern}\nFile: {abs_path}"

        results = f"Found {len(matches)} match(es) for pattern: {pattern}\n"
        results += f"File: {abs_path}\n"
        results += "=" * 70 + "\n"

        for match_num, (line_num, line) in enumerate(matches, 1):
            results += f"\nMatch #{match_num} (Line {line_num}):\n"
            results += "-" * 70 + "\n"

            # Add context lines before
            start = max(0, line_num - context_lines - 1)
            for i in range(start, line_num - 1):
                results += f"  {i + 1:>5} | {lines[i].rstrip()}\n"

            # Highlight the matching line
            results += f"→ {line_num:>5} | {line}\n"

            # Add context lines after
            end = min(len(lines), line_num + context_lines)
            for i in range(line_num, end):
                results += f"  {i + 1:>5} | {lines[i].rstrip()}\n"

        return results

    except re.error as e:
        return f"✗ Invalid regex pattern: {str(e)}"
    except Exception as e:
        return f"✗ Error during file search: {str(e)}"


@tool
def extract_pattern_matches(
    text: Annotated[str, "The text to search in"],
    pattern: Annotated[str, "The regex pattern to extract"],
    group_number: Annotated[
        int, "Which capture group to extract (0 for full match, default: 0)"
    ] = 0,
) -> str:
    """Extract all matches of a regex pattern from text.

    Finds all occurrences of a pattern in provided text and extracts them.
    Useful for extracting specific data like emails, URLs, function names, etc.

    Args:
        text (str): The text to search in.
        pattern (str): The regex pattern to extract matches from.
        group_number (int): Which capture group to return (0 for entire match).

    Returns:
        str: Formatted list of extracted matches.
    """
    try:
        # Compile pattern
        compiled_pattern = re.compile(pattern)

        # Find all matches
        if group_number == 0:
            matches = compiled_pattern.findall(text)
        else:
            all_matches = compiled_pattern.finditer(text)
            matches = []
            for match in all_matches:
                if group_number <= len(match.groups()):
                    matches.append(match.group(group_number))

        if not matches:
            return f"No matches found for pattern: {pattern}"

        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                unique_matches.append(match)

        results = f"Extracted {len(unique_matches)} unique match(es) from {len(matches)} total match(es)\n"
        results += f"Pattern: {pattern}\n"
        results += "=" * 70 + "\n"

        for i, match in enumerate(unique_matches, 1):
            results += f"{i:>3}. {match}\n"

        return results

    except re.error as e:
        return f"✗ Invalid regex pattern: {str(e)}"
    except Exception as e:
        return f"✗ Error during extraction: {str(e)}"


@tool
def replace_pattern_in_file(
    file_path: Annotated[str, "Path to the file to modify"],
    pattern: Annotated[str, "The regex pattern to find"],
    replacement: Annotated[str, "The replacement string (can use groups: \\1, \\2, etc.)"],
    dry_run: Annotated[bool, "If True, show changes without modifying file"] = True,
) -> str:
    """Replace regex pattern matches in a file.

    Finds all occurrences of a regex pattern in a file and replaces them.
    By default runs in dry-run mode to preview changes before applying.

    Args:
        file_path (str): Path to the file to modify.
        pattern (str): The regex pattern to find.
        replacement (str): The replacement string (can use groups: \\1, \\2, etc.).
        dry_run (bool): If True, preview changes without modifying file.

    Returns:
        str: Summary of changes made or would be made.
    """
    try:
        abs_path = os.path.abspath(file_path)

        if not os.path.exists(abs_path):
            return f"✗ File not found: {abs_path}"

        if not os.path.isfile(abs_path):
            return f"✗ Path is not a file: {abs_path}"

        # Read file
        with open(abs_path, encoding="utf-8", errors="ignore") as f:
            original_content = f.read()

        # Compile pattern and perform replacement
        compiled_pattern = re.compile(pattern)
        new_content = compiled_pattern.sub(replacement, original_content)

        if original_content == new_content:
            return f"No matches found for pattern: {pattern}"

        # Count changes
        change_count = len(compiled_pattern.findall(original_content))

        if dry_run:
            results = f"[DRY RUN] Would replace {change_count} occurrence(s)\n"
            results += f"File: {abs_path}\n"
            results += "=" * 70 + "\n"
            results += f"Pattern: {pattern}\n"
            results += f"Replacement: {replacement}\n"
            results += "=" * 70 + "\n"
            results += "\nPreview of changes:\n"
            results += "-" * 70 + "\n"

            # Show before/after for first few matches
            original_lines = original_content.split("\n")
            new_lines = new_content.split("\n")

            for _, (orig, new) in enumerate(zip(original_lines[:20], new_lines[:20], strict=False)):
                if orig != new:
                    results += f"- {orig}\n"
                    results += f"+ {new}\n"

            results += "\nTo apply changes, set dry_run=False"
            return results
        else:
            # Write changes to file
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return f"✓ Successfully replaced {change_count} occurrence(s) in file: {abs_path}"

    except re.error as e:
        return f"✗ Invalid regex pattern: {str(e)}"
    except Exception as e:
        return f"✗ Error during replacement: {str(e)}"


@tool
def validate_and_explain_pattern(
    pattern: Annotated[str, "The regex pattern to explain"],
) -> str:
    """Validate a regex pattern and provide a detailed explanation.

    Validates a regex pattern and provides a human-readable breakdown of what
    the pattern matches and common examples.

    Args:
        pattern (str): The regex pattern to validate and explain.

    Returns:
        str: Detailed explanation of the pattern.
    """
    try:
        # Validate compilation
        compiled_pattern = re.compile(pattern)

        # Analyze the pattern
        explanation = "Regex Pattern Analysis\n"
        explanation += "=" * 70 + "\n"
        explanation += f"Pattern: {pattern}\n"
        explanation += "-" * 70 + "\n"

        # Provide common examples based on pattern
        examples = []
        if r"\\.py" in pattern or ".py" in pattern:
            examples = ["file.py", "script.py", "test.py"]
            explanation += "Pattern Type: Python files\n"
        elif r"\\.(txt|doc)" in pattern or ".txt" in pattern:
            examples = ["document.txt", "readme.txt", "notes.txt"]
            explanation += "Pattern Type: Text files\n"
        elif r"\\d+" in pattern:
            examples = ["123", "0", "999999"]
            explanation += "Pattern Type: Numbers\n"
        elif r"\\w+" in pattern:
            examples = ["word", "test", "hello123"]
            explanation += "Pattern Type: Words/Alphanumeric\n"
        elif r"\\S+" in pattern:
            examples = ["text", "123", "special!chars"]
            explanation += "Pattern Type: Non-whitespace\n"
        elif "@" in pattern:
            examples = ["user@example.com", "test@domain.org"]
            explanation += "Pattern Type: Email-like\n"
        else:
            explanation += "Pattern Type: Custom pattern\n"

        explanation += "\nCommon Components Found:\n"
        if ".*" in pattern:
            explanation += "  • .* = Match any character (zero or more times)\n"
        if ".+" in pattern:
            explanation += "  • .+ = Match any character (one or more times)\n"
        if r"\\d" in pattern:
            explanation += "  • \\d = Match digits (0-9)\n"
        if r"\\w" in pattern:
            explanation += "  • \\w = Match word characters (a-z, A-Z, 0-9, _)\n"
        if r"\\s" in pattern:
            explanation += "  • \\s = Match whitespace\n"
        if "[" in pattern and "]" in pattern:
            explanation += "  • [...] = Character class (match any character inside)\n"
        if "(" in pattern and ")" in pattern:
            explanation += "  • (...) = Capture group\n"
        if "|" in pattern:
            explanation += "  • | = Alternation (OR operator)\n"
        if "$" in pattern:
            explanation += "  • $ = End of string anchor\n"
        if "^" in pattern:
            explanation += "  • ^ = Start of string anchor\n"

        explanation += "\n" + "-" * 70 + "\n"
        explanation += "✓ Pattern is valid and ready to use!\n"

        if examples:
            explanation += "\nExample matches:\n"
            for example in examples:
                if compiled_pattern.search(example):
                    explanation += f"  ✓ '{example}' matches\n"
                else:
                    explanation += f"  ✗ '{example}' does not match\n"

        return explanation

    except re.error as e:
        return f"✗ Invalid regex pattern: {str(e)}\n\nTip: Ensure special characters are properly escaped."
    except Exception as e:
        return f"✗ Error: {str(e)}"
