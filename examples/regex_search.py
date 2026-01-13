"""Example usage of Regex Search Tools.

Demonstrates regex pattern validation, file searching, and text extraction.
"""

from core.tools.regex import (
    compile_regex_pattern,
    search_files_by_pattern,
    search_text_in_file,
    extract_pattern_matches,
    validate_and_explain_pattern,
)


def main():
    """Run regex search examples."""
    print("REGEX SEARCH TOOL EXAMPLES")

    # Example 1: Validate and explain a regex pattern
    print("Example 1: Validate and Explain a Regex Pattern")
    pattern = r".*\.py$"
    result = validate_and_explain_pattern.invoke({"pattern": pattern})
    print(result)

    # Example 2: Compile a regex pattern
    print("Example 2: Compile a Regex Pattern")
    pattern = r"^test_.*\.py$"
    result = compile_regex_pattern.invoke({"pattern": pattern})
    print(result)

    # Example 3: Search for Python files
    print("Example 3: Search for Python Files in src directory")
    pattern = r".*\.py$"
    result = search_files_by_pattern.invoke(
        {"pattern": pattern, "search_path": "src", "max_results": 20}
    )
    print(result)

    # Example 4: Search for test files
    print("Example 4: Search for Test Files")
    pattern = r"^test_.*\.py$"
    result = search_files_by_pattern.invoke(
        {"pattern": pattern, "search_path": "tests", "max_results": 15}
    )
    print(result)

    # Example 5: Extract patterns from text
    print("Example 5: Extract Python Function Definitions")
    sample_code = """
def hello_world():
    pass

def calculate_sum(a, b):
    return a + b

async def fetch_data():
    pass

class MyClass:
    def method(self):
        pass
"""
    pattern = r"def (\w+)\("
    result = extract_pattern_matches.invoke(
        {"text": sample_code, "pattern": pattern, "group_number": 1}
    )
    print(result)

    # Example 6: Extract email-like patterns
    print("Example 6: Extract Email Addresses")
    sample_text = """
Contact us at:
- support@example.com
- info@company.org
- admin@test.co.uk
For urgent matters: urgent@example.com
"""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    result = extract_pattern_matches.invoke({"text": sample_text, "pattern": pattern})
    print(result)

    # Example 7: Extract numbers and digits
    print("Example 7: Extract Numbers from Text")
    sample_text = """
Prices: $10.99, $25.00, $100.50
Quantities: 5 items, 12 boxes, 999 units
Phone: 555-123-4567
"""
    pattern = r"\d+"
    result = extract_pattern_matches.invoke({"text": sample_text, "pattern": pattern})
    print(result)

    # Example 8: Search for patterns in a file
    print("Example 8: Search for Imports in a Python File")
    try:
        pattern = r"^from .* import|^import .*"
        result = search_text_in_file.invoke(
            {
                "file_path": "src/core/tools/regex.py",
                "pattern": pattern,
                "context_lines": 1,
            }
        )
        # Only show first part since file is large
        lines = result.split("\n")
        print("\n".join(lines[:30]))
        print("\n... (truncated for display)\n")
    except Exception as e:
        print(f"Error: {e}")

    # Example 9: Validate complex patterns
    print("Example 9: Validate Complex Patterns")

    patterns = [
        (r"^\d{3}-\d{3}-\d{4}$", "US Phone Number"),
        (r"^[A-Z][a-z]*$", "Capitalized Word"),
        (r"\b[A-Z]{2,}\b", "Uppercase Words"),
        (r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", "IPv4 Address"),
    ]

    for pattern, description in patterns:
        print(f"\nPattern: {description}")
        print(f"Regex: {pattern}")
        result = compile_regex_pattern.invoke({"pattern": pattern})
        # Extract just the first line
        first_line = result.split("\n")[0]
        print(f"Status: {first_line}")

    print("Examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
