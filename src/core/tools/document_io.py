from langchain_core.tools import tool
from typing import Annotated, Dict, List, Optional
from pathlib import Path

WORKING_DIRECTORY = Path("../workspace")
WORKING_DIRECTORY.mkdir(parents=True, exist_ok=True)

@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline.

    Writes a numbered list of main points or sections to a text file in the working directory.

    Args:
        points (Annotated[List[str]]): List of main points or sections to be included in the outline.
        file_name (Annotated[str]): File path to save the outline inside the workspace.

    Returns:
        Annotated[str]: Path of the saved outline file.
    """
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document.

    Reads the content of a text file from the working directory,
    optionally retrieving only a subset of lines specified by start and end.

    Args:
        file_name (Annotated[str]): File path to read the document from inside the workspace.
        start (Annotated[Optional[int]], optional): The start line number (0-based). Defaults to None (start from first line).
        end (Annotated[Optional[int]], optional): The end line number (0-based, exclusive). Defaults to None (to the end of file).

    Returns:
        str: The content of the specified document or section as a string.
    """
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    full_path = WORKING_DIRECTORY / file_name
    with full_path.open("w") as file:
        file.write(content)
    print(f"[INFO] Document saved to: {full_path}")  # For debug visibility
    return f"Document saved to {full_path}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""

    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"