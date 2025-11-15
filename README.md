# LangGraph Agents

## Table of Contents
* [Overview](#overview)
* [Technical Architecture](#technical-architecture)
* [Installation](#installation)
  * [Using **uv** (recommended)](#using-uv-recommended)
  * [Using `pip` / virtual-env](#using-pip--virtualenv)
* [Quick Start](#quick-start)
* [Troubleshooting](#troubleshooting)
* [Directory Tree](#directory-tree)

## Overview
`langgraph-agents` is a modular AI agent framework built on **LangGraph** and **LLM models**, designed to create and orchestrate intelligent agents for tasks like document writing, research, chart generation, and retrieval-augmented generation (RAG). It's ideal for developers who want:
- **Composable agents** for complex workflows, using ReAct-style reasoning and tool calling
- **Seamless integration** with tools for web search, scraping, document I/O, and more
- **Supervisor-based teams** to manage multi-agent collaboration (e.g., paper writing or research teams)
- **Extensible structure** for adding new agents in the future


This project is actively evolving, with plans to incorporate additional agents for specialized domains.

## Technical Architecture
| Layer              | Purpose                                                                 | File/Directory                  |
|--------------------|-------------------------------------------------------------------------|---------------------------------|
| Agent Classes      | Core ReAct-style agents for tasks like writing, note-taking, and search | `src/core/agents` (e.g., DocWriterAgent, SearchAgent) |
| Tools              | Utilities for web search, scraping, document I/O, and Python execution, etc.  | `src/core/tools/` (e.g., document_io.py, scrape.py) |
| Prompts              | Utilities for prompts  | `core/tools/` (e.g., document_io.py, scrape.py) |
| Supervisors/Teams  | Routers to manage multi-agent workflows and team invocations            | `src/core/supervisor.py` (e.g., SupervisorAgent) |
| Configuration      | LLM setup and API integrations (e.g., Groq, Google Search)              | `infra/llm_clients/groq.py`, `infra/api/google_search.py` |

## Installation
`langgraph-agents` works with **Python 3.10 – 3.12**. It requires dependencies like LangGraph, LangChain, Pydantic, and Groq SDK.

### Using **uv** (recommended)
[`uv`](https://docs.astral.sh/uv/getting-started/features/#projects) is a Rust-powered fast dependency manager and virtual environment tool.
```bash
# 1 - Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# 2 - Set up and activate virtual environment
uv venv .venv
source .venv/bin/activate
# 3 - Lock and install dependencies
uv lock
uv sync
```

### Using `pip` / virtual-env
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **OpenAI users:** Don’t forget to export your OpenAI key before running:
>
> ```bash
> export OPENAI_API_KEY="..."
> ```

> **Groq users:** Don’t forget to export your Groq key before running:
>
> ```bash
> export GROQ_API_KEY="gsk-..."
> ```

## Quick Start
Here's how to instantiate and run a team agent for a paper-writing task.
```python
from infra.llm_clients.groq import get_llm
from core.agents.paper_writing import PaperWritingTeamAgent
from langchain_core.messages import HumanMessage

llm = get_llm()
paper_team = PaperWritingTeamAgent(llm)
query = "Write an outline for a poem about cats and then write the poem to disk."
result = paper_team.graph.invoke({"messages": [HumanMessage(content=query)]})

# ✅ Expected output: A dictionary with messages, including the final poem content
print(result["messages"][-1].content)
```

For a simple individual agent:
```python
from infra.llm_clients.groq import get_llm
from core.agents.web_search import SearchAgent
from langchain_core.messages import HumanMessage

llm = get_llm()
search_agent = SearchAgent(llm)
query = "When is the next FIFA World Cup?"
result = search_agent.graph.invoke({"messages": [HumanMessage(content=query)]})
print(result["messages"][-1].content)
```

## Examples
Examples for all agents are provided in the `examples/` directory. Each example demonstrates how to use a specific agent with realistic scenarios.

### Running Examples
Set your API keys and run any example with:
```bash
export GROQ_API_KEY="gsk-..."
PYTHONPATH=src python3 uv run examples/<agent_name>.py
```

### Available Examples

#### 1. **Document Writer** (`examples/doc_writer.py`)
Generate comprehensive documents using the DocWriterAgent.
```bash
PYTHONPATH=src python3 uv run examples/doc_writer.py
```
**Use case:** Creating detailed documents, reports, or content.

---

#### 2. **Paper Writing Team** (`examples/paper_writing_team.py`)
Collaborate with a team of agents to write poems and related content using PaperWritingTeamAgent.
```bash
PYTHONPATH=src python3 uv run examples/paper_writing_team.py
```
**Use case:** Complex writing tasks requiring multiple perspectives (outlines, poems, documents).

---

#### 3. **Research Team** (`examples/research_team.py`)
Research a topic using the ResearchTeamAgent with web search and scraping capabilities.
```bash
PYTHONPATH=src python3 uv run examples/research_team.py
```
**Use case:** In-depth research with web search and content extraction.

---

#### 4. **Web Search** (`examples/web_search.py`)
Perform web searches using the SearchAgent.
```bash
PYTHONPATH=src python3 uv run examples/web_search.py
```
**Use case:** Finding information on the web, answering current event questions.

---

#### 5. **RAG (Retrieval-Augmented Generation)** (`examples/rag.py`)
Load documents from URLs and prepare them for retrieval-augmented generation using RagAgent.
```bash
PYTHONPATH=src python3 uv run examples/rag.py
```
**Use case:** Question-answering over specific documents or knowledge bases.

---

#### 6. **Note Taker** (`examples/note_taker.py`)
Take structured notes from content using NoteTakerAgent.
```bash
PYTHONPATH=src python3 uv run examples/note_taker.py
```
**Use case:** Extracting key points and organizing information.

---

#### 7. **Chart Generator** (`examples/chart_generator.py`)
Generate charts and visualizations using ChartGeneratorAgent.
```bash
PYTHONPATH=src python3 uv run examples/chart_generator.py
```
**Use case:** Creating visual representations of data and trends.

---

#### 8. **Web Scraper** (`examples/web_scraper.py`)
Scrape and extract content from web pages using WebScraperAgent.
```bash
PYTHONPATH=src python3 uv run examples/web_scraper.py
```
**Use case:** Extracting specific information from websites.

---

### Running All Examples
To run all examples at once:
```bash
for example in examples/*.py; do
    echo "Running $example..."
    PYTHONPATH=src python3 uv run "$example"
    echo "---"
done
```

## Troubleshooting
- **API Key Errors:** Ensure `GROQ_API_KEY` is set in your environment.
- **Dependency Issues:** If using uv, run `uv sync` again. For pip, verify `requirements.txt` includes `langgraph`, `langchain`, and `groq`.
- **Graph Recursion Limits:** If loops exceed defaults, pass `{"recursion_limit": 100}` to `graph.invoke()`.
- **Model Availability:** Confirm Groq models (e.g., LLaMA) are accessible; fallback to OpenAI if needed.
- For more, check console logs or raise an issue on the repo.

## Directory Tree
```text
langgraph-agents/
├── Makefile
├── README.md
├── development.md
├── devtools/
│   └── lint.py
├── examples/
│   ├── __init__.py
│   ├── chart_generator.py
│   ├── doc_writer.py
│   ├── note_taker.py
│   ├── paper_writing_team.py
│   ├── rag.py
│   ├── research_team.py
│   ├── web_scraper.py
│   └── web_search.py
├── installation.md
├── pyproject.toml
├── src/
│   ├── core/
│   │   ├── supervisor.py
│   │   ├── agents/
│   │   │   ├── base.py
│   │   │   ├── chart_generator.py
│   │   │   ├── doc_writer.py
│   │   │   ├── note_taker.py
│   │   │   ├── paper_writing.py
│   │   │   ├── rag.py
│   │   │   ├── research_team.py
│   │   │   ├── supervisor.py
│   │   │   ├── web_scraper.py
│   │   │   └── web_search.py
│   ├── prompts/
│   │   └── rag.py
│   ├── tools/
│   │   ├── document_io.py
│   │   ├── math.py
│   │   ├── python_repl.py
│   │   └── scrape.py
│   └── utils/
│       └── env.py
│   ├── infra/
│   │   ├── api/
│   │   │   └── google_search.py
│   │   └── llm_clients/
│   │       └── groq.py
│   └── services/
├── tests/
│   ├── core/
│   │   ├── agents/
│   │   │   ├── test_base.py
│   │   │   ├── test_chart_generator.py
│   │   │   ├── test_doc_writer.py
│   │   │   └── test_note_taker.py
│   │   └── test_supervisor.py
├── uv.lock
└── workspace/
```
