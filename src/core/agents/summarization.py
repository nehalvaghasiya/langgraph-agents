"""Summarization Agent with dynamic strategy selection.

This module implements a smart summarization agent that dynamically selects
between MAP_REDUCE, REFINE, and HIERARCHICAL strategies based on document
analysis. It uses a Plan-Execute-Reflect architecture with conditional routing.
"""

import json
import logging
import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.prompts.summarization import SummarizationPrompts
from core.tools.summarization import (
    MAX_REVISION_COUNT,
    PREVIEW_TOKEN_COUNT,
    TOKEN_THRESHOLD_LONG,
    TOKEN_THRESHOLD_MASSIVE,
    SmartChunker,
    SummarizationStrategy,
    get_doc_metadata,
)

# Configure logging
logger = logging.getLogger(__name__)


class DocMetadata(TypedDict):
    """Document metadata for analysis."""

    char_count: int
    estimated_tokens: int
    line_count: int
    paragraph_count: int


class SummarizationState(TypedDict):
    """Unified state for the Summarization Graph.

    This state supports all three summarization strategies and maintains
    all necessary data for the complete workflow including reflection and revision.

    Attributes:
        original_text: The full input document to summarize.
        doc_metadata: Document statistics (length, tokens, etc.).

        content_type: Classification of the document type.
        selected_strategy: The chosen summarization strategy.
        summary_focus: Planner-generated focus instructions.

        chunks: Text split into segments based on strategy.
        chunk_summaries: Intermediate summaries from map/refine steps.
        running_summary: Accumulated summary for REFINE strategy.

        draft_summary: The pre-reflection result.
        critique_feedback: Feedback from the Reflector node.
        revision_count: Loop guard to prevent infinite revisions.
        final_output: The final approved summary.
    """

    # Input
    original_text: str
    doc_metadata: NotRequired[DocMetadata]

    # Analysis & Planning
    content_type: NotRequired[str]
    selected_strategy: NotRequired[SummarizationStrategy]
    summary_focus: NotRequired[str]

    # Execution Data
    chunks: NotRequired[list[str]]
    chunk_summaries: NotRequired[list[str]]
    running_summary: NotRequired[str]

    # Finalization
    draft_summary: NotRequired[str]
    critique_feedback: NotRequired[str]
    revision_count: NotRequired[int]
    final_output: NotRequired[str]


class SummarizationAgent:
    """Summarization Agent with intelligent strategy selection.

    This agent analyzes input documents and automatically selects the optimal
    summarization strategy:
    - REFINE: For narratives, preserves chronological flow
    - MAP_REDUCE: For informational content, parallel processing
    - HIERARCHICAL: For massive documents (>50k tokens)

    The workflow includes:
    1. Router/Planner: Analyzes document and selects strategy
    2. Chunker: Splits text based on strategy requirements
    3. Strategy Execution: Runs the selected summarization approach
    4. Reflector: Quality assurance with critique
    5. Reviser: Addresses critique if needed (max 2 revisions)

    Attributes:
        model: The language model to use.
        chunker: SmartChunker for strategy-aware text splitting.
        graph: The compiled LangGraph workflow.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> model = ChatOpenAI(model="gpt-4")
        >>> agent = SummarizationAgent(model)
        >>> result = agent.invoke("Long document text here...")
        >>> print(result)
    """

    def __init__(self, model: BaseChatModel):
        """Initialize the SummarizationAgent.

        Args:
            model: The language model to use for all operations.
        """
        self.model = model
        self.chunker = SmartChunker()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph workflow.

        Returns:
            Compiled StateGraph for summarization.
        """
        workflow = StateGraph(SummarizationState)

        # Add nodes
        workflow.add_node("router_node", self._router_node)
        workflow.add_node("chunk_node", self._chunk_node)
        workflow.add_node("direct_summarize_node", self._direct_summarize_node)
        workflow.add_node("map_reduce_node", self._map_reduce_node)
        workflow.add_node("refine_node", self._refine_node)
        workflow.add_node("hierarchical_node", self._hierarchical_node)
        workflow.add_node("reflect_node", self._reflect_node)
        workflow.add_node("revise_node", self._revise_node)

        # Define edges
        workflow.add_edge(START, "router_node")
        workflow.add_conditional_edges(
            "router_node",
            self._route_after_router,
            {
                "direct": "direct_summarize_node",
                "chunk": "chunk_node",
            },
        )
        workflow.add_edge("direct_summarize_node", END)
        workflow.add_conditional_edges(
            "chunk_node",
            self._route_to_strategy,
            {
                "MAP_REDUCE": "map_reduce_node",
                "REFINE": "refine_node",
                "HIERARCHICAL": "hierarchical_node",
            },
        )
        workflow.add_edge("map_reduce_node", "reflect_node")
        workflow.add_edge("refine_node", "reflect_node")
        workflow.add_edge("hierarchical_node", "reflect_node")
        workflow.add_conditional_edges(
            "reflect_node",
            self._route_after_reflection,
            {"revise": "revise_node", "end": END},
        )
        workflow.add_edge("revise_node", "reflect_node")

        return workflow.compile()

    def _route_after_router(
        self, state: SummarizationState
    ) -> Literal["direct", "chunk"]:
        """Route after router based on document length.

        Args:
            state: Current graph state.

        Returns:
            "direct" for short docs, "chunk" for long docs.
        """
        metadata = state.get("doc_metadata", {})
        tokens = metadata.get("estimated_tokens", 0)

        if tokens < TOKEN_THRESHOLD_LONG:
            return "direct"
        return "chunk"

    def _route_to_strategy(
        self, state: SummarizationState
    ) -> SummarizationStrategy:
        """Route to the selected strategy.

        Args:
            state: Current graph state.

        Returns:
            The selected strategy name.
        """
        return state.get("selected_strategy", "MAP_REDUCE")

    def _route_after_reflection(
        self, state: SummarizationState
    ) -> Literal["revise", "end"]:
        """Route after reflection based on critique.

        Args:
            state: Current graph state.

        Returns:
            "end" if approved or max revisions, "revise" otherwise.
        """
        critique = state.get("critique_feedback", "").strip().upper()
        revision_count = state.get("revision_count", 0)

        if critique == "APPROVED" or revision_count >= MAX_REVISION_COUNT:
            return "end"
        return "revise"

    def _router_node(self, state: SummarizationState) -> dict:
        """Analyze document and select summarization strategy.

        This is the "Brain" of the agent. It:
        1. Extracts document metadata
        2. Classifies content type (narrative/informational/massive)
        3. Selects optimal strategy (REFINE/MAP_REDUCE/HIERARCHICAL)
        4. Generates focus instructions for summarization

        Args:
            state: Current graph state with original_text.

        Returns:
            Updated state with metadata, strategy, and focus.
        """
        print("[Router] Analyzing document and selecting strategy...")

        original_text = state["original_text"]
        metadata = get_doc_metadata(original_text)
        tokens = metadata["estimated_tokens"]

        # Short document - direct summarization
        if tokens < TOKEN_THRESHOLD_LONG:
            print(f"[Router] Short document ({tokens} tokens) - using direct summarization")
            return {
                "doc_metadata": metadata,
                "content_type": "informational",
                "selected_strategy": "MAP_REDUCE",
                "summary_focus": "Capture all key points concisely.",
                "revision_count": 0,
            }

        # Extract preview for analysis
        preview_chars = PREVIEW_TOKEN_COUNT * 4
        beginning = original_text[:preview_chars]
        ending = original_text[-preview_chars:] if len(original_text) > preview_chars else ""

        # Force HIERARCHICAL for massive documents
        if tokens >= TOKEN_THRESHOLD_MASSIVE:
            print(f"[Router] Massive document ({tokens} tokens) - forcing HIERARCHICAL strategy")
            return {
                "doc_metadata": metadata,
                "content_type": "massive_dataset",
                "selected_strategy": "HIERARCHICAL",
                "summary_focus": "Extract and organize key information from this large document.",
                "revision_count": 0,
            }

        # Use LLM to classify and plan
        prompt = SummarizationPrompts.ROUTER_PROMPT.format(
            beginning=beginning,
            ending=ending,
            estimated_tokens=tokens,
            char_count=metadata["char_count"],
            paragraph_count=metadata["paragraph_count"],
        )

        try:
            response = self.model.invoke([
                SystemMessage(content=SummarizationPrompts.ROUTER_SYSTEM),
                HumanMessage(content=prompt),
            ])

            # Parse JSON response
            result = self._parse_router_response(response.content)
            content_type = result.get("content_type", "informational")
            strategy = result.get("selected_strategy", "MAP_REDUCE")
            focus = result.get("summary_focus", "Summarize the key points.")

            print(f"[Router] Selected Strategy: {strategy}")
            print(f"[Router] Content Type: {content_type}")
            print(f"[Router] Focus: {focus[:100]}...")

            return {
                "doc_metadata": metadata,
                "content_type": content_type,
                "selected_strategy": strategy,
                "summary_focus": focus,
                "revision_count": 0,
            }

        except Exception as e:
            # Fallback to MAP_REDUCE on error
            print(f"[Router] Error during analysis: {e}. Defaulting to MAP_REDUCE.")
            return {
                "doc_metadata": metadata,
                "content_type": "informational",
                "selected_strategy": "MAP_REDUCE",
                "summary_focus": "Extract main ideas and key points.",
                "revision_count": 0,
            }

    def _parse_router_response(self, response: str) -> dict:
        """Parse JSON response from router LLM.

        Args:
            response: The LLM response string.

        Returns:
            Parsed dictionary with strategy info.
        """
        try:
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Try to find JSON object in response
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {}

    def _chunk_node(self, state: SummarizationState) -> dict:
        """Split document into chunks based on strategy.

        Args:
            state: Current graph state.

        Returns:
            Updated state with chunks.
        """
        strategy = state.get("selected_strategy", "MAP_REDUCE")
        print(f"[Chunker] Splitting text for {strategy} strategy...")

        chunks = self.chunker.chunk(state["original_text"], strategy)
        print(f"[Chunker] Created {len(chunks)} chunks")

        return {"chunks": chunks}

    def _direct_summarize_node(self, state: SummarizationState) -> dict:
        """Directly summarize short documents.

        Args:
            state: Current graph state.

        Returns:
            Updated state with final_output.
        """
        print("[DirectSummarize] Processing short document...")

        focus = state.get("summary_focus", "Capture all key points.")
        prompt = SummarizationPrompts.DIRECT_SUMMARY_PROMPT.format(
            summary_focus=focus,
            text=state["original_text"],
        )

        try:
            response = self.model.invoke([
                SystemMessage(content="You are a concise summarization expert."),
                HumanMessage(content=prompt),
            ])
            summary = response.content
        except Exception as e:
            summary = f"Error during summarization: {e}"

        print("[DirectSummarize] Complete.")
        return {"final_output": summary}

    def _map_reduce_node(self, state: SummarizationState) -> dict:
        """Execute MAP_REDUCE strategy.

        Map: Summarize each chunk in parallel (conceptually).
        Reduce: Combine all summaries into a draft.

        Args:
            state: Current graph state.

        Returns:
            Updated state with chunk_summaries and draft_summary.
        """
        print("[MapReduce] Starting parallel summarization...")

        chunks = state.get("chunks", [])
        focus = state.get("summary_focus", "")
        chunk_summaries = []

        # MAP phase - summarize each chunk
        for i, chunk in enumerate(chunks):
            print(f"[MapReduce] Processing chunk {i + 1}/{len(chunks)}")
            prompt = SummarizationPrompts.MAP_PROMPT.format(
                summary_focus=focus,
                chunk=chunk,
            )

            try:
                response = self.model.invoke([
                    SystemMessage(content="You are a focused summarization assistant."),
                    HumanMessage(content=prompt),
                ])
                chunk_summaries.append(response.content)
            except Exception as e:
                chunk_summaries.append(f"[Chunk {i + 1} failed: {e}]")

        # REDUCE phase - combine summaries
        print("[MapReduce] Reducing summaries...")
        formatted_summaries = "\n\n".join(
            f"Section {i + 1}:\n{summary}"
            for i, summary in enumerate(chunk_summaries)
        )

        reduce_prompt = SummarizationPrompts.REDUCE_PROMPT.format(
            summary_focus=focus,
            chunk_summaries=formatted_summaries,
        )

        try:
            response = self.model.invoke([
                SystemMessage(content="You are an expert at synthesizing information."),
                HumanMessage(content=reduce_prompt),
            ])
            draft = response.content
        except Exception:
            draft = "\n\n".join(chunk_summaries)

        print("[MapReduce] Complete.")
        return {
            "chunk_summaries": chunk_summaries,
            "draft_summary": draft,
        }

    def _refine_node(self, state: SummarizationState) -> dict:
        """Execute REFINE strategy.

        Sequential processing where each chunk updates a running summary.
        Best for narratives where chronological order matters.

        Args:
            state: Current graph state.

        Returns:
            Updated state with running_summary and draft_summary.
        """
        print("[Refine] Starting sequential refinement...")

        chunks = state.get("chunks", [])
        focus = state.get("summary_focus", "")
        running_summary = ""

        for i, chunk in enumerate(chunks):
            print(f"[Refine] Processing chunk {i + 1}/{len(chunks)}")

            if i == 0:
                # Initial summary
                prompt = SummarizationPrompts.REFINE_INITIAL_PROMPT.format(
                    summary_focus=focus,
                    chunk=chunk,
                )
            else:
                # Refine existing summary
                prompt = SummarizationPrompts.REFINE_UPDATE_PROMPT.format(
                    summary_focus=focus,
                    running_summary=running_summary,
                    chunk=chunk,
                )

            try:
                response = self.model.invoke([
                    SystemMessage(content="You are a narrative summarization expert."),
                    HumanMessage(content=prompt),
                ])
                running_summary = response.content
            except Exception as e:
                if i == 0:
                    running_summary = f"[Initial chunk failed: {e}]"
                # On later chunks, keep existing summary

        print("[Refine] Complete.")
        return {
            "running_summary": running_summary,
            "draft_summary": running_summary,
        }

    def _hierarchical_node(self, state: SummarizationState) -> dict:
        """Execute HIERARCHICAL strategy.

        Tree-based recursive summarization for massive documents.
        Creates layers of summaries until converging to a single summary.

        Args:
            state: Current graph state.

        Returns:
            Updated state with chunk_summaries and draft_summary.
        """
        print("[Hierarchical] Starting tree-based summarization...")

        chunks = state.get("chunks", [])
        focus = state.get("summary_focus", "")
        group_size = 5

        # Layer 0: Summarize leaf chunks
        print(f"[Hierarchical] Layer 0: Summarizing {len(chunks)} leaf chunks...")
        current_level = []

        for i, chunk in enumerate(chunks):
            prompt = SummarizationPrompts.HIERARCHICAL_LEAF_PROMPT.format(
                summary_focus=focus,
                chunk=chunk,
            )

            try:
                response = self.model.invoke([
                    SystemMessage(content="You are a concise summarization assistant."),
                    HumanMessage(content=prompt),
                ])
                current_level.append(response.content)
            except Exception:
                current_level.append(f"[Leaf {i + 1} summary]")

        all_summaries = list(current_level)
        layer = 1

        # Recursive merge until single summary
        while len(current_level) > 1:
            print(f"[Hierarchical] Layer {layer}: Merging {len(current_level)} summaries...")
            next_level = []

            for i in range(0, len(current_level), group_size):
                group = current_level[i : i + group_size]
                formatted_group = "\n\n".join(
                    f"Summary {j + 1}:\n{s}" for j, s in enumerate(group)
                )

                prompt = SummarizationPrompts.HIERARCHICAL_MERGE_PROMPT.format(
                    summary_focus=focus,
                    summaries=formatted_group,
                )

                try:
                    response = self.model.invoke([
                        SystemMessage(content="You are an expert at merging summaries."),
                        HumanMessage(content=prompt),
                    ])
                    next_level.append(response.content)
                except Exception:
                    # Fallback: concatenate
                    next_level.append(" ".join(group))

            current_level = next_level
            all_summaries.extend(current_level)
            layer += 1

        draft = current_level[0] if current_level else "No summary generated."
        print(f"[Hierarchical] Complete. Total layers: {layer}")

        return {
            "chunk_summaries": all_summaries,
            "draft_summary": draft,
        }

    def _reflect_node(self, state: SummarizationState) -> dict:
        """Reflect on draft summary quality.

        Acts as a Senior Editor reviewing the summary against goals.

        Args:
            state: Current graph state.

        Returns:
            Updated state with critique_feedback and possibly final_output.
        """
        print("[Reflector] Evaluating draft summary...")

        draft = state.get("draft_summary", "")
        focus = state.get("summary_focus", "")
        original = state.get("original_text", "")

        # Sample from original for reference
        sample_size = min(2000, len(original))
        sample = original[:sample_size]
        if len(original) > sample_size:
            sample += "\n...[truncated]..."

        prompt = SummarizationPrompts.REFLECTOR_PROMPT.format(
            summary_focus=focus,
            draft_summary=draft,
            original_sample=sample,
        )

        try:
            response = self.model.invoke([
                SystemMessage(content="You are a Senior Editor with high standards."),
                HumanMessage(content=prompt),
            ])
            critique = response.content.strip()
        except Exception:
            critique = "APPROVED"

        result = {"critique_feedback": critique}

        if critique.upper() == "APPROVED":
            print("[Reflector] Summary APPROVED.")
            result["final_output"] = draft
        else:
            print(f"[Reflector] Critique: {critique[:100]}...")

        return result

    def _revise_node(self, state: SummarizationState) -> dict:
        """Revise draft based on critique feedback.

        Args:
            state: Current graph state.

        Returns:
            Updated state with revised draft_summary and incremented revision_count.
        """
        revision_count = state.get("revision_count", 0) + 1
        print(f"[Reviser] Revision #{revision_count}...")

        draft = state.get("draft_summary", "")
        focus = state.get("summary_focus", "")
        critique = state.get("critique_feedback", "")

        prompt = SummarizationPrompts.REVISER_PROMPT.format(
            summary_focus=focus,
            draft_summary=draft,
            critique_feedback=critique,
        )

        try:
            response = self.model.invoke([
                SystemMessage(content="You are an expert editor improving summaries."),
                HumanMessage(content=prompt),
            ])
            revised = response.content
        except Exception:
            revised = draft

        print("[Reviser] Revision complete.")
        return {
            "draft_summary": revised,
            "revision_count": revision_count,
            "final_output": revised,  # Set as final in case max revisions hit
        }

    def invoke(self, text: str) -> str:
        """Invoke the summarization agent.

        This is the main entry point for using the agent.

        Args:
            text: The document text to summarize.

        Returns:
            The final summary string.

        Example:
            >>> agent = SummarizationAgent(model)
            >>> summary = agent.invoke("Long document content...")
            >>> print(summary)
        """
        if not text or not text.strip():
            return "Error: Empty text provided for summarization."

        initial_state: SummarizationState = {
            "original_text": text,
        }

        result = self.graph.invoke(initial_state)
        return result.get("final_output", "Error: No summary generated.")

    def invoke_with_state(self, text: str) -> SummarizationState:
        """Invoke the agent and return full state for debugging.

        Args:
            text: The document text to summarize.

        Returns:
            The complete final state dictionary.
        """
        if not text or not text.strip():
            return {"original_text": text, "final_output": "Error: Empty text."}

        initial_state: SummarizationState = {
            "original_text": text,
        }

        return self.graph.invoke(initial_state)
