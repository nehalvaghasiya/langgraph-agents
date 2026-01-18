"""Prompts for the Summarization Agent with dynamic strategy selection.

This module contains all prompt templates for the router, strategy execution
(MAP_REDUCE, REFINE, HIERARCHICAL), reflection, and revision nodes.
"""


class SummarizationPrompts:
    """Centralized prompts for the Summarization Agent."""

    ROUTER_SYSTEM = """You are an expert document analyst. Your task is to classify a document 
and determine the optimal summarization strategy based on its content type and length."""

    ROUTER_PROMPT = """Analyze the following document preview and metadata to determine the best summarization approach.

DOCUMENT PREVIEW (beginning):
---
{beginning}
---

DOCUMENT PREVIEW (ending):
---
{ending}
---

DOCUMENT METADATA:
- Estimated tokens: {estimated_tokens}
- Character count: {char_count}
- Paragraphs: {paragraph_count}

Based on this analysis, provide your assessment as a JSON object with these fields:

1. "content_type": Choose one of:
   - "narrative" - Stories, biographies, fiction, chronological accounts where flow matters
   - "informational" - Technical docs, news, reports, factual content
   - "massive_dataset" - Very large documents (>50k tokens), logs, transcripts

2. "selected_strategy": Choose one of:
   - "REFINE" - Best for narratives (preserves chronological flow, sequential processing)
   - "MAP_REDUCE" - Best for informational content (parallel processing, efficient)
   - "HIERARCHICAL" - Best for massive documents (tree-based, handles extreme length)

3. "summary_focus": Specific instructions for what to focus on in the summary.
   Examples: "Track the protagonist's emotional journey", "Extract key financial figures", 
   "Identify main arguments and evidence", "Summarize technical specifications"

Respond with ONLY a valid JSON object, no additional text."""

    DIRECT_SUMMARY_PROMPT = """Summarize the following text concisely while capturing all key points.

SUMMARIZATION FOCUS:
{summary_focus}

TEXT:
---
{text}
---

Provide a clear, well-structured summary that addresses the focus goals."""

    MAP_PROMPT = """You are summarizing a section of a larger document.
Strictly adhere to the following summarization focus:

SUMMARIZATION FOCUS:
{summary_focus}

TEXT SECTION:
---
{chunk}
---

Provide a focused summary of this section that aligns with the stated focus.
Extract key information relevant to the goals. Be concise but comprehensive."""

    REDUCE_PROMPT = """You are combining multiple section summaries into a cohesive final summary.

SUMMARIZATION FOCUS:
{summary_focus}

SECTION SUMMARIES:
{chunk_summaries}

Create a unified, coherent summary that:
1. Flows naturally without repetition
2. Maintains the focus defined in the summarization goals
3. Preserves all critical information from the section summaries
4. Is well-structured with clear organization

Provide the synthesized summary:"""

    REFINE_INITIAL_PROMPT = """You are creating an initial summary of a document section.
This will be refined as more sections are processed.

SUMMARIZATION FOCUS:
{summary_focus}

FIRST SECTION:
---
{chunk}
---

Create a comprehensive summary of this section that aligns with the focus.
This summary will be updated as we process additional sections."""

    REFINE_UPDATE_PROMPT = """You are refining an existing summary with new information from the next section.

SUMMARIZATION FOCUS:
{summary_focus}

CURRENT RUNNING SUMMARY:
---
{running_summary}
---

NEW SECTION TO INCORPORATE:
---
{chunk}
---

Update the running summary to incorporate the new information from this section.
Maintain coherent flow and avoid redundancy. Preserve chronological order if relevant.
The updated summary should be comprehensive but not repetitive."""

    HIERARCHICAL_LEAF_PROMPT = """You are summarizing a granular section of a very large document.
Keep this summary focused and concise as it will be combined with others.

SUMMARIZATION FOCUS:
{summary_focus}

SECTION:
---
{chunk}
---

Provide a concise summary (2-3 sentences) capturing the key points."""

    HIERARCHICAL_MERGE_PROMPT = """You are merging multiple summaries into a higher-level summary.
This is part of a hierarchical summarization of a very large document.

SUMMARIZATION FOCUS:
{summary_focus}

SUMMARIES TO MERGE:
{summaries}

Combine these summaries into a coherent higher-level summary.
Remove redundancy and maintain the key information from each.
Keep the result focused and well-organized."""

    REFLECTOR_PROMPT = """You are a Senior Editor reviewing a summary for quality and completeness.

ORIGINAL SUMMARIZATION FOCUS:
{summary_focus}

DRAFT SUMMARY:
---
{draft_summary}
---

SAMPLE FROM ORIGINAL TEXT (for reference):
---
{original_sample}
---

Evaluate the summary:
1. Does it address the stated focus goals?
2. Is it coherent and well-organized?
3. Are there obvious omissions or inaccuracies?
4. Is there unnecessary repetition?

If the summary is satisfactory and meets the goals, respond with exactly: APPROVED

If improvements are needed, provide specific, actionable critique points.
Be concise but precise about what needs to change."""

    REVISER_PROMPT = """You are revising a summary based on editorial feedback.

SUMMARIZATION FOCUS:
{summary_focus}

CURRENT DRAFT:
---
{draft_summary}
---

EDITORIAL FEEDBACK:
---
{critique_feedback}
---

Revise the summary to address the feedback while:
1. Maintaining alignment with the summarization focus
2. Improving coherence and flow
3. Addressing specific critique points

Provide the revised summary:"""
