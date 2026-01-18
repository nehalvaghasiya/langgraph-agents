"""Tests for the Summarization Agent.

Tests cover:
- State and metadata handling
- SmartChunker with different strategies
- Router logic and strategy selection
- Map-Reduce, Refine, and Hierarchical strategies
- Reflection loop and revision handling
- Edge cases and error handling
"""

from core.agents.summarization import (
    SummarizationAgent,
    SummarizationState,
)
from core.tools.summarization import (
    CHUNK_CONFIG,
    MAX_REVISION_COUNT,
    TOKEN_THRESHOLD_LONG,
    TOKEN_THRESHOLD_MASSIVE,
    SmartChunker,
    count_tokens,
    get_doc_metadata,
)


class MockLLMResponse:
    """Mock LLM response object."""

    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """Mock language model for testing."""

    def __init__(self, responses: list[str] | None = None):
        """Initialize with optional sequence of responses."""
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.messages_log = []

    def invoke(self, messages):
        """Return next response in sequence."""
        self.messages_log.append(messages)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return MockLLMResponse(response)


class TestCountTokens:
    """Tests for count_tokens utility."""

    def test_empty_string(self):
        """Test token count for empty string."""
        assert count_tokens("") == 0

    def test_short_text(self):
        """Test token count for short text."""
        text = "Hello world"  # 11 chars
        assert count_tokens(text) == 2  # 11 / 4 = 2

    def test_custom_ratio(self):
        """Test token count with custom ratio."""
        text = "Test text"  # 9 chars
        assert count_tokens(text, chars_per_token=3.0) == 3


class TestGetDocMetadata:
    """Tests for get_doc_metadata utility."""

    def test_empty_text(self):
        """Test metadata for empty text."""
        meta = get_doc_metadata("")
        assert meta["char_count"] == 0
        assert meta["estimated_tokens"] == 0

    def test_multiline_text(self):
        """Test metadata for multi-paragraph text."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        meta = get_doc_metadata(text)
        assert meta["char_count"] == len(text)
        assert meta["paragraph_count"] == 3
        assert meta["line_count"] == 5  # Including empty lines


class TestSmartChunker:
    """Tests for SmartChunker class."""

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = SmartChunker()
        chunks = chunker.chunk("", "MAP_REDUCE")
        assert chunks == []

    def test_short_text_single_chunk(self):
        """Test that short text returns single chunk."""
        chunker = SmartChunker()
        text = "Short text here."
        chunks = chunker.chunk(text, "MAP_REDUCE")
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_different_strategies_different_sizes(self):
        """Test that different strategies use different chunk sizes."""
        # Verify configurations exist
        assert "MAP_REDUCE" in CHUNK_CONFIG
        assert "REFINE" in CHUNK_CONFIG
        assert "HIERARCHICAL" in CHUNK_CONFIG

        # REFINE should have larger chunks
        assert CHUNK_CONFIG["REFINE"]["chunk_size"] > CHUNK_CONFIG["MAP_REDUCE"]["chunk_size"]
        # HIERARCHICAL should have smaller chunks
        assert CHUNK_CONFIG["HIERARCHICAL"]["chunk_size"] < CHUNK_CONFIG["MAP_REDUCE"]["chunk_size"]

    def test_long_text_multiple_chunks(self):
        """Test that long text is split into multiple chunks."""
        chunker = SmartChunker()
        text = "This is a sentence. " * 200  # Long text
        chunks = chunker.chunk(text, "MAP_REDUCE")
        assert len(chunks) > 1

    def test_hierarchical_grouping(self):
        """Test hierarchical chunk grouping."""
        chunker = SmartChunker()
        text = "Section content. " * 500
        groups = chunker.chunk_for_hierarchical(text, group_size=5)
        assert len(groups) > 0
        # Each group should have at most group_size chunks
        for group in groups:
            assert len(group) <= 5


class TestSummarizationState:
    """Tests for state TypedDict."""

    def test_minimal_state(self):
        """Test creating state with minimal fields."""
        state: SummarizationState = {"original_text": "Test"}
        assert state["original_text"] == "Test"

    def test_full_state(self):
        """Test creating state with all fields."""
        state: SummarizationState = {
            "original_text": "Test document",
            "doc_metadata": {
                "char_count": 13,
                "estimated_tokens": 3,
                "line_count": 1,
                "paragraph_count": 1,
            },
            "content_type": "informational",
            "selected_strategy": "MAP_REDUCE",
            "summary_focus": "Key points",
            "chunks": ["chunk1", "chunk2"],
            "chunk_summaries": ["summary1", "summary2"],
            "running_summary": "",
            "draft_summary": "Draft",
            "critique_feedback": "APPROVED",
            "revision_count": 0,
            "final_output": "Final summary",
        }
        assert state["selected_strategy"] == "MAP_REDUCE"
        assert state["revision_count"] == 0


class TestAgentInitialization:
    """Tests for SummarizationAgent initialization."""

    def test_agent_creation(self):
        """Test that agent can be created."""
        model = MockLLM()
        agent = SummarizationAgent(model)
        assert agent.model == model
        assert agent.chunker is not None
        assert agent.graph is not None

    def test_agent_has_chunker(self):
        """Test that agent has SmartChunker."""
        model = MockLLM()
        agent = SummarizationAgent(model)
        assert isinstance(agent.chunker, SmartChunker)


class TestRouting:
    """Tests for routing logic."""

    def test_route_short_doc_to_direct(self):
        """Test short documents route to direct summarization."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Short",
            "doc_metadata": {"estimated_tokens": 100, "char_count": 400, "line_count": 1, "paragraph_count": 1},
        }
        result = agent._route_after_router(state)
        assert result == "direct"

    def test_route_long_doc_to_chunk(self):
        """Test long documents route to chunking."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Long",
            "doc_metadata": {"estimated_tokens": 5000, "char_count": 20000, "line_count": 100, "paragraph_count": 20},
        }
        result = agent._route_after_router(state)
        assert result == "chunk"

    def test_route_to_strategy(self):
        """Test routing to specific strategy."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        for strategy in ["MAP_REDUCE", "REFINE", "HIERARCHICAL"]:
            state: SummarizationState = {
                "original_text": "Test",
                "selected_strategy": strategy,
            }
            result = agent._route_to_strategy(state)
            assert result == strategy

    def test_route_after_reflection_approved(self):
        """Test routing to end when approved."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Test",
            "critique_feedback": "APPROVED",
            "revision_count": 0,
        }
        result = agent._route_after_reflection(state)
        assert result == "end"

    def test_route_after_reflection_needs_revision(self):
        """Test routing to revise when critique given."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Test",
            "critique_feedback": "Missing key details about the topic.",
            "revision_count": 0,
        }
        result = agent._route_after_reflection(state)
        assert result == "revise"

    def test_route_after_reflection_max_revisions(self):
        """Test routing to end when max revisions reached."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Test",
            "critique_feedback": "Still needs work",
            "revision_count": MAX_REVISION_COUNT,
        }
        result = agent._route_after_reflection(state)
        assert result == "end"


class TestStrategySelection:
    """Tests for strategy selection in router node."""

    def test_router_selects_narrative_refine(self):
        """Test that narrative content suggests REFINE strategy."""
        # Mock LLM returns REFINE for narrative
        model = MockLLM(responses=[
            '{"content_type": "narrative", "selected_strategy": "REFINE", "summary_focus": "Track character arc"}'
        ])
        agent = SummarizationAgent(model)

        # Create long narrative text
        narrative = "Once upon a time, there was a hero. " * 300

        state: SummarizationState = {"original_text": narrative}
        result = agent._router_node(state)

        assert result["selected_strategy"] == "REFINE"
        assert result["content_type"] == "narrative"

    def test_router_selects_informational_map_reduce(self):
        """Test that informational content suggests MAP_REDUCE strategy."""
        model = MockLLM(responses=[
            '{"content_type": "informational", "selected_strategy": "MAP_REDUCE", "summary_focus": "Key facts"}'
        ])
        agent = SummarizationAgent(model)

        # Create long technical text
        technical = "The study found that the results were statistically significant. " * 300

        state: SummarizationState = {"original_text": technical}
        result = agent._router_node(state)

        assert result["selected_strategy"] == "MAP_REDUCE"
        assert result["content_type"] == "informational"

    def test_router_forces_hierarchical_for_massive(self):
        """Test that massive documents force HIERARCHICAL strategy."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        # Create massive text (>50k tokens worth)
        massive = "Content. " * 60000  # ~60k tokens

        state: SummarizationState = {"original_text": massive}
        result = agent._router_node(state)

        assert result["selected_strategy"] == "HIERARCHICAL"
        assert result["content_type"] == "massive_dataset"

    def test_router_fallback_on_invalid_json(self):
        """Test fallback to MAP_REDUCE on invalid JSON response."""
        model = MockLLM(responses=["This is not valid JSON at all!"])
        agent = SummarizationAgent(model)

        text = "Some content here. " * 300

        state: SummarizationState = {"original_text": text}
        result = agent._router_node(state)

        # Should fallback to MAP_REDUCE
        assert result["selected_strategy"] == "MAP_REDUCE"


class TestNodeExecution:
    """Tests for individual node execution."""

    def test_chunk_node(self):
        """Test chunking node."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Test content. " * 200,
            "selected_strategy": "MAP_REDUCE",
        }
        result = agent._chunk_node(state)

        assert "chunks" in result
        assert len(result["chunks"]) > 0

    def test_direct_summarize_node(self):
        """Test direct summarization node."""
        model = MockLLM(responses=["This is a direct summary."])
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Short text to summarize.",
            "summary_focus": "Key points",
        }
        result = agent._direct_summarize_node(state)

        assert "final_output" in result
        assert result["final_output"] == "This is a direct summary."

    def test_map_reduce_node(self):
        """Test MAP_REDUCE strategy execution."""
        # Responses: chunk summaries + reduce
        model = MockLLM(responses=[
            "Summary of chunk 1",
            "Summary of chunk 2",
            "Combined final summary",
        ])
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Full text",
            "chunks": ["Chunk 1 content", "Chunk 2 content"],
            "summary_focus": "Key points",
        }
        result = agent._map_reduce_node(state)

        assert "chunk_summaries" in result
        assert "draft_summary" in result
        assert len(result["chunk_summaries"]) == 2

    def test_refine_node(self):
        """Test REFINE strategy execution."""
        model = MockLLM(responses=[
            "Initial summary",
            "Refined with chunk 2",
        ])
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Full text",
            "chunks": ["Chunk 1", "Chunk 2"],
            "summary_focus": "Character arc",
        }
        result = agent._refine_node(state)

        assert "running_summary" in result
        assert "draft_summary" in result
        assert result["draft_summary"] == result["running_summary"]

    def test_hierarchical_node(self):
        """Test HIERARCHICAL strategy execution."""
        # Multiple responses for leaf summaries + merges
        model = MockLLM(responses=[
            "Leaf 1 summary",
            "Leaf 2 summary",
            "Merged summary",
        ])
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Full text",
            "chunks": ["Chunk 1", "Chunk 2"],
            "summary_focus": "Key information",
        }
        result = agent._hierarchical_node(state)

        assert "chunk_summaries" in result
        assert "draft_summary" in result

    def test_reflect_node_approved(self):
        """Test reflection node with approval."""
        model = MockLLM(responses=["APPROVED"])
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Original text here",
            "draft_summary": "Good summary",
            "summary_focus": "Key points",
        }
        result = agent._reflect_node(state)

        assert result["critique_feedback"] == "APPROVED"
        assert "final_output" in result

    def test_reflect_node_critique(self):
        """Test reflection node with critique."""
        model = MockLLM(responses=["Missing important details about the main topic."])
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Original text here",
            "draft_summary": "Incomplete summary",
            "summary_focus": "Key points",
        }
        result = agent._reflect_node(state)

        assert "Missing" in result["critique_feedback"]
        assert "final_output" not in result

    def test_revise_node(self):
        """Test revision node."""
        model = MockLLM(responses=["Improved summary with more details."])
        agent = SummarizationAgent(model)

        state: SummarizationState = {
            "original_text": "Original",
            "draft_summary": "Old draft",
            "summary_focus": "Key points",
            "critique_feedback": "Add more details",
            "revision_count": 0,
        }
        result = agent._revise_node(state)

        assert result["revision_count"] == 1
        assert "draft_summary" in result
        assert "final_output" in result


class TestIntegration:
    """Integration tests for complete flows."""

    def test_short_document_flow(self):
        """Test complete flow for short document."""
        model = MockLLM(responses=[
            "This is a concise summary of the short document."
        ])
        agent = SummarizationAgent(model)

        result = agent.invoke("A brief paragraph about testing.")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_long_document_map_reduce_flow(self):
        """Test complete flow for long document with MAP_REDUCE."""
        long_text = "Technical content about AI. " * 400

        model = MockLLM(responses=[
            # Router
            '{"content_type": "informational", "selected_strategy": "MAP_REDUCE", "summary_focus": "Key AI concepts"}',
            # Map chunks (several)
            "AI concept 1",
            "AI concept 2",
            "AI concept 3",
            # Reduce
            "Combined AI summary",
            # Reflect
            "APPROVED",
        ])
        agent = SummarizationAgent(model)

        result = agent.invoke(long_text)
        assert isinstance(result, str)

    def test_invoke_with_state_returns_full_state(self):
        """Test that invoke_with_state returns complete state."""
        model = MockLLM(responses=["Short summary"])
        agent = SummarizationAgent(model)

        result = agent.invoke_with_state("Brief text.")

        assert isinstance(result, dict)
        assert "original_text" in result

    def test_empty_text_handling(self):
        """Test handling of empty input."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        result = agent.invoke("")
        assert "error" in result.lower() or "empty" in result.lower()

    def test_whitespace_only_handling(self):
        """Test handling of whitespace-only input."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        result = agent.invoke("   \n\t   ")
        assert "error" in result.lower() or "empty" in result.lower()


class TestErrorHandling:
    """Tests for error handling."""

    def test_json_parsing_with_markdown(self):
        """Test JSON parsing from markdown code blocks."""
        model = MockLLM(responses=[
            '```json\n{"content_type": "narrative", "selected_strategy": "REFINE", "summary_focus": "Story"}\n```'
        ])
        agent = SummarizationAgent(model)

        result = agent._parse_router_response(model.responses[0])
        assert result["selected_strategy"] == "REFINE"

    def test_json_parsing_plain(self):
        """Test JSON parsing from plain JSON."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        result = agent._parse_router_response(
            '{"content_type": "informational", "selected_strategy": "MAP_REDUCE", "summary_focus": "Facts"}'
        )
        assert result["selected_strategy"] == "MAP_REDUCE"

    def test_json_parsing_invalid(self):
        """Test JSON parsing fallback for invalid JSON."""
        model = MockLLM()
        agent = SummarizationAgent(model)

        result = agent._parse_router_response("Not JSON at all")
        assert result == {}


class TestConstants:
    """Tests for configuration constants."""

    def test_token_thresholds(self):
        """Test token threshold values are sensible."""
        assert TOKEN_THRESHOLD_LONG < TOKEN_THRESHOLD_MASSIVE
        assert TOKEN_THRESHOLD_LONG > 0
        assert TOKEN_THRESHOLD_MASSIVE > 0

    def test_max_revision_count(self):
        """Test max revision count is reasonable."""
        assert MAX_REVISION_COUNT >= 1
        assert MAX_REVISION_COUNT <= 5

    def test_chunk_config_all_strategies(self):
        """Test chunk config exists for all strategies."""
        strategies = ["MAP_REDUCE", "REFINE", "HIERARCHICAL"]
        for strategy in strategies:
            assert strategy in CHUNK_CONFIG
            assert "chunk_size" in CHUNK_CONFIG[strategy]
            assert "chunk_overlap" in CHUNK_CONFIG[strategy]
