"""Example usage of the SummarizationAgent.

This script demonstrates the Smart Dynamic Summarization Agent with:
- Router-based strategy selection (MAP_REDUCE, REFINE, HIERARCHICAL)
- Automatic content analysis and strategy optimization
- Reflection loop with iterative improvement
"""

from core.agents.summarization import SummarizationAgent
from core.tools.summarization import count_tokens, get_doc_metadata
from infra.llm_clients.openai import get_llm


def main():
    """Run the summarization agent examples."""
    # Initialize the LLM
    llm = get_llm()

    # Create the summarization agent
    agent = SummarizationAgent(model=llm)


    # EXAMPLE 1: Short Text (Direct Summarization)
    short_text = """
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on developing algorithms that can access data and use it to learn
    for themselves. The learning process begins with observations or data, such
    as examples, direct experience, or instruction, to look for patterns in data
    and make better decisions in the future.
    """

    print("EXAMPLE 1: Short Text (Direct Summarization)")
    print(f"\nToken count: {count_tokens(short_text)}")
    print("\nOriginal text:")
    print(short_text.strip())
    print("\n>>> Summary:")
    summary = agent.invoke(short_text)
    print(summary)


    # EXAMPLE 2: Technical Document (MAP_REDUCE Strategy)
    technical_text = """
    Artificial Intelligence (AI) has emerged as one of the most transformative
    technologies of the 21st century. Its applications span across virtually
    every industry, from healthcare to finance, from transportation to entertainment.

    In healthcare, AI systems are being used to analyze medical images, predict
    disease outcomes, and assist in drug discovery. Machine learning algorithms
    can detect patterns in X-rays and MRI scans that might be missed by human
    radiologists. Natural language processing helps extract insights from medical
    records and research papers.

    The financial sector has embraced AI for fraud detection, algorithmic trading,
    and risk assessment. Banks use machine learning models to detect unusual
    transaction patterns that might indicate fraudulent activity. Robo-advisors
    leverage AI to provide personalized investment recommendations to clients.

    Transportation is being revolutionized by autonomous vehicles. Companies like
    Tesla, Waymo, and numerous others are developing self-driving cars that use
    computer vision, sensor fusion, and deep learning to navigate roads safely.
    AI is also optimizing logistics and supply chain operations worldwide.

    In entertainment, AI powers recommendation systems on platforms like Netflix,
    Spotify, and YouTube. Content creators use AI tools for video editing, music
    composition, and even writing. Generative AI models can create artwork,
    write stories, and produce realistic synthetic media.

    However, the rise of AI also brings challenges. Concerns about job displacement,
    algorithmic bias, privacy, and the potential for misuse are actively debated.
    Ethical AI development frameworks are being established to ensure these
    powerful technologies benefit humanity while minimizing risks.

    The future of AI holds immense promise. Advances in areas like general AI,
    quantum computing integration, and neuromorphic computing could lead to even
    more capable systems. As AI continues to evolve, it will undoubtedly reshape
    how we live, work, and interact with technology.
    """ * 5  # Repeat to make it longer

    print("EXAMPLE 2: Technical Document (likely MAP_REDUCE Strategy)")
    metadata = get_doc_metadata(technical_text)
    print(f"\nDocument stats: {metadata['estimated_tokens']} tokens, {metadata['paragraph_count']} paragraphs")
    print("\nProcessing... (this may take a moment)")

    result = agent.invoke_with_state(technical_text)
    print(f"\nSelected strategy: {result.get('selected_strategy', 'N/A')}")
    print(f"Content type: {result.get('content_type', 'N/A')}")
    print(f"Revisions made: {result.get('revision_count', 0)}")
    print(f"\n>>> Summary:\n{result.get('final_output', 'No output')}")


    # EXAMPLE 3: Narrative Content (REFINE Strategy)

    narrative_text = """
    Chapter 1: The Beginning

    Sarah had always known she was different. From the moment she could remember,
    she had the ability to see things others couldn't—shadows that whispered,
    lights that danced just beyond the corner of her eye. Her grandmother,
    before she passed, had called it "the sight."

    Growing up in the small coastal town of Millbrook, Sarah learned to hide her
    gift. The townspeople were superstitious, and her mother warned her that
    revealing her abilities would only bring trouble. So she kept quiet, even
    when she saw the darkness gathering at the old lighthouse.

    It wasn't until her sixteenth birthday that everything changed. The storm
    that night was unlike any the town had seen in decades. Lightning split the
    sky, and the wind howled like a wounded animal. And in the heart of the
    tempest, Sarah saw something that would alter the course of her life forever.

    A ship appeared on the horizon, its sails torn and masts broken, yet somehow
    it moved against the wind, heading straight for the treacherous rocks below
    the lighthouse. Sarah knew, with a certainty that went beyond mere intuition,
    that she was the only one who could help the souls aboard that phantom vessel.

    As she ran through the rain toward the cliff's edge, her grandmother's words
    echoed in her mind: "The sight is both a gift and a burden. Use it wisely,
    child, for there are forces in this world that prey on those who can see
    beyond the veil."

    Chapter 2: The Discovery

    The phantom ship vanished before Sarah reached the cliffs, leaving nothing
    but the storm's fury. But something had changed within her. The next morning,
    she awoke to find a weathered leather journal on her nightstand—a journal
    that hadn't been there the night before.

    Its pages were filled with her grandmother's handwriting, detailing centuries
    of family history. The women of her line had always possessed the sight,
    and they had always been called to protect the boundary between the living
    and the dead.
    """ * 3

    print("EXAMPLE 3: Narrative Content (likely REFINE Strategy)")
    metadata = get_doc_metadata(narrative_text)
    print(f"\nDocument stats: {metadata['estimated_tokens']} tokens, {metadata['paragraph_count']} paragraphs")
    print("\nProcessing narrative content...")

    result = agent.invoke_with_state(narrative_text)
    print(f"\nSelected strategy: {result.get('selected_strategy', 'N/A')}")
    print(f"Content type: {result.get('content_type', 'N/A')}")
    print(f"Summary focus: {result.get('summary_focus', 'N/A')}")
    print(f"\n>>> Summary:\n{result.get('final_output', 'No output')}")


    # EXAMPLE 4: Using the Graph Directly
    print("EXAMPLE 4: Inspecting Agent Graph Structure")

    # Access the compiled graph for visualization
    print("\nGraph nodes:")
    for node in ["router", "direct_summarize", "chunk", "map_reduce", "refine", "hierarchical", "reflect", "revise"]:
        print(f"  - {node}")

    print("\nGraph flow:")
    print("  START → router → [direct_summarize | chunk]")
    print("  chunk → [map_reduce | refine | hierarchical]")
    print("  [strategy] → reflect → [end | revise]")
    print("  revise → reflect")

    print("Examples completed!")


if __name__ == "__main__":
    main()
