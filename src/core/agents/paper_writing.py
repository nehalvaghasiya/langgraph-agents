
from typing import Literal
from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from core.agents.chart_generator import ChartGeneratorAgent
from core.agents.doc_writer import DocWriterAgent
from core.agents.note_taker import NoteTakerAgent
from core.agents.supervisor import SupervisorAgent

from core.supervisor import State as SupervisorState, make_supervisor_node


class PaperWritingTeamAgent:
    """Agent for Paper Writing Team."""
    def __init__(self, model):
        """Initialized the Agent for Paper Writing Team."""
        self.model = model
        
        # Define members as sub-agents
        self.doc_writer = DocWriterAgent(model)
        self.note_taker = NoteTakerAgent(model)
        self.chart_generator = ChartGeneratorAgent(model)
        self.members = ["doc_writer", "note_taker", "chart_generator"]
        self.supervisor = SupervisorAgent(model, self.members)
        
        # Build team graph
        graph = StateGraph(SupervisorState)  # Use SupervisorState from your code
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("doc_writer", self.doc_writer_node)
        graph.add_node("note_taker", self.note_taker_node)
        graph.add_node("chart_generator", self.chart_generator_node)
        graph.add_edge(START, "supervisor")
        
        # Conditional edges would be added based on supervisor routing (adapt from make_supervisor_node)
        self.graph = graph.compile()

    def supervisor_node(self, state: SupervisorState) -> Command[Literal["doc_writer", "note_taker", "chart_generator"]]:
        """Create Supervisor Node."""
        # Use your make_supervisor_node logic here
        return make_supervisor_node(self.model, self.members)(state)

    def doc_writer_node(self, state: SupervisorState) -> Command[Literal["supervisor"]]:
        """Create a node for writing doc."""
        result = self.doc_writer.graph.invoke(state)
        return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="doc_writer")]}, goto="supervisor")

    def note_taker_node(self, state: SupervisorState) -> Command[Literal["supervisor"]]:
        """Create a node for taking Note."""
        result = self.note_taker.graph.invoke(state)
        return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="note_taker")]}, goto="supervisor")

    def chart_generator_node(self, state: SupervisorState) -> Command[Literal["supervisor"]]:
        """Create a node for generating chart."""
        result = self.chart_generator.graph.invoke(state)
        return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="chart_generator")]}, goto="supervisor")
