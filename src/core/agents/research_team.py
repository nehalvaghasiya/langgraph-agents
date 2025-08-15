from core.agents.superwiser import SupervisorAgent
from core.agents.web_scraper import WebScraperAgent
from core.agents.web_search import SearchAgent

from typing import Literal
from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from core.supervisor import State as SupervisorState, make_supervisor_node


class ResearchTeamAgent:
    def __init__(self, model):
        self.model = model
        self.search = SearchAgent(model)
        self.web_scraper = WebScraperAgent(model)
        self.members = ["search", "web_scraper"]
        self.supervisor = SupervisorAgent(model, self.members)
        graph = StateGraph(SupervisorState)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("search", self.search_node)
        graph.add_node("web_scraper", self.web_scraper_node)
        graph.add_edge(START, "supervisor")
        self.graph = graph.compile()

    def supervisor_node(self, state: SupervisorState) -> Command[Literal["search", "web_scraper", "FINISH"]]:
        return make_supervisor_node(self.model, self.members)(state)

    def search_node(self, state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = self.search.graph.invoke(state)
        return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="search")]}, goto="supervisor")

    def web_scraper_node(self, state: SupervisorState) -> Command[Literal["supervisor"]]:
        result = self.web_scraper.graph.invoke(state)
        return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="web_scraper")]}, goto="supervisor")
