from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent


def make_member_tool(member: str):
    def tool(args: dict) -> str:
        print(f"Routing to member: {member}")
        return f"Result from {member}"

    tool.__name__ = member
    tool.name = member  # type: ignore[attr-defined]
    return tool


class SupervisorAgent(BaseAgent):
    """Class for Supervisor Agent."""

    def __init__(self, model: BaseChatModel, members: list[str], system: str = ""):
        # Members are treated as "tools" (sub-agents)
        self.members = members
        system = (
            system
            or f"You are a supervisor managing workers: {members}. Route to the next worker or FINISH."
        )
        # Override tools to be member invocations
        member_tools = [make_member_tool(member) for member in members]
        super().__init__(model, member_tools, system)
