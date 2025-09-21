from typing import Any

from langchain_core.language_models import BaseChatModel

from core.agents.base import BaseAgent


class MemberTool:
    def __init__(self, member: str):
        self.name = member
        self.member = member

    def __call__(self, args: Any) -> str:
        print(f"Routing to member: {self.member}")
        return f"Result from {self.member}"

    def invoke(self, args: Any) -> str:
        return self(args)


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
        member_tools = [self.create_member_tool(member) for member in members]
        super().__init__(model, member_tools, system)

    def create_member_tool(self, member: str):
        """Create member tool that invokes a sub-agent."""
        return MemberTool(member)
