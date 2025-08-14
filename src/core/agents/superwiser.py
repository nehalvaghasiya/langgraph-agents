from core.agents.base import BaseAgent


class SupervisorAgent(BaseAgent):
    def __init__(self, model, members: list[str], system: str = ""):
        # Members are treated as "tools" (sub-agents)
        self.members = members
        system = system or f"You are a supervisor managing workers: {members}. Route to the next worker or FINISH."
        # Override tools to be member invocations
        member_tools = [self.create_member_tool(member) for member in members]
        super().__init__(model, member_tools, system)

    def create_member_tool(self, member: str):
        # Dummy tool that invokes a sub-agent (you'd replace with actual sub-agent invocation)
        def member_tool(args):
            print(f"Routing to member: {member}")
            # Simulate sub-agent call (replace with actual agent.graph.invoke)
            return f"Result from {member}"
        member_tool.name = member
        member_tool.invoke = member_tool
        return member_tool