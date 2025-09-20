from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, MessagesState
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict


class State(MessagesState):
    """State for Graph execution."""

    next: NotRequired[str]


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> Callable[[State], Command[str]]:
    """Make supervisor node."""
    # options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: str

    def supervisor_node(state: State) -> Command[str]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response.next
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node
