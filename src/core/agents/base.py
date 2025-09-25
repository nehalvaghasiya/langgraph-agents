from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """AgentState to store messages.

    A TypedDict that defines the state structure for the agent graph.

    Attributes:
        messages (list): A list of messages representing the conversation history.
    """

    messages: list


class BaseAgent:
    """Base agent for langgraph agents.

    A base class for creating LangGraph agents with tool integration capabilities.
    This class sets up a state graph with LLM and action nodes, handles tool calls,
    and manages the conversation flow.

    Attributes:
        system (str): System message/prompt for the agent.
        graph: Compiled LangGraph state graph.
        tools (dict): Dictionary mapping tool names to tool objects.
        model: Language model bound with available tools.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain.tools import Tool
        >>>
        >>> model = ChatOpenAI()
        >>> tools = [some_tool]
        >>> agent = BaseAgent(model, tools, "You are a helpful assistant")
        >>> result = agent.graph.invoke({"messages": [HumanMessage("Hello")]})
    """

    def __init__(self, model: BaseChatModel, tools: list, system: str = ""):
        """Initialize the BaseAgent with model, tools, and optional system message.

        Args:
            model (BaseChatModel): The language model to use for generating responses.
            tools (list): List of tools available to the agent.
            system (str, optional): System message/prompt for the agent. Defaults to "".
        """
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState) -> bool:
        """Check if action exists.

        Determines whether the last message in the state contains tool calls
        that need to be executed.

        Args:
            state (AgentState): The current agent state containing messages.

        Returns:
            bool: True if tool calls exist in the last message, False otherwise.
        """
        """Check if action exists."""
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def call_model(self, state: AgentState) -> dict:
        """Call the language model with the current state.

        Invokes the language model with the conversation history, optionally
        prepending a system message if one was provided during initialization.

        Args:
            state (AgentState): The current agent state containing messages.

        Returns:
            dict: Dictionary containing the new message from the model in a list
                  under the 'messages' key.
        """
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState) -> dict:
        """Take action by executing tool calls.

        Processes tool calls from the last message in the state, executes each tool,
        and returns the results as ToolMessage objects. If a tool raises an exception,
        the error is caught and returned as a string in the ToolMessage content.

        Args:
            state (AgentState): The current agent state containing messages with tool calls.

        Returns:
            dict: Dictionary containing a list of ToolMessage objects under the 'messages' key,
                  each representing the result of a tool execution.

        Note:
            If a tool name is not found in available tools, returns an error message
            instead of executing the tool.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t["name"] not in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                try:
                    result = self.tools[t["name"]].invoke(t["args"])
                except Exception as e:
                    result = f"Exception: {e}"
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
        print("Back to the model!")
        return {"messages": results}
