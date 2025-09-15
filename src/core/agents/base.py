
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

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
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        """Check if action exists."""
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_model(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t['name'] not in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}