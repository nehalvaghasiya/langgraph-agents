
from typing import Literal
from langgraph.prebuilt import ToolNode, tools_condition

from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from core.prompts.rag import RAGPrompts

class RagAgent:
    """Class for RAG agent."""
    def __init__(self, model, doc_splits):
        """Initialized RAG agent."""
        self.model = model
        
        model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
        encode_kwargs = {"normalize_embeddings": False}
        vectorstore = InMemoryVectorStore.from_documents(
            documents=doc_splits, embedding=HuggingFaceEmbeddings(
                model_name="jinaai/jina-embeddings-v2-base-en", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        self.retriever_tool = create_retriever_tool(
            retriever, "retrieve_blog_posts", "Search and return information about Lilian Weng blog posts."
        )
        
        # Build the graph (mirroring your workflow)
        from langgraph.graph import MessagesState, StateGraph, START, END
        workflow = StateGraph(MessagesState)
        workflow.add_node(self.generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node(self.rewrite_question)
        workflow.add_node(self.generate_answer)
        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_conditional_edges(
            "generate_query_or_respond", tools_condition, {"tools": "retrieve", END: END}
        )
        workflow.add_conditional_edges("retrieve", self.grade_documents)
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        self.graph = workflow.compile()

    def generate_query_or_respond(self, state):
        """Generate query or respond."""
        response = self.model.bind_tools([self.retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    class GradeDocuments(BaseModel):
        """Pydantic structure for Grade Documents."""
        binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")

    def grade_documents(self, state) -> Literal["generate_answer", "rewrite_question"]:
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = RAGPrompts.GRADE_PROMPT.format(question=question, context=context)
        response = self.model.with_structured_output(self.GradeDocuments).invoke([{"role": "user", "content": prompt}])
        return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

    def rewrite_question(self, state):
        messages = state["messages"]
        question = messages[0].content
        prompt = RAGPrompts.REWRITE_PROMPT.format(question=question)
        response = self.model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}

    def generate_answer(self, state):
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = RAGPrompts.GENERATE_PROMPT.format(question=question, context=context)
        response = self.model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}