from typing import Literal

from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from core.agents.base import AgentState
from core.prompts.rag import RAGPrompts


class RagAgent:
    """Class for RAG agent."""

    def __init__(self, model: BaseChatModel, doc_splits: list[Document]):
        """Initialized RAG agent."""
        self.model = model

        # Try CPU first to avoid CUDA issues
        model_kwargs = {"device": "cpu", "trust_remote_code": True}
        encode_kwargs = {"normalize_embeddings": False}

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        except Exception as e:
            print(f"Embeddings creation failed: {e}")
            raise

        try:
            vectorstore = InMemoryVectorStore.from_documents(
                documents=doc_splits,
                embedding=embeddings,
            )
        except Exception as e:
            print(f"Vectorstore creation failed: {e}")
            raise

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        self.retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_blog_posts",
            "Search and return information about Lilian Weng blog posts.",
        )

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

    def generate_query_or_respond(self, state: AgentState) -> dict:
        """Generate query or respond."""
        response = self.model.bind_tools([self.retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    class GradeDocuments(BaseModel):
        """Pydantic structure for Grade Documents."""

        binary_score: str = Field(
            description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
        )

    def grade_documents(self, state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
        """Grade documents to generate answer or rewrite question."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = RAGPrompts.GRADE_PROMPT.format(question=question, context=context)
        response = self.model.with_structured_output(self.GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

    def rewrite_question(self, state: AgentState) -> dict:
        """Rewrite question using LLM."""
        messages = state["messages"]
        question = messages[0].content
        prompt = RAGPrompts.REWRITE_PROMPT.format(question=question)
        response = self.model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}

    def generate_answer(self, state: AgentState):
        """Generate answer using LLM."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = RAGPrompts.GENERATE_PROMPT.format(question=question, context=context)
        response = self.model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}
