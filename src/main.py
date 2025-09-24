from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.agents.doc_writer import DocWriterAgent
from core.agents.paper_writing import PaperWritingTeamAgent
from core.agents.rag import RagAgent
from infra.llm_clients.groq import get_llm

# Get LLM instance
llm = get_llm()

# For Doc Writer agent
doc_writer = DocWriterAgent(llm)
query = "Write a document about cats."
result = doc_writer.graph.invoke({"messages": [HumanMessage(content=query)]})
print(result)

# For Paper writing Team
paper_team = PaperWritingTeamAgent(llm)
team_result = paper_team.graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Write an outline for poem about cats and then write the poem to disk."
            )
        ]
    }
)
print(team_result)

# For RAG
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=30
)
doc_splits = text_splitter.split_documents(docs_list)

rag_agent = RagAgent(llm, doc_splits)
rag_result = rag_agent.graph.invoke(
    {"messages": [HumanMessage(content="What does Lilian Weng say about types of reward hacking?")]}
)
print("RAG result:", rag_result)


