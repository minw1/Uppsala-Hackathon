import dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langdetect import detect, DetectorFactory
from pathlib import Path
import json
#merge fixed?

DetectorFactory.seed = 0  # make detection deterministic

dotenv.load_dotenv()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise. The context documents are in Swedish, but it is essential that you answer in the same "
    "language that the question is in."
)
is_first_question = True

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma(
    collection_name="support_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_hackathon_db",  # Where to save data locally, remove if not necessary
)


LANG_FILE = Path(__file__).parent / "languages.json"

try:
    with open(LANG_FILE, "r", encoding="utf-8") as f:
        LANG_NAME = json.load(f)
except FileNotFoundError:
    LANG_NAME = {}

def last_human_text(state) -> str | None:
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
            return getattr(msg, "content", "") or (msg.get("content") if isinstance(msg, dict) else "")
    return None

def detect_lang_code(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"  # TODO: swap this to swedish


graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


llm_with_tools = llm.bind_tools([retrieve])


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def generate(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

graph_builder.add_node(generate)
graph_builder.add_node(tools)

graph_builder.set_entry_point("generate")
graph_builder.add_conditional_edges(
    "generate",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "adz123"}}


def ask_question(question):
    response = graph.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(question)] if is_first_question else [
            HumanMessage(question)]},
        config=config)
    return response['messages'][-1].content