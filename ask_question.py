import dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

DetectorFactory.seed = 0  # make detection deterministic

dotenv.load_dotenv()

system_prompt = (
    "You are a question-answering assistant for the municipality of Uppsala. "
    "You are providing useful information and points of contact for people that want to make use of the municipal "
    "support system in Uppsala. Individuals should be able to search for and find relevant "
    "information based on their needs and also receive guidance on what the next step might be, e.g., "
    "applying for a specific measure or contacting a person/organization to find out more. If you don't know answer "
    "answer, say that you don't know. Use three sentences maximum and keep the  answer concise. The context documents "
    "are in Swedish, but it is essential that you answer in the same language that the question is in. If the "
    "question is in English, answer in English."
)
is_first_question = True

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma(
    collection_name="support_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_hackathon_db",  # Where to save data locally, remove if not necessary
)

LANG_FILE = Path(__file__).parent / "config" / "languages.json"

try:
    with open(LANG_FILE, "r", encoding="utf-8") as f:
        LANG_NAME = json.load(f)
except FileNotFoundError:
    LANG_NAME = {}


@tool
def determine_language(state) -> str:
    """Determine the language of the last human/user message."""
    for msg in reversed(state["messages"]):
        if getattr(msg, "type") == "human" or getattr(msg, "role") == "user":
            return LANG_NAME[detect(msg.content)]
    return "Swedish"


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


llm_with_tools = llm.bind_tools([retrieve, determine_language])


# Generate an AIMessage that may include a tool-call to be sent.
def generate(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


tools = ToolNode([determine_language, retrieve])

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


def ask_question(question: str):
    question_lang = detect(question)
    print(f'Please answer in {LANG_NAME[question_lang] if question_lang in LANG_NAME else "Swedish"}.')
    human_message = HumanMessage(question + f'\nPlease answer in {LANG_NAME[question_lang] if question_lang in LANG_NAME else "Swedish"}.')
    # noinspection PyTypeChecker
    response = graph.invoke(
        {"messages": [SystemMessage(system_prompt),
                      AIMessage("Hej, I'm your Support Guide Chatbot from the municipality of Uppsala! I can help you "
                                "find information about municipal support and servies - like activities, contact "
                                "persons or applications. Vi kan också chatta på svenska!"),
                      human_message] if is_first_question else [human_message]},
        config=config)
    return response['messages'][-1].content


print(ask_question("ye kaise kaam krta hai"))
print(ask_question("Pouvez-vous me donner un contact concret pour obtenir du soutien en santé mentale ?"))
