import dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil
import json

# Put your API keys in a .env file in the same directory as this script
dotenv.load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# empty ./chroma_hackathon_db directory before running this script to start fresh
if os.path.exists("./chroma_hackathon_db"):
    shutil.rmtree("./chroma_hackathon_db")

vector_store = Chroma(
    collection_name="support_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_hackathon_db",
)

# load web paths from json
try:
    with open("links.json", "r") as f:
        web_paths = json.load(f)
except FileNotFoundError:
    print("links.json file not found. Please create the file with a list of URLs to scrape.")
    web_paths = ["https://www.uppsala.se/stod-och-omsorg/funktionsnedsattning/stod-och-omsorg-for-dig-med-funktionsnedsattning/", "https://www.uppsala.se/kampanjsidor/en-manad-for-psykisk-halsa/"]

loader = WebBaseLoader(
    web_paths=web_paths,
    bs_get_text_kwargs={"separator": " ", "strip": True}
)

docs = loader.load()

print(f"Total characters in all documents: {sum([len(doc.page_content) for doc in docs])}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

vector_store.add_documents(documents=all_splits)
