# config.py
"""
Centralized configuration and service initialization.
"""
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
# --- Embedding Model ---
# Initializes the sentence-transformer model for creating vector embeddings.
EMBED_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Vector Database ---
# Initializes the Chroma vector store with a persistent directory.
VECTOR_DB = Chroma(
    collection_name="news_plus_static",
    persist_directory="./news_chroma",
    embedding_function=EMBED_MODEL
)

# --- Large Language Model ---
# Initializes the Google Gemini model for the agent's reasoning capabilities.
LLM_GEMINI = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

