from __future__ import annotations
import os
import datetime
import requests
import feedparser
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool, Tool
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------#
# 1.  Initialise services
# ------------------------------------------------------------------#
# Use a fast, free, and effective local model for creating embeddings.
EMBED = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Local vector store for persisting ingested news and documents.
VDB = Chroma(
    collection_name="news_plus_static",
    persist_directory="./news_chroma",
    embedding_function=EMBED
)

# Initialize the Google Gemini Pro model for agentic reasoning.
# It's powerful, has a large context window, and excels at tool use.
LLM_GEMINI = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0,
    #api_version="v1"
    #convert_system_message_to_human=True # Recommended for Gemini
)

# ------------------------------------------------------------------#
# 2.  Helper to push docs into Chroma with standardized metadata
# ------------------------------------------------------------------#
def _upsert(text: str, url: str, date: str, src: str) -> None:
    """Creates a LangChain Document and upserts it into the Chroma vector store."""
    # Ensure date is in YYYY-MM-DD format, taking first 10 chars.
    iso_date = date[:10] if isinstance(date, str) else date.isoformat()
    doc = Document(
        page_content=text,
        metadata={"url": url, "date": iso_date, "source": src}
    )
    VDB.add_documents([doc])

# ------------------------------------------------------------------#
# 3.  NEWSAPI – Structured Tool for mainstream news
# ------------------------------------------------------------------#
class NewsQuery(BaseModel):
    query: str = Field(..., description="News search terms, e.g., 'interest rate hikes'")
    from_date: str = Field(..., description="Earliest date to search from in YYYY-MM-DD format.")

def _newsapi_tool(query: str, from_date: str) -> str:
    """Fetch headlines from the NewsAPI 'Everything' endpoint."""
    if not os.getenv("NEWSAPI_KEY"):
        return "NEWSAPI_KEY environment variable not set. Cannot use this tool."
    url = "https://newsapi.org/v2/everything"
    try:
        r = requests.get(url, params={
            "q": query, "from": from_date, "sortBy": "publishedAt",
            "language": "en", "pageSize": 30, "apiKey": os.getenv("NEWSAPI_KEY")
        }, timeout=30).json()

        if r.get("status") != "ok":
            return f"Error from NewsAPI: {r.get('message', 'Unknown error')}"

        articles = r.get("articles", [])
        for art in articles:
            # Some articles lack descriptions; handle gracefully.
            content = f"{art.get('title', '')} – {art.get('description', '')}"
            if content != " – ":
                _upsert(content, art["url"], art["publishedAt"], "newsapi")
        return f"Loaded {len(articles)} articles from NewsAPI into the vector store."
    except requests.RequestException as e:
        return f"Failed to connect to NewsAPI: {e}"

NEWS_TOOL = StructuredTool.from_function(
    func=_newsapi_tool,
    name="news_api_search",
    description="Searches a 1 month archive of real-time headlines from 150k outlets. Best for mainstream English-language news.",
    args_schema=NewsQuery,
)

# ------------------------------------------------------------------#
# 4.  GDELT – Structured Tool for global news
# ------------------------------------------------------------------#
class GdeltQuery(BaseModel):
    query: str = Field(..., description="Search string, e.g., 'global supply chain disruptions'")

def _gdelt_tool(query: str) -> str:
    """Fetch articles from the GDELT project's document API."""
    url = "https://api.gdeltproject.org/api/v2/doc/docsearch"
    try:
        r = requests.get(url, params={
            "query": query, "mode": "ArtList", "maxrecords": 30, "format": "json"
        }, timeout=30).json()
        articles = r.get("articles", [])
        for art in articles:
            _upsert(art["title"], art["url"], art["seendate"], "gdelt")
        return f"Loaded {len(articles)} articles from GDELT into the vector store."
    except requests.RequestException as e:
        return f"Failed to connect to GDELT: {e}"

GDELT_TOOL = StructuredTool.from_function(
    func=_gdelt_tool,
    name="gdelt_search",
    description="Searches global web news in 100+ languages, updated every 15 mins. Great for fast-breaking, international, or non-US sources.",
    args_schema=GdeltQuery,
)

# ------------------------------------------------------------------#
# 5.  RSS – Structured Tool for specialist sources
# ------------------------------------------------------------------#
class RssQuery(BaseModel):
    feed_url: str = Field(..., description="A valid public RSS or Atom feed URL.")

def _rss_tool(feed_url: str) -> str:
    """Parses and ingests items from any public RSS or Atom feed."""
    try:
        parsed = feedparser.parse(feed_url)
        if parsed.bozo: # Indicates a potential problem with the feed
             return f"Error parsing RSS feed. It might be malformed or unavailable. Reason: {parsed.bozo_exception}"

        entries = parsed.entries[:20]
        for ent in entries:
            # Find the best available date, defaulting to today.
            date = ent.get("published_parsed") or ent.get("updated_parsed")
            if date:
                iso_date = datetime.datetime(*date[:6]).isoformat()
            else:
                iso_date = datetime.date.today().isoformat()
            _upsert(ent.title, ent.link, iso_date, "rss")
        return f"Loaded {len(entries)} items from RSS feed into the vector store."
    except Exception as e:
        return f"An unexpected error occurred while processing the RSS feed: {e}"


RSS_TOOL = StructuredTool.from_function(
    func=_rss_tool,
    name="rss_import",
    description="Pulls the latest 20 items from a public RSS/Atom feed (e.g., from the FT, ECB, IMF blogs, or specific government agencies). Use for specialist outlets.",
    args_schema=RssQuery,
)

# ------------------------------------------------------------------#
# 6.  Vector search retriever – Final Answer Tool
# ------------------------------------------------------------------#
class VectorQuery(BaseModel):
    query: str = Field(..., description="The user's original question or a query to search the vector store with.")

def _vector_search(query: str) -> str:
    """Searches the local vector store for relevant documents to answer a question."""
    one_month_ago = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
    # Retrieve the 8 most relevant documents from the last 30 days.
    docs = VDB.similarity_search(
        query, k=8, filter={"date": {"$gt": one_month_ago}}
    )
    if not docs:
        return "No relevant documents found in the local vector store for that query. You may need to use another tool to load some news first."
    # Format documents for the LLM to easily parse.
    return "\n\n".join(
        f"Source: {d.metadata['source']} ({d.metadata['date']})\nContent: {d.page_content}\nURL: {d.metadata['url']}"
        for d in docs
    )

VEC_TOOL = StructuredTool.from_function(
    func=_vector_search,
    name="vector_database_search",
    description="After loading news with other tools, use this tool to search the local vector store for context to answer the user's final question.",
    args_schema=VectorQuery,
)

# Define the list of tools available to the agent.
TOOLS: list[Tool] = [NEWS_TOOL, GDELT_TOOL, RSS_TOOL, VEC_TOOL]

# ------------------------------------------------------------------#
# 7.  Create the Tool-Calling Agent and Executor
# ------------------------------------------------------------------#
def make_tool_agent(llm) -> AgentExecutor:
    """Creates a modern, tool-calling agent using a structured prompt."""
    # This prompt template is key to guiding the LLM's reasoning process.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful research assistant specializing in finance, policy, and economics. "
                "Your goal is to answer the user's question accurately by using the provided tools. "
                "Follow these steps:\n"
                "1. Analyze the user's question to determine what information is needed.\n"
                "2. Choose the best tool to find that information (NewsAPI for mainstream, GDELT for global, RSS for specialist).\n"
                "3. Call the tool to load relevant articles into the vector store. The tool will return a confirmation message.\n"
                "4. Once you have loaded the necessary information, use the 'vector_database_search' tool with the user's original question to retrieve the most relevant snippets.\n"
                "5. Synthesize the retrieved snippets into a concise, well-written answer. Cite your sources with dates and mention the source (e.g., newsapi, gdelt). Do not make up information."
            ),
            ("user", "{input}"),
            # MessagesPlaceholder is where the agent's internal thought process (tool calls, observations) will go.
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # The core agent constructor. It binds the LLM to the tools and uses the prompt to guide it.
    agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)

    # The AgentExecutor is the runtime that actually invokes the agent and executes the tools.
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True, # Set to False for less console output
        handle_parsing_errors=True # Gracefully handles LLM output errors
    )

# ------------------------------------------------------------------#
# 8.  Public entry-point and interactive loop
# ------------------------------------------------------------------#
if __name__ == "__main__":
    # Ensure necessary API keys are set
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("NEWSAPI_KEY"):
        print("ERROR: Please set the GOOGLE_API_KEY and NEWSAPI_KEY environment variables.")
    else:
        print("Initializing Gemini-powered RAG agent...")
        agent_executor = make_tool_agent(llm=LLM_GEMINI)
        print("Agent ready. Ask a news, policy, or finance question.")
        print("Type 'exit' or 'quit' to end the session.")

        while True:
            try:
                question = input("\n➜  ")
                if question.lower() in ["exit", "quit"]:
                    break
                # The agent needs a dictionary input, with the key matching the prompt's input variable.
                response = agent_executor.invoke({"input": question})
                print("\nAnswer:\n" + response["output"])
            except Exception as e:
                print(f"\nAn error occurred: {e}")