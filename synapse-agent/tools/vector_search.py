# tools/vector_search.py
"""
Defines the tool for searching the local vector database.
"""
import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from config import VECTOR_DB

class VectorQuery(BaseModel):
    """Input model for the vector database search tool."""
    query: str = Field(..., description="The user's original question or a query to search the vector store with.")

def search_vector_store(query: str) -> str:
    """
    Searches the local vector store for relevant documents to answer a question.
    Filters results to include only documents from the last 30 days.
    """
    # Calculate the Unix timestamp for 30 days ago for numerical filtering.
    thirty_days_ago_ts = int((datetime.datetime.now() - datetime.timedelta(days=30)).timestamp())

    # Use the 'timestamp' field with the $gt operator for a valid numerical comparison.
    docs = VECTOR_DB.similarity_search(
        query, k=8, filter={"timestamp": {"$gt": thirty_days_ago_ts}}
    )

    if not docs:
        return "No relevant documents were found in the local vector store. Use other tools to load information first."

    # Format the retrieved documents for the agent.
    return "\n\n".join(
        f"Source: {d.metadata.get('source', 'N/A')} ({d.metadata.get('date', 'N/A')})\n"
        f"Content: {d.page_content}\nURL: {d.metadata.get('url', 'N/A')}"
        for d in docs
    )

VECTOR_SEARCH_TOOL = StructuredTool.from_function(
    func=search_vector_store,
    name="vector_database_search",
    description="After loading info with other tools, use this to find context to answer the user's final question.",
    args_schema=VectorQuery,
)
