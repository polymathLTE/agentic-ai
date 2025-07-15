# tools/tavily.py
"""
Defines the Tavily web search tool for real-time information retrieval.
"""
import os
import datetime
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain_core.tools import StructuredTool
from tools.common import upsert_document

class TavilyQuery(BaseModel):
    """Input model for the Tavily web search tool."""
    query: str = Field(..., description="The search query for the Tavily search engine.")

def tavily_web_search(query: str) -> str:
    """
    Uses the Tavily API to search the web, then ingests the results
    into the vector store.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY environment variable not set. Cannot use this tool."

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="advanced", max_results=7)
        results = response.get("results", [])

        if not results:
            return "Tavily search returned no results."

        search_date = datetime.date.today()
        for res in results:
            upsert_document(res["content"], res["url"], search_date, "tavily")

        return f"Loaded {len(results)} search results from Tavily into the vector store."
    except Exception as e:
        return f"An error occurred during the Tavily search: {e}"

TAVILY_TOOL = StructuredTool.from_function(
    func=tavily_web_search,
    name="tavily_web_search",
    description="A powerful web search engine for real-time information. Best for general news, finance, or policy questions.",
    args_schema=TavilyQuery,
)
