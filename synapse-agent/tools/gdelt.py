# tools/gdelt.py
"""
Defines the GDELT tool for searching global news articles.
"""
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from tools.common import upsert_document

class GdeltQuery(BaseModel):
    """Input model for the GDELT search tool."""
    query: str = Field(..., description="Search string, e.g., 'global supply chain disruptions'")

def gdelt_news_search(query: str) -> str:
    """
    Fetches news articles from the GDELT project's document API and
    ingests them into the vector store.
    """
    api_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": 30,
        "format": "json",
    }
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        articles = response.json().get("articles", [])
        
        if not articles:
            return "GDELT search returned no articles."

        for art in articles:
            upsert_document(art["title"], art["url"], art["seendate"], "gdelt")
        
        return f"Loaded {len(articles)} articles from GDELT into the vector store."
    except requests.RequestException as e:
        return f"Failed to connect to GDELT: {e}"
    except Exception as e:
        return f"An error occurred during the GDELT search: {e}"

GDELT_TOOL = StructuredTool.from_function(
    func=gdelt_news_search,
    name="gdelt_news_search",
    description="Searches global web news in over 100 languages. Good for international perspectives if Tavily is insufficient.",
    args_schema=GdeltQuery,
)
