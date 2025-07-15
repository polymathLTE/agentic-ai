# tools/rss.py
"""
Defines the RSS tool for importing articles from a specific feed URL.
"""
import datetime
import feedparser
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from tools.common import upsert_document

class RssQuery(BaseModel):
    """Input model for the RSS feed tool."""
    feed_url: str = Field(..., description="A valid public RSS or Atom feed URL.")

def import_from_rss(feed_url: str) -> str:
    """
    Parses and ingests the latest 20 items from any public RSS or Atom feed
    into the vector store.
    """
    try:
        parsed_feed = feedparser.parse(feed_url)
        if parsed_feed.bozo:
            return f"Error parsing RSS feed: {parsed_feed.bozo_exception}"

        entries = parsed_feed.entries[:20]
        if not entries:
            return "No items found in the RSS feed."

        for entry in entries:
            # Get the published or updated date, falling back to now.
            date_struct = entry.get("published_parsed") or entry.get("updated_parsed")
            dt_obj = datetime.datetime(*date_struct[:6]) if date_struct else datetime.datetime.now()
            
            upsert_document(entry.title, entry.link, dt_obj, "rss")

        return f"Loaded {len(entries)} items from the RSS feed into the vector store."
    except Exception as e:
        return f"An unexpected error occurred while processing the RSS feed: {e}"

RSS_TOOL = StructuredTool.from_function(
    func=import_from_rss,
    name="rss_feed_import",
    description="Pulls the latest 20 items from a specific public RSS/Atom feed. Use this for highly specialized outlets when the user provides a URL.",
    args_schema=RssQuery,
)
