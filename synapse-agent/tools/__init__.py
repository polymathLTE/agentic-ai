# tools/__init__.py
"""
Makes the 'tools' directory a package and exports all tools.
"""
from .tavily import TAVILY_TOOL
from .gdelt import GDELT_TOOL
from .rss import RSS_TOOL
from .vector_search import VECTOR_SEARCH_TOOL
from .stock_tool import STOCK_PRICE_TOOL

# A comprehensive list of all tools available to the agent.
ALL_TOOLS = [TAVILY_TOOL, GDELT_TOOL, RSS_TOOL, VECTOR_SEARCH_TOOL]
