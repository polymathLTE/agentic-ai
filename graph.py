# In a new graph.py file
from typing import TypedDict, List

class AgentState(TypedDict):
    original_query: str
    research_plan: str
    search_summary: str
    final_report: str
    # This field could be used for more complex routing
    next_agent: str
