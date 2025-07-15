# agents/search_specialist.py
from tools import TAVILY_TOOL

def create_search_specialist(state):
    """
    The search agent that executes the research plan.
    This agent is now a simple, direct tool call.
    """
    print("---SEARCH SPECIALIST: EXECUTING SEARCH---")
    queries = state['research_plan'].get('search_queries', [])
    
    # We'll just use the first query from the plan for simplicity and speed.
    if not queries:
        return {"search_summary": "No search queries were provided in the plan."}

    query = queries[0]
    print(f"Searching with query: {query}")

    # Call the tavily tool directly
    try:
        result = TAVILY_TOOL.invoke({"query": query})
        state['search_summary'] = result
    except Exception as e:
        print(f"Error during Tavily search: {e}")
        state['search_summary'] = "Failed to retrieve search results."
        
    print(f"Search specialist summary: {state['search_summary']}")
    return state
