# graph.py
from typing import TypedDict, Dict, List 
from langgraph.graph import StateGraph, END
from agents.research_manager import create_research_manager
from agents.search_specialist import create_search_specialist
from agents.financial_analyst import create_financial_analyst

class AgentState(TypedDict):
    """
    A shared state object passed between agents in the graph.
    """
    original_query: str
    research_plan: Dict
    search_summary: str
    final_report: str
    # This field can be used for more complex routing if needed later
    next_agent: str

def create_agent_graph():
    """
    Creates and compiles the multi-agent graph.
    """
    # Initialize the graph with the state object
    workflow = StateGraph(AgentState)

    # Add the agent nodes to the graph
    workflow.add_node("ResearchManager", create_research_manager)
    workflow.add_node("SearchSpecialist", create_search_specialist)
    workflow.add_node("FinancialAnalyst", create_financial_analyst)

    # Set the entry point of the graph
    workflow.set_entry_point("ResearchManager")

    # Define the edges (the flow of control)
    workflow.add_edge("ResearchManager", "SearchSpecialist")
    workflow.add_edge("SearchSpecialist", "FinancialAnalyst")
    workflow.add_edge("FinancialAnalyst", END) # The analyst is the final step

    # Compile the graph into a runnable object
    app = workflow.compile()
    return app
