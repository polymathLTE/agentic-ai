# agents/research_manager.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import LLM_GEMINI
from typing import TypedDict, Dict, List

class ResearchPlan(BaseModel):
    """
    Structured research plan.
    """
    search_queries: List[str] = Field(description="A list of 3-5 concise search queries for the web.")
    stock_tickers: List[str] = Field(description="A list of any stock ticker symbols mentioned in the query.")

def create_research_manager(state):
    """Creates a structured research plan."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the Research Manager. Your role is to analyze a user's query and create a structured research plan. "
         "Based on the query: '{original_query}', generate a list of search queries for the web search specialist and identify any stock tickers. "
         "Output ONLY the structured plan."),
        ("user", "{original_query}")
    ])
    
    # Use the structured_output feature to force a JSON-like response
    structured_llm = LLM_GEMINI.with_structured_output(ResearchPlan)
    chain = prompt | structured_llm
    
    plan = chain.invoke(state)
    
    state['research_plan'] = plan.dict() # Store the plan as a dictionary
    return state
