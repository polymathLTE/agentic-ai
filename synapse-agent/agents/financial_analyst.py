# agents/financial_analyst.py
from langchain_core.prompts import ChatPromptTemplate
from config import LLM_GEMINI
from tools import VECTOR_SEARCH_TOOL, STOCK_PRICE_TOOL

def create_financial_analyst(state):
    """
    The analyst agent that synthesizes the final report.
    This is no longer a tool-calling agent but a direct, two-step process.
    """
    print("---FINANCIAL ANALYST: SYNTHESIZING REPORT---")
    
    # STEP 1: Explicitly call the tools to gather context.
    # -----------------------------------------------------
    
    # Retrieve news context from the vector database
    print("Gathering news context from vector database...")
    context = VECTOR_SEARCH_TOOL.invoke({"query": state['original_query']})
    
    # Retrieve stock price context if tickers are present
    stock_context = ""
    tickers = state.get('research_plan', {}).get('stock_tickers', [])
    if tickers:
        print(f"Gathering stock data for tickers: {tickers}")
        for ticker in tickers:
            stock_context += f"\n\n{STOCK_PRICE_TOOL.invoke({'ticker': ticker})}"
            
    # STEP 2: Synthesize the report using the gathered context.
    # ---------------------------------------------------------
    
    report_prompt = ChatPromptTemplate.from_template(
        """You are a master financial analyst. Your task is to write a clear, concise, and insightful report based *only* on the provided context.

        Here is the user's original query:
        {original_query}

        Here is the context you have gathered from your tools:
        ---
        News Context: {context}
        ---
        Stock Market Context: {stock_context}
        ---

        Synthesize this information into a final report. Do not mention your tools or the context directly. Just provide the report.
        If the context is empty or unhelpful, state that you could not find relevant information.
        """
    )

    chain = report_prompt | LLM_GEMINI
    
    report = chain.invoke({
        "original_query": state['original_query'],
        "context": context,
        "stock_context": stock_context if stock_context else "No stock data requested."
    })
    
    state['final_report'] = report.content
    return state
