# tools/stock_tool.py

import yfinance as yf
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class StockQuery(BaseModel):
    """Input model for the stock price tool."""
    ticker: str = Field(..., description="The stock ticker symbol, e.g., 'AAPL' or 'GOOGL'.")

def get_stock_price(ticker: str) -> str:
    """
    Fetches the last closing price and other key info for a given stock ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.history(period="1d")
        if info.empty:
            return f"Could not find data for ticker: {ticker}. It may be delisted or invalid."
        
        last_close = info['Close'].iloc[-1]
        prev_close = info['Close'].iloc[0] if len(info['Close']) > 1 else last_close
        change = last_close - prev_close
        change_percent = (change / prev_close) * 100 if prev_close != 0 else 0

        return (
            f"Latest data for {ticker}:\n"
            f"Closing Price: ${last_close:.2f}\n"
            f"Change: ${change:.2f} ({change_percent:.2f}%)\n"
            f"Day's High: ${info['High'].iloc[-1]:.2f}\n"
            f"Day's Low: ${info['Low'].iloc[-1]:.2f}\n"
            f"Volume: {info['Volume'].iloc[-1]:,}"
        )
    except Exception as e:
        return f"An error occurred while fetching stock data for {ticker}: {e}"

STOCK_PRICE_TOOL = StructuredTool.from_function(
    func=get_stock_price,
    name="get_stock_price",
    description="Fetches the latest closing price and daily trading info for a given stock ticker symbol.",
    args_schema=StockQuery,
)
