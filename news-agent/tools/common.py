# tools/common.py
"""
A module for common, shared utility functions used by different tools.
"""
import datetime
from langchain_core.documents import Document
from config import VECTOR_DB

def upsert_document(text: str, url: str, dt_obj: datetime.datetime | datetime.date | str, src: str) -> None:
    """
    Creates a LangChain Document with standardized metadata and upserts it
    into the Chroma vector store.

    Handles date parsing from different formats.
    """
    # Create a Unix timestamp (integer) for numerical filtering in Chroma
    if isinstance(dt_obj, datetime.datetime):
        timestamp = int(dt_obj.timestamp())
        date_str = dt_obj.strftime('%Y-%m-%d')
    elif isinstance(dt_obj, datetime.date):
        timestamp = int(datetime.datetime.combine(dt_obj, datetime.time.min).timestamp())
        date_str = dt_obj.strftime('%Y-%m-%d')
    elif isinstance(dt_obj, str):
        # Attempt to parse common string date formats (e.g., from GDELT)
        try:
            parsed_dt = datetime.datetime.strptime(dt_obj, '%Y%m%d%H%M%S')
            timestamp = int(parsed_dt.timestamp())
            date_str = parsed_dt.strftime('%Y-%m-%d')
        except ValueError:
            # Fallback if parsing fails
            timestamp = int(datetime.datetime.now().timestamp())
            date_str = datetime.date.today().strftime('%Y-%m-%d')
    else:
        # Fallback for unexpected types
        timestamp = int(datetime.datetime.now().timestamp())
        date_str = datetime.date.today().strftime('%Y-%m-%d')


    doc = Document(
        page_content=text,
        metadata={"url": url, "date": date_str, "timestamp": timestamp, "source": src}
    )
    VECTOR_DB.add_documents([doc])
