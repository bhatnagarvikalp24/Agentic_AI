"""
Data models and state definitions for the Agentic AI Assistant.
Contains TypedDict definitions and data structures used throughout the application.
"""

from typing import TypedDict, Optional, List, Dict, Any, Union
import pandas as pd
from datetime import datetime


class GraphState(TypedDict):
    """
    Main state definition for LangGraph workflow.
    Contains all the data that flows between nodes in the graph.
    """
    # User input
    user_prompt: str
    doc_loaded: bool
    document_path: Optional[str]
    
    # Routing and prompts
    vanna_prompt: Optional[str]
    fuzzy_prompt: Optional[str]
    route: Optional[str]
    
    # SQL results
    sql_result: Optional[pd.DataFrame]
    sql_query: Optional[str]
    
    # Web search results
    web_links: Optional[List[tuple[str, str]]]
    general_summary: Optional[str]
    
    # Document processing
    updated_doc_path: Optional[str]
    
    # Chart information
    chart_info: Optional[Dict[str, Any]]
    
    # Comparison results
    comparison_summary: Optional[str]
    
    # FAISS knowledge base results
    faiss_summary: Optional[str]
    faiss_sources: Optional[List[tuple[str, str, str]]]  # (doc_name, snippet, path)
    faiss_images: Optional[List[Dict[str, Any]]]


class ChatEntry(TypedDict):
    """
    Structure for chat history entries.
    """
    prompt: str
    title: str
    route: Optional[str]
    result: Optional[Union[pd.DataFrame, List[tuple[str, str]]]]
    sql_query: Optional[str]
    web_links: Optional[List[tuple[str, str]]]
    general_summary: Optional[str]
    comparison_summary: Optional[str]
    timestamp: str
    faiss_summary: Optional[str]
    faiss_sources: Optional[List[tuple[str, str, str]]]
    faiss_images: Optional[List[Dict[str, Any]]]
    pinned: Optional[bool]


class ChartInfo(TypedDict):
    """
    Structure for chart configuration.
    """
    type: str  # "bar", "line", "pie"
    x: str     # x-axis column name
    y: Union[str, List[str]]  # y-axis column name(s)


class DatabaseSchema(TypedDict):
    """
    Structure for database schema information.
    """
    table_name: str
    columns: List[str]
    description: Optional[str]


class SearchResult(TypedDict):
    """
    Structure for search results.
    """
    title: str
    link: str
    snippet: str
    summary: Optional[str]


class FAISSDocument(TypedDict):
    """
    Structure for FAISS document metadata.
    """
    doc_name: str
    content: str
    file_path: Optional[str]
    metadata: Dict[str, Any]


class FAISSImage(TypedDict):
    """
    Structure for FAISS image metadata.
    """
    original_doc: str
    extracted_image_path: str
    caption: Optional[str]
    metadata: Dict[str, Any]


# Constants for state management
STATE_KEYS_SET_AT_ENTRY = [
    "user_prompt", 
    "doc_loaded", 
    "document_path", 
    "vanna_prompt", 
    "fuzzy_prompt",
    "route",
    "sql_result",
    "sql_query",
    "web_links",
    "updated_doc_path",
    "chart_info",
    "comparison_summary",
    "general_summary",
    "faiss_summary", 
    "faiss_sources",
    "faiss_images"
]

# Route types
ROUTE_TYPES = {
    "sql": "Structured data query",
    "search": "External web search", 
    "document": "Document processing",
    "comp": "Comparison analysis",
    "faissdb": "Internal knowledge base"
}

# Chart types
CHART_TYPES = {
    "bar": "Bar Chart",
    "line": "Line Chart", 
    "pie": "Pie Chart"
}

# Money-related keywords for formatting
MONEY_KEYWORDS = [
    "loss", "premium", "amount", "cost", "ibnr", "ult", 
    "total", "claim", "reserve", "payment", "earned", "budget"
]

# Insurance-related keywords
INSURANCE_KEYWORDS = [
    "insurance", "insurer", "claim", "premium", "underwriting",
    "policy", "fraud", "broker", "actuary", "reinsurance", "coverage", 
    "Actuarial", "reserving", "P&L", "Profit and Loss"
]


def create_initial_state(user_prompt: str, doc_path: Optional[str] = None) -> GraphState:
    """
    Create initial GraphState for a new query.
    
    Args:
        user_prompt: The user's input query
        doc_path: Optional path to uploaded document
        
    Returns:
        Initial GraphState dictionary
    """
    return {
        "user_prompt": user_prompt,
        "doc_loaded": doc_path is not None,
        "document_path": doc_path,
        "vanna_prompt": None,
        "fuzzy_prompt": None,
        "route": None,
        "sql_result": None,
        "sql_query": None,
        "web_links": None,
        "updated_doc_path": None,
        "comparison_summary": None,
        "general_summary": None,
        "faiss_summary": None,
        "faiss_sources": None,
        "faiss_images": None,
        "chart_info": None
    }


def create_chat_entry(
    prompt: str,
    title: str,
    route: Optional[str],
    result: Optional[Union[pd.DataFrame, List[tuple[str, str]]]],
    sql_query: Optional[str] = None,
    web_links: Optional[List[tuple[str, str]]] = None,
    general_summary: Optional[str] = None,
    comparison_summary: Optional[str] = None,
    faiss_summary: Optional[str] = None,
    faiss_sources: Optional[List[tuple[str, str, str]]] = None,
    faiss_images: Optional[List[Dict[str, Any]]] = None,
    pinned: bool = False
) -> ChatEntry:
    """
    Create a chat history entry.
    
    Args:
        prompt: User's original prompt
        title: Generated title for the chat
        route: Route taken by the agent
        result: Query results (DataFrame or web links)
        sql_query: Generated SQL query
        web_links: Web search results
        general_summary: Generated summary
        comparison_summary: Comparison analysis
        faiss_summary: FAISS knowledge base summary
        faiss_sources: FAISS document sources
        faiss_images: FAISS images
        pinned: Whether this chat is pinned
        
    Returns:
        ChatEntry dictionary
    """
    return {
        "prompt": prompt,
        "title": title,
        "route": route,
        "result": result,
        "sql_query": sql_query,
        "web_links": web_links,
        "general_summary": general_summary,
        "comparison_summary": comparison_summary,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "faiss_summary": faiss_summary,
        "faiss_sources": faiss_sources,
        "faiss_images": faiss_images,
        "pinned": pinned
    } 