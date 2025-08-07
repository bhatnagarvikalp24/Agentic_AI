import streamlit as st
import json
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

def format_date_label(chat_date: date) -> str:
    """Format date label for chat history grouping."""
    today = date.today()
    if chat_date == today:
        return "Today"
    elif chat_date == today - timedelta(days=1):
        return "Yesterday"
    else:
        return chat_date.strftime("%d %b %Y")

def serialize_chat_history(history: List[Dict[str, Any]]) -> str:
    """Serialize chat history to JSON format, handling pandas DataFrames."""
    safe_history = []
    for chat in history:
        safe_chat = chat.copy()
        if isinstance(safe_chat.get("result"), pd.DataFrame):
            safe_chat["result"] = safe_chat["result"].to_dict(orient="records")
        safe_history.append(safe_chat)
    return json.dumps(safe_history, indent=2)

def render_sidebar():
    """Render the complete sidebar with chat history, download, and clear functionality."""
    
    # Initialize chat history and active index if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "active_chat_index" not in st.session_state:
        st.session_state.active_chat_index = None

    with st.sidebar:
        st.header("🗂️ Session")
        
        # Clear chat history button
        if st.button("🧹 Clear Chat History"):
            clear_chat_history()
            st.success("Chat history cleared!")

        # Group and render chat history
        grouped = {}
        for chat in st.session_state.chat_history:
            chat_date = datetime.strptime(chat["timestamp"], "%d %b %Y, %I:%M %p").date()
            grouped.setdefault(chat_date, []).append(chat)

        # Render grouped chat history
        for group_date in sorted(grouped.keys(), reverse=True):
            label = format_date_label(group_date)
            with st.expander(f"📅 {label}"):
                entries = sorted(grouped[group_date], key=lambda e: not e.get("pinned", False))
                for idx, chat in enumerate(entries):
                    title = chat.get("title") or chat["prompt"][:40]
                    pin_icon = "📌 " if chat.get("pinned") else ""
                    if st.button(f"{pin_icon}{title}", key=f"chat_{group_date}_{idx}"):
                        st.session_state.active_chat_index = st.session_state.chat_history.index(chat)
                        st.session_state.user_prompt = chat["prompt"]
                        st.session_state.just_ran_agent = False

        # Export chat history as JSON
        if has_chat_history():
            history_json = serialize_chat_history(get_chat_history())
            st.download_button(
                "⬇️ Export Chat History", 
                history_json, 
                file_name="chat_history.json",
                mime="application/json"
            )

def get_active_chat_index() -> int:
    """Get the currently active chat index."""
    return st.session_state.get("active_chat_index")

def set_active_chat_index(index: int):
    """Set the active chat index."""
    st.session_state.active_chat_index = index

def clear_active_chat_index():
    """Clear the active chat index."""
    st.session_state.active_chat_index = None

def clear_chat_history():
    """Clear all chat history and reset active index."""
    st.session_state.chat_history = []
    clear_active_chat_index()

def has_active_chat_selected() -> bool:
    """Check if user has selected a chat from history."""
    return st.session_state.active_chat_index is not None

def is_viewing_history() -> bool:
    """Check if user is currently viewing a historical chat."""
    return has_active_chat_selected()

def is_entering_new_query() -> bool:
    """Check if user is entering a new query (not viewing history)."""
    return not has_active_chat_selected()

def get_current_chat_entry():
    """Get the currently selected chat entry from history."""
    if not has_active_chat_selected():
        return None
    return st.session_state.chat_history[st.session_state.active_chat_index]

def is_chat_history_empty() -> bool:
    """Check if chat history is empty."""
    return len(get_chat_history()) == 0

def has_chat_history() -> bool:
    """Check if there are any chats in history."""
    return not is_chat_history_empty()

def get_chat_count() -> int:
    """Get the total number of chats in history."""
    return len(get_chat_history())

def get_chat_history() -> List[Dict[str, Any]]:
    """Get the current chat history."""
    return st.session_state.get("chat_history", [])

def add_chat_entry(chat_entry: Dict[str, Any]):
    """Add a new chat entry to history."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append(chat_entry) 
