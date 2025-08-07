# Streamlit + LangGraph Agentic AI App
#To be added
##Vanna vs External doc
##Summary with web links
#History logs
#More agentic frontend
#Few shot prompting for all nodes
#Adding charts for vanna

# Latest working:
#VANNA: Show me exposure year wise incurred loss, ultimate loss and IBNR and plot a graph
#COMP: what are the ibnr trends as compared to market average in last five years
#faissdb: what is the process flow of Earmark module
#faissdb: What are the different projects in SPARTA
#Vanna: show me branch wise incurred loss  and plot a pie chart
#SERP: What are the expected incurred losses due to russia ukraine war

# Queries:
# Comp: Compare the Incurred loss with AXA XL competitors for last 5 years
# Vanna: Show me exposure year wise Incurred loss and plot a pie chart
# Search: What are the expected losses due to recent wars
# Vanna: Show me exposure year wise ultimate premium, ultimate loss and loss ratio and plot a graph

# Things we covered since last wednesday:
# 1. Comp Node: 1. Added a comp node to compare the internal data (vanna) with external docs/competitors (serp).
#               2. It will show SQL_Query, SQL output table, General summary, web links, Final Comparing summary  
# 2. Serp Node: Added a general summary along with links. And also a short summary of each of the link.
# 3. Vanna Node: 1. Added a SQl_Query to be printed along with SQL result/table. 
#                2. If a user asks to plot a chart, chart will be printed. If user explicitly mentioned any specific type 
#                   of chart then it will override the LLM's predicted chart. 
# 4. History logs: 1. Added all the chats in form of history with a time stamp. 
#                  2. Removed the user input text box and the "run agent" button when user view any historical tab,
#                     and new button is provided when user wants to ask a new query.
# 5. Routes Visualization Chart: Added new Comp node in the Route workflow chart.


#TO DO
# 1 Hit on enter instead of Run agent button : Done
# 2 Vanna output values to accounting format : Done
# 3 Comp node prompt optimize : Done
# 4 Fetch numerical numbers as well in Serp node : Done
# 5 few shots prompting wherever possible
# 6 follow up questions on historical tabs: Done
# 7 suggest next possible questions by llm : Done
# 8 Same session multiple questions/ new session
# 9. Move route chart from center to right : Done
# 10. Document Node
# DATA on priority
#11: LR
#12: Export PPT
#AvE
#impact of infllation(loss)
#
#Host on streamlit cloud
#PPT for Live + Historical tabs

#Things we covered post 25/07/25
# 1. Improve prompting - provided by MD
# 2. COMP node: passing vanna output to LLM for SERP search for better contextual output. Had to update SERP node as well.
# 3. Trained VANNA on the keywords provided by MD for COMP node
# 4. Passing "documentations" to router LLM on which Vanna was trained so that LLM has rough idea about each column in the table



import streamlit as st
import tempfile
import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from datetime import datetime, date, timedelta
from typing import Optional, List

# Import new modules
from models import (
    GraphState, ChatEntry, ChartInfo, SearchResult, 
    FAISSDocument, FAISSImage, STATE_KEYS_SET_AT_ENTRY,
    MONEY_KEYWORDS, INSURANCE_KEYWORDS, ROUTE_TYPES, CHART_TYPES,
    create_initial_state, create_chat_entry
)
from database import (
    get_schema_description, execute_query, format_money_columns,
    validate_sql_query, get_database_info, DATABASE_DOCUMENTATION,
    get_database_path, is_database_accessible, get_database_summary
)
from prompts import (
    get_follow_up_questions_prompt, get_title_generation_prompt
)
from presentation import generate_ppt

# Import configuration
from config import (
    vn_model, client, embedding, call_llm, prune_state,
    SEARCH_DOMAIN_FILTER, FAISS_INDEX_PATH, SERPAPI_API_KEY
)

# Import LangGraph components
from langgraph_setup import (
    RouterNode, vanna_node, serp_node, comp_node, faissdb_node, 
    document_node, build_langgraph, get_user_chart_type, suggest_chart, plot_chart
)

# Import UI components
from ui.sidebar import (
    render_sidebar, get_active_chat_index, is_viewing_history, 
    is_entering_new_query, get_current_chat_entry, add_chat_entry, 
    clear_active_chat_index, has_chat_history, get_chat_count
)

st.set_page_config(layout="wide")

# ---- Database accessibility check ----
db_path = get_database_path()
if not is_database_accessible(db_path):
    st.error(f"⚠️ Database not accessible at {db_path}. Please ensure the database file exists.")
    st.stop()

# ---- LangGraph State is now imported from models.py ----


# LangGraph components are now imported from langgraph.py

def generate_follow_up_questions(user_prompt: str) -> List[str]:
    followup_prompt = get_follow_up_questions_prompt(user_prompt)
    try:
        response = call_llm(followup_prompt)
        return re.findall(r"^\s*[-–•]?\s*(.+)", response, re.MULTILINE)[:3] or response.split("\n")[:3]
    except:
        return []

def visualize_workflow(active_route=None):
    """
    Visualize the LangGraph workflow with optional route highlighting.
    
    Args:
        active_route: The currently active route to highlight
    """
    route_to_node = {
        "sql": "vanna_sql",
        "search": "serp_search",
        "document": "doc_update",
        "faissdb": "faissdb",
        "comp": "comp"
    }

    highlight_node = route_to_node.get(active_route)

    G = nx.DiGraph()
    edge_styles = {}

    # Define all nodes
    nodes = ["__start__", "router", "vanna_sql", "serp_search", "doc_update", "comp", "faissdb", "__end__"]
    for node in nodes:
        G.add_node(node)

    # Define edges
    edges = [
        ("__start__", "router"),
        ("router", "vanna_sql"),
        ("router", "serp_search"),
        ("router", "doc_update"),
        ("router", "comp"),
        ("router", "faissdb"),
        ("vanna_sql", "__end__"),
        ("serp_search", "__end__"),
        ("doc_update", "__end__"),
        ("comp", "__end__"),
        ("faissdb", "__end__")
    ]

    # Add edges with styles
    for source, target in edges:
        G.add_edge(source, target)
        if source == "router":
            edge_styles[(source, target)] = {"style": "dashed", "color": "gray", "width": 1}
        else:
            edge_styles[(source, target)] = {"style": "solid", "color": "black", "width": 1.5}

    # Highlight the active route in red
    if highlight_node and ("router", highlight_node) in G.edges:
        edge_styles[("router", highlight_node)] = {"style": "solid", "color": "red", "width": 2.5}

    # Positions for nodes
    pos = {
        "__start__": (2, 4),
        "router": (2, 3),
        "vanna_sql": (0, 2),
        "serp_search": (1, 2),
        "doc_update": (2, 2),
        "comp": (3, 2),
        "faissdb": (4, 2),
        "__end__": (2, 1),
    }

    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="skyblue")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Draw styled edges
    for edge in G.edges:
        style = edge_styles.get(edge, {"style": "solid", "color": "black", "width": 1})
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[edge],
            arrows=True,
            arrowstyle='-|>',
            style=style["style"],
            edge_color=style["color"],
            width=style["width"]
        )

    plt.title("Agentic LangGraph Workflow")
    plt.axis("off")
    plt.tight_layout()
    st.pyplot(plt)

# ---- LangGraph Setup ----
agent_graph = build_langgraph()

# ---- Streamlit UI ----
st.title("\U0001F9E0 Agentic AI Assistant (Insurance)")


def generate_title(prompt: str) -> str:
    try:
        title_prompt = get_title_generation_prompt(prompt)
        return call_llm(title_prompt)
    except:
        return prompt[:40] + ("..." if len(prompt) > 40 else "")

# Render the sidebar
render_sidebar()

# Render before running agent (all dashed)
#with st.expander("🧭 Workflow Graph (Initial)"):
#    visualize_workflow(graph_builder)

# ✅ Initialize just_ran_agent flag if not already
if "just_ran_agent" not in st.session_state:
    st.session_state.just_ran_agent = False

# ✅ UI Control Logic: if user is entering a new query
if is_entering_new_query():
    with st.form(key="query_form"):
        user_prompt = st.text_input("Enter your query:")
        doc_file = st.file_uploader("Upload Insurance Document (.docx)", type=["docx"])
        submitted = st.form_submit_button("Run Agent")
    #user_prompt = st.text_input("Enter your query:", key="user_prompt")
    #doc_file = st.file_uploader("Upload Insurance Document (.docx)", type=["docx"])


    if submitted:
    # Only run when prompt is entered and changed
    #if user_prompt and (
    #    "last_prompt" not in st.session_state
    #    or st.session_state["last_prompt"] != user_prompt
    #):
        st.session_state["last_prompt"] = user_prompt

        if doc_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(doc_file.read())
                doc_path = tmp.name
        else:
            doc_path = None

        state: GraphState = create_initial_state(user_prompt, doc_path)

        with st.spinner("Running Agent..."):
            try:
                output = agent_graph.invoke(state)
                st.session_state.output = output
                followups = generate_follow_up_questions(user_prompt)
                st.session_state.followups = followups

            except Exception as e:
                st.error(f"Agent crashed due to error: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

        # ✅ Save to chat history
        chat_entry = create_chat_entry(
            prompt=user_prompt,
            title=generate_title(user_prompt),
            route=output.get("route"),
            result=output.get("sql_result") if output.get("route") in ["sql", "document", "comp"] else output.get("web_links"),
            sql_query=output.get("sql_query"),
            web_links=output.get("web_links"),
            general_summary=output.get("general_summary"),
            comparison_summary=output.get("comparison_summary"),
            faiss_summary=output.get("faiss_summary"),
            faiss_sources=output.get("faiss_sources"),
            faiss_images=output.get("faiss_images")
        )

        add_chat_entry(chat_entry)
        st.session_state.active_chat_index = len(st.session_state.chat_history) - 1
        st.session_state.just_ran_agent = True

        col_left, col_mid, col_right = st.columns([4, 0.4 ,1.5])

        with col_right:
            if st.session_state.get("output"):
                st.markdown("### 🧭 Workflow Diagram")
                visualize_workflow(active_route=st.session_state["output"].get("route"))


        with col_left:
            # ✅ Output rendering
            if output.get("route") in ["sql", "document", "comp"] and output.get("sql_result") is not None:
                st.subheader("SQL Query Result:")
                if output.get("sql_query"):  # For live session
                    st.code(output["sql_query"], language="sql")
                try:
                    sql_df = output["sql_result"]
                    if isinstance(sql_df, pd.DataFrame):
                        formatted_df = format_money_columns(sql_df, MONEY_KEYWORDS)
                        st.dataframe(formatted_df)
                    else:
                        st.write("Raw SQL output:")
                        st.write(sql_df)
                except Exception as e:
                    st.warning(f"Could not display table properly: {e}")
                    st.write(output["sql_result"])

                if any(word in output["user_prompt"].lower() for word in ["plot", "draw", "visualize", "chart", "bar graph", "line graph", "pie chart", "graph"]):
                    user_chart_type = get_user_chart_type(output["user_prompt"])
                    chart_info = suggest_chart(sql_df)
                    
                    if chart_info and user_chart_type:
                        chart_info["type"] = user_chart_type

                    if chart_info:
                        try:
                            plot_chart(sql_df, chart_info)
                        except Exception as e:
                            st.warning(f"Could not render chart: {e}")

            if output.get("route") in ["search", "comp"] and output.get("web_links"):
                st.subheader("🧠 General Summary:")
                summary = output.get("general_summary")
                if summary and summary.strip().lower() != "none":
                    st.markdown(summary)
                else:
                    st.markdown("_No summary could be generated from the results._")

                st.subheader("🔗 Top Web Links:")
                for i, (link, summary) in enumerate(output["web_links"], 1):
                    st.markdown(f"**{i}.** {link}")
                    st.markdown(f"_Summary:_\n{summary}")

            if output.get("route") == "comp" and output.get("comparison_summary"):
                st.subheader("🆚 Comparison Summary:")
                st.markdown(output["comparison_summary"])
            
            if output.get("route") == "faissdb":
                st.subheader("📘 Internal Knowledge Base Answer:")
                st.markdown(output.get("faiss_summary", "_No summary available._"))

                # Show images related to the most similar doc
                if output.get("faiss_images"):
                    most_similar_doc = output["faiss_sources"][0][0]  # get docname
                    st.subheader(f"🖼️ Images from: {most_similar_doc}")
                    for meta in output["faiss_images"]:
                        if meta["original_doc"] == most_similar_doc:
                            img_path = meta["extracted_image_path"]
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, caption=meta.get("caption", ""), use_container_width=True)

                st.subheader("📄 Document Sources:")
                base_dir = os.path.dirname(__file__)
                for i, (docname, snippet, path) in enumerate(output.get("faiss_sources", []), 1):
                    st.markdown(f"**{i}. {docname}**\n\n{snippet}")
                    #st.code(f"📁 File path: {path}")
                    #st.code(f"🧪 Exists: {os.path.exists(path) if path else 'No path'}")
                    if path:
                        full_path = os.path.join(base_dir, path).replace("\\", "/")
                        if os.path.exists(full_path):
                            with open(full_path, "rb") as f:
                                st.download_button(
                                    label=f"📥 Download {os.path.basename(path)}",
                                    data=f,
                                    file_name=os.path.basename(path),
                                    key=f"download_doc_{i}"
                                )




            if output.get("updated_doc_path"):
                with open(output["updated_doc_path"], "rb") as f:
                    st.download_button("Download Updated Document", f, file_name="updated.docx")

            if st.session_state.get("followups"):
                st.markdown("### 💬 You could also ask:")
                for q in followups:
                    st.markdown(f"- 👉 {q}")

            st.download_button("⬇️ Export to PPT", generate_ppt(chat_entry), file_name="agentic_ai_output.pptx")

            st.session_state.just_ran_agent = False
            st.session_state.active_chat_index = None

else:
    # ✅ If user is viewing previous chat, show message + unlock option
    st.info("📜 You're viewing a previous conversation. Click below to start a new query.")
    if st.button("Start New Query"):
        clear_active_chat_index()
        st.session_state.user_prompt = ""
        st.rerun() 

# ✅ Render selected chat in main area
if is_viewing_history() and not st.session_state.just_ran_agent:
    entry = get_current_chat_entry()
    st.markdown(f"### 📝 Prompt\n{entry['prompt']}")
    st.caption(f"🕒 {entry['timestamp']}")
    st.markdown(f"_Route_: `{entry['route']}`")

    if entry["route"] in ["sql", "document"]:
        st.subheader("SQL Query Result:")
        if entry.get("sql_query"):  # For history view
            st.code(entry["sql_query"], language="sql")
        result_df = entry.get("result")
        if isinstance(result_df, list):  # was serialized
            result_df = pd.DataFrame(result_df)
        if isinstance(result_df, pd.DataFrame):
            formatted_df = format_money_columns(result_df, MONEY_KEYWORDS)
            st.dataframe(formatted_df)
        else:
            st.text(result_df)

    elif entry["route"] == "faissdb":
        st.subheader("📘 Internal Knowledge Base Answer:")
        st.markdown(entry.get("faiss_summary", "_No summary available._"))

        # === Show Associated Images from Top Doc ===
        faiss_images = entry.get("faiss_images", [])
        faiss_sources = entry.get("faiss_sources", [])
        if faiss_images and faiss_sources:
            top_doc = faiss_sources[0][0]
            st.subheader(f"🖼️ Images from: {top_doc}")
            for meta in faiss_images:
                if meta.get("original_doc") == top_doc:
                    img_path = meta.get("extracted_image_path")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=meta.get("caption", ""), use_container_width=True)

        # === Show Document Sources with Download Buttons ===
        st.subheader("📄 Document Sources:")
        base_dir = os.path.dirname(__file__)
        for i, (docname, snippet, path) in enumerate(faiss_sources, 1):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(f"**{i}. {docname}**\n\n{snippet}")
            with col2:
                if path:
                    full_path = os.path.join(base_dir, path).replace("\\", "/")
                    if os.path.exists(full_path):
                        with open(path, "rb") as f:
                            st.download_button(
                                label="⬇️",
                                data=f,
                                file_name=os.path.basename(path),
                                key=f"download_history_{i}"
                            )

    elif entry["route"] == "search":
        if entry.get("general_summary"):
            st.subheader("🧠 General Summary:")
            st.markdown(entry["general_summary"])

        st.subheader("🔗 Top Web Links:")
        for i, (link, summary) in enumerate(entry["result"], 1):
            st.markdown(f"**{i}.** {link}")
            st.markdown(f"_Summary:_\n{summary}")

    elif entry["route"] == "comp":
        # ✅ Show SQL Query
        if entry.get("sql_query"):
            st.subheader("🧾 SQL Query:")
            st.code(entry["sql_query"], language="sql")

        # ✅ Show SQL Result
        st.subheader("SQL Query Result:")
        result_df = entry.get("result")
        if isinstance(result_df, list):  # was serialized
            result_df = pd.DataFrame(result_df)
        if isinstance(result_df, pd.DataFrame):
            formatted_df = format_money_columns(result_df, MONEY_KEYWORDS)
            st.dataframe(formatted_df)
        else:
            st.text(result_df)

        # ✅ Comparison Summary
        if entry.get("comparison_summary"):
            st.subheader("🆚 Comparison Summary:")
            st.markdown(entry["comparison_summary"])

        # ✅ General Summary
        if entry.get("general_summary"):
            st.subheader("🧠 General Summary:")
            st.markdown(entry["general_summary"])

        # ✅ Web Links
        st.subheader("🔗 Top Web Links:")
        web_links = entry.get("web_links")
        for i, (link, summary) in enumerate(web_links or [], 1):
            st.markdown(f"**{i}.** {link}")
            st.markdown(f"_Summary:_\n{summary}")

    ppt_buffer = generate_ppt(entry)
    st.download_button("⬇️ Export to PPT", ppt_buffer, file_name="agentic_ai_output.pptx")
