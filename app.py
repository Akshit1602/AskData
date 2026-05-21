import os
import streamlit as st
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
import openai
import json
import plotly.express as px
from io import StringIO

# --------------------------------------------------------------------------
# 1. ENVIRONMENT AND API CONFIGURATION
# --------------------------------------------------------------------------

# Set up OpenAI environment variables from Streamlit secrets
try:
    os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"]
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
    os.environ["OPENAI_API_VERSION"] = st.secrets["OPENAI_API_VERSION"]
    os.environ["OPENAI_DEPLOYMENT_NAME"] = st.secrets["OPENAI_DEPLOYMENT_NAME"]
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    secrets_are_set = True
except (KeyError, FileNotFoundError):
    secrets_are_set = False
    st.error("OpenAI API secrets are not configured. Please create a `.streamlit/secrets.toml` file with your credentials.")

# --------------------------------------------------------------------------
# 2. STREAMLIT UI SETUP
# --------------------------------------------------------------------------

st.title("Shell GenAI Demo 💬")
st.markdown("Welcome to the Shell GenAI Demo. Ask any question about your data, and the assistant will provide insights.")

# --------------------------------------------------------------------------
# 3. DATABASE SETUP (Using sqlite3)
# --------------------------------------------------------------------------

from metadata import get_metadata

@st.cache_resource
def setup_database():
    """
    Sets up an in-memory SQLite database, loads data based on metadata,
    and returns the standard sqlite3 connection object.
    """
    # Setup in-memory SQLite database using the standard library
    # check_same_thread=False is required for multi-threaded access in Streamlit
    conn = sqlite3.connect(":memory:", check_same_thread=False)

    metadata = get_metadata()

    for file_info in metadata["files"]:
        path = file_info["path"]
        table_name = file_info["table"]
        fmt = file_info["format"]

        if fmt == "csv":
            df = pd.read_csv(path)
            if "date_col" in file_info:
                date_col = file_info["date_col"]
                date_format = file_info.get("date_format")
                df[date_col] = pd.to_datetime(df[date_col], format=date_format).dt.strftime("%Y-%m-%d")
        elif fmt == "excel":
            sheet_name = file_info.get("sheet_name", 0)
            df = pd.read_excel(path, sheet_name=sheet_name)
        else:
            continue

        df.to_sql(table_name, conn, if_exists="replace", index=False)

    # Return the connection object
    return conn

# --------------------------------------------------------------------------
# 4. INITIALIZATION OF DB CONNECTION, ENGINE, and LLM
# --------------------------------------------------------------------------

# Get the single, populated sqlite3 connection object
db_connection = setup_database()

# Create a SQLAlchemy engine that uses our existing connection.
# The `creator` argument is the key to linking them without creating a new DB.
engine = create_engine("sqlite:///", creator=lambda: db_connection)

# LangChain's SQLDatabase wrapper now uses the engine that points to our data
db = SQLDatabase(engine=engine)

# Initialize Azure OpenAI LLM if secrets are set
if secrets_are_set:
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
        temperature=0
    )

# --------------------------------------------------------------------------
# 5. SCHEMA METADATA AND LANGGRAPH INTEGRATION
# --------------------------------------------------------------------------

from metadata import get_metadata
from graph_logic import create_graph

metadata = get_metadata()

# Provide table schema information to the LangChain SQLDatabase object
db.get_context()["table_info"] = metadata["table_info_combined"]

# Initialize the LangGraph
app_graph = create_graph(db_connection)


# --------------------------------------------------------------------------
# 7. CHART RENDERING HELPER
# --------------------------------------------------------------------------

def render_charts(configs, df):
    """Renders Plotly charts based on the provided configurations and DataFrame."""
    for config in configs:
        try:
            chart_type = config.get('type')
            title = config.get('title', 'Chart')

            if chart_type == 'line':
                fig = px.line(df, x=config.get('x'), y=config.get('y'), color=config.get('color'), title=title)
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'bar':
                fig = px.bar(df, x=config.get('x'), y=config.get('y'), color=config.get('color'), title=title)
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'pie':
                fig = px.pie(df, values=config.get('values'), names=config.get('names'), title=title)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering chart: {e}")

# --------------------------------------------------------------------------
# 8. STREAMLIT CHAT INTERFACE LOGIC
# --------------------------------------------------------------------------

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_history" not in st.session_state:
    st.session_state.graph_history = ""
if "last_dataframe_json" not in st.session_state:
    st.session_state.last_dataframe_json = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif message["role"] == "assistant":
            content = message["content"]
            # Handle the dictionary structure for assistant messages
            if isinstance(content, dict):
                with st.expander("Show SQL Query"):
                    st.code(content["sql"], language="sql")
                with st.expander("Show Tabular Output"):
                    # Recreate DataFrame from stored JSON
                    df = pd.read_json(StringIO(content["dataframe"]))
                    st.dataframe(df)

                # Check for visualizations in the stored message
                if "visualizations" in content and content["visualizations"]:
                    with st.expander("Show Visualizations"):
                        render_charts(content["visualizations"], df)

                st.markdown(content["insight"])
            else:
                st.markdown(content) # For simple string messages

# Main interaction loop
if secrets_are_set:
    # Warning message if conversation history exceeds 5 messages
    if len(st.session_state.messages) >= 10: # 5 turns = 10 messages (user + assistant)
        st.warning("The conversation history is getting long. This might affect the accuracy and performance of the assistant.")

    if prompt := st.chat_input("What is your question?"):
        # Display user message and add to history
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Generating response..."):
            # Prepare initial state for LangGraph
            initial_state = {
                "user_question": prompt,
                "history": st.session_state.graph_history,
                "plan": [],
                "current_step_index": 0,
                "sql_query": None,
                "dataframe_json": st.session_state.last_dataframe_json,
                "visualizations": None,
                "insight": None,
                "retry_count": 0,
                "error": None,
                "final_output": None
            }

            # Invoke the graph
            try:
                result = app_graph.invoke(initial_state)
                final_output = result.get("final_output", {})
                st.session_state.graph_history = result.get("history", "")

                sql_query = final_output.get("sql")
                dataframe_json = final_output.get("dataframe")
                if dataframe_json:
                    st.session_state.last_dataframe_json = dataframe_json
                viz_configs = final_output.get("visualizations", [])
                insight = final_output.get("insight", "No insight generated.")

                df = pd.DataFrame()
                if dataframe_json:
                    df = pd.read_json(StringIO(dataframe_json))

                # Display assistant response in the chat message container
                with st.chat_message("assistant"):
                    if sql_query:
                        with st.expander("Show SQL Query"):
                            st.code(sql_query, language="sql")
                    if not df.empty:
                        with st.expander("Show Tabular Output"):
                            st.dataframe(df)

                    if viz_configs:
                        with st.expander("Show Visualizations"):
                            render_charts(viz_configs, df)

                    st.markdown(insight)

                # Add the full assistant response to chat history
                full_response = {
                    "sql": sql_query,
                    "dataframe": dataframe_json,
                    "insight": insight,
                    "visualizations": viz_configs
                }
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"An error occurred during graph execution: {e}")