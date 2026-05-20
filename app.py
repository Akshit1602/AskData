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

@st.cache_resource
def setup_database():
    """
    Sets up an in-memory SQLite database, loads data from CSVs,
    and returns the standard sqlite3 connection object.
    """
    # Setup in-memory SQLite database using the standard library
    # check_same_thread=False is required for multi-threaded access in Streamlit
    conn = sqlite3.connect(":memory:", check_same_thread=False)

    # Read data from CSV files
    shell_dim_df = pd.read_csv("Shell__dim_station__preview_.csv")
    shell_fact_df = pd.read_csv("Shell__fact_station_day_product__preview_.csv")

    # Convert date format to be SQLite compatible
    shell_fact_df["date"] = pd.to_datetime(shell_fact_df["date"], format="%d-%m-%Y").dt.strftime("%Y-%m-%d")

    # Write DataFrames to tables using the sqlite3 connection
    shell_dim_df.to_sql("dim_station", conn, if_exists="replace", index=False)
    shell_fact_df.to_sql("fact_station", conn, if_exists="replace", index=False)

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
# 5. SCHEMA METADATA FOR THE LLM
# --------------------------------------------------------------------------

# Column descriptions and relationships to provide context to the LLM
dim_station_column_descriptions = {
    "station_id": "Unique identifier for each Shell fuel station (e.g., BLR-001).",
    "station_name": "Display name of the station used for business reporting.",
    "city": "Metro/urban location where the station operates.",
    "cluster": "Operational grouping of stations for performance management.",
    "latitude": "Geographic latitude of the station, useful for mapping and routing.",
    "longitude": "Geographic longitude of the station.",
    "opened_year": "Year the station started operations, helpful for lifecycle and maturity analysis.",
    "has_ev_charger": "Indicates if the station supports EV charging (1 = Yes, 0 = No).",
    "cstore_size_sqft": "Size of the Shell convenience store in square feet, used for revenue potential analysis."
}

fact_station_column_descriptions = {
    "date": "Daily date for transaction reporting in YYYY-MM-DD format.",
    "station_id": "Unique identifier linking to dim_station for each Shell site.",
    "city": "City-level filter for localized performance analytics.",
    "product_family": "Fuel type sold: Petrol, Diesel, or Premium.",
    "shell_price_inr_per_liter": "Shell retail selling price per liter in INR.",
    "comp_min_price_inr_per_liter_within_3km": "Minimum competitor price detected within a 3km radius.",
    "price_gap_inr_per_liter": "Pricing difference: Shell price minus competitor minimum price (positive → Shell is pricier).",
    "liters_sold": "Total fuel volume sold for the product that day (in liters).",
    "revenue_inr": "Total revenue generated from fuel sales in INR.",
    "gross_margin_inr": "Gross margin earned on fuel sales for the day in INR.",
    "downtime_minutes": "Pump/equipment downtime duration impacting sales opportunity.",
    "stockout_flag": "Indicates if a product was unavailable for any duration (1 = Stockout, 0 = Normal).",
    "promo_active": "Indicates whether a discount/offer/promotion was active (1 = Yes, 0 = No).",
    "competitors_within_3km": "Number of competitor stations competing for the same catchment area.",
    "weather_heat_index": "Approximate temperature/humidity index that influences fuel demand.",
    "rainfall_mm": "Rainfall amount that may impact footfall and demand fluctuations.",
    "holiday_flag": "Marks national/major holidays that drive demand changes (1 = Holiday).",
    "footfall_estimate": "Estimated number of customers visiting the station on that day.",
    "cstore_transactions": "Number of completed transactions in the convenience store.",
    "cstore_revenue_inr": "Revenue generated from non-fuel C-store sales.",
    "loyalty_signups": "Number of new enrollments into Shell loyalty programmes.",
    "ev_charger_sessions": "Count of EV charging sessions (if facility exists)."
}

relationships = {
    "fact_station_day_product": {
        "station_id": {
            "references": {
                "table": "dim_station",
                "column": "station_id"
            },
            "relationship_type": "many_to_one"
        }
    }
}

# Provide table schema information to the LangChain SQLDatabase object
table_info_combined = (
    "dim_station(station_id, station_name, city, cluster, latitude, longitude, opened_year, has_ev_charger, cstore_size_sqft)\n"
    "fact_station(date, station_id, city, product_family, shell_price_inr_per_liter, "
    "comp_min_price_inr_per_liter_within_3km, price_gap_inr_per_liter, liters_sold, revenue_inr, gross_margin_inr, "
    "downtime_minutes, stockout_flag, promo_active, competitors_within_3km, weather_heat_index, rainfall_mm, "
    "holiday_flag, footfall_estimate, cstore_transactions, cstore_revenue_inr, loyalty_signups, ev_charger_sessions)\n"
)
db.get_context()["table_info"] = table_info_combined

# --------------------------------------------------------------------------
# 6. CORE FUNCTIONS FOR QUERY AND INSIGHT GENERATION
# --------------------------------------------------------------------------

def generate_sql_query(question, history=None, top_k=50):
    """Generates an SQL query from a natural language question using an LLM, considering conversation history."""

    history_context = ""
    if history:
        history_context = "## Conversation History:\n"
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                history_context += f"User: {content}\n"
            elif role == "assistant" and isinstance(content, dict):
                sql = content.get("sql", "")
                # Recreate sample from JSON if available to keep context manageable
                try:
                    df_sample = pd.read_json(StringIO(content["dataframe"])).head(5).to_string()
                except:
                    df_sample = "No data available."
                history_context += f"Assistant SQL: {sql}\nAssistant Result (sample):\n{df_sample}\n"

    user_prompt = f"""
    You are an expert data analyst for Shell Retail (India) working with SQLite.

    {history_context}

    Here is the current user question:
    {question}

    Here are the details of the dataset:
    table_name1 = 'dim_station' and column description = {dim_station_column_descriptions}
    table_name2 = 'fact_station' and column description = {fact_station_column_descriptions}

    ## Relationships:
    Use ONLY the relationships provided here (many-to-one):
    {relationships}
    - fact_station.station_id → dim_station.station_id

    ## Consider the relevant table info:
    {table_info_combined}

    Strict Rules:
    - Use ONLY the tables/columns listed above. Do not invent tables or columns.
    - Enforce ONLY the relationship defined above when joining.
    - Dates are TEXT in 'YYYY-MM-DD'. Use SQLite date helpers when needed (e.g., DATE('now','start of month')).
    - Return a single valid SQLite SELECT query. No comments, no explanations, no markdown fences.
    - Use camelcase for city names.
    - Prefer explicit column lists; avoid SELECT *.
    - If the user asks for “Top/Bottom N,” use ORDER BY and LIMIT.
    - Round off all numerical values to two decimal places.
    - Unless the user specifies otherwise, LIMIT results to {top_k}.
    """
    
    response = llm.invoke([
        ("system", "You are a Shell Retail SQLite expert. Given an input question, conversation history and schema, return ONLY a syntactically correct SQLite SELECT query that follows the provided relationships and rules. Use the history to resolve ambiguities in the current question."),
        ("human", user_prompt)
    ])
    
    sql_query = response.content
    # Clean up potential markdown formatting from the LLM response
    return sql_query.replace('sql', '').replace('`', '').strip()

def generate_visualization_config(question, df):
    """Generates a list of Plotly chart configurations based on the user question and data."""
    if df.empty:
        return []

    # Prepare a sample of the data and column types
    data_sample = df.head(5).to_dict(orient='records')
    column_info = df.dtypes.apply(lambda x: str(x)).to_dict()

    viz_prompt = f"""
    You are a data visualization expert. Based on the user's question and the provided data sample, suggest the most appropriate Plotly Express charts.

    User Question: {question}

    Data Sample (first 5 rows):
    {data_sample}

    Column Data Types:
    {column_info}

    Rules:
    - Use ONLY the following chart types: 'line', 'bar', 'pie'.
    - Follow these cues:
        - Trends over time -> 'line'
        - Comparisons between categories -> 'bar'
        - Proportions or percentages of a total -> 'pie'
    - Return a JSON list of objects. Each object must have:
        - 'type': One of ['line', 'bar', 'pie']
        - 'x': Column name for x-axis (required for line and bar)
        - 'y': Column name for y-axis (required for line and bar)
        - 'values': Column name for values (required for pie)
        - 'names': Column name for labels (required for pie)
        - 'color': Optional column name for color coding
        - 'title': A descriptive title for the chart
    - Stick to as few charts as possible (usually 1, maximum 2).
    - If the data is not suitable for any of these charts (e.g., only one row and one column, or no clear categories/trends), return an empty list [].
    - Ensure the column names used exactly match the data sample.
    - Return ONLY the JSON list. No explanations, no markdown fences.
    """

    response = llm.invoke([
        ("system", "You are a data visualization expert for Shell Retail. Return ONLY a JSON list of Plotly chart configurations."),
        ("human", viz_prompt)
    ])

    try:
        # Attempt to parse the response as JSON
        content = response.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()

        config = json.loads(content)
        return config if isinstance(config, list) else []
    except Exception as e:
        # In case of parsing error, return an empty list
        return []

def generate_insight(question, result):
    """Generates a business insight from the query result using an LLM."""
    insight_prompt = f"""
    You are a senior Shell Retail business performance analyst.
    Your job is to interpret SQL results and explain *why* performance is the way it is,
    connecting data patterns to pricing, competition, sales leakage, conversion, and operational excellence.

    Context:
    - Audience: Shell Retail Operations and Cluster Leadership Teams.
    - These insights inform business reviews and decision boards.
    - You have deep knowledge of station operations, pricing dynamics, and KPIs.

    Inputs:
    • Business Question:
    {question}

    • SQL Output (tabular data):
    {result}

    Your Task:
    Generate an analytical summary that answers the question comprehensively.
    Go beyond reporting numbers — explain *why* these numbers look this way,
    which operational or market factors may be influencing them,
    and what specific actions Shell teams should consider next.

    Guidelines:
    1. Start with a brief snapshot of the key trend or variance.
    2. Diagnose likely drivers across these pillars:
       - Pricing & Competitiveness (price gap vs. nearby competition)
       - Customer Demand (footfall, conversion, loyalty signups)
       - Operational Efficiency (downtime, stockouts, EV charger usage)
       - Promotional Effectiveness (active promos vs. uplift)
       - Network Health (cluster performance, city-wise variance)
    3. Quantify differences or gaps where possible.
    4. End with 2–3 concrete recommendations or next steps:
       - Pricing adjustments
       - Promo targeting
       - Stockout reduction
       - EV charger optimization
       - Uptime improvements

    Rules:
    - Be concise and structured — 3–6 bullet points total.
    - Use the data in the result table; never speculate beyond it.
    - Avoid technical SQL language or metadata.
    - No generic “monitor further” statements; give specific, actionable insights.
    - Assume readers know Shell’s business context but not raw numbers.

    Output Format:
    📊 **What’s happening**
    📉 **Why it’s happening (drivers & root causes)**
    🎯 **Recommended business actions**
    """
    response = llm.invoke(
        [
            ("system", "You are a data-driven business analyst skilled at diagnostic storytelling for Shell Retail leadership."),
            ("human", insight_prompt)
        ]
    )
    return response.content

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

        # Capture current history before adding the new prompt
        history = st.session_state.messages.copy()
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Generating response..."):
            # Generate SQL query with history
            sql_query = generate_sql_query(prompt, history=history)
            
            # Execute query using the direct sqlite3 connection
            try:
                df = pd.read_sql_query(sql_query, db_connection)
            except Exception as e:
                st.error(f"Failed to execute query: {e}")
                df = pd.DataFrame() # Ensure df is initialized on error

            # Generate visualization configuration
            viz_configs = []
            if not df.empty:
                viz_configs = generate_visualization_config(prompt, df)

            # Generate insight based on the result
            if not df.empty:
                insight = generate_insight(prompt, df.to_string())
            else:
                insight = "The query returned no results, so no insights could be generated."
            
            # Display assistant response in the chat message container
            with st.chat_message("assistant"):
                with st.expander("Show SQL Query"):
                    st.code(sql_query, language="sql")
                with st.expander("Show Tabular Output"):
                    st.dataframe(df)

                if viz_configs:
                    with st.expander("Show Visualizations"):
                        render_charts(viz_configs, df)

                st.markdown(insight)
                
            # Add the full assistant response to chat history
            full_response = {
                "sql": sql_query,
                "dataframe": df.to_json(), # Store dataframe as JSON string
                "insight": insight,
                "visualizations": viz_configs
            }
            st.session_state.messages.append({"role": "assistant", "content": full_response})