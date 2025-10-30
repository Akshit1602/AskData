import os
import streamlit as st
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_sql_query_chain
import openai

# Set up OpenAI environment variables
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


st.title("Shell GenAI Demo  RAG")
st.markdown("Welcome to the Shell GenAI Demo. Ask any question about your data, and the assistant will provide insights.")

@st.cache_resource
def setup_database():
    # Read data from CSV files
    shell_dim_df = pd.read_csv("Shell__dim_station__preview_.csv")
    shell_fact_df = pd.read_csv("Shell__fact_station_day_product__preview_.csv")

    # Convert date format
    shell_fact_df["date"] = pd.to_datetime(shell_fact_df["date"], format="%d-%m-%Y").dt.strftime("%Y-%m-%d")

    # Setup in-memory SQLite database
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = conn.cursor()

    # Create table schemas
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dim_station (
        station_id TEXT PRIMARY KEY,
        station_name TEXT NOT NULL,
        city TEXT NOT NULL,
        cluster TEXT,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        opened_year INTEGER,
        has_ev_charger INTEGER,
        cstore_size_sqft INTEGER
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fact_station (
        date TEXT NOT NULL,
        station_id TEXT NOT NULL,
        city TEXT NOT NULL,
        product_family TEXT NOT NULL,
        shell_price_inr_per_liter REAL NOT NULL,
        comp_min_price_inr_per_liter_within_3km REAL,
        price_gap_inr_per_liter REAL,
        liters_sold INTEGER NOT NULL,
        revenue_inr REAL NOT NULL,
        gross_margin_inr REAL NOT NULL,
        downtime_minutes INTEGER DEFAULT 0,
        stockout_flag INTEGER DEFAULT 0,
        promo_active INTEGER DEFAULT 0,
        competitors_within_3km INTEGER,
        weather_heat_index REAL,
        rainfall_mm REAL,
        holiday_flag INTEGER DEFAULT 0,
        footfall_estimate INTEGER,
        cstore_transactions INTEGER,
        cstore_revenue_inr REAL,
        loyalty_signups INTEGER,
        ev_charger_sessions INTEGER,
        PRIMARY KEY (date, station_id, product_family),
        FOREIGN KEY (station_id) REFERENCES dim_station(station_id)
    );
    """)

    # Write DataFrames to tables
    shell_dim_df.to_sql("dim_station", conn, if_exists="replace", index=False)
    shell_fact_df.to_sql("fact_station", conn, if_exists="replace", index=False)

    conn.commit()

    # Create SQLDatabase object from the existing connection
    db = SQLDatabase(engine=create_engine("sqlite:///:memory:", creator=lambda: conn))
    return db

db = setup_database()

# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    temperature=0
)

# Column descriptions and relationships
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

table_info_combined = (
    "dim_station(station_id, station_name, city, cluster, latitude, longitude, opened_year, has_ev_charger, cstore_size_sqft)\\n"
    "fact_station(date, station_id, city, product_family, shell_price_inr_per_liter, "
    "comp_min_price_inr_per_liter_within_3km, price_gap_inr_per_liter, liters_sold, revenue_inr, gross_margin_inr, "
    "downtime_minutes, stockout_flag, promo_active, competitors_within_3km, weather_heat_index, rainfall_mm, "
    "holiday_flag, footfall_estimate, cstore_transactions, cstore_revenue_inr, loyalty_signups, ev_charger_sessions)\\n"
)
db.get_context()["table_info"] = table_info_combined

def generate_sql_query(question, top_k=50):
    user_prompt = f\"\"\"
    You are an expert data analyst for Shell Retail (India) working with SQLite.

    Here is the user question:
    {question}

    Here are the details of the dataset:
    table_name1 = 'dim_station' and column description = {dim_station_column_descriptions}
    table_name2 = 'fact_station' and column description = {fact_station_column_descriptions}

    ## Relationships:
    Use ONLY the relationships provided here (many-to-one):
    {relationships}
    - fact_station_day_product.station_id → dim_station.station_id

    ## Consider the relevant table info:
    {table_info_combined}

    Strict Rules:
    - Use ONLY the tables/columns listed above. Do not invent tables or columns.
    - Enforce ONLY the relationship defined above when joining.
    - Dates are TEXT in 'YYYY-MM-DD'. Use SQLite date helpers when needed (e.g., DATE('now','start of month')).
    - Return a single valid SQLite SELECT query. No comments, no explanations, no markdown fences.
    - Use camelcase fo city names
    - Prefer explicit column lists; avoid SELECT *.
    - If the user asks for “Top/Bottom N,” use ORDER BY and LIMIT.
    - Round off all the values upto two decimal places.
    - Unless the user specifies otherwise, LIMIT results to {top_k}.
    \"\"\"

    response = llm.invoke([
        ("system", "You are a Shell Retail SQLite expert. Given an input question and schema, return ONLY a syntactically correct SQLite SELECT query that follows the provided relationships and rules."),
        ("human", user_prompt)
    ])

    sql_query = response.content
    return sql_query.replace('sql', '').replace('`', '')

def generate_insight(question, result):
    insight_prompt = f\"\"\"
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
    \"\"\"
    response = llm.invoke(
        [
            ("system", "You are a data-driven business analyst skilled at diagnostic storytelling for Shell Retail leadership."),
            ("human", insight_prompt)
        ]
    )
    return response.content

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if secrets_are_set:
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        sql_query = generate_sql_query(prompt)
        sql_result = db.run(sql_query)
        insight = generate_insight(prompt, sql_result)

        response = f"Assistant: {insight}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(insight)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": insight})
