import os
import json
import pandas as pd
from io import StringIO
from typing import TypedDict, List, Optional, Literal
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
import sqlite3

# Define the state
class GraphState(TypedDict):
    user_question: str
    history: str  # Summarized history
    plan: List[str]
    current_step_index: int
    sql_query: Optional[str]
    dataframe_json: Optional[str]
    visualizations: Optional[List[dict]]
    insight: Optional[str]
    retry_count: int
    error: Optional[str]
    final_output: Optional[dict]

def get_llm():
    return AzureChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        temperature=0
    )

# --- Node Functions ---

def orchestrator_node(state: GraphState):
    llm = get_llm()
    user_question = state["user_question"]
    history = state["history"]

    prompt = f"""
    You are an orchestrator for a Shell Retail data assistant.
    Analyze the user's question and history to decide the execution plan.
    Available agents:
    - 'sql': For generating and executing SQL queries when new data is needed.
    - 'viz': For generating visualizations from data.
    - 'insight': For generating business insights from data.

    Rules:
    1. If the user asks for a new data query, the plan should be ["sql", "viz", "insight"].
    2. If the user asks for a change in visualization (e.g., "make it a bar chart") and data is already available in history, the plan should be ["viz"].
    3. If the user asks a follow-up that requires new data based on previous results, the plan should be ["sql", "viz", "insight"].
    4. If the question can be answered from existing data/history without a new SQL, skip 'sql'.
    5. Return ONLY a JSON object with the 'plan' key (a list of agent names).

    User Question: {user_question}
    History Summary: {history}
    """

    response = llm.invoke(prompt)
    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        plan_data = json.loads(content)
        plan = plan_data.get("plan", ["sql", "viz", "insight"])
    except:
        plan = ["sql", "viz", "insight"]

    return {
        "plan": plan,
        "current_step_index": 0,
        "retry_count": 0,
        "error": None
    }

def sql_node(state: GraphState, db_connection):
    llm = get_llm()
    user_question = state["user_question"]
    history = state["history"]
    retry_count = state.get("retry_count", 0)
    error = state.get("error")

    # Re-using metadata
    from metadata import dim_station_column_descriptions, fact_station_column_descriptions, relationships, table_info_combined

    error_context = f"\nPrevious attempt failed with error: {error}. Please fix the SQL." if error else ""

    user_prompt = f"""
    You are an expert data analyst for Shell Retail (India) working with SQLite.
    {history}
    User Question: {user_question}
    {error_context}

    Tables:
    dim_station: {dim_station_column_descriptions}
    fact_station: {fact_station_column_descriptions}
    Relationships: {relationships}
    Table Info: {table_info_combined}

    Rules:
    - Return ONLY valid SQLite SELECT query.
    - No markdown, no comments.
    - LIMIT results to 50 unless specified.
    - Round numerical values to 2 decimal places.
    """

    response = llm.invoke([
        ("system", "You are a Shell Retail SQLite expert. Return ONLY a syntactically correct SQLite SELECT query."),
        ("human", user_prompt)
    ])

    sql_query = response.content.replace('sql', '').replace('`', '').strip()

    try:
        df = pd.read_sql_query(sql_query, db_connection)
        return {
            "sql_query": sql_query,
            "dataframe_json": df.to_json(),
            "error": None,
            "retry_count": 0,
            "current_step_index": state["current_step_index"] + 1
        }
    except Exception as e:
        return {
            "sql_query": sql_query,
            "error": str(e),
            "retry_count": retry_count + 1
        }

def visualization_node(state: GraphState):
    llm = get_llm()
    user_question = state["user_question"]
    df_json = state.get("dataframe_json")

    if not df_json:
        return {"current_step_index": state["current_step_index"] + 1}

    df = pd.read_json(StringIO(df_json))
    if df.empty:
        return {"visualizations": [], "current_step_index": state["current_step_index"] + 1}

    data_sample = df.head(5).to_dict(orient='records')
    column_info = df.dtypes.apply(lambda x: str(x)).to_dict()

    viz_prompt = f"""
    You are a data visualization expert. Suggest Plotly Express charts (line, bar, pie).
    User Question: {user_question}
    Data Sample: {data_sample}
    Columns: {column_info}

    Mapping Rules:
    - For trends (e.g., over time) -> use 'line' chart.
    - For comparisons between categories -> use 'bar' chart.
    - For distributions or percentages of a total -> use 'pie' chart.

    Return ONLY a JSON list of objects with: 'type', 'x', 'y', 'values', 'names', 'color', 'title'.
    """

    response = llm.invoke([
        ("system", "Return ONLY a JSON list of Plotly chart configurations."),
        ("human", viz_prompt)
    ])

    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        configs = json.loads(content)
    except:
        configs = []

    return {
        "visualizations": configs,
        "current_step_index": state["current_step_index"] + 1
    }

def insight_node(state: GraphState):
    llm = get_llm()
    user_question = state["user_question"]
    df_json = state.get("dataframe_json")

    if not df_json:
        return {"insight": "No data available.", "current_step_index": state["current_step_index"] + 1}

    df = pd.read_json(StringIO(df_json))

    insight_prompt = f"""
    You are a senior Shell Retail business performance analyst.
    Interpret results for: {user_question}
    Data:
    {df.to_string()}

    Provide:
    📊 What’s happening
    📉 Why it’s happening
    🎯 Recommended business actions
    """

    response = llm.invoke(insight_prompt)
    return {
        "insight": response.content,
        "current_step_index": state["current_step_index"] + 1
    }

def clarification_node(state: GraphState):
    return {
        "sql_query": None, # Clear invalid SQL
        "insight": "I'm sorry, I'm having trouble generating a valid query for your request after several attempts. Could you please clarify your question or provide more details?",
        "current_step_index": len(state["plan"]) # Ensure we finish after this
    }

def summarizer_node(state: GraphState):
    llm = get_llm()
    history = state.get("history", "")
    user_question = state["user_question"]
    sql = state.get("sql_query", "")
    insight = state.get("insight", "")

    prompt = f"""
    Summarize the conversation so far. Include key findings and data points.
    Current History: {history}
    New Turn:
    User: {user_question}
    SQL: {sql}
    Insight: {insight}

    Keep the summary concise but informative, prioritizing user questions and key findings.
    """

    response = llm.invoke(prompt)

    # Also prepare the final output for Streamlit
    final_output = {
        "sql": state.get("sql_query"),
        "dataframe": state.get("dataframe_json"),
        "visualizations": state.get("visualizations"),
        "insight": state.get("insight")
    }

    return {
        "history": response.content,
        "final_output": final_output
    }

# --- Router Functions ---

def router(state: GraphState):
    plan = state["plan"]
    idx = state["current_step_index"]

    if idx >= len(plan):
        return "summarize"

    return plan[idx]

# --- Graph Construction ---

def create_graph(db_connection):
    workflow = StateGraph(GraphState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("sql_agent", lambda state: sql_node(state, db_connection))
    workflow.add_node("viz_agent", visualization_node)
    workflow.add_node("insight_agent", insight_node)
    workflow.add_node("summarize_agent", summarizer_node)
    workflow.add_node("ask_clarification", clarification_node)

    workflow.set_entry_point("orchestrator")

    def route_from_orchestrator(state):
        return router(state)

    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "sql": "sql_agent",
            "viz": "viz_agent",
            "insight": "insight_agent",
            "summarize": "summarize_agent"
        }
    )

    def route_from_sql(state):
        if state.get("error"):
            if state.get("retry_count", 0) <= 3: # 1 initial + 3 retries = 4 attempts
                return "retry"
            else:
                return "ask_clarification"
        return router(state)

    workflow.add_conditional_edges(
        "sql_agent",
        route_from_sql,
        {
            "retry": "sql_agent",
            "sql": "sql_agent",
            "viz": "viz_agent",
            "insight": "insight_agent",
            "summarize": "summarize_agent",
            "ask_clarification": "ask_clarification"
        }
    )

    workflow.add_edge("ask_clarification", "summarize_agent")

    workflow.add_conditional_edges(
        "viz_agent",
        router,
        {
            "sql": "sql_agent",
            "viz": "viz_agent",
            "insight": "insight_agent",
            "summarize": "summarize_agent"
        }
    )

    workflow.add_conditional_edges(
        "insight_agent",
        router,
        {
            "sql": "sql_agent",
            "viz": "viz_agent",
            "insight": "insight_agent",
            "summarize": "summarize_agent"
        }
    )

    workflow.add_edge("summarize_agent", END)

    return workflow.compile()
