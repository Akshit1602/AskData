import os
import json
import logging
import pandas as pd
from io import StringIO
from typing import TypedDict, List, Optional, Literal
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
import sqlite3

# Configure logging for graph_logic
logger = logging.getLogger("shell_genai_app.graph_logic")

# Define the state
class GraphState(TypedDict):
    user_question: str
    refined_question: Optional[str]
    history: str  # Summarized history
    structured_context: Optional[str] # JSON string of filters/entities
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

from metadata import get_metadata

def orchestrator_node(state: GraphState):
    logger.info("Entering orchestrator_node")
    llm = get_llm()
    user_question = state["user_question"]
    history = state["history"]
    structured_context = state.get("structured_context", "{}")
    metadata = get_metadata()
    domain_context = metadata["domain_context"]

    # --- Step 3: Automated Semantic Reset Detection ---
    reset_prompt = f"""
    Analyze the user's question against the current conversation state.
    Current State (JSON): {structured_context}
    User Question: {user_question}

    Determine if the user wants to:
    1. 'RETAIN': Continue the current thread or refine it.
    2. 'RESET': Start a completely new topic or explicitly asked for a reset.

    Return ONLY a JSON object with 'action' (RETAIN/RESET).
    """

    reset_response = llm.invoke(reset_prompt)
    logger.info(f"Reset check response: {reset_response.content.strip()}")
    try:
        reset_content = reset_response.content.strip()
        if "```json" in reset_content:
            reset_content = reset_content.split("```json")[1].split("```")[0].strip()
        reset_action = json.loads(reset_content).get('action', 'RETAIN')
    except Exception as e:
        logger.warning(f"Failed to parse reset action: {e}")
        reset_action = 'RETAIN'

    logger.info(f"Reset action determined: {reset_action}")
    current_history = history if reset_action == 'RETAIN' else ""
    current_context = structured_context if reset_action == 'RETAIN' else "{}"

    prompt = f"""
    You are an orchestrator for a {domain_context} data assistant.
    Analyze the user's question and history to decide the execution plan.
    Available agents:
    - 'refine': To resolve references, anaphora, and ellipses in follow-up questions. MANDATORY before 'sql'.
    - 'sql': For generating and executing SQL queries when new data is needed.
    - 'viz': For generating visualizations from data.
    - 'insight': For generating business insights, analysis, or explanations from data.

    Rules:
    1. If the user asks for a new data query (e.g., "What are the sales?"), the plan should be ["refine", "sql", "viz"].
    2. If the user asks for a change in visualization (e.g., "make it a bar chart") and data is already available in history, the plan should be ["viz"].
    3. If the user asks for business insights, analysis, "why", "explain", or recommendations, INCLUDE 'insight' in the plan (e.g., ["sql", "viz", "insight"] if new data is needed, or just ["insight"] if data exists in history).
    4. If the user asks a follow-up that requires new data but NOT insights, the plan should be ["refine", "sql", "viz"].
    5. If the question can be answered from existing data/history without a new SQL, skip 'sql'.
    6. Return ONLY a JSON object with the 'plan' key (a list of agent names).

    User Question: {user_question}
    History Summary: {current_history}
    """

    response = llm.invoke(prompt)
    logger.info(f"Orchestrator plan response: {response.content.strip()}")
    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        plan_data = json.loads(content)
        plan = plan_data.get("plan", ["refine", "sql", "viz"])
    except Exception as e:
        logger.warning(f"Failed to parse orchestrator plan: {e}")
        plan = ["refine", "sql", "viz"]

    logger.info(f"Final plan: {plan}")
    return {
        "plan": plan,
        "history": current_history,
        "structured_context": current_context,
        "current_step_index": 0,
        "retry_count": 0,
        "error": None
    }

def intent_refinement_node(state: GraphState):
    logger.info("Entering intent_refinement_node")
    llm = get_llm()
    user_question = state["user_question"]
    history = state["history"]
    structured_context = state.get("structured_context", "{}")
    metadata = get_metadata()
    domain_context = metadata["domain_context"]

    prompt = f"""
    You are a Query Refinement Expert for an {domain_context} data assistant.
    Your task is to rewrite the user's question to be self-contained, resolving any references (it, them, those), anaphora, or ellipses based on the conversation history and state.

    ### CONTEXT:
    History: {history}
    Structured State: {structured_context}

    ### USER QUESTION:
    {user_question}

    ### INSTRUCTIONS:
    1. Identify if the question is a follow-up or a new topic.
    2. If it's a follow-up, merge the previous constraints with the new question.
       Example:
       Turn 1: "Show sales for Bangalore"
       Turn 2: "What about Chennai?"
       Refined: "Show sales for Chennai"
    3. If the user uses pronouns (e.g., "their revenue"), resolve them to the specific entities previously mentioned.
    4. If the question is already self-contained, return it as is.
    5. Return ONLY a JSON object with a single key 'refined_question'.
    """

    response = llm.invoke(prompt)
    logger.info(f"Refinement response: {response.content.strip()}")
    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        refined_data = json.loads(content)
        refined_question = refined_data.get("refined_question", user_question)
    except Exception as e:
        logger.warning(f"Failed to parse refined question: {e}")
        refined_question = user_question

    logger.info(f"Refined question: {refined_question}")
    return {
        "refined_question": refined_question,
        "current_step_index": state["current_step_index"] + 1
    }

def sql_node(state: GraphState, db_connection):
    logger.info("Entering sql_node")
    llm = get_llm()
    user_question = state.get("refined_question") or state["user_question"]
    history = state["history"]
    structured_context = state.get("structured_context", "{}")
    retry_count = state.get("retry_count", 0)
    error = state.get("error")

    # Re-using metadata
    metadata = get_metadata()
    domain_context = metadata["domain_context"]
    column_descriptions = metadata["column_descriptions"]
    relationships = metadata["relationships"]
    table_info_combined = metadata["table_info_combined"]

    if error:
        system_msg = f"You are a SQL Self-Correction Expert. The previous SQL failed with error: {error}. Focus on fixing column names and join conditions."
        user_prompt = f"""
        ### ERROR DIAGNOSTIC:
        The previous query failed.
        Error: {error}

        ### TASK:
        1. Review the Schema below. Ensure every column used exists in the 'Tables and Columns' JSON.
        2. Check for common SQLite errors (e.g., string vs integer).
        3. Provide a CORRECTED SELECT query.

        ### DATABASE SCHEMA:
        {json.dumps(column_descriptions, indent=2)}
        {table_info_combined}

        Return ONLY the corrected SQLite SELECT query.
        """
    else:
        system_msg = f"You are an {domain_context} SQLite expert. Return ONLY a syntactically correct SQLite SELECT query."
        user_prompt = f"""
        You are an {domain_context} working with SQLite.

        ### CONTEXT FROM PREVIOUS TURNS:
        {history}
        Structured State: {structured_context}

        ### CURRENT REQUEST:
        User Question: {user_question}

        ### DATABASE SCHEMA:
        Tables and Columns:
        {json.dumps(column_descriptions, indent=2)}
        Relationships: {json.dumps(relationships, indent=2)}
        Table Info (DDL-like): {table_info_combined}

        Rules:
        - Use the 'Structured State' to resolve references like "them", "those", or "that city".
        - If the current question is a follow-up (e.g., "What about Bangalore?"), carry forward the previous metrics and constraints unless contradicted.
        - Return ONLY valid SQLite SELECT query.
        - No markdown, no comments.
        - LIMIT results to 50 unless specified.
        - Round numerical values to 2 decimal places.
        """

    response = llm.invoke([
        ("system", system_msg),
        ("human", user_prompt)
    ])
    logger.info(f"SQL generation response: {response.content.strip()}")

    sql_query = response.content.replace('sql', '').replace('`', '').strip()
    logger.info(f"Executing SQL: {sql_query}")

    try:
        df = pd.read_sql_query(sql_query, db_connection)
        logger.info(f"SQL execution successful, returned {len(df)} rows.")
        return {
            "sql_query": sql_query,
            "dataframe_json": df.to_json(),
            "error": None,
            "retry_count": 0,
            "current_step_index": state["current_step_index"] + 1
        }
    except Exception as e:
        logger.error(f"SQL execution failed: {str(e)}")
        return {
            "sql_query": sql_query,
            "error": str(e),
            "retry_count": retry_count + 1
        }

def visualization_node(state: GraphState):
    logger.info("Entering visualization_node")
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

    Strict Rules:
    - Limit to ONLY ONE best fitting visual.
    - Return ONLY a JSON list containing a single object (or an empty list if no visual is suitable).

    Output:
    Return ONLY the JSON list of objects with: 'type', 'x', 'y', 'values', 'names', 'color', 'title'.
    """

    response = llm.invoke([
        ("system", "Return ONLY a JSON list of Plotly chart configurations."),
        ("human", viz_prompt)
    ])
    logger.info(f"Visualization response: {response.content.strip()}")

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
    logger.info("Entering insight_node")
    llm = get_llm()
    user_question = state["user_question"]
    df_json = state.get("dataframe_json")
    metadata = get_metadata()
    domain_context = metadata["domain_context"]

    if not df_json:
        return {"insight": "No data available.", "current_step_index": state["current_step_index"] + 1}

    df = pd.read_json(StringIO(df_json))

    # Add domain-specific context for insights
    analysis_focus = ""
    if "experimental" in domain_context.lower():
        analysis_focus = """
        When interpreting results for experimental testing:
        - Compare 'Treatment' vs 'Control' groups.
        - Look for incremental lift in GMV, Household Counts, or Orders.
        - Analyze performance differences across 'Acquisition' and 'Retention' cohorts.
        - Mention if the results suggest the campaign was successful based on the delta between groups.
        """

    insight_prompt = f"""
    You are a senior {domain_context}.
    Interpret results for: {user_question}
    Data:
    {df.to_string()}
    {analysis_focus}

    Provide:
    📊 What’s happening
    📉 Why it’s happening
    🎯 Recommended business actions
    """

    response = llm.invoke(insight_prompt)
    logger.info(f"Insight response: {response.content.strip()[:100]}...")
    return {
        "insight": response.content,
        "current_step_index": state["current_step_index"] + 1
    }

def clarification_node(state: GraphState):
    logger.info("Entering clarification_node")
    llm = get_llm()
    metadata = get_metadata()
    table_info = metadata["table_info_combined"]

    prompt = f"""
    The SQL assistant failed to generate a valid query for the user.
    Based on the available tables and columns, suggest 3 sample questions the user could ask instead.

    Database Schema:
    {table_info}

    Return ONLY the 3 questions as a bulleted list.
    """

    suggestions = llm.invoke(prompt).content.strip()
    logger.info(f"Clarification suggestions: {suggestions}")

    message = (
        "I'm sorry, I'm having trouble generating a valid query for your request after several attempts. "
        "Here are some alternative questions you might find useful based on the data I have:\n\n"
        f"{suggestions}"
    )

    return {
        "sql_query": None, # Clear invalid SQL
        "insight": message,
        "current_step_index": len(state["plan"]) # Ensure we finish after this
    }

def summarizer_node(state: GraphState):
    logger.info("Entering summarizer_node")
    llm = get_llm()
    history = state.get("history", "")
    structured_context = state.get("structured_context", "{}")
    user_question = state["user_question"]
    sql = state.get("sql_query", "")
    insight = state.get("insight")
    df_json = state.get("dataframe_json")

    # If insight wasn't generated but we have data, generate a one-liner description
    if not insight and df_json:
        df = pd.read_json(StringIO(df_json))
        if not df.empty:
            desc_prompt = f"""
            You are a helpful data assistant. Provide a one-line description of the results for the user's question.
            User Question: {user_question}
            Data:
            {df.head(10).to_string()}

            Return ONLY the one-line description.
            """
            insight = llm.invoke(desc_prompt).content.strip()

    summary_prompt = f"""
    You are a conversation state manager for a SQL assistant.
    Analyze the new turn and update the conversation history AND the structured context.

    Current History: {history}
    Current Structured Context: {structured_context}

    New Turn:
    User: {user_question}
    SQL: {sql}
    Result Snapshot: {insight}

    Your Task:
    1. Update the 'History' summary (concise narrative).
    2. Update the 'StructuredContext' JSON. It must track:
       - 'active_filters': dictionary of column-value pairs (e.g. {{"city": "Bangalore"}})
       - 'active_metrics': list of columns user is interested in (e.g. ["revenue_inr", "liters_sold"])
       - 'last_entities': list of specific IDs or names mentioned.
       - 'intent': the current analytical goal (e.g. "comparing city performance")

    Return ONLY a JSON object with 'history' and 'structured_context' keys.
    """

    response = llm.invoke(summary_prompt)
    logger.info(f"Summarizer response: {response.content.strip()}")
    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        result = json.loads(content)
        new_history = result.get("history", history)
        new_structured_context = json.dumps(result.get("structured_context", {}))
    except Exception as e:
        logger.warning(f"Failed to parse summarizer output: {e}")
        new_history = history
        new_structured_context = structured_context

    # Also prepare the final output for Streamlit
    final_output = {
        "sql": state.get("sql_query"),
        "dataframe": state.get("dataframe_json"),
        "visualizations": state.get("visualizations"),
        "insight": insight
    }

    return {
        "history": new_history,
        "structured_context": new_structured_context,
        "final_output": final_output
    }

# --- Router Functions ---

def router(state: GraphState):
    logger.info("Entering router")
    plan = state["plan"]
    idx = state["current_step_index"]

    if idx >= len(plan):
        logger.info("Plan complete, routing to summarize.")
        return "summarize"

    next_node = plan[idx]
    logger.info(f"Routing to next node in plan: {next_node}")
    return next_node

# --- Graph Construction ---

def create_graph(db_connection):
    workflow = StateGraph(GraphState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("intent_refinement", intent_refinement_node)
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
            "refine": "intent_refinement",
            "sql": "sql_agent",
            "viz": "viz_agent",
            "insight": "insight_agent",
            "summarize": "summarize_agent"
        }
    )

    workflow.add_conditional_edges(
        "intent_refinement",
        router,
        {
            "refine": "intent_refinement",
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
