import os
import psycopg2
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_conn():
    return psycopg2.connect(DATABASE_URL)

def get_user_memory(user_id: str) -> str:
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT brand_preferences FROM user_memories WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else "No saved brand/niche preferences yet."

def save_user_memory(user_id: str, preference: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO user_memories (user_id, brand_preferences, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (user_id) DO UPDATE
        SET brand_preferences = EXCLUDED.brand_preferences, updated_at = NOW();
    """, (user_id, preference))
    conn.commit()
    cur.close()
    conn.close()

@tool
def research_keywords(topic: str) -> str:
    """Search the SEO keyword database for search volume, difficulty, and affiliate links matching a topic."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT keyword, search_volume, difficulty, affiliate_link FROM keywords WHERE keyword ILIKE %s",
        (f"%{topic}%",)
    )
    rows = cur.fetchall()
    if not rows:
        cur.execute("SELECT keyword, search_volume, difficulty, affiliate_link FROM keywords LIMIT 4")
        rows = cur.fetchall()
    cur.close()
    conn.close()
    return "\n".join([f"- {r[0]} (Vol: {r[1]}/mo, Diff: {r[2]}, Affiliate: {r[3]})" for r in rows])

@tool
def publish_article(user_id: str, title: str, target_keyword: str, content: str) -> str:
    """Publish a finished SEO & affiliate article to the database for a specific user_id. ONLY call after user confirms publishing."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO articles (user_id, title, target_keyword, content) VALUES (%s, %s, %s, %s) RETURNING id",
        (user_id, title, target_keyword, content)
    )
    article_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return f"Successfully published article #{article_id}: '{title}' for user {user_id}."

tools = [research_keywords, publish_article]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    blocked: bool

llm = ChatAnthropic(model="claude-haiku-5-5").bind_tools(tools)

def guardrail_node(state: AgentState):
    last_msg = state["messages"][-1].content.lower()
    injection_phrases = ["ignore previous", "system prompt", "jailbreak", "reveal your instructions", "act as"]
    off_topic_phrases = ["write python code", "solve calculus", "weather today", "recipe for"]

    if any(p in last_msg for p in injection_phrases):
        return {
            "blocked": True,
            "messages": [AIMessage(content="Security Guardrail: I cannot bypass my operational rules or reveal internal instructions. Ask me about SEO keywords, affiliate articles, or content strategy.")]
        }
    if any(p in last_msg for p in off_topic_phrases):
        return {
            "blocked": True,
            "messages": [AIMessage(content="I am RankForge, an SEO and affiliate content engine. I can only help with keyword research, drafting SEO articles, and publishing content.")]
        }
    return {"blocked": False}

def seo_agent_node(state: AgentState):
    user_id = state["user_id"]
    last_text = state["messages"][-1].content

    if any(k in last_text.lower() for k in ["my niche", "my blog", "prefer tone", "my audience", "remember that"]):
        save_user_memory(user_id, last_text)

    long_term_mem = get_user_memory(user_id)
    recent_messages = state["messages"][-10:]

    system_prompt = SystemMessage(content=f"""You are RankForge, an AI SEO and affiliate content engine.
Current User ID: {user_id}
Saved Long-Term Brand Preferences for this user: {long_term_mem}

Rules:
1. Use 
esearch_keywords to look up real keyword metrics and affiliate links before drafting content.
2. When writing an SEO article, include a catchy H1 title, target keyword, structured H2 subheadings, and the exact affiliate link naturally embedded.
3. RISKY ACTION GUARD: Publishing an article (publish_article) is a live action. You MUST ask the user "Would you like me to publish this article live?" BEFORE calling publish_article, unless the user explicitly confirmed ("yes", "confirm", "publish it now") in their latest message.
4. Always pass user_id="{user_id}" when calling publish_article. Never access or mention any other user's data.
5. Keep responses concise to save tokens.""")

    response = llm.invoke([system_prompt] + recent_messages)
    return {"messages": [response]}

def route_after_guardrail(state: AgentState):
    return END if state.get("blocked") else "seo_agent"

def route_after_agent(state: AgentState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

pool = ConnectionPool(conninfo=DATABASE_URL, max_size=10, kwargs={"autocommit": True})
checkpointer = PostgresSaver(pool)
checkpointer.setup()

workflow = StateGraph(AgentState)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("seo_agent", seo_agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("guardrail")
workflow.add_conditional_edges("guardrail", route_after_guardrail, {"seo_agent": "seo_agent", END: END})
workflow.add_conditional_edges("seo_agent", route_after_agent, {"tools": "tools", END: END})
workflow.add_edge("tools", "seo_agent")

graph = workflow.compile(checkpointer=checkpointer)
