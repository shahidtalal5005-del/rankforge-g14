import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from graph import graph, get_db_conn

app = FastAPI(title="RankForge API - Group G14")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.post("/chat")
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": f"{req.user_id}:{req.thread_id}"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=req.message)], "user_id": req.user_id, "blocked": False},
        config=config
    )

    tool_calls = []
    tokens_in = 0
    tokens_out = 0

    for msg in result["messages"][-4:]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(tc["name"])
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            tokens_in += msg.usage_metadata.get("input_tokens", 0)
            tokens_out += msg.usage_metadata.get("output_tokens", 0)

    raw_content = result["messages"][-1].content
    if isinstance(raw_content, list):
        final_reply = "\n".join(
            block.get("text", "") for block in raw_content if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    else:
        final_reply = str(raw_content)

    return {
        "reply": final_reply,
        "tool_calls": list(set(tool_calls)),
        "tokens": {"in": tokens_in, "out": tokens_out}
    }

@app.get("/admin/overview")
def admin_overview():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, keyword, search_volume, difficulty, affiliate_link FROM keywords ORDER BY id")
    keywords = [{"id": r[0], "keyword": r[1], "volume": r[2], "difficulty": r[3], "link": r[4]} for r in cur.fetchall()]

    cur.execute("SELECT id, user_id, title, target_keyword, content, clicks, created_at FROM articles ORDER BY id DESC")
    articles = [{"id": r[0], "user_id": r[1], "title": r[2], "keyword": r[3], "content": r[4], "clicks": r[5], "created_at": str(r[6])} for r in cur.fetchall()]

    cur.execute("SELECT user_id, brand_preferences, updated_at FROM user_memories ORDER BY updated_at DESC")
    memories = [{"user_id": r[0], "preferences": r[1], "updated_at": str(r[2])} for r in cur.fetchall()]
    cur.close()
    conn.close()
    return {"keywords": keywords, "articles": articles, "memories": memories}

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
