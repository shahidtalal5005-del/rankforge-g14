# RankForge — AI SEO & Affiliate Content Engine (Group G14)

RankForge is a production LangGraph + FastAPI + Neon Postgres AI agent that researches high-intent SEO keywords, drafts affiliate-monetized articles, enforces human-in-the-loop confirmation before publishing, and isolates per-creator memories.

## Architecture
- **Graph (ackend/graph.py):** 3-node LangGraph (guardrail -> seo_agent <-> 	ools) compiled with PostgresSaver checkpointing.
- **Tools:** esearch_keywords (queries Postgres keywords table) and publish_article (inserts verified articles into rticles table).
- **Memory & Isolation:** Short-term thread state persisted via PostgresSaver (user_id:thread_id); long-term brand preferences stored in user_memories keyed by user_id.
- **Guardrails:** Prompt-injection & off-topic filter node + explicit confirmation gate before executing publish_article.

## Endpoints
- GET /health -> returns ok
- POST /chat -> accepts {"user_id", "thread_id", "message"} and returns {"reply", "tool_calls", "tokens"}
- GET /admin/overview -> returns live keywords, published articles, and user memories

## Local Setup
1. Copy .env.example to .env and set ANTHROPIC_API_KEY and DATABASE_URL.
2. Install dependencies: pip install -r backend/requirements.txt
3. Run server: cd backend && uvicorn main:app --reload
4. Open http://127.0.0.1:8000 (Customer Workspace) and http://127.0.0.1:8000/admin.html (Admin Dashboard).
