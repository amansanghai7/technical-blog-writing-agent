# Blog Writing Agent

A production-grade multi-agent system that autonomously researches, plans, and writes full technical blog posts — complete with AI-generated diagrams — using LangGraph, OpenAI, and Google Gemini.

Built with **Human-in-the-Loop plan approval**, **Supabase-backed checkpoint persistence**, and a **Streamlit UI** that streams live agent progress.

---

## Architecture

```
User Input (topic)
       │
       ▼
  ┌─────────┐
  │  Router  │  Decides: closed_book / hybrid / open_book
  └────┬─────┘
       │
  ┌────▼──────┐
  │  Research  │  Tavily web search → LLM evidence synthesis
  └────┬───────┘   (skipped for evergreen topics)
       │
  ┌────▼──────────┐
  │  Orchestrator  │  Generates a structured blog plan (title, sections, word targets)
  └────┬───────────┘
       │
  ┌────▼──────────┐
  │  Plan Review   │  ⏸ HUMAN-IN-THE-LOOP — Approve / Edit / Reject & Replan
  └────┬───────────┘
       │
  ┌────▼────────────────────────────────┐
  │  Workers  (parallel fan-out via Send) │  Each section written by a dedicated LLM call
  └────┬────────────────────────────────┘
       │
  ┌────▼─────────────────────────────────────────┐
  │  Reducer Subgraph                             │
  │   merge_content → decide_images              │
  │                 → generate_and_place_images  │  Gemini generates diagrams; inserts after headings
  └────┬──────────────────────────────────────────┘
       │
  Final .md + images/ saved to BLOGS_DIR
```

State is checkpointed to **Supabase PostgreSQL** after every node — interrupted runs can be resumed from the exact failure point.

---

## Features

| Feature | Details |
|---|---|
| **Intelligent routing** | Classifies each topic as evergreen, hybrid, or news-roundup before doing any research |
| **Web research** | Tavily search + LLM evidence synthesis; deduplication and recency filtering |
| **Parallel section writing** | All blog sections written concurrently via LangGraph `Send` fan-out |
| **Human-in-the-Loop** | Plan approval gate before expensive worker fan-out — approve, edit, or reject |
| **AI image generation** | Google Gemini generates diagrams placed at the right heading in the blog |
| **Checkpoint & Resume** | Full graph state saved to Supabase after every node; interrupted runs resume from last checkpoint |
| **Past blogs viewer** | Sidebar lists all previously generated blogs; one-click load |
| **Live streaming UI** | Streamlit shows each node as it completes; progress JSON updated in real time |
| **Multi-format download** | Download as `.md` or bundled `.zip` with all images |

---

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — Stateful multi-agent orchestration (StateGraph, Send, interrupt)
- **[LangChain OpenAI](https://python.langchain.com/docs/integrations/chat/openai/)** — GPT-4.1-mini for all LLM calls with structured Pydantic output
- **[Tavily](https://tavily.com)** — Real-time web research API
- **[Google Gemini](https://ai.google.dev)** — `gemini-2.5-flash-image` for diagram generation
- **[Supabase](https://supabase.com)** — PostgreSQL backend for LangGraph checkpoint persistence
- **[psycopg v3](https://www.psycopg.org/psycopg3/)** + **psycopg-pool** — PostgreSQL connection pooling
- **[Streamlit](https://streamlit.io)** — Interactive UI with real-time streaming

---

## Project Structure

```
blog-writing-agent/
├── bwa_backend.py          # LangGraph graph: all nodes, state, checkpointer
├── bwa_frontend.py         # Streamlit UI: streaming, HITL review panel, past blogs
├── requirements.txt        # Direct dependencies (pinned)
├── .env.example            # Template — copy to .env and fill in keys
├── .streamlit/
│   └── config.toml         # Streamlit server config (port, headless, no watcher)
├── HITL_IMPLEMENTATION.md  # Deep-dive on the Human-in-the-Loop design
└── .gitignore
```

---

## Prerequisites

- Python 3.11+
- API keys for: OpenAI, Tavily, Google AI Studio
- A [Supabase](https://supabase.com) project (free tier is enough)

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in every value. See the [Environment Variables](#environment-variables) section below for details.

### 5. Run the app

```bash
streamlit run bwa_frontend.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | [platform.openai.com](https://platform.openai.com/api-keys) |
| `TAVILY_API_KEY` | Yes | [tavily.com](https://tavily.com) — enables web research |
| `GOOGLE_API_KEY` | Yes | [aistudio.google.com](https://aistudio.google.com/apikey) — Gemini image gen |
| `DATABASE_URI` | Yes | Supabase session-pooler connection string (see below) |
| `LANGGRAPH_ALLOWED_MSGPACK_MODULES` | Yes | Set to `bwa_backend` — allows Pydantic types in checkpoints |
| `LANGSMITH_TRACING` | No | Set `false` to disable LangSmith tracing |
| `BLOGS_DIR` | No | Output directory for generated blogs. Defaults to current directory. Set to an absolute path on EC2. |

### Getting the Supabase `DATABASE_URI`

1. Go to your Supabase project dashboard
2. **Settings → Database → Connection String**
3. Select **Session pooler** (port 5432)
4. Copy the URI and replace `[YOUR-PASSWORD]` with your database password

---

## Usage Guide

### Generate a blog

1. Enter a topic in the sidebar (e.g. *"How transformers work — a visual guide"*)
2. Set the as-of date (used for recency filtering in research)
3. Click **Generate Blog**
4. Watch the agent progress stream in real time

### Human-in-the-Loop plan approval

After the orchestrator creates a plan, the graph **pauses automatically**:

- A warning banner appears with the full plan (title, sections, word counts, tags)
- Edit the blog title, audience, or tone if needed
- Click one of:
  - **✅ Approve & Write** — proceed with the plan as generated
  - **✏️ Apply Edits & Write** — use your edited values
  - **🔄 Reject & Replan** — orchestrator generates a new plan (max 2 rejections)

Workers only start after you approve — no wasted API calls on a bad plan.

### Resume an interrupted run

If a run fails mid-way (API timeout, network issue), the last completed node is already checkpointed in Supabase. Click **Resume last run** in the sidebar — the graph continues from the last saved node, skipping everything that already succeeded.

### Past blogs

All generated blogs appear in the **Past blogs** sidebar list (newest first). Click any entry and then **Load selected blog** to view it in the tabs without regenerating.

---

## How It Works — Key Design Decisions

**Why `Send` for fan-out?**
Each blog section is an independent LLM task. LangGraph's `Send` dispatches all sections to worker nodes in parallel, cutting total generation time by ~70% compared to sequential writing.

**Why `interrupt()` for HITL?**
`interrupt()` serialises the full graph state to Supabase and pauses. The frontend resumes with `Command(resume=decision)`. This means the pause can last minutes or hours — the state is safe in the database. See [`HITL_IMPLEMENTATION.md`](HITL_IMPLEMENTATION.md) for a full deep-dive.

**Why separate `decide_images` from workers?**
The image placement LLM only receives section headings (not full text). This avoids the LLM truncating long blog content when asked to reproduce it inside a structured output schema.

---

## Deployment

This project is designed to deploy on **AWS EC2** behind an **Nginx reverse proxy** with **SSL**. The `BLOGS_DIR` environment variable ensures generated blogs persist in a fixed directory on the server, separate from the application code.

A full step-by-step deployment guide is in progress.

---

## License

MIT
