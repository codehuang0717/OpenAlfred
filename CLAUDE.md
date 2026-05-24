# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Environment

**Use `uv` for ALL package management. Never use `pip`.**

```bash
uv add <package>        # add dependency
uv sync                 # sync venv to lockfile
uv run python ...       # run code in venv
```

The venv is at `.venv/`. PyTorch uses CUDA 12.4 wheels via `[[tool.uv.index]]` in pyproject.toml.

## Running

```bash
# LangGraph dev server (agent graph + API on :2024)
uv run langgraph dev

# FastAPI directly (business API on :7788)
uv run python -m app

# RAG CLI
cd src
uv run python -m rag.cli demo
uv run python -m rag.cli ingest "path/to/file.md" --user <uuid>
uv run python -m rag.cli search "query" --resolve
```

## Architecture

### Graph Nodes (logic/nodes.py)

Five nodes in sequence:
1. **load_context_node** — injects current time, L1 memories (from `memory/`), conversation summary, and sets RAG user_id global
2. **agent_node** — binds all tools to LLM, invokes, returns response or tool calls. Voice calls use slim tool subset
3. **tools** (ToolNode) — executes requested tools
4. **extract_knowledge_node** — every N turns, LLM extracts user facts to `memory/{user_id}/*.md`
5. **summarize_node** — when conversation exceeds threshold, compresses old messages (skipped for voice calls)

Conditional edge: agent → tools (if tool_calls) or extract_knowledge (if text response). Tools loop back to agent.

### Tools (tools/)

Built-in tools listed in `tools/__init__.py` → `ALL_TOOLS` list.

RAG tools (`tools/rag.py`) use both a `contextvars.ContextVar` and a module-level `_user_id` global — set in `load_context_node` and `agent_node` before LLM invocation.

### Database (db/)

Single SQLite file (`todos.db`) with WAL mode. All DB access through `aiosqlite` — functions are `async`. Tables: todos, reminders, users, thread_memories, supervisor_sessions, settings, email_credentials, documents, image_lookup.

Schema is created/ migrated in `db/connection.py::init_db()` — called at app startup (lifespan) and CLI entry.

### RAG Module (rag/)

Markdown ingestion pipeline:
1. `md_parser.py` — splits by `#`/`##` headings into Sections, extracts image refs
2. `image_handler.py` — copies images from source dir to `data/images/{doc_id}/`, calls describer, inserts `image_lookup` row, replaces `![alt](path)` with `{"_img":{"i":id,"d":"desc"}}`
3. `chunker.py` — `chunk_sections()` for markdown (keeps text+heading+images together), `chunk_text()` for plain text. Separators include `{` to avoid splitting JSON placeholders
4. `embedding.py` — loads BGE-M3 once (global singleton), normalizes embeddings
5. `store.py` — ChromaDB collection (cosine distance) + SQLite documents table
6. `retriever.py` — embed query → ChromaDB query → return chunks with scores

`image_describer.py` uses Gemini via `langchain_google_genai.ChatGoogleGenerativeAI`. Cache by SHA256 of image file. Self-healing cache: if dirty data from older gemini responses is found, extracts pure text and overwrites.

### Auth

`core/auth.py` — LangGraph auth handler. Maps JWT `sub` → `config["configurable"]["langgraph_auth_user"]["identity"]`. Thread CRUD filtered by `owner` metadata.

`_get_user_id_from_config()` in nodes.py resolves identity from config or falls back to "default".

### Config

`core/config.py::Config` — reads from `.env` at project root. All keys have defaults so the app starts without any env vars set.

## Key Patterns

- **Logging**: `from utils.logger import get_logger` with name like `"rag.embedding"`. Log files in `logs/` with rotation (10MB, 5 backups).
- **Async**: All DB and ingestion functions are async. CLI uses `asyncio.run()`.
- **Structured output**: `utils/structured_output.py::structured_invoke()` for Pydantic model extraction from LLM.
- **User isolation**: All CRUD functions take `user_id` parameter.
