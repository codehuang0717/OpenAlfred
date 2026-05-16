"""
RAG CLI — Test ingestion, search, and management without the frontend.

Usage:
    uv run python -m rag.cli ingest <filepath> [--title TITLE] [--user USER_ID]
    uv run python -m rag.cli ingest-text --text "..." --title TITLE [--user USER_ID]
    uv run python -m rag.cli search "your query" [--top-k 5] [--user USER_ID]
    uv run python -m rag.cli list [--user USER_ID]
    uv run python -m rag.cli delete <doc_id> [--user USER_ID]
    uv run python -m rag.cli demo [--user USER_ID]
"""

import sys
import os
import argparse
import asyncio
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import config
from core.database import init_db
from rag.ingestion import ingest_file, ingest_text
from rag.retriever import search
from rag.store import delete_document
from db.rag import get_documents
from tools.rag import _resolve_content
from logic.prompts import RAG_SEARCH_RESULT_HEADER
from utils.logger import setup_logging, get_logger

logger = get_logger("rag.cli")


def cmd_ingest(args):
    path = os.path.abspath(args.filepath)
    print(f"Ingesting: {path}")
    doc = asyncio.run(ingest_file(args.user, path, args.title or ""))
    print(f"\nDone! Document ID: {doc['id']}")
    print(f"  Title: {doc['title']}")
    print(f"  Chunks: {doc['chunk_count']}")


def cmd_ingest_text(args):
    doc = asyncio.run(ingest_text(args.user, args.text, args.title))
    print(f"\nDone! Document ID: {doc['id']}")
    print(f"  Title: {doc['title']}")
    print(f"  Chunks: {doc['chunk_count']}")


def cmd_search(args):
    print(f'Searching: "{args.query}"')
    results = search(args.user, args.query, args.top_k)
    if not results:
        print("No results found.")
        return
    print(f"\nFound {len(results)} result(s):\n")

    if args.resolve:
        async def _resolve_and_show():
            lines = []
            for i, r in enumerate(results, 1):
                heading = f" (## {r['heading']})" if r.get("heading") else ""
                content = await _resolve_content(r["content"])
                block = f"[{i}] Source: {r['filename']}{heading} (relevance: {r['score']})\n{content}"
                lines.append(block)
            results_text = "\n\n---\n\n".join(lines)
            print(RAG_SEARCH_RESULT_HEADER.format(results=results_text))
        asyncio.run(_resolve_and_show())
    else:
        for i, r in enumerate(results, 1):
            print(f"{'='*60}")
            print(f"[{i}] {r['filename']}  (score: {r['score']})")
            print(f"{'='*60}")
            print(r['content'][:500])
            print()


def cmd_list(args):
    docs = asyncio.run(get_documents(args.user))
    if not docs:
        print("No documents in knowledge base.")
        return
    print(f"\n{len(docs)} document(s):")
    for d in docs:
        print(f"  [{d['id'][:8]}..] {d['title'] or d['filename']} ({d['file_type']}, {d['chunk_count']} chunks)")


def cmd_delete(args):
    ok = asyncio.run(delete_document(args.user, args.doc_id))
    print("Deleted." if ok else "Document not found.")


def cmd_demo(args):
    """Self-test: ingest a sample text, search it, then clean up."""
    print("=" * 60)
    print("RAG Demo — Testing ingestion + retrieval pipeline")
    print("=" * 60)

    sample_text = """
OpenAlfred is an open-source AI personal assistant. It combines text chat,
voice calls, and proactive monitoring. The system uses LangGraph for agent
orchestration and supports multiple LLM providers including OpenAI GPT,
DeepSeek, Cerebras, Google Gemini, and local models via Ollama.

The assistant can manage todos, set reminders, make phone calls via SIP,
read and send emails, capture screenshots, and search the web. It also has
a proactive supervisor that monitors the user's screen activity and can
intervene when the user appears distracted from their tasks.

OpenAlfred uses a three-layer memory architecture:
1. Short-term: sliding window of recent messages
2. Mid-term: automatic conversation summarization
3. Long-term: knowledge extraction to markdown memory files

The project is built with FastAPI, Next.js, LiveKit for voice, Redis for
events, and SQLite for structured data storage.
""".strip()

    async def _demo():
        print("\n[1/4] Ingesting sample text...")
        doc = await ingest_text(args.user, sample_text, "OpenAlfred Overview")
        print(f"  Created document: {doc['id']} ({doc['chunk_count']} chunks)")

        print("\n[2/4] Searching: 'What is OpenAlfred?'")
        results = search(args.user, "What is OpenAlfred?", top_k=3)
        for r in results:
            print(f"  [{r['score']}] {r['filename']}: {r['content'][:100]}...")

        print("\n[3/4] Searching: 'How does memory work?'")
        results = search(args.user, "How does memory work?", top_k=3)
        for r in results:
            print(f"  [{r['score']}] {r['filename']}: {r['content'][:100]}...")

        print("\n[4/4] Cleaning up...")
        await delete_document(args.user, doc["id"])
        print("  Deleted demo document.")

        print("\n" + "=" * 60)
        print("Demo complete! Pipeline is working.")
        print("=" * 60)

    asyncio.run(_demo())


def main():
    parser = argparse.ArgumentParser(description="RAG CLI — Test knowledge base operations")
    parser.add_argument("--user", default="default", help="User ID (default: 'default')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest a file")
    p_ingest.add_argument("filepath", help="Path to file")
    p_ingest.add_argument("--title", help="Document title")

    p_text = sub.add_parser("ingest-text", help="Ingest raw text")
    p_text.add_argument("--text", required=True, help="Text content")
    p_text.add_argument("--title", required=True, help="Document title")

    p_search = sub.add_parser("search", help="Search knowledge base")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_search.add_argument("--resolve", action="store_true", help="Resolve images and show full prompt")

    sub.add_parser("list", help="List all documents")

    p_del = sub.add_parser("delete", help="Delete a document")
    p_del.add_argument("doc_id", help="Document ID")

    sub.add_parser("demo", help="Run a self-test demo")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level, log_file="rag.log")

    asyncio.run(init_db())
    logger.debug("Database initialized. command=%s user=%s", args.command, args.user)

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ingest-text":
        cmd_ingest_text(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
