import httpx
import logging
from typing import Optional, List, Dict, Literal
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("eye-tool")

SCREENPIPE_URL = "http://localhost:3030"

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

async def get_recent_ocr_text(minutes: int = 10) -> str:
    """
    Fetch and summarize OCR text captured by Screenpipe in the last X minutes.
    """
    try:
        now_utc = datetime.now(timezone.utc)
        start_time_dt = now_utc - timedelta(minutes=minutes)
        start_time_str = start_time_dt.isoformat()
        
        # Diagnostic Log
        console.print(Panel(
            f"[bold cyan]Screenpipe Query[/bold cyan]\n"
            f"Local Time: {datetime.now().strftime('%H:%M:%S')}\n"
            f"UTC Time  : {now_utc.strftime('%H:%M:%S')} (Z)\n"
            f"Start Time: {start_time_str}\n"
            f"Window    : {minutes} minutes",
            title="[eye-tool] Diagnostic",
            border_style="cyan"
        ))

        async with httpx.AsyncClient() as client:
            params = {
                "limit": 50,
                "content_type": "ocr",
                "start_time": start_time_str
            }
            resp = await client.get(f"{SCREENPIPE_URL}/search", params=params, timeout=10.0)
            
            if resp.status_code != 200:
                console.print(f"[bold red]Screenpipe Error:[/bold red] {resp.status_code}")
                return ""
            
            data = resp.json()
            items = data.get("data", [])
            
            # Diagnostic Response Summary
            count = len(items)
            res_color = "green" if count > 0 else "yellow"
            console.print(f"[{res_color}][eye-tool] Screenpipe returned {count} OCR items.[/{res_color}]")
            
            if not items:
                return "No screen activity detected in the last few minutes."
            
            seen_text = set()
            unique_content = []
            
            for item in items:
                content = item.get("content", {}).get("text", "").strip()
                if content and content not in seen_text:
                    seen_text.add(content)
                    unique_content.append(content)
            
            summary = "\n".join(unique_content[:20]) 
            
            # Preview of what's going to AI
            console.print(Panel(
                summary[:500] + ("..." if len(summary) > 500 else ""),
                title="OCR Context Preview (Sent to AI)",
                border_style="dim"
            ))
            
            return summary if summary else "Screen was blank or no text detected."

    except Exception as e:
        console.print(f"[bold red]Error connecting to Screenpipe:[/bold red] {e}")
        return f"Error connecting to Screenpipe: {str(e)}"

# LangChain tool definitions (for the chat agent)
from langchain.tools import tool

@tool
async def view_screen(mode: Literal["current", "history"] = "current", query: str = "") -> str:
    """View user's screen content. mode='current' for live screen (last 2min OCR), mode='history' to search past screen text by query."""
    if mode == "history" and query:
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "q": query,
                    "limit": 10,
                    "content_type": "ocr"
                }
                resp = await client.get(f"{SCREENPIPE_URL}/search", params=params, timeout=10.0)
                if resp.status_code == 200:
                    items = resp.json().get("data", [])
                    if not items: return f"No results found for '{query}'"
                    
                    results = []
                    for item in items:
                        ts = item.get("content", {}).get("timestamp", "unknown")
                        txt = item.get("content", {}).get("text", "")
                        results.append(f"[{ts}] {txt}")
                    return "\n---\n".join(results)
                return f"Error: {resp.text}"
        except Exception as e:
            return f"Error querying screen history: {e}"
    else:
        return await get_recent_ocr_text(minutes=2)

screen_tools = [view_screen]
