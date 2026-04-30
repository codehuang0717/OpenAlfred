import httpx
import logging
from typing import Optional, List, Dict, Literal
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("eye-tool")

SCREENPIPE_URL = "http://localhost:3030"

from rich.console import Console
from rich.panel import Panel

console = Console()

async def get_enhanced_context(minutes: int = 10) -> str:
    """
    Fetch OCR, Audio, and UI context from Screenpipe for a comprehensive user activity view.
    """
    try:
        now_utc = datetime.now(timezone.utc)
        start_time_dt = now_utc - timedelta(minutes=minutes)
        start_time_str = start_time_dt.isoformat()
        
        async with httpx.AsyncClient() as client:
            # 1. Fetch OCR and Audio in one search call
            params = {
                "limit": 60,
                "content_type": "all", # Fetch OCR, Audio, and Input
                "start_time": start_time_str
            }
            resp = await client.get(f"{SCREENPIPE_URL}/search", params=params, timeout=10.0)
            
            if resp.status_code != 200:
                logger.error(f"Screenpipe Search Error: {resp.status_code}")
                return "Error fetching screen context."

            data = resp.json()
            items = data.get("data", [])
            
            # 2. Process items into structured context
            ocr_parts = []
            audio_parts = []
            input_events = 0
            apps_active = set()
            urls_active = set()
            
            for item in items:
                content_type = item.get("type", "").upper()
                content = item.get("content", {})
                
                if content_type == "OCR":
                    text = content.get("text", "").strip()
                    app = content.get("app_name", "Unknown")
                    window = content.get("window_name", "Unknown")
                    url = content.get("browser_url", "")
                    
                    if text:
                        ocr_parts.append(f"[{app} | {window}] {text[:200]}")
                        apps_active.add(app)
                        if url: urls_active.add(url)
                
                elif content_type == "AUDIO":
                    text = content.get("text", "").strip()
                    if text:
                        audio_parts.append(text)
                
                elif content_type == "INPUT":
                    input_events += 1
            
            # 3. Fetch Activity Summary (if available)
            activity_summary = ""
            try:
                summary_resp = await client.get(f"{SCREENPIPE_URL}/activity/get-activity-summary", timeout=5.0)
                if summary_resp.status_code == 200:
                    activity_data = summary_resp.json()
                    # activity_data format depends on version, usually it's a list of activities
                    # We'll just take a few if it's a list
                    if isinstance(activity_data, list):
                        summary_items = [f"{a.get('app_name')}: {a.get('duration')}s" for a in activity_data[:5]]
                        activity_summary = "Recent Apps: " + ", ".join(summary_items)
            except:
                pass # Optional feature

            # 4. Construct Final Context String
            context_blocks = []
            
            if apps_active:
                context_blocks.append(f"Active Apps: {', '.join(apps_active)}")
            
            if urls_active:
                context_blocks.append(f"Active URLs: {', '.join(urls_active)}")
            
            context_blocks.append(f"Physical Activity: {'Active' if input_events > 0 else 'Idle'} ({input_events} input events detected)")

            if activity_summary:
                context_blocks.append(f"Activity Summary: {activity_summary}")

            if ocr_parts:
                # Keep unique and relevant OCR snippets
                unique_ocr = list(dict.fromkeys(ocr_parts))[:15]
                context_blocks.append("--- Screen Content (OCR) ---\n" + "\n".join(unique_ocr))
            
            if audio_parts:
                unique_audio = list(dict.fromkeys(audio_parts))[:10]
                context_blocks.append("--- Audio Transcripts (Meetings/Speech) ---\n" + "\n".join(unique_audio))

            final_context = "\n\n".join(context_blocks)
            
            # Diagnostic Log
            console.print(f"[cyan][eye-tool] Fetched enhanced context: {len(ocr_parts)} OCR items, {len(audio_parts)} Audio items.[/cyan]")
            
            return final_context if final_context else "No activity detected."

    except Exception as e:
        logger.error(f"Error connecting to Screenpipe: {e}")
        return f"Error connecting to Screenpipe: {str(e)}"

async def get_recent_ocr_text(minutes: int = 10) -> str:
    """Legacy wrapper for backward compatibility or simple use cases."""
    return await get_enhanced_context(minutes)

# LangChain tool definitions
from langchain.tools import tool

@tool
async def view_screen(mode: Literal["current", "history"] = "current", query: str = "") -> str:
    """View user's screen content and audio activity. 
    mode='current' for live activity (last 2-5min), 
    mode='history' to search past screen/audio text by query."""
    if mode == "history" and query:
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "q": query,
                    "limit": 15,
                    "content_type": "all"
                }
                resp = await client.get(f"{SCREENPIPE_URL}/search", params=params, timeout=10.0)
                if resp.status_code == 200:
                    items = resp.json().get("data", [])
                    if not items: return f"No results found for '{query}'"
                    
                    results = []
                    for item in items:
                        itype = item.get("type", "OCR")
                        content = item.get("content", {})
                        ts = content.get("timestamp", "unknown")
                        txt = content.get("text", "")
                        app = content.get("app_name", "N/A")
                        results.append(f"[{ts}] [{itype}] [{app}] {txt}")
                    return "\n---\n".join(results)
                return f"Error: {resp.text}"
        except Exception as e:
            return f"Error querying history: {e}"
    else:
        return await get_enhanced_context(minutes=2)

screen_tools = [view_screen]
