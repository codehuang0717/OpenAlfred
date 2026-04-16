import httpx
import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("eye-tool")

SCREENPIPE_URL = "http://localhost:3030"

async def get_recent_ocr_text(minutes: int = 10) -> str:
    """
    Fetch and summarize OCR text captured by Screenpipe in the last X minutes.
    """
    try:
        # Calculate start time for the query
        # Screenpipe uses ISO 8601 strings
        start_time = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        
        async with httpx.AsyncClient() as client:
            # We search for OCR content within the time range
            params = {
                "limit": 50,
                "content_type": "ocr",
                "start_time": start_time
            }
            logger.info(f"Querying Screenpipe for OCR since {start_time}")
            resp = await client.get(f"{SCREENPIPE_URL}/search", params=params, timeout=10.0)
            
            if resp.status_code != 200:
                logger.error(f"Screenpipe API Error ({resp.status_code}): {resp.text}")
                return ""
            
            data = resp.json()
            items = data.get("data", [])
            
            if not items:
                return "No screen activity detected in the last few minutes."
            
            # Extract and deduplicate text
            # Screenpipe returns chunks of text from OCR
            seen_text = set()
            unique_content = []
            
            for item in items:
                content = item.get("content", {}).get("text", "").strip()
                if content and content not in seen_text:
                    seen_text.add(content)
                    unique_content.append(content)
            
            # Join the unique content into a summary string
            # We limit to the most recent items to avoid token bloat
            summary = "\n".join(unique_content[:20]) 
            return summary if summary else "Screen was blank or no text detected."

    except Exception as e:
        logger.error(f"Error fetching screen data: {e}")
        return f"Error connecting to Screenpipe: {str(e)}"

# LangChain tool definitions (for the chat agent)
from langchain.tools import tool

@tool
async def search_screen_history(query: str) -> str:
    """
    Search historical screen content (OCR) for a specific query.
    Useful for 'What did I see earlier?' or 'When did I look at X?'
    """
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

@tool
async def get_current_screen_context() -> str:
    """
    Get a summary of what is currently on the user's screen (last 2 minutes of OCR).
    """
    return await get_recent_ocr_text(minutes=2)
