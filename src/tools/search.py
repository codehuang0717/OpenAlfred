from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from core.config import config
import os

# Ensure API key is in environment for the underlying tool
if config.TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = config.TAVILY_API_KEY

@tool
async def web_search(query: str) -> str:
    """Search the web for real-time information, news, or specific facts."""
    search = TavilySearchResults(max_results=5)
    results = await search.ainvoke(query)
    
    if not results:
        return "No results found."
        
    formatted_results = []
    for res in results:
        formatted_results.append(f"Source: {res.get('url')}\nContent: {res.get('content')}")
        
    return "\n---\n".join(formatted_results)

search_tools = [web_search]
