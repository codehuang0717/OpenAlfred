from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from core.config import config
import os
from services.tool_observations import observed, fields, action

# Ensure API key is in environment for the underlying tool
if config.TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = config.TAVILY_API_KEY

@tool
async def web_search(query: str) -> str:
    """Search the web for real-time information, news, or specific facts."""
    search = TavilySearchResults(max_results=5)
    results = await search.ainvoke(query)
    
    if not results:
        return observed("No results found.", "未找到网页结果", outcome="empty")

    if not isinstance(results, list) or any(
        not isinstance(item, dict) for item in results
    ):
        raise ValueError("搜索服务未返回有效的来源列表")
    formatted_results = []
    for res in results:
        formatted_results.append(f"Source: {res.get('url')}\nContent: {res.get('content')}")

    return observed(
        "\n---\n".join(formatted_results),
        f"找到 {len(results)} 个网页来源",
        details=[
            {
                "label": f"来源 {i}",
                "value": str(r.get("title") or r.get("url") or "无标题")
                + "\n"
                + str(r.get("content") or "")[:1000],
            }
            for i, r in enumerate(results[:5], 1)
        ],
        actions=[
            action("external", str(r.get("title") or "打开来源")[:80], str(r["url"]))
            for r in results[:5]
            if str(r.get("url", "")).startswith("https://")
            and len(str(r["url"])) <= 2048
        ],
    )


search_tools = [web_search]
