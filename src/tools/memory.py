from core.config import config
from langchain.tools import tool, ToolRuntime
from mem0 import MemoryClient

mem0_client = MemoryClient(api_key=config.MEM0_API_KEY)

def _get_user_id(runtime: ToolRuntime) -> str:
    """Extract user_id from RunnableConfig populated by LangGraph Auth."""
    if hasattr(runtime, "config") and runtime.config:
        conf = runtime.config.get("configurable", {})
        auth_user = conf.get("langgraph_auth_user", {})
        if isinstance(auth_user, dict) and "identity" in auth_user:
            return auth_user["identity"]
            
        metadata = runtime.config.get("metadata", {})
        if "owner" in metadata:
            return metadata["owner"]
            
        if "thread_owner" in conf:
            return conf["thread_owner"]
    if hasattr(runtime, "state") and runtime.state:
        if isinstance(runtime.state, dict): return runtime.state.get("user_id", "default")
        return getattr(runtime.state, "user_id", "default")
    return "default"



@tool
def search_memory(query: str, runtime: ToolRuntime) -> str:
    """Search user's long-term memory and preferences."""
    user_id = _get_user_id(runtime)
    results = mem0_client.search(query=query, filters={"AND": [{"user_id": user_id}]})
    if results and isinstance(results, dict):
        results = results.get("results", results)
    if isinstance(results, list):
        return "\n".join([r["memory"] for r in results]) if results else "无相关记忆"
    return str(results) if results else "无相关记忆"


@tool
def add_memory(content: str, runtime: ToolRuntime) -> str:
    """Store user interaction content into long-term memory."""
    user_id = _get_user_id(runtime)
    messages = [{"role": "user", "content": content}]
    result = mem0_client.add(messages=messages, user_id=user_id)
    return str(result)


memTools = [search_memory, add_memory]
