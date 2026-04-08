from config import config
from langchain.tools import tool, ToolRuntime
from mem0 import MemoryClient

mem0_client = MemoryClient(api_key=config.MEM0_API_KEY)


@tool
def search_memory(query: str, runtime: ToolRuntime) -> str:
    """搜索用户的记忆和偏好"""
    state = runtime.state
    user_id = state.get("mem0_user_id", "default")
    results = mem0_client.search(query, user_id=user_id)
    if results and isinstance(results, dict):
        results = results.get("results", results)
    if isinstance(results, list):
        return "\n".join([r["memory"] for r in results]) if results else "无相关记忆"
    return str(results) if results else "无相关记忆"


@tool
def add_memory(content: str, runtime: ToolRuntime) -> str:
    """存储用户的交互内容到记忆"""
    state = runtime.state
    user_id = state.get("mem0_user_id", "default")
    messages = [{"role": "user", "content": content}]
    result = mem0_client.add(messages=messages, user_id=user_id)
    return str(result)


@tool
def delete_memory(id: str, runtime: ToolRuntime) -> str:
    """删除指定的记忆"""
    try:
        mem0_client.delete(memory_id=id)
        return f"Successfully deleted memory: {id}"
    except Exception as e:
        return f"Error deleting memory: {str(e)}"


memTools = [search_memory, add_memory, delete_memory]
