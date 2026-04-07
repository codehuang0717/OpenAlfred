from langchain.tools import ToolRuntime, tool
from langchain.messages import ToolMessage
from langgraph.types import Command


@tool
async def speak(runtime: ToolRuntime, text: str) -> Command:
    """
    将文本转换为语音播报。仅在需要语音通知用户时调用。
    普通的文字对话不需要调用此工具。

    Args:
        text: 用于语音播报的文本（50字以内），用自然口语表达。
              例如: "已帮你添加了任务xxx"
              例如: "你今天有5个待办事项"

    Returns:
        语音播报已生成
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"语音播报已生成: {text}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "tts_text": text,
        }
    )


speak_tools = [speak]
