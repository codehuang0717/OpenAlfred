from typing import Annotated, Optional, Literal
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel
from langgraph.graph.message import add_messages

class TodoDict(TypedDict):
    id: str
    title: str
    description: str
    emoji: str
    status: Literal["pending", "completed"]
    created_at: str
    completed_at: NotRequired[Optional[str]]
    deleted: int
    notes: str
    expected_completion_at: NotRequired[Optional[str]]
    scheduled_start_at: NotRequired[Optional[str]]

class AgentState(BaseModel):
    """State for the text-based chat agent."""
    messages: Annotated[list, add_messages] = []
    todos: list[TodoDict] = []
    model_selection: Optional[str] = "gpt-cloud"
    conversation_summary: str = ""
    summarized_count: int = 0
    user_id: str = ""
    system_instruction: str = ""
    extraction_counter: int = 0
    extracted_msg_count: int = 0

class VoiceAgentState(TypedDict):
    """Logically consistent state for the voice pipeline."""
    messages: Annotated[list, add_messages]
    session_id: str
    tts_text: Optional[str]
    model_selection: Optional[str]  # "gpt-cloud" or "gemma-local"
