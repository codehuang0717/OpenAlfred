# src/schema.py
from typing import Annotated, Optional, TypedDict, Literal
from pydantic import BaseModel
from langgraph.graph.message import add_messages

class TodoDict(TypedDict):
    id: str
    title: str
    description: str
    emoji: str
    status: Literal["pending", "completed"]
    created_at: str
    completed_at: Optional[str]
    deleted: int
    notes: str
    expected_completion_at: Optional[str]

class AgentState(BaseModel):
    """State for the text-based chat agent."""
    todos: list[TodoDict] = []
    mem0_user_id: str = "default"
    tts_text: Optional[str] = None
    jump_to: str = ""
    structured_response: dict = {}
    model_selection: Optional[str] = "gpt-cloud"
    chat_session_id: Optional[str] = None

class VoiceAgentState(TypedDict):
    """Logically consistent state for the voice pipeline."""
    messages: Annotated[list, add_messages]
    session_id: str
    tts_text: Optional[str]
    model_selection: Optional[str]  # "gpt-cloud" or "gemma-local"
