from typing import Annotated, Optional, Literal
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field
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

# ── Structured Output Schemas ──────────────────────────────────────────

class KnowledgeExtractionFact(BaseModel):
    """A single fact extracted from conversation about the user."""
    category: Literal["profile", "preferences", "relationship", "patterns"]
    fact: str = Field(description="The extracted fact about the user")
    importance: Literal["high", "medium", "low"] = "medium"


class KnowledgeExtractionResult(BaseModel):
    """Collection of newly extracted facts (empty if nothing new)."""
    facts: list[KnowledgeExtractionFact] = Field(
        default_factory=list,
        description="Newly extracted facts. Empty list if no new information found."
    )


class VoiceAgentState(TypedDict):
    """Logically consistent state for the voice pipeline."""
    messages: Annotated[list, add_messages]
    session_id: str
    tts_text: Optional[str]
    model_selection: Optional[str]  # "gpt-cloud" or "gemma-local"
