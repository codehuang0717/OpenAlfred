from typing import Annotated, Optional, Literal
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


def _last_write(existing, update):
    """Last write wins when parallel tools each snapshot the same channel."""
    return update if update is not None else existing

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
    todos: Annotated[list[TodoDict], _last_write] = []
    model_selection: Optional[str] = "gpt-cloud"
    conversation_summary: str = ""
    summarized_count: int = 0
    user_id: str = ""
    system_instruction: str = ""
    runtime_context: str = ""
    extraction_counter: int = 0
    extracted_msg_count: int = 0
    prepared_messages: list = []
    context_metrics: dict = {}
    context_error: str = ""

# ── Structured Output Schemas ──────────────────────────────────────────

class KnowledgeExtractionFact(BaseModel):
    """A single fact extracted from conversation about the user."""
    category: Literal["profile", "preferences", "relationship", "patterns"]
    fact: str = Field(min_length=2, max_length=250, description="One concise durable fact directly supported by the user quote")
    importance: Literal["high", "medium", "low"] = "medium"
    message_index: int = Field(ge=0, description="Index of the source user message")
    quote: str = Field(min_length=2, max_length=1000, description="Exact continuous quote from that user message")
    retention_basis: Literal["stable_identity", "lasting_preference", "stable_relationship", "self_reported_habit"]


class KnowledgeExtractionResult(BaseModel):
    """Collection of newly extracted facts (empty if nothing new)."""
    facts: list[KnowledgeExtractionFact] = Field(
        default_factory=list,
        max_length=3,
        description="Newly extracted facts. Empty list if no new information found."
    )


class VoiceAgentState(TypedDict):
    """Logically consistent state for the voice pipeline."""
    messages: Annotated[list, add_messages]
    session_id: str
    tts_text: Optional[str]
    model_selection: Optional[str]  # "gpt-cloud" or "gemma-local"
