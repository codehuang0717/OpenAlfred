"""Small owner-scoped tool observations shared by live events and history."""

from typing import Literal

from pydantic import BaseModel, Field

ToolStatus = Literal["running", "succeeded", "failed", "interrupted", "unknown"]


class ToolField(BaseModel):
    label: str = Field(max_length=80)
    value: str = Field(max_length=5000)


class ToolAction(BaseModel):
    kind: Literal["panel", "email", "email_draft", "image", "external", "settings"]
    label: str = Field(max_length=80)
    target: str = Field(max_length=2048)
    secondary: str | None = Field(default=None, max_length=128)


class ToolDisplay(BaseModel):
    version: Literal[1] = 1
    title: str = Field(max_length=80)
    status: ToolStatus
    outcome: Literal[
        "completed",
        "empty",
        "no_change",
        "exists",
        "blocked",
        "partial",
        "queued",
        "proposal",
        "truncated",
    ] = "completed"
    summary: str = Field(max_length=240)
    fields: list[ToolField] = Field(default_factory=list, max_length=40)
    actions: list[ToolAction] = Field(default_factory=list, max_length=10)
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_ms: int | None = Field(default=None, ge=0)
