"""Versioned email draft and send receipt contracts."""
from typing import Literal
from pydantic import BaseModel, Field, field_validator

MailStatus = Literal["draft", "proposal", "adopted", "queued", "sending", "accepted", "partial", "failed", "unknown", "cancelled"]


class DraftFields(BaseModel):
    account_id: str = Field(default="", max_length=128)
    to_address: str = Field(default="", max_length=320)
    subject: str = Field(default="", max_length=998)
    body: str = Field(default="", max_length=200_000)

    @field_validator("account_id", "to_address", "subject")
    @classmethod
    def single_line(cls, value: str) -> str:
        if "\r" in value or "\n" in value:
            raise ValueError("此字段不能包含换行")
        return value.strip()


class DraftCreate(DraftFields):
    client_key: str = Field(min_length=8, max_length=128)


class DraftUpdate(DraftFields):
    revision: int = Field(ge=1)


class DraftSend(BaseModel):
    revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=128)


class DraftAdopt(BaseModel):
    revision: int = Field(ge=1)


class SendJobResponse(BaseModel):
    id: str
    draft_id: str
    revision: int
    status: MailStatus
    phase: str
    account_id: str
    from_address: str
    to_address: str
    subject: str
    body: str
    message_id: str
    error: str | None
    created_at: str
    updated_at: str
    accepted_at: str | None


class DraftResponse(DraftFields):
    id: str
    revision: int
    status: MailStatus
    created_at: str
    updated_at: str
    proposal_for: str | None
    base_revision: int | None
    last_job: SendJobResponse | None = None
