"""Document existing JSON responses without changing serialization or filtering.

Router ``responses`` declarations provide the client contract. Runtime behavior
stays in the existing services; contract tests validate representative outputs.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    status: str


class TextResponse(BaseModel):
    text: str


class UserIdentity(BaseModel):
    id: str
    username: str
    display_name: str


class SipAccount(BaseModel):
    extension: str | None
    password: str | None
    server: str | None = None
    note: str | None = None


class AuthResponse(BaseModel):
    token: str
    user: UserIdentity
    sip: SipAccount | None = None


class ProfileResponse(UserIdentity):
    created_at: str | None
    sip_extension: str | None
    avatar_url: str


class ProfileUpdateResponse(StatusResponse):
    display_name: str


class AvatarResponse(StatusResponse):
    avatar_url: str


class AgentConfigResponse(BaseModel):
    agent_avatar_url: str


class AgentAvatarResponse(StatusResponse, AgentConfigResponse):
    pass


class TokenResponse(BaseModel):
    token: str


class TimezoneResponse(BaseModel):
    timezone: str


class MemoryFile(BaseModel):
    filename: str
    title: str
    content: str


class MemoryUpdateResponse(StatusResponse):
    filename: str


class CreatedThread(BaseModel):
    thread_id: str
    title: str
    created_at: str


class ThreadResponse(CreatedThread):
    updated_at: str


class CallThreadResponse(ThreadResponse):
    direction: Literal["inbound", "outbound"]
    room_name: str


class TitleResponse(BaseModel):
    title: str


class ThreadRenameResponse(StatusResponse, TitleResponse):
    pass


class ToolCallResponse(BaseModel):
    id: str | None = None
    name: str
    status: Literal["calling", "done"]


class TextStep(BaseModel):
    type: Literal["text"]
    id: str
    content: str


class ToolsStep(BaseModel):
    type: Literal["tools"]
    id: str
    tools: list[ToolCallResponse]


class CodingTaskReference(BaseModel):
    type: Literal["coding_task"]
    job_id: str
    app_id: str
    title: str


class CodingTaskStep(BaseModel):
    type: Literal["coding_task"]
    id: str
    task: CodingTaskReference


class EmailDraftReference(BaseModel):
    type: Literal["email_draft"]
    draft_id: str
    subject: str


class EmailDraftStep(BaseModel):
    type: Literal["email_draft"]
    id: str
    draft: EmailDraftReference


ChatStep = Annotated[TextStep | ToolsStep | CodingTaskStep | EmailDraftStep, Field(discriminator="type")]


class ChatMessageResponse(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]
    steps: list[ChatStep] | None = None
    tools: list[ToolCallResponse] | None = None
    outcome: Literal["tools", "completed", "failed"] | None = None
    failure: str | None = None


class ModelResponse(BaseModel):
    id: str
    name: str
    provider: str
    icon: str
    description: str


class ModelSelectionResponse(BaseModel):
    model_selection: str


class ModelSelectionUpdateResponse(StatusResponse, ModelSelectionResponse):
    pass


class OnlineResponse(BaseModel):
    online: bool


class TodoResponse(BaseModel):
    id: str
    user_id: str
    title: str
    description: str
    emoji: str
    status: Literal["pending", "completed"]
    created_at: str
    completed_at: str | None
    deleted: int
    notes: str
    expected_completion_at: str | None
    scheduled_start_at: str | None


class ReminderResponse(BaseModel):
    id: str
    user_id: str
    title: str | None
    subtitle: str | None
    body: str
    scheduled_at: str
    sent: int
    level: str
    sound: str | None
    created_at: str
    delivery_method: str
    audio_path: str


CodingStatus = Literal["queued", "generating", "ready", "failed", "cancelled", "interrupted"]


class UserAppBase(BaseModel):
    id: str
    user_id: str
    kind: str
    title: str
    spec: Any
    status: Literal["draft", "ready", "published", "failed"]
    published_revision_id: str | None
    has_unpublished_revision: int
    created_at: str
    updated_at: str


class UserAppResponse(UserAppBase):
    job_status: CodingStatus | None
    job_stage: str | None
    job_id: str | None
    job_report: str | None
    job_error: str | None


class UserAppRevisionResponse(BaseModel):
    id: str
    revision_number: int
    renderer: Literal["catalog", "html"]
    source: Any
    validation: dict[str, Any]
    status: Literal["ready", "published"]
    created_at: str


class CodingJobState(BaseModel):
    id: str
    status: CodingStatus
    stage: str
    report: str | None
    model: str
    error: str | None
    revision_id: str | None
    metrics: dict[str, Any]
    created_at: str
    updated_at: str


class LatestCodingJob(CodingJobState):
    epoch: int


class UserAppDetailsResponse(UserAppBase):
    revisions: list[UserAppRevisionResponse]
    latest_job: LatestCodingJob | None


class QueuedRevisionResponse(BaseModel):
    app_id: str
    job_id: str
    status: Literal["queued"]


class CodingJobResponse(CodingJobState):
    app_id: str
    title: str


class CodingJobEvent(BaseModel):
    seq: int
    stage: str
    message: str
    created_at: str


class CodingJobEventsResponse(BaseModel):
    job: CodingJobResponse
    events: list[CodingJobEvent]


class NotifyConfigResponse(BaseModel):
    bark_url: str


class NotifyUpdateResponse(StatusResponse, NotifyConfigResponse):
    pass


class NotifyTestResponse(StatusResponse):
    message: str


class WeatherLocationResponse(BaseModel):
    latitude: float
    longitude: float
    accuracy: float | None = None
    label: str
    source: str


class SavedWeatherLocationResponse(BaseModel):
    location: WeatherLocationResponse | None


class WeatherLocationUpdateResponse(StatusResponse):
    location: WeatherLocationResponse


class CurrentWeather(BaseModel):
    time: str | None
    weather: str
    temperature: float | None
    apparent_temperature: float | None
    humidity: float | None
    precipitation: float | None
    wind_speed: float | None


class DailyWeather(BaseModel):
    date: str | None = None
    weather: str | None = None
    temperature_min: float | None = None
    temperature_max: float | None = None
    rain_probability: float | None = None


class WeatherSummaryResponse(BaseModel):
    location: WeatherLocationResponse
    timezone: str
    current: CurrentWeather
    daily: DailyWeather
    suggestions: list[str]
    updated_at: str
    stale: bool


class WeatherResponse(BaseModel):
    weather: WeatherSummaryResponse | None
    needs_location: bool


class OnboardingResponse(BaseModel):
    seen: bool


class OnboardingUpdateResponse(StatusResponse, OnboardingResponse):
    pass


class SupervisorResponse(BaseModel):
    binding_required: bool
    recording_enabled: bool
    smart_supervision_enabled: bool
    supervisor_running: bool
    screenpipe_running: bool
    analysis_running: bool
    error: str | None


class EmailConfigResponse(BaseModel):
    account_id: str
    email_address: str
    provider: str
    imap_server: str
    imap_port: int
    smtp_server: str
    smtp_port: int
    created_at: str


class EmailConfigUpdateResponse(StatusResponse):
    account_id: str


class EmailResponse(BaseModel):
    id: str
    account_id: str
    account_email: str
    subject: str
    sender: str = Field(alias="from")
    date: str


class EmailContentResponse(EmailResponse):
    body: str
    html_body: str


class KnowledgeDocumentResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    title: str
    file_type: str
    chunk_count: int
    created_at: str


class KnowledgeSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    heading: str
    content: str
    score: float
    image_count: int


class KnowledgeSearchResponse(BaseModel):
    query: str
    results: list[KnowledgeSearchResult]


class IngestTaskResponse(BaseModel):
    task_id: str
    user_id: str
    status: Literal["processing", "completed", "failed"]
    stage: Literal["parsing", "images", "embedding", "done"]
    progress: int
    result: KnowledgeDocumentResponse | None
    error: str | None
    updated_at: float


class IngestAcceptedResponse(BaseModel):
    task_id: str
    status: Literal["processing", "completed"]


class SelectedFileResponse(BaseModel):
    filepath: str


def json_response(model: Any, status_code: int = 200) -> dict:
    """Attach a documented success schema without altering existing responses."""
    return {status_code: {"model": model}}
