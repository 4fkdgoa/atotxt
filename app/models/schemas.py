"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Task Status ---

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Transcript ---

class Utterance(BaseModel):
    """A single utterance from a speaker."""
    speaker: str = Field(..., description="Speaker identifier (e.g. SPEAKER_00)")
    text: str = Field(..., description="Transcribed text")
    start: Optional[float] = Field(None, description="Start time in seconds")
    end: Optional[float] = Field(None, description="End time in seconds")


class Transcript(BaseModel):
    """Full transcript with speaker-separated utterances."""
    language: str = Field("ko", description="Detected language")
    utterances: list[Utterance] = Field(default_factory=list)

    def to_dialogue_text(self) -> str:
        """Convert to formatted dialogue string for LLM input."""
        lines = []
        for u in self.utterances:
            lines.append(f"{u.speaker}: {u.text}")
        return "\n".join(lines)


# --- Summary ---

class TopicSummary(BaseModel):
    """Summary for a single topic."""
    topic: str = Field(..., description="Topic name")
    summary: str = Field(..., description="Summary of the topic")
    speakers: list[str] = Field(default_factory=list, description="Speakers involved")


class SummaryResult(BaseModel):
    """Full summary result with topics."""
    overall_summary: str = Field("", description="Overall summary of the conversation")
    topics: list[TopicSummary] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list, description="Action items extracted")


# --- IT Meeting Specific Schemas ---

class MeetingType(str, Enum):
    """Meeting type for prompt selection."""
    GENERAL = "general"
    IT_STANDUP = "it_standup"


class TechOverview(BaseModel):
    """Technical overview from the tech lead (usually SPEAKER_00)."""
    description: str = Field("", description="Technical context/architecture discussed")
    technologies: list[str] = Field(default_factory=list, description="Tech stack mentioned")


class ITTopicSummary(BaseModel):
    """Summary for a single topic in IT meetings."""
    topic: str = Field(..., description="Topic name (e.g., 'Login API bug fix')")
    summary: str = Field(..., description="Progress and content summary")
    speakers: list[str] = Field(default_factory=list)
    status: str = Field("in_progress", description="done | in_progress | blocked | planned")
    issue_refs: list[str] = Field(default_factory=list, description="Issue/PR numbers")


class Blocker(BaseModel):
    """A blocker identified in the meeting."""
    issue: str = Field(..., description="Blocker description")
    owner: str = Field("", description="Owner speaker")
    needs: str = Field("", description="Required support/resources")


class ActionItem(BaseModel):
    """Structured action item with assignee."""
    task: str = Field(..., description="Task description")
    assignee: str = Field("", description="Assignee (SPEAKER_XX or name)")
    deadline: Optional[str] = Field(None, description="Deadline if mentioned")
    priority: str = Field("medium", description="high | medium | low")


class ITSummaryResult(BaseModel):
    """Full summary result for IT standup/sprint meetings."""
    overall_summary: str = Field("", description="Overall meeting summary")
    tech_overview: Optional[TechOverview] = None
    topics: list[ITTopicSummary] = Field(default_factory=list)
    blockers: list[Blocker] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list, description="Decisions made")
    next_steps: list[str] = Field(default_factory=list, description="Next steps")


# --- API Response ---

class TranscriptionResponse(BaseModel):
    """Full response combining transcript and summary."""
    task_id: str
    status: TaskStatus
    transcript: Optional[Transcript] = None
    summary: Optional[SummaryResult] = None
    error: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Response for task status queries."""
    task_id: str
    status: TaskStatus
    progress: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Response returned immediately after upload."""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    message: str = "Audio file uploaded. Processing started."


# --- GPU Info ---

class GPUInfo(BaseModel):
    """GPU hardware information."""
    name: str
    vram_total_mb: int
    vram_free_mb: int
    cuda_available: bool


class SystemInfo(BaseModel):
    """System information including GPU details."""
    device: str
    whisper_model: str
    compute_type: str
    ollama_model: str
    gpus: list[GPUInfo] = Field(default_factory=list)
