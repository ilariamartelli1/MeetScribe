"""Typed, model-independent contracts for draft minutes and JSON export v1."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.asr_contract import Language

ReviewStatus = Literal["DRAFT / UNREVIEWED", "REVIEWED"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MinutesTopic(_StrictModel):
    text: str = Field(description="A concise topic supported by cited segments.")
    source_segment_ids: tuple[str, ...] = Field(
        description="IDs of source segments supporting this topic."
    )


class MinutesDecision(_StrictModel):
    text: str = Field(description="A decision that was actually made.")
    source_segment_ids: tuple[str, ...] = Field(
        description="IDs of source segments supporting this decision."
    )


class MinutesActionItem(_StrictModel):
    description: str = Field(description="The task explicitly agreed in the meeting.")
    owner_name: str | None = Field(
        description="An explicitly named owner, or null when not specified."
    )
    owner_speaker_id: str | None = Field(
        description="An anonymous voice ID, or null when attribution is unknown."
    )
    deadline_text: str | None = Field(
        description=(
            "Source-grounded deadline wording; a uniquely matched weekday "
            "transcription error may be normalized."
        )
    )
    deadline_date: date | None = Field(
        description="ISO calendar date only when explicitly stated and unambiguous."
    )
    source_segment_ids: tuple[str, ...] = Field(
        description="IDs of source segments supporting this action item."
    )


class MinutesDraft(_StrictModel):
    summary: str = Field(description="A concise summary based only on the transcript.")
    topics: tuple[MinutesTopic, ...] = Field(
        description="Important topics; use an empty array when none are supported."
    )
    decisions: tuple[MinutesDecision, ...] = Field(
        description="Decisions actually made; use an empty array when none are supported."
    )
    action_items: tuple[MinutesActionItem, ...] = Field(
        description="Explicit tasks; use an empty array when none are supported."
    )


class TranscriptSegmentV1(_StrictModel):
    id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker_id: str | None
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_segment(self) -> "TranscriptSegmentV1":
        if self.end_ms <= self.start_ms:
            raise ValueError("Transcript segment end must be after its start.")
        if not self.id.strip() or not self.text.strip():
            raise ValueError("Transcript segment ID and text must be non-empty.")
        if self.speaker_id is not None and not self.speaker_id.strip():
            raise ValueError("Transcript speaker ID must be non-empty or null.")
        return self


class MeetingExportV1(_StrictModel):
    schema_version: Literal[1]
    meeting_id: str = Field(min_length=1)
    language: Language
    review_status: ReviewStatus
    stale: bool
    transcript_revision: int = Field(ge=1)
    minutes_revision: int = Field(ge=1)
    minutes_source_transcript_revision: int = Field(ge=1)
    segments: tuple[TranscriptSegmentV1, ...]
    minutes: MinutesDraft

    @model_validator(mode="after")
    def validate_export_references(self) -> "MeetingExportV1":
        if not self.meeting_id.strip():
            raise ValueError("Meeting ID must be non-empty.")
        if not self.segments:
            raise ValueError("Meeting export must contain transcript segments.")
        if self.minutes_source_transcript_revision > self.transcript_revision:
            raise ValueError("Minutes cannot cite a future transcript revision.")
        expected_stale = (
            self.minutes_source_transcript_revision != self.transcript_revision
        )
        if self.stale != expected_stale:
            raise ValueError("Stale status must match the minutes source revision.")

        segment_ids = {segment.id for segment in self.segments}
        if len(segment_ids) != len(self.segments):
            raise ValueError("Transcript segment IDs must be unique.")
        speaker_ids = {
            segment.speaker_id
            for segment in self.segments
            if segment.speaker_id is not None
        }
        source_id_groups = (
            *(topic.source_segment_ids for topic in self.minutes.topics),
            *(decision.source_segment_ids for decision in self.minutes.decisions),
            *(item.source_segment_ids for item in self.minutes.action_items),
        )
        for source_ids in source_id_groups:
            if not set(source_ids).issubset(segment_ids):
                raise ValueError("Minutes contain an unsupported source segment ID.")
        for item in self.minutes.action_items:
            if (
                item.owner_speaker_id is not None
                and item.owner_speaker_id not in speaker_ids
            ):
                raise ValueError("Action item contains an unsupported speaker ID.")
        return self
