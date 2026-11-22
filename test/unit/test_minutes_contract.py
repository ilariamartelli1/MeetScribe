"""Verify the typed draft-minutes and JSON export contracts."""

import json
from datetime import date

import pytest
from pydantic import ValidationError

from app.core.minutes_contract import (
    MeetingExportV1,
    MinutesActionItem,
    MinutesDecision,
    MinutesDraft,
    MinutesTopic,
    TranscriptSegmentV1,
)


def _draft() -> MinutesDraft:
    return MinutesDraft(
        summary="The team agreed to review the plan.",
        topics=(MinutesTopic(text="Planning", source_segment_ids=("segment-1",)),),
        decisions=(
            MinutesDecision(text="Review the plan", source_segment_ids=("segment-1",)),
        ),
        action_items=(
            MinutesActionItem(
                description="Send the updated plan",
                owner_name="Elena",
                owner_speaker_id=None,
                deadline_text="March 5, 2027",
                deadline_date=date(2027, 3, 5),
                source_segment_ids=("segment-1",),
            ),
        ),
    )


def _export() -> MeetingExportV1:
    return MeetingExportV1(
        schema_version=1,
        meeting_id="job-123",
        language="en",
        review_status="DRAFT / UNREVIEWED",
        stale=False,
        transcript_revision=1,
        minutes_revision=1,
        minutes_source_transcript_revision=1,
        segments=(
            TranscriptSegmentV1(
                id="segment-1",
                start_ms=1250,
                end_ms=2500,
                speaker_id="voice-1",
                text="Elena will send the updated plan by March 5, 2027.",
            ),
        ),
        minutes=_draft(),
    )


def test_minutes_response_schema_requires_all_top_level_fields() -> None:
    response = {
        "summary": "A concise summary.",
        "topics": [],
        "decisions": [],
    }

    with pytest.raises(ValidationError):
        MinutesDraft.model_validate_json(json.dumps(response))


def test_minutes_response_schema_rejects_extra_fields() -> None:
    response = {
        "summary": "A concise summary.",
        "topics": [],
        "decisions": [],
        "action_items": [],
        "speaker_transcript": "not part of the minutes contract",
    }

    with pytest.raises(ValidationError):
        MinutesDraft.model_validate_json(json.dumps(response))


def test_minutes_action_item_requires_nullable_fields() -> None:
    response = {
        "summary": "A concise summary.",
        "topics": [],
        "decisions": [],
        "action_items": [
            {"description": "Review the plan", "source_segment_ids": ["segment-1"]}
        ],
    }

    with pytest.raises(ValidationError):
        MinutesDraft.model_validate_json(json.dumps(response))


def test_export_v1_round_trips_date_unicode_and_millisecond_timestamps() -> None:
    export = _export()

    encoded = export.model_dump_json()
    decoded = MeetingExportV1.model_validate_json(encoded)

    assert decoded.schema_version == 1
    assert decoded.meeting_id == "job-123"
    assert decoded.language == "en"
    assert decoded.segments[0].start_ms == 1250
    assert decoded.segments[0].end_ms == 2500
    assert decoded.minutes.summary == "The team agreed to review the plan."
    assert decoded.minutes.action_items[0].owner_name == "Elena"
    assert decoded.minutes.action_items[0].deadline_date == date(2027, 3, 5)

    italian = export.model_copy(
        update={
            "language": "it",
            "minutes": _draft().model_copy(
                update={"summary": "La riunione ha deciso di rivedere il piano."}
            ),
        }
    )
    italian_round_trip = MeetingExportV1.model_validate_json(italian.model_dump_json())
    assert (
        italian_round_trip.minutes.summary
        == "La riunione ha deciso di rivedere il piano."
    )


def test_export_v1_rejects_unsupported_versions_and_citations() -> None:
    export_data = json.loads(_export().model_dump_json())
    export_data["schema_version"] = 2
    with pytest.raises(ValidationError):
        MeetingExportV1.model_validate_json(json.dumps(export_data))

    export_data = json.loads(_export().model_dump_json())
    export_data["minutes"]["topics"][0]["source_segment_ids"] = ["missing"]
    with pytest.raises(ValidationError, match="unsupported source segment ID"):
        MeetingExportV1.model_validate_json(json.dumps(export_data))


def test_export_v1_stale_flag_must_match_transcript_revisions() -> None:
    export_data = json.loads(_export().model_dump_json())
    export_data["transcript_revision"] = 2

    with pytest.raises(ValidationError, match="Stale status"):
        MeetingExportV1.model_validate_json(json.dumps(export_data))
