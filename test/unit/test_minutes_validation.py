"""Verify minutes response parsing, evidence checks and export assembly."""

import json
from dataclasses import replace
from datetime import date

import pytest

from app.asr_contract import Language, ModelProvenance, TranscriptSegment
from app.diarization_contract import Diarization, DiarizationProvenance
from app.minutes_contract import (
    MinutesActionItem,
    MinutesDecision,
    MinutesDraft,
    MinutesTopic,
)
from app.minutes_validation import (
    InvalidTranscriptEvidenceError,
    MinutesCompletenessError,
    MinutesEvidenceError,
    MinutesResponseShapeError,
    build_meeting_export_v1,
    parse_minutes_response,
    transcript_segments_v1,
    validate_minutes_completeness,
    validate_minutes_evidence,
)
from app.speaker_alignment import AttributedTranscript, SpeakerAttribution

ASR_PROVENANCE = ModelProvenance(
    "fake", "1.0", "asr", "revision", "digest", "cpu", "int8", True, 5
)
DIARIZATION_PROVENANCE = DiarizationProvenance(
    "fake", "1.0", "diarizer", "revision", "cpu"
)


def _transcript(
    *texts: str,
    speakers: tuple[str | None, ...] | None = None,
    language: Language = "en",
) -> AttributedTranscript:
    if speakers is None:
        speakers = tuple("Voice 1" for _ in texts)
    assert len(speakers) == len(texts)
    segments = tuple(
        TranscriptSegment(index * 1000, (index + 1) * 1000, text)
        for index, text in enumerate(texts)
    )
    attributions = tuple(
        SpeakerAttribution(segment.start_ms, segment.end_ms, speaker)
        for segment, speaker in zip(segments, speakers, strict=True)
    )
    duration_ms = len(segments) * 1000
    return AttributedTranscript(
        language=language,
        duration_ms=duration_ms,
        segments=segments,
        provenance=ASR_PROVENANCE,
        diarization=Diarization(duration_ms, (), (), DIARIZATION_PROVENANCE),
        attributions=attributions,
    )


def _action(
    *,
    source_segment_ids: tuple[str, ...] = ("segment-1",),
    owner_name: str | None = None,
    owner_speaker_id: str | None = None,
    deadline_text: str | None = None,
    deadline_date: date | None = None,
) -> MinutesActionItem:
    return MinutesActionItem(
        description="Send the updated plan",
        owner_name=owner_name,
        owner_speaker_id=owner_speaker_id,
        deadline_text=deadline_text,
        deadline_date=deadline_date,
        source_segment_ids=source_segment_ids,
    )


def _minutes(
    action: MinutesActionItem | None = None,
    *,
    summary: str = "The meeting covered the project plan.",
    topics: tuple[MinutesTopic, ...] = (
        MinutesTopic(text="Project plan", source_segment_ids=("segment-1",)),
    ),
    decisions: tuple[MinutesDecision, ...] = (),
) -> MinutesDraft:
    return MinutesDraft(
        summary=summary,
        topics=topics,
        decisions=decisions,
        action_items=() if action is None else (action,),
    )


def test_malformed_json_is_reported_as_a_shape_error() -> None:
    with pytest.raises(MinutesResponseShapeError, match="required schema"):
        parse_minutes_response('{"summary":')


def test_incomplete_content_and_empty_citations_are_rejected_separately() -> None:
    with pytest.raises(MinutesCompletenessError, match="summary"):
        validate_minutes_completeness(_minutes(summary="  "))

    incomplete_topic = _minutes(
        topics=(MinutesTopic(text="Project plan", source_segment_ids=()),)
    )
    with pytest.raises(MinutesCompletenessError, match="at least one source segment"):
        validate_minutes_completeness(incomplete_topic)


@pytest.mark.parametrize("collection", ["topic", "decision", "action"])
def test_unsupported_source_segment_ids_are_rejected(collection: str) -> None:
    action = _action(source_segment_ids=("unknown-segment",))
    minutes = _minutes(action)
    if collection == "topic":
        minutes = _minutes(
            topics=(
                MinutesTopic(
                    text="Project plan", source_segment_ids=("unknown-segment",)
                ),
            )
        )
    elif collection == "decision":
        minutes = _minutes(
            decisions=(
                MinutesDecision(
                    text="Approve the plan", source_segment_ids=("unknown-segment",)
                ),
            )
        )

    with pytest.raises(MinutesEvidenceError, match="unknown transcript segment"):
        validate_minutes_evidence(minutes, transcript_segments_v1(_transcript("Plan")))


@pytest.mark.parametrize("pronoun", ["I", "we"])
def test_first_person_pronouns_are_not_named_owners(pronoun: str) -> None:
    transcript = _transcript(f"{pronoun} will send the updated plan.")

    with pytest.raises(MinutesEvidenceError, match="pronoun"):
        validate_minutes_evidence(
            _minutes(_action(owner_name=pronoun)), transcript_segments_v1(transcript)
        )


def test_named_nonparticipant_is_not_assigned_from_an_unrelated_mention() -> None:
    transcript = _transcript(
        "Alice is attending the next meeting.",
        "I will send the updated plan.",
    )
    action = _action(owner_name="Alice", source_segment_ids=("segment-2",))

    with pytest.raises(MinutesEvidenceError, match="owner_name"):
        validate_minutes_evidence(_minutes(action), transcript_segments_v1(transcript))


def test_anonymous_owner_remains_separate_from_named_owner() -> None:
    transcript = _transcript("I will send the updated plan.")
    action = _action(owner_speaker_id="voice-1")

    validate_minutes_evidence(_minutes(action), transcript_segments_v1(transcript))


@pytest.mark.parametrize(
    ("text", "deadline_text", "deadline_date"),
    [
        ("We will discuss the plan later.", None, None),
        ("Finish the draft by next Friday.", "next Friday", None),
        ("Finish the draft by 03/04/2027.", "03/04/2027", None),
    ],
)
def test_absent_relative_and_ambiguous_dates_remain_unresolved(
    text: str, deadline_text: str | None, deadline_date: date | None
) -> None:
    transcript = _transcript(text)
    action = _action(deadline_text=deadline_text, deadline_date=deadline_date)

    validate_minutes_evidence(_minutes(action), transcript_segments_v1(transcript))


@pytest.mark.parametrize(
    ("text", "deadline_text", "deadline_date"),
    [
        (
            "Finish the draft by next Friday.",
            "next Friday",
            date(2027, 3, 5),
        ),
        (
            "Finish the draft by 03/04/2027.",
            "03/04/2027",
            date(2027, 4, 3),
        ),
        (
            "Finish on March 5 or April 5, 2027.",
            "March 5 or April 5, 2027",
            date(2027, 4, 5),
        ),
    ],
)
def test_upload_time_or_ambiguous_numbers_cannot_supply_a_deadline_date(
    text: str, deadline_text: str, deadline_date: date
) -> None:
    transcript = _transcript(text)
    action = _action(deadline_text=deadline_text, deadline_date=deadline_date)

    with pytest.raises(MinutesEvidenceError, match="unambiguous absolute date"):
        validate_minutes_evidence(_minutes(action), transcript_segments_v1(transcript))


@pytest.mark.parametrize(
    ("text", "deadline_text", "deadline_date"),
    [
        (
            "Elena will send the plan by March 5, 2027.",
            "March 5, 2027",
            date(2027, 3, 5),
        ),
        (
            "Elena will send the plan by 13/04/2027.",
            "13/04/2027",
            date(2027, 4, 13),
        ),
        (
            "Elena invierà il piano entro il 5 marzo 2027.",
            "5 marzo 2027",
            date(2027, 3, 5),
        ),
        (
            "Elena invierà il piano entro il 5 mag 2027.",
            "5 mag 2027",
            date(2027, 5, 5),
        ),
    ],
)
def test_explicit_unambiguous_absolute_dates_are_supported(
    text: str, deadline_text: str, deadline_date: date
) -> None:
    transcript = _transcript(text)
    action = _action(
        owner_name="Elena",
        deadline_text=deadline_text,
        deadline_date=deadline_date,
    )

    validate_minutes_evidence(_minutes(action), transcript_segments_v1(transcript))


def test_deadline_wording_must_be_present_in_cited_source() -> None:
    transcript = _transcript("Elena will send the plan next Friday.")
    action = _action(deadline_text="the following Friday")

    with pytest.raises(MinutesEvidenceError, match="deadline_text"):
        validate_minutes_evidence(_minutes(action), transcript_segments_v1(transcript))


def test_transcript_export_assigns_stable_ids_and_keeps_original_milliseconds() -> None:
    transcript = _transcript(
        "Elena will send the plan by March 5, 2027.",
        "The team will review it.",
        speakers=("Voice 1", "Voice 2"),
        language="it",
    )
    action = _action(
        owner_name="Elena",
        owner_speaker_id="voice-1",
        deadline_text="March 5, 2027",
        deadline_date=date(2027, 3, 5),
    )
    minutes = _minutes(action)

    first = transcript_segments_v1(transcript)
    second = transcript_segments_v1(transcript)
    export = build_meeting_export_v1("job-456", transcript, minutes)

    assert first == second
    assert [(segment.id, segment.start_ms, segment.end_ms) for segment in first] == [
        ("segment-1", 0, 1000),
        ("segment-2", 1000, 2000),
    ]
    assert [segment.speaker_id for segment in first] == ["voice-1", "voice-2"]
    assert export.meeting_id == "job-456"
    assert export.language == "it"
    assert export.review_status == "DRAFT / UNREVIEWED"
    assert export.stale is False
    assert export.transcript_revision == export.minutes_revision == 1
    assert export.minutes_source_transcript_revision == 1


def test_export_rejects_mismatched_speaker_attribution_timestamps() -> None:
    transcript = _transcript("The plan is ready.")
    mismatched = replace(
        transcript,
        attributions=(SpeakerAttribution(1, 999, "Voice 1"),),
    )

    with pytest.raises(InvalidTranscriptEvidenceError, match="matching transcript"):
        transcript_segments_v1(mismatched)


def test_citation_presence_does_not_add_a_claim_verified_status() -> None:
    transcript = _transcript("The team has not approved the budget.")
    minutes = _minutes(
        decisions=(
            MinutesDecision(
                text="The team approved the budget.",
                source_segment_ids=("segment-1",),
            ),
        )
    )
    export = build_meeting_export_v1("job-789", transcript, minutes)
    serialized = json.loads(export.model_dump_json())

    assert serialized["minutes"]["decisions"][0]["source_segment_ids"] == ["segment-1"]
    assert "verified" not in serialized["minutes"]["decisions"][0]
