"""Verify speaker coverage in the two-voice integration check."""

import pytest

from test.integration.test_speaker_alignment import (
    assert_complete_transcript,
    interval_speaker,
)


def test_complete_transcript_detects_missing_words() -> None:
    assert_complete_transcript(
        "The project is on schedule", "The project is on schedule."
    )
    with pytest.raises(AssertionError, match="Expected:"):
        assert_complete_transcript(
            "The project is scheduled", "The project is on schedule."
        )


def test_speaker_coverage_combines_segments_without_counting_silence() -> None:
    assert (
        interval_speaker(
            (0, 8000),
            [
                {"start_ms": 2000, "end_ms": 4000, "speaker": "Voice 1"},
                {"start_ms": 4000, "end_ms": 6000, "speaker": "Voice 1"},
                {"start_ms": 6000, "end_ms": 7000, "speaker": None},
            ],
        )
        == "Voice 1"
    )


def test_speaker_coverage_rejects_insufficient_labeled_time() -> None:
    with pytest.raises(AssertionError):
        interval_speaker(
            (0, 8000),
            [
                {"start_ms": 1000, "end_ms": 4000, "speaker": "Voice 1"},
                {"start_ms": 4000, "end_ms": 6000, "speaker": None},
            ],
        )


def test_speaker_coverage_does_not_double_count_overlapping_segments() -> None:
    with pytest.raises(AssertionError):
        interval_speaker(
            (0, 8000),
            [
                {"start_ms": 1000, "end_ms": 3000, "speaker": "Voice 1"},
                {"start_ms": 1000, "end_ms": 3000, "speaker": "Voice 1"},
                {"start_ms": 3000, "end_ms": 5000, "speaker": None},
            ],
        )


def test_speaker_coverage_rejects_too_little_transcribed_time() -> None:
    with pytest.raises(AssertionError):
        interval_speaker(
            (0, 8000),
            [{"start_ms": 1000, "end_ms": 1999, "speaker": "Voice 1"}],
        )


def test_speaker_coverage_rejects_ambiguous_overlapping_speakers() -> None:
    with pytest.raises(AssertionError):
        interval_speaker(
            (0, 8000),
            [
                {"start_ms": 1000, "end_ms": 4000, "speaker": "Voice 1"},
                {"start_ms": 1000, "end_ms": 4000, "speaker": "Voice 2"},
            ],
        )
