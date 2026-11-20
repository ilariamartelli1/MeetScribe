"""Verify the final-speech timestamp check handles segment boundaries."""

import pytest

from test.integration.test_asr_long_recording import assert_final_speech_timing


def test_accepts_phrase_split_across_segments_in_expected_interval() -> None:
    assert_final_speech_timing(
        [
            {"start_ms": 288580, "end_ms": 294580, "text": "The team will meet"},
            {"start_ms": 294580, "end_ms": 296800, "text": "again after the results"},
        ]
    )


def test_rejects_phrase_outside_expected_interval() -> None:
    with pytest.raises(AssertionError):
        assert_final_speech_timing(
            [{"start_ms": 280000, "end_ms": 290000, "text": "The team will meet again"}]
        )


def test_rejects_phrase_partly_outside_expected_interval() -> None:
    with pytest.raises(AssertionError):
        assert_final_speech_timing(
            [
                {"start_ms": 290000, "end_ms": 292155, "text": "The team will meet"},
                {"start_ms": 292155, "end_ms": 295000, "text": "again"},
            ]
        )
