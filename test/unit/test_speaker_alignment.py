"""Verify conservative attribution on the original recording timeline."""

from dataclasses import replace

import pytest

from app.asr_contract import ModelProvenance, Transcript, TranscriptSegment
from app.diarization_contract import Diarization, DiarizationProvenance, SpeakerTurn
from app.speaker_alignment import SpeakerAttribution, align_speakers

ASR = ModelProvenance("fake", "1", "asr", "revision", "digest", "cpu", "int8", True, 5)
DIARIZATION = DiarizationProvenance("fake", "1", "diarizer", "revision", "cpu")


def test_aligns_silence_gaps_and_keeps_stable_first_appearance_labels() -> None:
    transcript = Transcript(
        "en",
        15000,
        (
            TranscriptSegment(100, 800, "first"),
            TranscriptSegment(5000, 5600, "second"),
            TranscriptSegment(10000, 10800, "first again"),
            TranscriptSegment(13000, 13500, "without turn"),
        ),
        ASR,
    )
    diarization = Diarization(
        15000,
        (
            SpeakerTurn(100, 500, "B"),
            SpeakerTurn(500, 800, "B"),
            SpeakerTurn(5000, 5600, "A"),
            SpeakerTurn(10000, 10800, "B"),
        ),
        (),
        DIARIZATION,
    )
    result = align_speakers(transcript, diarization)
    assert result.segments == transcript.segments
    assert result.diarization == diarization
    assert result.provenance == ASR
    assert result.attributions == (
        SpeakerAttribution(100, 800, "Voice 1"),
        SpeakerAttribution(5000, 5600, "Voice 2"),
        SpeakerAttribution(10000, 10800, "Voice 1"),
        SpeakerAttribution(13000, 13500, None),
    )


def test_mixed_speaker_turns_and_overlap_remain_unknown() -> None:
    transcript = Transcript(
        "en",
        5000,
        (
            TranscriptSegment(0, 1000, "adjacent"),
            TranscriptSegment(1000, 2000, "change"),
            TranscriptSegment(2500, 3500, "overlap"),
            TranscriptSegment(4000, 4500, "alone"),
        ),
        ASR,
    )
    diarization = Diarization(
        5000,
        (
            SpeakerTurn(0, 1500, "A"),
            SpeakerTurn(1500, 2000, "B"),
            SpeakerTurn(2500, 3500, "A"),
            SpeakerTurn(3000, 3200, "B"),
            SpeakerTurn(4000, 4500, "B"),
        ),
        (SpeakerTurn(0, 1500, "A"), SpeakerTurn(1500, 2000, "B")),
        DIARIZATION,
    )
    assert [
        a.speaker for a in align_speakers(transcript, diarization).attributions
    ] == ["Voice 1", None, None, "Voice 2"]


def test_time_boundaries_do_not_overlap_and_ties_are_stable() -> None:
    transcript = Transcript("en", 1000, (TranscriptSegment(100, 200, "word"),), ASR)
    diarization = Diarization(
        1000,
        (
            SpeakerTurn(0, 100, "B"),
            SpeakerTurn(0, 100, "A"),
            SpeakerTurn(100, 200, "B"),
        ),
        (),
        DIARIZATION,
    )
    assert align_speakers(transcript, diarization).attributions[0].speaker == "Voice 2"


def test_duration_mismatch_is_rejected_with_small_decoder_tolerance() -> None:
    transcript = Transcript("en", 1000, (TranscriptSegment(0, 100, "text"),), ASR)
    diarization = Diarization(1020, (), (), DIARIZATION)
    assert align_speakers(transcript, diarization).attributions[0].speaker is None
    with pytest.raises(ValueError, match="durations differ"):
        align_speakers(transcript, replace(diarization, duration_ms=1021))
