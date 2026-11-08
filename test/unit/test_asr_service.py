"""Verify language, speech and original-audio timestamp validation."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.asr_contract import ModelProvenance, Transcript, TranscriptSegment
from app.asr_service import InvalidTranscriptError, NoSpeechError, TranscriptionService

PROVENANCE = ModelProvenance(
    "fake", "1.0", "model", "revision", "digest", "cpu", "int8", True, 5
)


def test_negations_at_original_recording_boundaries_are_preserved() -> None:
    transcript = Transcript(
        "en",
        5000,
        (TranscriptSegment(0, 500, "No"), TranscriptSegment(4500, 5000, "not")),
        PROVENANCE,
    )
    fake = Mock()
    fake.transcribe.return_value = transcript
    service = TranscriptionService(fake)
    assert service.transcribe(Path("sample.wav")) == transcript
    fake.transcribe.assert_called_once_with(Path("sample.wav"), "en")


def test_italian_and_adapter_failures_are_propagated() -> None:
    fake = Mock()
    fake.transcribe.return_value = Transcript(
        "it", 1000, (TranscriptSegment(10, 500, "non"),), PROVENANCE
    )
    service = TranscriptionService(fake)
    assert service.transcribe(Path("sample.mp3"), "it").segments[0].text == "non"
    fake.transcribe.side_effect = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        service.transcribe(Path("sample.mp3"), "it")


@pytest.mark.parametrize("segments", [(), (TranscriptSegment(0, 1, " "),)])
def test_silence_is_reported_as_no_speech(
    segments: tuple[TranscriptSegment, ...],
) -> None:
    fake = Mock()
    fake.transcribe.return_value = Transcript("en", 1000, segments, PROVENANCE)
    with pytest.raises(NoSpeechError, match="No speech"):
        TranscriptionService(fake).transcribe(Path("silence.wav"))


@pytest.mark.parametrize(
    "segments",
    [
        (TranscriptSegment(-1, 100, "no"),),
        (TranscriptSegment(0, 1001, "no"),),
        (TranscriptSegment(10, 10, "no"),),
        (TranscriptSegment(500, 700, "no"), TranscriptSegment(100, 200, "not")),
        (TranscriptSegment(0, 600, "no"), TranscriptSegment(100, 500, "not")),
        (TranscriptSegment(0, 100, "no"), TranscriptSegment(100, 200, " ")),
    ],
)
def test_invalid_timestamps_and_blank_segments_are_rejected(
    segments: tuple[TranscriptSegment, ...],
) -> None:
    fake = Mock()
    fake.transcribe.return_value = Transcript("en", 1000, segments, PROVENANCE)
    with pytest.raises(InvalidTranscriptError):
        TranscriptionService(fake).transcribe(Path("sample.wav"))


def test_wrong_language_and_empty_duration_are_rejected() -> None:
    fake = Mock()
    transcript = Transcript("en", 1000, (TranscriptSegment(0, 100, "no"),), PROVENANCE)
    service = TranscriptionService(fake)
    fake.transcribe.return_value = replace(transcript, language="it")
    with pytest.raises(InvalidTranscriptError, match="language"):
        service.transcribe(Path("sample.wav"))
    fake.transcribe.return_value = replace(transcript, duration_ms=0)
    with pytest.raises(InvalidTranscriptError, match="duration"):
        service.transcribe(Path("sample.wav"))
