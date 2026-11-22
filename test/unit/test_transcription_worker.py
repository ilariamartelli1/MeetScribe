"""Verify the jobs: faster-whisper jobs unload their model after success or failure."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.asr_contract import ModelProvenance, Transcript, TranscriptSegment
from app.core.diarization_contract import (
    Diarization,
    DiarizationProvenance,
    SpeakerTurn,
)
from app.core.transcription_worker import FasterWhisperJob, SpeakerAttributedJob


@pytest.mark.parametrize("fails", [False, True])
def test_model_is_unloaded_after_transcription(fails: bool) -> None:
    transcript = Transcript(
        "en",
        1000,
        (TranscriptSegment(0, 100, "No"),),
        ModelProvenance(
            "fake", "1", "model", "revision", "digest", "cpu", "int8", True, 5
        ),
    )
    with patch(
        "app.adapters.faster_whisper_asr.FasterWhisperTranscriber"
    ) as constructor:
        transcriber = constructor.return_value
        transcriber.transcribe.return_value = transcript
        if fails:
            transcriber.transcribe.side_effect = RuntimeError("model failed")
        job = FasterWhisperJob(Path("/model"), "cpu")
        if fails:
            with pytest.raises(RuntimeError, match="model failed"):
                job(Path("/sample.wav"), "en")
        else:
            assert job(Path("/sample.wav"), "en") == transcript
        constructor.assert_called_once_with(Path("/model"), "cpu")
        transcriber.close.assert_called_once_with()


@pytest.mark.parametrize("fails", [False, True])
def test_speaker_job_unloads_asr_before_diarization_and_releases_diarizer(
    fails: bool,
) -> None:
    transcript = Transcript(
        "en",
        1000,
        (TranscriptSegment(0, 100, "No"),),
        ModelProvenance(
            "fake", "1", "model", "revision", "digest", "cpu", "int8", True, 5
        ),
    )
    diarization = Diarization(
        1000,
        (SpeakerTurn(0, 100, "speaker"),),
        (),
        DiarizationProvenance("fake", "1", "model", "revision", "cpu"),
    )
    with (
        patch(
            "app.adapters.faster_whisper_asr.FasterWhisperTranscriber"
        ) as asr_constructor,
        patch(
            "app.adapters.community1_diarizer.Community1Diarizer"
        ) as diarizer_constructor,
    ):
        transcriber = asr_constructor.return_value
        transcriber.transcribe.return_value = transcript
        diarizer = diarizer_constructor.return_value
        diarizer.diarize.return_value = diarization

        def create_diarizer(*args: object, **kwargs: object) -> object:
            transcriber.close.assert_called_once_with()
            return diarizer

        diarizer_constructor.side_effect = create_diarizer
        if fails:
            diarizer.diarize.side_effect = RuntimeError("diarization failed")
        job = SpeakerAttributedJob(Path("/asr"), Path("/diarization"), "cpu")
        if fails:
            with pytest.raises(RuntimeError, match="diarization failed"):
                job(Path("/sample.wav"), "en")
        else:
            result = job(Path("/sample.wav"), "en")
            assert result.segments == transcript.segments
            assert result.attributions[0].speaker == "Voice 1"
        diarizer_constructor.assert_called_once_with(Path("/diarization"), "cpu")
        diarizer.diarize.assert_called_once_with(Path("/sample.wav"))
        diarizer.close.assert_called_once_with()


def test_speaker_job_does_not_load_diarizer_when_asr_fails() -> None:
    with (
        patch(
            "app.adapters.faster_whisper_asr.FasterWhisperTranscriber"
        ) as asr_constructor,
        patch(
            "app.adapters.community1_diarizer.Community1Diarizer"
        ) as diarizer_constructor,
    ):
        transcriber = asr_constructor.return_value
        transcriber.transcribe.side_effect = RuntimeError("asr failed")
        with pytest.raises(RuntimeError, match="asr failed"):
            SpeakerAttributedJob(Path("/asr"), Path("/diarization"), "cpu")(
                Path("/sample.wav"), "en"
            )
        transcriber.close.assert_called_once_with()
        diarizer_constructor.assert_not_called()
