"""Verify the jobs: faster-whisper jobs unload their model after success or failure."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.asr_contract import ModelProvenance, Transcript, TranscriptSegment
from app.transcription_worker import FasterWhisperJob


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
    with patch("app.faster_whisper_asr.FasterWhisperTranscriber") as constructor:
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
