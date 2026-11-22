"""Verify the configured model factories are spawn-safe and lazy."""

import pickle
from functools import partial
from pathlib import Path
from unittest.mock import patch

from app.adapters.model_factories import create_community1, create_faster_whisper
from app.core.transcription_worker import SpeakerAttributedJob


def test_configured_factories_are_pickleable_and_load_lazily() -> None:
    job = SpeakerAttributedJob(
        partial(create_faster_whisper, Path("/asr"), "cpu"),
        partial(create_community1, Path("/diarization"), "cpu"),
    )
    restored = pickle.loads(pickle.dumps(job))
    with (
        patch("app.adapters.faster_whisper_asr.FasterWhisperTranscriber") as asr,
        patch("app.adapters.community1_diarizer.Community1Diarizer") as diarizer,
    ):
        restored.transcriber_factory()
        restored.diarizer_factory()
        asr.assert_called_once_with(Path("/asr"), "cpu")
        diarizer.assert_called_once_with(Path("/diarization"), "cpu")
