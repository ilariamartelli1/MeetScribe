"""Verify the configured model factories are spawn-safe and lazy."""

import pickle
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest

from app.adapters.model_factories import create_community1, create_faster_whisper
from app.core.transcription_worker import SpeakerAttributedJob
from app.infrastructure.config import ASR_MODEL_MANIFEST_ENV
from app.infrastructure.model_integrity import MANIFEST_DIR, load_manifest


def test_configured_factories_are_pickleable_and_load_lazily() -> None:
    model_manifest = load_manifest(MANIFEST_DIR / "faster_whisper.toml")
    job = SpeakerAttributedJob(
        partial(create_faster_whisper, Path("/asr"), "cpu", model_manifest),
        partial(create_community1, Path("/diarization"), "cpu"),
    )
    restored = pickle.loads(pickle.dumps(job))
    with (
        patch("app.adapters.faster_whisper_asr.FasterWhisperTranscriber") as asr,
        patch("app.adapters.community1_diarizer.Community1Diarizer") as diarizer,
    ):
        restored.transcriber_factory()
        restored.diarizer_factory()
        asr.assert_called_once_with(Path("/asr"), "cpu", model_manifest)
        diarizer.assert_called_once_with(Path("/diarization"), "cpu")


def test_faster_whisper_factory_loads_selected_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "custom.toml"
    manifest_path.write_text(
        'name = "custom"\nmodel_id = "custom/model"\nrevision = "custom-revision"\n'
        '[files]\n"model.bin" = "' + "a" * 64 + '"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(ASR_MODEL_MANIFEST_ENV, str(manifest_path))

    with patch("app.adapters.faster_whisper_asr.FasterWhisperTranscriber") as asr:
        create_faster_whisper(Path("/custom-model"), "cpu")

    asr.assert_called_once_with(
        Path("/custom-model"), "cpu", load_manifest(manifest_path)
    )
