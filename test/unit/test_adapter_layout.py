"""Check the model adapter package and its shared manifest paths."""

from importlib import import_module
from pathlib import Path

import pytest

from app.adapters import community1_diarizer, faster_whisper_asr, ollama_client
from app.infrastructure.model_integrity import MANIFEST_DIR, load_manifest


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("faster_whisper_asr", "FasterWhisperTranscriber"),
        ("community1_diarizer", "Community1Diarizer"),
        ("ollama_client", "OllamaClient"),
    ],
)
def test_provider_is_importable_from_adapters(module: str, name: str) -> None:
    provider = getattr(import_module(f"app.adapters.{module}"), name)
    assert provider.__module__ == f"app.adapters.{module}"


def test_adapters_use_the_shared_model_manifests() -> None:
    assert MANIFEST_DIR == Path(__file__).resolve().parents[2] / "app/model_manifests"
    assert faster_whisper_asr.MODEL_MANIFEST == load_manifest(
        MANIFEST_DIR / "faster_whisper.toml"
    )
    assert community1_diarizer.MODEL_MANIFEST == load_manifest(
        MANIFEST_DIR / "community1.toml"
    )
    assert ollama_client.OLLAMA_MANIFEST_PATH == MANIFEST_DIR / "summary_model.toml"
    assert ollama_client.OLLAMA_MANIFEST_PATH.is_file()
