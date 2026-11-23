"""Verify default application settings and their immutability."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.infrastructure.config import (
    ASR_MODEL_DIR_ENV,
    ASR_MODEL_MANIFEST_ENV,
    DEFAULT_ASR_MODEL_DIR,
    DEFAULT_ASR_MODEL_MANIFEST,
    Settings,
)


def test_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ASR_MODEL_DIR_ENV, raising=False)
    monkeypatch.delenv(ASR_MODEL_MANIFEST_ENV, raising=False)
    settings = Settings()

    assert settings.app_name == "MeetScribe"
    assert settings.api_prefix == "/api/v1"
    assert settings.asr_model_dir == DEFAULT_ASR_MODEL_DIR
    assert settings.asr_model_manifest == DEFAULT_ASR_MODEL_MANIFEST

    with pytest.raises(FrozenInstanceError):
        settings.api_prefix = "/other"  # type: ignore[misc]


def test_asr_model_settings_can_select_another_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "model"
    manifest_path = tmp_path / "manifest.toml"
    monkeypatch.setenv(ASR_MODEL_DIR_ENV, str(model_dir))
    monkeypatch.setenv(ASR_MODEL_MANIFEST_ENV, str(manifest_path))

    settings = Settings()

    assert settings.asr_model_dir == model_dir
    assert settings.asr_model_manifest == manifest_path
