"""Verify integration test path selection and transcript normalization."""

from pathlib import Path

import pytest

from app.infrastructure.config import ASR_MODEL_DIR_ENV, ASR_MODEL_MANIFEST_ENV
from test.integration.asr_model_config import (
    CONTAINER_MANIFEST_PATH,
    asr_model_mount_args,
)
from test.integration.test_asr_gpu_feasibility import DEFAULT_PATHS, integration_paths
from test.integration.test_asr_accuracy import normalize_text


def test_integration_paths_use_versioned_fixtures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key in DEFAULT_PATHS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(ASR_MODEL_DIR_ENV, str(tmp_path))

    paths = integration_paths()

    assert paths[ASR_MODEL_DIR_ENV] == tmp_path.resolve()
    assert paths["MEETSCRIBE_TEST_WAV"] == DEFAULT_PATHS["MEETSCRIBE_TEST_WAV"]
    assert paths["MEETSCRIBE_TEST_MP3"] == DEFAULT_PATHS["MEETSCRIBE_TEST_MP3"]


def test_integration_paths_allow_audio_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key in DEFAULT_PATHS:
        monkeypatch.setenv(key, str(tmp_path))

    assert all(path == tmp_path.resolve() for path in integration_paths().values())


def test_asr_mounts_use_the_selected_model_and_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "custom-model"
    model_dir.mkdir()
    manifest_path = tmp_path / "custom.toml"
    manifest_path.write_text("model manifest", encoding="utf-8")
    monkeypatch.setenv(ASR_MODEL_DIR_ENV, str(model_dir))
    monkeypatch.setenv(ASR_MODEL_MANIFEST_ENV, str(manifest_path))

    mounts = asr_model_mount_args("/model")

    assert f"type=bind,src={model_dir},dst=/model,readonly" in mounts
    assert (
        f"type=bind,src={manifest_path},dst={CONTAINER_MANIFEST_PATH},readonly"
        in mounts
    )
    assert f"{ASR_MODEL_MANIFEST_ENV}={CONTAINER_MANIFEST_PATH}" in mounts


def test_normalize_text_ignores_case_punctuation_and_whitespace() -> None:
    assert normalize_text("  Hello, WORLD!\n") == normalize_text("hello world")
    assert normalize_text("hello_world") == normalize_text("helloworld")
    assert normalize_text("do not") != normalize_text("do")
