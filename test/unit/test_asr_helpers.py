"""Verify integration test path selection and transcript normalization."""

from pathlib import Path

import pytest

from test.integration.test_asr_gpu_feasibility import DEFAULT_PATHS, integration_paths
from test.integration.test_asr_accuracy import normalize_text


def test_integration_paths_use_versioned_fixtures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key in DEFAULT_PATHS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MEETSCRIBE_MODEL_DIR", str(tmp_path))

    paths = integration_paths()

    assert paths["MEETSCRIBE_MODEL_DIR"] == tmp_path.resolve()
    assert paths["MEETSCRIBE_TEST_WAV"] == DEFAULT_PATHS["MEETSCRIBE_TEST_WAV"]
    assert paths["MEETSCRIBE_TEST_MP3"] == DEFAULT_PATHS["MEETSCRIBE_TEST_MP3"]


def test_integration_paths_allow_audio_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key in DEFAULT_PATHS:
        monkeypatch.setenv(key, str(tmp_path))

    assert all(path == tmp_path.resolve() for path in integration_paths().values())


def test_normalize_text_ignores_case_punctuation_and_whitespace() -> None:
    assert normalize_text("  Hello, WORLD!\n") == normalize_text("hello world")
    assert normalize_text("hello_world") == normalize_text("helloworld")
    assert normalize_text("do not") != normalize_text("do")
