"""Check Community-1 artifact integrity validation."""

import hashlib
from pathlib import Path

import pytest

from test.src import diarization_model


def test_model_integrity_reports_missing_and_modified_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"test fixture"
    monkeypatch.setattr(
        diarization_model,
        "MODEL_FILES",
        {"config.yaml": hashlib.sha256(content).hexdigest()},
    )
    with pytest.raises(FileNotFoundError, match="Missing Community-1"):
        diarization_model.verify_model(tmp_path)
    target = tmp_path / "config.yaml"
    target.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256"):
        diarization_model.verify_model(tmp_path)
    target.write_bytes(content)
    diarization_model.verify_model(tmp_path)
