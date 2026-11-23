"""Check model manifests and generic artifact integrity validation."""

import hashlib
from pathlib import Path

import pytest

from app.infrastructure.model_integrity import (
    MANIFEST_DIR,
    ModelManifest,
    load_manifest,
    verify_model,
)


@pytest.mark.parametrize("model_name", ["First model", "Another model"])
def test_model_integrity_reports_missing_and_modified_files(
    tmp_path: Path, model_name: str
) -> None:
    content = b"test fixture"
    manifest = ModelManifest(
        model_name,
        "test/model",
        "revision",
        {"config.yaml": hashlib.sha256(content).hexdigest()},
        {},
    )
    with pytest.raises(FileNotFoundError, match=f"Missing {model_name}"):
        verify_model(tmp_path, manifest)
    target = tmp_path / "config.yaml"
    target.write_bytes(b"modified")
    with pytest.raises(ValueError, match=f"{model_name}.*SHA-256"):
        verify_model(tmp_path, manifest)
    target.write_bytes(content)
    verify_model(tmp_path, manifest)


def test_optional_files_are_checked_when_present(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"correct").hexdigest()
    manifest = ModelManifest(
        "Test", "test/model", "revision", {"required": digest}, {"optional": digest}
    )
    (tmp_path / "required").write_bytes(b"correct")
    verify_model(tmp_path, manifest)
    (tmp_path / "optional").write_bytes(b"modified")
    with pytest.raises(ValueError, match="optional"):
        verify_model(tmp_path, manifest)
    (tmp_path / "optional").write_bytes(b"correct")
    verify_model(tmp_path, manifest)


@pytest.mark.parametrize(
    "filename",
    ["community1.toml", "faster_whisper.toml", "faster_whisper_turbo.toml"],
)
def test_versioned_manifests_are_valid(filename: str) -> None:
    manifest = load_manifest(MANIFEST_DIR / filename)
    assert manifest.name
    assert manifest.model_id
    assert manifest.revision
    assert manifest.files


@pytest.mark.parametrize(
    "content",
    [
        'name = ""\nmodel_id = "id"\nrevision = "rev"\n[files]\n',
        'name = "x"\nmodel_id = "id"\nrevision = "rev"\n[files]\n',
        'name = "x"\nmodel_id = "id"\nrevision = "rev"\n[files]\n"../file" = "'
        + "a" * 64
        + '"\n',
        'name = "x"\nmodel_id = "id"\nrevision = "rev"\n[files]\n"/absolute" = "'
        + "a" * 64
        + '"\n',
        'name = "x"\nmodel_id = "id"\nrevision = "rev"\n[files]\n"file" = "not-a-sha"\n',
        'name = "x"\nmodel_id = "id"\nrevision = "rev"\n[files]\n"file" = "'
        + "a" * 64
        + '"\n[optional_files]\n"file" = "'
        + "a" * 64
        + '"\n',
    ],
)
def test_invalid_manifests_are_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "manifest.toml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_manifest(path)
