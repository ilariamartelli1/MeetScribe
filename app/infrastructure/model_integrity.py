"""Load pinned model manifests and verify their local artifacts."""

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "model_manifests"


@dataclass(frozen=True)
class ModelManifest:
    name: str
    model_id: str
    revision: str
    files: dict[str, str]
    optional_files: dict[str, str]


def _file_hashes(value: object, section: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"Model manifest {section} must be a table.")
    hashes: dict[str, str] = {}
    for name, digest in value.items():
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or "\x00" in name
            or PurePosixPath(name).is_absolute()
            or any(part in (".", "..") for part in name.split("/"))
            or any(not part for part in name.split("/"))
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"Invalid file path or SHA-256 in {section}: {name!r}.")
        hashes[name] = digest
    return hashes


def load_manifest(path: Path) -> ModelManifest:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    name = data.get("name")
    model_id = data.get("model_id")
    revision = data.get("revision")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (name, model_id, revision)
    ):
        raise ValueError(f"Model manifest {path} needs a name, model ID and revision.")
    assert (
        isinstance(name, str)
        and isinstance(model_id, str)
        and isinstance(revision, str)
    )
    files = _file_hashes(data.get("files"), "files")
    optional_files = _file_hashes(data.get("optional_files", {}), "optional_files")
    if not files or files.keys() & optional_files.keys():
        raise ValueError(
            f"Model manifest {path} has missing or duplicate file entries."
        )
    return ModelManifest(name, model_id, revision, files, optional_files)


def verify_model(path: Path, manifest: ModelManifest) -> None:
    for name, expected in manifest.files.items():
        _verify_file(path / name, expected, manifest.name)
    for name, expected in manifest.optional_files.items():
        file = path / name
        if file.exists() or file.is_symlink():
            _verify_file(file, expected, manifest.name)


def _verify_file(path: Path, expected: str, model_name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {model_name} model file: {path}")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(
                f"{model_name} model file failed SHA-256 verification: {path}"
            )
