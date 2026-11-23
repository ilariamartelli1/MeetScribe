import os
from dataclasses import dataclass, field
from pathlib import Path

from app.infrastructure.model_integrity import MANIFEST_DIR

ASR_MODEL_DIR_ENV = "MEETSCRIBE_ASR_MODEL_DIR"
ASR_MODEL_MANIFEST_ENV = "MEETSCRIBE_ASR_MODEL_MANIFEST"
DEFAULT_ASR_MODEL_DIR = Path("models/faster-whisper-large-v3")
DEFAULT_ASR_MODEL_MANIFEST = MANIFEST_DIR / "faster_whisper.toml"


def _asr_model_dir() -> Path:
    return Path(os.environ.get(ASR_MODEL_DIR_ENV, str(DEFAULT_ASR_MODEL_DIR)))


def _asr_model_manifest() -> Path:
    return Path(os.environ.get(ASR_MODEL_MANIFEST_ENV, str(DEFAULT_ASR_MODEL_MANIFEST)))


@dataclass(frozen=True)
class Settings:
    app_name: str = "MeetScribe"
    api_prefix: str = "/api/v1"
    asr_model_dir: Path = field(default_factory=_asr_model_dir)
    asr_model_manifest: Path = field(default_factory=_asr_model_manifest)
