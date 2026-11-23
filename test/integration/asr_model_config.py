"""Select and mount the configured local ASR model for offline tests."""

from pathlib import Path

from app.infrastructure.config import (
    ASR_MODEL_MANIFEST_ENV,
    Settings,
)

CONTAINER_MANIFEST_PATH = "/asr-manifest.toml"


def configured_model_paths() -> tuple[Path, Path]:
    settings = Settings()
    return (
        settings.asr_model_dir.resolve(strict=True),
        settings.asr_model_manifest.resolve(strict=True),
    )


def asr_model_mount_args(container_model_dir: str) -> list[str]:
    model_dir, manifest_path = configured_model_paths()
    return [
        "--mount",
        f"type=bind,src={model_dir},dst={container_model_dir},readonly",
        "--mount",
        f"type=bind,src={manifest_path},dst={CONTAINER_MANIFEST_PATH},readonly",
        "--env",
        f"{ASR_MODEL_MANIFEST_ENV}={CONTAINER_MANIFEST_PATH}",
    ]
