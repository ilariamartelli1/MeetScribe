"""Spawn-safe, lazy factories for the configured local speech models."""

from pathlib import Path

from app.core.asr_contract import Device, ManagedTranscriber
from app.core.diarization_contract import Diarizer
from app.infrastructure.config import Settings
from app.infrastructure.model_integrity import ModelManifest, load_manifest


def create_faster_whisper(
    model_dir: Path,
    device: Device,
    model_manifest: ModelManifest | None = None,
) -> ManagedTranscriber:
    from app.adapters.faster_whisper_asr import FasterWhisperTranscriber

    if model_manifest is None:
        model_manifest = load_manifest(Settings().asr_model_manifest)
    return FasterWhisperTranscriber(model_dir, device, model_manifest)


def create_community1(model_dir: Path, device: Device) -> Diarizer:
    from app.adapters.community1_diarizer import Community1Diarizer

    return Community1Diarizer(model_dir, device)
