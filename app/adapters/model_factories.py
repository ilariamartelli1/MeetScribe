"""Spawn-safe, lazy factories for the configured local speech models."""

from pathlib import Path

from app.core.asr_contract import Device, ManagedTranscriber
from app.core.diarization_contract import Diarizer


def create_faster_whisper(model_dir: Path, device: Device) -> ManagedTranscriber:
    from app.adapters.faster_whisper_asr import FasterWhisperTranscriber

    return FasterWhisperTranscriber(model_dir, device)


def create_community1(model_dir: Path, device: Device) -> Diarizer:
    from app.adapters.community1_diarizer import Community1Diarizer

    return Community1Diarizer(model_dir, device)
