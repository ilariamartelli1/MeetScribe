"""Model-independent speaker turns on the original recording timeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.asr_contract import Device


@dataclass(frozen=True)
class SpeakerTurn:
    start_ms: int
    end_ms: int
    speaker_id: str


@dataclass(frozen=True)
class DiarizationProvenance:
    adapter: str
    adapter_version: str
    model_id: str
    model_revision: str
    device: Device


@dataclass(frozen=True)
class Diarization:
    duration_ms: int
    turns: tuple[SpeakerTurn, ...]
    exclusive_turns: tuple[SpeakerTurn, ...]
    provenance: DiarizationProvenance


class Diarizer(Protocol):
    def diarize(self, audio_path: Path) -> Diarization: ...

    def close(self) -> None: ...
