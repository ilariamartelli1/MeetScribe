from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

Language = Literal["en", "it"]


@dataclass(frozen=True)
class TranscriptSegment:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class ModelProvenance:
    adapter: str
    adapter_version: str
    model_id: str
    model_revision: str
    model_sha256: str
    device: str
    compute_type: str
    vad_enabled: bool
    beam_size: int


@dataclass(frozen=True)
class Transcript:
    language: Language
    duration_ms: int
    segments: tuple[TranscriptSegment, ...]
    provenance: ModelProvenance


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, language: Language) -> Transcript: ...
