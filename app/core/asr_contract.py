from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

CPU_DEVICE: Final = "cpu"
CUDA_DEVICE: Final = "cuda"
Device = Literal["cpu", "cuda"]
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
    model_sha256: str | None = None
    device: str | None = None


@dataclass(frozen=True)
class Transcript:
    language: Language
    duration_ms: int
    segments: tuple[TranscriptSegment, ...]
    provenance: ModelProvenance


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, language: Language) -> Transcript: ...


class ManagedTranscriber(Transcriber, Protocol):
    def close(self) -> None: ...
