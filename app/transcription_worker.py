from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.asr_contract import Language, Transcript
from app.asr_service import TranscriptionService


@dataclass(frozen=True)
class FasterWhisperJob:
    model_dir: Path
    device: Literal["cpu", "cuda"]

    def __call__(self, audio_path: Path, language: Language) -> Transcript:
        from app.faster_whisper_asr import FasterWhisperTranscriber

        transcriber = FasterWhisperTranscriber(self.model_dir, self.device)
        try:
            return TranscriptionService(transcriber).transcribe(audio_path, language)
        finally:
            transcriber.close()
