from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.asr_contract import Language, ManagedTranscriber, Transcript
from app.core.asr_service import TranscriptionService
from app.core.diarization_contract import Diarizer
from app.core.diarization_service import DiarizationService
from app.core.speaker_alignment import AttributedTranscript, align_speakers


@dataclass(frozen=True)
class TranscriptionJob:
    transcriber_factory: Callable[[], ManagedTranscriber]

    def __call__(self, audio_path: Path, language: Language) -> Transcript:
        transcriber = self.transcriber_factory()
        try:
            return TranscriptionService(transcriber).transcribe(audio_path, language)
        finally:
            transcriber.close()


@dataclass(frozen=True)
class SpeakerAttributedJob:
    transcriber_factory: Callable[[], ManagedTranscriber]
    diarizer_factory: Callable[[], Diarizer]

    def __call__(self, audio_path: Path, language: Language) -> AttributedTranscript:
        transcript = TranscriptionJob(self.transcriber_factory)(audio_path, language)
        diarizer = self.diarizer_factory()
        try:
            diarization = DiarizationService(diarizer).diarize(audio_path)
        finally:
            diarizer.close()
        return align_speakers(transcript, diarization)
