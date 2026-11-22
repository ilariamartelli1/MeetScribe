from dataclasses import dataclass
from pathlib import Path
from app.core.asr_contract import Device, Language, Transcript
from app.core.asr_service import TranscriptionService
from app.core.diarization_service import DiarizationService
from app.core.speaker_alignment import AttributedTranscript, align_speakers


@dataclass(frozen=True)
class FasterWhisperJob:
    model_dir: Path
    device: Device

    def __call__(self, audio_path: Path, language: Language) -> Transcript:
        from app.adapters.faster_whisper_asr import FasterWhisperTranscriber

        transcriber = FasterWhisperTranscriber(self.model_dir, self.device)
        try:
            return TranscriptionService(transcriber).transcribe(audio_path, language)
        finally:
            transcriber.close()


@dataclass(frozen=True)
class SpeakerAttributedJob:
    model_dir: Path
    diarization_model_dir: Path
    device: Device

    def __call__(self, audio_path: Path, language: Language) -> AttributedTranscript:
        transcript = FasterWhisperJob(self.model_dir, self.device)(audio_path, language)
        from app.adapters.community1_diarizer import Community1Diarizer

        diarizer = Community1Diarizer(self.diarization_model_dir, self.device)
        try:
            diarization = DiarizationService(diarizer).diarize(audio_path)
        finally:
            diarizer.close()
        return align_speakers(transcript, diarization)
