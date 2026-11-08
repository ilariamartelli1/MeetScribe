from pathlib import Path

from app.asr_contract import Language, Transcriber, Transcript


class NoSpeechError(ValueError):
    pass


class InvalidTranscriptError(ValueError):
    pass


class TranscriptionService:
    def __init__(self, transcriber: Transcriber) -> None:
        self._transcriber = transcriber

    def transcribe(self, audio_path: Path, language: Language = "en") -> Transcript:
        if language not in ("en", "it"):
            raise ValueError("Recording language must be 'en' or 'it'.")
        transcript = self._transcriber.transcribe(audio_path, language)
        if transcript.language != language:
            raise InvalidTranscriptError("Transcript language does not match request.")
        if not transcript.segments or not any(
            segment.text.strip() for segment in transcript.segments
        ):
            raise NoSpeechError("No speech was transcribed from the recording.")
        if transcript.duration_ms <= 0:
            raise InvalidTranscriptError("Recording duration must be positive.")
        previous_start = -1
        previous_end = -1
        for segment in transcript.segments:
            if (
                not segment.text.strip()
                or not 0 <= segment.start_ms < segment.end_ms <= transcript.duration_ms
                or segment.start_ms < previous_start
                or segment.end_ms < previous_end
            ):
                raise InvalidTranscriptError(
                    "Transcript segments must contain text and ordered timestamps "
                    "within the original recording."
                )
            previous_start = segment.start_ms
            previous_end = segment.end_ms
        return transcript
