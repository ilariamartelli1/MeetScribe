"""Run the production ASR adapter on synthetic quiet negations and silence."""

import json
import tempfile
import wave
from dataclasses import asdict
from pathlib import Path

from app.asr_contract import Language
from app.asr_service import NoSpeechError, TranscriptionService
from app.faster_whisper_asr import FasterWhisperTranscriber

transcriber = FasterWhisperTranscriber(Path("/model"), "cuda")
service = TranscriptionService(transcriber)
reports: dict[str, object] = {}
samples: tuple[tuple[Language, str], ...] = (
    ("en", "/audio/quiet-negations-en.wav"),
    ("it", "/audio/quiet-negations-it.wav"),
)
for language, filename in samples:
    reports[language] = asdict(service.transcribe(Path(filename), language))

with tempfile.NamedTemporaryFile(suffix=".wav") as silence:
    with wave.open(silence.name, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(b"\0\0" * 16000)
    try:
        service.transcribe(Path(silence.name))
    except NoSpeechError:
        reports["silence_rejected"] = True
    else:
        raise RuntimeError("Silent recording was incorrectly transcribed.")

transcriber.close()
print(json.dumps(reports))
