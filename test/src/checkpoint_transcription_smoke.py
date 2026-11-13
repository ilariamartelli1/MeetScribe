"""Transcribe the four synthetic checkpoint recordings inside the app container."""

import json
from dataclasses import asdict
from pathlib import Path

from app.asr_contract import Language
from app.asr_service import TranscriptionService
from app.faster_whisper_asr import FasterWhisperTranscriber


def main() -> None:
    transcriber = FasterWhisperTranscriber(Path("/model"), "cuda")
    try:
        service = TranscriptionService(transcriber)
        reports = []
        languages: tuple[tuple[Language, str], ...] = (
            ("en", "english"),
            ("it", "italian"),
        )
        for language, folder in languages:
            for extension in ("wav", "mp3"):
                name = f"meeting-{language}.{extension}"
                transcript = service.transcribe(
                    Path("/audio") / folder / name, language
                )
                reports.append({"file": name, "transcript": asdict(transcript)})
        print(json.dumps(reports, ensure_ascii=False))
    finally:
        transcriber.close()


if __name__ == "__main__":
    main()
