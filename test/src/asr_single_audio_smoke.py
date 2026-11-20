"""Transcribe one fixture through the production adapter in an offline container."""

import json
import sys
from pathlib import Path

from app.asr_service import TranscriptionService
from app.faster_whisper_asr import FasterWhisperTranscriber


def main() -> None:
    transcriber = FasterWhisperTranscriber(Path("/model"), "cuda")
    try:
        transcript = TranscriptionService(transcriber).transcribe(Path(sys.argv[1]))
        print(json.dumps({"text": " ".join(s.text for s in transcript.segments)}))
    finally:
        transcriber.close()


if __name__ == "__main__":
    main()
