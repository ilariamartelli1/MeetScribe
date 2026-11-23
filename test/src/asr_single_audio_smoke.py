"""Transcribe one fixture through the production adapter in an offline container."""

import json
import sys
from pathlib import Path

from app.adapters.model_factories import create_faster_whisper
from app.core.asr_service import TranscriptionService


def main() -> None:
    transcriber = create_faster_whisper(Path("/model"), "cuda")
    try:
        transcript = TranscriptionService(transcriber).transcribe(Path(sys.argv[1]))
        print(json.dumps({"text": " ".join(s.text for s in transcript.segments)}))
    finally:
        transcriber.close()


if __name__ == "__main__":
    main()
