"""Check long synthetic speech through the production adapter in an offline container."""

import json
from pathlib import Path

from app.asr_service import TranscriptionService
from app.faster_whisper_asr import FasterWhisperTranscriber


def main() -> None:
    transcriber = FasterWhisperTranscriber(Path("/model"), "cuda")
    try:
        transcript = TranscriptionService(transcriber).transcribe(
            Path("/audio/sample.mp3")
        )
        print(
            json.dumps(
                {
                    "duration_ms": transcript.duration_ms,
                    "text": " ".join(segment.text for segment in transcript.segments),
                    "segments": [
                        {
                            "start_ms": segment.start_ms,
                            "end_ms": segment.end_ms,
                            "text": segment.text,
                        }
                        for segment in transcript.segments
                    ],
                    "condition_on_previous_text": (
                        transcript.provenance.condition_on_previous_text
                    ),
                }
            )
        )
    finally:
        transcriber.close()


if __name__ == "__main__":
    main()
