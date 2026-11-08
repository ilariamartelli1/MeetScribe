"""Run the production transcription coordinator with a real offline GPU model."""

import json
import time
from pathlib import Path

from app.transcription_jobs import TranscriptionCoordinator
from app.transcription_worker import FasterWhisperJob


def main() -> None:
    coordinator = TranscriptionCoordinator(FasterWhisperJob(Path("/model"), "cuda"))
    try:
        job_id = coordinator.submit(Path("/audio/quiet-negations-en.wav"))
        deadline = time.monotonic() + 120
        while True:
            snapshot = coordinator.get(job_id)
            if snapshot.state == "failed":
                raise RuntimeError(f"Transcription failed: {snapshot.error}")
            if snapshot.state == "succeeded":
                assert snapshot.result is not None
                print(
                    json.dumps(
                        {
                            "state": snapshot.state,
                            "language": snapshot.result.language,
                            "text": " ".join(
                                segment.text for segment in snapshot.result.segments
                            ),
                        }
                    )
                )
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Transcription job did not finish in time.")
            time.sleep(0.05)
    finally:
        coordinator.shutdown()


if __name__ == "__main__":
    main()
