"""Run the speaker-attributed job with offline GPU models in an owned child."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from app.speaker_alignment import AttributedTranscript
from app.transcription_jobs import TranscriptionCoordinator
from app.transcription_worker import SpeakerAttributedJob

JOB_TIMEOUT_SECONDS = 240


def main() -> None:
    job = SpeakerAttributedJob(Path("/asr"), Path("/diarization"), "cuda")
    coordinator = TranscriptionCoordinator(job)
    try:
        job_id = coordinator.submit(Path("/audio/two-voices.wav"))
        deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
        while True:
            snapshot = coordinator.get(job_id)
            if snapshot.state == "failed":
                raise RuntimeError(f"Speaker-attributed job failed: {snapshot.error}")
            if snapshot.state == "succeeded":
                if not isinstance(snapshot.result, AttributedTranscript):
                    raise RuntimeError("The job did not return speaker attribution.")
                print(json.dumps(asdict(snapshot.result)))
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Speaker-attributed job did not finish in time.")
            time.sleep(0.05)
    finally:
        coordinator.shutdown()


if __name__ == "__main__":
    main()
