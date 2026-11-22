"""Run English and Italian speaker-attributed jobs with offline GPU models."""

import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

from app.adapters.model_factories import create_community1, create_faster_whisper
from app.core.asr_contract import Language
from app.core.speaker_alignment import AttributedTranscript
from app.core.transcription_jobs import TranscriptionCoordinator
from app.core.transcription_worker import SpeakerAttributedJob

JOB_TIMEOUT_SECONDS = 240


def run_job(
    coordinator: TranscriptionCoordinator, audio_path: Path, language: Language
) -> dict[str, object]:
    job_id = coordinator.submit(audio_path, language)
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    while True:
        snapshot = coordinator.get(job_id)
        if snapshot.state == "failed":
            raise RuntimeError(f"Speaker-attributed job failed: {snapshot.error}")
        if snapshot.state == "succeeded":
            if not isinstance(snapshot.result, AttributedTranscript):
                raise RuntimeError("The job did not return speaker attribution.")
            return asdict(snapshot.result)
        if time.monotonic() >= deadline:
            raise TimeoutError("Speaker-attributed job did not finish in time.")
        time.sleep(0.05)


def main() -> None:
    job = SpeakerAttributedJob(
        partial(create_faster_whisper, Path("/asr"), "cuda"),
        partial(create_community1, Path("/diarization"), "cuda"),
    )
    coordinator = TranscriptionCoordinator(job)
    try:
        jobs: tuple[tuple[Language, str], ...] = (
            ("en", "two-voices.wav"),
            ("it", "quiet-negations-it.wav"),
        )
        reports = {
            language: run_job(coordinator, Path("/audio") / filename, language)
            for language, filename in jobs
        }
        print(json.dumps(reports))
    finally:
        coordinator.shutdown()


if __name__ == "__main__":
    main()
