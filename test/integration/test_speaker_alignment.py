"""Check attributed transcription using two synthetic voices and offline models."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VOICE_INTERVALS_MS = ((0, 7062), (8062, 16062), (17062, 24124), (25124, 33124))
MIN_INTERVAL_COVERAGE = 0.8


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run the offline GPU integration test.",
)
def test_offline_job_attributes_alternating_synthetic_voices(asr_image: str) -> None:
    asr_model = Path(
        os.environ.get(
            "MEETSCRIBE_MODEL_DIR", str(ROOT / "models/faster-whisper-large-v3-turbo")
        )
    ).resolve(strict=True)
    diarization_model = Path(
        os.environ.get(
            "MEETSCRIBE_DIARIZATION_MODEL_DIR",
            str(ROOT / "models/pyannote-community-1"),
        )
    ).resolve(strict=True)
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--gpus",
            "all",
            "--memory",
            "6g",
            "--cpus",
            "4",
            "--mount",
            f"type=bind,src={asr_model},dst=/asr,readonly",
            "--mount",
            f"type=bind,src={diarization_model},dst=/diarization,readonly",
            "--mount",
            f"type=bind,src={ROOT / 'test/data/smoke'},dst=/audio,readonly",
            "--mount",
            f"type=bind,src={ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/speaker_alignment_smoke.py",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Speaker job failed:\n{result.stderr}\n{result.stdout}"
    )
    report = json.loads(result.stdout)
    assert report["provenance"]["device"] == "cuda"
    assert report["diarization"]["provenance"]["device"] == "cuda"
    segments = report["segments"]
    attributions = report["attributions"]
    assert segments and len(segments) == len(attributions)
    assert all(
        (segment["start_ms"], segment["end_ms"])
        == (attribution["start_ms"], attribution["end_ms"])
        for segment, attribution in zip(segments, attributions, strict=True)
    )
    interval_speakers = []
    for start, end in VOICE_INTERVALS_MS:
        speakers = {
            entry["speaker"]
            for entry in attributions
            if entry["speaker"] is not None
            and max(0, min(end, entry["end_ms"]) - max(start, entry["start_ms"]))
            >= (end - start) * MIN_INTERVAL_COVERAGE
        }
        assert len(speakers) == 1, (start, end, speakers, attributions)
        interval_speakers.append(speakers.pop())
    assert interval_speakers[0] == interval_speakers[2]
    assert interval_speakers[1] == interval_speakers[3]
    assert interval_speakers[0] != interval_speakers[1]
