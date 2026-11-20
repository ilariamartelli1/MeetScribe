"""Exercise the actual Community-1 adapter on two synthetic voices."""

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from app.community1_diarizer import MODEL_MANIFEST
from app.model_integrity import verify_model
from test.src.audio_fixtures import VOICE_INTERVALS_MS

ROOT = Path(__file__).resolve().parents[2]
TWO_VOICES_AUDIO = ROOT / "test/data/smoke/two-voices.wav"
TWO_VOICES_SHA256 = "a3067b55c8547ff6a40f0ee9195ceb02e02af3678827f09221e6e34590a9f21a"
MIN_SPEAKER_COVERAGE_MS = 1000
CONTAINER_MEMORY_LIMIT = "6g"
CONTAINER_CPU_LIMIT = "4"
CONTAINER_TIMEOUT_SECONDS = 300


def _dominant_speaker(turns: list[dict[str, object]], interval: tuple[int, int]) -> str:
    coverage: Counter[str] = Counter()
    start, end = interval
    for turn in turns:
        turn_start = turn["start_ms"]
        turn_end = turn["end_ms"]
        speaker = turn["speaker_id"]
        assert isinstance(turn_start, int)
        assert isinstance(turn_end, int)
        assert isinstance(speaker, str)
        overlap = max(0, min(end, turn_end) - max(start, turn_start))
        coverage[speaker] += overlap
    assert coverage, f"No speaker turns in interval {interval}"
    speaker, covered_ms = coverage.most_common(1)[0]
    assert covered_ms >= MIN_SPEAKER_COVERAGE_MS
    return speaker


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run GPU diarization checks.",
)
def test_real_community1_adapter_identifies_two_synthetic_voices(
    asr_image: str,
) -> None:
    with TWO_VOICES_AUDIO.open("rb") as audio:
        assert hashlib.file_digest(audio, "sha256").hexdigest() == TWO_VOICES_SHA256
    model = Path(
        os.environ.get(
            "MEETSCRIBE_DIARIZATION_MODEL_DIR",
            str(ROOT / "models/pyannote-community-1"),
        )
    ).resolve(strict=True)
    verify_model(model, MODEL_MANIFEST)
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--gpus",
            "all",
            "--memory",
            CONTAINER_MEMORY_LIMIT,
            "--cpus",
            CONTAINER_CPU_LIMIT,
            "--mount",
            f"type=bind,src={model},dst=/diarization,readonly",
            "--mount",
            f"type=bind,src={TWO_VOICES_AUDIO},dst=/audio/two-voices.wav,readonly",
            "--mount",
            f"type=bind,src={ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/diarization_adapter_smoke.py",
        ],
        capture_output=True,
        text=True,
        timeout=CONTAINER_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, (
        f"Community-1 adapter failed:\n{completed.stderr}\n{completed.stdout}"
    )
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["provenance"]["device"] == "cuda"
    assert result["turns"] and result["exclusive_turns"]
    speakers = [
        _dominant_speaker(result["turns"], interval) for interval in VOICE_INTERVALS_MS
    ]
    assert speakers[0] == speakers[2] and speakers[1] == speakers[3]
    assert speakers[0] != speakers[1], f"Distinct voices were not separated: {speakers}"
