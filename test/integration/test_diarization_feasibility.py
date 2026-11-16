"""Run offline Community-1 CPU diarization after CUDA ASR."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from test.src.diarization_model import verify_model

ROOT = Path(__file__).resolve().parents[2]
CONTAINER_MEMORY_LIMIT = "6g"
CONTAINER_CPU_LIMIT = "4"
CONTAINER_TIMEOUT_SECONDS = 1200
EXPECTED_TORCH_THREADS = 2
EXPECTED_JOB_FORMATS = ("wav", "mp3", "wav", "mp3")


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run GPU and diarization checks.",
)
def test_offline_asr_then_cpu_diarization(asr_image: str) -> None:
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
    verify_model(diarization_model)
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
            f"type=bind,src={asr_model},dst=/model,readonly",
            "--mount",
            f"type=bind,src={diarization_model},dst=/diarization,readonly",
            "--mount",
            f"type=bind,src={ROOT / 'test/data/smoke'},dst=/audio,readonly",
            "--mount",
            f"type=bind,src={ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app:/checks",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/diarization_feasibility.py",
        ],
        capture_output=True,
        text=True,
        timeout=CONTAINER_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, (
        f"Offline diarization feasibility failed:\n{completed.stderr}\n{completed.stdout}"
    )
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report["threads"] == EXPECTED_TORCH_THREADS
    assert tuple(job["format"] for job in report["jobs"]) == EXPECTED_JOB_FORMATS
    assert all(
        job["turns"] > 0 and job["exclusive_turns"] > 0 for job in report["jobs"]
    )
    print(json.dumps(report, indent=2))
