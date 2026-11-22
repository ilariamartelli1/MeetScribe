"""Run offline Community-1 diarization after ASR on the selected device."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from app.core.asr_contract import CUDA_DEVICE
from app.adapters.community1_diarizer import MODEL_MANIFEST
from app.infrastructure.model_integrity import verify_model

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
@pytest.mark.parametrize(
    ("device", "format_override"),
    (("cuda", None), ("cpu", None), ("cpu", "mp3")),
)
def test_offline_asr_then_diarization(
    asr_image: str, device: str, format_override: str | None
) -> None:
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
    verify_model(diarization_model, MODEL_MANIFEST)
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            *(["--gpus", "all"] if device == CUDA_DEVICE else []),
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
            "--device",
            device,
            *(
                ["--format", format_override, "--runs-per-format", "1"]
                if format_override
                else []
            ),
        ],
        capture_output=True,
        text=True,
        timeout=CONTAINER_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, (
        f"Offline diarization feasibility failed:\n{completed.stderr}\n{completed.stdout}"
    )
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report["device"] == device
    assert report["torch_cuda_version"] is not None
    assert report["threads"] == EXPECTED_TORCH_THREADS
    expected_formats = (format_override,) if format_override else EXPECTED_JOB_FORMATS
    assert tuple(job["format"] for job in report["jobs"]) == expected_formats
    assert all(
        job["turns"] > 0 and job["exclusive_turns"] > 0 for job in report["jobs"]
    )
    print(json.dumps(report, indent=2))


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run GPU and diarization checks.",
)
def test_cuda_diarization_fails_without_gpu(asr_image: str) -> None:
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--mount",
            f"type=bind,src={ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app:/checks",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/diarization_feasibility.py",
            "--device",
            "cuda",
        ],
        capture_output=True,
        text=True,
        timeout=CONTAINER_TIMEOUT_SECONDS,
    )
    assert completed.returncode != 0
    assert "CUDA GPU unavailable for diarization; CPU fallback is disabled" in (
        completed.stderr
    )


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run GPU and diarization checks.",
)
def test_diarization_rejects_nonpositive_run_count(asr_image: str) -> None:
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--mount",
            f"type=bind,src={ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app:/checks",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/diarization_feasibility.py",
            "--runs-per-format",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=CONTAINER_TIMEOUT_SECONDS,
    )
    assert completed.returncode != 0
    assert "--runs-per-format must be at least 1" in completed.stderr
