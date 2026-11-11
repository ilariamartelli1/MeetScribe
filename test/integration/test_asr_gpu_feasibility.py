"""Verify repeated offline GPU inference and report feasibility measurements.

Requires MEETSCRIBE_ASR_TESTS=1, a provisioned model, Docker, and an NVIDIA GPU.
Uses the session-scoped ASR image fixture from conftest.py.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = {
    "MEETSCRIBE_MODEL_DIR": PROJECT_ROOT / "models/faster-whisper-large-v3-turbo",
    "MEETSCRIBE_TEST_WAV": PROJECT_ROOT / "test/data/smoke/synthetic-meeting.wav",
    "MEETSCRIBE_TEST_MP3": PROJECT_ROOT / "test/data/smoke/synthetic-meeting.mp3",
}


def integration_paths() -> dict[str, Path]:
    return {
        key: Path(os.environ.get(key, str(default))).resolve(strict=True)
        for key, default in DEFAULT_PATHS.items()
    }


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run the offline GPU integration test.",
)
def test_offline_gpu_inference(asr_image: str) -> None:
    paths = integration_paths()

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--gpus",
            "all",
            "--mount",
            f"type=bind,src={PROJECT_ROOT / 'test/src'},dst=/checks,readonly",
            "--mount",
            f"type=bind,src={paths['MEETSCRIBE_MODEL_DIR']},dst=/model,readonly",
            "--mount",
            f"type=bind,src={paths['MEETSCRIBE_TEST_WAV']},dst=/audio/sample.wav,readonly",
            "--mount",
            f"type=bind,src={paths['MEETSCRIBE_TEST_MP3']},dst=/audio/sample.mp3,readonly",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/asr_feasibility.py",
            "--model-dir",
            "/model",
            "--wav",
            "/audio/sample.wav",
            "--mp3",
            "/audio/sample.mp3",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"GPU container failed:\n{result.stderr}\n{result.stdout}"
    )
    report = json.loads(result.stdout)
    print(json.dumps(report, indent=2))
    assert report["requested_device"] == "cuda"
    assert report["requested_compute_type"] == "int8_float16"
    assert len(report["jobs"]) == 2
    for job in report["jobs"]:
        assert job["effective_device"] == "cuda"
        assert job["effective_compute_type"] == "int8_float16"
        assert [audio["format"] for audio in job["recordings"]] == [".wav", ".mp3"]
        assert all(audio["segments"] > 0 for audio in job["recordings"])
