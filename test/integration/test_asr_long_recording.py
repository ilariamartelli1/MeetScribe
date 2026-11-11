"""Check the synthetic five-minute MP3 for the timestamp regression.

Uses the session-scoped ASR image fixture from conftest.py.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDING = PROJECT_ROOT / "test/data/meeting-en.mp3"


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run the offline GPU integration test.",
)
def test_five_minute_mp3_has_bounded_timestamps_and_final_speech(
    asr_image: str,
) -> None:
    recording = RECORDING.resolve(strict=True)
    model_dir = Path(
        os.environ.get(
            "MEETSCRIBE_MODEL_DIR",
            str(PROJECT_ROOT / "models/faster-whisper-large-v3-turbo"),
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
            "--mount",
            f"type=bind,src={model_dir},dst=/model,readonly",
            "--mount",
            f"type=bind,src={recording},dst=/audio/sample.mp3,readonly",
            "--mount",
            f"type=bind,src={PROJECT_ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/asr_long_recording_smoke.py",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"Long recording failed:\n{result.stderr}\n{result.stdout}"
    )
    report = json.loads(result.stdout)
    assert 290_000 <= report["duration_ms"] <= 300_000
    assert report["condition_on_previous_text"] is False
    assert all(
        0 <= segment["start_ms"] < segment["end_ms"] <= report["duration_ms"]
        for segment in report["segments"]
    )
    text = " ".join(re.findall(r"\w+", report["text"].casefold()))
    assert "the team will meet again after the results have been reviewed" in text
    assert text.count("the results of the results") == 0
    assert any(
        "the team will meet again" in segment["text"].casefold()
        and segment["start_ms"] < 296_955
        and segment["end_ms"] > 292_155
        for segment in report["segments"]
    )
