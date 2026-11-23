"""Verify the job coordinator runs the production adapter in an offline GPU child.

Uses the session-scoped ASR image fixture from conftest.py.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from test.integration.asr_model_config import asr_model_mount_args

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run the offline GPU integration test.",
)
def test_real_model_runs_in_an_owned_child(asr_image: str) -> None:
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--gpus",
            "all",
            *asr_model_mount_args("/model"),
            "--mount",
            f"type=bind,src={PROJECT_ROOT / 'test/data/smoke'},dst=/audio,readonly",
            "--mount",
            f"type=bind,src={PROJECT_ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/transcription_job_smoke.py",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"GPU transcription job failed:\n{result.stderr}\n{result.stdout}"
    )
    report = json.loads(result.stdout)
    assert report["state"] == "succeeded"
    assert report["language"] == "en"
    assert {"no", "not"}.issubset(re.findall(r"\w+", report["text"].casefold()))
