"""Document the known omission of repeated speech in isolation."""

import json
import os
import subprocess
from pathlib import Path
from typing import Literal

import pytest

from test.integration.asr_model_config import asr_model_mount_args
from test.integration.test_asr_accuracy import normalize_text

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "test/data/smoke"


def repetition_outcome(actual: str, script: str) -> Literal["complete", "omitted"]:
    lines = script.splitlines()
    if len(lines) != 2 or not lines[0].strip() or lines[0] != lines[1]:
        raise ValueError("Repetition script must contain the same sentence twice.")
    normalized = normalize_text(actual)
    if normalized == normalize_text(script):
        return "complete"
    if normalized == normalize_text(lines[0]):
        return "omitted"
    raise AssertionError(f"Unexpected repetition transcript: {actual!r}")


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run the offline GPU integration test.",
)
def test_repeated_sentence_on_real_model(asr_image: str) -> None:
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
            f"type=bind,src={SMOKE},dst=/audio,readonly",
            "--mount",
            f"type=bind,src={ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/asr_single_audio_smoke.py",
            "/audio/repeated-sentence.wav",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Repetition test failed:\n{result.stderr}"
    actual = json.loads(result.stdout)["text"]
    script = (SMOKE / "repeated-sentence-script.txt").read_text(encoding="utf-8")
    if repetition_outcome(actual, script) == "omitted":
        pytest.xfail("Known behavior: the model omitted one repeated sentence.")
