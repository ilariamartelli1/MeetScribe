"""Verify offline production ASR adapter with quiet synthetic negations.

Uses the session-scoped ASR image fixture from conftest.py.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run the offline GPU integration test.",
)
def test_quiet_negations_and_silence_on_real_model(asr_image: str) -> None:
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
            f"type=bind,src={PROJECT_ROOT / 'test/data/asr'},dst=/audio,readonly",
            "--mount",
            f"type=bind,src={PROJECT_ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/asr_adapter_smoke.py",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"GPU adapter failed:\n{result.stderr}\n{result.stdout}"
    )
    report = json.loads(result.stdout)
    print(json.dumps(report, indent=2))
    assert report["silence_rejected"] is True
    for language, annotations in (
        ("en", (("no", 3000, 3321), ("not", 6647, 6940))),
        ("it", (("no", 3000, 3265), ("non", 5965, 6333))),
    ):
        segments = report[language]["segments"]
        actual = " ".join(segment["text"] for segment in segments)
        tokens = re.findall(r"\w+", actual.casefold())
        missing = [word for word, _, _ in annotations if word not in tokens]
        assert not missing, f"{language}: missed {missing!r}; transcript: {actual!r}"
        for word, start_ms, end_ms in annotations:
            assert any(
                word in re.findall(r"\w+", segment["text"].casefold())
                and segment["start_ms"] < end_ms
                and segment["end_ms"] > start_ms
                for segment in segments
            ), (
                f"{language}: {word!r} missing at original-audio time {start_ms}-{end_ms} ms"
            )
