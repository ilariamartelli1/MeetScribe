"""Build the application image used by offline ASR integration tests."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_NAME = "meetscribe-app"
IMAGE_BUILD_TIMEOUT_SECONDS = 2400


def build_asr_image() -> str:
    subprocess.run(
        [
            "docker",
            "compose",
            "--project-name",
            "meetscribe",
            "-f",
            str(PROJECT_ROOT / "compose.yaml"),
            "build",
            "app",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        timeout=IMAGE_BUILD_TIMEOUT_SECONDS,
    )
    return IMAGE_NAME
