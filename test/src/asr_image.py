"""Build the application image used by offline ASR integration tests."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_NAME = "meetscribe-app"


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
        timeout=600,
    )
    return IMAGE_NAME
