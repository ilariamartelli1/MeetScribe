"""Verify ASR integration image builds and build failures are propagated."""

import subprocess
from unittest.mock import patch

import pytest

from test.src.asr_image import IMAGE_NAME, PROJECT_ROOT, build_asr_image


def test_build_asr_image_uses_project_compose_configuration() -> None:
    with patch("test.src.asr_image.subprocess.run") as run:
        image = build_asr_image()

    run.assert_called_once_with(
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
    assert image == IMAGE_NAME == "meetscribe-app"


def test_build_asr_image_propagates_build_failures() -> None:
    with (
        patch(
            "test.src.asr_image.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "docker"),
        ),
        pytest.raises(subprocess.CalledProcessError),
    ):
        build_asr_image()
