"""Evaluate real offline audio-to-minutes output against synthetic references.

Set MEETSCRIBE_MINUTES_QUALITY_TESTS=1 to run the Docker/GPU evaluation.
"""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = PROJECT_ROOT / "test/src/minutes_quality_smoke.py"
TEST_TIMEOUT_SECONDS = 1800
READY_CHECK = (
    "import urllib.request; "
    "urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2).read(); "
    "urllib.request.urlopen('http://ollama:11434/api/version', timeout=2).read()"
)
REPORT_PREFIX = "MEETSCRIBE_MINUTES_QUALITY_RESULT "


def _compose_command(project: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        project,
        "--profile",
        "summary",
        "-f",
        str(PROJECT_ROOT / "compose.yaml"),
        "-f",
        str(PROJECT_ROOT / "compose.ollama-offline.yaml"),
        "-f",
        str(PROJECT_ROOT / "compose.minutes-quality.yaml"),
    ]


def _wait_for_services(compose: list[str], env: dict[str, str]) -> None:
    deadline = time.monotonic() + 120
    last_error = ""
    while time.monotonic() < deadline:
        ready = subprocess.run(
            [*compose, "exec", "-T", "app", "python", "-c", READY_CHECK],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if ready.returncode == 0:
            return
        last_error = ready.stderr
        time.sleep(0.5)
    pytest.fail(f"The offline app/Ollama stack did not become ready: {last_error}")


def _parse_report(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(REPORT_PREFIX):
            report: object = json.loads(line.removeprefix(REPORT_PREFIX))
            if isinstance(report, dict):
                return report
    raise RuntimeError("The audio-to-minutes test returned no quality report.")


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_MINUTES_QUALITY_TESTS") != "1",
    reason=(
        "Set MEETSCRIBE_MINUTES_QUALITY_TESTS=1 to run the offline GPU minutes "
        "quality test."
    ),
)
def test_real_audio_to_minutes_quality_report() -> None:
    project = f"meetscribe-minutes-quality-{uuid.uuid4().hex[:12]}"
    env = os.environ.copy()
    compose = _compose_command(project)
    try:
        subprocess.run(
            [*compose, "up", "--build", "-d", "app", "ollama"],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        _wait_for_services(compose, env)
        result = subprocess.run(
            [*compose, "exec", "-T", "app", "python", "-"],
            cwd=PROJECT_ROOT,
            env=env,
            input=SMOKE_PATH.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
        )
        assert result.returncode == 0, (
            "Real audio-to-minutes evaluation failed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        report = _parse_report(result.stdout)
        evaluations = report.get("evaluations")
        assert isinstance(evaluations, list) and len(evaluations) == 3, report
        names = {entry.get("case") for entry in evaluations if isinstance(entry, dict)}
        assert names == {
            "english-checkpoint",
            "italian-checkpoint",
            "known-asr-error",
        }, report
        for entry in evaluations:
            assert isinstance(entry, dict), report
            assert entry.get("review_status") == "DRAFT / UNREVIEWED", entry
            assert entry.get("stale") is False, entry
            assert isinstance(entry.get("segments"), list) and entry["segments"], entry
            assert isinstance(entry.get("minutes"), dict), entry
            metrics = entry.get("extraction_metrics")
            assert isinstance(metrics, dict), entry
            assert metrics.get("context_size") == 4096, entry
            assert isinstance(entry.get("reference_script"), str), entry
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        subprocess.run(
            [*compose, "down", "--remove-orphans"],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
