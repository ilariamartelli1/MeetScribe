"""Evaluate real offline audio-to-minutes output against synthetic references.

Set MEETSCRIBE_MINUTES_QUALITY_TESTS=1 to run the Docker/GPU evaluation.
"""

import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from app.core.minutes_contract import MinutesDraft

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = PROJECT_ROOT / "test/src/minutes_quality_smoke.py"
TEST_TIMEOUT_SECONDS = 1800
CASE_SELECTION_ENV = "MEETSCRIBE_MINUTES_QUALITY_CASES"
CASE_NAMES = {
    "english-checkpoint",
    "italian-checkpoint",
    "known-asr-error",
}
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


def _deadline_matches_source_or_weekday(
    deadline_text: str | None, source_phrase: str
) -> bool:
    if deadline_text is None:
        return False
    expected = {
        source_phrase.casefold(),
        source_phrase.rsplit(" ", maxsplit=1)[-1].casefold(),
    }
    return deadline_text.casefold() in expected


def _assert_known_asr_error_quality(evaluation: dict[str, object]) -> None:
    transcript = evaluation.get("transcript_download")
    minutes_data = evaluation.get("minutes")
    assert isinstance(transcript, str), evaluation
    assert isinstance(minutes_data, dict), evaluation
    minutes = MinutesDraft.model_validate_json(json.dumps(minutes_data))

    results_actions = [
        item
        for item in minutes.action_items
        if re.search(r"\breview\w*\b", item.description, re.IGNORECASE)
        and re.search(r"\bresults?\b", item.description, re.IGNORECASE)
    ]
    assert results_actions, "The results review must remain a follow-up action."
    for item in minutes.action_items:
        is_description = re.search(
            r"\b(describ\w*|present\w*|explain\w*)\b", item.description, re.I
        )
        is_in_meeting_subject = re.search(
            r"\b(changes|updates|service)\b", item.description, re.I
        )
        assert not (is_description and is_in_meeting_subject), (
            f"An in-meeting description became a follow-up action: {item.description}"
        )
    assert all(
        not re.search(r"\bon schedule\b", decision.text, re.IGNORECASE)
        for decision in minutes.decisions
    ), "Project status must not be classified as a decision."

    output_text = " ".join(
        [
            minutes.summary,
            *(topic.text for topic in minutes.topics),
            *(decision.text for decision in minutes.decisions),
            *(
                f"{item.description} {item.deadline_text or ''}"
                for item in minutes.action_items
            ),
        ]
    )
    if re.search(r"\bfragi\b", transcript, re.IGNORECASE):
        assert not re.search(r"\bfragi\b", output_text, re.IGNORECASE), (
            "The minutes should normalize the clear weekday typo."
        )
        assert any(
            _deadline_matches_source_or_weekday(item.deadline_text, "on Friday")
            for item in results_actions
        ), "The results-review deadline should normalize to Friday."
    else:
        expected_deadline = re.search(r"\b(?:on|by)\s+Friday\b", transcript, re.I)
        assert expected_deadline, (
            "The transcript contains neither Fragi nor a source-grounded Friday."
        )
        assert any(
            _deadline_matches_source_or_weekday(
                item.deadline_text, expected_deadline.group(0)
            )
            for item in results_actions
        ), "The results-review deadline must preserve the transcript wording."
    assert all(item.deadline_date is None for item in results_actions), (
        "A weekday must not be converted to an unsupported calendar date."
    )
    assert re.search(r"\bfriday\b", minutes.summary, re.IGNORECASE), (
        "The summary should include the results-review weekday."
    )
    assert not re.search(r"\b(unverified|possible transcription)\b", output_text, re.I)


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
    selection = env.get(CASE_SELECTION_ENV)
    if selection is None:
        expected_names = CASE_NAMES
    else:
        expected_names = {name.strip() for name in selection.split(",") if name.strip()}
        assert expected_names and expected_names <= CASE_NAMES, (
            f"Invalid {CASE_SELECTION_ENV} selection: {selection!r}"
        )
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
        smoke_command = [*compose, "exec", "-T"]
        if selection is not None:
            smoke_command.extend(["-e", f"{CASE_SELECTION_ENV}={selection}"])
        smoke_command.extend(["app", "python", "-"])
        result = subprocess.run(
            smoke_command,
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
        print(json.dumps(report, indent=2, ensure_ascii=False))
        evaluations = report.get("evaluations")
        assert isinstance(evaluations, list) and len(evaluations) == len(
            expected_names
        ), report
        names = {entry.get("case") for entry in evaluations if isinstance(entry, dict)}
        assert names == expected_names, report
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
            if entry.get("case") == "known-asr-error":
                _assert_known_asr_error_quality(entry)
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
